"""Multi-camera Re-ID with probabilistic, revisable global identities ("G-17"). RESEARCH-GRADE.

What it does
  * Per camera, every person track becomes a SEGMENT (camera, track id, first/last seen, entry and exit border).
  * APPEARANCE: a few good crops per segment (not truncated, detector conf >= 0.5, minimum size) are embedded by
    argonix.reid (pretrained OSNet person Re-ID model; GPU fp16 batched, CPU fallback). A segment's appearance is the
    quality-weighted mean of its L2-normalised embeddings.
  * When a new segment B appears, its candidate predecessors are segments that ENDED shortly before: exits from other
    cameras inside the travel-time gate learned by the handoff module (argonix/handoff.py, reused here), same-camera
    re-entries, and same-camera tracker fragments (a track lost mid-image and a new one starting next to it).
  * Each candidate A gets Bayesian odds against "B is a new, unrelated person":
        odds(A) = P(A's camera+exit side -> B's camera)        topology prior (handoff, learned / cold-started)
                * f(dt | edge A->B) / rate of new arrivals on B    travel-time density vs. unrelated arrivals
                * LR_appearance(cosine similarity)                  N(s; same) / N(s; different person)
                * LR_direction                                      exit/entry motion consistent with the edge
        P(B continues A) = odds(A) / (1 + sum of all candidates' odds)
    B joins A's global ID only when P >= link_threshold; otherwise it starts a new ID. The top-k alternatives and
    P(new) are stored with the evidence of every candidate.
  * REVISABLE: when B ends, the decision is re-scored with all of B's crops. A link that drops below
    unlink_threshold is SPLIT off; a better candidate RE-LINKS; an ID that now matches an earlier one is MERGED.
    One exit feeds at most one entry (a later, clearly better claimant takes it; the loser is re-decided).
  * OPERATOR CORRECTIONS (link / unlink / merge IDs) are locked against automatic changes, persisted and audited.
  * The "different person" similarity distribution is re-estimated on site from tracks visible at the same time on
    the same camera (certainly different people). The "same person" distribution is a per-model prior.

What it is NOT (AGENTS.md)
  * Not face recognition, not a name, not a person's identity: a global ID is an opaque number for one journey through
    the cameras, purged after `retention_s`. Nothing here raises an alert or labels anyone.
  * Non-overlapping multi-camera association is research-grade. Every number is a model estimate and can be wrong.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import point_in_polygon
from .handoff import LEFT, SIDES, Lifecycle, border_distances, velocity

NOTE = ("Global IDs are opaque, probabilistic and revisable estimates from appearance + camera topology/travel time. "
        "They are not names or identities, can be wrong, and are purged after the retention period. "
        "Non-overlapping multi-camera association is research-grade.")
METHOD = "bayes-odds-v1"
EXIT_KEYS = set(SIDES) | {"zone"}

DEFAULTS: dict = {
    # appearance model / embedding service
    "model": "auto", "arch": None, "device": "auto", "fp16": True, "batch_size": 32, "max_wait_ms": 20.0,
    "max_queue": 512, "sync": False,
    "color_mode": "auto",       # auto: compare in gray when either side is monochrome/IR | color | gray | both (eval)
    "compare_mode": "color",    # which embedding "both" compares
    "mono_check_s": 5.0,
    "min_conf": 0.5, "min_height_px": 48.0, "edge_px": 3.0, "aspect_min": 1.2, "aspect_max": 5.0,
    "good_height_px": 128.0,
    "sample_every_s": 0.4, "max_crops": 12,
    # track lifecycle (same meaning as in argonix/handoff.py)
    "gap_s": 1.5, "border_margin": 0.08, "extrapolate_s": 0.6, "entry_margin": 0.12, "min_track_s": 0.6,
    "min_obs": 3, "exit_zones_px": {},
    # association
    "mode": "fused",            # fused | appearance (no topology/travel time/direction) | topology (no appearance)
    "min_crops_decide": 3, "decide_after_s": 1.0, "decide_max_wait_s": 5.0,
    "link_threshold": 0.6, "unlink_threshold": 0.35, "relink_margin": 0.15, "merge_margin": 0.05, "top_k": 3,
    "min_dt_s": 0.0, "max_dt_s": 60.0,   # travel-time window without learned topology (handoff disabled / appearance mode)
    "flat_prior": 0.5,          # appearance mode: P(continues to that camera) with no topology
    "midimage_entry_factor": 0.25,  # a new track that did not enter at a border / exit zone is a weaker continuation
    "direction_penalty_llr": -1.5, "min_edge_links": 5,
    "reentry": True, "reentry_max_s": 30.0, "reentry_prior": 0.1,
    "fragment_max_s": 4.0, "fragment_odds": 1.0, "fragment_radius": 0.06, "fragment_radius_per_s": 0.08,
    "appearance": None,         # {"color": {"same": [mu, sd], "diff": [mu, sd]}, "gray": {...}} (else model priors)
    "calib_min_neg": 50, "llr_clip": 6.0,
    # privacy / housekeeping
    "retention_s": 3600.0,      # journeys, crops and thumbnails older than this (data time) are deleted
    "embedding_ttl_s": 300.0,   # embeddings are dropped this long after a segment ends (no longer matchable)
    "save_crops": True, "save_every_s": 30.0, "max_events": 2000, "recent_s": 900.0,
    "keep_trace": False,        # evaluation only: keep every foot position of every segment
}


def _r(v, d=3):
    return None if v is None else round(float(v), d)


def _npdf(x: float, mu: float, sd: float) -> float:
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)


@dataclass
class Segment:
    id: str
    cam: str
    tid: int
    first_t: float
    last_t: float
    entry: Optional[str]                 # left/right/top/bottom/zone, or None (appeared mid-image)
    first_pos: tuple = (0.0, 0.0)
    last_pos: tuple = (0.0, 0.0)
    exit: Optional[str] = None           # side/zone, "lost" (vanished mid-image) or "open"
    closed: bool = False
    blip: bool = False
    gid: int = 0
    pred: Optional[str] = None
    succ: Optional[str] = None
    link: Optional[dict] = None          # evidence of pred -> self
    alternatives: list = field(default_factory=list)
    decided: bool = False
    decided_t: Optional[float] = None
    reevaluated: int = 0
    locked: bool = False                 # an operator decided this segment's predecessor
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)   # last bbox, normalised
    n_obs: int = 0
    hist: deque = field(default_factory=lambda: deque(maxlen=32))
    head: list = field(default_factory=list)    # first few (t, x, y) for the entry direction
    crops: list = field(default_factory=list)   # [(quality, e_color|None, e_gray|None, mono)] - memory only
    n_req: int = 0
    last_crop_t: float = -1e18
    thumb: Optional[bytes] = None
    thumb_q: float = 0.0
    thumb_path: Optional[str] = None
    thumb_dirty: bool = False
    trace: list = field(default_factory=list)
    neg_done: bool = False

    @property
    def mono(self) -> bool:
        return bool(self.crops) and sum(1 for c in self.crops if c[3]) * 2 > len(self.crops)

    def emb(self, mode: str) -> Optional[np.ndarray]:
        i = 1 if mode == "color" else 2
        rows = [(c[0], c[i]) for c in self.crops if c[i] is not None]
        if not rows:
            return None
        v = np.sum([q * e for q, e in rows], axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None

    def public(self, full: bool = False) -> dict:
        d = {"id": self.id, "camera": self.cam, "track_id": self.tid, "first_t": _r(self.first_t, 2),
             "last_t": _r(self.last_t, 2), "active": not self.closed, "gid": self.gid or None,
             "link_confidence": _r((self.link or {}).get("fused_confidence")),
             "decided_by": (self.link or {}).get("decided_by"), "locked": self.locked}
        if full:
            d.update(entry=self.entry, exit=self.exit, n_crops=len(self.crops) or None, crop=self.thumb_path,
                     link=self.link, alternatives=self.alternatives, decided=self.decided,
                     bbox=[_r(v) for v in self.bbox])
        return d


# -------------------------------------------------------------------------------------------------------------------
# Topology / travel-time gates
# -------------------------------------------------------------------------------------------------------------------
class StaticGate:
    """No learned topology: a uniform travel-time window and a uniform destination prior."""

    source = "uniform window (no learned topology)"

    def __init__(self, cfg: dict, cams: list[str]):
        self.cfg, self.cams = cfg, cams

    def window(self, a: str, b: str) -> tuple[float, float]:
        return float(self.cfg["min_dt_s"]), float(self.cfg["max_dt_s"])

    def density(self, a: str, b: str, dt: float) -> float:
        lo, hi = self.window(a, b)
        return 1.0 / max(1e-3, hi - lo)

    def prior(self, a: str, key: str, b: str) -> float:
        return 1.0 / (len(self.cams) + 1)

    def expected(self, a: str, b: str) -> Optional[dict]:
        return None


class HandoffGate:
    """Travel-time window/density and destination prior from the handoff engine's learned topology (argonix/handoff.py);
    the synthetic cold-start topology is used until an edge has enough online samples (`use_sim=False` ignores it)."""

    def __init__(self, engine, use_sim: bool = True):
        self.eng, self.use_sim = engine, use_sim
        self.source = "handoff topology"

    def _stats(self, a: str, b: str):
        e = self.eng
        topos = [("online", e.topology)]
        if self.use_sim and e.sim_topology is not None:
            topos.append(("synthetic cold start", e.sim_topology))
        for name, topo in topos:
            s = topo.tt_stats(a, b)
            if s and s["n"] >= e.cfg["min_edge_n"]:
                return s, name
        return None, None

    def window(self, a: str, b: str) -> tuple[float, float]:
        with self.eng.lock:
            c = self.eng.cfg
            s, _ = self._stats(a, b)
            if s is None:
                return float(c["min_dt_s"]), float(c["max_dt_s"])
            return max(c["min_dt_s"], 0.6 * s["p05"] - 1.0), min(c["max_dt_s"], 1.5 * s["p95"] + 2.0)

    def density(self, a: str, b: str, dt: float) -> float:
        with self.eng.lock:
            c = self.eng.cfg
            s, _ = self._stats(a, b)
            if s is None:
                return 1.0 / max(1e-3, c["max_dt_s"] - c["min_dt_s"])
            z = (math.log(max(dt, 0.05)) - s["mu"]) / s["sigma"]
            return math.exp(-0.5 * z * z) / (max(dt, 0.05) * s["sigma"] * math.sqrt(2 * math.pi))

    def prior(self, a: str, key: str, b: str) -> float:
        e = self.eng
        with e.lock:
            classes = list(e.cams) + ([b] if b not in e.cams else []) + [LEFT]
            sim = e.sim_topology if self.use_sim else None
            topo = e.topology if (e.topology.n_side(a, key) >= e.cfg["min_prior_n"] or sim is None) else sim
            return float(topo.dist(a, key, classes, alpha=1.0).get(b, 0.0))

    def expected(self, a: str, b: str) -> Optional[dict]:
        with self.eng.lock:
            s, name = self._stats(a, b)
        if s is None:
            return None
        return {"p10": _r(s["p10"], 1), "p50": _r(s["p50"], 1), "p90": _r(s["p90"], 1), "n": s["n"], "source": name}


# -------------------------------------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------------------------------------
class IdentityEngine:
    """Site-independent core: segments, candidate scoring, global IDs, merges/splits, operator corrections."""

    def __init__(self, cams: list[str], cfg: Optional[dict] = None, data_dir: Optional[Path] = None, gate=None,
                 sim_priors: Optional[dict] = None, clock=time.time):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cams = list(cams)
        self.dir = Path(data_dir) if data_dir else None
        self.gate = gate or StaticGate(self.cfg, self.cams)
        self.sim_priors = sim_priors or {}
        self.clock = clock
        self.lock = threading.RLock()
        self.life = Lifecycle(self.cfg)                       # reused for exit classification only
        self.segs: dict[str, Segment] = {}
        self.open: dict[tuple, str] = {}
        self.undecided: set[str] = set()
        self.gids: dict[int, dict] = {}
        self.alias: dict[int, int] = {}
        self.next_gid = 1
        self.cannot: set[tuple[str, str]] = set()
        self.events: deque = deque(maxlen=int(self.cfg["max_events"]))
        self.wm: dict[str, tuple[float, float]] = {}
        self.first_t: dict[str, float] = {}
        self.arrivals: deque = deque(maxlen=20000)            # (cam, t, entered_at_border)
        self.neg: dict[str, deque] = {"color": deque(maxlen=2000), "gray": deque(maxlen=2000)}
        self.edge_entry: dict[tuple, Counter] = defaultdict(Counter)
        self.entry_marg: dict[str, Counter] = defaultdict(Counter)
        self.mono_cams: dict[str, bool] = {}
        self.any_mono = self.cfg["color_mode"] in ("gray", "both")
        self.n = Counter()
        self.latencies: deque = deque(maxlen=2000)
        self._last_save = 0.0
        self._last_purge = -1e18
        self._load()

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _now(self) -> float:
        return max((t for t, _ in self.wm.values()), default=0.0)

    def _watermark(self) -> Optional[float]:
        now = time.monotonic()
        live = [t for t, wall in self.wm.values() if now - wall < 10.0]
        return min(live) if live else None

    def _event(self, typ: str, **kw) -> None:
        self.events.append({"type": typ, "t": _r(self._now(), 2), "wall": _r(self.clock(), 2), **kw})

    def _new_gid(self, t: float, **kw) -> int:
        g = self.next_gid
        self.next_gid += 1
        self.gids[g] = {"created_t": t, "merged_into": None, **kw}
        return g

    def resolve(self, gid: int) -> int:
        seen = set()
        while gid in self.alias and gid not in seen:
            seen.add(gid)
            gid = self.alias[gid]
        return gid

    def chain(self, s: Segment) -> list[Segment]:
        while s.pred and s.pred in self.segs:
            s = self.segs[s.pred]
        out = [s]
        while s.succ and s.succ in self.segs:
            s = self.segs[s.succ]
            out.append(s)
        return out

    def _forward(self, s: Segment) -> list[Segment]:
        out = [s]
        while s.succ and s.succ in self.segs:
            s = self.segs[s.succ]
            out.append(s)
        return out

    def need_gray(self, mono: bool) -> bool:
        return mono or self.cfg["color_mode"] in ("gray", "both") or (self.cfg["color_mode"] == "auto" and self.any_mono)

    def need_color(self) -> bool:
        return self.cfg["color_mode"] != "gray"

    def set_mono(self, cam: str, mono: bool) -> None:
        with self.lock:
            self.mono_cams[cam] = mono
            if mono:
                self.any_mono = True

    def pair_mode(self, a: Segment, b: Segment) -> str:
        m = self.cfg["color_mode"]
        if m in ("gray", "color"):
            return m
        if m == "both":
            return self.cfg["compare_mode"]
        return "gray" if (a.mono or b.mono) else "color"

    # ---- appearance model --------------------------------------------------------------------------------------
    def app_params(self, mode: str) -> tuple[list, list, str]:
        over = (self.cfg.get("appearance") or {}).get(mode) or {}
        pri = (self.sim_priors or {}).get(mode) or {}
        same = over.get("same") or pri.get("same") or [0.7, 0.1]
        diff = over.get("diff") or pri.get("diff") or [0.4, 0.12]
        src = "config" if over.get("diff") else "model prior"
        neg = self.neg.get(mode) or ()
        if len(neg) >= int(self.cfg["calib_min_neg"]) and not over.get("diff"):
            a = np.asarray(neg, np.float32)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med))) * 1.4826
            diff = [med, max(0.04, mad)]
            src = f"site ({len(neg)} co-visible pairs)"
        return list(same), list(diff), src

    def app_llr(self, s: float, mode: str) -> float:
        same, diff, _ = self.app_params(mode)
        c = float(self.cfg["llr_clip"])
        return max(-c, min(c, _npdf(s, *same) - _npdf(s, *diff)))

    def add_embedding(self, sid: str, e_color, e_gray, meta: dict) -> None:
        """Embedding service callback (any thread)."""
        with self.lock:
            s = self.segs.get(sid)
            if s is None or len(s.crops) >= int(self.cfg["max_crops"]):
                return
            s.crops.append((float(meta.get("q", 1.0)), None if e_color is None else np.asarray(e_color, np.float32),
                            None if e_gray is None else np.asarray(e_gray, np.float32), bool(meta.get("mono"))))
            if s.decided and not s.locked and not s.reevaluated and len(s.crops) >= int(self.cfg["max_crops"]):
                self._decide(s, reeval=True)

    def offer_thumb(self, sid: str, q: float, jpg: bytes) -> None:
        with self.lock:
            s = self.segs.get(sid)
            if s is not None and q > s.thumb_q:
                s.thumb, s.thumb_q, s.thumb_dirty = jpg, q, True

    # ---- per frame ---------------------------------------------------------------------------------------------
    def observe(self, cam: str, t: float, w: float, h: float, dets, zones=()) -> list[tuple[str, tuple, float]]:
        """Feed one analytics frame. Returns crop requests [(segment id, bbox px, quality)] for the caller to embed."""
        c = self.cfg
        with self.lock:
            if cam not in self.cams:
                self.cams.append(cam)
            self.first_t.setdefault(cam, t)
            prev = self.wm.get(cam)
            self.wm[cam] = (max(t, prev[0]) if prev else t, time.monotonic())
            want = []
            for tid, bb, cls, conf in dets:
                if cls != "person":
                    continue
                x1, y1, x2, y2 = (float(v) for v in bb)
                xn, yn = min(1.0, max(0.0, (x1 + x2) / 2.0 / w)), min(1.0, max(0.0, y2 / h))
                sid = self.open.get((cam, tid))
                s = self.segs.get(sid) if sid else None
                if s is None:
                    in_zone = any(point_in_polygon((xn, yn), z) for z in zones)
                    d = border_distances(xn, yn)
                    side = min(SIDES, key=lambda k: d[k])
                    entry = "zone" if in_zone else (side if d[side] <= c["entry_margin"] else None)
                    s = Segment(f"{cam}:{tid}:{int(round(t * 1000))}", cam, int(tid), t, t, entry, (xn, yn), (xn, yn))
                    self.segs[s.id] = s
                    self.open[(cam, tid)] = s.id
                    self.undecided.add(s.id)
                    self.n["segments"] += 1
                elif t <= s.last_t:
                    continue
                s.last_t = t
                s.hist.append((t, xn, yn, None, None))
                if len(s.head) < 6:
                    s.head.append((t, xn, yn))
                s.last_pos = (xn, yn)
                s.bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                s.n_obs += 1
                if c["keep_trace"]:
                    s.trace.append((round(t, 3), round(xn, 4), round(yn, 4)))
                q = self._quality(bb, w, h, conf)
                if q > 0 and t - s.last_crop_t >= c["sample_every_s"] and s.n_req < c["max_crops"]:
                    s.last_crop_t = t
                    s.n_req += 1
                    want.append((s.id, (x1, y1, x2, y2), q))
            for key, sid in [(k, v) for k, v in self.open.items() if k[0] == cam]:
                s = self.segs[sid]
                if t - s.last_t > c["gap_s"]:
                    self._close(s, zones)
            self._step()
            return want

    def _quality(self, bb, w, h, conf) -> float:
        from .reid import crop_quality
        c = self.cfg
        return crop_quality(bb, w, h, conf, {k: c[k] for k in ("min_conf", "min_height_px", "edge_px", "aspect_min",
                                                                "aspect_max", "good_height_px")})

    def _close(self, s: Segment, zones=(), exit_key: Optional[str] = None) -> None:
        self.open.pop((s.cam, s.tid), None)
        s.closed = True
        if exit_key is not None:
            s.exit = exit_key
        elif len(s.hist) >= 2:
            ok, side, in_zone = self.life._classify_exit(s, zones)
            s.exit = ("zone" if in_zone else side) if ok else "lost"
        else:
            s.exit = "lost"
        if s.n_obs < int(self.cfg["min_obs"]) or s.last_t - s.first_t < float(self.cfg["min_track_s"]):
            if not s.decided:  # detector blip: never becomes an identity
                s.blip = True
                self.undecided.discard(s.id)
                self.segs.pop(s.id, None)
                self.n["blips"] += 1
                return
        if s.entry is not None:
            self.entry_marg[s.cam][s.entry] += 1
            self.arrivals.append((s.cam, s.first_t, True))
        else:
            self.arrivals.append((s.cam, s.first_t, False))
        self._negatives(s)
        if s.decided and not s.locked:
            self._decide(s, reeval=True)

    def _negatives(self, b: Segment) -> None:
        """Self-calibration: tracks visible at the same time on the same camera are different people."""
        if b.neg_done or not b.crops:
            return
        b.neg_done = True
        for s in self.segs.values():
            if s is b or s.neg_done or s.cam != b.cam or not s.crops or s.first_t > b.last_t or s.last_t < b.first_t:
                continue  # s.neg_done: that segment already counted its pairs when it closed (no double counting)
            ov = min(s.last_t, b.last_t) - max(s.first_t, b.first_t)
            if ov < 0.5:
                continue
            for mode in ("color", "gray"):
                ea, eb = s.emb(mode), b.emb(mode)
                if ea is not None and eb is not None:
                    self.neg[mode].append(float(ea @ eb))

    def _step(self) -> None:
        c = self.cfg
        wm = self._watermark()
        if wm is not None and self.undecided:
            for sid in sorted(self.undecided, key=lambda k: self.segs[k].first_t if k in self.segs else 0.0):
                s = self.segs.get(sid)
                if s is None or s.decided:
                    self.undecided.discard(sid)
                    continue
                if wm < s.first_t + c["gap_s"] + 0.25:
                    continue  # a candidate that ended just before s started may not be closed yet
                dur = s.last_t - s.first_t
                ready = s.closed or (len(s.crops) >= c["min_crops_decide"] and dur >= c["decide_after_s"]) \
                    or dur >= c["decide_max_wait_s"]
                if ready and s.n_obs >= int(c["min_obs"]):
                    self._decide(s)
        now = self._now()
        if now - self._last_purge >= 10.0:
            self._last_purge = now
            self.purge(now)
        if self.dir and self.clock() - self._last_save >= float(c["save_every_s"]):
            self._save_locked()

    # ---- scoring -----------------------------------------------------------------------------------------------
    def _arrival_rate(self, cam: str, t: float, border: bool = True) -> float:
        """New tracks per second on `cam` (border/zone entries only when `border`): the rate at which unrelated people
        could explain a new track - the 'new identity' alternative of every decision."""
        span = max(60.0, min(600.0, t - self.first_t.get(cam, t)))
        n = sum(1 for c, ta, e in self.arrivals if c == cam and t - span <= ta <= t and (e or not border))
        return max(n, 1) / span

    def _entry_ok(self, b: Segment) -> Optional[bool]:
        if b.entry in (None, "zone") or len(b.head) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = b.head[0], b.head[-1]
        if t1 - t0 < 0.15:
            return None
        vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
        inward = {"left": vx, "right": -vx, "top": vy, "bottom": -vy}[b.entry]
        return inward > -0.02

    def _exit_ok(self, a: Segment) -> Optional[bool]:
        if a.exit not in SIDES or len(a.hist) < 2:
            return None
        vx, vy = velocity(a.hist)
        outward = {"left": -vx, "right": vx, "top": -vy, "bottom": vy}[a.exit]
        return outward > -0.02

    def _candidates(self, b: Segment) -> list[tuple[Segment, str, float]]:
        c = self.cfg
        horizon = max(float(c["max_dt_s"]), float(c["reentry_max_s"])) + 30.0
        out = []
        for a in self.segs.values():
            if a is b or not a.closed or a.blip or a.first_t >= b.first_t or b.first_t - a.last_t > horizon:
                continue
            if (a.id, b.id) in self.cannot:
                continue
            if a.succ and a.succ != b.id and self.segs.get(a.succ) is not None and self.segs[a.succ].locked:
                continue  # an operator decided where a went
            dt = b.first_t - a.last_t
            if a.cam != b.cam:
                if a.exit not in EXIT_KEYS:
                    continue  # vanished mid-image: occlusion / tracker loss, not a camera handoff
                if c["mode"] == "appearance":
                    lo, hi = float(c["min_dt_s"]), float(c["max_dt_s"])
                else:
                    lo, hi = self.gate.window(a.cam, b.cam)
                if lo <= dt <= hi:
                    out.append((a, "handoff", dt))
            elif dt > 0 and c["reentry"]:
                d = math.hypot(a.last_pos[0] - b.first_pos[0], a.last_pos[1] - b.first_pos[1])
                if dt <= c["fragment_max_s"] and d <= c["fragment_radius"] + c["fragment_radius_per_s"] * dt:
                    out.append((a, "fragment", dt))
                elif a.exit in EXIT_KEYS and b.entry is not None and dt <= c["reentry_max_s"]:
                    out.append((a, "reentry", dt))
        return out

    def _score(self, a: Segment, b: Segment, kind: str, dt: float) -> tuple[float, dict]:
        c, mode = self.cfg, self.cfg["mode"]
        ev: dict = {"kind": kind, "from_segment": a.id, "to_segment": b.id, "from_camera": a.cam, "to_camera": b.cam,
                    "from_track": a.tid, "to_track": b.tid, "dt_s": _r(dt, 2), "method": METHOD, "mode": mode}
        lam = self._arrival_rate(b.cam, b.first_t, border=b.entry is not None)
        ev["arrival_rate_per_min"] = _r(lam * 60, 2)
        if kind == "handoff":
            if mode == "appearance":
                prior, f = float(c["flat_prior"]), 1.0 / max(1e-3, c["max_dt_s"] - c["min_dt_s"])
                ev.update(topology_prob=None, expected_dt_s=None, travel_time_llr=None)
                odds = prior * f / lam
            else:
                prior = self.gate.prior(a.cam, a.exit, b.cam)
                f = self.gate.density(a.cam, b.cam, dt)
                lo, hi = self.gate.window(a.cam, b.cam)
                ev.update(topology_prob=_r(prior), expected_dt_s=self.gate.expected(a.cam, b.cam),
                          travel_time_llr=_r(math.log(max(1e-12, f * max(1e-3, hi - lo))), 2),
                          gate_s=[_r(lo, 1), _r(hi, 1)], topology_source=getattr(self.gate, "source", ""))
                odds = prior * f / lam
                if b.entry is None:
                    odds *= float(c["midimage_entry_factor"])
                eo, xo = self._entry_ok(b), self._exit_ok(a)
                llr = 0.0 if eo is not False and xo is not False else float(c["direction_penalty_llr"])
                learned = None
                cnt = self.edge_entry.get((a.cam, a.exit, b.cam))
                if cnt and sum(cnt.values()) >= c["min_edge_links"] and b.entry is not None:
                    n, marg = sum(cnt.values()), self.entry_marg.get(b.cam) or Counter()
                    m = sum(marg.values())
                    learned = math.log((cnt[b.entry] + 0.5) / (n + 2.5)) - math.log((marg[b.entry] + 0.5) / (m + 2.5))
                    learned = max(-2.0, min(2.0, learned))
                    llr += learned
                odds *= math.exp(llr)
                ev["direction"] = {"exit": a.exit, "entry": b.entry, "exit_motion_ok": xo, "entry_motion_ok": eo,
                                   "learned_entry_llr": _r(learned, 2), "llr": _r(llr, 2)}
            ev["entered_at_border"] = b.entry is not None
        elif kind == "reentry":
            tau = max(1.0, float(c["reentry_max_s"]) / 3.0)
            odds = float(c["reentry_prior"]) * math.exp(-dt / tau) / tau / lam
        else:  # fragment
            d = math.hypot(a.last_pos[0] - b.first_pos[0], a.last_pos[1] - b.first_pos[1])
            sig = float(c["fragment_radius"]) + float(c["fragment_radius_per_s"]) * dt
            odds = float(c["fragment_odds"]) * math.exp(-0.5 * (d / sig) ** 2)
            ev["gap_distance"] = _r(d)
        ev["prior_odds"] = _r(odds, 4)
        app = None
        if mode != "topology":
            pm = self.pair_mode(a, b)
            ea, eb = a.emb(pm), b.emb(pm)
            if ea is not None and eb is not None:
                sim = float(ea @ eb)
                llr = self.app_llr(sim, pm)
                _, _, src = self.app_params(pm)
                app = {"similarity": _r(sim), "mode": pm, "llr": _r(llr, 2), "n_crops": [len(a.crops), len(b.crops)],
                       "different_person_model": src}
                odds *= math.exp(llr)
        ev["appearance"] = app
        ev["odds"] = _r(odds, 4)
        return odds, ev

    # ---- decisions ---------------------------------------------------------------------------------------------
    def _posterior(self, b: Segment, exclude: set) -> tuple[list, float]:
        """Hypotheses are IDENTITIES: several earlier segments of one global ID are alternative departure points of
        the same hypothesis, so each ID contributes its best segment's odds once (they must not split the mass)."""
        scored, best = [], {}
        for a, kind, dt in self._candidates(b):
            if a.id in exclude:
                continue
            odds, ev = self._score(a, b, kind, dt)
            scored.append((a, odds, ev))
            key = a.gid or a.id
            best[key] = max(best.get(key, 0.0), odds)
        tot = 1.0 + sum(best.values())
        for a, o, ev in scored:
            ev["fused_confidence"] = _r(o / tot)
        scored.sort(key=lambda x: -x[1])
        return scored, 1.0 / tot

    def _alternatives(self, scored, p_new) -> list[dict]:
        k, seen, alts = int(self.cfg["top_k"]), set(), []
        for a, _, ev in scored:
            key = a.gid or a.id
            if key in seen:
                continue
            seen.add(key)
            alts.append({"segment": a.id, "gid": a.gid or None, "camera": a.cam, "track_id": a.tid,
                         "confidence": ev["fused_confidence"], "evidence": ev})
            if len(alts) >= k:
                break
        alts.append({"new_identity": True, "confidence": _r(p_new)})
        return alts

    def _decide(self, b: Segment, reeval: bool = False, depth: int = 0) -> None:
        c = self.cfg
        thr = float(c["link_threshold"])
        exclude: set = set()
        choice = None
        while True:
            scored, p_new = self._posterior(b, exclude)
            best = scored[0] if scored else None
            if best is None or best[2]["fused_confidence"] < thr:
                break
            a, _, ev = best
            if a.succ and a.succ != b.id:
                other = self.segs.get(a.succ)
                if other is None:
                    a.succ = None
                elif other.locked:
                    exclude.add(a.id)
                    continue
                elif ev["fused_confidence"] > ((other.link or {}).get("fused_confidence") or 0.0) + c["relink_margin"] \
                        and depth < 4:
                    choice = best  # b claims a; the current successor is re-decided afterwards
                    break
                else:
                    exclude.add(a.id)
                    continue
            choice = best
            break
        alts = self._alternatives(scored, p_new)
        first = not b.decided
        b.alternatives = alts
        b.decided = True
        self.undecided.discard(b.id)
        if first:
            b.decided_t = self._now()
            self.latencies.append(max(0.0, b.decided_t - b.first_t))
        else:
            b.reevaluated += 1
        cur = b.pred
        if choice is None:
            if first:
                b.gid = self._new_gid(b.first_t, first_segment=b.id)
                for s in self._forward(b)[1:]:
                    s.gid = b.gid
                self._event("new", gid=b.gid, segment=b.id, p_new=_r(p_new))
                self.n["new"] += 1
            elif cur is not None:
                cur_conf = next((ev["fused_confidence"] for a, _, ev in scored if a.id == cur), 0.0)
                if cur_conf < float(c["unlink_threshold"]):
                    old = b.gid
                    self._unlink(b, by="auto", reason=f"re-scored at {_r(cur_conf)} < unlink threshold")
                    self.n["auto_splits"] += 1
                    self._event("split", gid=old, new_gid=b.gid, segment=b.id, by="auto", confidence=_r(cur_conf))
                elif b.link:
                    b.link.update(fused_confidence=_r(cur_conf), rescored=True)
            return
        a, _, ev = choice
        if cur == a.id:
            if b.link:  # same decision: refresh the evidence
                keep = {k: b.link[k] for k in ("decided_by", "t", "operator_confirmed", "confirmed_by") if k in b.link}
                b.link = {**ev, **keep}
            return
        relink = cur is not None
        if relink:
            cur_conf = next((e["fused_confidence"] for s, _, e in scored if s.id == cur), 0.0)
            if ev["fused_confidence"] < cur_conf + float(c["relink_margin"]):
                return  # not clearly better: keep the current link (hysteresis)
            self._unlink(b, by="auto", reason=f"re-linked: {a.id} scored {ev['fused_confidence']}", assign=False)
        elif not first and ev["fused_confidence"] < thr + float(c["merge_margin"]):
            return  # merging an existing ID needs a little extra margin
        displaced = None
        if a.succ and a.succ != b.id:
            displaced = self.segs.get(a.succ)
            if displaced is not None:
                self._unlink(displaced, by="auto", reason=f"{b.id} matched {a.id} better")
                self.n["auto_displaced"] += 1
                self._event("split", gid=a.gid, new_gid=displaced.gid, segment=displaced.id, by="auto",
                            reason=f"{b.id} matched {a.id} better")
        old_gid = b.gid
        self._link(a, b, ev, by="auto")
        if relink:
            self.n["auto_relinks"] += 1
            self._event("relink", gid=b.gid, from_gid=old_gid, segment=b.id, from_segment=a.id, previous=cur,
                        by="auto", confidence=ev["fused_confidence"])
        elif old_gid and old_gid != b.gid:
            self.n["auto_merges"] += 1
            self._event("merge", gid=old_gid, into=b.gid, segment=b.id, by="auto",
                        confidence=ev["fused_confidence"])
        else:
            self._event("link", gid=b.gid, segment=b.id, from_segment=a.id, by="auto",
                        confidence=ev["fused_confidence"])
        self.n["auto_links"] += 1
        if displaced is not None and not displaced.locked:
            self._decide(displaced, reeval=True, depth=depth + 1)

    def _link(self, a: Segment, b: Segment, ev: dict, by: str) -> None:
        a.succ, b.pred = b.id, a.id
        b.link = {**ev, "decided_by": by, "t": _r(self._now(), 2)}
        if not a.gid:
            a.gid = self._new_gid(a.first_t, first_segment=a.id)
        old = b.gid
        for s in self._forward(b):
            s.gid = a.gid
        if old and old != a.gid and not any(s.gid == old for s in self.segs.values()):
            self.alias[old] = a.gid
            if old in self.gids:
                self.gids[old]["merged_into"] = a.gid
        if ev.get("kind") == "handoff" and b.entry is not None and (by != "auto" or (ev.get("fused_confidence") or 0) >= 0.8):
            self.edge_entry[(a.cam, a.exit, b.cam)][b.entry] += 1

    def _unlink(self, b: Segment, by: str, reason: str = "", assign: bool = True) -> Optional[Segment]:
        """Cut the link into b. With assign, b and its successors get a NEW global ID (a split)."""
        a = self.segs.get(b.pred) if b.pred else None
        if a is not None and a.succ == b.id:
            a.succ = None
        b.pred = None
        prev = b.link
        b.link = None
        if assign:
            g = self._new_gid(b.first_t, first_segment=b.id, split_from=b.gid or None,
                              split_reason=reason, split_by=by, previous_link=prev and {
                                  k: prev.get(k) for k in ("from_segment", "fused_confidence", "decided_by")})
            for s in self._forward(b):
                s.gid = g
        return a

    # ---- operator corrections ----------------------------------------------------------------------------------
    def _seg(self, sid: str) -> Segment:
        s = self.segs.get(sid)
        if s is None:
            raise KeyError(f"unknown or expired segment {sid!r}")
        return s

    def operator_link(self, a_id: str, b_id: str, by: str, persist: bool = True) -> dict:
        """The operator states that segments a and b are the same person (a before b). Locked; audited by the API."""
        with self.lock:
            a, b = self._seg(a_id), self._seg(b_id)
            if a.first_t > b.first_t:
                a, b = b, a
            if a is b:
                raise ValueError("cannot link a segment to itself")
            if a.last_t > b.first_t + 0.5:
                raise ValueError("the two segments overlap in time: one person cannot be in both tracks at once")
            if b.pred == a.id:
                if not b.locked and (b.link or {}).get("decided_by") == "auto":
                    self.n["operator_confirmed_auto"] += 1
                b.locked = True
                b.link = {**(b.link or {}), "operator_confirmed": True, "confirmed_by": by}
                self.n["operator_confirms"] += 1
                self._event("confirm", gid=b.gid, segment=b.id, from_segment=a.id, by=by)
            else:
                if a.succ:
                    s = self.segs.get(a.succ)
                    if s is not None:
                        self._reject(s, by)
                        self._unlink(s, by=by, reason="operator re-linked its predecessor")
                        self.cannot.add((a.id, s.id))
                if b.pred:
                    p = b.pred
                    self._reject(b, by)
                    self._unlink(b, by=by, reason="operator re-linked")
                    self.cannot.add((p, b.id))
                kind = "handoff" if a.cam != b.cam else "reentry"
                try:
                    odds, ev = self._score(a, b, kind, b.first_t - a.last_t)
                except Exception:  # noqa: BLE001 - evidence is informative only
                    ev = {"kind": kind}
                ev.update(kind="operator", model_kind=kind, fused_confidence=1.0, auto_origin=False)
                old = b.gid
                self._link(a, b, ev, by=by)
                b.locked = True
                self.n["operator_links"] += 1
                self._event("operator_link", gid=b.gid, merged=old if old != b.gid else None, segment=b.id,
                            from_segment=a.id, by=by)
            self.cannot.discard((a.id, b.id))
            res = {"gid": b.gid, "label": f"G-{b.gid}", "from_segment": a.id, "to_segment": b.id}
            if persist:
                self._correction({"action": "link", "a": a.id, "b": b.id, "by": by})
            return res

    def _reject(self, b: Segment, by: str) -> None:
        if b.link and b.link.get("decided_by") == "auto":
            self.n["operator_rejected_auto"] += 1

    def operator_unlink(self, b_id: str, by: str, persist: bool = True) -> dict:
        """The operator states that segment b is NOT the continuation of its current predecessor."""
        with self.lock:
            b = self._seg(b_id)
            if not b.pred:
                raise ValueError("this segment has no predecessor link to remove")
            old = b.gid
            self._reject(b, by)
            a = self._unlink(b, by=by, reason="operator")
            if a is not None:
                self.cannot.add((a.id, b.id))
            b.locked = True
            self.n["operator_unlinks"] += 1
            self._event("operator_unlink", gid=old, new_gid=b.gid, segment=b.id, from_segment=a.id if a else None, by=by)
            if persist:
                self._correction({"action": "unlink", "a": a.id if a else None, "b": b.id, "by": by})
            return {"gid": old, "new_gid": b.gid, "label": f"G-{b.gid}", "segment": b.id}

    def operator_merge(self, g1: int, g2: int, by: str) -> dict:
        """Merge two global IDs (their segments interleave in time order; they must not overlap)."""
        with self.lock:
            g1, g2 = self.resolve(int(g1)), self.resolve(int(g2))
            if g1 == g2:
                raise ValueError("already the same global ID")
            segs = sorted([s for s in self.segs.values() if s.gid in (g1, g2)], key=lambda s: s.first_t)
            if not any(s.gid == g1 for s in segs) or not any(s.gid == g2 for s in segs):
                raise KeyError("unknown or expired global ID")
            for x, y in zip(segs, segs[1:]):
                if x.last_t > y.first_t + 0.5:
                    raise ValueError(f"segments {x.id} and {y.id} overlap in time: one person cannot be in both")
            for x, y in zip(segs, segs[1:]):
                if y.pred != x.id:
                    self.operator_link(x.id, y.id, by, persist=False)
            gid = segs[-1].gid
            self._correction({"action": "merge", "gids": [g1, g2], "segments": [s.id for s in segs], "by": by})
            return {"gid": gid, "label": f"G-{gid}", "merged": [g1, g2]}

    def _correction(self, rec: dict) -> None:
        rec = {"wall": time.time(), **rec}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "corrections.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        self._save_locked()

    # ---- read-outs ---------------------------------------------------------------------------------------------
    def _identity_rows(self) -> dict[int, list[Segment]]:
        rows: dict[int, list[Segment]] = defaultdict(list)
        for s in self.segs.values():
            if s.gid and not s.blip:
                rows[s.gid].append(s)
        for v in rows.values():
            v.sort(key=lambda s: s.first_t)
        return rows

    def identities(self, active_only: bool = False, since_s: Optional[float] = None, limit: int = 200) -> list[dict]:
        with self.lock:
            now = self._now()
            since = float(self.cfg["recent_s"] if since_s is None else since_s)
            out = []
            for g, segs in self._identity_rows().items():
                active = any(not s.closed for s in segs)
                if active_only and not active:
                    continue
                if not active and now - segs[-1].last_t > since:
                    continue
                confs = [s.link["fused_confidence"] for s in segs if s.link and s.link.get("fused_confidence") is not None]
                cams = []
                for s in segs:
                    if not cams or cams[-1] != s.cam:
                        cams.append(s.cam)
                cur = next((s for s in reversed(segs) if not s.closed), None)
                out.append({"gid": g, "label": f"G-{g}", "active": active, "first_t": _r(segs[0].first_t, 2),
                            "last_t": _r(max(s.last_t for s in segs), 2), "cameras": cams,
                            "n_segments": len(segs), "n_cameras": len(set(cams)),
                            "min_link_confidence": _r(min(confs)) if confs else None,
                            "operator_verified": any(s.locked for s in segs),
                            "current": {"camera": cur.cam, "track_id": cur.tid, "bbox": [_r(v) for v in cur.bbox]}
                            if cur else None,
                            "segments": [s.public() for s in segs]})
            out.sort(key=lambda r: (not r["active"], -r["last_t"]))
            return out[:limit]

    def journey(self, gid: int) -> Optional[dict]:
        with self.lock:
            g = self.resolve(int(gid))
            segs = sorted([s for s in self.segs.values() if s.gid == g and not s.blip], key=lambda s: s.first_t)
            if not segs:
                return None
            self._flush_thumbs(segs)
            evs = [e for e in self.events if g in (e.get("gid"), e.get("new_gid"), e.get("into"))
                   or e.get("segment") in {s.id for s in segs}]
            return {"gid": g, "label": f"G-{g}", "requested": int(gid), "merged_into": g if g != int(gid) else None,
                    "active": any(not s.closed for s in segs), "first_t": _r(segs[0].first_t, 2),
                    "last_t": _r(max(s.last_t for s in segs), 2), "segments": [s.public(full=True) for s in segs],
                    "info": {k: v for k, v in self.gids.get(g, {}).items() if k != "previous_link"},
                    "events": evs[-100:], "note": NOTE, "kind": "estimate"}

    def gid_of(self, cam: str, tid: int) -> Optional[tuple[int, Optional[float]]]:
        with self.lock:
            s = self.segs.get(self.open.get((cam, tid), ""))
            if s is None or not s.decided or not s.gid:
                return None
            return s.gid, (s.link or {}).get("fused_confidence")

    def metrics(self) -> dict:
        with self.lock:
            rows = self._identity_rows()
            confs = [s.link["fused_confidence"] for s in self.segs.values() if s.link and s.link.get("decided_by") == "auto"]
            conf_ok, rej = self.n["operator_confirmed_auto"], self.n["operator_rejected_auto"]
            app = {}
            for m in ("color", "gray"):
                same, diff, src = self.app_params(m)
                app[m] = {"same": [_r(v) for v in same], "different": [_r(v) for v in diff], "different_source": src,
                          "n_negative_pairs": len(self.neg[m])}
            lat = list(self.latencies)
            return {"segments_total": self.n["segments"], "segments_in_memory": len(self.segs),
                    "segments_open": len(self.open), "undecided": len(self.undecided), "blips_dropped": self.n["blips"],
                    "identities": len(rows), "identities_active": sum(1 for v in rows.values() if any(not s.closed for s in v)),
                    "multi_camera_identities": sum(1 for v in rows.values() if len({s.cam for s in v}) > 1),
                    "auto": {"links": self.n["auto_links"], "new": self.n["new"], "merges": self.n["auto_merges"],
                             "splits": self.n["auto_splits"], "relinks": self.n["auto_relinks"],
                             "displaced": self.n["auto_displaced"],
                             "mean_link_confidence": _r(float(np.mean(confs))) if confs else None},
                    "operator": {"links": self.n["operator_links"], "unlinks": self.n["operator_unlinks"],
                                 "confirms": self.n["operator_confirms"],
                                 "auto_links_confirmed": conf_ok, "auto_links_rejected": rej,
                                 "auto_link_precision_from_feedback": _r(conf_ok / (conf_ok + rej)) if conf_ok + rej else None},
                    "decision_latency_s": {"median": _r(float(np.median(lat)), 2) if lat else None,
                                           "p90": _r(float(np.percentile(lat, 90)), 2) if lat else None},
                    "appearance_model": app, "color_mode": self.cfg["color_mode"], "mode": self.cfg["mode"],
                    "monochrome_cameras": sorted(c for c, v in self.mono_cams.items() if v),
                    "thresholds": {k: self.cfg[k] for k in ("link_threshold", "unlink_threshold", "relink_margin", "top_k")},
                    "retention_s": self.cfg["retention_s"], "embedding_ttl_s": self.cfg["embedding_ttl_s"],
                    "topology_source": getattr(self.gate, "source", ""), "method": METHOD, "note": NOTE}

    # ---- retention / persistence -------------------------------------------------------------------------------
    def purge(self, now: Optional[float] = None) -> int:
        """Drop embeddings of segments ended > embedding_ttl_s ago, and everything (incl. thumbnails) > retention_s."""
        with self.lock:
            now = self._now() if now is None else now
            ttl, ret = float(self.cfg["embedding_ttl_s"]), float(self.cfg["retention_s"])
            gone = []
            for s in list(self.segs.values()):
                if not s.closed:
                    continue
                if (s.crops or s.thumb is not None) and now - s.last_t > ttl:
                    self._flush_thumbs([s])  # a thumbnail offered after the crops expired is still written once
                    s.crops = []
                    s.thumb = None
                if now - s.last_t > ret:
                    gone.append(s)
            for s in gone:
                for k in (s.pred, s.succ):
                    o = self.segs.get(k) if k else None
                    if o is not None and o not in gone:
                        if o.pred == s.id:
                            o.pred = None
                        if o.succ == s.id:
                            o.succ = None
                self.segs.pop(s.id, None)
                self.undecided.discard(s.id)
                if s.thumb_path and self.dir:
                    try:
                        (self.dir.parent / s.thumb_path).unlink(missing_ok=True)
                    except OSError:
                        pass
            if gone:
                live = {s.gid for s in self.segs.values()}
                self.gids = {g: v for g, v in self.gids.items() if g in live}
                self.alias = {k: v for k, v in self.alias.items() if v in live}
                ids = {s.id for s in self.segs.values()}
                self.cannot = {p for p in self.cannot if p[0] in ids or p[1] in ids}
                self.n["purged"] += len(gone)
            return len(gone)

    def _flush_thumbs(self, segs) -> None:
        if not (self.dir and self.cfg["save_crops"]):
            return
        for s in segs:
            if s.thumb_dirty and s.thumb:
                name = s.id.replace(":", "_") + ".jpg"
                p = self.dir / "crops" / name
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(s.thumb)
                    s.thumb_path = f"{self.dir.name}/crops/{name}"
                    s.thumb_dirty = False
                except OSError:
                    pass

    def _state(self) -> dict:
        keep = ("id", "cam", "tid", "first_t", "last_t", "entry", "first_pos", "last_pos", "exit", "closed", "gid",
                "pred", "succ", "link", "alternatives", "decided", "decided_t", "locked", "bbox", "n_obs", "thumb_path")
        segs = []
        for s in self.segs.values():
            if s.blip:
                continue
            d = {k: getattr(s, k) for k in keep}
            d["n_crops"] = len(s.crops)
            segs.append(d)
        return {"version": 1, "saved": time.time(), "next_gid": self.next_gid, "segments": segs,
                "gids": {str(k): v for k, v in self.gids.items()}, "alias": {str(k): v for k, v in self.alias.items()},
                "cannot": sorted(self.cannot), "events": list(self.events)[-500:], "counters": dict(self.n),
                "note": "no embeddings are persisted"}

    def _save_locked(self) -> None:
        if not self.dir:
            return
        self._last_save = self.clock()
        self._flush_thumbs([s for s in self.segs.values() if s.decided or s.closed])
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "state.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self._state()))
        os.replace(tmp, p)

    def save(self) -> None:
        with self.lock:
            self._save_locked()

    def _load(self) -> None:
        if not self.dir or not (self.dir / "state.json").exists():
            return
        try:
            d = json.loads((self.dir / "state.json").read_text())
        except (OSError, ValueError):
            return
        self.next_gid = int(d.get("next_gid", 1))
        for r in d.get("segments", []):
            s = Segment(r["id"], r["cam"], int(r["tid"]), float(r["first_t"]), float(r["last_t"]), r.get("entry"),
                        tuple(r.get("first_pos") or (0, 0)), tuple(r.get("last_pos") or (0, 0)))
            for k in ("exit", "gid", "pred", "succ", "link", "alternatives", "decided", "decided_t", "locked",
                      "n_obs", "thumb_path"):
                setattr(s, k, r.get(k, getattr(s, k)))
            s.bbox = tuple(r.get("bbox") or (0, 0, 0, 0))
            if not r.get("closed"):
                s.exit = s.exit or "open"  # still visible when the service stopped
            s.closed, s.decided = True, True
            self.segs[s.id] = s
        self.gids = {int(k): v for k, v in (d.get("gids") or {}).items()}
        self.alias = {int(k): int(v) for k, v in (d.get("alias") or {}).items()}
        self.cannot = {tuple(p) for p in d.get("cannot", [])}
        self.events.extend(d.get("events") or [])
        self.n.update(d.get("counters") or {})

    def flush(self) -> None:
        """End of data: close every open segment and decide everything still pending."""
        with self.lock:
            for key, sid in list(self.open.items()):
                s = self.segs[sid]
                self._close(s, exit_key="open")
            for sid in sorted(self.undecided, key=lambda k: self.segs[k].first_t if k in self.segs else 0.0):
                s = self.segs.get(sid)
                if s is not None and not s.decided and s.n_obs >= int(self.cfg["min_obs"]):
                    self._decide(s)
            self.undecided.clear()

    def close(self) -> None:
        self.flush()
        try:
            self.save()
        except OSError:
            pass


# -------------------------------------------------------------------------------------------------------------------
# Site adapter
# -------------------------------------------------------------------------------------------------------------------
class GlobalIdentityAnalyzer:
    """Site analyzer: segments + crops -> shared embedding service -> IdentityEngine. Emits no events or alerts."""

    def __init__(self, site, cfg: dict, embedder=None):
        from .reid import EmbeddingService, make_embedder
        self.site = site
        self.cfg = {**DEFAULTS, **(cfg or {})}
        cams = [c["id"] for c in site.cfg.get("cameras", []) if c.get("enabled", True)]
        self.warnings: list[str] = []
        if embedder is None:
            models_dir = Path(getattr(site, "_base", Path.cwd())) / "data" / "models"
            embedder, self.warnings = make_embedder(self.cfg, models_dir)
        self.embedder = embedder
        ho = getattr(site, "handoff", None)
        gate = HandoffGate(ho.engine) if ho is not None else None
        if gate is None:
            self.warnings.append("handoff is not enabled: no learned topology/travel times, a uniform "
                                 f"{self.cfg['min_dt_s']}-{self.cfg['max_dt_s']} s window is used")
        self.engine = IdentityEngine(cams, self.cfg, site.data / "reid", gate, getattr(embedder, "sim", None))
        self.service = EmbeddingService(embedder, self.engine.add_embedding, int(self.cfg["batch_size"]),
                                        float(self.cfg["max_wait_ms"]), int(self.cfg["max_queue"]),
                                        sync=bool(self.cfg["sync"]))
        self._mono_at: dict[str, float] = {}
        self._zones: dict[str, tuple] = {}

    def zones(self, worker) -> list:
        ho = getattr(self.site, "handoff", None)
        if ho is not None:
            return ho.zones(worker)
        size, key = worker.source_size, (worker.id, worker.source_size)
        if self._zones.get(worker.id, (None,))[0] != key:
            polys = []
            if size:
                for z in (self.cfg.get("exit_zones_px") or {}).get(worker.id, []):
                    pts = z.get("polygon_px") if isinstance(z, dict) else z
                    polys.append([(float(x) / size[0], float(y) / size[1]) for x, y in pts])
            self._zones[worker.id] = (key, polys)
        return self._zones[worker.id][1]

    def on_frame(self, worker, t, frame, dets, obs):
        from .reid import extract_crop, is_monochrome
        import cv2
        h, w = frame.shape[:2]
        eng = self.engine
        if t - self._mono_at.get(worker.id, -1e18) >= float(self.cfg["mono_check_s"]):
            self._mono_at[worker.id] = t
            eng.set_mono(worker.id, is_monochrome(frame))
        want = eng.observe(worker.id, t, w, h, dets, self.zones(worker))
        for sid, bb, q in want:
            crop = extract_crop(frame, bb)
            if crop.size == 0:
                continue
            mono = eng.mono_cams.get(worker.id, False) or is_monochrome(crop)
            self.service.submit(sid, crop, {"q": q, "mono": mono, "color": eng.need_color(), "gray": eng.need_gray(mono)})
            if self.cfg["save_crops"]:
                ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    eng.offer_thumb(sid, q, jpg.tobytes())
        return []

    def annotate(self, tracks: list[dict]) -> list[dict]:
        """Add the (estimated) global ID to live track rows: gid, gid_label, gid_conf (None for a new ID)."""
        out = []
        for tr in tracks:
            g = self.engine.gid_of(tr.get("camera"), tr.get("track_id"))
            out.append({**tr, "gid": g[0], "gid_label": f"G-{g[0]}", "gid_conf": g[1]} if g else tr)
        return out

    def status(self) -> dict:
        return {"embedder": self.embedder.info, "service": {**self.service.stats, "queue": self.service.queue_len},
                "warnings": self.warnings}

    def close(self) -> None:
        self.service.close()
        self.engine.close()


def install(site, embedder=None) -> Optional[GlobalIdentityAnalyzer]:
    """Register the global-ID analyzer on a Site when its config has "reid": {"enabled": true, ...}."""
    cfg = site.cfg.get("reid") or {}
    if not cfg.get("enabled"):
        return None
    an = GlobalIdentityAnalyzer(site, cfg, embedder=embedder)
    site.analyzers.append(an)
    site.reid = an
    return an
