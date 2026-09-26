"""Next-camera prediction ("handoff"): which camera a person track leaving camera A appears in next, and roughly when.

What it is
  * A per-camera track LIFECYCLE built from successive analyzer calls: an EXIT is a person track that disappears
    for > gap_s after last being seen near an image border (or inside a configured exit zone, e.g. a door); an
    ENTRY is a new person track that first appears near a border / exit zone.
  * A fixed-length FEATURE vector at exit: last K normalised foot positions and velocities, exit side, speed,
    heading, ground velocity (when calibrated), dwell, hour of day and a camera one-hot.
  * HandoffNet: a tiny MLP (in -> 64 -> 32 -> K cameras + "leaves coverage") with a travel-time head
    (log-normal mean / log-std of transit seconds per destination camera). CPU only, a few thousand parameters.
  * SELF-SUPERVISED labels: an exit on A followed by an entry on B inside the allowed travel-time window becomes a
    soft label. Several candidate entries split the weight (Bayesian: prior x travel-time density / arrival rate
    of unrelated entries, vs. "left coverage"); no candidate -> "left coverage". The camera topology (transition
    counts, travel-time distributions) is learned as a by-product.
  * Cold start from a synthetic floor-plan simulator (tools/sim_handoff.py), then periodic fine-tuning on the
    online labels (replay buffer persisted under <site data>/handoff/).

What it is NOT (AGENTS.md)
  * No appearance or identity Re-ID. An association is a topology + travel-time gated guess between anonymous
    tracks; it is probabilistic, revisable and research-grade for non-overlapping cameras.
  * Predictions are labelled estimates with probabilities. Nothing here raises an alert about a person; the
    optional informational `handoff_expected` event type is off by default.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .geometry import point_in_polygon

LEFT = "__left__"                       # class label: "leaves coverage" (no entry on any camera in the window)
SIDES = ("left", "right", "top", "bottom")
K_HIST = 8
N_BASE = 4 * K_HIST + len(SIDES) + 1 + 3 + 3 + 1 + 2  # positions, velocities, side, zone, speed/heading, ground v, dwell, hour
NOTE = ("Estimate from exit motion + learned camera topology/travel times only. No appearance or identity matching; "
        "cross-camera association is probabilistic and revisable.")

DEFAULTS: dict[str, Any] = {
    "gap_s": 1.5,              # a track unseen this long is finished
    "border_margin": 0.08,     # exit: last foot position within this fraction of an image border ...
    "extrapolate_s": 0.6,      # ... or its extrapolated position crosses the border within this time
    "entry_margin": 0.12,      # entry: first foot position within this fraction of a border (or in an exit zone)
    "min_track_s": 0.4,        # ignore tracks shorter than this (detector blips)
    "min_dt_s": 0.5,           # allowed exit -> entry transit window (seconds) before topology is learned
    "max_dt_s": 60.0,
    "horizon_s": 2.5,          # live prediction for tracks expected to reach a border within this time
    "predict_every_s": 0.4,
    "hold_s": 12.0,            # keep an exited track's prediction visible this long (data time)
    "retrain_every": 50,       # new online labels between retrains
    "min_labels": 30,          # online labels needed before training without a synthetic cold start
    "buffer_max": 5000,
    "metrics_window": 200,
    "min_edge_n": 8,           # online samples before an edge's own travel-time distribution is used for gating
    "min_prior_n": 20,         # online exits from a (camera, side) before its own transition prior is used
    "epochs": 200,
    "finetune_epochs": 80,
    "cold_start": True,
    "cold_start_blocking": False,
    "sim_topology": None,      # dict | "demo" | path to a JSON spec (tools/sim_handoff.py)
    "sim_duration_s": 7200,
    "sim_seed": 0,
    "exit_zones_px": {},       # {camera: [polygon_px | {"id", "polygon_px"}]} in source-image pixels
    "emit_events": False,      # informational handoff_expected events (off by default)
    "eta_source": "auto",      # auto: per-edge empirical travel time when the edge has >= min_edge_n samples, else
                               # the NN travel-time head | nn | edge  (the head generalised worse on rendered video)
    "seed": 0,
}


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------------------------------------------
# Track lifecycle + features
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _Track:
    first_t: float
    last_t: float
    hist: deque                    # (t, xn, yn, gx|None, gy|None)
    entry: bool = False
    last_pred_t: float = -1e18


@dataclass
class ExitRec:
    cam: str
    tid: int
    t: float                       # last seen
    side: str                      # nearest border
    key: str                       # "zone" when it left through an exit zone, else side
    in_zone: bool
    x: list[float]                 # base features (no camera one-hot)
    pos: tuple[float, float]
    pred: Optional[dict] = None    # NN (or topology fallback) prediction at exit: {"probs": {cls: p}, "eta": {cam: s}}
    base: Optional[dict] = None    # baseline prediction at exit
    id: int = 0


@dataclass
class EntryRec:
    cam: str
    tid: int
    t: float                       # first seen
    key: str
    pos: tuple[float, float]


@dataclass
class Sample:
    x: list[float]
    cam: str
    key: str
    target: dict[str, float]       # soft label over destination cameras + LEFT (sums to 1)
    dt: dict[str, float]           # transit seconds per destination camera
    t: float = 0.0
    source: str = "online"         # online | sim
    w: float = 1.0


def velocity(hist, window_s: float = 1.0) -> tuple[float, float]:
    """Normalised-image velocity (units/s) over the last ~window_s."""
    if len(hist) < 2:
        return (0.0, 0.0)
    t1, x1, y1 = hist[-1][:3]
    old = None
    for e in hist:
        if t1 - e[0] <= window_s:
            old = e
            break
    if old is None or t1 - old[0] < 0.15:
        old = hist[-2]
    dt = t1 - old[0]
    if dt <= 1e-6:
        return (0.0, 0.0)
    return ((x1 - old[1]) / dt, (y1 - old[2]) / dt)


def ground_velocity(hist, window_s: float = 1.0) -> Optional[tuple[float, float]]:
    pts = [e for e in hist if e[3] is not None and hist[-1][0] - e[0] <= window_s]
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.15:
        return None
    dt = pts[-1][0] - pts[0][0]
    return ((pts[-1][3] - pts[0][3]) / dt, (pts[-1][4] - pts[0][4]) / dt)


def hour_of_day(t: float) -> float:
    lt = time.localtime(t)
    return lt.tm_hour + lt.tm_min / 60.0


def border_distances(x: float, y: float) -> dict[str, float]:
    return {"left": x, "right": 1.0 - x, "top": y, "bottom": 1.0 - y}


def base_features(hist, first_t: float, side: str, in_zone: bool, k: int = K_HIST) -> list[float]:
    """Fixed-length exit feature vector (len N_BASE). hist entries: (t, xn, yn, gx|None, gy|None)."""
    pts = list(hist)[-k:]
    while len(pts) < k:
        pts.insert(0, pts[0])
    pos, vel = [], []
    for i, e in enumerate(pts):
        pos += [e[1], e[2]]
        if i == 0:
            vel += [0.0, 0.0]
        else:
            p = pts[i - 1]
            dt = e[0] - p[0]
            vel += [0.0, 0.0] if dt <= 1e-6 else [_clip((e[1] - p[1]) / dt, -3, 3), _clip((e[2] - p[2]) / dt, -3, 3)]
    vx, vy = velocity(hist)
    sp = math.hypot(vx, vy)
    head = [vy / sp, vx / sp] if sp > 1e-3 else [0.0, 0.0]
    gv = ground_velocity(hist)
    g = [_clip(gv[0], -4, 4), _clip(gv[1], -4, 4), 1.0] if gv else [0.0, 0.0, 0.0]
    hr = hour_of_day(hist[-1][0]) * 2 * math.pi / 24
    return (pos + vel + [1.0 if side == s else 0.0 for s in SIDES] + [1.0 if in_zone else 0.0]
            + [_clip(sp, 0, 3)] + head + g + [math.log1p(max(0.0, hist[-1][0] - first_t))]
            + [math.sin(hr), math.cos(hr)])


class Lifecycle:
    """Per-camera anonymous track lifecycle: turns successive detections into EXIT / ENTRY records."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tracks: dict[str, dict[int, _Track]] = defaultdict(dict)

    def _classify_exit(self, tr: _Track, zones) -> tuple[bool, str, bool]:
        c = self.cfg
        _, x, y = tr.hist[-1][:3]
        vx, vy = velocity(tr.hist)
        in_zone = any(point_in_polygon((x, y), z) for z in zones)
        d = border_distances(x, y)
        de = border_distances(x + vx * c["extrapolate_s"], y + vy * c["extrapolate_s"])
        side = min(SIDES, key=lambda s: min(d[s], de[s]))
        near = d[side] <= c["border_margin"] or de[side] <= 0.0
        return (near or in_zone), side, in_zone

    def time_to_border(self, tr: _Track) -> tuple[float, str]:
        _, x, y = tr.hist[-1][:3]
        vx, vy = velocity(tr.hist)
        best, side = math.inf, "right"
        for s, dist, v in (("left", x, -vx), ("right", 1 - x, vx), ("top", y, -vy), ("bottom", 1 - y, vy)):
            if v > 0.02:
                tb = max(0.0, dist) / v
                if tb < best:
                    best, side = tb, s
        return best, side

    def observe(self, cam: str, t: float, w: float, h: float, dets, grounds: dict, zones) -> tuple[list, list, list]:
        """Returns (entries, exits, active) where active = [(tid, track)] seen on this frame."""
        c = self.cfg
        tracks = self.tracks[cam]
        entries, exits, active = [], [], []
        for tid, bb, cls, conf in dets:
            if cls != "person":
                continue
            x1, y1, x2, y2 = bb
            xn, yn = _clip((x1 + x2) / 2.0 / w, 0.0, 1.0), _clip(y2 / h, 0.0, 1.0)
            g = grounds.get(tid)
            e = (t, xn, yn, None if g is None else float(g[0]), None if g is None else float(g[1]))
            tr = tracks.get(tid)
            if tr is None:
                tr = tracks[tid] = _Track(t, t, deque(maxlen=32))
                in_zone = any(point_in_polygon((xn, yn), z) for z in zones)
                d = border_distances(xn, yn)
                side = min(SIDES, key=lambda s: d[s])
                if in_zone or d[side] <= c["entry_margin"]:
                    tr.entry = True
                    entries.append(EntryRec(cam, tid, t, "zone" if in_zone else side, (xn, yn)))
            elif t <= tr.last_t:
                continue
            tr.hist.append(e)
            tr.last_t = t
            active.append((tid, tr))
        for tid in [k for k, tr in tracks.items() if t - tr.last_t > c["gap_s"]]:
            tr = tracks.pop(tid)
            if len(tr.hist) < 2 or tr.last_t - tr.first_t < c["min_track_s"]:
                continue
            ok, side, in_zone = self._classify_exit(tr, zones)
            if ok:
                x = base_features(tr.hist, tr.first_t, side, in_zone)
                exits.append(ExitRec(cam, tid, tr.last_t, side, "zone" if in_zone else side, in_zone, x,
                                     (tr.hist[-1][1], tr.hist[-1][2])))
        return entries, exits, active

    def flush(self, cam: Optional[str] = None) -> None:
        """Forget open tracks (end of a recording: an open track at the end of data is not an exit)."""
        for c in ([cam] if cam else list(self.tracks)):
            self.tracks[c].clear()


# ---------------------------------------------------------------------------------------------------------------
# Topology (transition counts + travel-time distributions); also the per-exit-side baseline
# ---------------------------------------------------------------------------------------------------------------
def _wquantile(vals: list[tuple[float, float]], q: float) -> Optional[float]:
    vals = sorted((v, w) for v, w in vals if w > 0)
    if not vals:
        return None
    tot = sum(w for _, w in vals)
    acc = 0.0
    for v, w in vals:
        acc += w
        if acc >= q * tot:
            return v
    return vals[-1][0]


class Topology:
    """Weighted transition counts per (camera, exit key) and per camera edge, plus travel-time samples."""

    def __init__(self, max_tt: int = 1000):
        self.side: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.trans: dict[tuple[str, str], float] = defaultdict(float)
        self.tt: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        self.exits: dict[str, float] = defaultdict(float)
        self.max_tt = max_tt
        self._stats: dict = {}

    def add(self, cam: str, key: str, target: dict[str, float], dt: dict[str, float]) -> None:
        for k, w in target.items():
            if w <= 0:
                continue
            self.side[(cam, key)][k] += w
            self.trans[(cam, k)] += w
            if k != LEFT and k in dt and dt[k] > 0:
                lst = self.tt[(cam, k)]
                lst.append((float(dt[k]), float(w)))
                if len(lst) > self.max_tt:
                    del lst[: len(lst) - self.max_tt]
                self._stats.pop((cam, k), None)
        self.exits[cam] += sum(target.values())

    def n_side(self, cam: str, key: str) -> float:
        return sum(self.side.get((cam, key), {}).values())

    def dist(self, cam: str, key: str, classes: list[str], alpha: float = 0.5) -> dict[str, float]:
        cnt = self.side.get((cam, key), {})
        tot = sum(cnt.get(k, 0.0) for k in classes) + alpha * len(classes)
        return {k: (cnt.get(k, 0.0) + alpha) / tot for k in classes}

    def tt_stats(self, cam: str, dest: str) -> Optional[dict]:
        """{n, mu, sigma (log-seconds), p05, p10, p50, p90, p95} for an edge, or None."""
        if (cam, dest) in self._stats:
            return self._stats[(cam, dest)]
        vals = self.tt.get((cam, dest))
        if not vals:
            return None
        ws = np.array([w for _, w in vals])
        lg = np.log(np.maximum([v for v, _ in vals], 0.05))
        n = float(ws.sum())
        mu = float((ws * lg).sum() / n)
        var = float((ws * (lg - mu) ** 2).sum() / n)
        s = {"n": round(n, 2), "mu": mu, "sigma": max(0.12, math.sqrt(var)),
             **{f"p{int(q * 100):02d}": _wquantile(vals, q) for q in (0.05, 0.10, 0.50, 0.90, 0.95)}}
        self._stats[(cam, dest)] = s
        return s

    def predict(self, cam: str, key: str, classes: list[str]) -> dict:
        """Baseline prediction: destination distribution for this (camera, exit side) + median edge travel time."""
        probs = self.dist(cam, key, classes)
        eta = {}
        for k in classes:
            if k != LEFT:
                s = self.tt_stats(cam, k)
                if s:
                    eta[k] = s["p50"]
        return {"probs": probs, "eta": eta}

    def summary(self, cams: list[str]) -> dict:
        out = []
        for (a, b), n in sorted(self.trans.items()):
            if b == LEFT or n < 0.05:
                continue
            s = self.tt_stats(a, b) or {}
            out.append({"from": a, "to": b, "count": round(n, 2),
                        "travel_s": {k: (round(s[k], 2) if s.get(k) is not None else None) for k in ("p10", "p50", "p90")}})
        tot = sum(self.exits.values())
        left = sum(v for (a, b), v in self.trans.items() if b == LEFT)
        per_cam = {c: {"exits": round(self.exits.get(c, 0.0), 2),
                       "left_coverage": round(self.trans.get((c, LEFT), 0.0), 2)} for c in cams}
        return {"cameras": cams, "transitions": out, "left_coverage_rate": round(left / tot, 3) if tot else None,
                "exits_weighted": round(tot, 2), "per_camera": per_cam}

    def to_dict(self) -> dict:
        return {"side": {f"{c}\t{k}": dict(v) for (c, k), v in self.side.items()},
                "trans": {f"{a}\t{b}": v for (a, b), v in self.trans.items()},
                "tt": {f"{a}\t{b}": v for (a, b), v in self.tt.items()}, "exits": dict(self.exits)}

    @classmethod
    def from_dict(cls, d: dict) -> "Topology":
        t = cls()
        for k, v in (d.get("side") or {}).items():
            c, s = k.split("\t")
            t.side[(c, s)].update(v)
        for k, v in (d.get("trans") or {}).items():
            a, b = k.split("\t")
            t.trans[(a, b)] = float(v)
        for k, v in (d.get("tt") or {}).items():
            a, b = k.split("\t")
            t.tt[(a, b)] = [(float(x), float(w)) for x, w in v]
        t.exits.update({k: float(v) for k, v in (d.get("exits") or {}).items()})
        return t


# ---------------------------------------------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _make_net(d_in: int, n_cams: int):
    torch = _torch()
    nn = torch.nn

    class HandoffNet(nn.Module):
        """in -> 64 -> 32 -> (K cameras + leaves-coverage) logits, plus per-camera log-normal travel time."""

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.cls = nn.Linear(32, n_cams + 1)
            self.tt = nn.Linear(32, 2 * max(1, n_cams))

        def forward(self, x):
            z = self.body(x)
            tt = self.tt(z).view(-1, max(1, n_cams), 2)
            return self.cls(z), tt[..., 0] + 2.3, tt[..., 1].clamp(-2.5, 1.5)  # mu ~ log(10 s) at init

    return HandoffNet()


class HandoffModel:
    """HandoffNet + its camera vocabulary, feature standardisation and temperature calibration."""

    def __init__(self, cams: list[str], net=None, mean=None, std=None, temperature: float = 1.0, meta=None):
        self.cams = list(cams)
        self.classes = self.cams + [LEFT]
        self.d_in = N_BASE + len(self.cams)
        self.net = net or _make_net(self.d_in, len(self.cams))
        self.mean = np.zeros(self.d_in, np.float32) if mean is None else np.asarray(mean, np.float32)
        self.std = np.ones(self.d_in, np.float32) if std is None else np.asarray(std, np.float32)
        self.T = float(temperature)
        self.meta = dict(meta or {})

    def encode(self, xs: list[list[float]], cams: list[str]) -> np.ndarray:
        X = np.zeros((len(xs), self.d_in), np.float32)
        for i, (x, c) in enumerate(zip(xs, cams)):
            X[i, :N_BASE] = x
            if c in self.cams:
                X[i, N_BASE + self.cams.index(c)] = 1.0
        return (X - self.mean) / self.std

    def predict(self, xs: list[list[float]], cams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(probs [n, K+1] calibrated, mu [n, K] log-seconds, sigma [n, K])."""
        torch = _torch()
        if not xs:
            k = len(self.cams)
            return np.zeros((0, k + 1)), np.zeros((0, k)), np.zeros((0, k))
        self.net.eval()
        with torch.no_grad():
            logits, mu, ls = self.net(torch.from_numpy(self.encode(xs, cams)))
            p = torch.softmax(logits / self.T, dim=1).numpy()
        return p, mu.numpy()[:, : len(self.cams)], np.exp(ls.numpy())[:, : len(self.cams)]

    def predict_one(self, x: list[float], cam: str) -> dict:
        p, mu, sg = self.predict([x], [cam])
        probs = {k: float(p[0, i]) for i, k in enumerate(self.classes)}
        eta = {c: float(math.exp(mu[0, i])) for i, c in enumerate(self.cams)}
        rng = {c: (float(math.exp(mu[0, i] - 1.2816 * sg[0, i])), float(math.exp(mu[0, i] + 1.2816 * sg[0, i])))
               for i, c in enumerate(self.cams)}
        return {"probs": probs, "eta": eta, "eta_range": rng}

    def save(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"state": self.net.state_dict(), "cams": self.cams, "mean": self.mean.tolist(),
                    "std": self.std.tolist(), "T": self.T, "meta": json.dumps(self.meta)}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "HandoffModel":
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=True)
        m = cls(d["cams"], mean=d["mean"], std=d["std"], temperature=d["T"], meta=json.loads(d["meta"]))
        m.net.load_state_dict(d["state"])
        return m


def _targets(samples: list[Sample], classes: list[str]) -> np.ndarray:
    Y = np.zeros((len(samples), len(classes)), np.float32)
    idx = {k: i for i, k in enumerate(classes)}
    for i, s in enumerate(samples):
        for k, w in s.target.items():
            Y[i, idx.get(k, len(classes) - 1)] += w  # destinations outside the vocabulary count as "left"
        tot = Y[i].sum()
        Y[i] = Y[i] / tot if tot > 0 else np.eye(len(classes), dtype=np.float32)[-1]
    return Y


def train_model(samples: list[Sample], cams: list[str], init: Optional[HandoffModel] = None, epochs: int = 200,
                seed: int = 0, lr: float = 3e-3, smoothing: float = 0.02, meta: Optional[dict] = None) -> HandoffModel:
    """Train (or fine-tune from `init` when its camera vocabulary matches) on soft-labelled samples."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cams = list(cams)
    model = HandoffModel(cams)
    order = rng.permutation(len(samples))
    n_val = int(0.15 * len(samples)) if len(samples) >= 60 else 0
    val = [samples[i] for i in order[:n_val]]
    tr = [samples[i] for i in order[n_val:]]
    X = np.zeros((len(tr), model.d_in), np.float32)
    raw = model.encode([s.x for s in tr], [s.cam for s in tr])  # standardised with zero mean/unit std -> raw
    X[:] = raw
    if init is not None and init.cams == cams:
        model.net.load_state_dict(init.net.state_dict())
        model.mean, model.std = init.mean.copy(), init.std.copy()
    else:
        model.mean = X.mean(0)
        sd = X.std(0)
        model.std = np.where(sd < 1e-4, 1.0, sd).astype(np.float32)
    Xt = torch.from_numpy((X - model.mean) / model.std)
    Y = torch.from_numpy(_targets(tr, model.classes))
    Y = Y * (1 - smoothing) + smoothing / Y.shape[1]
    W = torch.tensor([s.w for s in tr], dtype=torch.float32)
    K = len(cams)
    LT = np.zeros((len(tr), max(1, K)), np.float32)
    MT = np.zeros((len(tr), max(1, K)), np.float32)
    for i, s in enumerate(tr):
        for j, c in enumerate(cams):
            if s.target.get(c, 0) >= 0.3 and s.dt.get(c, 0) > 0:
                LT[i, j], MT[i, j] = math.log(max(0.1, s.dt[c])), s.target[c]
    LT, MT = torch.from_numpy(LT), torch.from_numpy(MT)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=1e-4)
    bs = 512
    model.net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr)) if len(tr) > bs else torch.arange(len(tr))
        for b in range(0, len(tr), bs):
            ix = perm[b:b + bs]
            logits, mu, ls = model.net(Xt[ix])
            ce = -(Y[ix] * torch.log_softmax(logits, 1)).sum(1)
            nll = (MT[ix] * (ls[:, :MT.shape[1]] + 0.5 * ((LT[ix] - mu[:, :MT.shape[1]]) / ls[:, :MT.shape[1]].exp()) ** 2)).sum(1)
            loss = ((ce + 0.5 * nll) * W[ix]).sum() / W[ix].sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # temperature scaling on the held-out split
    model.T = 1.0
    if val:
        p_logits = None
        with torch.no_grad():
            model.net.eval()
            p_logits, _, _ = model.net(torch.from_numpy(model.encode([s.x for s in val], [s.cam for s in val])))
        Yv = torch.from_numpy(_targets(val, model.classes))
        best = (math.inf, 1.0)
        for T in np.exp(np.linspace(math.log(0.5), math.log(4.0), 36)):
            ll = float(-(Yv * torch.log_softmax(p_logits / float(T), 1)).sum(1).mean())
            if ll < best[0]:
                best = (ll, float(T))
        model.T = best[1]
    model.meta = {**(meta or {}), "n_train": len(tr), "n_val": len(val), "temperature": round(model.T, 3),
                  "trained_at": time.time(), "params": int(sum(p.numel() for p in model.net.parameters()))}
    return model


def score(pred: dict, target: dict[str, float], dt: dict[str, float]) -> dict:
    """Soft top-1 accuracy, log-loss and ETA error of one prediction against one (soft) label."""
    probs = pred["probs"]
    top = max(probs, key=probs.get)
    ll = -sum(w * math.log(max(1e-6, probs.get(k, 0.0))) for k, w in target.items() if w > 0)
    dest = max(target, key=target.get)
    eta_err = None
    if dest != LEFT and target[dest] >= 0.5 and dest in dt and dest in (pred.get("eta") or {}):
        eta_err = abs(pred["eta"][dest] - dt[dest])
    return {"top1": float(target.get(top, 0.0)), "logloss": ll, "eta_err": eta_err}


def evaluate(model: Optional[HandoffModel], baseline: Topology, samples: list[Sample]) -> dict:
    """NN vs per-exit-side baseline on labelled samples (held-out)."""
    out = {"n": len(samples)}
    classes = (model.cams if model else sorted({c for s in samples for c in s.target if c != LEFT})) + [LEFT]
    rows = {"nn": [], "baseline": []}
    if model and samples:
        p, mu, _ = model.predict([s.x for s in samples], [s.cam for s in samples])
        for i, s in enumerate(samples):
            pr = {"probs": {k: float(p[i, j]) for j, k in enumerate(model.classes)},
                  "eta": {c: float(math.exp(mu[i, j])) for j, c in enumerate(model.cams)}}
            rows["nn"].append(score(pr, s.target, s.dt))
    for s in samples:
        rows["baseline"].append(score(baseline.predict(s.cam, s.key, classes), s.target, s.dt))
    for k, r in rows.items():
        if not r:
            continue
        etas = [x["eta_err"] for x in r if x["eta_err"] is not None]
        out[k] = {"top1": round(float(np.mean([x["top1"] for x in r])), 4),
                  "logloss": round(float(np.mean([x["logloss"] for x in r])), 4),
                  "eta_mae_s": round(float(np.mean(etas)), 2) if etas else None, "n_eta": len(etas)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Engine: lifecycle -> association -> labels -> topology / model / metrics -> live predictions
# ---------------------------------------------------------------------------------------------------------------
class RollingMetrics:
    def __init__(self, n: int):
        self.rows: deque = deque(maxlen=n)

    def add(self, nn: Optional[dict], base: Optional[dict], ambiguous: bool) -> None:
        self.rows.append({"nn": nn, "baseline": base, "ambiguous": ambiguous})

    def summary(self) -> dict:
        out = {"window": self.rows.maxlen, "n_scored": len(self.rows)}
        for k in ("nn", "baseline"):
            r = [x[k] for x in self.rows if x[k]]
            etas = [x["eta_err"] for x in r if x["eta_err"] is not None]
            out[k] = {"n": len(r), "top1": round(float(np.mean([x["top1"] for x in r])), 4) if r else None,
                      "logloss": round(float(np.mean([x["logloss"] for x in r])), 4) if r else None,
                      "eta_mae_s": round(float(np.mean(etas)), 2) if etas else None, "n_eta": len(etas)}
        return out


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class HandoffEngine:
    """Site-independent core (the simulator and tests drive it directly; HandoffAnalyzer adapts it to a Site)."""

    def __init__(self, cams: list[str], cfg: Optional[dict] = None, data_dir: Optional[Path] = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cams = list(cams)
        self.dir = Path(data_dir) if data_dir else None
        self.lock = threading.RLock()
        self.life = Lifecycle(self.cfg)
        self.wm: dict[str, tuple[float, float]] = {}       # camera -> (latest data t, wall time of that update)
        self.first_t: dict[str, float] = {}
        self.pending: list[ExitRec] = []
        self.recent: deque = deque(maxlen=500)             # resolved exits (for entry-claim counting)
        self.entries: deque = deque(maxlen=5000)
        self.topology = Topology()                          # learned online (exposed)
        self.sim_topology: Optional[Topology] = None        # from the synthetic cold start (association prior only)
        self.baseline = Topology()                          # per-exit-side baseline, trained on the same data as the NN
        self.model: Optional[HandoffModel] = None
        self.buffer: list[Sample] = []                      # online labels (replay buffer)
        self.sim_samples: list[Sample] = []
        self.metrics = RollingMetrics(int(self.cfg["metrics_window"]))
        self.history: deque = deque(maxlen=20000)           # resolved exits (for offline evaluation)
        self.predictions: dict[str, dict] = {}
        self.n_labels = 0
        self.n_exits = self.n_entries = 0
        self.n_ambiguous = 0
        self._since_retrain = 0
        self._ids = 0
        self._training = False
        self._train_thread: Optional[threading.Thread] = None
        self.last_train: dict = {}
        self.cold_start_eval: Optional[dict] = None
        self.version = 0
        self._load()

    # ---- persistence -------------------------------------------------------------------------------------------
    def _p(self, name: str) -> Optional[Path]:
        return self.dir / name if self.dir else None

    def _load(self) -> None:
        if not self.dir:
            return
        try:
            if self._p("model.pt").exists():
                self.model = HandoffModel.load(self._p("model.pt"))
                self.version = int(self.model.meta.get("version", 0))
                self.cold_start_eval = self.model.meta.get("sim_eval")
        except Exception as e:  # a corrupt model must not stop the site: retrain from the buffer instead
            self.last_train = {"ok": False, "reason": f"model load failed: {e!r}"[:200]}
            self.model = None
        try:
            if self._p("buffer.json").exists():
                d = json.loads(self._p("buffer.json").read_text())
                self.buffer = [Sample(**s) for s in d.get("samples", [])]
                self.n_labels = int(d.get("n_labels", len(self.buffer)))
            if self._p("topology.json").exists():
                d = json.loads(self._p("topology.json").read_text())
                self.topology = Topology.from_dict(d.get("online") or {})
                self.baseline = Topology.from_dict(d.get("baseline") or {})
        except Exception as e:
            self.last_train = {"ok": False, "reason": f"state load failed: {e!r}"[:200]}

    def save(self) -> None:
        if not self.dir:
            return
        with self.lock:
            buf = [asdict(s) for s in self.buffer]
            topo = {"online": self.topology.to_dict(), "baseline": self.baseline.to_dict()}
            model = self.model
            n = self.n_labels
        _atomic_json(self._p("buffer.json"), {"n_labels": n, "samples": buf})
        _atomic_json(self._p("topology.json"), topo)
        if model is not None:
            model.save(self._p("model.pt"))

    # ---- cold start --------------------------------------------------------------------------------------------
    def cold_start(self, spec: dict, blocking: bool = False) -> None:
        """Generate synthetic trajectories for `spec`, fit the sim topology/baseline and (if no model is persisted
        yet) train the initial model. Runs in a background thread unless blocking."""
        def job():
            try:
                from_cache = self._load_sim_cache(spec)
                if from_cache is None:
                    sim = _sim_module()
                    tr = sim.simulate_dataset(spec, duration_s=float(self.cfg["sim_duration_s"]),
                                              seed=int(self.cfg["sim_seed"]), cfg=self.cfg)
                    ho = sim.simulate_dataset(spec, duration_s=max(1200.0, float(self.cfg["sim_duration_s"]) / 4),
                                              seed=int(self.cfg["sim_seed"]) + 1000, cfg=self.cfg)
                    self._save_sim_cache(spec, tr, ho)
                else:
                    tr, ho = from_cache
                st = Topology()
                for s in tr:
                    st.add(s.cam, s.key, s.target, s.dt)
                with self.lock:
                    self.sim_samples, self.sim_topology = tr, st
                    for s in tr:
                        self.baseline.add(s.cam, s.key, s.target, s.dt)
                    need_model = self.model is None
                if need_model:
                    m = train_model(tr, self.cams, epochs=int(self.cfg["epochs"]), seed=int(self.cfg["seed"]),
                                    meta={"source": "synthetic", "n_sim": len(tr), "n_online": 0})
                    ev = evaluate(m, st, ho)
                    m.meta["sim_eval"] = ev
                    with self.lock:
                        self.version += 1
                        m.meta["version"] = self.version
                        self.model, self.cold_start_eval = m, ev
                        self.last_train = {"ok": True, "reason": "cold start (synthetic)", "at": time.time(),
                                           "n_sim": len(tr), "n_online": 0}
                    self.save()
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.last_train = {"ok": False, "reason": f"cold start failed: {e!r}"[:300], "at": time.time()}
            finally:
                with self.lock:
                    self._training = False

        with self.lock:
            self._training = True
        if blocking:
            job()
        else:
            self._train_thread = threading.Thread(target=job, daemon=True, name="handoff-coldstart")
            self._train_thread.start()

    def _sim_key(self, spec: dict) -> str:
        import hashlib
        return hashlib.sha1(json.dumps([spec, self.cfg["sim_duration_s"], self.cfg["sim_seed"], self.cams,
                                        {k: self.cfg[k] for k in ("gap_s", "border_margin", "extrapolate_s",
                                                                  "entry_margin", "min_track_s")}],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _load_sim_cache(self, spec: dict):
        p = self._p("sim_samples.json")
        if not p or not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            if d.get("key") != self._sim_key(spec):
                return None
            return [Sample(**s) for s in d["train"]], [Sample(**s) for s in d["holdout"]]
        except Exception:
            return None

    def _save_sim_cache(self, spec: dict, tr, ho) -> None:
        if self.dir:
            _atomic_json(self._p("sim_samples.json"), {"key": self._sim_key(spec), "train": [asdict(s) for s in tr],
                                                        "holdout": [asdict(s) for s in ho]})

    # ---- per-frame ---------------------------------------------------------------------------------------------
    def observe(self, cam: str, t: float, w: float, h: float, dets, grounds: Optional[dict] = None,
                zones=()) -> list[dict]:
        """Feed one analytics frame of one camera. Returns informational handoff records for newly exited tracks
        (the analyzer turns them into events only when emit_events is on)."""
        with self.lock:
            if cam not in self.cams:  # camera added at runtime: a destination class after the next (full) retrain
                self.cams.append(cam)
            self.first_t.setdefault(cam, t)
            prev = self.wm.get(cam)
            self.wm[cam] = (max(t, prev[0]) if prev else t, time.monotonic())
            entries, exits, active = self.life.observe(cam, t, w, h, dets, grounds or {}, zones)
            self.entries.extend(entries)
            self.n_entries += len(entries)
            out = []
            for ex in exits:
                self._ids += 1
                ex.id = self._ids
                ex.pred = self._predict(ex.x, ex.cam, ex.key)
                ex.base = self.baseline.predict(ex.cam, ex.key, self.cams + [LEFT])
                self.pending.append(ex)
                self.n_exits += 1
                if ex.pred:
                    self.predictions[f"{cam}:{ex.tid}"] = self._pred_row(cam, ex.tid, ex.pred, ex.t, "exited", ex.key)
                else:
                    self.predictions.pop(f"{cam}:{ex.tid}", None)
                out.append({"exit": ex, "pred": ex.pred})
            live = {f"{cam}:{tid}" for tid in self.life.tracks[cam]}
            for k in [k for k, v in self.predictions.items() if v["camera"] == cam and v["state"] == "leaving"
                      and k not in live]:
                del self.predictions[k]
            for k in [k for k, v in self.predictions.items() if v["camera"] == cam and v["state"] == "exited"
                      and t - v["t_exit"] > self.cfg["hold_s"]]:
                del self.predictions[k]
            self._live_predictions(cam, t, active, zones)
            self._resolve()
            return out

    def _live_predictions(self, cam: str, t: float, active, zones) -> None:
        todo = []
        for tid, tr in active:
            if len(tr.hist) < 3 or t - tr.first_t < 0.6 or t - tr.last_pred_t < self.cfg["predict_every_s"]:
                continue
            tb, side = self.life.time_to_border(tr)
            x, y = tr.hist[-1][1:3]
            in_zone = any(point_in_polygon((x, y), z) for z in zones)
            if tb > self.cfg["horizon_s"] and not in_zone:
                self.predictions.pop(f"{cam}:{tid}", None)
                continue
            tr.last_pred_t = t
            todo.append((tid, tr, "zone" if in_zone else side, side, in_zone))
        for tid, tr, key, side, in_zone in todo:
            pred = self._predict(base_features(tr.hist, tr.first_t, side, in_zone), cam, key)
            if pred:
                self.predictions[f"{cam}:{tid}"] = self._pred_row(cam, tid, pred, t, "leaving", key)

    def _predict(self, x: list[float], cam: str, key: str) -> Optional[dict]:
        classes = self.cams + [LEFT]
        if self.model is not None:
            p = self.model.predict_one(x, cam)
            p["source"] = "nn"
            p["eta_nn"] = dict(p["eta"])
            if self.cfg["eta_source"] != "nn":
                for c in list(p["eta"]):
                    s = self._edge_tt(cam, c)
                    if s is not None:
                        p["eta"][c] = s["p50"]
                        p["eta_range"][c] = (s["p10"], s["p90"])
                    elif self.cfg["eta_source"] == "edge":
                        p["eta"].pop(c)
                        p["eta_range"].pop(c, None)
            return p
        if self.topology.n_side(cam, key) >= 1 or self.baseline.n_side(cam, key) >= 1:
            topo = self.topology if self.topology.n_side(cam, key) >= self.cfg["min_prior_n"] else self.baseline
            p = topo.predict(cam, key, classes)
            p["source"] = "topology"
            return p
        return None

    def _edge_tt(self, cam: str, dest: str) -> Optional[dict]:
        for topo in (self.topology, self.sim_topology):
            s = topo.tt_stats(cam, dest) if topo else None
            if s and s["n"] >= self.cfg["min_edge_n"]:
                return s
        return None

    def _pred_row(self, cam, tid, pred, t, state, key) -> dict:
        items = sorted(pred["probs"].items(), key=lambda kv: -kv[1])[:3]
        rows = []
        for k, p in items:
            r = {"camera": None if k == LEFT else k, "leaves_coverage": k == LEFT, "prob": round(min(0.99, p), 3),
                 "eta_s": None}
            if k != LEFT and k in (pred.get("eta") or {}):
                r["eta_s"] = round(pred["eta"][k], 1)
                lo_hi = (pred.get("eta_range") or {}).get(k)
                if lo_hi:
                    r["eta_p10_s"], r["eta_p90_s"] = round(lo_hi[0], 1), round(lo_hi[1], 1)
            rows.append(r)
        return {"camera": cam, "track_id": tid, "state": state, "exit_side": key, "predictions": rows,
                "t_exit": t if state == "exited" else None, "updated": t, "source": pred.get("source"),
                "model_version": self.version if pred.get("source") == "nn" else None, "kind": "estimate"}

    # ---- association -------------------------------------------------------------------------------------------
    def _watermark(self) -> Optional[float]:
        now = time.monotonic()
        live = [t for t, wall in self.wm.values() if now - wall < 10.0]
        return min(live) if live else None

    def _gate(self, cam: str, dest: str) -> tuple[float, float]:
        c = self.cfg
        s = self.topology.tt_stats(cam, dest)
        if s is None or s["n"] < c["min_edge_n"]:
            s = self.sim_topology.tt_stats(cam, dest) if self.sim_topology else None
            if s is None or s["n"] < c["min_edge_n"]:
                return c["min_dt_s"], c["max_dt_s"]
        return max(c["min_dt_s"], 0.6 * s["p05"] - 1.0), min(c["max_dt_s"], 1.5 * s["p95"] + 2.0)

    def _density(self, cam: str, dest: str, dt: float) -> float:
        c = self.cfg
        for topo in (self.topology, self.sim_topology):
            s = topo.tt_stats(cam, dest) if topo else None
            if s and s["n"] >= c["min_edge_n"]:
                z = (math.log(max(dt, 0.05)) - s["mu"]) / s["sigma"]
                return math.exp(-0.5 * z * z) / (dt * s["sigma"] * math.sqrt(2 * math.pi))
        return 1.0 / max(1e-3, c["max_dt_s"] - c["min_dt_s"])

    def _prior(self, cam: str, key: str) -> dict[str, float]:
        classes = self.cams + [LEFT]
        if self.topology.n_side(cam, key) >= self.cfg["min_prior_n"] or not self.sim_topology:
            return self.topology.dist(cam, key, classes, alpha=1.0)
        return self.sim_topology.dist(cam, key, classes, alpha=1.0)

    def _entry_rate(self, cam: str, t: float) -> float:
        span = max(30.0, min(2 * self.cfg["max_dt_s"], t - self.first_t.get(cam, t)))
        n = sum(1 for e in self.entries if e.cam == cam and t - span <= e.t <= t)
        return max(n, 1) / span

    def _in_gate(self, ex: ExitRec, en: EntryRec) -> bool:
        if en.cam == ex.cam:
            return False
        lo, hi = self._gate(ex.cam, en.cam)
        return lo <= en.t - ex.t <= hi

    def associate(self, ex: ExitRec) -> tuple[dict[str, float], dict[str, float], int]:
        """Soft label for one exit: ({class: weight}, {camera: transit s}, n_candidates)."""
        cands = [en for en in self.entries if self._in_gate(ex, en)]
        if not cands:
            return {LEFT: 1.0}, {}, 0
        prior = self._prior(ex.cam, ex.key)
        others = list(self.pending) + list(self.recent)
        score: dict[str, float] = defaultdict(float)
        dts: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for en in cands:
            k = max(1, sum(1 for o in others if self._in_gate(o, en)))  # exits competing for this entry
            dt = en.t - ex.t
            s = prior.get(en.cam, 0.0) * self._density(ex.cam, en.cam, dt) / (self._entry_rate(en.cam, en.t) * k)
            score[en.cam] += s
            dts[en.cam].append((dt, s))
        score[LEFT] = prior.get(LEFT, 0.0)
        tot = sum(score.values())
        target = {k: v / tot for k, v in score.items() if v / tot > 1e-4}
        dt = {c: sum(d * w for d, w in v) / max(1e-12, sum(w for _, w in v)) for c, v in dts.items()}
        return target, dt, len(cands)

    def _resolve(self, force: bool = False) -> None:
        wm = self._watermark()
        ready = [e for e in self.pending if force or (wm is not None and wm >= e.t + self.cfg["max_dt_s"])]
        if not ready:
            return
        for ex in sorted(ready, key=lambda e: e.t):
            target, dt, n = self.associate(ex)
            self.pending.remove(ex)
            self.recent.append(ex)
            self._record(ex, target, dt, n)
        if self.entries:
            oldest = min([e.t for e in self.pending] + [wm or self.entries[-1].t]) - 2 * self.cfg["max_dt_s"]
            while self.entries and self.entries[0].t < oldest:
                self.entries.popleft()

    def _record(self, ex: ExitRec, target: dict, dt: dict, n_cands: int) -> None:
        amb = max(target.values()) < 0.8
        s = Sample(ex.x, ex.cam, ex.key, target, dt, ex.t, "online", 1.0)
        nn = score(ex.pred, target, dt) if ex.pred and ex.pred.get("source") == "nn" else None
        base = score(ex.base, target, dt) if ex.base else None
        self.metrics.add(nn, base, amb)
        self.topology.add(ex.cam, ex.key, target, dt)
        self.baseline.add(ex.cam, ex.key, target, dt)
        self.buffer.append(s)
        if len(self.buffer) > self.cfg["buffer_max"]:
            del self.buffer[: len(self.buffer) - int(self.cfg["buffer_max"])]
        self.n_labels += 1
        self.n_ambiguous += int(amb)
        self._since_retrain += 1
        self.history.append({"cam": ex.cam, "tid": ex.tid, "t": ex.t, "key": ex.key, "pos": ex.pos,
                             "pred": ex.pred, "base": ex.base, "target": target, "dt": dt, "n_candidates": n_cands})
        if self._since_retrain >= self.cfg["retrain_every"] and not self._training:
            self.retrain(blocking=False, reason=f"{self._since_retrain} new labels")

    def flush(self) -> None:
        """End of data: forget open tracks and resolve every pending exit with what has been seen."""
        with self.lock:
            self.life.flush()
            self._resolve(force=True)

    # ---- training ----------------------------------------------------------------------------------------------
    def training_set(self) -> tuple[list[Sample], dict]:
        n_on = len(self.buffer)
        sims = []
        if self.sim_samples:
            w_sim = 0.2 + 0.8 * math.exp(-n_on / 300.0)  # synthetic data fades as real labels accumulate
            rng = random.Random(self.n_labels)
            pick = self.sim_samples if len(self.sim_samples) <= 3000 else rng.sample(self.sim_samples, 3000)
            sims = [Sample(**{**asdict(s), "w": w_sim}) for s in pick]
        return list(self.buffer) + sims, {"n_online": n_on, "n_sim": len(sims)}

    def retrain(self, blocking: bool = False, reason: str = "manual") -> dict:
        with self.lock:
            if self._training:
                return {"ok": False, "reason": "training already in progress"}
            data, info = self.training_set()
            if not info["n_sim"] and info["n_online"] < self.cfg["min_labels"]:
                return {"ok": False, "reason": f"need {self.cfg['min_labels']} labels, have {info['n_online']}"}
            if not data:
                return {"ok": False, "reason": "no training data"}
            self._training = True
            self._since_retrain = 0
            init = self.model
            cams = list(self.cams)

        def job():
            try:
                src = ("synthetic+online" if info["n_online"] else "synthetic") if info["n_sim"] else "online"
                ft = init is not None and init.cams == cams
                m = train_model(data, cams, init=init if ft else None, seed=int(self.cfg["seed"]) + self.version,
                                epochs=int(self.cfg["finetune_epochs" if ft else "epochs"]),
                                meta={"source": src, **info, "reason": reason, "fine_tuned": ft,
                                      "sim_eval": self.cold_start_eval})
                with self.lock:
                    self.version += 1
                    m.meta["version"] = self.version
                    self.model = m
                    self.last_train = {"ok": True, "reason": reason, "at": time.time(), **info}
                self.save()
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.last_train = {"ok": False, "reason": f"training failed: {e!r}"[:300], "at": time.time()}
            finally:
                with self.lock:
                    self._training = False

        if blocking:
            job()
        else:
            self._train_thread = threading.Thread(target=job, daemon=True, name="handoff-train")
            self._train_thread.start()
        return {"ok": True, "started": not blocking, "reason": reason, **info}

    def wait_training(self, timeout: float = 300.0) -> None:
        th = self._train_thread
        if th is not None and th.is_alive():
            th.join(timeout)

    # ---- read-outs ---------------------------------------------------------------------------------------------
    def predictions_list(self) -> list[dict]:
        with self.lock:
            return sorted((dict(v) for v in self.predictions.values()), key=lambda r: (r["camera"], r["track_id"]))

    def topology_summary(self) -> dict:
        with self.lock:
            d = self.topology.summary(self.cams)
        d["source"] = "online self-supervised labels (topology + travel-time association, no identity)"
        return d

    def metrics_summary(self) -> dict:
        with self.lock:
            m = self.model
            d = {**self.metrics.summary(), "n_labels": self.n_labels, "n_ambiguous_labels": self.n_ambiguous,
                 "n_exits": self.n_exits, "n_entries": self.n_entries, "pending": len(self.pending),
                 "buffer": len(self.buffer), "training": self._training, "last_train": self.last_train,
                 "retrain_every": self.cfg["retrain_every"],
                 "next_retrain_in": max(0, int(self.cfg["retrain_every"]) - self._since_retrain),
                 "model": None if m is None else {
                     "version": m.meta.get("version", self.version), "trained_at": m.meta.get("trained_at"),
                     "source": m.meta.get("source"), "n_train": m.meta.get("n_train"),
                     "n_online": m.meta.get("n_online"), "n_sim": m.meta.get("n_sim"), "cams": m.cams,
                     "temperature": round(m.T, 3), "params": m.meta.get("params")},
                 "sim_eval": self.cold_start_eval,
                 "label_note": "Scored against self-supervised labels (associations), not ground truth.",
                 "note": NOTE}
        return d

    def close(self) -> None:
        self.wait_training(60)
        self.flush()
        self.wait_training(120)
        try:
            self.save()
        except Exception:
            pass


def _sim_module():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools import sim_handoff
    return sim_handoff


# ---------------------------------------------------------------------------------------------------------------
# Site adapter
# ---------------------------------------------------------------------------------------------------------------
class HandoffAnalyzer:
    """Site analyzer: feeds each camera's person tracks to a HandoffEngine. Emits no alerts by default."""

    def __init__(self, site, cfg: dict):
        self.site = site
        cams = [c["id"] for c in site.cfg.get("cameras", []) if c.get("enabled", True)]
        self.engine = HandoffEngine(cams, cfg, site.data / "handoff")
        self.cfg = self.engine.cfg
        self._zones: dict[str, tuple] = {}
        spec = self._spec(site)
        if spec and self.cfg["cold_start"]:
            if self.engine.model is None or not self.engine.sim_samples:
                self.engine.cold_start(spec, blocking=bool(self.cfg["cold_start_blocking"]))
        if self.cfg["emit_events"]:
            from .rules import register_rule
            register_rule("handoff_expected", "Expected camera handoff (estimate)", 1)

    def _spec(self, site) -> Optional[dict]:
        s = self.cfg.get("sim_topology")
        if not s:
            return None
        sim = _sim_module()
        if isinstance(s, str) and s != "demo":
            p = Path(s)
            s = json.loads((p if p.is_absolute() else site.config_path.resolve().parent / p).read_text())
        return sim.load_spec(s, site.cfg)

    def zones(self, worker) -> list:
        size = worker.source_size
        key = (worker.id, size)
        if self._zones.get(worker.id, (None,))[0] != key:
            polys = []
            if size:
                for z in (self.cfg.get("exit_zones_px") or {}).get(worker.id, []):
                    pts = z.get("polygon_px") if isinstance(z, dict) else z
                    polys.append([(float(x) / size[0], float(y) / size[1]) for x, y in pts])
            self._zones[worker.id] = (key, polys)
        return self._zones[worker.id][1]

    def on_frame(self, worker, t, frame, dets, obs):
        h, w = frame.shape[:2]
        grounds = {o.track_id: o.ground for o in obs if o.ground is not None}
        recs = self.engine.observe(worker.id, t, w, h, dets, grounds, self.zones(worker))
        if not self.cfg["emit_events"] or not recs:
            return []
        from .models import Event
        out = []
        for r in recs:
            ex, pred = r["exit"], r["pred"]
            if not pred:
                continue
            top = max(pred["probs"], key=pred["probs"].get)
            dest = "leaving coverage" if top == LEFT else top
            out.append(Event("handoff_expected", worker.id, "_handoff", f"{worker.name} exit ({ex.key})", [ex.tid],
                             ex.t, ex.t, f"person track #{ex.tid} left the {worker.id} view ({ex.key}); "
                                         f"estimated next: {dest} ({pred['probs'][top]:.0%})",
                             {"rule": "handoff_expected", "predictions": self.engine._pred_row(
                                 worker.id, ex.tid, pred, ex.t, "exited", ex.key)["predictions"],
                              "model_version": self.engine.version, "note": NOTE,
                              "confidence_method": "temperature-scaled softmax (estimate)"},
                             round(min(0.99, pred["probs"][top]), 3), 1, 1, None, None))
        return out

    def close(self) -> None:
        self.engine.close()


def install(site) -> Optional[HandoffAnalyzer]:
    """Register the handoff analyzer on a Site when its config has "handoff": {"enabled": true, ...}."""
    cfg = site.cfg.get("handoff") or {}
    if not cfg.get("enabled"):
        return None
    an = HandoffAnalyzer(site, cfg)
    site.analyzers.append(an)
    site.handoff = an
    return an
