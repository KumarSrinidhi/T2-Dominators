"""Person-on-the-ground / possible-fall detection (EXPERIMENTAL, off by default, operator-verified).

What it reports is a POSTURE OBSERVATION, never a diagnosis: "a person's box and body keypoints changed from upright to
horizontal quickly and stayed horizontal near the ground for N seconds". It says nothing about injury, health, cause
or intent. People lying down on purpose, sitting on the floor, crawling children, exercise, occlusion and a camera
looking straight down can all look similar. (The project manual advises against fall detection for a hackathon demo;
it is built here on request, disabled by default and clearly labelled.)

Per person track (a `Site.analyzers` plug-in), from the tracker box and, when a pose model is available, YOLO pose
keypoints (COCO-17) matched to the track by IoU:

  upright sample   box aspect w/h <= upright_aspect and (torso within upright_torso_deg of vertical, if keypoints)
  lying sample     box aspect w/h >= lying_aspect, box height <= lying_height_frac x the person's upright height,
                   and (if keypoints) torso >= torso_deg from vertical with the hips within hip_height_frac upright
                   heights of the box bottom (bending over keeps the hips high; sitting keeps the torso upright)
  rapid descent    the box top dropped >= descent_frac upright heights within descent_window_s before lying began
  event            lying persists >= persist_s (short flicker up to grace_s tolerated) after a rapid descent
                   (or, with report_without_descent, a person lying without an observed descent: "person on ground")

Heuristic events are deterministic rules on observed geometry (`learned_model` unset). An optional classifier (any
callable features[T, F] -> probability, e.g. `TorchFallClassifier`) can corroborate; an event raised by the
classifier ALONE sets `learned_model` so the correlator caps it at P3. No classifier is shipped/trained.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .models import Event
from .rules import register_rule

RULE = "person_down"
register_rule(RULE, "Person on the ground / possible fall", 3)
FEATURE_VERSION = "fallfeat-v1"
KP_CONF = 0.3
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
NOTE = ("Posture observation from the person's box shape and body keypoints (experimental). It does not establish "
        "injury, health or cause: lying down on purpose, sitting on the floor, exercise or occlusion can look "
        "similar. An operator must review the clip.")


@dataclass
class FallParams:
    upright_aspect: float = 0.75
    upright_torso_deg: float = 35.0
    lying_aspect: float = 1.0
    lying_height_frac: float = 0.65
    torso_deg: float = 55.0
    hip_height_frac: float = 0.3
    descent_frac: float = 0.35
    descent_window_s: float = 2.0
    persist_s: float = 3.0
    grace_s: float = 1.0
    recover_s: float = 1.0
    min_upright_samples: int = 3
    report_without_descent: bool = False
    rearm_s: float = 60.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "FallParams":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (cfg or {}).items() if k in names})


def torso_angle(kps: Optional[np.ndarray]) -> Optional[float]:
    """Angle (degrees) between the hip-mid -> shoulder-mid vector and image-up; None if keypoints are unreliable."""
    if kps is None:
        return None
    k = np.asarray(kps, np.float64)
    if k.shape[0] <= R_HIP or min(k[L_SHO, 2], k[R_SHO, 2], k[L_HIP, 2], k[R_HIP, 2]) < KP_CONF:
        return None
    sh = (k[L_SHO, :2] + k[R_SHO, :2]) / 2
    hp = (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    v = sh - hp
    n = float(np.hypot(*v))
    if n < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -v[1] / n))))


def hip_y(kps: Optional[np.ndarray]) -> Optional[float]:
    if kps is None:
        return None
    k = np.asarray(kps, np.float64)
    if k.shape[0] <= R_HIP or min(k[L_HIP, 2], k[R_HIP, 2]) < KP_CONF:
        return None
    return float((k[L_HIP, 1] + k[R_HIP, 1]) / 2)


@dataclass
class Sample:
    t: float
    box: tuple
    torso: Optional[float] = None
    hip_y: Optional[float] = None
    truncated: bool = False  # box touches the bottom/side image edge: shape is not the person's posture

    @property
    def w(self) -> float:
        return max(1e-6, self.box[2] - self.box[0])

    @property
    def h(self) -> float:
        return max(1e-6, self.box[3] - self.box[1])

    @property
    def aspect(self) -> float:
        return self.w / self.h


@dataclass
class FallTrack:
    """Per-track state machine: "unknown" -> "upright" <-> "down" -> fires once per down episode."""
    p: FallParams
    hist: deque = field(default_factory=lambda: deque(maxlen=80))
    upright_h: deque = field(default_factory=lambda: deque(maxlen=25))
    state: str = "unknown"
    down_since: Optional[float] = None
    last_lying: Optional[float] = None
    up_since: Optional[float] = None
    descent: Optional[dict] = None
    fired: bool = False
    clf_scores: deque = field(default_factory=lambda: deque(maxlen=40))
    clf_since: Optional[float] = None

    def ref_height(self) -> Optional[float]:
        return float(np.median(self.upright_h)) if len(self.upright_h) >= self.p.min_upright_samples else None

    def is_upright(self, s: Sample) -> bool:
        if s.aspect > self.p.upright_aspect:
            return False
        return s.torso is None or s.torso <= self.p.upright_torso_deg

    def is_lying(self, s: Sample) -> bool:
        p, H = self.p, self.ref_height()
        if s.truncated or s.aspect < p.lying_aspect:
            return False
        if H is not None and s.h > p.lying_height_frac * H:
            return False
        if s.torso is not None:
            if s.torso < p.torso_deg:
                return False  # torso upright (e.g. sitting on the floor with legs out)
            if H is not None and s.hip_y is not None and (s.box[3] - s.hip_y) > p.hip_height_frac * H:
                return False  # hips high above the ground (e.g. bending over)
        return True

    def _find_descent(self, t_on: float, s_on: Sample) -> Optional[dict]:
        H = self.ref_height()
        if H is None:
            return None
        best = None
        for u in self.hist:
            if u.t >= t_on or t_on - u.t > self.p.descent_window_s or not self.is_upright(u):
                continue
            drop = (s_on.box[1] - u.box[1]) / H
            if drop >= self.p.descent_frac:
                best = {"drop_upright_heights": round(drop, 2), "duration_s": round(t_on - u.t, 2),
                        "from_t": u.t}  # keep the latest qualifying upright sample (shortest transition)
        return best

    def update(self, s: Sample) -> Optional[dict]:
        """Add a sample; returns an episode dict the first time the event condition holds, else None."""
        p = self.p
        self.hist.append(s)
        if self.is_upright(s):
            self.upright_h.append(s.h)
            if self.state == "down":
                self.up_since = self.up_since if self.up_since is not None else s.t
                if s.t - self.up_since >= p.recover_s:
                    self._reset("upright")
            else:
                self.state = "upright"
            return None
        self.up_since = None  # recovery needs consecutive upright samples
        if self.is_lying(s):
            if self.state != "down":
                self.state, self.down_since, self.fired = "down", s.t, False
                self.descent = self._find_descent(s.t, s)
            self.last_lying = s.t
            dur = s.t - self.down_since
            if not self.fired and dur >= p.persist_s and (self.descent or p.report_without_descent):
                self.fired = True
                return {"t_start": self.down_since, "t": s.t, "lying_s": round(dur, 2), "descent": self.descent,
                        "aspect": round(s.aspect, 2), "torso_deg": None if s.torso is None else round(s.torso, 1),
                        "ref_height_px": None if self.ref_height() is None else round(self.ref_height(), 1),
                        "height_frac": None if self.ref_height() is None else round(s.h / self.ref_height(), 2)}
            return None
        if self.state == "down" and self.last_lying is not None and s.t - self.last_lying > p.grace_s:
            self._reset("unknown")
        return None

    def _reset(self, state: str) -> None:
        self.state, self.down_since, self.last_lying, self.up_since = state, None, None, None
        self.descent, self.fired = None, False


def track_features(hist: list[Sample], ref_h: Optional[float], t_now: float, window_s: float = 4.0,
                   fps: float = 5.0) -> np.ndarray:
    """Classifier input for one track: [T, 5] = aspect, height/ref, top drop/ref, torso/90 (-1 if unknown), has_pose,
    sampled at `fps` over the last `window_s` (nearest earlier sample). Shared by runtime and any training script."""
    T = int(round(window_s * fps))
    H = ref_h or max(s.h for s in hist)
    top0 = min(s.box[1] for s in hist)
    out = np.zeros((T, 5), np.float32)
    ts = [s.t for s in hist]
    for i in range(T):
        tt = t_now - (T - 1 - i) / fps
        j = max(0, int(np.searchsorted(ts, tt, side="right")) - 1)
        s = hist[j]
        out[i] = [s.aspect, s.h / H, (s.box[1] - top0) / H, -1.0 if s.torso is None else s.torso / 90.0,
                  0.0 if s.torso is None else 1.0]
    return out


class TorchFallClassifier:
    """Optional learned corroboration: loads {"state_dict", "meta"} (a small GRU over `track_features`)."""

    def __init__(self, path: str | Path):
        import torch
        from torch import nn
        ck = torch.load(str(path), map_location="cpu", weights_only=True)  # a copied checkpoint must not be able to run code on load
        self.meta = dict(ck.get("meta", {}))
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"fall classifier features {self.meta.get('feature_version')} != {FEATURE_VERSION}")
        hid = int(self.meta.get("hidden", 32))

        class FallGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(5, hid, batch_first=True)
                self.head = nn.Linear(hid, 1)

            def forward(self, x):
                _, h = self.gru(x)
                return self.head(h[-1]).squeeze(-1)

        self.model = FallGRU()
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.name, self.version = self.meta.get("name", Path(path).stem), self.meta.get("version", "unknown")
        self._torch = torch

    def __call__(self, feats: np.ndarray) -> float:
        with self._torch.no_grad():
            x = self._torch.from_numpy(np.ascontiguousarray(feats, np.float32))[None]
            return float(self._torch.sigmoid(self.model(x))[0])


class PoseCache:
    """Wraps a pose callable so two analyzers asking for the same frame run the model once."""

    def __init__(self, pose: Callable, size: int = 8):
        self.pose, self.size = pose, size
        self.name = getattr(pose, "name", "pose model")
        self._c: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.runs = 0

    def __call__(self, frame):
        key = id(frame)
        with self.lock:
            hit = self._c.get(key)
            if hit is not None and hit[0] is frame:
                return hit[1]
        res = self.pose(frame)
        with self.lock:
            self.runs += 1
            self._c[key] = (frame, res)  # holding the frame keeps its id unique while cached
            while len(self._c) > self.size:
                self._c.popitem(last=False)
        return res


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_poses(dets, poses) -> dict[int, np.ndarray]:
    """Greedy IoU (>= 0.3) match of pose detections [(box, kps)] to person tracks. Returns tid -> kps."""
    pairs = sorted(((_iou(d[1], p[0]), i, j) for i, d in enumerate(dets) for j, p in enumerate(poses)), reverse=True)
    out, ut, up = {}, set(), set()
    for iou, i, j in pairs:
        if iou < 0.3:
            break
        if i in ut or j in up:
            continue
        ut.add(i); up.add(j)
        out[dets[i][0]] = np.asarray(poses[j][1], np.float32)
    return out


@dataclass
class _Cam:
    tracks: dict = field(default_factory=dict)
    last_seen: dict = field(default_factory=dict)
    last_pose: float = -1e18
    last_fired: dict = field(default_factory=dict)


class FallAnalyzer:
    """See module docstring. `pose`: callable frame -> [(box xyxy, kps [17, 3])] or None (boxes-only mode)."""

    def __init__(self, pose: Optional[Callable] = None, classifier: Optional[Callable] = None, *,
                 params: Optional[FallParams] = None, pose_fps: float = 5.0, pose_aspect: float = 0.6,
                 classifier_threshold: float = 0.8, criticality: int = 2, stale_s: float = 5.0,
                 model_info: Optional[dict] = None):
        self.pose, self.clf = pose, classifier
        self.p = params or FallParams()
        self.pose_period = 1.0 / max(0.1, float(pose_fps))
        self.pose_aspect = float(pose_aspect)
        self.clf_thr = float(classifier_threshold)
        self.criticality, self.stale_s = int(criticality), float(stale_s)
        self.model_info = dict(model_info or {})
        self.state: dict[str, _Cam] = {}
        self.lock = threading.Lock()
        self.stats = {"pose_runs": 0, "samples": 0, "events": 0, "tracks_down": 0}

    def mode(self) -> str:
        return ("pose keypoints + box" if self.pose is not None else "boxes only (no pose model)") + \
               (" + classifier" if self.clf is not None else "")

    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        with self.lock:
            st = self.state.setdefault(worker.id, _Cam())
        people = [d for d in dets if d[2] == "person"]
        for tid in [k for k, ts in st.last_seen.items() if t - ts > self.stale_s]:
            st.tracks.pop(tid, None); st.last_seen.pop(tid, None)
        kps: dict[int, np.ndarray] = {}
        # pose only when someone looks non-upright (wide box) or is already being watched as "down" (saves compute)
        need = any((d[1][2] - d[1][0]) / max(1e-6, d[1][3] - d[1][1]) >= self.pose_aspect
                   or getattr(st.tracks.get(d[0]), "state", "") == "down" for d in people)
        if self.pose is not None and people and need and t - st.last_pose >= self.pose_period - 1e-3:
            st.last_pose = t
            try:
                kps = match_poses(people, self.pose(frame) or [])
                self.stats["pose_runs"] += 1
            except Exception as e:  # noqa: BLE001 - fall back to boxes for this frame
                self.stats["pose_error"] = repr(e)[:200]
        events = []
        fh, fw = (frame.shape[:2] if frame is not None and hasattr(frame, "shape") else (None, None))
        for tid, bb, cls, conf in people:
            k = kps.get(tid)
            trunc = fh is not None and (bb[3] >= fh - 3 or bb[0] <= 2 or bb[2] >= fw - 3)
            s = Sample(t, tuple(float(v) for v in bb), torso_angle(k), hip_y(k), trunc)
            tr = st.tracks.get(tid)
            if tr is None:
                tr = st.tracks[tid] = FallTrack(self.p)
            st.last_seen[tid] = t
            ep = tr.update(s)
            self.stats["samples"] += 1
            learned_only = False
            score = None
            if self.clf is not None and len(tr.hist) >= 3:
                try:
                    score = float(self.clf(track_features(list(tr.hist), tr.ref_height(), t)))
                    tr.clf_scores.append(score)
                    if score >= self.clf_thr:
                        tr.clf_since = tr.clf_since if tr.clf_since is not None else t
                    else:
                        tr.clf_since = None
                except Exception as e:  # noqa: BLE001
                    self.stats["classifier_error"] = repr(e)[:200]
            if ep is None and tr.clf_since is not None and t - tr.clf_since >= self.p.persist_s and not tr.fired:
                tr.fired = True
                learned_only = True
                ep = {"t_start": tr.clf_since, "t": t, "lying_s": round(t - tr.clf_since, 2), "descent": None,
                      "aspect": round(s.aspect, 2), "torso_deg": None if s.torso is None else round(s.torso, 1),
                      "ref_height_px": tr.ref_height(), "height_frac": None}
            if ep is None:
                continue
            if t - st.last_fired.get(tid, -1e18) < self.p.rearm_s:
                continue
            st.last_fired[tid] = t
            events.append(self._event(worker, t, frame, dets, obs, tid, bb, ep, score, learned_only))
        self.stats["tracks_down"] = sum(1 for c in self.state.values() for tr in c.tracks.values() if tr.state == "down")
        self.stats["events"] += len(events)
        return events

    def _event(self, worker, t, frame, dets, obs, tid, bb, ep: dict, score, learned_only: bool) -> Event:
        d = ep.get("descent")
        if learned_only:
            what = (f"a learned posture classifier scored the track as on the ground for {ep['lying_s']:.1f}s "
                    f"(score {score:.2f}); the geometric rule did not confirm it")
        elif d:
            what = (f"posture changed from upright to horizontal within {d['duration_s']:.1f}s (box top dropped "
                    f"{d['drop_upright_heights']:.2f} standing heights) and has stayed horizontal near the ground for "
                    f"{ep['lying_s']:.1f}s")
        else:
            what = f"horizontal posture near the ground for {ep['lying_s']:.1f}s (no rapid descent was observed)"
        extra = []
        if ep.get("torso_deg") is not None:
            extra.append(f"torso {ep['torso_deg']:.0f}° from vertical")
        extra.append(f"box width/height {ep['aspect']:.2f}")
        obs_txt = f"Person track #{tid} ({worker.id}): {what}; {', '.join(extra)}. {NOTE}"
        g = next((o.ground for o in obs if o.track_id == tid and o.ground), None)
        deriv: dict[str, Any] = {
            "rule": RULE, "method": f"posture state machine ({self.mode()})",
            "threshold": {k: getattr(self.p, k) for k in ("lying_aspect", "lying_height_frac", "torso_deg",
                                                          "hip_height_frac", "descent_frac", "descent_window_s",
                                                          "persist_s")},
            "episode": {k: v for k, v in ep.items() if k != "t"}, "classifier_score": score,
            "learned_model": (self.model_info.get("classifier") or "fall-classifier") if learned_only else None,
            "confidence_method": "classifier score (uncalibrated)" if learned_only
            else "deterministic geometry thresholds; fixed heuristic confidence",
            "pose_model": self.model_info.get("pose"), "box_xyxy_analytics_px": [round(float(v), 1) for v in bb],
            "calib_error_m": getattr(worker, "calib_error", None), "experimental": True, "note": NOTE}
        conf = round(score, 3) if learned_only else (0.7 if d else 0.5)
        if not learned_only and score is not None and score >= self.clf_thr:
            conf = 0.8
        e = Event(RULE, worker.id, "_view", f"{worker.name} (whole view)", [tid], ep["t_start"], t, obs_txt, deriv,
                  conf, 3, self.criticality, None, g)
        ev: dict[str, Any] = {}
        if frame is not None:
            try:
                from .threats import write_crop
                rel = write_crop(worker, frame, tuple(bb), [tuple(bb)], f"{RULE}_{tid}", t)
                if rel:
                    ev["crop"] = rel
            except Exception:  # noqa: BLE001 - evidence crop is optional
                pass
        e.evidence = ev
        return e

    def status(self) -> dict:
        return {"enabled": True, "mode": self.mode(), "params": self.p.__dict__, "stats": dict(self.stats),
                "model": self.model_info, "note": NOTE, "experimental": True}

    def close(self) -> None:
        self.state.clear()


def install(site, pose_factory: Optional[Callable] = None, classifier_factory: Optional[Callable] = None
            ) -> Optional[FallAnalyzer]:
    """Register the analyzer when the site config has "fall": {"enabled": true, ...} (off by default).

        "fall": {"enabled": true, "pose_weights": "yolo26n-pose.pt", "pose_fps": 5, "persist_s": 3.0,
                 "classifier": null, "classifier_threshold": 0.8, "report_without_descent": false}

    Reuses the altercation analyzer's pose model when one is installed (one model, cached per frame). A pose model
    that cannot load leaves the analyzer in boxes-only mode (warning in site.fall_status).
    """
    cfg = site.cfg.get("fall") or {}
    site.fall = None
    site.fall_status = {"enabled": False, "warnings": []}
    if not cfg.get("enabled", False):
        return None
    warns = site.fall_status["warnings"]
    base = Path(getattr(site, "_base", Path.cwd()))
    pose = None
    if cfg.get("pose", True):
        shared = next((a for a in site.analyzers if type(a).__name__ == "AltercationAnalyzer"), None)
        if shared is not None and getattr(shared, "pose", None) is not None:
            if not isinstance(shared.pose, PoseCache):
                shared.pose = PoseCache(shared.pose)
            pose = shared.pose
        else:
            w = cfg.get("pose_weights", "yolo26n-pose.pt")
            try:
                if pose_factory is not None:
                    raw = pose_factory(w)
                else:
                    from .threat_install import _exists_or_name, _resolve, make_pose
                    wp = _resolve(site, w)
                    if not _exists_or_name(wp):
                        raise FileNotFoundError(wp)
                    raw = make_pose(wp, cfg.get("device", "auto"), int(cfg.get("pose_imgsz", 640)), base / "data" / "models")
                pose = PoseCache(raw) if raw is not None else None
            except Exception as e:  # noqa: BLE001 - boxes-only fallback
                warns.append(f"pose model {w} unavailable, running boxes-only: {e!r}"[:300])
    clf, clf_name = None, None
    if cfg.get("classifier"):
        cp = Path(cfg["classifier"]) if Path(cfg["classifier"]).is_absolute() else base / cfg["classifier"]
        try:
            clf = (classifier_factory or TorchFallClassifier)(cp)
            clf_name = f"{getattr(clf, 'name', cp.stem)} {getattr(clf, 'version', '')}".strip()
        except Exception as e:  # noqa: BLE001
            warns.append(f"fall classifier {cfg['classifier']} unavailable, heuristic only: {e!r}"[:300])
    an = FallAnalyzer(pose, clf, params=FallParams.from_cfg(cfg), pose_fps=float(cfg.get("pose_fps", 5.0)),
                      classifier_threshold=float(cfg.get("classifier_threshold", 0.8)),
                      criticality=int(cfg.get("criticality", 2)),
                      model_info={"pose": getattr(pose, "name", None), "classifier": clf_name})
    site.analyzers.append(an)
    site.fall = an
    site.fall_status = {"enabled": True, "mode": an.mode(), "warnings": warns}
    return an
