"""Pose-based "possible physical altercation" analyzer (model inference, operator-verified).

Pipeline per camera (runs as a `Site.analyzers` plug-in):
  person tracks -> candidate pairs (boxes within `pair_max_dist` mean body-heights) -> YOLO26n-pose keypoints on the
  analytics frame (only while a candidate pair exists) matched to tracks by IoU -> per-track keypoint history ->
  per-pair window of T samples resampled to `model_fps` -> pair features -> small GRU (or the heuristic baseline) ->
  score -> persistence (score >= threshold for >= persist_s) -> `altercation` event.

It uses keypoints and boxes only (no appearance). The event states observable cues (arm speed toward the other
person, distance, how long) and the model score/version. It never labels a person or infers intent: fights,
sports, dancing, play-fighting and hugs can look alike to pose features.

`pair_features`, `resample_track`, `window_candidate` and `heuristic_score` are shared with
tools/train_altercation.py so training and runtime compute exactly the same features.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .models import Event

FEATURE_VERSION = "pairfeat-v1"
N_KP = 17
L_SHO, R_SHO, L_WRI, R_WRI = 5, 6, 9, 10
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO-17 left/right swap
KP_CONF = 0.3
N_FEAT = 68 + 68 + 34 + 4 + 4 + 4


# ---------------------------------------------------------------------------------------------- features
def _centers(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def _heights(b: np.ndarray) -> np.ndarray:
    return np.maximum(b[..., 3] - b[..., 1], 1.0)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: np.clip(z[..., 2] - z[..., 0], 0, None) * np.clip(z[..., 3] - z[..., 1], 0, None)
    return inter / np.maximum(area(a) + area(b) - inter, 1e-6)


def pair_features(kA: np.ndarray, bA: np.ndarray, kB: np.ndarray, bB: np.ndarray, fps: float = 5.0) -> np.ndarray:
    """Features for a pair window. kX: [..., T, 17, 3] keypoints (x, y, conf) in image px; bX: [..., T, 4] xyxy.

    Returns [..., T, N_FEAT] float32: pair-centred, body-height-normalised keypoints of both people (68), keypoint
    velocities in body-heights/s (68), keypoint confidences (34), distance / distance change / box IoU / log height
    ratio (4), wrist extension toward the other person (4) and wrist speeds (4). Scale- and translation-invariant.
    """
    kA, kB = np.asarray(kA, np.float32), np.asarray(kB, np.float32)
    bA, bB = np.asarray(bA, np.float32), np.asarray(bB, np.float32)
    s = ((_heights(bA) + _heights(bB)) / 2).mean(-1)[..., None, None]           # [..., 1, 1]
    cA, cB = _centers(bA), _centers(bB)                                          # [..., T, 2]
    c = (cA + cB) / 2
    vA = (kA[..., 2] >= KP_CONF)[..., None]
    vB = (kB[..., 2] >= KP_CONF)[..., None]
    pA = (kA[..., :2] - c[..., None, :]) / s[..., None] * vA                   # [..., T, 17, 2]
    pB = (kB[..., :2] - c[..., None, :]) / s[..., None] * vB

    def vel(k, v):
        xy = k[..., :2] / s[..., None]
        d = np.zeros_like(xy)
        d[..., 1:, :, :] = (xy[..., 1:, :, :] - xy[..., :-1, :, :]) * fps
        ok = np.zeros_like(v)
        ok[..., 1:, :, :] = v[..., 1:, :, :] & v[..., :-1, :, :]
        return d * ok

    velA, velB = vel(kA, vA), vel(kB, vB)
    T = kA.shape[-3]
    dvec = cB - cA
    dist = np.linalg.norm(dvec, axis=-1) / s[..., 0]                            # [..., T]
    ddist = np.zeros_like(dist)
    ddist[..., 1:] = (dist[..., 1:] - dist[..., :-1]) * fps
    iou = _iou(bA, bB)
    hr = np.log(_heights(bA) / _heights(bB))
    u = dvec / np.maximum(np.linalg.norm(dvec, axis=-1, keepdims=True), 1e-6)   # A -> B unit vector

    def ext(k, v, uu):
        out = []
        for w, sh in ((L_WRI, L_SHO), (R_WRI, R_SHO)):
            arm = (k[..., w, :2] - k[..., sh, :2]) / s
            ok = (v[..., w, 0] & v[..., sh, 0]).astype(np.float32)
            out.append((arm * uu).sum(-1) * ok)
        return out

    extras = [dist, ddist, iou, hr] + ext(kA, vA, u) + ext(kB, vB, -u) + [
        np.linalg.norm(velA[..., L_WRI, :], axis=-1), np.linalg.norm(velA[..., R_WRI, :], axis=-1),
        np.linalg.norm(velB[..., L_WRI, :], axis=-1), np.linalg.norm(velB[..., R_WRI, :], axis=-1)]
    lead = pA.shape[:-2]
    f = np.concatenate([pA.reshape(*lead, 34), pB.reshape(*lead, 34), velA.reshape(*lead, 34),
                        velB.reshape(*lead, 34), kA[..., 2], kB[..., 2], np.stack(extras, -1)], -1)
    assert f.shape[-1] == N_FEAT and f.shape[-2] == T
    return np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def flip_lr(k: np.ndarray, b: np.ndarray, width: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal mirror (augmentation): x -> width - x and left/right keypoints swapped."""
    k = k[..., FLIP_IDX, :].copy()
    k[..., 0] = width - k[..., 0]
    b = b.copy()
    b[..., 0], b[..., 2] = width - b[..., 2], width - b[..., 0]
    return k, b


def cues(kA, bA, kB, bB, fps: float = 5.0) -> dict:
    """Human-readable observable cues for one pair window (single window, no batch dims)."""
    f = pair_features(kA, bA, kB, bB, fps)
    dist, ddist = f[:, 170], f[:, 171]
    extA, extB = f[:, 174:176].max(1), f[:, 176:178].max(1)
    wr = f[:, 178:182]
    return {"min_distance_body_heights": round(float(dist.min()), 2),
            "distance_change_bh_per_s": round(float(ddist.min()), 2),
            "peak_wrist_speed_bh_per_s": round(float(wr.max()), 2),
            "peak_arm_extension_toward_other_bh": round(float(max(extA.max(), extB.max())), 2),
            "both_moving_arms": bool(wr[:, :2].max() > 1.0 and wr[:, 2:].max() > 1.0)}


def heuristic_score(kA, bA, kB, bB, fps: float = 5.0) -> np.ndarray:
    """Baseline without learning: proximity x fastest wrist motion toward the other person.

    Per step: prox = clip(1.5 - distance/body-height, 0, 1); toward = max over the 4 wrists of the wrist velocity
    component toward the other person (body-heights/s, >= 0). Score = max over time of the 3-step moving average of
    prox * toward. Returns [...] scores (unbounded; thresholds are chosen on validation data).
    """
    kA, kB = np.asarray(kA, np.float32), np.asarray(kB, np.float32)
    bA, bB = np.asarray(bA, np.float32), np.asarray(bB, np.float32)
    s = ((_heights(bA) + _heights(bB)) / 2).mean(-1)[..., None]
    cA, cB = _centers(bA), _centers(bB)
    dvec = cB - cA
    dist = np.linalg.norm(dvec, axis=-1) / s
    u = dvec / np.maximum(np.linalg.norm(dvec, axis=-1, keepdims=True), 1e-6)
    prox = np.clip(1.5 - dist, 0, 1)

    def toward(k, uu):
        v = np.zeros(k.shape[:-2] + (2,), np.float32)
        best = np.zeros(k.shape[:-2], np.float32)
        for w in (L_WRI, R_WRI):
            xy = k[..., w, :2] / s[..., None]
            ok = k[..., w, 2] >= KP_CONF
            v = np.zeros_like(xy)
            v[..., 1:, :] = (xy[..., 1:, :] - xy[..., :-1, :]) * fps
            okv = np.zeros_like(ok)
            okv[..., 1:] = ok[..., 1:] & ok[..., :-1]
            best = np.maximum(best, np.clip((v * uu).sum(-1), 0, None) * okv)
        return best

    m = np.maximum(toward(kA, u), toward(kB, -u))
    x = prox * m
    if x.shape[-1] >= 3:
        x = (x[..., 2:] + x[..., 1:-1] + x[..., :-2]) / 3
    return x.max(-1)


def resample_track(times: np.ndarray, boxes: np.ndarray, kps: np.ndarray, sample_t: np.ndarray,
                   max_gap: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate a track's boxes/keypoints at `sample_t`. Returns (boxes [T,4], kps [T,17,3], valid [T]).

    A sample is valid when it lies within the track's time span and the neighbouring observations are at most
    `max_gap` s apart (or it is within max_gap/2 of the nearest observation at the ends). Invalid samples are filled
    with the nearest valid one so features stay finite; callers decide how many invalid samples to accept.
    """
    times = np.asarray(times, np.float64)
    T = len(sample_t)
    ob, ok = np.zeros((T, 4), np.float32), np.zeros((T, N_KP, 3), np.float32)
    valid = np.zeros(T, bool)
    if len(times) == 0:
        return ob, ok, valid
    for i, ts in enumerate(sample_t):
        j = int(np.searchsorted(times, ts))
        if j < len(times) and abs(times[j] - ts) < 1e-6:
            ob[i], ok[i], valid[i] = boxes[j], kps[j], True
        elif 0 < j < len(times) and times[j] - times[j - 1] <= max_gap:
            w = (ts - times[j - 1]) / (times[j] - times[j - 1])
            ob[i] = (1 - w) * boxes[j - 1] + w * boxes[j]
            ok[i, :, :2] = (1 - w) * kps[j - 1, :, :2] + w * kps[j, :, :2]
            ok[i, :, 2] = np.minimum(kps[j - 1, :, 2], kps[j, :, 2])
            valid[i] = True
        else:
            n = min(max(j, 0), len(times) - 1) if j < len(times) else len(times) - 1
            if j > 0 and (j == len(times) or abs(times[j - 1] - ts) < abs(times[n] - ts)):
                n = j - 1
            if abs(times[n] - ts) <= max_gap / 2:
                ob[i], ok[i], valid[i] = boxes[n], kps[n], True
    if valid.any() and not valid.all():
        idx = np.nonzero(valid)[0]
        for i in np.nonzero(~valid)[0]:
            n = idx[np.argmin(np.abs(idx - i))]
            ob[i], ok[i] = ob[n], ok[n]
            ok[i, :, 2] = 0.0  # filled keypoints count as not observed
    return ob, ok, valid


def window_candidate(bA: np.ndarray, bB: np.ndarray, pair_max_dist: float = 1.5) -> bool:
    """A pair window is scored only if the people come within `pair_max_dist` mean body-heights of each other."""
    s = (_heights(bA) + _heights(bB)).mean() / 2
    d = np.linalg.norm(_centers(bA) - _centers(bB), axis=-1) / s
    return bool(d.min() <= pair_max_dist)


# ---------------------------------------------------------------------------------------------- classifier
def build_gru(n_in: int = N_FEAT, hidden: int = 64):
    import torch
    from torch import nn

    class PairGRU(nn.Module):
        """LayerNorm -> Linear+ReLU -> GRU -> [last hidden, mean over time] -> logit. ~40k parameters."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(n_in)
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2))
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

        def forward(self, x):  # x [B, T, F] -> logits [B]
            h, last = self.gru(self.inp(self.norm(x)))
            return self.head(torch.cat([last[-1], h.mean(1)], -1)).squeeze(-1)

    return PairGRU()


class GRUScorer:
    """Loads a checkpoint written by tools/train_altercation.py: {"state_dict", "meta"}. Scores = sigmoid(logit)."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=True)  # a copied checkpoint must not be able to run code on load
        self.meta = dict(ck.get("meta", {}))
        if self.meta.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
            raise ValueError(f"checkpoint features {self.meta.get('feature_version')} != {FEATURE_VERSION}")
        self.model = build_gru(int(self.meta.get("n_feat", N_FEAT)), int(self.meta.get("hidden", 64)))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.name = self.meta.get("name", Path(path).stem)
        self.version = self.meta.get("version", "unknown")
        self._torch = torch

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(np.ascontiguousarray(feats, np.float32)).to(self.device)
            return self._torch.sigmoid(self.model(x)).cpu().numpy()


# ---------------------------------------------------------------------------------------------- pose model
class UltralyticsPose:
    """YOLO26n-pose on the analytics frame: returns [(box xyxy, kps [17,3])] for detected people."""

    def __init__(self, weights: str = "yolo26n-pose.pt", device: str = "auto", imgsz: int = 640, conf: float = 0.25,
                 models_dir: Optional[Path] = None):
        import os
        from ultralytics import YOLO
        from .detect import resolve_device
        self.device = resolve_device(device)
        w = Path(weights)
        if not w.exists() and models_dir and (Path(models_dir) / w.name).exists():
            w = Path(models_dir) / w.name
        self.model = YOLO(str(w))
        self.kw = dict(imgsz=imgsz, conf=conf, classes=[0], device=self.device, verbose=False,
                       half=self.device.startswith("cuda"))
        self.name = os.path.basename(str(w))
        self.lock = threading.Lock()

    def __call__(self, frame) -> list[tuple[np.ndarray, np.ndarray]]:
        with self.lock:  # one model shared by all camera threads
            r = self.model.predict(frame, **self.kw)[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        kd = r.keypoints.data.cpu().numpy()  # [N, 17, 3]
        return [(boxes[i].astype(np.float32), kd[i].astype(np.float32)) for i in range(len(boxes))]


# ---------------------------------------------------------------------------------------------- runtime analyzer
@dataclass
class _PairEp:
    first: float
    last: float
    scores: list = field(default_factory=list)
    fired: bool = False
    cues: dict = field(default_factory=dict)


@dataclass
class _CamState:
    hist: dict = field(default_factory=dict)       # tid -> deque[(t, box, kps)]
    last_pose: float = -1e18
    eps: dict = field(default_factory=dict)        # (a, b) -> _PairEp
    last_fired: dict = field(default_factory=dict)  # (a, b) -> t


class AltercationAnalyzer:
    """See module docstring. `pose` is a callable frame -> [(box, kps[17,3])]; `scorer` maps features
    [N, T, F] -> scores [N] in [0, 1] (GRUScorer), or None to use the heuristic baseline (`heuristic_score`)."""

    RULE = "altercation"

    def __init__(self, pose: Callable, scorer: Optional[Callable] = None, *, threshold: float = 0.6,
                 model_fps: float = 5.0, window: int = 8, persist_s: float = 1.5, grace_s: float = 0.6,
                 pair_max_dist: float = 1.5, min_valid: float = 0.75, rearm_s: float = 30.0,
                 criticality: int = 2, evidence: bool = True, heuristic_scale: float = 1.0,
                 model_info: Optional[dict] = None):
        self.pose, self.scorer = pose, scorer
        self.thr, self.fps, self.T = float(threshold), float(model_fps), int(window)
        self.persist_s, self.grace_s = float(persist_s), float(grace_s)
        self.pair_max_dist, self.min_valid = float(pair_max_dist), float(min_valid)
        self.rearm_s, self.criticality, self.write_crop = float(rearm_s), int(criticality), evidence
        self.heuristic_scale = float(heuristic_scale)
        self.model_info = dict(model_info or {})
        self.state: dict[str, _CamState] = {}
        self.lock = threading.Lock()
        self.stats = {"pose_runs": 0, "windows_scored": 0, "events": 0}

    # ---- helpers ------------------------------------------------------------------------------
    def _mode(self) -> str:
        return "gru" if self.scorer is not None else "heuristic"

    def _score(self, feats_in: list[tuple]) -> np.ndarray:
        if not feats_in:
            return np.zeros(0)
        kA = np.stack([x[0] for x in feats_in]); bA = np.stack([x[1] for x in feats_in])
        kB = np.stack([x[2] for x in feats_in]); bB = np.stack([x[3] for x in feats_in])
        if self.scorer is not None:
            return np.asarray(self.scorer(pair_features(kA, bA, kB, bB, self.fps)), np.float64)
        return np.clip(heuristic_score(kA, bA, kB, bB, self.fps) / self.heuristic_scale, 0, 1)

    @staticmethod
    def _match(dets, poses) -> dict[int, np.ndarray]:
        """Greedy IoU match of pose detections to person tracks (IoU >= 0.3). Returns tid -> kps."""
        if not dets or not poses:
            return {}
        tb = np.array([d[1] for d in dets], np.float32)
        pb = np.array([p[0] for p in poses], np.float32)
        iou = _iou(tb[:, None, :], pb[None, :, :])
        out, used_t, used_p = {}, set(), set()
        for flat in np.argsort(-iou, axis=None):
            i, j = divmod(int(flat), len(poses))
            if iou[i, j] < 0.3:
                break
            if i in used_t or j in used_p:
                continue
            used_t.add(i); used_p.add(j)
            out[dets[i][0]] = poses[j][1]
        return out

    def _candidates(self, dets) -> list[tuple[int, int]]:
        people = [(d[0], np.asarray(d[1], np.float32)) for d in dets if d[2] == "person"]
        out = []
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                (ta, ba), (tb, bb) = people[i], people[j]
                s = (_heights(ba) + _heights(bb)) / 2
                if np.linalg.norm(_centers(ba) - _centers(bb)) / s <= self.pair_max_dist:
                    out.append((min(ta, tb), max(ta, tb)))
        return out

    # ---- main hook ----------------------------------------------------------------------------
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        cam = worker.id
        with self.lock:
            st = self.state.setdefault(cam, _CamState())
        # forget tracks not seen for a while
        for tid in [k for k, h in st.hist.items() if not h or t - h[-1][0] > 3.0]:
            st.hist.pop(tid, None)
        cands = self._candidates(dets)
        live_pairs = set(cands) | {k for k, ep in st.eps.items() if t - ep.last <= self.grace_s}
        if not live_pairs:
            self._expire(st, t)
            return []
        if t - st.last_pose < 1.0 / self.fps - 1e-3:
            return []
        st.last_pose = t
        poses = self.pose(frame)
        self.stats["pose_runs"] += 1
        for tid, kps in self._match(dets, poses).items():
            box = next(np.asarray(d[1], np.float32) for d in dets if d[0] == tid)
            st.hist.setdefault(tid, deque(maxlen=64)).append((t, box, np.asarray(kps, np.float32)))
        # build one window per candidate pair ending now
        sample_t = t - (self.T - 1 - np.arange(self.T)) / self.fps
        win, keys = [], []
        for a, b in sorted(set(cands)):
            ha, hb = st.hist.get(a), st.hist.get(b)
            if not ha or not hb:
                continue
            ra = resample_track(np.array([x[0] for x in ha]), np.stack([x[1] for x in ha]),
                                np.stack([x[2] for x in ha]), sample_t)
            rb = resample_track(np.array([x[0] for x in hb]), np.stack([x[1] for x in hb]),
                                np.stack([x[2] for x in hb]), sample_t)
            if min(ra[2].mean(), rb[2].mean()) < self.min_valid:
                continue
            if not window_candidate(ra[0], rb[0], self.pair_max_dist):
                continue
            win.append((ra[1], ra[0], rb[1], rb[0])); keys.append((a, b))
        scores = self._score(win)
        self.stats["windows_scored"] += len(win)
        events = []
        for (a, b), sc, w in zip(keys, scores, win):
            if sc < self.thr:
                continue
            ep = st.eps.get((a, b))
            if ep is None or t - ep.last > self.grace_s:
                ep = st.eps[(a, b)] = _PairEp(first=t, last=t)
            ep.last = t
            ep.scores.append(round(float(sc), 3))
            ep.cues = cues(w[0], w[1], w[2], w[3], self.fps)
            if not ep.fired and t - ep.first >= self.persist_s:
                ep.fired = True
                if t - st.last_fired.get((a, b), -1e18) < self.rearm_s:
                    continue
                st.last_fired[(a, b)] = t
                events.append(self._event(worker, t, frame, dets, obs, a, b, ep))
        self._expire(st, t)
        self.stats["events"] += len(events)
        return events

    def _expire(self, st: _CamState, t: float) -> None:
        for k in [k for k, ep in st.eps.items() if t - ep.last > self.grace_s]:
            del st.eps[k]

    def _event(self, worker, t, frame, dets, obs, a, b, ep: _PairEp) -> Event:
        boxes = {d[0]: [round(float(v), 1) for v in d[1]] for d in dets if d[0] in (a, b)}
        ga = next((o.ground for o in obs if o.track_id == a and o.ground), None)
        gb = next((o.ground for o in obs if o.track_id == b and o.ground), None)
        c = ep.cues
        dist_txt = f"closest {c['min_distance_body_heights']} body-heights apart"
        dist_m = None
        if ga and gb:
            dist_m = round(float(np.hypot(ga[0] - gb[0], ga[1] - gb[1])), 2)
            dist_txt = f"≈{dist_m} m apart on the ground plane (±{worker.calib_error or '?'} m)"
        motion = []
        if c["peak_wrist_speed_bh_per_s"] >= 1.0:
            motion.append(f"rapid arm motion (peak wrist speed {c['peak_wrist_speed_bh_per_s']} body-heights/s)")
        if c["peak_arm_extension_toward_other_bh"] >= 0.25:
            motion.append("arm extended toward the other person")
        if c["distance_change_bh_per_s"] <= -0.5:
            motion.append("closing distance quickly")
        if c["both_moving_arms"]:
            motion.append("both moving their arms")
        dur = t - ep.first
        name = self.model_info.get("name", self._mode())
        ver = self.model_info.get("version", "")
        obs_txt = (f"Person tracks #{a} and #{b} ({worker.id}): {', '.join(motion) or 'close-range body motion'}, "
                   f"{dist_txt}, for {dur:.1f}s; pose-model score {max(ep.scores):.2f} "
                   f"({name}{' ' + ver if ver else ''}). Model inference from body keypoints only; "
                   f"it does not establish what is happening or why.")
        deriv = {"rule": self.RULE, "method": f"pose keypoints -> {self._mode()} pair classifier",
                 "threshold": {"score": self.thr, "persist_s": self.persist_s, "window": f"{self.T} @ {self.fps} fps",
                               "pair_max_dist_body_heights": self.pair_max_dist},
                 "scores": ep.scores[-12:], "peak_score": max(ep.scores), "duration_s": round(dur, 2),
                 "cues": c, "distance_m": dist_m, "boxes_xyxy_analytics_px": boxes,
                 "model": {**self.model_info, "mode": self._mode(), "features": FEATURE_VERSION},
                 "learned_model": self.scorer is not None,
                 "confidence_method": "peak window score of the pose pair model (uncalibrated probability)"
                                      if self.scorer is not None else "heuristic proximity x wrist speed, scaled",
                 "calib_error_m": worker.calib_error,
                 "note": "Model inference, not a fact. Sports, play-fighting, dancing and hugs can score high; "
                         "occlusion and crowding can hide real incidents. An operator must review the clip."}
        ground = ((ga[0] + gb[0]) / 2, (ga[1] + gb[1]) / 2) if ga and gb else None
        e = Event(self.RULE, worker.id, "_view", f"{worker.name} (whole view)", [a, b], ep.first, t, obs_txt, deriv,
                  round(float(np.mean(ep.scores)), 3), 3, self.criticality, None, ground)
        ev = {"model": {"detector": f"{name} {ver}".strip() + f" ({self._mode()})",
                        "tracker": f"pose: {getattr(self.pose, 'name', 'pose model')}; person tracks: "
                                   f"{worker.backend.info.get('detector', '?')} + {worker.backend.info.get('tracker', '?')}"}}
        if self.write_crop and frame is not None and boxes:
            from .threats import write_crop
            bb = np.array(list(boxes.values()), np.float32)
            union = (bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max())
            rel = write_crop(worker, frame, union, [tuple(v) for v in boxes.values()], f"{self.RULE}_{a}_{b}", t)
            if rel:
                ev["crop"] = rel
        e.evidence = ev
        return e

    def close(self) -> None:
        self.state.clear()


def load_meta(path: str | Path) -> dict:
    p = Path(path)
    j = p.with_suffix(".json")
    return json.loads(j.read_text()) if j.exists() else {}
