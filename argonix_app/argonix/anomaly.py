"""Learned trajectory-anomaly CORROBORATION: an Isolation Forest over per-track motion features.

AGENTS.md: deterministic rules and statistical baselines are the primary alert layer; learned anomaly detection is
corroboration only. So this analyzer
  * emits only INFO-level `trajectory_anomaly` events (severity 1) with `derivation["learned_model"]` set, which the
    correlator caps at P3 until a deterministic rule agrees;
  * says what was unusual (the features furthest from typical, with typical values), never who or why;
  * stays silent until it has `min_train` completed normal tracks for that camera (cold start).

Model (Liu, Ting & Zhou 2008): a small numpy Isolation Forest (scikit-learn is not a dependency of this app),
`n_trees` trees on sub-samples of `max_samples` rows, anomaly score s = 2^(-E[h(x)] / c(psi)). Following manual §09
BL-14, the score is turned into a percentile against held-out rows (the most recent `calib_fraction` of the training
window, not used to grow trees); an event needs percentile >= `emit_percentile`. The percentile is relative to this
camera's recent traffic and is NOT a probability.

Features per track (one model per camera, fitted on a rolling window of tracks not involved in any fired rule):
speed mean / p90 / std, path straightness, log dwell, heading changes per minute, fraction of time stopped, and the
hour of day (sin/cos) + weekend flag. Speeds are m/s on calibrated cameras and body-heights/s otherwise. Tracks are
resampled at 1 s so detector jitter does not dominate. Refit runs in a background thread; the training window is
persisted under `<data>/anomaly/` and the forest is refitted from it at start-up (no pickles).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import foot_point
from .models import Event
from .rules import register_rule

RULE = "trajectory_anomaly"
register_rule(RULE, "Unusual trajectory (model)", 1)
FEATURES = ("speed_mean", "speed_p90", "speed_std", "straightness", "log_dwell_s", "dir_changes_per_min",
            "stop_fraction", "hour_sin", "hour_cos", "weekend")
MODEL_NAME = "iforest-trajectory"
EULER = 0.5772156649


def c_factor(n: int) -> float:
    """Average path length of an unsuccessful BST search over n points (normalises IF path lengths)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER) - 2.0 * (n - 1) / n


class IsolationForest:
    """Minimal Isolation Forest: random axis-aligned splits, height limit ceil(log2(psi)), vectorised scoring."""

    def __init__(self, n_trees: int = 100, max_samples: int = 256, seed: int = 20260925):
        self.n_trees, self.max_samples, self.seed = int(n_trees), int(max_samples), int(seed)
        self.trees: list[tuple[np.ndarray, ...]] = []
        self.psi = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n < 2:
            raise ValueError("need at least 2 rows")
        rng = np.random.default_rng(self.seed)
        self.psi = min(self.max_samples, n)
        limit = int(math.ceil(math.log2(max(self.psi, 2))))
        self.trees = [self._build(X[rng.choice(n, self.psi, replace=False)], limit, rng) for _ in range(self.n_trees)]
        return self

    @staticmethod
    def _build(X: np.ndarray, limit: int, rng) -> tuple[np.ndarray, ...]:
        feat, thr, left, right, adj = [], [], [], [], []

        def node(idx: np.ndarray, depth: int) -> int:
            k = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); adj.append(0.0)
            sub = X[idx]
            if depth >= limit or len(idx) <= 1:
                adj[k] = depth + c_factor(len(idx))
                return k
            lo, hi = sub.min(axis=0), sub.max(axis=0)
            cand = np.flatnonzero(hi > lo)
            if cand.size == 0:  # all rows identical
                adj[k] = depth + c_factor(len(idx))
                return k
            f = int(rng.choice(cand))
            s = float(rng.uniform(lo[f], hi[f]))
            mask = sub[:, f] < s
            feat[k], thr[k] = f, s
            left[k] = node(idx[mask], depth + 1)
            right[k] = node(idx[~mask], depth + 1)
            return k

        node(np.arange(len(X)), 0)
        return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(adj))

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        rows = np.arange(len(X))
        total = np.zeros(len(X))
        for feat, thr, left, right, adj in self.trees:
            cur = np.zeros(len(X), dtype=np.int64)
            while True:
                f = feat[cur]
                inner = f >= 0
                if not inner.any():
                    break
                ci, fi = cur[inner], f[inner]
                go_left = X[rows[inner], fi] < thr[ci]
                cur[inner] = np.where(go_left, left[ci], right[ci])
            total += adj[cur]
        return total / max(len(self.trees), 1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in (0, 1]; higher = easier to isolate = more unusual."""
        return np.power(2.0, -self.path_lengths(X) / max(c_factor(self.psi), 1e-9))


# ---------------------------------------------------------------------------------------------- features
def track_features(samples: list[tuple[float, float, float, float]], metric: bool,
                   stop_speed: Optional[float] = None) -> Optional[np.ndarray]:
    """samples: [(t, x, y, scale)] with position in metres (metric) or px, and scale = 1 (metric) or box height px.

    Returns the FEATURES vector (without the time-of-day part, see `time_features`) or None if the track is too short.
    """
    if len(samples) < 4:
        return None
    a = np.asarray(samples, dtype=np.float64)
    t, x, y, sc = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    dur = t[-1] - t[0]
    if dur < 3.0:
        return None
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0)
    gx, gy = np.interp(grid, t, x), np.interp(grid, t, y)
    norm = 1.0 if metric else max(float(np.median(sc)), 1.0)
    dx, dy = np.diff(gx) / norm, np.diff(gy) / norm
    sp = np.hypot(dx, dy)  # per 1 s step: m/s or body-heights/s
    if sp.size < 2:
        return None
    path = float(sp.sum())
    net = math.hypot((gx[-1] - gx[0]) / norm, (gy[-1] - gy[0]) / norm)
    straight = net / path if path > 1e-6 else 1.0
    stop = stop_speed if stop_speed is not None else (0.2 if metric else 0.12)
    moving = sp >= stop
    ang = np.arctan2(dy, dx)
    turns = 0
    prev = None
    for i in range(len(sp)):
        if not moving[i]:
            continue
        if prev is not None:
            d = abs((ang[i] - ang[prev] + math.pi) % (2 * math.pi) - math.pi)
            if d > math.radians(60):
                turns += 1
        prev = i
    return np.array([sp.mean(), np.percentile(sp, 90), sp.std(), straight, math.log1p(dur),
                     turns / max(dur / 60.0, 1 / 6), 1.0 - moving.mean()], dtype=np.float64)


def time_features(t: float) -> np.ndarray:
    lt = time.localtime(t)
    hod = lt.tm_hour + lt.tm_min / 60.0
    return np.array([math.sin(2 * math.pi * hod / 24), math.cos(2 * math.pi * hod / 24),
                     1.0 if lt.tm_wday >= 5 else 0.0])


# ---------------------------------------------------------------------------------------------- per-camera model
class _CamModel:
    def __init__(self, cam: str, window: int):
        self.cam = cam
        self.X: deque = deque(maxlen=window)
        self.T: deque = deque(maxlen=window)
        self.forest: Optional[IsolationForest] = None
        self.calib: Optional[np.ndarray] = None
        self.med: Optional[np.ndarray] = None
        self.mad: Optional[np.ndarray] = None
        self.version: Optional[str] = None
        self.fitted_at: Optional[float] = None
        self.n_train = self.n_calib = 0
        self.added_since_fit = 0
        self.fitting = False
        self.metric: Optional[bool] = None
        self.recent: deque = deque(maxlen=20)
        self.emitted = 0
        self.scored = 0


class TrajectoryAnomalyAnalyzer:
    name = "anomaly"

    def __init__(self, data_dir: Optional[Path] = None, min_train: int = 250, window: int = 5000,
                 n_trees: int = 100, max_samples: int = 256, calib_fraction: float = 0.2,
                 emit_percentile: float = 0.99, refit_every: int = 50, refit_s: float = 600.0,
                 min_duration_s: float = 4.0, stale_s: float = 3.0, rescore_s: float = 60.0, criticality: int = 1,
                 seed: int = 20260925, background: bool = True):
        self.dir = Path(data_dir) / "anomaly" if data_dir else None
        self.min_train, self.window = int(min_train), int(window)
        self.n_trees, self.max_samples, self.calib_fraction = int(n_trees), int(max_samples), float(calib_fraction)
        self.emit_percentile, self.refit_every, self.refit_s = float(emit_percentile), int(refit_every), float(refit_s)
        self.min_duration, self.stale, self.rescore_s = float(min_duration_s), float(stale_s), float(rescore_s)
        self.criticality, self.seed, self.background = int(criticality), int(seed), bool(background)
        self.lock = threading.RLock()
        self.models: dict[str, _CamModel] = {}
        self.tracks: dict[str, dict[int, dict]] = {}
        self._threads: list[threading.Thread] = []
        if self.dir:
            self._load_all()

    # ---- persistence ----
    def _path(self, cam: str) -> Optional[Path]:
        return self.dir / f"{cam}.npz" if self.dir else None

    def _load_all(self) -> None:
        if not self.dir or not self.dir.is_dir():
            return
        for p in self.dir.glob("*.npz"):
            try:
                d = np.load(p, allow_pickle=False)
                cam = p.stem
                m = self.models[cam] = _CamModel(cam, self.window)
                for row, t in zip(d["X"], d["T"]):
                    m.X.append(np.asarray(row, dtype=np.float64)); m.T.append(float(t))
                meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
                m.metric = meta.get("metric")
                if len(m.X) >= self.min_train:
                    self._fit(m)
            except Exception:  # noqa: BLE001 - a corrupt window file must not stop the site
                continue

    def _save(self, m: _CamModel) -> None:
        p = self._path(m.cam)
        if p is None or not m.X:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, X=np.array(m.X), T=np.array(m.T),
                 meta=np.array(json.dumps({"features": FEATURES, "metric": m.metric, "version": m.version})))
        tmp.replace(p)

    # ---- fitting ----
    def _fit(self, m: _CamModel) -> None:
        with self.lock:
            X = np.array(m.X, dtype=np.float64)
        n = len(X)
        n_cal = max(50, int(round(self.calib_fraction * n)))
        if n - n_cal < 50:
            return
        train, cal = X[: n - n_cal], X[n - n_cal:]  # rows are in arrival order: calibrate on the most recent
        forest = IsolationForest(self.n_trees, self.max_samples, self.seed).fit(train)
        calib = np.sort(forest.score(cal))
        med = np.median(train, axis=0)
        mad = np.median(np.abs(train - med), axis=0) * 1.4826
        h = hashlib.sha1(X.tobytes() + json.dumps([self.n_trees, self.max_samples, self.seed, FEATURES]).encode())
        with self.lock:
            m.forest, m.calib, m.med, m.mad = forest, calib, med, np.maximum(mad, 1e-3)
            m.version = h.hexdigest()[:12]
            m.fitted_at, m.n_train, m.n_calib, m.added_since_fit = time.time(), len(train), len(cal), 0
            m.fitting = False
        self._save(m)

    def _maybe_refit(self, m: _CamModel) -> None:
        if m.fitting or len(m.X) < self.min_train:
            return
        due = m.forest is None or m.added_since_fit >= self.refit_every or (
            m.fitted_at is not None and time.time() - m.fitted_at >= self.refit_s and m.added_since_fit > 0)
        if not due:
            return
        m.fitting = True
        if self.background:
            th = threading.Thread(target=self._fit_safe, args=(m,), daemon=True, name=f"iforest-{m.cam}")
            self._threads.append(th)
            th.start()
        else:
            self._fit_safe(m)

    def _fit_safe(self, m: _CamModel) -> None:
        try:
            self._fit(m)
        except Exception:  # noqa: BLE001
            m.fitting = False

    def wait_idle(self, timeout: float = 30.0) -> None:
        for th in list(self._threads):
            th.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    def percentile(self, m: _CamModel, s: float) -> float:
        n = len(m.calib)
        return float((np.searchsorted(m.calib, s, side="left") + 0.5) / (n + 1))

    # ---- runtime ----
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        cam = worker.id
        metric = getattr(worker, "H", None) is not None
        scale = float(getattr(worker, "scale", 1.0) or 1.0)
        ground = {o.track_id: o.ground for o in obs if o.cls == "person"}
        flagged = {k[2] for k, ep in getattr(getattr(worker, "engine", None), "eps", {}).items()
                   if len(k) == 3 and getattr(ep, "fired", False)}
        events: list[Event] = []
        with self.lock:
            m = self.models.get(cam)
            if m is None:
                m = self.models[cam] = _CamModel(cam, self.window)
            if m.metric is not None and m.metric != metric:  # calibration changed: old features have other units
                m.X.clear(); m.T.clear(); m.forest = None
            m.metric = metric
            tr = self.tracks.setdefault(cam, {})
            for tid, bb, cls, conf in dets:
                if cls != "person":
                    continue
                if metric:
                    g = ground.get(tid)
                    if g is None:
                        continue
                    s = (t, g[0], g[1], 1.0)
                else:
                    fx, fy = foot_point(bb)
                    s = (t, fx * scale, fy * scale, max(1.0, (bb[3] - bb[1]) * scale))
                rec = tr.setdefault(tid, {"samples": deque(maxlen=900), "first": t, "last": t, "scored_at": None,
                                          "emitted": False, "flagged": False, "conf": []})
                rec["samples"].append(s)
                rec["last"] = t
                rec["conf"] = (rec["conf"] + [conf])[-20:]
            for tid in flagged:
                if tid in tr:
                    tr[tid]["flagged"] = True
            for tid, rec in list(tr.items()):
                ended = t - rec["last"] > self.stale
                long_lived = (t - rec["first"] >= self.rescore_s
                              and (rec["scored_at"] is None or t - rec["scored_at"] >= self.rescore_s))
                if not ended and not long_lived:
                    continue
                ev = self._score_track(worker, m, tid, rec, t, final=ended)
                if ev:
                    events.append(ev)
                if ended:
                    del tr[tid]
            self._maybe_refit(m)
        return events

    def _score_track(self, worker, m: _CamModel, tid: int, rec: dict, t: float, final: bool) -> Optional[Event]:
        rec["scored_at"] = t
        samples = list(rec["samples"])
        if not samples or samples[-1][0] - samples[0][0] < self.min_duration:
            return None
        base = track_features(samples, bool(m.metric))
        if base is None:
            return None
        x = np.concatenate([base, time_features(samples[0][0])])
        if final and not rec["flagged"]:  # completed, normal-looking traffic joins the rolling training window
            m.X.append(x); m.T.append(samples[-1][0]); m.added_since_fit += 1
        if m.forest is None or rec["emitted"]:
            return None
        s = float(m.forest.score(x[None, :])[0])
        pct = self.percentile(m, s)
        z = (x - m.med) / m.mad
        top = sorted(range(len(FEATURES)), key=lambda i: -abs(z[i]))[:3]
        contrib = [{"feature": FEATURES[i], "value": round(float(x[i]), 3), "typical": round(float(m.med[i]), 3),
                    "robust_z": round(float(z[i]), 2)} for i in top]
        m.scored += 1
        m.recent.append({"track_id": tid, "t": round(t, 1), "percentile": round(pct, 4), "score": round(s, 4),
                         "top": contrib[:2]})
        if pct < self.emit_percentile:
            return None
        rec["emitted"] = True
        m.emitted += 1
        unit = "m/s" if m.metric else "body-heights/s"
        parts = ", ".join(f"{c['feature']} {c['value']} (typical {c['typical']})" for c in contrib)
        obs = (f"person track #{tid} ({worker.id}) moved unlike this camera's recent tracks: model percentile "
               f"{pct:.3f}; most different: {parts} [{unit}]. Model inference for corroboration only")
        conf = min(0.99, pct)
        deriv = {
            "rule": RULE, "learned_model": {"name": MODEL_NAME, "version": m.version, "features": list(FEATURES),
                                            "n_train": m.n_train, "n_calib": m.n_calib, "fitted_at": m.fitted_at},
            "anomaly_score": round(s, 4), "percentile": round(pct, 4),
            "threshold": {"emit_percentile": self.emit_percentile, "min_train": self.min_train},
            "top_features": contrib, "speed_unit": unit, "track_complete": final,
            "duration_s": round(samples[-1][0] - samples[0][0], 1),
            "confidence_method": "held-out percentile of the Isolation Forest score (relative, not a probability)",
            "calib_error_m": getattr(worker, "calib_error", None),
            "note": "corroboration only: capped at P3 unless a deterministic rule agrees",
        }
        return Event(RULE, worker.id, "_view", f"{getattr(worker, 'name', worker.id)} (whole view)", [tid],
                     samples[0][0], samples[-1][0], obs, deriv, round(conf, 3), 1, self.criticality, None, None)

    # ---- API ----
    def status(self) -> dict:
        with self.lock:
            cams = []
            for cam, m in self.models.items():
                cams.append({"camera": cam, "trained": m.forest is not None, "version": m.version,
                             "fitted_at": m.fitted_at, "n_window": len(m.X), "n_train": m.n_train,
                             "n_calib": m.n_calib, "min_train": self.min_train, "fitting": m.fitting,
                             "speed_unit": None if m.metric is None else ("m/s" if m.metric else "body-heights/s"),
                             "scored": m.scored, "emitted": m.emitted, "recent": list(m.recent)[::-1],
                             "typical": ({f: round(float(v), 3) for f, v in zip(FEATURES, m.med)}
                                         if m.med is not None else None)})
            return {"enabled": True, "model": MODEL_NAME, "features": list(FEATURES),
                    "params": {"n_trees": self.n_trees, "max_samples": self.max_samples,
                               "calib_fraction": self.calib_fraction, "emit_percentile": self.emit_percentile,
                               "refit_every": self.refit_every, "refit_s": self.refit_s, "window": self.window},
                    "note": ("Corroboration only (AGENTS.md): events are INFO level and capped at P3 unless a "
                             "deterministic rule agrees. Percentiles are relative to this camera's recent traffic."),
                    "cameras": cams}

    def close(self) -> None:
        self.wait_idle(10.0)
        with self.lock:
            for m in self.models.values():
                try:
                    self._save(m)
                except Exception:  # noqa: BLE001
                    pass
