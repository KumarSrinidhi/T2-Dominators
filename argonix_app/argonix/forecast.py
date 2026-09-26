"""Trajectory forecasting (EXPERIMENTAL, research-grade, off by default).

Predicts where each tracked person is likely to be over the next few seconds, from their own recent motion only
(no appearance, no identity, no interaction with other people). Two predictors share one interface:

  * "cv"  - constant-velocity extrapolation of the last observed step (the standard strong baseline);
  * "gru" - a small GRU (~30k parameters, CPU real-time) trained by tools/train_forecast.py on the public ETH/UCY
            pedestrian trajectories. It predicts a RESIDUAL on top of constant velocity in a heading-aligned frame,
            plus a per-step spread (sigma), so it can only bend the CV path, and its uncertainty grows with horizon.

Protocol (the standard ETH/UCY one): positions sampled every DT = 0.4 s, OBS_LEN = 8 observed steps (3.2 s),
PRED_LEN = 12 predicted steps (4.8 s); ADE/FDE in metres on a held-out scene (leave-one-scene-out).

Runtime (`ForecastAnalyzer`, a `Site.analyzers` plug-in):
  person tracks -> per-track position history in the rule engine's space (ground metres on calibrated cameras,
  source-image pixels otherwise) -> smoothed samples every 0.4 s -> forecaster -> per-track path + sigma, served by
  /api/forecast/tracks and drawn as a faint dashed path on Live tiles. Uncalibrated cameras are converted to
  approximate metres with the person's box height (1.7 m assumed body height); that scale is a rough estimate.

Optional "predicted zone entry" event: a person currently OUTSIDE a zone that has the intrusion rule, whose forecast
enters it within `horizon_s` with estimated probability >= `min_prob` on `persist` consecutive updates. It is a
PREDICTION, never an observation: severity 1 and `learned_model` set, so the correlator caps it at P3 unless the
deterministic intrusion rule later fires in the same zone (which then corroborates it).
"""
from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .geometry import apply_h, foot_point, invert_h, point_in_polygon
from .models import Event
from .rules import register_rule

DT = 0.4
OBS_LEN = 8
PRED_LEN = 12
FEATURE_VERSION = "fcst-v1"
BODY_HEIGHT_M = 1.7
RULE = "predicted_zone_entry"
register_rule(RULE, "Predicted zone entry (forecast)", 1)
NOTE = ("Experimental forecast from this person's own recent motion only (no identity, appearance or interaction "
        "with others). A prediction, not an observation; the spread grows with the horizon.")


# ---------------------------------------------------------------------------------------------- geometry helpers
def heading_frames(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample rotation that maps the last observed step onto +x. obs [B, T, 2] -> (R [B, 2, 2], origin [B, 2]).

    World -> local: (p - origin) @ R.T ; local -> world: q @ R + origin. Near-stationary tracks keep identity.
    """
    obs = np.asarray(obs, np.float64)
    v = obs[:, -1] - obs[:, -2]
    n = np.linalg.norm(v, axis=-1)
    c = np.where(n > 1e-3, v[:, 0] / np.maximum(n, 1e-9), 1.0)
    s = np.where(n > 1e-3, v[:, 1] / np.maximum(n, 1e-9), 0.0)
    R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], 1)  # rows: heading, left normal
    return R, obs[:, -1].copy()


def to_local(p: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return np.einsum("btj,bij->bti", np.asarray(p, np.float64) - origin[:, None, :], R)


def to_world(q: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return np.einsum("bti,bij->btj", np.asarray(q, np.float64), R) + origin[:, None, :]


def cv_predict(obs: np.ndarray, pred_len: int = PRED_LEN) -> np.ndarray:
    """Constant velocity from the last observed step. obs [B, T, 2] -> [B, pred_len, 2]."""
    obs = np.asarray(obs, np.float64)
    v = obs[:, -1] - obs[:, -2]
    k = np.arange(1, pred_len + 1)[None, :, None]
    return obs[:, -1:, :] + k * v[:, None, :]


def ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Average / final displacement error (mean over samples), same units as the inputs."""
    d = np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(gt, np.float64), axis=-1)
    return float(d.mean()), float(d[:, -1].mean())


def model_inputs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heading-aligned observed displacements [B, OBS_LEN-1, 2] (metres per step) + the frame (R, origin)."""
    R, o = heading_frames(obs)
    loc = to_local(obs, R, o)
    return np.diff(loc, axis=1).astype(np.float32), R, o


# ---------------------------------------------------------------------------------------------- learned model
def build_model(hidden: int = 64, pred_len: int = PRED_LEN):
    """GRU encoder over heading-aligned displacements -> residual offsets over CV (local frame) + log sigma."""
    import torch
    from torch import nn

    class ForecastGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.pred_len = pred_len
            self.inp = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
            self.gru = nn.GRU(32, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, pred_len * 3))

        def forward(self, disp):  # disp [B, T-1, 2] -> (residual [B, P, 2], log_sigma [B, P])
            _, h = self.gru(self.inp(disp))
            z = self.head(torch.cat([h[-1], disp[:, -1]], -1)).view(-1, self.pred_len, 3)
            return z[..., :2], z[..., 2].clamp(-5, 3)

    return ForecastGRU()


def decode(disp_local: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """CV in the local frame (last displacement repeated) + cumulative residual -> local future positions."""
    v = disp_local[:, -1][:, None, :]
    k = np.arange(1, residual.shape[1] + 1)[None, :, None]
    return k * v + np.cumsum(residual, axis=1)


class Forecaster:
    """Batch predictor. method "gru" needs a checkpoint written by tools/train_forecast.py; "cv" needs nothing.

    predict(obs [B, OBS_LEN, 2] in metres, sampled every DT) -> (mean [B, P, 2], sigma [B, P]) in metres.
    """

    def __init__(self, checkpoint: Optional[str | Path] = None, method: str = "auto"):
        self.meta: dict[str, Any] = {}
        self.model = None
        self.cv_sigma = [round(0.12 * (k + 1), 3) for k in range(PRED_LEN)]  # default spread, overwritten by meta
        if checkpoint and Path(checkpoint).exists() and method in ("auto", "gru"):
            import torch
            ck = torch.load(str(checkpoint), map_location="cpu", weights_only=True)  # a copied checkpoint must not be able to run code on load
            self.meta = dict(ck.get("meta", {}))
            if self.meta.get("feature_version") != FEATURE_VERSION:
                raise ValueError(f"forecast checkpoint features {self.meta.get('feature_version')} != {FEATURE_VERSION}")
            self.model = build_model(int(self.meta.get("hidden", 64)), int(self.meta.get("pred_len", PRED_LEN)))
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.cv_sigma = list(self.meta.get("cv_rmse_per_step", self.cv_sigma))
            self._torch = torch
        elif method == "gru":
            raise FileNotFoundError(f"forecast checkpoint not found: {checkpoint}")
        self.method = "gru" if self.model is not None else "cv"

    @property
    def info(self) -> dict:
        m = self.meta
        return {"method": self.method, "name": m.get("name", "constant-velocity"), "version": m.get("version", "cv-v1"),
                "trained_on": m.get("trained_on"), "params": m.get("params"), "dt_s": DT, "obs_len": OBS_LEN,
                "pred_len": PRED_LEN}

    def predict(self, obs: np.ndarray, method: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, np.float64)
        method = method or self.method
        if len(obs) == 0:
            return np.zeros((0, PRED_LEN, 2)), np.zeros((0, PRED_LEN))
        if method == "gru" and self.model is not None and obs.shape[1] == OBS_LEN:
            disp, R, o = model_inputs(obs)
            with self._torch.no_grad():
                res, ls = self.model(self._torch.from_numpy(disp))
            loc = decode(disp.astype(np.float64), res.numpy().astype(np.float64))
            return to_world(loc, R, o), np.exp(ls.numpy().astype(np.float64))
        mean = cv_predict(obs, PRED_LEN)
        return mean, np.tile(np.asarray(self.cv_sigma[:PRED_LEN], np.float64), (len(obs), 1))


# ---------------------------------------------------------------------------------------------- datasets (ETH/UCY)
def load_eth_ucy(path: str | Path) -> dict[int, np.ndarray]:
    """Read one ETH/UCY file (frame, ped_id, x, y; whitespace/tab separated, metres) -> ped -> [N, 3] (frame, x, y)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) >= 4:
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                continue
    out: dict[int, list] = {}
    for f, pid, x, y in rows:
        out.setdefault(int(pid), []).append((f, x, y))
    return {k: np.array(sorted(v), np.float64) for k, v in out.items()}


def frame_step(tracks: dict[int, np.ndarray]) -> float:
    """The file's annotation step in frames (the most common frame gap; 10 for UCY, 6 for ETH biwi_eth)."""
    gaps = np.concatenate([np.diff(a[:, 0]) for a in tracks.values() if len(a) > 1] or [np.array([10.0])])
    gaps = gaps[gaps > 0]
    vals, counts = np.unique(np.round(gaps, 3), return_counts=True)
    return float(vals[np.argmax(counts)]) if len(vals) else 10.0


def windows(tracks: dict[int, np.ndarray], seq_len: int = OBS_LEN + PRED_LEN, stride: int = 1) -> np.ndarray:
    """All windows of `seq_len` consecutive annotated steps (no gaps) -> [N, seq_len, 2]."""
    step = frame_step(tracks)
    out = []
    for a in tracks.values():
        if len(a) < seq_len:
            continue
        ok = np.isclose(np.diff(a[:, 0]), step)
        for i in range(0, len(a) - seq_len + 1, stride):
            if ok[i:i + seq_len - 1].all():
                out.append(a[i:i + seq_len, 1:3])
    return np.array(out, np.float64).reshape(-1, seq_len, 2)


def synthetic_walkers(n: int = 2000, seed: int = 0, seq_len: int = OBS_LEN + PRED_LEN) -> np.ndarray:
    """Fallback training data when ETH/UCY cannot be downloaded: smooth random walkers (speed 0.5-1.8 m/s, slowly
    turning, occasional stops) with 3 cm position noise. Clearly NOT real pedestrian data."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, seq_len, 2))
    for i in range(n):
        p = rng.uniform(-10, 10, 2)
        hd = rng.uniform(-np.pi, np.pi)
        sp = rng.uniform(0.5, 1.8)
        turn = rng.normal(0, 0.15)
        stop_at = rng.integers(4, seq_len) if rng.random() < 0.15 else None
        for k in range(seq_len):
            out[i, k] = p
            if stop_at is not None and k >= stop_at:
                sp *= 0.5
            hd += turn * DT + rng.normal(0, 0.05)
            p = p + sp * DT * np.array([math.cos(hd), math.sin(hd)])
    return out + rng.normal(0, 0.03, out.shape)


# ---------------------------------------------------------------------------------------------- runtime analyzer
def sample_history(times: np.ndarray, pts: np.ndarray, t_now: float, n: int = OBS_LEN, dt: float = DT,
                   smooth_s: float = 0.2) -> Optional[np.ndarray]:
    """Positions at t_now - (n-1-k)*dt, k = 0..n-1: a local straight-line fit to the observations within +-smooth_s
    of each sample time, evaluated at that time (reduces detector jitter without lagging at the newest sample, where
    only past observations exist); linear interpolation when fewer than two fall inside. None when the history does
    not cover the first sample time."""
    times = np.asarray(times, np.float64)
    pts = np.asarray(pts, np.float64)
    if len(times) < 2 or times[0] > t_now - (n - 1) * dt + 1e-6 - smooth_s:
        return None
    out = np.zeros((n, 2))
    for k in range(n):
        ts = t_now - (n - 1 - k) * dt
        m = np.abs(times - ts) <= smooth_s + 1e-9
        if m.sum() >= 2 and np.ptp(times[m]) > 1e-6:
            tt = times[m] - ts
            A = np.stack([np.ones_like(tt), tt], -1)
            coef, *_ = np.linalg.lstsq(A, pts[m], rcond=None)
            out[k] = coef[0]
        elif m.any():
            out[k] = pts[m].mean(0)
        else:
            out[k] = [np.interp(ts, times, pts[:, 0]), np.interp(ts, times, pts[:, 1])]
    return out


def entry_probability(mean: np.ndarray, sigma: np.ndarray, poly: list, n_samples: int = 64,
                      rng: Optional[np.random.Generator] = None) -> tuple[float, Optional[int]]:
    """Monte-Carlo probability that a forecast step lies inside `poly` (isotropic Gaussian per step).

    Returns (max over steps of the per-step inside fraction, first step index reaching that max) for ONE track:
    mean [P, 2], sigma [P].
    """
    rng = rng or np.random.default_rng(0)
    best, best_k = 0.0, None
    z = rng.standard_normal((n_samples, 2))
    for k in range(len(mean)):
        pts = mean[k][None, :] + z * float(sigma[k])
        frac = sum(point_in_polygon((float(x), float(y)), poly) for x, y in pts) / n_samples
        if frac > best + 1e-9:
            best, best_k = frac, k
    return best, best_k


@dataclass
class _Track:
    hist: deque = field(default_factory=lambda: deque(maxlen=64))  # (t, x, y, box_h_source_px)
    last_seen: float = 0.0


@dataclass
class _CamState:
    tracks: dict = field(default_factory=dict)
    last_run: float = -1e18
    latest: list = field(default_factory=list)
    frame_size: Optional[list] = None
    entry_hits: dict = field(default_factory=dict)   # (tid, zone) -> consecutive updates over threshold
    last_fired: dict = field(default_factory=dict)   # (tid, zone) -> t


class ForecastAnalyzer:
    """See module docstring. `forecaster` is a Forecaster (or any object with predict(obs) and info)."""

    def __init__(self, forecaster: Forecaster, *, horizon_s: float = 4.8, update_hz: float = 2.5,
                 min_history_s: float = 1.2, stale_s: float = 1.5, zone_entry: Optional[dict] = None,
                 classes: tuple = ("person",)):
        self.fc = forecaster
        self.steps = max(1, min(PRED_LEN, int(round(float(horizon_s) / DT))))
        self.update_period = 1.0 / max(0.1, float(update_hz))
        self.min_history_s, self.stale_s = float(min_history_s), float(stale_s)
        ze = dict(zone_entry or {})
        self.ze_enabled = bool(ze.get("enabled", False))
        self.ze_steps = max(1, min(self.steps, int(round(float(ze.get("horizon_s", 3.2)) / DT))))
        self.ze_min_prob = float(ze.get("min_prob", 0.5))
        self.ze_persist = max(1, int(ze.get("persist", 2)))
        self.ze_rearm = float(ze.get("rearm_s", 60.0))
        self.ze_rules = tuple(ze.get("zone_rules", ("intrusion",)))
        self.ze_criticality = ze.get("criticality")
        self.classes = tuple(classes)
        self.state: dict[str, _CamState] = {}
        self.lock = threading.Lock()
        self.stats = {"updates": 0, "tracks_forecast": 0, "events": 0}
        self._rng = np.random.default_rng(0)

    # ---- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _engine_point(worker, bb, o) -> Optional[tuple[float, float]]:
        if getattr(worker, "H", None) is not None:
            return tuple(o.ground) if o is not None and o.ground else None
        fx, fy = foot_point(bb)
        s = float(getattr(worker, "scale", 1.0) or 1.0)
        return fx * s, fy * s

    def _to_px(self, worker, pts: np.ndarray) -> Optional[np.ndarray]:
        """Engine-space points -> analytics-frame pixels (for drawing on the Live tile)."""
        s = float(getattr(worker, "scale", 1.0) or 1.0)
        if getattr(worker, "H", None) is None:
            return np.asarray(pts, np.float64) / s
        try:
            hinv = invert_h(worker.H)
            return np.array([apply_h(hinv, (float(x), float(y))) for x, y in pts]) / s
        except ValueError:
            return None

    # ---- main hook ----------------------------------------------------------------------------
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        with self.lock:
            st = self.state.setdefault(worker.id, _CamState())
        by_tid = {o.track_id: o for o in obs}
        for tid, bb, cls, conf in dets:
            if cls not in self.classes:
                continue
            p = self._engine_point(worker, bb, by_tid.get(tid))
            if p is None:
                continue
            tr = st.tracks.setdefault(tid, _Track())
            s = float(getattr(worker, "scale", 1.0) or 1.0)
            tr.hist.append((t, float(p[0]), float(p[1]), max(1.0, (bb[3] - bb[1]) * s)))
            tr.last_seen = t
        for tid in [k for k, tr in st.tracks.items() if t - tr.last_seen > self.stale_s]:
            del st.tracks[tid]
        if t - st.last_run < self.update_period - 1e-3:
            return []
        st.last_run = t
        live = {tid for tid, *_ in dets}
        rows, obs_list, meta = [], [], []
        pixel = getattr(worker, "H", None) is None
        for tid, tr in st.tracks.items():
            if tid not in live or len(tr.hist) < 3 or tr.hist[-1][0] - tr.hist[0][0] < self.min_history_s:
                continue
            a = np.array(tr.hist, np.float64)
            m_per_unit = 1.0
            if pixel:  # approximate metres from the person's box height (assumes BODY_HEIGHT_M); rough estimate
                m_per_unit = BODY_HEIGHT_M / float(np.median(a[-10:, 3]))
            full = sample_history(a[:, 0], a[:, 1:3] * m_per_unit, t)
            if full is not None:
                obs_list.append(full); meta.append((tid, m_per_unit, "full"))
            else:  # short history: constant velocity from the samples we have
                n = int((a[-1, 0] - a[0, 0]) // DT) + 1
                short = sample_history(a[:, 0], a[:, 1:3] * m_per_unit, t, n=max(2, n))
                if short is None:
                    continue
                pad = np.vstack([np.repeat(short[:1], OBS_LEN - len(short), 0), short]) if len(short) < OBS_LEN \
                    else short[-OBS_LEN:]
                obs_list.append(pad); meta.append((tid, m_per_unit, "short"))
        out, events = [], []
        if obs_list:
            O = np.array(obs_list)
            full_idx = [i for i, m in enumerate(meta) if m[2] == "full"]
            mean = np.zeros((len(O), PRED_LEN, 2)); sig = np.zeros((len(O), PRED_LEN))
            if full_idx:
                mean[full_idx], sig[full_idx] = self.fc.predict(O[full_idx])
            short_idx = [i for i, m in enumerate(meta) if m[2] == "short"]
            if short_idx:
                mean[short_idx], sig[short_idx] = self.fc.predict(O[short_idx], method="cv")
            for i, (tid, mpu, kind) in enumerate(meta):
                P = self.steps
                path = mean[i, :P] / mpu
                sigma = sig[i, :P] / mpu
                cur = O[i, -1] / mpu
                method = self.fc.method if kind == "full" else "cv"
                row = {"camera": worker.id, "track_id": int(tid), "t": t, "method": method, "dt_s": DT,
                       "horizon_s": round(P * DT, 2), "space": "image_px" if pixel else "ground_m",
                       "approx_scale": pixel, "current": [round(float(v), 3) for v in cur],
                       "path": [[round(float(x), 3), round(float(y), 3)] for x, y in path],
                       "sigma": [round(float(v), 3) for v in sigma]}
                px = self._to_px(worker, np.vstack([cur[None, :], path]))
                if px is not None:
                    # spread in analytics px: sigma projected along +x at each step (ellipse radius for drawing)
                    side = self._to_px(worker, path + np.stack([sigma, np.zeros_like(sigma)], -1))
                    rad = np.linalg.norm(side - px[1:], axis=-1) if side is not None else np.zeros(P)
                    row["current_px"] = [round(float(v), 1) for v in px[0]]
                    row["path_px"] = [[round(float(x), 1), round(float(y), 1)] for x, y in px[1:]]
                    row["sigma_px"] = [round(float(v), 1) for v in rad]
                out.append(row)
                if self.ze_enabled:
                    events += self._zone_entry(worker, st, t, int(tid), cur, path, sigma, method)
        fs = getattr(frame, "shape", None)
        with self.lock:
            st.latest = out
            st.frame_size = [int(fs[1]), int(fs[0])] if fs is not None and len(fs) >= 2 else None
        self.stats["updates"] += 1
        self.stats["tracks_forecast"] = sum(len(s.latest) for s in self.state.values())
        self.stats["events"] += len(events)
        return events

    def _zone_entry(self, worker, st: _CamState, t: float, tid: int, cur, path, sigma, method) -> list[Event]:
        events = []
        zones = getattr(getattr(worker, "engine", None), "zones", []) or []
        for z in zones:
            if not any(r in z.rules for r in self.ze_rules):
                continue
            key = (tid, z.id)
            if point_in_polygon((float(cur[0]), float(cur[1])), z.polygon):
                st.entry_hits.pop(key, None)  # already inside: the deterministic rule's job
                continue
            p, k = entry_probability(path[:self.ze_steps], sigma[:self.ze_steps], z.polygon, rng=self._rng)
            if p < self.ze_min_prob:
                st.entry_hits.pop(key, None)
                continue
            st.entry_hits[key] = st.entry_hits.get(key, 0) + 1
            if st.entry_hits[key] != self.ze_persist or t - st.last_fired.get(key, -1e18) < self.ze_rearm:
                continue
            st.last_fired[key] = t
            eta = round((k + 1) * DT, 1)
            unit = "px (approx.)" if getattr(worker, "H", None) is None else "m"
            model = self.fc.info
            name = model["name"] if method == "gru" else "constant-velocity extrapolation"
            obs_txt = (f"Person track #{tid} ({worker.id}) is currently outside zone '{z.name}'; its forecast path "
                       f"enters the zone in about {eta}s (estimated probability {p:.2f}, {method} forecast). "
                       f"This is a prediction from recent motion, not an observed entry.")
            deriv = {"rule": RULE, "method": f"trajectory forecast ({method}) + Monte-Carlo zone overlap",
                     "prediction": True, "learned_model": name,
                     "threshold": {"min_prob": self.ze_min_prob, "horizon_s": round(self.ze_steps * DT, 2),
                                   "persist_updates": self.ze_persist},
                     "probability": round(p, 3), "eta_s": eta, "units": unit,
                     "path": [[round(float(x), 2), round(float(y), 2)] for x, y in path[:self.ze_steps]],
                     "sigma": [round(float(v), 3) for v in sigma[:self.ze_steps]],
                     "model": {**model, "method_used": method},
                     "confidence_method": "fraction of Gaussian forecast samples inside the zone (uncalibrated)",
                     "calib_error_m": getattr(worker, "calib_error", None), "note": NOTE}
            crit = int(self.ze_criticality) if self.ze_criticality is not None else int(z.criticality)
            ground = (float(path[k][0]), float(path[k][1])) if getattr(worker, "H", None) is not None else None
            events.append(Event(RULE, worker.id, z.id, z.name, [tid], t, t, obs_txt, deriv, round(p, 3), 1, crit,
                                z.site_zone, ground))
        return events

    # ---- API ----------------------------------------------------------------------------------
    def tracks(self, camera: str = "") -> list[dict]:
        with self.lock:
            out = []
            for cam, st in self.state.items():
                if camera and cam != camera:
                    continue
                fs = getattr(st, "frame_size", None)
                out += [{**r, "frame_size": fs} for r in st.latest]
            return out

    def status(self) -> dict:
        return {"enabled": True, "model": self.fc.info, "metrics": self.fc.meta.get("metrics"),
                "horizon_s": round(self.steps * DT, 2),
                "zone_entry": {"enabled": self.ze_enabled, "horizon_s": round(self.ze_steps * DT, 2),
                               "min_prob": self.ze_min_prob, "persist_updates": self.ze_persist},
                "stats": dict(self.stats), "note": NOTE, "experimental": True}

    def close(self) -> None:
        self.state.clear()


# ---------------------------------------------------------------------------------------------- install
DEFAULT_CHECKPOINT = "data/models/argonix_forecast_gru.pt"
BUNDLED_CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "forecast" / "argonix_forecast_gru.pt"


def install(site, forecaster_factory: Optional[Callable] = None) -> Optional[ForecastAnalyzer]:
    """Register the forecaster when the site config has "forecast": {"enabled": true, ...} (off by default).

        "forecast": {"enabled": true, "method": "auto|gru|cv", "checkpoint": "data/models/argonix_forecast_gru.pt",
                     "horizon_s": 4.8, "update_hz": 2.5, "min_history_s": 1.2,
                     "zone_entry": {"enabled": false, "horizon_s": 3.2, "min_prob": 0.5, "persist": 2}}

    A missing checkpoint falls back to constant velocity (reported in site.forecast_status["warnings"]).
    """
    cfg = site.cfg.get("forecast") or {}
    site.forecast = None
    if not cfg.get("enabled", False):
        return None
    warnings = []
    ck = cfg.get("checkpoint", DEFAULT_CHECKPOINT)
    base = Path(getattr(site, "_base", Path.cwd()))
    ckp = Path(ck) if Path(ck).is_absolute() else base / ck
    if not ckp.exists() and "checkpoint" not in cfg and BUNDLED_CHECKPOINT.exists():
        ckp = BUNDLED_CHECKPOINT  # the small ETH/UCY model shipped in the repo (assets/forecast/)
    method = cfg.get("method", "auto")
    try:
        fc = (forecaster_factory or Forecaster)(ckp, "cv" if method == "cv" else "auto")
    except Exception as e:  # noqa: BLE001 - a bad checkpoint must not stop the site
        warnings.append(f"forecast checkpoint {ck} failed to load, using constant velocity: {e!r}"[:300])
        fc = Forecaster(None, "cv")
    if method != "cv" and fc.method == "cv" and not warnings:
        warnings.append(f"forecast checkpoint not found ({ck}); using constant velocity. "
                        "Train one with tools/train_forecast.py")
    an = ForecastAnalyzer(fc, horizon_s=float(cfg.get("horizon_s", 4.8)), update_hz=float(cfg.get("update_hz", 2.5)),
                          min_history_s=float(cfg.get("min_history_s", 1.2)), zone_entry=cfg.get("zone_entry"))
    an.warnings = warnings
    site.analyzers.append(an)
    site.forecast = an
    return an


def save_checkpoint(path: str | Path, model, meta: dict) -> None:
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": json.loads(json.dumps(meta, default=float))}, str(p))
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
