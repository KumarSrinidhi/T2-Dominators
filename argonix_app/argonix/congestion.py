"""Zone congestion forecast: occupancy (queue length for queue zones) 5-120 min ahead. A PREDICTION, not an observation.

Data: the 1-minute mean occupancy per (camera, zone) that the seasonal baseline (baseline.py) already stores in
baseline.db. Nothing is trained or kept between requests except a short cache of the backtest.

Model, per zone, from a forecast origin o (the newest stored minute):
  S(T)   seasonal-naive: median occupancy in T's hour of the week, from samples older than `seasonal_gap_s` before o
         (so the hours that feed the recent window never feed their own reference; with < `min_seasonal_samples` there
         is no seasonal reference yet);
  R(h)   damped Holt linear smoothing (alpha, beta, phi) of the last `recent_min` minutes of the deviation y - S:
         how much busier or quieter than usual it is now, and whether that deviation is growing;
  F(h) = S(o + h) + w_h * R(h), clipped at 0. The weight w_h in [0, 1] (0 = pure seasonal-naive, 1 = the current
         deviation persists) is picked per horizon on the OLDER half of a rolling-origin backtest.
  80% interval: F + the 10th / 90th percentile of the older half's forecast errors at that horizon; its coverage on the
         NEWER half is measured and reported.

Skill gate: on the NEWER half of the backtest origins (held out from the weight and interval fit) the forecast must beat
BOTH references by `min_skill` (relative MAE): naive (last value) and seasonal-naive, with >= `min_backtest` held-out
origins each. Otherwise the API answers `useful: false` with "insufficient history" or "no skill" and the dashboard
draws no band. The seasonal reference needs about a week of history before the hour in question.

Congestion note: when the forecast is useful and the zone has a limit (crowding `max_count`, or queue_long
`max_length` for queue zones), the first minute whose upper 80% bound exceeds it. The optional `congestion_forecast`
event (config "congestion_forecast": {"emit_events": true}; off by default) is informational: severity 1, marked
`prediction` and `learned_model`, so the correlator caps it at P3 until a deterministic rule agrees.
"""
from __future__ import annotations

import bisect
import math
import statistics
import threading
import time
from typing import Any, Optional

from . import flow  # noqa: F401  (registers queue_long and its defaults in rules.DEFAULTS)
from .models import Event
from .rules import DEFAULTS as RULE_DEFAULTS, register_rule

RULE = "congestion_forecast"
register_rule(RULE, "Congestion expected (forecast)", 1)
LABEL = "forecast — prediction, not observation"
MODEL = "congestion-forecast-v1: seasonal-naive (hour of week) + damped Holt on the deviation; statistical"
NOTE = ("A forecast is a prediction from past occupancy, not an observation. It is shown only when a backtest on this "
        "zone's own history beats both the last value and the usual-for-this-hour value; the band is an 80% "
        "prediction interval from past errors (its measured coverage is shown). Counts are person tracks.")
W_GRID = tuple(i / 10 for i in range(11))
EVAL_H_MIN = (5, 15, 30, 60)

DEFAULTS: dict[str, Any] = {
    "enabled": True,              # forecast API + dashboard panel (passive statistics)
    "emit_events": False,         # informational congestion_forecast events (off by default)
    "horizon_min": 30,            # horizon used for events
    "max_horizon_min": 120,
    "lookback_days": 28.0,        # history loaded per zone
    "backtest_days": 7.0,         # rolling-origin backtest window (newest part of the history)
    "backtest_max_origins": 400,
    "min_backtest": 30,           # held-out origins needed per reference before the forecast can be called useful
    "min_skill": 0.05,            # required relative MAE improvement over each reference
    "recent_min": 60,             # smoothing window for the deviation (minutes)
    "min_recent": 10,             # samples needed in that window
    "alpha": 0.5, "beta": 0.1, "phi": 0.9,
    "default_tau_min": 30.0,      # weight w_h = exp(-h / tau) when the fit half is too small
    "min_seasonal_samples": 10,
    "max_seasonal_samples": 200,
    "seasonal_gap_s": 7200.0,
    "coverage": 0.8,
    "stale_s": 600.0,             # newest sample older than this -> no forecast
    "backtest_refresh_min": 10,   # reuse a zone's backtest until this many new minutes arrive
    "eval_interval_s": 60.0,      # event check period per camera
    "rearm_s": 1800.0,            # one event per zone per this period
    "confidence": 0.3,
}


def holt(values: list[float], steps: int, alpha: float, beta: float, phi: float) -> list[float]:
    """Damped Holt linear forecasts 1..steps ahead (level from the first value, trend from 0)."""
    if not values:
        return [0.0] * steps
    level, trend = float(values[0]), 0.0
    for y in values[1:]:
        prev = level
        level = alpha * y + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
    out, damp, p = [], 0.0, 1.0
    for _ in range(steps):
        p *= phi
        damp += p
        out.append(level + damp * trend)
    return out


def quantile(vals: list[float], q: float) -> float:
    """Linear-interpolation quantile of an unsorted list."""
    v = sorted(vals)
    if not v:
        return float("nan")
    x = q * (len(v) - 1)
    i = int(math.floor(x))
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (x - i)


def _mae(errs: list[float]) -> Optional[float]:
    return sum(abs(e) for e in errs) / len(errs) if errs else None


class ZoneSeries:
    """One zone's occupancy on a regular grid (index = t // step) with an hour-of-week index for seasonal medians."""

    def __init__(self, rows, step: float, cfg: dict):
        self.step, self.cfg = float(step), cfg
        self.v: dict[int, float] = {}
        for t, val in rows:
            self.v[int(t // self.step)] = float(val)
        self.idx = sorted(self.v)
        self._off: dict[int, int] = {}
        self._loc: dict[int, tuple[int, float]] = {}
        self.by_how: dict[int, list[int]] = {}
        for m in self.idx:
            self.by_how.setdefault(self.local(m)[0], []).append(m)
        self._cache: dict[tuple[int, int], Optional[float]] = {}
        self.steps_per_min = 60.0 / self.step

    def local(self, m: int) -> tuple[int, float]:
        """(hour of week as baseline.hour_of_week, seconds into the local hour) of index m; the UTC offset is looked up
        once per hour (localtime per sample made long histories slow)."""
        r = self._loc.get(m)
        if r is None:
            t = m * self.step
            k = int(t // 3600)
            off = self._off.get(k)
            if off is None:
                off = self._off[k] = time.localtime(t).tm_gmtoff
            lt = t + off
            r = self._loc[m] = (int(((lt // 86400) + 3) % 7 * 24 + (lt % 86400) // 3600), lt % 3600)
        return r

    @property
    def last(self) -> Optional[int]:
        return self.idx[-1] if self.idx else None

    def seasonal(self, m: int, cutoff: int) -> Optional[float]:
        """Median of stored samples in m's hour of the week with index < cutoff (None if too few)."""
        how = self.local(m)[0]
        key = (how, cutoff)
        if key not in self._cache:
            lst = self.by_how.get(how, [])
            i = bisect.bisect_left(lst, cutoff)
            sel = lst[max(0, i - int(self.cfg["max_seasonal_samples"])):i]
            self._cache[key] = (statistics.median(self.v[k] for k in sel)
                                if len(sel) >= int(self.cfg["min_seasonal_samples"]) else None)
        return self._cache[key]

    def profile(self, m: int, cutoff: int) -> Optional[float]:
        """Seasonal level used by the model: the hour-of-week medians linearly interpolated between hour centres (the
        plain median is a step at every hour boundary). Falls back to m's own hour median."""
        own = self.seasonal(m, cutoff)
        if own is None:
            return None
        frac = self.local(m)[1] / 3600.0 - 0.5                        # -0.5 .. +0.5 around the hour centre
        hour = int(round(3600.0 / self.step))
        other = self.seasonal(m + (hour if frac > 0 else -hour), cutoff)
        return own if other is None else own + (other - own) * abs(frac)

    def components(self, o: int, n: int) -> Optional[dict]:
        """Forecast ingredients from origin o (inclusive) for 1..n steps, or None without enough recent samples."""
        c = self.cfg
        if o not in self.v:
            return None
        recent = max(1, int(round(c["recent_min"] * self.steps_per_min)))
        rec = [m for m in range(o - recent + 1, o + 1) if m in self.v]
        if len(rec) < int(c["min_recent"]):
            return None
        cutoff = o - int(round(c["seasonal_gap_s"] / self.step))
        p_rec = [self.profile(m, cutoff) for m in rec]
        a, b, phi = float(c["alpha"]), float(c["beta"]), float(c["phi"])
        dev = None
        if all(p is not None for p in p_rec):
            dev = holt([self.v[m] - p for m, p in zip(rec, p_rec)], n, a, b, phi)
        return {"naive": self.v[o],
                "seasonal": [self.seasonal(o + h, cutoff) for h in range(1, n + 1)],   # the reference
                "profile": [self.profile(o + h, cutoff) for h in range(1, n + 1)],     # the model's seasonal level
                "dev": dev, "raw": holt([self.v[m] for m in rec], n, a, b, phi)}

    @staticmethod
    def combine(comp: dict, h: int, w: Optional[float]) -> tuple[float, bool]:
        """(forecast h steps ahead, True if the seasonal+deviation model was used; else damped Holt on the raw series)."""
        p, dev = comp["profile"][h - 1], comp["dev"]
        if p is not None and dev is not None and w is not None:
            return max(0.0, p + w * dev[h - 1]), True
        return max(0.0, comp["raw"][h - 1]), False


def backtest(zs: ZoneSeries, horizons: list[int]) -> dict:
    """Rolling-origin backtest over the newest `backtest_days`. Origins are split by time: the older half fits the
    per-horizon weight and the interval quantiles, the newer half measures MAE vs naive / seasonal-naive and coverage."""
    c = zs.cfg
    last = zs.last
    out: dict[str, Any] = {"horizons": {}, "n_origins": 0}
    if last is None or not horizons:
        return out
    hmax, hmin = max(horizons), min(horizons)
    start = last - int(c["backtest_days"] * 86400 / zs.step)
    cands = [m for m in zs.idx if start <= m <= last - hmin]
    stride = max(1, math.ceil(len(cands) / int(c["backtest_max_origins"])))
    origins = cands[::stride]
    rows = []  # (origin, comp)
    for o in origins:
        comp = zs.components(o, hmax)
        if comp is not None:
            rows.append((o, comp))
    out["n_origins"] = len(rows)
    half = len(rows) // 2
    fit, ev = rows[:half], rows[half:]
    lo_q, hi_q = (1 - c["coverage"]) / 2, 1 - (1 - c["coverage"]) / 2
    for h in horizons:
        f_rows = [(zs.v[o + h], comp) for o, comp in fit if o + h in zs.v]
        e_rows = [(zs.v[o + h], comp) for o, comp in ev if o + h in zs.v]
        seas_fit = [(y, comp) for y, comp in f_rows if comp["profile"][h - 1] is not None and comp["dev"] is not None]
        h_min = h / zs.steps_per_min
        if len(seas_fit) >= 10:
            w = min(W_GRID, key=lambda w: _mae([y - zs.combine(comp, h, w)[0] for y, comp in seas_fit]))
            w_src = "fitted on the older half of the backtest"
        else:
            w = math.exp(-h_min / float(c["default_tau_min"]))
            w_src = "default exp(-h / tau): too little seasonal history to fit"
        fit_err = [y - zs.combine(comp, h, w)[0] for y, comp in f_rows]
        q_lo = quantile(fit_err, lo_q) if len(fit_err) >= 10 else None
        q_hi = quantile(fit_err, hi_q) if len(fit_err) >= 10 else None
        err, err_naive, err_s, err_f_s, inside = [], [], [], [], 0
        for y, comp in e_rows:
            f, _ = zs.combine(comp, h, w)
            err.append(y - f)
            err_naive.append(y - comp["naive"])
            s = comp["seasonal"][h - 1]
            if s is not None:
                err_s.append(y - s)
                err_f_s.append(y - f)
            if q_lo is not None and f + q_lo <= y <= f + q_hi:
                inside += 1
        mae, mae_n, mae_s, mae_fs = _mae(err), _mae(err_naive), _mae(err_s), _mae(err_f_s)
        out["horizons"][h] = {
            "h_min": round(h_min, 2), "n_fit": len(f_rows), "n_eval": len(e_rows), "n_eval_seasonal": len(err_s),
            "mae": _r(mae), "mae_naive": _r(mae_n), "mae_seasonal": _r(mae_s), "mae_on_seasonal_subset": _r(mae_fs),
            "skill_vs_naive": _r(_skill(mae, mae_n)), "skill_vs_seasonal": _r(_skill(mae_fs, mae_s)),
            "weight": round(w, 3), "weight_source": w_src,
            "interval": {"q_lo": _r(q_lo), "q_hi": _r(q_hi), "nominal": c["coverage"],
                         "coverage_eval": _r(inside / len(e_rows)) if e_rows and q_lo is not None else None},
        }
    return out


def _skill(mae: Optional[float], ref: Optional[float]) -> Optional[float]:
    """Relative MAE improvement over a reference (1 = perfect, 0 = as good, < 0 = worse)."""
    if mae is None or ref is None:
        return None
    if ref == 0:
        return 0.0 if mae == 0 else -1.0
    return 1 - mae / ref


def _r(v, d: int = 3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), d)


def gate(bt: dict, h: int, cfg: dict) -> tuple[str, str]:
    """(status, message) for horizon h: ok | insufficient_history | no_skill."""
    r = bt["horizons"].get(h)
    need, ms = int(cfg["min_backtest"]), float(cfg["min_skill"])
    if r is None or r["n_eval"] < need:
        n = 0 if r is None else r["n_eval"]
        return "insufficient_history", (f"Insufficient history: {n} held-out backtest origins at this horizon "
                                        f"(needs {need}).")
    if r["n_eval_seasonal"] < need:
        return "insufficient_history", (f"Insufficient history for the seasonal reference: {r['n_eval_seasonal']} "
                                        f"held-out origins with a usual-for-this-hour value (needs {need}; about a "
                                        "week of history per hour of the week).")
    sn, ss = r["skill_vs_naive"], r["skill_vs_seasonal"]
    if sn is None or ss is None or sn < ms or ss < ms:
        return "no_skill", (f"Insufficient skill: backtest MAE {r['mae']} vs last value {r['mae_naive']} and vs "
                            f"usual-for-this-hour {r['mae_seasonal']} (needs {ms:.0%} better than both). "
                            "No forecast is shown.")
    return "ok", (f"Backtest on {r['n_eval']} held-out origins: MAE {r['mae']} vs {r['mae_naive']} (last value) and "
                  f"{r['mae_seasonal']} (usual for this hour).")


def forecast_zone(rows, step: float, horizon_min: float, cfg: dict, now: Optional[float] = None,
                  limit: Optional[float] = None, bt_cache: Optional[dict] = None) -> dict:
    """Forecast one zone's series `rows` [(t, value)] `horizon_min` ahead. `now` (epoch) enables the staleness check.
    `bt_cache` (a dict owned by the caller) reuses the backtest until `backtest_refresh_min` new minutes arrive."""
    zs = ZoneSeries(rows, step, cfg)
    spm = zs.steps_per_min
    H = max(1, int(round(horizon_min * spm)))
    out: dict[str, Any] = {"horizon_min": horizon_min, "label": LABEL, "model": MODEL, "useful": False,
                           "forecast": [], "history": [], "backtest": None, "congestion": None, "now": None,
                           "as_of": None, "stale": False}
    last = zs.last
    if last is None:
        out.update(status="no_data", message="No occupancy samples stored for this zone yet.")
        return out
    out["history"] = [{"t": m * step, "v": round(zs.v[m], 3)} for m in range(last - int(round(60 * spm)) + 1, last + 1)
                      if m in zs.v]
    out["now"] = round(zs.v[last], 3)
    out["as_of"] = (last + 1) * step
    if now is not None and now - (last + 1) * step > float(cfg["stale_s"]):
        out.update(status="stale", stale=True,
                   message=f"No occupancy sample for {int((now - (last + 1) * step) // 60)} min: no forecast.")
        return out
    horizons = sorted({int(round(x * spm)) for x in EVAL_H_MIN if x * spm <= H} | {H})
    key = (H, step)
    cached = (bt_cache or {}).get(key)
    if cached and cached[0] <= last < cached[0] + cfg["backtest_refresh_min"] * spm:
        bt = cached[1]
    else:
        bt = backtest(zs, horizons)
        if bt_cache is not None:
            bt_cache[key] = (last, bt)
    out["backtest"] = {"n_origins": bt["n_origins"], "backtest_days": cfg["backtest_days"],
                       "split": "older half fits weights and interval; newer half is held out for MAE and coverage",
                       "horizons": [{"h": h, **r} for h, r in sorted(bt["horizons"].items())]}
    status, msg = gate(bt, H, cfg)
    comp = zs.components(last, H)
    if comp is None:
        status, msg = "insufficient_history", (f"Insufficient recent data: fewer than {cfg['min_recent']} samples in the "
                                               f"last {cfg['recent_min']} min.")
    out.update(status=status, message=msg, useful=status == "ok")
    if comp is None:
        return out
    ev_h = sorted(bt["horizons"])
    pts = [(0, 1.0)] + [(h, bt["horizons"][h]["weight"]) for h in ev_h]

    def weight(h: int) -> float:
        for (h0, w0), (h1, w1) in zip(pts, pts[1:]):
            if h <= h1:
                return w0 + (w1 - w0) * (h - h0) / max(1, h1 - h0)
        return pts[-1][1]

    def band(h: int) -> tuple[Optional[float], Optional[float]]:
        for hh in ev_h:  # the nearest evaluated horizon at or beyond h (conservative)
            if hh >= h:
                iv = bt["horizons"][hh]["interval"]
                return iv["q_lo"], iv["q_hi"]
        return (bt["horizons"][ev_h[-1]]["interval"]["q_lo"], bt["horizons"][ev_h[-1]]["interval"]["q_hi"]) \
            if ev_h else (None, None)

    used_seasonal = False
    for h in range(1, H + 1):
        f, seas = zs.combine(comp, h, weight(h))
        used_seasonal |= seas
        lo, hi = band(h)
        out["forecast"].append({"t": (last + h) * step, "h_min": round(h / spm, 2), "mean": round(f, 3),
                                "lo": round(max(0.0, f + lo), 3) if lo is not None else None,
                                "hi": round(max(0.0, f + hi), 3) if hi is not None else None})
    out["method"] = ("seasonal-naive + weighted damped-Holt deviation" if used_seasonal else
                     "damped Holt on the raw series (no seasonal history yet)")
    if limit is not None:
        cong: dict[str, Any] = {"limit": limit, "observed_now_above": out["now"] > limit, "expected": False,
                                "in_min": None, "mean_in_min": None, "bound": f"upper {cfg['coverage']:.0%} bound"}
        if out["useful"]:
            first = next((p for p in out["forecast"] if p["hi"] is not None and p["hi"] > limit), None)
            mean_first = next((p for p in out["forecast"] if p["mean"] > limit), None)
            if first:
                cong.update(expected=True, in_min=first["h_min"], upper=first["hi"], mean=first["mean"])
            if mean_first:
                cong["mean_in_min"] = mean_first["h_min"]
        out["congestion"] = cong
    return out


def zone_limit(raw: dict) -> Optional[dict]:
    """The zone's occupancy limit: crowding max_count, else queue_long max_length (count strictly above = over)."""
    rules = raw.get("rules") or {}
    if "crowding" in rules:
        p = {**RULE_DEFAULTS.get("crowding", {}), **(rules["crowding"] or {})}
        return {"value": float(p["max_count"]), "rule": "crowding", "label": "crowding limit"}
    if "queue_long" in rules:
        p = {**RULE_DEFAULTS.get("queue_long", {}), **(rules["queue_long"] or {})}
        if float(p.get("max_length") or 0) > 0:
            return {"value": float(p["max_length"]), "rule": "queue_long", "label": "queue length limit"}
    return None


class CongestionForecaster:
    """Site-level forecaster: zone discovery, per-zone backtest cache, the API payload and the optional event."""

    def __init__(self, site, cfg: Optional[dict] = None):
        self.site = site
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.lock = threading.Lock()
        self.bt_cache: dict[tuple[str, str], dict] = {}
        self.last_eval: dict[str, float] = {}
        self.last_fired: dict[tuple[str, str], float] = {}

    def zones(self) -> list[dict]:
        out = []
        for w in list(self.site.workers):
            live = {z.id for z in getattr(getattr(w, "engine", None), "zones", [])}
            for raw in w.cfg.get("zones", []):
                zid = raw.get("id")
                if zid not in live:
                    continue
                rules = raw.get("rules") or {}
                queue = raw.get("type") == "queue" or "queue_long" in rules
                out.append({"camera": w.id, "camera_name": w.name, "zone": zid, "name": raw.get("name", zid),
                            "kind": "queue" if queue else "area",
                            "metric": "queue length (people in the queue zone)" if queue else "occupancy (people)",
                            "limit": zone_limit(raw), "criticality": int(raw.get("criticality", 1)),
                            "site_zone": raw.get("site_zone")})
        return out

    def forecast(self, z: dict, horizon_min: float, now: Optional[float] = None) -> dict:
        b = self.site.baseline
        t_ref = time.time() if now is None else now
        rows = b.series(z["camera"], z["zone"], since=t_ref - float(self.cfg["lookback_days"]) * 86400)
        with self.lock:
            cache = self.bt_cache.setdefault((z["camera"], z["zone"]), {})
        lim = z["limit"]["value"] if z.get("limit") else None
        f = forecast_zone(rows, b.bucket_s, horizon_min, self.cfg, now=t_ref, limit=lim, bt_cache=cache)
        return {**z, **f}

    def all(self, horizon_min: float, camera: str = "", zone: str = "", now: Optional[float] = None) -> list[dict]:
        return [self.forecast(z, horizon_min, now) for z in self.zones()
                if (not camera or z["camera"] == camera) and (not zone or z["zone"] == zone)]

    # ---- optional informational event (Site.analyzers hook; installed only with emit_events) ----------------------
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        if not self.cfg.get("emit_events") or t - self.last_eval.get(worker.id, -1e18) < float(self.cfg["eval_interval_s"]):
            return []
        self.last_eval[worker.id] = t
        events = []
        for z in self.zones():
            if z["camera"] != worker.id or not z.get("limit"):
                continue
            key = (z["camera"], z["zone"])
            if t - self.last_fired.get(key, -1e18) < float(self.cfg["rearm_s"]):
                continue
            f = self.forecast(z, float(self.cfg["horizon_min"]), now=t)
            cong = f.get("congestion") or {}
            if not (f["useful"] and cong.get("expected")) or cong.get("observed_now_above"):
                continue  # already above the limit: the deterministic crowding / queue rule reports it
            self.last_fired[key] = t
            hz = max(f["backtest"]["horizons"], key=lambda r: r["h"], default={})
            obs_txt = (f"Forecast (prediction, not an observation): {z['metric']} in '{z['name']}' may exceed the "
                       f"{z['limit']['label']} of {z['limit']['value']:g} in about {cong['in_min']:g} min (upper "
                       f"{self.cfg['coverage']:.0%} bound {cong['upper']:.1f}, expected {cong['mean']:.1f}); "
                       f"{f['now']:.1f} now.")
            deriv = {"rule": RULE, "prediction": True, "learned_model": MODEL, "method": f.get("method"),
                     "horizon_min": f["horizon_min"], "limit": z["limit"], "congestion": cong,
                     "backtest": {k: hz.get(k) for k in ("n_eval", "mae", "mae_naive", "mae_seasonal",
                                                         "skill_vs_naive", "skill_vs_seasonal")},
                     "confidence_method": "fixed: statistical forecast, uncalibrated", "note": NOTE}
            events.append(Event(RULE, worker.id, z["zone"], z["name"], [], t, t, obs_txt, deriv,
                                float(self.cfg["confidence"]), 1, z["criticality"], z.get("site_zone")))
        return events


def install(site) -> Optional[CongestionForecaster]:
    cfg = {**DEFAULTS, **(site.cfg.get("congestion_forecast") or {})}
    site.congestion = None
    if not cfg.get("enabled", True):
        return None
    site.congestion = CongestionForecaster(site, cfg)
    if cfg.get("emit_events"):
        site.analyzers.append(site.congestion)
    return site.congestion
