"""Seasonal baselines per (camera, zone, hour-of-week).

Occupancy: robust median/MAD of 1-minute mean occupancy from PREVIOUS weeks (samples newer than `history_gap_s`,
i.e. earlier minutes of the current hour, are not part of the reference). Two detectors run on each completed 1-minute bucket, both
only after `min_samples` history exists for that hour-of-week (cold start = silent), each with its own cooldown:
  * point test: the Iglewicz-Hoaglin modified z-score |0.6745 (x - median) / MAD| >= `z_threshold`
    -> `occupancy_deviation`;
  * two-sided tabular CUSUM (manual §09 BL-10 / §5.9) on the same standardized residual r = clip(z, +-8):
    S+ = max(0, S+ + r - k), S- = max(0, S- - r - k); an alarm when S > h (that side restarts at 0)
    -> `occupancy_shift`. It catches a sustained small shift that never crosses the point threshold. The 1-minute
    buckets do not overlap (the manual's warning is about overlapping windows), but consecutive minutes are
    autocorrelated, so false alarms are more frequent than the i.i.d. ARL tables suggest; h defaults to 8.

Event rate: events per zone per hour (`rate_bucket_s`), zero-count hours included, summarised per hour-of-week.
Dwell: completed stays per zone (from the flow analyzer), quantiles per hour-of-week of the ENTRY time
(`dwell_quantile` is the provider a dwell rule can use once `n` is large enough).

All of these are low-severity corroboration signals and descriptive statistics, not primary alarms.
"""
from __future__ import annotations

import math
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

from .models import Event
from .rules import register_rule

register_rule("occupancy_shift", "Sustained occupancy shift for this hour", 1)
RATE_EXCLUDE = {"occupancy_deviation", "occupancy_shift", "camera_health"}  # not zone activity


def hour_of_week(t: float) -> int:
    lt = time.localtime(t)
    return lt.tm_wday * 24 + lt.tm_hour


def modified_z(x: float, hist: list[float], mad_floor: float = 0.5) -> tuple[float, float, float]:
    med = statistics.median(hist)
    mad = max(statistics.median([abs(h - med) for h in hist]), mad_floor)
    return 0.6745 * (x - med) / mad, med, mad


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Inverted-CDF quantile of an already sorted list."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, max(0, math.ceil(q * n - 1e-9) - 1))]


class Cusum:
    """Two-sided tabular CUSUM on standardized residuals."""

    def __init__(self, k: float = 0.5, h: float = 8.0, clip: float = 8.0):
        self.k, self.h, self.clip = float(k), float(h), float(clip)
        self.pos = self.neg = 0.0
        self.n = 0

    def update(self, r: float) -> Optional[str]:
        r = max(-self.clip, min(self.clip, r))
        self.n += 1
        self.pos = max(0.0, self.pos + r - self.k)
        self.neg = max(0.0, self.neg - r - self.k)
        if self.pos > self.h:
            self.pos = 0.0
            return "up"
        if self.neg > self.h:
            self.neg = 0.0
            return "down"
        return None


class Baseline:
    def __init__(self, path: Path, bucket_s: float = 60.0, z_threshold: float = 3.5, min_samples: int = 8,
                 cooldown_s: float = 900.0, max_history: int = 200, cusum_k: float = 0.5, cusum_h: float = 8.0,
                 cusum_cooldown_s: float = 1800.0, cusum_enabled: bool = True, rate_bucket_s: float = 3600.0,
                 dwell_max_s: float = 6 * 3600.0, history_gap_s: float = 3600.0):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS occupancy(camera TEXT, zone TEXT, how INTEGER, t REAL, value REAL);
            CREATE INDEX IF NOT EXISTS ix_occ ON occupancy(camera, zone, how);
            CREATE TABLE IF NOT EXISTS event_rate(camera TEXT, zone TEXT, how INTEGER, t REAL, count INTEGER);
            CREATE INDEX IF NOT EXISTS ix_rate ON event_rate(camera, zone, how);
            CREATE TABLE IF NOT EXISTS dwell(camera TEXT, zone TEXT, how INTEGER, t REAL, dwell_s REAL);
            CREATE INDEX IF NOT EXISTS ix_dwell ON dwell(camera, zone, how);
        """)
        self.lock = threading.Lock()
        self.bucket_s, self.z, self.min_samples = bucket_s, z_threshold, min_samples
        self.cooldown, self.max_history = cooldown_s, max_history
        self.cusum_k, self.cusum_h, self.cusum_cooldown = float(cusum_k), float(cusum_h), float(cusum_cooldown_s)
        self.cusum_enabled = bool(cusum_enabled)
        self.rate_bucket_s, self.dwell_max_s = float(rate_bucket_s), float(dwell_max_s)
        # history excludes samples newer than this: "usual for this hour of the week" = previous weeks, so a shift
        # that is happening now cannot absorb itself into its own reference (0 = include everything, old behaviour)
        self.history_gap_s = float(history_gap_s)
        self.acc: dict[tuple[str, str], list] = {}   # (cam, zone) -> [bucket_start, sum, n]
        self.last_fire: dict[tuple[str, str], float] = {}
        self.cusums: dict[tuple[str, str], Cusum] = {}
        self.cusum_last_fire: dict[tuple[str, str], float] = {}
        self.rate_acc: dict[tuple[str, str], list] = {}  # (cam, zone) -> [bucket_start, count]
        self.zone_names: dict[tuple[str, str], str] = {}

    # ---- occupancy ----
    def history(self, cam: str, zone: str, how: int, before: float = float("inf")) -> list[float]:
        with self.lock:
            rows = self.db.execute("SELECT value FROM occupancy WHERE camera=? AND zone=? AND how=? AND t < ? "
                                   "ORDER BY t DESC LIMIT ?", (cam, zone, how, before, self.max_history)).fetchall()
        return [r[0] for r in rows]

    def series(self, cam: str, zone: str, since: float = 0.0) -> list[tuple[float, float]]:
        """Stored 1-bucket mean occupancy samples (bucket start time, value), oldest first (congestion forecast)."""
        with self.lock:
            return self.db.execute("SELECT t, value FROM occupancy WHERE camera=? AND zone=? AND t >= ? ORDER BY t",
                                   (cam, zone, float(since))).fetchall()

    def add_sample(self, cam: str, zone: str, t: float, value: float) -> None:
        with self.lock:
            self.db.execute("INSERT INTO occupancy VALUES(?,?,?,?,?)", (cam, zone, hour_of_week(t), t, value))
            self.db.commit()

    def observe(self, cam: str, zone_id: str, zone_name: str, t: float, count: int,
                criticality: int = 1) -> Optional[Event]:
        """Feed an instantaneous count; returns the point-test Event (if any) of a completed bucket.

        Use `observe_all` to also receive CUSUM events (the pipeline does)."""
        evs = self.observe_all(cam, zone_id, zone_name, t, count, criticality)
        return next((e for e in evs if e.rule == "occupancy_deviation"), None)

    def observe_all(self, cam: str, zone_id: str, zone_name: str, t: float, count: int,
                    criticality: int = 1) -> list[Event]:
        """Feed an instantaneous count; returns the events of a completed bucket (point test and/or CUSUM)."""
        k = (cam, zone_id)
        self.zone_names[k] = zone_name
        self._roll_rates(cam, t, k)
        a = self.acc.get(k)
        if a is None:
            self.acc[k] = [t, count, 1]
            return []
        if t - a[0] < self.bucket_s:
            a[1] += count; a[2] += 1
            return []
        value = a[1] / a[2]
        self.acc[k] = [t, count, 1]
        how = hour_of_week(a[0])
        hist = self.history(cam, zone_id, how, before=a[0] - self.history_gap_s if self.history_gap_s else float("inf"))
        self.add_sample(cam, zone_id, a[0], value)
        if len(hist) < self.min_samples:
            return []
        z, med, mad = modified_z(value, hist)
        events = []
        if abs(z) >= self.z and t - self.last_fire.get(k, -1e18) >= self.cooldown:
            self.last_fire[k] = t
            direction = "higher" if z > 0 else "lower"
            events.append(Event(
                "occupancy_deviation", cam, zone_id, zone_name, [], a[0], t,
                f"Mean occupancy {value:.1f} in '{zone_name}' is {direction} than usual for this hour "
                f"(median {med:.1f}, n={len(hist)})",
                {"rule": "occupancy_deviation", "hour_of_week": how, "value": round(value, 2),
                 "median": round(med, 2), "mad": round(mad, 2), "modified_z": round(z, 2),
                 "threshold": {"modified_z": self.z, "min_samples": self.min_samples, "bucket_s": self.bucket_s},
                 "confidence_method": "min(1, |z| / (2 x threshold)); statistical, uncalibrated",
                 "duration_s": round(t - a[0], 1)},
                round(min(1.0, abs(z) / (2 * self.z)), 3), 1, criticality))
        if self.cusum_enabled:
            cs = self.cusums.setdefault(k, Cusum(self.cusum_k, self.cusum_h))
            before = (cs.pos, cs.neg)
            side = cs.update(z)
            if side and t - self.cusum_last_fire.get(k, -1e18) >= self.cusum_cooldown:
                self.cusum_last_fire[k] = t
                word = "higher" if side == "up" else "lower"
                s_val = before[0] + max(-8.0, min(8.0, z)) - self.cusum_k if side == "up" else \
                    before[1] - max(-8.0, min(8.0, z)) - self.cusum_k
                events.append(Event(
                    "occupancy_shift", cam, zone_id, zone_name, [], a[0], t,
                    f"Occupancy in '{zone_name}' has stayed {word} than usual for this hour of the week over "
                    f"recent minutes (CUSUM {s_val:.1f} > h={self.cusum_h}; latest minute {value:.1f} vs median "
                    f"{med:.1f}): a sustained shift, not a single spike",
                    {"rule": "occupancy_shift", "side": side, "cusum_value": round(s_val, 2),
                     "threshold": {"cusum_h": self.cusum_h, "cusum_k": self.cusum_k, "min_samples": self.min_samples,
                                   "cooldown_s": self.cusum_cooldown, "bucket_s": self.bucket_s},
                     "hour_of_week": how, "value": round(value, 2), "median": round(med, 2),
                     "residual_z": round(z, 2), "evaluations": cs.n,
                     "confidence_method": "fixed 0.6: CUSUM alarm on robust residuals; statistical, uncalibrated",
                     "duration_s": round(t - a[0], 1)},
                    0.6, 1, criticality))
        return events

    # ---- event rate ----
    def record_event(self, e: Event) -> None:
        """Count an event towards its zone's hourly event rate (baseline-generated and health events excluded)."""
        if e.rule in RATE_EXCLUDE or (e.derivation or {}).get("demo_scenario"):  # demo events are not activity
            return
        k = (e.camera_id, e.zone_id)
        self.zone_names.setdefault(k, e.zone_name)
        with self.lock:
            a = self.rate_acc.get(k)
            if a is None:  # first event of a zone never observed yet: the bucket is partial, it is not persisted
                a = self.rate_acc[k] = [self._rate_start(e.t_end), 0, False]
            a[1] += 1

    def _rate_start(self, t: float) -> float:
        return (t // self.rate_bucket_s) * self.rate_bucket_s

    def _roll_rates(self, cam: str, t: float, key: tuple[str, str]) -> None:
        """Close finished hourly buckets for every zone of `cam` (zero-count hours are recorded too)."""
        start = self._rate_start(t)
        keys = [key] + [(cam, z) for (c, z) in list(self.acc) + list(self.rate_acc) if c == cam]
        rows = []
        with self.lock:
            for k in set(keys):
                a = self.rate_acc.get(k)
                if a is None:  # observation started mid-bucket: count it, but do not persist the partial hour
                    self.rate_acc[k] = [start, 0, t - start < 60.0]
                    continue
                if a[0] < start:
                    if a[2]:
                        rows.append((k[0], k[1], hour_of_week(a[0]), a[0], a[1]))
                    self.rate_acc[k] = [start, 0, t - start < max(60.0, 2 * self.bucket_s)]
            if rows:
                self.db.executemany("INSERT INTO event_rate VALUES(?,?,?,?,?)", rows)
                self.db.commit()

    # ---- dwell ----
    def add_dwell(self, cam: str, zone: str, entry_t: float, dwell_s: float) -> None:
        if not 0 < dwell_s <= self.dwell_max_s:
            return
        with self.lock:
            self.db.execute("INSERT INTO dwell VALUES(?,?,?,?,?)", (cam, zone, hour_of_week(entry_t), entry_t, dwell_s))
            self.db.commit()

    def dwell_quantile(self, cam: str, zone: str, how: int, q: float) -> tuple[Optional[float], int]:
        """(q-quantile of completed stays in this hour-of-week cell, number of stays)."""
        with self.lock:
            vals = sorted(r[0] for r in self.db.execute(
                "SELECT dwell_s FROM dwell WHERE camera=? AND zone=? AND how=?", (cam, zone, how)).fetchall())
        return (_quantile(vals, q) if vals else None), len(vals)

    # ---- summaries ----
    def summary(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT camera, zone, how, COUNT(*), AVG(value) FROM occupancy "
                                   "GROUP BY camera, zone, how ORDER BY camera, zone, how").fetchall()
        return [{"camera": c, "zone": z, "hour_of_week": h, "samples": n, "mean": round(m, 2),
                 "ready": n >= self.min_samples} for c, z, h, n, m in rows]

    def summary_rates(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT camera, zone, how, COUNT(*), AVG(count), MAX(count) FROM event_rate "
                                   "GROUP BY camera, zone, how ORDER BY camera, zone, how").fetchall()
            cur = {k: v[1] for k, v in self.rate_acc.items()}
        return [{"camera": c, "zone": z, "hour_of_week": h, "hours": n, "mean_per_hour": round(m, 2),
                 "max_per_hour": mx, "current_hour": cur.get((c, z))} for c, z, h, n, m, mx in rows]

    def summary_dwell(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT camera, zone, how, dwell_s FROM dwell ORDER BY camera, zone, how").fetchall()
        cells: dict[tuple, list[float]] = {}
        for c, z, h, d in rows:
            cells.setdefault((c, z, h), []).append(d)
        out = []
        for (c, z, h), vals in cells.items():
            vals.sort()
            out.append({"camera": c, "zone": z, "hour_of_week": h, "stays": len(vals),
                        "p50_s": round(_quantile(vals, 0.5), 1), "p90_s": round(_quantile(vals, 0.9), 1),
                        "p95_s": round(_quantile(vals, 0.95), 1)})
        return out

    def summary_cusum(self) -> dict:
        return {"enabled": self.cusum_enabled, "k": self.cusum_k, "h": self.cusum_h,
                "cooldown_s": self.cusum_cooldown,
                "series": [{"camera": c, "zone": z, "s_pos": round(cs.pos, 2), "s_neg": round(cs.neg, 2),
                            "evaluations": cs.n} for (c, z), cs in sorted(self.cusums.items())]}
