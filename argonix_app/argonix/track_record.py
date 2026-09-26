"""Operator track record: how often reviewed alerts from the same rule on the same camera were confirmed.

Every alert's priority breakdown ("Why P2?") gets a `track_record` factor: of the alerts from its top rule on its
camera that operators closed with a disposition in the last `window_days`, how many were true positives, with a
Wilson 95 % interval. It is feedback, not a model: it only counts decisions people already made.

Optional demotion (site config "feedback_demotion": {"enabled": true, ...}, off by default): a *single, uncorroborated*
alert (one rule, one camera, one modality) from a rule whose reviewed precision here is reliably low (at least
`min_reviewed` decisions and the upper end of the 95 % interval below `max_precision_upper`) drops one priority band,
and the breakdown says why. Corroborated alerts, demo alerts and P4 are never touched, so a noisy rule can't bury a
situation that another rule, camera or sensor confirms.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .correlate import ORDER
from .feedback import DISPOSITIONS, wilson
from .models import alert_is_demo

DEFAULT_DEMOTION = {"enabled": False, "min_reviewed": 20, "max_precision_upper": 0.2}


def top_camera(d: dict) -> Optional[str]:
    top = (d.get("factors") or {}).get("top_event")
    for e in d.get("events", []):
        if e.get("rule") == top:
            return e.get("camera_id")
    return (d.get("cameras") or [None])[0]


def tally(alerts: list[dict]) -> dict[tuple[str, str], list[int]]:
    """(rule, camera) -> [confirmed, reviewed]; also (rule, "*") for the whole site. Demo alerts and dispositions
    that say nothing about correctness (duplicate, drill, unclear) are left out."""
    out: dict[tuple[str, str], list[int]] = {}
    for a in alerts:
        if alert_is_demo(a):
            continue
        tp = DISPOSITIONS.get(a.get("disposition") or "")
        if tp is None:
            continue
        seen = set()
        for e in a.get("events", []):
            for key in ((e["rule"], e.get("camera_id") or ""), (e["rule"], "*")):
                if key in seen:
                    continue
                seen.add(key)
                c = out.setdefault(key, [0, 0])
                c[0] += int(tp)
                c[1] += 1
    return out


class TrackRecord:
    def __init__(self, store, window_days: float = 30.0, ttl_s: float = 60.0, demotion: Optional[dict] = None):
        self.store = store
        self.window_s = float(window_days) * 86400.0
        self.ttl_s = float(ttl_s)
        self.demotion = {**DEFAULT_DEMOTION, **(demotion or {})}
        self._cache: tuple[float, dict] = (-1e18, {})
        self._lock = threading.Lock()

    def counts(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            t, c = self._cache
            if now - t < self.ttl_s:
                return c
        alerts = self.store.list_alerts(status="resolved,dismissed", since=now - self.window_s, limit=20000)
        c = tally(alerts)
        with self._lock:
            self._cache = (now, c)
        return c

    def invalidate(self) -> None:
        with self._lock:
            self._cache = (-1e18, {})

    def record_for(self, rule: str, camera: str, now: Optional[float] = None) -> dict:
        c = self.counts(now)
        k, n = c.get((rule, camera), [0, 0])
        ks, ns = c.get((rule, "*"), [0, 0])
        return {"rule": rule, "camera": camera, "window_days": round(self.window_s / 86400.0, 1),
                "reviewed": n, "confirmed": k, "precision": round(k / n, 3) if n else None,
                "ci95": list(wilson(k, n)),
                "site_reviewed": ns, "site_confirmed": ks, "site_precision": round(ks / ns, 3) if ns else None}

    def annotate(self, d: dict, now: Optional[float] = None) -> dict:
        """Add factors.track_record to an alert dict and apply the (optional) demotion in place."""
        f = d.setdefault("factors", {})
        rule, cam = f.get("top_event"), top_camera(d)
        if not rule or not cam:
            return d
        tr = self.record_for(rule, cam, now)
        f["track_record"] = tr
        dm = self.demotion
        if not dm.get("enabled") or alert_is_demo(d):
            return d
        prio = d.get("priority")
        corroborated = (f.get("distinct_rules", 1) > 1 or f.get("distinct_cameras", 1) > 1
                        or len(f.get("modalities") or []) > 1)
        if (prio in ORDER[:-1] and not corroborated and tr["reviewed"] >= int(dm["min_reviewed"])
                and tr["ci95"][1] < float(dm["max_precision_upper"])):
            new = ORDER[ORDER.index(prio) + 1]
            f["demoted"] = {"from": prio, "to": new, "reason": (
                f"operators confirmed {tr['confirmed']} of the last {tr['reviewed']} reviewed '{rule}' alerts on "
                f"{cam} (95% CI upper bound {tr['ci95'][1]:.2f} < {float(dm['max_precision_upper']):.2f}); a single "
                f"uncorroborated alert from this rule here drops one band")}
            d["priority"] = new
        return d
