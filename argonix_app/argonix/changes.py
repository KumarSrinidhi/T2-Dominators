""""What changed vs normal?": the last few hours compared with this site's own history for the same hours of the week.

Four comparisons, each only where enough history exists (otherwise the zone is listed under `not_enough_history`,
never guessed):

* occupancy: mean people count per zone in the window vs the per-minute samples from previous weeks at the same
  hours of the week (robust median/MAD, the same modified z-score the occupancy baseline uses);
* event rate: events per zone in the window (rules, sensors; demo, health and baseline-generated events excluded)
  vs the expected count from the hourly event-rate history of the same hours of the week (Poisson-style z);
* dwell: median completed stay per zone in the window vs earlier stays at the same hours of the week;
* camera health: cameras whose view is degraded right now, and view alerts raised in the window.

Everything is a descriptive statistic about places and counts. It says nothing about who anyone is or what they
intend, and it raises no alert: it is a briefing for an operator starting a shift or checking a hunch.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Optional

from .baseline import RATE_EXCLUDE, hour_of_week, modified_z
from .models import alert_is_demo, is_demo

DISCLAIMER = ("Statistical comparison with this site's own history at the same hours of the week. It describes counts "
              "and durations in places, not people or intent, and it raises no alerts. Verify against live views.")


def _hours(since: float, until: float, step_s: float = 300.0) -> list[tuple[int, float]]:
    """(hour of week, fraction of an hour the window spends in it), in local time like the baselines (5-min steps,
    so time zones with half-hour offsets are handled)."""
    acc: dict[int, float] = {}
    t = since
    while t < until:
        dt = min(step_s, until - t)
        h = hour_of_week(t + dt / 2)
        acc[h] = acc.get(h, 0.0) + dt / 3600.0
        t += dt
    return sorted(acc.items())


def _zone_names(site) -> dict[tuple[str, str], str]:
    names = {}
    for w in site.workers:
        for z in w.cfg.get("zones", []) or []:
            names[(w.id, z.get("id"))] = z.get("name") or z.get("id")
    return names


def what_changed(site, hours: float = 1.0, now: Optional[float] = None, top: int = 12,
                 z_threshold: float = 2.0, min_history: int = 8) -> dict:
    now = time.time() if now is None else float(now)
    hours = max(0.25, min(float(hours), 24.0))
    since = now - hours * 3600.0
    b = site.baseline
    names = _zone_names(site)
    cam_names = {w.id: w.name for w in site.workers}
    hw = _hours(since, now)
    cells = sorted({h for h, _ in hw})
    changes: list[dict] = []
    steady: list[str] = []
    thin: list[dict] = []

    def label(cam: str, zone: str) -> str:
        zn = names.get((cam, zone)) or b.zone_names.get((cam, zone)) or zone
        return f"'{zn}' ({cam_names.get(cam, cam)})"

    # ---- occupancy ---------------------------------------------------------------------------------------------
    with b.lock:
        rows = b.db.execute("SELECT camera, zone, value FROM occupancy WHERE t >= ? AND t < ?", (since, now)).fetchall()
    recent: dict[tuple[str, str], list[float]] = {}
    for cam, zone, v in rows:
        recent.setdefault((cam, zone), []).append(v)
    for (cam, zone), vals in recent.items():
        hist = [x for h in cells for x in b.history(cam, zone, h, before=since)]
        if len(hist) < min_history:
            thin.append({"kind": "occupancy", "camera": cam, "zone": zone, "history": len(hist)})
            continue
        x = statistics.fmean(vals)
        z, med, mad = modified_z(x, hist)
        item = f"occupancy in {label(cam, zone)}"
        if abs(z) >= z_threshold:
            word = "higher" if z > 0 else "lower"
            changes.append({"kind": "occupancy", "camera": cam, "zone": zone, "observed": round(x, 2),
                            "usual": round(med, 2), "direction": word, "score": round(z, 2),
                            "basis": f"{len(hist)} one-minute samples from previous weeks at the same hours",
                            "text": f"Mean occupancy in {label(cam, zone)} was {x:.1f} people, {word} than the usual "
                                    f"{med:.1f} for these hours (modified z {z:+.1f})."})
        else:
            steady.append(item)

    # ---- event rate --------------------------------------------------------------------------------------------
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for a in site.store.list_alerts(since=since, limit=5000):
        if alert_is_demo(a):
            continue
        for e in a.get("events", []):
            if e["rule"] in RATE_EXCLUDE or is_demo(e) or not (since <= float(e.get("t_end") or 0) <= now):
                continue
            k = (e["camera_id"], e["zone_id"])
            b.zone_names.setdefault(k, e.get("zone_name") or e["zone_id"])
            by = counts.setdefault(k, {})
            by[e["rule"]] = by.get(e["rule"], 0) + 1
    with b.lock:
        hist_rows = b.db.execute(
            f"SELECT camera, zone, how, count FROM event_rate WHERE t < ? AND how IN ({','.join('?' * len(cells))})",
            (since, *cells)).fetchall() if cells else []
    per_cell: dict[tuple[str, str], dict[int, list[int]]] = {}
    for cam, zone, h, c in hist_rows:
        per_cell.setdefault((cam, zone), {}).setdefault(h, []).append(c)
    for k in set(counts) | set(per_cell):
        cam, zone = k
        obs = sum(counts.get(k, {}).values())
        cell = per_cell.get(k, {})
        n_hours = sum(len(cell.get(h, [])) for h in cells)
        if n_hours < max(2, len(cells)):
            if obs:
                thin.append({"kind": "event_rate", "camera": cam, "zone": zone, "history": n_hours, "observed": obs})
            continue
        exp = sum(frac * (statistics.fmean(cell[h]) if cell.get(h) else 0.0) for h, frac in hw)
        z = (obs - exp) / math.sqrt(max(exp, 1.0))
        item = f"events in {label(cam, zone)}"
        if abs(z) >= z_threshold and abs(obs - exp) >= 2:
            word = "more" if obs > exp else "fewer"
            rules = ", ".join(f"{r} x{n}" for r, n in sorted(counts.get(k, {}).items(), key=lambda kv: -kv[1]))
            changes.append({"kind": "event_rate", "camera": cam, "zone": zone, "observed": obs, "usual": round(exp, 2),
                            "direction": "higher" if obs > exp else "lower", "score": round(z, 2), "rules": counts.get(k, {}),
                            "basis": f"{n_hours} hourly counts from previous weeks at the same hours",
                            "text": f"{obs} event(s) in {label(cam, zone)}, {word} than the usual {exp:.1f} for these "
                                    f"hours" + (f" ({rules})" if rules else "") + "."})
        elif obs or exp:
            steady.append(item)

    # ---- dwell -------------------------------------------------------------------------------------------------
    with b.lock:
        drows = b.db.execute("SELECT camera, zone, dwell_s FROM dwell WHERE t >= ? AND t < ?", (since, now)).fetchall()
        dhist = b.db.execute(
            f"SELECT camera, zone, dwell_s FROM dwell WHERE t < ? AND how IN ({','.join('?' * len(cells))})",
            (since, *cells)).fetchall() if cells else []
    drec: dict[tuple[str, str], list[float]] = {}
    dh: dict[tuple[str, str], list[float]] = {}
    for cam, zone, v in drows:
        drec.setdefault((cam, zone), []).append(v)
    for cam, zone, v in dhist:
        dh.setdefault((cam, zone), []).append(v)
    for k, vals in drec.items():
        cam, zone = k
        hist = dh.get(k, [])
        if len(vals) < 3 or len(hist) < max(min_history, 10):
            if len(hist) < max(min_history, 10):
                thin.append({"kind": "dwell", "camera": cam, "zone": zone, "history": len(hist)})
            continue
        x = statistics.median(vals)
        z, med, _ = modified_z(x, hist)
        if abs(z) >= z_threshold:
            word = "longer" if z > 0 else "shorter"
            changes.append({"kind": "dwell", "camera": cam, "zone": zone, "observed": round(x, 1), "usual": round(med, 1),
                            "direction": "higher" if z > 0 else "lower", "score": round(z, 2),
                            "basis": f"{len(hist)} earlier stays at the same hours; {len(vals)} stays in the window",
                            "text": f"Median stay in {label(cam, zone)} was {x:.0f} s, {word} than the usual {med:.0f} s "
                                    f"for these hours ({len(vals)} stays, modified z {z:+.1f})."})
        else:
            steady.append(f"stays in {label(cam, zone)}")

    # ---- camera health (no baseline needed: any degraded view is a change) --------------------------------------
    for w in site.workers:
        st = getattr(w, "stats", {}) or {}
        reasons = st.get("health_reasons") or []
        if st.get("health") not in (None, "ok") and (reasons or st.get("health") == "down"):
            what = ", ".join(reasons) or st.get("health")
            changes.append({"kind": "camera_health", "camera": w.id, "zone": "_view", "observed": what, "usual": "ok",
                            "direction": "degraded", "score": 99.0, "basis": "current view check",
                            "text": f"Camera {w.name} ({w.id}) view is degraded now: {what}."})

    changes.sort(key=lambda c: -abs(c["score"]))
    return {"window": {"since": since, "until": now, "hours": hours, "hours_of_week": cells},
            "changes": changes[:top], "more": max(0, len(changes) - top), "steady": sorted(set(steady)),
            "not_enough_history": thin, "z_threshold": z_threshold, "disclaimer": DISCLAIMER}
