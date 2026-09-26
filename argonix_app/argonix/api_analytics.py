"""Analytics API: zone flow/dwell/density/queue statistics, counting lines, event hotspots, unattended objects and
the trajectory-anomaly model status. Read endpoints need `alert.read`; editing counting lines needs `zone.edit`.

`/api/analytics/forecast` is the zone congestion forecast (argonix/congestion.py): a labelled prediction with its own
backtest, presented as useful only when it beats the naive and seasonal-naive references.

`/api/analytics/zones` and `/lines` return 404 (plain "Not Found", so the UI treats the feature as absent) when the
flow analytics are disabled (`"analytics": {"enabled": false}`). The anomaly and objects endpoints always answer and
say whether their analyzer is enabled.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from .api_cameras import ZONE_ID_RE, _num
from .auth import Principal
from .flow import parse_line_events
from .models import alert_is_demo

WINDOWS = {"24h": 86400.0, "7d": 7 * 86400.0}
MAX_LINES = 32


def validate_lines(lines: Any, calibrated: bool) -> tuple[list[dict], list[str]]:
    """Validate counting lines: [{"id", "name", "points_px": [[x0, y0], [x1, y1]] | "points": [...] (metres),
    "in_label", "out_label", "band", "events": {"direction", "min_gap_s", "criticality", "schedule"}}].
    Returns (clean lines, warnings)."""
    if not isinstance(lines, list):
        raise ValueError("lines must be a list")
    if len(lines) > MAX_LINES:
        raise ValueError(f"at most {MAX_LINES} lines per camera")
    clean, warnings, ids = [], [], set()
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            raise ValueError(f"line {i} must be an object")
        lid = str(ln.get("id", ""))
        if not ZONE_ID_RE.match(lid):
            raise ValueError(f"line {i}: id must be 1-40 characters of letters, digits, '_' or '-'")
        if lid in ids:
            raise ValueError(f"duplicate line id '{lid}'")
        ids.add(lid)
        out: dict[str, Any] = {"id": lid, "name": str(ln.get("name") or lid)[:80]}
        key = "points_px" if "points_px" in ln else "points" if "points" in ln else None
        if key is None:
            raise ValueError(f"line '{lid}' needs points_px (image pixels) or points (ground metres)")
        pts = ln[key]
        if (not isinstance(pts, (list, tuple)) or len(pts) != 2
                or not all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pts)):
            raise ValueError(f"line '{lid}': {key} must be [[x0, y0], [x1, y1]]")
        out[key] = [[_num(q[0], f"line '{lid}' {key}"), _num(q[1], f"line '{lid}' {key}")] for q in pts]
        if out[key][0] == out[key][1]:
            raise ValueError(f"line '{lid}': the two points must differ")
        if key == "points" and not calibrated:
            warnings.append(f"line '{lid}' uses ground metres but the camera is uncalibrated: it is skipped until the "
                            "camera is calibrated (draw it in pixels instead)")
        for lab in ("in_label", "out_label"):
            if ln.get(lab) is not None:
                out[lab] = str(ln[lab])[:40]
        if ln.get("band") is not None:
            b = _num(ln["band"], f"line '{lid}' band")
            if b < 0:
                raise ValueError(f"line '{lid}': band must be >= 0")
            out["band"] = b
        ev = parse_line_events(ln.get("events"), lid)  # opt-in line_cross events (argonix/flow.py)
        if ev is not None:
            out["events"] = {"direction": ev["direction"], "min_gap_s": ev["min_gap_s"],
                             "criticality": ev["criticality"]}
            if ev["schedule"] is not None:
                out["events"]["schedule"] = ln["events"]["schedule"]
        clean.append(out)
    return clean, warnings


def _cell(v: float, size: float) -> float:
    return math.floor(v / size) * size


def spatial_hotspots(site, since: float, until: Optional[float] = None, cell_m: float = 1.0, cell_px: float = 48.0,
                     include_dismissed: bool = False) -> dict:
    """Where rules fired: per-camera ground (metre) / image (pixel) grid cells and per-zone event counts since `since`.
    Shared by /api/analytics/hotspots and /api/hotspots (the incident hotspot view attaches it as `spatial`)."""
    now = time.time() if until is None else until
    alerts = site.store.list_alerts(updated_since=since, limit=100000)
    cams: dict[str, dict] = {}
    zones: dict[tuple, dict] = {}
    n_events = 0
    for a in alerts:
        if (a.get("status") == "dismissed" and not include_dismissed) or alert_is_demo(a):
            continue
        for e in a.get("events", []):
            if (e.get("t_end") or 0) < since:
                continue
            n_events += 1
            cam = e.get("camera_id", "?")
            cd = cams.setdefault(cam, {"camera": cam, "events": 0, "ground": {}, "image": {}, "unplaced": 0})
            cd["events"] += 1
            rule = e.get("rule", "?")
            g = e.get("ground")
            foot = (e.get("derivation") or {}).get("foot_px")
            if g and len(g) == 2:
                key, grid, size = (_cell(g[0], cell_m), _cell(g[1], cell_m)), cd["ground"], cell_m
            elif foot and len(foot) == 2:
                key, grid, size = (_cell(foot[0], cell_px), _cell(foot[1], cell_px)), cd["image"], cell_px
            else:
                cd["unplaced"] += 1
                key = None
            if key is not None:
                c = grid.setdefault(key, {"x0": round(key[0], 3), "y0": round(key[1], 3), "size": size,
                                          "count": 0, "rules": {}})
                c["count"] += 1
                c["rules"][rule] = c["rules"].get(rule, 0) + 1
            zk = (cam, e.get("zone_id", "?"))
            zd = zones.setdefault(zk, {"camera": cam, "zone_id": zk[1], "zone_name": e.get("zone_name", zk[1]),
                                       "events": 0, "alerts": set(), "by_rule": {}, "last": 0.0})
            zd["events"] += 1
            zd["alerts"].add(a["id"])
            zd["by_rule"][rule] = zd["by_rule"].get(rule, 0) + 1
            zd["last"] = max(zd["last"], e.get("t_end") or 0.0)
    out_cams = []
    for cd in cams.values():
        out_cams.append({"camera": cd["camera"], "events": cd["events"], "unplaced": cd["unplaced"],
                         "ground_cells": sorted(cd["ground"].values(), key=lambda c: -c["count"])[:100],
                         "image_cells": sorted(cd["image"].values(), key=lambda c: -c["count"])[:100]})
    out_zones = sorted(({**z, "alerts": len(z["alerts"])} for z in zones.values()),
                       key=lambda z: (-z["events"], z["camera"], z["zone_id"]))
    names = {w.id: w.name for w in site.workers}
    for c in out_cams:
        c["name"] = names.get(c["camera"], c["camera"])
    return {"since": since, "until": now, "cell_m": cell_m, "cell_px": cell_px,
            "include_dismissed": include_dismissed, "events": n_events, "cameras": out_cams, "zones": out_zones,
            "note": ("Event frequency, not risk: a hotspot shows where rules fired most (true or false "
                     "positives). Ground cells use each camera's own ground frame; image cells are for "
                     "uncalibrated cameras.")}


def analytics_router(site, need: Callable[[str], Callable]) -> APIRouter:
    r = APIRouter()

    def flow():
        an = getattr(site, "flow", None)
        if an is None:
            raise HTTPException(404)
        return an

    @r.get("/api/analytics/zones")
    def zones(p: Principal = Depends(need("alert.read"))):
        d = flow().summary_zones()
        names = {w.id: w.name for w in site.workers}
        for c in d["cameras"]:
            c["name"] = names.get(c["camera"], c["camera"])
        d["note"] = ("Counts are person tracks (tracker ID switches split or merge people). Density needs a "
                     "calibrated camera; queue waits are Little's-law estimates unless marked measured.")
        return d

    @r.get("/api/analytics/lines")
    def lines(p: Principal = Depends(need("alert.read"))):
        d = flow().summary_lines()
        names = {w.id: w.name for w in site.workers}
        for c in d["cameras"]:
            c["name"] = names.get(c["camera"], c["camera"])
        return d

    @r.put("/api/cameras/{cid}/lines")
    def put_lines(cid: str, body: dict = Body(...), p: Principal = Depends(need("zone.edit"))):
        c = site.camera_config(cid)
        if c is None:
            raise HTTPException(404, f"unknown camera {cid}")
        try:
            clean, warnings = validate_lines(body.get("lines"), calibrated=bool(c.get("calibration")))
            site.update_camera_config(cid, {"lines": clean})
        except KeyError:
            raise HTTPException(404, f"unknown camera {cid}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        site.store.audit(p.username, "camera_lines", "", {"camera": cid, "lines": [ln["id"] for ln in clean]})
        return {"camera": cid, "lines": clean, "warnings": warnings}

    @r.get("/api/analytics/hotspots")
    def hotspots(window: str = "24h", cell_m: float = 1.0, cell_px: float = 48.0, include_dismissed: bool = False,
                 p: Principal = Depends(need("alert.read"))):
        if window not in WINDOWS:
            raise HTTPException(400, f"window must be one of {sorted(WINDOWS)}")
        if not (0.1 <= cell_m <= 50) or not (4 <= cell_px <= 512):
            raise HTTPException(400, "cell_m must be 0.1-50 and cell_px 4-512")
        now = time.time()
        return {"window": window, **spatial_hotspots(site, now - WINDOWS[window], now, cell_m, cell_px, include_dismissed)}

    @r.get("/api/analytics/forecast")
    def forecast(zone: str = "", camera: str = "", horizon_min: float = 30.0,
                 p: Principal = Depends(need("alert.read"))):
        """Zone occupancy / queue-length forecast (a prediction). `zone` is a zone id, optionally "camera:zone";
        without it every zone is returned. 404 when "congestion_forecast": {"enabled": false}."""
        from .congestion import LABEL, MODEL, NOTE
        cf = getattr(site, "congestion", None)
        if cf is None:
            raise HTTPException(404)
        if not 1 <= horizon_min <= float(cf.cfg["max_horizon_min"]):
            raise HTTPException(400, f"horizon_min must be 1-{cf.cfg['max_horizon_min']:g}")
        if ":" in zone and not camera:
            camera, zone = zone.split(":", 1)
        rows = cf.all(horizon_min, camera, zone)
        if (zone or camera) and not rows:
            raise HTTPException(404, f"unknown zone {camera + ':' if camera else ''}{zone}")
        return {"horizon_min": horizon_min, "label": LABEL, "model": MODEL, "note": NOTE,
                "emit_events": bool(cf.cfg.get("emit_events")), "min_skill": cf.cfg["min_skill"],
                "coverage": cf.cfg["coverage"], "generated_at": time.time(), "zones": rows}

    @r.get("/api/analytics/anomaly")
    def anomaly(p: Principal = Depends(need("alert.read"))):
        an = getattr(site, "anomaly", None)
        if an is None:
            return {"enabled": False, "note": "Trajectory-anomaly corroboration is off (\"anomaly\": {\"enabled\": "
                                              "true} in the site config turns it on). Corroboration only."}
        return an.status()

    @r.get("/api/analytics/objects")
    def objects(p: Principal = Depends(need("alert.read"))):
        an = getattr(site, "objects", None)
        if an is None:
            return {"enabled": False, "note": "Unattended-object analysis is off (\"objects\": {\"enabled\": true} "
                                              "adds the bag classes to the detector)."}
        return {"enabled": True, **an.summary()}

    return r
