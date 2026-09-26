"""Camera management API: connect Argus to existing CCTV (RTSP/ONVIF/NVR channels) from the dashboard.

Sources are always returned REDACTED (rtsp://user:***@host/...). A client that sends a redacted source back
unchanged keeps the stored credentials. Every mutating call is audited with credentials redacted.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Optional

import cv2
from fastapi import APIRouter, Body, Depends, HTTPException, Response

from . import cameras
from .auth import Principal
from .pipeline import CAMERA_ID_RE, PIXEL_RULES
from .restream import restreamable
from .rules import DEFAULTS, VEHICLE_RULES, vehicle_classes
from .schedule import Schedule

ZONE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_POLY_POINTS = 64
ZONE_TYPES = ("area", "queue")
INT_PARAMS = {"max_count", "min_tracks", "min_against", "min_count", "min_people"}
UNIT_PARAMS = {"min_fraction", "min_concentration", "still_box_frac"}  # fractions in [0, 1]
# per-camera live-stream options an operator may tune for a flaky link: (min, max) accepted values in seconds
STREAM_OPTS = {"reconnect_min_s": (0.1, 60.0), "reconnect_max_s": (1.0, 300.0), "down_grace_s": (0.5, 60.0)}


def _slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:32] or "cam"
    if not base.startswith("cam"):
        base = f"cam_{base}"[:36]
    cid, i = base, 2
    while cid in taken:
        cid, i = f"{base}_{i}", i + 1
    return cid


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number")
    return float(v)


def _polygon(poly, what: str) -> list[list[float]]:
    if not isinstance(poly, list) or not 3 <= len(poly) <= MAX_POLY_POINTS:
        raise ValueError(f"{what} must be a list of 3-{MAX_POLY_POINTS} [x, y] points")
    out = []
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{what}: each point must be [x, y]")
        out.append([_num(p[0], what), _num(p[1], what)])
    return out


def validate_zones(zones: Any, calibrated: bool) -> tuple[list[dict], list[str]]:
    """Validate zones against rules.DEFAULTS. Returns (clean zones, warnings)."""
    if not isinstance(zones, list):
        raise ValueError("zones must be a list")
    clean, warnings, ids = [], [], set()
    for i, z in enumerate(zones):
        if not isinstance(z, dict):
            raise ValueError(f"zone {i} must be an object")
        zid = str(z.get("id", ""))
        if not ZONE_ID_RE.match(zid):
            raise ValueError(f"zone {i}: id must be 1-40 characters of letters, digits, '_' or '-'")
        if zid in ids:
            raise ValueError(f"duplicate zone id '{zid}'")
        ids.add(zid)
        out: dict[str, Any] = {"id": zid, "name": str(z.get("name") or zid)[:80]}
        if "polygon_px" in z:
            out["polygon_px"] = _polygon(z["polygon_px"], f"zone '{zid}' polygon_px")
        elif "polygon" in z:
            out["polygon"] = _polygon(z["polygon"], f"zone '{zid}' polygon")
            if not calibrated:
                warnings.append(f"zone '{zid}' uses ground metres but the camera is uncalibrated: it is skipped "
                                "until the camera is calibrated (draw it in pixels with polygon_px instead)")
        else:
            raise ValueError(f"zone '{zid}' needs polygon_px (image pixels) or polygon (ground metres)")
        rules = z.get("rules") or {}
        if not isinstance(rules, dict):
            raise ValueError(f"zone '{zid}': rules must be an object, e.g. {{\"intrusion\": {{}}}}")
        out["rules"] = {}
        for r, params in rules.items():
            if r not in DEFAULTS:
                raise ValueError(f"zone '{zid}': unknown rule '{r}' (known: {sorted(DEFAULTS)})")
            params = params or {}
            if not isinstance(params, dict):
                raise ValueError(f"zone '{zid}' rule '{r}': parameters must be an object")
            params = dict(params)
            rule_sched = params.pop("schedule", None)  # optional per-rule schedule (replaces the zone's)
            bad = set(params) - set(DEFAULTS[r])
            if bad:
                raise ValueError(f"zone '{zid}' rule '{r}': unknown parameter(s) {sorted(bad)} "
                                 f"(allowed: {sorted(DEFAULTS[r])})")
            clean_p = {}
            for k, v in params.items():
                v = _num(v, f"zone '{zid}' {r}.{k}")
                if k != "cos_threshold" and v < 0:
                    raise ValueError(f"zone '{zid}' {r}.{k} must be >= 0")
                if k == "cos_threshold" and not -1 <= v <= 1:
                    raise ValueError(f"zone '{zid}' {r}.cos_threshold must be in [-1, 1]")
                if k in UNIT_PARAMS and v > 1:
                    raise ValueError(f"zone '{zid}' {r}.{k} must be in [0, 1]")
                clean_p[k] = int(v) if k in INT_PARAMS else v
            if Schedule.parse(rule_sched, f"zone '{zid}' rule '{r}' schedule") is not None:
                clean_p["schedule"] = rule_sched
            out["rules"][r] = clean_p
            if not calibrated and r not in PIXEL_RULES and not (r == "running" and "speed_bh_s" in clean_p):
                warnings.append(f"zone '{zid}': rule '{r}' needs calibration (metric speed/direction) and is "
                                "skipped until the camera is calibrated")
        if z.get("direction_px") is not None:  # arrow drawn in the editor: [[x0,y0],[x1,y1]] source px
            dp = z["direction_px"]
            if (not isinstance(dp, (list, tuple)) or len(dp) != 2
                    or not all(isinstance(q, (list, tuple)) and len(q) == 2 for q in dp)):
                raise ValueError(f"zone '{zid}': direction_px must be [[x0, y0], [x1, y1]]")
            out["direction_px"] = [[_num(q[0], "direction_px"), _num(q[1], "direction_px")] for q in dp]
            if out["direction_px"][0] == out["direction_px"][1]:
                raise ValueError(f"zone '{zid}': direction_px points must differ")
        if "wrong_way" in out["rules"]:
            if not z.get("direction") and not out.get("direction_px"):
                raise ValueError(f"zone '{zid}': wrong_way needs a permitted 'direction' [dx, dy] (ground frame) "
                                 "or 'direction_px' [[x0,y0],[x1,y1]] (image arrow)")
        if z.get("direction") is not None:
            d = z["direction"]
            if not isinstance(d, (list, tuple)) or len(d) != 2:
                raise ValueError(f"zone '{zid}': direction must be [dx, dy]")
            out["direction"] = [_num(d[0], "direction"), _num(d[1], "direction")]
            if out["direction"] == [0.0, 0.0]:
                raise ValueError(f"zone '{zid}': direction must be non-zero")
        ztype = z.get("type", "area") or "area"
        if ztype not in ZONE_TYPES:
            raise ValueError(f"zone '{zid}': type must be one of {list(ZONE_TYPES)}")
        if ztype != "area" or "queue_long" in out["rules"]:
            out["type"] = "queue"
        for key in ("service_line_px", "service_line"):  # queue service point: [[x0, y0], [x1, y1]]
            if z.get(key) is None:
                continue
            sl = z[key]
            if (not isinstance(sl, (list, tuple)) or len(sl) != 2
                    or not all(isinstance(q, (list, tuple)) and len(q) == 2 for q in sl)):
                raise ValueError(f"zone '{zid}': {key} must be [[x0, y0], [x1, y1]]")
            out[key] = [[_num(q[0], key), _num(q[1], key)] for q in sl]
            if out[key][0] == out[key][1]:
                raise ValueError(f"zone '{zid}': {key} points must differ")
            if key == "service_line" and not calibrated:
                warnings.append(f"zone '{zid}': service_line is in ground metres and needs calibration")
        crit = z.get("criticality", 1)
        if isinstance(crit, bool) or not isinstance(crit, int) or not 1 <= crit <= 3:
            raise ValueError(f"zone '{zid}': criticality must be an integer 1-3")
        out["criticality"] = crit
        if z.get("site_zone"):
            if not ZONE_ID_RE.match(str(z["site_zone"])):
                raise ValueError(f"zone '{zid}': invalid site_zone")
            out["site_zone"] = str(z["site_zone"])
        if Schedule.parse(z.get("schedule"), f"zone '{zid}' schedule") is not None:
            out["schedule"] = z["schedule"]
        clean.append(out)
    return clean, warnings


def validate_stream_opts(body: dict) -> dict:
    """Pick and validate the reconnect/outage options (STREAM_OPTS) present in a request body."""
    out = {}
    for k, (lo, hi) in STREAM_OPTS.items():
        if body.get(k) is not None:
            v = _num(body[k], k)
            if not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo:g} and {hi:g} seconds")
            out[k] = v
    if out.get("reconnect_min_s", 0) > out.get("reconnect_max_s", float("inf")):
        raise ValueError("reconnect_min_s must not exceed reconnect_max_s")
    return out


def validate_capture(capture: Any) -> Optional[dict]:
    """Webcam mode request {"width", "height", "fps", "fourcc"} (see cameras.apply_capture_settings); None clears it."""
    if capture is None:
        return None
    if not isinstance(capture, dict) or set(capture) - {"width", "height", "fps", "fourcc"}:
        raise ValueError('capture must be an object with width, height, fps and/or fourcc (e.g. "MJPG")')
    out: dict[str, Any] = {}
    for k, (lo, hi) in {"width": (160, 7680), "height": (120, 4320), "fps": (1, 240)}.items():
        if capture.get(k) is not None:
            v = _num(capture[k], f"capture.{k}")
            if not lo <= v <= hi:
                raise ValueError(f"capture.{k} must be between {lo} and {hi}")
            out[k] = int(v) if k != "fps" else v
    if capture.get("fourcc") is not None:
        f = str(capture["fourcc"])
        if len(f) != 4 or not f.isalnum():
            raise ValueError('capture.fourcc must be 4 letters or digits, e.g. "MJPG" or "YUY2"')
        out["fourcc"] = f
    return out or None


def validate_masks(masks: Any, key: str = "masks_px", what: str = "mask") -> list[list[list[float]]]:
    if not isinstance(masks, list):
        raise ValueError(f"{key} must be a list of polygons")
    if len(masks) > 64:
        raise ValueError(f"{key}: at most 64 polygons")
    return [_polygon(m, f"{what} {i}") for i, m in enumerate(masks)]


def validate_privacy_masks(masks: Any) -> list[list[list[float]]]:
    """Privacy masks (source px): painted black before anything else sees the frame."""
    return validate_masks(masks, "privacy_masks_px", "privacy mask")


def camera_router(site, need: Callable[[str], Callable]) -> APIRouter:
    r = APIRouter()

    def cam_cfg(cid: str) -> dict:
        c = site.camera_config(cid)
        if c is None:
            raise HTTPException(404, f"unknown camera {cid}")
        return c

    def view(c: dict) -> dict:
        w = site.worker(c["id"])
        enabled = bool(c.get("enabled", True))
        st = w.stats if w else {}
        size = w.source_size if w else None
        return {"id": c["id"], "name": c.get("name", c["id"]), "source": cameras.redact(c["source"]),
                "main_source": cameras.redact(c["main_source"]) if c.get("main_source") else "",
                "enabled": enabled, "status": st.get("status", "disabled" if not enabled else "stopped"),
                "health": st.get("health"), "fps": st.get("fps"), "error": cameras.redact(st.get("error", "")),
                "width": size[0] if size else None, "height": size[1] if size else None,
                "calibrated": (w.H is not None) if w else bool(c.get("calibration")),
                "zones": c.get("zones", []), "masks_px": c.get("masks_px", []), "lines": c.get("lines", []),
                "privacy_masks_px": c.get("privacy_masks_px", []),
                "vehicles_enabled": bool(w.vehicle_labels) if w else False,
                **{k: c[k] for k in STREAM_OPTS if k in c}}

    def apply(cid: str, patch: dict) -> None:
        try:
            site.update_camera_config(cid, patch)
        except KeyError:
            raise HTTPException(404, f"unknown camera {cid}")
        except ValueError as e:
            raise HTTPException(400, str(e))

    def clean_source(src, old=None) -> str:
        try:
            return cameras.validate_source(cameras.unredact(str(src), old))
        except ValueError as e:
            raise HTTPException(400, str(e))

    def clean_main_source(src, old=None):
        """Optional full-resolution stream for WebRTC live view and main-stream recording (restream). ''/None clears
        it (the restreamer then uses `source`)."""
        if src is None or not str(src).strip():
            return None
        s = clean_source(src, old)
        ok, why = restreamable(s)
        if not ok:
            raise HTTPException(400, f"main_source: {why}")
        return s

    @r.get("/api/cameras")
    def list_cameras(p: Principal = Depends(need("alert.read"))):
        return [view(c) for c in site.cfg["cameras"]]

    @r.post("/api/cameras")
    def add_camera(body: dict = Body(...), p: Principal = Depends(need("settings.edit"))):
        if "source" not in body:
            raise HTTPException(400, "source is required")
        name = str(body.get("name") or "").strip()[:80]
        taken = {c["id"] for c in site.cfg["cameras"]}
        cid = str(body.get("id") or "").strip() or _slug(name or "cam", taken)
        if not CAMERA_ID_RE.match(cid):
            raise HTTPException(400, "id must be 1-40 characters of letters, digits, '_' or '-'")
        if cid in taken:
            raise HTTPException(409, f"camera '{cid}' already exists")
        cfg: dict[str, Any] = {"id": cid, "name": name or cid, "source": clean_source(body["source"]),
                               "enabled": bool(body.get("enabled", True))}
        try:
            if body.get("zones") is not None:
                cfg["zones"], _ = validate_zones(body["zones"], calibrated=False)
            if body.get("masks_px") is not None:
                cfg["masks_px"] = validate_masks(body["masks_px"])
            if body.get("privacy_masks_px") is not None:
                cfg["privacy_masks_px"] = validate_privacy_masks(body["privacy_masks_px"])
            cfg.update(validate_stream_opts(body))
            if body.get("capture") is not None:
                cfg["capture"] = validate_capture(body["capture"])
            main = clean_main_source(body.get("main_source"))
            if main:
                cfg["main_source"] = main
            site.add_camera(cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        site.store.audit(p.username, "camera_add", "", {"camera": cid, "name": cfg["name"],
                                                        "source": cameras.redact(cfg["source"]),
                                                        "main_source": cameras.redact(cfg.get("main_source", "")),
                                                        "enabled": cfg["enabled"]})
        return view(site.camera_config(cid))

    # fixed paths before /{cid} routes (no clash today, but keep the order explicit)
    @r.post("/api/cameras/test")
    def test_source(body: dict = Body(...), p: Principal = Depends(need("settings.edit"))):
        src = body.get("source")
        old = None
        if body.get("id"):
            old = cam_cfg(str(body["id"]))["source"]
            src = src or old
        if not src:
            raise HTTPException(400, "source is required")
        src = clean_source(src, old)
        try:
            timeout = min(30.0, max(1.0, _num(body.get("timeout_s", 10), "timeout_s")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        res = cameras.probe_source(src, timeout_s=timeout)
        site.store.audit(p.username, "camera_test", "", {"source": cameras.redact(src), "ok": res["ok"],
                                                         "error": res["error"]})
        return res

    @r.get("/api/cameras/discover")
    def discover(timeout_s: float = 3.0, p: Principal = Depends(need("settings.edit"))):
        devices = cameras.discover(min(10.0, max(0.5, timeout_s)))
        site.store.audit(p.username, "camera_discover", "", {"found": len(devices)})
        return {"devices": devices, "note": "ONVIF WS-Discovery only reaches devices on this LAN segment/VLAN "
                                            "with discovery enabled; an empty list does not mean no cameras."}

    @r.get("/api/cameras/templates")
    def templates(p: Principal = Depends(need("settings.edit"))):
        return cameras.templates()

    @r.put("/api/cameras/{cid}")
    def update_camera(cid: str, body: dict = Body(...), p: Principal = Depends(need("settings.edit"))):
        c = cam_cfg(cid)
        allowed = {"name", "source", "main_source", "enabled", "capture", *STREAM_OPTS}
        bad = set(body) - allowed
        if bad:
            raise HTTPException(400, f"unsupported field(s) {sorted(bad)}; use /zones and /masks for geometry")
        patch: dict[str, Any] = {}
        if "name" in body:
            patch["name"] = str(body["name"] or cid).strip()[:80] or cid
        if "enabled" in body:
            patch["enabled"] = bool(body["enabled"])
        if "source" in body:
            new_src = clean_source(body["source"], c["source"])
            if new_src != c["source"]:
                patch["source"] = new_src
        if "main_source" in body:
            new_main = clean_main_source(body["main_source"], c.get("main_source"))
            if new_main != c.get("main_source"):
                patch["main_source"] = new_main   # None deletes the key (deep_merge)
        try:
            opts = validate_stream_opts({**{k: c[k] for k in STREAM_OPTS if k in c}, **body})
        except ValueError as e:
            raise HTTPException(400, str(e))
        patch.update({k: v for k, v in opts.items() if k in body and c.get(k) != v})
        if "capture" in body:
            try:
                cap = validate_capture(body["capture"])
            except ValueError as e:
                raise HTTPException(400, str(e))
            if cap != c.get("capture"):
                patch["capture"] = cap   # None deletes the key (deep_merge)
        if patch:
            apply(cid, patch)
        audit = {k: v for k, v in patch.items() if k not in ("source", "main_source")}
        for k in ("source", "main_source"):
            if k in patch:
                audit[k] = cameras.redact(patch[k] or "")
        site.store.audit(p.username, "camera_update", "", {"camera": cid, **audit})
        return view(site.camera_config(cid))

    @r.delete("/api/cameras/{cid}")
    def delete_camera(cid: str, p: Principal = Depends(need("settings.edit"))):
        c = cam_cfg(cid)
        src = cameras.redact(c["source"])
        site.remove_camera(cid)
        site.store.audit(p.username, "camera_delete", "", {"camera": cid, "source": src})
        return {"ok": True, "id": cid}

    @r.put("/api/cameras/{cid}/zones")
    def put_zones(cid: str, body: dict = Body(...), p: Principal = Depends(need("zone.edit"))):
        c = cam_cfg(cid)
        try:
            zones, warnings = validate_zones(body.get("zones"), calibrated=bool(c.get("calibration")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not vehicle_classes(site.cfg):
            warnings += [f"zone '{z['id']}': rule '{r}' needs vehicle detection (site config \"vehicles\": "
                         "{\"enabled\": true}); it sees no vehicles until then"
                         for z in zones for r in z["rules"] if r in VEHICLE_RULES]
        apply(cid, {"zones": zones})
        site.store.audit(p.username, "camera_zones", "", {"camera": cid, "zones": [z["id"] for z in zones],
                                                          "rules": {z["id"]: sorted(z["rules"]) for z in zones}})
        return {**view(site.camera_config(cid)), "warnings": warnings}

    @r.put("/api/cameras/{cid}/masks")
    def put_masks(cid: str, body: dict = Body(...), p: Principal = Depends(need("zone.edit"))):
        cam_cfg(cid)
        try:
            masks = validate_masks(body.get("masks_px"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        apply(cid, {"masks_px": masks})
        site.store.audit(p.username, "camera_masks", "", {"camera": cid, "masks": len(masks)})
        return view(site.camera_config(cid))

    @r.put("/api/cameras/{cid}/privacy_masks")
    def put_privacy_masks(cid: str, body: dict = Body(...), p: Principal = Depends(need("zone.edit"))):
        """Privacy masks are blanked on the frame itself: detection, recording, evidence, live view and exports
        never see those pixels again (earlier recordings are not rewritten)."""
        cam_cfg(cid)
        try:
            masks = validate_privacy_masks(body.get("privacy_masks_px"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        apply(cid, {"privacy_masks_px": masks})
        site.store.audit(p.username, "camera_privacy_masks", "", {"camera": cid, "privacy_masks": len(masks)})
        return view(site.camera_config(cid))

    @r.post("/api/schedule/check")
    def schedule_check(body: dict = Body(...), p: Principal = Depends(need("alert.read"))):
        """Validate a schedule object for the editors and say whether it is active right now (server clock)."""
        try:
            sch = Schedule.parse(body.get("schedule"), "schedule")
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if sch is None:
            return {"ok": True, "schedule": None, "active": True, "note": "no schedule: always armed"}
        return {"ok": True, **sch.status(time.time())}

    @r.get("/api/cameras/{cid}/snapshot.jpg")
    def snapshot(cid: str, p: Principal = Depends(need("alert.read"))):
        if not (p.can("evidence.view") or p.can("zone.edit")):  # zone editors (admin) must see what they draw on
            raise HTTPException(403, "permission 'evidence.view' or 'zone.edit' required")
        cam_cfg(cid)
        w = site.worker(cid)
        snap = w.snapshot() if w else None
        if snap is None:
            raise HTTPException(503, "no frame yet")
        frame, _, _, scale = snap
        if scale != 1.0 and w.source_size:  # hand the editor source-pixel geometry (zones/masks are in source px)
            frame = cv2.resize(frame, w.source_size)
        return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                        media_type="image/jpeg", headers={"Cache-Control": "no-store",
                                                          "X-Coordinate-Space": "source-px"})

    return r
