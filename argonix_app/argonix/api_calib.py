"""API for automatic calibration (operator-approved) and the automatic bird's-eye floor map."""
from __future__ import annotations

import math

import cv2
from fastapi import APIRouter, Body, Depends, HTTPException, Response

from .autocalib import AutoCalibrator
from .birdseye import SiteMap, _calibration, as_h, birdseye, ground_origin


def calib_router(site, need) -> APIRouter:
    r = APIRouter()
    calibrators: dict[str, AutoCalibrator] = {}
    smap = SiteMap(site)
    site.site_map = smap  # lets other components (e.g. the WS track feed) reuse the same layout
    smap.start()

    def worker(cam: str):
        w = next((w for w in getattr(site, "workers", []) if w.id == cam), None)
        if not w:
            raise HTTPException(404, f"unknown camera {cam}")
        return w

    def apply_patch(cam: str, patch: dict) -> None:
        upd = getattr(site, "update_camera_config", None)
        if upd is None:
            raise HTTPException(501, "this site cannot apply live configuration changes (update_camera_config missing)")
        try:
            upd(cam, patch)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"config update rejected: {e}")
        smap.invalidate()

    # ---- auto calibration ------------------------------------------------------------------------------------
    @r.post("/api/cameras/{cam}/autocalibrate/start")
    def autocal_start(cam: str, body: dict = Body(default={}), p=Depends(need("calibration.approve"))):
        w = worker(cam)
        body = body or {}
        kw = {}
        for k in ("person_height_m", "hfov_deg", "min_samples", "min_tracks", "max_duration_s"):
            if body.get(k) is not None:
                try:
                    v = float(body[k])
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{k} must be a number")
                if not math.isfinite(v) or v <= 0:
                    raise HTTPException(400, f"{k} must be positive")
                kw[k] = int(v) if k in ("min_samples", "min_tracks") else v
        if "hfov_deg" in kw and not 10 <= kw["hfov_deg"] <= 170:
            raise HTTPException(400, "hfov_deg must be between 10 and 170")
        if cam in calibrators:
            calibrators[cam].stop()
        cal = AutoCalibrator(**kw)
        calibrators[cam] = cal
        cal.start(w)
        site.store.audit(p.username, "autocalibration_started", "", {"camera": cam, **kw})
        return cal.status()

    @r.get("/api/cameras/{cam}/autocalibrate")
    def autocal_status(cam: str, p=Depends(need("alert.read"))):
        worker(cam)
        cal = calibrators.get(cam)
        if cal is None:
            return {"state": "idle", "samples": 0, "tracks": 0, "coverage": 0.0, "estimate": None,
                    "reason": "not started", "collecting": False, "camera": cam}
        return cal.status()

    @r.post("/api/cameras/{cam}/autocalibrate/apply")
    def autocal_apply(cam: str, p=Depends(need("calibration.approve"))):
        w = worker(cam)
        cal = calibrators.get(cam)
        st = cal.status() if cal else None
        if not st or st["state"] != "ready" or not st["estimate"]:
            raise HTTPException(409, "no auto-calibration estimate is ready for this camera")
        est = dict(st["estimate"])
        est["approved_by"] = p.username
        previous = _calibration(w)
        apply_patch(cam, {"calibration": est})
        cal.stop()
        warnings = []
        if any("polygon" in z and "polygon_px" not in z for z in (getattr(w, "cfg", {}) or {}).get("zones", [])):
            warnings.append("zones defined in ground metres keep their old coordinates; review them against the new "
                            "camera-centred frame (zones drawn in image pixels are re-projected automatically)")
        site.store.audit(p.username, "calibration_applied", "", {
            "camera": cam, "method": est.get("method"), "calib_error_m": est.get("calib_error_m"),
            "params": est.get("params"), "samples": est.get("samples"), "tracks": est.get("tracks"),
            "previous_method": previous.get("method", "manual") if previous else None,
            "previous_calib_error_m": previous.get("calib_error_m") if previous else None})
        return {"applied": True, "camera": cam, "calibration": est, "warnings": warnings}

    # ---- bird's-eye / site map ---------------------------------------------------------------------------------
    @r.get("/api/cameras/{cam}/birdseye.png")
    def cam_birdseye(cam: str, px_per_m: float = 20, max_range_m: float = 40, p=Depends(need("evidence.view"))):
        w = worker(cam)
        Hm = as_h(getattr(w, "H", None))
        if Hm is None:
            raise HTTPException(409, "camera has no ground calibration yet")
        fr = getattr(w, "last_frame", None)
        if fr is None:
            raise HTTPException(503, "no frame yet")
        scale = float(getattr(w, "scale", 1.0) or 1.0)
        px_per_m = min(max(px_per_m, 2.0), 100.0)
        max_range_m = min(max(max_range_m, 2.0), 200.0)
        src = (fr.shape[1] * scale, fr.shape[0] * scale)
        try:
            img, b, ppm = birdseye(fr, Hm, px_per_m, max_range_m, scale, ground_origin(Hm, src, _calibration(w)))
        except ValueError as e:
            raise HTTPException(409, str(e))
        return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png",
                        headers={"X-Bounds-M": ",".join(f"{v:.3f}" for v in b), "X-Px-Per-M": f"{ppm:.4f}",
                                 "Cache-Control": "no-store"})

    @r.get("/api/map")
    def site_map(p=Depends(need("alert.read"))):
        return smap.info()

    @r.get("/api/map/background.png")
    def map_background(p=Depends(need("evidence.view"))):
        png, b = smap.background_png()
        return Response(png, media_type="image/png",
                        headers={"X-Bounds-M": ",".join(f"{v:.3f}" for v in b), "Cache-Control": "no-store"})

    @r.get("/api/map/heatmap")
    def map_heatmap(p=Depends(need("alert.read"))):
        return smap.heatmap()

    @r.get("/api/map/heatmap/history")
    def map_heatmap_history(hours: float = 24.0, step_s: float = 3600.0, p=Depends(need("alert.read"))):
        """Time-lapse: per-step occupancy grids (person-seconds per cell) for the last `hours` (max 7 days)."""
        if not (math.isfinite(hours) and 0 < hours <= 168):
            raise HTTPException(400, "hours must be in (0, 168]")
        if not (math.isfinite(step_s) and 60 <= step_s <= 86400):
            raise HTTPException(400, "step_s must be between 60 and 86400 seconds")
        return smap.history(hours, step_s)

    @r.put("/api/cameras/{cam}/placement")
    def set_placement(cam: str, body: dict = Body(...), p=Depends(need("zone.edit"))):
        worker(cam)
        try:
            off = [float(v) for v in body.get("offset_m", [0, 0])]
            rot = float(body.get("rotation_deg", 0.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "body must be {offset_m: [x, y], rotation_deg}")
        if len(off) != 2 or not all(math.isfinite(v) and abs(v) < 1e5 for v in off + [rot]):
            raise HTTPException(400, "offset_m must be two finite numbers and rotation_deg finite")
        placement = {"offset_m": off, "rotation_deg": rot % 360}
        persisted = hasattr(site, "update_camera_config")
        if persisted:
            apply_patch(cam, {"placement": placement})
        else:  # placement only affects the map, so an in-memory change is safe until live config updates exist
            w = worker(cam)
            w.cfg["placement"] = placement
            smap.invalidate()
        site.store.audit(p.username, "placement_changed", "", {"camera": cam, **placement, "persisted": persisted})
        return {"camera": cam, "placement": placement, "persisted": persisted}

    return r
