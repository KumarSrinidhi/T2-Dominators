"""Bird's-eye warp, site composite/placement, occupancy heatmap and the calibration/map API."""
import functools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from argonix import api_calib
from argonix.api_calib import calib_router
from argonix.autocalib import AutoCalibrator, image_to_ground_h, pixel_from_world
from argonix.birdseye import SiteMap, birdseye, camera_layout, site_xy, visible_polygon
from argonix.geometry import apply_h, homography_from_points

from .test_autocalib import HH, W, StreamWorker, synth_frames

ROOT = Path(__file__).resolve().parents[1]
F, TILT, CAM_H = 700.0, 35.0, 8.0
SCALE = 2.0


def camera_frame(marker_xy=(0.0, 10.0), half=1.0):
    """Analytics-size frame of a grey floor with a white 2x2 m square at marker_xy, seen by the known camera."""
    Hm = image_to_ground_h(F, math.radians(TILT), CAM_H, W / 2, HH / 2)
    ppm = 20
    top = np.full((60 * ppm, 60 * ppm, 3), 90, np.uint8)  # ground texture: x in [-30, 30], y in [0, 60]
    cx, cy = marker_xy
    cv2.rectangle(top, (int((cx - half + 30) * ppm), int((60 - cy - half) * ppm)),
                  (int((cx + half + 30) * ppm), int((60 - cy + half) * ppm)), (255, 255, 255), -1)
    T = np.array([[ppm, 0, 30 * ppm], [0, -ppm, 60 * ppm], [0, 0, 1.0]])  # ground -> texture px
    S = np.diag([SCALE, SCALE, 1.0])  # analytics px -> source px
    M = np.linalg.inv(T @ Hm @ S)  # texture px -> analytics px
    return cv2.warpPerspective(top, M, (int(W / SCALE), int(HH / SCALE))), Hm


def test_birdseye_puts_ground_marker_at_its_metric_position():
    frame, Hm = camera_frame((1.5, 10.0))
    img, b, ppm = birdseye(frame, Hm, px_per_m=20, max_range_m=40, scale=SCALE, origin=(0.0, 0.0))
    assert img.shape[2] == 4 and ppm == 20
    white = (img[..., 0] > 200) & (img[..., 3] > 0)
    rows, cols = np.nonzero(white)
    x = b[0] + (cols.mean() + 0.5) / ppm
    y = b[3] - (rows.mean() + 0.5) / ppm
    assert math.dist((x, y), (1.5, 10.0)) < 0.3
    # nothing beyond range or behind the camera is drawn
    rr, cc = np.nonzero(img[..., 3] > 0)
    xs, ys = b[0] + (cc + 0.5) / ppm, b[3] - (rr + 0.5) / ppm
    assert np.hypot(xs, ys).max() <= 40.5 and ys.min() > 0


def test_visible_polygon_is_clipped_below_horizon_and_in_range():
    Hm = image_to_ground_h(F, math.radians(10), 3.0, W / 2, HH / 2)  # horizon inside the image
    poly = np.array(visible_polygon(Hm, (W, HH), max_range_m=25, origin=(0, 0)))
    assert len(poly) > 3
    assert np.hypot(poly[:, 0], poly[:, 1]).max() <= 25.001 and poly[:, 1].min() > 0


class FakeStore:
    def __init__(self):
        self.audits = []

    def audit(self, actor, action, alert_id="", details=None):
        self.audits.append((actor, action, alert_id, details))


class FakeWorker(StreamWorker):
    def __init__(self, cam_id, site, frames=(), calibration=None):
        super().__init__(list(frames), cam_id, SCALE)
        self.site, self.cfg = site, {"id": cam_id, "calibration": calibration or {}}
        self.H, self.calib_info = None, {}
        self.fixed_dets = None
        if calibration:
            self.set_calibration(calibration)

    def set_calibration(self, cal):
        self.cfg["calibration"] = cal
        self.H = homography_from_points([tuple(p) for p in cal["image_points"]], [tuple(p) for p in cal["ground_points"]])
        self.calib_info = {"calib_error_m": cal.get("calib_error_m", 1.0)}

    @property
    def last_dets(self):
        if self.fixed_dets is not None:
            return self.fixed_dets
        return StreamWorker.last_dets.fget(self)


class FakeSite:
    def __init__(self):
        self.cfg, self.store, self.workers, self.patches = {"map": {"width_m": 30, "height_m": 20}}, FakeStore(), [], []

    def update_camera_config(self, cam, patch):
        self.patches.append((cam, patch))
        w = next(w for w in self.workers if w.id == cam)
        if "calibration" in patch:
            w.set_calibration(patch["calibration"])
        w.cfg.update({k: v for k, v in patch.items() if k != "calibration"})


def auto_cal():
    """The calibration dict the auto-calibrator would produce for the known camera (exact)."""
    Hm = image_to_ground_h(F, math.radians(TILT), CAM_H, W / 2, HH / 2)
    img = [(128, 684), (1152, 684), (1152, 200), (128, 200)]
    return {"image_points": [list(p) for p in img], "ground_points": [list(apply_h(Hm.tolist(), p)) for p in img],
            "calib_error_m": 1.0, "method": "auto-pedestrian", "params": {"image_size": [W, HH]}}


def make_site(n=2, uncalibrated=True):
    site = FakeSite()
    for i in range(n):
        w = FakeWorker(f"cam{i + 1}", site, calibration=auto_cal())
        w.last_frame = camera_frame()[0]
        site.workers.append(w)
    if uncalibrated:
        site.workers.append(FakeWorker("cam_nocal", site))
    return site


def test_default_layout_places_overlapping_cameras_side_by_side_and_skips_uncalibrated():
    site = make_site(2)
    lay = camera_layout(site)
    a, b = np.array(lay["cam1"]["fov_site"]), np.array(lay["cam2"]["fov_site"])
    assert a[:, 0].max() < b[:, 0].min()  # no overlap
    assert lay["cam1"]["placement_source"] == "default"
    assert lay["cam_nocal"]["placement"] is None and "no ground calibration" in lay["cam_nocal"]["reason"]
    # explicit placement is honoured and site_xy agrees with it
    site.workers[1].cfg["placement"] = {"offset_m": [100.0, 5.0], "rotation_deg": 90}
    lay = camera_layout(site)
    assert lay["cam2"]["placement_source"] == "config"
    x, y = site_xy(site.workers[1], (0.0, 10.0))
    assert (x, y) == pytest.approx((90.0, 5.0), abs=1e-6)  # (0,10) rotated 90 deg CCW -> (-10, 0), then offset


def test_background_png_is_cached_and_matches_bounds():
    site = make_site(2)
    smap = SiteMap(site, refresh_s=10)
    png1, b = smap.background_png()
    png2, _ = smap.background_png()
    assert png1 is png2  # served from cache within refresh_s
    img = cv2.imdecode(np.frombuffer(png1, np.uint8), cv2.IMREAD_COLOR)
    ppm = smap.ppm(b)
    assert img.shape[1] == math.ceil((b[2] - b[0]) * ppm) and img.shape[0] == math.ceil((b[3] - b[1]) * ppm)
    info = smap.info()
    assert info["bounds_m"] == pytest.approx(list(b)) and info["background_url"] == "/api/map/background.png"
    cams = {c["id"]: c for c in info["cameras"]}
    assert cams["cam1"]["calibrated"] and cams["cam1"]["method"] == "auto-pedestrian"
    assert not cams["cam_nocal"]["calibrated"] and cams["cam_nocal"]["reason"]


def test_heatmap_accumulates_in_site_coordinates_and_decays():
    site = make_site(1, uncalibrated=False)
    w = site.workers[0]
    u, v, _ = pixel_from_world(2.0, 10.0, 0.0, F, math.radians(TILT), CAM_H, W / 2, HH / 2)
    fu, fv = float(u) / SCALE, float(v) / SCALE
    w.fixed_dets = [(1, (fu - 10, fv - 60, fu + 10, fv), "person", 0.9)]
    smap = SiteMap(site, half_life_s=10)
    smap.tick(now=100.0)
    smap.tick(now=101.0)
    hm = smap.heatmap()
    grid = np.array(hm["values"]).reshape(hm["height"], hm["width"])
    iy, ix = np.unravel_index(grid.argmax(), grid.shape)
    cell_centre = (hm["origin_m"][0] + (ix + 0.5) * hm["cell_m"], hm["origin_m"][1] + (iy + 0.5) * hm["cell_m"])
    assert math.dist(cell_centre, site_xy(w, (2.0, 10.0))) <= hm["cell_m"]
    peak = grid.max()
    w.fixed_dets = []
    smap.tick(now=111.0)  # 10 s = one half-life (dt capped at 5 s per tick -> tick twice)
    smap.tick(now=116.0)
    assert smap.heatmap()["max"] == pytest.approx(peak * 0.5 ** ((5 + 5) / 10), rel=0.02)


# ---- API --------------------------------------------------------------------------------------------------------
class P:
    def __init__(self, name):
        self.username = name


def api_client(site, denied=frozenset()):
    def need(perm):
        def dep():
            if perm in denied:
                raise HTTPException(403, f"permission '{perm}' required")
            return P("tester")
        return dep
    app = FastAPI()
    app.include_router(calib_router(site, need))
    return TestClient(app)


def test_autocalibrate_api_flow_requires_approval(monkeypatch):
    monkeypatch.setattr(api_calib, "AutoCalibrator", functools.partial(AutoCalibrator, poll_s=0.0, per_track_interval_s=0.0))
    site = FakeSite()
    w = FakeWorker("cam1", site, frames=synth_frames(F, TILT, CAM_H, seed=1))
    site.workers.append(w)
    c = api_client(site)
    assert c.get("/api/cameras/cam1/autocalibrate").json()["state"] == "idle"
    assert c.post("/api/cameras/cam1/autocalibrate/apply").status_code == 409  # nothing to apply yet
    assert c.post("/api/cameras/cam1/autocalibrate/start", json={"hfov_deg": 500}).status_code == 400
    assert c.post("/api/cameras/cam1/autocalibrate/start", json={}).json()["state"] == "collecting"
    deadline = time.time() + 60
    st = {}
    while time.time() < deadline:
        st = c.get("/api/cameras/cam1/autocalibrate").json()
        if st["state"] in ("ready", "failed"):
            break
        time.sleep(0.05)
    assert st["state"] == "ready", st
    assert w.H is None  # an estimate is NOT applied until an operator approves it
    r = c.post("/api/cameras/cam1/autocalibrate/apply")
    assert r.status_code == 200 and r.json()["applied"]
    assert site.patches[-1][0] == "cam1" and site.patches[-1][1]["calibration"]["method"] == "auto-pedestrian"
    assert w.H is not None and w.cfg["calibration"]["approved_by"] == "tester"
    assert [a[1] for a in site.store.audits] == ["autocalibration_started", "calibration_applied"]
    assert c.get("/api/cameras/nope/autocalibrate").status_code == 404


def test_calibration_permissions_enforced():
    site = make_site(1, uncalibrated=False)
    c = api_client(site, denied={"calibration.approve", "zone.edit"})
    assert c.post("/api/cameras/cam1/autocalibrate/start").status_code == 403
    assert c.post("/api/cameras/cam1/autocalibrate/apply").status_code == 403
    assert c.put("/api/cameras/cam1/placement", json={"offset_m": [1, 2], "rotation_deg": 0}).status_code == 403


def test_map_endpoints_and_placement():
    site = make_site(2)
    c = api_client(site)
    m = c.get("/api/map").json()
    assert set(m) >= {"bounds_m", "px_per_m", "cameras", "background_url", "updated_at"}
    cams = {x["id"]: x for x in m["cameras"]}
    assert cams["cam1"]["fov_polygon_m"] and cams["cam1"]["calib_error_m"] == 1.0
    r = c.get("/api/map/background.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    hm = c.get("/api/map/heatmap").json()
    assert hm["width"] * hm["height"] == len(hm["values"]) and hm["cell_m"] == 0.5
    r = c.get("/api/cameras/cam1/birdseye.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG" and len(r.headers["x-bounds-m"].split(",")) == 4
    assert c.get("/api/cameras/cam_nocal/birdseye.png").status_code == 409
    assert c.put("/api/cameras/cam2/placement", json={"offset_m": ["a", 2]}).status_code == 400
    r = c.put("/api/cameras/cam2/placement", json={"offset_m": [50, 0], "rotation_deg": -90})
    assert r.status_code == 200 and r.json()["placement"]["rotation_deg"] == 270
    cams = {x["id"]: x for x in c.get("/api/map").json()["cameras"]}
    assert cams["cam2"]["placement"] == {"offset_m": [50.0, 0.0], "rotation_deg": 270.0}
    assert cams["cam2"]["placement_source"] == "config"
    assert site.store.audits[-1][1] == "placement_changed"


def test_server_includes_map_routes_for_real_site(tmp_path):
    from argonix.pipeline import Site
    from argonix.server import create_app
    from .test_server import FakeBackend
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["record"] = False
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    c = TestClient(create_app(site, open_demo=True))
    m = c.get("/api/map").json()
    assert {x["id"] for x in m["cameras"]} == {w.id for w in site.workers}
    assert all(x["calibrated"] and x["method"] == "manual" for x in m["cameras"])
    assert c.get("/api/map/background.png").status_code == 200
    assert c.get("/api/map/heatmap").status_code == 200
