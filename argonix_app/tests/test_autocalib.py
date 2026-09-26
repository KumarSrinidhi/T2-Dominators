"""Auto-calibration from walking people, checked against SYNTHETIC ground truth (known f, tilt, height)."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from argonix.autocalib import AutoCalibrator, calibrate_video, image_to_ground_h, pixel_from_world
from argonix.geometry import apply_h, homography_from_points

ROOT = Path(__file__).resolve().parents[1]
W, HH = 1280, 720


def synth_frames(f=700.0, tilt_deg=35.0, h=8.0, n_tracks=30, seed=0, scale=2.0, noise_px=1.5, y_range=(3, 30),
                 x_range=(-10, 10), lateral_only=False, dt=0.2, dur=20.0, standing=False):
    """People = vertical 1.70 +/- 0.09 m segments (0.5 m wide) walking on flat ground, projected by a known camera.
    Returns [(t, dets)] with boxes in ANALYTICS pixels (source px / scale)."""
    rng = np.random.default_rng(seed)
    th = math.radians(tilt_deg)
    cx, cy = W / 2, HH / 2
    people = []
    for k in range(n_tracks):
        ang = rng.choice([0.0, math.pi]) if lateral_only else rng.uniform(0, 2 * math.pi)
        people.append(dict(tid=k + 1, t0=k * 2.0, x=rng.uniform(*x_range), y=rng.uniform(*y_range),
                           vx=0.0 if standing else 1.3 * math.cos(ang), vy=0.0 if standing else 1.3 * math.sin(ang),
                           ph=rng.normal(1.70, 0.09)))
    frames = []
    t_end = max(p["t0"] for p in people) + dur
    for t in np.arange(0.0, t_end, dt):
        dets = []
        for p in people:
            if not p["t0"] <= t < p["t0"] + dur:
                continue
            X, Y = p["x"] + p["vx"] * (t - p["t0"]), p["y"] + p["vy"] * (t - p["t0"])
            uf, vf, ok1 = pixel_from_world(X, Y, 0.0, f, th, h, cx, cy)
            _, vh, ok2 = pixel_from_world(X, Y, p["ph"], f, th, h, cx, cy)
            if not (ok1 and ok2):
                continue
            zc = Y * math.cos(th) + h * math.sin(th)
            bw = f * 0.5 / zc
            n = rng.normal(0, noise_px, 4)
            box = (float(uf) - bw / 2 + n[0], float(vh) + n[1], float(uf) + bw / 2 + n[2], float(vf) + n[3])
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > HH:
                continue
            dets.append((p["tid"], tuple(v / scale for v in box), "person", 0.9))
        frames.append((float(t), dets))
    return frames


def run_offline(frames, scale=2.0, **kw):
    cal = AutoCalibrator(**kw)
    cal.start(None)
    for t, dets in frames:
        cal.add(dets, scale, (W, HH), t)
    return cal.finish()


def ground_err_at(est, f, tilt_deg, h, gx, gy):
    """Distance between the true ground point and where the estimated calibration puts its true pixel."""
    u, v, _ = pixel_from_world(gx, gy, 0.0, f, math.radians(tilt_deg), h, W / 2, HH / 2)
    He = homography_from_points([tuple(p) for p in est["image_points"]], [tuple(p) for p in est["ground_points"]])
    return math.dist(apply_h(He, (float(u), float(v))), (gx, gy))


def test_recovers_known_camera_without_lens_info():
    f, tilt, h = 700.0, 35.0, 8.0
    st = run_offline(synth_frames(f, tilt, h, seed=1))
    assert st["state"] == "ready", st["reason"]
    est = st["estimate"]
    p = est["params"]
    assert abs(p["tilt_deg"] - tilt) < 3.0
    assert abs(p["height_m"] - h) / h < 0.15
    err10 = ground_err_at(est, f, tilt, h, 2.0, 10.0)
    assert err10 < 1.0
    # honesty: the stated error is conservative (>= 0.5 m floor) and covers the true error at the median distance
    D = est["uncertainty"]["median_foot_distance_m"]
    assert est["calib_error_m"] >= 0.5
    assert ground_err_at(est, f, tilt, h, 0.0, D) <= est["calib_error_m"]
    # existing config format + metadata
    assert len(est["image_points"]) == 4 and len(est["ground_points"]) == 4
    assert est["method"] == "auto-pedestrian" and est["samples"] >= 150 and est["tracks"] >= 8
    assert all(v < HH for _, v in est["image_points"]) and min(v for _, v in est["image_points"]) > p["horizon_y_px"]
    assert est["walking_speed_check"]["median_mps"] == pytest.approx(1.3, abs=0.3)
    json.dumps(est)  # serialisable for the config file
    print(f"\nsynthetic f=700 tilt=35 h=8: est f={p['f_px']} tilt={p['tilt_deg']} h={p['height_m']} "
          f"err@10m={err10:.2f} m stated={est['calib_error_m']} m")


def test_low_tilt_needs_lens_fov_and_then_is_accurate():
    f, tilt, h = 1400.0, 12.0, 3.5
    frames = synth_frames(f, tilt, h, seed=2)
    st = run_offline(frames)
    # people's heights barely constrain f at low tilt: the fit must refuse rather than guess
    assert st["state"] == "failed" and "HFOV" in st["reason"]
    assert st["rejected_fit"] and "tilt_deg" in st["rejected_fit"]
    hfov = math.degrees(2 * math.atan(W / 2 / f))
    st = run_offline(frames, hfov_deg=hfov)
    assert st["state"] == "ready", st["reason"]
    p = st["estimate"]["params"]
    assert abs(p["tilt_deg"] - tilt) < 3.0 and abs(p["height_m"] - h) / h < 0.15
    err10 = ground_err_at(st["estimate"], f, tilt, h, 2.0, 10.0)
    assert err10 < 1.0
    print(f"\nsynthetic f=1400 tilt=12 h=3.5 with HFOV: est tilt={p['tilt_deg']} h={p['height_m']} err@10m={err10:.2f} m")


def test_fails_with_too_few_samples():
    st = run_offline(synth_frames(n_tracks=4, seed=3))
    assert st["state"] == "failed" and st["estimate"] is None
    assert "not enough data" in st["reason"] and "tracks" in st["reason"]


def test_fails_without_vertical_coverage():
    # everyone walks sideways at ~12 m: plenty of samples but no spread of foot rows
    st = run_offline(synth_frames(y_range=(12, 12.5), lateral_only=True, x_range=(-4, 4), n_tracks=30, seed=4))
    assert st["state"] == "failed" and "span" in st["reason"] and st["samples"] >= 150


def test_one_loiterer_cannot_dominate():
    cal = AutoCalibrator()
    cal.start(None)
    for t, dets in synth_frames(n_tracks=1, standing=True, dur=600, seed=5):
        cal.add(dets, 2.0, (W, HH), t)
    assert cal.status()["samples"] <= cal.per_track_max


def test_rejects_truncated_low_confidence_and_non_person_boxes():
    cal = AutoCalibrator()
    cal.start(None)
    n = cal.add([(1, (0.0, 10, 20, 80), "person", 0.9),  # touches the left border
                 (2, (100, 10, 120, 80), "person", 0.3),  # low confidence
                 (3, (200, 10, 220, 80), "car", 0.9),  # not a person
                 (4, (300, 10, 380, 40), "person", 0.9),  # wrong aspect ratio
                 (5, (400, 100, 420, 160), "person", 0.9)], 2.0, (W, HH), 0.0)
    assert n == 1
    assert cal.samples[0][1:4] == (820.0, 320.0, 200.0)  # source px = analytics px * scale


class StreamWorker:
    """Fake CameraWorker: each read of last_dets advances a synthetic stream (contract from pipeline.py)."""

    def __init__(self, frames, cam_id="camX", scale=2.0):
        self.id, self.name, self._frames, self._i = cam_id, cam_id, frames, 0
        self.scale, self.source_size = scale, (W, HH)
        self.last_frame = np.zeros((int(HH / scale), int(W / scale), 3), np.uint8)

    @property
    def last_dets(self):
        if self._i >= len(self._frames):
            return []
        d = list(self._frames[self._i][1])
        self._i += 1
        return d


def test_background_collector_thread_reaches_ready():
    w = StreamWorker(synth_frames(seed=1))
    cal = AutoCalibrator(poll_s=0.0, per_track_interval_s=0.0, refit_every=100)
    cal.start(w)
    deadline = time.time() + 60
    while time.time() < deadline and cal.status()["state"] != "ready":
        time.sleep(0.05)
    st = cal.status()
    cal.stop()
    assert st["state"] == "ready", st
    assert st["estimate"]["method"] == "auto-pedestrian" and st["camera"] == "camX"


def test_homography_matches_camera_model():
    f, th, h = 800.0, math.radians(30), 6.0
    Hm = image_to_ground_h(f, th, h, W / 2, HH / 2)
    for X, Y in [(0, 5), (3, 12), (-4, 20)]:
        u, v, _ = pixel_from_world(X, Y, 0.0, f, th, h, W / 2, HH / 2)
        g = Hm @ np.array([float(u), float(v), 1.0])
        assert np.allclose(g[:2] / g[2], (X, Y), atol=1e-6)


VTEST = ROOT / "data" / "real" / "vtest.avi"
WEIGHTS = ROOT / "data" / "models" / "yolo26n.pt"


@pytest.mark.slow
@pytest.mark.skipif(not (VTEST.exists() and WEIGHTS.exists()), reason="data/real/vtest.avi or model weights missing")
def test_real_clip_vtest_reports_estimate_or_reason():
    """Not a pass/fail accuracy test: vtest.avi has no survey data. Reports the estimate vs the hand calibration."""
    from argonix.detect import UltralyticsTracker
    st = calibrate_video(VTEST, UltralyticsTracker(str(WEIGHTS), "bytetrack.yaml", 0.3, 480))
    assert st["state"] in {"ready", "failed"} and st["reason"]
    assert st["samples"] > 0
    cam = next(c for c in json.loads((ROOT / "configs" / "real_site.json").read_text())["cameras"] if c["id"] == "cam_campus")
    hand = homography_from_points([tuple(p) for p in cam["calibration"]["image_points"]],
                                  [tuple(p) for p in cam["calibration"]["ground_points"]])
    print(f"\nvtest auto-calibration: state={st['state']} samples={st['samples']} tracks={st['tracks']} "
          f"coverage={st['coverage']} reason={st['reason']}\n  rejected_fit={st.get('rejected_fit')}")
    print(f"  hand: row520->row185 at centre column = {math.dist(apply_h(hand, (384, 520)), apply_h(hand, (384, 185))):.2f} m")
    if st["state"] == "ready":
        est = st["estimate"]
        assert est["calib_error_m"] >= 0.5
        He = homography_from_points([tuple(p) for p in est["image_points"]], [tuple(p) for p in est["ground_points"]])
        print(f"  auto: params={est['params']} err={est['calib_error_m']} m "
              f"row520->row185 = {math.dist(apply_h(He, (384, 520)), apply_h(He, (384, 185))):.2f} m")
