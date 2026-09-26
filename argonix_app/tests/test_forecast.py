"""Trajectory forecasting: CV baseline, frames, metrics, GRU shapes/training step, checkpoint, runtime analyzer,
predicted-zone-entry (capped as a learned inference), API. Synthetic data only; no network, no model weights."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from argonix import forecast as F
from argonix.api_forecast import forecast_router
from argonix.correlate import explain_priority
from argonix.models import TrackObs
from argonix.rules import LABEL, SEVERITY, Zone

ROOT = Path(__file__).resolve().parents[1]
IDENT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def line(n=20, v=(1.0, 0.5), p0=(0.0, 0.0)):
    k = np.arange(n)[:, None]
    return np.asarray(p0) + k * np.asarray(v) * F.DT


# ---------------------------------------------------------------------------------------------- core maths
def test_cv_and_metrics_exact_on_straight_line():
    seq = line()[None]
    pred = F.cv_predict(seq[:, :F.OBS_LEN])
    assert pred.shape == (1, F.PRED_LEN, 2)
    assert np.allclose(pred, seq[:, F.OBS_LEN:])
    assert F.ade_fde(pred, seq[:, F.OBS_LEN:]) == pytest.approx((0.0, 0.0))
    ade, fde = F.ade_fde(pred + [3.0, 4.0], seq[:, F.OBS_LEN:])
    assert ade == pytest.approx(5.0) and fde == pytest.approx(5.0)


def test_heading_frame_round_trip():
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 3, (5, F.OBS_LEN, 2)).cumsum(1)
    R, o = F.heading_frames(obs)
    loc = F.to_local(obs, R, o)
    assert np.allclose(F.to_world(loc, R, o), obs)
    last = loc[:, -1] - loc[:, -2]
    assert np.allclose(last[:, 1], 0, atol=1e-9) and (last[:, 0] > 0).all()  # last step is +x
    still = np.zeros((1, F.OBS_LEN, 2))
    R2, _ = F.heading_frames(still)
    assert np.allclose(R2[0], np.eye(2))


def test_gru_shapes_decode_and_training_step():
    torch = pytest.importorskip("torch")
    m = F.build_model(hidden=32)
    assert sum(p.numel() for p in m.parameters()) < 100_000
    seqs = F.synthetic_walkers(256, seed=3)
    disp, R, o = F.model_inputs(seqs[:, :F.OBS_LEN])
    assert disp.shape == (256, F.OBS_LEN - 1, 2)
    res, ls = m(torch.from_numpy(disp))
    assert res.shape == (256, F.PRED_LEN, 2) and ls.shape == (256, F.PRED_LEN)
    # zero residual decodes to exactly constant velocity
    cvw = F.to_world(F.decode(disp.astype(np.float64), np.zeros((256, F.PRED_LEN, 2))), R, o)
    assert np.allclose(cvw, F.cv_predict(seqs[:, :F.OBS_LEN]), atol=1e-5)
    from tools.train_forecast import prepare
    d, fut, _, _ = prepare(seqs)
    X, Y = torch.from_numpy(d), torch.from_numpy(fut)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    steps = torch.arange(1, F.PRED_LEN + 1, dtype=torch.float32)[None, :, None]

    def loss():
        r, _ = m(X)
        return ((steps * X[:, -1:, :] + torch.cumsum(r, 1)) - Y).norm(dim=-1).mean()

    l0 = float(loss())
    for _ in range(30):
        opt.zero_grad(); lo = loss(); lo.backward(); opt.step()
    assert float(loss()) < l0


def test_checkpoint_roundtrip_and_fallbacks(tmp_path):
    pytest.importorskip("torch")
    m = F.build_model(hidden=16)
    meta = {"name": "t", "version": "v0", "feature_version": F.FEATURE_VERSION, "hidden": 16,
            "pred_len": F.PRED_LEN, "cv_rmse_per_step": [0.1] * F.PRED_LEN, "metrics": {"average": {}}}
    ck = tmp_path / "f.pt"
    F.save_checkpoint(ck, m, meta)
    assert json.loads(ck.with_suffix(".json").read_text())["name"] == "t"
    fc = F.Forecaster(ck)
    assert fc.method == "gru" and fc.info["params"] is None and fc.info["name"] == "t"
    mean, sig = fc.predict(F.synthetic_walkers(4)[:, :F.OBS_LEN])
    assert mean.shape == (4, F.PRED_LEN, 2) and sig.shape == (4, F.PRED_LEN) and (sig > 0).all()
    _, sig_cv = fc.predict(F.synthetic_walkers(2)[:, :F.OBS_LEN], method="cv")
    assert np.allclose(sig_cv, 0.1)
    assert F.Forecaster(tmp_path / "missing.pt").method == "cv"
    with pytest.raises(FileNotFoundError):
        F.Forecaster(tmp_path / "missing.pt", method="gru")
    F.save_checkpoint(tmp_path / "bad.pt", m, {**meta, "feature_version": "other"})
    with pytest.raises(ValueError):
        F.Forecaster(tmp_path / "bad.pt")


def test_eth_ucy_parser_and_windows(tmp_path):
    rows = [(f * 10, 1, 0.1 * f, 0.0) for f in range(25)]            # 25 steps -> 6 windows of 20
    rows += [(f * 10, 2, 0.0, 0.2 * f) for f in list(range(12)) + list(range(13, 25))]  # a gap -> no window
    p = tmp_path / "scene.txt"
    p.write_text("\n".join("\t".join(str(v) for v in r) for r in rows))
    tr = F.load_eth_ucy(p)
    assert set(tr) == {1, 2} and F.frame_step(tr) == 10
    w = F.windows(tr)
    assert w.shape == (6, 20, 2) and np.allclose(w[0, :, 0], 0.1 * np.arange(20))


def test_sample_history_smoothing_and_coverage():
    t = np.arange(0, 4.0, 0.2)
    pts = np.stack([t * 1.0, np.zeros_like(t)], -1)
    s = F.sample_history(t, pts, t[-1])
    assert s.shape == (F.OBS_LEN, 2) and np.allclose(np.diff(s[:, 0]), F.DT, atol=1e-6)
    assert F.sample_history(t[-5:], pts[-5:], t[-1]) is None  # 0.8 s of history does not cover 2.8 s


def test_entry_probability():
    poly = [(10, -2), (14, -2), (14, 2), (10, 2)]
    mean = np.stack([np.linspace(1, 12, 8), np.zeros(8)], -1)
    p, k = F.entry_probability(mean, np.full(8, 0.3), poly)
    assert p > 0.9 and mean[k][0] >= 9.5
    p2, _ = F.entry_probability(mean * [1, 0] + [0, 20], np.full(8, 0.3), poly)
    assert p2 == 0.0


# ---------------------------------------------------------------------------------------------- analyzer
def worker(calibrated=True, zones=(), cam="cam1"):
    return SimpleNamespace(id=cam, name=f"Camera {cam}", H=IDENT if calibrated else None, scale=1.0,
                           calib_error=0.3 if calibrated else None, engine=SimpleNamespace(zones=list(zones)))


def drive(an, w, steps=40, fps=5.0, v=(1.2, 0.0), p0=(0.0, 0.0), pixel_box_h=None):
    events, t = [], 1000.0
    frame = np.zeros((540, 960, 3), np.uint8)
    for i in range(steps):
        x, y = p0[0] + v[0] * i / fps, p0[1] + v[1] * i / fps
        if pixel_box_h:
            bb = (x - 20, y - pixel_box_h, x + 20, y)
            obs = [TrackObs(w.id, 7, t, bb, "person", 0.9, None)]
        else:
            bb = (100.0, 100.0, 140.0, 200.0)
            obs = [TrackObs(w.id, 7, t, bb, "person", 0.9, (x, y))]
        events += an.on_frame(w, t, frame, [(7, bb, "person", 0.9)], obs)
        t += 1 / fps
    return events


RESTRICTED = Zone.from_dict({"id": "lab", "name": "Lab", "polygon": [[12, -3], [18, -3], [18, 3], [12, 3]],
                             "rules": {"intrusion": {}}, "criticality": 3, "site_zone": "S-lab"})


def test_analyzer_forecasts_calibrated_track():
    an = F.ForecastAnalyzer(F.Forecaster(None, "cv"), horizon_s=3.2)
    w = worker()
    assert drive(an, w, steps=25) == []
    rows = an.tracks()
    assert len(rows) == 1
    r = rows[0]
    assert r["space"] == "ground_m" and r["method"] == "cv" and len(r["path"]) == 8 and r["horizon_s"] == 3.2
    # walking +x at 1.2 m/s: 3.2 s ahead is ~3.84 m beyond the current position
    assert r["path"][-1][0] - r["current"][0] == pytest.approx(3.84, abs=0.05)
    assert len(r["path_px"]) == 8 and r["frame_size"] == [960, 540] and len(r["sigma_px"]) == 8
    assert r["sigma"][-1] > r["sigma"][0]  # spread grows with horizon


def test_analyzer_pixel_mode_uses_body_height_scale():
    an = F.ForecastAnalyzer(F.Forecaster(None, "cv"), horizon_s=2.0)
    w = worker(calibrated=False)
    drive(an, w, steps=25, v=(50.0, 0.0), p0=(100.0, 400.0), pixel_box_h=170)
    r = an.tracks("cam1")[0]
    assert r["space"] == "image_px" and r["approx_scale"] is True
    assert r["path"][-1][0] - r["current"][0] == pytest.approx(100.0, abs=3)  # 50 px/s x 2 s, back in pixels
    assert an.tracks("other") == []


def test_predicted_zone_entry_is_prediction_capped_at_p3():
    an = F.ForecastAnalyzer(F.Forecaster(None, "cv"), horizon_s=4.8,
                            zone_entry={"enabled": True, "horizon_s": 3.2, "min_prob": 0.5, "persist": 2})
    w = worker(zones=[RESTRICTED])
    ev = drive(an, w, steps=40, v=(1.5, 0.0))  # reaches x=12 at 8 s; forecast sees it ~3 s earlier
    assert len(ev) == 1
    e = ev[0]
    assert e.rule == "predicted_zone_entry" and e.severity == 1 and e.zone_id == "lab" and e.site_zone == "S-lab"
    assert e.derivation["learned_model"] and e.derivation["prediction"] is True
    assert "prediction" in e.observation and "not an observed entry" in e.observation
    assert e.t_end < 1000 + 8.0  # fired before the person reached the zone
    _, prio, factors = explain_priority([e])
    assert prio in ("P3", "P4") and factors["learned_model_only"]
    e.confidence, e.criticality = 1.0, 3
    e.severity = 3  # even if someone raised it, learned-only stays capped
    assert explain_priority([e])[1] == "P3"
    assert LABEL["predicted_zone_entry"] and SEVERITY["predicted_zone_entry"] == 1


def test_no_zone_entry_when_walking_away_or_disabled():
    w = worker(zones=[RESTRICTED])
    an = F.ForecastAnalyzer(F.Forecaster(None, "cv"), zone_entry={"enabled": True})
    assert drive(an, w, steps=40, v=(-1.5, 0.0)) == []
    an2 = F.ForecastAnalyzer(F.Forecaster(None, "cv"))  # zone entry off by default
    assert drive(an2, w, steps=40, v=(1.5, 0.0)) == []


# ---------------------------------------------------------------------------------------------- install + API
def test_install_off_by_default_and_cv_fallback(tmp_path):
    site = SimpleNamespace(cfg={}, analyzers=[], _base=tmp_path)
    assert F.install(site) is None and site.forecast is None and site.analyzers == []
    site.cfg = {"forecast": {"enabled": True, "checkpoint": "data/models/nope.pt"}}
    an = F.install(site)
    assert an in site.analyzers and site.forecast is an and an.fc.method == "cv"
    assert any("constant velocity" in w for w in an.warnings)


@pytest.mark.skipif(not F.BUNDLED_CHECKPOINT.exists(), reason="bundled forecast checkpoint not present")
def test_bundled_checkpoint_loads_and_stays_near_cv_on_a_straight_walk(tmp_path):
    pytest.importorskip("torch")
    site = SimpleNamespace(cfg={"forecast": {"enabled": True}}, analyzers=[], _base=tmp_path)
    an = F.install(site)
    assert an.fc.method == "gru" and an.warnings == [] and an.status()["metrics"]["average"]["cv"]["ade"] > 0
    seq = line(20, v=(1.3, 0.0))[None]
    mean, sig = an.fc.predict(seq[:, :F.OBS_LEN])
    assert np.abs(mean - seq[:, F.OBS_LEN:]).max() < 1.0 and (np.diff(sig[0]) > -1e-6).all()


def make_app(site):
    def need(perm):
        def dep(x_role: str = Header(default="")):
            if not x_role:
                raise HTTPException(401)
            if perm != "alert.read":
                raise HTTPException(403)
            return x_role
        return dep
    app = FastAPI()
    app.include_router(forecast_router(site, need))
    return TestClient(app)


def test_api_shapes_permissions_and_404(tmp_path):
    site = SimpleNamespace(cfg={"forecast": {"enabled": True, "checkpoint": "nope.pt"}}, analyzers=[], _base=tmp_path)
    an = F.install(site)
    drive(an, worker(), steps=25)
    c = make_app(site)
    assert c.get("/api/forecast/tracks").status_code == 401
    r = c.get("/api/forecast/tracks", headers={"x-role": "viewer"}).json()
    assert r["experimental"] and r["tracks"][0]["track_id"] == 7 and "prediction" in r["note"]
    s = c.get("/api/forecast/status", headers={"x-role": "viewer"}).json()
    assert s["model"]["method"] == "cv" and s["warnings"] and s["experimental"]
    assert c.get("/api/fall/status", headers={"x-role": "viewer"}).status_code == 404
    off = make_app(SimpleNamespace(cfg={}, analyzers=[], forecast=None))
    assert off.get("/api/forecast/tracks", headers={"x-role": "viewer"}).status_code == 404


class Walker:
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self):
        self.n = 0

    def track(self, frame):
        self.n += 1
        x = 200 + (self.n % 60) * 8
        return [(1 + self.n // 60, (x, 300, x + 30, 400), "person", 0.9)]


def test_site_integration(tmp_path):
    from argonix.pipeline import Site
    from argonix.server import create_app
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    cfg["forecast"] = {"enabled": True, "method": "cv"}
    cfg["fall"] = {"enabled": True, "pose": False}
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
        c["loop"] = False
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Walker())
    assert site.forecast in site.analyzers and site.fall in site.analyzers
    site.start(); site.wait(timeout=180); site.stop()
    assert site.forecast.stats["updates"] > 0 and site.fall.stats["samples"] > 0
    kw = {"open_demo": True} if "open_demo" in inspect.signature(create_app).parameters else {}
    client = TestClient(create_app(site, **kw))
    st = client.get("/api/forecast/status")
    if st.status_code == 200:  # (with accounts enabled this needs a login; covered by the router test above)
        assert st.json()["model"]["method"] == "cv"
        assert client.get("/api/fall/status").json()["mode"].startswith("boxes only")
    assert not any(k.startswith("analyzer_error") for k in site.workers[0].stats)
