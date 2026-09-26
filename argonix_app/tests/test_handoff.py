"""Next-camera (handoff) prediction: lifecycle, features, association, training vs baseline, persistence, API."""
import json
from collections import deque
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from argonix.auth import Auth, hash_password
from argonix.handoff import (DEFAULTS, LEFT, N_BASE, SIDES, HandoffEngine, HandoffModel, Lifecycle, Sample,
                             Topology, base_features, evaluate, train_model)
from argonix.pipeline import Site
from argonix.server import create_app
from tools.sim_handoff import simulate_dataset

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
PW = "correct-horse-battery"  # test fixture only


def det(tid, xn, yn):
    x, y = xn * W, yn * H
    return (tid, (x - 15, y - 100, x + 15, y), "person", 0.9)


def feed(eng, t0, t1, fn, cams=("camA", "camB", "camC"), fps=5.0, zones=None):
    """Advance every camera from t0 to t1; fn(cam, t) -> list of dets."""
    n = int(round((t1 - t0) * fps))
    for i in range(n + 1):
        t = t0 + i / fps
        for c in cams:
            eng.observe(c, t, W, H, fn(c, t), {}, (zones or {}).get(c, []))


# ---------------------------------------------------------------------------------------------------------------
def test_features_fixed_length_and_camera_onehot():
    hist = deque([(100.0 + i * 0.2, 0.5 + 0.04 * i, 0.6, 1.0 + 0.3 * i, 2.0) for i in range(10)])
    x = base_features(hist, 100.0, "right", False)
    assert len(x) == N_BASE
    side = x[4 * 8: 4 * 8 + len(SIDES)]
    assert side == [0.0, 1.0, 0.0, 0.0]            # exit side one-hot
    assert x[4 * 8 + 4] == 0.0                     # not an exit zone
    assert abs(x[4 * 8 + 5] - 0.2) < 1e-6          # speed: 0.04 per 0.2 s
    assert np.allclose(x[4 * 8 + 8: 4 * 8 + 11], [1.5, 0.0, 1.0])  # ground velocity 0.3 m / 0.2 s + calibrated flag
    assert all(-1 <= v <= 1 for v in x[-2:])       # hour of day sin/cos
    short = base_features(deque([(5.0, 0.1, 0.9, None, None), (5.2, 0.08, 0.9, None, None)]), 5.0, "left", True)
    assert len(short) == N_BASE and short[4 * 8 + 8: 4 * 8 + 11] == [0.0, 0.0, 0.0]
    m = HandoffModel(["camA", "camB"])
    X = m.encode([x], ["camB"]) * m.std + m.mean
    assert X.shape == (1, N_BASE + 2) and list(X[0, N_BASE:]) == [0.0, 1.0]


def test_exit_and_entry_detection():
    life = Lifecycle(dict(DEFAULTS))
    zone = [(0.4, 0.3), (0.6, 0.3), (0.6, 0.45), (0.4, 0.45)]
    exits, entries = [], []

    def step(t, dets):
        en, ex, _ = life.observe("camA", t, W, H, dets, {}, [zone])
        entries.extend(en)
        exits.extend(ex)

    for i in range(12):  # 1: walks right to the border, vanishes; 2: vanishes mid-image; 3: walks into the door zone
        t = i * 0.2
        step(t, [det(1, 0.5 + 0.04 * i, 0.7), det(2, 0.3, 0.6 + 0.01 * i), det(3, 0.5, 0.6 - 0.02 * i),
                 *([det(4, 0.03, 0.6)] if i >= 5 else []), *([det(5, 0.5, 0.8)] if i >= 5 else [])])
    for i in range(12, 25):
        step(i * 0.2, [])
    kinds = {e.tid: e.key for e in exits}
    assert kinds[1] == "right" and kinds[3] == "zone" and 2 not in kinds and 5 not in kinds
    assert kinds[4] == "left"
    assert {e.tid: e.key for e in entries} == {4: "left"}  # 5 appeared mid-image; 1-3 existed at start mid-image
    ex1 = next(e for e in exits if e.tid == 1)
    assert len(ex1.x) == N_BASE and abs(ex1.t - 2.2) < 1e-6


def _engine(**kw):
    return HandoffEngine(["camA", "camB", "camC"], {"max_dt_s": 20.0, "cold_start": False, **kw})


def _exit_on_a(c, t):
    if c == "camA" and t <= 2.0:
        return [det(1, 0.6 + 0.18 * t, 0.7)]  # reaches x = 0.96 at t = 2.0, then gone
    return []


def test_association_time_gated_with_ambiguity_splitting():
    # single candidate on camB 5 s after the exit: most weight, but never certainty
    eng = _engine()
    feed(eng, 0, 30, lambda c, t: _exit_on_a(c, t) + ([det(7, 0.02 + 0.05 * (t - 7), 0.7)] if c == "camB" and 7.0 <= t <= 9 else []))
    assert eng.n_labels == 1 and not eng.pending  # resolved automatically once every camera passed exit + max_dt
    tgt = eng.history[-1]["target"]
    assert max(tgt, key=tgt.get) == "camB" and 0.5 < tgt["camB"] < 1.0 and tgt[LEFT] > 0
    assert abs(eng.history[-1]["dt"]["camB"] - 5.0) < 0.3

    # two candidates (camB and camC) at similar times: the weight is split, neither is claimed
    eng = _engine()
    feed(eng, 0, 30, lambda c, t: _exit_on_a(c, t) + ([det(7, 0.02 + 0.05 * (t - 7), 0.7)] if c == "camB" and 7.0 <= t <= 9 else [])
         + ([det(8, 0.5, 0.98 - 0.05 * (t - 8))] if c == "camC" and 8.0 <= t <= 10 else []))
    tgt = eng.history[-1]["target"]
    assert abs(tgt["camB"] - tgt["camC"]) < 0.05 and tgt["camB"] < 0.5 and eng.history[-1]["n_candidates"] == 2

    # entry outside the allowed travel-time window -> "left coverage"
    eng = _engine()
    feed(eng, 0, 40, lambda c, t: _exit_on_a(c, t) + ([det(7, 0.02 + 0.05 * (t - 25), 0.7)] if c == "camB" and 25.0 <= t <= 27 else []))
    assert eng.history[-1]["target"] == {LEFT: 1.0}
    topo = eng.topology_summary()
    assert topo["left_coverage_rate"] == 1.0 and topo["transitions"] == []


def test_association_is_order_independent_across_cameras():
    """Camera workers run at different speeds: camB may be processed far ahead of camA."""
    eng = _engine()
    fn = lambda c, t: _exit_on_a(c, t) + ([det(7, 0.02 + 0.05 * (t - 7), 0.7)] if c == "camB" and 7.0 <= t <= 9 else [])
    feed(eng, 0, 30, fn, cams=("camB", "camC"))  # B and C first, entirely
    assert eng.n_labels == 0
    feed(eng, 0, 30, fn, cams=("camA",))
    eng.flush()
    tgt = eng.history[-1]["target"]
    assert max(tgt, key=tgt.get) == "camB"
    assert eng.topology_summary()["transitions"][0]["from"] == "camA"


def test_live_prediction_for_leaving_track_and_scoring():
    eng = _engine()
    for _ in range(30):  # prior knowledge: right-side exits of camA go to camB
        eng.baseline.add("camA", "right", {"camB": 1.0}, {"camB": 5.0})
    feed(eng, 0, 1.8, _exit_on_a)
    rows = eng.predictions_list()
    assert rows and rows[0]["state"] == "leaving" and rows[0]["kind"] == "estimate"
    top = rows[0]["predictions"][0]
    assert top["camera"] == "camB" and 0.5 < top["prob"] < 1.0 and top["eta_s"] == 5.0
    assert len(rows[0]["predictions"]) <= 3 and sum(p["prob"] for p in rows[0]["predictions"]) <= 1.0001
    feed(eng, 2.0, 30, lambda c, t: _exit_on_a(c, t) + ([det(7, 0.02 + 0.05 * (t - 7), 0.7)] if c == "camB" and 7.0 <= t <= 9 else []))
    m = eng.metrics_summary()
    assert m["n_labels"] == 1 and m["baseline"]["n"] == 1 and m["baseline"]["top1"] > 0.5
    assert m["model"] is None  # no NN without cold start / enough labels


def test_nn_beats_side_baseline_on_heldout_simulation():
    cams = ["camA", "camB", "camC"]
    tr = simulate_dataset("demo", duration_s=2700, seed=0)
    ho = simulate_dataset("demo", duration_s=1200, seed=1000)
    assert len(tr) > 300 and len(ho) > 100
    assert {s.key for s in tr} >= {"left", "right", "zone", "bottom"}
    base = Topology()
    for s in tr:
        base.add(s.cam, s.key, s.target, s.dt)
    m = train_model(tr, cams, epochs=150, seed=0)
    assert m.meta["params"] < 20000
    res = evaluate(m, base, ho)
    assert res["nn"]["top1"] > res["baseline"]["top1"] + 0.02
    assert res["nn"]["logloss"] < res["baseline"]["logloss"]
    assert res["nn"]["eta_mae_s"] is not None and res["nn"]["eta_mae_s"] < 6.0
    p, mu, sg = m.predict([ho[0].x], [ho[0].cam])
    assert abs(p.sum() - 1) < 1e-5 and p.max() < 1.0 and (sg > 0).all()


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        right = i % 2 == 0
        x = list(rng.normal(0, 0.1, N_BASE))
        x[4 * 8 + (1 if right else 0)] = 1.0
        dest = "camB" if right else LEFT
        out.append(Sample(x, "camA", "right" if right else "left", {dest: 1.0},
                          {"camB": 6.0} if right else {}, 1000.0 + i))
    return out


def test_online_retrain_persistence_and_reload(tmp_path):
    eng = HandoffEngine(["camA", "camB"], {"min_labels": 20, "cold_start": False}, tmp_path / "handoff")
    assert eng.retrain(blocking=True)["ok"] is False  # not enough labels yet
    for s in _samples(40):
        eng.buffer.append(s)
        eng.topology.add(s.cam, s.key, s.target, s.dt)
    res = eng.retrain(blocking=True, reason="test")
    assert res["ok"] and eng.model is not None and eng.model.meta["source"] == "online"
    probe = _samples(4, seed=5)
    p1, _, _ = eng.model.predict([s.x for s in probe], [s.cam for s in probe])
    assert p1[0].argmax() == 1 and p1[1].argmax() == 2  # right -> camB, left -> leaves coverage
    eng.save()
    for f in ("model.pt", "buffer.json", "topology.json"):
        assert (tmp_path / "handoff" / f).exists()

    eng2 = HandoffEngine(["camA", "camB"], {"min_labels": 20, "cold_start": False}, tmp_path / "handoff")
    assert eng2.version == eng.version == 1 and len(eng2.buffer) == 40
    p2, _, _ = eng2.model.predict([s.x for s in probe], [s.cam for s in probe])
    assert np.allclose(p1, p2, atol=1e-6)
    assert eng2.topology_summary()["transitions"] == eng.topology_summary()["transitions"]
    res = eng2.retrain(blocking=True)  # fine-tunes from the persisted weights
    assert res["ok"] and eng2.version == 2 and eng2.model.meta["fine_tuned"] is True


# ---------------------------------------------------------------------------------------------------------------
# API through a real Site (fake tracker): a person leaves cam1 at the right edge and enters cam2 at the left edge
class Walker:
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self, cam):
        self.cam, self.n = cam, 0

    def track(self, frame):
        self.n += 1
        t = self.n / 5.0
        if self.cam == "cam1" and t <= 4.0:
            return [(1, (480 + 115 * t - 15, 350, 480 + 115 * t + 15, 450), "person", 0.9)]
        if self.cam == "cam2" and 7.0 <= t <= 12.0:
            return [(9, (10 + 60 * (t - 7), 350, 40 + 60 * (t - 7), 450), "person", 0.9)]
        return []


def make_site(tmp_path, handoff=True):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["record"] = False
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
        c["zones"] = []
    if handoff:
        cfg["handoff"] = {"enabled": True, "cold_start": False, "max_dt_s": 15, "min_labels": 1000}
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Walker(c["id"]))


def login(app, user):
    c = TestClient(app, headers={"X-Argonix-CSRF": "1"})  # what the dashboard sends (cookie requests need it)
    assert c.post("/api/login", json={"username": user, "password": PW}).status_code == 200
    return c


def test_handoff_api_and_permissions(tmp_path):
    site = make_site(tmp_path)
    assert site.handoff in site.analyzers
    site.start(); site.wait(timeout=120); site.stop()
    users = {n: {**hash_password(PW), "roles": [r]} for n, r in [("vic", "viewer"), ("olga", "operator"), ("ada", "admin")]}
    (tmp_path / "users.json").write_text(json.dumps(users))
    app = create_app(site, auth=Auth(tmp_path / "users.json", tmp_path / "secret.key"))
    anon = TestClient(app)
    for path in ("/api/handoff/predictions", "/api/handoff/topology", "/api/handoff/metrics"):
        assert anon.get(path).status_code == 401
    vic, olga, ada = (login(app, u) for u in ("vic", "olga", "ada"))
    assert isinstance(vic.get("/api/handoff/predictions").json(), list)
    topo = vic.get("/api/handoff/topology").json()
    assert topo["cameras"] == ["cam1", "cam2"]
    edge = next(e for e in topo["transitions"] if e["from"] == "cam1" and e["to"] == "cam2")
    assert edge["count"] > 0.5 and 2.0 < edge["travel_s"]["p50"] < 4.0
    m = vic.get("/api/handoff/metrics").json()
    assert m["n_labels"] >= 1 and m["n_exits"] >= 1 and m["n_entries"] >= 1
    assert "identity" in m["note"] and m["nn"]["n"] == 0 and m["model"] is None
    assert vic.post("/api/handoff/retrain").status_code == 403
    assert olga.post("/api/handoff/retrain").status_code == 403
    r = ada.post("/api/handoff/retrain")
    assert r.status_code == 409 and "labels" in r.json()["detail"]  # admin allowed; too few labels to train
    assert any(a["action"] == "handoff_retrain" for a in site.store.list_audit(limit=50))
    assert not site.store.list_alerts()  # handoff emits no alerts by default


def test_handoff_endpoints_absent_when_disabled(tmp_path):
    site = make_site(tmp_path, handoff=False)
    from argonix.handoff import HandoffAnalyzer
    assert not hasattr(site, "handoff") and not any(isinstance(a, HandoffAnalyzer) for a in site.analyzers)
    c = TestClient(create_app(site, open_demo=True))
    r = c.get("/api/handoff/metrics")
    assert r.status_code == 404 and r.json()["detail"] == "Not Found"  # the UI treats this as "feature missing"
