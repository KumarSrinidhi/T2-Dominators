"""Weapon-like object + altercation analyzers with fake models (CPU only, no weights needed)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from argonix import threat_install
from argonix.altercation import (N_FEAT, AltercationAnalyzer, heuristic_score, pair_features, resample_track,
                                 window_candidate)
from argonix.models import TrackObs
from argonix.rules import LABEL, SEVERITY
from argonix.threats import ObjDet, ObjectThreatAnalyzer, associate

ROOT = Path(__file__).resolve().parents[1]
# words an observation must never use about a person (report the object/motion, not the person)
BANNED = ("dangerous", "criminal", "suspect", "attacker", "aggressor", "armed person", "gunman", "violent person",
          "threatening", "intent to")


class FakeModel:
    """Scripted object model: `script[i]` = list of ObjDet for the i-th full-frame call."""

    def __init__(self, script, name="fake-weapons", version="v0-test", crop_script=None):
        self.script, self.i = list(script), 0
        self.crop_script = crop_script
        self.info = {"name": name, "version": version, "classes": {0: "knife", 1: "handgun"}, "device": "cpu"}
        self.crop_calls = 0

    def predict(self, images, crop=False):
        if crop:
            self.crop_calls += 1
            return [list(self.crop_script or []) for _ in images]
        out = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return [list(out)]


def worker(tmp_path, cam="cam1"):
    site = SimpleNamespace(data=tmp_path)
    return SimpleNamespace(id=cam, name=f"Camera {cam}", site=site, calib_error=0.5,
                           backend=SimpleNamespace(info={"detector": "fake-det", "tracker": "fake-trk"}))


FRAME = np.full((480, 640, 3), 90, np.uint8)
PERSON = (7, (300.0, 100.0, 360.0, 300.0), "person", 0.9)
KNIFE = ObjDet("knife", 0.8, (355.0, 180.0, 380.0, 200.0), "fake-weapons")  # right next to the person's hand


def run(an, w, schedule, dets=(PERSON,), dt=0.34):
    events, t = [], 1000.0
    for _ in schedule:
        events += an.on_frame(w, t, FRAME, list(dets), [])
        t += dt
    return events


# ---------------------------------------------------------------------------------------------- objects
def test_association_uses_hand_region():
    people = [(1, (0, 0, 50, 150)), (2, (300, 100, 360, 300))]
    assert associate((355, 180, 380, 200), people)[0] == 2
    assert associate((10, 60, 30, 80), people)[0] == 1
    assert associate((600, 400, 620, 420), people) == (None, 0.0)   # nowhere near anybody
    assert associate((330, 290, 340, 299), people)[0] is None         # at the feet, below the hand region


def test_persistence_k_of_n(tmp_path):
    w = worker(tmp_path)
    # hits at checks 1, 3 -> only 2 of 5: nothing
    an = ObjectThreatAnalyzer([FakeModel([[KNIFE], [], [KNIFE], [], [], []])], fps=3, k=3, n=5)
    assert run(an, w, range(6)) == []
    # hits at 1, 2, 4 -> the 3rd hit (4th check) fires, exactly once
    an = ObjectThreatAnalyzer([FakeModel([[KNIFE], [KNIFE], [], [KNIFE], [KNIFE], [KNIFE]])], fps=3, k=3, n=5)
    ev = run(an, w, range(6))
    assert len(ev) == 1 and ev[0].rule == "object_threat"
    assert ev[0].derivation["persistence"] == {"seen": 3, "of": 4}


def test_fps_gating(tmp_path):
    m = FakeModel([[KNIFE]] * 20)
    an = ObjectThreatAnalyzer([m], fps=2, k=3, n=5)
    run(an, worker(tmp_path), range(10), dt=0.1)  # 1 s of analytics at 10 fps -> 2 checks
    assert m.i == 2


def test_min_conf_and_person_required(tmp_path):
    w = worker(tmp_path)
    weak = ObjDet("knife", 0.3, KNIFE.box, "fake-weapons")
    an = ObjectThreatAnalyzer([FakeModel([[weak]] * 6)], fps=3, min_conf=0.5, k=3, n=5)
    assert run(an, w, range(6)) == []
    far = ObjDet("knife", 0.9, (600, 400, 620, 420), "fake-weapons")
    an = ObjectThreatAnalyzer([FakeModel([[far]] * 6)], fps=3, k=3, n=5)
    assert run(an, w, range(6)) == []
    m = FakeModel([[KNIFE]] * 6)
    an = ObjectThreatAnalyzer([m], fps=3, k=3, n=5)
    assert run(an, w, range(6), dets=()) == [] and m.i == 0  # no people: the model is not even run


def test_event_fields_and_crop(tmp_path):
    an = ObjectThreatAnalyzer([FakeModel([[KNIFE]] * 5)], fps=3, k=3, n=5)
    ev = run(an, worker(tmp_path), range(5))
    assert len(ev) == 1
    e = ev[0]
    assert e.track_ids == [7] and e.severity == 3 and e.zone_id == "_view"
    assert e.observation.startswith("Object resembling a knife near person track #7 (cam1)")
    assert "3 of the last 3 checks" in e.observation and "fake-weapons v0-test" in e.observation
    assert not any(b in e.observation.lower() for b in BANNED)
    d = e.derivation
    assert d["learned_model"] is True and d["object_label"] == "knife"
    assert d["models"][0]["version"] == "v0-test"
    assert d["per_check_conf"] == [0.8, 0.8, 0.8] and d["box_xyxy_analytics_px"] == [355.0, 180.0, 380.0, 200.0]
    assert d["person_track"] == 7 and d["association"]["score"] == 1.0
    crop = e.evidence["crop"]
    assert crop.startswith("evidence/cam1_object_threat_knife_") and (tmp_path / crop).is_file()
    assert "fake-weapons" in e.evidence["model"]["detector"]


def test_rearm_no_duplicate(tmp_path):
    an = ObjectThreatAnalyzer([FakeModel([[KNIFE]] * 30)], fps=3, k=3, n=5, rearm_s=60)
    assert len(run(an, worker(tmp_path), range(30))) == 1  # 10 s continuously visible -> one event


def test_person_crop_boxes_mapped_to_frame(tmp_path):
    # small person -> crop around it; the crop-space knife must land next to the person in frame coordinates
    small = (3, (400.0, 200.0, 430.0, 260.0), "person", 0.9)
    crop_knife = ObjDet("knife", 0.9, (30.0, 20.0, 40.0, 30.0), "fake-weapons")
    m = FakeModel([[]] * 6, crop_script=[crop_knife])
    an = ObjectThreatAnalyzer([m], fps=3, k=3, n=5, crop_below_px=100)
    ev = run(an, worker(tmp_path), range(4), dets=(small,))
    assert m.crop_calls >= 3 and len(ev) == 1
    x1, y1, x2, y2 = ev[0].derivation["box_xyxy_analytics_px"]
    assert 380 <= x1 <= 440 and 180 <= y1 <= 260 and ev[0].track_ids == [3]
    assert ev[0].derivation["detected_in"] == ["person_crop"]


# ---------------------------------------------------------------------------------------------- altercation
def skeleton(box, arm_phase=0.0, reach=0.0):
    """A crude COCO-17 skeleton inside `box`; `reach` pushes the right wrist sideways by that many box heights."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    k = np.zeros((17, 3), np.float32)
    rel = {0: (0, .08), 1: (-.05, .06), 2: (.05, .06), 3: (-.1, .07), 4: (.1, .07), 5: (-.2, .22), 6: (.2, .22),
           7: (-.25, .38), 8: (.25, .38), 9: (-.27, .52), 10: (.27, .52), 11: (-.12, .55), 12: (.12, .55),
           13: (-.12, .75), 14: (.12, .75), 15: (-.12, .97), 16: (.12, .97)}
    for i, (dx, dy) in rel.items():
        k[i] = (cx + dx * w, y1 + dy * h, 0.9)
    k[10, 0] += reach * h * np.sin(arm_phase)
    k[9, 0] -= reach * h * np.sin(arm_phase)
    return k


class FakePose:
    name = "fake-pose"

    def __init__(self, reach=0.6):
        self.calls, self.reach = 0, reach

    def __call__(self, frame):
        self.calls += 1
        return [(np.array(b, np.float32), skeleton(b, self.calls * 1.7, self.reach)) for b in self.boxes]


def test_pair_features_invariant_to_translation_and_scale():
    rng = np.random.default_rng(0)
    T = 8
    bA = np.tile([100, 100, 160, 300], (T, 1)).astype(np.float32) + rng.normal(0, 3, (T, 4)).astype(np.float32)
    bB = bA + np.array([70, 0, 70, 0], np.float32)
    kA = np.stack([skeleton(b, i, 0.5) for i, b in enumerate(bA)])
    kB = np.stack([skeleton(b, i + 1, 0.5) for i, b in enumerate(bB)])
    f = pair_features(kA, bA, kB, bB)
    assert f.shape == (T, N_FEAT) and np.isfinite(f).all()
    s, off = 2.5, np.array([400, -30], np.float32)
    tf = lambda k: np.concatenate([k[..., :2] * s + off, k[..., 2:]], -1)
    tb = lambda b: b * s + np.tile(off, 2)
    f2 = pair_features(tf(kA), tb(bA), tf(kB), tb(bB))
    np.testing.assert_allclose(f, f2, atol=1e-3)
    # batched call equals per-window calls
    fb = pair_features(np.stack([kA, kB]), np.stack([bA, bB]), np.stack([kB, kA]), np.stack([bB, bA]))
    np.testing.assert_allclose(fb[0], f, atol=1e-5)


def test_resample_track_interpolates_and_flags_gaps():
    times = np.array([0.0, 0.1, 0.2, 1.5, 1.6])
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [2, 0, 12, 10], [15, 0, 25, 10], [16, 0, 26, 10]], np.float32)
    kps = np.zeros((5, 17, 3), np.float32); kps[..., 2] = 1
    b, k, v = resample_track(times, boxes, kps, np.array([0.05, 0.2, 0.8, 1.55]))
    assert v.tolist() == [True, True, False, True]
    assert b[0, 0] == pytest.approx(0.5) and b[3, 0] == pytest.approx(15.5)
    assert k[2, :, 2].max() == 0.0  # the filled-in gap sample is marked unobserved


def test_heuristic_prefers_close_fast_arms():
    T = 8
    bA = np.tile([100, 100, 160, 300], (T, 1)).astype(np.float32)
    near, far = bA + [60, 0, 60, 0], bA + [600, 0, 600, 0]
    kA_fast = np.stack([skeleton(bA[0], i * 1.7, 0.6) for i in range(T)])
    kA_still = np.stack([skeleton(bA[0], 0, 0) for _ in range(T)])
    kN = np.stack([skeleton(near[0], 0, 0) for _ in range(T)])
    kF = np.stack([skeleton(far[0], 0, 0) for _ in range(T)])
    fast_near = heuristic_score(kA_fast, bA, kN, near)
    assert fast_near > heuristic_score(kA_still, bA, kN, near)
    assert fast_near > heuristic_score(kA_fast, bA, kF, far)
    assert window_candidate(bA, near) and not window_candidate(bA, far)


def run_pairs(an, w, boxes, n=25, dt=0.2):
    events, t = [], 500.0
    dets = [(i + 1, tuple(b), "person", 0.9) for i, b in enumerate(boxes)]
    obs = [TrackObs(w.id, i + 1, t, tuple(b), "person", 0.9, (float(i), 0.0)) for i, b in enumerate(boxes)]
    for _ in range(n):
        events += an.on_frame(w, t, FRAME, dets, obs)
        t += dt
    return events


def test_altercation_persistence_pairing_and_fields(tmp_path):
    pose = FakePose()
    pose.boxes = [(200, 100, 260, 300), (250, 100, 310, 300)]
    scores = []

    def scorer(feats):
        assert feats.ndim == 3 and feats.shape[1:] == (8, N_FEAT)
        scores.append(len(feats))
        return np.full(len(feats), 0.9)

    an = AltercationAnalyzer(pose, scorer, threshold=0.6, persist_s=1.5, model_info={"name": "fake-gru", "version": "v0"})
    w = worker(tmp_path)
    ev = run_pairs(an, w, pose.boxes)
    assert len(ev) == 1  # one event per pair episode
    e = ev[0]
    assert e.rule == "altercation" and e.track_ids == [1, 2] and e.severity == 3
    assert e.t_end - e.t_start >= 1.5
    assert e.observation.startswith("Person tracks #1 and #2 (cam1)")
    assert "fake-gru v0" in e.observation and "does not establish" in e.observation
    assert not any(b in e.observation.lower() for b in BANNED)
    d = e.derivation
    assert d["learned_model"] is True and d["peak_score"] == 0.9 and d["threshold"]["persist_s"] == 1.5
    assert d["distance_m"] == 1.0 and "min_distance_body_heights" in d["cues"]
    assert (tmp_path / e.evidence["crop"]).is_file()
    # the first windows only appear once 8 samples (1.4 s) of keypoints exist
    assert pose.calls > len(scores)


def test_altercation_low_score_or_far_apart_is_silent(tmp_path):
    pose = FakePose()
    pose.boxes = [(200, 100, 260, 300), (250, 100, 310, 300)]
    an = AltercationAnalyzer(pose, lambda f: np.full(len(f), 0.2), threshold=0.6)
    assert run_pairs(an, worker(tmp_path), pose.boxes) == []
    far = FakePose()
    far.boxes = [(0, 100, 60, 300), (500, 100, 560, 300)]
    an = AltercationAnalyzer(far, lambda f: np.full(len(f), 0.99), threshold=0.6)
    assert run_pairs(an, worker(tmp_path), far.boxes) == [] and far.calls == 0  # no candidate pair: no pose run


def test_altercation_score_must_persist(tmp_path):
    pose = FakePose()
    pose.boxes = [(200, 100, 260, 300), (250, 100, 310, 300)]
    flip = {"n": 0}

    def scorer(f):  # above threshold only every third window: never 1.5 s continuous (grace 0.6 s < gaps)
        flip["n"] += 1
        return np.full(len(f), 0.9 if flip["n"] % 4 == 0 else 0.1)

    an = AltercationAnalyzer(pose, scorer, threshold=0.6, persist_s=1.5, grace_s=0.3)
    assert run_pairs(an, worker(tmp_path), pose.boxes, n=40) == []


def test_heuristic_mode_without_classifier(tmp_path):
    pose = FakePose(reach=0.8)
    pose.boxes = [(200, 100, 260, 300), (250, 100, 310, 300)]
    an = AltercationAnalyzer(pose, None, threshold=0.5, heuristic_scale=0.2)
    ev = run_pairs(an, worker(tmp_path), pose.boxes)
    assert len(ev) == 1 and ev[0].derivation["learned_model"] is False
    assert ev[0].derivation["model"]["mode"] == "heuristic"


def test_gru_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from argonix.altercation import FEATURE_VERSION, GRUScorer, build_gru
    m = build_gru()
    assert sum(p.numel() for p in m.parameters()) < 1_000_000
    p = tmp_path / "gru.pt"
    torch.save({"state_dict": m.state_dict(), "meta": {"feature_version": FEATURE_VERSION, "name": "g", "version": "v1"}}, p)
    s = GRUScorer(p)
    out = s(np.zeros((3, 8, N_FEAT), np.float32))
    assert out.shape == (3,) and ((out > 0) & (out < 1)).all() and s.version == "v1"
    torch.save({"state_dict": m.state_dict(), "meta": {"feature_version": "other"}}, p)
    with pytest.raises(ValueError):
        GRUScorer(p)


# ---------------------------------------------------------------------------------------------- wiring
def fake_site(tmp_path, threats=None):
    cfg = {"cameras": []}
    if threats is not None:
        cfg["threats"] = threats
    return SimpleNamespace(cfg=cfg, analyzers=[], _base=tmp_path)


def test_rules_registered_for_ui():
    assert LABEL["object_threat"] == "Possible weapon-like object" and SEVERITY["object_threat"] == 3
    assert LABEL["altercation"] == "Possible physical altercation" and SEVERITY["altercation"] == 3


def test_install_disabled_by_default(tmp_path):
    s = fake_site(tmp_path)
    boom = lambda *a, **k: pytest.fail("no model may load when threats are not configured")
    assert threat_install.install(s, boom, boom, boom) == [] and s.analyzers == []
    s = fake_site(tmp_path, {"objects": {"enabled": False}, "altercation": {}})
    assert threat_install.install(s, boom, boom, boom) == []
    assert s.threat_status["objects"] == "disabled" and s.threat_status["altercation"] == "disabled"


def test_install_from_config(tmp_path):
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "w.pt").write_bytes(b"x")
    (tmp_path / "data" / "models" / "clf.pt").write_bytes(b"x")
    seen = {}

    def omf(spec, device, models_dir):
        seen.setdefault("obj", []).append(spec)
        return FakeModel([], name=spec.get("name", "m"))

    def pf(weights, device, imgsz, models_dir):
        seen["pose"] = weights
        return FakePose()

    def sf(path, device):
        seen["clf"] = path
        return SimpleNamespace(meta={"name": "gru", "version": "v9", "threshold": 0.73, "window": 8, "fps": 5.0},
                               __call__=None)

    s = fake_site(tmp_path, {
        "objects": {"enabled": True, "fps": 2, "k": 2, "n": 4, "min_conf": 0.6,
                    "models": [{"weights": "data/models/w.pt", "name": "weap", "classes": {"0": "knife"}},
                               {"weights": "data/models/missing.pt", "classes": {"0": "knife"}}]},
        "altercation": {"enabled": True, "pose_weights": "yolo26n-pose.pt", "classifier": "data/models/clf.pt",
                        "persist_s": 2.0}})
    added = threat_install.install(s, omf, pf, sf)
    assert [type(a).__name__ for a in added] == ["ObjectThreatAnalyzer", "AltercationAnalyzer"]
    assert s.analyzers == added
    obj, alt = added
    assert (obj.fps, obj.k, obj.n, obj.min_conf) == (2.0, 2, 4, 0.6) and len(obj.models) == 1
    assert seen["obj"][0]["weights"] == str(tmp_path / "data" / "models" / "w.pt")
    assert any("missing.pt" in w for w in s.threat_status["warnings"])
    assert alt.thr == 0.73 and alt.persist_s == 2.0 and alt.model_info["version"] == "v9"
    assert seen["clf"] == str(tmp_path / "data" / "models" / "clf.pt")


def test_site_wires_threats_end_to_end(tmp_path, monkeypatch):
    """Real Site + FakeBackend (person walking) + fake knife model: an object_threat alert with a crop is stored."""
    from argonix.pipeline import Site

    class Backend:
        info = {"detector": "fake", "tracker": "fake"}

        def track(self, frame):
            return [(1, (633, 200, 673, 302), "person", 0.9)]

    class Model:
        info = {"name": "fake-knife", "version": "v1", "classes": {0: "knife"}, "device": "cpu"}

        def predict(self, images, crop=False):
            return [[ObjDet("knife", 0.85, (668, 240, 690, 256), "fake-knife")] for _ in images]

    monkeypatch.setattr(threat_install, "make_object_model", lambda spec, device, md: Model())
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    cfg["threats"] = {"objects": {"enabled": True, "models": [{"weights": "yolo26n.pt", "classes": {"43": "knife"}}]}}
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Backend())
    from argonix.threats import ObjectThreatAnalyzer
    assert sum(isinstance(a, ObjectThreatAnalyzer) for a in site.analyzers) == 1  # (+ default-on flow analytics)
    site.start(); site.wait(timeout=120); site.stop()
    alerts = [a for a in site.store.list_alerts() if any(e["rule"] == "object_threat" for e in a["events"])]
    assert alerts, "expected an object_threat alert"
    a = alerts[0]
    e = next(e for e in a["events"] if e["rule"] == "object_threat")
    assert e["evidence"]["snapshot"] and e["evidence"]["crop"]  # pipeline evidence merged with analyzer evidence
    assert (tmp_path / "run" / e["evidence"]["crop"]).is_file()
    assert "learned-model" in a["disclaimer"]


def test_example_threat_config_is_valid():
    cfg = json.loads((ROOT / "configs" / "real_site_gpu_threats.json").read_text())
    t = cfg["threats"]
    assert t["objects"]["enabled"] and t["altercation"]["enabled"]
    assert all("weights" in m and m["classes"] for m in t["objects"]["models"])
