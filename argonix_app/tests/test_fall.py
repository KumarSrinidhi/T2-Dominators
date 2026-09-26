"""Fall / person-on-the-ground detection on synthetic box + keypoint sequences (no pose model, no network)."""
from types import SimpleNamespace

import numpy as np
import pytest

from argonix import fall as FL
from argonix.correlate import explain_priority
from argonix.rules import LABEL, SEVERITY

FPS = 5.0
STAND = (300.0, 100.0, 340.0, 270.0)  # 40 x 170 px upright person, feet at y = 270
# words an observation must never use: no diagnosis, no judgement about the person
BANNED = ("unconscious", "collapsed", "fainted", "emergency", "victim", "drunk", "overdose", "seizure", "dead",
          "heart", "medical condition", "needs help", "dangerous", "suspect")


def kps(box, posture):
    """COCO-17 keypoints with only shoulders (5, 6) and hips (11, 12) visible."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    k = np.zeros((17, 3), np.float32)
    if posture == "upright":
        sh, hp = [(cx - 8, y1 + 0.2 * h), (cx + 8, y1 + 0.2 * h)], [(cx - 6, y1 + 0.5 * h), (cx + 6, y1 + 0.5 * h)]
    elif posture == "lying":
        sh, hp = [(x1 + 0.2 * w, cy - 5), (x1 + 0.2 * w, cy + 5)], [(x1 + 0.5 * w, cy - 5), (x1 + 0.5 * w, cy + 5)]
    elif posture == "sit_floor":  # legs stretched out: wide box, torso upright
        sh, hp = [(x1 + 15, y1 + 12), (x1 + 25, y1 + 12)], [(x1 + 15, y1 + 55), (x1 + 25, y1 + 55)]
    elif posture == "bend":       # bent over at the hips: torso horizontal but hips high
        sh, hp = [(x1 + 10, y1 + 20), (x1 + 10, y1 + 30)], [(x1 + 70, y1 + 20), (x1 + 70, y1 + 30)]
    else:
        raise ValueError(posture)
    for i, (x, y) in zip((5, 6, 11, 12), sh + hp):
        k[i] = (x, y, 0.9)
    return k


def lerp(a, b, u):
    return tuple(float(p + (q - p) * u) for p, q in zip(a, b))


def sequence(end_box, end_posture, transition_s=0.6, stand_s=2.0, hold_s=5.0, then=None):
    """[(t, box, posture)]: stand, move to end_box over transition_s, hold; optional (box, posture, s) afterwards."""
    out, t = [], 0.0
    for _ in range(int(stand_s * FPS)):
        out.append((t, STAND, "upright")); t += 1 / FPS
    n = max(1, int(round(transition_s * FPS)))
    for i in range(1, n):
        out.append((t, lerp(STAND, end_box, i / n), None)); t += 1 / FPS
    for _ in range(int(hold_s * FPS)):
        out.append((t, end_box, end_posture)); t += 1 / FPS
    if then:
        for _ in range(int(then[2] * FPS)):
            out.append((t, then[0], then[1])); t += 1 / FPS
    return out


LYING = (235.0, 228.0, 405.0, 270.0)       # 170 x 42: horizontal, near the ground
SIT_CHAIR = (300.0, 160.0, 350.0, 270.0)   # 50 x 110, aspect 0.45
SIT_FLOOR = (300.0, 190.0, 390.0, 270.0)   # 90 x 80, aspect 1.1
BEND = (290.0, 175.0, 390.0, 270.0)        # 100 x 95, aspect 1.05


class FakePose:
    """Returns the scripted keypoints for the current box (set by the driver before each frame)."""
    name = "fake-pose"

    def __init__(self):
        self.cur, self.calls = [], 0

    def __call__(self, frame):
        self.calls += 1
        return list(self.cur)


def worker(tmp_path=None, cam="cam1"):
    return SimpleNamespace(id=cam, name=f"Camera {cam}", calib_error=None,
                           site=SimpleNamespace(data=tmp_path), backend=SimpleNamespace(info={}))


def run(seq, pose=True, tmp_path=None, **kw):
    fp = FakePose() if pose else None
    an = FL.FallAnalyzer(fp, **kw)
    w = worker(tmp_path)
    frame = np.zeros((480, 640, 3), np.uint8)
    events = []
    for t, box, posture in seq:
        if fp is not None:
            fp.cur = [(np.array(box, np.float32), kps(box, posture))] if posture else []
        events += an.on_frame(w, 1000.0 + t, frame, [(3, box, "person", 0.9)], [])
    return events, an


def test_fall_then_lying_fires_once_with_observable_text(tmp_path):
    ev, an = run(sequence(LYING, "lying"), tmp_path=tmp_path)
    assert len(ev) == 1
    e = ev[0]
    assert e.rule == "person_down" and e.severity == 3 and e.track_ids == [3]
    d = e.derivation
    assert d["learned_model"] is None and d["episode"]["descent"]["drop_upright_heights"] >= 0.35
    assert d["episode"]["lying_s"] >= 3.0 and d["method"].startswith("posture state machine (pose")
    low = e.observation.lower()
    assert "upright to horizontal" in low and "does not establish" in low
    assert not any(b in low for b in BANNED)
    assert e.evidence.get("crop", "").startswith("evidence/")
    assert LABEL["person_down"] == "Person on the ground / possible fall" and SEVERITY["person_down"] == 3
    _, prio, f = explain_priority([e])
    assert not f["learned_model_only"] and prio in ("P1", "P2")  # a deterministic rule is not capped


def test_lying_shorter_than_persist_then_getting_up_does_not_fire():
    ev, _ = run(sequence(LYING, "lying", hold_s=2.0, then=(STAND, "upright", 3.0)))
    assert ev == []


def test_sitting_and_bending_do_not_fire():
    for box, posture in ((SIT_CHAIR, "upright"), (SIT_FLOOR, "sit_floor"), (BEND, "bend")):
        ev, _ = run(sequence(box, posture, hold_s=8.0))
        assert ev == [], posture


def test_boxes_only_mode_still_detects_the_fall_but_cannot_tell_bending():
    ev, an = run(sequence(LYING, None), pose=False)
    assert len(ev) == 1 and "boxes only" in ev[0].derivation["method"]
    ev, _ = run(sequence(BEND, None, hold_s=8.0), pose=False)
    assert len(ev) == 1  # documented limitation: without keypoints a long deep bend looks like lying


def test_slow_lying_down_needs_opt_in_and_is_worded_differently():
    seq = sequence(LYING, "lying", transition_s=6.0)
    assert run(seq)[0] == []
    ev, _ = run(seq, params=FL.FallParams(report_without_descent=True))
    assert len(ev) == 1 and "no rapid descent" in ev[0].observation


def test_truncated_box_at_image_edge_is_ignored():
    edge = (200.0, 420.0, 400.0, 480.0)  # touches the bottom of the 480 px frame
    ev, _ = run(sequence(edge, None, hold_s=6.0), pose=False)
    assert ev == []


def test_rearm_and_recovery():
    seq = sequence(LYING, "lying", then=(STAND, "upright", 2.0))
    seq2 = [(t + seq[-1][0] + 0.2, b, p) for t, b, p in sequence(LYING, "lying")]
    ev, _ = run(seq + seq2, params=FL.FallParams(rearm_s=1.0))
    assert len(ev) == 2
    ev, _ = run(seq + seq2)  # default re-arm 60 s: the second episode on the same track is suppressed
    assert len(ev) == 1


def test_pose_runs_only_when_someone_looks_non_upright():
    standing = [(i / FPS, STAND, "upright") for i in range(20)]
    _, an = run(standing)
    assert an.stats["pose_runs"] == 0
    _, an = run(sequence(LYING, "lying"))
    assert an.stats["pose_runs"] > 0


def test_classifier_alone_is_learned_and_capped():
    clf = lambda f: 0.95 if f[-1, 0] > 2 else 0.05  # "wide box" classifier on the shared features
    seq = sequence(LYING, "lying", transition_s=6.0)  # no rapid descent: the heuristic stays silent
    ev, _ = run(seq, classifier=clf, classifier_threshold=0.8)
    assert len(ev) == 1
    e = ev[0]
    assert e.derivation["learned_model"] and "learned posture classifier" in e.observation
    e.criticality, e.confidence = 3, 1.0
    assert explain_priority([e])[1] == "P3"
    ev, _ = run(sequence(LYING, "lying"), classifier=clf)  # heuristic + classifier agree: deterministic, conf 0.8
    assert len(ev) == 1 and ev[0].derivation["learned_model"] is None and ev[0].confidence == 0.8


def test_features_and_torso_angle():
    assert FL.torso_angle(kps(STAND, "upright")) == pytest.approx(0.0, abs=1e-6)
    assert FL.torso_angle(kps(LYING, "lying")) == pytest.approx(90.0, abs=1e-6)
    assert FL.torso_angle(np.zeros((17, 3))) is None and FL.torso_angle(None) is None
    hist = [FL.Sample(i / FPS, STAND) for i in range(10)] + [FL.Sample(2 + i / FPS, LYING) for i in range(5)]
    f = FL.track_features(hist, 170.0, 2.8)
    assert f.shape == (20, 5) and f[-1, 0] > 3 and f[0, 0] < 0.3


def test_install_boxes_only_fallback_and_pose_sharing(tmp_path):
    site = SimpleNamespace(cfg={}, analyzers=[], _base=tmp_path)
    assert FL.install(site) is None and site.fall is None
    site.cfg = {"fall": {"enabled": True}}

    def broken(w):
        raise FileNotFoundError(w)
    an = FL.install(site, pose_factory=broken)
    assert an.pose is None and "boxes only" in an.mode() and site.fall_status["warnings"]

    class AltercationAnalyzer:  # stand-in for the real analyzer: only its pose callable is shared
        def __init__(self):
            self.pose = FakePose()
    alt = AltercationAnalyzer()
    site2 = SimpleNamespace(cfg={"fall": {"enabled": True}}, analyzers=[alt], _base=tmp_path)
    an2 = FL.install(site2)
    assert isinstance(alt.pose, FL.PoseCache) and an2.pose is alt.pose
    frame = np.zeros((4, 4, 3), np.uint8)
    alt.pose(frame); an2.pose(frame)
    assert alt.pose.runs == 1  # one model run per frame for both analyzers
