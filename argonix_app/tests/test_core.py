import unittest

from argonix.correlate import Correlator
from argonix.geometry import apply_h, homography_from_points, invert_h, point_in_polygon, rms_error, signed_depth
from argonix.models import TrackObs
from argonix.rules import RuleEngine, Zone

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def obs(tid, t, x, y, conf=0.9):
    return TrackObs("cam1", tid, t, (0, 0, 1, 1), "person", conf, (x, y))


def run(engine, frames):
    out = []
    for t, items in frames:
        out += engine.update(t, items)
    return out


class Geometry(unittest.TestCase):
    def test_homography_roundtrip(self):
        img = [(100, 400), (540, 400), (620, 470), (20, 470)]
        gnd = [(0, 10), (8, 10), (8, 0), (0, 0)]
        h = homography_from_points(img, gnd)
        for p, g in zip(img, gnd):
            q = apply_h(h, p)
            self.assertAlmostEqual(q[0], g[0], 6); self.assertAlmostEqual(q[1], g[1], 6)
        back = apply_h(invert_h(h), (4, 5))
        self.assertAlmostEqual(apply_h(h, back)[0], 4, 6)
        self.assertLess(rms_error(h, img, gnd), 1e-6)

    def test_degenerate(self):
        with self.assertRaises(ValueError):
            homography_from_points([(0, 0), (1, 1), (2, 2), (3, 3)], SQ)

    def test_polygon(self):
        self.assertTrue(point_in_polygon((5, 5), SQ))
        self.assertFalse(point_in_polygon((11, 5), SQ))
        self.assertAlmostEqual(signed_depth((5, 1), SQ), 1.0)
        self.assertAlmostEqual(signed_depth((12, 5), SQ), -2.0)


class Rules(unittest.TestCase):
    def zone(self, **rules):
        return Zone.from_dict({"id": "z", "name": "Dock", "polygon": SQ, "criticality": 3,
                               "direction": [1, 0], "rules": rules})

    def test_intrusion_fires_once_after_persistence(self):
        e = RuleEngine("cam1", [self.zone(intrusion={"min_seconds": 1.0})], calib_error_m=0.3)
        frames = [(i * 0.2, [obs(1, i * 0.2, 5, 5)]) for i in range(20)]
        ev = run(e, frames)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].rule, "intrusion")
        self.assertGreaterEqual(ev[0].t_end - ev[0].t_start, 1.0)
        self.assertIn("track #1", ev[0].observation)

    def test_intrusion_respects_calibration_margin(self):
        e = RuleEngine("cam1", [self.zone(intrusion={"min_seconds": 0.5})], calib_error_m=0.5)
        frames = [(i * 0.2, [obs(1, i * 0.2, 5, 0.2)]) for i in range(20)]  # only 0.2 m inside
        self.assertEqual(run(e, frames), [])

    def test_brief_entry_does_not_fire(self):
        e = RuleEngine("cam1", [self.zone(intrusion={"min_seconds": 2.0})])
        frames = [(i * 0.2, [obs(1, i * 0.2, 5, 5)]) for i in range(5)]
        frames += [(1 + i * 0.2, [obs(1, 1 + i * 0.2, 20, 5)]) for i in range(20)]
        self.assertEqual(run(e, frames), [])

    def test_flicker_and_reentry_do_not_duplicate(self):
        e = RuleEngine("cam1", [self.zone(intrusion={"min_seconds": 1.0})], rearm_s=30)
        frames = [(i * 0.2, [obs(1, i * 0.2, 5, 5)]) for i in range(10)]            # fires at ~1 s
        frames += [(5 + i * 0.2, [obs(1, 5 + i * 0.2, 5, 5)]) for i in range(10)]    # 3 s gap, back inside
        self.assertEqual(len(run(e, frames)), 1)
        later = [(40 + i * 0.2, [obs(1, 40 + i * 0.2, 5, 5)]) for i in range(10)]    # after re-arm window
        self.assertEqual(len(run(e, later)), 1)

    def test_loitering(self):
        e = RuleEngine("cam1", [self.zone(loitering={"min_seconds": 10})])
        frames = [(i * 0.5, [obs(2, i * 0.5, 4 + (i % 3) * 0.1, 4)]) for i in range(30)]
        ev = run(e, frames)
        self.assertEqual([x.rule for x in ev], ["loitering"])

    def test_running_and_walking(self):
        e = RuleEngine("cam1", [self.zone(running={"speed_mps": 2.5, "min_seconds": 0.6})])
        walk = [(i * 0.2, [obs(3, i * 0.2, 1 + 1.2 * i * 0.2, 5)]) for i in range(30)]
        self.assertEqual(run(e, walk), [])
        e2 = RuleEngine("cam1", [self.zone(running={"speed_mps": 2.5, "min_seconds": 0.6})])
        run_ = [(i * 0.2, [obs(4, i * 0.2, 0.5 + 4 * i * 0.2, 5)]) for i in range(12)]
        ev = run(e2, run_)
        self.assertEqual(len(ev), 1)
        self.assertGreater(ev[0].derivation["speed_mps"], 2.5)

    def test_running_pixel_space_body_heights(self):
        big = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        z = Zone.from_dict({"id": "z", "name": "View", "polygon": big, "criticality": 1,
                            "rules": {"running": {"speed_bh_s": 1.5, "min_seconds": 0.6}}})

        def px(tid, t, x, h=60.0, scale=2.0):  # bbox in analytics px, ground = foot point in source px
            y2 = 200.0
            return TrackObs("cam1", tid, t, (x / scale - 10, y2 - h, x / scale + 10, y2), "person", 0.9,
                            (x, y2 * scale))
        # 120 source-px tall person (60 analytics px x 2): walking 100 px/s = 0.83 bh/s, running 240 px/s = 2 bh/s
        walk = [(i * 0.2, [px(1, i * 0.2, 100 + 100 * i * 0.2)]) for i in range(30)]
        self.assertEqual(run(RuleEngine("cam1", [z], calib_error_m=8, units="px"), walk), [])
        fast = [(i * 0.2, [px(2, i * 0.2, 100 + 240 * i * 0.2)]) for i in range(15)]
        ev = run(RuleEngine("cam1", [z], calib_error_m=8, units="px"), fast)
        self.assertEqual([x.rule for x in ev], ["running"])
        self.assertAlmostEqual(ev[0].derivation["box_height_px"], 120.0)
        self.assertGreater(ev[0].derivation["speed_bh_s"], 1.5)
        self.assertEqual(ev[0].derivation["space"], "image_px")
        # without speed_bh_s, running stays metric-only and is skipped on an uncalibrated camera
        z2 = Zone.from_dict({"id": "z", "name": "View", "polygon": big, "criticality": 1,
                             "rules": {"running": {"speed_mps": 2.5, "min_seconds": 0.6}}})
        self.assertEqual(run(RuleEngine("cam1", [z2], calib_error_m=8, units="px"), fast), [])

    def test_wrong_way(self):
        e = RuleEngine("cam1", [self.zone(wrong_way={"min_seconds": 1.0})])
        with_flow = [(i * 0.2, [obs(5, i * 0.2, 1 + i * 0.2, 5)]) for i in range(20)]
        self.assertEqual(run(e, with_flow), [])
        against = [(10 + i * 0.2, [obs(6, 10 + i * 0.2, 9 - i * 0.2, 5)]) for i in range(20)]
        self.assertEqual([x.rule for x in run(e, against)], ["wrong_way"])

    def test_crowding(self):
        e = RuleEngine("cam1", [self.zone(crowding={"max_count": 3, "min_seconds": 1.0})])
        frames = [(i * 0.25, [obs(k, i * 0.25, 2 + k, 5) for k in range(5)]) for i in range(10)]
        ev = run(e, frames)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].track_ids, [0, 1, 2, 3, 4])

    def test_non_person_ignored(self):
        e = RuleEngine("cam1", [self.zone(intrusion={"min_seconds": 0.2})])
        o = TrackObs("cam1", 9, 0, (0, 0, 1, 1), "car", 0.9, (5, 5))
        self.assertEqual(run(e, [(i * 0.2, [o]) for i in range(10)]), [])


class Correlation(unittest.TestCase):
    def ev(self, rule, cam, t, tid=1, site="dock"):
        e = RuleEngine(cam, [Zone.from_dict({"id": "z", "name": "Dock", "polygon": SQ, "criticality": 3,
                                             "site_zone": site, "rules": {rule: {"min_seconds": 0.2}}})])
        out = run(e, [(t + i * 0.2, [TrackObs(cam, tid, t + i * 0.2, (0, 0, 1, 1), "person", 0.9, (5, 5))])
                      for i in range(5)])
        return out[0]

    def test_cross_camera_corroboration_raises_priority(self):
        c = Correlator(window_s=10)
        a1, new1 = c.add(self.ev("loitering", "cam1", 0))
        s1 = a1.score
        a2, new2 = c.add(self.ev("intrusion", "cam2", 2, tid=7))
        self.assertTrue(new1); self.assertFalse(new2)
        self.assertIs(a1, a2)
        self.assertEqual(a2.cameras, ["cam1", "cam2"])
        self.assertGreater(a2.score, s1)
        self.assertEqual(a2.priority, "P1")

    def test_window_expiry_creates_new_alert(self):
        c = Correlator(window_s=5)
        a1, _ = c.add(self.ev("intrusion", "cam1", 0))
        a2, new = c.add(self.ev("intrusion", "cam1", 30))
        self.assertTrue(new); self.assertIsNot(a1, a2)

    def test_alert_dict_has_explainability_fields(self):
        a, _ = Correlator().add(self.ev("intrusion", "cam1", 0))
        d = a.to_dict()
        e = d["events"][0]
        for k in ("observation", "derivation", "confidence", "camera_id", "zone_id"):
            self.assertIn(k, e)
        self.assertIn("threshold", e["derivation"])
        self.assertIn("disclaimer", d)


if __name__ == "__main__":
    unittest.main()


class LearnedModelGate(unittest.TestCase):
    def ev(self, rule, learned, conf=1.0, crit=3, tid=1):
        from argonix.models import Event
        from argonix.rules import register_rule
        register_rule("object_threat", "Possible weapon-like object", 3)
        return Event(rule, "cam1", "z", "Dock", [tid], 0, 1, "obs", {"learned_model": "m" if learned else None,
                     "threshold": {}}, conf, 3, crit)

    def test_learned_only_alert_capped_until_rule_agrees(self):
        c = Correlator()
        a, _ = c.add(self.ev("object_threat", True))
        self.assertEqual(a.priority, "P3")  # score 9 would be P1, but learned-only is capped
        self.assertTrue(a.factors["learned_model_only"] and a.factors["cap"])
        a, _ = c.add(self.ev("intrusion", False))  # a deterministic rule on the same zone corroborates
        self.assertEqual(a.priority, "P1")
        self.assertIsNone(a.factors["cap"])
        self.assertEqual(a.factors["distinct_rules"], 2)

    def test_factors_explain_score(self):
        a, _ = Correlator().add(self.ev("intrusion", False, conf=0.8))
        f = a.factors
        self.assertAlmostEqual(f["base_score"], f["severity"] * f["zone_criticality"] * f["confidence"], 3)
        self.assertEqual(a.score, f["score"])
