"""Crowd flow / counting lines / density / queue analytics, unattended objects, trajectory-anomaly corroboration,
CUSUM + event-rate + dwell baselines, and the analytics API. CPU only, synthetic tracks, no model weights."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argonix.anomaly import IsolationForest, TrajectoryAnomalyAnalyzer, track_features
from argonix.baseline import Baseline, Cusum, hour_of_week
from argonix.correlate import Correlator
from argonix.flow import FlowAnalyzer, dir_bin, polygon_area, screen_angle
from argonix.models import Event, TrackObs
from argonix.objects import UnattendedObjectAnalyzer
from argonix.rules import LABEL, SEVERITY, Zone

ROOT = Path(__file__).resolve().parents[1]
PPM = 100.0  # synthetic calibrated camera: 100 image px per ground metre, ground = image / 100
H_CAL = [[1 / PPM, 0, 0], [0, 1 / PPM, 0], [0, 0, 1.0]]
FRAME = np.full((1200, 2000, 3), 90, np.uint8)
BANNED = ("dangerous", "criminal", "suspect", "attacker", "thief", "terror", "intent to", "abandoned by")
T0 = 1_790_000_000.0


def worker(tmp_path=None, calibrated=True, zones=(), lines=(), cam="cam1"):
    raw = [dict(z) for z in zones]
    zobjs = [Zone.from_dict({**z, "polygon": z.get("polygon") or z["polygon_px"]}) for z in raw]
    return SimpleNamespace(id=cam, name=f"Camera {cam}", H=H_CAL if calibrated else None,
                           calib_error=0.3 if calibrated else None, zones=zobjs if calibrated else [],
                           px_zones=[] if calibrated else zobjs, cfg={"zones": raw, "lines": list(lines)},
                           scale=1.0, site=SimpleNamespace(data=tmp_path), engine=SimpleNamespace(eps={}),
                           backend=SimpleNamespace(info={"detector": "fake"}))


def person(tid, gx, gy, conf=0.9, h_m=1.7):
    """A person det whose foot point is at ground (gx, gy) metres (= image px / 100)."""
    fx, fy = gx * PPM, gy * PPM
    return (tid, (fx - 20, fy - h_m * PPM, fx + 20, fy), "person", conf)


def bag(tid, gx, gy, conf=0.6, label="backpack"):
    fx, fy = gx * PPM, gy * PPM
    return (tid, (fx - 15, fy - 30, fx + 15, fy), label, conf)


def step(an, w, t, dets):
    obs = []
    for tid, bb, cls, conf in dets:
        g = ((bb[0] + bb[2]) / 2 / PPM, bb[3] / PPM) if w.H is not None else None
        obs.append(TrackObs(w.id, tid, t, bb, cls, conf, g))
    return an.on_frame(w, t, FRAME, list(dets), obs)


def run(an, w, t0, seconds, positions, fps=5.0):
    """positions(t_rel) -> list of dets; returns all events."""
    ev, n = [], int(seconds * fps)
    for i in range(n):
        t = t0 + i / fps
        ev += step(an, w, t, positions(i / fps))
    return ev


def no_labels(text):
    return not any(b in text.lower() for b in BANNED)


# ---------------------------------------------------------------------------------------------- lines
def test_line_crossing_counts_both_directions():
    ln = {"id": "door", "name": "Door", "points_px": [[0, 500], [1000, 500]], "in_label": "entering"}
    w = worker(calibrated=False, lines=[ln])
    an = FlowAnalyzer()

    def pos(t):
        d = []
        d.append(person(1, 5.0, 3.0 + 0.8 * t))            # walks down across y=500 px -> "in" (n = (0, 1))
        d.append(person(2, 7.0, 7.0 - 0.8 * t))            # walks up -> "out"
        d.append(person(3, 3.0, 5.0 + 0.02 * math.sin(4 * t)))  # jitters on the line (within the band): nothing
        d.append(person(4, 15.0, 3.0 + 0.8 * t))           # crosses the line's extension (x = 1500 px): nothing
        return d
    run(an, w, T0, 5.0, pos)
    s = an.summary_lines()["cameras"][0]["lines"][0]
    assert s["counts"] == {"in": 1, "out": 1}, s
    assert s["labels"]["in"] == "entering" and s["counts_60m"] == {"in": 1, "out": 1}
    assert len(s["series"]) == 60 and sum(r["in"] for r in s["series"]) == 1

    # back and forth: each real crossing counts
    an2 = FlowAnalyzer()
    run(an2, worker(calibrated=False, lines=[ln]), T0, 10.0,
        lambda t: [person(9, 5.0, 5.0 - 1.5 * math.cos(2 * math.pi * t / 5))])  # crosses at 1.25, 3.75, 6.25, 8.75 s
    assert an2.summary_lines()["cameras"][0]["lines"][0]["counts"] == {"in": 2, "out": 2}


def test_ground_metre_line_needs_calibration():
    ln = {"id": "g", "points": [[0, 5], [10, 5]]}
    an = FlowAnalyzer()
    run(an, worker(calibrated=True, lines=[ln]), T0, 5.0, lambda t: [person(1, 5.0, 3.0 + 0.8 * t)])
    assert an.summary_lines()["cameras"][0]["lines"][0]["counts"]["in"] == 1
    an = FlowAnalyzer()
    run(an, worker(calibrated=False, lines=[ln]), T0, 1.0, lambda t: [person(1, 5.0, 3.0)])
    cam = an.summary_lines()["cameras"][0]
    assert cam["lines"] == [] and "need calibration" in cam["warnings"][0]


# ---------------------------------------------------------------------------------------------- zones / flow
def test_zone_entries_exits_dwell_and_direction():
    z = {"id": "plaza", "name": "Plaza", "polygon": [[2, 0], [8, 0], [8, 10], [2, 10]], "rules": {}}
    an = FlowAnalyzer()
    w = worker(zones=[z])
    # 4 people walk east (+x) through the 6 m zone at 1.2 m/s, one after another
    run(an, w, T0, 40.0, lambda t: [person(i, -1 + 1.2 * (t - 4 * i), 2 + i) for i in range(4) if t >= 4 * i])
    s = an.summary_zones()["cameras"][0]["zones"][0]
    assert s["entries_total"] == 4 and s["exits_total"] == 4, s
    assert s["dwell"]["n"] == 4 and 4.0 <= s["dwell"]["p50_s"] <= 5.5  # 6 m at 1.2 m/s = 5 s
    assert s["flow"]["dominant_deg"] is not None and (s["flow"]["dominant_deg"] < 10 or s["flow"]["dominant_deg"] > 350)
    assert s["flow"]["concentration"] > 0.9 and s["flow"]["hist"][0] > 0.9  # bin E
    assert s["flow"]["frame"].startswith("map")
    assert s["density"]["area_m2"] == 60.0 and s["density"]["level"] == "normal"


def test_direction_conventions():
    assert dir_bin(screen_angle((1, 0), "ground_m")) == 0            # E
    assert dir_bin(screen_angle((0, 1), "ground_m")) == 2            # ground +y = up on the map = N
    assert dir_bin(screen_angle((0, -1), "image_px")) == 2           # image up = N
    assert dir_bin(screen_angle((-1, 1), "image_px")) == 5           # image left-down = SW
    assert polygon_area([(0, 0), (2, 0), (2, 3), (0, 3)]) == 6.0


def test_counterflow_learned_and_configured_reference():
    z = {"id": "corr", "name": "Corridor", "polygon": [[0, 0], [30, 0], [30, 4], [0, 4]],
         "rules": {"counterflow": {"min_seconds": 2.0}}}
    an = FlowAnalyzer(half_life_s=300)
    w = worker(zones=[z])

    def east(t, n=6, off=0):  # a new track id per lap, so the wrap-around is not a (backwards) jump
        return [person(off + i + 1000 * int((1.3 * t + 5 * i) // 30), (1.3 * t + 5 * i) % 30, 0.5 + 0.5 * i)
                for i in range(n)]
    ev = run(an, w, T0, 60.0, lambda t: east(t))
    assert ev == []  # everyone with the flow: nothing
    s = an.summary_zones()["cameras"][0]["zones"][0]["flow"]
    assert s["concentration"] > 0.9
    # now 4 people walk west through the eastbound crowd
    ev = run(an, w, T0 + 60, 8.0, lambda t: east(t + 60) + [person(100 + i, 28 - 1.4 * t, 0.7 + 0.8 * i) for i in range(4)])
    cf = [e for e in ev if e.rule == "counterflow"]
    assert len(cf) == 1, [e.rule for e in ev]
    e = cf[0]
    d = e.derivation
    assert d["reference"]["source"] == "learned dominant flow" and d["fraction_against"] >= 0.3
    assert set(e.track_ids) == {100, 101, 102, 103} and e.severity == SEVERITY["counterflow"]
    assert no_labels(e.observation) and LABEL["counterflow"] == "Counter-flow in zone"

    # configured permitted direction wins over the learned one (here: permitted = west, so the eastbound crowd is
    # the counter-flow and the four westbound people are with it)
    z2 = {**z, "direction": [-1, 0]}
    an2 = FlowAnalyzer()
    ev = run(an2, worker(zones=[z2]), T0, 6.0, lambda t: east(t))
    assert [e.derivation["reference"]["source"] for e in ev] == ["permitted direction (configured)"]


def test_bidirectional_corridor_has_no_dominant_flow():
    z = {"id": "corr", "name": "Corridor", "polygon": [[0, 0], [30, 0], [30, 4], [0, 4]],
         "rules": {"counterflow": {"min_seconds": 1.0}}}
    an = FlowAnalyzer()
    ev = run(an, worker(zones=[z]), T0, 60.0,
             lambda t: [person(i, (1.3 * t + 6 * i) % 30, 0.5) for i in range(4)]
             + [person(10 + i, 30 - (1.3 * t + 6 * i) % 30, 3.0) for i in range(4)])
    assert [e for e in ev if e.rule == "counterflow"] == []
    assert an.summary_zones()["cameras"][0]["zones"][0]["flow"]["concentration"] < 0.3


def test_density_with_and_without_calibration():
    z = {"id": "pen", "name": "Pen", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
         "rules": {"density": {"min_seconds": 2.0}}}
    an = FlowAnalyzer()
    w = worker(zones=[z])
    grid = [(0.25 + 0.5 * (i % 4), 0.25 + 0.5 * (i // 4)) for i in range(16)]
    ev = run(an, w, T0, 4.0, lambda t: [person(i, *grid[i]) for i in range(10)])  # 10 in 4 m2 = 2.5 /m2
    assert [(e.rule, e.derivation["level"]) for e in ev] == [("density", "warning")]
    e = ev[0]
    assert e.derivation["density_ppm2"] == 2.5 and e.derivation["area_m2"] == 4.0 and e.severity == 2
    lo, hi = e.derivation["density_range_ppm2"]
    assert lo < 2.5 < hi  # the calibration error widens the area -> an honest range
    ev = run(an, w, T0 + 4, 4.0, lambda t: [person(i, *grid[i]) for i in range(16)])  # 4 /m2: critical
    assert [(e.rule, e.derivation["level"], e.severity) for e in ev] == [("density", "critical", 3)]
    s = an.summary_zones()["cameras"][0]["zones"][0]
    assert s["density"]["ppm2"] == 4.0 and s["density"]["level"] == "critical" and s["density_note"] is None

    # uncalibrated: the same people counted, no density (count only), no density events
    zpx = {**z, "polygon_px": [[0, 0], [200, 0], [200, 200], [0, 200]], "polygon": None,
           "rules": {}}  # the pipeline strips metric rules from pixel zones
    an = FlowAnalyzer()
    ev = run(an, worker(calibrated=False, zones=[zpx]), T0, 3.0, lambda t: [person(i, *grid[i]) for i in range(16)])
    s = an.summary_zones()["cameras"][0]["zones"][0]
    assert ev == [] and s["occupancy"] == 16 and s["density"] is None and "count" in s["density_note"]
    assert s["space"] == "image_px"


def test_queue_littles_law_wait_and_rule():
    # queue lane x in [0, 10] m; service line at x = 10 (people leave the queue by crossing it to +x)
    z = {"id": "q", "name": "Ticket queue", "type": "queue", "polygon": [[0, 0], [10, 0], [10, 2], [0, 2]],
         "service_line": [[10, -1], [10, 3]], "rules": {"queue_long": {"max_length": 3, "min_seconds": 5}}}
    an = FlowAnalyzer(queue_window_s=300)
    w = worker(zones=[z])
    W, gap = 30.0, 6.0  # each person spends 30 s in the queue; one arrival every 6 s -> L = 5 (Little)

    def pos(t):
        out = []
        for i in range(int(t // gap) + 1):
            a = i * gap
            r = t - a
            if r < 0 or r > W + 3:
                continue
            x = 10.0 * min(r, W) / W if r <= W else 10.0 + 1.0 * (r - W)  # walk through the lane, then leave
            out.append(person(i, x, 1.0))
        return out
    ev = run(an, w, T0, 400.0, pos)
    q = an.summary_zones()["cameras"][0]["zones"][0]["queue"]
    assert q["method"] == "service-line crossings"
    assert 4.0 <= q["length_mean_window"] <= 6.0, q
    assert 0.8 * W <= q["wait_est_s"] <= 1.2 * W, q          # Little's law estimate
    assert abs(q["measured_wait_p50_s"] - W) <= 2.0, q         # per-track entry -> service crossing
    assert q["service_rate_per_min"] == pytest.approx(10.0, rel=0.15) and q["steady_state"] is True
    ql = [e for e in ev if e.rule == "queue_long"]
    assert len(ql) >= 1 and ql[0].derivation["condition"] == "length" and ql[0].derivation["queue"]["note"]
    assert no_labels(ql[0].observation)
    # a longer limit: nothing
    z2 = {**z, "rules": {"queue_long": {"max_length": 10, "min_seconds": 5}}}
    an2 = FlowAnalyzer()
    assert [e for e in run(an2, worker(zones=[z2]), T0, 200.0, pos) if e.rule == "queue_long"] == []


def test_dispersal_and_convergence_cues():
    z = {"id": "sq", "name": "Square", "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]],
         "rules": {"dispersal": {}, "convergence": {}}}
    ang = [2 * math.pi * i / 6 for i in range(6)]

    def ring(r):
        return [person(i, 10 + r * math.cos(a), 10 + r * math.sin(a)) for i, a in enumerate(ang)]
    an = FlowAnalyzer()
    w = worker(zones=[z])
    ev = run(an, w, T0, 10.0, lambda t: ring(1.0 + 0.02 * math.sin(t)))  # standing in a group
    ev += run(an, w, T0 + 10, 2.5, lambda t: ring(1.0 + 3.0 * t))         # everyone runs outward at 3 m/s
    rules = [e.rule for e in ev]
    assert rules == ["dispersal"], rules
    assert ev[0].derivation["median_radial_speed_mps"] >= 2.5 and no_labels(ev[0].observation)
    an = FlowAnalyzer()
    ev = run(an, worker(zones=[z]), T0, 10.0, lambda t: ring(9.0))
    ev += run(an, worker(zones=[z]), T0 + 10, 2.5, lambda t: ring(9.0 - 3.0 * t))
    assert [e.rule for e in ev] == ["convergence"]
    # ordinary walking through the square in several directions: no cue
    an = FlowAnalyzer()
    ev = run(an, worker(zones=[z]), T0, 30.0,
             lambda t: [person(i, 10 + (1.2 * t % 18 - 9) * math.cos(a), 10 + (1.2 * t % 18 - 9) * math.sin(a))
                        for i, a in enumerate(ang)])
    assert ev == []


# ---------------------------------------------------------------------------------------------- objects
def obj_run(an, w, t0, seconds, fn, fps=5.0):
    ev = []
    for i in range(int(seconds * fps)):
        ev += step(an, w, t0 + i / fps, fn(i / fps))
    return ev


def test_unattended_object_owner_leaves(tmp_path):
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10, leave_dist_m=3.0)
    w = worker(tmp_path)

    def scene(t):
        if t < 5:    # person 7 walks in carrying the bag (bag moves with them)
            px = 2 + 0.8 * t
            return [person(7, px, 5.0), bag(1_000_001, px + 0.1, 4.6)]
        if t < 8:    # stands still, bag put down at their feet
            return [person(7, 6.0, 5.0), bag(1_000_001, 6.3, 5.0)]
        if t < 18:   # walks away (> 3 m) and out of view
            return [person(7, 6.0 + 1.2 * (t - 8), 5.0), bag(1_000_001, 6.3, 5.0)] if t < 14 else [bag(1_000_001, 6.3, 5.0)]
        return [bag(1_000_001, 6.3, 5.0)]
    ev = obj_run(an, w, T0, 60.0, scene)
    assert [e.rule for e in ev] == ["unattended_object"], an.summary()
    e = ev[0]
    d = e.derivation
    assert d["last_associated_person_track"] == 7 and d["owner_dist_at_drop_m"] <= 0.5
    assert d["object_label"] == "backpack" and e.track_ids == [1_000_001] and e.severity == 2
    assert "last associated person track #7" in e.observation and no_labels(e.observation)
    assert "not an accusation" in d["note"] and d["distance_method"].startswith("ground")
    assert (tmp_path / e.evidence["crop"]).is_file() and (tmp_path / e.evidence["drop_crop"]).is_file()
    # fired after static (>= 5 s from the drop at ~t=5) + unattended (10 s from when the owner passed 3 m)
    assert 15.0 <= e.t_end - T0 <= 25.0, e.t_end - T0
    assert an.summary()["cameras"][0]["objects"][0]["alerted"] is True


def test_unattended_object_owner_stays_or_id_switch(tmp_path):
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10)
    w = worker(tmp_path)
    ev = obj_run(an, w, T0, 60.0, lambda t: [person(3, 6.0, 5.0), bag(1_000_002, 6.4, 5.0)])
    assert ev == []  # owner right next to it the whole time
    assert an.summary()["cameras"][0]["objects"][0]["state"] == "attended"

    # the owner's track ID switches while they stand next to the bag: the new track stays > attend_s -> attended
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10, attend_s=3)
    ev = obj_run(an, w, T0, 60.0, lambda t: [person(3 if t < 10 else 44, 6.0, 5.0), bag(1_000_002, 6.4, 5.0)])
    assert ev == []
    assert an.summary()["cameras"][0]["objects"][0]["last_associated_person_track"] == 44

    # passers-by within 1 m for < attend_s do not reset the timer once the owner has left
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10, attend_s=3)

    def scene(t):
        d = [bag(1_000_003, 6.4, 5.0)]
        if t < 3:
            d.append(person(5, 6.0, 5.0))
        if int(t) % 6 == 0 and t > 8:  # someone walks past every 6 s
            d.append(person(100 + int(t), 5.5 + (t % 1) * 2, 5.5))
        return d
    ev = obj_run(an, w, T0, 60.0, scene)
    assert [e.rule for e in ev] == ["unattended_object"]


def test_unattended_object_background_object_needs_owner(tmp_path):
    w = worker(tmp_path)
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10)  # require_owner=True (default)
    assert obj_run(an, w, T0, 60.0, lambda t: [bag(1_000_009, 3.0, 3.0, label="suitcase")]) == []
    an = UnattendedObjectAnalyzer(static_s=5, unattended_s=10, require_owner=False)
    ev = obj_run(an, w, T0, 60.0, lambda t: [bag(1_000_009, 3.0, 3.0, label="suitcase")])
    assert len(ev) == 1 and ev[0].derivation["last_associated_person_track"] is None
    assert "no person was associated" in ev[0].observation


def test_unattended_object_uncalibrated_uses_approx_distance(tmp_path):
    an = UnattendedObjectAnalyzer(static_s=3, unattended_s=5)
    w = worker(tmp_path, calibrated=False)

    def scene(t):
        d = [bag(1_000_004, 6.3, 5.0)]
        if t < 5:
            d.append(person(2, 6.0, 5.0))
        elif t < 12:
            d.append(person(2, 6.0 + (t - 5) * 1.0, 5.0))  # walks away 7 "m" (= 700 px, 4 body heights)
        return d
    ev = obj_run(an, w, T0, 40.0, scene)
    assert len(ev) == 1 and ev[0].derivation["distance_method"].startswith("image approx")
    assert ev[0].ground is None


# ---------------------------------------------------------------------------------------------- anomaly
def _walk(rng, t0, kind="normal"):
    pts, t = [], t0
    if kind == "normal":
        v = rng.uniform(1.1, 1.5)
        a = rng.uniform(0, 2 * math.pi)
        x, y = rng.uniform(0, 20), rng.uniform(0, 20)
        for i in range(int(rng.uniform(10, 20) * 5)):
            pts.append((t + i / 5, x + v * i / 5 * math.cos(a) + rng.normal(0, 0.03),
                        y + v * i / 5 * math.sin(a) + rng.normal(0, 0.03)))
    else:  # zig-zag with stops and bursts
        x, y = 10.0, 10.0
        for i in range(100):
            ph = (i // 10) % 4
            dx, dy = [(3.5, 0), (0, 0), (0, 3.5), (0, 0)][ph]
            sgn = 1 if (i // 20) % 2 == 0 else -1
            x += sgn * dx / 5; y += sgn * dy / 5
            pts.append((t + i / 5, x, y))
    return pts


def feed_track(an, w, tid, pts):
    ev = []
    for (t, x, y) in pts:
        ev += step(an, w, t, [person(tid, x, y)])
    ev += step(an, w, pts[-1][0] + 5, [])  # track ends (stale)
    return ev


def test_isolation_forest_numpy_scores_outliers_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 4))
    f = IsolationForest(n_trees=50, max_samples=128, seed=1).fit(X)
    s_in = f.score(np.zeros((1, 4)))[0]
    s_out = f.score(np.full((1, 4), 6.0))[0]
    assert s_out > 0.6 > s_in and 0 < s_in < 1


def test_trajectory_anomaly_emits_learned_event_capped_at_p3(tmp_path):
    rng = np.random.default_rng(3)
    an = TrajectoryAnomalyAnalyzer(data_dir=tmp_path, min_train=120, n_trees=60, max_samples=128, background=False,
                                   refit_every=1000)
    w = worker(tmp_path)
    t, ev = T0, []
    for tid in range(1, 131):
        ev += feed_track(an, w, tid, _walk(rng, t))
        t += 30
    st = an.status()["cameras"][0]
    assert st["trained"] and st["n_train"] + st["n_calib"] == 120 and st["n_window"] == 130 and ev == []
    normal = feed_track(an, w, 500, _walk(rng, t))
    odd = feed_track(an, w, 501, _walk(rng, t + 30, kind="odd"))
    assert normal == [] and len(odd) == 1
    e = odd[0]
    assert e.rule == "trajectory_anomaly" and e.severity == 1
    lm = e.derivation["learned_model"]
    assert lm["name"] == "iforest-trajectory" and lm["version"] and e.derivation["percentile"] >= 0.99
    assert len(e.derivation["top_features"]) == 3 and no_labels(e.observation)
    assert "corroboration only" in e.observation.lower() or "corroboration" in e.derivation["note"]

    # correlation: learned-only evidence stays at or below P3, even when corroborated by another learned event
    c = Correlator(window_s=60)
    e.criticality, e.site_zone = 3, "lobby"
    twin = Event(**{**e.__dict__, "camera_id": "cam2"})  # the same site zone seen by a second camera
    a, _ = c.add(e)
    a, _ = c.add(twin)
    assert a.factors["learned_model_only"] and a.priority == "P3" and a.factors["cap"]
    det = Event("running", "cam1", "_view", "x", [501], e.t_end, e.t_end + 1, "fast", {"rule": "running"}, 0.9, 2, 3)
    a, _ = c.add(det)
    assert not a.factors["learned_model_only"] and a.priority in {"P1", "P2"}

    # persistence: a new analyzer reloads the training window and refits on start-up
    an.close()
    an2 = TrajectoryAnomalyAnalyzer(data_dir=tmp_path, min_train=120, n_trees=60, max_samples=128, background=False)
    st2 = an2.status()["cameras"][0]
    assert st2["trained"] and st2["n_window"] >= 130


def test_track_features_units():
    pts = [(i / 5, 1.2 * i / 5, 0.0, 1.0) for i in range(50)]
    f = track_features(pts, metric=True)
    assert f[0] == pytest.approx(1.2, abs=0.05) and f[3] == pytest.approx(1.0, abs=0.01)
    pts_px = [(i / 5, 120 * i / 5, 0.0, 170.0) for i in range(50)]  # 120 px/s with 170 px boxes = 0.7 bh/s
    assert track_features(pts_px, metric=False)[0] == pytest.approx(120 / 170, abs=0.03)


# ---------------------------------------------------------------------------------------------- baselines
def feed_minutes(b, t0, value, minutes):
    evs = []
    for m in range(minutes):
        for s in range(0, 60, 5):
            evs += b.observe_all("cam1", "lobby", "Lobby", t0 + m * 60 + s, value)
    return evs


def test_cusum_detects_sustained_shift_point_test_does_not(tmp_path):
    b = Baseline(tmp_path / "b.db", bucket_s=60, min_samples=8, cooldown_s=0, cusum_h=8.0, cusum_cooldown_s=0)
    week = 7 * 24 * 3600
    for i in range(10):  # history: 2 or 3 people in this hour-of-week
        feed_minutes(b, T0 + i * week, 2 + (i % 2), 1)
    t = T0 + 10 * week
    evs = feed_minutes(b, t, 4, 12)  # modified z = 0.6745 * (4 - 2.5) / 0.5 = 2.0 < 3.5 every minute
    rules = [e.rule for e in evs]
    assert "occupancy_deviation" not in rules and rules.count("occupancy_shift") >= 1, rules
    e = next(e for e in evs if e.rule == "occupancy_shift")
    assert e.derivation["side"] == "up" and e.derivation["threshold"]["cusum_h"] == 8.0
    assert "sustained shift" in e.observation and e.severity == 1
    assert b.observe("cam1", "lobby", "Lobby", t + 13 * 60, 4) is None  # observe() keeps its old contract
    cs = Cusum(k=0.5, h=5)
    assert all(cs.update(0.3) is None for _ in range(200))  # in-control residuals: no alarm
    assert b.summary_cusum()["series"][0]["zone"] == "lobby"


def test_event_rate_and_dwell_baselines(tmp_path):
    b = Baseline(tmp_path / "b.db", rate_bucket_s=3600)
    h0 = (T0 // 3600) * 3600
    b.observe_all("cam1", "lobby", "Lobby", h0 + 1, 0)        # observation starts at the top of the hour
    for k in range(3):
        b.record_event(Event("loitering", "cam1", "lobby", "Lobby", [1], h0 + 100 * k, h0 + 100 * k + 1, "", {}, 0.8, 1))
    b.record_event(Event("occupancy_deviation", "cam1", "lobby", "Lobby", [], h0, h0, "", {}, 0.8, 1))  # excluded
    b.observe_all("cam1", "lobby", "Lobby", h0 + 3600 + 5, 0)  # next hour: the first one is closed
    b.observe_all("cam1", "lobby", "Lobby", h0 + 7200 + 5, 0)  # and an empty hour
    rates = b.summary_rates()
    assert sum(r["hours"] for r in rates) == 2
    assert {r["max_per_hour"] for r in rates} == {3, 0}
    for d in (10, 20, 30, 40, 1000):
        b.add_dwell("cam1", "lobby", h0, d)
    b.add_dwell("cam1", "lobby", h0, 10 * 3600)  # longer than dwell_max_s: dropped
    q50, n = b.dwell_quantile("cam1", "lobby", hour_of_week(h0), 0.5)
    assert n == 5 and q50 == 30
    assert b.summary_dwell()[0]["p90_s"] == 1000


# ---------------------------------------------------------------------------------------------- API
class WalkBackend:
    """Two people walking down the image through a counting line and a queue zone (image px, uncalibrated OK)."""
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self):
        self.n = 0

    def track(self, frame):
        self.n += 1
        y = 120 + (self.n * 8) % 520  # down through the queue zone (100-500 px), out below it, then back to the top
        return [(1, (300, y - 100, 340, y), "person", 0.9), (2, (500, y - 90, 540, y + 10), "person", 0.9)]


def make_analytics_site(tmp_path, extra=None):
    from argonix.pipeline import Site
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cam = cfg["cameras"][0]
    cam["source"] = str(ROOT / cam["source"])
    cam["lines"] = [{"id": "mid", "name": "Middle", "points_px": [[0, 330], [960, 330]]}]
    cam["zones"].append({"id": "queue_a", "name": "Queue A", "type": "queue",
                         "polygon_px": [[250, 100], [600, 100], [600, 500], [250, 500]],
                         "service_line_px": [[250, 480], [600, 480]], "rules": {"queue_long": {"max_length": 1}}})
    cfg["cameras"] = [cam]
    cfg["record"] = False
    cfg.update(extra or {})
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: WalkBackend())


def test_analytics_api_shapes(tmp_path):
    site = make_analytics_site(tmp_path, {"anomaly": {"enabled": True, "min_train": 5}})
    site.start(); site.wait(timeout=120); site.stop()
    assert site.flow is not None and site.anomaly is not None and site.objects is None
    c = TestClient(__import__("argonix.server", fromlist=["create_app"]).create_app(site, open_demo=True))
    z = c.get("/api/analytics/zones").json()
    cam = z["cameras"][0]
    assert cam["camera"] == "cam1" and cam["calibrated"] is True and cam["name"]
    q = next(x for x in cam["zones"] if x["id"] == "queue_a")
    assert q["kind"] == "queue" and q["queue"]["method"] == "service-line crossings"
    assert q["entries_total"] >= 2 and len(q["series"]) == 60 and set(q["flow"]) >= {"bins", "hist", "dominant_deg"}
    assert q["density"]["area_m2"] > 0
    ln = c.get("/api/analytics/lines").json()["cameras"][0]["lines"][0]
    assert ln["id"] == "mid" and ln["counts"]["in"] >= 2  # walking down across a left-to-right line = "in"
    hs = c.get("/api/analytics/hotspots?window=7d").json()
    assert hs["events"] >= 1 and hs["zones"] and hs["cameras"][0]["ground_cells"]
    assert any(zz["zone_id"] == "queue_a" and "queue_long" in zz["by_rule"] for zz in hs["zones"])
    assert c.get("/api/analytics/hotspots?window=1y").status_code == 400
    an = c.get("/api/analytics/anomaly").json()
    assert an["enabled"] and an["model"] == "iforest-trajectory" and an["cameras"][0]["camera"] == "cam1"
    assert c.get("/api/analytics/objects").json()["enabled"] is False
    bl = c.get("/api/baseline").json()
    assert {"cusum", "event_rates", "dwell", "buckets"} <= set(bl) and bl["cusum"]["h"] == 8.0
    assert any(d["zone"] == "queue_a" for d in bl["dwell"])
    # lines editing: validation + persistence (restarts the camera worker)
    assert c.put("/api/cameras/cam1/lines", json={"lines": [{"id": "bad id!", "points_px": [[0, 0], [1, 1]]}]}).status_code == 400
    assert c.put("/api/cameras/cam1/lines", json={"lines": [{"id": "a", "points_px": [[5, 5], [5, 5]]}]}).status_code == 400
    r = c.put("/api/cameras/cam1/lines", json={"lines": [{"id": "door", "name": "Door", "points_px": [[0, 200], [900, 200]],
                                                           "in_label": "entering"}]})
    assert r.status_code == 200, r.text
    assert json.loads(site.config_path.read_text())["cameras"][0]["lines"][0]["id"] == "door"
    assert c.get("/api/cameras").json()[0]["lines"][0]["in_label"] == "entering"
    # zone validation knows queue zones and the new rules
    zones = c.get("/api/cameras").json()[0]["zones"]
    assert c.put("/api/cameras/cam1/zones", json={"zones": zones}).status_code == 200
    bad = [{**zones[0], "rules": {"counterflow": {"min_fraction": 2}}}]
    assert c.put("/api/cameras/cam1/zones", json={"zones": bad}).status_code == 400
    bad = [{**zones[0], "type": "stadium"}]
    assert c.put("/api/cameras/cam1/zones", json={"zones": bad}).status_code == 400
    site.stop()


def test_analytics_api_permissions_and_disabled(tmp_path):
    from argonix.auth import Auth, hash_password
    from argonix.server import create_app
    site = make_analytics_site(tmp_path, {"analytics": {"enabled": False}})
    assert site.flow is None
    pw = "correct-horse-battery"
    users = {n: {**hash_password(pw), "roles": [r]} for n, r in [("vic", "viewer"), ("olga", "operator"), ("ada", "admin")]}
    (tmp_path / "users.json").write_text(json.dumps(users))
    app = create_app(site, auth=Auth(tmp_path / "users.json", tmp_path / "secret.key"))

    def login(u):
        cl = TestClient(app, headers={"X-Argonix-CSRF": "1"})  # cookie-authenticated writes need the CSRF header
        assert cl.post("/api/login", json={"username": u, "password": pw}).status_code == 200
        return cl
    anon, vic, olga, ada = TestClient(app), login("vic"), login("olga"), login("ada")
    for path in ("/api/analytics/zones", "/api/analytics/lines", "/api/analytics/hotspots", "/api/analytics/anomaly",
                 "/api/analytics/objects"):
        assert anon.get(path).status_code == 401
    r = vic.get("/api/analytics/zones")
    assert r.status_code == 404 and r.json()["detail"] == "Not Found"  # disabled: the UI treats it as absent
    assert vic.get("/api/analytics/hotspots").status_code == 200
    assert vic.get("/api/analytics/anomaly").json()["enabled"] is False
    body = {"lines": [{"id": "l1", "points_px": [[0, 0], [100, 0]]}]}
    assert vic.put("/api/cameras/cam1/lines", json=body).status_code == 403
    assert olga.put("/api/cameras/cam1/lines", json=body).status_code == 403
    assert ada.put("/api/cameras/cam1/lines", json=body).status_code == 200
    assert ada.put("/api/cameras/nope/lines", json=body).status_code == 404


def test_objects_enable_extra_detector_classes(tmp_path):
    from argonix.analytics_install import detector_config
    cfg = {"detector": {"conf": 0.3}, "objects": {"enabled": True}}
    det = detector_config(cfg)
    assert det["extra_classes"] == [24, 26, 28] and "extra_classes" not in cfg["detector"]  # config not modified
    assert "extra_classes" not in detector_config({"detector": {}})
    site = make_analytics_site(tmp_path, {"objects": {"enabled": True, "static_s": 2}})
    assert site.objects is not None and site.objects.static_s == 2.0
    site.stop()
