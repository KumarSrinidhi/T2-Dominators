"""Rule schedules (after-hours arming), counting-line `line_cross` events and the vehicle rules. CPU only, synthetic
tracks fed straight into the rule engine / flow analyzer, no model weights."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argonix.flow import FlowAnalyzer, parse_line_events
from argonix.models import TrackObs
from argonix.rules import LABEL, SEVERITY, RuleEngine, Zone, vehicle_classes
from argonix.schedule import Schedule, Window

ROOT = Path(__file__).resolve().parents[1]
SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]
AFTER_HOURS = {"tz": "UTC", "windows": ["mon-fri 19:00-07:00", "sat,sun 00:00-24:00"], "label": "after-hours"}
FRAME = np.full((1200, 2000, 3), 90, np.uint8)


def ts(day: int, hh: int, mm: int = 0) -> float:
    """Epoch seconds of weekday `day` (0 = Monday) hh:mm UTC in the week of Mon 2026-09-21."""
    d = datetime(2026, 9, 21 + day, hh, mm, tzinfo=timezone.utc)
    assert d.weekday() == day
    return d.timestamp()


def pobs(tid, t, x, y, conf=0.9, cls="person"):
    return TrackObs("cam1", tid, t, (0, 0, 1, 1), cls, conf, (x, y))


def vobs(tid, t, x, y, cls="car", conf=0.8, box=None):
    return TrackObs("cam1", tid, t, box or (0, 0, 1, 1), cls, conf, (x, y))


def feed(engine, t0, seconds, fn, fps=5.0):
    out = []
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        out += engine.update(t, fn(t, i / fps))
    return out


# ---------------------------------------------------------------------------------------------- schedule parsing
def test_schedule_windows_and_overnight_wrap():
    s = Schedule.parse(AFTER_HOURS)
    assert s.active(ts(0, 19, 0)) and s.active(ts(0, 23, 59)) and s.active(ts(1, 6, 59))  # Mon 19:00 -> Tue 07:00
    assert not s.active(ts(1, 7, 0)) and not s.active(ts(2, 12, 0))                    # end exclusive, daytime off
    assert not s.active(ts(0, 3, 0))  # Mon 03:00 would belong to a SUNDAY evening window (not listed on this rule)
    assert s.active(ts(5, 6, 59)) and s.active(ts(5, 12, 0)) and s.active(ts(6, 23, 59))  # Fri night + weekend
    assert s.active_window(ts(5, 3, 0)).text == "mon-fri 19:00-07:00"  # Sat 03:00: Friday's overnight window first
    assert s.active_window(ts(5, 12, 0)).text == "sat,sun 00:00-24:00"
    d = s.describe(ts(2, 22, 30), "zone")
    assert d["text"] == "after-hours window mon-fri 19:00-07:00" and d["local_time"] == "Wed 22:30"
    assert s.describe(ts(2, 12, 0)) is None

    w = Window.parse("fri-mon 22:00-02:00")  # day ranges wrap around the week too
    assert w.days == {4, 5, 6, 0} and w.overnight
    assert w.contains(1, 60) and not w.contains(2, 60)  # Tue 01:00 = Monday night's tail; Wed 01:00 is not
    assert Window.parse("22:00-06:00").days == frozenset(range(7))  # no day part = daily
    assert Window.parse("Weekdays 08:00-18:00").days == {0, 1, 2, 3, 4}
    assert Schedule.parse(None) is None and Schedule.parse({}) is None
    assert Schedule.parse("sat 10:00-11:00").windows[0].start == 600  # bare string shorthand, server local time


def test_schedule_time_zones_and_errors():
    ist = Schedule.parse({"tz": "Asia/Kolkata", "windows": ["daily 19:00-07:00"]})  # zoneinfo or no-DST fallback
    assert ist.active(ts(0, 14, 0)) and not ist.active(ts(0, 12, 0))  # 14:00 UTC = 19:30 IST; 12:00 UTC = 17:30 IST
    off = Schedule.parse({"tz": "+05:30", "windows": ["daily 19:00-07:00"]})
    assert off.active(ts(0, 14, 0)) and not off.active(ts(0, 12, 0))
    bad = [({"windows": ["mon-fri 19:00-07:00"], "tz": "Mars/Olympus"}, "unknown time zone"),
           ({"windows": ["funday 10:00-11:00"]}, "unknown day"),
           ({"windows": ["mon 25:00-26:00"]}, "invalid time"),
           ({"windows": ["mon 24:00-06:00"]}, "invalid time"),
           ({"windows": ["mon 10:00-10:00"]}, "same start and end"),
           ({"windows": ["every evening"]}, "cannot parse"),
           ({"windows": []}, "non-empty"),
           ({"windows": ["mon 10:00-11:00"], "when": "x"}, "unknown key"),
           (42, "must be an object")]
    for d, msg in bad:
        with pytest.raises(ValueError, match=msg):
            Schedule.parse(d, "zone 'dock' schedule")
    with pytest.raises(ValueError, match="zone 'dock' schedule"):
        Zone.from_dict({"id": "dock", "polygon": SQ, "rules": {}, "schedule": {"windows": ["x"]}})
    with pytest.raises(ValueError, match="rule 'intrusion' schedule"):
        Zone.from_dict({"id": "dock", "polygon": SQ, "rules": {"intrusion": {"schedule": {"windows": ["x"]}}}})


# ---------------------------------------------------------------------------------------------- rule gating
def test_schedule_gates_zone_rules_and_states_window():
    z = Zone.from_dict({"id": "z", "name": "Dock", "polygon": SQ, "schedule": AFTER_HOURS,
                        "rules": {"intrusion": {"min_seconds": 1.0}, "crowding": {"max_count": 1, "min_seconds": 1.0}}})
    assert "schedule" not in z.rules["intrusion"]

    def inside(t, rel):
        return [pobs(1, t, 5, 5), pobs(2, t, 6, 6)]

    day = feed(RuleEngine("cam1", [z], calib_error_m=0.3), ts(2, 12, 0), 5.0, inside)  # Wednesday noon: not armed
    assert day == []
    night = feed(RuleEngine("cam1", [z], calib_error_m=0.3), ts(2, 22, 0), 5.0, inside)
    assert sorted(e.rule for e in night) == ["crowding", "intrusion", "intrusion"]
    for e in night:
        assert e.derivation["schedule"]["window"] == "mon-fri 19:00-07:00"
        assert "after-hours window mon-fri 19:00-07:00" in e.observation

    # a person already inside when the window opens: the rule arms at 19:00 and needs min_seconds from then
    eng = RuleEngine("cam1", [z], calib_error_m=0.3)
    ev = feed(eng, ts(2, 18, 59, ) + 55, 10.0, lambda t, r: [pobs(1, t, 5, 5)])
    fired = [e for e in ev if e.rule == "intrusion"]
    assert len(fired) == 1 and fired[0].t_start >= ts(2, 19, 0)

    # the window closing ends an open episode (grace period does not carry it over)
    eng = RuleEngine("cam1", [Zone.from_dict({"id": "z", "polygon": SQ, "schedule": AFTER_HOURS,
                                              "rules": {"loitering": {"min_seconds": 20}}})], calib_error_m=0.3)
    assert feed(eng, ts(1, 6, 59, ) + 45, 30.0, lambda t, r: [pobs(1, t, 5, 5)]) == []


def test_rule_schedule_replaces_zone_schedule():
    z = Zone.from_dict({"id": "z", "name": "Yard", "polygon": SQ, "schedule": AFTER_HOURS,
                        "rules": {"intrusion": {"min_seconds": 1.0},
                                  "loitering": {"min_seconds": 2.0,
                                                "schedule": {"tz": "UTC", "windows": ["daily 09:00-17:00"],
                                                             "label": "opening-hours"}}}})
    ev = feed(RuleEngine("cam1", [z], calib_error_m=0.3), ts(2, 12, 0), 4.0, lambda t, r: [pobs(1, t, 5, 5)])
    assert [e.rule for e in ev] == ["loitering"]  # intrusion follows the zone (after hours): off at noon
    assert ev[0].derivation["schedule"]["source"] == "rule"
    assert "opening-hours window daily 09:00-17:00" in ev[0].observation
    assert Zone.from_dict({"id": "a", "polygon": SQ, "rules": {"intrusion": {}}}).schedule_info("intrusion", 0) is None


def test_schedule_gates_flow_rules():
    zone = {"id": "q", "name": "Queue", "polygon_px": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]], "type": "queue",
            "schedule": {"tz": "UTC", "windows": ["mon-fri 09:00-17:00"], "label": "service-hours"},
            "rules": {"queue_long": {"max_length": 1, "min_seconds": 2.0}}}
    zobj = Zone.from_dict({**zone, "polygon": zone["polygon_px"]})
    w = SimpleNamespace(id="cam1", H=None, calib_error=None, zones=[], px_zones=[zobj], scale=1.0,
                        cfg={"zones": [zone], "lines": []})

    def people(t):
        return [(i, (100 * i, 300, 100 * i + 40, 400), "person", 0.9) for i in range(1, 4)]

    def run(t0):
        an, ev = FlowAnalyzer(), []
        for i in range(25):
            t = t0 + i / 5
            ev += an.on_frame(w, t, FRAME, people(t), [])
        return an, ev

    an, ev = run(ts(5, 12, 0))  # Saturday: outside the window
    assert ev == []
    an, ev = run(ts(2, 12, 0))
    assert [e.rule for e in ev] == ["queue_long"] and ev[0].derivation["schedule"]["label"] == "service-hours"
    zs = an.summary_zones()["cameras"][0]["zones"][0]["schedule"]
    assert zs["active"] is True and zs["window"] == "mon-fri 09:00-17:00"


# ---------------------------------------------------------------------------------------------- line_cross
def line_worker(events, calibrated=False):
    ln = {"id": "door1", "name": "Door 1", "points_px": [[0, 500], [1000, 500]], "in_label": "entering",
          "events": events}
    return SimpleNamespace(id="cam1", H=None, calib_error=None, zones=[], px_zones=[], scale=1.0,
                           cfg={"zones": [], "lines": [ln]})


def walker(tid, x, y0, vy):
    """A person det whose foot point moves along y (image px) at vy px/s."""
    def at(t):
        y = y0 + vy * t
        return (tid, (x - 20, y - 150, x + 20, y), "person", 0.85)
    return at


def run_lines(w, t0, seconds, walkers, fps=5.0):
    an, ev = FlowAnalyzer(), []
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        ev += an.on_frame(w, t, FRAME, [f(i / fps) for f in walkers], [])
    return an, ev


def test_line_cross_direction_and_gap():
    down = [walker(1, 100, 300, 80), walker(2, 300, 330, 80), walker(3, 500, 380, 80)]  # "in" at ~2.5, 2.1, 1.5 s
    up = [walker(4, 700, 700, -80)]                                                      # "out" at ~2.5 s
    an, ev = run_lines(line_worker({"direction": "in", "min_gap_s": 0}), ts(2, 12, 0), 5.0, down + up)
    assert [e.rule for e in ev] == ["line_cross"] * 3
    e = ev[0]
    assert e.zone_id == "line:door1" and e.zone_name == "Door 1" and e.severity == SEVERITY["line_cross"]
    assert "crossed line 'Door 1' inbound ('entering')" in e.observation and e.derivation["direction"] == "in"
    assert an.summary_lines()["cameras"][0]["lines"][0]["counts"] == {"in": 3, "out": 1}  # counting unchanged
    assert LABEL["line_cross"] == "Counting-line crossing"

    # min_gap_s: the 2nd and 3rd crossings fall inside the gap: tallied and reported with the next event
    an, ev = run_lines(line_worker({"direction": "in", "min_gap_s": 5}), ts(2, 12, 0), 5.0, down)
    assert len(ev) == 1 and an.summary_lines()["cameras"][0]["lines"][0]["events"]["events_raised"] == 1
    late = [walker(1, 100, 300, 80), walker(2, 300, 330, 80), walker(5, 800, 0, 80)]  # 3rd crosses at ~6.25 s
    an, ev = run_lines(line_worker({"direction": "in", "min_gap_s": 3}), ts(2, 12, 0), 8.0, late)
    assert len(ev) == 2 and ev[1].derivation["crossings_suppressed_by_gap"] == 1
    assert "1 more matching crossing(s)" in ev[1].observation

    an, ev = run_lines(line_worker({"direction": "out"}), ts(2, 12, 0), 5.0, down + up)
    assert [e.derivation["direction"] for e in ev] == ["out"] and "outbound" in ev[0].observation
    an, ev = run_lines(line_worker({"direction": "both", "min_gap_s": 0}), ts(2, 12, 0), 5.0, down + up)
    assert sorted(e.derivation["direction"] for e in ev) == ["in", "in", "in", "out"]
    an, ev = run_lines(line_worker(None), ts(2, 12, 0), 5.0, down + up)  # no events option: counting only
    assert ev == [] and an.summary_lines()["cameras"][0]["lines"][0]["events"] is None


def test_line_cross_respects_schedule_and_validation():
    after = {"direction": "in", "min_gap_s": 0, "schedule": AFTER_HOURS, "criticality": 3}
    down = [walker(1, 100, 300, 80)]
    assert run_lines(line_worker(after), ts(2, 12, 0), 5.0, down)[1] == []  # daytime door entry: counted only
    an, ev = run_lines(line_worker(after), ts(2, 23, 0), 5.0, down)
    assert len(ev) == 1 and ev[0].criticality == 3
    assert "after-hours window mon-fri 19:00-07:00" in ev[0].observation
    assert ev[0].derivation["schedule"]["source"] == "line"
    assert an.summary_lines()["cameras"][0]["lines"][0]["events"]["schedule"]["active"] is True

    assert parse_line_events(None, "l") is None and parse_line_events(True, "l")["direction"] == "both"
    for ev, msg in [({"direction": "sideways"}, "direction must be one of"), ({"min_gap_s": -1}, "min_gap_s"),
                    ({"criticality": 7}, "criticality"), ({"colour": "red"}, "unknown key"),
                    ({"schedule": {"windows": ["nope"]}}, "line 'l' events schedule"), ("yes", "must be an object")]:
        with pytest.raises(ValueError, match=msg):
            parse_line_events(ev, "l")


# ---------------------------------------------------------------------------------------------- vehicles
def vzone(**rules):
    return Zone.from_dict({"id": "plaza", "name": "Plaza", "polygon": SQ, "criticality": 2, "rules": rules})


def test_vehicle_in_zone_and_people_unaffected():
    z = vzone(vehicle_in_zone={"min_seconds": 2.0}, crowding={"max_count": 1, "min_seconds": 1.0},
              intrusion={"min_seconds": 1.0})
    eng = RuleEngine("cam1", [z], calib_error_m=0.3, vehicle_classes=("car", "motorcycle", "bus", "truck"))
    ev = feed(eng, ts(2, 12, 0), 4.0, lambda t, r: [vobs(1_000_001, t, 5, 5), vobs(1_000_002, t, 6, 6, "truck"),
                                                    vobs(1_000_003, t, 20, 5), pobs(7, t, 30, 30)])
    rules = sorted(e.rule for e in ev)
    assert rules == ["vehicle_in_zone", "vehicle_in_zone"]  # no crowding/intrusion from vehicles; outside car ignored
    car = next(e for e in ev if e.track_ids == [1_000_001])
    assert "car track #1000001" in car.observation and "pedestrian-only 'Plaza'" in car.observation
    assert car.severity == SEVERITY["vehicle_in_zone"] and car.derivation["object_class"] == "car"
    # without vehicle classes (vehicles disabled) the rule never sees them
    eng = RuleEngine("cam1", [z], calib_error_m=0.3)
    assert feed(eng, ts(2, 12, 0), 4.0, lambda t, r: [vobs(1_000_001, t, 5, 5)]) == []


def test_vehicle_stopped_metres_and_pixels():
    z = vzone(vehicle_stopped={"min_seconds": 10.0, "window_s": 3.0, "still_m": 1.0})

    def car(x_of_t):
        return lambda t, r: [vobs(1_000_004, t, x_of_t(r), 5)]

    eng = RuleEngine("cam1", [z], calib_error_m=0.3, vehicle_classes=("car",))
    ev = feed(eng, ts(2, 12, 0), 30.0, car(lambda r: min(1 + 2.0 * r, 5) + 0.05 * (r % 1)))  # drives in, parks at x=5
    assert [e.rule for e in ev] == ["vehicle_stopped"]
    e = ev[0]
    assert e.t_start >= ts(2, 12, 0) + 2 + 3 * 0.8  # stillness is only tested once the window is covered
    assert "stationary in no-stopping 'Plaza'" in e.observation and e.derivation["displacement_m"] <= 1.0
    # a slow-rolling vehicle (0.5 m/s = 1.5 m per 3 s window) is not "stopped"
    eng = RuleEngine("cam1", [z], calib_error_m=0.3, vehicle_classes=("car",))
    assert feed(eng, ts(2, 12, 0), 18.0, car(lambda r: 0.5 + 0.5 * r)) == []
    # parked OUTSIDE the zone: nothing
    eng = RuleEngine("cam1", [z], calib_error_m=0.3, vehicle_classes=("car",))
    assert feed(eng, ts(2, 12, 0), 20.0, car(lambda r: 15.0)) == []

    # uncalibrated camera: stillness relative to the box size (box centre in analytics px)
    zpx = Zone.from_dict({"id": "bay", "name": "Bay", "polygon": [(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
                          "rules": {"vehicle_stopped": {"min_seconds": 5.0, "window_s": 2.0, "still_box_frac": 0.15}}})

    def box_car(dx_per_s):
        def fn(t, r):
            x = 300 + dx_per_s * r
            return [vobs(1_000_009, t, x + 100, 600, box=(x, 500, x + 200, 600))]  # 200 px wide box
        return fn
    eng = RuleEngine("cam1", [zpx], calib_error_m=8.0, units="px", vehicle_classes=("car",))
    ev = feed(eng, ts(2, 12, 0), 10.0, box_car(5))  # 10 px per 2 s window = 5% of the box: stationary
    assert [e.rule for e in ev] == ["vehicle_stopped"] and "displacement_box_frac" in ev[0].derivation
    assert ev[0].derivation["space"] == "image_px" and ev[0].ground is None
    eng = RuleEngine("cam1", [zpx], calib_error_m=8.0, units="px", vehicle_classes=("car",))
    assert feed(eng, ts(2, 12, 0), 10.0, box_car(40)) == []  # 80 px per window = 40% of the box: moving


def test_vehicle_config_detector_classes_and_drawing():
    from argonix.analytics_install import detector_config
    from argonix.media import VEHICLE_COL, draw
    assert vehicle_classes({}) == {} and vehicle_classes({"vehicles": {"enabled": False}}) == {}
    assert vehicle_classes({"vehicles": {"enabled": True}}) == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert vehicle_classes({"vehicles": {"enabled": True, "classes": {"80": "forklift"}}}) == {80: "forklift"}
    with pytest.raises(ValueError, match="not a COCO vehicle class"):
        vehicle_classes({"vehicles": {"enabled": True, "classes": [0]}})
    det = detector_config({"detector": {"conf": 0.3}, "vehicles": {"enabled": True, "detector_conf": 0.4},
                           "objects": {"enabled": True, "detector_conf": 0.25}})
    assert det["extra_classes"] == [24, 26, 28, 2, 3, 5, 7] and det["extra_conf"] == 0.25
    assert "extra_classes" not in detector_config({"detector": {}, "vehicles": {"enabled": False}})

    img = np.zeros((200, 300, 3), np.uint8)
    out = draw(img, [(1, (10, 10, 60, 90), "person", 0.9), (1_000_001, (100, 20, 250, 120), "car", 0.8)], [], set(),
               vehicle_labels=("car",))
    assert tuple(out[50, 10]) == (80, 220, 80) and tuple(out[70, 100]) == VEHICLE_COL


def test_flow_ignores_vehicles():
    w = line_worker({"direction": "both", "min_gap_s": 0})
    an, ev = run_lines(w, ts(2, 12, 0), 5.0, [lambda t: (1_000_001, (80, 150 + 80 * t, 220, 300 + 80 * t), "car", 0.9)])
    assert ev == [] and an.summary_lines()["cameras"][0]["lines"][0]["counts"] == {"in": 0, "out": 0}


# ---------------------------------------------------------------------------------------------- site + API
class StillBackend:
    info = {"detector": "fake", "tracker": "fake"}

    def track(self, frame):
        return [(1, (300, 200, 340, 300), "person", 0.9), (1_000_001, (500, 300, 700, 420), "car", 0.8)]


def make_site(tmp_path, extra=None, cam_patch=None):
    from argonix.pipeline import Site
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cam = cfg["cameras"][0]
    cam["source"] = str(ROOT / cam["source"])
    cam.update(cam_patch or {})
    cfg["cameras"] = [cam]
    cfg["record"] = False
    cfg.update(extra or {})
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: StillBackend())


def test_site_config_validation_and_api(tmp_path):
    with pytest.raises(ValueError, match="zone 'bad' schedule"):
        make_site(tmp_path, cam_patch={"zones": [{"id": "bad", "polygon_px": [[0, 0], [9, 0], [9, 9]],
                                                  "schedule": {"windows": ["mon 7pm"]}}]})
    with pytest.raises(ValueError, match="line 'door' events"):
        make_site(tmp_path, cam_patch={"lines": [{"id": "door", "points_px": [[0, 0], [9, 0]],
                                                  "events": {"direction": "up"}}]})

    site = make_site(tmp_path, {"vehicles": {"enabled": True}})
    assert site.workers[0].engine.vehicle_classes == ("car", "motorcycle", "bus", "truck")
    from argonix.server import create_app
    c = TestClient(create_app(site, open_demo=True))
    cam = c.get("/api/cameras").json()[0]
    assert cam["vehicles_enabled"] is True
    zones = cam["zones"]
    good = [{**zones[0], "schedule": AFTER_HOURS,
             "rules": {**zones[0]["rules"], "vehicle_stopped": {"min_seconds": 30, "schedule": "daily 00:00-06:00"}}}]
    r = c.put("/api/cameras/cam1/zones", json={"zones": good})
    assert r.status_code == 200, r.text
    saved = json.loads(site.config_path.read_text())["cameras"][0]["zones"][0]
    assert saved["schedule"] == AFTER_HOURS and saved["rules"]["vehicle_stopped"]["schedule"] == "daily 00:00-06:00"
    z = site.workers[0].zones[0]
    assert z.schedule is not None and "vehicle_stopped" in z.rule_schedules
    r = c.put("/api/cameras/cam1/zones", json={"zones": [{**zones[0], "schedule": {"windows": ["mon 19:00-19:00"]}}]})
    assert r.status_code == 400 and "same start and end" in r.json()["detail"]

    lines = [{"id": "door1", "name": "Door 1", "points_px": [[0, 200], [900, 200]],
              "events": {"direction": "in", "min_gap_s": 5, "schedule": AFTER_HOURS}}]
    r = c.put("/api/cameras/cam1/lines", json={"lines": lines})
    assert r.status_code == 200, r.text
    assert r.json()["lines"][0]["events"] == {"direction": "in", "min_gap_s": 5.0, "criticality": 1,
                                              "schedule": AFTER_HOURS}
    bad = [{**lines[0], "events": {"direction": "in", "min_gap_s": "soon"}}]
    assert c.put("/api/cameras/cam1/lines", json={"lines": bad}).status_code == 400

    chk = c.post("/api/schedule/check", json={"schedule": AFTER_HOURS}).json()
    assert chk["ok"] and isinstance(chk["active"], bool) and chk["windows"] == AFTER_HOURS["windows"]
    chk = c.post("/api/schedule/check", json={"schedule": {"windows": ["noon"]}}).json()
    assert chk["ok"] is False and "cannot parse" in chk["error"]
    assert c.post("/api/schedule/check", json={}).json()["active"] is True

    # without `vehicles` enabled the API accepts vehicle rules but warns that they see nothing
    site2 = make_site(tmp_path / "b")
    c2 = TestClient(create_app(site2, open_demo=True))
    z0 = c2.get("/api/cameras").json()[0]["zones"][0]
    r = c2.put("/api/cameras/cam1/zones", json={"zones": [{**z0, "rules": {"vehicle_in_zone": {}}}]})
    assert r.status_code == 200 and any("needs vehicle detection" in w for w in r.json()["warnings"])
    site.stop(); site2.stop()
