"""Synchronised incident replay: media selection per camera (recording segments, evidence clips, gaps, related
cameras, markers) and the /api/incidents/{id}/replay endpoint. No video decoding."""
import time

from fastapi.testclient import TestClient

from argonix.models import Event
from argonix.replay import build_plan, gaps, overlapping, window
from argonix.rules import SEVERITY
from argonix.server import create_app
from tests.test_server import make_site


def test_overlapping_and_gaps():
    segs = [{"start": 900, "end": 960}, {"start": 1020, "end": 1080}, {"start": 960, "end": 1020},
            {"start": 1100, "end": 1160}, {"start": 1035, "end": 1040}]
    got = overlapping(segs, 990, 1035)
    assert [(s["start"], s["end"]) for s in got] == [(960, 1020), (1020, 1080)]  # touching 1035 start excluded
    assert gaps([(960, 1020), (1020, 1080)], 990, 1035) == []
    assert gaps([(1000, 1010), (1005, 1012), (1020, 1030)], 990, 1035) == [
        {"start": 990, "end": 1000}, {"start": 1012, "end": 1020}, {"start": 1030, "end": 1035}]
    assert gaps([], 990, 1035) == [{"start": 990, "end": 1035}]
    assert gaps([(990.2, 1035)], 990, 1035) == []  # sub-second jitter is not a gap


def fake_incident():
    return {"id": "I1000-1", "opened_at": 1000.0, "last_event_at": 1025.0, "cameras": ["cam1", "cam2"],
            "alert_ids": ["A1", "A2"],
            "events": [{"alert_id": "A1", "rule": "intrusion", "camera_id": "cam1", "zone_name": "Door",
                        "t_start": 1000.0, "t_end": 1005.0},
                       {"alert_id": "A2", "rule": "loitering", "camera_id": "cam2", "zone_name": "Hall",
                        "t_start": 1020.0, "t_end": 1025.0, "learned_model": False}],
            "predictions": {"cam2:7": {"predictions": [{"camera": "cam3", "prob": 0.7, "eta_s": 12},
                                                       {"camera": "cam1", "prob": 0.2},
                                                       {"leaves_coverage": True, "prob": 0.1}]}}}


def test_build_plan_selects_media_per_camera():
    recs = {"cam1": [{"path": "recordings/cam1/a.mp4", "start": 900.0, "end": 960.0},
                     {"path": "recordings/cam1/b.mp4", "start": 960.0, "end": 1020.0},
                     {"path": "recordings/cam1/c.mp4", "start": 1020.0, "end": 1080.0},
                     {"path": "recordings/cam1/gone.mp4", "start": 1080.0, "end": 1140.0}],
            "cam2": [{"path": "recordings/cam2/a.mp4", "start": 1000.0, "end": 1060.0}]}
    alerts = [{"id": "A2", "events": [{"camera_id": "cam2", "rule": "loitering",
                                       "evidence": {"clip": "evidence/cam2_l.mp4", "frame_time": 1025.0}}]},
              {"id": "A1", "events": [{"camera_id": "cam1", "rule": "intrusion",
                                       "evidence": {"clip": "evidence/missing.mp4", "frame_time": 1003.0}}]}]
    files = {r["path"] for rs in recs.values() for r in rs if "gone" not in r["path"]} | {"evidence/cam2_l.mp4"}
    plan = build_plan(fake_incident(), lambda c, a, b: recs.get(c, []), alerts, files.__contains__,
                      {"cam1": "Server wing", "cam2": "Lobby", "cam3": "Yard", "cam4": "Gate"},
                      pre_s=10, post_s=50, extra=["cam4", "nope"], known={"cam1", "cam2", "cam3", "cam4"},
                      open_segments={"cam4": {"start": 1060.0, "end": 1100.0}})
    assert (plan["start"], plan["end"]) == (990.0, 1075.0) and not plan["clamped"]
    by = {c["camera"]: c for c in plan["cameras"]}
    assert [c["camera"] for c in plan["cameras"]] == ["cam1", "cam2", "cam3", "cam4"]   # unknown extra dropped
    assert [c["role"] for c in plan["cameras"]] == ["primary", "primary", "predicted_next", "requested"]
    c1 = by["cam1"]
    assert [s["path"] for s in c1["segments"]] == ["recordings/cam1/b.mp4", "recordings/cam1/c.mp4"]
    assert c1["segments"][0]["start"] == 960.0 and c1["gaps"] == [] and c1["name"] == "Server wing"
    assert c1["clips"] == [] and c1["missing_files"] == 1                    # the evidence clip is not on disk
    c2 = by["cam2"]
    assert c2["gaps"] == [{"start": 990.0, "end": 1000.0}, {"start": 1060.0, "end": 1075.0}]
    assert c2["clips"][0]["start"] == 1021.0 and c2["clips"][0]["approx"] and c2["clips"][0]["alert_id"] == "A2"
    c3 = by["cam3"]
    assert c3["prediction"]["prob"] == 0.7 and c3["prediction"]["estimate"] and c3["segments"] == []
    assert c3["gaps"] == [{"start": 990.0, "end": 1075.0}]
    assert by["cam4"]["in_progress"] == {"kind": "in_progress", "start": 1060.0, "end": 1075.0}
    assert [m["alert_id"] for m in plan["markers"]] == ["A1", "A2"] and plan["markers"][0]["t"] == 1000.0


def test_window_is_clamped():
    inc = {**fake_incident(), "last_event_at": 1000.0 + 5 * 3600}
    start, end, clamped = window(inc, 10, 10)
    assert clamped and end - start == 2 * 3600


def test_replay_endpoint(tmp_path):
    site = make_site(tmp_path)
    now = time.time()
    data = site.data
    (data / "evidence").mkdir(parents=True, exist_ok=True)
    (data / "evidence" / "cam1_intrusion.mp4").write_bytes(b"\0" * 64)
    for i, (a, b) in enumerate([(now - 200, now - 140), (now - 140, now - 80), (now - 80, now - 20)]):
        rel = f"recordings/cam1/seg{i}.mp4"
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(b"\0" * 64)
        site.store.add_recording("cam1", rel, a, b)
    e = Event("intrusion", "cam1", "server_door", "Server room door", [1], now - 100, now - 95, "person track #1",
              {"rule": "intrusion"}, 0.9, SEVERITY["intrusion"], 3, "server_door", None,
              {"snapshot": None, "clip": "evidence/cam1_intrusion.mp4", "frame_time": now - 95})
    site.on_event(e)
    iid = site.store.list_alerts()[0]["incident_id"]
    c = TestClient(create_app(site, open_demo=True))
    plan = c.get(f"/api/incidents/{iid}/replay?pre_s=10&post_s=10").json()
    assert plan["incident_id"] == iid and plan["end"] - plan["start"] >= 20
    cam = plan["cameras"][0]
    assert cam["camera"] == "cam1" and cam["role"] == "primary"
    assert [s["path"] for s in cam["segments"]] == ["recordings/cam1/seg1.mp4"]
    assert cam["clips"][0]["path"] == "evidence/cam1_intrusion.mp4" and cam["gaps"] == []
    assert plan["markers"][0]["rule"] == "intrusion"
    assert c.get(f"/api/incidents/{iid}/replay?pre_s=-1").status_code == 400
    assert c.get("/api/incidents/I0-404/replay").status_code == 404
