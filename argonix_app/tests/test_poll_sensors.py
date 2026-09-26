"""ESP32-CAM integration (argonix/poll_sensors.py): PIR polling -> sensor events, correlation with video, and the
MJPEG camera path. Ported from binay/webcam-capture."""
import json
import socket
import time
from pathlib import Path

import pytest

from argonix.correlate import Correlator
from argonix.models import Event
from argonix.pipeline import Site
from argonix.poll_sensors import HttpPollSensor, validate_poll_sensor

ROOT = Path(__file__).resolve().parents[1]


class FakeSite:
    def __init__(self, tmp_path):
        self.data = tmp_path

    def worker(self, cam):
        return None


def sensor(tmp_path, readings, **cfg):
    it = iter(readings)

    def fetch(url, timeout):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r
    base = {"id": "esp_pir", "url": "http://10.0.0.50/pir", "camera": "esp32cam", "zone": "entrance",
            "zone_name": "Entrance", "site_zone": "entrance"}
    return HttpPollSensor({**base, **cfg}, FakeSite(tmp_path), fetch=fetch)


def test_rising_edges_become_events(tmp_path):
    s = sensor(tmp_path, [{"motion": False, "count": 3}, {"motion": True, "count": 4}, {"motion": True, "count": 4},
                          {"motion": False, "count": 4}, {"motion": True, "count": 5}])
    got = [s.poll_once(now=100 + i) for i in range(5)]
    assert [e is not None for e in got] == [False, True, False, False, True]
    e = got[1]
    assert e.rule == "sensor_motion" and e.camera_id == "esp32cam" and e.site_zone == "entrance"
    assert "trigger #4" in e.observation and e.severity == 1 and s.status["triggers"] == 2


def test_missed_pulse_is_caught_by_the_counter(tmp_path):
    # motion went on and off between two polls: the state looks unchanged but the device's counter moved
    s = sensor(tmp_path, [{"motion": False, "count": 7}, {"motion": False, "count": 8}])
    assert s.poll_once() is None and s.poll_once() is not None


def test_first_reading_is_only_a_baseline(tmp_path):
    s = sensor(tmp_path, [{"motion": True, "count": 12}])
    assert s.poll_once() is None and s.status["online"]


def test_device_offline_is_reported_not_fatal(tmp_path):
    s = sensor(tmp_path, [{"motion": False, "count": 1}, OSError("timed out"), {"motion": True, "count": 2}])
    s.poll_once()
    assert s.poll_once() is None and not s.status["online"] and "timed out" in s.status["error"]
    assert s.poll_once() is not None and s.status["online"]


def test_poll_sensor_config_validation():
    with pytest.raises(ValueError):
        validate_poll_sensor({"id": "x", "url": "file:///etc/passwd"})
    with pytest.raises(ValueError):
        validate_poll_sensor({"id": "bad id!", "url": "http://10.0.0.5/pir"})
    assert validate_poll_sensor({"id": "ok", "url": "http://10.0.0.5/pir"})["poll_s"] == 0.5


def test_sensor_corroborates_video_on_the_same_site_zone():
    motion = Event("sensor_motion", "esp32cam", "entrance", "Entrance", [], 100, 100, "PIR triggered", {}, 0.9, 1, 1,
                   "entrance")
    alone, _ = Correlator(15).add(motion)
    assert alone.priority == "P4"
    c = Correlator(15)
    video = Event("intrusion", "arducam", "door", "Door", [3], 101, 101, "person in door", {}, 0.8, 3, 2, "entrance")
    a1, _ = c.add(video)
    before = a1.score
    a2, new = c.add(Event(**{**motion.__dict__, "t_start": 102, "t_end": 102}))
    assert not new and a2.id == a1.id and a2.score > before
    assert set(a2.cameras) == {"arducam", "esp32cam"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_simulated_esp32cam_end_to_end(tmp_path):
    """The real pipeline reads the node's MJPEG stream and polls its PIR (tools/sim_esp32cam.py)."""
    from tools.sim_esp32cam import serve
    port = free_port()
    srv, state = serve(port, ROOT / "data" / "synth" / "cam1.mp4", fps=10)
    try:
        cfg = {"site": "esp32 test", "analytics_fps": 5, "record": False, "correlation_window_s": 15,
               "cameras": [{"id": "esp32cam", "name": "ESP32-CAM", "source": f"http://127.0.0.1:{port}/stream",
                            "zones": [{"id": "entrance", "name": "Entrance", "site_zone": "entrance",
                                       "polygon_px": [[0, 0], [640, 0], [640, 480], [0, 480]],
                                       "rules": {"loitering": {"min_seconds": 60}}}]}],
               "poll_sensors": [{"id": "esp_pir", "url": f"http://127.0.0.1:{port}/pir", "camera": "esp32cam",
                            "zone": "entrance", "zone_name": "Entrance", "site_zone": "entrance", "poll_s": 0.2}]}
        p = tmp_path / "configs" / "esp.json"
        p.parent.mkdir()
        p.write_text(json.dumps(cfg), encoding="utf-8")

        class NoPeople:
            info = {"detector": "fake", "tracker": "fake"}

            def track(self, frame):
                return []
        site = Site(p, data_dir=tmp_path / "run", realtime=True, backend_factory=lambda c: NoPeople())
        site.start()
        try:
            w, deadline = site.workers[0], time.time() + 30
            while time.time() < deadline and not (w.stats["frames"] >= 5 and site.poll_sensors[0].status["online"]):
                time.sleep(0.2)
            assert w.stats["frames"] >= 5 and w.source_size == (640, 480), w.stats
            state.set_motion(True)
            deadline = time.time() + 10
            alerts = []
            while time.time() < deadline and not alerts:
                alerts = [a for a in site.store.list_alerts() if "sensor_motion" in {e["rule"] for e in a["events"]}]
                time.sleep(0.2)
            assert alerts, site.poll_sensors[0].status
            ev = alerts[0]["events"][0]
            assert ev["evidence"]["snapshot"] and (site.data / ev["evidence"]["snapshot"]).is_file()
        finally:
            site.stop()
    finally:
        srv.shutdown()
