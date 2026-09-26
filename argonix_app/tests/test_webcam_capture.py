"""Webcam capture settings (resolution / fps / MJPEG) are requested before the first frame."""
import cv2

from argonix import cameras


class FakeCap:
    def __init__(self):
        self.calls = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True


def test_capture_settings_are_applied_in_order():
    cap = FakeCap()
    cameras.apply_capture_settings(cap, {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"})
    assert cap.calls == [(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                         (cv2.CAP_PROP_FRAME_WIDTH, 1280.0), (cv2.CAP_PROP_FRAME_HEIGHT, 720.0),
                         (cv2.CAP_PROP_FPS, 30.0)]


def test_no_capture_settings_leaves_camera_untouched():
    cap = FakeCap()
    cameras.apply_capture_settings(cap, None)
    cameras.apply_capture_settings(cap, {})
    assert cap.calls == []


def test_open_capture_applies_settings_to_webcam_index(monkeypatch):
    made = []

    def fake_vc(*args):
        c = FakeCap()
        made.append((args, c))
        return c
    monkeypatch.setattr(cameras.cv2, "VideoCapture", fake_vc)
    cameras.open_capture("2", {"width": 1920, "height": 1080})
    (args, cap), = made
    assert args == (2,) and (cv2.CAP_PROP_FRAME_WIDTH, 1920.0) in cap.calls


def test_camera_api_validates_and_stores_capture(tmp_path):
    from fastapi.testclient import TestClient
    from argonix.server import create_app
    from tests.test_server import make_site
    site = make_site(tmp_path)
    c = TestClient(create_app(site, open_demo=True), headers={"X-Argonix-CSRF": "1"})
    cap = {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}
    r = c.post("/api/cameras", json={"id": "usb", "source": "2", "capture": cap, "enabled": False})
    assert r.status_code == 200, r.text
    assert site.camera_config("usb")["capture"] == cap
    assert c.post("/api/cameras", json={"id": "usb2", "source": "3", "capture": {"width": 99999}}).status_code == 400
    assert c.post("/api/cameras", json={"id": "usb3", "source": "3", "capture": {"fourcc": "M J"}}).status_code == 400
    assert c.put("/api/cameras/usb", json={"capture": {"fps": 15}}).status_code == 200
    assert site.camera_config("usb")["capture"]["fps"] == 15
    assert c.put("/api/cameras/usb", json={"capture": None}).status_code == 200
    assert "capture" not in site.camera_config("usb")
