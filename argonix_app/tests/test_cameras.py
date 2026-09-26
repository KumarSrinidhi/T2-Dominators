"""Connecting existing CCTV: redaction, templates, probing, ONVIF parsing, live grabber, camera CRUD API,
uncalibrated cameras with pixel zones, config persistence and permissions. No real cameras or network needed."""
import base64
import json
import socket
import threading
import time
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from argonix import cameras
from argonix import pipeline as pipeline_mod
from argonix.pipeline import LatestFrameGrabber, Site
from argonix.server import create_app
from tests.test_auth import client_with_users, login
from tests.test_server import FakeBackend, make_site

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "synth" / "cam1.mp4"
SECRET = "S3cr@t/pw"  # test fixture only
RTSP = f"rtsp://admin:{SECRET}@192.0.2.10:554/Streaming/Channels/102"  # TEST-NET-1, never routed


@pytest.fixture(scope="module", autouse=True)
def synth_clips():
    if not CLIP.exists():
        from tools.make_synthetic import main
        main()


# ---- redaction / templates / source validation --------------------------------------------------------------
def test_redact_hides_passwords():
    assert cameras.redact(RTSP) == "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/102"
    assert cameras.redact("rtsp://u:p@h/cam/realmonitor?channel=1&subtype=1") == \
        "rtsp://u:***@h/cam/realmonitor?channel=1&subtype=1"
    assert cameras.redact("rtsp://192.0.2.1/live") == "rtsp://192.0.2.1/live"
    assert cameras.redact("http://h/video.cgi?user=admin&password=hunter2&x=1") == \
        "http://h/video.cgi?user=admin&password=***&x=1"
    assert cameras.redact("0") == "0" and cameras.redact(str(CLIP)) == str(CLIP)
    msg = cameras.redact(f"failed to open {RTSP} (timeout)")
    assert SECRET not in msg and "admin:***@192.0.2.10" in msg
    # a client echoing the redacted form back means "unchanged"; any other masked value is refused
    assert cameras.unredact(cameras.redact(RTSP), RTSP) == RTSP
    with pytest.raises(ValueError):
        cameras.unredact("rtsp://admin:***@other/x", RTSP)


def test_templates_and_build_url():
    t = cameras.templates()
    assert t["verified"] is False and "not guarantees" in t["disclaimer"]
    ids = {x["id"] for x in t["templates"]}
    assert {"hikvision", "dahua", "axis", "reolink", "uniview", "tapo", "onvif"} <= ids
    assert all(x["main"].startswith("rtsp://") and x["sub"].startswith("rtsp://") for x in t["templates"])
    assert cameras.build_url("hikvision", "10.0.0.5", "admin", "pw") == \
        "rtsp://admin:pw@10.0.0.5:554/Streaming/Channels/102"
    assert cameras.build_url("hikvision", "nvr", "admin", "pw", channel=3, stream="main").endswith("/Channels/301")
    assert cameras.build_url("dahua", "h", "a", "p", channel=2).endswith("channel=2&subtype=1")
    assert cameras.build_url("reolink", "h", "a", "p").endswith("/h264Preview_01_sub")
    assert "p%40ss%3Aw%2Fd@" in cameras.build_url("tapo", "h", "a", "p@ss:w/d")  # credentials are percent-encoded
    with pytest.raises(ValueError):
        cameras.build_url("nope", "h")


def test_validate_source_rejects_exotic_inputs():
    assert cameras.validate_source("0") == "0"
    assert cameras.validate_source(RTSP) == RTSP
    assert cameras.validate_source(str(CLIP)) == str(CLIP)
    for bad in ["", "concat:/etc/passwd|x", "file:///etc/passwd", "ftp://h/x", "rtsp://", "no/such/file.mp4"]:
        with pytest.raises(ValueError):
            cameras.validate_source(bad)


# ---- probing ----------------------------------------------------------------------------------------------------
def test_probe_local_video_file():
    r = cameras.probe_source(str(CLIP), timeout_s=10)
    assert r["ok"] and r["error"] == ""
    assert (r["width"], r["height"]) == (960, 540) and r["fps"]
    jpg = base64.b64decode(r["snapshot_jpeg_b64"])
    assert jpg[:2] == b"\xff\xd8" and cv2.imdecode(__import__("numpy").frombuffer(jpg, "uint8"), 1) is not None


def test_probe_failure_is_redacted_and_bounded():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); port = srv.getsockname()[1]; srv.close()  # closed port
    src = f"rtsp://admin:{SECRET}@127.0.0.1:{port}/stream1"
    t0 = time.time()
    r = cameras.probe_source(src, timeout_s=5)
    assert not r["ok"] and r["error"] and time.time() - t0 < 6.5
    assert SECRET not in json.dumps(r) and r["source"] == f"rtsp://admin:***@127.0.0.1:{port}/stream1"
    assert not cameras.probe_source("does/not/exist.mp4")["ok"]


# ---- ONVIF WS-Discovery -----------------------------------------------------------------------------------------
PROBE_MATCHES = """<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
 xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl" xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
<SOAP-ENV:Header>
 <wsa:MessageID>uuid:6a0b0e3c-0000-4000-8000-a0b1c2d3e4f5</wsa:MessageID>
 <wsa:RelatesTo>uuid:11111111-2222-3333-4444-555555555555</wsa:RelatesTo>
 <wsa:To>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</wsa:To>
 <wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</wsa:Action>
</SOAP-ENV:Header>
<SOAP-ENV:Body><wsdd:ProbeMatches>
 <wsdd:ProbeMatch>
  <wsa:EndpointReference><wsa:Address>urn:uuid:4a3c-cam-front</wsa:Address></wsa:EndpointReference>
  <wsdd:Types>dn:NetworkVideoTransmitter tds:Device</wsdd:Types>
  <wsdd:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/Profile/Streaming
   onvif://www.onvif.org/name/HIKVISION%20Front%20Door onvif://www.onvif.org/hardware/DS-2CD2143G2-I
   onvif://www.onvif.org/location/city/somewhere</wsdd:Scopes>
  <wsdd:XAddrs>http://192.168.1.64/onvif/device_service http://[fe80::1]/onvif/device_service</wsdd:XAddrs>
  <wsdd:MetadataVersion>10</wsdd:MetadataVersion>
 </wsdd:ProbeMatch>
 <wsdd:ProbeMatch>
  <wsa:EndpointReference><wsa:Address>urn:uuid:dahua-nvr</wsa:Address></wsa:EndpointReference>
  <wsdd:Types>dn:NetworkVideoTransmitter</wsdd:Types>
  <wsdd:Scopes>onvif://www.onvif.org/name/Dahua onvif://www.onvif.org/hardware/NVR4108HS</wsdd:Scopes>
  <wsdd:XAddrs>http://192.168.1.108:80/onvif/device_service</wsdd:XAddrs>
 </wsdd:ProbeMatch>
</wsdd:ProbeMatches></SOAP-ENV:Body></SOAP-ENV:Envelope>"""


def test_parse_probe_matches():
    devs = cameras.parse_probe_matches(PROBE_MATCHES.encode())
    assert len(devs) == 2
    a, b = devs
    assert a["endpoint"] == "urn:uuid:4a3c-cam-front" and a["host"] == "192.168.1.64"
    assert a["name"] == "HIKVISION Front Door" and a["hardware"] == "DS-2CD2143G2-I"
    assert a["location"] == "city/somewhere" and a["vendor_guess"] == "hikvision"
    assert "dn:NetworkVideoTransmitter" in a["types"] and len(a["xaddrs"]) == 2
    assert b["host"] == "192.168.1.108" and b["vendor_guess"] == "dahua" and b["hardware"] == "NVR4108HS"
    assert cameras.parse_probe_matches(b"<not-xml") == [] and cameras.parse_probe_matches(b"<a/>") == []


def test_discover_against_local_responder(monkeypatch):
    """Loopback stand-in for the multicast group: checks the Probe we send and that replies are collected."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0)); srv.settimeout(5)
    seen = {}

    def responder():
        try:
            data, addr = srv.recvfrom(65535)
            seen["probe"] = data.decode()
            srv.sendto(PROBE_MATCHES.encode(), addr)
        except OSError:
            pass
    th = threading.Thread(target=responder, daemon=True); th.start()
    monkeypatch.setattr(cameras, "WSD_ADDR", srv.getsockname())
    monkeypatch.setattr(cameras, "_interface_ips", lambda: ["127.0.0.1"])
    devs = cameras.discover(timeout_s=1.5)
    th.join(2); srv.close()
    assert "dn:NetworkVideoTransmitter" in seen["probe"] and "/discovery/Probe<" in seen["probe"]
    assert {d["host"] for d in devs} == {"192.168.1.64", "192.168.1.108"}
    assert all(d["responder"] == "127.0.0.1" for d in devs)


# ---- live grabber -----------------------------------------------------------------------------------------------
class _ClosedCap:
    def isOpened(self): return False
    def release(self): pass


def test_grabber_backoff_is_exponential_and_capped():
    g = LatestFrameGrabber("rtsp://x", "t", backoff_min=0.01, backoff_max=0.04, opener=lambda s: _ClosedCap())
    g.start()
    time.sleep(0.4)
    g.stop()
    d = list(g.delays)
    assert d[:4] == [0.01, 0.02, 0.04, 0.04] and not g.connected and g.error == "cannot open source"


def test_grabber_keeps_latest_frame_and_reconnects_after_stream_end():
    g = LatestFrameGrabber(str(CLIP), "t", backoff_min=0.01, backoff_max=0.02, opener=cv2.VideoCapture)
    g.start()
    got = g.wait_newer(0, timeout=5)
    deadline = time.time() + 10
    while g.reconnects < 1 and time.time() < deadline:
        time.sleep(0.05)
    g.stop()
    assert got is not None and got[0].shape[:2] == (540, 960)
    assert g.reconnects >= 1 and g.seq > 100  # file played to the end, stream "dropped", reopened, kept going


def test_live_worker_marks_down_then_recovers(tmp_path, monkeypatch):
    """A live (rtsp) camera whose first connects fail raises a 'down' health alert, then recovers and analyses."""
    calls = {"n": 0}

    def opener(src):
        calls["n"] += 1
        return _ClosedCap() if calls["n"] <= 6 else cv2.VideoCapture(str(CLIP))  # ~1.2 s outage first
    monkeypatch.setattr(pipeline_mod, "open_capture", opener)
    site = make_site(tmp_path)
    site.cfg["cameras"][0].update(source=RTSP, reconnect_min_s=0.05, reconnect_max_s=0.3, down_grace_s=0.3)
    site.workers = [site._make_worker(site.cfg["cameras"][0])]
    w = site.workers[0]
    site.start()
    deadline = time.time() + 30
    while w.stats["frames"] < 8 and time.time() < deadline:
        time.sleep(0.1)
    site.stop()
    assert w.stats["frames"] >= 8 and w.stats["status"] == "stopped"
    assert w.grabber.connections >= 1 and not w.grabber.thread.is_alive()
    down = [e for a in site.store.list_alerts(rule="camera_health") for e in a["events"]
            if e["derivation"]["condition"] == "down"]
    assert down, "stream loss should raise a camera-down health alert"
    assert w.health.state != "DOWN"  # recovered once frames flowed
    assert SECRET not in json.dumps(w.stats)
    assert w.last_frame is not None and w.source_size == (960, 540) and w.scale == 1.0


# ---- uncalibrated camera with pixel zones -----------------------------------------------------------------------
def test_uncalibrated_camera_with_pixel_zone_raises_intrusion(tmp_path):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["record"] = False
    cfg["cameras"] = [{"id": "gate", "name": "Gate (uncalibrated)", "source": str(CLIP),
                       "masks_px": [[[0, 0], [50, 0], [50, 50], [0, 50]]],
                       "zones": [{"id": "door", "name": "Door", "criticality": 3,
                                  "polygon_px": [[600, 250], [720, 250], [720, 360], [600, 360]],  # FakeBackend foot (653, 302)
                                  "rules": {"intrusion": {"min_seconds": 1.0}, "running": {}}},
                                 {"id": "metric_only", "polygon": [[0, 0], [5, 0], [5, 5]], "rules": {"loitering": {}}}]}]
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(); p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    w = site.workers[0]
    assert w.H is None and w.zones == [] and w.calib_info["calibrated"] is False
    assert any("running" in s for s in w.calib_info["skipped"]) and any("metric_only" in s for s in w.calib_info["skipped"])
    site.start(); site.wait(timeout=120); site.stop()
    alerts = site.store.list_alerts(rule="intrusion")
    assert alerts, "pixel-zone intrusion should fire on an uncalibrated camera"
    ev = next(e for e in alerts[0]["events"] if e["rule"] == "intrusion")
    d = ev["derivation"]
    assert d["space"] == "image_px" and d["calib_error_m"] is None and "margin_px" in d["threshold"]
    assert ev["ground"] is None and "px past boundary" in ev["observation"]
    assert ev["evidence"]["calibration"]["calibrated"] is False
    assert not site.store.list_alerts(rule="running")
    assert w.tracks == []  # no map output without calibration
    assert w.last_dets and w.last_dets[0][0] == 1 and w.source_size == (960, 540)
    client = TestClient(create_app(site, open_demo=True))
    cams = client.get("/api/cameras").json()
    assert cams[0]["calibrated"] is False and cams[0]["zones"][0]["polygon_px"]
    snap = client.get("/api/cameras/gate/snapshot.jpg")
    assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"


# ---- camera CRUD API + persistence ------------------------------------------------------------------------------
def test_camera_crud_api_persists_and_redacts(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    cfg_path = site.config_path

    r = client.post("/api/cameras", json={"name": "Loading Dock", "source": RTSP})
    assert r.status_code == 200, r.text
    cam = r.json()
    assert cam["id"] == "cam_loading_dock" and cam["source"] == "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/102"
    assert cam["calibrated"] is False and cam["enabled"] is True
    saved = json.loads(cfg_path.read_text())
    assert [c["id"] for c in saved["cameras"]] == ["cam1", "cam_loading_dock"]
    # the git-tracked config stores the redacted URL; the working secret is in the git-ignored secrets file
    assert saved["cameras"][1]["source"] == "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/102"
    secrets = json.loads(cfg_path.with_name(cfg_path.stem + ".secrets.json").read_text())
    assert secrets["cam_loading_dock"]["source"] == RTSP and SECRET not in cfg_path.read_text()
    assert site.camera_config("cam_loading_dock")["source"] == RTSP  # the running camera keeps the real URL
    assert cfg_path.with_name(cfg_path.name + ".bak").exists()
    assert client.post("/api/cameras", json={"id": "cam1", "source": RTSP}).status_code == 409
    assert client.post("/api/cameras", json={"source": "concat:/etc/passwd"}).status_code == 400
    assert client.post("/api/cameras", json={"id": "bad id!", "source": RTSP}).status_code == 400

    listed = client.get("/api/cameras").json()
    assert {c["id"] for c in listed} == {"cam1", "cam_loading_dock"}
    assert SECRET not in json.dumps(listed)

    # echoing the redacted source back keeps the secret; renaming/disabling persists
    r = client.put("/api/cameras/cam_loading_dock", json={"name": "Dock", "source": cam["source"], "enabled": False})
    assert r.status_code == 200 and r.json()["name"] == "Dock" and r.json()["status"] == "disabled"
    saved = json.loads(cfg_path.read_text())
    assert saved["cameras"][1]["source"] == cameras.redact(RTSP) and saved["cameras"][1]["enabled"] is False
    assert site.worker("cam_loading_dock") is None
    assert client.put("/api/cameras/cam_loading_dock", json={"source": "rtsp://admin:***@elsewhere/x"}).status_code == 400
    assert client.put("/api/cameras/cam_loading_dock", json={"calibration": {}}).status_code == 400
    assert client.put("/api/cameras/nope", json={"name": "x"}).status_code == 404

    # zones: validated against rules.DEFAULTS
    zones = [{"id": "dock", "name": "Dock edge", "polygon_px": [[10, 10], [200, 10], [200, 200]],
              "rules": {"intrusion": {"min_seconds": 2}, "running": {}}}]
    r = client.put("/api/cameras/cam_loading_dock/zones", json={"zones": zones})
    assert r.status_code == 200 and any("running" in w for w in r.json()["warnings"])
    assert json.loads(cfg_path.read_text())["cameras"][1]["zones"][0]["rules"]["intrusion"]["min_seconds"] == 2
    for bad in [[{**zones[0], "rules": {"teleport": {}}}],
                [{**zones[0], "rules": {"intrusion": {"speed_mps": 1}}}],
                [{**zones[0], "rules": {"wrong_way": {}}}],
                [{**zones[0], "polygon_px": [[0, 0], [1, 1]]}],
                [zones[0], zones[0]]]:
        assert client.put("/api/cameras/cam_loading_dock/zones", json={"zones": bad}).status_code == 400, bad
    r = client.put("/api/cameras/cam_loading_dock/masks", json={"masks_px": [[[0, 0], [40, 0], [40, 40]]]})
    assert r.status_code == 200 and r.json()["masks_px"] == [[[0.0, 0.0], [40.0, 0.0], [40.0, 40.0]]]
    assert client.put("/api/cameras/cam_loading_dock/masks", json={"masks_px": [[[0, "a"], [1, 1], [2, 2]]]}).status_code == 400

    # re-enable: a worker exists again (not started: the site is not running) and is uncalibrated
    assert client.put("/api/cameras/cam_loading_dock", json={"enabled": True}).status_code == 200
    w = site.worker("cam_loading_dock")
    assert w is not None and w.H is None and [z.id for z in w.px_zones] == ["dock"] and not w.is_alive()

    assert client.delete("/api/cameras/cam_loading_dock").status_code == 200
    assert client.delete("/api/cameras/cam_loading_dock").status_code == 404
    assert [c["id"] for c in json.loads(cfg_path.read_text())["cameras"]] == ["cam1"]
    bak = json.loads(cfg_path.with_name(cfg_path.name + ".bak").read_text())
    assert [c["id"] for c in bak["cameras"]] == ["cam1", "cam_loading_dock"]  # previous version kept

    log = site.store.list_audit(limit=100)
    actions = {e["action"] for e in log}
    assert {"camera_add", "camera_update", "camera_zones", "camera_masks", "camera_delete"} <= actions
    assert SECRET not in json.dumps(log)
    assert client.get("/api/cameras/templates").json()["templates"]


def test_update_restarts_running_worker_and_keeps_relative_paths(tmp_path):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    proj = tmp_path / "proj"
    (proj / "configs").mkdir(parents=True)
    (proj / "clips").mkdir()
    (proj / "clips" / "a.mp4").write_bytes(CLIP.read_bytes())
    cfg["cameras"][0]["source"] = "clips/a.mp4"  # relative to the project root, like the shipped configs
    p = proj / "configs" / "site.json"
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=True, backend_factory=lambda c: FakeBackend())
    old = site.worker("cam1")
    site.start()
    time.sleep(1.0)
    new = site.update_camera_config("cam1", {"name": "Renamed", "calibration": {"calib_error_m": 0.5}})
    assert old.done.is_set() and not old.is_alive() and old.stats["status"] == "stopped"
    assert new is site.worker("cam1") and new.is_alive() and new.name == "Renamed"
    assert new.calib_info["calib_error_m"] == 0.5
    saved = json.loads(p.read_text())
    assert saved["cameras"][0]["source"] == "clips/a.mp4"  # not rewritten to an absolute path
    assert saved["cameras"][0]["calibration"]["image_points"]  # deep merge kept sibling keys
    with pytest.raises(ValueError):  # invalid calibration is rejected before anything changes
        site.update_camera_config("cam1", {"calibration": {"image_points": [[0, 0], [1, 1], [2, 2], [3, 3]]}})
    assert site.worker("cam1") is new and new.is_alive()
    assert site.update_camera_config("cam1", {"calibration": None}).H is None  # None deletes -> uncalibrated
    site.stop()
    assert all(not w.is_alive() for w in site.workers)


def test_stream_reconnect_options_via_api(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    body = {"id": "cam_flaky", "name": "Flaky link", "source": RTSP, "enabled": False,
            "reconnect_min_s": 0.5, "reconnect_max_s": 4, "down_grace_s": 3}
    r = client.post("/api/cameras", json=body)
    assert r.status_code == 200, r.text
    assert (r.json()["reconnect_max_s"], r.json()["down_grace_s"]) == (4.0, 3.0)
    saved = json.loads(site.config_path.read_text())["cameras"][-1]
    assert (saved["reconnect_min_s"], saved["reconnect_max_s"], saved["down_grace_s"]) == (0.5, 4.0, 3.0)
    for bad in ({"reconnect_max_s": 0}, {"reconnect_min_s": 10, "reconnect_max_s": 5}, {"down_grace_s": "x"},
                {"reconnect_max_s": 1e9}):
        assert client.post("/api/cameras", json={**body, "id": "cam_bad", **bad}).status_code == 400, bad
        assert client.put("/api/cameras/cam_flaky", json=bad).status_code == 400, bad
    r = client.put("/api/cameras/cam_flaky", json={"reconnect_max_s": 8, "enabled": True})
    assert r.status_code == 200 and r.json()["reconnect_max_s"] == 8.0
    assert site.worker("cam_flaky").cfg["reconnect_max_s"] == 8.0  # the restarted worker gets the new backoff cap
    assert any(e["action"] == "camera_update" and "reconnect_max_s" in e["details"]
               for e in site.store.list_audit(limit=20))


def test_cameras_page_response_shapes(tmp_path):
    """Pins the shapes argonix/static/js/cameras.js reads: templates/discover are objects wrapping lists, and every
    template placeholder is one that the wizard's fillTemplate() substitutes."""
    import re
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    t = client.get("/api/cameras/templates").json()
    assert isinstance(t["templates"], list) and isinstance(t["disclaimer"], str)
    used = {m for x in t["templates"] for m in re.findall(r"\{(\w+)\}", x["main"] + x["sub"])}
    js = (ROOT / "argonix" / "static" / "js" / "cameras.js").read_text(encoding="utf-8")
    assert all(f'"{{{m}}}"' in js for m in used), used  # fillTemplate replaces each placeholder
    assert "t?.templates" in js and "r?.devices" in js  # the wizard unwraps both response objects


def test_add_camera_via_api_runs_uncalibrated_with_pixel_zone(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    r = client.post("/api/cameras", json={"id": "cam_px", "name": "Pixel cam", "source": str(CLIP),
                                          "zones": [{"id": "door", "polygon_px": [[600, 250], [720, 250], [720, 360], [600, 360]],
                                                     "rules": {"intrusion": {}}}]})
    assert r.status_code == 200, r.text
    site.start(); site.wait(timeout=120); site.stop()
    evs = [e for a in site.store.list_alerts(camera="cam_px") for e in a["events"] if e["camera_id"] == "cam_px"]
    assert any(e["rule"] == "intrusion" and e["derivation"]["space"] == "image_px" for e in evs)


# ---- permissions ------------------------------------------------------------------------------------------------
def test_camera_endpoints_permissions(tmp_path):
    site, app = client_with_users(tmp_path)
    vic, olga, ada = (login(app, u) for u in ("vic", "olga", "ada"))
    body = {"name": "New", "source": str(CLIP)}
    assert TestClient(app).get("/api/cameras").status_code == 401
    assert vic.get("/api/cameras").status_code == 200
    assert vic.post("/api/cameras", json=body).status_code == 403
    assert olga.post("/api/cameras", json=body).status_code == 403
    assert vic.post("/api/cameras/test", json={"source": str(CLIP)}).status_code == 403
    assert vic.get("/api/cameras/templates").status_code == 403
    assert vic.get("/api/cameras/discover").status_code == 403
    assert olga.put("/api/cameras/cam1/zones", json={"zones": []}).status_code == 403
    assert olga.put("/api/cameras/cam1/masks", json={"masks_px": []}).status_code == 403
    assert vic.delete("/api/cameras/cam1").status_code == 403
    assert vic.get("/api/cameras/cam1/snapshot.jpg").status_code == 200
    r = ada.post("/api/cameras", json=body)
    assert r.status_code == 200, r.text
    assert ada.post("/api/cameras/test", json={"source": str(CLIP)}).json()["ok"] is True
    assert ada.put("/api/cameras/cam1/masks", json={"masks_px": []}).status_code == 200
    log = login(app, "aud").get("/api/audit").json()
    assert any(e["actor"] == "ada" and e["action"] == "camera_add" for e in log)
    assert any(e["actor"] == "ada" and e["action"] == "camera_test" for e in log)


def test_direction_px_arrow_projects_to_ground_direction(tmp_path):
    from tools.make_synthetic import H_G2I
    from argonix.geometry import apply_h
    from argonix.pipeline import Site
    from tests.test_server import FakeBackend
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cam = cfg["cameras"][1]  # corridor zone, permitted direction +x in ground
    cam["source"] = str(ROOT / cam["source"])
    a, b = apply_h(H_G2I, (4, 3)), apply_h(H_G2I, (10, 3))  # arrow drawn along +x on the image
    z = next(z for z in cam["zones"] if z["id"] == "corridor")
    z.pop("direction")
    z["direction_px"] = [list(a), list(b)]
    cfg["cameras"] = [cam]
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    d = next(z for z in site.workers[0].zones if z.id == "corridor").direction
    assert d[0] == pytest.approx(6, abs=1e-6) and d[1] == pytest.approx(0, abs=1e-6)


def test_site_without_cameras_prewarms_the_shared_detector(tmp_path):
    """A GPU site started empty builds its shared model in the background, so the first API-added camera doesn't."""
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"], cfg["record"] = [], False
    p = tmp_path / "site.json"
    p.write_text(json.dumps(cfg))
    warmed = threading.Event()

    def factory(c):
        return FakeBackend()
    factory.prewarm = lambda: (time.sleep(0.2), warmed.set())
    site = Site(p, data_dir=tmp_path / "run", realtime=True, backend_factory=factory)
    site.start()
    try:
        assert warmed.wait(5), "prewarm did not run for a site with no cameras"
    finally:
        site.stop()
    # a site that already has cameras built its backends in the constructor: no extra prewarm
    cfg["cameras"] = json.loads((ROOT / "configs" / "demo_site.json").read_text())["cameras"][:1]
    cfg["cameras"][0]["source"] = str(ROOT / cfg["cameras"][0]["source"])
    p.write_text(json.dumps(cfg))
    calls = []
    factory.prewarm = lambda: calls.append(1)
    site = Site(p, data_dir=tmp_path / "run2", realtime=True, backend_factory=factory)
    site.start()
    time.sleep(0.3)
    site.stop()
    assert calls == []
