"""API tests using a fake tracker backend (fast, no model)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from argonix.pipeline import Site
from argonix.server import create_app

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Walks one person straight into cam1's restricted zone; returns image-space boxes."""
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self):
        self.n = 0

    def track(self, frame):
        self.n += 1
        return [(1, (633, 200, 673, 302), "person", 0.9)]  # foot ~ ground (13, 6.5), inside server-door zone


def make_site(tmp_path):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())


def test_api_flow(tmp_path):
    site = make_site(tmp_path)
    site.start(); site.wait(timeout=120); site.stop()
    client = TestClient(create_app(site, open_demo=True))
    assert client.get("/").status_code == 200
    cfg = client.get("/api/config").json()
    assert cfg["cameras"][0]["zones"] and cfg["explain_enabled"] is False
    alerts = client.get("/api/alerts").json()
    assert alerts and alerts[0]["events"][0]["rule"] in {"intrusion", "loitering"}
    aid = alerts[0]["id"]
    inc = client.get(f"/api/incidents/{alerts[0]['incident_id']}").json()  # the pipeline groups alerts into incidents
    assert aid in inc["alert_ids"] and inc["phase"] in {"open", "active", "cooling", "closed"}
    assert client.get("/api/alerts?priority=P9").json() == []
    assert client.get("/api/alerts?rule=intrusion").json()
    snap = alerts[0]["events"][0]["evidence"]["snapshot"]
    assert client.get("/media/" + snap).status_code == 200
    r = client.post(f"/api/alerts/{aid}/status", json={"status": "acknowledged", "note": "checked"})
    assert r.json()["status"] == "acknowledged" and r.json()["operator_note"] == "checked"
    assert client.post(f"/api/alerts/{aid}/status", json={"status": "bogus"}).status_code == 400
    assert client.post(f"/api/alerts/{aid}/explain").status_code == 503
    assert client.get("/api/cameras/cam1/latest.jpg").headers["content-type"] == "image/jpeg"
    s = client.get("/api/stats").json()
    assert s["cameras"]["cam1"]["frames"] > 0
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "tick"


def test_explain_endpoint_stores_assistive_text(tmp_path):
    site = make_site(tmp_path)
    site.start(); site.wait(timeout=120); site.stop()
    client = TestClient(create_app(site, explainer=lambda a, d: "One person stands near the door.", open_demo=True))
    assert client.get("/api/config").json()["explain_enabled"] is True
    aid = client.get("/api/alerts").json()[0]["id"]
    before = client.get(f"/api/alerts/{aid}").json()
    after = client.post(f"/api/alerts/{aid}/explain").json()
    assert after["explanation"] == "One person stands near the door."
    assert after["priority"] == before["priority"] and after["score"] == before["score"]  # never affects scoring


def test_vlm_client_request_shape(tmp_path, monkeypatch):
    import httpx
    from argonix.vlm import RemoteExplainer
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "s.jpg").write_bytes(b"\xff\xd8jpg")
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(json)
        return httpx.Response(200, json={"text": "ok", "model": "m", "latency_s": 1.0}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    out = RemoteExplainer("http://x/explain")({"events": [{"observation": "person track #1", "evidence": {"snapshot": "evidence/s.jpg"}}]}, tmp_path)
    assert out.startswith("ok") and "people" in seen["prompt"] and seen["image_b64"]


def test_tamper_injection_raises_health_alert(tmp_path):
    site = make_site(tmp_path)
    w = site.workers[0]
    orig = w.backend.track

    def track(frame):
        if w.stats["frames"] == 40:  # ~8 s of media time, after the 5 s warm-up
            w.inject("cover", 999)
        return orig(frame)
    w.backend.track = track
    site.start(); site.wait(timeout=120); site.stop()
    client = TestClient(create_app(site, open_demo=True))
    hal = client.get("/api/alerts?rule=camera_health").json()
    assert hal, "cover fault not detected"
    ev = hal[0]["events"][0]
    assert ev["derivation"]["condition"] in {"covered", "blur", "moved"} and ev["evidence"]["snapshot"]
    assert client.post("/api/cameras/cam1/inject", json={"fault": "nope"}).status_code == 400
    assert client.post("/api/cameras/cam1/inject", json={"fault": "blur", "seconds": 1}).status_code == 200


def test_detection_mask_suppresses_static_false_positive(tmp_path):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    cfg["cameras"][0]["source"] = str(ROOT / cfg["cameras"][0]["source"])
    cfg["cameras"][0]["masks_px"] = [[[600, 280], [700, 280], [700, 320], [600, 320]]]  # covers FakeBackend's foot
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    site.start(); site.wait(timeout=120); site.stop()
    assert site.store.list_alerts(rule="intrusion") == [] and site.workers[0].stats["masked"] > 50
