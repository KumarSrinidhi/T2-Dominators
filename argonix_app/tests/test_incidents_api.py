"""Incident / hotspot / investigation API with RBAC. The Site is built but not started: events are injected through
Site.on_event (the pipeline's own entry point), so these tests need no video decode or model."""
import json
import time

from fastapi.testclient import TestClient

import argonix.auth as auth_mod
from argonix.auth import Auth
from argonix.models import Event
from argonix.rules import SEVERITY
from argonix.server import create_app
from tests.test_server import ROOT, FakeBackend, make_site

PW = "correct-horse-battery"  # test fixture only


def ev(rule, t, cam, zone, zname, tracks=(1,), crit=1, site_zone=None):
    return Event(rule, cam, zone, zname, list(tracks), t - 1, t, f"person track #{tracks[0] if tracks else '-'} {rule}",
                 {"rule": rule}, 0.9, SEVERITY[rule], crit, site_zone, None, {"snapshot": None, "clip": None})


def fast_hash(pw):
    """Test users with few PBKDF2 iterations (verify_password honours the stored count) to keep the test fast."""
    old, auth_mod.ITER = auth_mod.ITER, 1000
    try:
        return auth_mod.hash_password(pw)
    finally:
        auth_mod.ITER = old


def build(tmp_path, users=True):
    site = make_site(tmp_path)
    now = time.time()
    site.on_event(ev("intrusion", now - 30, "cam1", "server_door", "Server room door", (1,), 3, "server_door"))
    site.on_event(ev("loitering", now - 28, "cam1", "walkway", "Walkway", (2,)))
    auth = None
    if users:
        u = {name: {**fast_hash(PW), "roles": [role]} for name, role in
             [("vic", "viewer"), ("olga", "operator"), ("sam", "supervisor"), ("ada", "admin"), ("aud", "auditor")]}
        (tmp_path / "users.json").write_text(json.dumps(u))
        auth = Auth(tmp_path / "users.json", tmp_path / "secret.key")
    return site, create_app(site, auth=auth)


def login(app, user):
    c = TestClient(app, headers={"X-Argonix-CSRF": "1"})  # cookie-authenticated state changes need the CSRF header
    assert c.post("/api/login", json={"username": user, "password": PW}).status_code == 200
    return c


def test_incident_endpoints_and_permissions(tmp_path):
    site, app = build(tmp_path)
    assert TestClient(app).get("/api/incidents").status_code == 401
    vic, olga, sam, ada, aud = (login(app, u) for u in ("vic", "olga", "sam", "ada", "aud"))
    incs = vic.get("/api/incidents").json()
    assert len(incs) == 2 and all("events" not in i and i["n_alerts"] == 1 for i in incs)
    alerts = vic.get("/api/alerts").json()
    assert all(a["incident_id"] for a in alerts)
    intr = next(a for a in alerts if a["events"][0]["rule"] == "intrusion")
    i1 = intr["incident_id"]
    i2 = next(a["incident_id"] for a in alerts if a["id"] != intr["id"])
    assert [x["id"] for x in vic.get(f"/api/incidents?alert_id={intr['id']}").json()] == [i1]
    d = vic.get(f"/api/incidents/{i1}").json()
    assert d["alerts_detail"][0]["id"] == intr["id"] and d["phase"] == "active" and d["priority"] == "P1"
    assert d["notifications"][0]["decision"] == "would_push" and d["disclaimer"]
    assert vic.get("/api/incidents/I0-404").status_code == 404
    g = vic.get(f"/api/incidents/{i1}/graph").json()
    assert {n["kind"] for n in g["nodes"]} >= {"event", "camera", "zone", "track"}
    assert vic.get("/api/incidents/I0-404/graph").status_code == 404

    assert vic.patch(f"/api/incidents/{i1}", json={"status": "acknowledged"}).status_code == 403
    assert ada.patch(f"/api/incidents/{i1}", json={"status": "acknowledged"}).status_code == 403  # separation of duties
    assert olga.patch(f"/api/incidents/{i1}", json={"status": "acknowledged"}).json()["status"] == "acknowledged"
    assert olga.patch(f"/api/incidents/{i1}", json={"assignee": "olga"}).status_code == 403
    assert olga.post(f"/api/incidents/{i1}/merge", json={"sources": [i2]}).status_code == 403
    assert sam.patch(f"/api/incidents/{i1}", json={"bogus": 1}).status_code == 400
    assert sam.patch(f"/api/incidents/{i1}", json={"status": "nope"}).status_code == 400
    r = sam.patch(f"/api/incidents/{i1}", json={"assignee": "olga", "summary": "Door checked on camera."}).json()
    assert r["assignee"] == "olga" and r["summary"] == "Door checked on camera."
    assert sam.post(f"/api/incidents/{i1}/merge", json={"sources": "x"}).status_code == 400
    assert sam.post(f"/api/incidents/{i1}/merge", json={"sources": ["I0-404"]}).status_code == 404
    m = sam.post(f"/api/incidents/{i1}/merge", json={"sources": [i2]}).json()
    assert len(m["alert_ids"]) == 2 and vic.get(f"/api/incidents/{i2}").json()["merged_into"] == i1
    moved = next(a for a in m["alert_ids"] if a != intr["id"])
    assert vic.get(f"/api/alerts/{moved}").json()["incident_id"] == i1
    assert sam.post(f"/api/incidents/{i1}/split", json={"alert_ids": m["alert_ids"]}).status_code == 400
    new = sam.post(f"/api/incidents/{i1}/split", json={"alert_ids": [moved]}).json()
    assert new["split_from"] == i1 and vic.get(f"/api/alerts/{moved}").json()["incident_id"] == new["id"]
    log = aud.get("/api/audit").json()
    assert {"incident_merge", "incident_split", "incident_assign", "incident_status:acknowledged"} <= \
        {e["action"] for e in log}
    assert any(e["action"] == "incident_merge" and e["actor"] == "sam" for e in vic.get(f"/api/incidents/{i1}/audit").json())
    assert vic.get("/api/notifications").json()


def test_hotspots_and_investigate_api(tmp_path):
    site, app = build(tmp_path)
    vic, aud = login(app, "vic"), login(app, "aud")
    h = vic.get("/api/hotspots?by=camera").json()
    assert h["rows"][0]["key"] == "cam1" and h["rows"][0]["total"] == 2 and len(h["overall"]) == 168
    assert vic.get("/api/hotspots?by=nope").status_code == 400
    sp = vic.get("/api/hotspots").json()["spatial"]  # shared with /api/analytics/hotspots
    assert sp["events"] >= 2 and {z["zone_id"] for z in sp["zones"]} >= {"server_door", "walkway"}
    sch = vic.get("/api/investigate/schema").json()
    assert "cam1" in {c["id"] for c in sch["vocab"]["cameras"]} and sch["llm_configured"] is False
    r = vic.post("/api/investigate", json={"question": "intrusions at the server room door today"}).json()
    assert r["engine"] == "rules" and r["results"]["total"] == 1 and r["citations"]
    assert r["citations"][0]["id"] in r["summary"]
    r = vic.post("/api/investigate", json={"question": "open incidents on the server wing"}).json()
    assert r["query"]["target"] == "incidents" and r["results"]["total"] == 2
    assert all(c["kind"] == "incident" for c in r["citations"])
    bad = vic.post("/api/investigate", json={"query": {"sql": "DROP TABLE alerts"}})
    assert bad.status_code == 400 and "unknown field" in bad.json()["detail"]
    assert vic.post("/api/investigate", json={"question": 5}).status_code == 400
    acts = [e["action"] for e in aud.get("/api/audit").json()]
    assert "investigate" in acts and "investigate_rejected" in acts
    # an LLM configured on the site is used through the same validator (fake, no network)

    class LLM:
        model_id = "fake"

        def chat(self, messages, max_new_tokens=200):
            return {"text": '{"rules": ["intrusion"], "time": {"preset": "today"}, "aggregation": "count"}',
                    "model": "fake"}
    site.llm = LLM()
    r = vic.post("/api/investigate", json={"question": "how many door entries today?", "summarize": "template"}).json()
    assert r["engine"] == "llm" and r["results"]["total"] == 1


def test_incidents_disabled_returns_404(tmp_path):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    cfg["incidents"] = {"enabled": False}
    cfg["cameras"][0]["source"] = str(ROOT / cfg["cameras"][0]["source"])
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    from argonix.pipeline import Site
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    site.on_event(ev("intrusion", time.time(), "cam1", "server_door", "Server room door", (1,), 3, "server_door"))
    c = TestClient(create_app(site, open_demo=True))
    assert c.get("/api/incidents").status_code == 404 and c.get("/api/incidents").json()["detail"] == "Not Found"
    assert "incident_id" not in c.get("/api/alerts").json()[0]
    assert c.get("/api/hotspots").json()["metric"] == "alerts"  # falls back to the alert store
    assert c.post("/api/investigate", json={"question": "intrusions today"}).json()["results"]["total"] == 1
