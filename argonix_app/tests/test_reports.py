"""Incident report, assistive multi-camera incident summary (cite-or-refuse, fake VLM) and shift handover API.
The Site is built but not started: events are injected through Site.on_event, so no video decode or model runs."""
import hashlib
import json
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

import argonix.auth as auth_mod
from argonix.auth import Auth
from argonix.models import Event
from argonix.report import verify_report
from argonix.rules import SEVERITY
from argonix.server import create_app
from tests.test_incidents_api import PW, fast_hash, login
from tests.test_server import make_site


def ev(rule, t, cam, zone, zname, tracks=(1,), crit=1, snapshot=None, ground=None):
    return Event(rule, cam, zone, zname, list(tracks), t - 1, t, f"person track #{tracks[0]} {rule}",
                 {"rule": rule, "threshold": 1}, 0.9, SEVERITY[rule], crit, zone, ground,
                 {"snapshot": snapshot, "clip": None})


class FakeVLM:
    """Stands in for RemoteExplainer: describe(images, prompt) returns whatever the test sets."""

    def __init__(self, text="", fail=False):
        self.text, self.fail, self.calls = text, fail, []

    def __call__(self, alert, data_dir, model=None):   # per-alert explain (unused here)
        return "unused"

    def features(self):
        return {"multi_image"}

    def describe(self, images, prompt, max_new_tokens=260):
        self.calls.append((len(images), prompt))
        if self.fail:
            raise RuntimeError("connection refused")
        return {"text": self.text(prompt) if callable(self.text) else self.text, "model": "fake-vl",
                "device": "cpu", "n_images": len(images)}


def build(tmp_path, explainer=None, snapshots=True, old_alert=False):
    site = make_site(tmp_path)
    snap = None
    if snapshots:
        (site.data / "snaps").mkdir(parents=True, exist_ok=True)
        img = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(img, (40, 30), (90, 100), (0, 255, 0), 2)
        cv2.imwrite(str(site.data / "snaps" / "s1.jpg"), img)
        snap = "snaps/s1.jpg"
    w = site.worker("cam1")
    z = w.zones[0]
    g = tuple(np.mean(np.array(z.polygon, float), axis=0))
    now = time.time()
    site.on_event(ev("intrusion", now - 60, "cam1", z.id, z.name, (1,), 3, snap, g))   # two alerts (> 15 s apart),
    site.on_event(ev("loitering", now - 26, "cam1", z.id, z.name, (1,), 3, None, g))  # one incident (same zone)
    if old_alert:  # a P1 alert nobody acknowledged for 2 h (its incident has closed by quiescence)
        site.on_event(ev("intrusion", now - 7200, "cam1", w.zones[1].id, w.zones[1].name, (7,), 3))
    u = {name: {**fast_hash(PW), "roles": [role]} for name, role in
         [("vic", "viewer"), ("olga", "operator"), ("aud", "auditor")]}
    (tmp_path / "users.json").write_text(json.dumps(u))
    auth = Auth(tmp_path / "users.json", tmp_path / "secret.key")
    return site, create_app(site, explainer=explainer, auth=auth)


def incident_id(site):
    return next(a["incident_id"] for a in site.store.list_alerts() if a["events"][0]["rule"] == "intrusion")


# ---- incident report ------------------------------------------------------------------------------------------------
def test_incident_report(tmp_path, monkeypatch):
    site, app = build(tmp_path)
    iid = incident_id(site)
    inc = site.incidents.get(iid)
    assert TestClient(app).get(f"/api/incidents/{iid}/report").status_code == 401
    vic, aud = login(app, "vic"), login(app, "aud")
    r = vic.get(f"/api/incidents/{iid}/report")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    doc = r.text
    assert iid in doc and all(a in doc for a in inc["alert_ids"])
    assert "Why this priority" in doc and "Corroboration multiplier" in doc and "heuristic, uncalibrated" in doc
    assert "observation" in doc and "derivation" in doc and "@page" in doc and "size:A4" in doc
    snap_hash = hashlib.sha256((site.data / "snaps" / "s1.jpg").read_bytes()).hexdigest()
    assert snap_hash in doc and "data:image/jpeg;base64," in doc      # evidence thumbnail + original's hash
    assert "data:image/png;base64," in doc                            # site map with the incident marked
    assert "src=\"http" not in doc and "<link" not in doc              # self-contained
    assert r.headers["x-report-sha256"] in doc and verify_report(doc) is True
    assert verify_report(doc.replace("Why this priority", "Why this priorities")) is False
    assert "script-src 'sha256-" in r.headers["content-security-policy"]
    log = aud.get("/api/audit").json()
    rec = next(e for e in log if e["action"] == "incident_report_generated")
    assert rec["actor"] == "vic" and rec["alert_id"] == iid and rec["details"]["sha256"] == r.headers["x-report-sha256"]
    assert "attachment" in vic.get(f"/api/incidents/{iid}/report?download=1").headers["content-disposition"]
    assert vic.get("/api/incidents/I0-404/report").status_code == 404
    monkeypatch.setitem(auth_mod.ROLE_PERMISSIONS, "viewer", frozenset({"alert.read"}))
    assert vic.get(f"/api/incidents/{iid}/report").status_code == 403


# ---- assistive incident summary -------------------------------------------------------------------------------------
def test_incident_summary_accepted_with_keyframes(tmp_path):
    fake = FakeVLM()
    site, app = build(tmp_path, explainer=fake)
    iid = incident_id(site)
    aids = site.incidents.get(iid)["alert_ids"]
    fake.text = f"A person entered the server room door zone on cam1 [{aids[0]}] and stayed there [{aids[-1]}]."
    vic, olga, aud = login(app, "vic"), login(app, "olga"), login(app, "aud")
    assert vic.post(f"/api/incidents/{iid}/summary").status_code == 403    # viewers cannot run the VLM
    r = olga.post(f"/api/incidents/{iid}/summary").json()
    assert r["engine"] == "vlm" and r["model"] == "fake-vl" and r["rejected"] is None
    assert r["citations"] == list(dict.fromkeys([aids[0], aids[-1]])) and "not evidence" in r["label"]
    assert fake.calls and fake.calls[0][0] == 1 and aids[0] in fake.calls[0][1]  # 1 keyframe + the alert records
    assert r["keyframes"][0]["alert_id"] == aids[0]
    stored = vic.get(f"/api/incidents/{iid}").json()["assistive_summary"]
    assert stored["text"] == fake.text and stored["generated_by"] == "olga"
    assert any(t["kind"] == "assistive_summary" for t in vic.get(f"/api/incidents/{iid}").json()["timeline"])
    log = [e for e in aud.get("/api/audit").json() if e["action"] == "incident_summary_generated"]
    assert log and log[0]["alert_id"] == iid and log[0]["details"]["engine"] == "vlm"
    assert olga.post("/api/incidents/I0-404/summary").status_code == 404


def test_incident_summary_rejects_and_falls_back(tmp_path):
    fake = FakeVLM("Two people were seen [A999999-1].")
    site, app = build(tmp_path, explainer=fake)
    iid = incident_id(site)
    aids = site.incidents.get(iid)["alert_ids"]
    olga = login(app, "olga")
    r = olga.post(f"/api/incidents/{iid}/summary").json()             # cites an id outside the incident
    assert r["engine"] == "template" and "not in the results" in r["rejected"] and r["model_attempted"] == "fake-vl"
    assert all(c in aids for c in r["citations"]) and r["citations"] and "no model" in r["label"]
    fake.text = f"A suspicious man loitered at the door [{aids[0]}]."    # person-labelling language
    r = olga.post(f"/api/incidents/{iid}/summary").json()
    assert r["engine"] == "template" and "person-labelling" in r["rejected"]
    fake.text = "Someone walked past the door."                          # cites nothing
    assert "cites no" in olga.post(f"/api/incidents/{iid}/summary").json()["rejected"]
    fake.fail = True                                                     # VLM unreachable
    r = olga.post(f"/api/incidents/{iid}/summary").json()
    assert r["engine"] == "template" and "model unavailable" in r["rejected"] and aids[0] in r["text"]


def test_incident_summary_text_model_and_no_model(tmp_path):
    site, app = build(tmp_path, snapshots=False)
    iid = incident_id(site)
    aids = site.incidents.get(iid)["alert_ids"]
    olga = login(app, "olga")
    r = olga.post(f"/api/incidents/{iid}/summary").json()             # nothing configured -> deterministic
    assert r["engine"] == "template" and r["rejected"] == "no VLM or text model is configured"

    class LLM:
        model_id = "fake-llm"

        def chat(self, messages, max_new_tokens=200):
            assert aids[0] in messages[-1]["content"]
            return {"text": f"Entry into the door zone was recorded [{aids[0]}].", "model": "fake-llm"}
    site.llm = LLM()                                                     # no keyframes -> text model (/chat)
    r = olga.post(f"/api/incidents/{iid}/summary").json()
    assert r["engine"] == "llm" and r["model"] == "fake-llm" and r["citations"] == [aids[0]]


# ---- shift handover -------------------------------------------------------------------------------------------------
def test_handover_summary_notes_and_print(tmp_path):
    site, app = build(tmp_path, old_alert=True)
    vic, olga, aud = login(app, "vic"), login(app, "olga"), login(app, "aud")
    assert TestClient(app).get("/api/handover/summary").status_code == 401
    s = vic.get("/api/handover/summary").json()
    assert s["counts"]["alerts"] == 3 and s["counts"]["open_alerts"] == 3 and s["hours"] == 8
    assert s["counts"]["open_incidents"] == 1 and s["incidents_enabled"]
    old = next(a for a in site.store.list_alerts() if a["created_at"] < time.time() - 3600)
    assert old["priority"] == "P1" and [o["id"] for o in s["overdue"] if o["kind"] == "alert"] == [old["id"]]
    assert s["top_zones"][0]["alerts"] == 2 and s["top_zones"][0]["camera_id"] == "cam1"
    assert s["camera_problems"] == [] and s["lines"]
    recent = vic.get(f"/api/handover/summary?since={time.time() - 600}").json()
    assert recent["counts"]["alerts"] == 2
    assert vic.get("/api/handover/summary?hours=0").status_code == 400
    assert vic.get(f"/api/handover/summary?since={time.time() + 60}").status_code == 400
    site.worker("cam1").health.active.add("blur")                        # a camera view problem now
    s = vic.get("/api/handover/summary").json()
    assert s["camera_problems"][0]["id"] == "cam1" and "degraded" in s["camera_problems"][0]["problem"]

    assert vic.post("/api/handover", json={"text": "hello"}).status_code == 403
    assert olga.post("/api/handover", json={"text": "  "}).status_code == 400
    n = olga.post("/api/handover", json={"text": "Door alarm on cam1 still open; checked twice.", "hours": 8}).json()
    assert n["author"] == "olga" and n["snapshot"]["counts"]["open_alerts"] == 3
    notes = vic.get("/api/handover").json()
    assert notes[0]["id"] == n["id"] and notes[0]["text"].startswith("Door alarm")
    assert any(e["action"] == "handover_note" and e["actor"] == "olga" for e in aud.get("/api/audit").json())
    r = vic.get("/api/handover/print")
    assert r.status_code == 200 and "Door alarm on cam1" in r.text and verify_report(r.text) is True
    assert "Unresolved at shift end" in r.text
