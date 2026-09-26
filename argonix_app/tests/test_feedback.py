import io
import zipfile

from fastapi.testclient import TestClient

from argonix.feedback import precision_by_rule, verify_bundle, wilson
from argonix.server import create_app
from tests.test_server import make_site


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0


def test_precision_by_rule_excludes_duplicates():
    mk = lambda rule, disp: {"events": [{"rule": rule}], "disposition": disp}
    s = {r["rule"]: r for r in precision_by_rule(
        [mk("intrusion", "true_positive_actioned"), mk("intrusion", "false_positive_rule"),
         mk("intrusion", "duplicate"), mk("intrusion", None), mk("running", "false_positive_detection")])}
    assert s["intrusion"]["alerts"] == 4 and s["intrusion"]["labelled"] == 2 and s["intrusion"]["precision"] == 0.5
    assert s["running"]["precision"] == 0.0 and s["running"]["enough_data"] is False


def test_disposition_workflow_audit_and_export(tmp_path):
    site = make_site(tmp_path)
    site.start(); site.wait(timeout=120); site.stop()
    c = TestClient(create_app(site, open_demo=True))
    aid = c.get("/api/alerts?rule=intrusion").json()[0]["id"]
    H = {"X-Operator": "alice"}
    assert c.post(f"/api/alerts/{aid}/status", json={"status": "resolved", "disposition": "true_positive_actioned"},
                  headers=H).status_code == 409                      # new -> resolved skips acknowledgement
    assert c.post(f"/api/alerts/{aid}/status", json={"status": "acknowledged"}, headers=H).status_code == 200
    assert c.post(f"/api/alerts/{aid}/status", json={"status": "resolved"}, headers=H).status_code == 400
    assert c.post(f"/api/alerts/{aid}/status", json={"status": "resolved", "disposition": "bogus"}, headers=H).status_code == 400
    c.post(f"/api/alerts/{aid}/viewed", headers=H)
    r = c.post(f"/api/alerts/{aid}/status", json={"status": "resolved", "disposition": "true_positive_actioned",
                                                   "note": "guard sent"}, headers=H)
    assert r.status_code == 200 and r.json()["disposition"] == "true_positive_actioned"
    audit = c.get(f"/api/alerts/{aid}/audit").json()
    assert [a["action"] for a in audit][:2] == ["status:resolved", "evidence_viewed"] and audit[0]["actor"].startswith("alice")
    stats = {s["rule"]: s for s in c.get("/api/feedback/stats").json()["rules"]}
    assert stats["intrusion"]["tp"] == 1

    z = c.get(f"/api/alerts/{aid}/export", headers=H)
    assert z.headers["content-type"] == "application/zip" and verify_bundle(z.content)
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert {"alert.json", "audit.json", "manifest.json"} <= set(names) and any(n.endswith(".jpg") for n in names)
    # tampering with any file breaks the chain
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(z.content)) as src, zipfile.ZipFile(buf, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, b"tampered" if n == "alert.json" else src.read(n))
    assert not verify_bundle(buf.getvalue())
