"""Evidence integrity, retention, legal hold, tamper-evident audit log and anonymised export."""
import base64
import io
import json
import os
import sqlite3
import time
import zipfile

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from argonix import privacy
from argonix.feedback import DERIVED, export_bundle, verify_bundle
from argonix.retention import DAY, sweep
from argonix.server import create_app
from argonix.store import Store
from tests.test_server import make_site


@pytest.fixture(scope="module")
def ran_site(tmp_path_factory):
    site = make_site(tmp_path_factory.mktemp("ev"))
    site.start(); site.wait(timeout=120); site.stop()
    return site


def _img(path):
    return cv2.imread(str(path))


def test_raw_and_annotated_evidence_saved_and_exported(ran_site):
    site = ran_site
    a = site.store.list_alerts(rule="intrusion")[0]
    ev = next(e for e in a["events"] if e["rule"] == "intrusion")["evidence"]
    assert ev["snapshot_raw"].endswith("_raw.jpg") and "software overlay" in ev["overlay"]
    ann, raw = _img(site.data / ev["snapshot"]), _img(site.data / ev["snapshot_raw"])
    assert ann.shape == raw.shape  # same resolution
    # the fake detection box (green/red rectangle) is burned into the annotated one only
    x1, y1, x2, y2 = (int(v) for v in ev["person_boxes"][0])
    assert np.abs(ann[y1:y2 + 1, x1].astype(int) - raw[y1:y2 + 1, x1].astype(int)).mean() > 20
    # the "software overlay" label sits bottom-left of the annotated snapshot only
    h = ann.shape[0]
    assert np.abs(ann[h - 12:h, 0:100].astype(int) - raw[h - 12:h, 0:100].astype(int)).mean() > 20

    c = TestClient(create_app(site, open_demo=True))
    assert c.get("/media/" + ev["snapshot_raw"]).status_code == 200
    z = c.get(f"/api/alerts/{a['id']}/export")
    assert z.status_code == 200 and verify_bundle(z.content)
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    m = json.loads(zf.read("manifest.json"))
    kinds = {e["file"]: e["kind"] for e in m["entries"]}
    assert kinds[f"evidence/{os.path.basename(ev['snapshot_raw'])}"] == "original frame (no overlay)"
    assert kinds[f"evidence/{os.path.basename(ev['snapshot'])}"] == "annotated snapshot (software overlay)"
    # the audit chain head at export time is in the bundle and matches the log (the export itself was audited)
    assert m["audit_chain"]["hash"] == site.store.audit_head()["hash"]
    assert site.store.last_audit("evidence_exported")["alert_id"] == a["id"]
    # an extra, unlisted file is rejected
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for n in zf.namelist():
            dst.writestr(n, zf.read(n))
        dst.writestr("evidence/extra.jpg", b"x")
    assert not verify_bundle(buf.getvalue())


def test_legal_hold_api_audit_and_retention_status(ran_site):
    site = ran_site
    c = TestClient(create_app(site, open_demo=True))
    aid = site.store.list_alerts()[0]["id"]
    r = c.post(f"/api/alerts/{aid}/hold", json={"reason": "requested by legal"}, headers={"X-Operator": "sue"})
    assert r.status_code == 200 and r.json()["legal_hold"]["reason"] == "requested by legal"
    assert c.get(f"/api/alerts/{aid}").json()["legal_hold"]
    assert c.get("/api/retention").json()["held_alerts"] == 1
    assert c.delete(f"/api/alerts/{aid}/hold").status_code == 200
    assert "legal_hold" not in c.get(f"/api/alerts/{aid}").json()
    acts = [x["action"] for x in c.get(f"/api/alerts/{aid}/audit").json()]
    assert acts[:2] == ["legal_hold_released", "legal_hold_set"]
    assert c.post("/api/alerts/nope/hold", json={}).status_code == 404
    st = c.get("/api/retention").json()
    assert st["policy"]["recordings_days"] == 7 and st["policy"]["max_disk_gb"] == 20 and st["held_alerts"] == 0
    res = c.post("/api/retention/sweep").json()  # fresh files: nothing is old enough
    assert res["evidence"] == 0 and res["recordings"] == 0
    assert c.get("/api/retention").json()["last_sweep"]["t"] == res["t"]
    v = c.get("/api/audit/verify").json()
    assert v["ok"] and v["rows"] > 0 and v["first_bad_seq"] is None


# ---- retention --------------------------------------------------------------------------------------------------
def _file(path, mtime, size=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    os.utime(path, (mtime, mtime))
    return path


def _alert(aid, t, status, evidence, cam="cam1"):
    return {"id": aid, "priority": "P3", "status": status, "created_at": t, "updated_at": t, "cameras": [cam],
            "events": [{"rule": "intrusion", "camera_id": cam, "evidence": evidence}]}


def test_retention_deletes_old_files_but_respects_holds(tmp_path):
    now = time.time()
    old, recent = now - 40 * DAY, now - 3600
    st = Store(tmp_path / "argonix.db")
    ev_held = {"snapshot": "evidence/h.jpg", "snapshot_raw": "evidence/h_raw.jpg", "clip": "evidence/h.mp4"}
    ev_free = {"snapshot": "evidence/f.jpg", "snapshot_raw": "evidence/f_raw.jpg", "clip": "evidence/f.mp4"}
    for rel in [*ev_held.values(), *ev_free.values()]:
        _file(tmp_path / rel, old)
    _file(tmp_path / "evidence" / "new.jpg", recent)
    st.upsert_alert(_alert("held", old, "resolved", ev_held))
    st.upsert_alert(_alert("free", old, "resolved", ev_free))
    st.upsert_alert(_alert("open", old, "new", {}))
    st.set_hold("held", "sue", "case 12")
    # recordings: one old segment overlapping the held alert's window, one old unrelated, one recent
    _file(tmp_path / "recordings/cam1/a.mp4", old); st.add_recording("cam1", "recordings/cam1/a.mp4", old - 30, old + 30)
    _file(tmp_path / "recordings/cam1/b.mp4", old - DAY); st.add_recording("cam1", "recordings/cam1/b.mp4", old - DAY, old - DAY + 60)
    _file(tmp_path / "recordings/cam1/c.mp4", recent); st.add_recording("cam1", "recordings/cam1/c.mp4", recent, recent + 60)

    res = sweep(st, tmp_path, {"recordings_days": 7, "evidence_days": 30, "max_disk_gb": 0, "alerts_days": 20,
                               "min_age_s": 300}, now)
    assert res["evidence"] == 3 and res["recordings"] == 1 and res["alerts"] == 1 and res["kept_on_hold"] == 4
    for rel in ev_held.values():
        assert (tmp_path / rel).exists()
    assert not any((tmp_path / rel).exists() for rel in ev_free.values())
    assert (tmp_path / "evidence/new.jpg").exists()
    assert (tmp_path / "recordings/cam1/a.mp4").exists() and not (tmp_path / "recordings/cam1/b.mp4").exists()
    assert {r["path"] for r in st.list_recordings()} == {"recordings/cam1/a.mp4", "recordings/cam1/c.mp4"}
    assert st.get_alert("free") is None and st.get_alert("held") and st.get_alert("open")  # open alerts are kept

    # disk cap: oldest first, still never the held files, never files younger than min_age_s
    _file(tmp_path / "evidence/x1.jpg", now - 5 * DAY, 4000)
    _file(tmp_path / "evidence/x2.jpg", now - 4 * DAY, 4000)
    _file(tmp_path / "evidence/busy.mp4", now - 10, 4000)
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file() and p.suffix in {".jpg", ".mp4"})
    res = sweep(st, tmp_path, {"recordings_days": 0, "evidence_days": 0, "alerts_days": 0, "min_age_s": 300,
                               "max_disk_gb": (total - 5000) / 1e9}, now)
    assert res["disk_bytes"] <= total - 5000
    assert not (tmp_path / "evidence/x1.jpg").exists() and (tmp_path / "evidence/busy.mp4").exists()
    for rel in ev_held.values():
        assert (tmp_path / rel).exists()
    # once the hold is released the next age sweep may delete it
    st.release_hold("held")
    res = sweep(st, tmp_path, {"recordings_days": 7, "evidence_days": 30, "alerts_days": 0, "min_age_s": 300}, now)
    assert not (tmp_path / ev_held["snapshot"]).exists() and res["kept_on_hold"] == 0


def test_retention_job_audits_each_sweep(ran_site):
    res = ran_site.retention.run_once()
    last = ran_site.store.last_audit("retention_swept")
    assert last["actor"] == "system" and last["details"]["bytes"] == res["bytes"] and "policy" in last["details"]


# ---- tamper-evident audit log -----------------------------------------------------------------------------------
def test_audit_chain_detects_tampering(tmp_path):
    st = Store(tmp_path / "a.db")
    for i in range(5):
        st.audit("alice", f"act{i}", "A1", {"i": i})
    v = st.verify_audit()
    assert v == {"ok": True, "rows": 5, "first_bad_seq": None, "head": st.audit_head()["hash"]}
    st.db.execute("UPDATE audit SET details=? WHERE id=3", (json.dumps({"i": 99}),)); st.db.commit()
    assert st.verify_audit()["first_bad_seq"] == 3 and not st.verify_audit()["ok"]
    st.db.execute("UPDATE audit SET details=? WHERE id=3", (json.dumps({"i": 2}),)); st.db.commit()
    assert st.verify_audit()["ok"]
    st.db.execute("DELETE FROM audit WHERE id=2"); st.db.commit()  # a removed row breaks the next link
    assert st.verify_audit()["first_bad_seq"] == 3
    # also detected after a restart (the head is re-read, the broken middle is not re-chained)
    st.db.close()
    assert Store(tmp_path / "a.db").verify_audit()["first_bad_seq"] == 3


def test_audit_chain_migrates_old_databases(tmp_path):
    db = sqlite3.connect(str(tmp_path / "old.db"))
    db.execute("CREATE TABLE audit(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, actor TEXT, action TEXT, "
               "alert_id TEXT, details TEXT)")
    for i in range(3):
        db.execute("INSERT INTO audit(t,actor,action,alert_id,details) VALUES(?,?,?,?,?)",
                   (1000.0 + i, "bob", f"old{i}", "", "{}"))
    db.commit(); db.close()
    st = Store(tmp_path / "old.db")
    assert st.audit_chained_on_start == 3 and st.verify_audit()["ok"]
    st.audit("bob", "new", "", {})
    assert st.verify_audit() == {"ok": True, "rows": 4, "first_bad_seq": None, "head": st.audit_head()["hash"]}
    assert [a["seq"] for a in st.list_audit()] == [4, 3, 2, 1]
    assert Store(tmp_path / "old.db").audit_chained_on_start == 0  # chained once


def test_audit_verify_api_reports_break(ran_site):
    c = TestClient(create_app(ran_site, open_demo=True))
    assert c.get("/api/audit/verify").json()["ok"]
    with ran_site.store.lock:
        ran_site.store.db.execute("UPDATE audit SET actor='mallory' WHERE id=1"); ran_site.store.db.commit()
    v = c.get("/api/audit/verify").json()
    assert v == {"ok": False, "rows": v["rows"], "first_bad_seq": 1, "head": None}


# ---- anonymised export and VLM privacy --------------------------------------------------------------------------
def _scene(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)  # texture: blurring measurably lowers local detail
    (tmp_path / "evidence").mkdir(exist_ok=True)
    cv2.imwrite(str(tmp_path / "evidence/s_raw.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(str(tmp_path / "evidence/s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    ev = {"snapshot": "evidence/s.jpg", "snapshot_raw": "evidence/s_raw.jpg", "person_boxes": [[100, 40, 160, 200]]}
    return img, {"id": "A9", "events": [{"rule": "intrusion", "camera_id": "cam1", "evidence": ev}]}


def _detail(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_anonymised_export_contains_blurred_copies(tmp_path, monkeypatch):
    img, alert = _scene(tmp_path)

    class FakeFaces:  # stands in for Haar/YuNet (OpenCV 5 wheels ship no Haar cascades)
        method, available = "fake", True

        def detect(self, im):
            return [(220, 150, 40, 40)]
    monkeypatch.setattr(privacy, "get_detector", lambda model_path=None: FakeFaces())
    data = export_bundle(alert, [], tmp_path, anonymise=True)
    assert verify_bundle(data)
    zf = zipfile.ZipFile(io.BytesIO(data))
    m = json.loads(zf.read("manifest.json"))
    derived = [e for e in m["entries"] if e["kind"] == DERIVED]
    assert {e["derived_from"] for e in derived} == {"evidence/s.jpg", "evidence/s_raw.jpg"}
    assert m["anonymisation"]["face_detector"] == "fake"
    orig = zf.read("evidence/s_raw.jpg")  # originals unchanged in the bundle
    assert orig == (tmp_path / "evidence/s_raw.jpg").read_bytes()
    anon = cv2.imdecode(np.frombuffer(zf.read("anonymised/s_raw.jpg"), np.uint8), cv2.IMREAD_COLOR)
    head = (slice(40, 75), slice(100, 160))    # top 22 % of the person box
    face = (slice(150, 190), slice(220, 260))  # the fake face detection
    body = (slice(120, 200), slice(20, 80))    # untouched
    assert _detail(anon[head]) < 0.2 * _detail(img[head]) and _detail(anon[face]) < 0.2 * _detail(img[face])
    assert _detail(anon[body]) > 0.5 * _detail(img[body])
    info = next(e for e in derived if e["derived_from"] == "evidence/s_raw.jpg")["anonymisation"]
    assert info == {"faces": 1, "face_detector": "fake", "person_heads": 1}

    # without any face detector: head regions are still blurred and the manifest says what was not covered
    none = privacy.FaceDetector.__new__(privacy.FaceDetector)
    none.method, none._haar, none._yunet = "none", None, None
    monkeypatch.setattr(privacy, "get_detector", lambda model_path=None: none)
    m2 = json.loads(zipfile.ZipFile(io.BytesIO(export_bundle(alert, [], tmp_path, anonymise=True))).read("manifest.json"))
    assert m2["anonymisation"]["notes"] and any(e["kind"] == DERIVED for e in m2["entries"])
    # not requested -> no derived copies
    m3 = json.loads(zipfile.ZipFile(io.BytesIO(export_bundle(alert, [], tmp_path))).read("manifest.json"))
    assert not any(e["kind"] == DERIVED for e in m3["entries"]) and "anonymisation" not in m3


def test_export_api_anonymise_flag(ran_site):
    c = TestClient(create_app(ran_site, open_demo=True))
    aid = ran_site.store.list_alerts(rule="intrusion")[0]["id"]
    z = c.get(f"/api/alerts/{aid}/export?anonymise=1")
    assert z.status_code == 200 and "_anonymised.zip" in z.headers["content-disposition"] and verify_bundle(z.content)
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert any(n.startswith("anonymised/") for n in names)
    assert ran_site.store.last_audit("evidence_exported")["details"] == {"anonymised_copies": True}


def test_vlm_frames_are_blurred_when_configured(tmp_path, monkeypatch):
    from argonix.vlm import RemoteExplainer
    img, alert = _scene(tmp_path)
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return httpx.Response(200, json={"text": "ok", "model": "m", "latency_s": 1.0}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200, json={}, request=httpx.Request("GET", url)))
    out = RemoteExplainer("http://x:8100")(alert, tmp_path)
    assert base64.b64decode(sent["image_b64"]) == (tmp_path / "evidence/s.jpg").read_bytes() and "blurred" not in out
    sent.clear()
    out = RemoteExplainer("http://x:8100", privacy={"blur_faces_for_vlm": True})(alert, tmp_path)
    got = cv2.imdecode(np.frombuffer(base64.b64decode(sent["image_b64"]), np.uint8), cv2.IMREAD_COLOR)
    head = (slice(40, 75), slice(100, 160))
    assert _detail(got[head]) < 0.2 * _detail(img[head]) and out.endswith("faces blurred before sending]")
