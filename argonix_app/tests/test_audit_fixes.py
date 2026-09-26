"""Regression tests for the 2026-09-25 code audit (binay/audit-fixes), re-applied on current main.

H1 open demo binds locally · H2 camera passwords out of the site config · H3 camera restarts / recorder isolation ·
M1 rule state pruned · M2 unique evidence names · M3 whole-view events kept apart · M5 safe checkpoint loading ·
M8 signed export, missing files listed, paths contained · L2 ordered alert paging · L3 VLM concurrency cap ·
L5 no DTDs from the network · L6 bad timeout is a 400. Each fails on the code before the fix."""
import hashlib
import io
import json
import pickle
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import argonix.threat_install  # noqa: F401  registers object_threat / altercation labels and severities
from argonix import cameras
from argonix.correlate import Correlator
from argonix.feedback import export_bundle, verify_bundle
from argonix.models import Event, TrackObs
from argonix.pipeline import Site
from argonix.rules import RuleEngine, Zone
from argonix.server import create_app
from argonix.store import Store

ROOT = Path(__file__).resolve().parents[1]
DOOR = (633, 200, 673, 302)  # a person box whose foot point is inside cam1's server-door zone (demo site)


def site_cfg(tmp_path, cameras_n=1, **extra) -> Path:
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text(encoding="utf-8"))
    cfg["cameras"] = cfg["cameras"][:cameras_n]
    cfg["record"] = False
    cfg.pop("sensors", None)
    cfg.update(extra)
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class Boxes:
    """Fake backend returning fixed person boxes."""
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self, boxes):
        self.boxes = boxes

    def track(self, frame):
        return list(self.boxes)


def client(site):
    return TestClient(create_app(site, open_demo=True), headers={"X-Argonix-CSRF": "1"})


# ---- H1: open demo stays on this computer unless asked -------------------------------------------------------------
def test_open_demo_binds_locally_by_default():
    sys.path.insert(0, str(ROOT))
    from run import bind_host
    assert bind_host(None, open_now=True) == ("127.0.0.1", "")
    assert bind_host(None, open_now=False) == ("0.0.0.0", "")  # sign-in required: serve the network
    host, warning = bind_host("0.0.0.0", open_now=True)
    assert host == "0.0.0.0" and "full access" in warning
    assert bind_host("127.0.0.1", open_now=True) == ("127.0.0.1", "")


# ---- H2: camera passwords never reach the git-tracked config ---------------------------------------------------------
def test_camera_password_goes_to_ignored_secrets_file(tmp_path):
    p = site_cfg(tmp_path)
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Boxes([]))
    full = "rtsp://admin:s3cret-pass@10.0.0.5:554/stream1"
    site.cfg["cameras"][0]["source"] = full
    site.cfg["cameras"][0]["main_source"] = "rtsp://admin:s3cret-pass@10.0.0.5:554/stream0"
    site.save_config()
    saved = p.read_text(encoding="utf-8")
    assert "s3cret-pass" not in saved and "rtsp://admin:***@10.0.0.5:554/stream1" in saved
    assert "s3cret-pass" not in p.with_name(p.name + ".bak").read_text(encoding="utf-8") if p.with_name(
        p.name + ".bak").exists() else True
    sec = json.loads((p.parent / "s.secrets.json").read_text(encoding="utf-8"))
    assert sec["cam1"] == {"source": full, "main_source": "rtsp://admin:s3cret-pass@10.0.0.5:554/stream0"}
    again = Site(p, data_dir=tmp_path / "run2", realtime=False, backend_factory=lambda c: Boxes([]))
    assert again.camera_config("cam1")["source"] == full  # restored on load
    # the in-memory config still has the real URL, so the camera keeps working after a save
    assert site.camera_config("cam1")["source"] == full
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "configs/*.secrets.json" in ignored and "configs/*.bak" in ignored


def test_env_placeholders_are_not_treated_as_secrets(tmp_path):
    p = site_cfg(tmp_path)
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Boxes([]))
    site.cfg["cameras"][0]["source"] = "rtsp://${RTSP_USER}:${RTSP_PASSWORD}@${RTSP_HOST}/vtest"
    site.save_config()
    assert "${RTSP_PASSWORD}" in p.read_text(encoding="utf-8")


# ---- H3: a camera survives processing and recording errors -----------------------------------------------------------
class FailsFirst(Boxes):
    def __init__(self, fail_calls):
        super().__init__([(1, DOOR, "person", 0.9)])
        self.calls, self.fail_calls = 0, fail_calls

    def track(self, frame):
        self.calls += 1
        if self.calls <= self.fail_calls:
            raise RuntimeError("GPU batch timed out")
        return super().track(frame)


def test_live_camera_restarts_after_error(tmp_path):
    p = site_cfg(tmp_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["cameras"][0].update(restart_min_s=0.05, restart_max_s=0.2)
    p.write_text(json.dumps(cfg), encoding="utf-8")
    be = FailsFirst(fail_calls=2)
    site = Site(p, data_dir=tmp_path / "run", realtime=True, backend_factory=lambda c: be)
    site.start()
    try:
        w = site.workers[0]
        deadline = time.time() + 30
        while time.time() < deadline and not (w.stats.get("restarts", 0) >= 2 and w.stats.get("frames", 0) > 3):
            time.sleep(0.1)
        assert w.stats.get("restarts", 0) >= 2 and w.stats.get("frames", 0) > 3 and w.is_alive()
    finally:
        site.stop()


def test_recorder_failure_pauses_recording_not_analytics(tmp_path, monkeypatch):
    import argonix.pipeline as pl

    def broken_write(self, frame):
        raise BrokenPipeError("ffmpeg exited")
    monkeypatch.setattr(pl.H264Writer, "write", broken_write)
    p = site_cfg(tmp_path, record=True)
    site = Site(p, data_dir=tmp_path / "run", realtime=False,
                backend_factory=lambda c: Boxes([(1, DOOR, "person", 0.9)]))
    site.start(); site.wait(timeout=180); site.stop()
    w = site.workers[0]
    assert w.stats["frames"] > 50 and w.stats.get("status") != "error"
    assert "BrokenPipeError" in w.stats.get("recording_error", "")
    assert site.store.list_alerts(), "analytics kept producing alerts while recording was broken"


# ---- M1: rule state is pruned --------------------------------------------------------------------------------------------
def test_rule_state_does_not_grow_with_visitors():
    z = Zone.from_dict({"id": "z", "name": "z", "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)],
                        "rules": {"intrusion": {}, "loitering": {"min_seconds": 2}}})
    e, t = RuleEngine("c", [z]), 0.0
    for tid in range(1, 2001):  # each visitor stays 3 s, then leaves forever
        for _ in range(15):
            t += 0.2
            e.update(t, [TrackObs("c", tid, t, (0, 0, 1, 1), "person", 0.9, (5.0, 5.0))])
        for _ in range(20):
            t += 0.2
            e.update(t, [])
    assert len(e.eps) == 0 and len(e.last_fired) <= 60, (len(e.eps), len(e.last_fired))


def test_rearm_still_suppresses_quick_reentry():
    z = Zone.from_dict({"id": "z", "name": "z", "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)],
                        "rules": {"intrusion": {}}})
    e, t, fired = RuleEngine("c", [z]), 0.0, 0
    for _ in range(2):  # the same track leaves for 8 s (longer than stale + grace) and returns within the 30 s re-arm
        for _ in range(10):
            t += 0.2
            fired += len(e.update(t, [TrackObs("c", 7, t, (0, 0, 1, 1), "person", 0.9, (5.0, 5.0))]))
        for _ in range(40):
            t += 0.2
            e.update(t, [])
    assert fired == 1


# ---- M2: one evidence file per event -------------------------------------------------------------------------------------
def test_simultaneous_events_get_distinct_evidence_files(tmp_path):
    two = Boxes([(1, DOOR, "person", 0.9), (2, (650, 205, 690, 305), "person", 0.9)])
    site = Site(site_cfg(tmp_path), data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: two)
    site.start(); site.wait(timeout=180); site.stop()
    evs = [ev for a in site.store.list_alerts() for ev in a["events"] if ev["rule"] == "intrusion"]
    snaps = [ev["evidence"]["snapshot"] for ev in evs]
    assert len(snaps) >= 2 and len(set(snaps)) == len(snaps), snaps
    clips = [ev["evidence"]["clip"] for ev in evs]
    assert len(set(clips)) == len(clips)


# ---- M3: whole-view events do not merge with or boost each other -----------------------------------------------------------
def test_health_and_object_events_stay_separate():
    c = Correlator(15)
    h = Event("camera_health", "cam1", "_view", "Cam 1 (whole view)", [], 100, 100, "view covered", {}, 0.9, 2, 2)
    o = Event("object_threat", "cam1", "_view", "Cam 1 (whole view)", [7], 105, 105, "object near track #7",
              {"learned_model": True}, 0.9, 3, 3)
    a1, _ = c.add(h)
    a2, new = c.add(o)
    assert new and a2.id != a1.id
    assert a2.priority in ("P3", "P4"), "a health event must not lift the learned-only cap"


def test_same_track_still_corroborates():
    c = Correlator(15)
    i = Event("intrusion", "cam1", "server_door", "Server room door", [7], 100, 100, "in zone", {}, 0.9, 3, 3)
    o = Event("object_threat", "cam1", "_view", "Cam 1 (whole view)", [7], 101, 101, "object near track #7",
              {"learned_model": True}, 0.6, 3, 2)
    a1, _ = c.add(i)
    a2, new = c.add(o)
    assert not new and a2.id == a1.id


# ---- M5: checkpoints load without unpickling code ---------------------------------------------------------------------------
class _Boom:
    def __reduce__(self):
        return (print, ("code ran while loading a checkpoint",))


def test_checkpoints_refuse_pickled_code(tmp_path):
    torch = pytest.importorskip("torch")
    from argonix.altercation import GRUScorer
    from argonix.forecast import Forecaster
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {}, "meta": {"x": _Boom()}}, bad)
    with pytest.raises(pickle.UnpicklingError):
        GRUScorer(bad)
    with pytest.raises(pickle.UnpicklingError):
        Forecaster(bad)
    shipped = Forecaster(ROOT / "assets" / "forecast" / "argonix_forecast_gru.pt")  # the real one still loads
    assert shipped.model is not None


def test_altercation_checkpoint_with_json_meta_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from argonix.altercation import FEATURE_VERSION, N_FEAT, GRUScorer, build_gru
    m = build_gru(N_FEAT, 16)
    meta = json.loads(json.dumps({"feature_version": FEATURE_VERSION, "n_feat": N_FEAT, "hidden": 16,
                                  "metrics": {"auc": 0.81}}, default=float))
    torch.save({"state_dict": m.state_dict(), "meta": meta}, tmp_path / "g.pt")
    assert GRUScorer(tmp_path / "g.pt").meta["hidden"] == 16


# ---- M8: export is signed, lists missing files, stays inside the data dir -----------------------------------------------------
def _rechain(blob: bytes, edit: dict) -> bytes:
    z = zipfile.ZipFile(io.BytesIO(blob))
    m = json.loads(z.read("manifest.json"))
    files = {n: z.read(n) for n in z.namelist()}
    files.update(edit)
    prev, entries = "0" * 64, []
    for e in m["entries"]:
        h = hashlib.sha256(files[e["file"]]).hexdigest()
        prev = hashlib.sha256((prev + h + e["file"]).encode()).hexdigest()
        entries.append({**e, "sha256": h, "chain": prev})
    m2 = {**m, "entries": entries, "head": prev}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n, d in files.items():
            out.writestr(n, json.dumps(m2) if n == "manifest.json" else d)
    return buf.getvalue()


def test_export_signed_missing_listed_and_contained(tmp_path):
    data = tmp_path / "run"
    (data / "evidence").mkdir(parents=True)
    (data / "evidence" / "a.jpg").write_bytes(b"jpeg")
    (tmp_path / "outside.txt").write_text("private")
    alert = {"id": "A1", "events": [{"rule": "intrusion", "evidence": {
        "snapshot": "evidence/a.jpg", "clip": "evidence/not_yet.mp4", "crop": "../outside.txt"}}]}
    key = b"k" * 32
    blob = export_bundle(alert, [], data, key=key)
    z = zipfile.ZipFile(io.BytesIO(blob))
    m = json.loads(z.read("manifest.json"))
    assert {x["path"] for x in m["missing"]} == {"evidence/not_yet.mp4", "../outside.txt"}
    assert not any("outside" in n for n in z.namelist())
    assert verify_bundle(blob, key=key) and not verify_bundle(blob, key=b"x" * 32)
    forged = _rechain(blob, {"alert.json": b'{"id": "A1", "edited": true}'})
    assert verify_bundle(forged) and not verify_bundle(forged, key=key)  # the chain alone is fooled; the signature not


def test_verify_endpoint_checks_the_server_signature(tmp_path):
    site = Site(site_cfg(tmp_path), data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Boxes([]))
    c = client(site)
    auth = c.app.state.auth
    alert = {"id": "A9", "priority": "P3", "status": "new", "created_at": 1, "updated_at": 1, "cameras": ["cam1"],
             "events": [{"rule": "intrusion", "evidence": {}}]}
    blob = export_bundle(alert, [], site.data, key=auth.secret)
    r = c.post("/api/evidence/verify", content=blob, headers={"Content-Type": "application/zip"})
    assert r.status_code == 200 and r.json()["chain_ok"] and r.json()["signature_ok"]
    r = c.post("/api/evidence/verify", content=_rechain(blob, {"alert.json": b"{}"}),
               headers={"Content-Type": "application/zip"})
    assert r.json()["chain_ok"] and not r.json()["signature_ok"]
    assert c.post("/api/evidence/verify", content=b"not a zip").status_code == 400


# ---- L2: the live feed pages through a burst oldest-first ------------------------------------------------------------------------
def test_alerts_since_pages_without_gaps(tmp_path):
    st = Store(tmp_path / "a.db")
    for i in range(120):
        st.upsert_alert({"id": f"A{i}", "priority": "P4", "status": "new", "created_at": i, "updated_at": i,
                         "cameras": ["cam1"], "events": [{"rule": "loitering"}]})
    seen, seq = [], 0
    while True:
        page = st.alerts_since(seq, limit=50)
        seen += [a["id"] for a in page]
        if page:
            seq = page[-1]["seq"]
        if len(page) < 50:
            break
    assert seen == [f"A{i}" for i in range(120)]


# ---- L3: a slow VLM can't take every API thread ---------------------------------------------------------------------------------
def test_vlm_calls_are_capped(tmp_path):
    from argonix.vlm import VlmBusy, vlm_slot
    class S:
        cfg = {"vlm_max_concurrent": 1}
    s, held, release = S(), threading.Event(), threading.Event()

    def slow():
        with vlm_slot(s):
            held.set()
            release.wait(5)
    t = threading.Thread(target=slow)
    t.start()
    assert held.wait(5)
    with pytest.raises(VlmBusy):
        with vlm_slot(s, wait_s=0.1):
            pass
    release.set(); t.join()
    with vlm_slot(s, wait_s=0.1):  # free again
        pass


# ---- L5 / L6: untrusted input ------------------------------------------------------------------------------------------------------
def test_onvif_reply_with_a_dtd_is_ignored():
    bomb = ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">]>'
            '<Envelope><Body><ProbeMatches><ProbeMatch><XAddrs>&b;</XAddrs></ProbeMatch></ProbeMatches></Body></Envelope>')
    assert cameras.parse_probe_matches(bomb) == []


def test_camera_test_rejects_bad_timeout(tmp_path):
    site = Site(site_cfg(tmp_path), data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Boxes([]))
    r = client(site).post("/api/cameras/test", json={"source": "rtsp://10.0.0.9/x", "timeout_s": "abc"})
    assert r.status_code == 400


# ---- found by the RTSP end-to-end run on the GPU host: editing a camera on a batched (GPU) site ---------------------------
def test_camera_edit_on_a_batched_site(tmp_path):
    from argonix.inference import BatchedInference
    from tests.test_inference import FakeModel
    svc = BatchedInference(FakeModel(), device="cpu", warmup=False)
    site = Site(site_cfg(tmp_path), data_dir=tmp_path / "run", realtime=True,
                backend_factory=lambda c: svc.client(c["id"]))
    site.start()
    try:
        old = site.worker("cam1")
        new = site.update_camera_config("cam1", {"name": "Renamed"})  # used to fail: "already has a client"
        assert new is site.worker("cam1") and new is not old and new.name == "Renamed" and not old.is_alive()
        with pytest.raises(ValueError):  # an invalid edit still changes nothing and keeps the camera running
            site.update_camera_config("cam1", {"calibration": {"image_points": [[0, 0], [1, 1], [2, 2], [3, 3]]}})
        assert site.worker("cam1") is new and new.is_alive()
    finally:
        site.stop()
        svc.close()


# ---- field test (Arducam): a person the tracker keeps losing re-fires the same rule under new ids -------------------------
def _visits(engine, ids, stay=15, gap=5):
    t, fired = 0.0, []
    for tid in ids:  # the same real person, a new track id after each short loss
        for _ in range(stay):
            t += 0.2
            fired += engine.update(t, [TrackObs("c", tid, t, (0, 0, 1, 1), "person", 0.9, (5.0, 5.0))])
        for _ in range(gap):
            t += 0.2
            engine.update(t, [])
    return fired


def test_zone_rearm_suppresses_id_churn_only_when_configured():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    plain = RuleEngine("c", [Zone.from_dict({"id": "z", "name": "z", "polygon": poly, "rules": {"intrusion": {}}})])
    assert len(_visits(plain, range(1, 6))) == 5  # default unchanged: every new id fires
    z = Zone.from_dict({"id": "z", "name": "z", "polygon": poly, "rules": {"intrusion": {"zone_rearm_s": 60}}})
    quiet = RuleEngine("c", [z])
    assert len(_visits(quiet, range(1, 6))) == 1  # ~20 s of churn: one event


def test_zone_rearm_is_accepted_by_the_zone_api():
    from argonix.api_cameras import validate_zones
    zones, _ = validate_zones([{"id": "door", "name": "Door", "polygon_px": [[0, 0], [100, 0], [100, 100]],
                                "rules": {"loitering": {"min_seconds": 60, "zone_rearm_s": 120}}}], calibrated=False)
    assert zones[0]["rules"]["loitering"]["zone_rearm_s"] == 120


# ---- run.py --infer-node: the remote detector is a per-run override, never written to the site config ----------------
def test_detector_overrides_reach_the_factory_but_not_the_config(tmp_path, monkeypatch):
    import argonix.pipeline as pl
    seen = {}

    def fake_factory(det, models_dir=None):
        seen.update(det)
        return lambda c: Boxes([])
    monkeypatch.setattr(pl, "make_backend_factory", fake_factory)
    p = site_cfg(tmp_path)
    remote = {"url": "ws://127.0.0.1:8200", "token_file": "data/infer_token", "fallback": "local"}
    site = Site(p, data_dir=tmp_path / "run", realtime=False, detector_overrides={"remote": remote})
    assert seen["remote"] == remote and seen["weights"]  # merged into the configured detector
    site.save_config()
    assert "remote" not in json.loads(p.read_text(encoding="utf-8"))["detector"]
