"""Access-control sensor stream: signature checks, mapping to events, fusion with video, denial bursts, offline."""
import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from argonix.auth import Auth
from argonix.correlate import Correlator
from argonix.incidents import webhook_headers
from argonix.models import Event
from argonix.pipeline import Site
from argonix.rules import SEVERITY
from argonix.sensors import IngestError, SensorHub, verify_signature
from argonix.server import create_app
from tests.test_incidents_api import fast_hash
from tests.test_server import ROOT, FakeBackend

SECRET = "whsec_c2Vuc29yLXRlc3Qtc2VjcmV0LTMyLWJ5dGVzLWxvbmch"  # test fixture only
ENV = "ARGONIX_SENSOR_SECRET_TEST"
T0 = 1_790_000_000.0
SOURCES = [{"id": "door-ctrl-1", "secret_env": ENV, "heartbeat_s": 30,
            "doors": {"D1": {"name": "Server room door", "camera": "cam1", "zone": "server_door"},
                      "D2": {"site_zone": "lobby"}}}]


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def signed(body: dict, msg_id="msg-1", ts=None, secret=SECRET):
    raw = json.dumps(body)
    return webhook_headers(secret, msg_id, int(T0 if ts is None else ts), raw), raw.encode()


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setenv(ENV, SECRET)
    clock = Clock()
    emitted = []
    h = SensorHub({"sources": SOURCES}, emit=lambda e: emitted.append(e) or {"id": f"A{len(emitted)}"}, clock=clock,
                  zone_lookup=lambda cam, z: {"name": "Server room door", "site_zone": "server_door", "criticality": 3})
    h.emitted, h.clk = emitted, clock
    return h


def test_signature_good_bad_skew_replay(hub):
    body = {"source_id": "door-ctrl-1", "type": "door_forced", "ts": T0, "door": "D1"}
    hdr, raw = signed(body)
    assert verify_signature(SECRET, hdr, raw, T0) == "msg-1"
    assert hub.ingest_signed(hdr, raw)["rule"] == "door_forced"
    with pytest.raises(IngestError) as e:                        # same webhook-id again: replay
        hub.ingest_signed(hdr, raw)
    assert e.value.status == 409
    bad_hdr, _ = signed(body, "msg-2", secret="whsec_" + "QUJD" * 8)
    with pytest.raises(IngestError) as e:                        # wrong key
        hub.ingest_signed(bad_hdr, raw)
    assert e.value.status == 401 and "mismatch" in str(e.value)
    hdr3, raw3 = signed(body, "msg-3")
    with pytest.raises(IngestError):                             # body changed after signing
        hub.ingest_signed(hdr3, raw3.replace(b"D1", b"D2"))
    old_hdr, old_raw = signed(body, "msg-4", ts=T0 - 301)
    with pytest.raises(IngestError) as e:                        # 5 min + 1 s old: rejected even though signed
        hub.ingest_signed(old_hdr, old_raw)
    assert e.value.status == 401 and "server time" in str(e.value)
    with pytest.raises(IngestError):                             # missing headers
        hub.ingest_signed({}, raw)
    st = hub.status()[0]
    assert st["rejected"] == 5 and st["counts"] == {"door_forced": 1} and len(hub.emitted) == 1


def test_mapping_is_factual_and_strict(hub):
    iso = datetime.fromtimestamp(T0 - 10, timezone.utc).isoformat().replace("+00:00", "Z")
    hdr, raw = signed({"source_id": "door-ctrl-1", "type": "door_held_open", "ts": iso, "door": "D1",
                       "card_id": "0412345678", "details": {"held_s": 45}})
    out = hub.ingest_signed(hdr, raw)
    e = hub.emitted[-1]
    assert out["status"] == "accepted" and e.rule == "door_held_open" and e.camera_id == "sensor:door-ctrl-1"
    assert e.observation == "Door D1 (Server room door) reported held open for 45 s (door controller door-ctrl-1)"
    assert e.site_zone == "server_door" and e.zone_name == "Server room door" and e.criticality == 3
    assert e.derivation["modality"] == "access_control" and not e.derivation.get("learned_model")
    assert e.severity == SEVERITY["door_held_open"] and e.t_end == T0 - 10 and e.derivation["clock"] == "panel"
    assert "0412345678" not in json.dumps(e.to_dict())           # raw card id never kept

    def post(body, n=[0]):
        n[0] += 1
        h, r = signed(body, f"v{n[0]}", ts=hub.clk.t)
        return hub.ingest_signed(h, r)
    base = {"source_id": "door-ctrl-1", "door": "D1", "type": "door_forced"}
    for bad in ({**base, "type": "door_exploded"}, {**base, "door": "D9"}, {**base, "extra": 1},
                {**base, "ts": hub.clk.t + 3600}, {**base, "ts": "2026-09-22T09:46:40"},
                {**base, "details": {"badge_no": "123"}}, {**base, "details": {"x": {"nested": 1}}},
                {"source_id": "door-ctrl-1", "type": "door_forced"}):
        with pytest.raises(IngestError) as e:
            post(bad)
        assert e.value.status == 422, bad
    assert post({"source_id": "door-ctrl-1", "type": "access_granted", "door": "D2"})["status"] == "recorded"
    assert post({"source_id": "door-ctrl-1", "type": "heartbeat"})["rule"] is None


def test_access_denied_burst(hub):
    def deny(t, card, i):
        hub.clk.t = t
        h, r = signed({"source_id": "door-ctrl-1", "type": "access_denied", "door": "D2", "card_id": card, "ts": t},
                      f"d{i}", ts=t)
        return hub.ingest_signed(h, r)
    assert deny(T0, "A", 0)["rule"] is None
    assert deny(T0 + 70, "B", 1)["rule"] is None                  # first one fell out of the 60 s window
    assert deny(T0 + 80, "B", 2)["rule"] is None
    out = deny(T0 + 90, "C", 3)
    assert out["rule"] == "access_denied_burst"
    e = hub.emitted[-1]
    assert e.t_start == T0 + 70 and e.t_end == T0 + 90 and e.site_zone == "lobby"
    assert e.observation.startswith("3 access denials at door D2 within 20 s") and "2 distinct credential" in e.observation
    assert all(c.startswith("card:") for c in e.derivation["card_hashes"])
    assert deny(T0 + 95, "C", 4)["rule"] is None                  # counting restarts after a burst


def test_sensor_offline_and_recovery(hub):
    assert hub.check(T0 + 29) == [] and hub.health("door-ctrl-1") == "WARMUP"
    (e,) = hub.check(T0 + 31)
    assert e.rule == "sensor_offline" and e.camera_id == "sensor:door-ctrl-1" and "31 s" in e.observation
    assert hub.check(T0 + 60) == [] and hub.health("door-ctrl-1") == "DOWN"   # once per outage
    hub.clk.t = T0 + 61
    h, r = signed({"source_id": "door-ctrl-1", "type": "heartbeat"}, "hb", ts=hub.clk.t)
    hub.ingest_signed(h, r)
    st = hub.status()[0]
    assert st["state"] == "ok" and st["recovered_at"] == T0 + 61 and hub.health("door-ctrl-1") == "OK"
    assert hub.check(T0 + 80) == [] and len(hub.check(T0 + 92)) == 1


def video(rule, t, cam="cam1", zone="server_door", site_zone="server_door"):
    return Event(rule, cam, zone, "Server room door", [7], t - 1, t, f"person track #7 ({cam}) inside zone", {"rule": rule},
                 0.9, SEVERITY[rule], 3, site_zone, None, {"snapshot": None, "clip": None})


def test_correlates_with_video_sensor_plus_video(hub):
    corr = Correlator()
    h, r = signed({"source_id": "door-ctrl-1", "type": "door_forced", "ts": T0, "door": "D1"})
    hub.ingest_signed(h, r)
    a1, new1 = corr.add(hub.emitted[-1])
    a2, new2 = corr.add(video("intrusion", T0 + 3))
    assert new1 and not new2 and a1 is a2
    f = a2.factors
    assert f["modalities"] == ["access_control", "video"] and f["cross_modal"] == "sensor + video"
    assert f["distinct_cameras"] == 1 and f["distinct_rules"] == 2 and f["corroboration_multiplier"] == 1.5
    assert a2.priority == "P1" and "Door reported forced open" in a2.title


def build_site(tmp_path, monkeypatch, users=None):
    monkeypatch.setenv(ENV, SECRET)
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = False
    cfg["sensors"] = {"sources": SOURCES}
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())
    auth = None
    if users:
        (tmp_path / "users.json").write_text(json.dumps(users))
        auth = Auth(tmp_path / "users.json", tmp_path / "secret.key")
    return site, create_app(site, auth=auth, open_demo=users is None)


def test_api_one_incident_two_modalities(tmp_path, monkeypatch):
    site, app = build_site(tmp_path, monkeypatch)
    c = TestClient(app)
    now = time.time()
    body = json.dumps({"source_id": "door-ctrl-1", "type": "door_forced", "ts": now, "door": "D1"})
    r = c.post("/api/sensors/events", content=body, headers=webhook_headers(SECRET, "api-1", int(now), body))
    assert r.status_code == 202, r.text
    assert c.post("/api/sensors/events", content=body,
                  headers=webhook_headers(SECRET, "api-1", int(now), body)).status_code == 409
    assert c.post("/api/sensors/events", content=body).status_code == 401   # unsigned, no token
    d = site.on_event(video("intrusion", now + 2))
    assert d["id"] == r.json()["alert_id"] and d["incident_id"] == r.json()["incident_id"]
    assert d["factors"]["cross_modal"] == "sensor + video"
    g = c.get(f"/api/incidents/{d['incident_id']}/graph").json()
    kinds = {n["kind"] for n in g["nodes"]}
    types = {e["type"] for e in g["edges"]}
    assert {"sensor", "camera", "event"} <= kinds and {"reported_by", "covered_by", "corroborates"} <= types
    st = c.get("/api/sensors").json()["sources"][0]
    assert st["counts"] == {"door_forced": 1} and st["rejected"] == 1 and st["last_event"]["alert_id"] == d["id"]
    ev = c.get("/api/sensors/events").json()
    assert ev[0]["type"] == "door_forced" and ev[0]["via"] == "hmac"
    # the offline event opens a separate integrity incident titled for the source, not a camera
    site.sensors.check(time.time() + 120)
    inc = [i for i in c.get("/api/incidents").json() if "sensor_offline" in i["rules"]]
    assert inc and inc[0]["kind"] == "integrity" and inc[0]["title"].startswith("Access-control source silent")


def test_api_token_needs_sensor_ingest(tmp_path, monkeypatch):
    users = {name: {**fast_hash("correct-horse-battery"), "roles": [role]} for name, role in
             [("panel", "integration"), ("vic", "viewer")]}
    site, app = build_site(tmp_path, monkeypatch, users)
    auth = app.state.auth
    _, tok = auth.create_token("panel", "door controller", ["integration"])
    _, vtok = auth.create_token("vic", "viewer script", ["viewer"])
    body = {"source_id": "door-ctrl-1", "type": "access_granted", "door": "D1", "card_id": "12345"}
    c = TestClient(app)
    r = c.post("/api/sensors/events", json=body, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 202 and r.json()["status"] == "recorded"
    assert c.post("/api/sensors/events", json=body, headers={"Authorization": f"Bearer {vtok}"}).status_code == 403
    assert c.post("/api/sensors/events", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/api/sensors/events", json={**body, "source_id": "x"},
                  headers={"Authorization": f"Bearer {tok}"}).status_code == 422
    row = site.sensors.recent()[0]
    assert row["via"] == "token:panel" and row["card_hash"].startswith("card:") and "12345" not in json.dumps(row)
