"""Demo scenarios, phone push (Telegram / ntfy), privacy masks and the heatmap time-lapse."""
import json
import math
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from argonix.api_demo import demo_router
from argonix.feedback import precision_by_rule
from argonix.models import Event
from argonix.notify import Notifier
from argonix.pipeline import Site
from argonix.scenario import ScenarioEngine, validate
from argonix.server import create_app

from .test_server import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "configs" / "scenarios"
TOKEN = "123456:SECRET-token-value"
PRIVACY = [[20, 20], [200, 20], [200, 120], [20, 120]]  # top-left of cam1, away from zones and the fake track


def make_site(tmp_path, record=False, **extra):
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["record"] = record
    cfg["demo"] = {"scenarios_dir": str(SCEN)}
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    cam_extra = extra.pop("camera", {})
    cfg["cameras"][0].update(cam_extra)
    cfg.update(extra)
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: FakeBackend())


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def real_event(rule="intrusion", zone="server_door", t=None):
    t = time.time() if t is None else t
    return Event(rule, "cam1", zone, "Server room door", [7], t, t, "Person track #7 inside restricted zone",
                 {"rule": rule, "threshold": {"min_seconds": 1}}, 0.9, 3, 3, "server_door")


# ---- scenarios -----------------------------------------------------------------------------------------------------
def test_scenario_files_validate():
    files = sorted(SCEN.glob("*.json"))
    assert files
    for f in files:
        sc = validate(json.loads(f.read_text()), f.stem)
        assert sc["duration_s"] > 0 and sc["steps"][0]["at"] == 0
    with pytest.raises(ValueError):
        validate({"name": "x", "steps": [{"at": 0, "action": "explode"}]})
    with pytest.raises(ValueError):
        validate({"name": "x", "steps": [{"at": 0, "action": "tamper", "fault": "melt"}]})


def test_scenario_runs_demo_events_through_the_real_alert_path(tmp_path):
    site = make_site(tmp_path)
    w = site.workers[0]
    faults = []
    orig_inject = w.inject
    w.inject = lambda f, s: (faults.append((f, s)), orig_inject(f, s))
    eng = ScenarioEngine(site, SCEN, speed=100)
    names = [s["name"] for s in eng.scenarios()]
    assert "intrusion_door_tamper" in names
    eng.start("intrusion_door_tamper", "tester")
    with pytest.raises(RuntimeError):
        eng.start("intrusion_door_tamper", "tester")  # one at a time
    assert wait_for(lambda: not eng.running)
    st = eng.status()["run"]
    assert st["state"] == "done" and st["progress"] == 1.0
    assert [s["state"] for s in st["steps"]] == ["done"] * 4, st["steps"]
    assert faults == [("cover", 15.0)] and w.fault is None  # injected, then recovered
    assert w.demo_tag and w.demo_tag[0] == "intrusion_door_tamper"

    alerts = site.store.list_alerts()
    assert len(alerts) == 1, [a["title"] for a in alerts]  # intrusion + door report correlate into one alert
    a = alerts[0]
    assert a["demo"] and a["title"].startswith("[DEMO] ") and a["priority"] == "P1"
    assert {e["rule"] for e in a["events"]} == {"intrusion", "door_forced"}
    assert all(e["observation"].startswith("[DEMO]") and e["derivation"]["demo_scenario"] for e in a["events"])
    assert st["alerts"] == [a["id"]]
    inc = site.store.get_incident(a["incident_id"])
    assert inc["demo"] and inc["title"].startswith("[DEMO] ")
    # the simulated door report (no sensor module) is a readable door node covered by the scenario's camera
    g = site.incidents.graph(a["incident_id"])
    sensors = [n for n in g["nodes"] if n["kind"] == "sensor"]
    assert len(sensors) == 1 and sensors[0]["label"].startswith("door near "), sensors
    assert {"source": sensors[0]["id"], "target": "cam:cam1"} in [{"source": e["source"], "target": e["target"]}
                                                                  for e in g["edges"] if e["type"] == "covered_by"]
    # excluded from precision statistics and the event-rate baseline
    site.store.set_status(a["id"], "resolved", "", "true_positive_actioned", "tester")
    assert precision_by_rule(site.store.list_alerts()) == []
    assert ("cam1", "server_door") not in site.baseline.rate_acc


def test_demo_events_never_merge_with_observed_ones(tmp_path):
    site = make_site(tmp_path)
    site.on_event(real_event())
    eng = ScenarioEngine(site, SCEN, speed=1000)
    eng.start("intrusion_door_tamper", "tester")
    assert wait_for(lambda: not eng.running)
    alerts = site.store.list_alerts()
    demo = [a for a in alerts if a.get("demo")]
    real = [a for a in alerts if not a.get("demo")]
    assert len(demo) == 1 and len(real) == 1
    assert demo[0]["incident_id"] != real[0]["incident_id"]
    assert not real[0]["title"].startswith("[DEMO]")
    assert precision_by_rule(alerts)[0]["alerts"] == 1  # only the observed alert counts


def test_stopped_scenario_clears_its_camera_fault(tmp_path):
    site = make_site(tmp_path)
    d = tmp_path / "scen"
    d.mkdir()
    (d / "long.json").write_text(json.dumps({"name": "long", "steps": [
        {"at": 0, "action": "tamper", "fault": "freeze", "seconds": 60}, {"at": 50, "action": "recover"}]}))
    eng = ScenarioEngine(site, d)
    eng.start("long", "tester")
    assert wait_for(lambda: site.workers[0].fault is not None)
    eng.stop("tester")
    st = eng.status()["run"]
    assert st["state"] == "stopped" and st["steps"][1]["state"] == "skipped"
    assert site.workers[0].fault is None


def test_health_events_during_a_scripted_tamper_are_labelled_demo(tmp_path):
    site = make_site(tmp_path)
    w = site.workers[0]
    w._pending_health = []
    w._health_step(time.time(), None, [("covered", True)])
    assert not w._pending_health[0].derivation.get("demo_scenario")  # an observed condition stays observed
    w.demo_tag = ("intrusion_door_tamper", time.time() + 30)
    w._pending_health = []
    w._health_step(time.time(), None, [("covered", True)])
    e = w._pending_health[0]
    assert e.derivation["demo_scenario"] and e.derivation["injected_fault"] and e.observation.startswith("[DEMO]")
    w.demo_tag = ("intrusion_door_tamper", time.time() - 1)  # label window over
    w._pending_health = []
    w._health_step(time.time(), None, [("covered", True)])
    assert not w._pending_health[0].derivation.get("demo_scenario")


class P:
    username = "tester"


def router_client(site, open_demo=False, denied=frozenset()):
    def need(perm):
        def dep():
            if perm in denied:
                raise HTTPException(403, f"permission '{perm}' required")
            return P()
        return dep
    app = FastAPI()
    app.include_router(demo_router(site, need, lambda: open_demo))
    return TestClient(app)


def test_demo_api_refuses_unless_enabled_and_needs_settings_edit(tmp_path):
    site = make_site(tmp_path)
    c = router_client(site, open_demo=False)
    body = c.get("/api/demo/scenarios").json()
    assert body["enabled"] is False and body["hint"]
    assert c.post("/api/demo/scenarios/intrusion_door_tamper/run").status_code == 409
    c = router_client(site, open_demo=True, denied={"settings.edit"})
    assert c.post("/api/demo/scenarios/intrusion_door_tamper/run").status_code == 403
    assert c.get("/api/demo/scenarios").status_code == 403
    site.cfg["demo"]["scenarios"] = True  # config switch works without open demo mode
    c = router_client(site, open_demo=False)
    assert c.post("/api/demo/scenarios/nope/run").status_code == 404


def test_demo_api_run_status_stop_and_audit(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    site.scenarios.speed = 200
    lst = client.get("/api/demo/scenarios").json()
    assert lst["enabled"] and any(s["name"] == "intrusion_door_tamper" and len(s["steps"]) == 4
                                  for s in lst["scenarios"])
    r = client.post("/api/demo/scenarios/intrusion_door_tamper/run")
    assert r.status_code == 200 and r.json()["run"]["name"] == "intrusion_door_tamper"
    assert wait_for(lambda: not client.get("/api/demo/status").json()["running"])
    assert client.get("/api/demo/status").json()["run"]["state"] == "done"
    assert client.post("/api/demo/stop").status_code == 200
    actions = [x["action"] for x in client.get("/api/audit").json()]
    assert "demo_scenario_run" in actions
    alerts = client.get("/api/alerts").json()
    assert alerts and all(a["demo"] for a in alerts)
    assert client.get("/api/feedback/stats").json()["rules"] == []


# ---- notifications ----------------------------------------------------------------------------------------------------
class Recorder:
    """httpx mock transport: records requests, answers from a script of status codes (last one repeats)."""

    def __init__(self, codes=(200,), exc=None):
        self.codes, self.exc, self.requests = list(codes), exc, []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        if self.exc:
            raise self.exc(f"connection failed for {request.url}", request=request)
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        body = {"ok": code == 200} if code == 200 else {"ok": False, "description": f"error {code}"}
        return httpx.Response(code, json=body)


def notifier(site, rec, telegram=None, http=None, **kw):
    cfg = {"console_url": "https://argonix.lan:8000",
           "telegram": {"enabled": True, "token_env": "ARGONIX_TG_TOKEN", "chat_id": "987654321", **(telegram or {})},
           "http": http or {}}
    audits = []
    n = Notifier(cfg, audit=lambda *a: audits.append(a), record=site.store.add_notification, data_dir=site.data,
                 camera_names=lambda: {"cam1": "Server wing"}, env={"ARGONIX_TG_TOKEN": TOKEN},
                 transport=httpx.MockTransport(rec), backoff_s=0.001, **kw)
    n.audits = audits
    return n


def test_p1_incident_pushes_to_telegram_with_details(tmp_path):
    site = make_site(tmp_path)
    rec = Recorder()
    n = notifier(site, rec)
    site.incidents.notifier = n
    site.on_event(real_event())  # intrusion in a criticality-3 zone: P1, exempt from the push budget
    assert wait_for(lambda: rec.requests)
    req = rec.requests[0]
    assert req.url.path == f"/bot{TOKEN}/sendMessage"
    body = json.loads(req.content)
    assert body["chat_id"] == "987654321"
    text = body["text"]
    a = site.store.list_alerts()[0]
    for part in ("P1", "Restricted-zone entry", "Server wing (cam1)", "Server room door", a["incident_id"],
                 f"https://argonix.lan:8000/#/incidents/{a['incident_id']}", "Time: "):
        assert part in text, (part, text)
    assert wait_for(lambda: n.audits)
    assert n.audits[0][1] == "notify_sent" and TOKEN not in json.dumps(n.audits)
    rows = site.store.list_notifications(a["incident_id"])
    assert any(r["decision"] == "would_push" and "telegram: queued" in json.dumps(r["details"]) for r in rows)
    assert wait_for(lambda: any(r["decision"] == "telegram_sent" for r in site.store.list_notifications(a["incident_id"])))
    n.close()


def test_min_priority_and_suppressed_pushes_are_not_sent(tmp_path):
    site = make_site(tmp_path)
    rec = Recorder()
    n = notifier(site, rec)
    inc = {"id": "I1-1", "cameras": ["cam1"], "zones": ["Lobby"], "events": [], "rules": ["crowding"]}
    assert n(inc, {"priority": "P2", "alert_id": "A1", "trigger": "new", "t": time.time()}) is None
    n2 = notifier(site, rec, telegram={"min_priority": "P2"})
    assert "queued" in n2(inc, {"priority": "P2", "alert_id": "A1", "trigger": "new", "t": time.time()})
    assert wait_for(lambda: len(rec.requests) == 1)
    # a suppressed decision never reaches the notifier: IncidentManager only calls it on "would_push"
    site.incidents.notifier = n2
    site.incidents._decide({**site.incidents._blank("behaviour", time.time(), "P4"), "cameras": ["cam1"]},
                           time.time(), "new", "A2")
    time.sleep(0.2)
    assert len(rec.requests) == 1
    n.close(); n2.close()


def test_retries_with_backoff_and_no_retry_on_client_error(tmp_path):
    site = make_site(tmp_path)
    rec = Recorder(codes=(500, 429, 200))
    n = notifier(site, rec)
    msg = n.message({"id": "I1-1", "cameras": ["cam1"], "zones": [], "events": [], "rules": ["intrusion"]},
                    {"priority": "P1", "alert_id": "A1", "trigger": "new"})
    res = n.deliver(n.channels[0], msg)
    assert res["ok"] and res["attempts"] == 3 and len(rec.requests) == 3
    rec2 = Recorder(codes=(400,))
    n2 = notifier(site, rec2)
    res = n2.deliver(n2.channels[0], msg)
    assert not res["ok"] and res["attempts"] == 1 and "400" in res["detail"]
    n.close(); n2.close()


def test_network_errors_never_leak_the_token(tmp_path):
    site = make_site(tmp_path)
    rec = Recorder(exc=httpx.ConnectError)
    n = notifier(site, rec, retries=2)
    res = n.test("tester")
    assert not res[0]["ok"] and "ConnectError" in res[0]["detail"]
    assert TOKEN not in json.dumps(res) and TOKEN not in json.dumps(n.audits)
    assert TOKEN not in json.dumps(n.status())
    n.close()


def test_snapshot_and_ntfy_channel(tmp_path):
    site = make_site(tmp_path)
    (site.data / "evidence").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(site.data / "evidence" / "s.jpg"), np.zeros((40, 60, 3), np.uint8))
    rec = Recorder()
    n = notifier(site, rec, telegram={"send_snapshot": True},
                 http={"enabled": True, "url": "https://ntfy.example/secret-topic", "min_priority": "P1"})
    inc = {"id": "I9-1", "cameras": ["cam1"], "zones": ["Server room door"], "rules": ["intrusion"], "demo": True,
           "events": [{"alert_id": "A9", "rule": "intrusion", "camera_id": "cam1", "zone_name": "Server room door",
                       "t_end": time.time(), "evidence": {"snapshot": "evidence/s.jpg"}}]}
    assert n(inc, {"priority": "P1", "alert_id": "A9", "trigger": "new"}) == "telegram: queued; http: queued"
    assert wait_for(lambda: len(rec.requests) == 2)
    tg = next(r for r in rec.requests if r.url.host == "api.telegram.org")
    assert tg.url.path.endswith("/sendPhoto") and b"image/jpeg" in tg.content and b"[DEMO]" in tg.content
    nt = next(r for r in rec.requests if r.url.host == "ntfy.example")
    assert nt.headers["Priority"] == "5" and nt.headers["Click"].endswith("/#/incidents/I9-1")
    assert "Restricted-zone entry" in nt.content.decode() and nt.headers["Title"].startswith("[DEMO] P1")
    assert n.status()["channels"][1]["host"] == "ntfy.example"
    assert "secret-topic" not in json.dumps(n.status())
    n.close()


def test_notify_api_status_and_test_button(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    st = client.get("/api/notify").json()
    assert [c["kind"] for c in st["channels"]] == ["telegram", "http"] and not st["channels"][0]["enabled"]
    assert client.post("/api/notify/test", json={}).status_code == 400  # nothing enabled
    rec = Recorder()
    site.notifier = notifier(site, rec)
    r = client.post("/api/notify/test", json={})
    assert r.status_code == 200 and r.json()["ok"] and len(rec.requests) == 1
    assert "test notification" in json.loads(rec.requests[0].content)["text"]
    assert site.notifier.audits[0][0] == "operator (unauthenticated)" and site.notifier.audits[0][1] == "notify_sent"
    site.notifier.close()


# ---- privacy masks ---------------------------------------------------------------------------------------------------
def region(img):
    x0, y0, x1, y1 = 40, 40, 180, 100  # inside PRIVACY
    return img[y0:y1, x0:x1]


def test_privacy_mask_blanks_pixels_everywhere(tmp_path):
    src = cv2.VideoCapture(str(ROOT / "data" / "synth" / "cam1.mp4"))
    ok, raw = src.read()
    src.release()
    assert ok and np.median(region(raw)) > 20  # the area has content without the mask
    site = make_site(tmp_path, record=True, camera={"privacy_masks_px": [PRIVACY]})
    site.start(); site.wait(timeout=180); site.stop()
    w = site.workers[0]
    snaps = [e["evidence"]["snapshot"] for a in site.store.list_alerts() for e in a["events"]]
    assert snaps
    for rel in snaps[:3]:
        img = cv2.imread(str(site.data / rel))
        assert np.median(region(img)) < 8, rel
    recs = site.store.list_recordings("cam1")
    assert recs
    cap = cv2.VideoCapture(str(site.data / recs[0]["path"]))
    ok, fr = cap.read()
    cap.release()
    assert ok and np.median(region(fr)) < 8
    live = cv2.imdecode(np.frombuffer(w.latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert np.median(region(live)) < 8
    client = TestClient(create_app(site, open_demo=True))
    snap = cv2.imdecode(np.frombuffer(client.get("/api/cameras/cam1/snapshot.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    assert np.median(region(snap)) < 8
    # the rest of the frame is untouched
    assert np.median(snap[300:400, 300:500]) > 20


def test_privacy_mask_api_validation_and_audit(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    bad = client.put("/api/cameras/cam1/privacy_masks", json={"privacy_masks_px": [[[0, 0], [1, 1]]]})
    assert bad.status_code == 400
    assert client.put("/api/cameras/cam1/privacy_masks", json={"privacy_masks_px": "x"}).status_code == 400
    r = client.put("/api/cameras/cam1/privacy_masks", json={"privacy_masks_px": [PRIVACY]})
    assert r.status_code == 200 and r.json()["privacy_masks_px"] == [PRIVACY]
    assert site.workers[0].privacy_masks and site.camera_config("cam1")["privacy_masks_px"] == [PRIVACY]
    assert json.loads(site.config_path.read_text())["cameras"][0]["privacy_masks_px"] == [PRIVACY]
    assert any(x["action"] == "camera_privacy_masks" for x in client.get("/api/audit").json())
    cams = client.get("/api/cameras").json()
    assert cams[0]["privacy_masks_px"] == [PRIVACY] and cams[0]["masks_px"] == []


# ---- heatmap time-lapse ----------------------------------------------------------------------------------------------
def heat_site():
    from argonix.autocalib import pixel_from_world
    from .test_birdseye import CAM_H, F, HH, SCALE, TILT, W, make_site as bsite
    site = bsite(1, uncalibrated=False)
    u, v, _ = pixel_from_world(2.0, 10.0, 0.0, F, math.radians(TILT), CAM_H, W / 2, HH / 2)
    fu, fv = float(u) / SCALE, float(v) / SCALE
    site.workers[0].fixed_dets = [(1, (fu - 10, fv - 60, fu + 10, fv), "person", 0.9)]
    return site


def test_heatmap_history_buckets_persist_and_aggregate(tmp_path):
    from argonix.birdseye import SiteMap
    site = heat_site()
    db = tmp_path / "heat.db"
    smap = SiteMap(site, bucket_s=900, history_path=db)
    T = 1_000_000 * 900.0  # bucket-aligned
    for dt in (10, 11, 12, 13):          # bucket A: 0.5 + 3 x 1 s
        smap.tick(now=T + dt)
    for dt in (1000, 1001, 1002):        # bucket B (next 15 min): dt capped at 5 s, then 1 s, 1 s
        smap.tick(now=T + dt)
    h = smap.history(hours=1, step_s=900, now=T + 1003)
    assert h["step_s"] == 900 and len(h["frames"]) == 4
    by_start = {f["t_start"]: f for f in h["frames"]}
    assert by_start[T]["total"] == pytest.approx(3.5) and by_start[T + 900]["total"] == pytest.approx(7.0)
    idx = by_start[T]["cells"][0][0]
    assert by_start[T]["cells"] == [[idx, 3.5]] and by_start[T + 900]["cells"][0][0] == idx  # same place
    assert h["width"] * h["height"] > idx and h["max"] == pytest.approx(7.0)
    hourly = smap.history(hours=2, step_s=3600, now=T + 1003)
    assert sum(f["total"] for f in hourly["frames"]) == pytest.approx(10.5) and len(hourly["frames"]) == 2
    smap.stop()  # flushes the running bucket
    again = SiteMap(site, bucket_s=900, history_path=db)  # e.g. after a restart
    h2 = again.history(hours=1, step_s=900, now=T + 1003)
    assert [f["total"] for f in h2["frames"]] == [f["total"] for f in h["frames"]]


def test_heatmap_history_api(tmp_path):
    from .test_birdseye import api_client
    site = heat_site()
    c = api_client(site)
    r = c.get("/api/map/heatmap/history?hours=24&step_s=3600")
    assert r.status_code == 200
    h = r.json()
    assert len(h["frames"]) == 24 and h["step_s"] == 3600 and h["cell_m"] == 0.5
    assert all(f["t_end"] - f["t_start"] == 3600 for f in h["frames"])
    assert c.get("/api/map/heatmap/history?hours=0").status_code == 400
    assert c.get("/api/map/heatmap/history?hours=500").status_code == 400
    assert c.get("/api/map/heatmap/history?step_s=5").status_code == 400
