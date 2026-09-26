"""Incident lifecycle, merge/split, reopen, inhibition, push budget, webhook signing, graph and hotspots (CPU, fast)."""
import time

import pytest

from argonix.correlate import Correlator
from argonix.incidents import (EDGE_TYPES, IncidentManager, TokenBucket, WebhookSender, hotspots, sign_webhook,
                               webhook_headers)
from argonix.models import Event
from argonix.rules import SEVERITY
from argonix.store import Store

T0 = 1_790_000_000.0  # fixed epoch for deterministic event time


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def ev(rule, t, cam="cam1", zone="z1", tracks=(1,), crit=1, conf=0.9, site_zone=None):
    return Event(rule, cam, zone, zone.upper(), list(tracks), t - 1, t, f"person track #{tracks[0] if tracks else '-'} "
                 f"{rule} in {zone}", {"rule": rule}, conf, SEVERITY[rule], crit, site_zone, None,
                 {"snapshot": f"evidence/{cam}_{rule}_{int(t)}.jpg", "clip": f"evidence/{cam}_{rule}_{int(t)}.mp4"})


class Rig:
    """Correlator -> IncidentManager -> Store, as Site.on_event wires them."""

    def __init__(self, tmp_path, cfg=None, **kw):
        self.clock = Clock()
        self.store = Store(tmp_path / "a.db")
        self.corr = Correlator()
        self.mgr = IncidentManager(self.store, cfg or {}, clock=self.clock, **kw)

    def feed(self, e):
        self.clock.t = max(self.clock.t, e.t_end)
        a, _ = self.corr.add(e)
        d = a.to_dict()
        self.mgr.on_alert(d)
        self.store.upsert_alert(d)
        return d

    def inc(self, d):
        return self.mgr.get(d["incident_id"])

    def tick(self, t):
        self.clock.t = t
        self.mgr.tick()


def test_lifecycle_hold_active_cooling_closed(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(ev("loitering", T0))                       # severity 1 x crit 1 x 0.9 -> P4: held
    assert d["priority"] == "P4" and d["incident_id"]
    inc = r.inc(d)
    assert inc["phase"] == "open" and inc["pushes"] == []
    r.tick(T0 + 9.9)
    assert r.inc(d)["phase"] == "open"
    r.tick(T0 + 10)
    inc = r.inc(d)
    assert inc["phase"] == "active" and inc["activated_at"] == pytest.approx(T0 + 10)
    assert inc["pushes"][0]["decision"] == "not_push_tier"   # P4 is visible, never pushed
    r.tick(T0 + 59.9)
    assert r.inc(d)["phase"] == "active"
    r.tick(T0 + 60)
    assert r.inc(d)["phase"] == "cooling"
    r.tick(T0 + 299.9)
    assert r.inc(d)["phase"] == "cooling"
    r.tick(T0 + 300)
    inc = r.inc(d)
    assert inc["phase"] == "closed" and inc["closed_by"] == "quiescence" and inc["status"] == "closed"
    kinds = [x["kind"] for x in inc["timeline"]]
    assert kinds.index("opened") < kinds.index("active") < kinds.index("cooling") < kinds.index("closed")


def test_cooling_back_to_active_on_new_event(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(ev("running", T0, crit=2))                 # 2 x 2 x 0.9 = 3.6 -> P2: no hold-down
    assert r.inc(d)["phase"] == "active" and r.inc(d)["pushes"][0]["decision"] == "would_push"
    r.tick(T0 + 70)
    assert r.inc(d)["phase"] == "cooling"
    d2 = r.feed(ev("running", T0 + 80, tracks=(7,), crit=2))  # same zone, new alert (correlator window passed)
    assert d2["id"] != d["id"] and d2["incident_id"] == d["incident_id"]
    inc = r.inc(d2)
    assert inc["phase"] == "active" and len(inc["alert_ids"]) == 2
    assert len(inc["pushes"]) == 1  # an update is not a new push


def test_reopen_within_window_and_new_incident_after(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(ev("loitering", T0))
    r.tick(T0 + 300)
    first = r.inc(d)
    assert first["phase"] == "closed"
    d2 = r.feed(ev("loitering", T0 + 700, tracks=(2,)))    # 400 s after the close: within the 600 s window
    assert d2["incident_id"] == first["id"]
    inc = r.inc(d2)
    assert inc["reopen_count"] == 1 and inc["phase"] == "active" and inc["status"] == "open"
    r.tick(T0 + 700 + 300)
    assert r.inc(d2)["phase"] == "closed"
    d3 = r.feed(ev("loitering", T0 + 1000 + 601, tracks=(3,)))
    assert d3["incident_id"] != first["id"]               # window over: a new incident


def test_operator_status_merge_split(tmp_path):
    r = Rig(tmp_path)
    a = r.feed(ev("loitering", T0, cam="cam1", zone="z1", tracks=(1,)))
    b = r.feed(ev("loitering", T0 + 1, cam="cam2", zone="z2", tracks=(5,)))
    ia, ib = a["incident_id"], b["incident_id"]
    assert ia != ib
    with pytest.raises(ValueError):
        r.mgr.merge(ia, [ia], "sam")
    with pytest.raises(KeyError):
        r.mgr.merge(ia, ["I0-404"], "sam")
    m = r.mgr.merge(ia, [ib], "sam")
    assert set(m["alert_ids"]) == {a["id"], b["id"]} and m["merged_from"] == [ib]
    assert set(m["cameras"]) == {"cam1", "cam2"}
    src = r.mgr.get(ib)
    assert src["merged_into"] == ia and src["phase"] == "closed" and src["closed_by"] == "merge"
    assert r.store.incident_for_alert(b["id"]) == ia and r.store.get_alert(b["id"])["incident_id"] == ia
    b2 = r.feed(ev("loitering", T0 + 5, cam="cam2", zone="z2", tracks=(5,)))  # same alert keeps correlating
    assert b2["id"] == b["id"] and b2["incident_id"] == ia
    with pytest.raises(ValueError):
        r.mgr.split(ia, [a["id"], b["id"]], "sam")          # must leave one alert behind
    with pytest.raises(ValueError):
        r.mgr.split(ia, ["A-nope"], "sam")
    new = r.mgr.split(ia, [b["id"]], "sam")
    assert new["split_from"] == ia and new["alert_ids"] == [b["id"]] and new["cameras"] == ["cam2"]
    assert r.mgr.get(ia)["alert_ids"] == [a["id"]]
    b3 = r.feed(ev("loitering", T0 + 8, cam="cam2", zone="z2", tracks=(5,)))
    assert b3["incident_id"] == new["id"] and r.store.get_alert(b["id"])["incident_id"] == new["id"]
    inc = r.mgr.update(ia, {"status": "acknowledged", "assignee": "olga", "summary": "Checked on camera."}, "sam")
    assert inc["status"] == "acknowledged" and inc["assignee"] == "olga" and inc["summary"]
    inc = r.mgr.update(ia, {"status": "resolved", "note": "fine"}, "sam")
    assert inc["phase"] == "closed" and inc["closed_by"] == "operator" and inc["status"] == "resolved"
    with pytest.raises(ValueError):
        r.mgr.update(ia, {"status": "bogus"}, "sam")
    acts = {x["action"] for x in r.store.list_audit()}
    assert {"incident_merge", "incident_split", "incident_assign", "incident_status:resolved"} <= acts


def test_camera_health_inhibits_rule_alerts_on_that_camera(tmp_path):
    health = {"cam1": "DEGRADED"}
    r = Rig(tmp_path, health=lambda c: health.get(c, "OK"))
    h = r.feed(ev("camera_health", T0, cam="cam1", zone="_view", tracks=(), crit=2))
    hi = r.inc(h)
    assert hi["kind"] == "integrity" and hi["phase"] == "active" and "inhibited_by" not in h
    x = r.feed(ev("intrusion", T0 + 5, cam="cam1", zone="door", crit=3))   # 3 x 3 x 0.9 -> P1
    assert x["inhibited_by"] == hi["id"] and x["suppressed"] and "camera view problem" in x["suppressed_reason"]
    xi = r.inc(x)
    assert xi["inhibited_by"] == hi["id"] and xi["pushes"][-1]["decision"] == "suppressed"
    y = r.feed(ev("intrusion", T0 + 6, cam="cam2", zone="door2", crit=3))
    assert "inhibited_by" not in y and r.inc(y)["pushes"][-1]["decision"] == "would_push"
    r.tick(T0 + 200)                                        # camera still degraded: integrity incident stays active
    assert r.mgr.get(hi["id"])["phase"] == "active"
    health["cam1"] = "OK"
    r.tick(T0 + 261)
    assert r.mgr.get(hi["id"])["phase"] == "cooling"
    r.tick(T0 + 500)
    assert r.mgr.get(hi["id"])["phase"] == "closed"
    z = r.feed(ev("intrusion", T0 + 900, cam="cam1", zone="door9", tracks=(9,), crit=3))
    assert "inhibited_by" not in z                          # source closed: no more inhibition


def test_push_budget_token_bucket_p1_exempt(tmp_path):
    r = Rig(tmp_path)
    decisions = []
    for i in range(5):  # five distinct P2 situations within 5 s
        d = r.feed(ev("running", T0 + i, zone=f"z{i}", tracks=(10 + i,), crit=2))
        decisions.append(r.inc(d)["pushes"][-1]["decision"])
    assert decisions == ["would_push"] * 3 + ["budget_denied"] * 2
    p1 = r.feed(ev("intrusion", T0 + 6, zone="vault", tracks=(50,), crit=3))
    assert p1["priority"] == "P1" and r.inc(p1)["pushes"][-1]["decision"] == "would_push"
    d = r.feed(ev("running", T0 + 4 + 600, zone="z9", tracks=(60,), crit=2))  # one token refills per 10 min
    assert r.inc(d)["pushes"][-1]["decision"] == "would_push"
    rows = r.store.list_notifications()
    assert sum(x["decision"] == "budget_denied" for x in rows) == 2 and all(x["reason"] for x in rows)
    b = TokenBucket(6, 3)
    assert [b.try_take(0) for _ in range(4)] == [True, True, True, False]
    assert b.try_take(600) and not b.try_take(601)


def test_escalation_and_reopen_push_triggers(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(ev("loitering", T0, zone="door", tracks=(1,)))
    r.tick(T0 + 10)
    assert r.inc(d)["pushes"][-1]["decision"] == "not_push_tier"
    d2 = r.feed(ev("intrusion", T0 + 12, zone="door", tracks=(1,), crit=3))  # same alert escalates to P1
    assert d2["id"] == d["id"]
    last = r.inc(d2)["pushes"][-1]
    assert last["trigger"] == "escalated" and last["priority"] == "P1" and last["decision"] == "would_push"


def test_webhook_signature_standard_vector_and_sender():
    # test vector from the Standard Webhooks reference libraries
    sig = sign_webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330,
                       '{"test": 2432232314}')
    assert sig == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="
    seen = []

    class R:
        status_code = 204

    def post(url, content, headers, timeout):
        seen.append((url, content, headers))
        return R()

    results = []
    s = WebhookSender("https://hook.example/x", "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", post=post,
                      on_result=lambda p, st, det: results.append(st))
    assert s.send("m1", {"type": "t", "data": {"incident_id": "I1-1"}}) == ("sent", "HTTP 204")
    url, body, hd = seen[0]
    assert hd["webhook-id"] == "m1" and hd["webhook-signature"] == webhook_headers(
        "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", "m1", int(hd["webhook-timestamp"]), body)["webhook-signature"]
    s.close()


def test_webhook_disabled_by_default_and_queued_when_configured(tmp_path):
    r = Rig(tmp_path)
    assert r.mgr.webhook is None
    sent = []

    class R:
        status_code = 200

    (tmp_path / "b").mkdir()
    r2 = Rig(tmp_path / "b",
             cfg={"webhook": {"url": "https://hook.example/y", "secret": "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"}},
             webhook_post=lambda url, content, headers, timeout: sent.append(content) or R())
    r2.feed(ev("intrusion", T0, zone="vault", crit=3))
    for _ in range(50):
        if any(x["decision"] == "webhook_sent" for x in r2.store.list_notifications()):
            break
        time.sleep(0.02)
    assert sent and '"requires_human_decision":true' in sent[0] and "evidence/" not in sent[0]  # minimal payload
    r2.mgr.close()


def test_graph_shape_with_handoff_and_reid(tmp_path):
    preds = [{"camera": "cam1", "track_id": 1, "state": "leaving", "exit_side": "right", "updated": T0,
              "source": "nn", "kind": "estimate",
              "predictions": [{"camera": "cam2", "leaves_coverage": False, "prob": 0.7, "eta_s": 6.0},
                              {"camera": None, "leaves_coverage": True, "prob": 0.2, "eta_s": None}]}]
    r = Rig(tmp_path, predictions=lambda: preds, reid=lambda cam, tid: {"global_id": "g7", "confidence": 0.61},
            camera_names=lambda: {"cam1": "Server wing", "cam2": "Lobby"})
    r.feed(ev("intrusion", T0, zone="door", tracks=(1,), crit=3))
    d = r.feed(ev("loitering", T0 + 2, zone="door", tracks=(1,)))       # joins the same alert: corroboration
    g = r.mgr.graph(d["incident_id"])
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"event", "camera", "zone", "track", "evidence", "global_track", "coverage"} <= kinds
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    types = {e["type"] for e in g["edges"]}
    assert types <= set(EDGE_TYPES) and {"observed_in", "same_track", "corroborates", "predicted_next",
                                         "evidence"} <= types
    pn = [e for e in g["edges"] if e["type"] == "predicted_next"]
    assert all(e["estimate"] for e in pn) and any(e["target"] == "cam:cam2" and e["prob"] == 0.7 for e in pn)
    cam2 = next(n for n in g["nodes"] if n["id"] == "cam:cam2")
    assert cam2["predicted_only"] and cam2["label"] == "Lobby"
    gid = [e for e in g["edges"] if e["target"] == "gid:g7"]
    assert gid and gid[0]["probabilistic"] and gid[0]["confidence"] == 0.61
    # inhibited events link to the inhibiting incident
    r.feed(ev("camera_health", T0 + 20, cam="cam3", zone="_view", tracks=(), crit=2))
    x = r.feed(ev("intrusion", T0 + 25, cam="cam3", zone="d3", tracks=(4,), crit=3))
    g2 = r.mgr.graph(x["incident_id"])
    inh = [e for e in g2["edges"] if e["type"] == "inhibited_by"]
    assert inh and any(n["kind"] == "incident" for n in g2["nodes"])


def test_warm_start_reloads_live_incidents(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(ev("running", T0, crit=2))
    m2 = IncidentManager(r.store, {}, clock=Clock(T0 + 5))
    assert d["id"] in m2.alert_map and d["incident_id"] in m2.live
    a, _ = r.corr.add(ev("running", T0 + 6, crit=2))
    x = a.to_dict()
    m2.on_alert(x)
    assert x["incident_id"] == d["incident_id"]


def test_hotspots_hour_of_week(tmp_path):
    r = Rig(tmp_path)
    lt, lt2 = time.localtime(T0), time.localtime(T0 + 7 * 86400)
    how = lt.tm_wday * 24 + lt.tm_hour
    if lt2.tm_wday * 24 + lt2.tm_hour != how:
        pytest.skip("DST change between the two sample weeks")
    r.feed(ev("loitering", T0, zone="z1", tracks=(1,)))
    r.feed(ev("loitering", T0 + 1, zone="z2", tracks=(2,)))
    r.tick(T0 + 400)
    r.feed(ev("loitering", T0 + 7 * 86400, zone="z1", tracks=(3,)))  # same slot one week later
    h = hotspots(r.store, T0 + 7 * 86400 + 10, days=28, by="zone")
    top = h["rows"][0]
    assert top["key"] == "cam1:z1" and top["total"] == 2 and top["counts"][how] == 2
    assert h["overall"][how] == 3 and h["top"][0]["how"] == how and h["top"][0]["examples"]
    hc = hotspots(r.store, T0 + 7 * 86400 + 10, days=28, by="camera", metric="alerts")
    assert hc["rows"][0]["key"] == "cam1" and hc["rows"][0]["total"] == 3
