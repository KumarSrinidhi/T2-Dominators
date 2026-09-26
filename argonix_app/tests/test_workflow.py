"""Operator workflow: alert status state machine (legal / illegal transitions, escalation reason, "unclear" second
review, reopen permission), acknowledge SLA escalation on incidents, and incident messages on the WebSocket."""
import pytest
from fastapi.testclient import TestClient

from argonix.auth import ROLES, STATUS_PERM
from argonix.server import create_app
from argonix.workflow import TRANSITIONS, TransitionError, plan, sla_config
from tests.test_incidents import T0, Rig, ev as iev
from tests.test_incidents_api import build, login

ALL = lambda p: True  # noqa: E731 - every permission


def status_of(st, **kw):
    return {"id": "A1", "status": st, **kw}


@pytest.mark.parametrize("cur,to,body,code", [
    ("new", "resolved", {"disposition": "true_positive_actioned"}, 409),   # must be acknowledged first
    ("new", "new", {}, 409),
    ("acknowledged", "new", {}, 409),
    ("escalated", "dismissed", {"disposition": "duplicate"}, 409),
    ("escalated", "escalated", {"reason": "x"}, 409),
    ("resolved", "escalated", {"reason": "x"}, 409),
    ("resolved", "dismissed", {"disposition": "duplicate"}, 409),
    ("new", "dismissed", {}, 400),                                          # closing needs a disposition
    ("new", "escalated", {}, 400),                                          # escalation needs a reason
    ("new", "escalated", {"reason": "r", "to_role": "boss"}, 400),          # unknown role
    ("new", "resolved", {"disposition": "unclear"}, 400),                   # unclear never closes
    ("new", "bogus", {}, 400),
])
def test_illegal_or_incomplete_transitions(cur, to, body, code):
    with pytest.raises(TransitionError) as e:
        plan(status_of(cur), {"status": to, **body}, ALL, STATUS_PERM, ROLES, "olga")
    assert e.value.code == code and e.value.msg


def test_legal_transitions_cover_the_table():
    extra = {"escalated": {"reason": "needs a supervisor"}, "resolved": {"disposition": "true_positive_actioned"},
             "dismissed": {"disposition": "false_positive_rule"}}
    for cur, targets in TRANSITIONS.items():
        for to in targets:
            ch = plan(status_of(cur), {"status": to, **extra.get(to, {})}, ALL, STATUS_PERM, ROLES, "olga")
            assert ch["from"] == cur and ch["to"] == to and ch["fields"]["status"] == to


def test_escalation_unclear_reopen_fields():
    ch = plan(status_of("new"), {"status": "escalated", "reason": "possible forced entry", "to_role": "supervisor",
                                 "to_user": "sam"}, ALL, STATUS_PERM, ROLES, "olga")
    esc = ch["fields"]["escalation"]
    assert esc["reason"] == "possible forced entry" and esc["to_role"] == "supervisor" and esc["by"] == "olga"
    assert ch["fields"]["acknowledged_by"] == "olga" and ch["fields"]["acknowledged_at"]   # first response
    un = plan(status_of("acknowledged"), {"status": "acknowledged", "disposition": "unclear"}, ALL, STATUS_PERM)
    assert un["fields"]["second_review"] is True and un["fields"]["disposition"] == "unclear"
    fin = plan(status_of("acknowledged", second_review=True, disposition="unclear"),
               {"status": "resolved", "disposition": "true_positive_no_action"}, ALL, STATUS_PERM, actor="sam")
    assert fin["fields"]["second_review"] is False and fin["fields"]["second_reviewed_by"] == "sam"
    re = plan(status_of("dismissed", disposition="duplicate"), {"status": "new"}, ALL, STATUS_PERM)
    assert re["fields"]["disposition"] is None and re["fields"]["acknowledged_at"] is None
    assert re["fields"]["reopen_count"] == 1 and re["fields"]["reopened_at"]
    operator = {"alert.acknowledge", "alert.escalate", "alert.resolve", "alert.dismiss"}.__contains__
    with pytest.raises(TransitionError) as e:
        plan(status_of("resolved"), {"status": "acknowledged"}, operator, STATUS_PERM)
    assert e.value.code == 403 and "reopen" in e.value.msg


def test_status_api_state_machine_and_rbac(tmp_path):
    site, app = build(tmp_path)
    olga, sam, ada = login(app, "olga"), login(app, "sam"), login(app, "ada")
    aid = olga.get("/api/alerts").json()[0]["id"]
    url = f"/api/alerts/{aid}/status"
    r = olga.post(url, json={"status": "resolved", "disposition": "true_positive_actioned"})
    assert r.status_code == 409 and "illegal transition new -> resolved" in r.json()["detail"]
    assert olga.post(url, json={"status": "escalated"}).status_code == 400            # no reason
    r = olga.post(url, json={"status": "escalated", "reason": "door forced?", "to_role": "supervisor"})
    assert r.status_code == 200 and r.json()["escalation"]["to_role"] == "supervisor"
    assert r.json()["acknowledged_by"] == "olga"
    assert olga.post(url, json={"status": "escalated", "reason": "again"}).status_code == 409
    assert sam.post(url, json={"status": "acknowledged", "disposition": "unclear"}).status_code == 200
    flagged = olga.get("/api/alerts?review=second").json()
    assert [a["id"] for a in flagged] == [aid] and flagged[0]["status"] == "acknowledged"
    assert olga.post(url, json={"status": "resolved", "disposition": "unclear"}).status_code == 400
    r = olga.post(url, json={"status": "resolved", "disposition": "true_positive_no_action", "note": "checked"})
    assert r.status_code == 200 and r.json()["second_review"] is False
    assert olga.get("/api/alerts?review=second").json() == []
    assert olga.post(url, json={"status": "new"}).status_code == 403                  # operators cannot reopen
    assert ada.post(url, json={"status": "acknowledged"}).status_code == 200          # admin can reopen
    assert olga.get(f"/api/alerts/{aid}").json()["reopen_count"] == 1
    acts = [(x["action"], x["details"].get("from")) for x in olga.get(f"/api/alerts/{aid}/audit").json()]
    assert acts[:4] == [("status:acknowledged", "resolved"), ("status:resolved", "acknowledged"),
                        ("status:acknowledged", "escalated"), ("status:escalated", "new")]
    wf = olga.get("/api/config").json()["workflow"]
    assert wf["transitions"]["new"] == list(TRANSITIONS["new"]) and wf["sla_s"]["P1"] == 60
    # re-correlation: a later event joining the alert rewrites its body but keeps the operator fields
    from tests.test_incidents_api import ev as aev
    a = site.store.get_alert(aid)
    e = a["events"][-1]
    site.on_event(aev(e["rule"], a["updated_at"] + 1, e["camera_id"], e["zone_id"], e["zone_name"],
                      tuple(e["track_ids"]), e["criticality"], e.get("site_zone")))
    a2 = site.store.get_alert(aid)
    assert len(a2["events"]) == len(a["events"]) + 1
    assert a2["escalation"]["reason"] == "door forced?" and a2["reopen_count"] == 1 and a2["status"] == "acknowledged"


def test_priority_list_filter(tmp_path):
    site, app = build(tmp_path)
    c = login(app, "vic")
    rows = c.get("/api/alerts").json()
    prios = {a["priority"] for a in rows}
    got = c.get("/api/alerts?priority=" + ",".join(sorted(prios))).json()
    assert len(got) == len(rows)
    assert all(a["priority"] in ("P1", "P2") for a in c.get("/api/alerts?priority=P1,P2").json())


def test_sla_config_override():
    assert sla_config({}) == {"P1": 60.0, "P2": 300.0, "P3": 1800.0, "P4": None}
    assert sla_config({"sla_s": {"P1": 30, "P4": 0}})["P1"] == 30.0


def test_sla_overdue_escalates_once_and_audits(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(iev("intrusion", T0, zone="vault", tracks=(50,), crit=3))          # P1: SLA 60 s
    assert d["priority"] == "P1"
    r.tick(T0 + 59)
    assert "sla" not in r.inc(d)
    r.tick(T0 + 61)
    inc = r.inc(d)
    assert inc["sla"]["state"] == "overdue" and inc["pushes"][-1]["trigger"] == "sla_overdue"
    assert inc["pushes"][-1]["decision"] == "would_push"
    r.tick(T0 + 120)
    rows = [n for n in r.store.list_notifications(d["incident_id"]) if n["trigger"] == "sla_overdue"]
    assert len(rows) == 1 and rows[0]["details"]["sla_s"] == 60                  # once per incident
    assert any(a["action"] == "incident_sla_overdue" and a["actor"] == "system"
               for a in r.store.list_audit(d["incident_id"]))


def test_sla_met_when_an_alert_is_acknowledged(tmp_path):
    r = Rig(tmp_path)
    d = r.feed(iev("intrusion", T0, zone="vault", tracks=(50,), crit=3))
    r.store.set_status(d["id"], "acknowledged", actor="olga", fields={"acknowledged_by": "olga"})
    r.tick(T0 + 61)
    inc = r.inc(d)
    assert inc["sla"]["state"] == "acknowledged" and inc["sla"]["by"] == "olga"
    assert not any(n["trigger"] == "sla_overdue" for n in r.store.list_notifications(d["incident_id"]))
    low = r.feed(iev("loitering", T0 + 100, zone="lobby", tracks=(7,)))           # P4: no SLA, no escalation
    r.tick(T0 + 5000)
    assert "sla" not in r.inc(low)


def test_sla_refires_webhook(tmp_path):
    sent = []

    class R:
        status_code = 200

    r = Rig(tmp_path, cfg={"webhook": {"url": "https://hook.example/y"}, "sla_s": {"P1": 10}},
            webhook_post=lambda url, content, headers, timeout: sent.append(content) or R())
    d = r.feed(iev("intrusion", T0, zone="vault", crit=3))
    r.tick(T0 + 11)
    import time
    for _ in range(100):
        if len(sent) >= 2:
            break
        time.sleep(0.02)
    assert len(sent) == 2 and '"trigger":"sla_overdue"' in sent[1] and '"sla_s":10' in sent[1]
    assert r.inc(d)["sla"]["sla_s"] == 10
    r.mgr.close()


def test_websocket_carries_incident_messages(tmp_path):
    from tests.test_incidents_api import ev as aev
    import time
    site, _ = build(tmp_path, users=False)
    c = TestClient(create_app(site, open_demo=True))
    with c.websocket_connect("/ws") as ws:
        site.on_event(aev("intrusion", time.time(), "cam1", "server_door", "Server room door", (9,), 3, "server_door"))
        seen = set()
        for _ in range(12):
            m = ws.receive_json()
            seen.add(m["type"])
            if m["type"] == "incident":
                assert m["incident"]["id"] and "timeline" not in m["incident"]    # compact view
                break
        assert "incident" in seen
