"""Decision-support additions: robust N-point calibration, baseline-aware dwell, operator track record (+ optional
feedback demotion), camera recovery, "what changed vs normal?" and the footage access log."""
import json
import math
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from argonix.baseline import Baseline, hour_of_week
from argonix.changes import what_changed
from argonix.geometry import apply_h, fit_homography, homography_from_points
from argonix.models import Event, TrackObs
from argonix.rules import RuleEngine, Zone
from argonix.server import create_app
from argonix.store import Store
from argonix.track_record import TrackRecord
from tests.test_rule_additions import make_site

IMG = [(100, 400), (540, 400), (620, 470), (20, 470)]
GND = [(0, 10), (8, 10), (8, 0), (0, 0)]
H_TRUE = homography_from_points(IMG, GND)
H_INV = homography_from_points(GND, IMG)  # ground -> image, to make extra exact correspondences


# ---------------------------------------------------------------------------------------------- calibration
def extra_points(n=6):
    g = [(1 + 6 * (i % 3) / 2, 1 + 8 * (i // 3) / 2) for i in range(n)]
    return [apply_h(H_INV, p) for p in g], g


def test_fit_exact_four_matches_solve():
    h, info = fit_homography(IMG, GND)
    assert info["fit"].startswith("exact") and info["loo_rmse_m"] is None
    assert all(math.isclose(a, b, abs_tol=1e-9) for r1, r2 in zip(h, H_TRUE) for a, b in zip(r1, r2))


def test_fit_many_points_flags_outlier_and_reports_loo():
    ei, eg = extra_points()
    img, gnd = IMG + ei, GND + eg
    gnd = list(gnd)
    gnd[5] = (gnd[5][0] + 3.0, gnd[5][1])  # a mistyped ground coordinate
    h, info = fit_homography(img, gnd, robust_threshold_m=0.5)
    assert info["points"] == 10 and info["inliers"] == 9 and info["outliers"] == [5]
    assert info["loo_rmse_m"] is not None and info["loo_rmse_m"] < 1e-3
    q = apply_h(h, (300, 430))
    assert math.dist(q, apply_h(H_TRUE, (300, 430))) < 1e-4


def test_fit_rejects_no_consensus():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], [(0, 0), (1, 0), (1, 1), (0, 1), (5, 5)])


def test_site_calibration_uses_leave_one_out_error(tmp_path):
    site = make_site(tmp_path)
    cam = site.cfg["cameras"][0]
    cal = cam["calibration"]
    h = homography_from_points([tuple(p) for p in cal["image_points"]], [tuple(p) for p in cal["ground_points"]])
    hi = homography_from_points([tuple(p) for p in cal["ground_points"]], [tuple(p) for p in cal["image_points"]])
    g_extra = [(3.0, 3.0), (12.0, 9.0)]
    # slightly perturbed extra image points: a realistic, non-zero leave-one-out error
    i_extra = [(apply_h(hi, g)[0] + 4, apply_h(hi, g)[1] - 3) for g in g_extra]
    cal2 = {**cal, "image_points": cal["image_points"] + [list(p) for p in i_extra],
            "ground_points": cal["ground_points"] + [list(g) for g in g_extra], "robust_threshold_m": 2.0}
    site2 = make_site(tmp_path / "b", cam_patch={"calibration": cal2})
    info = site2.workers[0].calib_info
    assert info["fit"]["points"] == 6 and info["fit"]["inliers"] == 6
    assert info["calib_error_m"] >= info["fit"]["loo_rmse_m"] > 0
    assert apply_h(h, (480, 300))  # the 4-point site still works
    assert site.workers[0].calib_info["fit"]["fit"].startswith("exact")


# ---------------------------------------------------------------------------------------------- dwell baseline
SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def obs(tid, t):
    return TrackObs("cam1", tid, t, (0, 0, 1, 1), "person", 0.9, (4.0, 4.0))


def loiter_engine(provider, **p):
    e = RuleEngine("cam1", [Zone.from_dict({"id": "z", "name": "Waiting area", "polygon": SQ,
                                            "rules": {"loitering": {"min_seconds": 10, **p}}})])
    e.dwell_baseline = provider
    return e


def first_fire(e, until=200.0):
    for i in range(int(until * 2)):
        t = i * 0.5
        ev = e.update(t, [obs(1, t)])
        if ev:
            return t, ev[0]
    return None, None


def test_dwell_threshold_uses_site_baseline():
    e = loiter_engine(lambda z, t, q: (60.0, 50), baseline_quantile=0.95, baseline_factor=1.5, baseline_min_stays=30)
    thr, info = e.dwell_threshold("z", e.zones[0].rules["loitering"], 0.0)
    assert thr == 90.0 and info["dwell_baseline"]["stays"] == 50
    t, ev = first_fire(e)
    assert 90.0 <= t < 91.0
    assert "usual here at this hour: p95 60s over 50 stays" in ev.observation
    assert ev.derivation["threshold"]["configured_min_seconds"] == 10


def test_dwell_threshold_falls_back_when_not_ready_or_off():
    e = loiter_engine(lambda z, t, q: (60.0, 5), baseline_quantile=0.95)
    thr, info = e.dwell_threshold("z", e.zones[0].rules["loitering"], 0.0)
    assert thr == 10 and "not ready (5 of 30" in info["dwell_baseline"]
    assert 10.0 <= first_fire(e)[0] < 11.0
    off = loiter_engine(lambda z, t, q: (60.0, 500))  # no baseline_quantile: plain min_seconds
    assert off.dwell_threshold("z", off.zones[0].rules["loitering"], 0.0)[0] == 10
    low = loiter_engine(lambda z, t, q: (2.0, 500), baseline_quantile=0.9)  # never below the configured minimum
    assert low.dwell_threshold("z", low.zones[0].rules["loitering"], 0.0)[0] == 10


# ---------------------------------------------------------------------------------------------- track record
def alert(aid, rule, cam, disp=None, t=None, status="resolved", rules_n=1):
    t = time.time() - 3600 if t is None else t
    evs = [{"rule": rule, "camera_id": cam, "zone_id": "z", "zone_name": "Z", "t_start": t, "t_end": t,
            "derivation": {}}]
    if rules_n > 1:
        evs.append({**evs[0], "rule": "running"})
    d = {"id": aid, "priority": "P2", "status": status, "created_at": t, "updated_at": t, "cameras": [cam],
         "events": evs, "factors": {"top_event": rule, "distinct_rules": rules_n, "distinct_cameras": 1,
                                    "modalities": ["video"]}}
    if disp:
        d["disposition"] = disp
    return d


def seeded_store(tmp_path, fp=19, tp=1):
    st = Store(tmp_path / "a.db")
    for i in range(fp):
        st.upsert_alert(alert(f"A{i}", "loitering", "cam1", "false_positive_rule"))
    for i in range(tp):
        st.upsert_alert(alert(f"B{i}", "loitering", "cam1", "true_positive_actioned"))
    st.upsert_alert(alert("C1", "loitering", "cam2", "true_positive_no_action"))
    st.upsert_alert(alert("D1", "loitering", "cam1", "duplicate"))  # says nothing about correctness
    return st


def test_track_record_counts_reviewed_decisions(tmp_path):
    tr = TrackRecord(seeded_store(tmp_path))
    d = alert("N1", "loitering", "cam1", status="new")
    tr.annotate(d)
    r = d["factors"]["track_record"]
    assert (r["confirmed"], r["reviewed"]) == (1, 20) and r["precision"] == 0.05
    assert (r["site_confirmed"], r["site_reviewed"]) == (2, 21)
    assert d["priority"] == "P2" and "demoted" not in d["factors"]  # demotion is off by default


def test_feedback_demotion_only_for_single_uncorroborated_alerts(tmp_path):
    tr = TrackRecord(seeded_store(tmp_path), demotion={"enabled": True, "min_reviewed": 20, "max_precision_upper": 0.3})
    d = tr.annotate(alert("N1", "loitering", "cam1", status="new"))
    assert d["priority"] == "P3" and d["factors"]["demoted"]["from"] == "P2"
    assert "confirmed 1 of the last 20" in d["factors"]["demoted"]["reason"]
    both = tr.annotate(alert("N2", "loitering", "cam1", status="new", rules_n=2))
    assert both["priority"] == "P2"  # another rule agrees: never demoted
    other = tr.annotate(alert("N3", "loitering", "cam2", status="new"))
    assert other["priority"] == "P2"  # too few decisions on cam2
    few = TrackRecord(seeded_store(tmp_path / "x", fp=5), demotion={"enabled": True})
    assert few.annotate(alert("N4", "loitering", "cam1", status="new"))["priority"] == "P2"


# ---------------------------------------------------------------------------------------------- camera recovery
def health_event(t, cond="cover"):
    return Event("camera_health", "cam1", "_view", "Cam (whole view)", [], t, t,
                 f"Camera view condition '{cond}' detected on cam1", {"rule": "camera_health", "condition": cond},
                 0.9, 2, 2, None, None)


def test_health_recovery_marks_the_alert(tmp_path):
    site = make_site(tmp_path)
    t0 = time.time()
    d = site.on_event(health_event(t0))
    seen = []
    site.listeners.append(seen.append)
    out = site.on_health_recovered("cam1", "cover", t0 + 42.0, t0)
    assert out["id"] == d["id"] and out["health_recovery"][0]["outage_s"] == 42.0
    assert site.store.get_alert(d["id"])["health_recovery"][0]["condition"] == "cover"
    assert seen and any(a["action"] == "camera_recovered" for a in site.store.list_audit(d["id"]))
    assert site.on_health_recovered("cam1", "cover", t0 + 50.0, t0) is None  # recorded once
    assert site.on_health_recovered("cam1", "blur", t0 + 50.0, t0) is None   # no alert for that condition
    # a later event on the same alert keeps the recovery note
    again = site.on_event(health_event(t0 + 1))
    assert again["id"] == d["id"] and again.get("health_recovery")


# ---------------------------------------------------------------------------------------------- what changed
def changes_site(tmp_path, now):
    b = Baseline(tmp_path / "b.db")
    st = Store(tmp_path / "a.db")
    week = 7 * 86400.0
    for w in range(1, 5):  # four previous weeks at the same hours of the week
        for k in range(60):
            t = now - w * week - k * 60.0
            b.add_sample("cam1", "lobby", t, 2.0 + 0.2 * (k % 3))
            b.add_sample("cam1", "hall", t, 1.0 + 0.2 * (k % 2))
        for k in range(2):
            t = now - w * week - k * 3600.0 + 1
            b.db.execute("INSERT INTO event_rate VALUES(?,?,?,?,?)", ("cam1", "lobby", hour_of_week(t), t, 1))
    b.db.commit()
    for k in range(30):  # the last half hour: lobby much busier, hall as usual
        b.add_sample("cam1", "lobby", now - k * 60.0 - 30, 9.0)
        b.add_sample("cam1", "hall", now - k * 60.0 - 30, 1.1)
        b.add_sample("cam1", "new_zone", now - k * 60.0 - 30, 3.0)  # no history yet
    for i in range(12):
        t = now - 60 * (i + 1)
        st.upsert_alert({"id": f"A{i}", "priority": "P3", "status": "new", "created_at": t, "updated_at": t,
                         "cameras": ["cam1"], "events": [{"rule": "intrusion", "camera_id": "cam1", "zone_id": "lobby",
                                                          "zone_name": "Lobby", "t_start": t, "t_end": t,
                                                          "derivation": {}}]})
    w = SimpleNamespace(id="cam1", name="Entrance", stats={"health": "degraded", "health_reasons": ["cover"]},
                        cfg={"zones": [{"id": "lobby", "name": "Lobby"}, {"id": "hall", "name": "Hall"}]})
    return SimpleNamespace(baseline=b, store=st, workers=[w])


def test_what_changed_vs_normal(tmp_path):
    now = time.time()
    r = what_changed(changes_site(tmp_path, now), hours=1.0, now=now)
    kinds = {(c["kind"], c["zone"]): c for c in r["changes"]}
    occ = kinds[("occupancy", "lobby")]
    assert occ["direction"] == "higher" and occ["observed"] == 9.0 and "'Lobby' (Entrance)" in occ["text"]
    ev = kinds[("event_rate", "lobby")]
    assert ev["observed"] == 12 and ev["usual"] < 3 and "intrusion x12" in ev["text"]
    assert ("camera_health", "_view") in kinds and r["changes"][0]["kind"] == "camera_health"
    assert ("occupancy", "hall") not in kinds and any("Hall" in s for s in r["steady"])
    assert any(x["zone"] == "new_zone" for x in r["not_enough_history"])
    assert "raises no alerts" in r["disclaimer"]


# ---------------------------------------------------------------------------------------------- access log
def test_media_and_live_reads_are_audited_once_per_window(tmp_path):
    site = make_site(tmp_path)
    ev = site.data / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "x.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    c = TestClient(create_app(site, open_demo=True))
    h = {"X-Argonix-CSRF": "1"}
    assert c.get("/media/evidence/x.jpg", headers=h).status_code == 200
    assert c.get("/media/evidence/x.jpg", headers=h).status_code == 200
    rows = [a for a in site.store.list_audit() if a["action"] == "media_read"]
    assert len(rows) == 1
    det = rows[0]["details"]
    assert (json.loads(det) if isinstance(det, str) else det)["path"] == "evidence/x.jpg"
    w = site.workers[0]
    w.latest_jpeg = b"\xff\xd8\xff\xd9"
    for _ in range(3):
        assert c.get(f"/api/cameras/{w.id}/latest.jpg", headers=h).status_code == 200
    assert len([a for a in site.store.list_audit() if a["action"] == "live_view"]) == 1


def test_access_log_can_be_turned_off(tmp_path):
    site = make_site(tmp_path, extra={"privacy": {"access_log": False}})
    w = site.workers[0]
    w.latest_jpeg = b"\xff\xd8\xff\xd9"
    c = TestClient(create_app(site, open_demo=True))
    assert c.get(f"/api/cameras/{w.id}/latest.jpg").status_code == 200
    assert not [a for a in site.store.list_audit() if a["action"] == "live_view"]
