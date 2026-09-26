"""Zone congestion forecast: damped Holt, the seasonal + deviation forecaster on synthetic series (beats naive and
seasonal-naive), the skill / history gate, congestion notes, the API and the optional capped event. CPU only."""
import math
import random
import time

from fastapi.testclient import TestClient

from argonix.baseline import hour_of_week
from argonix.congestion import DEFAULTS, LABEL, RULE, ZoneSeries, forecast_zone, holt
from argonix.server import create_app
from tests.test_server import make_site

T0 = 1_790_000_000.0 // 3600 * 3600


def synthetic(days: float, kind: str = "seasonal", seed: int = 1, end: float = None) -> list[tuple[float, float]]:
    """1-minute occupancy. seasonal: daily cycle + evening peak + a persistent AR(1) deviation + noise;
    noise: a constant level with white noise (nothing to forecast beyond the mean)."""
    rnd, rows, dev = random.Random(seed), [], 0.0
    n = int(days * 1440)
    start = (end - n * 60) if end is not None else T0
    for m in range(n):
        t = start + m * 60
        hod = (t % 86400) / 3600
        s = 12 + 10 * math.sin(2 * math.pi * (hod - 8) / 24) + (6 if 17 <= hod < 19 else 0)
        dev = 0.97 * dev + rnd.gauss(0, 0.8)
        v = s + dev + rnd.gauss(0, 1) if kind == "seasonal" else 10 + rnd.gauss(0, 2)
        rows.append((t, max(0.0, v)))
    return rows


def test_holt_tracks_level_and_damps_trend():
    assert holt([5.0] * 30, 10, 0.5, 0.1, 0.9) == [5.0] * 10
    up = holt([float(i) for i in range(40)], 30, 0.5, 0.1, 0.9)
    assert up[0] > 38 and all(b > a for a, b in zip(up, up[1:]))      # keeps rising ...
    assert up[29] - up[28] < up[1] - up[0]                             # ... ever more slowly (damped)
    assert holt([], 3, 0.5, 0.1, 0.9) == [0.0, 0.0, 0.0]


def test_zone_series_hour_of_week_matches_baseline():
    zs = ZoneSeries([], 60, DEFAULTS)
    rnd = random.Random(3)
    for m in (rnd.randrange(29_000_000, 31_000_000) for _ in range(3000)):
        assert zs.local(m)[0] == hour_of_week(m * 60)


def test_forecaster_beats_naive_and_seasonal_on_seasonal_series():
    f = forecast_zone(synthetic(21), 60, 30, DEFAULTS)
    assert f["status"] == "ok" and f["useful"] and f["label"] == LABEL
    r = next(x for x in f["backtest"]["horizons"] if x["h"] == 30)
    assert r["n_eval"] >= DEFAULTS["min_backtest"] and r["n_eval_seasonal"] >= DEFAULTS["min_backtest"]
    assert r["mae"] < r["mae_naive"] * 0.95 and r["mae_on_seasonal_subset"] < r["mae_seasonal"] * 0.95
    assert 0.6 <= r["interval"]["coverage_eval"] <= 0.95  # nominal 80%, measured on held-out origins
    assert len(f["forecast"]) == 30 and f["forecast"][-1]["h_min"] == 30
    assert all(p["lo"] <= p["mean"] <= p["hi"] for p in f["forecast"])
    assert f["forecast"][0]["t"] == f["as_of"]  # first forecast minute starts where the observations end
    assert f["history"] and len(f["history"]) <= 60 and "seasonal-naive" in f["method"]


def test_gate_no_skill_on_noise():
    f = forecast_zone(synthetic(21, "noise"), 60, 30, DEFAULTS, limit=11)
    assert f["status"] == "no_skill" and not f["useful"] and "Insufficient skill" in f["message"]
    assert f["congestion"]["expected"] is False  # never a congestion note from an unskilled forecast


def test_gate_insufficient_history():
    f = forecast_zone(synthetic(3), 60, 30, DEFAULTS)  # no previous week: no seasonal reference yet
    assert f["status"] == "insufficient_history" and not f["useful"]
    assert "damped Holt" in f["method"]
    f = forecast_zone(synthetic(0.1), 60, 30, DEFAULTS)
    assert f["status"] == "insufficient_history" and not f["useful"]
    f = forecast_zone([], 60, 30, DEFAULTS)
    assert f["status"] == "no_data" and f["forecast"] == []


def test_stale_series_is_not_forecast():
    rows = synthetic(21)
    f = forecast_zone(rows, 60, 30, DEFAULTS, now=rows[-1][0] + 3600)
    assert f["status"] == "stale" and f["stale"] and not f["useful"] and f["forecast"] == []


def test_congestion_note_uses_upper_bound():
    rows = synthetic(21)
    base = forecast_zone(rows, 60, 30, DEFAULTS)
    hi = [p["hi"] for p in base["forecast"]]
    limit = max(base["now"], min(hi)) + 0.01
    assume = any(x > limit for x in hi)
    f = forecast_zone(rows, 60, 30, DEFAULTS, limit=limit)
    c = f["congestion"]
    assert c["limit"] == limit and c["observed_now_above"] is False
    if assume:
        first = next(p for p in f["forecast"] if p["hi"] > limit)
        assert c["expected"] and c["in_min"] == first["h_min"] and c["upper"] > limit
    f = forecast_zone(rows, 60, 30, DEFAULTS, limit=max(hi) + 1)
    assert f["congestion"]["expected"] is False


def seeded_site(tmp_path, cfg_patch=None):
    site = make_site(tmp_path)
    site.cfg["congestion_forecast"] = {**DEFAULTS, **(cfg_patch or {})}
    from argonix.congestion import install
    install(site)
    now = time.time()
    rows = synthetic(21, end=now // 60 * 60)
    site.baseline.db.executemany("INSERT INTO occupancy VALUES(?,?,?,?,?)",
                                 [("cam1", "server_door", hour_of_week(t), t, v) for t, v in rows])
    site.baseline.db.commit()
    return site, now


def test_forecast_api(tmp_path):
    site, _ = seeded_site(tmp_path)
    c = TestClient(create_app(site, open_demo=True))
    d = c.get("/api/analytics/forecast?horizon_min=30").json()
    assert d["label"] == LABEL and {z["zone"] for z in d["zones"]} == {"server_door", "walkway"}
    door = next(z for z in d["zones"] if z["zone"] == "server_door")
    assert door["status"] == "ok" and door["useful"] and len(door["forecast"]) == 30
    assert door["camera"] == "cam1" and door["kind"] == "area" and door["limit"] is None
    walk = next(z for z in d["zones"] if z["zone"] == "walkway")
    assert walk["status"] == "no_data" and not walk["useful"]
    one = c.get("/api/analytics/forecast?zone=cam1:server_door&horizon_min=15").json()
    assert [z["zone"] for z in one["zones"]] == ["server_door"] and len(one["zones"][0]["forecast"]) == 15
    assert c.get("/api/analytics/forecast?zone=nope").status_code == 404
    assert c.get("/api/analytics/forecast?horizon_min=0").status_code == 400
    assert c.get("/api/analytics/forecast?horizon_min=500").status_code == 400
    site.congestion = None  # "congestion_forecast": {"enabled": false}
    assert c.get("/api/analytics/forecast").status_code == 404


def test_congestion_event_is_informational_and_capped(tmp_path):
    site, now = seeded_site(tmp_path, {"emit_events": True})
    cf = site.congestion
    assert cf in site.analyzers
    z = next(z for z in cf.zones() if z["zone"] == "server_door")
    f = cf.forecast(z, 30, now=now)
    limit = max(f["now"], min(p["hi"] for p in f["forecast"])) + 0.01
    raw = next(r for r in site.workers[0].cfg["zones"] if r["id"] == "server_door")
    raw.setdefault("rules", {})["crowding"] = {"max_count": limit}
    worker = site.workers[0]
    evs = cf.on_frame(worker, now, None, [], [])
    assert len(evs) == 1
    e = evs[0]
    assert e.rule == RULE and e.severity == 1 and e.derivation["prediction"] and e.derivation["learned_model"]
    assert e.observation.startswith("Forecast (prediction, not an observation)")
    assert cf.on_frame(worker, now + 5, None, [], []) == []      # evaluated once per interval
    assert cf.on_frame(worker, now + 120, None, [], []) == []    # one event per zone per rearm period
    site.on_event(e)
    a = next(a for a in site.store.list_alerts() if a["events"][0]["rule"] == RULE)
    assert a["priority"] in ("P3", "P4")
    site.cfg["congestion_forecast"] = {"emit_events": False}
    from argonix.congestion import install
    site.analyzers.remove(cf)
    assert install(site) is not None and site.congestion not in site.analyzers  # off by default: API only
