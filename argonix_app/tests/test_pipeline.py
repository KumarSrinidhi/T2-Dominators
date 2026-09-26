"""Integration: YOLO + ByteTrack + homography + rules on the scripted synthetic clips (ground truth known)."""
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "demo_site.json"
WEIGHTS = ROOT / "data" / "models" / "yolo26n.pt"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not WEIGHTS.exists(), reason="data/models/yolo26n.pt missing (real detector test)")]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    if not (ROOT / "data" / "synth" / "cam1.mp4").exists():
        from tools.make_synthetic import main
        main()
    from argonix.pipeline import Site
    cfg = json.loads(CFG.read_text())
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    d = tmp_path_factory.mktemp("run")
    (d / "site.json").write_text(json.dumps(cfg))
    s = Site(d / "site.json", data_dir=d, realtime=False)
    s.start(); s.wait(timeout=600); s.stop()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def rules_fired(site):
    return {(e["camera_id"], e["rule"]) for a in site.store.list_alerts() for e in a["events"]}


def test_cameras_ran_without_error(site):
    for w in site.workers:
        assert w.stats["status"] == "stopped", w.stats
        assert w.stats["frames"] >= 90  # 20 s at 5 fps


def test_expected_behaviours_detected(site):
    fired = rules_fired(site)
    for expected in [("cam1", "intrusion"), ("cam1", "loitering"), ("cam1", "running"),
                     ("cam2", "wrong_way"), ("cam2", "crowding")]:
        assert expected in fired, fired


def test_evidence_and_explainability(site):
    alerts = site.store.list_alerts()
    assert alerts
    for a in alerts:
        assert a["priority"] in {"P1", "P2", "P3", "P4"} and a["model"]["rules"] == "rules-v1"
        for e in a["events"]:
            assert (site.data / e["evidence"]["snapshot"]).exists()
            assert (site.data / e["evidence"]["clip"]).exists()
            assert e["derivation"]["threshold"] and 0 < e["confidence"] <= 1


def test_intrusion_outranks_loitering_only(site):
    by_rule = {e["rule"]: a for a in site.store.list_alerts() for e in a["events"]}
    assert by_rule["intrusion"]["priority"] in {"P1", "P2"}


def test_recordings_written(site):
    recs = site.store.list_recordings()
    assert {r["camera"] for r in recs} == {"cam1", "cam2"}
    assert all((site.data / r["path"]).stat().st_size > 1000 for r in recs)


def test_no_false_camera_health_alerts_on_clean_video(site):
    assert not site.store.list_alerts(rule="camera_health")
    assert all(w.stats.get("health") == "OK" for w in site.workers), [w.stats.get("health") for w in site.workers]
