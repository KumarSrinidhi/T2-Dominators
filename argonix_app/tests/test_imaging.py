"""Black-and-white / IR CCTV support: channel normalisation, monochrome flag, CLAHE, and the full pipeline on grey video."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from argonix.imaging import Conditioner, is_monochrome, to_bgr

ROOT = Path(__file__).resolve().parents[1]


def test_channel_normalisation():
    assert to_bgr(np.zeros((10, 20), np.uint8)).shape == (10, 20, 3)
    assert to_bgr(np.zeros((10, 20, 4), np.uint8)).shape == (10, 20, 3)
    assert to_bgr(np.zeros((10, 20, 1), np.uint8)).shape == (10, 20, 3)


def test_monochrome_detection():
    rng = np.random.default_rng(0)
    grey = rng.integers(0, 255, (60, 80), np.uint8)
    assert is_monochrome(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR))
    colour = rng.integers(0, 255, (60, 80, 3), np.uint8)
    assert not is_monochrome(colour)


def test_conditioner_modes():
    dark = np.full((60, 80), 25, np.uint8)
    dark[20:40, 30:50] = 45
    c = Conditioner("auto")
    out = c(dark)
    assert out.shape == (60, 80, 3) and c.monochrome and c.enhanced
    assert out.std() > cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR).std()  # contrast stretched
    off = Conditioner("off")
    assert np.array_equal(off(dark), cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)) and not off.enhanced
    with pytest.raises(ValueError):
        Conditioner("magic")


@pytest.mark.slow
def test_pipeline_on_greyscale_video(tmp_path):
    """The synthetic cam1 scene converted to single-channel grey still yields the intrusion (real YOLO)."""
    src = ROOT / "data" / "synth" / "cam1.mp4"
    if not src.exists() or not (ROOT / "data" / "models" / "yolo26n.pt").exists():
        pytest.skip("synthetic clip or model missing")
    from argonix.media import write_clip
    from argonix.pipeline import Site
    cap, frames = cv2.VideoCapture(str(src)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    grey = tmp_path / "cam1_grey.mp4"
    write_clip(grey, frames, 25)
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["cameras"] = cfg["cameras"][:1]
    cfg["cameras"][0]["source"] = str(grey)
    cfg["record"] = False
    cfg["detector"] = {**cfg.get("detector", {}), "weights": str(ROOT / "data" / "models" / "yolo26n.pt"),
                       "batched": False}
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir()
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp_path / "run", realtime=False)
    site.start(); site.wait(timeout=600); site.stop()
    w = site.workers[0]
    assert w.stats["monochrome"] is True
    rules = {e["rule"] for a in site.store.list_alerts() for e in a["events"]}
    assert "intrusion" in rules, rules
