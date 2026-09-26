"""Shared test setup.

Many API/pipeline tests replay the synthetic demo clips (configs/demo_site.json -> data/synth/cam1|cam2.mp4) through
a fake detector. On a fresh checkout (CI) those clips do not exist yet, and before this fixture only the tests that
happened to run first generated them. Rendering needs data/models/yolo26n.pt to cut people out of Ultralytics'
sample images; Ultralytics downloads it on first use when it is missing. Offline without the weights the render is
skipped with a warning and the tests that replay the clips fail with "cannot open source".

Markers (registered in pyproject.toml): `slow` = runs a real YOLO detector over whole videos (CI deselects them with
-m "not slow"); `gpu` = needs CUDA.
"""
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synth"
WEIGHTS = ROOT / "data" / "models" / "yolo26n.pt"


def have_weights() -> bool:
    return WEIGHTS.exists()


@pytest.fixture(scope="session", autouse=True)
def synthetic_clips():
    if (SYNTH / "cam1.mp4").exists() and (SYNTH / "cam2.mp4").exists():
        return
    try:
        from tools.make_synthetic import main
        main()
    except Exception as e:  # noqa: BLE001 - offline without weights: the dependent tests report the missing clip
        warnings.warn(f"could not render the synthetic demo clips ({type(e).__name__}: {e})", stacklevel=1)


def pytest_collection_modifyitems(config, items):
    """`gpu` tests skip cleanly on machines without CUDA."""
    if not any(item.get_closest_marker("gpu") for item in items):
        return
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    if not cuda:
        skip = pytest.mark.skip(reason="needs a CUDA GPU")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip)
