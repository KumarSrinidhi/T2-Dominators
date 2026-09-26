"""Build the multi-camera Re-ID demo: a 3-camera scene of REAL pedestrians, and a site config that watches it.

The scene is the one `tools/eval_reid.py --video` evaluates (same functions, same output folder):
- walkers move on the handoff demo floor plan (tools/sim_handoff.py DEMO_TOPOLOGY: entrance hall camA, east corridor
  camB, north wing camC);
- each walker is pasted as crops of one real pedestrian tracked in data/real/vtest.avi (and the Kaggle street clip
  when present), with a different real frame of that pedestrian on every camera visit;
- every camera has its own colour/brightness/blur change.
Output: data/reid_eval/scene/camA|camB|camC.mp4 + truth.json, and configs/reid_demo.json (3 looping cameras, handoff
on with the demo topology, Re-ID on). Rendering runs on the CPU (YOLO26n to cut the pedestrians out once, then
compositing); an existing scene is reused unless --force is given.

    uv run python tools/make_reid_demo.py
    uv run python run.py --config configs/reid_demo.json --data data/run_reid --port 8004 --open-demo

This is a demo scene, not a benchmark: measured Re-ID numbers on it are in README ("Multi-camera Re-ID").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.eval_reid import OUT, build_bank, render_scene  # noqa: E402
from tools.make_synthetic_handoff import site_config  # noqa: E402
from tools.sim_handoff import load_spec  # noqa: E402

SCENE = OUT / "scene"
CFG = ROOT / "configs" / "reid_demo.json"
WEIGHTS = ROOT / "data" / "models" / "yolo26n.pt"

# Demo settings on top of argonix/global_ids.py DEFAULTS. Everything else (link threshold 0.6, retention 1 h,
# embeddings dropped 5 min after a segment ends) stays at the defaults.
REID = {
    "enabled": True,
    "model": "auto",        # OSNet weights from data/models/reid/ when present, else the labelled hand-crafted embedder
    "device": "auto",
    "color_mode": "auto",   # compare on luminance when either camera is monochrome/IR
    "max_dt_s": 40,         # travel-time window before the handoff topology is learned (same as handoff max_dt_s)
}


def build_config(scene: Path = SCENE) -> dict:
    """The site config for the rendered scene; source paths are relative to argonix_app/."""
    cfg = site_config(load_spec("demo"), scene)
    cfg = {"site": "Argus Re-ID demo (3 cameras, real pedestrian crops)",
           "_doc": ("Rendered by tools/make_reid_demo.py: walkers on the handoff demo floor plan, pasted as crops of "
                    "real pedestrians from data/real/vtest.avi with a per-camera colour/blur change. Global IDs are "
                    "probabilistic, revisable estimates (research-grade), never identities."),
           **{k: v for k, v in cfg.items() if k != "site"}}
    cfg["reid"] = dict(REID)
    return cfg


def scene_ready(scene: Path = SCENE) -> bool:
    return (scene / "truth.json").exists() and all((scene / f"{c}.mp4").exists() for c in ("camA", "camB", "camC"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--duration", type=float, default=480.0, help="scene length in seconds (clips loop in the demo)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", help="device for the YOLO pass that cuts out the pedestrians")
    ap.add_argument("--force", action="store_true", help="re-render the scene even if it exists")
    ap.add_argument("--config-only", action="store_true", help="only (re)write configs/reid_demo.json")
    a = ap.parse_args(argv)
    if not a.config_only and (a.force or not scene_ready()):
        if not (ROOT / "data" / "real" / "vtest.avi").exists():
            print("data/real/vtest.avi is missing: it is the source of the real pedestrian crops. Download it from "
                  "the OpenCV repository (samples/data/vtest.avi) into data/real/ first.", file=sys.stderr)
            return 2
        bank = build_bank(str(WEIGHTS), device=a.device)
        if not bank:
            print("no pedestrian crops were found in data/real/*: cannot render the scene", file=sys.stderr)
            return 2
        truth = render_scene(bank, SCENE, dur=a.duration, seed=a.seed)
        print(f"scene: {len(truth['visits'])} camera visits by {len(truth['walkers'])} walkers, "
              f"{len(bank)} real pedestrians in the bank -> {SCENE.relative_to(ROOT)}")
    elif not a.config_only:
        print(f"scene exists: {SCENE.relative_to(ROOT)} (use --force to re-render)")
    CFG.write_text(json.dumps(build_config(), indent=2) + "\n")
    print(f"wrote {CFG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
