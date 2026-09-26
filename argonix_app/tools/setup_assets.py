"""Fetch or generate everything a fresh clone needs that git does not carry (weights, public clips, rendered scenes).

    uv run python tools/setup_assets.py                  # core: detector weights, OpenCV vtest.avi, synthetic demo clips
    uv run python tools/setup_assets.py --all            # everything below
    uv run python tools/setup_assets.py --only reid,street
    uv run python tools/setup_assets.py --check          # report what is present / missing, change nothing
    (on a CUDA host: uv run --no-sync python tools/setup_assets.py ...)

Groups (all files land in the git-ignored data/ folder; nothing is committed):
    core     data/models/yolo26n.pt, data/real/vtest.avi, data/synth/cam1|cam2.mp4
             -> run.py --open-demo, configs/real_site.json (campus camera), the test suite
    models   data/models/yolo26s.pt (GPU configs), data/models/yolo26n-pose.pt (fall detection, altercation)
    street   data/real/social-distance/pedestrian.mp4 (Kaggle; needs KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY)
    reid     data/models/reid/osnet_ain_x1_0_msmt17_cosine.pth (torchreid model zoo; MSMT17 weights, non-commercial
             research terms) + the rendered 3-camera scene data/reid_eval/scene/ (configs/reid_demo.json)
    handoff  data/synth_handoff/camA|camB|camC.mp4 (configs/handoff_demo.json)
The scenes are rendered locally. Measured on a laptop CPU: reid scene about 15-20 min, handoff about 1 min.

Idempotent: files that are already there (and match their pinned SHA-256, where one is pinned) are skipped; partial
downloads resume. Deliberately NOT fetched: the fine-tuned weapon/altercation weights (trained on data whose licence
is unknown, so they are not redistributed: train them with tools/train_weapons.py / train_altercation.py), TensorRT
engines (GPU-specific: tools/bench_gpu.py --export), the VLM weights (tools/vlm_server.py downloads them on its host)
and the evaluation datasets (tools/datasets.py). Exit code 1 when a selected step fails, else 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "data" / "models"
REAL = ROOT / "data" / "real"
VTEST_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
VTEST_SHA = "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"
STREET_KAGGLE = "jayitabhattacharyya/social-distance"
STREET_SHA = "e7a9418237f7c08fe154dfd53eefd41cb8517d85f6b7ceb804333920177a359c"
OSNET = "osnet_ain_x1_0_msmt17_cosine.pth"
OSNET_GDRIVE_ID = "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal"
GROUPS = ("core", "models", "street", "reid", "handoff")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def ok_file(p: Path, sha: Optional[str] = None) -> bool:
    return p.is_file() and p.stat().st_size > 0 and (sha is None or sha256(p) == sha)


@contextmanager
def keep_config_text(name: str):
    """The scene generators rewrite their committed config. When the JSON content comes out the same, put the
    original bytes back so a fresh clone does not show a formatting-only diff."""
    p = ROOT / "configs" / name
    before = p.read_bytes() if p.exists() else None
    try:
        yield
    finally:
        if before is not None and p.exists():
            try:
                same = json.loads(before) == json.loads(p.read_bytes())
            except ValueError:
                same = False
            if same:
                p.write_bytes(before)


# ---------------------------------------------------------------------------------------------- steps
def yolo_weights(name: str) -> Callable[[], None]:
    def run() -> None:
        from ultralytics.utils.downloads import attempt_download_asset  # knows the release that carries yolo26*.pt
        dest = MODELS / name
        MODELS.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(dest))
        if not ok_file(dest):
            raise RuntimeError(f"Ultralytics did not download {name} (offline?)")
    return run


def fetch_url(url: str, dest: Path, sha: Optional[str]) -> None:
    from tools.datasets import download
    download(url, dest)
    if sha and sha256(dest) != sha:
        bad = dest.with_name(dest.name + ".bad")
        os.replace(dest, bad)
        raise RuntimeError(f"{rel(dest)}: SHA-256 does not match the pinned value (moved to {bad.name}); "
                           f"the source may have changed or served an error page: {url}")


def vtest() -> None:
    fetch_url(VTEST_URL, REAL / "vtest.avi", VTEST_SHA)


def synth() -> None:
    from tools.make_synthetic import main
    with keep_config_text("demo_site.json"):
        main()


def street() -> None:
    from tools.datasets import RemoteZip, _auth_headers, spec_url
    spec = {"kaggle": STREET_KAGGLE}
    headers = _auth_headers(spec)  # raises with a clear message when no Kaggle credentials are set
    rz = RemoteZip(spec_url(spec), headers)  # reads only the one member, not the whole dataset zip
    names = rz.select(["*pedestrian.mp4"])
    if not names:
        mp4 = [n for n in rz.entries if n.lower().endswith(".mp4")]
        raise RuntimeError(f"pedestrian.mp4 not found in the Kaggle zip (mp4 members: {mp4[:10]})")
    out = REAL / "social-distance"
    rz.extract(names[:1], out, strip=len(Path(names[0]).parts) - 1)
    dest = out / "pedestrian.mp4"
    if sha256(dest) != STREET_SHA:
        print(f"  note: {rel(dest)} differs from the clip the README numbers were measured on", flush=True)


def osnet() -> None:
    url = f"https://drive.usercontent.google.com/download?id={OSNET_GDRIVE_ID}&export=download&confirm=t"
    from argonix.reid import KNOWN
    fetch_url(url, MODELS / "reid" / OSNET, KNOWN[OSNET]["sha256"])


def reid_scene() -> None:
    from tools.make_reid_demo import main
    with keep_config_text("reid_demo.json"):
        rc = main([])
    if rc != 0:
        raise RuntimeError("tools/make_reid_demo.py failed (see above)")


def handoff() -> None:
    from tools.make_synthetic_handoff import DUR, SEED, make
    with keep_config_text("handoff_demo.json"):
        make(DUR, SEED)


@dataclass
class Step:
    group: str
    name: str
    targets: list[Path]
    run: Callable[[], None]
    sha: Optional[str] = None  # checked on the first target
    needs: tuple[str, ...] = ()  # names of steps whose files must exist first

    def present(self) -> bool:
        return all(ok_file(t, self.sha if i == 0 else None) for i, t in enumerate(self.targets))


def steps() -> list[Step]:
    from argonix.reid import KNOWN
    synth_dir, hand, scene = ROOT / "data" / "synth", ROOT / "data" / "synth_handoff", ROOT / "data" / "reid_eval" / "scene"
    return [
        Step("core", "yolo26n", [MODELS / "yolo26n.pt"], yolo_weights("yolo26n.pt")),
        Step("core", "vtest", [REAL / "vtest.avi"], vtest, VTEST_SHA),
        Step("core", "synthetic demo", [synth_dir / "cam1.mp4", synth_dir / "cam2.mp4"], synth, needs=("yolo26n",)),
        Step("models", "yolo26s", [MODELS / "yolo26s.pt"], yolo_weights("yolo26s.pt")),
        Step("models", "yolo26n-pose", [MODELS / "yolo26n-pose.pt"], yolo_weights("yolo26n-pose.pt")),
        Step("street", "street clip", [REAL / "social-distance" / "pedestrian.mp4"], street),
        Step("reid", "osnet", [MODELS / "reid" / OSNET], osnet, KNOWN[OSNET]["sha256"]),
        Step("reid", "reid scene", [scene / f"cam{c}.mp4" for c in "ABC"], reid_scene, needs=("yolo26n", "vtest")),
        Step("handoff", "handoff scene", [hand / f"cam{c}.mp4" for c in "ABC"], handoff, needs=("yolo26n",)),
    ]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="every group: " + ", ".join(GROUPS))
    ap.add_argument("--only", default="", help="comma-separated groups (default: core)")
    ap.add_argument("--check", action="store_true", help="only report what is present or missing")
    ap.add_argument("--force", action="store_true", help="redo the selected steps even if their files exist")
    a = ap.parse_args(argv)

    groups = set(GROUPS) if a.all else {g.strip() for g in a.only.split(",") if g.strip()} or {"core"}
    unknown = groups - set(GROUPS)
    if unknown:
        ap.error(f"unknown group(s) {sorted(unknown)}; choose from {', '.join(GROUPS)}")
    todo = [s for s in steps() if s.group in groups]
    by_name = {s.name: s for s in steps()}

    if not (ROOT / "assets" / "forecast" / "argonix_forecast_gru.pt").exists():
        print("WARN  assets/forecast/argonix_forecast_gru.pt is missing: it is committed, so the clone is incomplete "
              "(git lfs / sparse checkout?)", flush=True)

    failed = []
    for s in todo:
        where = ", ".join(rel(t) for t in s.targets)
        if s.present() and not a.force:
            print(f"ok    [{s.group}] {s.name}: {where}", flush=True)
            continue
        if a.check:
            print(f"MISS  [{s.group}] {s.name}: {where}", flush=True)
            failed.append(s.name)
            continue
        missing = [n for n in s.needs if not by_name[n].present()]
        if missing:
            print(f"SKIP  [{s.group}] {s.name}: needs {', '.join(missing)} first", flush=True)
            failed.append(s.name)
            continue
        print(f"get   [{s.group}] {s.name} -> {where}", flush=True)
        try:
            s.run()
            if not s.present():
                raise RuntimeError("finished but the expected files are not there")
            print(f"ok    [{s.group}] {s.name}", flush=True)
        except Exception as e:  # noqa: BLE001 - report every step, keep going
            print(f"FAIL  [{s.group}] {s.name}: {type(e).__name__}: {e}", flush=True)
            if os.environ.get("ARGONIX_DEBUG"):
                traceback.print_exc()
            failed.append(s.name)

    if failed:
        verb = "missing" if a.check else "failed"
        print(f"\n{len(failed)} of {len(todo)} {verb}: {', '.join(failed)}", flush=True)
        return 1
    print(f"\nall {len(todo)} selected assets present ({', '.join(sorted(groups))})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
