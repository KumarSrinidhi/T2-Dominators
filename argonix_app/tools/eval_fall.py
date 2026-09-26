"""Run the fall / person-on-the-ground analyzer over video clips with the real detector + tracker (CPU is fine).

    uv run python tools/eval_fall.py --download                      # fetch the 3 CAVIAR "on floor" clips (~37 MB)
    uv run python tools/eval_fall.py                                 # default clip set, pose if weights available
    uv run python tools/eval_fall.py --gray                          # same, frames converted to greyscale first
    uv run python tools/eval_fall.py --no-pose clip1.mp4 clip2.avi

Default clip set: CAVIAR (EC CAVIAR project IST 2001 37540, homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/, research use
with acknowledgement; footage not redistributed) "Rest_FallOnFloor", "Rest_SlumpOnFloor", "Rest_WiggleOnFloor"
(one person ends up on the floor in each) plus normal walking footage (CAVIAR Walk/Browse, OpenCV vtest.avi, the Kaggle
street clip) for false alarms per hour. Clips that are not on disk are skipped. Results are a small sanity check,
not a benchmark: a handful of clips from one lobby camera.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.fall import FallAnalyzer, FallParams, PoseCache  # noqa: E402
from argonix.imaging import Conditioner  # noqa: E402

CAVIAR = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/{n}/{n}.mpg"
FALLS = ["Rest_FallOnFloor", "Rest_SlumpOnFloor", "Rest_WiggleOnFloor"]
DATA_DIRS = [ROOT / "data"]


def find(rel: str) -> Path | None:
    return next((d / rel for d in DATA_DIRS if (d / rel).exists()), None)


def default_clips() -> list[tuple[Path, bool]]:
    out = []
    for n in FALLS:
        p = find(f"datasets/caviar/raw/{n}.mpg")
        if p:
            out.append((p, True))
    for rel in ["datasets/caviar/raw/Walk1.mpg", "datasets/caviar/raw/Walk2.mpg", "datasets/caviar/raw/Walk3.mpg",
                "datasets/caviar/raw/Browse1.mpg", "real/vtest.avi", "real/social-distance/pedestrian.mp4"]:
        p = find(rel)
        if p:
            out.append((p, False))
    return out


def download() -> None:
    d = ROOT / "data" / "datasets" / "caviar" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    for n in FALLS:
        p = d / f"{n}.mpg"
        if not p.exists():
            req = urllib.request.Request(CAVIAR.format(n=n), headers={"User-Agent": "argonix-eval-fall"})
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
            print(f"downloaded {p} ({p.stat().st_size / 1e6:.1f} MB)")


def run_clip(path: Path, pose, fps: float, gray: bool, imgsz: int) -> dict:
    from argonix.detect import UltralyticsTracker
    trk = UltralyticsTracker(str(ROOT / "data" / "models" / "yolo26n.pt"), conf=0.3, imgsz=imgsz, device="cpu",
                             models_dir=ROOT / "data" / "models")
    an = FallAnalyzer(pose, params=FallParams())
    cond = Conditioner("auto")
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_fps = src_fps if 1 <= src_fps <= 120 else 25.0
    step = max(1, round(src_fps / fps))
    w = SimpleNamespace(id=path.stem, name=path.stem, calib_error=None, site=SimpleNamespace(data=None),
                        backend=SimpleNamespace(info={}))
    n, events, t0 = 0, [], time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % step:
            continue
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cond(frame)
        dets = trk.track(small)
        for e in an.on_frame(w, n / src_fps, small, dets, []):
            events.append({"t": round(e.t_end, 1), "t_start": round(e.t_start, 1), "track": e.track_ids,
                           "descent": e.derivation["episode"].get("descent"), "text": e.observation[:160]})
    cap.release()
    dur = n / src_fps
    return {"clip": path.name, "duration_s": round(dur, 1), "events": events, "n_events": len(events),
            "wall_s": round(time.time() - t0, 1), "pose_runs": an.stats["pose_runs"], "mode": an.mode(),
            "gray": gray}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--pose-weights", default=str(ROOT / "data" / "models" / "yolo26n-pose.pt"))
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", action="append", default=[], help="extra data/ dirs to look for clips in")
    a = ap.parse_args()
    DATA_DIRS.extend(Path(d) for d in a.data_dir)
    if a.download:
        download()
    pose = None
    if not a.no_pose:
        try:
            from argonix.altercation import UltralyticsPose
            pose = PoseCache(UltralyticsPose(a.pose_weights, device="cpu", imgsz=a.imgsz,
                                             models_dir=ROOT / "data" / "models"))
        except Exception as e:  # noqa: BLE001
            print(f"pose model unavailable ({e!r}); boxes-only mode")
    clips = [(Path(c), "fall" in Path(c).stem.lower() or "floor" in Path(c).stem.lower()) for c in a.clips] \
        or default_clips()
    res = []
    for p, positive in clips:
        r = run_clip(p, pose, a.fps, a.gray, a.imgsz)
        r["expected"] = "person ends up on the floor" if positive else "normal activity (no fall)"
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "events"}), flush=True)
        for e in r["events"]:
            print("   ", json.dumps(e))
    pos = [r for r in res if "floor" in r["expected"]]
    neg = [r for r in res if "floor" not in r["expected"]]
    neg_h = sum(r["duration_s"] for r in neg) / 3600
    summary = {"fall_clips_with_event": f"{sum(1 for r in pos if r['n_events'])}/{len(pos)}",
               "normal_footage_min": round(neg_h * 60, 1), "false_alarms": sum(r["n_events"] for r in neg),
               "false_alarms_per_hour": round(sum(r["n_events"] for r in neg) / neg_h, 2) if neg_h else None,
               "mode": res[0]["mode"] if res else None, "gray": a.gray,
               "hardware": f"{platform.processor() or platform.machine()} CPU", "fps": a.fps}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "clips": res}, indent=2))


if __name__ == "__main__":
    main()
