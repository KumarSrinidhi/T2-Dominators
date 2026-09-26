"""Generate two scripted synthetic camera clips (no downloads) with a known floor homography.

Person crops come from the images bundled with ultralytics; they are pasted onto a rendered floor along
scripted ground-plane paths, so ground truth (who enters which zone, speeds) is known for testing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from argonix.geometry import apply_h, homography_from_points, invert_h  # noqa: E402
from argonix.media import write_clip  # noqa: E402

W, H, FPS, DUR = 960, 540, 25, 20.0
IMG_PTS = [[80, 520], [880, 520], [700, 200], [260, 200]]
GND_PTS = [[0, 0], [16, 0], [16, 12], [0, 12]]
H_G2I = invert_h(homography_from_points([tuple(p) for p in IMG_PTS], [tuple(p) for p in GND_PTS]))


def person_crops(weights: str) -> list[np.ndarray]:
    import ultralytics
    from ultralytics import YOLO
    assets = Path(ultralytics.__file__).parent / "assets"
    m = YOLO(weights)
    crops = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(assets / name))
        r = m(img, classes=[0], conf=0.5, verbose=False)[0]
        for x1, y1, x2, y2 in r.boxes.xyxy.int().tolist():
            if (y2 - y1) > 1.6 * (x2 - x1) or name == "zidane.jpg":
                crops.append(img[y1:y2, x1:x2].copy())
    return crops


def floor() -> np.ndarray:
    bg = np.full((H, W, 3), (70, 72, 76), np.uint8)
    bg[:200] = (110, 105, 100)
    for gx in range(0, 17, 2):
        a, b = apply_h(H_G2I, (gx, 0)), apply_h(H_G2I, (gx, 12))
        cv2.line(bg, tuple(map(int, a)), tuple(map(int, b)), (95, 97, 100), 1)
    for gy in range(0, 13, 2):
        a, b = apply_h(H_G2I, (0, gy)), apply_h(H_G2I, (16, gy))
        cv2.line(bg, tuple(map(int, a)), tuple(map(int, b)), (95, 97, 100), 1)
    return bg


def path_pos(path: list[tuple[float, float, float]], t: float):
    """path: list of (t, x, y) keyframes; None outside [t0, tN]."""
    if t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return path[-1][1:]


def render(actors, crops) -> list[np.ndarray]:
    bg = floor()
    frames = []
    for i in range(int(DUR * FPS)):
        t = i / FPS
        f = bg.copy()
        placed = []
        for idx, path in actors:
            p = path_pos(path, t)
            if p:
                u, v = apply_h(H_G2I, p)
                placed.append((v, u, crops[idx % len(crops)]))
        for v, u, c in sorted(placed, key=lambda z: z[0]):  # far first
            h = int(90 + (v - 200) / 320 * 150)
            w = max(8, int(c.shape[1] * h / c.shape[0]))
            c2 = cv2.resize(c, (w, h))
            x1, y1 = int(u - w / 2), int(v - h)
            xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x1 + w), min(H, y1 + h)
            if xb > xa and yb > ya:
                f[ya:yb, xa:xb] = c2[ya - y1:yb - y1, xa - x1:xb - x1]
        frames.append(f)
    return frames


SCENARIOS = {
    # cam1: A walks into the restricted server-door zone and stays; B runs across the walkway.
    "cam1": [(0, [(0, 2, 2), (9, 12.5, 6.5), (20, 12.5, 6.8)]),
             (0, [(3, 15, 1.2), (7, 1, 1.2)])],
    # cam2: C walks against the one-way corridor; four people gather in the lobby.
    "cam2": [(2, [(0, 14, 3), (8, 3, 3)]),
             (0, [(6, 4, 12), (9, 4, 8), (20, 4.2, 8)]), (1, [(6, 7, 12), (9, 7, 9), (20, 7, 9)]),
             (3, [(6, 10, 12), (9, 10, 8), (20, 10, 8)]), (4, [(6, 12.5, 12), (9, 12.5, 9.5), (20, 12.5, 9.5)])],
}


def site_config(out_dir: Path) -> dict:
    cal = {"image_points": IMG_PTS, "ground_points": GND_PTS, "calib_error_m": 0.3,
           "check_points": {"image": [list(map(round, apply_h(H_G2I, (8, 6))))], "ground": [[8, 6]]}}
    rel = lambda n: os.path.relpath(out_dir / f"{n}.mp4", ROOT).replace("\\", "/")
    return {
        "site": "Argus demo site (synthetic)", "analytics_fps": 5, "record": True, "correlation_window_s": 15,
        "detector": {"weights": "data/models/yolo26n.pt", "tracker": "bytetrack.yaml", "conf": 0.3, "imgsz": 480},
        "map": {"width_m": 16, "height_m": 12},
        "cameras": [
            {"id": "cam1", "name": "Server wing", "source": rel("cam1"), "loop": True, "calibration": cal,
             "zones": [
                 {"id": "server_door", "name": "Server room door", "criticality": 3, "site_zone": "server_door",
                  "polygon": [[11, 4], [15, 4], [15, 9], [11, 9]],
                  "rules": {"intrusion": {"min_seconds": 1.0}, "loitering": {"min_seconds": 6}}},
                 {"id": "walkway", "name": "Walkway", "criticality": 1,
                  "polygon": [[0, 0], [16, 0], [16, 3], [0, 3]], "rules": {"running": {"speed_mps": 2.5}}}]},
            {"id": "cam2", "name": "Lobby", "source": rel("cam2"), "loop": True, "calibration": cal,
             "zones": [
                 {"id": "corridor", "name": "One-way corridor", "criticality": 2, "direction": [1, 0],
                  "polygon": [[1, 1], [15, 1], [15, 5], [1, 5]], "rules": {"wrong_way": {"min_seconds": 1.5}}},
                 {"id": "lobby", "name": "Lobby", "criticality": 1,
                  "polygon": [[2, 6.5], [14, 6.5], [14, 11], [2, 11]],
                  "rules": {"crowding": {"max_count": 3, "min_seconds": 3}}}]},
        ],
    }


def main(out_dir: Path = ROOT / "data" / "synth", weights: str = str(ROOT / "data" / "models" / "yolo26n.pt")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = person_crops(weights)
    for cam, actors in SCENARIOS.items():
        write_clip(out_dir / f"{cam}.mp4", render(actors, crops), FPS)
    cfg_path = ROOT / "configs" / "demo_site.json"
    try:  # keep sections this script does not generate (e.g. "sensors", the demo door controller)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg.update(site_config(out_dir))
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"{len(crops)} person crops; wrote {out_dir} and {cfg_path}")
    return cfg_path


if __name__ == "__main__":
    main()
