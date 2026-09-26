"""Does the ETH/UCY forecaster transfer to Argus's own tracker output? GRU vs constant velocity on real footage.

    uv run python tools/eval_forecast_real.py                          # cameras of configs/real_site.json
    uv run python tools/eval_forecast_real.py --config configs/real_site.json --imgsz 480

Runs the real detector + ByteTrack at 5 fps over each calibrated camera's clip once, projects foot points to the
ground plane with the camera's homography, resamples every person track to 0.4 s with the same smoothing as the
runtime analyzer (`sample_history`), and scores all 20-step windows (8 observed -> 12 future) of the resulting ground
tracks. The "truth" is the tracker's own later positions (detector jitter, ID switches and homography error included),
so this measures agreement with what the pipeline will observe, not true positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.forecast import (DT, OBS_LEN, PRED_LEN, BUNDLED_CHECKPOINT, Forecaster, ade_fde,  # noqa: E402
                              sample_history)
from argonix.geometry import apply_h, foot_point, homography_from_points  # noqa: E402
from argonix.imaging import Conditioner  # noqa: E402


def ground_tracks(src: str, H, imgsz: int, fps: float = 5.0, max_w: int = 960) -> dict[int, list]:
    from argonix.detect import UltralyticsTracker
    trk = UltralyticsTracker(str(ROOT / "data" / "models" / "yolo26n.pt"), conf=0.3, imgsz=imgsz, device="cpu")
    cap = cv2.VideoCapture(src)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(sfps / fps))
    cond, n, out = Conditioner("auto"), 0, {}
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        if n % step:
            continue
        scale = max(1.0, f.shape[1] / max_w)
        small = f if scale == 1.0 else cv2.resize(f, (int(f.shape[1] / scale), int(f.shape[0] / scale)))
        for tid, bb, cls, conf in trk.track(cond(small)):
            fx, fy = foot_point(bb)
            try:
                g = apply_h(H, (fx * scale, fy * scale))
            except ValueError:
                continue
            out.setdefault(tid, []).append((n / sfps, g[0], g[1]))
    return out


def windows_from_tracks(tracks: dict[int, list]) -> np.ndarray:
    seqs = []
    for pts in tracks.values():
        a = np.array(pts)
        if len(a) < 10:
            continue
        t_end = a[-1, 0]
        # every 0.4 s end time with a full 20-step window inside the track's span (no gaps > 1 s)
        for t_last in np.arange(a[0, 0] + (OBS_LEN + PRED_LEN - 1) * DT, t_end + 1e-6, DT):
            seg = a[(a[:, 0] >= t_last - (OBS_LEN + PRED_LEN - 1) * DT - 0.3) & (a[:, 0] <= t_last + 0.3)]
            if len(seg) < 5 or np.diff(seg[:, 0]).max() > 1.0:
                continue
            s = sample_history(seg[:, 0], seg[:, 1:3], t_last, n=OBS_LEN + PRED_LEN)
            if s is not None:
                seqs.append(s)
    return np.array(seqs).reshape(-1, OBS_LEN + PRED_LEN, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "real_site.json"))
    ap.add_argument("--checkpoint", default=str(BUNDLED_CHECKPOINT))
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="where data/real/... clips live")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    fc = Forecaster(a.checkpoint)
    res = {}
    for c in cfg["cameras"]:
        cal = c.get("calibration")
        src = Path(a.data_dir).parent / c["source"] if not Path(c["source"]).is_absolute() else Path(c["source"])
        if not cal or not src.exists():
            print(f"skip {c['id']} (no calibration or clip missing: {src})")
            continue
        H = homography_from_points([tuple(p) for p in cal["image_points"]], [tuple(p) for p in cal["ground_points"]])
        seqs = windows_from_tracks(ground_tracks(str(src), H, a.imgsz))
        if not len(seqs):
            continue
        obs, fut = seqs[:, :OBS_LEN], seqs[:, OBS_LEN:]
        g, _ = fc.predict(obs)
        cv, _ = fc.predict(obs, method="cv")
        r = lambda v: round(float(v), 3)
        res[c["id"]] = {"n_windows": len(seqs), "gru": [r(x) for x in ade_fde(g, fut)],
                        "cv": [r(x) for x in ade_fde(cv, fut)],
                        "stationary_hold": [r(x) for x in ade_fde(np.repeat(obs[:, -1:], PRED_LEN, 1), fut)],
                        "calib_error_m": cal.get("calib_error_m")}
        print(c["id"], json.dumps(res[c["id"]]), flush=True)
    print(json.dumps({"model": fc.info, "results_ade_fde_m": res}, indent=2))


if __name__ == "__main__":
    main()
