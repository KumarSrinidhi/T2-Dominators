"""Render a 3-camera synthetic site for the handoff (next-camera) demo, and evaluate the model on it.

Walkers are simulated on the DEMO_TOPOLOGY floor plan of tools/sim_handoff.py (entrance hall camA, east corridor
camB, north wing camC; corridors and doors between them, exits to outside) and pasted as person crops (as in
tools/make_synthetic.py) into three clips with a known floor homography. Ground truth (which camera each walker
appears in next, and when) is written to data/synth_handoff/truth.json.

    uv run python tools/make_synthetic_handoff.py                 # render clips + configs/handoff_demo.json
    uv run python run.py --config configs/handoff_demo.json       # watch it live (loops the clips)
    uv run python tools/make_synthetic_handoff.py --eval          # run faster than real time, score vs truth

The evaluation runs the full pipeline (YOLO detector + ByteTrack + handoff analyzer) on the rendered video,
matches each detected exit to the ground-truth walker visit and reports next-camera top-1 accuracy, log-loss and
ETA error for the NN and for the most-frequent-destination-per-exit-side baseline, plus the accuracy of the
self-supervised association labels. The cold-start simulator uses the same floor plan that generated the video,
which is optimistic: a real site's simulator would only approximate the true topology.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from argonix.geometry import apply_h  # noqa: E402
from argonix.handoff import LEFT  # noqa: E402
from argonix.media import H264Writer  # noqa: E402
from tools.make_synthetic import H_G2I, floor, person_crops  # noqa: E402
from tools.sim_handoff import GND_PTS, IMG_PTS, Sim, door_zones_px, load_spec, path_pos, person_h  # noqa: E402

FPS, DUR, SEED = 10, 480.0, 7
OUT = ROOT / "data" / "synth_handoff"
CFG = ROOT / "configs" / "handoff_demo.json"


def background(spec: dict, cam: str) -> np.ndarray:
    bg = floor()
    for p in spec["portals"].values():
        if p["camera"] != cam or p.get("kind") != "door":
            continue
        (u0, v0), (u1, v1) = apply_h(H_G2I, tuple(p["a"])), apply_h(H_G2I, tuple(p["b"]))
        cv2.rectangle(bg, (int(u0), int(v0) - 70), (int(u1), int(v1)), (40, 45, 55), -1)  # doorway in the back wall
    name = spec["cameras"][cam].get("name", cam)
    cv2.putText(bg, f"{cam} - {name} (synthetic)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    return bg


def render_camera(sim: Sim, visits, cam: str, crops, path: Path, dur: float) -> None:
    bg = background(sim.spec, cam)
    W, H = sim.size[cam]
    mine = [v for v in visits if v.cam == cam]
    wr = H264Writer(path, W, H, FPS)
    try:
        for i in range(int(dur * FPS)):
            t = i / FPS
            f = bg.copy()
            placed = []
            for v in mine:
                if not (v.path[0][0] <= t <= v.path[-1][0]):
                    continue
                g = path_pos(v.path, t)
                if any(float((np.asarray(g) - a) @ n) > 0.15 for a, n in sim.doors[cam]):
                    continue  # through the doorway: out of sight
                Hg = sim.Hg[cam]
                if (Hg[2][0] * g[0] + Hg[2][1] * g[1] + Hg[2][2]) * sim.w_sign[cam] <= 0:
                    continue
                u, vv = apply_h(Hg, g)
                placed.append((vv, u, crops[v.walker % len(crops)]))
            for vv, u, c in sorted(placed, key=lambda z: z[0]):  # far first
                hh = int(person_h(vv))
                ww = max(8, int(c.shape[1] * hh / c.shape[0]))
                c2 = cv2.resize(c, (ww, hh))
                x1, y1 = int(u - ww / 2), int(vv - hh)
                xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x1 + ww), min(H, y1 + hh)
                if xb > xa and yb > ya:
                    f[ya:yb, xa:xb] = c2[ya - y1:yb - y1, xa - x1:xb - x1]
            wr.write(f)
    finally:
        wr.close()


def site_config(spec: dict, out_dir: Path) -> dict:
    cal = {"image_points": IMG_PTS, "ground_points": GND_PTS, "calib_error_m": 0.3,
           "check_points": {"image": [list(map(round, apply_h(H_G2I, (8, 6))))], "ground": [[8, 6]]}}
    rel = lambda n: os.path.relpath(out_dir / f"{n}.mp4", ROOT).replace("\\", "/")
    # an empty synthetic room is pixel-identical frame to frame (no sensor noise): relax the frozen-view check
    cams = [{"id": cid, "name": c.get("name", cid), "source": rel(cid), "loop": True, "health": {"freeze_s": 600},
             "calibration": cal,
             "placement": c.get("placement"), "zones": []} for cid, c in spec["cameras"].items()]
    return {
        "site": "Argus handoff demo (synthetic, 3 cameras)", "analytics_fps": 5, "record": False,
        "correlation_window_s": 15,
        "detector": {"weights": "data/models/yolo26n.pt", "tracker": "bytetrack.yaml", "conf": 0.3, "imgsz": 480},
        "map": {"width_m": 60, "height_m": 50},
        "cameras": cams,
        "handoff": {"enabled": True, "sim_topology": "demo", "exit_zones_px": door_zones_px(spec),
                    "max_dt_s": 40, "retrain_every": 25, "cold_start": True, "cold_start_blocking": False},
    }


def make(dur: float = DUR, seed: int = SEED, out_dir: Path = OUT,
         weights: str = str(ROOT / "data" / "models" / "yolo26n.pt")) -> Path:
    spec = load_spec("demo")
    sim = Sim(spec, seed)
    visits = sim.run(dur - 45.0, t0=0.0)          # arrivals stop 45 s before the end so most walks finish
    sim.observe(visits, 0.0, dur, fps=FPS, noise=False)  # fills true first/last visible times
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = [c for c in person_crops(weights) if 2.0 <= c.shape[0] / c.shape[1] <= 4.5]  # full-body crops only
    crops += [cv2.flip(c, 1) for c in crops]
    for cam in spec["cameras"]:
        t = time.time()
        render_camera(sim, visits, cam, crops, out_dir / f"{cam}.mp4", dur)
        print(f"rendered {cam} ({time.time() - t:.0f} s)")
    truth = {"fps": FPS, "duration_s": dur, "seed": seed, "visits": []}
    for v in visits:
        nxt = visits[v.next] if v.next is not None else None
        truth["visits"].append({"walker": v.walker, "cam": v.cam, "enter": v.enter, "exit": v.exit,
                                "t_first": v.t_first, "t_last": v.t_last,
                                "next_cam": nxt.cam if nxt is not None else None,
                                "next_t_first": nxt.t_first if nxt is not None else None})
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    CFG.write_text(json.dumps(site_config(spec, out_dir), indent=2))
    n_hand = sum(1 for v in truth["visits"] if v["next_cam"])
    print(f"{len(visits)} camera visits by {max((v.walker for v in visits), default=0)} walkers "
          f"({n_hand} camera-to-camera transitions); wrote {out_dir} and {CFG}")
    return CFG


# ---------------------------------------------------------------------------------------------------------------
# Evaluation on the rendered video
# ---------------------------------------------------------------------------------------------------------------
def _hard(pred: dict) -> str:
    return max(pred["probs"], key=pred["probs"].get)


def evaluate_run(engine, truth: dict, step: int = 2, src_fps: float = FPS, tol: float = 1.5) -> dict:
    visits = truth["visits"]
    by_cam: dict[str, list] = {}
    for i, v in enumerate(visits):
        if v["t_last"] is not None and v["t_last"] < truth["duration_s"] - 1.0:
            by_cam.setdefault(v["cam"], []).append((i, v))
    used, rows, unmatched = set(), [], 0
    for h in engine.history:
        st = h["t"] - engine.first_t[h["cam"]] + (step - 1) / src_fps
        cands = [(abs(v["t_last"] - st), i, v) for i, v in by_cam.get(h["cam"], []) if abs(v["t_last"] - st) <= tol]
        if not cands:
            unmatched += 1
            continue
        _, i, v = min(cands, key=lambda c: c[0])
        if i in used:
            unmatched += 1
            continue
        used.add(i)
        dest = v["next_cam"] or LEFT
        dt = (v["next_t_first"] - v["t_last"]) if v["next_cam"] and v["next_t_first"] is not None else None
        rows.append({"cam": h["cam"], "key": h["key"], "truth": dest, "dt": dt, "pred": h["pred"], "base": h["base"],
                     "label": h["target"], "n_candidates": h["n_candidates"]})
    n_truth = sum(len(v) for v in by_cam.values())

    def summarize(key: str) -> dict:
        r = [x for x in rows if x[key]]
        if key == "pred":
            r = [x for x in r if x["pred"].get("source") == "nn"]
        if not r:
            return {"n": 0}
        acc = float(np.mean([_hard(x[key]) == x["truth"] for x in r]))
        ll = float(np.mean([-math.log(max(1e-6, x[key]["probs"].get(x["truth"], 0.0))) for x in r]))
        eta = [abs(x[key]["eta"][x["truth"]] - x["dt"]) for x in r
               if x["truth"] != LEFT and x["dt"] is not None and x["truth"] in (x[key].get("eta") or {})]
        k = sum(_hard(x[key]) == x["truth"] for x in r)
        out = {"n": len(r), "top1": round(acc, 4), "top1_wilson95": wilson(k, len(r)), "logloss": round(ll, 4),
               "eta_mae_s": round(float(np.mean(eta)), 2) if eta else None, "n_eta": len(eta)}
        if key == "pred":  # the NN travel-time head itself (served ETAs may come from the edge distributions)
            eh = [abs(x[key]["eta_nn"][x["truth"]] - x["dt"]) for x in r
                  if x["truth"] != LEFT and x["dt"] is not None and x["truth"] in (x[key].get("eta_nn") or {})]
            out["eta_head_mae_s"] = round(float(np.mean(eh)), 2) if eh else None
        return out

    both = [x for x in rows if x["pred"] and x["pred"].get("source") == "nn" and x["base"]]
    lab_acc = float(np.mean([max(x["label"], key=x["label"].get) == x["truth"] for x in rows])) if rows else None
    lab_w = float(np.mean([x["label"].get(x["truth"], 0.0) for x in rows])) if rows else None
    per_key = {}
    for x in both:
        d = per_key.setdefault(f"{x['cam']}:{x['key']}", {"n": 0, "nn": 0, "baseline": 0})
        d["n"] += 1
        d["nn"] += int(_hard(x["pred"]) == x["truth"])
        d["baseline"] += int(_hard(x["base"]) == x["truth"])
    return {"truth_exits": n_truth, "detected_exits": len(engine.history), "matched_exits": len(rows),
            "unmatched_detected_exits": unmatched, "exit_recall": round(len(rows) / n_truth, 3) if n_truth else None,
            "nn": summarize("pred"), "baseline": summarize("base"),
            "paired": {"n": len(both), "nn_correct": sum(_hard(x["pred"]) == x["truth"] for x in both),
                       "baseline_correct": sum(_hard(x["base"]) == x["truth"] for x in both)},
            "association_label_top1": round(lab_acc, 4) if lab_acc is not None else None,
            "association_mean_weight_on_truth": round(lab_w, 4) if lab_w is not None else None,
            "per_exit_side": per_key}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def run_eval(cold_start: bool = True, imgsz: int = 480, data_dir: Path = ROOT / "data" / "handoff_eval") -> dict:
    from argonix.pipeline import Site
    cfg = json.loads(CFG.read_text())
    for c in cfg["cameras"]:
        c["loop"] = False
    cfg["detector"]["imgsz"] = imgsz
    cfg["handoff"].update(cold_start=cold_start, cold_start_blocking=True)
    run_dir = data_dir / ("cold" if cold_start else "online_only")
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "configs").mkdir(parents=True)
    p = run_dir / "configs" / "site.json"
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
    cfg["detector"]["weights"] = str(ROOT / cfg["detector"]["weights"])
    p.write_text(json.dumps(cfg))
    t = time.time()
    site = Site(p, data_dir=run_dir / "run", realtime=False)
    t_cold = time.time() - t
    eng = site.handoff.engine
    t = time.time()
    site.start()
    site.wait()
    t_run = time.time() - t
    site.stop()  # flushes pending exits, waits for training, persists
    truth = json.loads((OUT / "truth.json").read_text())
    res = evaluate_run(eng, truth)
    res.update(cold_start=cold_start, imgsz=imgsz, cold_start_s=round(t_cold, 1), video_s=truth["duration_s"],
               processing_s=round(t_run, 1), speedup_vs_realtime=round(truth["duration_s"] / t_run, 2),
               model_version=eng.version, online_labels=eng.n_labels, rolling_metrics=eng.metrics.summary(),
               sim_eval=eng.cold_start_eval, frames={w.id: w.stats["frames"] for w in site.workers})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--duration", type=float, default=DUR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval", action="store_true", help="run the pipeline on the rendered clips and score vs truth")
    ap.add_argument("--no-cold-start", action="store_true", help="evaluate learning from online labels only")
    ap.add_argument("--imgsz", type=int, default=480)
    a = ap.parse_args()
    if a.eval:
        if not (OUT / "truth.json").exists():
            make(a.duration, a.seed)
        res = run_eval(not a.no_cold_start, a.imgsz)
        name = "eval_online_only.json" if a.no_cold_start else "eval.json"
        (OUT / name).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    else:
        make(a.duration, a.seed)
