"""Evaluate Argus on the public test datasets fetched by tools/datasets.py (eval_real.py-style, GPU if available).

For each dataset it runs the production pipeline (detector -> ByteTrack -> zones -> rules [-> analyzers]) over the
converted H.264 clips faster than real time and reports, per clip set:
  * frames, tracks (>= 10 samples), detections per frame, detect ms;
  * calibration used: a published one, the automatic pedestrian self-calibration (argonix/autocalib.py, fitted over
    all clips of the same camera), or none (uncalibrated: pixel zones, no speeds) -- always stated;
  * median / p90 walking speed of tracks when calibrated (sanity: ~1.1-1.6 m/s for walkers);
  * events per rule;
  * where labels exist, an HONEST comparison: UCSD Ped1/Ped2 and CUHK Avenue frame-level anomaly labels vs frames
    covered by our events, MOT17/MOT20 boxes vs our person detections (recall / precision at IoU 0.5);
  * colour vs grayscale: the same clips re-encoded as grayscale (datasets.py `variants: ["gray"]`).

Two accuracy profiles write committed result artifacts to eval_runs/<date>-<label>/ (JSON + markdown):
  * `caviar`: event-level precision / recall of the deterministic rules (loitering, running, crowding, and the
    altercation analyzer with --threats) against CAVIAR CVML ground-truth intervals, per rule, with Wilson 95 %
    intervals -> caviar_rules.{json,md}. Matching rule: see `score_caviar` / eval_metrics.event_matches_episode.
  * `mot17`: tracking accuracy (MOTA/MOTP, IDF1, HOTA; TrackEval conventions, tools/eval_metrics.py) of our
    detector + ByteTrack on MOT17 train sequences at the analytics rate (5 fps) and at the full frame rate
    -> mot17_tracking.{json,md}.

    uv run python tools/eval_datasets.py --only caviar,mot17 --fetch --device cuda:0
    uv run python tools/eval_datasets.py --only caviar --clips Fight_RunAway1 --only mot17 ...   (see README)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from argonix.pipeline import Site  # noqa: E402
from tools import eval_metrics as M  # noqa: E402

# batched None = the app default: one shared batched model on CUDA, a per-camera tracker on CPU
DET = {"weights": "auto", "tracker": "bytetrack.yaml", "conf": 0.3, "imgsz": 640, "device": "auto", "batched": None,
       "max_batch": 16, "max_wait_ms": 8}


def video_info(p: Path) -> dict:
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps if 1 <= fps <= 120 else 25.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        cap.release()


def full_view_zone(w: int, h: int, rules: dict, zid: str = "view", margin: float = 0.02) -> dict:
    mx, my = w * margin, h * margin
    return {"id": zid, "name": "Whole view", "criticality": 1,
            "polygon_px": [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], "rules": rules}


# ------------------------------------------------------------------------------------------------ runner
def run_clips(clips: list[dict], site_extra: Optional[dict] = None, analytics_fps: float = 5.0,
              det: Optional[dict] = None) -> dict:
    """Run a set of clips as cameras of one site (batched GPU inference), not real time.

    clips: [{"id", "path", "zones", "calibration"?, "masks_px"?}]. Returns per-camera results with per-frame
    detections (analytics frames only), events with frame numbers, and per-track speeds.
    """
    cfg = {"site": "eval_datasets", "analytics_fps": analytics_fps, "record": False, "correlation_window_s": 15,
           "detector": {**DET, **(det or {})}, "cameras": []}
    cfg.update(site_extra or {})
    info = {}
    for c in clips:
        cam = {"id": c["id"], "name": c["id"], "source": str(c["path"]), "loop": False,
               "zones": c.get("zones", []), "health": {"freeze_s": 1e9}}
        for k in ("calibration", "masks_px"):
            if c.get(k):
                cam[k] = c[k]
        cfg["cameras"].append(cam)
        info[c["id"]] = video_info(Path(c["path"]))
    tmp = Path(tempfile.mkdtemp(prefix="argonix_evalds_"))
    (tmp / "configs").mkdir()
    p = tmp / "configs" / "site.json"
    p.write_text(json.dumps(cfg))
    site = Site(p, data_dir=tmp / "run", realtime=False)
    rec: dict = {w.id: {"speeds": defaultdict(list), "dets": {}, "t0": None, "detect_ms": []} for w in site.workers}
    for w in site.workers:
        orig_upd, orig_proc, orig_track = w.engine.update, w._process, w.backend.track
        r = rec[w.id]
        fi = info[w.id]
        step = max(1, round(fi["fps"] / analytics_fps))

        def upd(t, obs, w=w, orig=orig_upd, r=r):
            out = orig(t, obs)
            if w.H is not None:
                for o in obs:
                    v = w.engine.velocity(o.track_id)
                    if v:
                        r["speeds"][o.track_id].append(math.hypot(*v))
            return out

        def proc(frame, t, w=w, orig=orig_proc, r=r, step=step, fps=fi["fps"]):
            if r["t0"] is None:
                r["t0"] = t - step / fps  # file sources: t = start_wall + n / fps, first processed n = step
            r["_t"] = t
            return orig(frame, t)

        def track(small, w=w, orig=orig_track, r=r, fi=fi):
            t0 = time.perf_counter()
            dets = orig(small)
            r["detect_ms"].append((time.perf_counter() - t0) * 1000)
            scale = fi["w"] / small.shape[1]  # analytics px -> source px
            r["dets"][round((r["_t"] - r["t0"]) * fi["fps"])] = (
                [(tid, tuple(float(x) for x in bb), cls, float(cf)) for tid, bb, cls, cf in dets], scale)
            return dets
        w.engine.update, w._process, w.backend.track = upd, proc, track
    t0 = time.time()
    site.start(); site.wait(); site.stop()
    wall = time.time() - t0
    alerts = site.store.list_alerts(limit=100000)
    out = {"wall_s": round(wall, 1), "cameras": {}, "model": dict(site.workers[0].backend.info) if site.workers else {}}
    for w in site.workers:
        r, fi = rec[w.id], info[w.id]
        evs = []
        for a in alerts:
            for e in a["events"]:
                if e["camera_id"] != w.id or r["t0"] is None:
                    continue
                evs.append({"rule": e["rule"], "zone": e.get("zone_id"),
                            "f0": round((e["t_start"] - r["t0"]) * fi["fps"]),
                            "f1": round((e["t_end"] - r["t0"]) * fi["fps"]),
                            "tracks": e.get("track_ids", []), "condition": (e.get("derivation") or {}).get("condition"),
                            "confidence": e.get("confidence")})
        per_track = [statistics.median(v) for v in r["speeds"].values() if len(v) >= 10]
        tracks = Counter(tid for ds, _ in r["dets"].values() for tid, _, cls, _ in ds if cls == "person")
        dms = r["detect_ms"]
        out["cameras"][w.id] = {
            "video": fi, "frames": w.stats["frames"], "health": w.stats.get("health"),
            "monochrome": w.stats.get("monochrome"),
            "calibrated": w.H is not None, "calib_error_m": w.calib_error,
            "tracks_seen": len(tracks), "tracks_10plus": sum(1 for n in tracks.values() if n >= 10),
            "dets_per_frame": round(sum(len(ds) for ds, _ in r["dets"].values()) / max(1, len(r["dets"])), 2),
            "mean_det_conf": round(float(np.mean([cf for ds, _ in r["dets"].values() for *_, cf in ds])), 3)
            if any(ds for ds, _ in r["dets"].values()) else None,
            "detect_ms_median": round(statistics.median(dms), 1) if dms else None,
            "median_track_speed_mps": round(statistics.median(per_track), 2) if per_track else None,
            "p90_track_speed_mps": round(sorted(per_track)[int(0.9 * (len(per_track) - 1))], 2) if per_track else None,
            "moving_tracks_median_speed_mps": (round(statistics.median([s for s in per_track if s >= 0.3]), 2)
                                               if any(s >= 0.3 for s in per_track) else None),
            "events_by_rule": dict(Counter(e["rule"] for e in evs)),
            "events": evs, "dets": r["dets"]}
    return out


# ------------------------------------------------------------------------------------------------ calibration
def autocalibrate(paths: list[Path], hfov_deg: Optional[float] = None, max_clips: int = 40) -> dict:
    """Fit ONE pedestrian self-calibration over several clips of the same fixed camera (tracks kept distinct)."""
    from argonix.autocalib import AutoCalibrator
    from argonix.detect import make_backend_factory
    fac = make_backend_factory({**DET, "batched": False}, models_dir=ROOT / "data" / "models")
    cal = AutoCalibrator(hfov_deg=hfov_deg, max_samples=2000, per_track_max=20)
    cal.start(None)
    for k, p in enumerate(paths[:max_clips]):
        be = fac({"id": f"calib{k}"})
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step, n = max(1, round(fps / 5)), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % step:
                continue
            scale = max(1.0, frame.shape[1] / 960)
            small = frame if scale == 1.0 else cv2.resize(frame, (int(frame.shape[1] / scale), int(frame.shape[0] / scale)))
            dets = [(tid + 100000 * (k + 1), bb, cls, cf) for tid, bb, cls, cf in be.track(small)]
            cal.add(dets, scale, (frame.shape[1], frame.shape[0]), k * 1e5 + n / fps)
        cap.release()
    st = cal.finish()
    return st


# ------------------------------------------------------------------------------------------------ scoring helpers
def intervals(frames: list[int], max_gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of frame numbers (gaps <= max_gap merged)."""
    out = []
    for f in sorted(set(frames)):
        if out and f - out[-1][1] <= max_gap:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(x) for x in out]


def frame_level(gt: dict[str, set], pred: dict[str, set], total: dict[str, int]) -> dict:
    tp = sum(len(gt[c] & pred.get(c, set())) for c in gt)
    fp = sum(len(pred.get(c, set()) - gt.get(c, set())) for c in total)
    fn = sum(len(gt[c] - pred.get(c, set())) for c in gt)
    n = sum(total.values())
    return {"frames": n, "gt_anomalous": tp + fn, "pred_flagged": tp + fp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def det_match(gt_boxes: dict[int, list], cam: dict, thr: float = 0.5, min_h: float = 0.0) -> dict:
    """Greedy IoU matching of our person boxes (analytics px * scale = source px) vs GT boxes on analytics frames."""
    tp = fp = fn = 0
    for f, (ds, scale) in cam["dets"].items():
        g = [b for b in gt_boxes.get(f, []) if b[3] - b[1] >= min_h]
        p = sorted([(cf, tuple(x * scale for x in bb)) for _, bb, cls, cf in ds if cls == "person"], reverse=True)
        used = set()
        for _, pb in p:
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if i not in used:
                    v = iou(pb, gb)
                    if v > best:
                        best, bi = v, i
            if best >= thr:
                used.add(bi); tp += 1
            else:
                fp += 1
        fn += len(g) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None}


def strip_heavy(res: dict) -> dict:
    """Drop per-frame detections/events before writing results."""
    out = {k: v for k, v in res.items() if k != "cameras"}
    out["cameras"] = {c: {k: v for k, v in r.items() if k not in ("dets", "events")} for c, r in res.get("cameras", {}).items()}
    return out


# ------------------------------------------------------------------------------------------------ ground truth
def parse_cvml(path: Path) -> dict:
    """CAVIAR CVML ground truth -> {"objects": {id: {frame: rec}}, "groups": {id: {frame: rec}}, "frames": n}."""
    root = ET.parse(path).getroot()
    objs: dict = defaultdict(dict)
    groups: dict = defaultdict(dict)
    n = 0

    def labels(el) -> dict:
        h = el.find(".//hypothesis")
        rec = {}
        for k in ("movement", "role", "context", "situation"):
            x = h.find(k) if h is not None else None
            rec[k] = (x.text or "").strip().lower() if x is not None else ""
        b = el.find("box")
        if b is not None:
            xc, yc, w, hh = (float(b.get(k, 0)) for k in ("xc", "yc", "w", "h"))
            rec["box"] = (xc - w / 2, yc - hh / 2, xc + w / 2, yc + hh / 2)
        return rec
    for fr in root.iter("frame"):
        f = int(fr.get("number"))
        n = max(n, f + 1)
        ol = fr.find("objectlist")
        for o in (ol.findall("object") if ol is not None else []):
            objs[o.get("id")][f] = labels(o)
        gl = fr.find("grouplist")
        for g in (gl.findall("group") if gl is not None else []):
            rec = labels(g)
            m = g.find("members")
            rec["members"] = [x for x in re.split(r"[,\s]+", (m.text or "") if m is not None else "") if x]
            groups[g.get("id")][f] = rec
    return {"objects": objs, "groups": groups, "frames": n}


CAVIAR_BEHAVIOURS = {
    # behaviour: (source, predicate, min seconds, Argus counterpart rules)
    "running": ("objects", lambda r: r["movement"] == "running", 1.0, ["running"]),
    "fighting": ("groups", lambda r: r["context"] == "fighting" or r["situation"] == "fighting", 1.0,
                 ["altercation", "running"]),
    "meeting": ("groups", lambda r: r["context"] == "meeting", 3.0, ["crowding"]),
    "browsing_or_inactive_20s": ("objects", lambda r: r["role"] != "leaving object" and
                                 (r["situation"] in ("browsing", "inactive") or r["movement"] == "inactive"), 20.0,
                                 ["loitering"]),
    "left_object": ("objects", lambda r: r["role"] == "leaving object", 0.0, []),
}


def caviar_episodes(gt: dict, fps: float) -> dict[str, list[dict]]:
    """Ground-truth behaviour episodes: per object / group, runs of frames where the behaviour predicate holds
    (gaps <= 0.5 s merged), at least `min seconds` long. Each episode carries its per-frame boxes (for groups: the
    group box plus the member objects' boxes) for the spatial part of the matching rule."""
    out: dict = {}
    for beh, (src, pred, min_s, _) in CAVIAR_BEHAVIOURS.items():
        eps = []
        for oid, frames in gt[src].items():
            for a, b in intervals([f for f, r in frames.items() if pred(r)], max_gap=int(0.5 * fps)):
                if (b - a + 1) / fps < min_s:
                    continue
                boxes = {}
                for f in range(a, b + 1):
                    r = frames.get(f)
                    if not r:
                        continue
                    bs = [r["box"]] if r.get("box") else []
                    for mid in r.get("members", []):
                        mr = gt["objects"].get(mid, {}).get(f)
                        if mr and mr.get("box"):
                            bs.append(mr["box"])
                    if bs:
                        boxes[f] = bs
                eps.append({"id": oid, "f0": a, "f1": b, "boxes": boxes})
        out[beh] = eps
    return out


def ucsd_gt(mfile: Path) -> list[set]:
    """UCSDped{1,2}.m: TestVideoFile{end+1}.gt_frame = [a:b, c:d]; (1-based frame numbers)."""
    out = []
    for line in mfile.read_text(errors="replace").splitlines():
        if "gt_frame" in line:
            s = set()
            for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", line.split("=", 1)[1]):
                s |= set(range(int(a), int(b) + 1))
            out.append(s)
    return out


def avenue_gt(mat: Path) -> set:
    """CUHK Avenue testing_label_mask/<n>_label.mat -> set of 1-based anomalous frame numbers."""
    import scipy.io
    vol = scipy.io.loadmat(str(mat))["volLabel"]
    return {i + 1 for i, m in enumerate(vol.ravel()) if np.asarray(m).any()}


def mot_gt(seq_dir: Path) -> dict[int, list]:
    """MOTChallenge gt/gt.txt -> {frame (1-based): [xyxy boxes]} for pedestrians marked 'consider'."""
    out: dict = defaultdict(list)
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        v = line.split(",")
        f, x, y, w, h = int(v[0]), *map(float, v[2:6])
        if int(float(v[6])) == 1 and int(float(v[7])) == 1:
            out[f].append((x, y, x + w, y + h))
    return out


def mot_ids(seq_dir: Path) -> int:
    ids = set()
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        v = line.split(",")
        if int(float(v[6])) == 1 and int(float(v[7])) == 1:
            ids.add(int(v[1]))
    return len(ids)


# ------------------------------------------------------------------------------------------------ profiles
def _calib_for(paths: list[Path], hfov: Optional[float], cache: dict, key: str) -> tuple[Optional[dict], dict]:
    if key not in cache:
        t0 = time.time()
        st = autocalibrate(paths, hfov_deg=hfov)
        est = st.get("estimate")
        cache[key] = (est, {"state": st["state"], "reason": st["reason"], "samples": st["samples"],
                            "tracks": st["tracks"], "coverage": st["coverage"], "fit_s": round(time.time() - t0, 1),
                            "calib_error_m": est and est["calib_error_m"], "params": est and est["params"],
                            "walking_speed_check": est and est.get("walking_speed_check"),
                            "hfov_deg_assumed": hfov})
    return cache[key]


def zones_for(info: dict, calib: Optional[dict], rules: dict) -> list[dict]:
    """A whole-view zone; for a calibrated camera the calibration quadrilateral (always below the horizon)."""
    if calib:
        return [{"id": "view", "name": "Calibrated floor area", "criticality": 1,
                 "polygon_px": [list(p) for p in calib["image_points"]], "rules": rules}]
    return [full_view_zone(info["w"], info["h"], {k: v for k, v in rules.items()
                                                  if k in ("intrusion", "loitering", "crowding")})]


def _spec(p: Path, calib: Optional[dict], rules: dict, prefix: str = "") -> dict:
    info = video_info(p)
    return {"id": re.sub(r"[^A-Za-z0-9_-]", "_", prefix + p.stem)[:40], "path": p,
            "calibration": calib and {k: calib[k] for k in ("image_points", "ground_points", "calib_error_m")},
            "zones": zones_for(info, calib, rules)}


def summarize(res: dict, cams: list[str]) -> dict:
    cs = [res["cameras"][c] for c in cams if c in res["cameras"]]
    speeds = [c["moving_tracks_median_speed_mps"] for c in cs if c.get("moving_tracks_median_speed_mps")]
    dms = [c["detect_ms_median"] for c in cs if c.get("detect_ms_median")]
    ev: Counter = Counter()
    for c in cs:
        ev.update(c["events_by_rule"])
    confs = [c["mean_det_conf"] for c in cs if c["mean_det_conf"]]
    return {"clips": len(cs), "frames": sum(c["frames"] for c in cs),
            "duration_s": round(sum(c["video"]["frames"] / c["video"]["fps"] for c in cs), 1),
            "tracks_10plus": sum(c["tracks_10plus"] for c in cs),
            "dets_per_frame": round(statistics.mean([c["dets_per_frame"] for c in cs]), 2) if cs else None,
            "mean_det_conf": round(statistics.mean(confs), 3) if confs else None,
            "detect_ms_median": round(statistics.median(dms), 1) if dms else None,
            "median_moving_speed_mps": round(statistics.median(speeds), 2) if speeds else None,
            "monochrome": sorted({str(c.get("monochrome")) for c in cs}),
            "events_by_rule": dict(ev)}


CAVIAR_RULES_PX = {  # uncalibrated (CAVIAR publishes no ground-plane calibration): image-pixel zone + thresholds
    "loitering": {"min_seconds": 20},
    "crowding": {"max_count": 1, "min_seconds": 3},                  # > 1 person track in view for 3 s
    "running": {"speed_bh_s": 1.5, "min_seconds": 1.0},             # body heights / s (~2.5 m/s for 1.7 m)
}
CAVIAR_RULES_M = {"loitering": {"min_seconds": 20}, "crowding": {"max_count": 1, "min_seconds": 3},
                  "running": {"speed_mps": 2.5, "min_seconds": 1.0}}
# (ground-truth behaviour, Argus rule, role): "primary" = the rule meant to find that behaviour
CAVIAR_PAIRS = [("browsing_or_inactive_20s", "loitering", "primary: dwell >= 20 s"),
                ("running", "running", "primary"),
                ("meeting", "crowding", "proxy: > 1 person in view is not a meeting"),
                ("fighting", "altercation", "primary (only with --threats)"),
                ("fighting", "running", "secondary"),
                ("left_object", None, "no Argus rule")]


def event_track_boxes(cam: dict, ev: dict) -> dict[int, list]:
    """The event's tracks' boxes (source px) on the analytics frames inside the event interval."""
    tids = set(ev.get("tracks") or [])
    out = {}
    for f, (ds, scale) in cam["dets"].items():
        if ev["f0"] <= f <= ev["f1"]:
            bs = [tuple(x * scale for x in bb) for tid, bb, cls, _ in ds if tid in tids and cls == "person"]
            if bs:
                out[f] = bs
    return out


def score_caviar(gt_eps: dict[str, list[dict]], preds: dict[str, list[dict]], tol: int,
                 prox_px: Optional[float], rules_run: set) -> dict:
    """Per (behaviour, rule) pair: event-level precision/recall with Wilson 95 % intervals."""
    out = {}
    for beh, rule, role in CAVIAR_PAIRS:
        key = f"{beh} vs {rule}" if rule else f"{beh} (no Argus rule)"
        r = M.match_events(gt_eps.get(beh, []), preds.get(rule, []) if rule else [], tol, prox_px)
        r.update(behaviour=beh, rule=rule, role=role, rule_enabled=bool(rule) and rule in rules_run)
        out[key] = r
    return out


def profile_caviar(root: Path, a, cache: dict) -> dict:
    d = root / "caviar"
    clips = sorted(p for p in (d / "mp4").glob("*.mp4") if not p.stem.endswith("_gray"))
    if a.clips:
        want = {c.strip() for c in a.clips.split(",") if c.strip()}
        clips = [p for p in clips if p.stem in want]
    clips = [p for p in clips if (d / "gt" / f"{p.stem}.xml").exists()]
    if a.max_clips:
        clips = clips[:a.max_clips]
    if not clips:
        return {"error": f"no converted CAVIAR clips with ground truth under {d} (use --fetch)"}
    if a.caviar_space == "auto":
        calib, cinfo = _calib_for(clips, a.hfov, cache, "caviar")
        if not calib:
            return {"error": "auto-calibration refused on CAVIAR", "calibration": cinfo}
        rules = dict(CAVIAR_RULES_M)
    else:
        calib, cinfo = None, {"state": "none", "note": "CAVIAR publishes no ground-plane calibration: zones and "
                                                       "thresholds are in image pixels; running uses body-heights/s"}
        rules = dict(CAVIAR_RULES_PX)
    extra = {"threats": {"altercation": {"enabled": True, "pose_weights": a.pose}}} if a.threats else {}
    tol = int(round(a.tol_s * 25))
    out = {"space": "image_px" if calib is None else "ground_m", "calibration": cinfo, "rules": rules,
           "altercation_enabled": bool(a.threats), "analytics_fps": 5.0,
           "matching": {"temporal_tolerance_s": a.tol_s, "proximity_px": a.prox_px,
                        "rule": "an event is a true positive when its interval overlaps a ground-truth episode "
                                "(+-tolerance) of the target behaviour in the same clip AND, in a shared frame of "
                                "that overlap, the centre of one of the event's track boxes is within proximity_px "
                                "of the episode's box (group box or a member's box). Many-to-one both ways: "
                                "precision = true events / events, recall = episodes hit / episodes.",
                        "episodes": "CVML labels: running = object movement 'running' >= 1 s; fighting = group "
                                    "context/situation 'fighting' >= 1 s; meeting = group context 'meeting' >= 3 s; "
                                    "browsing_or_inactive_20s = object situation browsing/inactive or movement "
                                    "inactive >= 20 s; left_object = role 'leaving object'. Runs merged over "
                                    "<= 0.5 s gaps."},
           "runs": {}}
    variants = {"colour": clips}
    grays = [p.with_name(p.stem + "_gray.mp4") for p in clips if p.with_name(p.stem + "_gray.mp4").exists()]
    if grays and a.gray:
        variants["gray"] = grays
        variants["colour_same_clips"] = [p.with_name(p.stem[:-5] + ".mp4") for p in grays]
    for vname, paths in variants.items():
        specs = []
        for p in paths:
            info = video_info(p)
            spec = {"id": re.sub(r"[^A-Za-z0-9_-]", "_", p.stem)[:40], "path": p}
            if calib:
                spec["calibration"] = {k: calib[k] for k in ("image_points", "ground_points", "calib_error_m")}
                spec["zones"] = [{"id": "view", "name": "Calibrated floor area", "criticality": 1,
                                  "polygon_px": [list(q) for q in calib["image_points"]], "rules": rules}]
            else:
                spec["zones"] = [full_view_zone(info["w"], info["h"], rules)]
            specs.append(spec)
        res = run_clips(specs, site_extra=extra)
        gts, preds = defaultdict(list), defaultdict(list)
        per_clip = {}
        for s, p in zip(specs, paths):
            stem = p.stem[:-5] if p.stem.endswith("_gray") else p.stem
            cam = res["cameras"][s["id"]]
            eps = caviar_episodes(parse_cvml(d / "gt" / f"{stem}.xml"), cam["video"]["fps"])
            for beh, lst in eps.items():
                gts[beh] += [{**e, "clip": s["id"]} for e in lst]
            for e in cam["events"]:
                preds[e["rule"]].append({"clip": s["id"], "f0": e["f0"], "f1": e["f1"],
                                         "boxes": event_track_boxes(cam, e)})
            per_clip[s["id"]] = {"gt_episodes": {b: len(v) for b, v in eps.items() if v},
                                 "events_by_rule": cam["events_by_rule"], "tracks_10plus": cam["tracks_10plus"],
                                 "dets_per_frame": cam["dets_per_frame"], "frames": cam["video"]["frames"],
                                 "events": [{k: e[k] for k in ("rule", "f0", "f1", "tracks")} for e in cam["events"]
                                            if e["rule"] != "camera_health"]}
        rules_run = set(rules) | ({"altercation"} if a.threats else set())
        out["runs"][vname] = {"summary": summarize(res, [s["id"] for s in specs]),
                              "scores": score_caviar(gts, preds, tol, a.prox_px, rules_run),
                              "per_clip": per_clip, "wall_s": res["wall_s"], "model": res["model"]}
    return out


# ------------------------------------------------------------------------------------------------ MOT17 tracking
MOT17_TRAIN = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN",
               "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def seqinfo(seq_dir: Path) -> dict:
    out = {}
    p = seq_dir / "seqinfo.ini"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def mot17_to_mp4(seq_dir: Path, dst: Path, fps: float, max_frames: int = 0, crf: int = 18) -> dict:
    """img1/%06d.jpg -> H.264 mp4 (one video frame per image, so frame k of the mp4 = MOT frame k)."""
    import subprocess
    from tools import datasets as D
    n_img = len(list((seq_dir / "img1").glob("*.jpg")))
    if not n_img and dst.exists():  # shipped mp4 + gt without the JPEG frames
        return {"frames": video_info(dst)["frames"], "converted": False}
    want = min(n_img, max_frames) if max_frames else n_img
    if dst.exists() and video_info(dst)["frames"] == want:
        return {"frames": want, "converted": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [D.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-start_number", "1",
           "-i", str(seq_dir / "img1" / "%06d.jpg"), "-frames:v", str(want), "-an",
           "-vf", D.video_filter("", None), *D.x264(crf), str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg failed for {seq_dir.name}: {r.stderr.strip()[-400:]}")
    os.replace(tmp, dst)
    return {"frames": video_info(dst)["frames"], "converted": True}


def profile_mot17(root: Path, a, cache: dict) -> dict:
    seqs = [s.strip() for s in a.mot_seqs.split(",") if s.strip()] if a.mot_seqs else MOT17_TRAIN
    settings = [s.strip() for s in a.mot_settings.split(",") if s.strip()]
    out = {"conventions": {
        "gt": "MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched "
              "(Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, "
              "distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing)"
              + (f"; pedestrians with visibility < {a.mot_min_vis} are also treated as distractors"
                 if a.mot_min_vis > 0 else "; no visibility threshold (as the official evaluation)"),
        "metrics": "CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; "
                   "MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts",
        "tracker": f"the app's production path: frames downscaled to analytics_width 960, detector (conf {DET['conf']}, "
                   "person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed "
                   "frames), person tracks only; no public detections are used (private-detector setting)",
        "settings": {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are "
                             "scored (GT subsampled the same way)",
                     "full": "every frame processed and scored"},
        "video": "img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline"},
        "sequences": {}, "combined": {}}
    for seq in seqs:
        sd = root / "mot17" / "raw" / "train" / seq
        if not (sd / "gt" / "gt.txt").exists():
            out["sequences"][seq] = {"error": f"missing {sd / 'gt' / 'gt.txt'} (use --fetch)"}
            continue
        si = seqinfo(sd)
        fps = float(si.get("frameRate", 30))
        mp4 = root / "mot17" / "mp4" / (f"{seq}_first{a.mot_max_frames}.mp4" if a.mot_max_frames else f"{seq}.mp4")
        conv = mot17_to_mp4(sd, mp4, fps, a.mot_max_frames)
        n = conv["frames"]
        gt_raw = M.parse_mot_gt((sd / "gt" / "gt.txt").read_text(), max_frame=n)
        so = {"fps": fps, "frames_scored_max": n, "seq_length": int(si.get("seqLength", 0) or 0),
              "resolution": f"{si.get('imWidth', '?')}x{si.get('imHeight', '?')}", "runs": {}}
        for setting in settings:
            afps = 5.0 if setting == "5fps" else fps
            spec = {"id": seq[:40], "path": mp4, "zones": []}
            res = run_clips([spec], analytics_fps=afps, det={"max_batch": 1})  # one camera: skip batch warm-ups
            cam = res["cameras"][spec["id"]]
            frames = sorted(f for f in cam["dets"] if 1 <= f <= n)
            tr = {f: [(tid, tuple(x * cam["dets"][f][1] for x in bb)) for tid, bb, cls, _ in cam["dets"][f][0]
                      if cls == "person"] for f in frames}
            if a.save_tracks:  # MOTChallenge result format, e.g. to cross-check with TrackEval / py-motmetrics
                a.save_tracks.mkdir(parents=True, exist_ok=True)
                with open(a.save_tracks / f"{seq}_{setting}.txt", "w") as fh:
                    for f in frames:
                        ds, sc = cam["dets"][f]
                        for tid, bb, cls, cf in ds:
                            if cls == "person":
                                x1, y1, x2, y2 = (v * sc for v in bb)
                                fh.write(f"{f},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{cf:.3f},-1,-1,-1\n")
            gt, trp, st = M.mot17_preprocess(gt_raw, tr, frames, min_vis=a.mot_min_vis)
            met = M.tracking_metrics(gt, trp)
            so["runs"][setting] = {"analytics_fps": round(afps, 3), "frames_scored": len(frames), **st, **met,
                                   "detect_ms_median": cam["detect_ms_median"], "wall_s": res["wall_s"],
                                   "processing_fps": round(len(frames) / max(1e-6, res["wall_s"]), 1),
                                   "model": res["model"]}
        out["sequences"][seq] = so
    for setting in settings:
        parts = [s["runs"][setting] for s in out["sequences"].values() if setting in s.get("runs", {})]
        if parts:
            out["combined"][setting] = {"sequences": len(parts), **M.combine_counts(parts)}
    for s in out["sequences"].values():  # drop the per-alpha HOTA arrays from the written JSON
        for r in s.get("runs", {}).values():
            r.pop("_hota_alpha", None)
    return out


# ------------------------------------------------------------------------------------------------ artifacts
def hardware(device: str) -> dict:
    hw = {"host": platform.node(), "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
          "cpu_count": os.cpu_count()}
    try:
        import torch
        hw["torch"] = torch.__version__
        if torch.cuda.is_available() and device != "cpu":
            hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return hw


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _ci(ci) -> str:
    return "" if not ci else f" ({ci[0]:.2f}-{ci[1]:.2f})"


def caviar_markdown(r: dict, meta: dict) -> str:
    L = [f"# CAVIAR rule precision / recall{' - SMOKE TEST, NOT A RESULT' if meta.get('smoke') else ''}", ""]
    if meta.get("smoke"):
        L += ["> **Smoke test, not a result.** Tiny sample run on a CPU laptop only to prove the script works "
              "end to end. Do not quote these numbers.", ""]
    L += [f"- Measured {meta['date']} on {meta['hardware'].get('gpu') or meta['hardware'].get('cpu')} "
          f"(host `{meta['hardware'].get('host')}`, {meta['hardware'].get('os')})",
          f"- Detector `{Path(str(meta['detector']['weights'])).name}` imgsz {meta['detector']['imgsz']} conf "
          f"{meta['detector']['conf']} + ByteTrack, analytics 5 fps, device {meta['detector']['device']}",
          f"- Space: **{r.get('space')}** ({r.get('calibration', {}).get('note', r.get('calibration'))})",
          f"- Rules: `{json.dumps(r.get('rules'))}`; altercation analyzer "
          f"{'on' if r.get('altercation_enabled') else 'off'}",
          f"- Matching: {r['matching']['rule']} Tolerance {r['matching']['temporal_tolerance_s']} s, proximity "
          f"{r['matching']['proximity_px']} px.",
          f"- Ground-truth episodes: {r['matching']['episodes']}",
          f"- Dataset: CAVIAR first set (EC Funded CAVIAR project/IST 2001 37540), clips: "
          f"{', '.join(r['runs']['colour']['per_clip'])}", ""]
    for vname, run in r["runs"].items():
        L += [f"## {vname}", "", "| Ground truth | Argus rule | Role | GT episodes | Events | Precision (Wilson 95 %) "
              "| Recall (Wilson 95 %) |", "|---|---|---|---|---|---|---|"]
        for s in run["scores"].values():
            rule = s["rule"] or "-"
            if s["rule"] and not s["rule_enabled"]:
                rule += " (not run)"
            L.append(f"| {s['behaviour']} | {rule} | {s['role']} | {s['gt_episodes']} | {s['events']} | "
                     f"{_pct(s['precision'])}{_ci(s['precision_wilson95'])} | "
                     f"{_pct(s['recall'])}{_ci(s['recall_wilson95'])} |")
        sm = run["summary"]
        L += ["", f"{sm['clips']} clips, {sm['duration_s']} s, {sm['tracks_10plus']} tracks (>= 10 samples), "
                  f"{sm['dets_per_frame']} person detections / frame, detect {sm['detect_ms_median']} ms median, "
                  f"wall {run['wall_s']} s. Events by rule: {json.dumps(sm['events_by_rule'])}", ""]
    return "\n".join(L) + "\n"


def mot17_markdown(r: dict, meta: dict) -> str:
    L = [f"# MOT17 tracking accuracy{' - SMOKE TEST, NOT A RESULT' if meta.get('smoke') else ''}", ""]
    if meta.get("smoke"):
        L += ["> **Smoke test, not a result.** Truncated sequence(s) on a CPU laptop, only to prove the script "
              "works end to end. Do not quote these numbers.", ""]
    L += [f"- Measured {meta['date']} on {meta['hardware'].get('gpu') or meta['hardware'].get('cpu')} "
          f"(host `{meta['hardware'].get('host')}`, {meta['hardware'].get('os')})",
          f"- Detector `{Path(str(meta['detector']['weights'])).name}` imgsz {meta['detector']['imgsz']} conf "
          f"{meta['detector']['conf']}, device {meta['detector']['device']}"]
    L += [f"- {k}: {v if isinstance(v, str) else json.dumps(v)}" for k, v in r["conventions"].items()]
    L += ["- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)", ""]
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "MOTP", "IDSW", "FP", "FN", "det_recall", "det_precision"]
    L += ["| Sequence | Setting | Frames scored | " + " | ".join(cols) + " | Detect ms |",
          "|---|---|---|" + "---|" * len(cols) + "---|"]
    for seq, s in r["sequences"].items():
        if "error" in s:
            L.append(f"| {seq} | - | {s['error']} |" + " |" * (len(cols) + 1))
            continue
        for setting, m in s["runs"].items():
            L.append(f"| {seq} | {setting} | {m['frames_scored']} | " + " | ".join(str(m.get(c)) for c in cols)
                     + f" | {m.get('detect_ms_median')} |")
    for setting, m in r["combined"].items():
        L.append(f"| **combined ({m['sequences']} seq)** | {setting} | - | "
                 + " | ".join(str(m.get(c)) for c in cols) + " | - |")
    return "\n".join(L) + "\n"


def write_artifacts(out_dir: Path, name: str, r: dict, meta: dict, md) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"meta": meta, **r}, indent=1, default=str), encoding="utf-8")
    if "error" not in r:
        (out_dir / f"{name}.md").write_text(md(r, meta), encoding="utf-8")
    print(f"wrote {out_dir / name}.json/.md", flush=True)


# ------------------------------------------------------------------------------------------------ fetching
def fetch_caviar(root: Path, clips: Optional[str]) -> None:
    from tools import datasets as D
    ds = next(x for x in D.load_manifest()["datasets"] if x["id"] == "caviar")
    if clips:
        want = {c.strip() for c in clips.split(",") if c.strip()}
        ds = {**ds, "fetch": [s for s in ds["fetch"] if Path(s["dest"]).stem in want]}
    st = D.State(root)
    with D.Lock(root / "caviar" / ".lock"):
        D.fetch_dataset(ds, root, st, False, [])
        D.convert_dataset(ds, root, st, D.ffmpeg_exe())


def fetch_mot17(root: Path, seqs: list[str], max_frames: int) -> None:
    """Only the selected sequences (and, with max_frames, only their first frames) out of the 5.5 GB MOT17.zip,
    through HTTP Range requests (datasets.RemoteZip)."""
    from tools import datasets as D
    ds = next(x for x in D.load_manifest()["datasets"] if x["id"] == "mot17")
    base = ds["fetch"][0]
    out, strip = root / base["dest_dir"], int(base["strip"])

    def local(n: str) -> Path:
        return out / Path(*Path(n).parts[strip:])
    with D.Lock(root / "mot17" / ".lock"):
        rz = D.RemoteZip(base["zip_url"])
        names = []
        for s in seqs:
            p = f"MOT17/train/{s}/"
            seq_names = sorted(n for n in rz.entries if n.startswith(p) and not n.endswith("/"))
            if not seq_names:
                raise RuntimeError(f"{s} not found in {base['zip_url']}")
            if max_frames:
                keep = {f"{p}img1/{i:06d}.jpg" for i in range(1, max_frames + 1)}
                seq_names = [n for n in seq_names if "/img1/" not in n or n in keep]
            names += seq_names
        todo = [n for n in names if not (local(n).exists() and local(n).stat().st_size == rz.entries[n]["usize"])]
        print(f"  mot17: {len(names)} members ({D.human(sum(rz.entries[n]['usize'] for n in names))}) of "
              f"{D.human(rz.size)} MOT17.zip, {len(todo)} to fetch", flush=True)
        todo.sort(key=lambda n: rz.entries[n]["offset"])
        for i in range(0, len(todo), 200):  # bounded Range requests (~40 MB of 1080p JPEGs each)
            rz.extract(todo[i:i + 200], out, strip)
            print(f"  mot17: {min(i + 200, len(todo))}/{len(todo)} members", flush=True)


def profile_frames_anomaly(root: Path, a, cache: dict, name: str) -> dict:
    """UCSD Ped1/Ped2 and CUHK Avenue: frame-level anomaly labels vs frames covered by any Argus event."""
    out = {"subsets": {}}
    if name == "ucsd":
        sets = []
        for ped in ("ped1", "ped2"):
            allp = sorted((root / "ucsd" / "mp4" / ped).glob("*.mp4"))
            test = [p for p in allp if p.stem.startswith("Test") and not p.stem.endswith("_gray")]
            train = [p for p in allp if p.stem.startswith("Train") and not p.stem.endswith("_gray")]
            mf = next(iter(glob.glob(str(root / "ucsd" / "raw" / "**" / f"UCSD{ped}.m"), recursive=True)), None)
            gt = ucsd_gt(Path(mf)) if mf else []
            num = {p.stem: int(re.sub(r"\D", "", p.stem)) for p in test}
            sets.append((ped, test, train, {s: gt[i - 1] for s, i in num.items() if 0 < i <= len(gt)}))
    else:
        test = sorted(p for p in (root / "avenue" / "mp4" / "testing").glob("*.mp4") if not p.stem.endswith("_gray"))
        train = sorted((root / "avenue" / "mp4" / "training").glob("*.mp4"))
        gtd = next(iter(glob.glob(str(root / "avenue" / "raw" / "**" / "testing_label_mask"), recursive=True)), None)
        gt = {p.stem: avenue_gt(Path(gtd) / f"{int(p.stem)}_label.mat") for p in test
              if gtd and (Path(gtd) / f"{int(p.stem)}_label.mat").exists()}
        sets = [("avenue", test, train, gt)]
    for sub, test, train, gt in sets:
        if a.max_clips:
            test = test[:a.max_clips]
        if not test:
            out["subsets"][sub] = {"error": "no converted test clips"}
            continue
        calib, cinfo = (None, {"state": "skipped"}) if a.no_calib else _calib_for(train or test, a.hfov, cache, sub)
        rules = {"running": {"speed_mps": 2.5, "min_seconds": 1.0}, "loitering": {"min_seconds": 20},
                 "crowding": {"max_count": 8, "min_seconds": 3}}
        specs = [_spec(p, calib, rules, prefix=f"{sub}_") for p in test]
        res = run_clips(specs)
        g, pr, tot, by_rule = {}, {}, {}, defaultdict(dict)
        for s, p in zip(specs, test):
            cam = res["cameras"][s["id"]]
            n = cam["video"]["frames"]
            tot[s["id"]] = n
            g[s["id"]] = gt.get(p.stem, set())
            flagged = set()
            for e in cam["events"]:
                if e["rule"] == "camera_health":
                    continue
                fr = set(range(max(1, e["f0"]), min(n, e["f1"]) + 1))
                flagged |= fr
                by_rule[e["rule"]].setdefault(s["id"], set()).update(fr)
            pr[s["id"]] = flagged
        out["subsets"][sub] = {"calibration": cinfo, "rules": rules, "summary": summarize(res, list(tot)),
                               "frame_level_any_event": frame_level(g, pr, tot),
                               "frame_level_by_rule": {r: frame_level(g, v, tot) for r, v in by_rule.items()},
                               "labelled_clips": sum(1 for x in g.values() if x), "wall_s": res["wall_s"],
                               "model": res["model"]}
    return out


def profile_mot(root: Path, a, cache: dict) -> dict:
    out = {"sequences": {}}
    for seq in sorted(glob.glob(str(root / "mot" / "raw" / "**" / "gt" / "gt.txt"), recursive=True)):
        sd = Path(seq).parent.parent
        mp4 = root / "mot" / "mp4" / f"{sd.name}.mp4"
        if not mp4.exists():
            continue
        variants = {"colour": mp4}
        if mp4.with_name(mp4.stem + "_gray.mp4").exists():
            variants["gray"] = mp4.with_name(mp4.stem + "_gray.mp4")
        gtb = mot_gt(sd)
        seq_out = {"gt_identities": mot_ids(sd), "runs": {}}
        for vname, p in variants.items():
            info = video_info(p)
            spec = {"id": sd.name[:40], "path": p,
                    "zones": [full_view_zone(info["w"], info["h"], {"crowding": {"max_count": 10, "min_seconds": 3}})]}
            res = run_clips([spec])
            cam = res["cameras"][spec["id"]]
            seq_out["runs"][vname] = {"summary": summarize(res, [spec["id"]]),
                                      "det_vs_gt_iou50": det_match(gtb, cam),
                                      "det_vs_gt_iou50_h50px": det_match(gtb, cam, min_h=50),
                                      "tracks_per_gt_identity": round(cam["tracks_10plus"] / max(1, seq_out["gt_identities"]), 2),
                                      "wall_s": res["wall_s"], "model": res["model"]}
        out["sequences"][sd.name] = seq_out
    return out


def profile_generic(root: Path, a, cache: dict, name: str, groups: dict, rules: dict) -> dict:
    """Datasets without behaviour labels: tracks, detections, speeds (auto-calibrated per camera), events."""
    out = {"groups": {}}
    for gname, patterns in groups.items():
        paths = sorted({Path(p) for pat in patterns for p in glob.glob(str(root / name / "mp4" / pat))})
        if a.max_clips:
            paths = paths[:a.max_clips]
        if not paths:
            continue
        specs, calibs = [], {}
        for p in paths:
            base = p.with_name(p.stem[:-5] + ".mp4") if p.stem.endswith("_gray") else p
            calib, cinfo = (None, {"state": "skipped"}) if a.no_calib else \
                _calib_for([base], a.hfov, cache, f"{name}:{base.stem}")
            calibs[p.stem] = cinfo
            specs.append(_spec(p, calib, rules))
        res = run_clips(specs)
        out["groups"][gname] = {"summary": summarize(res, [s["id"] for s in specs]), "calibration": calibs,
                                "per_clip": {s["id"]: {k: res["cameras"][s["id"]][k] for k in
                                                       ("video", "tracks_10plus", "dets_per_frame", "mean_det_conf",
                                                        "events_by_rule", "median_track_speed_mps",
                                                        "moving_tracks_median_speed_mps", "calibrated", "monochrome")}
                                             for s in specs},
                                "wall_s": res["wall_s"], "model": res["model"]}
    return out


RULES_GENERIC = {"running": {"speed_mps": 2.5, "min_seconds": 1.0}, "loitering": {"min_seconds": 20},
                 "crowding": {"max_count": 5, "min_seconds": 3}}
GENERIC = {  # dataset -> (clip groups by glob under <root>/<id>/mp4, rules)
    "pets2009": ({"S2L1": ["S2_L1_View_00[1-8].mp4"], "S2L1_gray": ["S2_L1_View_001_gray.mp4"],
                  "dense": ["S2_L[23]_View_001.mp4"]}, RULES_GENERIC),
    "epfl": ({"terrace": ["terrace1-c[0-3].mp4"], "passageway": ["passageway1-c[0-3].mp4"],
              "laboratory": ["6p-c[0-3].mp4"], "gray": ["*_gray.mp4"]}, RULES_GENERIC),
    "virat": ({"ground": ["VIRAT_S_*[0-9].mp4"], "gray": ["VIRAT_S_*_gray.mp4"]}, RULES_GENERIC),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.home() / "argonix_datasets")
    ap.add_argument("--only", default="caviar,ucsd,avenue,mot,pets2009,epfl,virat")
    ap.add_argument("--weights", default=None, help="detector weights (default: data/models/yolo26s.pt if present)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3, help="detector confidence cut before ByteTrack (default 0.3)")
    ap.add_argument("--stitch", action="store_true", help="re-link briefly lost tracks (argonix/stitch.py)")
    ap.add_argument("--stitch-cfg", default="", help='stitcher settings as JSON, e.g. {"max_gap_frames": 8}')
    ap.add_argument("--hfov", type=float, default=None, help="assumed lens HFOV for auto-calibration (default: none)")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--threats", action="store_true", help="enable the altercation analyzer on CAVIAR")
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="also write every profile's raw result to this JSON")
    ap.add_argument("--device", default="auto", help="detector device: auto, cpu, cuda:0, ...")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch + convert what the caviar / mot17 profiles need first (datasets.py, manifest)")
    # accuracy profiles: caviar (rules) and mot17 (tracking)
    ap.add_argument("--clips", default="", help="caviar: comma-separated clip stems (default: every clip)")
    ap.add_argument("--caviar-space", choices=["px", "auto"], default="px",
                    help="caviar: image-pixel zone/thresholds (default; CAVIAR has no calibration) or the "
                         "automatic pedestrian self-calibration (needs --hfov)")
    ap.add_argument("--gray", action="store_true", help="caviar: also score the grayscale variants")
    ap.add_argument("--tol-s", type=float, default=2.0, help="caviar: temporal matching tolerance (s)")
    ap.add_argument("--prox-px", type=float, default=30.0, help="caviar: spatial matching radius (source px)")
    ap.add_argument("--mot-seqs", default="", help="mot17: sequences (default: the 7 FRCNN train sequences)")
    ap.add_argument("--mot-max-frames", type=int, default=0, help="mot17: only the first N frames (smoke tests)")
    ap.add_argument("--mot-settings", default="5fps,full")
    ap.add_argument("--mot-min-vis", type=float, default=0.0)
    ap.add_argument("--save-tracks", type=Path, default=None,
                    help="mot17: also write our tracks in MOTChallenge format (<seq>_<setting>.txt) to this dir")
    ap.add_argument("--label", default=None, help="eval_runs/<date>-<label>/ (default: host name)")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "eval_runs")
    ap.add_argument("--smoke", action="store_true", help="mark the artifacts 'smoke test, not a result'")
    a = ap.parse_args(argv)
    w = a.weights or next((str(p) for p in (ROOT / "data" / "models" / "yolo26s.pt",
                                            ROOT / "data" / "models" / "yolo26n.pt") if p.exists()), "auto")
    DET.update(weights=w, imgsz=a.imgsz, device=a.device, conf=a.conf)
    if a.stitch:
        DET["stitch"] = {"enabled": True, **(json.loads(a.stitch_cfg) if a.stitch_cfg else {})}
    a.root = a.root.expanduser()
    date = time.strftime("%Y-%m-%d")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", a.label or platform.node().lower() or "host")
    out_dir = a.runs_dir / f"{date}-{label}"
    meta = {"date": date, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "hardware": hardware(a.device),
            "detector": dict(DET), "smoke": a.smoke, "command": " ".join(sys.argv)}
    results = {"started": meta["started"], "detector": dict(DET), "datasets": {}}
    cache: dict = {}
    names = [x.strip() for x in a.only.split(",") if x.strip()]
    if a.fetch:
        if "caviar" in names:
            fetch_caviar(a.root, a.clips)
        if "mot17" in names:
            fetch_mot17(a.root, [s.strip() for s in a.mot_seqs.split(",") if s.strip()] or MOT17_TRAIN,
                        a.mot_max_frames)
    for name in names:
        t0 = time.time()
        print(f"== {name}", flush=True)
        try:
            if name == "caviar":
                r = profile_caviar(a.root, a, cache)
                r["eval_wall_s"] = round(time.time() - t0, 1)
                write_artifacts(out_dir, "caviar_rules", r, meta, caviar_markdown)
            elif name == "mot17":
                r = profile_mot17(a.root, a, cache)
                r["eval_wall_s"] = round(time.time() - t0, 1)
                write_artifacts(out_dir, "mot17_tracking", r, meta, mot17_markdown)
            elif name in ("ucsd", "avenue"):
                r = profile_frames_anomaly(a.root, a, cache, name)
            elif name == "mot":
                r = profile_mot(a.root, a, cache)
            elif name in GENERIC:
                r = profile_generic(a.root, a, cache, name, *GENERIC[name])
            else:
                r = {"error": f"no profile for {name}"}
        except Exception:  # noqa: BLE001 - keep going, report the failure
            import traceback
            r = {"error": traceback.format_exc()[-1500:]}
        r["eval_wall_s"] = round(time.time() - t0, 1)
        results["datasets"][name] = r
        print(json.dumps(r, indent=1, default=str)[:3000], flush=True)
        if a.out:
            a.out.write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
