"""Accuracy metrics for tools/eval_datasets.py: multi-object tracking (CLEAR-MOT, IDF1, HOTA) and event-level
interval matching with Wilson intervals. Pure numpy; the assignment solver is scipy's when installed, else `lap`
(a project dependency).

Tracking metrics follow the TrackEval reference implementation (Luiten & Hoffhues, github.com/JonathonLuiten/
TrackEval, MIT) so numbers are comparable with published MOT17 results:
  * CLEAR-MOT (Bernardin & Stiefelhagen 2008): per frame, a GT keeps last frame's tracker ID when IoU >= 0.5
    (bonus in the assignment), the rest is a Hungarian assignment on IoU >= 0.5. An ID switch is counted when a GT is
    matched to a different tracker ID than the last one it was matched to (at any earlier frame).
    MOTA = 1 - (FN + FP + IDSW) / GT.  MOTP = mean IoU of matched pairs (TrackEval; py-motmetrics reports 1 - IoU).
    MT/PT/ML: GT trajectories tracked > 80 % / 20-80 % / < 20 % of their frames. Frag: tracked-again count.
  * Identity (Ristani et al. 2016): one global one-to-one assignment of GT IDs to tracker IDs maximising the number
    of frames where the pair overlaps with IoU >= 0.5.  IDF1 = 2 IDTP / (GT dets + tracker dets).
  * HOTA (Luiten et al. 2021, IJCV, doi 10.1007/s11263-020-01375-2): mean over alpha = 0.05 ... 0.95 of
    sqrt(DetA * AssA), with the per-frame assignment weighted by the global alignment score.
  * MOT17 preprocessing (TrackEval `MotChallenge2DBox`): tracker boxes matched (IoU >= 0.5, Hungarian) to a GT box of
    a distractor class (2 person on vehicle, 7 static person, 8 distractor, 12 reflection) are removed; then only GT
    rows with class 1 (pedestrian) and the 'consider' flag set are kept. No visibility threshold, as in the official
    evaluation (an optional `min_vis` treats less visible pedestrians like distractors: stated when used).

Frames are dicts {frame: [(id, (x1, y1, x2, y2)), ...]} over the frames that are evaluated; a frame with GT but no
entry in the tracker dict counts as "tracker output nothing".
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

EPS = np.finfo(float).eps
MOT17_PEDESTRIAN = 1
MOT17_DISTRACTORS = (2, 7, 8, 12)
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)


# ------------------------------------------------------------------------------------------------ basics
def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson score 95 % interval for k successes of n (None when n == 0). Same formula as argonix.feedback."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment on a (possibly rectangular) matrix -> (rows, cols)."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.empty(0, int), np.empty(0, int)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return np.asarray(r), np.asarray(c)
    except ImportError:
        import lap
        _, x, _ = lap.lapjv(cost, extend_cost=True)
        rows = np.array([i for i, j in enumerate(x) if j >= 0], int)
        return rows, np.array([x[i] for i in rows], int)


def iou_matrix(a, b) -> np.ndarray:
    """IoU between xyxy boxes a (n, 4) and b (m, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, EPS), 0.0)


def _frame_arrays(frames: dict, f) -> tuple[list, np.ndarray]:
    rows = frames.get(f, [])
    return [r[0] for r in rows], np.array([r[1] for r in rows], dtype=float).reshape(-1, 4)


# ------------------------------------------------------------------------------------------------ MOT17 GT
def parse_mot_gt(text: str, max_frame: Optional[int] = None) -> dict[int, list]:
    """MOTChallenge gt.txt -> {frame: [(id, xyxy, cls, consider, visibility)]} (every row, all classes)."""
    out: dict = defaultdict(list)
    for line in text.splitlines():
        v = line.strip().split(",")
        if len(v) < 6:
            continue
        f, tid = int(float(v[0])), int(float(v[1]))
        if max_frame and f > max_frame:
            continue
        x, y, w, h = map(float, v[2:6])
        consider = int(float(v[6])) if len(v) > 6 else 1
        cls = int(float(v[7])) if len(v) > 7 else 1
        vis = float(v[8]) if len(v) > 8 else 1.0
        out[f].append((tid, (x, y, x + w, y + h), cls, consider, vis))
    return dict(out)


def mot17_preprocess(gt_raw: dict[int, list], tracker: dict[int, list], frames: Iterable[int],
                     thr: float = 0.5, min_vis: float = 0.0) -> tuple[dict, dict, dict]:
    """TrackEval MOT17 preprocessing on the evaluated frames -> (gt, tracker, stats).

    Tracker boxes matched to a distractor-class GT box (or, with min_vis > 0, to a pedestrian less visible than
    min_vis) are removed; GT keeps class-1 rows with consider == 1 (and visibility >= min_vis)."""
    gt_out, tr_out = {}, {}
    removed = 0
    for f in frames:
        g = gt_raw.get(f, [])
        t = tracker.get(f, [])
        drop = set()
        if g and t:
            sim = iou_matrix([r[1] for r in g], [r[1] for r in t])
            score = np.where(sim >= thr - EPS, sim, 0.0)
            rows, cols = linear_assignment(-score)
            for r, c in zip(rows, cols):
                if score[r, c] > EPS:
                    cls, vis = g[r][2], g[r][4]
                    if cls in MOT17_DISTRACTORS or (min_vis > 0 and cls == MOT17_PEDESTRIAN and vis < min_vis):
                        drop.add(c)
        removed += len(drop)
        tr_out[f] = [(r[0], r[1]) for i, r in enumerate(t) if i not in drop]
        gt_out[f] = [(r[0], r[1]) for r in g if r[2] == MOT17_PEDESTRIAN and r[3] != 0 and r[4] >= min_vis]
    return gt_out, tr_out, {"tracker_boxes_removed_on_distractors": removed}


# ------------------------------------------------------------------------------------------------ CLEAR-MOT
def clear_mot(gt: dict, tr: dict, thr: float = 0.5) -> dict:
    frames = sorted(set(gt) | set(tr))
    gt_ids = sorted({i for f in frames for i, _ in gt.get(f, [])})
    gidx = {g: k for k, g in enumerate(gt_ids)}
    n_g = len(gt_ids)
    prev_tr = np.full(n_g, np.nan)          # last tracker id ever matched to each GT
    prev_step = np.full(n_g, np.nan)        # tracker id matched in the previous evaluated frame
    gt_count = np.zeros(n_g)
    matched_count = np.zeros(n_g)
    frag = np.zeros(n_g)
    tp = fp = fn = idsw = 0
    motp_sum = 0.0
    for f in frames:
        g_ids, g_box = _frame_arrays(gt, f)
        t_ids, t_box = _frame_arrays(tr, f)
        gi = np.array([gidx[i] for i in g_ids], int)
        if not len(g_ids):  # TrackEval keeps the previous-step correspondences across such frames
            fp += len(t_ids)
            continue
        gt_count[gi] += 1
        if not len(t_ids):
            fn += len(g_ids)
            continue
        sim = iou_matrix(g_box, t_box)
        t_arr = np.array(t_ids, dtype=float)
        score = 1000.0 * (t_arr[None, :] == prev_step[gi][:, None]) + sim
        score[sim < thr - EPS] = 0.0
        rows, cols = linear_assignment(-score)
        ok = score[rows, cols] > EPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = gi[rows], t_arr[cols]
        prev_m = prev_tr[mg]
        idsw += int(np.sum(~np.isnan(prev_m) & (mt != prev_m)))
        matched_count[mg] += 1
        not_prev = np.isnan(prev_step)
        prev_tr[mg] = mt
        prev_step[:] = np.nan
        prev_step[mg] = mt
        frag += not_prev & ~np.isnan(prev_step)
        n = len(rows)
        tp += n
        fn += len(g_ids) - n
        fp += len(t_ids) - n
        motp_sum += float(np.sum(sim[rows, cols]))
    present = gt_count > 0
    ratio = matched_count[present] / gt_count[present]
    mt_n = int(np.sum(ratio > 0.8))
    pt_n = int(np.sum(ratio >= 0.2)) - mt_n
    n_gt_dets = tp + fn
    return {"gt_dets": n_gt_dets, "tracker_dets": tp + fp, "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw,
            "MOTA": round((tp - fp - idsw) / max(1, n_gt_dets), 4), "MOTP": round(motp_sum / max(1, tp), 4),
            "gt_ids": int(np.sum(present)), "MT": mt_n, "PT": pt_n, "ML": int(np.sum(present)) - mt_n - pt_n,
            "Frag": int(np.sum(np.clip(frag[frag > 0] - 1, 0, None))),
            "det_recall": round(tp / max(1, n_gt_dets), 4), "det_precision": round(tp / max(1, tp + fp), 4)}


# ------------------------------------------------------------------------------------------------ identity
def _ids(gt: dict, tr: dict):
    frames = sorted(set(gt) | set(tr))
    g_ids = sorted({i for f in frames for i, _ in gt.get(f, [])})
    t_ids = sorted({i for f in frames for i, _ in tr.get(f, [])})
    return frames, {g: k for k, g in enumerate(g_ids)}, {t: k for k, t in enumerate(t_ids)}


def identity(gt: dict, tr: dict, thr: float = 0.5) -> dict:
    frames, gidx, tidx = _ids(gt, tr)
    pot = np.zeros((len(gidx), len(tidx)))
    n_gt = n_tr = 0
    for f in frames:
        g_ids, g_box = _frame_arrays(gt, f)
        t_ids, t_box = _frame_arrays(tr, f)
        n_gt += len(g_ids)
        n_tr += len(t_ids)
        if g_ids and t_ids:
            sim = iou_matrix(g_box, t_box)
            gi = np.array([gidx[i] for i in g_ids])
            ti = np.array([tidx[i] for i in t_ids])
            pot[gi[:, None], ti[None, :]] += sim >= thr - EPS
    idtp = 0
    if pot.size:
        r, c = linear_assignment(-pot)
        idtp = int(round(pot[r, c].sum()))
    idfn, idfp = n_gt - idtp, n_tr - idtp
    return {"IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
            "IDP": round(idtp / max(1, idtp + idfp), 4), "IDR": round(idtp / max(1, idtp + idfn), 4),
            "IDF1": round(2 * idtp / max(1, n_gt + n_tr), 4)}


# ------------------------------------------------------------------------------------------------ HOTA
def hota(gt: dict, tr: dict) -> dict:
    frames, gidx, tidx = _ids(gt, tr)
    ng, nt, na = len(gidx), len(tidx), len(HOTA_ALPHAS)
    pot = np.zeros((ng, nt))
    g_cnt, t_cnt = np.zeros(ng), np.zeros(nt)
    cache = []
    for f in frames:
        g_ids, g_box = _frame_arrays(gt, f)
        t_ids, t_box = _frame_arrays(tr, f)
        gi = np.array([gidx[i] for i in g_ids], int)
        ti = np.array([tidx[i] for i in t_ids], int)
        sim = iou_matrix(g_box, t_box)
        if len(gi) and len(ti):
            den = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(den > EPS, sim / np.maximum(den, EPS), 0.0)
            pot[gi[:, None], ti[None, :]] += sim_iou
        g_cnt[gi] += 1
        t_cnt[ti] += 1
        cache.append((gi, ti, sim))
    gas = pot / np.maximum(EPS, g_cnt[:, None] + t_cnt[None, :] - pot) if pot.size else pot
    TP, FN, FP, loc = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    mc = np.zeros((na, ng, nt))
    for gi, ti, sim in cache:
        if not len(gi):
            FP += len(ti)
            continue
        if not len(ti):
            FN += len(gi)
            continue
        score = gas[gi[:, None], ti[None, :]] * sim
        rows, cols = linear_assignment(-score)
        for a, alpha in enumerate(HOTA_ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            TP[a] += len(r)
            FN[a] += len(gi) - len(r)
            FP[a] += len(ti) - len(r)
            if len(r):
                loc[a] += sim[r, c].sum()
                mc[a, gi[r], ti[c]] += 1
    ass = np.zeros(na)
    for a in range(na):
        m = mc[a]
        ass_a = m / np.maximum(1, g_cnt[:, None] + t_cnt[None, :] - m)
        ass[a] = (m * ass_a).sum() / max(1, TP[a])
    det = TP / np.maximum(1, TP + FN + FP)
    hota_a = np.sqrt(det * ass)
    loca = np.maximum(1e-10, loc) / np.maximum(1e-10, TP)
    return {"HOTA": round(float(hota_a.mean()), 4), "DetA": round(float(det.mean()), 4),
            "AssA": round(float(ass.mean()), 4), "LocA": round(float(loca.mean()), 4),
            "HOTA(0.5)": round(float(hota_a[int(np.argmin(np.abs(HOTA_ALPHAS - 0.5)))]), 4),
            # per-alpha sums for pooling over sequences (combine_counts); dropped from the written JSON
            "_hota_alpha": {"TP": TP.tolist(), "FN": FN.tolist(), "FP": FP.tolist(), "AssA": ass.tolist(),
                            "loc": loc.tolist()}}


def pool_hota(parts: list[dict]) -> dict:
    """TrackEval's sequence combination: sum TP/FN/FP per alpha, AssA weighted by TP, then HOTA per alpha."""
    al = [p["_hota_alpha"] for p in parts if "_hota_alpha" in p]
    if not al:
        return {}
    TP = np.sum([a["TP"] for a in al], 0)
    FN = np.sum([a["FN"] for a in al], 0)
    FP = np.sum([a["FP"] for a in al], 0)
    ass = np.sum([np.array(a["AssA"]) * np.array(a["TP"]) for a in al], 0) / np.maximum(1, TP)
    loc = np.sum([a["loc"] for a in al], 0)
    det = TP / np.maximum(1, TP + FN + FP)
    h = np.sqrt(det * ass)
    return {"HOTA": round(float(h.mean()), 4), "DetA": round(float(det.mean()), 4),
            "AssA": round(float(ass.mean()), 4),
            "LocA": round(float((np.maximum(1e-10, loc) / np.maximum(1e-10, TP)).mean()), 4)}


def tracking_metrics(gt: dict, tr: dict) -> dict:
    return {**clear_mot(gt, tr), **identity(gt, tr), **hota(gt, tr)}


def combine_counts(parts: list[dict]) -> dict:
    """Pool CLEAR/identity counts over sequences (MOTA/IDF1 from summed counts, as MOTChallenge 'combined')."""
    s = defaultdict(float)
    for p in parts:
        for k in ("gt_dets", "tracker_dets", "TP", "FP", "FN", "IDSW", "IDTP", "IDFP", "IDFN", "MT", "PT", "ML",
                  "gt_ids", "Frag"):
            s[k] += p.get(k, 0)
    motp = sum(p["MOTP"] * p["TP"] for p in parts) / max(1, s["TP"])
    out = {k: int(v) for k, v in s.items()}
    out.update(MOTA=round((s["TP"] - s["FP"] - s["IDSW"]) / max(1, s["gt_dets"]), 4), MOTP=round(motp, 4),
               IDF1=round(2 * s["IDTP"] / max(1, s["gt_dets"] + s["tracker_dets"]), 4),
               IDP=round(s["IDTP"] / max(1, s["IDTP"] + s["IDFP"]), 4),
               IDR=round(s["IDTP"] / max(1, s["IDTP"] + s["IDFN"]), 4),
               det_recall=round(s["TP"] / max(1, s["gt_dets"]), 4),
               det_precision=round(s["TP"] / max(1, s["tracker_dets"]), 4),
               **pool_hota(parts))
    return out


# ------------------------------------------------------------------------------------------------ events
def box_distance(box, pt_box) -> float:
    """Distance from the centre of pt_box to the rectangle `box` (0 when inside)."""
    cx, cy = (pt_box[0] + pt_box[2]) / 2, (pt_box[1] + pt_box[3]) / 2
    dx = max(box[0] - cx, 0.0, cx - box[2])
    dy = max(box[1] - cy, 0.0, cy - box[3])
    return math.hypot(dx, dy)


def event_matches_episode(ev: dict, ep: dict, tol: int, prox_px: Optional[float]) -> bool:
    """Temporal overlap within +-tol frames, and (when prox_px is set and both sides have positions) some event
    track box centre within prox_px of the GT box in a shared frame of the overlap window.

    ev: {"clip", "f0", "f1", "boxes": {frame: [xyxy, ...]}}; ep: {"clip", "f0", "f1", "boxes": {frame: [xyxy, ...]}}"""
    if ev["clip"] != ep["clip"] or not (ev["f0"] - tol <= ep["f1"] and ep["f0"] - tol <= ev["f1"]):
        return False
    if prox_px is None or not ev.get("boxes") or not ep.get("boxes"):
        return True
    lo, hi = max(ev["f0"], ep["f0"]) - tol, min(ev["f1"], ep["f1"]) + tol
    gtf = sorted(f for f in ep["boxes"] if lo <= f <= hi)
    if not gtf:
        return False
    for f, pboxes in ev["boxes"].items():
        if not lo <= f <= hi:
            continue
        # nearest annotated frame (CAVIAR annotates every frame; analytics frames are a subset)
        g = min(gtf, key=lambda x: abs(x - f))
        if abs(g - f) > tol:
            continue
        for gb in ep["boxes"][g]:
            if any(box_distance(gb, pb) <= prox_px for pb in pboxes):
                return True
    return False


def match_events(gt_eps: list[dict], preds: list[dict], tol: int, prox_px: Optional[float]) -> dict:
    """Event-level precision/recall. A prediction is a true positive when it matches ANY GT episode; a GT episode
    is detected when ANY prediction matches it (many-to-one allowed both ways, reported as such)."""
    hit_gt = [False] * len(gt_eps)
    tp_pred = 0
    for p in preds:
        m = False
        for i, g in enumerate(gt_eps):
            if event_matches_episode(p, g, tol, prox_px):
                hit_gt[i] = m = True
        tp_pred += m
    tp_gt = sum(hit_gt)
    return {"gt_episodes": len(gt_eps), "events": len(preds), "true_events": tp_pred, "detected_episodes": tp_gt,
            "precision": round(tp_pred / len(preds), 3) if preds else None,
            "precision_wilson95": wilson(tp_pred, len(preds)),
            "recall": round(tp_gt / len(gt_eps), 3) if gt_eps else None,
            "recall_wilson95": wilson(tp_gt, len(gt_eps))}
