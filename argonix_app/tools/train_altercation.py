"""Pose sequences -> small pair GRU for "possible physical altercation" (run on the GPU host).

    uv run python tools/train_altercation.py extract --datasets ../datasets --only rwf_train   # pose + tracks, cached
    uv run python tools/train_altercation.py extract --datasets ../datasets --only rwf_val,hockey
    uv run python tools/train_altercation.py train                                            # GRU + heuristic, metrics

Data (Kaggle):
- rmilcb/rwf-2000-10fps: RWF-2000 (Cheng et al., ICPR 2020; surveillance-camera clips, 5 s) re-encoded at 10 fps.
  Official train (1600 clips) / val (400 clips) split; source-video ids do not overlap between them (checked).
  We split TRAIN by source-video id 85/15 into train/val (model selection, threshold); official VAL is the held-out
  TEST set. Licence on Kaggle: "unknown"; RWF-2000 is distributed for research use.
- yassershrief/hockey-fight-vidoes: Hockey Fights (Bermejo Nievas et al., CAIP 2011), 1000 clips of ~1.6 s.
  Used only as an out-of-domain test (broadcast ice hockey, not CCTV). Licence "unknown".

Weak labels: a clip is labelled Fight/NonFight, not a pair. Training therefore uses multiple-instance learning: every
candidate pair window in a clip is scored and the clip logit is a soft maximum (LogSumExp, r=5) over its windows.
Clips with no candidate pair window (people never within `pair_max_dist` body-heights, or pose/tracking failed) are
dropped from training and count as "no alert" (score 0) at evaluation, which is what the runtime would do.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.altercation import (FEATURE_VERSION, N_FEAT, build_gru, flip_lr, heuristic_score,  # noqa: E402
                                 pair_features, resample_track, window_candidate)

CACHE = ROOT / "data" / "altercation_cache"


# ---------------------------------------------------------------------------------------------- extraction
def list_clips(datasets: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"rwf_train": [], "rwf_val": [], "hockey": []}
    rwf = datasets / "rwf-2000-10fps"
    for split, key in (("train", "rwf_train"), ("val", "rwf_val")):
        for lab in ("Fight", "NonFight"):
            for p in sorted(list((rwf / split / lab).glob("*.avi")) + list((rwf / split / lab).glob("*.mp4"))):
                out[key].append({"path": str(p), "name": p.stem, "label": int(lab == "Fight"),
                                 "src": p.stem.rsplit("_", 1)[0]})
    hk = datasets / "hockey-fight-vidoes" / "data"
    for p in sorted(list(hk.glob("*.avi")) + list(hk.glob("*.mp4"))):
        out["hockey"].append({"path": str(p), "name": p.stem, "label": int(p.name.startswith("fi")), "src": p.stem})
    return out


def shrink(a) -> None:
    """Re-encode clips to --target-fps and <= --max-w px wide (H.264) into --out, keeping the dataset layout.

    For hosts with a slow link: the pose model runs at imgsz 640 and the runtime at 5 fps, so a 640-px, 5-fps copy
    carries what extraction uses (up to resampling differences) at a fraction of the size. --limit keeps a
    deterministic subset of clips per label and split."""
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src, out = Path(a.datasets), Path(a.out)
    clips = list_clips(src)
    n, t0 = 0, time.time()
    for key in a.only.split(","):
        sel = clips[key]
        if a.limit:
            rng = random.Random(0)
            sel = []
            for lab in (0, 1):
                cl = [c for c in clips[key] if c["label"] == lab]
                rng.shuffle(cl)
                sel += cl[:a.limit]
        jobs = []
        for c in sel:
            rel = Path(c["path"]).relative_to(src)
            dst = (out / rel).with_suffix(".mp4")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                jobs.append([ff, "-v", "error", "-y", "-threads", "1", "-i", c["path"], "-vf",
                             f"fps={a.target_fps},scale='min({a.max_w},iw)':-2", "-c:v", "libx264", "-crf",
                             str(a.crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(dst)])
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda cmd: subprocess.run(cmd, check=True), jobs))
        n += len(jobs)
    print(f"shrunk {n} clips into {out} in {(time.time() - t0) / 60:.1f} min")


def extract(a) -> None:
    import cv2
    from ultralytics import YOLO
    clips = list_clips(Path(a.datasets))
    model = YOLO(a.pose)
    CACHE.mkdir(parents=True, exist_ok=True)
    cuda = str(a.device) not in ("cpu", "") and __import__("torch").cuda.is_available()
    for key in a.only.split(","):
        items, t0, frames = [], time.time(), 0
        sel = clips[key]
        if a.limit:  # deterministic subset: up to `limit` clips per label, drawn by source video
            rng = random.Random(0)
            sel = []
            for lab in (0, 1):
                cl = [c for c in clips[key] if c["label"] == lab]
                rng.shuffle(cl)
                sel += sorted(cl[:a.limit], key=lambda c: c["name"])
        for n, c in enumerate(sel):
            cap = cv2.VideoCapture(c["path"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = a.step or max(1, round(fps / a.target_fps))
            tracks: dict[int, list] = {}
            i, used = 0, 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if i % step:
                    i += 1
                    continue
                r = model.track(fr, persist=used > 0, tracker="bytetrack.yaml", imgsz=a.imgsz, conf=0.25,
                                classes=[0], half=cuda, verbose=False, device=a.device if cuda else "cpu")[0]
                used += 1
                if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                    ids = r.boxes.id.int().tolist()
                    bx = r.boxes.xyxy.cpu().numpy()
                    kp = r.keypoints.data.cpu().numpy()
                    for j, tid in enumerate(ids):
                        tracks.setdefault(tid, []).append((i / fps, bx[j], kp[j]))
                i += 1
            cap.release()
            frames += used
            tr = {tid: (np.array([x[0] for x in v], np.float32), np.stack([x[1] for x in v]).astype(np.float32),
                        np.stack([x[2] for x in v]).astype(np.float16))
                  for tid, v in tracks.items() if len(v) >= 3}
            items.append({**c, "fps": fps, "frames": i, "width": 0, "tracks": tr})
            if n % 100 == 0:
                print(f"{key}: {n}/{len(sel)} clips, {frames / max(1e-6, time.time() - t0):.0f} fps", flush=True)
        with open(CACHE / f"{key}.pkl", "wb") as f:
            pickle.dump({"pose": a.pose, "imgsz": a.imgsz, "step": a.step or f"auto {a.target_fps} fps", "clips": items}, f)
        print(f"{key}: done {len(items)} clips, {frames} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------------------------- windows
def clip_windows(clip: dict, T: int, fps: float, pair_max_dist: float, min_valid: float, phases=(0.0,),
                 cap: int = 0, rng=None) -> list[tuple]:
    tr = {k: (v[0].astype(np.float64), v[1], v[2].astype(np.float32)) for k, v in clip["tracks"].items()}
    out = []
    for a, b in itertools.combinations(sorted(tr), 2):
        ta, tb = tr[a], tr[b]
        lo, hi = max(ta[0][0], tb[0][0]), min(ta[0][-1], tb[0][-1])
        if hi - lo < (T - 1) / fps * min_valid:
            continue
        for ph in phases:
            for te in np.arange(lo + (T - 1) / fps + ph, hi + 1e-6, 1.0 / fps):
                st = te - (T - 1 - np.arange(T)) / fps
                bA, kA, vA = resample_track(ta[0], ta[1], ta[2], st)
                bB, kB, vB = resample_track(tb[0], tb[1], tb[2], st)
                if min(vA.mean(), vB.mean()) < min_valid or not window_candidate(bA, bB, pair_max_dist):
                    continue
                out.append((kA, bA, kB, bB))
    if cap and len(out) > cap:
        rng = rng or random.Random(0)
        out = rng.sample(out, cap)
    return out


def load_windows(key: str, a, phases, cap: int) -> list[dict]:
    with open(CACHE / f"{key}.pkl", "rb") as f:
        d = pickle.load(f)
    rng = random.Random(1)
    res = []
    for c in d["clips"]:
        w = clip_windows(c, a.window, a.fps, a.pair_max_dist, a.min_valid, phases, cap, rng)
        arr = None
        if w:
            arr = tuple(np.stack([x[i] for x in w]).astype(np.float32) for i in range(4))
        res.append({"name": c["name"], "src": c["src"], "label": c["label"], "w": arr})
    return res


# ---------------------------------------------------------------------------------------------- metrics
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y); s = np.asarray(s, np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_metrics(y, s, thr) -> dict:
    y = np.asarray(y); p = np.asarray(s) >= thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum()); tn = int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "threshold": round(float(thr), 4), "accuracy": round((tp + tn) / len(y), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0, "roc_auc": round(roc_auc(y, s), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pick_threshold(y, s) -> float:
    """Threshold maximising Youden's J (TPR - FPR) on validation scores."""
    y = np.asarray(y); s = np.asarray(s)
    best, thr = -1.0, 0.5
    for t in np.unique(s):
        p = s >= t
        tpr = (p & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr = (p & (y == 0)).sum() / max(1, (y == 0).sum())
        if tpr - fpr > best:
            best, thr = tpr - fpr, float(t)
    return thr


# ---------------------------------------------------------------------------------------------- training
def feats_of(w, fps, flip=False, swap=False):
    kA, bA, kB, bB = w
    if swap:
        kA, bA, kB, bB = kB, bB, kA, bA
    if flip:
        kA, bA = flip_lr(kA, bA); kB, bB = flip_lr(kB, bB)
    return pair_features(kA, bA, kB, bB, fps)


def clip_scores_gru(model, clips, fps, device, chunk=4096) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for c in clips:
            if c["w"] is None:
                out.append(0.0); continue
            f = feats_of(c["w"], fps)
            logits = torch.cat([model(torch.from_numpy(f[i:i + chunk]).to(device)) for i in range(0, len(f), chunk)])
            out.append(float(torch.sigmoid(logits.max()).cpu()))
    return np.array(out)


def clip_scores_heuristic(clips, fps) -> np.ndarray:
    return np.array([float(heuristic_score(*c["w"], fps=fps).max()) if c["w"] is not None else 0.0 for c in clips])


def train(a) -> None:
    import torch
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda:0" if torch.cuda.is_available() and a.device != "cpu" else "cpu"
    t0 = time.time()
    rwf_train = load_windows("rwf_train", a, phases=(0.0, 0.1), cap=a.train_cap)
    rwf_test = load_windows("rwf_val", a, phases=(0.0,), cap=a.eval_cap)
    hockey = load_windows("hockey", a, phases=(0.0,), cap=a.eval_cap) if (CACHE / "hockey.pkl").exists() else []
    srcs = sorted({c["src"] for c in rwf_train})
    random.Random(0).shuffle(srcs)
    val_src = set(srcs[:int(len(srcs) * 0.15)])
    tr = [c for c in rwf_train if c["src"] not in val_src]
    va = [c for c in rwf_train if c["src"] in val_src]
    tr_w = [c for c in tr if c["w"] is not None]
    stats = {"train_clips": len(tr), "train_clips_with_windows": len(tr_w), "val_clips": len(va),
             "test_clips": len(rwf_test), "hockey_clips": len(hockey),
             "clips_without_windows": {k: sum(c["w"] is None for c in v) for k, v in
                                       (("train", tr), ("val", va), ("test", rwf_test), ("hockey", hockey))},
             "windows_train": int(sum(len(c["w"][0]) for c in tr_w)), "window_load_min": round((time.time() - t0) / 60, 1)}
    print(json.dumps(stats), flush=True)

    model = build_gru(N_FEAT, a.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    y_tr = np.array([c["label"] for c in tr_w])
    pos_w = torch.tensor((y_tr == 0).sum() / max(1, (y_tr == 1).sum()), device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    rng = random.Random(0)
    best = (-1.0, None, 0)
    y_va = np.array([c["label"] for c in va])
    for ep in range(a.epochs):
        model.train()
        rng.shuffle(tr_w)
        tot = 0.0
        for i in range(0, len(tr_w), a.clips_per_batch):
            batch = tr_w[i:i + a.clips_per_batch]
            feats, seg = [], []
            for c in batch:
                n = len(c["w"][0])
                idx = rng.sample(range(n), min(n, a.windows_per_clip))
                w = tuple(x[idx] for x in c["w"])
                f = feats_of(w, a.fps, flip=rng.random() < 0.5, swap=rng.random() < 0.5)
                feats.append(f); seg.append(len(idx))
            x = torch.from_numpy(np.concatenate(feats)).to(device)
            x = x + 0.02 * torch.randn_like(x)  # small feature noise
            logits = model(x)
            parts = torch.split(logits, seg)
            r = 5.0
            clip_logit = torch.stack([(torch.logsumexp(r * p, 0) - np.log(len(p))) / r for p in parts])
            y = torch.tensor([c["label"] for c in batch], dtype=torch.float32, device=device)
            loss = lossf(clip_logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        s_va = clip_scores_gru(model, va, a.fps, device)
        auc = roc_auc(y_va, s_va)
        if auc > best[0]:
            best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep}: loss {tot / len(tr_w):.4f} val AUC {auc:.4f} (best {best[0]:.4f} @ {best[2]})", flush=True)
    model.load_state_dict(best[1])

    # ---- evaluation: GRU vs heuristic, thresholds picked on val only ----
    res = {"params": n_params, "best_epoch": best[2], "data": stats}
    s_va = clip_scores_gru(model, va, a.fps, device)
    thr = pick_threshold(y_va, s_va)
    h_va = clip_scores_heuristic(va, a.fps)
    hthr = pick_threshold(y_va, h_va)
    for name, clips in (("val", va), ("rwf_test", rwf_test), ("hockey_ood", hockey)):
        if not clips:
            continue
        y = np.array([c["label"] for c in clips])
        res[name] = {"gru": binary_metrics(y, clip_scores_gru(model, clips, a.fps, device), thr),
                     "heuristic": binary_metrics(y, clip_scores_heuristic(clips, a.fps), hthr)}
    print(json.dumps(res, indent=2), flush=True)

    out = ROOT / "data" / "models" / "argonix_altercation_gru.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    meta = {"name": "argonix-altercation-posegru", "feature_version": FEATURE_VERSION, "n_feat": N_FEAT,
            "hidden": a.hidden, "window": a.window, "fps": a.fps, "pair_max_dist": a.pair_max_dist,
            "min_valid": a.min_valid, "threshold": round(thr, 4), "heuristic_threshold": round(hthr, 4),
            "pose": "yolo26n-pose.pt", "trained_on": "RWF-2000 train (10 fps Kaggle copy), MIL clip labels",
            "metrics": res, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "train_minutes": round((time.time() - t0) / 60, 1)}
    torch.save({"state_dict": best[1], "meta": json.loads(json.dumps(meta, default=float))}, out)
    meta["version"] = "v1-" + hashlib.sha256(out.read_bytes()).hexdigest()[:8]
    torch.save({"state_dict": best[1], "meta": json.loads(json.dumps(meta, default=float))}, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out} ({meta['version']})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--datasets", default="../datasets")
    e.add_argument("--only", default="rwf_train,rwf_val,hockey")
    e.add_argument("--pose", default="yolo26n-pose.pt")
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--device", default="0")
    e.add_argument("--step", type=int, default=0, help="use every n-th frame (0 = auto: sample at --target-fps)")
    e.add_argument("--target-fps", type=float, default=5.0, help="runtime analytics rate the model is trained for")
    e.add_argument("--limit", type=int, default=0, help="max clips per label per split (0 = all)")
    sh = sub.add_parser("shrink")
    sh.add_argument("--datasets", default="../datasets")
    sh.add_argument("--out", default="data/altercation_small")
    sh.add_argument("--only", default="rwf_train,rwf_val,hockey")
    sh.add_argument("--limit", type=int, default=0)
    sh.add_argument("--target-fps", type=float, default=5.0)
    sh.add_argument("--max-w", type=int, default=640)
    sh.add_argument("--crf", type=int, default=26)
    sh.add_argument("--jobs", type=int, default=6)
    t = sub.add_parser("train")
    for k, v in (("window", 8), ("hidden", 64), ("epochs", 60), ("clips_per_batch", 16), ("windows_per_clip", 16),
                 ("train_cap", 96), ("eval_cap", 600)):
        t.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    for k, v in (("fps", 5.0), ("pair_max_dist", 1.5), ("min_valid", 0.75), ("lr", 2e-3)):
        t.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    t.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    {"extract": extract, "train": train, "shrink": shrink}[a.cmd](a)


if __name__ == "__main__":
    main()
