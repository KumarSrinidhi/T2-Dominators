"""Measure the threat analyzers (run on the GPU host; CPU works but is slow).

    uv run python tools/eval_threats.py weapons --datasets ../datasets     # held-out test split + CCTV synthetic OOD
    uv run python tools/eval_threats.py normal --datasets ../datasets      # alerts/hour on normal footage + RWF val

`weapons`: for the fine-tuned model and the zero-training COCO baseline, on the held-out test split made by
tools/train_weapons.py (and on simuletic/cctv-weapon-dataset, synthetic CCTV views, labelled only "weapon", scored
class-agnostically): AP50 per class (all-point interpolation, IoU 0.5) and precision/recall at the runtime
confidence (default 0.5), plus Ultralytics' own `val` numbers for the fine-tuned model.

`normal`: runs the real runtime path (person tracker -> ObjectThreatAnalyzer / AltercationAnalyzer with the
configured persistence) over footage with no weapons and no fights, at the analytics fps, and counts events per hour.
It also runs the altercation analyzer over RWF-2000 val Fight clips (clip-level: did it raise >= 1 event).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.threats import UltralyticsObjectModel  # noqa: E402

MODELS = ROOT / "data" / "models"
WEAPONS = {"weights": str(MODELS / "argonix_weapons_yolo26n.pt"), "classes": {0: "knife", 1: "handgun"}}
COCO = {"weights": str(MODELS / "yolo26n.pt"), "name": "coco-yolo26n", "classes": {43: "knife", 34: "bat", 76: "scissors"}}


# ---------------------------------------------------------------------------------------------- held-out detection
def ap50(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
    rec = tpc / n_gt; prec = tpc / (tpc + fpc)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)


def read_labels(lbl: Path, w: int, h: int, cls_map: dict) -> list[tuple[str, tuple]]:
    out = []
    if lbl.exists():
        for ln in lbl.read_text().split("\n"):
            p = ln.split()
            if len(p) == 5 and int(p[0]) in cls_map:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((cls_map[c], ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
    return out


def eval_detector(model: UltralyticsObjectModel, img_dir: Path, lbl_dir: Path, gt_map: dict, labels: list[str],
                  op_conf: float, agnostic: bool = False) -> dict:
    """AP50 per label + precision/recall at op_conf. agnostic: every prediction/GT counts as label 'weapon'."""
    model.conf = 0.001
    preds = {l: [] for l in labels}   # label -> [(conf, tp)]
    n_gt = {l: 0 for l in labels}
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    n_bg_imgs, bg_fp_imgs = 0, 0
    for i in range(0, len(imgs), 16):
        batch = [cv2.imread(str(p)) for p in imgs[i:i + 16]]
        res = model.predict(batch)
        for p, im, dets in zip(imgs[i:i + 16], batch, res):
            h, w = im.shape[:2]
            gts = read_labels(lbl_dir / (p.stem + ".txt"), w, h, gt_map)
            if agnostic:
                gts = [("weapon", b) for _, b in gts]
                dets = [SimpleNamespace(label="weapon", conf=d.conf, box=d.box) for d in dets]
            if not gts:
                n_bg_imgs += 1
                bg_fp_imgs += any(d.conf >= op_conf for d in dets)
            for l in labels:
                g = [b for c, b in gts if c == l]
                n_gt[l] += len(g)
                ds = sorted([d for d in dets if d.label == l], key=lambda d: -d.conf)
                used: set = set()
                m = iou_matrix([d.box for d in ds], g) if ds and g else None
                for k, d in enumerate(ds):  # greedy by confidence, each GT matched once (IoU >= 0.5)
                    tp = False
                    cand = [(m[k, j], j) for j in range(len(g)) if j not in used] if m is not None else []
                    if cand:
                        best, j = max(cand)
                        if best >= 0.5:
                            used.add(j); tp = True
                    preds[l].append((d.conf, tp))
    out = {}
    for l in labels:
        arr = np.array(preds[l], dtype=float).reshape(-1, 2)
        conf, tp = arr[:, 0], arr[:, 1].astype(bool)
        sel = conf >= op_conf
        tps = int(tp[sel].sum()); fps_ = int((~tp[sel]).sum())
        out[l] = {"gt": n_gt[l], "ap50": round(ap50(tp, conf, n_gt[l]), 4),
                  f"precision@{op_conf}": round(tps / max(1, tps + fps_), 4),
                  f"recall@{op_conf}": round(tps / max(1, n_gt[l]), 4), "tp": tps, "fp": fps_}
    out["_images"] = len(imgs)
    out["_background_images"] = n_bg_imgs
    out[f"_background_images_with_any_det@{op_conf}"] = bg_fp_imgs
    return out


def weapons(a) -> None:
    ds = ROOT / "data" / "weapons_ds"
    res = {"op_conf": a.conf}
    ft = UltralyticsObjectModel(WEAPONS["weights"], WEAPONS["classes"], device=a.device)
    coco = UltralyticsObjectModel(COCO["weights"], COCO["classes"], name=COCO["name"], device=a.device)
    res["finetuned_model"] = ft.info
    t0 = time.time()
    res["test_finetuned"] = eval_detector(ft, ds / "images" / "test", ds / "labels" / "test",
                                          {0: "knife", 1: "handgun"}, ["knife", "handgun"], a.conf)
    res["test_coco_baseline"] = eval_detector(coco, ds / "images" / "test", ds / "labels" / "test",
                                              {0: "knife", 1: "handgun"}, ["knife", "handgun"], a.conf)
    sim = Path(a.datasets) / "cctv-weapon-dataset" / "Dataset"
    if sim.exists():
        res["cctv_synthetic_ood_finetuned"] = eval_detector(ft, sim / "images", sim / "labels", {1: "weapon"},
                                                            ["weapon"], a.conf, agnostic=True)
        res["cctv_synthetic_ood_coco_baseline"] = eval_detector(coco, sim / "images", sim / "labels", {1: "weapon"},
                                                                ["weapon"], a.conf, agnostic=True)
    from ultralytics import YOLO
    m = YOLO(WEAPONS["weights"])
    v = m.val(data=str(ds / "weapons.yaml"), split="test", imgsz=640, batch=8, device=a.device, workers=2,
              plots=False, verbose=False)
    res["ultralytics_val_test"] = {"mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
                                   "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
                                   "per_class_ap50": {m.names[int(c)]: round(float(x), 4)
                                                      for c, x in zip(v.box.ap_class_index, v.box.ap50)}}
    res["eval_minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(res, indent=2))
    (ROOT / "data" / "eval_weapons.json").write_text(json.dumps(res, indent=2))


# ---------------------------------------------------------------------------------------------- normal footage
def fake_worker(cam: str, data: Path, backend_info: dict):
    return SimpleNamespace(id=cam, name=cam, site=SimpleNamespace(data=data), calib_error=None,
                           backend=SimpleNamespace(info=backend_info))


def run_clip(path: str, analyzers: list, afps: float, device: str, data: Path, max_w: int = 960) -> dict:
    from argonix.detect import UltralyticsTracker
    trk = UltralyticsTracker(str(MODELS / "yolo26n.pt"), conf=0.3, imgsz=640, device=device)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1 <= src_fps <= 120:
        src_fps = 25.0
    step = max(1, round(src_fps / afps))
    w = fake_worker(Path(path).stem, data, trk.info)
    n, used, events = 0, 0, {type(an).__name__: [] for an in analyzers}
    ms = {type(an).__name__: 0.0 for an in analyzers}
    t0 = 1_000_000.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n % step:
            continue
        used += 1
        h0, w0 = fr.shape[:2]
        scale = max(1.0, w0 / max_w)
        small = fr if scale == 1.0 else cv2.resize(fr, (int(w0 / scale), int(h0 / scale)))
        dets = trk.track(small)
        t = t0 + n / src_fps
        for an in analyzers:
            s = time.perf_counter()
            evs = an.on_frame(w, t, small, dets, [])
            ms[type(an).__name__] += (time.perf_counter() - s) * 1000
            events[type(an).__name__] += [{"t": round(e.t_end - t0, 1), "obs": e.observation,
                                           "conf": e.confidence, "crop": e.evidence.get("crop")} for e in evs]
    cap.release()
    for an in analyzers:
        an.close()
        getattr(an, "last_run", {}).clear()
    return {"seconds": n / src_fps, "frames_analysed": used, "events": events,
            "ms_per_frame": {k: round(v / max(1, used), 1) for k, v in ms.items()}}


def poisson_upper95(k: int) -> float:
    """One-sided 95% upper bound on a Poisson count (Wilson-Hilferty; 3.0 for k = 0, the 'rule of three')."""
    n = k + 1
    return n * (1 - 1 / (9 * n) + 1.645 / (3 * n ** 0.5)) ** 3


def rate(events: int, seconds: float) -> dict:
    h = seconds / 3600
    return {"events": events, "hours": round(h, 4), "per_hour": round(events / h, 2) if h else None,
            "per_hour_95pct_upper": round(poisson_upper95(events) / h, 1) if h else None}


def normal(a) -> None:
    from argonix.threat_install import install
    cfg = json.loads(Path(a.config).read_text())
    t = cfg.setdefault("threats", {})
    if a.only:
        for k in list(t):
            if k != a.only:
                t[k] = {**t[k], "enabled": False}
    if a.heuristic:  # altercation without the learned classifier (baseline)
        t.setdefault("altercation", {})["classifier"] = None
    site = SimpleNamespace(cfg=cfg, analyzers=[], _base=ROOT)
    added = install(site)
    names = [type(x).__name__ for x in added]
    print("installed:", names, getattr(site, "threat_status", {}), flush=True)
    data = Path(tempfile.mkdtemp(prefix="argonix_eval_"))
    res = {"config": a.config, "analytics_fps": a.afps, "status": site.threat_status, "clips": {}}
    normal_clips = [str(ROOT / "data" / "real" / "vtest.avi"), str(ROOT / "data" / "real" / "social-distance" / "pedestrian.mp4")]
    extra = sorted((ROOT / "data" / "real" / "pedestrian-dataset").glob("*.avi"))
    groups = {"project_normal_clips": normal_clips, "pedestrian_dataset_clips": [str(p) for p in extra]}
    rwf = Path(a.datasets) / "rwf-2000-10fps" / "val"
    if rwf.exists():
        groups["rwf2000_val_nonfight"] = [str(p) for p in sorted((rwf / "NonFight").glob("*.avi"))][:a.rwf_limit]
        groups["rwf2000_val_fight"] = [str(p) for p in sorted((rwf / "Fight").glob("*.avi"))][:a.rwf_limit]
    summary = {}
    for g, clips in groups.items():
        secs, per = 0.0, {n: 0 for n in names}
        clips_with = {n: 0 for n in names}
        ms = {n: [] for n in names}
        before = [dict(getattr(an, "stats", {})) for an in added]
        for c in clips:
            r = run_clip(c, added, a.afps, a.device, data)
            secs += r["seconds"]
            for n in names:
                per[n] += len(r["events"][n]); clips_with[n] += bool(r["events"][n])
                ms[n].append(r["ms_per_frame"][n])
            if g in ("project_normal_clips", "pedestrian_dataset_clips") or any(r["events"].values()):
                res["clips"][c] = r
        summary[g] = {"clips": len(clips), **{n: {**rate(per[n], secs), "clips_with_event": clips_with[n],
                                                  "mean_ms_per_frame": round(float(np.mean(ms[n])), 1) if ms[n] else None}
                                              for n in names}}
        summary[g]["analyzer_stats"] = {type(an).__name__: {k: v - b.get(k, 0) for k, v in an.stats.items()}
                                        for an, b in zip(added, before)}
        print(g, json.dumps(summary[g]), flush=True)
    res["summary"] = summary
    res["evidence_dir"] = str(data)
    tag = (a.only or "all") + ("_heuristic" if a.heuristic else "")
    out = ROOT / "data" / f"eval_normal_{Path(a.config).stem}_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}; crops in {data}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weapons")
    w.add_argument("--datasets", default="../datasets")
    w.add_argument("--conf", type=float, default=0.5)
    w.add_argument("--device", default="auto")
    n = sub.add_parser("normal")
    n.add_argument("--datasets", default="../datasets")
    n.add_argument("--config", default=str(ROOT / "configs" / "real_site_gpu_threats.json"))
    n.add_argument("--afps", type=float, default=5.0)
    n.add_argument("--device", default="auto")
    n.add_argument("--rwf-limit", type=int, default=200)
    n.add_argument("--only", choices=["objects", "altercation"], default=None)
    n.add_argument("--heuristic", action="store_true", help="altercation analyzer without the learned classifier")
    a = ap.parse_args(argv)
    {"weapons": weapons, "normal": normal}[a.cmd](a)


if __name__ == "__main__":
    main()
