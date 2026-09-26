"""Build a held-out weapon dataset and fine-tune YOLO26n on it (run on the GPU host).

    uv run python tools/train_weapons.py prepare --datasets ../datasets --out data/weapons_ds
    uv run python tools/train_weapons.py train --data data/weapons_ds/weapons.yaml --epochs 50 --batch 16

Sources (Kaggle, downloaded with `uvx kaggle datasets download <ref> --unzip` into --datasets):
- raghavnanjappan/weapon-dataset-for-yolov5: ~4k internet images, classes Knife / Handgun, YOLO labels. Licence
  "unknown" on Kaggle (personal demo use only; do not redistribute images or weights trained on them).
- rmilcb/rwf-2000-10fps (RWF-2000 at 10 fps): frames from TRAIN-split NonFight clips are added as background
  (empty-label) images so the model sees normal CCTV scenes. RWF-2000 VAL clips are never used here (they are the
  normal-footage false-positive set in tools/eval_threats.py).

`prepare` pools the weapon images, removes exact and near duplicates (64-bit dHash, Hamming <= 4, grouped with
union-find so near-duplicates never straddle splits), then splits 80/10/10 into train/val/test (seed 0).
`train` fine-tunes from yolo26n.pt and copies best.pt to data/models/argonix_weapons_yolo26n.pt with a JSON sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "knife", 1: "handgun"}
SRC_CLASS = {0: 0, 1: 1}  # raghavnanjappan: 0 Knife, 1 Handgun -> ours: 0 knife, 1 handgun


def dhash(img: np.ndarray) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def group_near_duplicates(hashes: list[int], md5s: list[str], max_ham: int = 4) -> list[int]:
    """Union-find group id per item: same md5 or dHash Hamming distance <= max_ham."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_md5: dict[str, int] = {}
    for i, m in enumerate(md5s):
        if m in by_md5:
            union(by_md5[m], i)
        else:
            by_md5[m] = i
    h = np.array(hashes, dtype=np.uint64)
    for i in range(n):
        x = np.bitwise_xor(h[i + 1:], h[i])
        # popcount on uint64 via byte view
        ham = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(ham <= max_ham)[0]:
            union(i, i + 1 + int(j))
    return [find(i) for i in range(n)]


def background_frames(a, rng) -> dict:
    """{split: [(stem, BGR frame)]}. From --bg-frames (a dir with train/val/test/*.jpg written by `export-bg`) if
    given, else sampled from RWF-2000 train NonFight clips under --datasets (one frame per source video)."""
    if getattr(a, "bg_frames", None):
        root = Path(a.bg_frames)
        return {s: [(f"bg_{p.stem}", cv2.imread(str(p))) for p in sorted((root / s).glob("*.jpg"))]
                for s in ("train", "val", "test") if (root / s).is_dir()}
    bg = sorted((Path(a.datasets) / "rwf-2000-10fps" / "train" / "NonFight").glob("*.avi"))
    by_src: dict[str, list[Path]] = {}
    for p in bg:
        by_src.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    srcs = sorted(by_src)
    rng.shuffle(srcs)
    want = {"train": a.bg_train, "val": a.bg_eval, "test": a.bg_eval}
    out = {"train": [], "val": [], "test": []}
    si = 0
    for s in ("val", "test", "train"):
        while len(out[s]) < want[s] and si < len(srcs):
            clip = rng.choice(by_src[srcs[si]]); si += 1
            cap = cv2.VideoCapture(str(clip))
            n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 50
            cap.set(cv2.CAP_PROP_POS_FRAMES, rng.randrange(max(1, n_fr)))
            ok, fr = cap.read()
            cap.release()
            if ok:
                out[s].append((f"bg_rwf_{clip.stem}", fr))
    return out


def export_bg(a) -> None:
    """Write the RWF background frames as small JPEGs (to ship to a host that lacks the videos)."""
    rng = random.Random(0)
    root = Path(a.out)
    n = 0
    for s, frames in background_frames(a, rng).items():
        (root / s).mkdir(parents=True, exist_ok=True)
        for stem, fr in frames:
            h, w = fr.shape[:2]
            if w > a.max_w:
                fr = cv2.resize(fr, (a.max_w, int(h * a.max_w / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(root / s / (stem.removeprefix("bg_") + ".jpg")), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
    print(f"exported {n} background frames to {root}")


def prepare(a) -> None:
    src_root = Path(a.datasets)
    wd = src_root / "weapon-dataset-for-yolov5" / "dataset" / "dataset"
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    items = []  # (img_path, label_lines)
    for split in ("train", "val"):
        for img in sorted((wd / "images" / split).glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            lab = wd / "labels" / split / (img.stem + ".txt")
            lines = []
            if lab.exists():
                for ln in lab.read_text().splitlines():
                    p = ln.split()
                    if len(p) == 5 and int(p[0]) in SRC_CLASS:
                        lines.append(" ".join([str(SRC_CLASS[int(p[0])])] + p[1:]))
            items.append((img, lines))
    print(f"weapon images found: {len(items)}")
    hashes, md5s, keep = [], [], []
    for img, lines in items:
        data = img.read_bytes()
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None or not lines:
            continue
        keep.append((img, lines))
        md5s.append(hashlib.md5(data).hexdigest())
        hashes.append(dhash(im))
    groups = group_near_duplicates(hashes, md5s)
    uniq = sorted(set(groups))
    rng = random.Random(0)
    rng.shuffle(uniq)
    n = len(uniq)
    split_of = {g: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, g in enumerate(uniq)}
    # one representative per near-duplicate group in val/test (duplicates would inflate held-out metrics)
    seen_eval: set[int] = set()
    counts = {"train": 0, "val": 0, "test": 0}
    inst = {s: {0: 0, 1: 0} for s in counts}
    for (img, lines), g in zip(keep, groups):
        s = split_of[g]
        if s != "train":
            if g in seen_eval:
                continue
            seen_eval.add(g)
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        stem = f"w_{img.parent.name}_{img.stem}"
        shutil.copy2(img, out / "images" / s / (stem + img.suffix.lower()))
        (out / "labels" / s / (stem + ".txt")).write_text("\n".join(lines) + "\n")
        counts[s] += 1
        for ln in lines:
            inst[s][int(ln.split()[0])] += 1
    print(f"kept {len(keep)} labelled images, {len(uniq)} near-duplicate groups; split images {counts}")

    # background (no-weapon) frames from RWF-2000 TRAIN NonFight clips, split by source video id
    bgc = {"train": 0, "val": 0, "test": 0}
    for s, frames in background_frames(a, rng).items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for stem, fr in frames:
            cv2.imwrite(str(out / "images" / s / (stem + ".jpg")), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / s / (stem + ".txt")).write_text("")
            bgc[s] += 1
    print(f"background frames {bgc}")
    yaml = out / "weapons.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"names:\n  0: knife\n  1: handgun\n")
    (out / "split_stats.json").write_text(json.dumps({"images": counts, "instances": inst, "backgrounds": bgc,
                                                      "groups": len(uniq), "labelled_kept": len(keep)}, indent=2))
    print(f"wrote {yaml}")


def train(a) -> None:
    from ultralytics import YOLO
    import ultralytics
    model = YOLO(a.model)
    t0 = time.time()
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, cache=False,
                device=a.device, patience=a.patience, seed=0, deterministic=True, project=str(Path(a.project).resolve()),
                name=a.name, exist_ok=True, plots=False, amp=True, verbose=False)
    run = Path(a.project).resolve() / a.name
    best = run / "weights" / "best.pt"
    dst = ROOT / "data" / "models" / "argonix_weapons_yolo26n.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
    meta = {"name": "argonix-weapons-yolo26n", "version": f"v1-{sha[:8]}", "sha256_16": sha, "base": a.model,
            "classes": NAMES, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
            "ultralytics": ultralytics.__version__, "train_minutes": round((time.time() - t0) / 60, 1),
            "data": a.data, "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--datasets", default="../datasets")
    p.add_argument("--out", default="data/weapons_ds")
    p.add_argument("--bg-train", type=int, default=400)
    p.add_argument("--bg-eval", type=int, default=50)
    p.add_argument("--bg-frames", default=None, help="dir with train/val/test background JPEGs (from export-bg)")
    e = sub.add_parser("export-bg")
    e.add_argument("--datasets", default="../datasets")
    e.add_argument("--out", default="data/weapons_bg")
    e.add_argument("--bg-train", type=int, default=400)
    e.add_argument("--bg-eval", type=int, default=50)
    e.add_argument("--max-w", type=int, default=960)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/weapons_ds/weapons.yaml")
    t.add_argument("--model", default="data/models/yolo26n.pt")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--device", default="0")
    t.add_argument("--project", default="data/train_runs")
    t.add_argument("--name", default="weapons_yolo26n")
    a = ap.parse_args(argv)
    {"prepare": prepare, "train": train, "export-bg": export_bg}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
