"""Head-to-head: our trained threat models vs popular third-party pretrained models, on the same held-out data.

    # weapons (app venv): our fine-tuned YOLO26n vs a third-party YOLO weapon model
    python tools/compare_pretrained.py weapons --thirdparty data/models/thirdparty/threat_yolov8n_subh775.pt \
        --datasets ~/argonix_datasets --device cuda:0
    # violence, step 1 (app venv, needs cv2): sample frames from RWF-2000 val, Hockey Fights and normal clips
    python tools/compare_pretrained.py frames --datasets ~/argonix_datasets --out data/vit_frames
    # violence, step 2 (any venv with transformers + PIL, e.g. the VLM venv): classify the frames
    python tools/compare_pretrained.py vit --frames data/vit_frames --model data/models/thirdparty/vit_violence
    # violence, step 3 (any venv): clip-level scores -> ROC AUC / precision / recall / FPR
    python tools/compare_pretrained.py score --frames data/vit_frames

Third-party models tested (downloaded from Hugging Face; licences per their model cards):
- Subh775/Threat-Detection-YOLOv8n (MIT): YOLOv8n, classes Gun / Explosive / Grenade / Knife (Roboflow data).
- jaranohaal/vit-base-violence-detection (Apache-2.0): ViT-B/16 frame classifier trained on Kaggle "Real Life
  Violence Situations" (not RWF-2000, so RWF-2000 val is out-of-sample for it). Clip score = max over sampled
  frames of P(violent) (also reported: mean). A frame classifier cannot say WHO is involved.
Results are written to data/compare_pretrained_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------- shared metrics
def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def binary_stats(scores: list[float], labels: list[int], thr: float) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    return {"n": len(y), "threshold": thr, "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4), "fpr": round(fp / max(1, fp + tn), 4),
            "accuracy": round((tp + tn) / max(1, len(y)), 4), "roc_auc": round(roc_auc(scores, labels), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------------------------- weapons
def weapons(a) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    from argonix.threats import UltralyticsObjectModel
    import eval_threats as et
    from ultralytics import YOLO

    names = YOLO(a.thirdparty).names
    cmap = {}
    for k, v in names.items():
        n = str(v).lower()
        if any(w in n for w in ("gun", "pistol", "handgun", "rifle", "firearm")):
            cmap[int(k)] = "handgun"
        elif "knife" in n or "blade" in n:
            cmap[int(k)] = "knife"
    ds = ROOT / "data" / "weapons_ds"
    ours = UltralyticsObjectModel(et.WEAPONS["weights"], et.WEAPONS["classes"], device=a.device)
    theirs = UltralyticsObjectModel(a.thirdparty, cmap, name=Path(a.thirdparty).stem, device=a.device)
    res = {"op_conf": a.conf, "thirdparty": a.thirdparty, "thirdparty_names": names, "thirdparty_class_map": cmap,
           "note": "third-party 'Gun' is scored as handgun (it also covers rifles); held-out split is OUR test split, "
                   "which may overlap the third-party model's (unknown) Roboflow training images"}
    for tag, m in (("ours", ours), ("thirdparty", theirs)):
        t0 = time.time()
        res[f"test_{tag}"] = et.eval_detector(m, ds / "images" / "test", ds / "labels" / "test",
                                              {0: "knife", 1: "handgun"}, ["knife", "handgun"], a.conf)
        sim = Path(a.datasets) / "cctv-weapon-dataset" / "Dataset"
        if sim.exists():
            res[f"cctv_synthetic_ood_{tag}"] = et.eval_detector(m, sim / "images", sim / "labels", {1: "weapon"},
                                                                ["weapon"], a.conf, agnostic=True)
        res[f"{tag}_eval_s"] = round(time.time() - t0, 1)
    out = ROOT / "data" / "compare_pretrained_weapons.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str), flush=True)

    # false events per hour on normal footage, each model ALONE through the real runtime path (eval_threats normal)
    base = json.loads((ROOT / "configs" / "real_site_gpu_threats.json").read_text())
    for tag, model in (("ours", {"weights": et.WEAPONS["weights"], "classes": {"0": "knife", "1": "handgun"}}),
                       ("thirdparty", {"weights": a.thirdparty, "name": Path(a.thirdparty).stem,
                                       "classes": {str(k): v for k, v in cmap.items()}})):
        cfg = json.loads(json.dumps(base))
        cfg["threats"]["objects"]["models"] = [{**model, "conf": 0.25}]
        p = ROOT / "data" / f"compare_cfg_objects_{tag}.json"
        p.write_text(json.dumps(cfg, indent=1))
        et.normal(argparse.Namespace(config=str(p), datasets=a.datasets, afps=5.0, device=a.device,
                                     rwf_limit=a.rwf_limit, only="objects", heuristic=False))


# ---------------------------------------------------------------------------------------------- violence (ViT)
def safe_stem(stem: str) -> str:
    """ASCII file-name stem: OpenCV on Windows silently fails to write non-ASCII paths (some RWF-2000 clips have
    Chinese names). A short hash keeps renamed stems unique."""
    import hashlib
    import re
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return clean if clean == stem else f"{clean[:40]}_{hashlib.sha1(stem.encode()).hexdigest()[:8]}"


def frames(a) -> None:
    import cv2
    ds, out = Path(a.datasets), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    groups = {}
    rwf = ds / "rwf-2000-10fps" / "val"
    for lab in ("Fight", "NonFight"):
        for p in sorted((rwf / lab).glob("*.avi")):
            groups.setdefault("rwf_val", []).append((p, int(lab == "Fight")))
    # calibration only: the model's label names are just LABEL_0/1, so which output means "violent" is chosen on
    # RWF-2000 TRAIN clips (disjoint from every scored group) and then fixed for the evaluation
    for lab in ("Fight", "NonFight"):
        for p in sorted((ds / "rwf-2000-10fps" / "train" / lab).glob("*.avi"))[:a.calib_clips]:
            groups.setdefault("calib_rwf_train", []).append((p, int(lab == "Fight")))
    for p in sorted((ds / "hockey-fight-vidoes" / "data").glob("*.avi")):
        groups.setdefault("hockey", []).append((p, int(p.name.startswith("fi"))))
    for p in [ROOT / "data" / "real" / "vtest.avi", ROOT / "data" / "real" / "social-distance" / "pedestrian.mp4",
              *sorted((ROOT / "data" / "real" / "pedestrian-dataset").glob("*.avi"))]:
        if p.exists():
            groups.setdefault("normal_long", []).append((p, 0))
    index, n = [], 0
    for g, clips in groups.items():
        for p, lab in clips:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, int(round(fps / a.sample_fps)))
            k, fi = 0, 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if k % step == 0:
                    h, w = fr.shape[:2]
                    s = 256 / min(h, w)
                    fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                    name = f"{g}_{safe_stem(p.stem)}_{fi:05d}.jpg"
                    if not cv2.imwrite(str(out / "img" / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError(f"could not write frame {name}")
                    index.append({"group": g, "clip": p.stem, "label": lab, "t": round(k / fps, 2), "file": name})
                    fi += 1; n += 1
                k += 1
            cap.release()
        print(g, len(clips), "clips", flush=True)
    (out / "index.json").write_text(json.dumps({"sample_fps": a.sample_fps, "frames": index}))
    print(f"{n} frames -> {out}")


TIMM_TO_HF = [("cls_token", "vit.embeddings.cls_token"), ("pos_embed", "vit.embeddings.position_embeddings"),
              ("patch_embed.proj.", "vit.embeddings.patch_embeddings.projection."), ("norm.", "vit.layernorm."),
              ("head.", "classifier.")]
# transformers 4.x and 5.x name the ViT encoder differently; both are supported (style picked from the model)
HF_BLOCK = {
    "v4": {"prefix": "vit.encoder.layer.{i}.", "qkv": ("attention.attention.query.", "attention.attention.key.",
                                                        "attention.attention.value."),
           "map": [("norm1.", "layernorm_before."), ("attn.proj.", "attention.output.dense."),
                   ("norm2.", "layernorm_after."), ("mlp.fc1.", "intermediate.dense."), ("mlp.fc2.", "output.dense.")]},
    "v5": {"prefix": "vit.layers.{i}.", "qkv": ("attention.q_proj.", "attention.k_proj.", "attention.v_proj."),
           "map": [("norm1.", "layernorm_before."), ("attn.proj.", "attention.o_proj."),
                   ("norm2.", "layernorm_after."), ("mlp.fc1.", "mlp.fc1."), ("mlp.fc2.", "mlp.fc2.")]},
}


def timm_vit_to_hf(sd: dict, style: str = "v4") -> dict:
    """timm ViT state dict (blocks.*, patch_embed, cls_token, head) -> transformers ViTForImageClassification keys.
    The fused attn.qkv projection is split into query / key / value."""
    st, out = HF_BLOCK[style], {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            pre = st["prefix"].format(i=i)
            if rest.startswith("attn.qkv."):
                kind = rest.rsplit(".", 1)[1]
                for name, part in zip(st["qkv"], v.chunk(3, dim=0)):
                    out[f"{pre}{name}{kind}"] = part.clone()
                continue
            for a, b in st["map"]:
                if rest.startswith(a):
                    out[pre + b + rest[len(a):]] = v
                    break
            else:
                raise KeyError(f"unmapped timm key {k}")
            continue
        for a, b in TIMM_TO_HF:
            if k == a or (a.endswith(".") and k.startswith(a)):
                out[b + k[len(a):] if a.endswith(".") else b] = v
                break
        else:
            raise KeyError(f"unmapped timm key {k}")
    return out


def load_vit_classifier(path: Path):
    """Load the third-party ViT and PROVE its weights loaded. The published jaranohaal checkpoint stores timm-format
    keys under a transformers config: from_pretrained then skips every weight and silently returns a randomly
    initialised network (which is what the first comparison run scored). Converted and loaded strictly here."""
    import transformers
    cfg_d = json.loads((path / "config.json").read_text())
    cfg = transformers.ViTConfig(**{k: v for k, v in cfg_d.items() if not k.startswith("_") and k != "architectures"})
    from safetensors.torch import load_file
    sd = load_file(str(path / "model.safetensors"))
    timm = any(k.startswith("blocks.") for k in sd)
    if timm:
        cfg.layer_norm_eps = 1e-6  # timm's ViT LayerNorm epsilon
    model = transformers.ViTForImageClassification(cfg)
    if timm:
        style = "v5" if any(k.startswith("vit.layers.") for k in model.state_dict()) else "v4"
        sd = timm_vit_to_hf(sd, style)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ViT weights did not load: {len(missing)} missing (e.g. {missing[:3]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    return model


def vit(a) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    fr = Path(a.frames)
    (fr / "scores.json").unlink(missing_ok=True)  # a failed run must never leave the previous scores looking current
    idx = json.loads((fr / "index.json").read_text())["frames"]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        proc = AutoImageProcessor.from_pretrained(a.model)
    except (ValueError, KeyError, OSError):
        # old checkpoints name a processor class newer transformers no longer registers (e.g. ViTFeatureExtractor):
        # apply their preprocessor_config by hand (resize to size, rescale 1/255, normalise mean/std)
        pc = json.loads((Path(a.model) / "preprocessor_config.json").read_text())
        size = pc.get("size", 224)
        hw = (size["height"], size["width"]) if isinstance(size, dict) else (int(size), int(size))
        mean = torch.tensor(pc.get("image_mean", [0.5] * 3)).view(1, 3, 1, 1)
        std = torch.tensor(pc.get("image_std", [0.5] * 3)).view(1, 3, 1, 1)

        def proc(images, return_tensors="pt"):
            x = np.stack([np.asarray(im.resize((hw[1], hw[0]), Image.BILINEAR), np.float32) / 255.0 for im in images])
            return {"pixel_values": (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std}
    model = load_vit_classifier(Path(a.model))
    model = model.to(dev).eval()
    if dev.startswith("cuda"):
        model = model.half()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    viol = [k for k, v in id2label.items() if "viol" in v.lower() and "non" not in v.lower()]
    vid = viol[0] if viol else max(id2label)
    t_inf, scores = 0.0, []
    for i in range(0, len(idx), a.batch):
        ims = [Image.open(fr / "img" / r["file"]).convert("RGB") for r in idx[i:i + a.batch]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        if dev.startswith("cuda"):
            x = x.half(); torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            p = model(pixel_values=x).logits.float().softmax(-1)[:, vid].cpu().numpy()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        scores.extend(float(v) for v in p)
    for r, s in zip(idx, scores):
        r["p_violent"] = round(s, 5)
    meta = {"model": a.model, "id2label": id2label, "violent_id": vid, "device": dev,
            "ms_per_frame": round(1000 * t_inf / max(1, len(idx)), 2),
            "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu"}
    (fr / "scores.json").write_text(json.dumps({"meta": meta, "frames": idx}))
    print(json.dumps(meta), flush=True)


def score(a) -> None:
    d = json.loads((Path(a.frames) / "scores.json").read_text())
    clips: dict = {}
    for r in d["frames"]:
        c = clips.setdefault((r["group"], r["clip"]), {"label": r["label"], "p": [], "t": []})
        c["p"].append(r["p_violent"]); c["t"].append(r["t"])
    res = {"meta": d["meta"], "clip_score": "max over frames (and mean) of P(violent)"}
    cal = [v for k, v in clips.items() if k[0] == "calib_rwf_train"]
    if cal:
        sc, lab = [float(np.mean(c["p"])) for c in cal], [c["label"] for c in cal]
        auc1 = roc_auc(sc, lab)
        flip = auc1 < 0.5
        res["label_mapping"] = {
            "chosen_on": f"RWF-2000 train, {len(cal)} clips (not scored below)",
            "auc_if_output_is_violent": round(auc1, 4), "auc_if_other_output_is_violent": round(1 - auc1, 4),
            "violent_output": ("the other output (flipped)" if flip else f"output {d['meta']['violent_id']}")}
        if flip:
            for c in clips.values():
                c["p"] = [1.0 - x for x in c["p"]]
    for g in sorted({k[0] for k in clips if k[0] != "calib_rwf_train"}):
        cs = [v for k, v in clips.items() if k[0] == g]
        labels = [c["label"] for c in cs]
        for agg in ("max", "mean"):
            sc = [float(np.max(c["p"]) if agg == "max" else np.mean(c["p"])) for c in cs]
            if g == "normal_long":
                # long normal clips: count "events" = runs of consecutive frames with P >= 0.5, per hour of video
                ev = sum(int(np.sum(np.diff((np.asarray(c["p"]) >= 0.5).astype(int), prepend=0) == 1)) for c in cs)
                hours = sum((c["t"][-1] if c["t"] else 0) for c in cs) / 3600
                res[f"{g}_{agg}"] = {"clips": len(cs), "hours": round(hours, 4),
                                     "frames_ge_0.5": int(sum(np.sum(np.asarray(c["p"]) >= 0.5) for c in cs)),
                                     "frames": int(sum(len(c["p"]) for c in cs)),
                                     "onsets_per_hour": round(ev / max(hours, 1e-9), 1) if agg == "max" else None}
            else:
                res[f"{g}_{agg}"] = binary_stats(sc, labels, 0.5)
    out = ROOT / "data" / "compare_pretrained_violence.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weapons")
    w.add_argument("--thirdparty", required=True)
    w.add_argument("--datasets", default="../datasets")
    w.add_argument("--conf", type=float, default=0.5)
    w.add_argument("--device", default="auto")
    w.add_argument("--rwf-limit", type=int, default=200)
    f = sub.add_parser("frames")
    f.add_argument("--datasets", default="../datasets")
    f.add_argument("--out", default=str(ROOT / "data" / "vit_frames"))
    f.add_argument("--sample-fps", type=float, default=2.0)
    f.add_argument("--calib-clips", type=int, default=60, help="RWF-2000 train clips per class for the label mapping")
    v = sub.add_parser("vit")
    v.add_argument("--frames", default=str(ROOT / "data" / "vit_frames"))
    v.add_argument("--model", required=True)
    v.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("score")
    s.add_argument("--frames", default=str(ROOT / "data" / "vit_frames"))
    a = ap.parse_args(argv)
    {"weapons": weapons, "frames": frames, "vit": vit, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
