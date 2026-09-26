"""Appearance embeddings of person crops for cross-camera Re-ID (research-grade; the vectors are biometric-adjacent).

  * `OSNetEmbedder`: a small pretrained person Re-ID network (OSNet / OSNet-AIN from torchreid, trained on MSMT17),
    fp16 on CUDA, fp32 on CPU. Output: 512-d, L2-normalised. Whole-body appearance (clothing, shape, gait-free);
    no face recognition anywhere.
  * `HandcraftedEmbedder`: no learned weights (HSV stripe histograms + gradient-orientation histograms). A weak
    fallback when no Re-ID checkpoint is installed, and an ablation baseline.
  * `EmbeddingService`: ONE embedder shared by all cameras. A background thread batches queued crops (GPU-friendly)
    and hands the vectors back through a callback, so camera threads never wait on the Re-ID model.
  * Crop quality gate (`crop_quality`): only crops that are not truncated by the image border, have detector
    confidence >= 0.5, a minimum height and a plausible aspect ratio are embedded.
  * Grayscale / IR: `is_monochrome` detects colourless crops; `to_gray3` makes a 3-channel gray image so a colour
    camera and an IR camera can be compared on luminance only (colour cannot be used in that case).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

# Known checkpoints (torchreid model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). The code that
# runs them (argonix/osnet.py) is MIT; the WEIGHTS were trained on MSMT17, whose terms restrict use to non-commercial
# research. Similarity priors ("same"/"diff": mean, sd of cosine similarity) are starting values for the Bayesian
# appearance term; the "diff" side is re-estimated on site from tracks seen at the same time on the same camera.
KNOWN: dict[str, dict] = {
    "osnet_ain_x1_0_msmt17_cosine.pth": {
        "arch": "osnet_ain_x1_0", "trained_on": "MSMT17 (all identities; torchreid cross-domain setting, colour jitter)",
        "source": "torchreid model zoo, Google Drive id 1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
        "sha256": "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
        "sim": {"color": {"same": [0.70, 0.10], "diff": [0.36, 0.11]}, "gray": {"same": [0.68, 0.10], "diff": [0.42, 0.11]}}},
    "osnet_x1_0_msmt17.pth": {
        "arch": "osnet_x1_0", "trained_on": "MSMT17 (train split)",
        "source": "torchreid model zoo, Google Drive id 112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
        "sha256": "b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2",
        "sim": {"color": {"same": [0.72, 0.10], "diff": [0.38, 0.11]}, "gray": {"same": [0.70, 0.10], "diff": [0.45, 0.11]}}},
    "osnet_x0_25_msmt17.pth": {
        "arch": "osnet_x0_25", "trained_on": "MSMT17 (train split)",
        "source": "torchreid model zoo, Google Drive id 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
        "sha256": "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99",
        "sim": {"color": {"same": [0.74, 0.10], "diff": [0.42, 0.11]}, "gray": {"same": [0.72, 0.10], "diff": [0.48, 0.11]}}},
}
AUTO_ORDER = ["osnet_ain_x1_0_msmt17_cosine.pth", "osnet_x1_0_msmt17.pth", "osnet_x0_25_msmt17.pth"]
HANDCRAFTED_SIM = {"color": {"same": [0.80, 0.08], "diff": [0.62, 0.10]}, "gray": {"same": [0.82, 0.08], "diff": [0.70, 0.09]}}

QUALITY = {"min_conf": 0.5, "min_height_px": 48.0, "edge_px": 3.0, "aspect_min": 1.2, "aspect_max": 5.0,
           "good_height_px": 128.0}

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------------------------------------------- crops
def crop_quality(bb, w: float, h: float, conf: float, q: Optional[dict] = None) -> float:
    """0 when the crop must not be embedded (low confidence, truncated by the border, too small, odd shape), else a
    weight in (0, 1]: detector confidence x size factor (saturating at good_height_px)."""
    q = {**QUALITY, **(q or {})}
    x1, y1, x2, y2 = (float(v) for v in bb)
    bw, bh = x2 - x1, y2 - y1
    if conf < q["min_conf"] or bh < q["min_height_px"] or bw <= 1.0:
        return 0.0
    e = q["edge_px"]
    if x1 <= e or y1 <= e or x2 >= w - 1 - e or y2 >= h - 1 - e:
        return 0.0  # truncated by the image border: part of the person is missing
    if not q["aspect_min"] <= bh / bw <= q["aspect_max"]:
        return 0.0
    return float(conf * min(1.0, bh / q["good_height_px"]))


def extract_crop(frame: np.ndarray, bb, pad: float = 0.04) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bb)
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    xa, ya = max(0, int(x1 - px)), max(0, int(y1 - py))
    xb, yb = min(w, int(x2 + px + 1)), min(h, int(y2 + py + 1))
    return frame[ya:yb, xa:xb].copy()


def is_monochrome(img: np.ndarray, thresh: float = 4.0) -> bool:
    """True for gray / IR images: mean |B-G| + |G-R| (8-bit units) below `thresh` on a pixel subsample."""
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    s = img[::4, ::4].astype(np.int16)
    if s.size == 0:
        return False
    return float((np.abs(s[..., 0] - s[..., 1]) + np.abs(s[..., 1] - s[..., 2])).mean()) < thresh


def to_gray3(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.merge([g, g, g])


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------------------------------- embedders
class Embedder(Protocol):
    info: dict

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...  # [n, d] float32, rows L2-normalised


class OSNetEmbedder:
    """Pretrained OSNet person Re-ID backbone. Batched; fp16 on CUDA."""

    def __init__(self, weights: str, arch: Optional[str] = None, device: str = "auto", fp16: bool = True,
                 batch_size: int = 32, size: tuple[int, int] = (256, 128)):
        import torch
        from . import osnet
        from .detect import resolve_device
        self.device = resolve_device(device)
        arch = arch or KNOWN.get(Path(weights).name, {}).get("arch") or osnet.arch_from_name(weights)
        self.model = osnet.load(weights, arch).to(self.device)
        self.half = bool(fp16 and self.device.startswith("cuda"))
        if self.half:
            self.model.half()
        self.size, self.bs, self._torch = tuple(size), max(1, int(batch_size)), torch
        meta = KNOWN.get(Path(weights).name, {})
        self.sim = meta.get("sim")
        self.info = {"name": arch, "weights": Path(weights).name, "kind": "osnet", "device": self.device,
                     "fp16": self.half, "input_hw": list(self.size), "dim": int(self.model.feature_dim),
                     "trained_on": meta.get("trained_on", "unknown"), "source": meta.get("source", "user-supplied")}

    def _prep(self, crops: list[np.ndarray]) -> np.ndarray:
        H, W = self.size
        arr = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in crops])[..., ::-1]
        arr = (arr.astype(np.float32) / 255.0 - _MEAN) / _STD
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.info["dim"]), np.float32)
        torch, outs = self._torch, []
        with torch.inference_mode():
            for i in range(0, len(crops), self.bs):
                x = torch.from_numpy(self._prep(crops[i:i + self.bs])).to(self.device)
                if self.half:
                    x = x.half()
                outs.append(self.model(x).float().cpu().numpy())
        return l2n(np.concatenate(outs))


class HandcraftedEmbedder:
    """No learned weights: HSV colour histograms in head/torso/legs stripes + gradient-orientation histograms."""

    def __init__(self):
        self.sim = HANDCRAFTED_SIM
        self.info = {"name": "hsv-hog-v1", "kind": "handcrafted", "device": "cpu", "fp16": False, "dim": 192,
                     "trained_on": "nothing (hand-crafted features)", "source": "argonix"}

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), 192), np.float32)
        for i, c in enumerate(crops):
            r = cv2.resize(c, (32, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
            feats = []
            for s in range(3):
                part = np.ascontiguousarray(hsv[s * 21:(s + 1) * 21 + (1 if s == 2 else 0)])
                hs = cv2.calcHist([part], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
                v = cv2.calcHist([part], [2], None, [8], [0, 256]).ravel()
                feats += [hs / (hs.sum() + 1e-6), v / (v.sum() + 1e-6)]
            g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mag, ang = cv2.cartToPolar(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1), angleInDegrees=True)
            ang = ang % 180.0
            for cy in range(4):
                for cx in range(2):
                    sl = (slice(cy * 16, (cy + 1) * 16), slice(cx * 16, (cx + 1) * 16))
                    hh, _ = np.histogram(ang[sl], bins=9, range=(0, 180), weights=mag[sl])
                    feats.append(hh / (hh.sum() + 1e-6))
            out[i] = np.sqrt(np.concatenate(feats))  # Hellinger kernel -> cosine
        return l2n(out)


def resolve_model(model: str, models_dir: Path) -> Optional[Path]:
    """'auto' -> the first known checkpoint present in <models_dir>/reid; a name or path otherwise (None if absent)."""
    d = Path(models_dir) / "reid"
    if model in (None, "", "auto"):
        for n in AUTO_ORDER:
            if (d / n).exists():
                return d / n
        return None
    if model == "handcrafted":
        return None
    p = Path(model)
    for c in (p, d / p.name, Path(models_dir) / p.name):
        if c.exists():
            return c
    return None


def make_embedder(cfg: dict, models_dir: Path) -> tuple[Embedder, list[str]]:
    """(embedder, warnings). Falls back to the hand-crafted embedder when no checkpoint can be loaded."""
    warn: list[str] = []
    model = cfg.get("model", "auto")
    if model != "handcrafted":
        p = resolve_model(model, models_dir)
        if p is None:
            warn.append(f"Re-ID model {model!r} not found under {Path(models_dir) / 'reid'}; using the hand-crafted "
                        "embedder (weak)")
        else:
            try:
                return OSNetEmbedder(str(p), cfg.get("arch"), cfg.get("device", "auto"), bool(cfg.get("fp16", True)),
                                     int(cfg.get("batch_size", 32))), warn
            except Exception as e:  # noqa: BLE001 - a broken checkpoint must not stop the site
                warn.append(f"Re-ID model {p.name} failed to load ({e!r})"[:300] + "; using the hand-crafted embedder")
    return HandcraftedEmbedder(), warn


# ---------------------------------------------------------------------------------------------------- service
class EmbeddingService:
    """Shared batching thread in front of one embedder.

    submit(key, crop, meta) never blocks: when the queue is full the OLDEST request is dropped (counted). meta keys:
    "color" (embed the crop as is), "gray" (also embed its 3-channel gray version), "mono" (the crop is already
    colourless: its colour embedding doubles as the gray one). The callback receives (key, e_color, e_gray, meta).
    `sync=True` embeds in the caller's thread (tests, offline evaluation).
    """

    def __init__(self, embedder: Embedder, callback: Callable, max_batch: int = 32, max_wait_ms: float = 20.0,
                 max_queue: int = 512, sync: bool = False):
        self.embedder, self.callback = embedder, callback
        self.max_batch, self.max_wait = max(1, int(max_batch)), max(0.0, float(max_wait_ms)) / 1000.0
        self.max_queue, self.sync = int(max_queue), sync
        self._q: deque = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.stats = {"crops": 0, "images": 0, "batches": 0, "dropped": 0, "errors": 0, "last_error": "",
                      "embed_ms": 0.0, "ms_per_image": 0.0, "max_batch_seen": 0}
        self._thread = None
        if not sync:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="reid-embed")
            self._thread.start()

    @property
    def queue_len(self) -> int:
        return len(self._q)

    def submit(self, key, crop: np.ndarray, meta: dict) -> None:
        if self.sync:
            self._run([(key, crop, meta)])
            return
        with self._cond:
            if self._closed:
                return
            if len(self._q) >= self.max_queue:
                self._q.popleft()
                self.stats["dropped"] += 1
            self._q.append((key, crop, meta, time.perf_counter()))
            self._cond.notify_all()

    def _gather(self):
        with self._cond:
            while not self._q and not self._closed:
                self._cond.wait(0.5)
            if self._closed:
                return None
            deadline = self._q[0][3] + self.max_wait
            while len(self._q) < self.max_batch and not self._closed:
                now = time.perf_counter()
                if now >= deadline:
                    break
                self._cond.wait(deadline - now)
            n = min(self.max_batch, len(self._q))
            return [self._q.popleft()[:3] for _ in range(n)]

    def _loop(self) -> None:
        while True:
            items = self._gather()
            if items is None:
                return
            self._run(items)

    def _run(self, items) -> None:
        imgs, slots = [], []
        for i, (key, crop, meta) in enumerate(items):
            color, mono = meta.get("color", True), bool(meta.get("mono"))
            if color:
                slots.append((i, "color"))
                imgs.append(crop)
            if meta.get("gray") and not (mono and color):  # a colourless crop's colour embedding IS its gray one
                slots.append((i, "gray"))
                imgs.append(to_gray3(crop))
        if not imgs:
            return
        t0 = time.perf_counter()
        try:
            E = self.embedder.embed(imgs)
        except Exception as e:  # noqa: BLE001 - keep serving
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(e)[:300]
            return
        dt = (time.perf_counter() - t0) * 1000
        s = self.stats
        s["crops"] += len(items)
        s["images"] += len(imgs)
        s["batches"] += 1
        s["max_batch_seen"] = max(s["max_batch_seen"], len(imgs))
        s["embed_ms"] = round(dt if s["batches"] == 1 else 0.9 * s["embed_ms"] + 0.1 * dt, 2)
        per = dt / len(imgs)
        s["ms_per_image"] = round(per if s["batches"] == 1 else 0.9 * s["ms_per_image"] + 0.1 * per, 3)
        out: dict[int, dict] = {}
        for (i, mode), e in zip(slots, E):
            out.setdefault(i, {})[mode] = e
        for i, (key, crop, meta) in enumerate(items):
            got = out.get(i, {})
            ec = got.get("color")
            eg = got.get("gray", ec if meta.get("mono") and meta.get("gray") else None)
            try:
                self.callback(key, ec, eg, meta)
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_error"] = f"callback: {e!r}"[:300]

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
