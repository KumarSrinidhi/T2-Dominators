"""Model-agnostic detector+tracker interface. One backend object per camera so tracker state never mixes streams.

Three implementations share the `track(frame) -> list[Det]` interface:
- `UltralyticsTracker`: one model + tracker per camera (CPU default, and the path the tests use).
- `argonix.inference.BatchedInference` clients: one model on the GPU batching frames from every camera, with a
  per-camera tracker (GPU default).
- `argonix.remote_inference.RemoteTrackerBackend`: detection + tracking on a remote GPU node
  (argonix/infer_server.py) over a WebSocket per camera, with an optional local CPU fallback.
`make_backend_factory` picks between them from the `detector` config.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Protocol

Det = tuple[int, tuple[float, float, float, float], str, float]  # track_id, bbox xyxy, class, conf

# Object classes requested via `detector.extra_classes` (e.g. COCO backpack 24, handbag 26, suitcase 28) are tracked
# by a SEPARATE tracker whose IDs are offset by this constant, so they never collide with person track IDs and never
# perturb person tracking (persons are filtered and tracked exactly as without extra classes).
OBJECT_ID_OFFSET = 1_000_000

AUTO_VRAM_GB = 4.0  # "auto" picks the small (s) model on CUDA with at least this much total VRAM, else nano


class TrackerBackend(Protocol):
    info: dict

    def track(self, frame) -> list[Det]: ...


# ---------------------------------------------------------------------------------------------- model selection
def resolve_device(device: str = "auto") -> str:
    device = str(device).strip()
    if device.isdigit():  # Ultralytics-style GPU index ("0") -> torch device string, so torch models accept it too
        return f"cuda:{device}"
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cuda_vram_gb(device: str = "cuda:0") -> float:
    """Total VRAM of `device` in GiB (0 when not CUDA)."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch
        return torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
    except Exception:
        return 0.0


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def engine_name(stem: str, imgsz: int, batch: int) -> str:
    """File name used by tools/bench_gpu.py --export, e.g. yolo26s-640-b16.engine."""
    return f"{stem}-{imgsz}-b{batch}.engine"


def engine_batch(path: str) -> Optional[int]:
    """Max batch baked into an exported engine, parsed from its name (None when unknown)."""
    m = re.search(r"-b(\d+)\.engine$", os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def find_engine(models_dir: Path, stem: str, imgsz: int) -> Optional[Path]:
    """Largest-batch engine for `stem` exported at `imgsz` (a dynamic engine also serves smaller inputs)."""
    if not models_dir.is_dir():
        return None
    cands = []
    for p in models_dir.glob(f"{stem}-*-b*.engine"):
        m = re.match(rf"{re.escape(stem)}-(\d+)-b(\d+)\.engine$", p.name)
        if m and int(m.group(1)) >= imgsz:
            cands.append((int(m.group(1)) != imgsz, -int(m.group(2)), p))  # exact imgsz first, then larger batch
    return sorted(cands)[0][2] if cands else None


def resolve_weights(weights: str = "auto", device: str = "cpu", imgsz: int = 640,
                    models_dir: Optional[Path] = None) -> str:
    """Turn the configured `weights` into a loadable file.

    - "auto": yolo26s on CUDA with >= AUTO_VRAM_GB total VRAM, otherwise yolo26n; on CUDA with the `tensorrt`
      package importable, a matching exported engine in `models_dir` (see `engine_name`) is preferred.
    - A configured `.engine` that is missing, or unusable (no CUDA / no tensorrt), falls back to the sibling `.pt`
      (run in fp16 on CUDA).
    - An explicit `.pt` is used as given. A bare name that is not on disk is returned as-is (Ultralytics downloads
      official weights by name).
    """
    models_dir = Path(models_dir) if models_dir else None
    cuda = device.startswith("cuda")
    trt_ok = cuda and tensorrt_available()
    if weights in (None, "", "auto"):
        stem = "yolo26s" if cuda and cuda_vram_gb(device) >= AUTO_VRAM_GB else "yolo26n"
        eng = find_engine(models_dir, stem, imgsz) if trt_ok and models_dir else None
        if eng is not None:
            return str(eng)
        return str(models_dir / f"{stem}.pt") if models_dir and (models_dir / f"{stem}.pt").exists() else f"{stem}.pt"
    w = Path(weights)
    if w.suffix == ".engine":
        if trt_ok and w.exists():
            return str(w)
        stem = re.sub(r"-\d+-b\d+$", "", w.stem)
        for pt in (w.with_name(stem + ".pt"), models_dir / f"{stem}.pt" if models_dir else None):
            if pt is not None and pt.exists():
                return str(pt)
        return stem + ".pt"
    return str(w)


# ---------------------------------------------------------------------------------------------- split tracking
def split_track(res, frame, person_trk, object_trk, person_cls, object_cls, person_conf: float,
                object_conf: float) -> list[Det]:
    """Split one detector result into person and object boxes and update each group's OWN tracker.

    Person boxes below `person_conf` are dropped before tracking, so person tracks are the same as with a
    person-only detector run at that confidence. Object track IDs are offset by OBJECT_ID_OFFSET.
    """
    import numpy as np
    from ultralytics.engine.results import Boxes

    boxes = res.boxes
    if boxes is None:
        boxes = Boxes(np.zeros((0, 6), dtype=np.float32), frame.shape[:2])
    b = boxes.cpu().numpy()
    cls = np.asarray(b.cls).astype(int).reshape(-1)
    cf = np.asarray(b.conf).reshape(-1)
    names = res.names
    out: list[Det] = []
    groups = [(person_trk, np.isin(cls, list(person_cls)) & (cf >= person_conf), 0)]
    if object_trk is not None:
        groups.append((object_trk, np.isin(cls, list(object_cls)) & (cf >= object_conf), OBJECT_ID_OFFSET))
    for trk, mask, off in groups:
        for t in trk.update(b[mask], frame):  # x1, y1, x2, y2, track_id, score, cls, idx
            out.append((int(t[4]) + off, (float(t[0]), float(t[1]), float(t[2]), float(t[3])), names[int(t[6])],
                        float(t[5])))
    return out


# ---------------------------------------------------------------------------------------------- per-camera backend
class UltralyticsTracker:
    """YOLO + ByteTrack/BoT-SORT via Ultralytics (AGPL-3.0; acceptable for this personal demo).

    With `extra_classes` (object classes such as bags) the model runs in predict mode on persons + extras and each
    group gets its own tracker (`split_track`); without them the original `model.track` path is used unchanged.
    """

    def __init__(self, weights: str = "yolo11n.pt", tracker: str = "bytetrack.yaml", conf: float = 0.3,
                 imgsz: int = 640, classes: tuple[int, ...] = (0,), device: str = "auto",
                 models_dir: Optional[Path] = None, extra_classes: tuple[int, ...] = (),
                 extra_conf: Optional[float] = None):
        from ultralytics import YOLO
        device = resolve_device(device)
        weights = resolve_weights(weights, device, imgsz, models_dir)
        self.model = YOLO(weights, task="detect")
        engine = weights.endswith(".engine")
        half = device.startswith("cuda")  # .pt runs fp16 via quantize=16; an engine's precision is baked in at export
        self.kw = dict(persist=True, tracker=tracker, conf=conf, imgsz=imgsz, classes=list(classes),
                       device=device, quantize=16 if half and not engine else None, verbose=False)
        import ultralytics
        self.info = {"detector": os.path.basename(weights), "tracker": tracker, "framework": f"ultralytics {ultralytics.__version__}",
                     "conf_threshold": conf, "imgsz": imgsz, "device": device, "fp16": half,
                     "runtime": "tensorrt" if engine else "torch", "batched": False}
        self.classes = tuple(int(c) for c in classes)
        self.extra = tuple(int(c) for c in (extra_classes or ()) if int(c) not in self.classes)
        if self.extra:
            from .inference import _make_tracker
            self.conf, self.extra_conf = float(conf), float(conf if extra_conf is None else extra_conf)
            self.pkw = {k: v for k, v in self.kw.items() if k not in ("persist", "tracker")}
            self.pkw.update(conf=min(self.conf, self.extra_conf), classes=list(self.classes + self.extra))
            self.trk, self.otrk = _make_tracker(tracker), _make_tracker(tracker)
            self.info.update(extra_classes=list(self.extra), extra_conf=self.extra_conf,
                             object_id_offset=OBJECT_ID_OFFSET)

    def track(self, frame) -> list[Det]:
        if self.extra:
            r = self.model.predict(frame, **self.pkw)[0]
            return split_track(r, frame, self.trk, self.otrk, self.classes, self.extra, self.conf, self.extra_conf)
        r = self.model.track(frame, **self.kw)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        names = r.names
        out = []
        for tid, xyxy, c, cf in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist(),
                                    r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            out.append((tid, tuple(xyxy), names[c], float(cf)))
        return out


def make_backend_factory(det: dict, models_dir: Optional[Path] = None) -> Callable[[dict], TrackerBackend]:
    """The backend factory for `detector`, with optional track stitching (`detector.stitch`, argonix/stitch.py)
    applied to every backend it makes: local, batched or remote."""
    from .stitch import wrap_factory
    return wrap_factory(_make_backend_factory(det, models_dir), det)


def _make_backend_factory(det: dict, models_dir: Optional[Path] = None) -> Callable[[dict], TrackerBackend]:
    """Default backend factory for `Site` from the config's `detector` section.

    `detector.remote` ({"url": "ws://host:8200", "token_file": ..., "fallback": "local"}) sends every camera's
    frames to a remote inference node. Otherwise `detector.batched` (default: true on CUDA, false on CPU) hands
    every camera a client of ONE shared `BatchedInference`, or each camera gets its own `UltralyticsTracker`.
    The returned callable has a `close()` that shuts the shared service / connections down (a no-op for the
    per-camera path).
    """
    if det.get("remote"):
        from .remote_inference import make_remote_factory
        return make_remote_factory(det, models_dir)
    device = resolve_device(det.get("device", "auto"))
    imgsz = int(det.get("imgsz", 640))
    weights = det.get("weights", "auto")
    tracker = det.get("tracker", "bytetrack.yaml")
    conf = float(det.get("conf", 0.3))
    classes = tuple(det.get("classes", (0,)))
    extra = tuple(int(c) for c in (det.get("extra_classes") or ()))  # object classes with their own tracker
    extra_conf = det.get("extra_conf")
    extra_conf = float(extra_conf) if extra_conf is not None else None
    batched = det.get("batched")
    if batched is None:
        batched = device.startswith("cuda")

    if not batched:
        def per_camera(cam: dict) -> TrackerBackend:
            return UltralyticsTracker(weights, tracker, conf, imgsz, classes, device, models_dir, extra, extra_conf)
        per_camera.close = lambda: None
        return per_camera

    from .inference import BatchedInference
    state: dict = {}
    lock = threading.Lock()  # one service even when a camera is added while prewarm() is still building it

    def service() -> BatchedInference:
        with lock:
            if "svc" not in state:
                state["svc"] = BatchedInference(weights=weights, device=device, imgsz=imgsz, conf=conf,
                                                classes=classes, tracker=tracker, models_dir=models_dir,
                                                max_batch=int(det.get("max_batch", 16)),
                                                max_wait_ms=float(det.get("max_wait_ms", 8.0)),
                                                extra_classes=extra, extra_conf=extra_conf)
            return state["svc"]

    def shared(cam: dict) -> TrackerBackend:
        return service().client(cam["id"])

    shared.prewarm = service  # load + warm the model before the first camera exists (Site.start, no cameras)
    shared.close = lambda: state["svc"].close() if "svc" in state else None
    return shared
