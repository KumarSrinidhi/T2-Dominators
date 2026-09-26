"""Shared batched inference: ONE detector on the GPU serving every camera, with a per-camera tracker.

Each camera gets an `InferenceClient` exposing the same `track(frame) -> list[Det]` interface as
`UltralyticsTracker`, so `CameraWorker` is unchanged. `track()` enqueues the frame and blocks. A single inference
thread gathers pending frames from all cameras (up to `max_batch`, waiting at most `max_wait_ms` after the oldest
request, and not at all once every recently active camera has a frame queued), runs one batched `predict`, then
updates each camera's OWN ByteTrack instance in submission order and hands the tracks back to the caller.

Track IDs are numbered per camera (1, 2, ...): every camera's tracker gets its own ID counter, so a track can never
continue across cameras and cameras cannot perturb each other's numbering. Detector or tracker exceptions are
re-raised in the calling camera thread.
"""
from __future__ import annotations

import atexit
import itertools
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

from .detect import Det, engine_batch, resolve_device, resolve_weights, split_track

_TRACKER_INIT_LOCK = threading.Lock()
_LIVE: "weakref.WeakSet[BatchedInference]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:  # weak registry: atexit must not keep closed services (and their models) alive
    for svc in list(_LIVE):
        svc.close(timeout=2.0)


class _Request:
    __slots__ = ("cam", "frame", "t_submit", "done", "result", "error", "callback", "t_start", "t_infer", "t_track",
                 "t_done", "batch_n")

    def __init__(self, cam: str, frame, callback: Optional[Callable[["_Request"], None]] = None):
        self.cam, self.frame, self.t_submit = cam, frame, time.perf_counter()
        self.done = threading.Event()
        self.result: Optional[list[Det]] = None
        self.error: Optional[BaseException] = None
        self.callback = callback
        # timings (perf_counter): batch start, end of the batched predict, start/end of this camera's tracker update
        self.t_start = self.t_infer = self.t_track = self.t_done = 0.0
        self.batch_n = 0

    def finish(self) -> None:
        """Mark done and run the completion callback (never raises into the inference thread)."""
        self.t_done = self.t_done or time.perf_counter()
        self.frame = None  # the result no longer needs the pixels; do not keep them alive through the caller
        self.done.set()
        cb, self.callback = self.callback, None
        if cb is not None:
            try:
                cb(self)
            except Exception:  # noqa: BLE001 - a broken callback must not kill the shared inference thread
                pass


def _make_tracker(tracker_yaml: str):
    """A fresh Ultralytics tracker from its YAML (ByteTrack by default) with a PRIVATE track-ID counter.

    Ultralytics numbers tracks from one class-level counter shared by every tracker in the process, and each new
    tracker resets it. We save/restore that counter and give this tracker a track class whose `next_id` draws from
    its own counter instead.
    """
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
    if cfg.tracker_type not in TRACKER_MAP:
        raise ValueError(f"unsupported tracker_type {cfg.tracker_type!r}")
    if getattr(cfg, "with_reid", False):
        cfg.with_reid = False  # ReID needs detector features we do not route through the batch; use plain IoU tracking
    with _TRACKER_INIT_LOCK:
        saved = BaseTrack._count
        trk = TRACKER_MAP[cfg.tracker_type](args=cfg)
        BaseTrack._count = saved
    counter = itertools.count(1)
    base = trk.track_class
    trk.track_class = type(f"Camera{base.__name__}", (base,), {"next_id": staticmethod(lambda: next(counter))})
    return trk


class InferenceClient:
    """Per-camera handle with the `TrackerBackend` interface."""

    def __init__(self, service: "BatchedInference", camera_id: str):
        self.service, self.camera_id = service, camera_id
        self.info = dict(service.info)

    def track(self, frame) -> list[Det]:
        return self.service.submit(self.camera_id, frame)

    def close(self) -> None:
        self.service.release(self.camera_id)


class BatchedInference:
    """One model, many cameras. `model` may be injected (tests); otherwise Ultralytics YOLO is loaded from `weights`.

    An injected model must provide `predict(frames: list, **kw) -> list[result]` where each result has `.boxes`
    (Ultralytics `Boxes`-like: `xyxy/xywh/conf/cls`, boolean indexing, `.cpu().numpy()`) and `.names`.
    """

    def __init__(self, model: Any = None, *, weights: str = "auto", device: str = "auto", imgsz: int = 640,
                 conf: float = 0.3, classes: Optional[tuple[int, ...]] = (0,), tracker: str = "bytetrack.yaml",
                 models_dir: Optional[Path] = None, max_batch: int = 16, max_wait_ms: float = 8.0,
                 request_timeout_s: float = 60.0, active_window_s: float = 2.0, warmup: bool = True,
                 extra_classes: tuple[int, ...] = (), extra_conf: Optional[float] = None):
        self.device = resolve_device(device)
        # extra (object) classes: detected in the same batch, tracked by a second per-camera tracker (split_track)
        self.person_classes = tuple(int(c) for c in (classes or ()))
        self.extra = tuple(int(c) for c in (extra_classes or ()) if int(c) not in self.person_classes)
        self.conf, self.extra_conf = float(conf), float(conf if extra_conf is None else extra_conf)
        self.tracker_yaml = tracker
        runtime, fp16, name = "custom", False, type(model).__name__
        if model is None:
            from ultralytics import YOLO
            weights = resolve_weights(weights, self.device, imgsz, models_dir)
            model = YOLO(weights, task="detect")
            engine = weights.endswith(".engine")
            runtime, name = ("tensorrt" if engine else "torch"), os.path.basename(weights)
            fp16 = self.device.startswith("cuda")  # .pt: quantize=16 below; engines were exported fp16
            eb = engine_batch(weights) if engine else None
            if eb:
                max_batch = min(max_batch, eb)
            quantize = 16 if fp16 and not engine else None
        else:
            quantize = None
        self.model = model
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, float(max_wait_ms)) / 1000.0
        self.request_timeout_s = request_timeout_s
        self.active_window_s = active_window_s
        # rect=False: always letterbox to a fixed imgsz x imgsz input. With rect=True the input shape depends on the
        # batch's mix of camera aspect ratios, and every new shape cost a ~0.8-1.7 s stall on the RTX 3050 bench.
        pred_classes = list(self.person_classes + self.extra) if classes else None
        self.predict_kw = dict(conf=min(self.conf, self.extra_conf) if self.extra else conf, imgsz=imgsz,
                               device=self.device, verbose=False, quantize=quantize, classes=pred_classes, rect=False)
        try:
            import ultralytics
            fw = f"ultralytics {ultralytics.__version__}"
        except ImportError:  # pragma: no cover
            fw = "unknown"
        self.info = {"detector": name, "tracker": tracker, "framework": fw, "conf_threshold": conf, "imgsz": imgsz,
                     "device": self.device, "fp16": fp16, "runtime": runtime, "batched": True,
                     "max_batch": self.max_batch, "max_wait_ms": max_wait_ms}
        if self.extra:
            self.info.update(extra_classes=list(self.extra), extra_conf=self.extra_conf)

        self._cond = threading.Condition()
        self._pending: list[_Request] = []
        self._trackers: dict[str, Any] = {}
        self._otrackers: dict[str, Any] = {}  # per-camera object trackers (only with extra classes)
        self._last_seen: dict[str, float] = {}
        self._closed = False
        self.stats = {"frames": 0, "batches": 0, "batch_hist": {}, "infer_ms": 0.0, "track_ms": 0.0,
                      "errors": 0, "last_error": ""}
        self._warmup = (imgsz if warmup and runtime != "custom" else 0)
        self._ready = threading.Event()
        self._warmup_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="batched-inference")
        self._thread.start()
        self._ready.wait()
        if self._warmup_error is not None:
            raise self._warmup_error
        _LIVE.add(self)

    # -------------------------------------------------------------------------------------- public API
    def client(self, camera_id: str) -> InferenceClient:
        with self._cond:
            if self._closed:
                raise RuntimeError("inference service is closed")
            if camera_id in self._trackers:
                raise ValueError(f"camera {camera_id!r} already has a client")
        trk = _make_tracker(self.tracker_yaml)
        otrk = _make_tracker(self.tracker_yaml) if self.extra else None
        with self._cond:
            self._trackers[camera_id] = trk
            if otrk is not None:
                self._otrackers[camera_id] = otrk
        return InferenceClient(self, camera_id)

    def release(self, camera_id: str) -> None:
        with self._cond:
            self._trackers.pop(camera_id, None)
            self._otrackers.pop(camera_id, None)
            self._last_seen.pop(camera_id, None)

    def has_client(self, camera_id: str) -> bool:
        with self._cond:
            return camera_id in self._trackers

    @property
    def queue_depth(self) -> int:
        """Frames waiting for a batch (not yet on the GPU)."""
        with self._cond:
            return len(self._pending)

    def _enqueue(self, req: _Request) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("inference service is closed")
            if req.cam not in self._trackers:
                raise KeyError(f"unknown camera {req.cam!r}; call client() first")
            self._last_seen[req.cam] = req.t_submit
            self._pending.append(req)
            self._cond.notify_all()

    def submit_nowait(self, camera_id: str, frame, callback: Callable[[_Request], None]) -> _Request:
        """Queue a frame without blocking. `callback(req)` runs exactly once, on the inference thread (or on the
        closing thread), with `req.result` or `req.error` set. Used by the network node (argonix/infer_server.py).
        Frames of one camera are tracked in submission order."""
        req = _Request(camera_id, frame, callback)
        self._enqueue(req)
        return req

    def submit(self, camera_id: str, frame) -> list[Det]:
        req = _Request(camera_id, frame)
        self._enqueue(req)
        if not req.done.wait(self.request_timeout_s):
            with self._cond:
                if req in self._pending:
                    self._pending.remove(req)
            raise TimeoutError(f"inference for {camera_id} not served within {self.request_timeout_s}s")
        if req.error is not None:
            raise req.error
        return req.result  # type: ignore[return-value]

    def close(self, timeout: float = 10.0) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)
        with self._cond:
            left, self._pending = self._pending, []
        for r in left:  # requests that never reached the model
            r.error = RuntimeError("inference service closed")
            r.finish()

    @property
    def closed(self) -> bool:
        return self._closed

    # -------------------------------------------------------------------------------------- inference thread
    def _expected(self, now: float) -> int:
        """Cameras that submitted recently; once all of them are queued, waiting longer cannot grow the batch."""
        return sum(1 for t in self._last_seen.values() if now - t <= self.active_window_s)

    def _gather(self) -> Optional[list[_Request]]:
        with self._cond:
            while not self._pending and not self._closed:
                self._cond.wait()
            if self._closed:
                return None
            deadline = self._pending[0].t_submit + self.max_wait
            while len(self._pending) < self.max_batch and not self._closed:
                now = time.perf_counter()
                if len({r.cam for r in self._pending}) >= self._expected(now) or now >= deadline:
                    break
                self._cond.wait(deadline - now)
            batch, self._pending = self._pending[:self.max_batch], self._pending[self.max_batch:]
            return batch

    def _warm(self) -> None:
        """Run every batch size 1..max_batch once, ON THE INFERENCE THREAD.

        Measured on the RTX 3050 (torch 2.11, .pt fp16): the first call of each new batch size costs ~1.4-1.7 s, and
        that cost is per thread (cuDNN/cuBLAS handles and plans are thread-local), so warming on the constructing
        thread does not help. Left to run time, the stalls land on live cameras whenever load produces a new size.
        """
        import numpy as np
        t0 = time.perf_counter()
        blank = np.zeros((self._warmup, self._warmup, 3), dtype=np.uint8)
        for b in range(1, self.max_batch + 1):
            self.model.predict([blank] * b, **self.predict_kw)
        self.stats["warmup_s"] = round(time.perf_counter() - t0, 1)

    def _loop(self) -> None:
        try:
            if self._warmup:
                self._warm()
        except BaseException as e:
            self._warmup_error = e
            self._closed = True
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                batch = self._gather()
                if batch is None:
                    return
                self._run_batch(batch)
        except BaseException as e:  # never leave callers hanging
            with self._cond:
                self._closed = True
                left, self._pending = self._pending, []
            for r in left:
                r.error = RuntimeError(f"inference thread died: {e!r}")
                r.finish()
            raise

    def _run_batch(self, batch: list[_Request]) -> None:
        t0 = time.perf_counter()
        for r in batch:
            r.t_start, r.batch_n = t0, len(batch)
        try:
            results = self.model.predict([r.frame for r in batch], **self.predict_kw)
            if len(results) != len(batch):
                raise RuntimeError(f"model returned {len(results)} results for a batch of {len(batch)}")
        except Exception as e:
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(e)
            for r in batch:
                r.error = e
                r.finish()
            return
        t1 = time.perf_counter()
        for r, res in zip(batch, results):  # submission order == per-camera frame order
            r.t_infer, r.t_track = t1, time.perf_counter()
            try:
                trk = self._trackers.get(r.cam)
                if trk is None:
                    raise RuntimeError(f"camera {r.cam!r} was released")
                if self.extra:
                    r.result = split_track(res, r.frame, trk, self._otrackers.get(r.cam), self.person_classes,
                                           self.extra, self.conf, self.extra_conf)
                else:
                    r.result = self._update_tracker(trk, res, r.frame)
            except Exception as e:
                self.stats["errors"] += 1
                self.stats["last_error"] = repr(e)
                r.error = e
            r.finish()
        t2 = time.perf_counter()
        s = self.stats
        s["frames"] += len(batch)
        s["batches"] += 1
        s["batch_hist"][len(batch)] = s["batch_hist"].get(len(batch), 0) + 1
        for k, v in (("infer_ms", (t1 - t0) * 1000), ("track_ms", (t2 - t1) * 1000)):
            s[k] = round(v if s["batches"] == 1 else 0.9 * s[k] + 0.1 * v, 2)

    @staticmethod
    def _update_tracker(trk, res, frame) -> list[Det]:
        import numpy as np
        from ultralytics.engine.results import Boxes

        boxes = res.boxes
        if boxes is None:
            boxes = Boxes(np.zeros((0, 6), dtype=np.float32), frame.shape[:2])
        tracks = trk.update(boxes.cpu().numpy(), frame)
        names = res.names
        out: list[Det] = []
        for t in tracks:  # x1, y1, x2, y2, track_id, score, cls, idx
            out.append((int(t[4]), (float(t[0]), float(t[1]), float(t[2]), float(t[3])), names[int(t[6])], float(t[5])))
        return out

    @property
    def mean_batch(self) -> float:
        return self.stats["frames"] / self.stats["batches"] if self.stats["batches"] else 0.0
