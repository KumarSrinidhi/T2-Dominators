"""BatchedInference: batching, routing, per-camera track isolation, wait/max-batch behaviour, errors, shutdown.

Runs on CPU with a fake model (no weights). The fake detector reads a box position encoded in the frame's first
pixel, so every result can be traced back to the frame (and camera) that produced it. Tracking uses the real
Ultralytics ByteTrack.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from ultralytics.engine.results import Boxes

from argonix.detect import engine_batch, engine_name, make_backend_factory, resolve_device, resolve_weights
from argonix.inference import BatchedInference, InferenceClient


class _Res:
    def __init__(self, boxes, names):
        self.boxes, self.names = boxes, names


class FakeModel:
    """Detects one 'person' whose x position is frame[0, 0, 0] (0 = nothing); frame[0, 0, 1] == 255 raises."""

    def __init__(self, delay_s: float = 0.0):
        self.batches: list[list[int]] = []  # x positions seen per batch
        self.delay_s = delay_s
        self.lock = threading.Lock()

    def predict(self, frames, **kw):
        assert kw.get("classes") == [0]
        if any(int(f[0, 0, 1]) == 255 for f in frames):
            raise ValueError("boom")
        with self.lock:
            self.batches.append([int(f[0, 0, 0]) for f in frames])
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for f in frames:
            x = int(f[0, 0, 0])
            data = np.array([[x, 50, x + 30, 150, 0.9, 0]], np.float32) if x else np.zeros((0, 6), np.float32)
            out.append(_Res(Boxes(data, f.shape[:2]), {0: "person"}))
        return out


def frame(x: int, bad: bool = False):
    f = np.zeros((240, 320, 3), np.uint8)
    f[0, 0, 0] = x
    if bad:
        f[0, 0, 1] = 255
    return f


def service(**kw) -> tuple[BatchedInference, FakeModel]:
    m = FakeModel(kw.pop("delay_s", 0.0))
    return BatchedInference(m, device="cpu", warmup=False, **kw), m


def run_parallel(fns):
    out, errs = [None] * len(fns), []
    barrier = threading.Barrier(len(fns))

    def go(i, fn):
        barrier.wait()
        try:
            out[i] = fn()
        except Exception as e:  # pragma: no cover - surfaced below
            errs.append(e)

    ts = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(fns)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert not errs, errs
    return out


def test_three_cameras_batched_and_routed():
    svc, m = service(max_batch=8, max_wait_ms=300)
    clients = [svc.client(c) for c in ("a", "b", "c")]
    xs = {"a": 20, "b": 100, "c": 180}
    for c in clients:  # prime: every camera becomes "active" so the service waits for all three
        c.track(frame(xs[c.camera_id]))
    m.batches.clear()
    res = run_parallel([lambda c=c: c.track(frame(xs[c.camera_id])) for c in clients])
    assert sorted(m.batches[0]) == [20, 100, 180]  # one batch carried all three cameras
    for c, dets in zip(clients, res):
        assert len(dets) == 1
        assert abs(dets[0][1][0] - xs[c.camera_id]) < 1  # routed back to the camera that sent it
        assert dets[0][2] == "person"
    assert svc.stats["batch_hist"].get(3) == 1
    svc.close()


def test_track_ids_isolated_per_camera():
    svc, _ = service(max_batch=4, max_wait_ms=50)
    a, b = svc.client("a"), svc.client("b")
    ids_a, ids_b = set(), set()
    for _ in range(6):  # the SAME box in both cameras: a shared tracker would merge them into one track
        ra, rb = run_parallel([lambda: a.track(frame(60)), lambda: b.track(frame(60))])
        ids_a |= {d[0] for d in ra}
        ids_b |= {d[0] for d in rb}
        assert len(ra) == 1 and len(rb) == 1
    assert ids_a == {1} and ids_b == {1}  # stable, and numbered independently per camera
    # adding a camera does not reset anyone's numbering; a new object in camera a gets the next id in a's own space
    c = svc.client("c")
    assert c.track(frame(0)) == []
    assert [d[0] for d in a.track(frame(60))] == [1]
    a.track(frame(250))  # object jumps far away: new (unconfirmed) track
    assert [d[0] for d in a.track(frame(250))] == [2]
    assert [d[0] for d in b.track(frame(60))] == [1]
    svc.close()


def test_max_batch_splits_and_single_camera_does_not_wait():
    svc, m = service(max_batch=2, max_wait_ms=300, delay_s=0.05)
    clients = [svc.client(str(i)) for i in range(5)]
    run_parallel([lambda c=c, i=i: c.track(frame(10 + i)) for i, c in enumerate(clients)])
    assert all(len(b) <= 2 for b in m.batches)
    assert sum(len(b) for b in m.batches) == 5
    svc.close()

    svc, m = service(max_batch=8, max_wait_ms=2000)
    one = svc.client("solo")
    svc.client("never-submits")  # registered but idle cameras must not hold the batch open
    t0 = time.perf_counter()
    one.track(frame(30))
    assert time.perf_counter() - t0 < 1.0
    svc.close()


def test_wait_window_times_out_when_active_camera_is_late():
    svc, m = service(max_batch=8, max_wait_ms=200)
    a, b = svc.client("a"), svc.client("b")
    run_parallel([lambda: a.track(frame(10)), lambda: b.track(frame(20))])  # both active now
    m.batches.clear()
    t0 = time.perf_counter()
    a.track(frame(10))  # b stays silent: a is served after ~max_wait, alone
    dt = time.perf_counter() - t0
    assert 0.15 <= dt < 1.5, dt
    assert m.batches == [[10]]

    m.batches.clear()  # b arriving inside the window joins a's batch
    def late_b():
        time.sleep(0.05)
        return b.track(frame(20))
    run_parallel([lambda: a.track(frame(10)), late_b])
    assert sorted(m.batches[0]) == [10, 20]
    svc.close()


def test_errors_propagate_to_caller_and_service_survives():
    svc, _ = service(max_batch=4, max_wait_ms=5)
    a = svc.client("a")
    with pytest.raises(ValueError, match="boom"):
        a.track(frame(10, bad=True))
    assert a.track(frame(10))[0][2] == "person"
    assert svc.stats["errors"] == 1
    with pytest.raises(KeyError):
        svc.submit("unknown", frame(10))
    with pytest.raises(ValueError):
        svc.client("a")
    svc.close()


def test_shutdown_fails_pending_and_rejects_new_calls():
    svc, _ = service(max_batch=1, max_wait_ms=5, delay_s=0.3)
    a, b = svc.client("a"), svc.client("b")
    errs = []

    def call(c):
        try:
            c.track(frame(10))
        except RuntimeError as e:
            errs.append(e)

    ts = [threading.Thread(target=call, args=(c,)) for c in (a, b)]
    [t.start() for t in ts]
    time.sleep(0.1)  # one request is inside predict, the other is queued behind max_batch=1
    svc.close()
    [t.join(5) for t in ts]
    assert not svc._thread.is_alive()
    assert len(errs) == 1  # the in-flight batch completed; the queued one was failed, not left hanging
    with pytest.raises(RuntimeError):
        a.track(frame(10))
    svc.close()  # idempotent


def test_factory_selection_on_cpu(tmp_path):
    f = make_backend_factory({"weights": "data/models/yolo26n.pt", "device": "cpu"})
    assert f.close() is None  # per-camera path on CPU by default
    assert engine_batch(engine_name("yolo26s", 640, 16)) == 16
    # an engine that cannot be used here (no CUDA) falls back to the sibling .pt
    (tmp_path / "yolo26s.pt").write_bytes(b"")
    assert resolve_weights(str(tmp_path / "yolo26s-640-b16.engine"), "cpu") == str(tmp_path / "yolo26s.pt")
    assert Path(resolve_weights("auto", "cpu", models_dir=tmp_path)).name == "yolo26n.pt"


def test_batched_factory_hands_out_clients_of_one_service(monkeypatch):
    import argonix.inference as inf
    made = []

    class Svc(BatchedInference):
        def __init__(self, **kw):
            super().__init__(FakeModel(), device="cpu", warmup=False, max_batch=kw["max_batch"])
            made.append(self)

    monkeypatch.setattr(inf, "BatchedInference", Svc)
    f = make_backend_factory({"device": "cpu", "batched": True, "max_batch": 4, "stitch": {"enabled": False}})
    c1, c2 = f({"id": "a"}), f({"id": "b"})
    assert isinstance(c1, InferenceClient) and c1.service is c2.service and len(made) == 1
    assert c1.info["batched"] is True
    f.close()
    assert made[0].closed


def test_resolve_device_maps_gpu_index_to_torch_string():
    # "0" is valid for Ultralytics but not for torch (Re-ID/pose models call .to(device)); "cuda:0" works for both
    assert resolve_device("0") == "cuda:0"
    assert resolve_device(1) == "cuda:1"
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda:0")
