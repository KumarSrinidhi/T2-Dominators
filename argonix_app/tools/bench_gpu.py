"""GPU benchmark for Argus detection+tracking: per-camera vs batched (.pt fp16) vs batched TensorRT.

Streams are synthetic: the real clips in data/real are decoded once, resized like CameraWorker (<= 960 px wide)
and replayed as N independent streams (each from its own start offset). Every call goes through the production
code path (`UltralyticsTracker` or `BatchedInference` clients), tracking included.

    uv run python tools/bench_gpu.py --export                 # TensorRT fp16 engines (needs the tensorrt package)
    uv run python tools/bench_gpu.py                          # saturation grid -> markdown table
    uv run python tools/bench_gpu.py --paced --fps 10 15      # max cameras sustained at a fixed analytics rate
    uv run python tools/bench_gpu.py --decode                 # CPU decode cost per stream (OpenCV, no NVDEC)

Numbers are only valid for the machine they were measured on; the table header records the GPU, driver, CPU and
what else was using the GPU at the time.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.detect import UltralyticsTracker, find_engine, engine_name, tensorrt_available  # noqa: E402
from argonix.inference import BatchedInference  # noqa: E402

MODELS_DIR = ROOT / "data" / "models"
CLIPS = [ROOT / "data" / "real" / "vtest.avi", ROOT / "data" / "real" / "social-distance" / "pedestrian.mp4"]
MAX_W = 960  # CameraWorker's default analytics_width


# ------------------------------------------------------------------------------------------------ helpers
def load_clips(n_frames: int) -> list[list[np.ndarray]]:
    clips = []
    for p in CLIPS:
        if not p.exists():
            continue
        cap, frames = cv2.VideoCapture(str(p)), []
        while len(frames) < n_frames:
            ok, f = cap.read()
            if not ok:
                break
            s = max(1.0, f.shape[1] / MAX_W)
            frames.append(f if s == 1.0 else cv2.resize(f, (int(f.shape[1] / s), int(f.shape[0] / s))))
        cap.release()
        if frames:
            clips.append(frames)
    if not clips:
        raise SystemExit(f"no clips found in {ROOT / 'data' / 'real'}")
    return clips


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


class GpuSampler(threading.Thread):
    """Samples nvidia-smi: whole-GPU utilisation / memory, and this process's own VRAM."""

    def __init__(self, period_s: float = 0.25):
        super().__init__(daemon=True)
        self.period, self.stop_evt = period_s, threading.Event()
        self.util, self.mem_total, self.mem_self = [], [], []
        self.pid = os.getpid()

    @staticmethod
    def query() -> tuple[float, float]:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        u, m = (float(x) for x in out.split(","))
        return u, m

    def self_mem(self) -> float:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if pid.isdigit() and int(pid) == self.pid:
                return float(mem)
        return 0.0

    def run(self):
        while not self.stop_evt.is_set():
            try:
                u, m = self.query()
                self.util.append(u); self.mem_total.append(m); self.mem_self.append(self.self_mem())
            except Exception:
                pass
            self.stop_evt.wait(self.period)

    def summary(self) -> dict:
        return {"gpu_util": statistics.mean(self.util) if self.util else float("nan"),
                "vram_total_peak": max(self.mem_total, default=float("nan")),
                "vram_self_peak": max(self.mem_self, default=float("nan"))}


def gpu_baseline(seconds: float = 3.0) -> dict:
    s = GpuSampler(); s.start(); time.sleep(seconds); s.stop_evt.set(); s.join()
    return s.summary()


def machine_info() -> dict:
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "tensorrt": None}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    try:  # via nvidia-smi, so this (parent) process never creates a CUDA context of its own
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, mem, drv = (x.strip() for x in out.split(","))
        info.update(gpu=name, vram_gb=round(float(mem) / 1024, 2), driver=drv)
    except Exception:
        pass
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except Exception:
        pass
    return info


def other_gpu_processes() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
        return "; ".join(l.strip() for l in out.strip().splitlines() if l.strip() and not l.startswith(str(os.getpid())))
    except Exception:
        return "?"


# ------------------------------------------------------------------------------------------------ backends
def build(mode: str, model: str, imgsz: int, n: int, max_batch: int, max_wait_ms: float):
    """-> (list of per-camera backends, closer, weights actually used, service or None)"""
    pt = str(MODELS_DIR / f"{model}.pt") if (MODELS_DIR / f"{model}.pt").exists() else f"{model}.pt"
    if mode == "unbatched":
        bks = [UltralyticsTracker(pt, "bytetrack.yaml", 0.3, imgsz, (0,), "cuda:0") for _ in range(n)]
        return bks, (lambda: None), pt, None
    if mode == "batched":
        w = pt
    elif mode == "trt":
        eng = find_engine(MODELS_DIR, model, imgsz)
        if eng is None:
            return None, None, None, None
        w = str(eng)
    else:
        raise ValueError(mode)
    svc = BatchedInference(weights=w, device="cuda:0", imgsz=imgsz, conf=0.3, classes=(0,), models_dir=None,
                           max_batch=max_batch, max_wait_ms=max_wait_ms)
    if mode == "batched" and svc.info["runtime"] != "torch":
        raise RuntimeError("expected a .pt model for mode=batched")
    return [svc.client(f"cam{i}") for i in range(n)], svc.close, w, svc


# ------------------------------------------------------------------------------------------------ runs
def saturate(backends, clips, seconds: float) -> dict:
    """Every stream calls track() back-to-back: the achievable aggregate rate and its latency."""
    n = len(backends)
    log: list[list[tuple[float, float]]] = [[] for _ in range(n)]  # (t_start, t_end) per call
    stop = threading.Event()
    errs = []

    def cam(i):
        frames = clips[i % len(clips)]
        k = (i * 37) % len(frames)
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                backends[i].track(frames[k % len(frames)])
                log[i].append((t0, time.perf_counter()))
                k += 1
        except Exception as e:
            errs.append(repr(e)); stop.set()

    # the same threads warm up and then get measured: the first CUDA call on a new thread creates its cuDNN/cuBLAS
    # handles (a one-off stall of up to ~2 s that a long-lived CameraWorker thread pays once)
    ths = [threading.Thread(target=cam, args=(i,)) for i in range(n)]
    [t.start() for t in ths]
    t_end = time.perf_counter() + 90
    while time.perf_counter() < t_end and not errs and min(len(l) for l in log) < 3:  # every stream warmed up
        time.sleep(0.1)
    if min(len(l) for l in log) < 3:
        stop.set(); [t.join() for t in ths]
        raise RuntimeError(errs[0] if errs else "not every stream completed 3 calls within 90 s of warm-up")
    time.sleep(1.0)
    sampler = GpuSampler(); sampler.start()
    w0 = time.perf_counter(); time.sleep(seconds); w1 = time.perf_counter()
    stop.set(); [t.join() for t in ths]
    sampler.stop_evt.set(); sampler.join()
    if errs:
        raise RuntimeError(errs[0])
    wall = w1 - w0
    flat = [(b - a) * 1000 for l in log for a, b in l if w0 <= a and b <= w1]
    return {"frames": len(flat), "agg_fps": len(flat) / wall, "p50": pct(flat, 50), "p95": pct(flat, 95), "p99": pct(flat, 99),
            "mean": float(np.mean(flat)) if flat else float("nan"), "max": max(flat, default=float("nan")),
            **sampler.summary()}


def paced(backends, clips, fps: float, seconds: float, decode: bool, decode_paths=None) -> dict:
    """Each stream submits at `fps` (random phase), dropping a tick when it is more than one interval late, like
    CameraWorker. With `decode`, each stream really decodes its clip (all source frames, as CameraWorker does) and
    JPEG-encodes the result, so camera-side CPU work competes with inference."""
    n = len(backends)
    interval = 1.0 / fps
    lat: list[list[float]] = [[] for _ in range(n)]
    done, dropped = [0] * n, [0] * n
    stop = threading.Event()
    errs = []
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, interval, n)

    def cam(i):
        frames = clips[i % len(clips)]
        cap = None
        if decode:
            paths = decode_paths or CLIPS
            cap = cv2.VideoCapture(str(paths[i % len(paths)]))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            per_tick = src_fps / fps  # decode every source frame, like CameraWorker
        k, tick = (i * 37) % len(frames), 0
        try:
            for j in range(3):  # unpaced warm-up (first CUDA call on a new thread is slow), then a common start
                wf = frames[(k + j) % len(frames)]
                if cap is not None:  # warm with the real input shape: the per-camera path letterboxes with rect=True
                    ok, f = cap.read()
                    s = max(1.0, f.shape[1] / MAX_W)
                    wf = f if s == 1.0 else cv2.resize(f, (int(f.shape[1] / s), int(f.shape[0] / s)))
                backends[i].track(wf)
            with ready_lock:
                ready[0] += 1
            go.wait()
            start = go_at[0] + phase[i]
            while not stop.is_set():
                due = start + tick * interval
                now = time.perf_counter()
                if due > now:
                    time.sleep(due - now)
                if stop.is_set():
                    break
                tick += 1
                k += 1
                frame = frames[k % len(frames)]
                if cap is not None:  # CameraWorker decodes every source frame, even for ticks it then drops
                    for _ in range(max(1, int(tick * per_tick) - int((tick - 1) * per_tick))):
                        ok, f = cap.read()
                        if not ok:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
                    s = max(1.0, f.shape[1] / MAX_W)
                    frame = f if s == 1.0 else cv2.resize(f, (int(f.shape[1] / s), int(f.shape[0] / s)))
                if time.perf_counter() - due > interval:  # behind: drop this tick, stay live
                    dropped[i] += 1
                    continue
                t0 = time.perf_counter()
                backends[i].track(frame)
                lat[i].append((time.perf_counter() - t0) * 1000)
                if cap is not None:
                    cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                done[i] += 1
        except Exception as e:
            errs.append(repr(e)); stop.set()
        finally:
            if cap is not None:
                cap.release()

    ready, ready_lock, go, go_at = [0], threading.Lock(), threading.Event(), [0.0]
    ths = [threading.Thread(target=cam, args=(i,)) for i in range(n)]
    [t.start() for t in ths]
    t_end = time.perf_counter() + 90
    while ready[0] < n and not errs and time.perf_counter() < t_end:
        time.sleep(0.1)
    if ready[0] < n:
        stop.set(); go.set(); [t.join() for t in ths]
        raise RuntimeError(errs[0] if errs else "streams did not warm up within 90 s")
    go_at[0] = time.perf_counter() + 0.2
    go.set()
    sampler = GpuSampler(); sampler.start()
    time.sleep(2.0)  # settle into the paced schedule, then measure
    base_done, base_drop = list(done), list(dropped)
    for l in lat:
        l.clear()
    t0 = time.perf_counter()
    time.sleep(seconds)
    wall = time.perf_counter() - t0
    got = [d - b for d, b in zip(done, base_done)]
    drop = [d - b for d, b in zip(dropped, base_drop)]
    stop.set(); [t.join() for t in ths]
    sampler.stop_evt.set(); sampler.join()
    if errs:
        raise RuntimeError(errs[0])
    flat = [x for l in lat for x in l]
    per_cam = [g / wall for g in got]
    return {"per_cam_min_fps": min(per_cam), "agg_fps": sum(got) / wall, "drop_pct": 100 * sum(drop) / max(1, sum(got) + sum(drop)),
            "p50": pct(flat, 50), "p95": pct(flat, 95), "p99": pct(flat, 99),
            "mean": float(np.mean(flat)) if flat else float("nan"), "max": max(flat, default=float("nan")), **sampler.summary()}


def _make_1080p(tmp: Path, seconds: int = 12) -> Path | None:
    """Upscale the street clip to a 1920x1080 25 fps H.264 file (typical camera main stream) with bundled ffmpeg."""
    src = CLIPS[1]
    if not src.exists():
        return None
    try:
        import imageio_ffmpeg
        out = tmp / "street_1080p_h264.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-t", str(seconds), "-i", str(src),
                        "-vf", "scale=1920:1080", "-r", "25", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(out)], check=True, timeout=300)
        return out
    except Exception as e:
        print(f"1080p transcode skipped: {e!r}", file=sys.stderr)
        return None


def decode_cost(n_frames: int = 300) -> list[dict]:
    """Per-frame decode cost with OpenCV's FFmpeg backend (CPU; the pip wheels have no NVDEC). CPU ms counts
    process CPU time, which includes FFmpeg's decoder threads; wall ms is what one CameraWorker thread waits."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="argonix_bench_"))
    rows = []
    for p in [*CLIPS, _make_1080p(tmp)]:
        if p is None or not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = {}
        for how in ("read", "grab"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            w0, c0, k = time.perf_counter(), time.process_time(), 0
            while k < n_frames and (cap.read()[0] if how == "read" else cap.grab()):
                k += 1
            res[how] = ((time.perf_counter() - w0) * 1000 / max(1, k), (time.process_time() - c0) * 1000 / max(1, k))
        cap.release()
        rows.append({"clip": p.name, "res": f"{w}x{h}", "src_fps": round(fps, 1),
                     "read_wall_ms": res["read"][0], "read_cpu_ms": res["read"][1],
                     "grab_wall_ms": res["grab"][0], "grab_cpu_ms": res["grab"][1],
                     "core_pct_at_src_fps": res["read"][1] * fps / 10})
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return rows


def export_engines(models, imgszs, batch: int, workspace: float) -> None:
    if not tensorrt_available():
        raise SystemExit("tensorrt is not importable in this environment; see README (GPU section)")
    from ultralytics import YOLO
    for m in models:
        for sz in imgszs:
            dst = MODELS_DIR / engine_name(m, sz, batch)
            if dst.exists():
                print(f"exists: {dst}")
                continue
            pt = MODELS_DIR / f"{m}.pt"
            t0 = time.time()
            out = YOLO(str(pt) if pt.exists() else f"{m}.pt").export(
                format="engine", dynamic=True, batch=batch, imgsz=sz, quantize=16, workspace=workspace, device=0,
                verbose=False)
            Path(out).replace(dst)
            for tmp in (Path(out).with_suffix(".onnx"), Path(out).with_suffix(".fp16.onnx")):  # export by-products
                if tmp.exists():
                    tmp.unlink()
            print(f"exported {dst} in {time.time() - t0:.0f}s")



# ------------------------------------------------------------------------------------------------ isolation
def run_case(case: dict, timeout_s: float = 420.0) -> dict:
    """Run one configuration in a fresh interpreter: CUDA memory, cuDNN/cuBLAS per-thread handles and TensorRT
    contexts from earlier cases would otherwise accumulate in one process and distort later measurements."""
    bg = gpu_baseline(1.5)["gpu_util"]  # what the GPU's other tenants were doing just before this case
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", json.dumps(case)],
                           capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_s:.0f}s"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT "):
            return {**json.loads(line[7:]), "bg_util": bg}
    tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
    return {"error": f"exit {p.returncode}: {' / '.join(tail)}"}


def child(case: dict) -> None:
    try:
        clips = load_clips(case["frames"])
        bks, closer, w, svc = build(case["mode"], case["model"], case["imgsz"], case["n"], case["batch"],
                                    case["max_wait_ms"])
        if case["fps"]:
            r = paced(bks, clips, case["fps"], case["seconds"], case["with_decode"],
                      [Path(p) for p in case["decode_paths"]] or None)
        else:
            r = saturate(bks, clips, case["seconds"])
        r.update(weights=Path(w).name, mean_batch=svc.mean_batch if svc else 1.0,
                 batch_hist=svc.stats["batch_hist"] if svc else {}, infer_ms=svc.stats["infer_ms"] if svc else None)
        closer()
    except Exception as e:
        r = {"error": repr(e)}
    print("RESULT " + json.dumps(r, default=str), flush=True)
    os._exit(0)  # skip interpreter teardown (daemon threads, CUDA/TensorRT destructors)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="export TensorRT fp16 engines and exit")
    ap.add_argument("--paced", action="store_true", help="find the max cameras sustained at --fps")
    ap.add_argument("--decode", action="store_true", help="measure CPU decode cost per stream and exit")
    ap.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[480, 640])
    ap.add_argument("--modes", nargs="+", default=["unbatched", "batched", "trt"])
    ap.add_argument("--streams", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--fps", nargs="+", type=float, default=[10, 15])
    ap.add_argument("--with-decode", action="store_true", help="paced: decode + JPEG-encode on the camera threads")
    ap.add_argument("--decode-1080p", action="store_true",
                    help="paced --with-decode: every stream decodes a 1920x1080 25 fps H.264 clip (made with ffmpeg)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=16, help="max batch (service) / engine max batch (export)")
    ap.add_argument("--max-wait-ms", type=float, default=8.0)
    ap.add_argument("--workspace", type=float, default=1.0, help="TensorRT builder workspace, GiB")
    ap.add_argument("--frames", type=int, default=120, help="frames decoded per clip for synthetic streams")
    ap.add_argument("--out", type=Path, default=None, help="append markdown + JSON results here")
    ap.add_argument("--case", default=None, help=argparse.SUPPRESS)  # internal: one isolated measurement
    a = ap.parse_args(argv)

    if a.case:
        child(json.loads(a.case))
        return
    if a.export:
        export_engines(a.models, a.imgsz, a.batch, a.workspace)
        return
    lines = []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    info = machine_info()
    emit(f"### Argus GPU bench — {time.strftime('%Y-%m-%d %H:%M')}")
    emit(f"- GPU: {info.get('gpu')} ({info.get('vram_gb')} GiB), driver {info.get('driver')}; CPU: {info['cpu']} "
         f"({info['cpu_count']} threads); torch {info['torch']} (CUDA {info['cuda']}); ultralytics {info['ultralytics']}; "
         f"TensorRT {info['tensorrt'] or 'not installed'}")
    base = gpu_baseline()
    emit(f"- GPU shared with: {other_gpu_processes() or 'nothing'}; idle baseline util {base['gpu_util']:.0f}%, "
         f"VRAM used {base['vram_total_peak']:.0f} MiB before this bench")
    rows = []

    if a.decode:
        emit("\n| Clip | Source | src fps | read() wall / CPU ms per frame | grab() wall / CPU ms per frame | CPU core % per stream decoding every frame |")
        emit("|---|---|---|---|---|---|")
        for r in decode_cost():
            emit(f"| {r['clip']} | {r['res']} | {r['src_fps']} | {r['read_wall_ms']:.2f} / {r['read_cpu_ms']:.2f} | "
                 f"{r['grab_wall_ms']:.2f} / {r['grab_cpu_ms']:.2f} | {r['core_pct_at_src_fps']:.1f}% |")
            rows.append(r)
    else:
        clips = load_clips(a.frames)
        decode_paths = None
        if a.paced and a.with_decode and a.decode_1080p:
            import tempfile
            p1080 = _make_1080p(Path(tempfile.mkdtemp(prefix="argonix_bench_")))
            decode_paths = [p1080] if p1080 else None
        emit(f"- Streams: {len(clips)} real clips ({', '.join(f'{c[0].shape[1]}x{c[0].shape[0]}' for c in clips)}) "
             f"replayed as N synthetic streams; batched max_batch={a.batch}, max_wait={a.max_wait_ms} ms; conf 0.3, class person")
        if a.paced:
            emit("- Paced: each stream at a fixed rate; 'sustained' = every stream >= 95% of target and p95 latency <= "
                 "1 frame interval" + ("" if not a.with_decode else
                 f"; camera threads also decode {'a 1920x1080 25 fps H.264 clip' if decode_paths else 'their clip'} "
                 f"(every frame, like CameraWorker), resize to 960 px and JPEG-encode"))
            emit("\n| Mode | Model | imgsz | target fps | N cams | min per-cam fps | agg fps | dropped | p50 ms | p95 ms | GPU util (idle before) | VRAM self MiB | sustained |")
            emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        else:
            emit("\n| Mode | Model | imgsz | N streams | agg FPS | per-stream FPS | p50 ms | p95 ms | mean batch | GPU util (idle before) | VRAM self MiB | VRAM GPU total MiB |")
            emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for mode in a.modes:
            for model in a.models:
                for sz in a.imgsz:
                    if mode == "trt" and not tensorrt_available():
                        emit(f"| trt | {model} | {sz} | — | skipped: tensorrt not installed |")
                        continue
                    if mode == "trt" and find_engine(MODELS_DIR, model, sz) is None:
                        emit(f"| trt | {model} | {sz} | — | skipped: no engine for imgsz {sz} (run --export) |")
                        continue
                    for fps in (a.fps if a.paced else [None]):
                        for n in sorted(a.streams):
                            case = {"mode": mode, "model": model, "imgsz": sz, "n": n, "fps": fps, "seconds": a.seconds,
                                    "batch": a.batch, "max_wait_ms": a.max_wait_ms, "frames": a.frames,
                                    "with_decode": a.with_decode, "decode_paths": [str(p) for p in decode_paths or []]}
                            r = run_case(case)
                            if "error" in r:
                                emit(f"| {mode} | {model} | {sz} | {n} | failed: {r['error'][:300]} |")
                                ok = False
                            elif a.paced:
                                ok = r["per_cam_min_fps"] >= 0.95 * fps and r["p95"] <= 1000 / fps
                                emit(f"| {mode} | {r['weights']} | {sz} | {fps:g} | {n} | {r['per_cam_min_fps']:.1f} | "
                                     f"{r['agg_fps']:.0f} | {r['drop_pct']:.1f}% | {r['p50']:.1f} | {r['p95']:.1f} | "
                                     f"{r['gpu_util']:.0f}% ({r['bg_util']:.0f}%) | {r['vram_self_peak']:.0f} | {'yes' if ok else 'NO'} |")
                            else:
                                ok = None
                                emit(f"| {mode} | {r['weights']} | {sz} | {n} | {r['agg_fps']:.0f} | {r['agg_fps'] / n:.1f} | "
                                     f"{r['p50']:.1f} | {r['p95']:.1f} | {r['mean_batch']:.1f} | {r['gpu_util']:.0f}% ({r['bg_util']:.0f}%) | "
                                     f"{r['vram_self_peak']:.0f} | {r['vram_total_peak']:.0f} |")
                            rows.append({**case, "sustained": ok, **r})
                            if a.paced and ok is False:
                                break  # more cameras will not do better
    if a.out:
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n\n")
        with open(a.out.with_suffix(".json"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"info": info, "baseline": base, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
