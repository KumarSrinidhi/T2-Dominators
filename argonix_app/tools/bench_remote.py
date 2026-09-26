"""Measure remote inference (argonix/infer_server.py) end to end from THIS machine, and compare with local CPU.

    uv run python tools/bench_remote.py --url ws://<gpu-host>:8200 --token-file data/infer_token --out bench_remote.md

Every camera is one thread + one `RemoteTrackerBackend` (one WebSocket), exactly like a `CameraWorker`: it decodes
its clip locally (data/real/*, looped, resized to 960 px wide like `analytics_width`), calls `track()` and waits
for the result before the next frame. Phases (the first `--warmup` seconds of every run are not counted):
  link     WebSocket ping/pong round trips on an open camera connection (network + node event loop, no model)
  encoding 1 camera back-to-back for each JPEG quality x max side: RTT p50/p95, fps, KB/frame, Mbit/s at 10 fps
  scaling  N cameras back-to-back: per-camera fps and RTT, aggregate fps, uplink Mbit/s, node mean batch/GPU util
  paced    N cameras at --paced-fps: achieved fps per camera and the share of frames without a result
  local    N cameras through a per-camera UltralyticsTracker on this machine's CPU (the no-GPU alternative)
RTT = track() wall time for frames that got a result: downscale + JPEG encode + upload + node queue/decode/infer/
track + reply. Client CPU is this process's CPU time / wall time, in cores (psutil). Numbers are for this network
and these machines at the time of the run, not capacity claims.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.remote_inference import RemoteTrackerBackend, fetch_info, http_base, load_token  # noqa: E402

DEFAULT_CLIPS = ["data/real/vtest.avi", "data/real/social-distance/pedestrian.mp4"]


def pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


class Clip:
    """Looping decoder of one clip, resized to at most `width` px wide (what CameraWorker hands the detector)."""

    def __init__(self, path: str, offset: int, width: int = 960):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.width = width
        for _ in range(offset):
            self.cap.grab()

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        h, w = f.shape[:2]
        if w > self.width:
            f = cv2.resize(f, (self.width, round(h * self.width / w)))
        return f


def node_metrics(url: str, token: str):
    import httpx
    try:
        return httpx.get(http_base(url) + "/metrics", headers={"Authorization": f"Bearer {token}"}, timeout=10).json()
    except Exception:  # noqa: BLE001
        return None


def batch_delta(m0, m1):
    if not m0 or not m1:
        return None, None
    h0, h1 = m0.get("batch_hist", {}), m1.get("batch_hist", {})
    d = {int(k): v - h0.get(k, 0) for k, v in h1.items() if v - h0.get(k, 0) > 0}
    n = sum(d.values())
    return (round(sum(k * v for k, v in d.items()) / n, 2) if n else None), d


def run_cameras(n, make_backend, clips, seconds, warmup, pace_fps=0.0):
    """n camera threads; returns per-camera samples (latencies of frames with a result, frame counts, bytes)."""
    stop, measuring = threading.Event(), threading.Event()
    res = [None] * n
    backends = [make_backend(i) for i in range(n)]

    def cam(i):
        clip = Clip(clips[i % len(clips)], offset=37 * i)
        b = backends[i]
        lat, calls, empty_or_fail = [], 0, 0
        ok0 = bytes0 = 0
        started = False
        next_due = time.perf_counter()
        while not stop.is_set():
            f = clip.read()
            if measuring.is_set() and not started:
                started = True
                ok0 = getattr(b, "counters", {}).get("ok", 0)
                bytes0 = getattr(b, "counters", {}).get("bytes_sent", 0)
                t_start = time.perf_counter()
            okc = getattr(b, "counters", {}).get("ok", None)
            t0 = time.perf_counter()
            b.track(f)
            dt = (time.perf_counter() - t0) * 1000
            got = (getattr(b, "counters", {}).get("ok", 0) > okc) if okc is not None else True
            if started:
                calls += 1
                if got:
                    lat.append(dt)
                else:
                    empty_or_fail += 1
            if pace_fps:
                next_due += 1.0 / pace_fps
                wait = next_due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_due = time.perf_counter()  # behind: do not try to catch up (CameraWorker drops instead)
        dur = time.perf_counter() - t_start if started else 0.0
        c = getattr(b, "counters", {})
        res[i] = {"lat": lat, "calls": calls, "no_result": empty_or_fail, "dur": dur,
                  "ok": c.get("ok", 0) - ok0 if c else len(lat), "bytes": c.get("bytes_sent", 0) - bytes0 if c else 0}

    import psutil
    proc = psutil.Process()
    ts = [threading.Thread(target=cam, args=(i,), daemon=True) for i in range(n)]
    [t.start() for t in ts]
    time.sleep(warmup)
    cpu0, w0 = proc.cpu_times(), time.perf_counter()
    psutil.cpu_percent(None)
    measuring.set()
    time.sleep(seconds)
    cpu1, w1 = proc.cpu_times(), time.perf_counter()
    host_cpu = psutil.cpu_percent(None)  # whole machine, every process (other workloads show up here)
    stop.set()
    [t.join(30) for t in ts]
    for b in backends:
        getattr(b, "close", lambda: None)()
    cores = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / (w1 - w0)
    lat = [x for r in res for x in r["lat"]]
    per_fps = [r["ok"] / r["dur"] for r in res if r["dur"]]
    return {"n": n, "per_cam_fps": round(float(np.mean(per_fps)), 2) if per_fps else 0.0,
            "min_cam_fps": round(float(np.min(per_fps)), 2) if per_fps else 0.0,
            "agg_fps": round(sum(per_fps), 1), "p50": pct(lat, 50), "p95": pct(lat, 95),
            "no_result_pct": round(100 * sum(r["no_result"] for r in res) / max(1, sum(r["calls"] for r in res)), 1),
            "kb_per_frame": round(sum(r["bytes"] for r in res) / max(1, sum(r["ok"] for r in res)) / 1024, 1),
            "uplink_mbps": round(sum(r["bytes"] / r["dur"] for r in res if r["dur"]) * 8 / 1e6, 2),
            "client_cores": round(cores, 2), "host_cpu_pct": round(host_cpu)}


def link_rtt(url, token, n=30):
    b = RemoteTrackerBackend(url, "bench-ping", token=token, client_id=f"bench-{socket.gethostname()}")
    b.start()
    ws = b._ws
    if ws is None:
        raise SystemExit(f"cannot connect to {url}: {b.camera_stats()['remote_error']}")
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        if ws.ping().wait(10):
            out.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.05)
    b.close()
    return {"p50": pct(out, 50), "p95": pct(out, 95), "min": round(min(out), 1) if out else None, "n": len(out)}


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join("-" if v is None else str(v) for v in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", default=str(ROOT / "data" / "infer_token"))
    ap.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    ap.add_argument("--cams", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--qualities", nargs="+", type=int, default=[70, 85])
    ap.add_argument("--sizes", nargs="+", type=int, default=[480, 640])
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality for the scaling/paced phases")
    ap.add_argument("--size", type=int, default=480, help="max side for the scaling/paced phases")
    ap.add_argument("--paced-fps", type=float, default=5.0)
    ap.add_argument("--timeout-ms", type=float, default=2000)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--local-cams", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--local-weights", default=str(ROOT / "data" / "models" / "yolo26n.pt"))
    ap.add_argument("--local-imgsz", nargs="+", type=int, default=[480, 320])
    ap.add_argument("--phases", nargs="+", default=["link", "encoding", "scaling", "paced", "local"])
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    clips = [str(ROOT / c) if not Path(c).is_absolute() else c for c in a.clips]
    need_node = any(p != "local" for p in a.phases)
    token = load_token(None, a.token_file, (ROOT,)) if need_node else ""
    info = fetch_info(a.url, token, 10) if need_node else None
    if need_node and info is None:
        raise SystemExit(f"node at {a.url} not reachable or token rejected (GET /v1/info)")
    lines = [f"# Remote inference bench {a.label}".rstrip(), "",
             f"- Client: {socket.gethostname()} ({platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs), "
             f"{time.strftime('%Y-%m-%d %H:%M')}",
             f"- Node: {a.url} ({info['node'] if info else '?'}): {json.dumps(info['model']) if info else '?'}",
             f"- Clips (decoded here, 960 px max width): {', '.join(Path(c).name for c in clips)}",
             f"- {a.seconds:.0f} s per run after {a.warmup:.0f} s warm-up; client timeout {a.timeout_ms:.0f} ms", ""]
    print("\n".join(lines), flush=True)
    results = {"info": info, "args": vars(a)}

    def remote_backend(q, size):
        def mk(i):
            return RemoteTrackerBackend(a.url, f"bench{i}", token=token, client_id=f"bench-{socket.gethostname()}",
                                        jpeg_quality=q, max_side=size, timeout_s=a.timeout_ms / 1000,
                                        connect_timeout_s=10, max_timeouts=5)
        return mk

    def section(title, head, rows, note=""):
        s = f"## {title}\n\n{md_table(head, rows)}\n" + (f"\n{note}\n" if note else "")
        print(s, flush=True)
        lines.append(s)

    if "link" in a.phases:
        r = link_rtt(a.url, token)
        results["link"] = r
        section("Link (WebSocket ping, no inference)", ["p50 ms", "p95 ms", "min ms", "n"],
                [[r["p50"], r["p95"], r["min"], r["n"]]])

    if "encoding" in a.phases:
        rows = []
        for q in a.qualities:
            for size in a.sizes:
                m0 = node_metrics(a.url, token)
                r = run_cameras(1, remote_backend(q, size), clips, a.seconds, a.warmup)
                r["mean_batch"], _ = batch_delta(m0, node_metrics(a.url, token))
                results.setdefault("encoding", []).append({"q": q, "size": size, **r})
                rows.append([q, size, r["p50"], r["p95"], r["per_cam_fps"], r["kb_per_frame"],
                             round(r["kb_per_frame"] * 1024 * 8 * 10 / 1e6, 2), r["uplink_mbps"], r["client_cores"]])
                print(f"  encoding q={q} size={size}: {r}", flush=True)
        section("Encoding (1 camera, back-to-back)",
                ["JPEG q", "max side px", "RTT p50 ms", "RTT p95 ms", "fps", "KB/frame", "Mbit/s @10 fps", "Mbit/s measured",
                 "client cores"], rows)

    for phase, pace in (("scaling", 0.0), ("paced", a.paced_fps)):
        if phase not in a.phases:
            continue
        rows = []
        for n in a.cams:
            m0 = node_metrics(a.url, token)
            r = run_cameras(n, remote_backend(a.quality, a.size), clips, a.seconds, a.warmup, pace)
            m1 = node_metrics(a.url, token)
            r["mean_batch"], r["batch_hist"] = batch_delta(m0, m1)
            g = (m1 or {}).get("gpu") or {}
            r["gpu_util_pct"], r["node_p50"], r["node_p95"] = (g.get("util_pct"), (m1 or {}).get("latency_ms", {}).get("p50"),
                                                               (m1 or {}).get("latency_ms", {}).get("p95"))
            results.setdefault(phase, []).append(r)
            print(f"  {phase} n={n}: {r}", flush=True)
            rows.append([n, r["per_cam_fps"], r["min_cam_fps"], r["agg_fps"], r["p50"], r["p95"], r["no_result_pct"],
                         r["uplink_mbps"], r["mean_batch"], r["node_p50"], r["node_p95"], r["gpu_util_pct"], r["client_cores"],
                         r["host_cpu_pct"]])
        head = ["Cameras", "fps/cam (mean)", "fps/cam (min)", "total fps", "RTT p50", "RTT p95", "% no result",
                "uplink Mbit/s", "node mean batch", "node p50 ms", "node p95 ms", "GPU util %", "client cores",
                "client host CPU %"]
        title = (f"Scaling (back-to-back, JPEG q{a.quality} @{a.size} px)" if phase == "scaling"
                 else f"Paced at {a.paced_fps:g} fps per camera (JPEG q{a.quality} @{a.size} px)")
        section(title, head, rows, "node p50/p95 and GPU util come from the node's /metrics at the end of the run "
                                   "(latency over its last 60 s; GPU util = mean of the last 10 one-second samples, "
                                   "all GPU tenants).")

    if "local" in a.phases and a.local_cams:
        from argonix.detect import UltralyticsTracker
        import torch
        rows = []
        for imgsz in a.local_imgsz:
            for n in a.local_cams:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) // n))  # what Site does
                r = run_cameras(n, lambda i: UltralyticsTracker(a.local_weights, imgsz=imgsz, device="cpu"),
                                clips, a.seconds, a.warmup)
                results.setdefault("local", []).append({"imgsz": imgsz, **r})
                print(f"  local imgsz={imgsz} n={n}: {r}", flush=True)
                rows.append([f"{Path(a.local_weights).name} @{imgsz}", n, r["per_cam_fps"], r["agg_fps"], r["p50"], r["p95"],
                             r["client_cores"], r["host_cpu_pct"]])
        section("Local CPU on this machine (per-camera UltralyticsTracker, back-to-back)",
                ["Model", "Cameras", "fps/cam", "total fps", "call p50 ms", "call p95 ms", "cores used", "host CPU %"], rows)

    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
