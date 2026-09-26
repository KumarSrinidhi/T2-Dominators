"""Remote GPU inference node: ONE shared `BatchedInference` (argonix/inference.py) served over WebSockets, so camera
workers (ingest, rules, UI) can run on one machine and detection + tracking on another.

    python -m argonix.infer_server --host <tailscale-ip> --port 8200 --weights data/models/yolo26s.pt --imgsz 480 --max-batch 16

Protocol v1 (client: argonix/remote_inference.py)
- `GET /health`: unauthenticated liveness only ({"ok", "service", "protocol", "uptime_s"}).
- `GET /v1/info`, `GET /metrics`: need `Authorization: Bearer <token>`.
- `WS /v1/stream?client=<client_id>&camera=<camera_id>`: one persistent connection per camera, same bearer header.
  Connections without a valid token are refused before the handshake completes (HTTP 403).
  1. server -> client (text): {"type": "hello", "node", "model", "session", "resumed", "max_inflight", "max_age_ms"}
  2. client -> server (binary), one message per frame: 4-byte big-endian header length, a JSON header
     {"seq": int, "ts": float (echoed back), "fmt": "jpeg" | "raw", "w": int, "h": int}, then the payload
     (JPEG bytes, or raw BGR uint8 h*w*3).
  3. server -> client (text), one per frame: {"type": "result", "seq", "ts", "status": "ok" | "dropped" | "error",
     "dets": [[track_id, x1, y1, x2, y2, cls, conf], ...] in the SENT frame's pixels, "server": {queue_ms, decode_ms,
     infer_ms, track_ms, total_ms, batch}, "reason"}.

Tracking: one ByteTrack per (client_id, camera_id), numbered independently. A reconnect within `tracker_ttl_s`
resumes the same tracker (same `session` in the hello), so track IDs survive a network blip. A second connection
for the same key replaces the first (the old one is closed with code 4409).

Backpressure: at most `max_inflight` frames per camera are decoding or queued for the GPU. A newer frame arriving
while the camera is full replaces the one waiting behind them (latest wins; the older one is answered "dropped",
reason "superseded"), and a frame that waited longer than `max_age_ms` before it could be submitted is answered
"dropped", reason "stale". Replies always carry the frame's `seq`, so a client never waits for a frame the node
discarded.

Security: a shared bearer token, generated once into a git-ignored file (default data/infer_token). Frames are
sent unencrypted over the socket: use Tailscale (WireGuard) or an SSH tunnel, never an internet-exposed port.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import re
import secrets
import socket
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, WebSocket

from .inference import BatchedInference

PROTOCOL = 1
ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_SIDE = 4096
CLOSE_UNAUTHORIZED, CLOSE_BAD_REQUEST, CLOSE_REPLACED, CLOSE_SHUTDOWN = 4401, 4400, 4409, 4503


# ---------------------------------------------------------------------------------------------- token
def load_or_create_token(path: Path) -> str:
    """Read the shared token, creating a random one (0600 where the OS supports it) on first use."""
    path = Path(path)
    if path.exists():
        tok = path.read_text(encoding="utf-8").strip()
        if len(tok) < 16:
            raise SystemExit(f"token in {path} is too short (need >= 16 characters)")
        return tok
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tok + "\n")
    return tok


def bearer_ok(header: Optional[str], token: str) -> bool:
    if not header or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode(), token.encode())


# ---------------------------------------------------------------------------------------------- GPU telemetry
class GpuMonitor:
    """GPU util/memory via NVML (pynvml) sampled once a second in a daemon thread, else `nvidia-smi` on demand.

    Utilisation is averaged over the last 10 samples. Under Windows WDDM, NVML cannot attribute memory to a
    process, so `mem_used_mb` is the whole GPU (every tenant); `torch_reserved_mb` is this process's allocator.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.index = int(device.split(":")[1]) if device.startswith("cuda:") else 0
        self.samples: deque = deque(maxlen=10)
        self.static: dict = {}
        self._nvml = None
        self._smi_cache: tuple[float, dict] = (0.0, {})
        if not device.startswith("cuda"):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            name = pynvml.nvmlDeviceGetName(h)
            self.static = {"name": name.decode() if isinstance(name, bytes) else name,
                           "driver": _s(pynvml.nvmlSystemGetDriverVersion())}
            self._nvml = (pynvml, h)
            threading.Thread(target=self._sample_loop, daemon=True, name="gpu-monitor").start()
        except Exception:  # noqa: BLE001 - NVML missing/broken: nvidia-smi fallback
            self._nvml = None

    def _sample_loop(self) -> None:
        pynvml, h = self._nvml
        while True:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.samples.append((time.time(), u.gpu, m.used / 2**20, m.total / 2**20))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    def _smi(self) -> dict:
        t, d = self._smi_cache
        if time.time() - t < 2.0:
            return d
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", f"--id={self.index}", "--query-gpu=name,utilization.gpu,memory.used,"
                                  "memory.total,driver_version", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            name, util, used, total, drv = [x.strip() for x in out.split(",")]
            d = {"name": name, "driver": drv, "util_pct": float(util), "mem_used_mb": round(float(used)),
                 "mem_total_mb": round(float(total)), "source": "nvidia-smi"}
        except Exception as e:  # noqa: BLE001
            d = {"available": False, "error": repr(e)[:120]}
        self._smi_cache = (time.time(), d)
        return d

    def snapshot(self) -> dict:
        if not self.device.startswith("cuda"):
            return {"available": False, "device": self.device}
        if self._nvml and self.samples:
            s = list(self.samples)
            d = {**self.static, "util_pct": round(sum(x[1] for x in s) / len(s), 1), "util_window_s": len(s),
                 "mem_used_mb": round(s[-1][2]), "mem_total_mb": round(s[-1][3]), "source": "nvml"}
        else:
            d = self._smi()
        try:
            import torch
            d["torch_reserved_mb"] = round(torch.cuda.memory_reserved(self.device) / 2**20)
        except Exception:  # noqa: BLE001
            pass
        return {"available": "error" not in d, "device": self.device, **d}


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


# ---------------------------------------------------------------------------------------------- stats helpers
def percentiles(values, qs=(50, 95)) -> dict:
    if not values:
        return {f"p{q}": None for q in qs} | {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(a, q)), 1) for q in qs} | {"n": int(a.size)}


class Rate:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float = 5.0):
        self.window, self.t = window_s, deque()

    def add(self, now: float) -> None:
        self.t.append(now)
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()

    def fps(self, now: float) -> float:
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()
        if len(self.t) < 2:
            return 0.0
        return round((len(self.t) - 1) / max(1e-3, now - self.t[0]), 2)


class _Frame:
    __slots__ = ("seq", "ts", "fmt", "w", "h", "payload", "t_recv", "decode_ms")

    def __init__(self, seq, ts, fmt, w, h, payload, t_recv):
        self.seq, self.ts, self.fmt, self.w, self.h = seq, ts, fmt, w, h
        self.payload, self.t_recv, self.decode_ms = payload, t_recv, 0.0


class _Conn:
    def __init__(self, key: str, client: str, camera: str, ws: WebSocket, session: str):
        self.key, self.client, self.camera, self.ws, self.session = key, client, camera, ws, session
        self.inflight = 0            # frames decoding or submitted to the batch engine
        self.waiting: Optional[_Frame] = None  # newest frame not yet submitted (latest wins)
        self.decoding = False        # one decode at a time keeps submission (= tracking) order == arrival order
        self.closed = False
        self.out: asyncio.Queue = asyncio.Queue()
        self.t_open = time.time()
        self.frames = self.ok = self.dropped = self.errors = self.bytes_in = 0
        self.rate_in, self.rate_out = Rate(), Rate()
        self.lat: deque = deque(maxlen=300)
        self.peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"


def parse_frame(data: bytes, t_recv: float) -> _Frame:
    if len(data) < 4:
        raise ValueError("short message")
    (n,) = struct.unpack("!I", data[:4])
    if n > 4096 or 4 + n > len(data):
        raise ValueError("bad header length")
    hdr = json.loads(data[4:4 + n])
    fmt = hdr.get("fmt", "jpeg")
    if fmt not in ("jpeg", "raw"):
        raise ValueError(f"unsupported fmt {fmt!r}")
    w, h = int(hdr.get("w", 0)), int(hdr.get("h", 0))
    if fmt == "raw" and not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
        raise ValueError("raw frames need 0 < w, h <= 4096")
    return _Frame(int(hdr["seq"]), hdr.get("ts"), fmt, w, h, data[4 + n:], t_recv)


def decode_frame(fr: _Frame) -> np.ndarray:
    if fr.fmt == "raw":
        a = np.frombuffer(fr.payload, dtype=np.uint8)
        if a.size != fr.w * fr.h * 3:
            raise ValueError(f"raw payload has {a.size} bytes, expected {fr.w * fr.h * 3}")
        return a.reshape(fr.h, fr.w, 3)
    import cv2
    img = cv2.imdecode(np.frombuffer(fr.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG decode failed")
    if max(img.shape[:2]) > MAX_SIDE:
        raise ValueError("frame larger than 4096 px")
    return img


# ---------------------------------------------------------------------------------------------- the node
class InferenceNode:
    def __init__(self, svc: BatchedInference, token: str, *, max_inflight: int = 2, max_age_ms: float = 1000.0,
                 tracker_ttl_s: float = 30.0, node_name: Optional[str] = None, decode_workers: int = 4,
                 gpu: Optional[GpuMonitor] = None):
        if not token or len(token) < 16:
            raise ValueError("a bearer token of at least 16 characters is required")
        self.svc, self.token = svc, token
        self.max_inflight = max(1, int(max_inflight))
        self.max_age_ms = float(max_age_ms)
        self.tracker_ttl_s = float(tracker_ttl_s)
        self.name = node_name or socket.gethostname()
        self.decode_pool = ThreadPoolExecutor(max(1, decode_workers), thread_name_prefix="infer-decode")
        self.gpu = gpu or GpuMonitor(svc.device)
        self.conns: dict[str, _Conn] = {}
        self.detached: dict[str, float] = {}   # key -> when its last connection closed (tracker kept for the TTL)
        self.sessions: dict[str, str] = {}     # key -> session id of its tracker
        self.lat: deque = deque(maxlen=5000)   # (t, total_ms) for the node-wide p50/p95
        self.started = time.time()
        self.counters = {"frames": 0, "ok": 0, "dropped_superseded": 0, "dropped_stale": 0, "errors": 0,
                         "bytes_in": 0, "connections": 0, "replaced": 0, "auth_rejected": 0}

    # ---- helpers -------------------------------------------------------------------------------------
    def authorized(self, header: Optional[str]) -> bool:
        return bearer_ok(header, self.token)

    def info(self) -> dict:
        return {"node": self.name, "protocol": PROTOCOL, "model": dict(self.svc.info),
                "max_inflight": self.max_inflight, "max_age_ms": self.max_age_ms, "tracker_ttl_s": self.tracker_ttl_s}

    def _sweep(self) -> None:
        now = time.time()
        for key, t in list(self.detached.items()):
            if key not in self.conns and now - t > self.tracker_ttl_s:
                self.svc.release(key)
                self.detached.pop(key, None)
                self.sessions.pop(key, None)

    def _reply(self, conn: _Conn, msg: dict) -> None:
        if not conn.closed:
            conn.out.put_nowait(msg)

    def _drop(self, conn: _Conn, fr: _Frame, reason: str) -> None:
        conn.dropped += 1
        self.counters["dropped_" + reason] += 1
        self._reply(conn, {"type": "result", "seq": fr.seq, "ts": fr.ts, "status": "dropped", "reason": reason,
                           "dets": []})

    def _error(self, conn: _Conn, fr: _Frame, e: BaseException) -> None:
        conn.errors += 1
        self.counters["errors"] += 1
        self._reply(conn, {"type": "result", "seq": fr.seq, "ts": fr.ts, "status": "error",
                           "reason": repr(e)[:200], "dets": []})

    # ---- per-frame state machine (runs on the event loop) -------------------------------------------
    def on_frame(self, conn: _Conn, fr: _Frame) -> None:
        conn.frames += 1
        conn.bytes_in += len(fr.payload)
        conn.rate_in.add(time.perf_counter())
        self.counters["frames"] += 1
        self.counters["bytes_in"] += len(fr.payload)
        if conn.waiting is not None:
            self._drop(conn, conn.waiting, "superseded")
        conn.waiting = fr
        self._pump(conn)

    def _pump(self, conn: _Conn) -> None:
        loop = asyncio.get_running_loop()
        while (not conn.closed and not conn.decoding and conn.waiting is not None
               and conn.inflight < self.max_inflight):
            fr, conn.waiting = conn.waiting, None
            if (time.perf_counter() - fr.t_recv) * 1000 > self.max_age_ms:
                self._drop(conn, fr, "stale")
                continue
            conn.inflight += 1
            conn.decoding = True
            loop.create_task(self._decode_submit(conn, fr))

    async def _decode_submit(self, conn: _Conn, fr: _Frame) -> None:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            img = await loop.run_in_executor(self.decode_pool, decode_frame, fr)
            fr.decode_ms = (time.perf_counter() - t0) * 1000
            fr.payload = b""
            if conn.closed:
                raise ConnectionError("client gone")
            self.svc.submit_nowait(conn.key, img,
                                   lambda req: loop.call_soon_threadsafe(self._done, conn, fr, req))
        except Exception as e:  # noqa: BLE001 - bad frame, closed service, released tracker
            conn.inflight -= 1
            self._error(conn, fr, e)
        finally:
            conn.decoding = False
            self._pump(conn)

    def _done(self, conn: _Conn, fr: _Frame, req) -> None:
        conn.inflight -= 1
        now = time.perf_counter()
        if req.error is not None:
            self._error(conn, fr, req.error)
        else:
            total = (now - fr.t_recv) * 1000
            conn.ok += 1
            conn.rate_out.add(now)
            conn.lat.append(total)
            self.lat.append((time.time(), total))
            self.counters["ok"] += 1
            dets = [[tid, round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1), cls, round(conf, 3)]
                    for tid, b, cls, conf in req.result]
            self._reply(conn, {"type": "result", "seq": fr.seq, "ts": fr.ts, "status": "ok", "dets": dets,
                               "server": {"queue_ms": round((req.t_start - req.t_submit) * 1000, 2),
                                          "decode_ms": round(fr.decode_ms, 2),
                                          "infer_ms": round((req.t_infer - req.t_start) * 1000, 2),
                                          "track_ms": round((req.t_done - req.t_track) * 1000, 2),
                                          "total_ms": round(total, 2), "batch": req.batch_n}})
        self._pump(conn)

    # ---- connection lifecycle ------------------------------------------------------------------------
    async def serve(self, ws: WebSocket, client: str, camera: str) -> None:
        if not self.authorized(ws.headers.get("authorization")):
            self.counters["auth_rejected"] += 1
            await ws.close(code=CLOSE_UNAUTHORIZED)  # before accept(): the handshake is refused with HTTP 403
            return
        if not (ID_RE.match(client or "") and ID_RE.match(camera or "")):
            await ws.close(code=CLOSE_BAD_REQUEST)
            return
        await ws.accept()
        self._sweep()
        key = f"{client}/{camera}"
        old = self.conns.get(key)
        if old is not None:  # latest connection wins; the old one must not keep feeding the same tracker
            old.closed = True
            self.counters["replaced"] += 1
            try:
                await old.ws.close(code=CLOSE_REPLACED)
            except Exception:  # noqa: BLE001
                pass
        resumed = self.svc.has_client(key)
        if not resumed:
            try:
                self.svc.client(key)
            except Exception as e:  # noqa: BLE001 - e.g. the service is closed
                await ws.close(code=CLOSE_SHUTDOWN, reason=repr(e)[:100])
                return
            self.sessions[key] = secrets.token_hex(4)
        self.detached.pop(key, None)
        conn = _Conn(key, client, camera, ws, self.sessions.setdefault(key, secrets.token_hex(4)))
        self.conns[key] = conn
        self.counters["connections"] += 1
        await ws.send_text(json.dumps({"type": "hello", **self.info(), "session": conn.session, "resumed": resumed,
                                       "client": client, "camera": camera}))
        sender = asyncio.get_running_loop().create_task(self._sender(conn))
        try:
            while not conn.closed:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:  # text: only a JSON ping is understood
                    try:
                        if json.loads(msg.get("text") or "{}").get("type") == "ping":
                            self._reply(conn, {"type": "pong", "t": time.time()})
                    except ValueError:
                        pass
                    continue
                try:
                    fr = parse_frame(data, time.perf_counter())
                except Exception as e:  # noqa: BLE001
                    conn.errors += 1
                    self.counters["errors"] += 1
                    self._reply(conn, {"type": "result", "seq": -1, "status": "error", "reason": repr(e)[:200],
                                       "dets": []})
                    continue
                self.on_frame(conn, fr)
        except Exception:  # noqa: BLE001 - socket errors end this connection only
            pass
        finally:
            conn.closed = True
            conn.waiting = None
            sender.cancel()
            if self.conns.get(key) is conn:
                del self.conns[key]
                self.detached[key] = time.time()

    async def _sender(self, conn: _Conn) -> None:
        try:
            while True:
                msg = await conn.out.get()
                await conn.ws.send_text(json.dumps(msg))
        except Exception:  # noqa: BLE001 - cancelled or the socket closed
            pass

    # ---- metrics -------------------------------------------------------------------------------------
    def metrics(self) -> dict:
        self._sweep()
        now_p, now = time.perf_counter(), time.time()
        recent = [ms for t, ms in self.lat if now - t <= 60]
        s = self.svc.stats
        return {
            "node": self.name, "protocol": PROTOCOL, "uptime_s": round(now - self.started),
            "model": dict(self.svc.info), "warmup_s": s.get("warmup_s"),
            "queue_depth": self.svc.queue_depth,
            "waiting": sum(1 for c in self.conns.values() if c.waiting is not None),
            "inflight": sum(c.inflight for c in self.conns.values()),
            "batches": s["batches"], "frames_inferred": s["frames"], "mean_batch": round(self.svc.mean_batch, 2),
            "batch_hist": {str(k): v for k, v in sorted(s["batch_hist"].items())},
            "infer_ms_ema": s["infer_ms"], "track_ms_ema": s["track_ms"], "engine_errors": s["errors"],
            "last_error": s["last_error"],
            "latency_ms": {**percentiles(recent), "window_s": 60, "what": "frame received -> result queued"},
            "counters": dict(self.counters),
            "fps_in": round(sum(c.rate_in.fps(now_p) for c in self.conns.values()), 2),
            "clients": [{"client": c.client, "camera": c.camera, "peer": c.peer, "session": c.session,
                         "connected_s": round(now - c.t_open), "fps_in": c.rate_in.fps(now_p),
                         "fps_out": c.rate_out.fps(now_p), "frames": c.frames, "ok": c.ok, "dropped": c.dropped,
                         "errors": c.errors, "inflight": c.inflight, "kb_per_frame": round(c.bytes_in / c.frames / 1024, 1)
                         if c.frames else None, "latency_ms": percentiles(list(c.lat))}
                        for c in self.conns.values()],
            "detached_trackers": {k: round(now - t) for k, t in self.detached.items()},
            "gpu": self.gpu.snapshot(),
            "limits": {"max_inflight": self.max_inflight, "max_age_ms": self.max_age_ms,
                       "tracker_ttl_s": self.tracker_ttl_s, "max_batch": self.svc.max_batch,
                       "max_wait_ms": round(self.svc.max_wait * 1000, 1)},
        }


def create_infer_app(node: InferenceNode) -> FastAPI:
    app = FastAPI(title="Argus inference node", docs_url=None, redoc_url=None, openapi_url=None)

    def auth(authorization: Optional[str]) -> None:
        if not node.authorized(authorization):
            node.counters["auth_rejected"] += 1
            raise HTTPException(401, "bearer token required", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health():
        return {"ok": not node.svc.closed, "service": "argonix-infer", "protocol": PROTOCOL,
                "uptime_s": round(time.time() - node.started)}

    @app.get("/v1/info")
    def info(authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        return node.info()

    @app.get("/metrics")
    def metrics(authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        return node.metrics()

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket, client: str = "", camera: str = ""):
        await node.serve(ws, client, camera)

    app.state.node = node
    return app


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Argus remote GPU inference node (WebSocket, bearer token)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; prefer the Tailscale/LAN IP over 0.0.0.0 and never expose it to the internet")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--weights", default="auto")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="person confidence cut before ByteTrack (0.1: its low-score second stage keeps small and "
                         "occluded people tracked; new tracks still need 0.25)")
    ap.add_argument("--classes", default="0", help="comma-separated COCO class ids ('' = all)")
    ap.add_argument("--extra-classes", default="",
                    help="comma-separated object classes tracked separately, e.g. 24,26,28 (backpack, handbag, suitcase)"
                         " for sites with unattended-object analysis (objects.enabled), and/or 2,3,5,7 (car, motorcycle,"
                         " bus, truck) for sites with vehicle rules (vehicles.enabled)")
    ap.add_argument("--extra-conf", type=float, default=None, help="confidence for --extra-classes (default: --conf)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-wait-ms", type=float, default=8.0)
    ap.add_argument("--max-inflight", type=int, default=2, help="frames per camera decoding or queued for the GPU")
    ap.add_argument("--max-age-ms", type=float, default=1000.0, help="drop a frame that waited longer than this")
    ap.add_argument("--tracker-ttl-s", type=float, default=30.0, help="keep a disconnected camera's tracker this long")
    ap.add_argument("--token-file", default=str(ROOT / "data" / "infer_token"))
    ap.add_argument("--node-name", default=None)
    ap.add_argument("--decode-workers", type=int, default=4)
    a = ap.parse_args(argv)
    from .hosttune import fine_timers, prefer_performance_cores  # Windows: 1 ms waits, performance cores only
    fine_timers()
    pcores = prefer_performance_cores()
    if pcores:
        print(f"host: pinned to performance cores {pcores[0]}-{pcores[-1]} (ARGONIX_ANY_CORES=1 to disable)", flush=True)

    token_path = Path(a.token_file)
    created = not token_path.exists()
    token = load_or_create_token(token_path)
    classes = tuple(int(c) for c in a.classes.split(",") if c.strip()) or None
    t0 = time.time()
    svc = BatchedInference(weights=a.weights, device=a.device, imgsz=a.imgsz, conf=a.conf, classes=classes,
                           tracker=a.tracker, models_dir=ROOT / "data" / "models", max_batch=a.max_batch,
                           max_wait_ms=a.max_wait_ms,
                           extra_classes=tuple(int(c) for c in a.extra_classes.split(",") if c.strip()),
                           extra_conf=a.extra_conf)
    node = InferenceNode(svc, token, max_inflight=a.max_inflight, max_age_ms=a.max_age_ms,
                         tracker_ttl_s=a.tracker_ttl_s, node_name=a.node_name, decode_workers=a.decode_workers)
    print(f"Argus inference node: {svc.info['detector']} @{a.imgsz} on {svc.device} ({svc.info['runtime']}), "
          f"max_batch {svc.max_batch}, ready in {time.time() - t0:.1f} s", flush=True)
    print(f"  ws://{a.host}:{a.port}/v1/stream  (token {'created in' if created else 'read from'} {token_path}; "
          f"copy that file to each client)", flush=True)
    if a.host in ("0.0.0.0", "::"):
        print("  WARNING: listening on every interface. Keep this port off the internet (Tailscale/LAN only).",
              flush=True)
    import uvicorn
    try:
        uvicorn.run(create_infer_app(node), host=a.host, port=a.port, log_level="warning", ws_max_size=32 * 2**20,
                    ws_per_message_deflate=False)
    finally:
        svc.close()


if __name__ == "__main__":
    main()
