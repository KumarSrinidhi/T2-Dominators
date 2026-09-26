"""Client for a remote inference node (argonix/infer_server.py): a `TrackerBackend` whose detector and tracker run
on another machine, with reconnect/backoff and an optional local CPU fallback.

Config (site JSON):
    "detector": {"remote": {"url": "ws://100.x.y.z:8200", "token_file": "data/infer_token", "fallback": "local"}}

Optional `remote` keys: `client_id` (default: this host's name), `jpeg_quality` (80), `max_side` (0 = the node's
imgsz), `fmt` ("jpeg" | "raw"), `timeout_ms` (1000), `connect_timeout_s` (10), `backoff_min_s` (0.5),
`backoff_max_s` (10), `max_timeouts` (3 consecutive timeouts = node treated as down), `fallback_after_s` (10),
`fallback_weights` ("yolo26n.pt"), `fallback_imgsz` (320), `adaptive` (true), `target_rtt_ms` (250),
`min_side` (256), `max_timeout_ms` (5000), `uplink_kbps` (0 = no budget). The token may also come from the
environment variable ARGONIX_INFER_TOKEN.

Behaviour per camera (one WebSocket each; `track()` is called by that camera's worker thread):
- Each frame is downscaled so its longest side is `max_side`, JPEG-encoded, sent, and the call blocks for the
  reply. Boxes come back in the sent frame's pixels and are scaled to the caller's frame, so detections always
  belong to the frame the worker is processing (no pipelining, no reused boxes).
- Adaptive (default on), for slow links: every 12 results (or 10 s with at least 3) the median RTT is compared
  with `target_rtt_ms`. Above
  it, the frame gets smaller (resolution + JPEG quality ladder, down to `min_side`); well below it, larger again. A
  step down that does not cut the RTT by 5% (the link is latency-bound, not bandwidth-bound) is undone and held
  for a minute, so detection quality is not traded away for nothing. The reply timeout follows the measured RTT
  (3 x p95, between `timeout_ms` and `max_timeout_ms`), so a slow but working link is not declared dead.
- Frame rate is self-clocked: one frame per camera is in flight, and the worker always takes its freshest frame,
  so under a high RTT the camera's analytics rate falls to about 1 / RTT instead of queueing stale frames.
  `uplink_kbps` adds a budget shared by all cameras of the site: a send waits for budget (lowering that camera's
  fps), and a frame that would wait more than 2 s is skipped (returns [], counted in `skipped_budget`).
- A timeout or a frame the node dropped returns [] for that frame and counts in `dropped_remote`.
  `max_timeouts` consecutive timeouts close the connection (a hung node is treated like a dead one).
- The connection is (re)opened by a background thread with exponential backoff, so `track()` never blocks on a
  dead node. While it is down, frames return [] until `fallback_after_s` have passed; then, with
  `fallback: "local"`, a CPU model on this machine takes over until the node is back.
- Track IDs are remapped to one per-camera sequence. A switch to another tracker (the fallback, or a node that lost
  this camera's tracker) starts new IDs, so the rule engine never sees one ID reused for two different people.
  A reconnect that resumed the node-side tracker keeps its IDs.
- `info` always describes the model that produced the current detections (remote node or local fallback), so the
  evidence attached to each alert names the right model.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from .detect import Det, TrackerBackend

CLOSE_UNAUTHORIZED, CLOSE_REPLACED = 4401, 4409
log = logging.getLogger(__name__)


def http_base(url: str) -> str:
    """ws(s)://host:port[/...] -> http(s)://host:port (for /health, /metrics, /v1/info)."""
    u = urlsplit(url if "://" in url else "ws://" + url)
    scheme = {"ws": "http", "wss": "https"}.get(u.scheme, u.scheme)
    return urlunsplit((scheme, u.netloc, "", "", ""))


def ws_base(url: str) -> str:
    u = urlsplit(url if "://" in url else "ws://" + url)
    scheme = {"http": "ws", "https": "wss"}.get(u.scheme, u.scheme)
    return urlunsplit((scheme, u.netloc, "", "", ""))


def load_token(token: Optional[str] = None, token_file: Optional[str] = None,
               base_dirs: tuple[Path, ...] = ()) -> str:
    """Token from the explicit value, else ARGONIX_INFER_TOKEN, else `token_file` (relative paths are tried against
    the working directory, then each of `base_dirs`)."""
    if token:
        return token.strip()
    env = os.environ.get("ARGONIX_INFER_TOKEN", "").strip()
    if env:
        return env
    if token_file:
        p = Path(token_file)
        for cand in [p] + ([] if p.is_absolute() else [b / p for b in base_dirs]):
            if cand.is_file():
                return cand.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"inference token file not found: {token_file}")
    raise ValueError("remote inference needs a token: set detector.remote.token_file or ARGONIX_INFER_TOKEN")


def fetch_info(url: str, token: str, timeout_s: float = 3.0) -> Optional[dict]:
    """The node's model description (GET /v1/info), or None when unreachable."""
    import httpx
    try:
        r = httpx.get(http_base(url) + "/v1/info", headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s)
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def encode_frame(frame: np.ndarray, max_side: int, fmt: str = "jpeg", quality: int = 80) -> tuple[bytes, float, int, int]:
    """Downscale so the longest side is <= max_side, then encode. Returns (payload, scale back to the input, w, h)."""
    import cv2
    h0, w0 = frame.shape[:2]
    s = min(1.0, max_side / max(h0, w0)) if max_side else 1.0
    img = frame if s >= 1.0 else cv2.resize(frame, (max(1, round(w0 * s)), max(1, round(h0 * s))),
                                            interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if fmt == "raw":
        payload = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        payload = buf.tobytes()
    return payload, (w0 / w), w, h


def pack_frame(seq: int, payload: bytes, fmt: str, w: int, h: int, ts: Optional[float] = None) -> bytes:
    hdr = json.dumps({"seq": seq, "ts": time.time() if ts is None else ts, "fmt": fmt, "w": w, "h": h},
                     separators=(",", ":")).encode()
    return struct.pack("!I", len(hdr)) + hdr + payload


class _IdMap:
    """(epoch, remote track id) -> small per-camera id; entries unseen for `ttl_s` are forgotten."""

    def __init__(self, ttl_s: float = 600.0):
        self.ttl = ttl_s
        self.m: dict[tuple, list] = {}
        self.next = itertools.count(1)
        self._prune_at = 0.0

    def get(self, key: tuple, now: float) -> int:
        e = self.m.get(key)
        if e is None:
            e = self.m[key] = [next(self.next), now]
        e[1] = now
        if now >= self._prune_at:
            self._prune_at = now + 30.0
            for k in [k for k, v in self.m.items() if now - v[1] > self.ttl]:
                del self.m[k]
        return e[0]


LADDER = ((640, 85), (640, 75), (480, 75), (480, 65), (416, 55), (352, 50), (320, 45), (256, 40))


def ladder(max_side: int, quality: int, min_side: int = 256) -> list[tuple[int, int]]:
    """Encoding rungs from the configured (max_side, quality) down to `min_side`, each strictly cheaper."""
    rungs = [(int(max_side), int(quality))]
    for side, q in LADDER:
        ps, pq = rungs[-1]
        if min_side <= side <= ps and q <= pq and (side, q) != (ps, pq):
            rungs.append((side, q))
    return rungs


class AdaptiveEncoder:
    """Chooses the encoding rung from measured round-trip times (see the module docstring).

    `observe(rtt_ms, now)` returns True when the rung changed. Decisions use the median of the samples taken at the
    current rung: `window` of them, or at least `min_samples` once `max_window_s` has passed (slow links deliver few
    frames), so a single slow frame does not move it.
    """

    def __init__(self, rungs: list[tuple[int, int]], target_ms: float = 250.0, window: int = 12,
                 hold_s: float = 60.0, gain: float = 0.95, min_samples: int = 3, max_window_s: float = 10.0):
        self.rungs, self.target, self.window, self.hold_s, self.gain = rungs, float(target_ms), window, hold_s, gain
        self.min_samples, self.max_window_s = min_samples, max_window_s
        self.level = 0
        self.t_first: Optional[float] = None
        self.samples: list[float] = []
        self.rtt: dict[int, float] = {}      # median RTT last measured at each rung
        self.came_from: Optional[int] = None
        self.no_down_until = self.no_up_until = 0.0
        self.changes = 0

    @property
    def rung(self) -> tuple[int, int]:
        return self.rungs[self.level]

    def _move(self, to: int, came_from: Optional[int]) -> bool:
        self.came_from, self.level, self.samples = came_from, to, []
        self.changes += 1
        return True

    def observe(self, rtt_ms: float, now: float) -> bool:
        if not self.samples:
            self.t_first = now
        self.samples.append(float(rtt_ms))
        if len(self.samples) < self.window and (len(self.samples) < self.min_samples
                                                or now - self.t_first < self.max_window_s):
            return False
        med = float(np.median(self.samples))
        self.samples = []
        lv, prev = self.level, self.came_from
        self.rtt[lv] = med
        self.came_from = None
        if prev == lv - 1 and prev in self.rtt and med > self.gain * self.rtt[prev]:
            self.no_down_until = now + self.hold_s  # smaller frames did not help (latency-bound): go back
            return self._move(prev, None)
        if prev == lv + 1 and med > self.target:
            self.no_up_until = now + self.hold_s  # the bigger frames are too slow again: go back
            return self._move(prev, None)
        if med > self.target and lv < len(self.rungs) - 1 and now >= self.no_down_until:
            return self._move(lv + 1, lv)
        if med < 0.5 * self.target and lv > 0 and now >= self.no_up_until:
            return self._move(lv - 1, lv)
        return False


class UplinkBudget:
    """Token bucket (bytes) shared by the cameras of one site; `reserve(n)` says how long to wait before sending."""

    def __init__(self, kbps: float, burst_s: float = 1.0):
        self.rate = kbps * 1000 / 8  # bytes per second
        self.burst = self.rate * burst_s
        self.tokens, self.t = self.burst, time.monotonic()
        self.lock = threading.Lock()

    def reserve(self, n: int, max_wait_s: float) -> Optional[float]:
        """Seconds to wait before sending `n` bytes, or None (nothing reserved) if that would exceed `max_wait_s`."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
            self.t = now
            wait = max(0.0, (n - self.tokens) / self.rate)
            if wait > max_wait_s:
                return None
            self.tokens -= n
            return wait


class RemoteTrackerBackend:
    """`TrackerBackend` that runs detection + tracking on a remote node. See the module docstring."""

    def __init__(self, url: str, camera_id: str, *, token: str, client_id: Optional[str] = None,
                 fmt: str = "jpeg", jpeg_quality: int = 80, max_side: int = 0, timeout_s: float = 1.0,
                 connect_timeout_s: float = 10.0, backoff_min_s: float = 0.5, backoff_max_s: float = 10.0,
                 max_timeouts: int = 3, fallback: Optional[Callable[[], TrackerBackend]] = None,
                 fallback_after_s: float = 10.0, node_info: Optional[dict] = None, autostart: bool = False,
                 adaptive: bool = False, target_rtt_ms: float = 250.0, min_side: int = 256,
                 max_timeout_s: float = 5.0, budget: Optional[UplinkBudget] = None):
        if fmt not in ("jpeg", "raw"):
            raise ValueError("fmt must be 'jpeg' or 'raw'")
        self.url, self.camera_id = ws_base(url), str(camera_id)
        self.client_id = client_id or socket.gethostname()
        self.token = token
        self.fmt, self.quality, self.max_side = fmt, int(jpeg_quality), int(max_side)
        self.timeout_s, self.connect_timeout_s = float(timeout_s), float(connect_timeout_s)
        self.backoff_min_s, self.backoff_max_s = float(backoff_min_s), float(backoff_max_s)
        self.max_timeouts = max(1, int(max_timeouts))
        self._fallback_factory, self.fallback_after_s = fallback, float(fallback_after_s)
        self._fallback: Optional[TrackerBackend] = None
        self._node: dict = dict(node_info or {})   # the node's /v1/info or hello: {"node", "model", ...}
        self._lock = threading.Lock()
        self._ws = None
        self._session: Optional[str] = None
        self._remote_epoch = 0                     # bumped when the node-side tracker is new (or went stale)
        self._fallback_epoch = 0                   # bumped each time the local fallback takes over
        self._active = "none"                      # remote | local-fallback | none
        self._ids = _IdMap()
        self._seq = 0
        self._down_since: Optional[float] = time.monotonic()
        self._consec_timeouts = 0
        self._state = "connecting"
        self._last_error = ""
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._first = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rtts: deque = deque(maxlen=200)
        self.adaptive, self.target_rtt_ms, self.min_side = bool(adaptive) and fmt == "jpeg", target_rtt_ms, min_side
        self.max_timeout_s = max(float(max_timeout_s), self.timeout_s)
        self._adapt: Optional[AdaptiveEncoder] = None  # built on first use (the node's imgsz may be needed)
        self.budget = budget
        self.counters = {"frames": 0, "ok": 0, "dropped_remote": 0, "timeouts": 0, "errors": 0, "late": 0,
                         "connects": 0, "disconnects": 0, "fallback_frames": 0, "bytes_sent": 0,
                         "skipped_budget": 0}
        self._ema = {"rtt_ms": 0.0, "server_ms": 0.0, "encode_ms": 0.0, "kb": 0.0}
        if autostart:
            self.start()

    # ---- connection management ---------------------------------------------------------------------
    @property
    def stream_url(self) -> str:
        return f"{self.url}/v1/stream?client={quote(self.client_id, safe='')}&camera={quote(self.camera_id, safe='')}"

    def start(self, wait: bool = True) -> None:
        """Start the connector thread (done lazily by the first `track()`); optionally wait for its first attempt."""
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._connector, daemon=True, name=f"remote-{self.camera_id}")
            self._thread.start()
        if wait:
            self._first.wait(self.connect_timeout_s + 1.0)

    def _open(self):
        from websockets.sync.client import connect
        ws = connect(self.stream_url, additional_headers={"Authorization": f"Bearer {self.token}"},
                     open_timeout=self.connect_timeout_s, close_timeout=2, compression=None, proxy=None,
                     max_size=2**22, ping_interval=10, ping_timeout=20,
                     user_agent_header="argonix-remote-inference")
        try:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # small replies must not wait on Nagle
        except OSError:
            pass
        try:
            hello = json.loads(ws.recv(timeout=self.connect_timeout_s))
            if hello.get("type") != "hello":
                raise RuntimeError(f"unexpected first message {hello!r:.100}")
        except BaseException:
            ws.close()
            raise
        return ws, hello

    def _connector(self) -> None:
        from websockets.exceptions import InvalidStatus
        delay = self.backoff_min_s
        while not self._stop.is_set():
            if self._ws is None and self._state not in ("replaced", "closed"):
                try:
                    ws, hello = self._open()
                except Exception as e:  # noqa: BLE001 - refused, timed out, unauthorized, DNS...
                    if isinstance(e, InvalidStatus) and getattr(e.response, "status_code", 0) == 403:
                        self._state, self._last_error = "unauthorized", "node refused the token (HTTP 403)"
                    else:
                        self._state, self._last_error = "down", repr(e)[:200]
                    self._first.set()
                    if self._stop.wait(delay):
                        break
                    delay = min(self.backoff_max_s, delay * 2)
                    continue
                with self._lock:
                    if self._stop.is_set():
                        ws.close()
                        break
                    if hello.get("session") != self._session or not hello.get("resumed"):
                        self._remote_epoch += 1  # the node has a fresh tracker for us: IDs must not continue
                    self._session = hello.get("session")
                    self._node = {k: v for k, v in hello.items() if k != "type"}
                    self._ws, self._consec_timeouts = ws, 0
                    self._state, self._last_error = "up", ""
                    self.counters["connects"] += 1
                delay = self.backoff_min_s
                self._first.set()
            self._wake.wait(1.0)
            self._wake.clear()

    def _drop_connection(self, why: str, state: str = "down") -> None:
        with self._lock:
            ws, self._ws = self._ws, None
            if ws is not None:
                self.counters["disconnects"] += 1
                if self._down_since is None:
                    self._down_since = time.monotonic()
            if self._state not in ("replaced", "closed"):
                self._state = state
            self._last_error = why[:200]
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._wake.set()

    # ---- TrackerBackend ------------------------------------------------------------------------------
    def track(self, frame) -> list[Det]:
        if self._thread is None:
            self.start(wait=True)
        self.counters["frames"] += 1
        ws = self._ws
        if ws is not None:
            dets = self._remote_track(ws, frame)
            if dets is not None:
                return dets
            if self._ws is not None:  # connected, but this frame timed out / was dropped: skip it
                return []
        # node down (or just went down)
        if self._down_since is None:
            self._down_since = time.monotonic()
        if (self._fallback_factory is not None and self._state not in ("replaced", "closed")
                and time.monotonic() - self._down_since >= self.fallback_after_s):
            return self._fallback_track(frame)
        self._active = "none"
        return []

    def _encoding(self) -> tuple[int, int]:
        """(max_side, jpeg quality) for the next frame."""
        base = self.max_side or int((self._node.get("model") or {}).get("imgsz", 640))
        if not self.adaptive:
            return base, self.quality
        if self._adapt is None:
            self._adapt = AdaptiveEncoder(ladder(base, self.quality, self.min_side), self.target_rtt_ms)
        return self._adapt.rung

    def _timeout(self) -> float:
        if not self.adaptive:
            return self.timeout_s
        if len(self._rtts) < 5:
            return self.max_timeout_s  # link not measured yet: be patient
        r = sorted(self._rtts)
        p95 = r[min(len(r) - 1, int(0.95 * len(r)))]
        return min(self.max_timeout_s, max(self.timeout_s, 3 * p95 / 1000))

    def _remote_track(self, ws, frame) -> Optional[list[Det]]:
        from websockets.exceptions import ConnectionClosed
        t0 = time.perf_counter()
        side, quality = self._encoding()
        payload, scale, w, h = encode_frame(frame, side, self.fmt, quality)
        if self.budget is not None:
            wait = self.budget.reserve(len(payload), max_wait_s=2.0)
            if wait is None:  # over the site's uplink budget: skip this frame rather than stall the camera
                self.counters["skipped_budget"] += 1
                return None
            if wait > 0:
                time.sleep(wait)
        t1 = time.perf_counter()
        self._seq += 1
        seq = self._seq
        timeout = self._timeout()
        try:
            ws.send(pack_frame(seq, payload, self.fmt, w, h))
            deadline = t1 + timeout
            while True:
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError
                r = json.loads(ws.recv(timeout=left))
                if r.get("type") != "result":
                    continue
                if r.get("seq") != seq:
                    self.counters["late"] += 1  # a reply for a frame we already gave up on
                    continue
                break
        except TimeoutError:
            self.counters["timeouts"] += 1
            self.counters["dropped_remote"] += 1
            self._consec_timeouts += 1
            if self._consec_timeouts >= self.max_timeouts:
                self._drop_connection(f"{self._consec_timeouts} consecutive timeouts ({timeout:.1f} s)")
            return None
        except ConnectionClosed as e:
            code = getattr(getattr(e, "rcvd", None), "code", None)
            if code == CLOSE_REPLACED:  # another connection took this camera over: do not fight it
                self._drop_connection("replaced by a newer connection for this camera", state="replaced")
            else:
                self._drop_connection(f"connection closed ({code})")
            return None
        except Exception as e:  # noqa: BLE001 - socket errors
            self._drop_connection(repr(e))
            return None
        self._consec_timeouts = 0
        self.counters["bytes_sent"] += len(payload)
        if self._active == "local-fallback":
            # back from the fallback: the node's tracker has not seen the frames in between, so a lost track could
            # re-attach to someone else. Start new IDs rather than risk one ID for two people.
            with self._lock:
                self._remote_epoch += 1
        rtt = (time.perf_counter() - t1) * 1000
        if r.get("status") != "ok":
            self.counters["dropped_remote" if r.get("status") == "dropped" else "errors"] += 1
            if r.get("status") == "error":
                self._last_error = str(r.get("reason", ""))[:200]
            return None
        self._rtts.append(rtt)
        if self._adapt is not None:
            self._adapt.observe(rtt, time.monotonic())
        self._ema_add("rtt_ms", rtt)
        self._ema_add("server_ms", float((r.get("server") or {}).get("total_ms", 0.0)))
        self._ema_add("encode_ms", (t1 - t0) * 1000)
        self._ema_add("kb", len(payload) / 1024)
        self.counters["ok"] += 1
        self._down_since = None
        self._active = "remote"
        now = time.monotonic()
        with self._lock:
            epoch = self._remote_epoch
        out: list[Det] = []
        for tid, x1, y1, x2, y2, cls, conf in r.get("dets", []):
            out.append((self._ids.get(("r", epoch, int(tid)), now),
                        (x1 * scale, y1 * scale, x2 * scale, y2 * scale), str(cls), float(conf)))
        return out

    def _fallback_track(self, frame) -> list[Det]:
        if self._fallback is None:
            try:
                self._fallback = self._fallback_factory()
            except Exception as e:  # noqa: BLE001 - e.g. weights missing: stay without detections
                self._fallback_factory = None
                self._last_error = f"fallback unavailable: {e!r}"[:200]
                return []
        if self._active != "local-fallback":
            self._fallback_epoch += 1  # the local tracker's IDs never continue a remote (or older fallback) track
        self._active = "local-fallback"
        self.counters["fallback_frames"] += 1
        now = time.monotonic()
        return [(self._ids.get(("f", self._fallback_epoch, int(tid)), now), bb, cls, conf)
                for tid, bb, cls, conf in self._fallback.track(frame)]

    def _ema_add(self, k: str, v: float) -> None:
        self._ema[k] = v if self._ema[k] == 0.0 else 0.9 * self._ema[k] + 0.1 * v

    # ---- introspection ---------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        if self._state in ("up", "connecting", "down") and self._active == "local-fallback" and self._ws is None:
            return "fallback"
        return self._state

    @property
    def info(self) -> dict:
        """Describes the model that produced the latest detections (remote node or local fallback)."""
        base = {"runtime": "remote", "node": self.url, "node_name": self._node.get("node"),
                "client_id": self.client_id, "active": self._active, "state": self.state,
                "transport": self._transport()}
        if self._active == "local-fallback" and self._fallback is not None:
            fb = dict(getattr(self._fallback, "info", {}) or {})
            return {**fb, **base, "fallback_runtime": fb.get("runtime")}
        m = dict(self._node.get("model") or {})
        if not m:
            return {**base, "detector": "unknown (node not reached yet)"}
        return {**m, **base, "node_runtime": m.get("runtime")}

    def _transport(self) -> str:
        if self.fmt == "raw":
            return f"raw <= {self.max_side or 'node imgsz'} px"
        if self._adapt is not None:
            side, q = self._adapt.rung
            return f"jpeg q{q} <= {side} px (adaptive, rung {self._adapt.level + 1}/{len(self._adapt.rungs)})"
        side = self.max_side or (self._node.get("model") or {}).get("imgsz", "?")
        return f"jpeg q{self.quality} <= {side} px"

    def camera_stats(self) -> dict:
        """Fields merged into the camera's stats (System page / `/api/stats`)."""
        r = sorted(self._rtts)
        p95 = r[min(len(r) - 1, int(0.95 * len(r)))] if r else None
        return {"remote_state": self.state, "rtt_ms": round(self._ema["rtt_ms"], 1),
                "rtt_p95_ms": round(p95, 1) if p95 is not None else None,
                "dropped_remote": self.counters["dropped_remote"], "remote_timeouts": self.counters["timeouts"],
                "remote_reconnects": max(0, self.counters["connects"] - 1),
                "fallback_frames": self.counters["fallback_frames"], "remote_error": self._last_error,
                "kb_per_frame": round(self._ema["kb"], 1), "node_ms": round(self._ema["server_ms"], 1),
                "remote_fps_max": round(1000 / self._ema["rtt_ms"], 2) if self._ema["rtt_ms"] else None,
                "transport": self._transport(), "skipped_budget": self.counters["skipped_budget"]}

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._state = "closed"
        self._drop_connection("closed", state="closed")
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3.0)
        fb, self._fallback = self._fallback, None
        if fb is not None and hasattr(fb, "close"):
            fb.close()


def make_remote_factory(det: dict, models_dir: Optional[Path] = None) -> Callable[[dict], TrackerBackend]:
    """Backend factory for `detector.remote` (see the module docstring). The returned callable exposes `close()`,
    `remote` ({url, token, client_id}) and `backends` ({camera_id: RemoteTrackerBackend}) for the node registry."""
    r = dict(det["remote"])
    if not r.get("url"):
        raise ValueError("detector.remote.url is required, e.g. ws://gpu-host:8200")
    project = Path(models_dir).resolve().parents[1] if models_dir else Path.cwd()
    token = load_token(r.get("token"), r.get("token_file"), (project,))
    client_id = r.get("client_id") or socket.gethostname()
    connect_timeout = float(r.get("connect_timeout_s", 10.0))
    budget = UplinkBudget(float(r["uplink_kbps"])) if float(r.get("uplink_kbps", 0) or 0) > 0 else None
    info = fetch_info(r["url"], token, connect_timeout)  # model description for the site's model card; may be None
    # Object/vehicle classes (detector.extra_classes, added when unattended-object analysis or `vehicles` is on) are
    # detected by the NODE: it must be started with --extra-classes (e.g. 2,3,5,7,24,26,28). The node's class list is
    # fixed per node, so only warn here; the local CPU fallback detects them itself.
    extra = tuple(int(c) for c in (det.get("extra_classes") or ()))
    extra_conf = det.get("extra_conf")
    extra_conf = float(extra_conf) if extra_conf is not None else None
    if extra:
        node_extra = set(int(c) for c in (((info or {}).get("model") or {}).get("extra_classes") or ()))
        missing = [c for c in extra if c not in node_extra]
        if missing and info is not None:
            log.warning("inference node %s does not detect object classes %s (detector.extra_classes): unattended-object "
                        "and vehicle analysis get no such detections from it. Start the node with --extra-classes %s.",
                        r["url"], missing, ",".join(str(c) for c in extra))
        elif info is None:
            log.warning("inference node %s unreachable at start-up: cannot confirm it detects object classes %s "
                        "(start it with --extra-classes %s).", r["url"], list(extra), ",".join(str(c) for c in extra))

    fb = r.get("fallback", "local")
    fallback = None
    if fb == "local":
        from .detect import UltralyticsTracker
        fb_w = r.get("fallback_weights", "yolo26n.pt")
        if models_dir and not Path(fb_w).is_absolute() and (Path(models_dir) / fb_w).exists():
            fb_w = str(Path(models_dir) / fb_w)

        def fallback():
            return UltralyticsTracker(fb_w, det.get("tracker", "bytetrack.yaml"), float(det.get("conf", 0.3)),
                                      int(r.get("fallback_imgsz", 320)), tuple(det.get("classes", (0,))), "cpu",
                                      models_dir, extra, extra_conf)
    elif fb not in (None, False, "none", ""):
        raise ValueError("detector.remote.fallback must be 'local' or 'none'")

    backends: dict[str, RemoteTrackerBackend] = {}

    def factory(cam: dict) -> TrackerBackend:
        b = RemoteTrackerBackend(
            r["url"], cam["id"], token=token, client_id=client_id, fmt=r.get("fmt", "jpeg"),
            jpeg_quality=int(r.get("jpeg_quality", 80)), max_side=int(r.get("max_side", 0)),
            timeout_s=float(r.get("timeout_ms", 1000)) / 1000, connect_timeout_s=connect_timeout,
            backoff_min_s=float(r.get("backoff_min_s", 0.5)), backoff_max_s=float(r.get("backoff_max_s", 10.0)),
            max_timeouts=int(r.get("max_timeouts", 3)), fallback=fallback,
            fallback_after_s=float(r.get("fallback_after_s", 10.0)), node_info=info,
            adaptive=bool(r.get("adaptive", True)), target_rtt_ms=float(r.get("target_rtt_ms", 250)),
            min_side=int(r.get("min_side", 256)), max_timeout_s=float(r.get("max_timeout_ms", 5000)) / 1000,
            budget=budget)
        backends[cam["id"]] = b  # connects on its first track(), i.e. when its worker starts
        return b

    def close() -> None:
        for b in list(backends.values()):
            b.close()

    factory.close = close
    factory.remote = {"url": ws_base(r["url"]), "token": token, "client_id": client_id}
    factory.backends = backends
    return factory
