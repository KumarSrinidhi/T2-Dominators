"""Simulated ESP32-CAM node: the same HTTP API as firmware/esp32cam_argonix, for testing without the board.

    uv run python tools/sim_esp32cam.py --port 8090 [--video data/synth/cam1.mp4]

    http://127.0.0.1:8090/stream   MJPEG (the real board serves this on :81)
    http://127.0.0.1:8090/pir      {"motion":..,"count":..}
    http://127.0.0.1:8090/pir/on   set motion (simulate someone walking past the PIR); /pir/off clears it
    http://127.0.0.1:8090/status
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SimState:
    def __init__(self, video: Path | None, fps: float = 10.0):
        self.video, self.fps = video, fps
        self.motion, self.count, self.last_change = False, 0, time.time()
        self.lock = threading.Lock()
        # Decode everything up front: OpenCV serialises FFmpeg use, so decoding the video inside a server thread
        # would block (and time out) a client in the same process that is opening the stream.
        self.jpegs = self._load()

    def set_motion(self, on: bool) -> None:
        with self.lock:
            if on and not self.motion:
                self.count += 1
            self.motion, self.last_change = on, time.time()

    def _load(self, max_frames: int = 600) -> list[bytes]:
        out = []
        if self.video and Path(self.video).exists():
            cap = cv2.VideoCapture(str(self.video))
            while len(out) < max_frames:
                ok, f = cap.read()
                if not ok:
                    break
                out.append(cv2.imencode(".jpg", cv2.resize(f, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            cap.release()
        if not out:  # no video: a moving test pattern
            for n in range(80):
                f = np.full((480, 640, 3), 60, np.uint8)
                cv2.circle(f, (40 + (n * 7) % 560, 240), 30, (0, 200, 255), -1)
                out.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        return out

    def frames(self):
        """Endless JPEG frames (the video looped)."""
        while True:
            yield from self.jpegs


def make_handler(state: SimState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, d: dict) -> None:
            body = json.dumps(d).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/pir/on"):
                state.set_motion(True)
            elif self.path.startswith("/pir/off"):
                state.set_motion(False)
            if self.path.startswith("/pir"):
                with state.lock:
                    return self._json({"motion": state.motion, "count": state.count,
                                       "last_change_ms": int(state.last_change * 1000), "uptime_ms": 0})
            if self.path.startswith("/status"):
                return self._json({"ip": "127.0.0.1", "simulated": True, "motion": state.motion})
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=argonixframe")
                self.end_headers()
                try:
                    for jpg in state.frames():
                        self.wfile.write(b"\r\n--argonixframe\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg)
                        time.sleep(1.0 / state.fps)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                return
            self.send_error(404)
    return H


def serve(port: int, video: Path | None = None, fps: float = 10.0) -> tuple[ThreadingHTTPServer, SimState]:
    state = SimState(video, fps)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--video", default=str(ROOT / "data" / "synth" / "cam1.mp4"))
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    srv, _ = serve(a.port, Path(a.video), a.fps)
    print(f"Simulated ESP32-CAM on http://127.0.0.1:{a.port}  (stream: /stream, sensor: /pir, toggle: /pir/on /pir/off)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
