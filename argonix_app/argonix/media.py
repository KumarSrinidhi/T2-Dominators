"""Browser-playable H.264 writing via an ffmpeg subprocess, plus drawing helpers."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG = None


class H264Writer:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if FFMPEG:
            cmd = [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.cv = None
        else:  # fallback: not browser-playable everywhere
            self.proc = None
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.size = (w, h)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        elif self.cv:
            self.cv.release()


OVERLAY_LABEL = "software overlay"


def stamp(frame: np.ndarray, text: str = OVERLAY_LABEL) -> np.ndarray:
    """Copy of `frame` with a small label in the bottom-left corner, e.g. to mark boxes/zones as drawn by software
    (so an annotated evidence image is never mistaken for the camera's original picture)."""
    out = frame.copy()
    h = out.shape[0]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(out, (0, h - th - base - 6), (tw + 8, h), (0, 0, 0), -1)
    cv2.putText(out, text, (4, h - base - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_clip(path: Path, frames: list[np.ndarray], fps: float, label: str = "") -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    wr = H264Writer(path, w, h, fps)
    for f in frames:
        wr.write(stamp(f, label) if label else f)
    wr.close()


VEHICLE_COL = (255, 160, 0)  # BGR: blue-cyan boxes for vehicles, distinct from people (green) and alerts (red)


def draw(frame: np.ndarray, dets, zones_px, alert_tids: set[int], masks_px=(), vehicle_labels=()) -> np.ndarray:
    out = frame.copy()
    vehicle_labels = set(vehicle_labels)
    for m in masks_px:  # hatched dark overlay: detections here are ignored
        pts = np.array(m, np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (40, 40, 40))
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
        cv2.polylines(out, [pts], True, (160, 160, 160), 1)
        cv2.putText(out, "mask", tuple(pts[0] + [3, 14]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    for name, poly in zones_px:
        pts = np.array(poly, np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, (0, 140, 255), 2)
        cv2.putText(out, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    for tid, (x1, y1, x2, y2), cls, conf in dets:
        col = (0, 0, 255) if tid in alert_tids else VEHICLE_COL if cls in vehicle_labels else (80, 220, 80)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(out, f"{cls} #{tid} {conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out
