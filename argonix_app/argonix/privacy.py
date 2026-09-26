"""Face anonymisation for derived copies (evidence export with ?anonymise=1, frames sent to the VLM).

The originals stay the evidence record; anonymised images are labelled derived copies. What gets blurred:
  * faces found by a face detector: OpenCV's Haar frontal-face cascade (cv2.data.haarcascades) when the installed
    OpenCV ships it, else OpenCV's YuNet (cv2.FaceDetectorYN) when its ONNX model file is present locally
    (site config "privacy": {"face_model": "data/models/face_detection_yunet_2023mar.onnx"}); nothing is downloaded;
  * the head region (top `head_frac` of each person box) when the evidence recorded person boxes for that frame.
    CCTV looks down on people and a frontal-face detector misses most faces there, so this is the dependable part.
Neither is a guarantee: a face outside a detected person box and missed by the face detector stays visible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

HAAR = "haarcascade_frontalface_default.xml"
YUNET_DEFAULT = "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """Best available local face detector, or none (`method` says which)."""

    def __init__(self, model_path: Optional[Path] = None):
        self.method, self._haar, self._yunet = "none", None, None
        base = getattr(getattr(cv2, "data", None), "haarcascades", "")
        if hasattr(cv2, "CascadeClassifier") and base and (Path(base) / HAAR).exists():
            c = cv2.CascadeClassifier(str(Path(base) / HAAR))
            if not c.empty():
                self._haar, self.method = c, "opencv_haar_frontalface"
        if self._haar is None and model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.7)
                self.method = "opencv_yunet"
            except cv2.error:
                self._yunet = None

    @property
    def available(self) -> bool:
        return self.method != "none"

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face boxes (x, y, w, h) in image pixels."""
        if self._haar is not None:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return [tuple(int(v) for v in r) for r in self._haar.detectMultiScale(
                cv2.equalizeHist(grey), scaleFactor=1.1, minNeighbors=5, minSize=(16, 16))]
        if self._yunet is not None:
            h, w = img.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            return [tuple(int(v) for v in f[:4]) for f in (faces if faces is not None else [])]
        return []


_detectors: dict = {}


def get_detector(model_path: Optional[Path] = None) -> FaceDetector:
    key = str(model_path or "")
    if key not in _detectors:
        _detectors[key] = FaceDetector(model_path)
    return _detectors[key]


def head_boxes(person_boxes: Iterable, frac: float = 0.22) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of the top `frac` of each person box (x1, y1, x2, y2)."""
    out = []
    for b in person_boxes or []:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        hh = max(4.0, (y2 - y1) * frac)
        out.append((int(x1), int(y1), max(1, int(x2 - x1)), int(hh)))
    return out


def blur_regions(img: np.ndarray, rects: Iterable, pad: float = 0.15) -> np.ndarray:
    """Pixelate + Gaussian-blur each (x, y, w, h) region (padded), in place on a copy."""
    out = img.copy()
    H, W = out.shape[:2]
    for x, y, w, h in rects:
        px, py = int(w * pad), int(h * pad)
        x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 8), max(1, (y1 - y0) // 8)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = max(3, (min(x1 - x0, y1 - y0) // 3) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def anonymise(img: np.ndarray, person_boxes=None, detector: Optional[FaceDetector] = None) -> tuple[np.ndarray, dict]:
    det = detector or get_detector()
    faces = det.detect(img) if det.available else []
    heads = head_boxes(person_boxes) if person_boxes else []
    info = {"faces": len(faces), "face_detector": det.method, "person_heads": len(heads)}
    return (blur_regions(img, list(faces) + heads) if faces or heads else img.copy()), info


def anonymise_jpeg(data: bytes, person_boxes=None, detector: Optional[FaceDetector] = None,
                   quality: int = 90) -> tuple[Optional[bytes], dict]:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, {"error": "not a decodable image"}
    out, info = anonymise(img, person_boxes, detector)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (buf.tobytes() if ok else None), info


def anonymise_clip_frames(path: Path, n: int = 6, detector: Optional[FaceDetector] = None) -> list[tuple[bytes, dict]]:
    """n frames spread over a clip, each face-blurred (no per-frame person boxes are stored for clips, so this
    relies on the face detector alone). Returns [] when no face detector is available."""
    det = detector or get_detector()
    if not det.available:
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while cap.isOpened() and len(frames) < 3000:
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:
        return []
    idx = sorted({round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(max(1, min(n, len(frames))))})
    out = []
    for i in idx:
        img, info = anonymise(frames[i], None, det)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            out.append((buf.tobytes(), {**info, "frame_index": i}))
    return out
