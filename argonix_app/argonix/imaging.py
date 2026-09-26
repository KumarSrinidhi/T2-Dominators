"""Frame conditioning for real CCTV: black-and-white / IR night mode and low light.

- Single-channel or BGRA frames are converted to 3-channel BGR (detectors expect 3 channels; YOLO works on
  replicated grey reasonably, but accuracy on IR is lower than on colour — measure per site).
- Monochrome is detected from inter-channel differences (many cameras send IR as 3 identical channels).
- Optional CLAHE contrast enhancement for dark or low-contrast (typical IR) frames, used for detection only when
  `enhance` is "auto" (dark/low-contrast frames) or "clahe" (always); "off" disables it.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame


def is_monochrome(frame: np.ndarray, thr: float = 3.0) -> bool:
    """True when the colour channels carry (almost) no colour information."""
    if frame.ndim == 2 or frame.shape[2] == 1:
        return True
    s = frame[::4, ::4].astype(np.int16)
    diff = (np.abs(s[..., 0] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 2]).mean()) / 2
    return float(diff) < thr


class Conditioner:
    """Per-camera: normalises channels, tracks monochrome state, applies CLAHE when useful."""

    def __init__(self, mode: str = "auto", check_every: int = 25):
        if mode not in ("auto", "clahe", "off"):
            raise ValueError('enhance must be "auto", "clahe" or "off"')
        self.mode = mode
        self.check_every = check_every
        self.n = 0
        self.monochrome = False
        self.enhanced = False
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        mono_input = frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)
        frame = to_bgr(frame)
        if self.n % self.check_every == 0:
            self.monochrome = mono_input or is_monochrome(frame)
        self.n += 1
        if self.mode == "off":
            self.enhanced = False
            return frame
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_or_flat = float(grey.mean()) < 70 or float(grey.std()) < 35
        if self.mode == "clahe" or dark_or_flat:
            self.enhanced = True
            if self.monochrome:
                return cv2.cvtColor(self._clahe.apply(grey), cv2.COLOR_GRAY2BGR)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.enhanced = False
        return frame
