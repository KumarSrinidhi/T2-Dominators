"""Track stitching: keep one ID for a person the tracker briefly lost.

ByteTrack re-finds a lost track only when its Kalman-predicted box still overlaps the new detection. After a
longer occlusion, a crowd crossing or fast movement at 5-10 fps the prediction has drifted, the IoU is zero, and the
same person comes back under a NEW id, which re-fires rules such as loitering and breaks dwell times.

This layer sits after the tracker and re-links such tracks. When an id appears that has not been seen before, it is
matched against people lost in the last `max_gap_frames` frames: same class, similar box height, and the new foot
point near where the lost person was heading (distance measured in body heights, allowed to grow with the gap). The
nearest match keeps its old id; otherwise the track gets a new id. Deterministic, no appearance model, no identity.
Object tracks (ids >= OBJECT_ID_OFFSET) pass through unchanged.

On by default for every detector path; turn it off with "detector": {"stitch": {"enabled": false}} or tune it with
"detector": {"stitch": {"max_gap_frames": 5, "base_bh": 0.25, "per_frame_bh": 0.08, "min_size_ratio": 0.75}}.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Optional

from .detect import OBJECT_ID_OFFSET


class TrackStitcher:
    # Defaults from the MOT17 sweep (2026-09-26, 5 fps, YOLO26s 960 px, conf 0.1): with them ID switches fell 834 ->
    # 718 and IDF1 rose 0.444 -> 0.460, HOTA 0.372 -> 0.380. Looser settings (15-frame gap, 0.6 + 0.25/frame body
    # heights) removed more switches but joined different people (IDF1 0.412), so they are not the default.
    def __init__(self, max_gap_frames: int = 5, base_bh: float = 0.25, per_frame_bh: float = 0.08,
                 min_size_ratio: float = 0.75, max_speed_bh: float = 0.6):
        self.max_gap = int(max_gap_frames)
        self.base, self.per_frame = float(base_bh), float(per_frame_bh)
        self.min_ratio = float(min_size_ratio)
        self.max_speed = float(max_speed_bh)  # cap on the velocity used for prediction, body heights per frame
        self.frame = 0
        self.ids = itertools.count(1)
        self.map: dict[int, int] = {}                 # raw tracker id -> stable id
        self.raw_seen: dict[int, int] = {}            # raw id -> last frame seen
        self.state: dict[int, dict[str, Any]] = {}    # stable id -> last position / size / velocity / class
        self.stats = {"stitched": 0, "new": 0}

    def _match(self, cls: str, cx: float, cy: float, h: float, taken: set) -> Optional[int]:
        best, best_score = None, 1.0
        for sid, s in self.state.items():
            gap = self.frame - s["frame"]
            if sid in taken or s["cls"] != cls or gap < 1 or gap > self.max_gap:
                continue
            ratio = h / max(s["h"], 1e-6)
            if not self.min_ratio <= ratio <= 1 / self.min_ratio:
                continue
            px, py = s["cx"] + s["vx"] * gap, s["cy"] + s["vy"] * gap
            ref_h = max(h, s["h"], 1.0)
            dist = math.hypot(cx - px, cy - py) / ref_h
            score = dist / (self.base + self.per_frame * gap)
            if score < best_score:
                best, best_score = sid, score
        return best

    def update(self, dets: list) -> list:
        """dets: [(track_id, (x1, y1, x2, y2), cls, conf)] from the tracker -> the same with stable ids."""
        self.frame += 1
        out, taken = [], set()
        for tid, box, cls, conf in dets:
            if tid >= OBJECT_ID_OFFSET:
                out.append((tid, box, cls, conf))
                continue
            x1, y1, x2, y2 = box
            cx, cy, h = (x1 + x2) / 2.0, float(y2), max(float(y2 - y1), 1.0)
            sid = self.map.get(tid)
            if sid is None or sid in taken:
                matched = self._match(cls, cx, cy, h, taken) if sid is None else None
                if matched is not None:
                    sid = matched
                    self.stats["stitched"] += 1
                else:
                    sid = next(self.ids)
                    self.stats["new"] += 1
                self.map[tid] = sid
            prev = self.state.get(sid)
            vx = vy = 0.0
            if prev is not None and self.frame > prev["frame"]:
                g = self.frame - prev["frame"]
                cap = self.max_speed * h
                vx = max(-cap, min(cap, (cx - prev["cx"]) / g))
                vy = max(-cap, min(cap, (cy - prev["cy"]) / g))
                vx, vy = 0.5 * vx + 0.5 * prev["vx"], 0.5 * vy + 0.5 * prev["vy"]  # smooth detector jitter
            self.state[sid] = {"frame": self.frame, "cx": cx, "cy": cy, "h": h, "cls": cls, "vx": vx, "vy": vy}
            self.raw_seen[tid] = self.frame
            taken.add(sid)
            out.append((sid, box, cls, conf))
        # forget people lost longer than the gap, and raw ids the tracker has dropped
        horizon = self.frame - self.max_gap
        for sid in [k for k, s in self.state.items() if s["frame"] < horizon]:
            del self.state[sid]
        for tid in [k for k, f in self.raw_seen.items() if f < horizon - self.max_gap]:
            del self.raw_seen[tid]
            self.map.pop(tid, None)
        return out


class StitchedBackend:
    """Wraps any TrackerBackend so its person track ids go through a TrackStitcher. Everything else is forwarded."""

    def __init__(self, backend, cfg: Optional[dict] = None):
        self._backend = backend
        c = {k: v for k, v in (cfg or {}).items() if k != "enabled"}
        self._stitcher = TrackStitcher(**c)
        self.info = {**dict(getattr(backend, "info", {}) or {}), "stitch": {"enabled": True, **c}}

    def track(self, frame) -> list:
        return self._stitcher.update(self._backend.track(frame))

    def __getattr__(self, name):  # close(), camera_stats, ... of the wrapped backend
        return getattr(self._backend, name)


def wrap_factory(factory, det: dict):
    """Apply `detector.stitch` (when enabled) to every backend the factory makes; keeps the factory's attributes."""
    cfg = {"enabled": True, **(det.get("stitch") or {})}
    if not cfg.get("enabled"):
        return factory

    def stitched(cam: dict):
        return StitchedBackend(factory(cam), cfg)
    for attr in ("close", "prewarm", "remote", "backends"):
        if hasattr(factory, attr):
            setattr(stitched, attr, getattr(factory, attr))
    return stitched
