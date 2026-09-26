"""Automatic floor map: per-camera bird's-eye warps, a site composite with a metre grid, and an occupancy heatmap.

Everything here is a 2D ground-plane (homography) model, NOT a 3D twin: anything that is not on the flat ground
(people, walls, furniture) is smeared along the viewing ray in the warped image. Positions carry each camera's
calibration error.

Frames
  * local (per camera): the ground metres of that camera's calibration (auto calibration: camera-centred,
    x right / y forward);
  * site: local rotated by placement.rotation_deg (counter-clockwise) then shifted by placement.offset_m.
  * rasters are north-up: column = (x - xmin) * px_per_m, row = (ymax - y) * px_per_m.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

GAP_M = 2.0  # spacing between cameras laid side by side by default


# ---------------------------------------------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------------------------------------------
def as_h(H) -> Optional[np.ndarray]:
    if H is None:
        return None
    return np.asarray(H, float).reshape(3, 3)


def placement_matrix(placement: Optional[dict]) -> np.ndarray:
    p = placement or {}
    ox, oy = (list(p.get("offset_m") or [0.0, 0.0]) + [0.0, 0.0])[:2]
    a = math.radians(float(p.get("rotation_deg", 0.0)))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, float(ox)], [s, c, float(oy)], [0.0, 0.0, 1.0]])


def raster_matrix(bounds, ppm: float) -> np.ndarray:
    xmin, _, _, ymax = bounds
    return np.array([[ppm, 0.0, -xmin * ppm], [0.0, -ppm, ymax * ppm], [0.0, 0.0, 1.0]])


def _apply(M: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.column_stack([pts, np.ones(len(pts))]) @ M.T
    return q[:, :2] / q[:, 2:3]


def ground_origin(H, source_size, calibration: Optional[dict] = None) -> tuple[float, float]:
    """Reference point for the range limit: the camera itself for auto calibrations (camera-centred frame),
    otherwise the ground point under the bottom-centre of the image."""
    if calibration and calibration.get("method") == "auto-pedestrian":
        return (0.0, 0.0)
    W, Hh = source_size
    g = _apply(as_h(H), [(W / 2.0, Hh - 1.0)])[0]
    return (float(g[0]), float(g[1]))


def _front_sign(Hm: np.ndarray, source_size) -> float:
    """Sign of the homogeneous w for image points below the horizon (reference: bottom-centre pixel)."""
    W, Hh = source_size
    return 1.0 if (Hm[2] @ np.array([W / 2.0, Hh - 1.0, 1.0])) > 0 else -1.0


def visible_polygon(H, source_size, max_range_m: float = 40.0, origin=None, n: int = 24) -> list[list[float]]:
    """Ground polygon (local metres) seen by the camera: the image rectangle clipped below the horizon, projected
    to the ground and radially clamped to max_range_m around origin."""
    Hm = as_h(H)
    W, Hh = source_size
    s0 = _front_sign(Hm, source_size)
    corners = [(0, Hh - 1), (W - 1, Hh - 1), (W - 1, 0), (0, 0)]
    border = []
    for i in range(4):
        a, b = np.array(corners[i], float), np.array(corners[(i + 1) % 4], float)
        for t in np.linspace(0, 1, n, endpoint=False):
            border.append(a + (b - a) * t)
    border = np.array(border)
    w_ref = abs(Hm[2] @ np.array([W / 2.0, Hh - 1.0, 1.0]))
    eps = 1e-3 * w_ref

    # Sutherland-Hodgman clip against the half plane s0 * w >= eps (w is linear in pixel coordinates)
    def wv(p):
        return s0 * (Hm[2, 0] * p[0] + Hm[2, 1] * p[1] + Hm[2, 2]) - eps
    out = []
    for i in range(len(border)):
        p, q = border[i], border[(i + 1) % len(border)]
        wp, wq = wv(p), wv(q)
        if wp >= 0:
            out.append(p)
        if (wp >= 0) != (wq >= 0):
            out.append(p + (q - p) * (wp / (wp - wq)))
    if len(out) < 3:
        return []
    g = _apply(Hm, out)
    o = np.array(origin if origin is not None else ground_origin(Hm, source_size), float)
    d = g - o
    r = np.hypot(d[:, 0], d[:, 1])
    k = np.where(r > max_range_m, max_range_m / np.maximum(r, 1e-9), 1.0)
    g = o + d * k[:, None]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in g]


def _bbox(poly) -> tuple[float, float, float, float]:
    a = np.asarray(poly, float)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _valid_mask(Hm, bounds, ppm, size_px, source_size, origin, max_range_m, P: Optional[np.ndarray] = None):
    """Per raster pixel: True where the ground point is in front of the camera and within range."""
    Wr, Hr = size_px
    xs = bounds[0] + (np.arange(Wr) + 0.5) / ppm
    ys = bounds[3] - (np.arange(Hr) + 0.5) / ppm
    X, Y = np.meshgrid(xs, ys)
    if P is not None:  # site -> local
        Pi = np.linalg.inv(P)
        X, Y = Pi[0, 0] * X + Pi[0, 1] * Y + Pi[0, 2], Pi[1, 0] * X + Pi[1, 1] * Y + Pi[1, 2]
    Hi = np.linalg.inv(Hm)
    w = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
    # ground points in front map to image points whose homogeneous w has the sign of the bottom-centre's
    g0 = _apply(Hm, [(source_size[0] / 2.0, source_size[1] - 1.0)])[0]
    s0 = np.sign(Hi[2] @ np.array([g0[0], g0[1], 1.0]))
    return (w * s0 > 0) & (np.hypot(X - origin[0], Y - origin[1]) <= max_range_m)


# ---------------------------------------------------------------------------------------------------------------
# Single-camera bird's-eye view
# ---------------------------------------------------------------------------------------------------------------
def birdseye(frame, H, px_per_m: float = 20, max_range_m: float = 40, scale: float = 1.0, origin=None,
             max_px: int = 2000):
    """Warp an (analytics) frame onto the ground plane. H maps SOURCE px -> ground metres; source px = frame px *
    scale. Returns (BGRA image, bounds_m (xmin, ymin, xmax, ymax), px_per_m actually used)."""
    Hm = as_h(H)
    src = (frame.shape[1] * scale, frame.shape[0] * scale)
    o = origin if origin is not None else ground_origin(Hm, src)
    poly = visible_polygon(Hm, src, max_range_m, o)
    if not poly:
        raise ValueError("no ground visible below the horizon")
    xmin, ymin, xmax, ymax = _bbox(poly)
    ppm = float(px_per_m)
    if max(xmax - xmin, ymax - ymin) * ppm > max_px:
        ppm = max_px / max(xmax - xmin, ymax - ymin)
    bounds = (xmin, ymin, xmax, ymax)
    size = (max(1, int(math.ceil((xmax - xmin) * ppm))), max(1, int(math.ceil((ymax - ymin) * ppm))))
    M = raster_matrix(bounds, ppm) @ Hm @ np.diag([scale, scale, 1.0])
    img = cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, size, flags=cv2.INTER_NEAREST)
    valid = _valid_mask(Hm, bounds, ppm, size, src, o, max_range_m) & (inside > 0)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    out[~valid, :3] = 0
    return out, tuple(round(v, 3) for v in bounds), ppm


# ---------------------------------------------------------------------------------------------------------------
# Site layout (placement)
# ---------------------------------------------------------------------------------------------------------------
def _worker_H(w):
    return as_h(getattr(w, "H", None))


def _source_size(w):
    s = getattr(w, "source_size", None)
    if s:
        return (float(s[0]), float(s[1]))
    cal = (getattr(w, "cfg", {}) or {}).get("calibration") or {}
    size = (cal.get("params") or {}).get("image_size")
    if size:
        return (float(size[0]), float(size[1]))
    fr = getattr(w, "last_frame", None)
    if fr is not None:
        sc = float(getattr(w, "scale", 1.0) or 1.0)
        return (fr.shape[1] * sc, fr.shape[0] * sc)
    pts = np.asarray(cal.get("image_points") or [[640, 480]], float)  # last resort: extent of calibration points
    return (float(max(pts[:, 0].max() * 1.05, 2)), float(max(pts[:, 1].max() * 1.05, 2)))


def _calibration(w) -> dict:
    return (getattr(w, "cfg", {}) or {}).get("calibration") or {}


_layout_cache: dict = {"key": None, "value": None}
_layout_lock = threading.Lock()


def camera_layout(site, max_range_m: float = 40.0) -> dict:
    """{cam_id: {placement, placement_source, fov_local, fov_site, origin_local, reason}} for all cameras.

    Explicit config placements are used as given. Cameras without one keep their own frame (identity) when no
    footprints overlap; otherwise they are laid side by side along +x without overlap."""
    workers = list(getattr(site, "workers", []) or [])
    key_parts = []
    for w in workers:
        Hm = _worker_H(w)
        key_parts.append((w.id, None if Hm is None else Hm.round(9).tobytes(),
                          json.dumps((getattr(w, "cfg", {}) or {}).get("placement"), sort_keys=True),
                          _source_size(w) if Hm is not None else None))
    key = (tuple(key_parts), max_range_m)
    with _layout_lock:
        if _layout_cache["key"] == key:
            return _layout_cache["value"]
    out, auto = {}, []
    for w in workers:
        Hm = _worker_H(w)
        if Hm is None:
            out[w.id] = {"placement": None, "placement_source": None, "fov_local": [], "fov_site": [],
                         "origin_local": None, "reason": "camera has no ground calibration (H) yet"}
            continue
        size = _source_size(w)
        origin = ground_origin(Hm, size, _calibration(w))
        fov = visible_polygon(Hm, size, max_range_m, origin)
        if not fov:
            out[w.id] = {"placement": None, "placement_source": None, "fov_local": [], "fov_site": [],
                         "origin_local": None, "reason": "no ground visible below the horizon"}
            continue
        explicit = (getattr(w, "cfg", {}) or {}).get("placement")
        out[w.id] = {"placement": explicit, "placement_source": "config" if explicit else None, "fov_local": fov,
                     "origin_local": list(origin), "reason": "",
                     "camera_at_origin": _calibration(w).get("method") == "auto-pedestrian"}
        if not explicit:
            auto.append(w.id)
    boxes = {cid: _bbox(v["fov_local"]) for cid, v in out.items() if v["fov_local"]}

    def overlaps(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    explicit_ids = [c for c in boxes if c not in auto]
    ids = list(boxes)
    any_overlap = any(overlaps(boxes[a], boxes[b]) for i, a in enumerate(ids) for b in ids[i + 1:])
    if not explicit_ids and not any_overlap:
        for cid in auto:
            out[cid]["placement"], out[cid]["placement_source"] = {"offset_m": [0.0, 0.0], "rotation_deg": 0.0}, "default"
    else:
        cursor = 0.0
        for cid in explicit_ids:
            sb = _bbox(_apply(placement_matrix(out[cid]["placement"]), out[cid]["fov_local"]))
            cursor = max(cursor, sb[2] + GAP_M)
        for cid in auto:
            b = boxes[cid]
            out[cid]["placement"] = {"offset_m": [round(cursor - b[0], 3), round(-b[1], 3)], "rotation_deg": 0.0}
            out[cid]["placement_source"] = "default"
            cursor += (b[2] - b[0]) + GAP_M
    for cid, v in out.items():
        if v["fov_local"]:
            v["placement"] = {"offset_m": [float(x) for x in (list(v["placement"].get("offset_m") or [0, 0]) + [0, 0])[:2]],
                              "rotation_deg": float(v["placement"].get("rotation_deg", 0.0))}
            v["fov_site"] = [[round(float(x), 3), round(float(y), 3)]
                             for x, y in _apply(placement_matrix(v["placement"]), v["fov_local"])]
            v["camera_site"] = ([round(float(c), 3) for c in _apply(placement_matrix(v["placement"]), [(0.0, 0.0)])[0]]
                                if v.get("camera_at_origin") else None)
    with _layout_lock:
        _layout_cache["key"], _layout_cache["value"] = key, out
    return out


def site_xy(worker, ground_xy, placement: Optional[dict] = None) -> tuple[float, float]:
    """Local (per-camera) ground metres -> site metres, using the camera's explicit or default placement."""
    if placement is None:
        site = getattr(worker, "site", None)
        lay = camera_layout(site).get(worker.id) if site is not None else None
        placement = (lay or {}).get("placement") or (getattr(worker, "cfg", {}) or {}).get("placement")
    x, y = _apply(placement_matrix(placement), [ground_xy])[0]
    return (float(x), float(y))


# ---------------------------------------------------------------------------------------------------------------
# Heatmap history (time-lapse): coarse per-bucket grids in SQLite
# ---------------------------------------------------------------------------------------------------------------
def regrid(grid: np.ndarray, origin, cell: float, target_origin, target_shape) -> np.ndarray:
    """Shift a grid (row 0 = lowest y) onto another cell-aligned origin/shape; cells outside are dropped."""
    out = np.zeros(target_shape)
    dx = int(round((origin[0] - target_origin[0]) / cell))
    dy = int(round((origin[1] - target_origin[1]) / cell))
    h, w = grid.shape
    r0, r1 = max(0, -dy), min(h, target_shape[0] - dy)
    c0, c1 = max(0, -dx), min(w, target_shape[1] - dx)
    if r1 > r0 and c1 > c0:
        out[r0 + dy:r1 + dy, c0 + dx:c1 + dx] = grid[r0:r1, c0:c1]
    return out


class HeatHistory:
    """One row per time bucket: the person-seconds grid accumulated in that bucket (not decayed), zlib float32."""

    def __init__(self, path: Optional[Path] = None, retention_s: float = 7 * 86400.0):
        self.retention_s = float(retention_s)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS heat_buckets(t_start REAL PRIMARY KEY, bucket_s REAL, "
                        "cell_m REAL, x0 REAL, y0 REAL, width INTEGER, height INTEGER, total REAL, peak REAL, "
                        "grid BLOB)")
        self.db.commit()

    def put(self, t_start: float, bucket_s: float, cell: float, origin, grid: np.ndarray) -> None:
        blob = zlib.compress(np.ascontiguousarray(grid, dtype=np.float32).tobytes(), 6)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO heat_buckets VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (float(t_start), float(bucket_s), float(cell), float(origin[0]), float(origin[1]),
                             int(grid.shape[1]), int(grid.shape[0]), float(grid.sum()), float(grid.max(initial=0.0)),
                             blob))
            self.db.execute("DELETE FROM heat_buckets WHERE t_start < ?", (float(t_start) - self.retention_s,))
            self.db.commit()

    def range(self, t0: float, t1: float) -> list[tuple]:
        """[(t_start, bucket_s, cell_m, (x0, y0), grid)] with t0 <= t_start < t1, oldest first."""
        with self.lock:
            rows = self.db.execute("SELECT t_start, bucket_s, cell_m, x0, y0, width, height, grid FROM heat_buckets "
                                   "WHERE t_start >= ? AND t_start < ? ORDER BY t_start", (t0, t1)).fetchall()
        return [(r[0], r[1], r[2], (r[3], r[4]),
                 np.frombuffer(zlib.decompress(r[7]), np.float32).reshape(r[6], r[5]).astype(np.float64))
                for r in rows]

    def close(self) -> None:
        with self.lock:
            self.db.close()


# ---------------------------------------------------------------------------------------------------------------
# Site map: composite background + occupancy heatmap
# ---------------------------------------------------------------------------------------------------------------
_COLORS = [(80, 200, 255), (120, 255, 120), (255, 160, 80), (200, 120, 255), (80, 255, 255), (255, 255, 120)]


class SiteMap:
    def __init__(self, site, px_per_m: float = 20.0, max_range_m: float = 40.0, refresh_s: float = 10.0,
                 cell_m: float = 0.5, half_life_s: float = 120.0, tick_s: float = 0.5, max_px: int = 2400,
                 bucket_s: Optional[float] = None, history_path: Optional[Path] = None, flush_s: float = 60.0):
        self.site, self.px_per_m, self.max_range_m, self.refresh_s = site, px_per_m, max_range_m, refresh_s
        self.cell_m, self.half_life_s, self.tick_s, self.max_px = cell_m, half_life_s, tick_s, max_px
        # time-lapse history: undecayed person-seconds per `bucket_s` (map.heat_bucket_s, default 15 min), persisted
        # to <data>/heatmap.db (in memory when the site has no data dir)
        mcfg = (getattr(site, "cfg", {}) or {}).get("map", {}) or {}
        self.bucket_s = float(bucket_s or mcfg.get("heat_bucket_s", 900.0))
        data = getattr(site, "data", None)
        self.history_db = HeatHistory(history_path or (Path(data) / "heatmap.db" if data else None),
                                      float(mcfg.get("heat_retention_days", 7)) * 86400.0)
        self.flush_s = float(flush_s)
        self._bstart: Optional[float] = None
        self._bgrid: Optional[np.ndarray] = None
        self._bbounds = None
        self._bflushed = 0.0
        self.lock = threading.Lock()
        self._bg: Optional[tuple[float, bytes, tuple]] = None  # (built_at, png, bounds)
        self._heat: Optional[np.ndarray] = None
        self._heat_bounds = None
        self._last_tick: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- layout ----------------------------------------------------------------------------------------------
    def layout(self) -> dict:
        return camera_layout(self.site, self.max_range_m)

    def bounds(self, lay: Optional[dict] = None) -> tuple[float, float, float, float]:
        lay = lay or self.layout()
        pts = [p for v in lay.values() for p in v["fov_site"] + ([v["camera_site"]] if v.get("camera_site") else [])]
        if not pts:
            m = (getattr(self.site, "cfg", {}) or {}).get("map", {}) or {}
            return (0.0, 0.0, float(m.get("width_m", 20)), float(m.get("height_m", 20)))
        xmin, ymin, xmax, ymax = _bbox(pts)
        c = self.cell_m
        return (math.floor((xmin - 1) / c) * c, math.floor((ymin - 1) / c) * c,
                math.ceil((xmax + 1) / c) * c, math.ceil((ymax + 1) / c) * c)

    def ppm(self, bounds) -> float:
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1e-6)
        return min(self.px_per_m, self.max_px / span)

    def invalidate(self) -> None:
        with self.lock:
            self._bg = None

    def info(self) -> dict:
        lay = self.layout()
        b = self.bounds(lay)
        cams = []
        for w in getattr(self.site, "workers", []):
            v = lay.get(w.id, {})
            cal = _calibration(w)
            ci = getattr(w, "calib_info", {}) or {}
            cams.append({"id": w.id, "name": getattr(w, "name", w.id), "placement": v.get("placement"),
                         "placement_source": v.get("placement_source"), "fov_polygon_m": v.get("fov_site", []),
                         "fov_polygon_local_m": v.get("fov_local", []),
                         "camera_xy_m": v.get("camera_site"),
                         "calibrated": bool(v.get("fov_local")), "reason": v.get("reason", ""),
                         "calib_error_m": ci.get("calib_error_m", cal.get("calib_error_m")),
                         "method": cal.get("method", "manual") if cal else None})
        with self.lock:
            built = self._bg[0] if self._bg else None
        return {"bounds_m": [round(v, 3) for v in b], "px_per_m": round(self.ppm(b), 4), "cameras": cams,
                "background_url": "/api/map/background.png", "heatmap_url": "/api/map/heatmap",
                "raster": "north-up: column=(x-xmin)*px_per_m, row=(ymax-y)*px_per_m",
                "model": "2D ground-plane homography (not a 3D model); off-ground objects are smeared",
                "updated_at": built}

    # -- background ------------------------------------------------------------------------------------------
    def background_png(self) -> tuple[bytes, tuple]:
        now = time.time()
        with self.lock:
            bg = self._bg
        lay = self.layout()
        b = self.bounds(lay)
        if bg and now - bg[0] < self.refresh_s and bg[2] == b:
            return bg[1], b
        png = self._render(lay, b)
        with self.lock:
            self._bg = (now, png, b)
        return png, b

    def _render(self, lay: dict, b) -> bytes:
        ppm = self.ppm(b)
        Wr, Hr = max(1, int(math.ceil((b[2] - b[0]) * ppm))), max(1, int(math.ceil((b[3] - b[1]) * ppm)))
        canvas = np.full((Hr, Wr, 3), (28, 24, 22), np.uint8)
        T = raster_matrix(b, ppm)
        for w in getattr(self.site, "workers", []):
            v = lay.get(w.id) or {}
            fr = getattr(w, "last_frame", None)
            Hm = _worker_H(w)
            if fr is None or Hm is None or not v.get("fov_local"):
                continue
            scale = float(getattr(w, "scale", 1.0) or 1.0)
            P = placement_matrix(v["placement"])
            M = T @ P @ Hm @ np.diag([scale, scale, 1.0])
            img = cv2.warpPerspective(fr, M, (Wr, Hr), flags=cv2.INTER_LINEAR)
            inside = cv2.warpPerspective(np.full(fr.shape[:2], 255, np.uint8), M, (Wr, Hr), flags=cv2.INTER_NEAREST)
            valid = _valid_mask(Hm, b, ppm, (Wr, Hr), _source_size(w), v["origin_local"], self.max_range_m, P) & (inside > 0)
            canvas[valid] = img[valid]
        self._grid(canvas, b, ppm)
        for i, w in enumerate(getattr(self.site, "workers", [])):
            v = lay.get(w.id) or {}
            if not v.get("fov_site"):
                continue
            col = _COLORS[i % len(_COLORS)]
            pts = _apply(T, v["fov_site"]).round().astype(np.int32)
            cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
            err = (getattr(w, "calib_info", {}) or {}).get("calib_error_m")
            label = f"{w.id}" + (f" +/-{err:.1f} m" if isinstance(err, (int, float)) else "")
            if v.get("camera_at_origin"):
                cxy = _apply(T, [site_xy(w, (0.0, 0.0), v["placement"])])[0].round().astype(int)
                cv2.circle(canvas, tuple(int(c) for c in cxy), 6, col, -1, cv2.LINE_AA)
                anchor = (int(cxy[0]) + 8, int(cxy[1]) - 8)
            else:
                low = pts[pts[:, 1] >= pts[:, 1].max() - 2]  # bottom edge of the footprint
                anchor = (int(low[:, 0].mean()) - 30, int(pts[:, 1].max()) + 16)
            anchor = (min(max(anchor[0], 2), Wr - 2), min(max(anchor[1], 14), Hr - 2))
            cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return cv2.imencode(".png", canvas)[1].tobytes()

    @staticmethod
    def _grid(canvas, b, ppm) -> None:
        Hr, Wr = canvas.shape[:2]
        minor = 1.0 if ppm >= 8 else 5.0
        overlay = canvas.copy()
        for x in np.arange(math.ceil(b[0] / minor) * minor, b[2] + 1e-9, minor):
            major = abs(x / 5 - round(x / 5)) < 1e-6
            c = int(round((x - b[0]) * ppm))
            cv2.line(overlay, (c, 0), (c, Hr - 1), (150, 150, 150) if major else (90, 90, 90), 1)
            if major and abs(x / 10 - round(x / 10)) < 1e-6:
                cv2.putText(canvas, f"{x:g}", (c + 2, Hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        for y in np.arange(math.ceil(b[1] / minor) * minor, b[3] + 1e-9, minor):
            major = abs(y / 5 - round(y / 5)) < 1e-6
            r = int(round((b[3] - y) * ppm))
            cv2.line(overlay, (0, r), (Wr - 1, r), (150, 150, 150) if major else (90, 90, 90), 1)
            if major and abs(y / 10 - round(y / 10)) < 1e-6:
                cv2.putText(canvas, f"{y:g}", (2, r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)

    # -- heatmap ---------------------------------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> None:
        """Decay the grid and add the current person foot points (person-seconds, exponentially decayed)."""
        now = time.time() if now is None else now
        lay = self.layout()
        b = self.bounds(lay)
        c = self.cell_m
        nx, ny = max(1, int(round((b[2] - b[0]) / c))), max(1, int(round((b[3] - b[1]) / c)))
        with self.lock:
            if self._heat is None or self._heat_bounds != b:
                self._heat, self._heat_bounds, self._last_tick = np.zeros((ny, nx)), b, None
            dt = 0.0 if self._last_tick is None else max(0.0, min(now - self._last_tick, 5.0))
            self._last_tick = now
            if dt > 0:
                self._heat *= 0.5 ** (dt / self.half_life_s)
            weight = dt if dt > 0 else self.tick_s
            self._roll(now, b, (ny, nx))
            for w in getattr(self.site, "workers", []):
                v = lay.get(w.id) or {}
                Hm = _worker_H(w)
                dets = getattr(w, "last_dets", None)
                if Hm is None or not dets or not v.get("fov_local"):
                    continue
                scale = float(getattr(w, "scale", 1.0) or 1.0)
                feet = [((bb[0] + bb[2]) / 2 * scale, bb[3] * scale) for _, bb, cls, _ in dets if cls == "person"]
                if not feet:
                    continue
                s0 = _front_sign(Hm, _source_size(w))
                q = np.column_stack([np.asarray(feet), np.ones(len(feet))]) @ Hm.T
                ok = q[:, 2] * s0 > 1e-12
                if not ok.any():
                    continue
                g = q[ok, :2] / q[ok, 2:3]
                o = np.asarray(v["origin_local"], float)
                g = g[np.hypot(g[:, 0] - o[0], g[:, 1] - o[1]) <= self.max_range_m]
                if not len(g):
                    continue
                sg = _apply(placement_matrix(v["placement"]), g)
                ix = np.floor((sg[:, 0] - b[0]) / c).astype(int)
                iy = np.floor((sg[:, 1] - b[1]) / c).astype(int)
                m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                np.add.at(self._heat, (iy[m], ix[m]), weight)
                np.add.at(self._bgrid, (iy[m], ix[m]), weight)
            if now - self._bflushed >= self.flush_s:  # persist the running bucket now and then (survives a restart)
                self._flush(now)

    def _roll(self, now: float, b, shape) -> None:
        """Start a new history bucket at bucket boundaries (persisting the finished one); follow bounds changes."""
        start = math.floor(now / self.bucket_s) * self.bucket_s
        if self._bstart is not None and start != self._bstart:
            self._flush(now)
            self._bstart = None
        if self._bstart is None:
            self._bstart, self._bgrid, self._bbounds = start, np.zeros(shape), b
            self._bflushed = now
        elif self._bbounds != b:
            self._bgrid = regrid(self._bgrid, self._bbounds[:2], self.cell_m, b[:2], shape)
            self._bbounds = b

    def _flush(self, now: float) -> None:
        if self._bstart is not None and self._bgrid is not None:
            self.history_db.put(self._bstart, self.bucket_s, self.cell_m, self._bbounds[:2], self._bgrid)
        self._bflushed = now

    def history(self, hours: float = 24.0, step_s: float = 3600.0, now: Optional[float] = None) -> dict:
        """Time-lapse frames for the last `hours`: person-seconds per cell per `step_s` (a multiple of the stored
        bucket), re-gridded onto the current map bounds. Cells are sparse [index, value], row 0 = lowest y."""
        now = time.time() if now is None else now
        step = max(1, round(float(step_s) / self.bucket_s)) * self.bucket_s
        n = max(1, min(800, int(math.ceil(float(hours) * 3600.0 / step))))
        end = (math.floor(now / step) + 1) * step
        start = end - n * step
        with self.lock:
            b = self._heat_bounds or self.bounds()
            live = (self._bstart, self._bgrid.copy(), self._bbounds) if self._bgrid is not None else None
        c = self.cell_m
        nx, ny = max(1, int(round((b[2] - b[0]) / c))), max(1, int(round((b[3] - b[1]) / c)))
        rows = {r[0]: r for r in self.history_db.range(start, end)}
        if live and start <= live[0] < end:  # the running bucket is newer than its last flush
            rows[live[0]] = (live[0], self.bucket_s, c, live[2][:2], live[1])
        frames = [np.zeros((ny, nx)) for _ in range(n)]
        for t0, _, cell, origin, grid in rows.values():
            if abs(cell - c) > 1e-9:
                continue  # stored with another cell size: not comparable
            frames[min(n - 1, int((t0 - start) // step))] += regrid(grid, origin, c, b[:2], (ny, nx))
        peak = max((float(f.max()) for f in frames), default=0.0)
        out = []
        for i, f in enumerate(frames):
            nz = np.flatnonzero(f > 1e-6)
            out.append({"t_start": start + i * step, "t_end": start + (i + 1) * step, "total": round(float(f.sum()), 2),
                        "max": round(float(f.max()), 3),
                        "cells": [[int(k), round(float(f.flat[k]), 3)] for k in nz]})
        return {"cell_m": c, "origin_m": [b[0], b[1]], "width": nx, "height": ny, "step_s": step,
                "bucket_s": self.bucket_s, "from": start, "to": end, "now": now, "max": round(peak, 3),
                "frames": out, "row_order": "cell index = row * width + column; row 0 = y at origin_m[1]",
                "units": f"person-seconds per {c:g} m cell within each frame (not decayed)",
                "note": "Accumulated while the server runs (nothing is recorded while it is stopped). Cameras with "
                        "overlapping views count the same person more than once. Occupancy, not risk."}

    def heatmap(self) -> dict:
        with self.lock:
            if self._heat is None:
                b = self.bounds()
                nx = max(1, int(round((b[2] - b[0]) / self.cell_m)))
                ny = max(1, int(round((b[3] - b[1]) / self.cell_m)))
                grid = np.zeros((ny, nx))
            else:
                b, grid = self._heat_bounds, self._heat.copy()
        return {"cell_m": self.cell_m, "origin_m": [b[0], b[1]], "width": int(grid.shape[1]),
                "height": int(grid.shape[0]), "values": [round(float(v), 3) for v in grid.ravel()],
                "max": round(float(grid.max()), 3) if grid.size else 0.0,
                "row_order": "row 0 = y at origin_m[1] (ascending y); column 0 = x at origin_m[0]",
                "units": f"person-seconds, exponentially decayed (half-life {self.half_life_s:g} s)",
                "note": "cameras with overlapping views count the same person more than once"}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:  # never let a heatmap glitch kill the thread
                    pass
                self._stop.wait(self.tick_s)
        self._thread = threading.Thread(target=loop, daemon=True, name="sitemap-heat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self.lock:
            self._flush(time.time())
