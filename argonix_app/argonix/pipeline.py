"""Per-camera analytics workers and the Site that wires detector -> tracker -> ground plane -> rules ->
correlation -> evidence -> store.

Live sources (RTSP/HTTP/webcam) are read by a LatestFrameGrabber thread that keeps only the newest frame, so
analytics never lag behind a buffered stream, and reconnects with exponential backoff. Calibration is optional:
an uncalibrated camera still runs detection, tracking, health and recording, and evaluates the non-metric rules
(intrusion, loitering, crowding) on zones drawn in image pixels. It produces no ground/map positions.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import re
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .analytics_install import detector_config, install as install_analytics
from .cameras import is_live, is_redacted, open_capture, redact
from .correlate import Correlator
from .flow import parse_line_events
from .detect import TrackerBackend, make_backend_factory
from .baseline import Baseline, hour_of_week
from .geometry import (apply_h, fit_homography, foot_point, invert_h, point_in_polygon, rms_error,
                       signed_depth)
from .media import OVERLAY_LABEL, H264Writer, draw, stamp, write_clip
from .health import FAULTS, HealthMonitor, apply_fault
from .imaging import Conditioner
from .models import Event, TrackObs
from .rules import RuleEngine, Zone, vehicle_classes
from .store import Store
from .track_record import TrackRecord
from .workflow import OPERATOR_FIELDS

RULES_VERSION = "rules-v1"
# rules that make sense without a metric ground plane
PIXEL_RULES = ("intrusion", "loitering", "crowding", "queue_long", "vehicle_in_zone", "vehicle_stopped")


def _px_rule_ok(name: str, params) -> bool:
    """Rules usable on an uncalibrated camera; running opts in with an image-space speed (speed_bh_s, rules.py)."""
    return name in PIXEL_RULES or (name == "running" and isinstance(params, dict) and "speed_bh_s" in params)


CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")



# A burst of events (crowding, a busy door) used to start one ffmpeg encoder per clip at once and spike the CPU the
# analytics need. Clips still get a thread each (the frames are already in memory), but only two encode at a time.
_CLIP_ENCODERS = threading.BoundedSemaphore(2)


def _write_clip_bounded(path, frames, fps, label) -> None:
    with _CLIP_ENCODERS:
        write_clip(path, frames, fps, label)


class _NoBackend:
    """Stands in for a detector while a camera config is only being validated (never runs a frame)."""
    info = {"detector": "none", "tracker": "none"}

    def track(self, frame):  # pragma: no cover - never called
        raise RuntimeError("validation-only backend")


_NO_BACKEND = _NoBackend()

class LatestFrameGrabber:
    """Reads a live source continuously and keeps only the newest frame (no queue, so no lag build-up).

    On open failure or stream loss it reconnects with exponential backoff (backoff_min -> backoff_max, reset once
    frames flow again). `connected` is False while down; `attempts` counts open attempts.
    """

    def __init__(self, source, name: str, backoff_min: float = 1.0, backoff_max: float = 30.0,
                 opener: Optional[Callable] = None):
        self.source, self.opener = source, opener
        self.backoff = (float(backoff_min), float(backoff_max))
        self._cond = threading.Condition()
        self._frame = None
        self.seq, self.t = 0, 0.0
        self.connected = False
        self.attempts = 0
        self.connections = 0
        self.error = ""
        self.src_fps: Optional[float] = None
        self.delays: deque = deque(maxlen=32)  # recent backoff waits (for tests / diagnostics)
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"grab-{name}")

    @property
    def reconnects(self) -> int:
        return max(0, self.connections - 1)

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self.stop_evt.set()
        with self._cond:
            self._cond.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout)

    def latest(self):
        with self._cond:
            return (self._frame, self.seq, self.t) if self._frame is not None else None

    def wait_newer(self, last_seq: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self.seq > last_seq or self.stop_evt.is_set(), timeout)
            return (self._frame, self.seq, self.t) if self.seq > last_seq else None

    def _run(self) -> None:
        delay = self.backoff[0]
        while not self.stop_evt.is_set():
            self.attempts += 1
            cap, got = None, 0
            try:
                cap = (self.opener or open_capture)(self.source)
                if cap is not None and cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    self.src_fps = float(fps) if 1 <= fps <= 120 else None
                    while not self.stop_evt.is_set():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        if not got:
                            self.connections += 1
                        got += 1
                        with self._cond:
                            self._frame, self.seq, self.t = frame, self.seq + 1, time.time()
                            self.connected, self.error = True, ""
                            self._cond.notify_all()
                    self.error = "stream ended or read timed out" if got else "opened but no frames received"
                else:
                    self.error = "cannot open source"
            except Exception as e:  # noqa: BLE001 - a broken backend must not kill the reconnect loop
                self.error = redact(f"{type(e).__name__}: {e}")
            finally:
                if cap is not None:
                    cap.release()
            self.connected = False
            if self.stop_evt.is_set():
                break
            if got:
                delay = self.backoff[0]
            self.delays.append(delay)
            self.stop_evt.wait(delay)
            delay = min(delay * 2, self.backoff[1])


class CameraWorker(threading.Thread):
    def _ground_zone(self, z: dict) -> dict:
        """Project an editor-drawn zone (polygon_px, direction_px arrow in source px) onto the ground plane."""
        z = dict(z)
        if "polygon_px" in z:
            z["polygon"] = [list(apply_h(self.H, tuple(p))) for p in z["polygon_px"]]
        if z.get("direction_px") and not z.get("direction"):
            (x0, y0), (x1, y1) = (apply_h(self.H, tuple(p)) for p in z["direction_px"])
            z["direction"] = [x1 - x0, y1 - y0]
        return z

    def __init__(self, cfg: dict, site: "Site", backend: TrackerBackend):
        super().__init__(daemon=True, name=f"cam-{cfg['id']}")
        self.cfg, self.site, self.backend = cfg, site, backend
        self.id, self.name = cfg["id"], cfg.get("name", cfg["id"])
        cal = cfg.get("calibration")
        self.H = None
        self.px_zones: list[Zone] = []  # zones evaluated in source-image pixels (uncalibrated cameras only)
        # vehicle labels (site `vehicles`): only the vehicle rules see these tracks, never person counts/crowding/flow
        self.vehicle_labels = tuple(vehicle_classes(getattr(site, "cfg", {}) or {}).values())
        # counting-line events (argonix/flow.py): validated here so a bad config/API edit fails with a clear error
        for i, ln in enumerate(cfg.get("lines", []) or []):
            parse_line_events(ln.get("events"), str(ln.get("id") or f"line_{i + 1}"))
        if cal:
            # 4 points: exact; 5+: RANSAC + least squares with outliers and a leave-one-out error (geometry.py)
            self.H, fit = fit_homography([tuple(p) for p in cal["image_points"]],
                                         [tuple(p) for p in cal["ground_points"]],
                                         robust_threshold_m=float(cal.get("robust_threshold_m", 0.5)))
            chk = cal.get("check_points") or {}
            measured = rms_error(self.H, [tuple(p) for p in chk.get("image", [])],
                                 [tuple(p) for p in chk.get("ground", [])])
            # the stated error never undercuts what was measured: held-out check points or leave-one-out
            self.calib_error = max(float(cal.get("calib_error_m", 0.3)), measured, fit["loo_rmse_m"] or 0.0)
            self.calib_info = {"calibrated": True, "calib_error_m": round(self.calib_error, 3),
                               "measured_rms_m": round(measured, 3), "check_points": len(chk.get("image", [])),
                               "fit": fit}
            # zones may be drawn in image pixels (polygon_px, as an operator would) or given in ground metres
            self.zones = [Zone.from_dict(self._ground_zone(z)) for z in cfg.get("zones", [])]
            self.engine = RuleEngine(self.id, self.zones, calib_error_m=self.calib_error,
                                     vehicle_classes=self.vehicle_labels)
        else:
            self.calib_error = None
            self.zones = []  # ground-plane zones: none without calibration (nothing goes on the map)
            margin = float(cfg.get("zone_margin_px", 8.0))
            skipped = []
            for z in cfg.get("zones", []):
                zid = z.get("id", "?")
                if "polygon_px" not in z:
                    skipped.append(f"zone '{zid}': ground-metre polygon needs calibration")
                    continue
                rules = {k: v for k, v in (z.get("rules") or {}).items() if _px_rule_ok(k, v)}
                skipped += [f"zone '{zid}': rule '{k}' needs calibration "
                            f"({'metric area' if k == 'density' else 'metric speed/direction'})"
                            for k, v in (z.get("rules") or {}).items() if not _px_rule_ok(k, v)]
                self.px_zones.append(Zone.from_dict({**z, "polygon": z["polygon_px"], "rules": rules}))
            self.engine = RuleEngine(self.id, self.px_zones, calib_error_m=margin, units="px",
                                     vehicle_classes=self.vehicle_labels)
            self.calib_info = {"calibrated": False, "calib_error_m": None, "measured_rms_m": None, "check_points": 0,
                               "zone_space": "image_px", "zone_margin_px": margin, "skipped": skipped,
                               "note": "Uncalibrated: no ground/map positions; intrusion, loitering and crowding "
                                       "use image-pixel zones; running and wrong_way need calibration."}
        self._dwell_cache: dict[tuple, tuple[float, tuple]] = {}
        self.engine.dwell_baseline = self._dwell_baseline  # loitering rules with `baseline_quantile`
        # Frigate-style detection masks (source-image pixels): detections whose foot point falls inside are ignored
        self.masks = [[tuple(p) for p in m] for m in cfg.get("masks_px", [])]
        self.masked = 0
        # Privacy masks (source-image pixels): painted black on every frame right after grab/conditioning, so those
        # pixels never reach detection, recording, snapshots, clips, live view, the VLM or exports.
        self.privacy_masks = [np.asarray(m, dtype=np.float64).reshape(-1, 2) for m in cfg.get("privacy_masks_px", [])]
        self._privacy_cache: tuple = (None, [])
        self.stop_evt = threading.Event()
        self.latest_jpeg: Optional[bytes] = None
        self._ev_seq = itertools.count(1)  # several events on one frame must not share an evidence file
        self.tracks: list[dict] = []
        # Read by other components (e.g. auto-calibration): latest analytics frame + detections on it.
        self._last_lock = threading.Lock()
        self.last_frame = None            # latest analytics-resolution BGR frame (unannotated, what the detector saw)
        self.last_dets: list = []         # [(track_id, (x1, y1, x2, y2) in analytics px, cls, conf)]
        self.last_t: Optional[float] = None
        self.scale = 1.0                  # source px = analytics px * scale
        self.source_size: Optional[tuple[int, int]] = None  # (w, h) of the source frames
        self.grabber: Optional[LatestFrameGrabber] = None
        self.stats = {"fps": 0.0, "detect_ms": 0.0, "frames": 0, "events": 0, "dropped": 0, "status": "starting",
                      "error": "", "calibrated": self.H is not None}
        self.done = threading.Event()
        self.conditioner = Conditioner(cfg.get("enhance", "auto"))
        self.health = HealthMonitor(**cfg.get("health", {}))
        self.health_criticality = int(cfg.get("criticality", 2))
        self._next_health = 0.0
        self._pending_health: list[Event] = []
        self._health_on: dict[str, float] = {}  # active health condition -> when it was raised (for recovery)
        self.fault: Optional[tuple[str, float]] = None  # (fault, until_wall_time) for demo tamper injection
        # (scenario name, until_wall_time): health events raised before then are labelled as a demo scenario's
        self.demo_tag: Optional[tuple[str, float]] = None
        self._frozen = None
        self._began = False
        self._ext_evidence: deque = deque()  # evidence stems requested by non-video sources (request_evidence)

    def inject(self, fault: str, seconds: float) -> None:
        if fault not in FAULTS:
            raise ValueError(f"fault must be one of {FAULTS}")
        self.fault = (fault, time.time() + seconds)
        self._frozen = None

    def request_evidence(self, stem: str) -> Optional[dict]:
        """Thread-safe: ask this camera for a snapshot + pre/post clip around *now*, for an event that did not come
        from video (e.g. an access-control door event). The files are written by the worker on its next frame, so the
        returned paths appear a moment later (the clip after `clip_post_s`). None when the worker is not running."""
        if not (self._began and self.is_alive()):
            return None
        self._ext_evidence.append(stem)
        return {"snapshot": f"evidence/{stem}.jpg", "snapshot_raw": f"evidence/{stem}_raw.jpg", "clip": f"evidence/{stem}.mp4",
                "frame_time": time.time(),
                "camera": self.id, "calibration": self.calib_info}

    def _capture(self, stem: str, stamped, raw) -> tuple[str, str, str]:
        """Write the snapshot pair now (annotated + 'software overlay' stamp, and the raw frame as analysed) and queue
        a clip (pre-buffer + `clip_post_s` of frames still to come)."""
        snap_rel, raw_rel, clip_rel = f"evidence/{stem}.jpg", f"evidence/{stem}_raw.jpg", f"evidence/{stem}.mp4"
        (self.site.data / "evidence").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.site.data / snap_rel), stamped)
        # the frame exactly as analysed (analytics resolution, after B/W conditioning), with nothing drawn on it
        cv2.imwrite(str(self.site.data / raw_rel), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self._pending.append({"path": self.site.data / clip_rel, "frames": list(self._pre), "left": self._post_n})
        return snap_rel, raw_rel, clip_rel

    def clear_fault(self) -> None:
        """End an injected tamper fault now (demo recovery)."""
        self.fault = None
        self._frozen = None

    def _privacy(self, small, scale: float):
        """Fill the privacy polygons (source px) black on the analytics frame, in place (copied if shared)."""
        key = (small.shape[1], small.shape[0], scale)
        if self._privacy_cache[0] != key:
            self._privacy_cache = (key, [np.round(m / scale).astype(np.int32) for m in self.privacy_masks])
        cv2.fillPoly(small, self._privacy_cache[1], (0, 0, 0))
        return small

    def snapshot(self):
        """Consistent (frame, dets, t, scale) of the latest processed analytics frame, or None."""
        with self._last_lock:
            if self.last_frame is None:
                return None
            return self.last_frame, list(self.last_dets), self.last_t, self.scale

    def stop(self, timeout: float = 30.0) -> None:
        """Stop and join the worker (its grabber is stopped and its recorder/clips are flushed on the way out)."""
        self.stop_evt.set()
        if self.is_alive():
            self.join(timeout)

    def _health_step(self, t: float, frame, transitions=None) -> None:
        trans = transitions if transitions is not None else self.health.update(t, frame)
        self.stats.update(health=self.health.state, health_reasons=sorted(self.health.active),
                          health_metrics=self.health.metrics)
        for cond, active in trans:
            if not active:
                since = self._health_on.pop(cond, None)
                if since is not None:
                    try:
                        self.site.on_health_recovered(self.id, cond, t, since)
                    except Exception:  # noqa: BLE001 - bookkeeping must never stop the camera loop
                        pass
                continue
            self._health_on[cond] = t
            der = {"rule": "camera_health", "condition": cond, "metrics": self.health.metrics,
                   "threshold": {"persist_s": self.health.persist_s},
                   "confidence_method": "deterministic image statistics vs warm-up reference", "duration_s": 0,
                   "calib_error_m": self.calib_error}
            obs = f"Camera view condition '{cond}' detected on {self.id}"
            tag = self.demo_tag
            if tag and time.time() < tag[1]:  # raised by a demo scenario's scripted tamper injection
                der.update(demo_scenario=True, scenario=tag[0], injected_fault=True)
                obs = f"[DEMO] {obs} (fault injected by demo scenario '{tag[0]}')"
            e = Event("camera_health", self.id, "_view", f"{self.name} (whole view)", [], t, t, obs, der,
                      0.9, 2, self.health_criticality, None, None)
            self._pending_health.append(e)

    def _dwell_baseline(self, zone_id: str, t: float, q: float) -> tuple:
        """Site dwell baseline for this camera's zone in the hour of week of t (cached 5 min per zone/hour)."""
        k = (zone_id, hour_of_week(t), q)
        hit = self._dwell_cache.get(k)
        if hit and 0 <= t - hit[0] < 300:
            return hit[1]
        r = self.site.baseline.dwell_quantile(self.id, zone_id, k[1], q)
        self._dwell_cache[k] = (t, r)
        return r

    def _emit_down(self, now: float) -> None:
        self._pending_health = []
        self._health_step(now, None, self.health.mark_down(now))
        for e in self._pending_health:
            e.evidence = {"snapshot": None, "clip": None, "frame_time": e.t_end}
            self.site.on_event(e)
        self._pending_health = []

    # ------------------------------------------------------------------------------------------
    def _zones_px(self, scale: float):
        out = []
        if self.H is not None:
            hinv = invert_h(self.H)
            for z in self.zones:
                try:
                    out.append((z.name, [(x / scale, y / scale) for x, y in (apply_h(hinv, p) for p in z.polygon)]))
                except ValueError:
                    pass
        out += [(z.name, [(x / scale, y / scale) for x, y in z.polygon]) for z in self.px_zones]
        for ln in self.cfg.get("lines", []) or []:  # counting lines (argonix/flow.py), drawn as 2-point shapes
            try:
                pts = ln.get("points_px") or ([apply_h(invert_h(self.H), tuple(p)) for p in ln["points"]]
                                              if ln.get("points") and self.H is not None else None)
                if pts:
                    out.append((f"line: {ln.get('name', ln.get('id', ''))}", [(x / scale, y / scale) for x, y in pts]))
            except (ValueError, KeyError, TypeError):
                pass
        return out

    def run(self) -> None:
        """Run the camera. In live (realtime) operation a processing error restarts it with backoff.

        One exception anywhere in frame processing used to end the thread for good, leaving the camera in "error"
        until someone edited it. Batch/test runs (realtime=False) still stop at the first error: a restart would
        reprocess a file from its start."""
        lo = float(self.cfg.get("restart_min_s", 2.0))
        hi = max(lo, float(self.cfg.get("restart_max_s", 60.0)))
        delay = lo
        try:
            while not self.stop_evt.is_set():
                started = time.time()
                try:
                    self._run()
                    return  # the source ended normally, or stop was requested
                except Exception as e:  # keep the site alive if one camera fails
                    self.stats.update(status="error", error=redact(repr(e)))
                    if not self.site.realtime:
                        return
                if time.time() - started > hi:  # it had been running fine for a while: restart promptly
                    delay = lo
                self.stats["restarts"] = self.stats.get("restarts", 0) + 1
                self.stats["status"] = "restarting"
                if self.stop_evt.wait(delay):
                    return
                delay = min(delay * 2, hi)
        finally:
            close = getattr(self.backend, "close", None)  # per-camera resources: remote socket, batched tracker slot
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            self.done.set()

    def _run(self) -> None:
        src = str(self.cfg["source"])
        if is_live(src):
            self._run_live(src)
        else:
            self._run_file(src)

    def _run_file(self, src: str) -> None:
        cap = open_capture(src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source {redact(src)}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if src_fps > 120 or src_fps < 1:
            src_fps = 25.0
        afps = float(self.site.cfg.get("analytics_fps", 5))
        step = max(1, round(src_fps / afps))
        afps = src_fps / step
        realtime, loop = self.site.realtime, self.cfg.get("loop", True) and self.site.realtime
        start_wall, n = time.time(), 0
        self._begin(afps)
        try:
            while not self.stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    if loop and n > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                n += 1
                t = start_wall + n / src_fps
                if realtime:
                    delay = t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                if n % step:
                    continue
                if realtime and time.time() - t > 2.0 / afps:  # behind real time: drop, stay live
                    self.stats["dropped"] += 1
                    continue
                self._process(frame, t)
        finally:
            cap.release()
            self._end()

    def _run_live(self, src: str) -> None:
        afps = float(self.site.cfg.get("analytics_fps", 5))
        period = 1.0 / afps
        capture = self.cfg.get("capture")  # webcam mode request, e.g. {"width": 1280, "height": 720, "fourcc": "MJPG"}
        g = self.grabber = LatestFrameGrabber(src, self.id, float(self.cfg.get("reconnect_min_s", 1.0)),
                                              float(self.cfg.get("reconnect_max_s", 30.0)),
                                              opener=(lambda s: open_capture(s, capture)) if capture else None)
        # a stream that stays without frames this long is declared down (a sub-second reconnect is not an outage)
        down_grace = float(self.cfg.get("down_grace_s", 2.0))
        started = time.time()
        g.start()
        self._begin(afps)
        self.stats["status"] = "connecting"
        last_seq, next_due, last_down = 0, 0.0, 0.0
        try:
            while not self.stop_evt.is_set():
                self.stats.update(reconnects=g.reconnects, source_fps=g.src_fps)
                if not g.connected and g.attempts > 0 and g.error:
                    self.stats.update(status="reconnecting", error=g.error)
                    now = time.time()
                    if now - max(g.t, started) >= down_grace and now - last_down >= 1.0:
                        last_down = now
                        self._emit_down(now)
                got = g.wait_newer(last_seq, timeout=0.5)
                if got is None:
                    continue
                wait = next_due - time.time()
                if wait > 0:  # pace analytics to analytics_fps, then take the freshest frame
                    if self.stop_evt.wait(wait):
                        break
                    got = g.latest() or got
                frame, seq, t = got
                if last_seq and g.src_fps:  # frames skipped beyond the intended decimation = analytics too slow
                    expected = max(0, round(g.src_fps / afps) - 1)
                    self.stats["dropped"] += max(0, seq - last_seq - 1 - expected)
                last_seq = seq
                next_due = time.time() + period
                self.stats.update(status="running", error="")
                self._process(frame, t)
        finally:
            g.stop()
            self._end()

    # ------------------------------------------------------------------------------------------
    def _begin(self, afps: float) -> None:
        c = self.site.cfg
        self.afps = afps
        self._pre = deque(maxlen=max(1, int(c.get("clip_pre_s", 4) * afps)))
        self._post_n = int(c.get("clip_post_s", 4) * afps)
        self._pending: list[dict] = []
        self._seg_len = float(c.get("segment_s", 60))
        self._max_w = int(c.get("analytics_width", 960))
        self._rec: Optional[H264Writer] = None
        self._rec_start, self._rec_rel, self._rec_retry_at = 0.0, "", 0.0
        self._ema_fps, self._last_t, self._wall_last = 0.0, None, time.time()
        self._zones_px_cache: tuple = (None, [])
        self._began = True
        self.stats["status"] = "running"

    def _end(self) -> None:
        if self._began:
            for p in self._pending:
                write_clip(p["path"], p["frames"], self.afps, OVERLAY_LABEL)
            self._pending = []
            if self._rec:
                self._close_recorder(self._last_t or self._rec_start)
        self.stats["status"] = "stopped"

    def _close_recorder(self, t_end: float) -> None:
        rec, self._rec = self._rec, None
        try:
            rec.close()
            self.site.store.add_recording(self.id, self._rec_rel, self._rec_start, t_end)
        except Exception as e:  # noqa: BLE001 - a broken segment must not take the camera down
            self.stats["recording_error"] = redact(repr(e))[:200]

    def _process(self, frame, t: float) -> None:
        realtime = self.site.realtime
        h0, w0 = frame.shape[:2]
        scale = max(1.0, w0 / self._max_w)
        if self._zones_px_cache[0] != (w0, h0):
            self._zones_px_cache = ((w0, h0), self._zones_px(scale))
        zones_px = self._zones_px_cache[1]
        small = frame if scale == 1.0 else cv2.resize(frame, (int(w0 / scale), int(h0 / scale)))
        small = self.conditioner(small)  # grey/IR -> 3-channel, monochrome flag, optional CLAHE (imaging.py)
        if self.privacy_masks:
            small = self._privacy(small.copy() if small is frame else small, scale)
        self.stats.update(monochrome=self.conditioner.monochrome, enhanced=self.conditioner.enhanced)
        if self.fault:
            if time.time() > self.fault[1]:
                self.fault = None
            else:
                if self._frozen is None:
                    self._frozen = small.copy()
                small = apply_fault(small, self.fault[0], self._frozen)
        self._pending_health = []
        if t >= self._next_health:
            self._next_health = t + 1.0
            self._health_step(t, small)

        t0 = time.perf_counter()
        dets = self.backend.track(small)
        dt_ms = (time.perf_counter() - t0) * 1000
        backend_stats = getattr(self.backend, "camera_stats", None)  # e.g. remote_state / rtt_ms / dropped_remote
        if backend_stats is not None:
            self.stats.update(backend_stats())
        if self.masks:
            kept = [d for d in dets if not any(point_in_polygon(
                (foot_point(d[1])[0] * scale, foot_point(d[1])[1] * scale), m) for m in self.masks)]
            self.masked += len(dets) - len(kept)
            self.stats["masked"] = self.masked
            dets = kept
        with self._last_lock:
            self.last_frame, self.last_dets, self.last_t = small, list(dets), t
            self.scale, self.source_size = scale, (w0, h0)

        obs, eobs = [], []  # obs: ground-plane positions (map); eobs: positions in the rule engine's space
        for tid, bb, cls, conf in dets:
            fx, fy = foot_point(bb)
            fpx = (fx * scale, fy * scale)
            g = None
            if self.H is not None:
                try:
                    g = apply_h(self.H, fpx)
                except ValueError:
                    g = None
            obs.append(TrackObs(self.id, tid, t, bb, cls, conf, g))
            eobs.append(obs[-1] if self.H is not None else TrackObs(self.id, tid, t, bb, cls, conf, fpx))
        events = self.engine.update(t, eobs) + self._pending_health
        for an in self.site.analyzers:  # pluggable analyzers (handoff prediction, object/altercation models, ...)
            try:
                events += an.on_frame(self, t, small, dets, obs) or []
            except Exception as e:  # an analyzer must never take the camera down
                self.stats[f"analyzer_error:{type(an).__name__}"] = repr(e)[:200]
        for z in self.engine.zones:
            n_in = sum(1 for o in eobs if o.cls == "person" and o.ground and signed_depth(o.ground, z.polygon) >= 0)
            for dev in self.site.baseline.observe_all(self.id, z.id, z.name, t, n_in, z.criticality):
                dev.site_zone = z.site_zone
                events.append(dev)

        alert_tids = {k[2] for k, ep in self.engine.eps.items() if len(k) == 3 and ep.fired}
        ann = draw(small, dets, zones_px, alert_tids, [[(x / scale, y / scale) for x, y in m] for m in self.masks],
                   vehicle_labels=self.vehicle_labels)
        self._pre.append(ann)
        self.latest_jpeg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        self.tracks = [{"camera": self.id, "track_id": o.track_id, "cls": o.cls, "conf": round(o.conf, 2),
                        "x": round(o.ground[0], 2), "y": round(o.ground[1], 2), "alert": o.track_id in alert_tids}
                       for o in obs if o.ground]

        if self.site.cfg.get("record", True) and t >= self._rec_retry_at:
            # A recording failure (ffmpeg exited, disk full) pauses recording for `record_retry_s` instead of
            # ending analytics: a BrokenPipeError here used to stop the whole camera.
            try:
                if self._rec is None or t - self._rec_start >= self._seg_len:
                    if self._rec:
                        self._close_recorder(t)
                    self._rec_start = t
                    self._rec_rel = f"recordings/{self.id}/{time.strftime('%Y%m%d-%H%M%S', time.localtime(t))}.mp4"
                    self._rec = H264Writer(self.site.data / self._rec_rel, small.shape[1], small.shape[0], self.afps)
                self._rec.write(small)
                self.stats.pop("recording_error", None)
            except (OSError, ValueError) as e:
                self.stats["recording_error"] = redact(repr(e))[:200]
                self._rec_retry_at = t + float(self.site.cfg.get("record_retry_s", 30.0))
                rec, self._rec = self._rec, None
                if rec is not None:
                    try:
                        rec.proc.kill() if rec.proc else rec.close()
                    except Exception:  # noqa: BLE001
                        pass

        for p in self._pending:
            p["frames"].append(ann); p["left"] -= 1
        for p in [p for p in self._pending if p["left"] <= 0]:
            self._pending.remove(p)
            threading.Thread(target=_write_clip_bounded, args=(p["path"], p["frames"], self.afps, OVERLAY_LABEL),
                             daemon=True).start()

        stamped = stamp(ann) if (events or self._ext_evidence) else None
        while self._ext_evidence:  # evidence for non-video events (access control) covering this camera
            self._capture(self._ext_evidence.popleft(), stamped, small)
        for e in events:
            snap_rel, raw_rel, clip_rel = self._capture(f"{self.id}_{e.rule}_{int(t * 1000)}_{next(self._ev_seq)}", stamped, small)
            # analyzer events may carry their own evidence (e.g. an object crop, their model) - keep it
            e.evidence = {"snapshot": snap_rel, "snapshot_raw": raw_rel, "clip": clip_rel, "frame_time": t,
                          "overlay": f"snapshot and clip carry a '{OVERLAY_LABEL}' (boxes, zones, labels drawn by "
                                     "Argus); snapshot_raw is the same frame without it",
                          "enhanced": bool(self.conditioner.enhanced),
                          "person_boxes": [[round(float(v), 1) for v in bb] for _, bb, cls, _ in dets if cls == "person"],
                          "calibration": self.calib_info, "model": self.backend.info, **(e.evidence or {})}
            self.stats["events"] += 1
            self.site.on_event(e)

        self.stats["frames"] += 1
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t) if not realtime else 1.0 / max(1e-3, time.time() - self._wall_last)
            self._ema_fps = inst if self._ema_fps == 0 else 0.9 * self._ema_fps + 0.1 * inst
        self._last_t, self._wall_last = t, time.time()
        self.stats.update(fps=round(self._ema_fps, 2), detect_ms=round(dt_ms if self.stats["detect_ms"] == 0
                                                                        else 0.9 * self.stats["detect_ms"] + 0.1 * dt_ms, 1))


def deep_merge(dst: dict, patch: dict) -> dict:
    """Merge `patch` into `dst` in place: dicts merge recursively, lists/scalars replace, None deletes the key."""
    for k, v in patch.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


class Site:
    def __init__(self, config_path: Path, data_dir: Optional[Path] = None, realtime: bool = True,
                 backend_factory: Optional[Callable[[dict], TrackerBackend]] = None,
                 detector_overrides: Optional[dict] = None):
        """`detector_overrides` are merged into the `detector` section for this run only (never saved), e.g.
        run.py --infer-node puts `remote` there so every camera's detection runs on a GPU inference node."""
        self.config_path = Path(config_path)
        self.cfg = json.loads(self.config_path.read_text())
        base = self._base = self.config_path.resolve().parent.parent  # configs/ lives in the project root
        self._orig_sources: dict[str, str] = {}  # relative sources as written in the config (restored on save)
        # Camera passwords never go into the (git-tracked) site config: it stores rtsp://user:***@host/..., and the
        # full URLs live in the git-ignored <config>.secrets.json beside it (see save_config).
        self.secrets_path = self.config_path.with_name(self.config_path.stem + ".secrets.json")
        self._secrets: dict[str, dict] = (json.loads(self.secrets_path.read_text(encoding="utf-8"))
                                          if self.secrets_path.exists() else {})
        for c in self.cfg["cameras"]:
            self._restore_secrets(c)
            self._resolve_source(c)
        w = self.cfg.get("detector", {}).get("weights")
        self._orig_weights: Optional[tuple[str, str]] = None
        if w and not Path(w).is_absolute() and (base / w).exists():
            self.cfg["detector"]["weights"] = str(base / w)
            self._orig_weights = (w, self.cfg["detector"]["weights"])
        self.data = Path(data_dir or self.cfg.get("data_dir", "data/run"))
        self.data.mkdir(parents=True, exist_ok=True)
        self.realtime = realtime
        self.store = Store(self.data / "argonix.db")
        self.baseline = Baseline(self.data / "baseline.db", **self.cfg.get("baseline", {}))
        try:  # split CPU threads between camera workers to avoid oversubscription
            import os, torch
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(1, len(self.cfg["cameras"]))))
        except ImportError:
            pass
        # detector.batched (default: on with CUDA) -> one shared BatchedInference; else one UltralyticsTracker per camera
        # detector_config adds object classes (detector.extra_classes) when the unattended-object analyzer is on
        factory = backend_factory or make_backend_factory({**detector_config(self.cfg), **(detector_overrides or {})},
                                                          models_dir=base / "data" / "models")
        self.backend_factory = factory
        self.cfg_lock = threading.RLock()
        self.started = False
        self.workers = [self._make_worker(c) for c in self.cfg["cameras"] if c.get("enabled", True)]
        model = dict(self.workers[0].backend.info) if self.workers else {}
        model["rules"] = RULES_VERSION
        self.correlator = Correlator(window_s=float(self.cfg.get("correlation_window_s", 15)), model_info=model)
        # operator track record per (rule, camera) on every alert's "Why P?" (+ optional feedback demotion)
        self.track_record = TrackRecord(self.store, demotion=self.cfg.get("feedback_demotion"),
                                        **(self.cfg.get("track_record") or {}))
        self.lock = threading.Lock()
        self.listeners: list[Callable[[dict], None]] = []
        # analyzers: objects with on_frame(worker, t, frame, dets, obs) -> list[Event], optional close()
        self.analyzers: list = []
        if (self.cfg.get("handoff") or {}).get("enabled"):  # next-camera prediction (argonix/handoff.py)
            from .handoff import install
            install(self)
        if (self.cfg.get("reid") or {}).get("enabled"):  # multi-camera Re-ID + global IDs (argonix/global_ids.py)
            from .global_ids import install as install_reid
            install_reid(self)
        from .threat_install import install as install_threats  # weapon-like objects / altercations (config "threats")
        install_threats(self)
        install_analytics(self)  # flow/zone/line/queue stats + crowd rules, unattended objects, anomaly (argonix/flow.py ...)
        from .incidents import install as install_incidents  # incident lifecycle over alerts (config "incidents")
        install_incidents(self)
        from .experimental import install as install_experimental  # forecast + fall (configs "forecast", "fall")
        install_experimental(self)
        from .retention import RetentionJob  # recordings/evidence/alert retention sweep (config "retention")
        self.retention = RetentionJob(self)
        from .restream import install as install_restream  # MediaMTX WebRTC live + main-stream recording ("restream")
        install_restream(self)
        from .sensors import install as install_sensors  # access-control / door events (config "sensors")
        install_sensors(self)
        from .poll_sensors import install as install_poll_sensors  # polled motion sensors, e.g. ESP32-CAM PIR
        install_poll_sensors(self)
        from .notify import install as install_notify  # Telegram / HTTP phone push on incident push decisions
        install_notify(self)

    # ---- camera management -----------------------------------------------------------------
    def _make_worker(self, c: dict) -> CameraWorker:
        return CameraWorker(c, self, self.backend_factory(c))

    def _resolve_source(self, c: dict) -> None:
        src = str(c["source"])
        if not src.isdigit() and "://" not in src and not Path(src).is_absolute() and (self._base / src).exists():
            self._orig_sources[c["id"]] = src
            c["source"] = str(self._base / src)

    def worker(self, cam_id: str) -> Optional[CameraWorker]:
        return next((w for w in self.workers if w.id == cam_id), None)

    def camera_config(self, cam_id: str) -> Optional[dict]:
        return next((c for c in self.cfg["cameras"] if c["id"] == cam_id), None)

    def _cameras_changed(self) -> None:
        rs = getattr(self, "restream", None)  # MediaMTX paths follow camera add/update/remove
        if rs is not None:
            rs.sync_soon()

    SECRET_KEYS = ("source", "main_source")

    def _restore_secrets(self, c: dict) -> None:
        """Put the full URL back for a source the config stores redacted (only if it is the same URL, masked)."""
        for k, full in (self._secrets.get(str(c.get("id"))) or {}).items():
            if k in self.SECRET_KEYS and is_redacted(c.get(k, "")) and redact(full) == c.get(k):
                c[k] = full

    @staticmethod
    def _atomic_write(p: Path, text: str, backup: bool) -> None:
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if backup and p.exists():
            shutil.copy2(p, p.with_name(p.name + ".bak"))
        os.replace(tmp, p)

    def save_config(self) -> None:
        """Atomically persist the site config (previous version kept as <name>.bak).

        Sources with credentials are written redacted; the full URLs go to the git-ignored <config>.secrets.json
        (written first, owner-only where the OS allows), so a `git add -A` can never publish a camera password.
        `${VAR}` placeholders are not secrets and stay as written."""
        with self.cfg_lock:
            out = copy.deepcopy(self.cfg)
            secrets: dict[str, dict] = {}
            for c in out["cameras"]:
                orig = self._orig_sources.get(c["id"])
                if orig and c["source"] == str(self._base / orig):
                    c["source"] = orig
                for k in self.SECRET_KEYS:
                    v = c.get(k)
                    if isinstance(v, str) and "${" not in v and redact(v) != v:
                        secrets.setdefault(c["id"], {})[k], c[k] = v, redact(v)
            if self._orig_weights and out.get("detector", {}).get("weights") == self._orig_weights[1]:
                out["detector"]["weights"] = self._orig_weights[0]
            if secrets or self.secrets_path.exists():
                self._atomic_write(self.secrets_path, json.dumps(secrets, indent=2), backup=False)
                try:
                    os.chmod(self.secrets_path, 0o600)
                except OSError:
                    pass
                self._secrets = secrets
            self._atomic_write(self.config_path, json.dumps(out, indent=2), backup=True)

    def add_camera(self, cfg: dict) -> Optional[CameraWorker]:
        """Add a camera, persist, and start its worker if the site is running. Returns the worker (None if disabled)."""
        cfg = copy.deepcopy(cfg)
        with self.cfg_lock:
            cid = str(cfg.get("id", ""))
            if not CAMERA_ID_RE.match(cid):
                raise ValueError("camera id must be 1-40 characters of letters, digits, '_' or '-'")
            if self.camera_config(cid):
                raise ValueError(f"camera '{cid}' already exists")
            if not str(cfg.get("source", "")).strip():
                raise ValueError("camera source is required")
            self._resolve_source(cfg)
            w = self._make_worker(cfg) if cfg.get("enabled", True) else None  # validates calibration/zones
            self.cfg["cameras"].append(cfg)
            try:
                self.save_config()
            except Exception:
                self.cfg["cameras"].remove(cfg)
                raise
            if w:
                self.workers = self.workers + [w]
                if self.started:
                    w.start()
            self._cameras_changed()
            return w

    def remove_camera(self, cam_id: str) -> None:
        with self.cfg_lock:
            c = self.camera_config(cam_id)
            if c is None:
                raise KeyError(cam_id)
            w = self.worker(cam_id)
            if w:
                w.stop()
                self.workers = [x for x in self.workers if x is not w]
            self.cfg["cameras"].remove(c)
            self._orig_sources.pop(cam_id, None)
            self.save_config()
            self._cameras_changed()

    def update_camera_config(self, cam_id: str, patch: dict) -> Optional[CameraWorker]:
        """Deep-merge `patch` into the camera's config, persist the site config, and restart that camera's worker.

        The new config is validated (a worker is built on a placeholder backend) before the old worker is stopped, so
        on a ValueError nothing changes. The real backend is built only after the old worker has stopped and released
        its own: a shared GPU service hands out one client per camera id, so building it first failed with "camera
        already has a client" on every edit of a batched site. If the backend then cannot be built, the old camera
        is restarted as it was. Returns the new worker, or None if the camera is disabled.
        """
        with self.cfg_lock:
            old_cfg = self.camera_config(cam_id)
            if old_cfg is None:
                raise KeyError(cam_id)
            if "id" in patch and patch["id"] != cam_id:
                raise ValueError("camera id cannot be changed")
            new_cfg = deep_merge(copy.deepcopy(old_cfg), patch)
            if not str(new_cfg.get("source", "")).strip():
                raise ValueError("camera source is required")
            orig_src = self._orig_sources.get(cam_id)
            if "source" in patch:
                self._orig_sources.pop(cam_id, None)
                self._resolve_source(new_cfg)
            enabled = new_cfg.get("enabled", True)
            try:
                if enabled:
                    CameraWorker(new_cfg, self, _NO_BACKEND)  # validates calibration / zones / rules only
            except Exception:
                if orig_src:
                    self._orig_sources[cam_id] = orig_src
                raise
            old_w = self.worker(cam_id)
            if old_w:
                old_w.stop()
            try:
                new_w = self._make_worker(new_cfg) if enabled else None
            except Exception:
                if orig_src:
                    self._orig_sources[cam_id] = orig_src
                if old_w:  # put the camera back as it was
                    back = self._make_worker(old_cfg)
                    self.workers = [back if x is old_w else x for x in self.workers]
                    if self.started:
                        back.start()
                raise
            idx = self.cfg["cameras"].index(old_cfg)
            self.cfg["cameras"][idx] = new_cfg
            self.save_config()
            if old_w:
                ws = [new_w if x is old_w else x for x in self.workers]
            else:
                ws = self.workers + [new_w]
            self.workers = [x for x in ws if x is not None]
            if new_w and self.started:
                new_w.start()
            self._cameras_changed()
            return new_w

    # ---- events / lifecycle ----------------------------------------------------------------
    def on_event(self, e) -> dict:
        """Correlate one event into an alert (+ incident), persist it, notify listeners. Returns the alert dict."""
        try:
            self.baseline.record_event(e)  # per-zone event-rate baseline
        except Exception:  # noqa: BLE001 - statistics must never block an alert
            pass
        with self.lock:
            alert, _ = self.correlator.add(e)
            d = alert.to_dict()
            try:
                self.track_record.annotate(d)
            except Exception:  # noqa: BLE001 - feedback statistics must never block an alert
                pass
            prev = self.store.get_alert(alert.id)
            if prev:  # keep operator decisions (status, note, disposition, escalation, second review, ...)
                for k in OPERATOR_FIELDS:
                    if k in prev:
                        d[k] = prev[k]
                d.setdefault("operator_note", "")
                alert.status = d["status"]
            incidents = getattr(self, "incidents", None)
            if incidents is not None:  # groups the alert into an incident; sets d["incident_id"] (+ inhibition)
                try:
                    incidents.on_alert(d)
                except Exception as ex:  # noqa: BLE001 - incident bookkeeping must never drop an alert
                    d["incident_error"] = repr(ex)[:200]
            self.store.upsert_alert(d)
        for fn in self.listeners:
            fn(d)
        return d

    def on_health_recovered(self, cam: str, cond: str, t: float, since: float) -> Optional[dict]:
        """A camera view condition cleared: mark it on the newest camera_health alert that reported it, so the
        operator sees "recovered after 42 s" instead of an alert that looks ongoing. Raises no new alert."""
        with self.lock:
            for d in self.store.list_alerts(camera=cam, rule="camera_health", limit=10):
                evs = [e for e in d["events"] if e["rule"] == "camera_health" and e["camera_id"] == cam
                       and (e.get("derivation") or {}).get("condition") == cond]
                if not evs:
                    continue
                rec = d.setdefault("health_recovery", [])
                if any(r["condition"] == cond and r["raised_at"] == since for r in rec):
                    return None
                rec.append({"condition": cond, "raised_at": since, "recovered_at": t,
                            "outage_s": round(max(0.0, t - since), 1)})
                self.store.upsert_alert(d)
                break
            else:
                return None
        self.store.audit("system", "camera_recovered", d["id"],
                         {"camera": cam, "condition": cond, "outage_s": round(max(0.0, t - since), 1)})
        for fn in self.listeners:
            fn(d)
        return d

    def start(self) -> None:
        self.started = True
        for w in self.workers:
            if w.ident is None:
                w.start()
        prewarm = getattr(self.backend_factory, "prewarm", None)
        if prewarm is not None and not self.workers and self.realtime:
            # a GPU site started with no cameras: build and warm the shared model now, so the first camera added
            # through the API does not wait for CUDA start-up and one warm-up pass per batch size inside its request
            threading.Thread(target=self._prewarm, args=(prewarm,), daemon=True, name="detector-prewarm").start()
        if getattr(self, "incidents", None) is not None:
            self.incidents.start()  # 1 Hz lifecycle tick (hold-down, cooling, closing without new events)
        if self.realtime:  # offline/batch runs (tests, evaluation) never sweep on their own
            self.retention.start()
        if getattr(self, "restream", None) is not None:
            self.restream.start()
        if getattr(self, "sensors", None) is not None:
            self.sensors.start()  # 1 Hz sensor heartbeat check (sensor_offline)
        for ps in getattr(self, "poll_sensors", []):
            if ps.ident is None:
                ps.start()

    @staticmethod
    def _prewarm(prewarm) -> None:
        t0 = time.time()
        try:
            prewarm()
            print(f"detector: shared model warmed in {time.time() - t0:.1f} s (no cameras yet)", flush=True)
        except Exception as e:  # noqa: BLE001 - the first camera add will retry and report it
            print(f"detector: prewarm failed ({type(e).__name__}: {e}); the first camera add will retry", flush=True)

    def stop(self) -> None:
        self.started = False
        self.retention.close()
        for w in self.workers:
            w.stop_evt.set()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout=30)
        getattr(self.backend_factory, "close", lambda: None)()  # shared BatchedInference thread, if any
        for an in self.analyzers:
            getattr(an, "close", lambda: None)()
        if getattr(self, "sensors", None) is not None:
            self.sensors.close()
        for ps in getattr(self, "poll_sensors", []):
            ps.stop()
        if getattr(self, "incidents", None) is not None:
            self.incidents.tick()
            self.incidents.close()
        if getattr(self, "restream", None) is not None:
            self.restream.close()

    def wait(self, timeout: float = None) -> None:
        for w in self.workers:
            w.done.wait(timeout)

    def tracks(self) -> list[dict]:
        out = [t for w in self.workers for t in w.tracks]
        reid = getattr(self, "reid", None)  # adds the estimated global ID ("gid") when Re-ID is enabled
        return reid.annotate(out) if reid is not None else out
