"""Crowd flow, counting lines, zone dwell/direction, density and queue analytics, plus the crowd-level rules built on
them (`counterflow`, `density`, `queue_long`, `dispersal`, `convergence`).

Runs as a `Site.analyzers` plug-in (installed by `argonix.analytics_install`). Everything here is computed from person
tracks only: foot points on the calibrated ground plane (metres) or, for an uncalibrated camera, in source-image
pixels. The statistics are passive (they never raise an alert by themselves). Events come only from zone rules the
operator has switched on, and each one states what was measured, over which reference, and with which thresholds.

Space conventions:
  * "ground_m": a calibrated camera. Positions in the camera's ground frame (metres). Directions are reported in the
    map's screen convention (x to the right, y up), so angle 90 deg = up on the map.
  * "image_px": an uncalibrated camera. Positions in source-image pixels. Directions are reported as seen on the
    screen (up = towards the top of the image). Speeds are in body-heights per second (box height), never m/s.

Counting lines (per camera `lines`): `{"id", "name", "points_px": [[x0, y0], [x1, y1]]}` in source pixels (works
with or without calibration), or `"points": [[x0, y0], [x1, y1]]` in ground metres (needs calibration). For a segment
A->B the "in" direction is along n = (-dy, dx) in the line's own coordinates: for an image-pixel line that is towards
the RIGHT-hand side when looking from A to B on screen. A crossing is counted when a track's foot point moves from
more than `band` on one side to more than `band` on the other side, through the segment (not its extension).

Line events (opt-in per line): `"events": {"direction": "in" | "out" | "both", "min_gap_s": 5, "criticality": 1,
"schedule": {...}}` raises a `line_cross` event for each counted crossing in that direction, at most one per
`min_gap_s` per line (crossings inside the gap are counted and reported in the next event), and only inside the
line's schedule windows when one is set (argonix/schedule.py; e.g. after-hours door entries).

Zone schedules (argonix/rules.Zone) also gate the crowd rules here: outside its window a rule cannot arm.
"""
from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .geometry import apply_h, foot_point, signed_depth
from .models import Event
from .rules import DEFAULTS, SEVERITY, _Episode, register_rule
from .schedule import Schedule

FLOW_RULES = ("counterflow", "density", "queue_long", "dispersal", "convergence")
PIXEL_OK_RULES = ("queue_long",)  # the only flow rule that is meaningful without a metric ground plane
LINE_DIRECTIONS = ("in", "out", "both")

register_rule("line_cross", "Counting-line crossing", 1)
register_rule("counterflow", "Counter-flow in zone", 2)
register_rule("density", "Crowd density above limit", 2)
register_rule("queue_long", "Queue longer than limit", 1)
register_rule("dispersal", "Rapid outward movement from a point (dispersal cue)", 2)
register_rule("convergence", "Rapid inward movement to a point (convergence cue)", 2)
DEFAULTS.update({
    # fraction of moving tracks against the zone's permitted direction (or its learned dominant flow) for N s
    "counterflow": {"min_fraction": 0.3, "min_tracks": 4, "min_against": 2, "min_seconds": 3.0,
                    "min_speed_mps": 0.4, "cos_threshold": -0.5, "min_concentration": 0.5},
    # people per m^2 on the calibrated ground plane
    "density": {"warn_ppm2": 2.0, "critical_ppm2": 4.0, "min_count": 3, "min_seconds": 5.0},
    # queue length (people in the queue zone) and/or Little's-law wait estimate; 0 disables that condition
    "queue_long": {"max_length": 10, "max_wait_s": 0.0, "min_seconds": 10.0},
    "dispersal": {"min_people": 4, "min_fraction": 0.6, "min_radial_mps": 1.5, "speed_ratio": 2.0,
                  "min_seconds": 1.0},
    "convergence": {"min_people": 4, "min_fraction": 0.6, "min_radial_mps": 1.5, "speed_ratio": 2.0,
                    "min_seconds": 1.0},
})

DWELL_BINS = (0, 5, 15, 30, 60, 120, 300, 600)  # seconds; the last bin is open-ended (>= 600 s)
DIR_BINS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")  # screen convention: N = up
NOTE_UNCAL = "uncalibrated camera: counts only (no metric area, speed or m/s thresholds)"


# ---------------------------------------------------------------------------------------------- geometry helpers
def polygon_area(poly) -> float:
    """Shoelace area (units^2) of a simple polygon."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2.0


def polygon_perimeter(poly) -> float:
    return sum(math.dist(poly[i - 1], poly[i]) for i in range(len(poly)))


def centroid(pts) -> tuple[float, float]:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def screen_angle(v, space: str) -> float:
    """Direction of vector v in degrees [0, 360), counter-clockwise from screen-right with 90 = screen-up."""
    vx, vy = v
    if space == "image_px":
        vy = -vy  # image y grows downwards
    return math.degrees(math.atan2(vy, vx)) % 360.0


def dir_bin(angle_deg: float) -> int:
    return int(((angle_deg + 22.5) % 360.0) // 45.0)


def parse_line_events(ev: Any, lid: str) -> Optional[dict]:
    """Validate a counting line's `events` option. None/{}/false -> None (counting only). Raises ValueError."""
    if ev in (None, False, {}):
        return None
    if ev is True:
        ev = {}
    if not isinstance(ev, dict):
        raise ValueError(f"line '{lid}': events must be an object like {{\"direction\": \"in\", \"min_gap_s\": 5}}")
    bad = set(ev) - {"direction", "min_gap_s", "criticality", "schedule"}
    if bad:
        raise ValueError(f"line '{lid}' events: unknown key(s) {sorted(bad)} "
                         "(allowed: direction, min_gap_s, criticality, schedule)")
    direction = ev.get("direction", "both")
    if direction not in LINE_DIRECTIONS:
        raise ValueError(f"line '{lid}' events: direction must be one of {list(LINE_DIRECTIONS)}")
    gap = ev.get("min_gap_s", 5.0)
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not 0 <= gap <= 86400:
        raise ValueError(f"line '{lid}' events: min_gap_s must be a number of seconds between 0 and 86400")
    crit = ev.get("criticality", 1)
    if isinstance(crit, bool) or not isinstance(crit, int) or not 1 <= crit <= 3:
        raise ValueError(f"line '{lid}' events: criticality must be an integer 1-3")
    sch = Schedule.parse(ev.get("schedule"), f"line '{lid}' events schedule")
    return {"direction": direction, "min_gap_s": float(gap), "criticality": crit, "schedule": sch}


class Episodes:
    """Condition-held-for-N-seconds bookkeeping with a grace period and a re-arm window (same semantics as
    `rules.RuleEngine._tick`): fires once per episode, and not again for the same key within `rearm_s`."""

    def __init__(self, grace_s: float = 2.0, rearm_s: float = 60.0):
        self.grace, self.rearm = grace_s, rearm_s
        self.eps: dict[tuple, _Episode] = {}
        self.last_fired: dict[tuple, float] = {}

    def tick(self, key: tuple, t: float, cond: bool, conf: float, value: float, min_s: float) -> Optional[_Episode]:
        ep = self.eps.get(key)
        if cond:
            if ep is None or t - ep.last > self.grace:
                ep = self.eps[key] = _Episode(first=t, last=t)
            ep.last = t
            ep.confs.append(conf)
            ep.peak = max(ep.peak, value)
            if not ep.fired and t - ep.first >= min_s:
                ep.fired = True
                if t - self.last_fired.get(key, -1e18) < self.rearm:
                    return None
                self.last_fired[key] = t
                return ep
        elif ep is not None and t - ep.last > self.grace:
            del self.eps[key]
        return None

    def active(self, key: tuple) -> bool:
        return key in self.eps


# ---------------------------------------------------------------------------------------------- state
@dataclass
class _Track:
    hist: deque = field(default_factory=lambda: deque(maxlen=40))  # (t, pos, h_px, conf); pos in the camera's space
    last_t: float = 0.0
    conf: float = 0.0
    h_px: float = 1.0


class _Line:
    """A directed counting segment A->B; 'in' = crossing along n = (-dy, dx)."""

    def __init__(self, lid: str, name: str, a, b, space: str, band: float, in_label: str = "in",
                 out_label: str = "out", events: Optional[dict] = None):
        self.id, self.name, self.space = lid, name, space
        self.events = events                 # parse_line_events() result, or None (counting only)
        self.last_event: Optional[float] = None
        self.suppressed = 0                  # matching crossings not raised because of min_gap_s
        self.n_events = 0
        self.a, self.b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        self.len = math.hypot(dx, dy)
        if self.len < 1e-6:
            raise ValueError(f"line '{lid}': points must differ")
        self.n = (-dy / self.len, dx / self.len)
        self.band = band
        self.labels = {"in": in_label, "out": out_label}
        self.counts = {"in": 0, "out": 0}
        self.side: dict[int, tuple[int, tuple[float, float]]] = {}  # tid -> (side, position when that side was set)
        self.series: dict[int, dict] = {}  # minute -> {"in", "out"}

    def signed(self, p) -> float:
        return (p[0] - self.a[0]) * self.n[0] + (p[1] - self.a[1]) * self.n[1]

    def update(self, tid: int, p) -> Optional[str]:
        """Feed a position; returns "in"/"out" when this track just crossed the segment, else None."""
        s = self.signed(p)
        side = 1 if s > self.band else -1 if s < -self.band else 0
        prev = self.side.get(tid)
        if side == 0:
            return None
        self.side[tid] = (side, (p[0], p[1]))
        if prev is None or prev[0] == side:
            return None
        # crossing: intersect the move prev_pos -> p with the infinite line, then require it to lie on the segment
        q = prev[1]
        sq, sp = self.signed(q), s
        f = sq / (sq - sp) if sq != sp else 0.5
        x = (q[0] + f * (p[0] - q[0]), q[1] + f * (p[1] - q[1]))
        u = ((x[0] - self.a[0]) * (self.b[0] - self.a[0]) + (x[1] - self.a[1]) * (self.b[1] - self.a[1])) / self.len ** 2
        tol = self.band / self.len
        if not -tol <= u <= 1 + tol:
            return None
        return "in" if side > 0 else "out"

    def forget(self, tid: int) -> None:
        self.side.pop(tid, None)


class _Zone:
    def __init__(self, z, raw: dict, space: str, service_line: Optional[_Line], half_life_s: float):
        self.id, self.name, self.polygon = z.id, z.name, [tuple(p) for p in z.polygon]
        self.rules, self.criticality, self.site_zone = z.rules, z.criticality, z.site_zone
        self.zone = z  # rules.Zone: schedules (armed / schedule_info)
        self.direction = z.direction if space == "ground_m" else None
        self.space = space
        self.kind = "queue" if (raw.get("type") == "queue" or "queue_long" in z.rules) else "area"
        self.area = polygon_area(self.polygon) if space == "ground_m" else None
        self.perimeter = polygon_perimeter(self.polygon) if space == "ground_m" else None
        self.service = service_line
        self.queue_side = None
        if service_line is not None:
            s = service_line.signed(centroid(self.polygon))
            self.queue_side = 1 if s >= 0 else -1
        self.tau = half_life_s / math.log(2)
        self.inside: dict[int, float] = {}       # tid -> entry time
        self.last_in: dict[int, float] = {}      # tid -> last time seen inside
        self.recent_exit: dict[int, tuple[float, float]] = {}  # tid -> (entry, exit) for late service crossings
        self.entries = self.exits = 0
        self.dwell_counts = [0] * len(DWELL_BINS)
        self.dwells: deque = deque(maxlen=200)   # (exit_t, dwell_s)
        self.dir_hist = [0.0] * 8                # decayed track-seconds per 45-degree bin (screen convention)
        self.mean_vec = [0.0, 0.0, 0.0]          # decayed sum of unit vectors (analysis frame) and weight
        self.hist_t: Optional[float] = None
        self.departures: deque = deque(maxlen=2000)  # queue departure times
        self.waits: deque = deque(maxlen=200)        # (t, measured wait s) from entry -> service crossing
        self.occ: deque = deque(maxlen=4000)         # (t, count) for queue smoothing / steady-state check
        self.speed_hist: deque = deque(maxlen=200)   # (t, median speed) for the dispersal "sudden" test
        self.series: dict[int, dict] = {}
        self.last: dict[str, Any] = {"count": 0, "t": None}

    # ---- decayed direction statistics ----
    def decay_to(self, t: float) -> None:
        if self.hist_t is None:
            self.hist_t = t
            return
        dt = t - self.hist_t
        if dt <= 0:
            return
        f = math.exp(-dt / self.tau)
        self.dir_hist = [v * f for v in self.dir_hist]
        self.mean_vec = [v * f for v in self.mean_vec]
        self.hist_t = t

    def learn(self, v, w: float) -> None:
        sp = math.hypot(*v)
        if sp <= 0:
            return
        self.dir_hist[dir_bin(screen_angle(v, self.space))] += w
        self.mean_vec[0] += w * v[0] / sp
        self.mean_vec[1] += w * v[1] / sp
        self.mean_vec[2] += w

    def dominant(self) -> tuple[Optional[tuple[float, float]], float, float]:
        """(unit vector in the analysis frame or None, concentration R in [0, 1], weight in track-seconds)."""
        sx, sy, w = self.mean_vec
        if w <= 1e-9:
            return None, 0.0, 0.0
        r = math.hypot(sx, sy) / w
        if r <= 1e-9:
            return None, 0.0, w
        n = math.hypot(sx, sy)
        return (sx / n, sy / n), r, w


class _Cam:
    def __init__(self, worker, an: "FlowAnalyzer"):
        self.worker_ref = id(worker)
        self.id = worker.id
        self.space = "ground_m" if getattr(worker, "H", None) is not None else "image_px"
        self.calib_error = getattr(worker, "calib_error", None)
        self.tracks: dict[int, _Track] = {}
        self.eps = Episodes(grace_s=an.grace_s, rearm_s=an.rearm_s)
        self.last_t: Optional[float] = None
        self.warnings: list[str] = []
        cfg = getattr(worker, "cfg", {}) or {}
        raw = {z.get("id"): z for z in cfg.get("zones", []) if isinstance(z, dict)}
        zones = worker.zones if self.space == "ground_m" else getattr(worker, "px_zones", [])
        self.zones: list[_Zone] = []
        for z in zones:
            rz = raw.get(z.id, {})
            svc = None
            pts = self._line_points(worker, rz.get("service_line_px"), rz.get("service_line"), f"zone '{z.id}' service line")
            if pts:
                band = an.line_band_m if self.space == "ground_m" else an.line_band_px
                svc = _Line(f"{z.id}:service", f"{z.name} service line", pts[0], pts[1], self.space, band)
            self.zones.append(_Zone(z, rz, self.space, svc, an.half_life_s))
        # counting lines: image-pixel lines always work; ground-metre lines need calibration
        self.lines: list[_Line] = []
        for i, ln in enumerate(cfg.get("lines", []) or []):
            lid = str(ln.get("id") or f"line_{i + 1}")
            try:
                events = parse_line_events(ln.get("events"), lid)
                if ln.get("points_px"):
                    a, b = ln["points_px"]
                    self.lines.append(_Line(lid, ln.get("name", lid), a, b, "image_px",
                                            float(ln.get("band", an.line_band_px)), ln.get("in_label", "in"),
                                            ln.get("out_label", "out"), events))
                elif ln.get("points"):
                    if self.space != "ground_m":
                        self.warnings.append(f"line '{lid}': ground-metre points need calibration (skipped)")
                        continue
                    a, b = ln["points"]
                    self.lines.append(_Line(lid, ln.get("name", lid), a, b, "ground_m",
                                            float(ln.get("band", an.line_band_m)), ln.get("in_label", "in"),
                                            ln.get("out_label", "out"), events))
            except (ValueError, TypeError) as e:
                self.warnings.append(f"line '{lid}' skipped: {e}")

    def _line_points(self, worker, px, metres, what):
        if px:
            if self.space == "ground_m":
                try:
                    return [apply_h(worker.H, tuple(p)) for p in px]
                except ValueError:
                    self.warnings.append(f"{what}: above the horizon, skipped")
                    return None
            return [tuple(p) for p in px]
        if metres:
            if self.space != "ground_m":
                self.warnings.append(f"{what}: ground-metre points need calibration (skipped)")
                return None
            return [tuple(p) for p in metres]
        return None

    def carry_over(self, old: "_Cam") -> None:
        """Keep counters and time series of zones/lines that still exist after a camera config change."""
        oz = {z.id: z for z in old.zones}
        for z in self.zones:
            o = oz.get(z.id)
            if o is not None:
                z.entries, z.exits, z.dwell_counts, z.dwells = o.entries, o.exits, o.dwell_counts, o.dwells
                z.series, z.dir_hist, z.mean_vec, z.hist_t = o.series, o.dir_hist, o.mean_vec, o.hist_t
                z.departures, z.waits, z.occ = o.departures, o.waits, o.occ
        ol = {ln.id: ln for ln in old.lines}
        for ln in self.lines:
            o = ol.get(ln.id)
            if o is not None and (o.a, o.b) == (ln.a, ln.b):
                ln.counts, ln.series = o.counts, o.series
                ln.last_event, ln.suppressed, ln.n_events = o.last_event, o.suppressed, o.n_events


# ---------------------------------------------------------------------------------------------- analyzer
class FlowAnalyzer:
    """Per-camera flow/zone/line/queue statistics and the crowd-level rules. Thread-safe summaries for the API."""

    name = "flow"

    def __init__(self, bucket_s: float = 60.0, series_minutes: int = 60, half_life_s: float = 300.0,
                 exit_grace_s: float = 2.0, stale_s: float = 3.0, min_speed_mps: float = 0.3,
                 min_speed_bhps: float = 0.2, line_band_px: float = 6.0, line_band_m: float = 0.15,
                 queue_window_s: float = 300.0, grace_s: float = 2.0, rearm_s: float = 60.0, baseline=None):
        self.bucket_s, self.series_minutes = float(bucket_s), int(series_minutes)
        self.half_life_s, self.exit_grace, self.stale = float(half_life_s), float(exit_grace_s), float(stale_s)
        self.min_speed_mps, self.min_speed_bhps = float(min_speed_mps), float(min_speed_bhps)
        self.line_band_px, self.line_band_m = float(line_band_px), float(line_band_m)
        self.queue_window = float(queue_window_s)
        self.grace_s, self.rearm_s = float(grace_s), float(rearm_s)
        self.baseline = baseline
        self.lock = threading.RLock()
        self.cams: dict[str, _Cam] = {}

    # ---- helpers ----
    def _cam(self, worker) -> _Cam:
        c = self.cams.get(worker.id)
        if c is None or c.worker_ref != id(worker):  # new camera, or its worker was rebuilt after a config change
            new = _Cam(worker, self)
            if c is not None:
                new.carry_over(c)
            c = self.cams[worker.id] = new
        return c

    @staticmethod
    def _velocity(tr: _Track, window_s: float = 1.0):
        h = tr.hist
        if len(h) < 2:
            return None
        t1, p1 = h[-1][0], h[-1][1]
        t0, p0 = next(((e[0], e[1]) for e in h if t1 - e[0] <= window_s), (h[-2][0], h[-2][1]))
        if t1 - t0 < 0.4:
            return None
        return ((p1[0] - p0[0]) / (t1 - t0), (p1[1] - p0[1]) / (t1 - t0))

    def _speed_unit(self, cam: _Cam, tr: _Track, v) -> float:
        """Speed in m/s (calibrated) or body-heights/s (uncalibrated)."""
        sp = math.hypot(*v)
        return sp if cam.space == "ground_m" else sp / max(tr.h_px, 1.0)

    def _bucket(self, series: dict, t: float) -> dict:
        m = int(t // self.bucket_s)
        b = series.get(m)
        if b is None:
            b = series[m] = {"occ_sum": 0.0, "occ_n": 0, "occ_max": 0, "entries": 0, "exits": 0, "dens_sum": 0.0,
                             "in": 0, "out": 0, "wait": None}
            for k in [k for k in series if k <= m - self.series_minutes - 1]:
                del series[k]
        return b

    # ---- main entry point ----
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        with self.lock:
            return self._on_frame(worker, t, dets, obs)

    def _on_frame(self, worker, t: float, dets, obs) -> list[Event]:
        cam = self._cam(worker)
        cam.last_t = t
        scale = float(getattr(worker, "scale", 1.0) or 1.0)
        ground = {o.track_id: o.ground for o in obs if o.cls == "person"}
        people = []  # (tid, pos in cam space, foot px (source), h_px (source), conf)
        for tid, bb, cls, conf in dets:
            if cls != "person":
                continue
            fx, fy = foot_point(bb)
            fpx = (fx * scale, fy * scale)
            h_px = max(1.0, (bb[3] - bb[1]) * scale)
            pos = ground.get(tid) if cam.space == "ground_m" else fpx  # None: above the horizon (calibrated)
            people.append((tid, pos, fpx, h_px, conf))

        for tid, pos, fpx, h_px, conf in people:
            if pos is None:
                continue
            tr = cam.tracks.setdefault(tid, _Track())
            tr.hist.append((t, pos))
            tr.last_t, tr.conf, tr.h_px = t, conf, h_px
        for tid in [k for k, tr in cam.tracks.items() if t - tr.last_t > self.stale]:
            del cam.tracks[tid]
            for ln in cam.lines:
                ln.forget(tid)
            for z in cam.zones:
                if z.service:
                    z.service.forget(tid)

        # counting lines (+ opt-in line_cross events)
        events: list[Event] = []
        for ln in cam.lines:
            for tid, pos, fpx, h_px, conf in people:
                q = fpx if ln.space == "image_px" else pos
                d = ln.update(tid, q) if q is not None else None
                if d:
                    ln.counts[d] += 1
                    self._bucket(ln.series, t)[d] += 1
                    if ln.events:
                        e = self._line_event(cam, ln, t, tid, d, pos, conf)
                        if e:
                            events.append(e)

        placed = [p for p in people if p[1] is not None]
        for z in cam.zones:
            events += self._zone_step(cam, z, t, placed)
        return events

    def _line_event(self, cam: _Cam, ln: _Line, t: float, tid: int, d: str, pos, conf: float) -> Optional[Event]:
        """A `line_cross` event for a counted crossing, when the line asks for this direction, its schedule (if any)
        is active, and the line's last event is at least min_gap_s old (otherwise the crossing is only tallied)."""
        ev = ln.events
        if ev["direction"] not in (d, "both"):
            return None
        sch = ev["schedule"]
        sched = sch.describe(t, "line") if sch is not None else None
        if sch is not None and sched is None:
            return None  # outside the line's schedule: counted, no event
        if ln.last_event is not None and t - ln.last_event < ev["min_gap_s"]:
            ln.suppressed += 1
            return None
        suppressed, ln.suppressed, ln.last_event = ln.suppressed, 0, t
        ln.n_events += 1
        word = "inbound" if d == "in" else "outbound"
        lab = ln.labels[d]
        obs = (f"person track #{tid} ({cam.id}) crossed line '{ln.name}' {word}"
               + (f" ('{lab}')" if lab != d else ""))
        if suppressed:
            obs += f"; {suppressed} more matching crossing(s) in the previous {ev['min_gap_s']:g}s gap"
        deriv = {"rule": "line_cross", "line": ln.id, "direction": d, "direction_label": lab,
                 "direction_rule": "in = crossing along n = (-dy, dx) of A->B (image lines: towards the right-hand "
                                   "side looking from A to B)",
                 "threshold": {"direction": ev["direction"], "min_gap_s": ev["min_gap_s"], "band": ln.band},
                 "crossings_suppressed_by_gap": suppressed, "line_counts": dict(ln.counts),
                 "duration_s": 0.0, "confidence_method": "detector conf of the crossing person track (uncalibrated)",
                 "space": ln.space, "calib_error_m": cam.calib_error, "analyzer": "flow-v1"}
        if sched:
            deriv["schedule"] = sched
            obs += f" [{sched['text']}]"
        g = (round(pos[0], 2), round(pos[1], 2)) if pos is not None and cam.space == "ground_m" else None
        return Event("line_cross", cam.id, f"line:{ln.id}", ln.name, [tid], t, t, obs, deriv, round(conf, 3),
                     SEVERITY["line_cross"], ev["criticality"], None, g)

    def _zone_step(self, cam: _Cam, z: _Zone, t: float, people) -> list[Event]:
        events: list[Event] = []
        inside = []
        for tid, pos, fpx, h_px, conf in people:
            if signed_depth(pos, z.polygon) >= 0:
                inside.append((tid, pos, conf))
                if tid not in z.inside:
                    z.inside[tid] = t
                    z.entries += 1
                    self._bucket(z.series, t)["entries"] += 1
                z.last_in[tid] = t
        # exits: outside (or unseen) for longer than the grace period
        for tid in [k for k in z.inside if t - z.last_in.get(k, t) > self.exit_grace]:
            entry, last = z.inside.pop(tid), z.last_in.pop(tid)
            self._record_exit(cam, z, tid, entry, last)
        for tid in [k for k, (e, x) in z.recent_exit.items() if t - x > 10.0]:
            del z.recent_exit[tid]

        n = len(inside)
        b = self._bucket(z.series, t)
        b["occ_sum"] += n; b["occ_n"] += 1; b["occ_max"] = max(b["occ_max"], n)
        z.occ.append((t, n))
        while z.occ and t - z.occ[0][0] > self.queue_window:
            z.occ.popleft()
        density = n / z.area if z.area else None
        if density is not None:
            b["dens_sum"] += density

        # velocities of tracks inside (analysis frame) + direction learning
        z.decay_to(t)
        moving = []  # (tid, pos, v, speed_unit, conf)
        speeds = []
        min_sp = self.min_speed_mps if cam.space == "ground_m" else self.min_speed_bhps
        for tid, pos, conf in inside:
            tr = cam.tracks.get(tid)
            v = self._velocity(tr) if tr else None
            if v is None:
                continue
            sp = self._speed_unit(cam, tr, v)
            speeds.append(sp)
            if sp >= min_sp:
                moving.append((tid, pos, v, sp, conf))
        dt = (t - z.last["t"]) if z.last.get("t") is not None else 0.0
        dt = min(max(dt, 0.0), 1.0)
        learning = not cam.eps.active(("counterflow", z.id))  # freeze the learned reference during counter-flow
        if learning and dt > 0:
            for _, _, v, _, _ in moving:
                z.learn(v, dt)
        if speeds:
            z.speed_hist.append((t, statistics.median(speeds)))

        # queue service line
        if z.service is not None:
            for tid, pos, fpx, h_px, conf in people:
                d = z.service.update(tid, pos)
                if d and (1 if d == "in" else -1) == -z.queue_side:  # crossed from the queue side to the far side
                    self._departure(z, tid, t)
        z.last = {"t": t, "count": n, "density": density, "inside": [i[0] for i in inside]}

        # ---- rules (a rule outside its schedule window cannot arm) ----
        mean_conf = (sum(c for _, _, c in inside) / n) if n else 0.0
        if "counterflow" in z.rules and cam.space == "ground_m" and self._armed(cam, z, "counterflow", t):
            events += self._counterflow(cam, z, t, moving, mean_conf)
        if "density" in z.rules and z.area and self._armed(cam, z, "density", t):
            events += self._density(cam, z, t, inside, density, mean_conf)
        if "queue_long" in z.rules and self._armed(cam, z, "queue_long", t):
            events += self._queue_rule(cam, z, t, inside, mean_conf)
        for rule in ("dispersal", "convergence"):
            if rule in z.rules and cam.space == "ground_m" and self._armed(cam, z, rule, t):
                events += self._radial(cam, z, t, rule, moving, speeds)
        q = self.queue_stats(z, t) if z.kind == "queue" else None
        if q:
            b["wait"] = q.get("wait_est_s")
        return events

    @staticmethod
    def _armed(cam: _Cam, z: _Zone, rule: str, t: float) -> bool:
        """Schedule gate. Outside the window the rule's open episodes are dropped, so it re-arms from scratch."""
        if z.zone.armed(rule, t):
            return True
        for k in [k for k in cam.eps.eps if k[0] == rule and k[-1] == z.id]:
            del cam.eps.eps[k]
        return False

    def _record_exit(self, cam: _Cam, z: _Zone, tid: int, entry: float, last: float) -> None:
        dwell = max(0.0, last - entry)
        z.exits += 1
        self._bucket(z.series, last)["exits"] += 1
        z.dwells.append((last, dwell))
        z.dwell_counts[max(i for i, lo in enumerate(DWELL_BINS) if dwell >= lo)] += 1
        z.recent_exit[tid] = (entry, last)
        if z.kind == "queue" and z.service is None and dwell >= 2.0:
            z.departures.append(last)
            z.waits.append((last, dwell))
        if self.baseline is not None and hasattr(self.baseline, "add_dwell") and dwell >= 1.0:
            try:
                self.baseline.add_dwell(cam.id, z.id, entry, dwell)
            except Exception:  # noqa: BLE001 - statistics must never stop the analyzer
                pass

    def _departure(self, z: _Zone, tid: int, t: float) -> None:
        z.departures.append(t)
        entry = z.inside.get(tid)
        if entry is None and tid in z.recent_exit:
            entry = z.recent_exit[tid][0]
        if entry is not None and t - entry >= 1.0:
            z.waits.append((t, t - entry))

    # ---- queue ----
    def queue_stats(self, z: _Zone, t: float) -> dict:
        win = self.queue_window
        occ = [c for (tt, c) in z.occ if t - tt <= win]
        span = (t - z.occ[0][0]) if z.occ else 0.0
        mean_len = sum(occ) / len(occ) if occ else 0.0
        recent = [(tt, c) for (tt, c) in z.occ if t - tt <= 60.0]
        length_60 = sum(c for _, c in recent) / len(recent) if recent else 0.0
        deps = [d for d in z.departures if t - d <= win]
        eff = max(min(win, span), 1e-6)
        rate = len(deps) / eff if span >= 30.0 else None  # departures per second
        wait = mean_len / rate if rate and len(deps) >= 3 else None
        half = [c for (tt, c) in z.occ if t - tt > eff / 2], [c for (tt, c) in z.occ if t - tt <= eff / 2]
        steady = None
        if half[0] and half[1]:
            m1, m2 = sum(half[0]) / len(half[0]), sum(half[1]) / len(half[1])
            steady = abs(m2 - m1) / max(mean_len, 1.0) <= 0.3
        waits = sorted(w for (tt, w) in z.waits if t - tt <= 3600)
        return {
            "length": z.last.get("count", 0), "length_mean_60s": round(length_60, 2),
            "length_mean_window": round(mean_len, 2), "window_s": round(eff, 1),
            "departures_window": len(deps),
            "service_rate_per_min": round(rate * 60, 2) if rate is not None else None,
            "wait_est_s": round(wait, 1) if wait is not None else None,
            "steady_state": steady,
            "measured_wait_p50_s": round(statistics.median(waits), 1) if waits else None,
            "measured_waits_n": len(waits),
            "method": ("service-line crossings" if z.service is not None else "zone exits (no service line)"),
            "note": ("Estimate: Little's law W = L / lambda (mean queue length over the window / departure rate). "
                     "Valid only in steady state; measured waits are per-track entry->service times."),
        }

    def _queue_rule(self, cam: _Cam, z: _Zone, t: float, inside, conf) -> list[Event]:
        p = z.rules["queue_long"]
        q = self.queue_stats(z, t)
        ln = q["length_mean_60s"] if q["window_s"] >= 10 else len(inside)
        by_len = p["max_length"] > 0 and ln > p["max_length"]
        by_wait = (p["max_wait_s"] > 0 and q["wait_est_s"] is not None and q["steady_state"] is not False
                   and q["wait_est_s"] > p["max_wait_s"])
        ep = cam.eps.tick(("queue_long", z.id), t, by_len or by_wait, conf or 0.5, ln, p["min_seconds"])
        if not ep:
            return []
        wait_txt = (f", estimated wait {q['wait_est_s']:.0f} s (Little's law, estimate)" if q["wait_est_s"] is not None
                    else ", no service observed yet (no wait estimate)")
        obs = (f"Queue in '{z.name}': {ln:.1f} people (60 s mean, peak {ep.peak:.0f}) for "
               f"{ep.last - ep.first:.0f} s{wait_txt}")
        return [self._event(cam, z, "queue_long", ep, sorted(i[0] for i in inside), obs,
                            {"threshold": {k: p[k] for k in ("max_length", "max_wait_s", "min_seconds")},
                             "queue": q, "condition": "length" if by_len else "wait"},
                            "mean detector conf of people in the zone (uncalibrated)",
                            ground=centroid([i[1] for i in inside]) if inside else centroid(z.polygon))]

    # ---- counter-flow ----
    def _counterflow(self, cam: _Cam, z: _Zone, t: float, moving, conf) -> list[Event]:
        p = z.rules["counterflow"]
        if z.direction:
            dn = math.hypot(*z.direction)
            ref, ref_src, conc, w = (z.direction[0] / dn, z.direction[1] / dn), "permitted direction (configured)", 1.0, None
        else:
            ref, conc, w = z.dominant()
            ref_src = "learned dominant flow"
            if ref is None or conc < p["min_concentration"] or w < 30.0:
                ref = None
        mv = [m for m in moving if m[3] >= p["min_speed_mps"]]
        against = []
        if ref is not None:
            for tid, pos, v, sp, c in mv:
                cos = (v[0] * ref[0] + v[1] * ref[1]) / max(math.hypot(*v), 1e-9)
                if cos <= p["cos_threshold"]:
                    against.append(tid)
        frac = len(against) / len(mv) if mv else 0.0
        cond = (ref is not None and len(mv) >= p["min_tracks"] and len(against) >= p["min_against"]
                and frac >= p["min_fraction"])
        ep = cam.eps.tick(("counterflow", z.id), t, cond, conf or 0.5, frac, p["min_seconds"])
        if not ep:
            return []
        ang = screen_angle(ref, "ground_m")
        obs = (f"{len(against)} of {len(mv)} moving person tracks in '{z.name}' moving against the "
               f"{ref_src} (towards {DIR_BINS[dir_bin((ang + 180) % 360)]} instead of {DIR_BINS[dir_bin(ang)]}); "
               f"peak {ep.peak:.0%} for {ep.last - ep.first:.1f} s")
        return [self._event(cam, z, "counterflow", ep, sorted(against), obs,
                            {"threshold": {k: p[k] for k in ("min_fraction", "min_tracks", "min_against",
                                                             "min_seconds", "min_speed_mps", "cos_threshold")},
                             "reference": {"source": ref_src, "direction_deg": round(ang, 1),
                                           "concentration_R": round(conc, 2),
                                           "learned_track_seconds": round(w, 1) if w is not None else None,
                                           "half_life_s": self.half_life_s if w is not None else None},
                             "fraction_against": round(frac, 2), "moving_tracks": len(mv)},
                            "mean detector conf of people in the zone (uncalibrated)",
                            ground=centroid([m[1] for m in mv]) if mv else None)]

    # ---- density ----
    def _density(self, cam: _Cam, z: _Zone, t: float, inside, density, conf) -> list[Event]:
        p = z.rules["density"]
        n = len(inside)
        out = []
        for level, thr_key, sev in (("warning", "warn_ppm2", 2), ("critical", "critical_ppm2", 3)):
            thr = p[thr_key]
            if thr <= 0:
                continue
            ep = cam.eps.tick(("density", level, z.id), t, n >= p["min_count"] and density >= thr, conf or 0.5,
                              density, p["min_seconds"])
            if not ep:
                continue
            err = float(cam.calib_error or 0.0)
            a_lo, a_hi = max(z.area - z.perimeter * err, 1e-6), z.area + z.perimeter * err
            obs = (f"{n} person tracks in '{z.name}' ({z.area:.1f} m2): {density:.2f} people/m2, peak {ep.peak:.2f} "
                   f"for {ep.last - ep.first:.1f} s (limit {thr} /m2, {level})")
            e = self._event(cam, z, "density", ep, sorted(i[0] for i in inside), obs,
                            {"level": level, "threshold": {thr_key: thr, "min_count": p["min_count"],
                                                           "min_seconds": p["min_seconds"]},
                             "count": n, "area_m2": round(z.area, 2), "density_ppm2": round(density, 3),
                             "density_range_ppm2": [round(n / a_hi, 3), round(n / a_lo, 3)],
                             "area_note": "area on the calibrated ground plane; range from calib_error_m x perimeter"},
                            "mean detector conf of people in the zone (uncalibrated)",
                            ground=centroid([i[1] for i in inside]) if inside else None)
            e.severity = sev
            out.append(e)
        return out

    # ---- dispersal / convergence ----
    def _radial(self, cam: _Cam, z: _Zone, t: float, rule: str, moving, speeds) -> list[Event]:
        p = z.rules[rule]
        cond, info = False, {}
        if len(moving) >= p["min_people"]:
            c = centroid([m[1] for m in moving])
            sign = 1.0 if rule == "dispersal" else -1.0
            radial, aligned = [], 0
            for tid, pos, v, sp, conf in moving:
                r = (pos[0] - c[0], pos[1] - c[1])
                rn = math.hypot(*r)
                if rn < 0.5:
                    continue
                rs = (v[0] * r[0] + v[1] * r[1]) / rn  # radial speed (+ outward)
                if sign * rs / max(sp, 1e-9) >= 0.7:
                    aligned += 1
                    radial.append(sign * rs)
            pre = [s for (tt, s) in z.speed_hist if 3.0 <= t - tt <= 8.0]
            pre_speed = statistics.median(pre) if pre else 0.0
            now_speed = statistics.median(m[3] for m in moving)
            frac = aligned / len(moving)
            med_radial = statistics.median(radial) if radial else 0.0
            sudden = now_speed >= p["speed_ratio"] * max(pre_speed, 0.5)
            cond = frac >= p["min_fraction"] and med_radial >= p["min_radial_mps"] and sudden
            info = {"centre": [round(c[0], 2), round(c[1], 2)], "fraction_aligned": round(frac, 2),
                    "median_radial_speed_mps": round(med_radial, 2), "median_speed_mps": round(now_speed, 2),
                    "median_speed_before_mps": round(pre_speed, 2), "moving_tracks": len(moving)}
        conf = (sum(m[4] for m in moving) / len(moving)) if moving else 0.5
        ep = cam.eps.tick((rule, z.id), t, cond, conf, info.get("median_radial_speed_mps", 0.0), p["min_seconds"])
        if not ep:
            return []
        word = "outward from" if rule == "dispersal" else "inward towards"
        obs = (f"{int(info['fraction_aligned'] * info['moving_tracks'])} of {info['moving_tracks']} moving person "
               f"tracks in '{z.name}' moving rapidly {word} a common point (median radial speed "
               f"{info['median_radial_speed_mps']:.1f} m/s, up from {info['median_speed_before_mps']:.1f} m/s); "
               "a motion cue only, the cause is unknown")
        return [self._event(cam, z, rule, ep, sorted(m[0] for m in moving), obs,
                            {"threshold": {k: p[k] for k in ("min_people", "min_fraction", "min_radial_mps",
                                                             "speed_ratio", "min_seconds")}, **info},
                            "mean detector conf of the moving tracks (uncalibrated)",
                            ground=tuple(info["centre"]))]

    def _event(self, cam: _Cam, z: _Zone, rule: str, ep, tids, obs: str, deriv: dict, conf_method: str,
               ground=None) -> Event:
        from .rules import SEVERITY
        conf = sum(ep.confs) / len(ep.confs) if ep.confs else 0.5
        d = {**deriv, "rule": rule, "duration_s": round(ep.last - ep.first, 2), "confidence_method": conf_method,
             "space": cam.space, "calib_error_m": cam.calib_error, "analyzer": "flow-v1"}
        sched = z.zone.schedule_info(rule, ep.last)
        if sched:
            d["schedule"] = sched
            obs = f"{obs} [{sched['text']}]"
        g = (round(ground[0], 2), round(ground[1], 2)) if ground and cam.space == "ground_m" else None
        return Event(rule, cam.id, z.id, z.name, list(tids), ep.first, ep.last, obs, d, round(conf, 3),
                     SEVERITY[rule], z.criticality, z.site_zone, g)

    # ---------------------------------------------------------------------------------------------- summaries
    def _series(self, series: dict, t: float, kind: str, dens: bool = False) -> list[dict]:
        """Dense list of the last `series_minutes` buckets ending at the camera's latest frame time."""
        cur = int(t // self.bucket_s)
        out = []
        for m in range(cur - self.series_minutes + 1, cur + 1):
            b = series.get(m)
            row: dict[str, Any] = {"t": m * self.bucket_s}
            if kind == "zone":
                seen = bool(b and b["occ_n"])
                row["occ_mean"] = round(b["occ_sum"] / b["occ_n"], 2) if seen else None
                row["occ_max"] = b["occ_max"] if seen else None
                row["entries"] = b["entries"] if b else 0
                row["exits"] = b["exits"] if b else 0
                row["density"] = round(b["dens_sum"] / b["occ_n"], 3) if seen and dens else None
                row["wait_est_s"] = b["wait"] if b else None
            else:
                row["in"] = b["in"] if b else 0
                row["out"] = b["out"] if b else 0
            out.append(row)
        return out

    def zone_summary(self, cam: _Cam, z: _Zone) -> dict:
        t = cam.last_t or 0.0
        ref, conc, w = z.dominant()
        total = sum(z.dir_hist)
        dwells = sorted(d for (_, d) in z.dwells)
        last60 = self._series(z.series, t, "zone", dens=bool(z.area))
        dens = None
        if z.area:
            p = z.rules.get("density", DEFAULTS["density"])
            d = z.last.get("density") or 0.0
            level = ("critical" if d >= p["critical_ppm2"] > 0 else "warning" if d >= p["warn_ppm2"] > 0 else "normal")
            dens = {"area_m2": round(z.area, 2), "ppm2": round(d, 3), "level": level,
                    "thresholds": {"warn_ppm2": p["warn_ppm2"], "critical_ppm2": p["critical_ppm2"]},
                    "rule_enabled": "density" in z.rules,
                    "calib_error_m": cam.calib_error}
        return {
            "id": z.id, "name": z.name, "kind": z.kind, "space": z.space, "criticality": z.criticality,
            "rules": sorted(z.rules),
            "occupancy": z.last.get("count", 0),
            "entries_total": z.entries, "exits_total": z.exits,
            "entries_60m": sum(r["entries"] for r in last60), "exits_60m": sum(r["exits"] for r in last60),
            "dwell": {"n": len(dwells), "p50_s": round(statistics.median(dwells), 1) if dwells else None,
                      "p90_s": round(dwells[int(0.9 * (len(dwells) - 1))], 1) if dwells else None,
                      "bins_s": list(DWELL_BINS), "counts": list(z.dwell_counts),
                      "note": "completed stays (entry -> last seen inside); tracker ID switches split stays"},
            "flow": {"bins": list(DIR_BINS),
                     "hist": [round(v / total, 3) if total else 0.0 for v in z.dir_hist],
                     "track_seconds": round(total, 1),
                     "dominant_deg": round(screen_angle(ref, z.space), 1) if ref else None,
                     "concentration": round(conc, 2), "half_life_s": self.half_life_s,
                     "frame": "map (x right, y up)" if z.space == "ground_m" else "image (as seen on screen)",
                     "permitted_deg": (round(screen_angle(z.direction, "ground_m"), 1) if z.direction else None)},
            "density": dens,
            "density_note": None if dens else NOTE_UNCAL,
            "schedule": z.zone.schedule.status(t) if z.zone.schedule else None,
            "rule_schedules": {r: s.status(t) for r, s in z.zone.rule_schedules.items()},
            "queue": self.queue_stats(z, t) if z.kind == "queue" else None,
            "series": last60,
        }

    def summary_zones(self) -> dict:
        with self.lock:
            cams = []
            for cam in self.cams.values():
                cams.append({"camera": cam.id, "space": cam.space, "calibrated": cam.space == "ground_m",
                             "calib_error_m": cam.calib_error, "t": cam.last_t, "warnings": list(cam.warnings),
                             "zones": [self.zone_summary(cam, z) for z in cam.zones]})
            return {"bucket_s": self.bucket_s, "series_minutes": self.series_minutes, "cameras": cams}

    def summary_lines(self) -> dict:
        with self.lock:
            cams = []
            for cam in self.cams.values():
                t = cam.last_t or 0.0
                lines = []
                for ln in cam.lines:
                    ser = self._series(ln.series, t, "line")
                    ev = None
                    if ln.events:
                        sch = ln.events["schedule"]
                        ev = {"direction": ln.events["direction"], "min_gap_s": ln.events["min_gap_s"],
                              "events_raised": ln.n_events, "schedule": sch.status(t) if sch else None}
                    lines.append({"id": ln.id, "name": ln.name, "space": ln.space,
                                  "points": [list(ln.a), list(ln.b)], "labels": dict(ln.labels),
                                  "counts": dict(ln.counts),
                                  "counts_60m": {"in": sum(r["in"] for r in ser), "out": sum(r["out"] for r in ser)},
                                  "series": ser, "events": ev})
                cams.append({"camera": cam.id, "t": cam.last_t, "lines": lines, "warnings": list(cam.warnings)})
            return {"bucket_s": self.bucket_s, "direction_rule": "in = crossing along n = (-dy, dx) of A->B "
                    "(image lines: towards the right-hand side looking from A to B)", "cameras": cams}
