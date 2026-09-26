"""Deterministic, zone-scoped behaviour rules over ground-plane tracks.

Each rule is a condition that must hold continuously for `min_seconds` (with a short grace period for
tracker flicker) before it fires once per episode. Rules describe observable behaviour only.

A zone (or one of its rules) may carry a `schedule` (argonix/schedule.py): the rule only arms inside an active window,
and its events state the window that was active. A rule's own schedule replaces the zone's schedule for that rule.

Vehicle rules (`vehicle_in_zone`, `vehicle_stopped`) run on the vehicle tracks the detector emits when the site's
`vehicles.enabled` is set (COCO car/motorcycle/bus/truck via `detector.extra_classes`). Vehicle tracks never enter the
person rules, counts or crowding.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .geometry import signed_depth
from .models import Event, TrackObs
from .schedule import Schedule

SEVERITY = {"intrusion": 3, "wrong_way": 2, "running": 2, "crowding": 2, "loitering": 1, "camera_health": 2,
            "occupancy_deviation": 1}
# zone_rearm_s (per-track rules, 0 = off): after the rule fires in a zone, the same rule in that zone stays quiet for
# this many seconds whatever the track id. For cameras where the tracker keeps losing people (occlusion, low
# mounting) and a new id would re-fire the same situation; a second real person inside the window is not reported.
DEFAULTS = {
    "intrusion": {"min_seconds": 1.0, "zone_rearm_s": 0.0},
    "loitering": {"min_seconds": 20.0, "zone_rearm_s": 0.0},
    "running": {"speed_mps": 2.5, "min_seconds": 1.0, "zone_rearm_s": 0.0},
    "wrong_way": {"min_seconds": 1.5, "cos_threshold": -0.5, "min_speed_mps": 0.4, "zone_rearm_s": 0.0},
    "crowding": {"max_count": 5, "min_seconds": 3.0},
    # a vehicle's foot point (bottom centre of its box) inside a pedestrian-only zone for min_seconds
    "vehicle_in_zone": {"min_seconds": 2.0},
    # a vehicle inside a no-stopping zone that moved less than still_m metres (calibrated) or still_box_frac x its box
    # size (uncalibrated) over the last window_s seconds, continuously for min_seconds
    "vehicle_stopped": {"min_seconds": 60.0, "window_s": 3.0, "still_m": 1.0, "still_box_frac": 0.15},
}
# COCO class ids of vehicles (the ids `vehicles.classes` may list without naming them)
COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat"}
DEFAULT_VEHICLE_CLASSES = (2, 3, 5, 7)
VEHICLE_RULES = ("vehicle_in_zone", "vehicle_stopped")
LABEL = {"intrusion": "Restricted-zone entry", "loitering": "Extended dwell", "running": "Fast movement",
         "wrong_way": "Movement against permitted direction", "crowding": "Occupancy above limit",
         "camera_health": "Camera view problem", "occupancy_deviation": "Unusual occupancy for this hour"}



def register_rule(name: str, label: str, severity: int) -> None:
    """Register an analyzer-produced event type so correlation, priority and the UI know it."""
    SEVERITY[name] = int(severity)
    LABEL[name] = label


register_rule("vehicle_in_zone", "Vehicle in pedestrian-only zone", 2)
register_rule("vehicle_stopped", "Vehicle stopped in no-stopping zone", 1)


def vehicle_classes(cfg: dict) -> dict[int, str]:
    """{class id: label} of the vehicle classes enabled by the site config's `vehicles` section ({} when off).
    `classes` is a list of COCO ids (named from COCO_VEHICLES) or a {"id": "label"} object for other models."""
    vc = (cfg or {}).get("vehicles") or {}
    if not vc.get("enabled"):
        return {}
    cls = vc.get("classes") or list(DEFAULT_VEHICLE_CLASSES)
    if isinstance(cls, dict):
        return {int(k): str(v) for k, v in cls.items()}
    out = {}
    for c in cls:
        c = int(c)
        if c not in COCO_VEHICLES:
            raise ValueError(f"vehicles.classes: id {c} is not a COCO vehicle class {COCO_VEHICLES}; name it "
                             f'explicitly, e.g. "classes": {{"{c}": "label"}}')
        out[c] = COCO_VEHICLES[c]
    return out


@dataclass
class Zone:
    id: str
    name: str
    polygon: list[tuple[float, float]]
    rules: dict[str, dict[str, Any]]
    criticality: int = 1
    direction: Optional[tuple[float, float]] = None  # permitted walking direction (ground frame)
    site_zone: Optional[str] = None                   # shared id for zones seen by several cameras
    schedule: Optional[Schedule] = None               # zone-wide arming schedule (None = always armed)
    rule_schedules: dict[str, Schedule] = field(default_factory=dict)  # per-rule schedules (replace the zone's)

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        """Build a zone from its config. Raises ValueError on an invalid schedule (the message names the zone)."""
        zid = d["id"]
        rules, rule_sched = {}, {}
        for k, v in (d.get("rules") or {}).items():
            p = dict(v or {})
            sch = Schedule.parse(p.pop("schedule", None), f"zone '{zid}' rule '{k}' schedule")
            if sch is not None:
                rule_sched[k] = sch
            rules[k] = {**DEFAULTS.get(k, {}), **p}
        direction = tuple(d["direction"]) if d.get("direction") else None
        return cls(zid, d.get("name", zid), [tuple(p) for p in d["polygon"]], rules,
                   int(d.get("criticality", 1)), direction, d.get("site_zone"),
                   Schedule.parse(d.get("schedule"), f"zone '{zid}' schedule"), rule_sched)

    def schedule_for(self, rule: str) -> tuple[Optional[Schedule], str]:
        if rule in self.rule_schedules:
            return self.rule_schedules[rule], "rule"
        return self.schedule, "zone"

    def armed(self, rule: str, t: float) -> bool:
        """True when `rule` may arm at time t (no schedule, or inside one of its windows)."""
        sch, _ = self.schedule_for(rule)
        return sch is None or sch.active(t)

    def schedule_info(self, rule: str, t: float) -> Optional[dict]:
        """The schedule window active at t for `rule` (for an event's derivation); None without a schedule."""
        sch, src = self.schedule_for(rule)
        return sch.describe(t, src) if sch is not None else None


@dataclass
class _Episode:
    first: float
    last: float
    fired: bool = False
    confs: list[float] = field(default_factory=list)
    peak: float = 0.0


class RuleEngine:
    def __init__(self, camera_id: str, zones: list[Zone], calib_error_m: float = 0.3,
                 grace_s: float = 2.0, stale_s: float = 3.0, classes: tuple[str, ...] = ("person",),
                 rearm_s: float = 30.0, units: str = "m", vehicle_classes: tuple[str, ...] = ()):
        """units="m": zones/positions on the calibrated ground plane (metres), margin = calibration error.
        units="px": uncalibrated camera; zones/positions in source-image pixels and margin in pixels. The metric
        rules (running, wrong_way) are skipped in px mode: pixel speed/direction depend on depth in the image.
        vehicle_classes: detector labels treated as vehicles (only the vehicle rules see them).
        Exception: a running rule configured with `speed_bh_s` runs in px mode on the foot-point speed divided by
        the track's median box height (body heights per second, a local, perspective-normalised proxy; e.g. 1.5
        bh/s ~ 2.5 m/s for a 1.7 m person). It is labelled as such in the event derivation."""
        if units not in ("m", "px"):
            raise ValueError('units must be "m" or "px"')
        self.units = units
        self.camera_id = camera_id
        self.zones = zones
        self.margin = calib_error_m
        self.grace = grace_s
        self.rearm = rearm_s
        self.last_fired: dict[tuple, float] = {}
        self.stale = stale_s
        self.classes = classes
        self.vehicle_classes = tuple(c for c in vehicle_classes if c not in classes)
        self.hist: dict[int, deque] = {}
        self.last_seen: dict[int, float] = {}
        self.vhist: dict[int, deque] = {}  # vehicle track -> (t, position in the engine's space, bbox)
        self.eps: dict[tuple, _Episode] = {}
        # (rule, zone id) -> zone_rearm_s, for the zones that set it; zone_fired: when that rule last fired there
        self.zone_rearm = {(r, z.id): float(p.get("zone_rearm_s") or 0) for z in zones for r, p in z.rules.items()
                           if isinstance(p, dict) and float(p.get("zone_rearm_s") or 0) > 0}
        self.zone_fired: dict[tuple, float] = {}
        # optional learned-from-this-site dwell threshold for loitering: (zone id, t, quantile) -> (seconds, stays)
        # (the pipeline wires it to Baseline.dwell_quantile; see dwell_threshold)
        self.dwell_baseline: Optional[Callable[[str, float, float], tuple[Optional[float], int]]] = None
        self.heights: dict[int, deque] = {}  # px mode: box heights in source px (for speed_bh_s)

    # ---- kinematics -------------------------------------------------------------------------
    def velocity(self, tid: int, window_s: float = 1.0) -> Optional[tuple[float, float]]:
        h = self.hist.get(tid)
        if not h or len(h) < 2:
            return None
        t1, p1, _ = h[-1]
        old = [e for e in h if t1 - e[0] <= window_s]
        t0, p0, _ = old[0]
        if t1 - t0 < 0.4:
            return None
        return ((p1[0] - p0[0]) / (t1 - t0), (p1[1] - p0[1]) / (t1 - t0))

    # ---- episode bookkeeping ----------------------------------------------------------------
    def dwell_threshold(self, zone_id: str, p: dict, t: float) -> tuple[float, dict]:
        """Loitering threshold for this zone now. Plain `min_seconds` unless the rule sets `baseline_quantile`
        (e.g. 0.95): then max(min_seconds, baseline_factor x that quantile of the stays recorded in this zone in
        this hour of the week), once at least `baseline_min_stays` stays exist. A busy waiting area then fires on
        stays that are long *for that place and hour*, a quiet corridor on the configured minimum."""
        base = float(p["min_seconds"])
        q = p.get("baseline_quantile")
        if not q or self.dwell_baseline is None:
            return base, {"min_seconds": base}
        need = int(p.get("baseline_min_stays", 30))
        try:
            val, n = self.dwell_baseline(zone_id, t, float(q))
        except Exception:  # noqa: BLE001 - a statistics failure falls back to the configured minimum
            val, n = None, 0
        if val is None or n < need:
            return base, {"min_seconds": base, "dwell_baseline": f"not ready ({n} of {need} stays recorded here "
                                                                  "in this hour of the week)"}
        factor = float(p.get("baseline_factor", 1.0))
        thr = max(base, factor * float(val))
        return thr, {"min_seconds": round(thr, 1), "configured_min_seconds": base,
                     "dwell_baseline": {"quantile": float(q), "seconds": round(float(val), 1), "factor": factor,
                                        "stays": n, "scope": "this zone, this hour of the week"}}

    def _tick(self, key: tuple, t: float, cond: bool, conf: float, value: float, min_s: float) -> Optional[_Episode]:
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
                    return None  # same track/zone/rule re-entered within the re-arm window: no duplicate event
                zr = self.zone_rearm.get(key[:2]) if len(key) == 3 else None
                if zr and t - self.zone_fired.get(key[:2], -1e18) < zr:
                    return None  # the same situation under a new track id (tracker lost the person): zone re-arm
                self.last_fired[key] = t
                if zr:
                    self.zone_fired[key[:2]] = t
                return ep
        elif ep is not None and t - ep.last > self.grace:
            del self.eps[key]
        return None

    def _event(self, rule: str, zone: Zone, tids: list[int], ep: _Episode, obs: str, deriv: dict,
               near_boundary: bool, ground) -> Event:
        conf = sum(ep.confs) / len(ep.confs)
        if near_boundary:
            conf *= 0.8
        deriv = {**deriv, "rule": rule, "duration_s": round(ep.last - ep.first, 2),
                 "calib_error_m": self.margin, "confidence_method": "mean detector conf x boundary factor (uncalibrated)"}
        sched = zone.schedule_info(rule, ep.last)
        if sched:  # the rule was armed by a schedule: say which window
            deriv["schedule"] = sched
            obs = f"{obs} [{sched['text']}]"
        if self.units == "px":  # no ground plane: report image-space geometry, never a map position
            deriv.update(calib_error_m=None, space="image_px", margin_px=self.margin, foot_px=ground,
                         note="uncalibrated camera: zone test on image pixels, no ground position")
            ground = None
        return Event(rule, self.camera_id, zone.id, zone.name, tids, ep.first, ep.last, obs, deriv,
                     round(conf, 3), SEVERITY[rule], zone.criticality, zone.site_zone, ground)

    def vehicle_motion(self, tid: int, window_s: float) -> Optional[tuple[float, float]]:
        """(largest displacement from the current position over the last window_s, box size) for a vehicle track,
        or None until the history covers the window. Metres on the ground plane when calibrated; when uncalibrated,
        the box centre in analytics pixels and the box's larger side (so the ratio is resolution-independent)."""
        h = self.vhist.get(tid)
        if not h or h[-1][0] - h[0][0] < 0.8 * window_s:
            return None
        t1, p1, b1 = h[-1]
        recent = [e for e in h if t1 - e[0] <= window_s]
        if self.units == "m":
            return max(math.dist(p1, e[1]) for e in recent), 0.0
        c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        size = max(b1[2] - b1[0], b1[3] - b1[1], 1.0)
        return max(math.dist(c(b1), c(e[2])) for e in recent), size

    def _vehicle_rules(self, t: float, z: Zone, vobs: list[TrackObs], armed: dict) -> list[Event]:
        events: list[Event] = []
        u = self.units
        for o in vobs:
            d = signed_depth(o.ground, z.polygon)
            g = (round(o.ground[0], 2), round(o.ground[1], 2))
            tag = f"{o.cls} track #{o.track_id} ({self.camera_id})"
            if "vehicle_in_zone" in z.rules:
                p = z.rules["vehicle_in_zone"]
                ep = self._tick(("vehicle_in_zone", z.id, o.track_id), t, d >= self.margin and armed["vehicle_in_zone"],
                                o.conf, d, p["min_seconds"])
                if ep:
                    events.append(self._event("vehicle_in_zone", z, [o.track_id], ep,
                        f"{tag} inside pedestrian-only '{z.name}' for {ep.last - ep.first:.1f}s, "
                        f"{ep.peak:.2f} {u} past boundary",
                        {"threshold": {"min_seconds": p["min_seconds"], f"margin_{u}": self.margin},
                         f"depth_{u}": round(d, 2), "object_class": o.cls,
                         "position_note": "vehicle position = bottom centre of its box (approximate on the ground)"},
                        d < 2 * self.margin, g))
            if "vehicle_stopped" in z.rules:
                p = z.rules["vehicle_stopped"]
                mot = self.vehicle_motion(o.track_id, p["window_s"])
                still, moved = False, None
                if mot is not None:
                    disp, size = mot
                    moved = disp if u == "m" else disp / size
                    still = moved <= (p["still_m"] if u == "m" else p["still_box_frac"])
                ep = self._tick(("vehicle_stopped", z.id, o.track_id), t, d >= 0 and still and armed["vehicle_stopped"],
                                o.conf, p["window_s"], p["min_seconds"])
                if ep:
                    held = ep.last - ep.first + p["window_s"]
                    how = (f"moved <= {p['still_m']} m" if u == "m" else f"moved <= {p['still_box_frac']:.0%} of its box size")
                    events.append(self._event("vehicle_stopped", z, [o.track_id], ep,
                        f"{tag} stationary in no-stopping '{z.name}' for about {held:.0f}s ({how} per {p['window_s']:g}s)",
                        {"threshold": {k: p[k] for k in ("min_seconds", "window_s", "still_m", "still_box_frac")},
                         "stillness_test": ("max displacement of the ground position over window_s (metres)" if u == "m"
                                            else "max displacement of the box centre over window_s / box's larger side"),
                         ("displacement_m" if u == "m" else "displacement_box_frac"): round(moved, 3),
                         "stationary_s": round(held, 1), "object_class": o.cls},
                        d < self.margin, g))
        return events

    # ---- main update ------------------------------------------------------------------------
    def update(self, t: float, observations: list[TrackObs]) -> list[Event]:
        events: list[Event] = []
        obs = [o for o in observations if o.cls in self.classes and o.ground is not None]
        vobs = [o for o in observations if o.cls in self.vehicle_classes and o.ground is not None]
        for o in obs:
            self.hist.setdefault(o.track_id, deque(maxlen=64)).append((t, o.ground, o.conf))
            self.last_seen[o.track_id] = t
            if self.units == "px" and o.bbox[3] > 0:
                # bbox is in analytics px, ground = foot point in source px: foot y = bbox y2 * scale
                self.heights.setdefault(o.track_id, deque(maxlen=16)).append(
                    (o.bbox[3] - o.bbox[1]) * o.ground[1] / o.bbox[3])
        for o in vobs:
            self.vhist.setdefault(o.track_id, deque(maxlen=256)).append((t, o.ground, o.bbox))
            self.last_seen[o.track_id] = t
        for tid in [k for k, v in self.last_seen.items() if t - v > self.stale]:
            self.hist.pop(tid, None); self.vhist.pop(tid, None); self.last_seen.pop(tid, None)
            self.heights.pop(tid, None)
        # An episode ends only when its track is seen again with the condition false; a track that leaves the view
        # never is. Drop per-track episodes of gone tracks and re-arm stamps past the re-arm window, or both grow by
        # an entry per visitor forever (and the pipeline scans the episodes every frame).
        for k in [k for k, ep in self.eps.items()
                  if len(k) == 3 and k[2] not in self.last_seen and t - ep.last > self.grace]:
            del self.eps[k]
        for k in [k for k, v in self.last_fired.items() if t - v > self.rearm]:
            del self.last_fired[k]
        for k in [k for k, v in self.zone_fired.items() if t - v > self.zone_rearm.get(k, 0)]:
            del self.zone_fired[k]

        metric, u = self.units == "m", self.units
        for z in self.zones:
            armed = {r: z.armed(r, t) for r in z.rules}  # schedules: a rule outside its window cannot arm
            if vobs and any(r in z.rules for r in VEHICLE_RULES):
                events += self._vehicle_rules(t, z, vobs, armed)
            inside_ids = []
            for o in obs:
                d = signed_depth(o.ground, z.polygon)
                if d >= 0:
                    inside_ids.append(o.track_id)
                v = self.velocity(o.track_id)
                speed = math.hypot(*v) if v else 0.0
                g = (round(o.ground[0], 2), round(o.ground[1], 2))
                tag = f"{o.cls} track #{o.track_id} ({self.camera_id})"

                if "intrusion" in z.rules:
                    p = z.rules["intrusion"]
                    # conservative boundary: must be inside by at least the calibration error
                    ep = self._tick(("intrusion", z.id, o.track_id), t, d >= self.margin and armed["intrusion"], o.conf, d, p["min_seconds"])
                    if ep:
                        events.append(self._event("intrusion", z, [o.track_id], ep,
                            f"{tag} inside '{z.name}' for {ep.last - ep.first:.1f}s, {ep.peak:.2f} {u} past boundary",
                            {"threshold": {"min_seconds": p["min_seconds"], f"margin_{u}": self.margin},
                             f"depth_{u}": round(d, 2)},
                            d < 2 * self.margin, g))
                if "loitering" in z.rules:
                    p = z.rules["loitering"]
                    min_s, thr = self.dwell_threshold(z.id, p, t)
                    ep = self._tick(("loitering", z.id, o.track_id), t, d >= 0 and armed["loitering"], o.conf, d, min_s)
                    if ep:
                        usual = thr.get("dwell_baseline")
                        usual = (f" (usual here at this hour: p{round(usual['quantile'] * 100)} "
                                 f"{usual['seconds']:.0f}s over {usual['stays']} stays)") if isinstance(usual, dict) else ""
                        events.append(self._event("loitering", z, [o.track_id], ep,
                            f"{tag} remained in '{z.name}' for {ep.last - ep.first:.1f}s{usual}",
                            {"threshold": thr}, d < self.margin, g))
                if "running" in z.rules and metric:
                    p = z.rules["running"]
                    ep = self._tick(("running", z.id, o.track_id), t, d >= 0 and speed >= p["speed_mps"] and armed["running"],
                                    o.conf, speed, p["min_seconds"])
                    if ep:
                        events.append(self._event("running", z, [o.track_id], ep,
                            f"{tag} moving at up to {ep.peak:.1f} m/s in '{z.name}'",
                            {"threshold": {"speed_mps": p["speed_mps"], "min_seconds": p["min_seconds"]},
                             "speed_mps": round(speed, 2)}, False, g))
                elif "running" in z.rules and "speed_bh_s" in z.rules["running"]:  # px mode, opt-in
                    p = z.rules["running"]
                    hs = sorted(self.heights.get(o.track_id, ()))
                    h = hs[len(hs) // 2] if hs else 0.0
                    bhs = speed / h if h >= float(p.get("min_height_px", 16)) else 0.0
                    ep = self._tick(("running", z.id, o.track_id), t, d >= 0 and bhs >= p["speed_bh_s"],
                                    o.conf, bhs, p["min_seconds"])
                    if ep:
                        events.append(self._event("running", z, [o.track_id], ep,
                            f"{tag} moving at up to {ep.peak:.1f} body-heights/s in '{z.name}' (image space)",
                            {"threshold": {"speed_bh_s": p["speed_bh_s"], "min_seconds": p["min_seconds"]},
                             "speed_bh_s": round(bhs, 2), "box_height_px": round(h, 1),
                             "speed_method": "foot-point px/s divided by median box height (uncalibrated proxy)"},
                            False, g))
                if "wrong_way" in z.rules and z.direction and metric:
                    p = z.rules["wrong_way"]
                    cos = -1.0
                    cond = False
                    if d >= 0 and speed >= p["min_speed_mps"]:
                        dn = math.hypot(*z.direction)
                        cos = (v[0] * z.direction[0] + v[1] * z.direction[1]) / (speed * dn)
                        cond = cos <= p["cos_threshold"] and armed["wrong_way"]
                    ep = self._tick(("wrong_way", z.id, o.track_id), t, cond, o.conf, speed, p["min_seconds"])
                    if ep:
                        events.append(self._event("wrong_way", z, [o.track_id], ep,
                            f"{tag} moving against permitted direction in '{z.name}' ({ep.peak:.1f} m/s)",
                            {"threshold": {"cos_threshold": p["cos_threshold"], "min_seconds": p["min_seconds"]},
                             "cos_to_permitted": round(cos, 2)}, False, g))

            if "crowding" in z.rules:
                p = z.rules["crowding"]
                n = len(inside_ids)
                mean_conf = (sum(o.conf for o in obs if o.track_id in inside_ids) / n) if n else 0.0
                ep = self._tick(("crowding", z.id), t, n > p["max_count"] and armed["crowding"], mean_conf, n, p["min_seconds"])
                if ep:
                    events.append(self._event("crowding", z, sorted(inside_ids), ep,
                        f"{int(ep.peak)} person tracks in '{z.name}' (limit {p['max_count']}) for {ep.last - ep.first:.1f}s",
                        {"threshold": {"max_count": p["max_count"], "min_seconds": p["min_seconds"]}, "count": n},
                        False, None))
        return events
