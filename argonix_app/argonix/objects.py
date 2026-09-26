"""Unattended-object analyzer: a bag-class object that stays put while the person last associated with it moves away.

Runs as a `Site.analyzers` plug-in on the object detections the detector emits when `detector.extra_classes` is set
(COCO backpack 24, handbag 26, suitcase 28 by default; `objects.enabled` adds them automatically). Objects are tracked
by their own tracker (IDs offset by `detect.OBJECT_ID_OFFSET`) and never reach the person rules.

State machine per object candidate (an object track, re-joined by box overlap when the tracker loses it):
  MOVING      the object's box centre moved more than `still_px` / `still_frac` x box diagonal from its anchor
  STATIONARY  it stayed within that tolerance; the drop time is the start of this still period
  ATTENDED    the associated person (nearest person at drop time, within `owner_max_dist_m`) is within
              `leave_dist_m`, or the object is inside a person box (held/worn), or another person has stayed within
              `near_dist_m` for >= `attend_s` (that person becomes the last associated person: handles ID switches)
  UNATTENDED  stationary >= `static_s` and not attended, continuously for >= `unattended_s` -> ONE event

The event reports the object, how long it has been stationary, and the last associated person track as a pointer for
the operator to review. It is not an accusation and says nothing about intent: people put bags down for many
reasons, and a missed detection or a tracker ID switch can make an attended bag look unattended.

Distances: ground metres when the camera is calibrated (object bottom-centre vs person foot point); otherwise an
approximation from the image (pixel distance / person box height x 1.70 m), labelled as such.
"""
from __future__ import annotations

import itertools
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from .geometry import apply_h, foot_point, point_in_polygon, signed_depth
from .models import Event
from .rules import register_rule

RULE = "unattended_object"
register_rule(RULE, "Object left unattended", 2)
DEFAULT_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
PERSON_H_M = 1.70

_EV_SEQ = itertools.count(1)  # unique evidence file names, even for events in the same millisecond


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


@dataclass
class _Cand:
    key: int
    first_seen: float
    last_seen: float
    box: tuple
    anchor: tuple
    still_since: float
    labels: Counter = field(default_factory=Counter)
    track_ids: set = field(default_factory=set)
    confs: deque = field(default_factory=lambda: deque(maxlen=30))
    owner: Optional[int] = None
    owner_dist0: Optional[float] = None
    owner_seen: Optional[float] = None       # last time the associated person was visible
    owner_dist: Optional[float] = None       # latest distance to the associated person
    owner_how: str = ""
    near_since: dict = field(default_factory=dict)  # tid -> first time within near_dist (continuous)
    unattended_since: Optional[float] = None
    fired_at: Optional[float] = None
    drop_jpeg: Optional[bytes] = None
    state: str = "moving"
    ground: Optional[tuple] = None
    dist_method: str = ""

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0] if self.labels else "object"


class UnattendedObjectAnalyzer:
    name = "objects"

    def __init__(self, labels=None, min_conf: float = 0.3, static_s: float = 10.0, unattended_s: float = 30.0,
                 leave_dist_m: float = 3.0, owner_max_dist_m: float = 2.0, near_dist_m: float = 1.0,
                 attend_s: float = 3.0, require_owner: bool = True, forget_s: float = 15.0, still_px: float = 6.0,
                 still_frac: float = 0.25, max_box_frac: float = 0.2, rearm_s: float = 300.0, criticality: int = 2,
                 owner_lookback_s: float = 3.0, save_crops: bool = True):
        self.labels = set(labels or DEFAULT_CLASSES.values())
        self.min_conf, self.static_s, self.unattended_s = float(min_conf), float(static_s), float(unattended_s)
        self.leave_dist, self.owner_max = float(leave_dist_m), float(owner_max_dist_m)
        self.near_dist, self.attend_s = float(near_dist_m), float(attend_s)
        self.require_owner, self.forget_s = bool(require_owner), float(forget_s)
        self.still_px, self.still_frac, self.max_box_frac = float(still_px), float(still_frac), float(max_box_frac)
        self.rearm_s, self.criticality = float(rearm_s), int(criticality)
        self.lookback, self.save_crops = float(owner_lookback_s), bool(save_crops)
        self.lock = threading.RLock()
        self.cams: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.stats = {"object_dets": 0, "candidates": 0, "events": 0}

    # ---- geometry ----
    def _ground(self, worker, pt_analytics) -> Optional[tuple]:
        H = getattr(worker, "H", None)
        if H is None:
            return None
        s = float(getattr(worker, "scale", 1.0) or 1.0)
        try:
            return apply_h(H, (pt_analytics[0] * s, pt_analytics[1] * s))
        except ValueError:
            return None

    def _dist(self, worker, cand: _Cand, person) -> tuple[float, str]:
        """Distance object -> person in metres: ground plane when calibrated, else approximate from box height."""
        tid, box, g = person
        if cand.ground is not None and g is not None:
            return math.dist(cand.ground, g), "ground plane (calibrated)"
        ob = foot_point(cand.box)
        pf = foot_point(box)
        h = max(box[3] - box[1], 1.0)
        return math.dist(ob, pf) / h * PERSON_H_M, "image approx. (px / person box height x 1.70 m)"

    @staticmethod
    def _held(cand: _Cand, box) -> bool:
        cx, cy = _centre(cand.box)
        mx, my = 0.1 * (box[2] - box[0]), 0.05 * (box[3] - box[1])
        return box[0] - mx <= cx <= box[2] + mx and box[1] - my <= cy <= box[3] + my

    def _zone(self, worker, cand: _Cand):
        if cand.ground is not None:
            for z in getattr(worker, "zones", []) or []:
                if signed_depth(cand.ground, z.polygon) >= 0:
                    return z
        s = float(getattr(worker, "scale", 1.0) or 1.0)
        fp = foot_point(cand.box)
        for z in getattr(worker, "px_zones", []) or []:
            if point_in_polygon((fp[0] * s, fp[1] * s), z.polygon):
                return z
        return None

    # ---- main ----
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        with self.lock:
            return self._step(worker, t, frame, dets, obs)

    def _step(self, worker, t, frame, dets, obs) -> list[Event]:
        st = self.cams.setdefault(worker.id, {"cands": {}, "people": deque(), "t": None})
        st["t"] = t
        ground = {o.track_id: o.ground for o in obs if o.cls == "person"}
        people = [(d[0], tuple(float(v) for v in d[1]), ground.get(d[0])) for d in dets if d[2] == "person"]
        st["people"].append((t, people))
        while st["people"] and t - st["people"][0][0] > self.lookback + 1.0:
            st["people"].popleft()
        fh, fw = (frame.shape[:2] if frame is not None else (1080, 1920))
        objs = []
        for tid, bb, cls, conf in dets:
            if cls not in self.labels or conf < self.min_conf:
                continue
            bb = tuple(float(v) for v in bb)
            if (bb[2] - bb[0]) * (bb[3] - bb[1]) > self.max_box_frac * fw * fh:
                continue
            objs.append((tid, bb, cls, conf))
        self.stats["object_dets"] += len(objs)
        cands: dict[int, _Cand] = st["cands"]

        # associate detections -> candidates: same object track id first, then box overlap
        used: set[int] = set()
        for tid, bb, cls, conf in sorted(objs, key=lambda o: -o[3]):
            c = next((c for c in cands.values() if tid in c.track_ids and c.key not in used), None)
            if c is None:
                best = max(((iou(bb, c.box), c) for c in cands.values() if c.key not in used),
                           key=lambda x: x[0], default=(0.0, None))
                c = best[1] if best[0] >= 0.3 else None
            if c is None:
                c = _Cand(next(self._ids), t, t, bb, bb, t)
                c.ground = self._ground(worker, foot_point(bb))
                cands[c.key] = c
                self.stats["candidates"] += 1
                self._assign_owner(worker, st, c, t)
            used.add(c.key)
            c.track_ids.add(tid)
            c.labels[cls] += 1
            c.confs.append(conf)
            c.box, c.last_seen = bb, t
            c.ground = self._ground(worker, foot_point(bb))
            ac, bc = _centre(c.anchor), _centre(bb)
            diag = math.hypot(c.anchor[2] - c.anchor[0], c.anchor[3] - c.anchor[1])
            if math.dist(ac, bc) > max(self.still_px, self.still_frac * diag):
                c.anchor, c.still_since, c.state = bb, t, "moving"   # moved: a new drop time
                c.unattended_since, c.drop_jpeg, c.near_since = None, None, {}
                self._assign_owner(worker, st, c, t)
        for k in [k for k, c in cands.items() if t - c.last_seen > self.forget_s]:
            del cands[k]

        events = []
        for c in cands.values():
            ev = self._update(worker, st, c, t, frame, people)
            if ev:
                events.append(ev)
        return events

    def _assign_owner(self, worker, st, c: _Cand, t: float) -> None:
        """Owner = person nearest to the object around the drop time (current frame and the last few seconds)."""
        best = None
        for tt, people in st["people"]:
            if t - tt > self.lookback:
                continue
            for p in people:
                d, how = self._dist(worker, c, p)
                if self._held(c, p[1]):
                    d = min(d, 0.0)
                if d <= self.owner_max and (best is None or d < best[0] or (d == best[0] and tt > best[2])):
                    best = (d, p[0], tt, how)
        if best is not None:
            c.owner, c.owner_dist0, c.owner_seen, c.owner_how = best[1], round(best[0], 2), best[2], best[3]
        elif c.owner is None:
            c.owner_dist0 = None

    def _update(self, worker, st, c: _Cand, t: float, frame, people) -> Optional[Event]:
        if t - c.last_seen > 1.0:  # not detected right now (occluded or gone): keep timers, do not decide
            return None
        still_for = t - c.still_since
        if still_for < self.static_s:
            c.state = "moving" if still_for < 1.0 else "stationary"
            if still_for >= 1.0 and c.drop_jpeg is None and self.save_crops and frame is not None:
                c.drop_jpeg = self._crop(frame, c, people)
            return None
        # attended?
        attended, by = False, None
        vis = {p[0]: p for p in people}
        if c.owner is not None and c.owner in vis:
            d, how = self._dist(worker, c, vis[c.owner])
            c.owner_dist, c.owner_seen, c.dist_method = round(d, 2), t, how
            if d <= self.leave_dist or self._held(c, vis[c.owner][1]):
                attended, by = True, c.owner
        for tid, p in vis.items():
            d, how = self._dist(worker, c, p)
            if d <= self.near_dist or self._held(c, p[1]):
                c.near_since.setdefault(tid, t)
                if t - c.near_since[tid] >= self.attend_s and not attended:
                    attended, by = True, tid
            else:
                c.near_since.pop(tid, None)
        for tid in [k for k in c.near_since if k not in vis]:
            del c.near_since[tid]
        if attended and by is not None and by != c.owner:  # someone else stayed with it: they are now associated
            c.owner, c.owner_how = by, "stayed next to the object"
            c.owner_dist0 = c.owner_dist = round(self._dist(worker, c, vis[by])[0], 2)
            c.owner_seen = t
        if attended:
            c.state, c.unattended_since = "attended", None
            return None
        if c.owner is None and self.require_owner:
            c.state = "stationary (no associated person)"
            return None
        c.state = "unattended"
        if c.unattended_since is None:
            c.unattended_since = t
        if t - c.unattended_since < self.unattended_s:
            return None
        if c.fired_at is not None and t - c.fired_at < self.rearm_s:
            return None
        c.fired_at = t
        c.state = "unattended (alerted)"
        self.stats["events"] += 1
        return self._event(worker, c, t, frame, people)

    def _crop(self, frame, c: _Cand, people, pad: float = 1.5) -> Optional[bytes]:
        boxes = [c.box] + [p[1] for p in people if p[0] == c.owner]
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        w, h = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        x1, y1 = int(max(0, x1 - pad * w / 2)), int(max(0, y1 - pad * h / 2))
        x2, y2 = int(min(W, x2 + pad * w / 2)), int(min(H, y2 + pad * h / 2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2].copy()
        bx = [int(v) for v in c.box]
        cv2.rectangle(crop, (bx[0] - x1, bx[1] - y1), (bx[2] - x1, bx[3] - y1), (0, 200, 255), 2)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def _event(self, worker, c: _Cand, t: float, frame, people) -> Event:
        z = self._zone(worker, c)
        zid, zname = (z.id, z.name) if z else ("_view", f"{getattr(worker, 'name', worker.id)} (whole view)")
        crit = z.criticality if z else self.criticality
        still_for = t - c.still_since
        owner_txt = "no person was associated with it"
        if c.owner is not None:
            seen_ago = t - (c.owner_seen or t)
            where = (f"{c.owner_dist:.1f} m away" if c.owner_dist is not None and seen_ago < 1.0
                     else f"not visible for {seen_ago:.0f} s")
            owner_txt = (f"last associated person track #{c.owner} ({c.owner_dist0:.1f} m from it at the drop time) "
                         f"is now {where}")
        obj_id = min(c.track_ids) if c.track_ids else c.key
        obs = (f"{c.label.capitalize()} (object track #{obj_id}, {worker.id}) stationary for {still_for:.0f} s, "
               f"unattended for {t - c.unattended_since:.0f} s; {owner_txt}. Review the evidence: the person may "
               "simply have set it down")
        conf = sum(c.confs) / len(c.confs) if c.confs else 0.3
        evidence = {}
        if self.save_crops and frame is not None:
            data = getattr(getattr(worker, "site", None), "data", None)
            if data is not None:
                stem = f"{worker.id}_{RULE}_{c.label}_{int(t * 1000)}_{next(_EV_SEQ)}"
                (Path(data) / "evidence").mkdir(parents=True, exist_ok=True)
                now = self._crop(frame, c, people)
                if now:
                    (Path(data) / "evidence" / f"{stem}_now.jpg").write_bytes(now)
                    evidence["crop"] = f"evidence/{stem}_now.jpg"
                if c.drop_jpeg:
                    (Path(data) / "evidence" / f"{stem}_drop.jpg").write_bytes(c.drop_jpeg)
                    evidence["drop_crop"] = f"evidence/{stem}_drop.jpg"
        deriv = {
            "rule": RULE, "object_label": c.label, "object_labels_seen": dict(c.labels),
            "object_track_ids": sorted(c.track_ids), "box_xyxy_analytics_px": [round(v, 1) for v in c.box],
            "stationary_s": round(still_for, 1), "unattended_s": round(t - c.unattended_since, 1),
            "drop_time": c.still_since, "last_associated_person_track": c.owner,
            "owner_association": c.owner_how or None, "owner_dist_at_drop_m": c.owner_dist0,
            "owner_dist_now_m": c.owner_dist, "distance_method": c.dist_method or c.owner_how,
            "threshold": {"static_s": self.static_s, "unattended_s": self.unattended_s,
                          "leave_dist_m": self.leave_dist, "owner_max_dist_m": self.owner_max,
                          "near_dist_m": self.near_dist, "attend_s": self.attend_s,
                          "require_owner": self.require_owner, "min_conf": self.min_conf},
            "duration_s": round(t - c.still_since, 1), "calib_error_m": getattr(worker, "calib_error", None),
            "confidence_method": "mean object-detector conf over recent detections (uncalibrated)",
            "note": "the associated person is a pointer for review, not an accusation; intent is unknown",
            "analyzer": "objects-v1",
        }
        return Event(RULE, worker.id, zid, zname, [obj_id], c.still_since, t, obs, deriv, round(conf, 3), 2, crit,
                     getattr(z, "site_zone", None) if z else None,
                     (round(c.ground[0], 2), round(c.ground[1], 2)) if c.ground else None, evidence)

    # ---- API ----
    def summary(self) -> dict:
        with self.lock:
            cams = []
            for cam, st in self.cams.items():
                t = st["t"] or time.time()
                cams.append({"camera": cam, "objects": [
                    {"key": c.key, "label": c.label, "state": c.state, "track_ids": sorted(c.track_ids),
                     "stationary_s": round(t - c.still_since, 1), "box": [round(v, 1) for v in c.box],
                     "ground": [round(c.ground[0], 2), round(c.ground[1], 2)] if c.ground else None,
                     "last_associated_person_track": c.owner, "owner_dist_m": c.owner_dist,
                     "unattended_s": round(t - c.unattended_since, 1) if c.unattended_since else None,
                     "alerted": c.fired_at is not None}
                    for c in sorted(st["cands"].values(), key=lambda c: c.first_seen)]})
            return {"labels": sorted(self.labels), "stats": dict(self.stats),
                    "params": {"static_s": self.static_s, "unattended_s": self.unattended_s,
                               "leave_dist_m": self.leave_dist, "require_owner": self.require_owner},
                    "cameras": cams}
