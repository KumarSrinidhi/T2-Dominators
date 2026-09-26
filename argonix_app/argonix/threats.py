"""Weapon-like object analyzer: "object resembling a knife near person track #12", operator-verified.

Runs as a `Site.analyzers` plug-in. Every 1/`fps` s per camera it runs one or more object models on the analytics
frame and (optionally) on enlarged person crops, keeps detections whose canonical label is in `labels`, associates
each with the person track whose hand region overlaps it most, and keeps a k-of-n history per (label, track).
An `object_threat` event is emitted once per (label, track) episode when >= k of the last n checks saw the object at
>= min_conf. The event reports the OBJECT (and the person track it is near); it never labels the person.

Zero-training baseline: COCO YOLO26n classes knife (43), baseball bat (34), scissors (76). The fine-tuned model
(tools/train_weapons.py) adds handgun and a better knife class. Both are learned models: toy/replica guns, phones,
tools, umbrellas and sticks can look alike, and small or occluded objects are missed. Operators must verify.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .models import Event

RULE = "object_threat"
COCO_THREAT_CLASSES = {43: "knife", 34: "bat", 76: "scissors"}
PHRASE = {"knife": "a knife", "handgun": "a handgun", "bat": "a bat or stick", "scissors": "scissors",
          "weapon": "a weapon"}


@dataclass
class ObjDet:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # xyxy in analytics-frame px
    model: str
    source: str = "frame"                   # "frame" | "person_crop"


_CROP_SEQ = itertools.count(1)  # two crops on one frame must never share a file


def write_crop(worker, frame, box, marks, tag: str, t: float, pad: float = 0.35, min_side: int = 96) -> Optional[str]:
    """Save a JPEG crop of `frame` around `box` (xyxy, padded) with thin rectangles on `marks`; returns the path
    relative to the site data dir (``evidence/...``), or None if the crop is empty."""
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    px, py = max(bw * pad, (min_side - bw) / 2, 0), max(bh * pad, (min_side - bh) / 2, 0)
    X1, Y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    X2, Y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if X2 - X1 < 2 or Y2 - Y1 < 2:
        return None
    crop = frame[Y1:Y2, X1:X2].copy()
    for m in marks or []:
        cv2.rectangle(crop, (int(m[0] - X1), int(m[1] - Y1)), (int(m[2] - X1), int(m[3] - Y1)), (0, 200, 255), 1)
    if crop.shape[0] < 160:  # upscale small crops so the operator can see them
        f = 160 / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * f), 160), interpolation=cv2.INTER_CUBIC)
    rel = f"evidence/{worker.id}_{tag}_{int(t * 1000)}_{next(_CROP_SEQ)}_crop.jpg"
    out = Path(worker.site.data) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rel


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    return inter / max(1e-6, ar(a) + ar(b) - inter)


def hand_region(pbox, margin: float = 0.35) -> tuple[float, float, float, float]:
    """Where a held object can appear: the person box widened by `margin` x width on each side (arms reach out),
    from 10% above the head down to 85% of the body height (hands rarely hold things at the feet)."""
    x1, y1, x2, y2 = pbox
    w, h = x2 - x1, y2 - y1
    return (x1 - margin * w, y1 - 0.1 * h, x2 + margin * w, y1 + 0.85 * h)


def associate(obj_box, people: list[tuple[int, tuple]], margin: float = 0.35) -> tuple[Optional[int], float]:
    """(track_id, score) of the person whose hand region covers most of the object box (intersection over the
    object's area); ties broken by distance from the object centre to the person's upper-body centre. (None, 0)
    when no hand region overlaps the object."""
    ox1, oy1, ox2, oy2 = obj_box
    oa = max(1e-6, (ox2 - ox1) * (oy2 - oy1))
    oc = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    best, best_key = None, (0.0, 0.0)
    for tid, pb in people:
        hx1, hy1, hx2, hy2 = hand_region(pb, margin)
        inter = max(0.0, min(ox2, hx2) - max(ox1, hx1)) * max(0.0, min(oy2, hy2) - max(oy1, hy1))
        ioa = inter / oa
        if ioa <= 0:
            continue
        pc = ((pb[0] + pb[2]) / 2, pb[1] + 0.4 * (pb[3] - pb[1]))
        dist = ((oc[0] - pc[0]) ** 2 + (oc[1] - pc[1]) ** 2) ** 0.5 / max(1.0, pb[3] - pb[1])
        key = (round(ioa, 3), -dist)
        if key > best_key or best is None:
            best, best_key = tid, key
    return best, (best_key[0] if best is not None else 0.0)


def nms(dets: list[ObjDet], iou: float = 0.5) -> list[ObjDet]:
    """Per-label greedy NMS (keeps the most confident of overlapping boxes, e.g. frame vs crop duplicates)."""
    keep: list[ObjDet] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if all(k.label != d.label or _iou(k.box, d.box) < iou for k in keep):
            keep.append(d)
    return keep


class UltralyticsObjectModel:
    """A YOLO detector restricted to `classes` {class_id: canonical label}. predict(images) -> [[ObjDet]] per image."""

    def __init__(self, weights: str, classes: dict, name: str = "", version: str = "", device: str = "auto",
                 imgsz: int = 640, conf: float = 0.25, crop_imgsz: int = 320, models_dir: Optional[Path] = None):
        from ultralytics import YOLO
        from .detect import resolve_device
        self.device = resolve_device(device)
        w = Path(weights)
        if not w.exists() and models_dir and (Path(models_dir) / w.name).exists():
            w = Path(models_dir) / w.name
        self.model = YOLO(str(w), task="detect")
        self.classes = {int(k): str(v) for k, v in classes.items()}
        meta = {}
        side = w.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except ValueError:
                meta = {}
        self.name = name or meta.get("name") or w.stem
        sha = hashlib.sha256(w.read_bytes()).hexdigest()[:8] if w.exists() else "download"
        self.version = version or meta.get("version") or f"sha-{sha}"
        self.imgsz, self.crop_imgsz, self.conf = int(imgsz), int(crop_imgsz), float(conf)
        self.lock = threading.Lock()

    @property
    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "classes": self.classes, "device": self.device}

    def predict(self, images: list, crop: bool = False) -> list[list[ObjDet]]:
        if not images:
            return []
        kw = dict(imgsz=self.crop_imgsz if crop else self.imgsz, conf=self.conf, classes=list(self.classes),
                  device=self.device, verbose=False, half=self.device.startswith("cuda"))
        with self.lock:  # shared by every camera thread
            res = self.model.predict(images, **kw)
        out = []
        for r in res:
            dets = []
            if r.boxes is not None and len(r.boxes):
                for xyxy, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    if c in self.classes:
                        dets.append(ObjDet(self.classes[c], float(cf), tuple(xyxy), self.name,
                                           "person_crop" if crop else "frame"))
            out.append(dets)
        return out


@dataclass
class _Hist:
    confs: deque
    boxes: deque
    first: float
    last_hit: float = 0.0
    fired_at: float = -1e18
    models: set = field(default_factory=set)
    sources: set = field(default_factory=set)
    assoc: float = 0.0
    person_box: Optional[tuple] = None


class ObjectThreatAnalyzer:
    """See module docstring. `models`: objects with `predict(images, crop=False) -> [[ObjDet]]` and `info`."""

    def __init__(self, models: list, *, fps: float = 3.0, min_conf: float = 0.5, k: int = 3, n: int = 5,
                 labels: Optional[list[str]] = None, label_min_conf: Optional[dict] = None,
                 require_person: bool = True, assoc_margin: float = 0.35, person_crops: bool = True,
                 crop_below_px: float = 260.0, crop_scale: float = 1.4, max_crops: int = 4, rearm_s: float = 60.0,
                 criticality: int = 2, write_crops: bool = True, max_box_frac: float = 0.25):
        if not 1 <= k <= n:
            raise ValueError("need 1 <= k <= n")
        self.models = list(models)
        self.fps, self.min_conf, self.k, self.n = float(fps), float(min_conf), int(k), int(n)
        self.labels = set(labels) if labels else None
        self.label_min_conf = {str(a): float(b) for a, b in (label_min_conf or {}).items()}
        self.require_person, self.assoc_margin = bool(require_person), float(assoc_margin)
        self.person_crops, self.crop_below_px = bool(person_crops), float(crop_below_px)
        self.crop_scale, self.max_crops = float(crop_scale), int(max_crops)
        self.rearm_s, self.criticality, self.write_crops = float(rearm_s), int(criticality), bool(write_crops)
        self.max_box_frac = float(max_box_frac)
        self.last_run: dict[str, float] = {}
        self.hist: dict[str, dict[tuple, _Hist]] = {}
        self.lock = threading.Lock()
        self.stats = {"checks": 0, "raw_detections": 0, "events": 0}

    def thr(self, label: str) -> float:
        return self.label_min_conf.get(label, self.min_conf)

    # ---- detection ---------------------------------------------------------------------------
    def detect(self, frame, people: list[tuple[int, tuple]]) -> list[ObjDet]:
        """All models on the full frame, plus enlarged crops of the smallest people (small objects)."""
        H, W = frame.shape[:2]
        out: list[ObjDet] = []
        crops, offsets = [], []
        if self.person_crops:
            small = sorted([p for p in people if p[1][3] - p[1][1] < self.crop_below_px],
                           key=lambda p: p[1][3] - p[1][1])[:self.max_crops]
            for _, (x1, y1, x2, y2) in small:
                w, h = x2 - x1, y2 - y1
                cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * self.crop_scale / 2
                X1, Y1, X2, Y2 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
                if X2 - X1 >= 16 and Y2 - Y1 >= 16:
                    crops.append(frame[Y1:Y2, X1:X2]); offsets.append((X1, Y1))
        for m in self.models:
            out += m.predict([frame])[0]
            if crops:
                for dets, (ox, oy) in zip(m.predict(crops, crop=True), offsets):
                    for d in dets:
                        b = d.box
                        out.append(ObjDet(d.label, d.conf, (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy), d.model,
                                          "person_crop"))
        area = float(W * H)
        out = [d for d in out if (self.labels is None or d.label in self.labels)
               and (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) <= self.max_box_frac * area]
        return nms(out)

    # ---- main hook ---------------------------------------------------------------------------
    def on_frame(self, worker, t: float, frame, dets, obs) -> list[Event]:
        cam = worker.id
        with self.lock:
            if t - self.last_run.get(cam, -1e18) < 1.0 / self.fps - 1e-3:
                return []
            self.last_run[cam] = t
            hist = self.hist.setdefault(cam, {})
        people = [(d[0], tuple(float(v) for v in d[1])) for d in dets if d[2] == "person"]
        if self.require_person and not people:
            self._decay(hist, t, set())
            return []
        found = self.detect(frame, people)
        self.stats["checks"] += 1
        self.stats["raw_detections"] += len(found)
        best: dict[tuple, tuple[ObjDet, float, Optional[tuple]]] = {}
        for d in found:
            tid, score = associate(d.box, people, self.assoc_margin)
            if tid is None:
                if self.require_person:
                    continue
                key = (d.label, f"cell{int((d.box[0] + d.box[2]) / 128)}_{int((d.box[1] + d.box[3]) / 128)}")
            else:
                key = (d.label, tid)
            pbox = next((p[1] for p in people if p[0] == tid), None)
            if key not in best or d.conf > best[key][0].conf:
                best[key] = (d, score, pbox)
        events = []
        for key, (d, score, pbox) in best.items():
            h = hist.get(key)
            if h is None:
                h = hist[key] = _Hist(deque(maxlen=self.n), deque(maxlen=self.n), first=t)
            h.confs.append(round(d.conf, 3)); h.boxes.append(tuple(round(v, 1) for v in d.box))
            h.models.add(d.model); h.sources.add(d.source)
            h.assoc, h.person_box = score, pbox
            if d.conf >= self.thr(d.label):
                h.last_hit = t
            hits = [c for c in h.confs if c >= self.thr(d.label)]
            if len(hits) >= self.k and t - h.fired_at >= self.rearm_s:
                h.fired_at = t
                events.append(self._event(worker, t, frame, obs, key, d, h, hits))
        self._decay(hist, t, set(best))
        self.stats["events"] += len(events)
        return events

    def _decay(self, hist: dict, t: float, seen: set) -> None:
        """Checks where a tracked (label, track) was not seen count as misses (0.0); drop long-empty histories."""
        for key in list(hist):
            if key in seen:
                continue
            h = hist[key]
            h.confs.append(0.0); h.boxes.append(None)
            if not any(h.confs) and t - h.fired_at >= self.rearm_s:
                del hist[key]

    def _event(self, worker, t, frame, obs, key, d: ObjDet, h: _Hist, hits: list[float]) -> Event:
        label, who = key
        tid = who if isinstance(who, int) else None
        near = f"near person track #{tid}" if tid is not None else "with no person track nearby"
        seen = len(hits)
        mnames = sorted(h.models)
        minfo = [m.info for m in self.models if m.info.get("name") in h.models] or [m.info for m in self.models]
        mtxt = ", ".join(f"{i['name']} {i['version']}" for i in minfo)
        obs_txt = (f"Object resembling {PHRASE.get(label, label)} {near} ({worker.id}): detector confidence "
                   f"{max(hits):.2f}, seen in {seen} of the last {len(h.confs)} checks at {self.fps:g}/s "
                   f"({mtxt}). Model inference on the object only; it says nothing about the person's intent.")
        ground = next((o.ground for o in obs if tid is not None and o.track_id == tid and o.ground), None)
        deriv = {"rule": RULE, "object_label": label, "method": "object detector + k-of-n persistence per track",
                 "threshold": {"min_conf": self.thr(label), "persistence": f"{self.k} of {self.n}",
                               "check_fps": self.fps},
                 "per_check_conf": list(h.confs), "persistence": {"seen": seen, "of": len(h.confs)},
                 "box_xyxy_analytics_px": [round(v, 1) for v in d.box],
                 "recent_boxes": [b for b in h.boxes if b],
                 "person_track": tid, "person_box_xyxy": [round(v, 1) for v in h.person_box] if h.person_box else None,
                 "association": {"method": "object area inside the person's hand region (box widened "
                                           f"{self.assoc_margin:.0%} each side)", "score": round(h.assoc, 3)},
                 "detected_in": sorted(h.sources), "models": minfo, "learned_model": True,
                 "confidence_method": "mean detector confidence over the persistent checks (uncalibrated)",
                 "duration_s": round(t - h.first, 2), "calib_error_m": worker.calib_error,
                 "note": "Model inference, not a fact. Toy or replica guns, phones, tools, umbrellas and sticks can "
                         "look alike; small or occluded objects are missed. An operator must verify."}
        e = Event(RULE, worker.id, "_view", f"{worker.name} (whole view)", [tid] if tid is not None else [],
                  h.first, t, obs_txt, deriv, round(float(np.mean(hits)), 3), 3, self.criticality, None, ground)
        ev = {"model": {"detector": " + ".join(f"{i['name']} {i['version']}" for i in minfo) + " (object model)",
                        "tracker": f"person tracks: {worker.backend.info.get('detector', '?')} + "
                                   f"{worker.backend.info.get('tracker', '?')}"},
              "object_models": mnames}
        if self.write_crops and frame is not None:
            box = d.box
            if h.person_box:
                pb = h.person_box
                box = (min(box[0], pb[0]), min(box[1], pb[1]), max(box[2], pb[2]), max(box[3], pb[3]))
            rel = write_crop(worker, frame, box, [d.box], f"{RULE}_{label}", t, pad=0.15)
            if rel:
                ev["crop"] = rel
        e.evidence = ev
        return e

    def close(self) -> None:
        self.hist.clear()
