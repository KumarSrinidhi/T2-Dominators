"""Synthetic floor-plan simulator for the handoff (next-camera) model: cold-start data + held-out evaluation.

A site is described in each camera's own ground frame (metres, the frame of its calibration homography):
  portals  door/edge segments on a camera's view where people enter or leave it ("edge": they walk on until they
           are out of the image; "door": they vanish just past the door line, e.g. a doorway in the back wall);
  links    portal <-> portal corridors (length_m, leak = probability of leaving coverage on the way) or
           portal -> outside (leaves coverage);
  routes   per entry portal, the probability of each exit portal of the same camera;
  sources  where people arrive from outside (relative rates), arrivals_per_min for the whole site.
Walkers (1.0-1.6 m/s, optional pauses) are rendered into per-camera foot positions with the camera homography,
partial visibility at the image borders, jitter, dropouts and occasional track-ID switches, and then run through
the SAME Lifecycle code as live video, so the features match what the pipeline produces. Labels come from the
simulator's ground truth (which camera the walker appears in next, and when).

    uv run python tools/sim_handoff.py [--duration 7200] [--holdout 2400] [--seed 0]
prints NN vs the most-frequent-destination-per-exit-side baseline on a held-out simulation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from argonix.geometry import homography_from_points, invert_h  # noqa: E402
from argonix.handoff import DEFAULTS, LEFT, Lifecycle, Sample, Topology, evaluate, train_model  # noqa: E402

IMG_PTS = [[80, 520], [880, 520], [700, 200], [260, 200]]  # same floor homography as tools/make_synthetic.py
GND_PTS = [[0, 0], [16, 0], [16, 12], [0, 12]]
_CAM = {"image_size": [960, 540], "image_points": IMG_PTS, "ground_points": GND_PTS}

DEMO_TOPOLOGY = {
    "fps": 5,
    "cameras": {
        "camA": {**_CAM, "name": "Entrance hall", "placement": {"offset_m": [0, 0], "rotation_deg": 0}},
        "camB": {**_CAM, "name": "East corridor", "placement": {"offset_m": [75, 0], "rotation_deg": 0}},
        "camC": {**_CAM, "name": "North wing", "placement": {"offset_m": [37, 45], "rotation_deg": 0}},
    },
    "portals": {
        "A_west": {"camera": "camA", "a": [-1.5, 1.0], "b": [-1.5, 4.5], "out": [-1, 0], "kind": "edge"},
        "A_east_near": {"camera": "camA", "a": [16.5, 0.5], "b": [16.5, 3.0], "out": [1, 0], "kind": "edge"},
        "A_east_far": {"camera": "camA", "a": [20.0, 8.0], "b": [20.0, 11.0], "out": [1, 0], "kind": "edge"},
        "A_back": {"camera": "camA", "a": [6.0, 12.0], "b": [10.0, 12.0], "out": [0, 1], "kind": "door"},
        "B_west": {"camera": "camB", "a": [-0.5, 0.5], "b": [-0.5, 3.0], "out": [-1, 0], "kind": "edge"},
        "B_east": {"camera": "camB", "a": [16.5, 1.0], "b": [16.5, 4.5], "out": [1, 0], "kind": "edge"},
        "B_back": {"camera": "camB", "a": [9.0, 12.0], "b": [13.0, 12.0], "out": [0, 1], "kind": "door"},
        "C_south": {"camera": "camC", "a": [5.0, -0.2], "b": [11.0, -0.2], "out": [0, -1], "kind": "edge"},
        "C_east": {"camera": "camC", "a": [18.0, 3.0], "b": [18.0, 6.5], "out": [1, 0], "kind": "edge"},
        "C_back": {"camera": "camC", "a": [3.0, 12.0], "b": [7.0, 12.0], "out": [0, 1], "kind": "door"},
    },
    "links": [
        {"a": "A_west", "outside": True}, {"a": "A_back", "outside": True},
        {"a": "B_east", "outside": True}, {"a": "C_back", "outside": True},
        {"a": "A_east_near", "b": "B_west", "length_m": 8.0, "leak": 0.10},
        {"a": "A_east_far", "b": "C_south", "length_m": 12.0, "leak": 0.05},
        {"a": "B_back", "b": "C_east", "length_m": 10.0, "leak": 0.05},
    ],
    "sources": {"A_west": 1.0, "B_east": 0.4, "C_back": 0.35},
    "arrivals_per_min": 6.0,
    "routes": {
        "A_west": {"A_east_near": 0.40, "A_east_far": 0.30, "A_back": 0.15, "A_west": 0.15},
        "A_east_near": {"A_west": 0.60, "A_east_far": 0.15, "A_back": 0.25},
        "A_east_far": {"A_west": 0.60, "A_east_near": 0.15, "A_back": 0.25},
        "A_back": {"A_west": 0.50, "A_east_near": 0.30, "A_east_far": 0.20},
        "B_west": {"B_east": 0.50, "B_back": 0.30, "B_west": 0.20},
        "B_east": {"B_west": 0.60, "B_back": 0.40},
        "B_back": {"B_west": 0.60, "B_east": 0.40},
        "C_south": {"C_back": 0.50, "C_east": 0.30, "C_south": 0.20},
        "C_east": {"C_south": 0.50, "C_back": 0.50},
        "C_back": {"C_south": 0.60, "C_east": 0.40},
    },
    "room": [2.0, 1.5, 14.0, 10.5],
    "speed_mps": [1.0, 1.6],
    "pause": {"prob": 0.2, "min_s": 1.0, "max_s": 4.0},
    "corridor_pause": {"prob": 0.1, "min_s": 2.0, "max_s": 8.0},
    "noise": {"jitter_px": 1.5, "dropout": 0.03, "id_switch": 0.03},
}
BEYOND = {"edge": 8.0, "door": 0.6}
DOOR_CUT = 0.15  # metres past a door line after which a walker is no longer visible


def person_h(v: float) -> float:
    """Rendered person height (px) for a foot at image row v (same law as tools/make_synthetic.py)."""
    return max(20.0, 90 + (v - 200) / 320 * 150)


def load_spec(spec, site_cfg: Optional[dict] = None) -> dict:
    """'demo' -> DEMO_TOPOLOGY; a dict is completed with calibrations from the site config where missing."""
    if spec == "demo":
        return copy.deepcopy(DEMO_TOPOLOGY)
    spec = copy.deepcopy(spec)
    site_cams = {c["id"]: c for c in (site_cfg or {}).get("cameras", [])}
    for cid, c in spec.get("cameras", {}).items():
        cal = (site_cams.get(cid) or {}).get("calibration") or {}
        if "image_points" not in c and cal.get("image_points"):
            c["image_points"], c["ground_points"] = cal["image_points"], cal["ground_points"]
        if "image_points" not in c:
            raise ValueError(f"sim topology camera {cid}: no calibration (image_points/ground_points)")
        c.setdefault("image_size", (cal.get("params") or {}).get("image_size") or [1920, 1080])
    return spec


def door_zones_px(spec: dict, inside_m: float = 1.5, outside_m: float = 0.5, pad_m: float = 0.7) -> dict:
    """Exit zones (source px) around every door portal: {camera: [{"id", "polygon_px"}]}."""
    out: dict[str, list] = {}
    for pid, p in spec["portals"].items():
        if p.get("kind") != "door":
            continue
        cam = spec["cameras"][p["camera"]]
        Hg = invert_h(homography_from_points([tuple(x) for x in cam["image_points"]],
                                             [tuple(x) for x in cam["ground_points"]]))
        a, b, n = np.array(p["a"], float), np.array(p["b"], float), np.array(p["out"], float)
        n = n / np.linalg.norm(n)
        u = (b - a) / np.linalg.norm(b - a)
        a2, b2 = a - u * pad_m, b + u * pad_m
        quad = [a2 - n * inside_m, b2 - n * inside_m, b2 + n * outside_m, a2 + n * outside_m]
        poly = [[round(float(x), 1), round(float(y), 1)] for x, y in (_proj(Hg, q) for q in quad)]
        out.setdefault(p["camera"], []).append({"id": pid, "polygon_px": poly})
    return out


def _proj(Hg, g) -> tuple[float, float]:
    x = Hg[0][0] * g[0] + Hg[0][1] * g[1] + Hg[0][2]
    y = Hg[1][0] * g[0] + Hg[1][1] * g[1] + Hg[1][2]
    w = Hg[2][0] * g[0] + Hg[2][1] * g[1] + Hg[2][2]
    return x / w, y / w


@dataclass
class Visit:
    walker: int
    cam: str
    tid: int
    path: list                      # [(t, gx, gy)] keyframes in the camera's ground frame
    enter: str
    exit: str
    next: Optional[int] = None      # index of the walker's next visit (None: leaves coverage)
    t_first: Optional[float] = None  # first / last time actually visible in the image
    t_last: Optional[float] = None
    switch: Optional[tuple] = None   # (t, new_tid) when the tracker would switch IDs mid-visit


def path_pos(path, t: float):
    if t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return path[-1][1:]


class Sim:
    def __init__(self, spec: dict, seed: int = 0):
        self.spec, self.rng = spec, random.Random(seed)
        self.Hg, self.size, self.w_sign, self.doors = {}, {}, {}, {}
        for cid, c in spec["cameras"].items():
            Hg = invert_h(homography_from_points([tuple(x) for x in c["image_points"]],
                                                 [tuple(x) for x in c["ground_points"]]))
            self.Hg[cid], self.size[cid] = Hg, tuple(c["image_size"])
            gc = np.mean(np.array(c["ground_points"], float), 0)  # a ground point known to be in view
            self.w_sign[cid] = math.copysign(1.0, Hg[2][0] * gc[0] + Hg[2][1] * gc[1] + Hg[2][2])
            self.doors[cid] = [(np.array(p["a"], float), np.array(p["out"], float) / np.linalg.norm(p["out"]))
                               for p in spec["portals"].values() if p["camera"] == cid and p.get("kind") == "door"]
        self.link = {}
        for ln in spec["links"]:
            if ln.get("outside"):
                self.link[ln["a"]] = None
            else:
                self.link[ln["a"]] = (ln["b"], float(ln["length_m"]), float(ln.get("leak", 0.0)))
                self.link[ln["b"]] = (ln["a"], float(ln["length_m"]), float(ln.get("leak", 0.0)))
        self._tid = 0

    # ---- walking -----------------------------------------------------------------------------------------------
    def _pick(self, probs: dict) -> str:
        r, acc = self.rng.random() * sum(probs.values()), 0.0
        for k, p in probs.items():
            acc += p
            if r <= acc:
                return k
        return k

    def _portal_point(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        p = self.spec["portals"][pid]
        a, b = np.array(p["a"], float), np.array(p["b"], float)
        n = np.array(p["out"], float)
        return a + self.rng.uniform(0.15, 0.85) * (b - a), n / np.linalg.norm(n)

    def _room_path(self, P: str, Q: str, t: float, speed: float) -> tuple[list, float]:
        sp = self.spec
        pP, nP = self._portal_point(P)
        pQ, nQ = self._portal_point(Q)
        pts = [pP + nP * BEYOND[sp["portals"][P]["kind"]], pP]
        pause_at = None
        if P == Q or self.rng.random() < 0.6:
            x0, y0, x1, y1 = sp["room"]
            pts.append(np.array([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)]))
            pz = sp.get("pause") or {}
            if self.rng.random() < pz.get("prob", 0.0):
                pause_at = (len(pts) - 1, self.rng.uniform(pz["min_s"], pz["max_s"]))
        pts += [pQ, pQ + nQ * BEYOND[sp["portals"][Q]["kind"]]]
        path = [(t, float(pts[0][0]), float(pts[0][1]))]
        for i in range(1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[i - 1]))
            t += d / (speed * self.rng.uniform(0.92, 1.08))
            path.append((t, float(pts[i][0]), float(pts[i][1])))
            if pause_at and pause_at[0] == i:
                t += pause_at[1]
                path.append((t, float(pts[i][0]), float(pts[i][1])))
        return path, t

    def run(self, duration_s: float, t0: float = 0.0) -> list[Visit]:
        sp, rng = self.spec, self.rng
        visits: list[Visit] = []
        rate = float(sp["arrivals_per_min"]) / 60.0
        t, walker = t0, 0
        lo, hi = sp["speed_mps"]
        while True:
            t += rng.expovariate(rate)
            if t > t0 + duration_s:
                break
            walker += 1
            speed = rng.uniform(lo, hi)
            P, tc, prev = self._pick(sp["sources"]), t, None
            for _hop in range(8):
                Q = self._pick(sp["routes"][P])
                path, t_end = self._room_path(P, Q, tc, speed * rng.uniform(0.95, 1.05))
                self._tid += 1
                v = Visit(walker, sp["portals"][P]["camera"], self._tid, path, P, Q)
                visits.append(v)
                if prev is not None:
                    visits[prev].next = len(visits) - 1
                prev = len(visits) - 1
                ln = self.link.get(Q)
                if ln is None or rng.random() < ln[2]:
                    break
                R, L, _ = ln
                tc = t_end + L / speed * rng.uniform(0.9, 1.3)
                cp = sp.get("corridor_pause") or {}
                if rng.random() < cp.get("prob", 0.0):
                    tc += rng.uniform(cp["min_s"], cp["max_s"])
                P = R
        return visits

    # ---- observation -------------------------------------------------------------------------------------------
    def foot_box(self, cam: str, g) -> Optional[tuple[float, float, float, float]]:
        """Detector-like bbox (image px, clipped) of a walker standing at ground g, or None when not visible."""
        for a, n in self.doors[cam]:
            if float((np.asarray(g) - a) @ n) > DOOR_CUT:
                return None
        Hg = self.Hg[cam]
        w = Hg[2][0] * g[0] + Hg[2][1] * g[1] + Hg[2][2]
        if w * self.w_sign[cam] <= 1e-9:
            return None
        u, v = _proj(Hg, g)
        W, H = self.size[cam]
        ph = person_h(v)
        pw = 0.4 * ph
        x1, x2 = max(0.0, u - pw / 2), min(W - 1.0, u + pw / 2)
        y1, y2 = max(0.0, v - ph), min(H - 1.0, v)
        if x2 - x1 < 0.35 * pw or y2 - y1 < 0.45 * ph or v < 0:
            return None
        return (x1, y1, x2, y2)

    def observe(self, visits: list[Visit], t_start: float, t_end: float, fps: Optional[float] = None,
                noise: bool = True):
        """Per-camera analytics frames: {cam: [(t, dets, grounds)]} with dets = [(tid, bbox, 'person', conf)]."""
        fps = float(fps or self.spec.get("fps", 5))
        n = int((t_end - t_start) * fps) + 1
        nz = self.spec.get("noise", {}) if noise else {}
        frames = {c: [[] for _ in range(n)] for c in self.spec["cameras"]}
        for v in visits:
            i0 = max(0, math.ceil((v.path[0][0] - t_start) * fps))
            i1 = min(n - 1, math.floor((v.path[-1][0] - t_start) * fps))
            if nz.get("id_switch") and self.rng.random() < nz["id_switch"]:
                self._tid += 1
                v.switch = (self.rng.uniform(v.path[0][0], v.path[-1][0]), self._tid)
            for i in range(i0, i1 + 1):
                t = t_start + i / fps
                g = path_pos(v.path, t)
                if g is None:
                    continue
                bb = self.foot_box(v.cam, g)
                if bb is None:
                    continue
                v.t_first = t if v.t_first is None else v.t_first
                v.t_last = t
                if nz.get("dropout") and self.rng.random() < nz["dropout"]:
                    continue
                j = nz.get("jitter_px", 0.0)
                if j:
                    dx, dy = self.rng.gauss(0, j), self.rng.gauss(0, j)
                    bb = (bb[0] + dx, bb[1] + dy, bb[2] + dx, min(self.size[v.cam][1] - 1.0, bb[3] + dy))
                tid = v.switch[1] if v.switch and t >= v.switch[0] else v.tid
                frames[v.cam][i].append((tid, bb, "person", 0.85, g))
        return {c: [(t_start + i / fps, [d[:4] for d in fr], {d[0]: d[4] for d in fr}) for i, fr in enumerate(f)]
                for c, f in frames.items()}


def norm_zones(spec: dict) -> dict:
    zones = door_zones_px(spec)
    return {c: [[(x / spec["cameras"][c]["image_size"][0], y / spec["cameras"][c]["image_size"][1])
                 for x, y in z["polygon_px"]] for z in zones.get(c, [])] for c in spec["cameras"]}


def label_exits(visits: list[Visit], exits) -> list[Sample]:
    """Ground-truth labels for lifecycle exits: the walker's next visible camera (or leaves coverage)."""
    by_tid = {}
    for i, v in enumerate(visits):
        by_tid[v.tid] = (i, v, v.switch is None)
        if v.switch:
            by_tid[v.switch[1]] = (i, v, True)
    out = []
    for ex in exits:
        hit = by_tid.get(ex.tid)
        if not hit or not hit[2] or v_last_gap(hit[1], ex.t) > 1.5:
            continue  # fragment before an ID switch, or a mid-visit loss: not a real exit
        v = hit[1]
        nxt = visits[v.next] if v.next is not None else None
        while nxt is not None and nxt.t_first is None:  # never visible (should not happen): skip ahead
            nxt = visits[nxt.next] if nxt.next is not None else None
        if nxt is None:
            out.append(Sample(ex.x, ex.cam, ex.key, {LEFT: 1.0}, {}, ex.t, "sim"))
        else:
            out.append(Sample(ex.x, ex.cam, ex.key, {nxt.cam: 1.0}, {nxt.cam: round(nxt.t_first - ex.t, 2)},
                              ex.t, "sim"))
    return out


def v_last_gap(v: Visit, t: float) -> float:
    return abs((v.t_last if v.t_last is not None else -1e9) - t)


def run_lifecycle(spec: dict, frames: dict, cfg: Optional[dict] = None) -> list:
    life = Lifecycle({**DEFAULTS, **(cfg or {})})
    zones = norm_zones(spec)
    cams = list(frames)
    exits = []
    n = len(frames[cams[0]])
    for i in range(n):
        for c in cams:
            t, dets, grounds = frames[c][i]
            W, H = spec["cameras"][c]["image_size"]
            _, ex, _ = life.observe(c, t, W, H, dets, grounds, zones[c])
            exits += ex
    return exits


def simulate_dataset(spec: dict, duration_s: float = 7200, seed: int = 0, cfg: Optional[dict] = None,
                     episode_s: float = 900.0) -> list[Sample]:
    """Labelled exit samples from `duration_s` of simulated traffic (episodes at random hours of the day)."""
    spec = load_spec(spec)
    rng = random.Random(seed)
    base = time.mktime((2026, 1, 5, 0, 0, 0, 0, 0, -1))
    samples: list[Sample] = []
    n_ep = max(1, math.ceil(duration_s / episode_s))
    for ep in range(n_ep):
        t0 = base + 86400 * ep + rng.uniform(0, 86400 - episode_s - 300)
        sim = Sim(spec, seed * 1009 + ep)
        dur = min(episode_s, duration_s - ep * episode_s)
        visits = sim.run(dur, t0)
        frames = sim.observe(visits, t0, t0 + dur + 120)
        samples += label_exits(visits, run_lifecycle(spec, frames, cfg))
    return samples


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", default="demo", help='"demo" or a JSON topology file')
    ap.add_argument("--duration", type=float, default=7200, help="simulated seconds for training")
    ap.add_argument("--holdout", type=float, default=2400, help="simulated seconds for the held-out test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args(argv)
    spec = load_spec("demo" if a.spec == "demo" else json.loads(Path(a.spec).read_text()))
    t = time.time()
    tr = simulate_dataset(spec, a.duration, a.seed)
    ho = simulate_dataset(spec, a.holdout, a.seed + 1000)
    t_sim = time.time() - t
    cams = list(spec["cameras"])
    base = Topology()
    for s in tr:
        base.add(s.cam, s.key, s.target, s.dt)
    t = time.time()
    m = train_model(tr, cams, epochs=a.epochs, seed=a.seed)
    t_train = time.time() - t
    res = evaluate(m, base, ho)
    per_key = {}
    for key in sorted({(s.cam, s.key) for s in ho}):
        sub = [s for s in ho if (s.cam, s.key) == key]
        e = evaluate(m, base, sub)
        per_key[f"{key[0]}:{key[1]}"] = {"n": e["n"], "nn_top1": e["nn"]["top1"], "baseline_top1": e["baseline"]["top1"]}
    out = {"train_samples": len(tr), "holdout_samples": len(ho), "sim_s": round(t_sim, 1), "train_s": round(t_train, 1),
           "params": m.meta["params"], "temperature": round(m.T, 3), "holdout": res, "per_exit_side": per_key}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
