"""Evaluate multi-camera Re-ID + global IDs (argonix/global_ids.py) honestly.

Three evaluations, each reporting cross-camera association precision / recall, an IDF1-style score, and the ablation
appearance-only vs topology-only vs fused:

  --sim      Simulated trajectories (tools/sim_handoff.py floor plan) with SYNTHETIC appearance vectors of chosen
             separability. No video; seconds. Tests the fusion/decision logic, not a Re-ID model.
  --video    Rendered 3-camera scene: walkers are pasted crops of REAL pedestrians (tracks from data/real/vtest.avi),
             a different real frame of the same pedestrian per camera visit, with a per-camera colour/brightness/blur
             change. Full pipeline (YOLO + ByteTrack + handoff + Re-ID analyzer). Also run on grayscale-converted
             footage (--gray) for the colour-dependence drop. Replays the recorded segments/embeddings offline for
             the ablations (appearance / topology / fused, colour vs gray embeddings).
  --real-appearance
             Single-camera REAL footage (vtest.avi): same-track vs different-track similarity separability (ROC AUC)
             for colour vs gray crops, per embedder. Single camera, so optimistic for "same" pairs.

    uv run python tools/eval_reid.py --sim
    uv run python tools/eval_reid.py --video [--gray] [--imgsz 480]
    uv run python tools/eval_reid.py --real-appearance
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from argonix.global_ids import HandoffGate, IdentityEngine  # noqa: E402
from argonix.handoff import HandoffEngine, Topology  # noqa: E402

OUT = ROOT / "data" / "reid_eval"


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def assoc_metrics(seg_truth: dict, seg_gid: dict, seg_meta: dict, links: list[tuple[str, str]],
                  weights: dict | None = None) -> dict:
    """seg_truth: segment -> true person id (None = no match); seg_gid: segment -> predicted global id;
    seg_meta: segment -> (camera, first_t); links: predicted (a, b) links in the final state.
    weights: segment -> number of observations (IDF1 weighting; default 1)."""
    weights = weights or {s: 1 for s in seg_truth}
    # true links: consecutive matched segments of the same person
    by_person = defaultdict(list)
    for s, p in seg_truth.items():
        if p is not None:
            by_person[p].append(s)
    true_links = set()
    for p, ss in by_person.items():
        ss.sort(key=lambda s: seg_meta[s][1])
        true_links |= {(a, b) for a, b in zip(ss, ss[1:])}
    same_person_pairs = {(a, b) for p, ss in by_person.items() for i, a in enumerate(ss) for b in ss[i + 1:]}

    def split(ls, cross):
        return {(a, b) for a, b in ls if (seg_meta[a][0] != seg_meta[b][0]) == cross}

    out = {}
    for name, cross in (("cross_camera", True), ("same_camera", False)):
        pl = split(links, cross)
        tl = split(true_links, cross)
        judged = [(a, b) for a, b in pl if seg_truth.get(a) is not None and seg_truth.get(b) is not None]
        correct = [(a, b) for a, b in judged if seg_truth[a] == seg_truth[b]]
        found = [(a, b) for a, b in tl if (a, b) in pl or _same_gid(seg_gid, a, b)]
        out[name] = {"predicted_links": len(pl), "judged": len(judged), "correct": len(correct),
                     "precision": round(len(correct) / len(judged), 3) if judged else None,
                     "precision_wilson95": wilson(len(correct), len(judged)),
                     "true_links": len(tl), "recovered": len(found),
                     "recall": round(len(found) / len(tl), 3) if tl else None,
                     "recall_wilson95": wilson(len(found), len(tl))}
    # IDF1 over segments weighted by observations: optimal one-to-one gid <-> person matching
    segs = [s for s in seg_truth if s in seg_gid]
    gids = sorted({seg_gid[s] for s in segs})
    persons = sorted({seg_truth[s] for s in segs if seg_truth[s] is not None})
    co = np.zeros((len(gids), len(persons)))
    gi, pi = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(persons)}
    for s in segs:
        if seg_truth[s] is not None:
            co[gi[seg_gid[s]], pi[seg_truth[s]]] += weights.get(s, 1)
    idtp = 0.0
    if co.size:
        import lap  # linear assignment (a project dependency), maximising co-occurrence
        _, x, _ = lap.lapjv(co.max() - co, extend_cost=True)
        idtp = float(sum(co[i, j] for i, j in enumerate(x) if j >= 0))
    n_pred = float(sum(weights.get(s, 1) for s in segs))
    n_true = float(sum(weights.get(s, 1) for s in segs if seg_truth[s] is not None))
    idfp, idfn = n_pred - idtp, n_true - idtp
    out["idf1"] = round(2 * idtp / (2 * idtp + idfp + idfn), 4) if (idtp + idfp + idfn) else None
    out["identities"] = {"predicted": len(gids), "true": len(persons),
                         "gids_per_person": round(float(np.mean([len({seg_gid[s] for s in segs if seg_truth[s] == p})
                                                                 for p in persons])), 3) if persons else None,
                         "persons_per_gid": round(float(np.mean([len({seg_truth[s] for s in segs if seg_gid[s] == g
                                                                      and seg_truth[s] is not None} or {None})
                                                                 for g in gids])), 3) if gids else None}
    out["pairwise_same_person_recall"] = round(
        sum(1 for a, b in same_person_pairs if _same_gid(seg_gid, a, b)) / len(same_person_pairs), 3) \
        if same_person_pairs else None
    return out


def _same_gid(seg_gid, a, b) -> bool:
    return seg_gid.get(a) is not None and seg_gid.get(a) == seg_gid.get(b)


def engine_state(eng: IdentityEngine) -> tuple[dict, list, dict]:
    seg_gid = {s.id: s.gid for s in eng.segs.values() if s.gid}
    links = [(s.pred, s.id) for s in eng.segs.values() if s.pred and s.pred in eng.segs]
    weights = {s.id: s.n_obs for s in eng.segs.values()}
    return seg_gid, links, weights


# ---------------------------------------------------------------------------------------------------------------
# 1) simulated trajectories + synthetic appearance vectors
# ---------------------------------------------------------------------------------------------------------------
def sim_gate(spec="demo", duration_s: float = 2700.0, seed: int = 11) -> HandoffEngine:
    from tools.sim_handoff import simulate_dataset
    samples = simulate_dataset(spec, duration_s=duration_s, seed=seed)
    ho = HandoffEngine(["camA", "camB", "camC"], {"cold_start": False})
    st = Topology()
    for s in samples:
        st.add(s.cam, s.key, s.target, s.dt)
    ho.sim_topology = st
    return ho


def run_sim(mode: str = "fused", sep: float = 2.0, dim: int = 64, duration_s: float = 900.0, seed: int = 3,
            gate_engine: HandoffEngine | None = None, noise: bool = True, cfg: dict | None = None) -> dict:
    """Simulated walkers; each walker has a random unit 'identity' vector; each crop embedding = identity * sep +
    per-camera offset + noise, normalised. sep controls separability (2.0 ~ weak, 4.0 ~ strong)."""
    from tools.sim_handoff import DEMO_TOPOLOGY, Sim, norm_zones
    spec = copy.deepcopy(DEMO_TOPOLOGY)
    sim = Sim(spec, seed)
    visits = sim.run(duration_s, t0=0.0)
    frames = sim.observe(visits, 0.0, duration_s + 60.0, fps=5, noise=noise)
    tid_walker = {}
    for v in visits:
        tid_walker[v.tid] = v.walker
        if v.switch:
            tid_walker[v.switch[1]] = v.walker
    rng = np.random.default_rng(seed)
    ident = {w: rng.normal(size=dim) for w in {v.walker for v in visits}}
    cam_off = {c: rng.normal(size=dim) * 0.6 for c in spec["cameras"]}
    zones = norm_zones(spec)
    gate = HandoffGate(gate_engine or sim_gate()) if mode != "appearance" else None
    mu_same = sep * sep / (sep * sep + 2.44)  # |identity|^2 / (|identity|^2 + |camera offset|^2 + |noise|^2)
    eng = IdentityEngine(list(spec["cameras"]), {"mode": mode, "sync": True, "keep_trace": True,
                                                  "embedding_ttl_s": 1e9, "retention_s": 1e9, **(cfg or {})},
                         None, gate, sim_priors={"color": {"same": [mu_same, 0.1], "diff": [0.0, 0.2]}})
    n = len(next(iter(frames.values())))
    W, H = 960, 540
    for i in range(n):
        for cam in spec["cameras"]:
            t, dets, _ = frames[cam][i]
            want = eng.observe(cam, t, W, H, dets, zones[cam])
            for sid, bb, q in want:
                w_ = tid_walker[int(sid.split(":")[1])]
                e = ident[w_] / np.linalg.norm(ident[w_]) * sep + cam_off[cam] / math.sqrt(dim) * 2 \
                    + rng.normal(size=dim) / math.sqrt(dim) * 1.0
                e = e / np.linalg.norm(e)
                eng.add_embedding(sid, e, None, {"q": q})
    eng.flush()
    seg_truth = {s.id: tid_walker.get(s.tid) for s in eng.segs.values()}
    seg_meta = {s.id: (s.cam, s.first_t) for s in eng.segs.values()}
    seg_gid, links, weights = engine_state(eng)
    res = assoc_metrics(seg_truth, seg_gid, seg_meta, links, weights)
    res["engine"] = {k: v for k, v in eng.metrics().items() if k in ("auto", "segments_total", "identities",
                                                                      "multi_camera_identities")}
    return res


def main_sim(args) -> dict:
    t0 = time.time()
    ho = sim_gate()
    out = {"setting": "simulated trajectories (demo floor plan), synthetic appearance vectors; 900 s, 5 fps",
           "results": {}}
    for sep in (1.0, 1.5, 2.0):
        for mode in ("appearance", "topology", "fused"):
            r = run_sim(mode, sep=sep, gate_engine=ho)
            out["results"][f"sep{sep}:{mode}"] = r
            cc = r["cross_camera"]
            print(f"sep {sep:>3} {mode:10s} cross-cam P {cc['precision']} R {cc['recall']} "
                  f"(n={cc['judged']}/{cc['true_links']}) IDF1 {r['idf1']}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2) identity bank: real pedestrians (tracks in real footage), several frames each
# ---------------------------------------------------------------------------------------------------------------
BANK = OUT / "bank"
BANK_SOURCES = ["data/real/vtest.avi", "data/real/social-distance/pedestrian.mp4"]


def build_bank(weights: str, sources=BANK_SOURCES, min_h: float = 80.0, spacing_s: float = 1.2, per_track: int = 8,
               min_crops: int = 3, device: str = "auto") -> list[dict]:
    """Track people in real clips (YOLO + ByteTrack) and keep, per track, up to `per_track` good crops >= spacing_s apart
    (different poses/positions). Returns [{"id", "source", "track", "crops": [png paths], "t": [...]}] (cached)."""
    import cv2
    from argonix.detect import UltralyticsTracker
    from argonix.reid import crop_quality, extract_crop
    idx = BANK / "index.json"
    if idx.exists():
        return json.loads(idx.read_text())
    BANK.mkdir(parents=True, exist_ok=True)
    bank = []
    for src in sources:
        p = ROOT / src
        if not p.exists():
            print(f"bank: {src} missing, skipped")
            continue
        trk = UltralyticsTracker(weights, "bytetrack.yaml", 0.35, 640, (0,), device)
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / 5))
        per: dict[int, list] = defaultdict(list)
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            n += 1
            if n % step:
                continue
            t = n / fps
            H_, W_ = f.shape[:2]
            for tid, bb, cls, conf in trk.track(f):
                if bb[3] - bb[1] < min_h or crop_quality(bb, W_, H_, conf, {"min_conf": 0.6}) <= 0:
                    continue
                lst = per[tid]
                if lst and t - lst[-1][0] < spacing_s:
                    continue
                if len(lst) < per_track:
                    lst.append((t, extract_crop(f, bb, pad=0.06)))
        cap.release()
        stem = Path(src).stem
        for tid, lst in per.items():
            if len(lst) < min_crops:
                continue
            paths = []
            for k, (t, c) in enumerate(lst):
                q = BANK / f"{stem}_{tid}_{k}.png"
                cv2.imwrite(str(q), c)
                paths.append(str(q.relative_to(ROOT)).replace("\\", "/"))
            bank.append({"id": f"{stem}:{tid}", "source": src, "track": tid, "crops": paths, "t": [x[0] for x in lst]})
        print(f"bank: {src}: {sum(1 for b in bank if b['source'] == src)} identities")
    idx.write_text(json.dumps(bank, indent=1))
    return bank


def sprite(crop):
    """Cut the person out of its original background (GrabCut from the box) -> (BGR, alpha in [0, 1])."""
    import cv2
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, int(0.06 * w)), max(1, int(0.03 * h)), max(2, int(0.88 * w)), max(2, int(0.95 * h)))
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except cv2.error:
        fg = np.zeros((h, w), np.float32)
    frac = fg.mean()
    if not 0.2 <= frac <= 0.9:  # GrabCut failed: soft ellipse instead
        fg = np.zeros((h, w), np.float32)
        cv2.ellipse(fg, (w // 2, h // 2), (int(0.42 * w), int(0.5 * h)), 0, 0, 360, 1.0, -1)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    return crop, fg


CAM_STYLE = {  # per-camera photometric differences: gain (B, G, R), gamma, blur sigma, noise sd, contrast
    "camA": {"gain": (1.0, 1.0, 1.0), "gamma": 1.0, "blur": 0.0, "noise": 0.0, "contrast": 1.0},
    "camB": {"gain": (1.18, 0.98, 0.82), "gamma": 1.25, "blur": 1.0, "noise": 2.0, "contrast": 0.95},
    "camC": {"gain": (0.85, 1.0, 1.15), "gamma": 0.85, "blur": 0.6, "noise": 4.0, "contrast": 0.85},
}


def style(frame, cam: str, rng) -> np.ndarray:
    import cv2
    s = CAM_STYLE.get(cam) or CAM_STYLE["camA"]
    f = frame.astype(np.float32) / 255.0
    f = np.clip(f * np.array(s["gain"], np.float32), 0, 1) ** s["gamma"]
    f = (f - 0.5) * s["contrast"] + 0.5
    f = np.clip(f * 255.0, 0, 255)
    if s["blur"]:
        f = cv2.GaussianBlur(f, (0, 0), s["blur"])
    if s["noise"]:
        f = f + rng.normal(0, s["noise"], f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def render_scene(bank: list[dict], out_dir: Path, dur: float = 480.0, seed: int = 7, fps: int = 10) -> dict:
    """3-camera scene on the handoff demo floor plan; each walker is one real pedestrian from the bank, shown with a
    different real frame of that pedestrian in every camera visit. Writes <cam>.mp4 + truth.json."""
    import cv2
    from argonix.geometry import apply_h
    from argonix.media import H264Writer
    from tools.make_synthetic_handoff import background
    from tools.sim_handoff import Sim, load_spec, path_pos, person_h
    spec = load_spec("demo")
    sim = Sim(spec, seed)
    visits = sim.run(dur - 45.0, t0=0.0)
    sim.observe(visits, 0.0, dur, fps=fps, noise=False)
    walkers = sorted({v.walker for v in visits})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bank))
    ident = {w: bank[order[i % len(bank)]] for i, w in enumerate(walkers)}
    sprites: dict[str, tuple] = {}
    cycle = {w: [ident[w]["crops"][j] for j in rng.permutation(len(ident[w]["crops"]))] for w in walkers}
    k_visit: Counter = Counter()
    vis_crop = {}
    for i, v in enumerate(visits):  # every visit of a walker shows the next real frame of its pedestrian
        cyc = cycle[v.walker]
        path = cyc[k_visit[v.walker] % len(cyc)]
        k_visit[v.walker] += 1
        vis_crop[i] = (path,)
        if path not in sprites:
            sprites[path] = sprite(cv2.imread(str(ROOT / path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ci, cam in enumerate(spec["cameras"]):
        bg = background(spec, cam)
        W, H = sim.size[cam]
        mine = [(i, v) for i, v in enumerate(visits) if v.cam == cam]
        wr = H264Writer(out_dir / f"{cam}.mp4", W, H, fps)
        crng = np.random.default_rng(seed * 10 + ci)
        try:
            for fi in range(int(dur * fps)):
                t = fi / fps
                f = bg.copy().astype(np.float32)
                placed = []
                for i, v in mine:
                    if not (v.path[0][0] <= t <= v.path[-1][0]):
                        continue
                    g = path_pos(v.path, t)
                    if any(float((np.asarray(g) - a) @ n) > 0.15 for a, n in sim.doors[cam]):
                        continue
                    Hg = sim.Hg[cam]
                    if (Hg[2][0] * g[0] + Hg[2][1] * g[1] + Hg[2][2]) * sim.w_sign[cam] <= 0:
                        continue
                    u, vv = apply_h(Hg, g)
                    placed.append((vv, u, sprites[vis_crop[i][0]]))
                for vv, u, (c, a) in sorted(placed, key=lambda z: z[0]):
                    hh = int(person_h(vv))
                    ww = max(8, int(c.shape[1] * hh / c.shape[0]))
                    c2 = cv2.resize(c, (ww, hh)).astype(np.float32)
                    a2 = cv2.resize(a, (ww, hh))[..., None]
                    x1, y1 = int(u - ww / 2), int(vv - hh)
                    xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x1 + ww), min(H, y1 + hh)
                    if xb > xa and yb > ya:
                        cs, al = c2[ya - y1:yb - y1, xa - x1:xb - x1], a2[ya - y1:yb - y1, xa - x1:xb - x1]
                        f[ya:yb, xa:xb] = al * cs + (1 - al) * f[ya:yb, xa:xb]
                wr.write(style(f.astype(np.uint8), cam, crng))
        finally:
            wr.close()
        print(f"rendered {cam}")
    truth = {"fps": fps, "duration_s": dur, "seed": seed, "n_identities": len(bank),
             "walkers": {str(w): ident[w]["id"] for w in walkers},
             "visits": [{"walker": v.walker, "cam": v.cam, "tid": v.tid, "path": v.path, "t_first": v.t_first,
                         "t_last": v.t_last, "crop": vis_crop[i][0]} for i, v in enumerate(visits)]}
    (out_dir / "truth.json").write_text(json.dumps(truth))
    return truth


def gray_copy(src_dir: Path, dst_dir: Path) -> None:
    import cv2
    from argonix.media import H264Writer
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        w, h = int(cap.get(3)), int(cap.get(4))
        wr = H264Writer(dst_dir / p.name, w, h, fps)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            wr.write(cv2.merge([g, g, g]))
        wr.close()
        cap.release()


# ---------------------------------------------------------------------------------------------------------------
# 3) full pipeline on the rendered clips, recorded for offline ablations
# ---------------------------------------------------------------------------------------------------------------
class Recorder:
    """Records the analyzer's engine inputs (frames' detections, embeddings) for deterministic offline replays."""

    def __init__(self, an):
        import threading
        eng = an.engine
        self.calls, self.emb = [], defaultdict(list)
        self.lock = threading.Lock()
        obs, add = eng.observe, eng.add_embedding

        def observe(cam, t, w, h, dets, zones=()):
            with self.lock:
                self.calls.append((cam, t, w, h, list(dets), zones))
                return obs(cam, t, w, h, dets, zones)

        def add_embedding(sid, ec, eg, meta):
            with self.lock:
                self.emb[sid].append((ec, eg, dict(meta)))
            return add(sid, ec, eg, meta)
        eng.observe, eng.add_embedding = observe, add_embedding
        an.service.callback = add_embedding


def seg_truth_from_trace(eng: IdentityEngine, truth: dict, step: int, fps: float, tol: float = 0.05) -> dict:
    """segment -> walker by majority of foot-point matches (>= 50 % of its observations), else None."""
    from tools.sim_handoff import DEMO_TOPOLOGY, Sim, _proj, path_pos
    sim = Sim(copy.deepcopy(DEMO_TOPOLOGY), 0)
    by_cam = defaultdict(list)
    for v in truth["visits"]:
        by_cam[v["cam"]].append(v)
    out = {}
    for s in eng.segs.values():
        t0 = eng.first_t[s.cam]
        votes = Counter()
        for t, xn, yn in s.trace:
            tv = t - t0 + (step - 1) / fps
            best, bw = tol, None
            for v in by_cam[s.cam]:
                path = v["path"]
                if not path[0][0] <= tv <= path[-1][0]:
                    continue
                g = path_pos(path, tv)
                if sim.foot_box(s.cam, g) is None:
                    continue
                u, vv = _proj(sim.Hg[s.cam], g)
                W, H = sim.size[s.cam]
                d = math.hypot(u / W - xn, vv / H - yn)
                if d < best:
                    best, bw = d, v["walker"]
            if bw is not None:
                votes[bw] += 1
        w, n = votes.most_common(1)[0] if votes else (None, 0)
        out[s.id] = w if n >= 0.5 * max(1, len(s.trace)) else None
    return out


def replay(rec: Recorder, cams, cfg: dict, gate, sim_priors, embed_override=None) -> IdentityEngine:
    eng = IdentityEngine(list(cams), {**cfg, "sync": True}, None, gate, sim_priors)
    used = Counter()
    for cam, t, w, h, dets, zones in rec.calls:
        for sid, bb, q in eng.observe(cam, t, w, h, dets, zones):
            k = used[sid]
            used[sid] += 1
            if k < len(rec.emb[sid]):
                ec, eg, meta = rec.emb[sid][k]
                if embed_override is not None:
                    ec, eg = embed_override(sid, k)
                eng.add_embedding(sid, ec, eg, meta)
    eng.flush()
    return eng


def summarize(eng: IdentityEngine, st: dict) -> dict:
    seg_gid, links, weights = engine_state(eng)
    meta = {s.id: (s.cam, s.first_t) for s in eng.segs.values()}
    truth = {k: st.get(k) for k in meta}
    r = assoc_metrics(truth, seg_gid, meta, links, weights)
    r["auto"] = eng.metrics()["auto"]
    r["appearance_model"] = eng.metrics()["appearance_model"]
    return r


def run_video(args, gray: bool = False) -> dict:
    import cv2  # noqa: F401
    from argonix.pipeline import Site
    from tools.make_synthetic_handoff import site_config
    from tools.sim_handoff import load_spec
    weights = str(ROOT / "data" / "models" / "yolo26n.pt")
    bank = build_bank(weights, device=args.device)
    scene = OUT / "scene"
    if not (scene / "truth.json").exists():
        render_scene(bank, scene, dur=args.duration)
    truth = json.loads((scene / "truth.json").read_text())
    clips = scene
    if gray:
        clips = OUT / "scene_gray"
        if not (clips / "camA.mp4").exists():
            gray_copy(scene, clips)
    spec = load_spec("demo")
    cfg = site_config(spec, clips)
    run_dir = OUT / ("run_gray" if gray else "run_color")
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "configs").mkdir(parents=True)
    for c in cfg["cameras"]:
        c["source"], c["loop"] = str(clips / f"{c['id']}.mp4"), False
    cfg["detector"] = {"weights": weights, "tracker": "bytetrack.yaml", "conf": 0.3, "imgsz": args.imgsz,
                       "device": args.device}
    cfg["handoff"].update(cold_start=True, cold_start_blocking=True, max_dt_s=40)
    cfg["reid"] = {"enabled": True, "model": args.model, "device": args.device, "color_mode": "auto" if gray else "both",
                   "compare_mode": "color", "keep_trace": True, "embedding_ttl_s": 1e9, "retention_s": 1e9,
                   "max_dt_s": 40, "max_queue": 100000}
    p = run_dir / "configs" / "site.json"
    p.write_text(json.dumps(cfg))
    t0 = time.time()
    site = Site(p, data_dir=run_dir / "run", realtime=False)
    an = site.reid
    rec = Recorder(an)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        torch = None
    t_init = time.time() - t0
    t0 = time.time()
    site.start()
    site.wait()
    t_run = time.time() - t0
    for _ in range(200):  # let the embedding queue drain before closing
        if an.service.queue_len == 0:
            break
        time.sleep(0.1)
    site.stop()
    step = max(1, round(truth["fps"] / cfg["analytics_fps"]))
    st = seg_truth_from_trace(an.engine, truth, step, truth["fps"])
    res = {"setting": ("GRAYSCALE-converted" if gray else "colour") + " rendered 3-camera scene, real pedestrian crops",
           "video_s": truth["duration_s"], "processing_s": round(t_run, 1), "init_s": round(t_init, 1),
           "speed_vs_realtime": round(truth["duration_s"] / max(1e-6, t_run), 2),
           "walkers": len(truth["walkers"]), "visits": len(truth["visits"]), "bank_identities": len(bank),
           "segments": len(an.engine.segs), "segments_matched_to_truth": sum(1 for v in st.values() if v is not None),
           "embedder": an.embedder.info, "service": an.service.stats,
           "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch and torch.cuda.is_available() else None,
           "online": summarize(an.engine, st), "replay": {}}
    ho = site.handoff.engine
    priors = getattr(an.embedder, "sim", None)
    base = {k: v for k, v in an.cfg.items() if k in ("max_dt_s", "keep_trace", "embedding_ttl_s", "retention_s")}
    modes = ["color", "gray"] if not gray else ["gray"]
    for cm in modes:
        for mode in ("appearance", "topology", "fused"):
            e = replay(rec, an.engine.cams, {**base, "mode": mode, "color_mode": "both" if not gray else "gray",
                                             "compare_mode": cm}, HandoffGate(ho), priors)
            res["replay"][f"{cm}:{mode}"] = summarize(e, st)
            cc = res["replay"][f"{cm}:{mode}"]["cross_camera"]
            print(f"{'gray' if gray else 'color'} footage, {cm} emb, {mode:10s}: P {cc['precision']} R {cc['recall']} "
                  f"(judged {cc['judged']}, true {cc['true_links']}) IDF1 {res['replay'][f'{cm}:{mode}']['idf1']}")
    e = replay(rec, an.engine.cams, {**base, "mode": "fused", "color_mode": "both" if not gray else "gray",
                                     "compare_mode": modes[0]}, HandoffGate(ho, use_sim=False), priors)
    res["replay"][f"{modes[0]}:fused_no_sim_topology"] = summarize(e, st)
    # similarity distributions of true same-person cross-camera pairs vs different-person pairs (calibration check)
    res["similarity"] = sim_stats(an.engine, st)
    (OUT / ("video_gray.json" if gray else "video_color.json")).write_text(json.dumps(res, indent=1, default=str))
    return res


def sim_stats(eng: IdentityEngine, st: dict) -> dict:
    segs = [s for s in eng.segs.values() if st.get(s.id) is not None and s.crops]
    out = {}
    for mode in ("color", "gray"):
        same, diff = [], []
        for i, a in enumerate(segs):
            ea = a.emb(mode)
            if ea is None:
                continue
            for b in segs[i + 1:]:
                if a.cam == b.cam:
                    continue
                eb = b.emb(mode)
                if eb is None:
                    continue
                (same if st[a.id] == st[b.id] else diff).append(float(ea @ eb))
        if same and diff:
            out[mode] = {"same_mean": round(float(np.mean(same)), 3), "same_sd": round(float(np.std(same)), 3),
                         "diff_mean": round(float(np.mean(diff)), 3), "diff_sd": round(float(np.std(diff)), 3),
                         "n_same": len(same), "n_diff": len(diff), "auc": auc(same, diff)}
    return out


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    rp = ranks[: len(pos)].sum()
    return round(float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


# ---------------------------------------------------------------------------------------------------------------
# 4) real single-camera footage: same-track vs different-track separability, colour vs gray
# ---------------------------------------------------------------------------------------------------------------
def real_appearance(args) -> dict:
    import cv2
    from argonix.reid import HandcraftedEmbedder, OSNetEmbedder, to_gray3
    weights = str(ROOT / "data" / "models" / "yolo26n.pt")
    bank = build_bank(weights, device=args.device)
    items = [(b["id"], t, cv2.imread(str(ROOT / p))) for b in bank for t, p in zip(b["t"], b["crops"])]
    embedders = {"handcrafted hsv-hog-v1": HandcraftedEmbedder()}
    for n in ("osnet_ain_x1_0_msmt17_cosine.pth", "osnet_x1_0_msmt17.pth", "osnet_x0_25_msmt17.pth"):
        p = ROOT / "data" / "models" / "reid" / n
        if p.exists():
            embedders[n] = OSNetEmbedder(str(p), device=args.device)
    out = {"setting": "real footage, single camera per source: pairs of crops of the same track >= 2 s apart vs "
                      "crops of different tracks from the same source (optimistic for 'same': one viewpoint)",
           "n_crops": len(items), "n_tracks": len(bank), "results": {}}
    for name, emb in embedders.items():
        r = {}
        for mode in ("color", "gray"):
            imgs = [c if mode == "color" else to_gray3(c) for _, _, c in items]
            E = emb.embed(imgs)
            same, diff = [], []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if a[0].split(":")[0] != b[0].split(":")[0]:
                        continue  # compare within one source clip only
                    s = float(E[i] @ E[j])
                    if a[0] == b[0]:
                        if abs(a[1] - b[1]) >= 2.0:
                            same.append(s)
                    else:
                        diff.append(s)
            r[mode] = {"auc": auc(same, diff), "same_mean": round(float(np.mean(same)), 3),
                       "diff_mean": round(float(np.mean(diff)), 3), "n_same": len(same), "n_diff": len(diff)}
        r["gray_auc_drop"] = round(r["color"]["auc"] - r["gray"]["auc"], 4)
        out["results"][name] = r
        print(name, r)
    (OUT / "real_appearance.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--gray", action="store_true", help="with --video: run on grayscale-converted footage")
    ap.add_argument("--real-appearance", action="store_true")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--duration", type=float, default=480.0)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.sim:
        res = main_sim(a)
        (OUT / "sim.json").write_text(json.dumps(res, indent=1))
    if a.real_appearance:
        real_appearance(a)
    if a.video:
        r = run_video(a, gray=a.gray)
        print(json.dumps({k: r[k] for k in ("processing_s", "speed_vs_realtime", "segments",
                                             "segments_matched_to_truth", "vram_peak_mib")}, indent=1))
        print(json.dumps(r["online"]["cross_camera"], indent=1), "IDF1", r["online"]["idf1"])
