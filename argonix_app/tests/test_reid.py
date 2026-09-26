"""Multi-camera Re-ID + global IDs: crops/embedders, fusion scoring, threshold/top-k, merge/split, one-to-one exits,
fragments/re-entry, operator corrections (lock + persistence), retention, calibration, API permissions, gid in tracks.
CPU only and fast: a fake (vector) appearance model everywhere except the small architecture checks."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argonix.auth import Auth, hash_password
from argonix.global_ids import IdentityEngine, StaticGate
from argonix.pipeline import Site
from argonix.reid import (EmbeddingService, HandcraftedEmbedder, OSNetEmbedder, crop_quality, is_monochrome,
                          to_gray3)
from argonix.server import create_app

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
PW = "correct-horse-battery"  # test fixture only
D = 16
E = np.eye(D, dtype=np.float32)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def cosvec(c: float, base: int = 0, other: int = 3):
    """A unit vector with cosine similarity c to E[base]."""
    return unit(c * E[base] + math.sqrt(max(0.0, 1 - c * c)) * E[other])


class Gate:
    """Deterministic topology stand-in: 1-30 s window, uniform density, P(dest) = 0.5."""
    source = "test gate"

    def window(self, a, b):
        return 1.0, 30.0

    def density(self, a, b, dt):
        return 1.0 / 29.0

    def prior(self, a, key, b):
        return 0.5

    def expected(self, a, b):
        return {"p50": 6.0, "source": "test"}


# llr(s) = 60 s - 30 (clipped to +-6): s = 0.5 is neutral, s >= 0.6 strongly "same", s <= 0.4 strongly "different"
APP = {"color": {"same": [0.8, 0.1], "diff": [0.2, 0.1]}, "gray": {"same": [0.8, 0.1], "diff": [0.2, 0.1]}}


def engine(tmp=None, **kw):
    return IdentityEngine(["camA", "camB"], {"appearance": APP, "calib_min_neg": 10 ** 6, **kw}, tmp, Gate())


def box(xn, yn=0.7, h=120.0):
    x, y = xn * W, yn * H
    return (x - 0.15 * h, y - h, x + 0.15 * h, y)


def run(eng, tracks, t0, t1, fps=5.0, cams=("camA", "camB"), mono=False):
    """tracks: [{cam, tid, t0, t1, x0, x1, y?, emb: vector | fn(k) -> vector}] (straight walks)."""
    counts = {}
    for i in range(int(round((t1 - t0) * fps)) + 1):
        t = round(t0 + i / fps, 6)
        for c in cams:
            dets = []
            for tr in tracks:
                if tr["cam"] == c and tr["t0"] <= t <= tr["t1"]:
                    a = (t - tr["t0"]) / max(1e-6, tr["t1"] - tr["t0"])
                    dets.append((tr["tid"], box(tr["x0"] + a * (tr["x1"] - tr["x0"]), tr.get("y", 0.7)), "person", 0.9))
            for sid, bb, q in eng.observe(c, t, W, H, dets):
                tr = next(x for x in tracks if x["cam"] == c and x["tid"] == int(sid.split(":")[1]))
                k = counts.get(sid, 0)
                counts[sid] = k + 1
                e = tr["emb"](k) if callable(tr["emb"]) else tr["emb"]
                eng.add_embedding(sid, e, e if mono else None, {"q": q, "mono": mono})


def seg(eng, cam, tid):
    return next(s for s in eng.segs.values() if s.cam == cam and s.tid == tid)


A_EXIT = {"cam": "camA", "tid": 1, "t0": 0.0, "t1": 3.0, "x0": 0.5, "x1": 0.97}      # leaves camA on the right
B_ENTRY = {"cam": "camB", "tid": 2, "t0": 8.0, "t1": 12.0, "x0": 0.03, "x1": 0.5}    # enters camB on the left


# ------------------------------------------------------------------------------------------------ crops / embedders
def test_crop_quality_monochrome_and_gray():
    assert crop_quality((100, 100, 140, 230), W, H, 0.9) == pytest.approx(0.9)  # 130 px tall (>= good height), inside
    assert crop_quality((100, 100, 140, 230), W, H, 0.4) == 0             # low confidence
    assert crop_quality((0, 100, 40, 230), W, H, 0.9) == 0                # truncated by the left border
    assert crop_quality((100, 400, 140, 539), W, H, 0.9) == 0             # truncated by the bottom border
    assert crop_quality((100, 100, 120, 130), W, H, 0.9) == 0             # too small
    assert crop_quality((100, 100, 300, 230), W, H, 0.9) == 0             # wider than tall: not a standing person
    assert 0 < crop_quality((100, 100, 130, 164), W, H, 0.9) < 0.5        # small: lower weight
    rng = np.random.default_rng(0)
    col = rng.integers(0, 255, (64, 32, 3), dtype=np.uint8)
    g = to_gray3(col)
    assert g.shape == col.shape and not is_monochrome(col) and is_monochrome(g)


def test_handcrafted_embedder_and_batching_service():
    rng = np.random.default_rng(1)
    crops = [rng.integers(0, 255, (120, 50, 3), dtype=np.uint8) for _ in range(5)]
    emb = HandcraftedEmbedder()
    X = emb.embed(crops)
    assert X.shape == (5, 192) and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    assert float(X[0] @ emb.embed([crops[0]])[0]) > 0.999
    got = []
    svc = EmbeddingService(emb, lambda k, ec, eg, m: got.append((k, ec, eg, m)), max_batch=8, max_wait_ms=200)
    for i, c in enumerate(crops):
        svc.submit(f"s{i}", c, {"q": 1.0, "gray": i % 2 == 0, "mono": False})
    for _ in range(100):
        if len(got) == 5:
            break
        time.sleep(0.05)
    svc.close()
    assert sorted(k for k, *_ in got) == [f"s{i}" for i in range(5)]
    assert svc.stats["batches"] <= 2 and svc.stats["images"] == 8           # 5 colour + 3 gray, batched
    assert all((eg is not None) == (int(k[1]) % 2 == 0) for k, ec, eg, m in got)
    sync = []
    EmbeddingService(emb, lambda *a: sync.append(a), sync=True).submit("m", to_gray3(crops[0]),
                                                                        {"mono": True, "gray": True})
    assert sync and sync[0][1] is sync[0][2]  # a colourless crop: its colour embedding doubles as the gray one


@pytest.mark.parametrize("arch", ["osnet_x0_25", "osnet_ain_x0_25"])
def test_osnet_architectures_load_torchreid_style_checkpoints(tmp_path, arch):
    torch = pytest.importorskip("torch")
    from argonix import osnet
    m = osnet.build(arch)
    sd = {"module." + k: v for k, v in m.state_dict().items()}           # torchreid checkpoints: DataParallel prefix
    sd["module.classifier.weight"] = torch.zeros(7, 512)                   # + a classifier head that we drop
    p = tmp_path / f"{arch}_test.pth"
    torch.save(sd, p)
    emb = OSNetEmbedder(str(p), device="cpu", batch_size=2)
    rng = np.random.default_rng(2)
    X = emb.embed([rng.integers(0, 255, (90 + 10 * i, 40, 3), dtype=np.uint8) for i in range(3)])
    assert X.shape == (3, 512) and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    assert emb.info["name"] == arch and emb.info["device"] == "cpu" and emb.info["fp16"] is False


# ------------------------------------------------------------------------------------------------ fusion + decisions
def test_fused_link_carries_evidence_and_mismatch_starts_new_id():
    eng = engine()
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[0]}], 0, 20)
    a, b = seg(eng, "camA", 1), seg(eng, "camB", 2)
    assert a.exit == "right" and b.entry == "left"
    assert b.gid == a.gid and b.pred == a.id and a.succ == b.id
    ev = b.link
    assert ev["kind"] == "handoff" and ev["decided_by"] == "auto" and ev["fused_confidence"] >= 0.6
    assert abs(ev["dt_s"] - 5.0) < 0.3 and ev["topology_prob"] == 0.5 and ev["expected_dt_s"]["p50"] == 6.0
    assert ev["appearance"]["similarity"] > 0.99 and ev["appearance"]["llr"] == 6.0
    assert ev["direction"]["exit_motion_ok"] and ev["direction"]["entry_motion_ok"]
    assert b.alternatives[-1]["new_identity"] and b.alternatives[-1]["confidence"] < 0.1

    eng = engine()
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[1]}], 0, 20)
    a, b = seg(eng, "camA", 1), seg(eng, "camB", 2)
    assert b.gid != a.gid and b.pred is None
    alt = b.alternatives[0]
    assert alt["segment"] == a.id and alt["confidence"] < 0.01 and alt["evidence"]["appearance"]["llr"] == -6.0


def test_threshold_and_top_k_alternatives():
    eng = engine(top_k=2)
    tracks = [{**A_EXIT, "emb": E[0]},
              {"cam": "camA", "tid": 3, "t0": 0.4, "t1": 3.4, "x0": 0.5, "x1": 0.97, "y": 0.8, "emb": E[1]},
              {"cam": "camA", "tid": 4, "t0": 0.2, "t1": 3.2, "x0": 0.5, "x1": 0.97, "y": 0.9, "emb": E[2]},
              {**B_ENTRY, "emb": unit(E[0] + E[1])}]                     # looks like 1 and 3 equally (cos 0.71)
    run(eng, tracks, 0, 20)
    b = seg(eng, "camB", 2)
    assert b.pred is None and b.gid not in {seg(eng, "camA", t).gid for t in (1, 3, 4)}  # below threshold: new ID
    alts = b.alternatives
    assert len(alts) == 3 and alts[-1]["new_identity"]                    # top-2 + "new identity"
    assert {x["segment"] for x in alts[:2]} == {seg(eng, "camA", 1).id, seg(eng, "camA", 3).id}
    assert abs(alts[0]["confidence"] - alts[1]["confidence"]) < 0.05 and alts[0]["confidence"] < 0.6
    assert alts[0]["confidence"] >= alts[1]["confidence"]


def test_ablation_modes():
    eng = engine(mode="topology")
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[1]}], 0, 20)
    ev = seg(eng, "camB", 2).alternatives[0]["evidence"]
    assert ev["appearance"] is None and ev["topology_prob"] == 0.5        # appearance ignored entirely
    eng = engine(mode="appearance")
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[0]}], 0, 20)
    b = seg(eng, "camB", 2)
    assert b.pred == seg(eng, "camA", 1).id and b.link["topology_prob"] is None and b.link["appearance"]["llr"] == 6.0


def test_merge_when_later_crops_match():
    eng = engine()
    amb = cosvec(0.5)                                                      # neutral appearance
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "t1": 14.0, "emb": lambda k: amb if k < 6 else E[0]}], 0, 11)
    a, b = seg(eng, "camA", 1), seg(eng, "camB", 2)
    g_first = b.gid
    assert b.decided and b.pred is None and g_first != a.gid              # topology alone ~0.5 < 0.6: new ID
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "t1": 14.0, "emb": lambda k: amb if k < 6 else E[0]}], 11.2, 20)
    assert b.pred == a.id and b.gid == a.gid                               # MERGED once the evidence arrived
    assert eng.resolve(g_first) == a.gid
    assert any(e["type"] == "merge" and e["gid"] == g_first and e["into"] == a.gid for e in eng.events)
    j = eng.journey(g_first)
    assert j["gid"] == a.gid and j["merged_into"] == a.gid and [s["id"] for s in j["segments"]] == [a.id, b.id]


def test_split_when_later_crops_disagree():
    eng = engine()
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "t1": 14.0, "emb": lambda k: E[0] if k < 5 else -E[0]}], 0, 20)
    a, b = seg(eng, "camA", 1), seg(eng, "camB", 2)
    assert b.pred is None and b.gid != a.gid                               # linked early, SPLIT on re-scoring
    assert any(e["type"] == "split" and e["segment"] == b.id for e in eng.events)
    assert eng.gids[b.gid]["split_from"] == a.gid and eng.metrics()["auto"]["splits"] == 1


def test_one_exit_feeds_one_entry_better_claimant_wins():
    eng = engine()
    b1 = {**B_ENTRY, "emb": cosvec(0.515)}                                 # ~0.72 alone: linked first
    b2 = {**B_ENTRY, "tid": 5, "t0": 9.0, "t1": 13.0, "y": 0.85, "emb": E[0]}
    run(eng, [{**A_EXIT, "emb": E[0]}, b1, b2], 0, 22)
    a, s1, s2 = seg(eng, "camA", 1), seg(eng, "camB", 2), seg(eng, "camB", 5)
    assert s2.pred == a.id and s2.gid == a.gid and a.succ == s2.id
    assert s1.pred is None and s1.gid != a.gid
    assert eng.metrics()["auto"]["displaced"] == 1


def test_same_camera_fragment_and_reentry():
    eng = engine()
    run(eng, [{"cam": "camA", "tid": 1, "t0": 0.0, "t1": 3.0, "x0": 0.3, "x1": 0.5, "emb": E[0]},
              {"cam": "camA", "tid": 2, "t0": 3.8, "t1": 7.0, "x0": 0.52, "x1": 0.6, "emb": E[0]}], 0, 15)
    a, b = seg(eng, "camA", 1), seg(eng, "camA", 2)
    assert a.exit == "lost" and b.entry is None and b.pred == a.id and b.link["kind"] == "fragment"
    eng = engine()
    run(eng, [{**A_EXIT, "emb": E[0]}, {"cam": "camA", "tid": 5, "t0": 12.0, "t1": 16.0, "x0": 0.97, "x1": 0.6,
                                        "emb": E[0]}], 0, 25)
    b = seg(eng, "camA", 5)
    assert b.entry == "right" and b.link["kind"] == "reentry" and b.gid == seg(eng, "camA", 1).gid


def test_direction_inconsistency_lowers_confidence():
    good, bad = engine(), engine()
    run(good, [{**A_EXIT, "emb": cosvec(0.52)}, {**B_ENTRY, "emb": E[0]}], 0, 20)
    # enters at the left border but keeps walking left (outward, 0.04 frame widths/s)
    run(bad, [{**A_EXIT, "emb": cosvec(0.52)}, {**B_ENTRY, "x0": 0.08, "x1": 0.02, "t1": 9.5, "emb": E[0]}], 0, 20)
    cg = seg(good, "camB", 2).alternatives[0]
    cb = seg(bad, "camB", 2).alternatives[0]
    assert cg["evidence"]["direction"]["llr"] == 0 and cb["evidence"]["direction"]["llr"] == -1.5
    assert cb["confidence"] < cg["confidence"]


# ------------------------------------------------------------------------------------------------ operator corrections
def test_operator_link_unlink_lock_and_persistence(tmp_path):
    d = tmp_path / "reid"
    eng = engine(d)
    tracks = [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "t1": 16.0, "emb": E[1]}]
    run(eng, tracks, 0, 11)
    a, b = seg(eng, "camA", 1), seg(eng, "camB", 2)
    assert b.gid != a.gid
    res = eng.operator_link(b.id, a.id, "olga")                           # order does not matter
    assert res["gid"] == a.gid and b.pred == a.id and b.locked and b.link["decided_by"] == "olga"
    assert b.link["kind"] == "operator" and b.link["model_kind"] == "handoff"
    run(eng, tracks, 11.2, 25)                                             # b closes: re-scoring must not undo it
    assert b.pred == a.id and b.gid == a.gid
    res = eng.operator_unlink(b.id, "olga")
    assert b.pred is None and b.gid == res["new_gid"] != a.gid and (a.id, b.id) in eng.cannot and b.locked
    with pytest.raises(ValueError):
        eng.operator_unlink(b.id, "olga")                                  # nothing left to unlink
    eng.save()
    lines = (d / "corrections.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["action"] for x in lines] == ["link", "unlink"]

    eng2 = engine(d)                                                       # restart
    a2, b2 = eng2.segs[a.id], eng2.segs[b.id]
    assert b2.locked and b2.pred is None and b2.gid == b.gid and (a.id, b.id) in eng2.cannot
    assert eng2.next_gid == eng.next_gid and not any(s.crops for s in eng2.segs.values())  # no embeddings persisted
    eng2.operator_link(a.id, b.id, "sam")
    assert b2.gid == a2.gid and (a.id, b.id) not in eng2.cannot
    assert eng2.metrics()["operator"]["links"] == 2


def test_operator_merge_and_overlap_rejected():
    eng = engine()
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[1]},
              {"cam": "camA", "tid": 7, "t0": 9.0, "t1": 11.0, "x0": 0.4, "x1": 0.45, "y": 0.9, "emb": E[2]}], 0, 20)
    a, b, c = seg(eng, "camA", 1), seg(eng, "camB", 2), seg(eng, "camA", 7)
    with pytest.raises(ValueError):
        eng.operator_merge(b.gid, c.gid, "olga")                           # b and c overlap in time
    res = eng.operator_merge(b.gid, a.gid, "olga")
    assert res["gid"] == a.gid == b.gid and b.pred == a.id and b.locked
    with pytest.raises(KeyError):
        eng.operator_link(a.id, "camB:99:1", "olga")


# ------------------------------------------------------------------------------------------------ privacy / calibration
def test_retention_drops_embeddings_then_journeys(tmp_path):
    eng = engine(tmp_path / "reid", embedding_ttl_s=10.0, retention_s=40.0)
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[0]}], 0, 20)
    a = seg(eng, "camA", 1)
    eng.offer_thumb(a.id, 0.9, b"\xff\xd8fakejpeg")
    run(eng, [], 20.2, 30)
    assert a.id in eng.segs and not a.crops                                 # embeddings gone after 10 s
    assert a.thumb_path and (tmp_path / a.thumb_path).exists()             # thumbnail kept until retention
    run(eng, [], 30.2, 70)
    assert not eng.segs and not eng.identities() and not (tmp_path / a.thumb_path).exists()


def test_negative_calibration_from_co_visible_tracks():
    eng = IdentityEngine(["camA"], {"appearance": None, "calib_min_neg": 1}, None, Gate(),
                         sim_priors={"color": {"same": [0.7, 0.1], "diff": [0.4, 0.1]}})
    run(eng, [{"cam": "camA", "tid": 1, "t0": 0, "t1": 5, "x0": 0.3, "x1": 0.4, "emb": E[0]},
              {"cam": "camA", "tid": 2, "t0": 0, "t1": 5, "x0": 0.6, "x1": 0.7, "y": 0.9, "emb": cosvec(0.1)}],
        0, 10, cams=("camA",))
    same, diff, src = eng.app_params("color")
    assert len(eng.neg["color"]) == 1 and src.startswith("site") and abs(diff[0] - 0.1) < 1e-4
    assert same == [0.7, 0.1]                                              # "same" stays the model prior


def test_gray_pairing_for_monochrome_segments():
    eng = engine(color_mode="auto")
    assert not eng.need_gray(False) and eng.need_gray(True)
    run(eng, [{**A_EXIT, "emb": E[0]}, {**B_ENTRY, "emb": E[0]}], 0, 20, mono=True)
    b = seg(eng, "camB", 2)
    assert b.mono and b.link["appearance"]["mode"] == "gray"
    eng.set_mono("camA", True)
    assert eng.need_gray(False)                                            # colour crops now also get gray vectors
    assert engine(color_mode="gray").pair_mode(b, b) == "gray"


def test_static_gate_without_handoff():
    g = StaticGate({"min_dt_s": 0.0, "max_dt_s": 60.0}, ["a", "b"])
    assert g.window("a", "b") == (0.0, 60.0) and abs(g.density("a", "b", 5) - 1 / 60) < 1e-9
    assert abs(g.prior("a", "right", "b") - 1 / 3) < 1e-9 and g.expected("a", "b") is None


# ------------------------------------------------------------------------------------------------ site + API
class FakeEmbedder:
    info = {"name": "fake", "kind": "test", "dim": D, "device": "cpu", "fp16": False}
    sim = APP

    def embed(self, crops):
        return np.tile(E[0], (len(crops), 1))


class Walker:
    info = {"detector": "fake", "tracker": "fake"}

    def __init__(self, cam):
        self.cam, self.n = cam, 0

    def track(self, frame):
        self.n += 1
        t = self.n / 5.0
        if self.cam == "cam1" and t <= 4.0:
            return [(1, (480 + 115 * t - 15, 350, 480 + 115 * t + 15, 450), "person", 0.9)]
        if self.cam == "cam2" and 7.0 <= t <= 12.0:
            return [(9, (10 + 60 * (t - 7), 350, 40 + 60 * (t - 7), 450), "person", 0.9)]
        return []


def make_site(tmp_path, monkeypatch, reid=True):
    import argonix.reid
    monkeypatch.setattr(argonix.reid, "make_embedder", lambda cfg, d: (FakeEmbedder(), []))
    cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
    cfg["record"] = False
    for c in cfg["cameras"]:
        c["source"] = str(ROOT / c["source"])
        c["zones"] = []
    cfg["handoff"] = {"enabled": True, "cold_start": False, "max_dt_s": 15, "min_labels": 1000}
    if reid:
        cfg["reid"] = {"enabled": True, "sync": True}
    p = tmp_path / "configs" / "s.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(cfg))
    return Site(p, data_dir=tmp_path / "run", realtime=False, backend_factory=lambda c: Walker(c["id"]))


def login(app, user):
    c = TestClient(app, headers={"X-Argonix-CSRF": "1"})  # cookie-authenticated writes need the CSRF header
    assert c.post("/api/login", json={"username": user, "password": PW}).status_code == 200
    return c


def test_site_reid_api_permissions_and_gid_in_tracks(tmp_path, monkeypatch):
    site = make_site(tmp_path, monkeypatch)
    assert site.reid in site.analyzers and site.analyzers.index(site.reid) > site.analyzers.index(site.handoff)
    site.start(); site.wait(timeout=120); site.stop()
    eng = site.reid.engine
    s1 = next(s for s in eng.segs.values() if s.cam == "cam1")
    s2 = next(s for s in eng.segs.values() if s.cam == "cam2")
    assert s2.pred == s1.id and s2.gid == s1.gid and s2.link["topology_source"] == "handoff topology"
    assert s1.crops and s1.thumb_path and (tmp_path / "run" / s1.thumb_path).exists()

    # live tracks carry the estimated global ID
    eng.observe("cam2", 100.0, W, H, [(42, box(0.5), "person", 0.9)])
    s = next(x for x in eng.segs.values() if x.tid == 42)
    s.decided, s.gid = True, 77
    site.workers[1].tracks = [{"camera": "cam2", "track_id": 42, "x": 1.0, "y": 2.0}]
    tr = next(t for t in site.tracks() if t["track_id"] == 42)
    assert tr["gid"] == 77 and tr["gid_label"] == "G-77" and tr["gid_conf"] is None

    users = {n: {**hash_password(PW), "roles": [r]} for n, r in
             [("vic", "viewer"), ("olga", "operator"), ("ada", "admin"), ("aud", "auditor")]}
    (tmp_path / "users.json").write_text(json.dumps(users))
    app = create_app(site, auth=Auth(tmp_path / "users.json", tmp_path / "secret.key"))
    anon = TestClient(app)
    for path in ("/api/reid/tracks", "/api/reid/metrics", f"/api/reid/journey/{s1.gid}"):
        assert anon.get(path).status_code == 401
    vic, olga, ada, aud = (login(app, u) for u in ("vic", "olga", "ada", "aud"))
    ids = vic.get("/api/reid/tracks?since_s=100000").json()["identities"]
    row = next(r for r in ids if r["gid"] == s1.gid)
    assert row["label"] == f"G-{s1.gid}" and row["cameras"] == ["cam1", "cam2"] and row["n_segments"] == 2
    assert row["min_link_confidence"] > 0.6 and "name" not in json.dumps(row).lower()
    j = vic.get(f"/api/reid/journey/{s1.gid}").json()
    assert [x["camera"] for x in j["segments"]] == ["cam1", "cam2"] and j["segments"][1]["link"]["dt_s"] > 0
    assert j["segments"][0]["crop"].endswith(".jpg") and vic.get("/media/" + j["segments"][0]["crop"]).status_code == 200
    assert aud.get(f"/api/reid/journey/{s1.gid}").status_code == 403       # auditor: no evidence.view
    assert vic.get("/api/reid/journey/99999").status_code == 404
    m = vic.get("/api/reid/metrics").json()
    assert m["auto"]["links"] >= 1 and m["embedder"]["name"] == "fake" and "research-grade" in m["note"]

    body = {"segment": s2.id}
    assert vic.post("/api/reid/unlink", json=body).status_code == 403
    assert ada.post("/api/reid/unlink", json=body).status_code == 403     # admin role lacks alert.acknowledge
    r = olga.post("/api/reid/unlink", json=body)
    assert r.status_code == 200 and r.json()["new_gid"] != s1.gid
    assert olga.post("/api/reid/unlink", json=body).status_code == 409
    assert olga.post("/api/reid/link", json={"a": s1.id, "b": "nope"}).status_code == 404
    assert olga.post("/api/reid/link", json={"a": s1.id}).status_code == 400
    r = olga.post("/api/reid/link", json={"gid_a": s1.gid, "gid_b": s2.gid, "note": "same coat"})
    assert r.status_code == 200 and r.json()["gid"] == s1.gid == s2.gid
    acts = [a["action"] for a in site.store.list_audit(limit=50)]
    assert {"reid_unlink", "reid_merge", "reid_journey_viewed"} <= set(acts)
    assert not site.store.list_alerts()                                     # Re-ID never raises alerts


def test_reid_endpoints_absent_when_disabled(tmp_path, monkeypatch):
    site = make_site(tmp_path, monkeypatch, reid=False)
    assert not hasattr(site, "reid")
    c = TestClient(create_app(site, open_demo=True))
    r = c.get("/api/reid/metrics")
    assert r.status_code == 404 and r.json()["detail"] == "Not Found"
    assert all("gid" not in t for t in site.tracks())
