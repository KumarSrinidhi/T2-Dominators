"""Known-answer tests for tools/eval_metrics.py (CLEAR-MOT, IDF1, HOTA, MOT17 preprocessing, event matching)."""
import math

import numpy as np
import pytest

from tools.eval_metrics import (clear_mot, combine_counts, event_matches_episode, hota, identity, iou_matrix,
                                match_events, mot17_preprocess, parse_mot_gt, tracking_metrics, wilson)

B1 = (0, 0, 10, 10)
B2 = (100, 100, 110, 110)
FAR = (500, 500, 510, 510)


def seq(*frames):
    """seq([(id, box), ...], ...) -> {1: [...], 2: [...], ...}"""
    return {i + 1: list(f) for i, f in enumerate(frames)}


def test_iou_matrix():
    m = iou_matrix([B1, (0, 0, 10, 20)], [B1, (5, 0, 15, 10), FAR])
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(50 / 150)
    assert m[1, 0] == pytest.approx(0.5)
    assert m[0, 2] == 0.0


def test_perfect_tracking():
    gt = seq([(1, B1), (2, B2)], [(1, B1), (2, B2)], [(1, B1)])
    tr = seq([(7, B1), (8, B2)], [(7, B1), (8, B2)], [(7, B1)])
    m = tracking_metrics(gt, tr)
    assert (m["MOTA"], m["MOTP"], m["IDF1"], m["IDSW"], m["FP"], m["FN"]) == (1.0, 1.0, 1.0, 0, 0, 0)
    assert m["HOTA"] == pytest.approx(1.0) and m["DetA"] == pytest.approx(1.0) and m["AssA"] == pytest.approx(1.0)
    assert (m["MT"], m["PT"], m["ML"]) == (2, 0, 0)


def test_id_switch():
    # one person, 4 frames; the tracker changes its ID halfway
    gt = seq([(1, B1)], [(1, B1)], [(1, B1)], [(1, B1)])
    tr = seq([(1, B1)], [(1, B1)], [(2, B1)], [(2, B1)])
    c = clear_mot(gt, tr)
    assert (c["TP"], c["FP"], c["FN"], c["IDSW"]) == (4, 0, 0, 1)
    assert c["MOTA"] == pytest.approx(0.75)
    i = identity(gt, tr)
    assert (i["IDTP"], i["IDFP"], i["IDFN"]) == (2, 2, 2)
    assert i["IDF1"] == pytest.approx(0.5)
    h = hota(gt, tr)
    assert h["DetA"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(0.5)
    assert h["HOTA"] == pytest.approx(math.sqrt(0.5), abs=1e-4)


def test_misses_and_false_positives():
    # two people for 2 frames; the tracker follows only person 1 and adds one far-away false box in frame 2
    gt = seq([(1, B1), (2, B2)], [(1, B1), (2, B2)])
    tr = seq([(5, B1)], [(5, B1), (6, FAR)])
    c = clear_mot(gt, tr)
    assert (c["TP"], c["FP"], c["FN"], c["IDSW"]) == (2, 1, 2, 0)
    assert c["MOTA"] == pytest.approx(1 - (2 + 1 + 0) / 4)
    assert (c["MT"], c["ML"]) == (1, 1)
    i = identity(gt, tr)
    assert i["IDF1"] == pytest.approx(2 * 2 / (4 + 3), abs=1e-4)


def test_iou_threshold_and_motp():
    half = (0, 0, 10, 5)  # IoU 0.5 with B1: counts as a match at the 0.5 threshold
    low = (0, 0, 10, 4)   # IoU 0.4: a miss + a false positive
    gt = seq([(1, B1)], [(1, B1)])
    c = clear_mot(gt, seq([(1, half)], [(1, low)]))
    assert (c["TP"], c["FP"], c["FN"]) == (1, 1, 1)
    assert c["MOTP"] == pytest.approx(0.5)


def test_previous_match_kept_when_still_valid():
    # frame 2: tracker 2 overlaps GT 1 better, but tracker 1 is still >= 0.5 IoU -> no switch (CLEAR-MOT rule)
    gt = seq([(1, B1)], [(1, B1)])
    tr = seq([(1, B1)], [(1, (0, 0, 10, 6)), (2, B1)])
    c = clear_mot(gt, tr)
    assert c["IDSW"] == 0 and c["FP"] == 1


def test_fragmentation_counts_tracked_again():
    gt = seq([(1, B1)], [(1, B1)], [(1, B1)])
    tr = seq([(1, B1)], [(1, FAR)], [(1, B1)])
    c = clear_mot(gt, tr)
    assert c["Frag"] == 1 and c["IDSW"] == 0


def test_mot17_preprocess_distractors_and_consider():
    text = "\n".join([
        "1,1,0,0,10,10,1,1,1.0",       # pedestrian
        "1,2,100,100,10,10,1,7,1.0",   # static person (distractor)
        "1,3,200,200,10,10,0,1,1.0",   # pedestrian with consider = 0 (ignored)
        "1,4,300,300,10,10,1,3,1.0",   # car (not a distractor: a box on it stays a false positive)
        "2,1,0,0,10,10,1,1,0.1",       # low-visibility pedestrian: still counted by default
    ])
    raw = parse_mot_gt(text)
    tr = {1: [(9, (0, 0, 10, 10)), (10, (100, 100, 110, 110)), (11, (300, 300, 310, 310))], 2: [(9, B1)]}
    gt, trp, st = mot17_preprocess(raw, tr, [1, 2])
    assert [i for i, _ in gt[1]] == [1]
    assert [i for i, _ in trp[1]] == [9, 11]
    assert st["tracker_boxes_removed_on_distractors"] == 1
    c = clear_mot(gt, trp)
    assert (c["TP"], c["FP"], c["FN"]) == (2, 1, 0)
    gt_v, trp_v, _ = mot17_preprocess(raw, tr, [1, 2], min_vis=0.25)
    assert gt_v[2] == [] and trp_v[2] == []


def test_parse_mot_gt_truncates():
    raw = parse_mot_gt("1,1,0,0,10,10,1,1,1\n5,1,0,0,10,10,1,1,1\n", max_frame=3)
    assert list(raw) == [1] and raw[1][0][1] == (0, 0, 10, 10)


def test_combined_counts_pool_sequences():
    a = tracking_metrics(seq([(1, B1)], [(1, B1)], [(1, B1)], [(1, B1)]),
                         seq([(1, B1)], [(1, B1)], [(2, B1)], [(2, B1)]))    # MOTA 0.75, IDF1 0.5, HOTA 0.707
    b = tracking_metrics(seq([(1, B1), (2, B2)]), seq([(5, B1), (6, B2)]))   # perfect, 2 detections
    c = combine_counts([a, b])
    assert (c["TP"], c["IDSW"], c["gt_dets"]) == (6, 1, 6)
    assert c["MOTA"] == pytest.approx(5 / 6, abs=1e-4)
    assert c["IDF1"] == pytest.approx(2 * (2 + 2) / 12, abs=1e-4)
    # TrackEval pooling: AssA weighted by TP -> (0.5 * 4 + 1 * 2) / 6
    assert c["AssA"] == pytest.approx(4 / 6, abs=1e-4) and c["DetA"] == pytest.approx(1.0)
    assert combine_counts([a])["HOTA"] == pytest.approx(a["HOTA"], abs=1e-4)


def test_agrees_with_py_motmetrics_on_random_tracks():
    """Optional cross-check (runs only where py-motmetrics is installed): FP, FN and IDF1 must agree exactly.
    IDSW (hence MOTA) may differ slightly: TrackEval, which we follow, keeps only last-frame correspondences,
    py-motmetrics keeps the last-ever one (on the seeds below 1 of 10 cases differs by 2 switches)."""
    if not hasattr(np, "asfarray"):  # motmetrics 1.4 still calls the NumPy-1 API
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
    mm = pytest.importorskip("motmetrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        gt, tr, nxt = {}, {}, 1000
        pos, vel = rng.uniform(0, 400, (8, 2)), rng.normal(0, 3, (8, 2))
        tid = {i: i + 100 for i in range(8)}
        for f in range(1, 61):
            pos += vel
            gt[f] = [(i, (pos[i, 0], pos[i, 1], pos[i, 0] + 30, pos[i, 1] + 60)) for i in range(8)]
            tr[f] = [(tid[i], tuple(np.array(b) + rng.normal(0, 6, 4))) for i, b in gt[f] if rng.random() < 0.85]
            for i in range(8):
                if rng.random() < 0.03:
                    tid[i], nxt = nxt, nxt + 1
        acc = mm.MOTAccumulator(auto_id=False)
        for f in gt:
            xywh = lambda rows: np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in rows]).reshape(-1, 4)
            acc.update([i for i, _ in gt[f]], [i for i, _ in tr[f]],
                       mm.distances.iou_matrix(xywh(gt[f]), xywh(tr[f]), max_iou=0.5), frameid=f)
        ref = mm.metrics.create().compute(acc, metrics=["idf1", "num_false_positives", "num_misses"], name="x")
        ours = tracking_metrics(gt, tr)
        assert ours["IDF1"] == pytest.approx(float(ref["idf1"].iloc[0]), abs=1e-3)
        assert ours["FP"] == int(ref["num_false_positives"].iloc[0])
        assert ours["FN"] == int(ref["num_misses"].iloc[0])


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97


def ep(f0, f1, box=(50, 50, 70, 100), clip="c"):
    return {"clip": clip, "f0": f0, "f1": f1, "boxes": {f: [box] for f in range(f0, f1 + 1)}}


def ev(f0, f1, box, clip="c", step=5):
    return {"clip": clip, "f0": f0, "f1": f1, "boxes": {f: [box] for f in range(f0, f1 + 1, step)}}


def test_interval_matching_time_and_space():
    g = ep(100, 200)
    assert event_matches_episode(ev(150, 160, (55, 60, 65, 95)), g, tol=10, prox_px=30)       # inside the box
    assert event_matches_episode(ev(150, 160, (85, 60, 95, 95)), g, tol=10, prox_px=30)       # 20 px away
    assert not event_matches_episode(ev(150, 160, (200, 60, 210, 95)), g, tol=10, prox_px=30)  # other person
    assert not event_matches_episode(ev(230, 240, (55, 60, 65, 95)), g, tol=10, prox_px=30)   # too late
    assert event_matches_episode(ev(205, 240, (55, 60, 65, 95)), g, tol=10, prox_px=30)       # within tolerance
    assert not event_matches_episode(ev(150, 160, (55, 60, 65, 95), clip="d"), g, tol=10, prox_px=30)
    assert event_matches_episode(ev(150, 160, (200, 60, 210, 95)), g, tol=10, prox_px=None)   # temporal only


def test_match_events_precision_recall():
    gts = [ep(100, 200), ep(400, 500, box=(200, 50, 220, 100)), ep(800, 900)]
    preds = [ev(120, 150, (55, 60, 65, 95)),       # hits GT 0
             ev(130, 190, (55, 60, 65, 95)),       # hits GT 0 again (many-to-one)
             ev(420, 480, (205, 60, 215, 95)),     # hits GT 1
             ev(600, 650, (55, 60, 65, 95))]       # false alarm
    m = match_events(gts, preds, tol=10, prox_px=30)
    assert (m["events"], m["true_events"], m["gt_episodes"], m["detected_episodes"]) == (4, 3, 3, 2)
    assert m["precision"] == 0.75 and m["recall"] == pytest.approx(0.667, abs=1e-3)
    assert m["precision_wilson95"][0] < 0.75 < m["precision_wilson95"][1]
    empty = match_events([], [], tol=10, prox_px=30)
    assert empty["precision"] is None and empty["recall"] is None
