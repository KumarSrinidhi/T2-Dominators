"""Remote inference node (argonix/infer_server.py) + client (argonix/remote_inference.py), CPU only, fake model.

The fake detector (tests/test_inference.py) reads a box position from the frame's first pixel, so every result
can be traced back to the frame, client and camera that produced it. Frames travel as raw BGR (lossless) unless a
test is about JPEG. Tracking uses the real Ultralytics ByteTrack on the node.
"""
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argonix.detect import make_backend_factory
from argonix.inference import BatchedInference
from argonix.infer_server import InferenceNode, create_infer_app, load_or_create_token
from argonix.remote_inference import RemoteTrackerBackend, encode_frame, pack_frame
from tests.test_inference import FakeModel, frame

TOKEN = "unit-test-token-0123456789abcdef"  # test fixture only
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ROOT = Path(__file__).resolve().parents[1]


def make_node(delay_s=0.0, max_batch=8, max_wait_ms=5.0, **kw):
    m = FakeModel(delay_s)
    svc = BatchedInference(m, device="cpu", warmup=False, max_batch=max_batch, max_wait_ms=max_wait_ms)
    return InferenceNode(svc, TOKEN, **kw), m


def raw_msg(seq, x):
    f = frame(x)
    return pack_frame(seq, f.tobytes(), "raw", f.shape[1], f.shape[0])


def results(ws, n):
    out = {}
    while len(out) < n:
        r = ws.receive_json()
        if r.get("type") == "result":
            out[r["seq"]] = r
    return out


def stream(c, client, camera):
    ws = c.websocket_connect(f"/v1/stream?client={client}&camera={camera}", headers=AUTH)
    return ws


class LiveServer:
    """A real uvicorn server in a thread (the client under test uses real sockets)."""

    def __init__(self, app, port=0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        self.port = sock.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws_per_message_deflate=False))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]}, daemon=True)
        self.thread.start()
        t0 = time.time()
        while not self.server.started:
            assert time.time() - t0 < 10, "server did not start"
            time.sleep(0.01)

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)


class FakeLocal:
    """Stands in for the CPU fallback model."""
    info = {"detector": "fake-cpu.pt", "runtime": "torch", "device": "cpu"}

    def __init__(self):
        self.calls = 0

    def track(self, f):
        self.calls += 1
        return [(1, (5.0, 5.0, 20.0, 40.0), "person", 0.5)]


def wait_for(fn, timeout=5.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not met in time")


# ------------------------------------------------------------------------------------------------ server
def test_auth_required_for_stream_and_metrics():
    node, _ = make_node()
    with TestClient(create_infer_app(node)) as c:
        assert c.get("/health").json()["ok"] is True  # liveness only, no token needed
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer wrong-token-wrong-token"}).status_code == 401
        assert c.get("/v1/info").status_code == 401
        for headers in ({}, {"Authorization": "Bearer nope-nope-nope-nope"}, {"Authorization": TOKEN}):
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect("/v1/stream?client=a&camera=c1", headers=headers) as ws:
                    ws.receive_json()
        assert node.counters["auth_rejected"] >= 6
        assert not node.svc.has_client("a/c1")  # nothing was allocated for the rejected connections
        info = c.get("/v1/info", headers=AUTH).json()
        assert info["model"]["detector"] == "FakeModel" and info["protocol"] == 1
    node.svc.close()


def test_batches_across_two_clients_and_isolates_their_trackers():
    node, m = make_node(max_wait_ms=300)
    with TestClient(create_infer_app(node)) as c:
        # the SAME camera id from two different clients must get two independent trackers
        with stream(c, "siteA", "cam1") as a, stream(c, "siteB", "cam1") as b:
            ha, hb = a.receive_json(), b.receive_json()
            assert ha["type"] == hb["type"] == "hello" and ha["resumed"] is False
            assert ha["model"]["detector"] == "FakeModel" and ha["session"] != hb["session"]
            ids = {"a": set(), "b": set()}
            for seq in range(1, 7):
                m.batches.clear()
                ts = [threading.Thread(target=ws.send_bytes, args=(raw_msg(seq, x),)) for ws, x in ((a, 60), (b, 200))]
                [t.start() for t in ts]
                [t.join() for t in ts]
                ra, rb = results(a, 1)[seq], results(b, 1)[seq]
                assert ra["status"] == rb["status"] == "ok"
                assert abs(ra["dets"][0][1] - 60) < 1 and abs(rb["dets"][0][1] - 200) < 1  # routed to its sender
                ids["a"].add(ra["dets"][0][0])
                ids["b"].add(rb["dets"][0][0])
                if seq > 1:  # both cameras active: they share ONE batched predict
                    assert sorted(m.batches[0]) == [60, 200]
                    assert ra["server"]["batch"] == 2
            # a shared tracker would see two far-apart boxes alternating and never keep one ID per client
            assert ids == {"a": {1}, "b": {1}}
            srv = ra["server"]
            assert set(srv) >= {"queue_ms", "decode_ms", "infer_ms", "track_ms", "total_ms", "batch"}
        met = c.get("/metrics", headers=AUTH).json()
    assert met["batch_hist"].get("2", 0) >= 5 and met["mean_batch"] > 1
    assert met["counters"]["ok"] == 12 and met["latency_ms"]["n"] == 12 and met["latency_ms"]["p95"] is not None
    assert set(met["detached_trackers"]) == {"siteA/cam1", "siteB/cam1"}  # kept for the TTL after disconnect
    for k in ("queue_depth", "gpu", "clients", "fps_in", "limits", "model", "uptime_s"):
        assert k in met
    node.svc.close()


def test_metrics_report_connected_clients():
    node, _ = make_node()
    with TestClient(create_infer_app(node)) as c:
        with stream(c, "lap", "cam9") as ws:
            ws.receive_json()
            for seq in range(1, 4):
                ws.send_bytes(raw_msg(seq, 50))
                results(ws, 1)
            met = c.get("/metrics", headers=AUTH).json()
            (cl,) = met["clients"]
            assert (cl["client"], cl["camera"], cl["frames"], cl["ok"]) == ("lap", "cam9", 3, 3)
            assert cl["fps_in"] > 0 and cl["latency_ms"]["n"] == 3 and cl["kb_per_frame"] > 0
            assert met["gpu"]["available"] is False  # CPU node: reported, not faked
    node.svc.close()


def test_latest_frame_wins_and_stale_frames_are_dropped():
    node, m = make_node(delay_s=0.3, max_inflight=1, max_age_ms=5000)
    with TestClient(create_infer_app(node)) as c:
        with stream(c, "lap", "cam1") as ws:
            ws.receive_json()
            for seq in range(1, 6):  # the node is busy with frame 1 while 2..5 arrive
                ws.send_bytes(raw_msg(seq, 10 * seq))
            r = results(ws, 5)
    assert r[1]["status"] == "ok" and r[5]["status"] == "ok"  # the in-flight frame, then the newest one
    assert [r[s]["reason"] for s in (2, 3, 4)] == ["superseded"] * 3
    assert [b for b in m.batches] == [[10], [50]]  # superseded frames never reached the model
    node.svc.close()

    node, m = make_node(delay_s=0.3, max_inflight=1, max_age_ms=100)
    with TestClient(create_infer_app(node)) as c:
        with stream(c, "lap", "cam1") as ws:
            ws.receive_json()
            ws.send_bytes(raw_msg(1, 10))
            ws.send_bytes(raw_msg(2, 20))  # waits ~0.3 s behind frame 1, longer than max_age_ms
            r = results(ws, 2)
    assert r[1]["status"] == "ok" and (r[2]["status"], r[2]["reason"]) == ("dropped", "stale")
    assert node.counters["dropped_stale"] == 1
    node.svc.close()


def test_reconnect_resumes_tracker_and_new_connection_replaces_old():
    node, _ = make_node()
    with TestClient(create_infer_app(node)) as c:
        with stream(c, "lap", "cam1") as ws:
            h1 = ws.receive_json()
            ws.send_bytes(raw_msg(1, 60))
            ws.send_bytes(raw_msg(2, 60))
            results(ws, 2)
        with stream(c, "lap", "cam1") as ws:  # within the TTL: same tracker, IDs continue
            h2 = ws.receive_json()
            assert h2["resumed"] is True and h2["session"] == h1["session"]
            ws.send_bytes(raw_msg(3, 60))
            assert results(ws, 1)[3]["dets"][0][0] == 1
            with stream(c, "lap", "cam1") as ws2:  # a second connection for the same camera takes over
                assert ws2.receive_json()["resumed"] is True
                with pytest.raises(WebSocketDisconnect) as ei:
                    ws.receive_json()
                assert ei.value.code == 4409
                ws2.send_bytes(raw_msg(4, 60))
                assert results(ws2, 1)[4]["status"] == "ok"
    node.svc.close()


def test_bad_frames_get_error_replies_not_disconnects():
    node, _ = make_node()
    with TestClient(create_infer_app(node)) as c:
        with stream(c, "lap", "cam1") as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00\x01")
            assert ws.receive_json()["status"] == "error"
            ws.send_bytes(pack_frame(7, b"not a jpeg", "jpeg", 0, 0))
            r = results(ws, 1)[7]
            assert r["status"] == "error" and "decode" in r["reason"]
            f = np.full((240, 320, 3), (90, 0, 0), np.uint8)  # uniform colour survives JPEG almost exactly
            payload, scale, w, h = encode_frame(f, 320, "jpeg", 90)
            ws.send_bytes(pack_frame(8, payload, "jpeg", w, h))
            r = results(ws, 1)[8]
            assert r["status"] == "ok" and abs(r["dets"][0][1] - 90) <= 3
    node.svc.close()


def test_token_file_created_once(tmp_path):
    p = tmp_path / "sub" / "infer_token"
    t1 = load_or_create_token(p)
    assert len(t1) >= 32 and load_or_create_token(p) == t1


# ------------------------------------------------------------------------------------------------ client
def test_client_track_info_and_stats():
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    b = RemoteTrackerBackend(srv.url, "cam1", token=TOKEN, client_id="lap", fmt="raw", max_side=320,
                             connect_timeout_s=2)
    try:
        assert b.info["runtime"] == "remote" and b.info["active"] == "none"
        dets = b.track(frame(60))
        assert len(dets) == 1 and dets[0][0] == 1 and abs(dets[0][1][0] - 60) < 1 and dets[0][2] == "person"
        # a frame larger than max_side is downscaled for the wire and boxes come back in the caller's pixels
        big = np.zeros((480, 640, 3), np.uint8)
        big[0:2, 0:2, 0] = 60
        (d,) = b.track(big)
        assert abs(d[1][0] - 120) < 3  # x=60 in the 320-px frame the node saw == 120 in the caller's 640-px frame
        info = b.info
        assert (info["runtime"], info["active"], info["node"], info["detector"]) == \
            ("remote", "remote", srv.url, "FakeModel")
        assert info["node_name"] and info["transport"] == "raw <= 320 px"
        st = b.camera_stats()
        assert st["remote_state"] == "up" and st["rtt_ms"] > 0 and st["dropped_remote"] == 0
        assert st["rtt_p95_ms"] is not None
    finally:
        b.close()
        srv.stop()
        node.svc.close()


def test_client_timeout_returns_empty_and_counts_drop():
    node, _ = make_node(delay_s=0.4)
    srv = LiveServer(create_infer_app(node))
    b = RemoteTrackerBackend(srv.url, "cam1", token=TOKEN, client_id="lap", fmt="raw", max_side=320,
                             timeout_s=0.1, max_timeouts=2, backoff_min_s=0.05, connect_timeout_s=2)
    try:
        b.start()  # connect first: only the frame round trip is timed below
        assert b.camera_stats()["remote_state"] == "up"
        t0 = time.perf_counter()
        assert b.track(frame(60)) == []
        assert time.perf_counter() - t0 < 0.3  # bounded by the 0.1 s timeout, not by the 0.4 s node
        st = b.camera_stats()
        assert st["dropped_remote"] == 1 and st["remote_timeouts"] == 1 and st["remote_state"] == "up"
        assert b.track(frame(60)) == []  # second consecutive timeout: the node is treated as down
        assert b.camera_stats()["remote_state"] == "down"
        wait_for(lambda: b.camera_stats()["remote_state"] == "up")  # ...and reconnected in the background
    finally:
        b.close()
        srv.stop()
        node.svc.close()


def test_client_unauthorized_state():
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    b = RemoteTrackerBackend(srv.url, "cam1", token="wrong-token-wrong-token", client_id="lap", fmt="raw",
                             backoff_min_s=0.05, connect_timeout_s=2)
    try:
        assert b.track(frame(60)) == []
        assert b.camera_stats()["remote_state"] == "unauthorized"
        assert "403" in b.camera_stats()["remote_error"]
    finally:
        b.close()
        srv.stop()
        node.svc.close()


def test_client_falls_back_locally_and_returns_to_the_node():
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    port = srv.port
    local = FakeLocal()
    b = RemoteTrackerBackend(srv.url, "cam1", token=TOKEN, client_id="lap", fmt="raw", max_side=320,
                             timeout_s=0.5, connect_timeout_s=1, backoff_min_s=0.05, backoff_max_s=0.2,
                             fallback=lambda: local, fallback_after_s=0.3)
    node2 = None
    try:
        remote_ids = {d[0] for d in b.track(frame(60))}
        assert remote_ids == {1}
        srv.stop()  # the node goes away
        node.svc.close()
        assert b.track(frame(60)) == []  # connection lost: nothing for this frame, no fallback yet
        assert b.camera_stats()["remote_state"] in ("down", "connecting")
        dets = wait_for(lambda: b.track(frame(60)), timeout=3)  # after fallback_after_s the CPU model takes over
        assert dets[0][1] == (5.0, 5.0, 20.0, 40.0) and local.calls >= 1
        fb_ids = {d[0] for d in dets}
        assert not fb_ids & remote_ids  # a new tracker never reuses an ID the rules have already seen
        assert b.info["active"] == "local-fallback" and b.info["detector"] == "fake-cpu.pt"
        assert b.info["runtime"] == "remote"
        st = b.camera_stats()
        assert st["remote_state"] == "fallback" and st["fallback_frames"] >= 1

        node2, _ = make_node()  # the node comes back on the same address
        srv = LiveServer(create_infer_app(node2), port=port)

        def remote_again():
            d = b.track(frame(60))
            return d if d and d[0][1][0] == 60 else None
        dets = wait_for(remote_again, timeout=5)
        assert b.info["active"] == "remote" and b.camera_stats()["remote_state"] == "up"
        assert not {d[0] for d in dets} & (remote_ids | fb_ids)
        assert b.camera_stats()["remote_reconnects"] >= 1
    finally:
        b.close()
        srv.stop()
        if node2 is not None:
            node2.svc.close()


def test_factory_config_and_site_integration(tmp_path):
    from argonix.pipeline import Site
    from argonix.server import create_app
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    (tmp_path / "tok").write_text(TOKEN + "\n")
    try:
        f = make_backend_factory({"remote": {"url": srv.url, "token_file": str(tmp_path / "tok"), "fallback": "none",
                                             "client_id": "unit"}, "stitch": {"enabled": False}})
        b = f({"id": "camX"})
        assert isinstance(b, RemoteTrackerBackend) and f.remote["url"] == srv.url and f.backends["camX"] is b
        assert b.info["detector"] == "FakeModel"  # known from /v1/info before the camera's first frame
        f.close()
        with pytest.raises(ValueError):
            make_backend_factory({"remote": {"url": srv.url, "token": TOKEN, "fallback": "cloud"}})

        cfg = json.loads((ROOT / "configs" / "demo_site.json").read_text())
        cfg["cameras"] = cfg["cameras"][:1]
        cfg["record"] = False
        cfg["detector"] = {"remote": {"url": srv.url, "token_file": str(tmp_path / "tok"), "fallback": "none",
                                      "client_id": "site"}}
        for c in cfg["cameras"]:
            c["source"] = str(ROOT / c["source"])
        p = tmp_path / "configs" / "s.json"
        p.parent.mkdir()
        p.write_text(json.dumps(cfg))
        site = Site(p, data_dir=tmp_path / "run", realtime=False)
        assert site.correlator.model_info["runtime"] == "remote"
        site.start()
        site.wait(timeout=120)
        w = site.workers[0]
        assert w.stats["frames"] > 0 and w.stats["remote_state"] in ("up", "closed")
        assert w.stats["rtt_ms"] > 0 and w.stats["dropped_remote"] == 0
        site.stop()
        assert w.backend.state == "closed"  # the worker released its socket when it stopped
        client = TestClient(create_app(site, open_demo=True))
        nodes = client.get("/api/nodes").json()["nodes"]
        (inf,) = [n for n in nodes if n["kind"] == "inference"]
        assert inf["state"] == "up" and inf["url"] == srv.url and inf["model"]["detector"] == "FakeModel"
        assert inf["cameras"][0]["camera"] == w.id and inf["cameras"][0]["rtt_ms"] > 0
        assert TOKEN not in json.dumps(nodes)  # the token never reaches the browser
    finally:
        srv.stop()
        node.svc.close()


def test_nodes_endpoint_degrades_without_remote_and_with_dead_vlm(tmp_path):
    from argonix.server import create_app
    from argonix.vlm import RemoteExplainer
    from tests.test_server import make_site
    site = make_site(tmp_path)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()  # nothing listens here
    client = TestClient(create_app(site, explainer=RemoteExplainer(f"http://127.0.0.1:{dead}"), open_demo=True))
    nodes = client.get("/api/nodes").json()["nodes"]
    assert [n["kind"] for n in nodes] == ["inference", "vlm"]
    assert nodes[0]["location"] == "in-process" and nodes[0]["model"]["detector"] == "fake"
    assert nodes[1]["state"] == "down" and nodes[1]["error"]


# ------------------------------------------------------------------------------------------------ adaptation
def test_ladder_is_monotonically_cheaper():
    from argonix.remote_inference import ladder
    r = ladder(480, 80)
    assert r[0] == (480, 80) and r[-1] == (256, 40)
    assert all(a[0] >= b[0] and a[1] >= b[1] and a != b for a, b in zip(r, r[1:]))
    assert ladder(320, 50, min_side=320) == [(320, 50), (320, 45)]


def test_adaptive_encoder_steps_down_on_a_bandwidth_bound_link_and_recovers():
    from argonix.remote_inference import AdaptiveEncoder, ladder
    a = AdaptiveEncoder(ladder(640, 85), target_ms=250, window=5)
    t = 0.0

    def feed(rtt_fn, n):
        nonlocal t
        for _ in range(n):
            t += 0.1
            a.observe(rtt_fn(a.rung), t)

    def link(rung):  # 60 ms latency + upload at 125 kB/s; JPEG bytes ~ side^2 * quality^1.5 (60 kB at 640/q85)
        side, q = rung
        return 60 + 60 * (side / 640) ** 2 * (q / 85) ** 1.5 * 8

    feed(link, 60)
    assert a.level > 0 and link(a.rung) <= 250       # stepped down until under the target...
    assert link(a.rungs[a.level - 1]) > 250          # ...and no further than needed
    feed(lambda rung: 40, 60)  # the link got fast: bigger frames again
    assert a.level == 0


def test_adaptive_encoder_does_not_degrade_a_latency_bound_link():
    from argonix.remote_inference import AdaptiveEncoder, ladder
    a = AdaptiveEncoder(ladder(480, 80), target_ms=250, window=5, hold_s=60)
    t = 0.0
    for _ in range(40):  # 600 ms whatever the frame size: smaller frames cannot help
        t += 0.1
        a.observe(600, t)
    assert a.level == 0 and a.changes == 2  # tried one rung down, saw no gain, went back and holds
    for _ in range(3):  # a slow link still decides: >= min_samples once max_window_s has passed
        t += 5
        a.observe(600, t)
    assert a.level == 0 and a.rtt[0] == 600


def test_uplink_budget_spaces_sends_and_skips_beyond_max_wait():
    from argonix.remote_inference import UplinkBudget
    b = UplinkBudget(kbps=80)  # 10 kB/s, 10 kB burst
    assert b.reserve(10_000, 2.0) == 0.0
    w = b.reserve(10_000, 2.0)
    assert 0.9 < w <= 1.0
    assert b.reserve(30_000, 2.0) is None  # would wait ~4 s: skipped, nothing reserved


def test_adaptive_client_reports_its_encoding_in_info_and_stats():
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    b = RemoteTrackerBackend(srv.url, "cam1", token=TOKEN, client_id="lap", jpeg_quality=80, max_side=480,
                             adaptive=True, target_rtt_ms=10_000, connect_timeout_s=2)
    try:
        f = np.full((480, 640, 3), (90, 0, 0), np.uint8)
        assert len(b.track(f)) == 1
        assert b.info["transport"].startswith("jpeg q80 <= 480 px (adaptive, rung 1/")
        assert b.camera_stats()["transport"] == b.info["transport"] and b.camera_stats()["remote_fps_max"] > 0
        b._adapt.level = len(b._adapt.rungs) - 1  # force the cheapest rung: smaller payload, same detections
        kb_hi = b.camera_stats()["kb_per_frame"]
        for _ in range(3):
            (d,) = b.track(f)
        assert abs(d[1][0] - 90 * 640 / 256) < 8  # x=90 in the 256-px frame the node saw, in the caller's pixels
        assert b.info["transport"].startswith("jpeg q40 <= 256 px") and b.camera_stats()["kb_per_frame"] < kb_hi
    finally:
        b.close()
        srv.stop()
        node.svc.close()


def test_remote_factory_warns_when_node_lacks_object_classes(caplog):
    """Unattended-object analysis adds detector.extra_classes; a node started without --extra-classes cannot supply them."""
    node, _ = make_node()
    srv = LiveServer(create_infer_app(node))
    try:
        with caplog.at_level("WARNING", logger="argonix.remote_inference"):
            f = make_backend_factory({"remote": {"url": srv.url, "token": TOKEN, "fallback": "none"},
                                      "extra_classes": [24, 26, 28]})
        assert any("--extra-classes 24,26,28" in r.getMessage() for r in caplog.records)
        f.close()
    finally:
        srv.stop()
