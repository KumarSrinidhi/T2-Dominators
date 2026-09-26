"""tools/preflight.py: the pure checks (no network, no model weights, no GPU)."""
import json
import socket

import cv2
import numpy as np

from tools import preflight as pf


def test_python_version_window():
    assert pf.check_python((3, 13, 1)).status == pf.PASS
    assert pf.check_python((3, 11, 0)).status == pf.PASS
    assert pf.check_python((3, 10, 9)).status == pf.FAIL
    assert pf.check_python((3, 14, 0)).status == pf.FAIL


def test_torch_and_cuda():
    assert pf.check_torch(None, False, "auto", False).status == pf.FAIL
    assert pf.check_torch("2.14.0+cu126", True, "auto", True).status == pf.PASS
    assert pf.check_torch("2.14.0+cpu", False, "0", False).status == pf.FAIL          # config wants a GPU
    assert pf.check_torch("2.14.0+cpu", False, "cuda:0", False).status == pf.FAIL
    w = pf.check_torch("2.14.0+cpu", False, "auto", True)                             # CPU wheel on a GPU host
    assert w.status == pf.WARN and "--no-sync" in w.detail
    assert pf.check_torch("2.14.0+cpu", False, "auto", False).status == pf.INFO


def test_classify_source():
    assert pf.classify_source("0") == "webcam"
    assert pf.classify_source("rtsp://u:p@10.0.0.2:554/Streaming/Channels/102") == "stream"
    assert pf.classify_source("HTTPS://cam/video") == "stream"
    assert pf.classify_source("rtsp://${RTSP_USER}:${RTSP_PASSWORD}@${RTSP_HOST}/x") == "placeholder"
    assert pf.classify_source("concat:a|b://x") == "bad_scheme"
    assert pf.classify_source("data/synth/cam1.mp4") == "file"


def test_camera_checks_without_io(tmp_path):
    c = pf.check_camera({"id": "c1", "source": "rtsp://${RTSP_USER}@h/x"}, tmp_path, 1.0)
    assert c.status == pf.FAIL and "placeholder" in c.detail
    assert pf.check_camera({"id": "c2", "source": "0"}, tmp_path, 1.0).status == pf.INFO
    assert pf.check_camera({"id": "c3", "source": "gopher://h/x"}, tmp_path, 1.0).status == pf.FAIL
    miss = pf.check_camera({"id": "c4", "source": "data/reid_eval/scene/camA.mp4"}, tmp_path, 1.0)
    assert miss.status == pf.FAIL and "make_reid_demo.py" in miss.detail
    assert "make_synthetic_handoff.py" in pf.generator_hint("data/synth_handoff/camB.mp4")
    assert "run.py" in pf.generator_hint("data\\synth\\cam1.mp4")


def test_file_source_opens_and_reads(tmp_path):
    p = tmp_path / "clip.avi"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    for _ in range(3):
        wr.write(np.full((48, 64, 3), 128, np.uint8))
    wr.release()
    c = pf.check_camera({"id": "f", "source": "clip.avi"}, tmp_path, 1.0)  # relative to the project root
    assert c.status == pf.PASS and "64x48" in c.detail
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    assert pf.check_camera({"id": "b", "source": str(tmp_path / "bad.mp4")}, tmp_path, 1.0).status == pf.FAIL


def test_weights_resolution(tmp_path):
    models = tmp_path / "data" / "models"
    assert pf.check_weights("auto", tmp_path, cuda=False).status == pf.WARN      # downloaded by name at startup
    assert pf.check_weights("data/models/yolo26n.pt", tmp_path, False).status == pf.WARN
    assert pf.check_weights("data/models/my_custom.pt", tmp_path, False).status == pf.FAIL
    assert pf.check_weights("x", tmp_path, False, remote=True).status == pf.INFO
    models.mkdir(parents=True)
    (models / "yolo26n.pt").write_bytes(b"x")
    (models / "yolo26s.pt").write_bytes(b"x")
    assert pf.check_weights("auto", tmp_path, cuda=False).detail.endswith("yolo26n.pt")
    assert pf.check_weights("auto", tmp_path, cuda=True).detail.endswith("yolo26s.pt")
    assert pf.check_weights("data/models/yolo26n.pt", tmp_path, False).status == pf.PASS
    eng = pf.check_weights("data/models/yolo26s-480-b16.engine", tmp_path, True)   # engine missing, .pt present
    assert eng.status == pf.WARN and "yolo26s.pt" in eng.detail
    assert pf.check_weights("data/models/yolo99x-480-b16.engine", tmp_path, True).status == pf.FAIL
    (models / "yolo26s-480-b16.engine").write_bytes(b"x")
    assert pf.check_weights("data/models/yolo26s-480-b16.engine", tmp_path, True).status == pf.PASS
    assert pf.check_weights("data/models/yolo26s-480-b16.engine", tmp_path, False).status == pf.WARN  # no CUDA


def test_disk_port_urls_and_exit_code(tmp_path):
    gib = 2**30
    assert pf.check_disk(20 * gib, 5).status == pf.PASS
    assert pf.check_disk(3 * gib, 5).status == pf.WARN
    assert pf.check_disk(gib // 2, 5).status == pf.FAIL
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        assert not pf.port_free("127.0.0.1", s.getsockname()[1])
    finally:
        s.close()
    assert pf.base_url("http://127.0.0.1:8100/explain") == "http://127.0.0.1:8100"
    assert pf.base_url("127.0.0.1:18000") == "http://127.0.0.1:18000"
    assert pf.check_data_dir(tmp_path / "run").status == pf.PASS
    ok = [pf.Check(pf.PASS, "a"), pf.Check(pf.WARN, "b"), pf.Check(pf.INFO, "c")]
    assert pf.exit_code(ok) == 0 and pf.summary_line(ok).startswith("READY with warnings")
    assert pf.exit_code(ok + [pf.Check(pf.FAIL, "d")]) == 1
    assert pf.summary_line([pf.Check(pf.FAIL, "d")]).startswith("NOT READY")


def test_app_and_vlm_probes_with_fake_http():
    def fake(status, body=None, err=""):
        return lambda url, timeout: (status, body, err)
    assert pf.check_app("http://h:1", 1, fake(None, None, "ConnectError")).status == pf.FAIL
    assert pf.check_app("http://h:1", 1, fake(200, {"setup_required": True, "open_demo": True})).status == pf.WARN
    setup = pf.check_app("http://h:1", 1, fake(200, {"setup_required": True, "open_demo": False}))
    assert setup.status == pf.WARN and "setup_code.txt" in setup.detail
    assert pf.check_app("http://h:1/x", 1, fake(200, {"setup_required": False})).status == pf.PASS
    assert pf.check_vlm("http://h:2", 1, fake(503)).status == pf.WARN
    assert pf.check_vlm("http://h:2", 1, fake(200, {"loaded": None, "loading": True})).status == pf.WARN
    up = pf.check_vlm("http://h:2/explain", 1, fake(200, {"loaded": "qwen", "mode": "4bit", "device": "cuda"}))
    assert up.status == pf.PASS and "qwen" in up.detail


def test_reid_and_accounts(tmp_path):
    assert pf.check_reid({}, tmp_path) is None
    assert pf.check_reid({"reid": {"enabled": True}}, tmp_path).status == pf.WARN
    (tmp_path / "data" / "models" / "reid").mkdir(parents=True)
    (tmp_path / "data" / "models" / "reid" / "osnet_x0_25_msmt17.pth").write_bytes(b"x")
    assert pf.check_reid({"reid": {"enabled": True}}, tmp_path).status == pf.PASS
    cfg = tmp_path / "configs" / "site.json"
    cfg.parent.mkdir()
    assert "setup code" in pf.check_accounts(cfg, open_demo=False).detail
    (cfg.parent / "users.json").write_text(json.dumps({"alice": {"roles": ["admin"]}}))
    assert "1 account" in pf.check_accounts(cfg, open_demo=False).detail


def test_main_fails_on_a_missing_clip(tmp_path, capsys):
    cfg = tmp_path / "configs" / "site.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"detector": {"weights": "data/models/yolo26n.pt", "device": "cpu"},
                               "cameras": [{"id": "cam1", "source": "data/synth/cam1.mp4"}]}))
    rc = pf.main(["--config", str(cfg), "--data", str(tmp_path / "run"), "--host", "127.0.0.1", "--port", "0",
                  "--min-free-gb", "0"])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL  camera cam1" in out and "NOT READY" in out
