"""MediaMTX restream: path config generation, Control API reconcile, credential redaction, the WHEP proxy, the
managed-process fallback, main-stream recording index + retention, and the recordings-table migration.

No MediaMTX binary or network is needed: MediaMTX's HTTP side is an httpx.MockTransport stand-in (FakeMediaMTX)
that implements the v3 Control API calls and the WHEP endpoints Argus uses."""
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from argonix.restream import (MediaMTXClient, Restreamer, conf_differs, desired_paths, go_duration, index_main_recordings,
                              load_config, managed_yaml, parse_segment_start, path_config, reconcile,
                              sweep_main_recordings)
from argonix.server import create_app
from argonix.store import Store
from tests.test_auth import CSRF, PW, fast_hashes, login, write_users  # noqa: F401 (fixture re-export)
from tests.test_server import make_site

SECRET = "Pa55:w@rd/x"   # test fixture only
SECRET_Q = "Pa55%3Aw%40rd%2Fx"
MAIN = f"rtsp://admin:{SECRET_Q}@192.0.2.10:554/Streaming/Channels/101"   # TEST-NET-1, never routed
SUB = f"rtsp://admin:{SECRET_Q}@192.0.2.10:554/Streaming/Channels/102"
ANSWER = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
OFFER = b"v=0\r\no=- 2 2 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
SID = "0f8c2a3e-1b2c-4d5e-8f90-a1b2c3d4e5f6"


def assert_no_secret(obj):
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    assert SECRET not in s and SECRET_Q not in s, s


class FakeMediaMTX:
    """In-memory MediaMTX v3 API + WHEP endpoints. Echoes durations the way MediaMTX does ("1m0s")."""

    def __init__(self, up=True):
        self.up = up
        self.paths: dict[str, dict] = {}
        self.requests: list[httpx.Request] = []
        self.fail_add = ""

    @staticmethod
    def _echo(conf):
        out = dict(conf)
        for k, v in conf.items():
            d = go_duration(v)
            if d is not None and d == 60:
                out[k] = "1m0s"   # MediaMTX re-serialises durations
        return out

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if not self.up:
            raise httpx.ConnectError("connection refused", request=req)
        p = req.url.path
        if p == "/v3/paths/list":
            items = [{"name": n, "online": True, "readers": [{"type": "webRTCSession"}],
                      "tracks2": [{"codec": "H264"}]} for n in self.paths]
            return httpx.Response(200, json={"pageCount": 1, "itemCount": len(items), "items": items})
        if p == "/v3/config/paths/list":
            items = [{"name": n, **c, "rtspAnyPort": False} for n, c in self.paths.items()]
            return httpx.Response(200, json={"pageCount": 1, "itemCount": len(items), "items": items})
        m = re.match(r"^/v3/config/paths/(add|replace|delete)/(.+)$", p)
        if m:
            op, name = m.groups()
            if op == "add":
                if self.fail_add:
                    return httpx.Response(400, json={"status": "error", "error": self.fail_add})
                if name in self.paths:
                    return httpx.Response(400, json={"status": "error", "error": "path already exists"})
                self.paths[name] = self._echo(json.loads(req.content))
            elif op == "replace":
                if name not in self.paths:
                    return httpx.Response(404, json={"status": "error", "error": "path not found"})
                self.paths[name] = self._echo(json.loads(req.content))
            else:
                if self.paths.pop(name, None) is None:
                    return httpx.Response(404, json={"status": "error", "error": "path not found"})
            return httpx.Response(200, json={"status": "ok"})
        m = re.match(r"^/(cam_[^/]+)/whep(?:/([^/]+))?$", p)
        if m:
            name, sid = m.groups()
            if name not in self.paths:
                return httpx.Response(404, json={"status": "error", "error": "no stream is available on path"})
            if req.method == "POST" and not sid:
                return httpx.Response(201, content=ANSWER, headers={
                    "Content-Type": "application/sdp", "Location": f"/{name}/whep/{SID}", "ETag": '"*"',
                    "Accept-Patch": "application/trickle-ice-sdpfrag",
                    "Link": '<stun:stun.example:3478>; rel="ice-server"', "Set-Cookie": "mtx=should-not-pass"})
            if req.method == "PATCH" and sid == SID:
                return httpx.Response(204)
            if req.method == "DELETE" and sid == SID:
                return httpx.Response(200)
            return httpx.Response(404, json={"status": "error", "error": "session not found"})
        return httpx.Response(404, json={"status": "error", "error": "not found"})


def rs_site(tmp_path, record=True, keep_days=7, **extra):
    site = make_site(tmp_path)
    site.cfg["cameras"].append({"id": "door", "name": "Door", "source": SUB, "main_source": MAIN})
    site.cfg["cameras"].append({"id": "lobby", "name": "Lobby", "source": SUB.replace("192.0.2.10", "192.0.2.11")})
    site.cfg["restream"] = {"enabled": True, "api": "http://mtx:9997", "webrtc": "http://mtx:8889",
                            "record": {"enabled": record, "path": "<data>/main_rec", "segment_s": 60,
                                       "keep_days": keep_days}, **extra}
    fake = FakeMediaMTX()
    site.restream = Restreamer(site, transport=httpx.MockTransport(fake))
    return site, fake


# ---------------------------------------------------------------------------------------------------------------
def test_config_defaults_off_and_path_generation(tmp_path):
    rc = load_config({}, tmp_path)
    assert rc["enabled"] is False and rc["record"]["enabled"] is False
    assert rc["record"]["root"] == (tmp_path / "main_rec").resolve()
    rc = load_config({"restream": {"enabled": True, "record": {"enabled": True, "segment_s": 30}}}, tmp_path)
    cam = {"id": "door", "source": SUB, "main_source": MAIN}
    conf = path_config(cam, rc)
    assert conf["source"] == MAIN, "the MAIN stream is restreamed, not the analytics sub-stream"
    assert conf["record"] is True and conf["sourceOnDemand"] is False, "recording needs a continuous pull"
    assert conf["recordFormat"] == "fmp4" and conf["recordSegmentDuration"] == "30s" and conf["recordDeleteAfter"] == "0s"
    assert conf["recordPath"] == (tmp_path / "main_rec").resolve().as_posix() + "/%path/%Y-%m-%d_%H-%M-%S-%f"
    assert conf["rtspTransport"] == "tcp"
    # live only: on-demand pull, no record keys
    live = path_config({"id": "x", "source": SUB}, load_config({"restream": {"enabled": True}}, tmp_path))
    assert live == {"source": SUB, "sourceOnDemand": True, "record": False, "rtspTransport": "tcp"}
    # recording requested but restream off -> recording off too
    assert load_config({"restream": {"record": {"enabled": True}}}, tmp_path)["record"]["enabled"] is False
    # files, webcams, disabled cameras and MJPEG-over-HTTP cannot be restreamed
    assert path_config({"id": "f", "source": "data/synth/cam1.mp4"}, rc) is None
    assert path_config({"id": "w", "source": "0"}, rc) is None
    assert path_config({"id": "d", "source": SUB, "enabled": False}, rc) is None
    assert path_config({"id": "m", "source": "http://cam/video.mjpg"}, rc) is None
    assert path_config({"id": "h", "source": "https://cdn/live/index.m3u8"}, rc)["source"].endswith(".m3u8")
    names = desired_paths({"cameras": [cam, {"id": "f", "source": "clip.mp4"}]}, rc)
    assert list(names) == ["cam_door"]
    assert go_duration("1m0s") == 60 and go_duration("1.5h") == 5400 and go_duration("abc") is None
    assert not conf_differs({"recordSegmentDuration": "1m0s", "x": 1}, {"recordSegmentDuration": "60s"})
    assert conf_differs({"source": "rtsp://a"}, {"source": "rtsp://b"})


def test_managed_yaml_has_no_paths_or_credentials(tmp_path):
    rc = load_config({"restream": {"enabled": True, "api": "http://127.0.0.1:9987", "webrtc": "http://127.0.0.1:8889",
                                   "webrtc_additional_hosts": ["192.168.137.135"], "password": SECRET}}, tmp_path)
    y = managed_yaml(rc, tmp_path / "mediamtx" / "mediamtx.log")
    assert "apiAddress: 127.0.0.1:9987" in y and "webrtcAddress: 127.0.0.1:8889" in y
    # default: the ICE/UDP media port listens on localhost only and is the only candidate offered
    assert 'webrtcLocalUDPAddress: "127.0.0.1:8189"' in y and "webrtcIPsFromInterfaces: false" in y
    assert 'webrtcAdditionalHosts: ["127.0.0.1", "192.168.137.135"]' in y
    # webrtc_udp_host "" restores every interface (LAN viewers), as before
    lan = managed_yaml(load_config({"restream": {"enabled": True, "webrtc_udp_host": "",
                                                 "webrtc_additional_hosts": ["192.168.137.135"]}}, tmp_path),
                       tmp_path / "mediamtx" / "mediamtx.log")
    assert 'webrtcLocalUDPAddress: ":8189"' in lan and "webrtcIPsFromInterfaces: true" in lan
    assert 'webrtcAdditionalHosts: ["192.168.137.135"]' in lan
    assert "paths: {}" in y and "recordDeleteAfter: 0s" in y
    for off in ("rtsp", "rtmp", "hls", "srt", "moq", "playback"):
        assert f"\n{off}: false" in y
    assert_no_secret(y)
    assert 'ips: ["127.0.0.1", "::1"]' in y   # only Argus (localhost) may read/use the API


def test_reconcile_adds_updates_removes_only_cam_paths(tmp_path):
    fake = FakeMediaMTX()
    client = MediaMTXClient("http://mtx:9997", transport=httpx.MockTransport(fake))
    rc = load_config({"restream": {"enabled": True, "record": {"enabled": True}}}, tmp_path)
    fake.paths = {"cam_gone": {"source": "rtsp://x/1"}, "someone_elses": {"source": "publisher"}}
    want = desired_paths({"cameras": [{"id": "door", "source": MAIN}, {"id": "lobby", "source": SUB}]}, rc)
    res = reconcile(client, want)
    assert sorted(res["added"]) == ["cam_door", "cam_lobby"] and res["removed"] == ["cam_gone"] and not res["errors"]
    assert "someone_elses" in fake.paths and "cam_gone" not in fake.paths
    # a second pass is a no-op even though MediaMTX echoes "1m0s" for "60s"
    assert reconcile(client, want) == {"added": [], "updated": [], "removed": [], "errors": []}
    # a changed source is replaced in place
    want["cam_lobby"] = {**want["cam_lobby"], "source": SUB.replace("102", "101")}
    res = reconcile(client, want)
    assert res["updated"] == ["cam_lobby"] and fake.paths["cam_lobby"]["source"].endswith("/101")
    # MediaMTX error texts that echo a source URL are redacted
    fake.paths.clear()
    fake.fail_add = f"invalid source '{MAIN}'"
    res = reconcile(client, want)
    assert res["errors"] and "admin:***@192.0.2.10" in res["errors"][0]
    assert_no_secret(res)


def test_status_and_camera_api_never_leak_credentials(tmp_path):
    site, fake = rs_site(tmp_path)
    rs = site.restream
    assert rs.ensure_running() == "external"
    res = rs.sync()
    assert sorted(res["added"]) == ["cam_door", "cam_lobby"]   # cam1 is a file source: analytics only
    assert fake.paths["cam_door"]["source"] == MAIN            # MediaMTX needs the real URL ...
    client = TestClient(create_app(site, open_demo=True))
    st = client.get("/api/restream").json()                     # ... but nothing that leaves Argus has it
    assert st["mode"] == "external" and st["record"]["enabled"] is True
    door = st["cameras"]["door"]
    assert door["available"] and door["online"] and door["codecs"] == ["H264"] and door["main_is_separate"]
    assert door["main_source"] == "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/101"
    assert st["cameras"]["cam1"]["available"] is False and "file" in st["cameras"]["cam1"]["reason"]
    assert_no_secret(st)
    assert client.get("/api/config").json()["restream"] is True
    cams = client.get("/api/cameras").json()
    assert next(c for c in cams if c["id"] == "door")["main_source"].startswith("rtsp://admin:***@")
    assert_no_secret(cams)


def test_camera_api_main_source(tmp_path):
    site = make_site(tmp_path)
    client = TestClient(create_app(site, open_demo=True))
    r = client.post("/api/cameras", json={"name": "Gate", "source": SUB, "main_source": MAIN, "enabled": False})
    assert r.status_code == 200, r.text
    assert r.json()["main_source"] == "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/101"
    assert site.camera_config("cam_gate")["main_source"] == MAIN
    # echoing the redacted value back keeps the stored secret; '' clears it
    r = client.put("/api/cameras/cam_gate", json={"main_source": r.json()["main_source"], "name": "Gate 2"})
    assert r.status_code == 200 and site.camera_config("cam_gate")["main_source"] == MAIN
    assert client.put("/api/cameras/cam_gate", json={"main_source": "data/synth/cam1.mp4"}).status_code == 400
    assert client.put("/api/cameras/cam_gate", json={"main_source": ""}).json()["main_source"] == ""
    assert "main_source" not in site.camera_config("cam_gate")
    assert_no_secret(site.store.list_audit(limit=100))
    assert_no_secret(client.get("/api/cameras").json())


def test_whep_proxy(tmp_path):
    site, fake = rs_site(tmp_path)
    site.restream.ensure_running(); site.restream.sync()
    client = TestClient(create_app(site, open_demo=True), headers=CSRF)
    r = client.post("/api/cameras/door/whep", content=OFFER, headers={"Content-Type": "application/sdp",
                                                                      "Origin": "http://testserver"})
    assert r.status_code == 201, r.text
    assert r.content == ANSWER and r.headers["content-type"].startswith("application/sdp")
    assert r.headers["location"] == f"/api/cameras/door/whep/{SID}"   # rewritten: PATCH/DELETE come back to us
    assert r.headers["etag"] == '"*"' and "ice-server" in r.headers["link"]
    assert "mtx=" not in r.headers.get("set-cookie", "")
    fwd = [q for q in fake.requests if q.url.path == "/cam_door/whep"][-1]
    assert str(fwd.url) == "http://mtx:8889/cam_door/whep" and fwd.content == OFFER
    # browser headers are not forwarded (MediaMTX sees a same-host request from Argus)
    assert not {"origin", "cookie", "x-argonix-csrf", "x-forwarded-for"} & {k.lower() for k in fwd.headers}
    # trickle ICE and teardown go through the proxy too
    p = client.patch(f"/api/cameras/door/whep/{SID}", content=b"a=candidate:1 1 udp 1 192.0.2.1 5000 typ host",
                     headers={"Content-Type": "application/trickle-ice-sdpfrag", "If-Match": '"*"'})
    assert p.status_code == 204
    assert fake.requests[-1].method == "PATCH" and fake.requests[-1].headers["if-match"] == '"*"'
    assert client.delete(f"/api/cameras/door/whep/{SID}").json() == {"ok": True}
    assert client.delete("/api/cameras/door/whep/bad..id").status_code in (400, 404)
    # errors
    assert client.post("/api/cameras/door/whep", content=OFFER, headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post("/api/cameras/nope/whep", content=OFFER, headers={"Content-Type": "application/sdp"}).status_code == 404
    r = client.post("/api/cameras/cam1/whep", content=OFFER, headers={"Content-Type": "application/sdp"})
    assert r.status_code == 409 and "file" in r.json()["detail"]
    fake.paths.pop("cam_lobby")
    r = client.post("/api/cameras/lobby/whep", content=OFFER, headers={"Content-Type": "application/sdp"})
    assert r.status_code == 404 and "no stream" in r.json()["detail"]
    fake.up = False
    r = client.post("/api/cameras/door/whep", content=OFFER, headers={"Content-Type": "application/sdp"})
    assert r.status_code == 502 and "unreachable" in r.json()["detail"]
    audit = [a for a in site.store.list_audit(limit=50) if a["action"] == "live_view_webrtc"]
    assert audit and audit[-1]["details"]["camera"] == "door"
    assert_no_secret(audit)


def test_whep_requires_evidence_view_and_csrf(tmp_path):
    site, fake = rs_site(tmp_path)
    site.restream.ensure_running(); site.restream.sync()
    write_users(tmp_path, [("vic", "viewer"), ("aud", "auditor")])
    from argonix.auth import Auth
    app = create_app(site, auth=Auth(tmp_path / "users.json", tmp_path / "secret.key"))
    hdr = {"Content-Type": "application/sdp"}
    assert TestClient(app, headers=CSRF).post("/api/cameras/door/whep", content=OFFER, headers=hdr).status_code == 401
    assert login(app, "aud").post("/api/cameras/door/whep", content=OFFER, headers=hdr).status_code == 403
    vic = login(app, "vic")
    assert vic.post("/api/cameras/door/whep", content=OFFER, headers=hdr).status_code == 201
    no_csrf = TestClient(app, cookies=vic.cookies)   # a cookie-carrying request without the CSRF header
    assert no_csrf.post("/api/cameras/door/whep", content=OFFER, headers=hdr).status_code == 403
    evil = {**hdr, "Origin": "https://evil.example"}
    assert vic.post("/api/cameras/door/whep", content=OFFER, headers=evil).status_code == 403


def test_whep_disabled_returns_503(tmp_path):
    site = make_site(tmp_path)
    assert site.restream is None   # off by default
    client = TestClient(create_app(site, open_demo=True), headers=CSRF)
    assert client.get("/api/restream").json()["enabled"] is False
    r = client.post("/api/cameras/cam1/whep", content=OFFER, headers={"Content-Type": "application/sdp"})
    assert r.status_code == 503


def test_managed_process_started_when_api_down(tmp_path):
    site, fake = rs_site(tmp_path, api="http://127.0.0.1:9987", mediamtx_bin=str(tmp_path / "mediamtx.exe"))
    (tmp_path / "mediamtx.exe").write_bytes(b"")
    fake.up = False
    calls = []

    class Proc:
        pid, returncode = 4242, None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate"); self.returncode = 0

        def wait(self, timeout=None):
            return 0

    def popen(args, **kw):
        calls.append(args)
        fake.up = True   # the "binary" brings the API up
        return Proc()

    rs = Restreamer(site, transport=httpx.MockTransport(fake), popen=popen)
    assert rs.ensure_running() == "managed"
    assert calls[0][0].endswith("mediamtx.exe") and calls[0][1].endswith("mediamtx.yml")
    yml = Path(calls[0][1]).read_text()
    assert "apiAddress: 127.0.0.1:9987" in yml
    assert_no_secret(yml)
    rs.sync()
    assert "cam_door" in fake.paths
    rs.close()
    assert "terminate" in calls and rs.mode == "stopped"
    # no binary, API down -> "down" with a clear reason
    fake.up = False
    site.restream = Restreamer(site, transport=httpx.MockTransport(fake))
    site.restream.cfg["mediamtx_bin"] = ""
    assert site.restream.ensure_running() == "down" and "not reachable" in site.restream.error


def _seg(root: Path, cam: str, t: float, size=2048, end=None) -> Path:
    d = root / f"cam_{cam}"
    d.mkdir(parents=True, exist_ok=True)
    dt = datetime.fromtimestamp(t)
    f = d / (dt.strftime("%Y-%m-%d_%H-%M-%S-") + f"{dt.microsecond:06d}.mp4")
    f.write_bytes(b"\0" * size)
    e = end if end is not None else t + 60
    os.utime(f, (e, e))
    return f


def test_segment_name_parsing():
    t = parse_segment_start("2026-09-25_14-03-07-250000.mp4")
    assert t == datetime(2026, 9, 25, 14, 3, 7, 250000).timestamp()
    assert parse_segment_start("2026-09-25_14-03-07-250000+0530.mp4") == \
        datetime.fromisoformat("2026-09-25T14:03:07.250000+05:30").timestamp()
    assert parse_segment_start("notes.txt") is None and parse_segment_start("2026-13-40_00-00-00-0.mp4") is None


def test_recording_index_api_and_retention(tmp_path):
    site, fake = rs_site(tmp_path, keep_days=7)
    root = site.restream.cfg["record"]["root"]
    now = time.time()
    t_old = now - 10 * 86400
    t0 = now - 200
    old = _seg(root, "door", t_old)
    a = _seg(root, "door", t0)                    # 60 s, contiguous with b
    b = _seg(root, "door", t0 + 60, end=now)      # still being written
    _seg(root, "lobby", t0 + 5)
    stray = root / "cam_door" / "keep-me.txt"
    stray.write_text("not a segment")
    site.store.add_recording("cam1", "recordings/cam1/x.mp4", t0, t0 + 60)   # analytics row
    st = index_main_recordings(site.store, root)
    assert st["indexed"] == 4
    assert index_main_recordings(site.store, root)["indexed"] == 0   # idempotent
    os.utime(b, (now + 3, now + 3))
    assert index_main_recordings(site.store, root)["updated"] == 1   # growing segment: end refreshed
    mains = site.store.list_recordings(stream="main")
    assert {r["camera"] for r in mains} == {"door", "lobby"} and all(r["stream"] == "main" for r in mains)
    ra = next(r for r in mains if r["path"].endswith(a.name))
    assert abs(ra["start"] - t0) < 1e-3 and abs(ra["end"] - (t0 + 60)) < 1e-3 and ra["size"] == 2048

    client = TestClient(create_app(site, open_demo=True))
    both = client.get("/api/recordings").json()
    assert {r["stream"] for r in both} == {"main", "analytics"}
    assert client.get("/api/recordings?stream=main&camera=door").json()[0]["url"].startswith("/api/recordings/")
    ana = client.get("/api/recordings?stream=analytics").json()
    assert ana[0]["url"] == "/media/recordings/cam1/x.mp4"
    assert all("/" not in r["path"] and ":" not in r["path"] for r in client.get("/api/recordings?stream=main").json())
    assert client.get("/api/recordings?stream=bogus").status_code == 400
    # "open main-stream recording at this time": evidence frame at t0+30, 4 s pre-roll -> segment a, offset 26 s
    at = client.get(f"/api/recordings/at?camera=door&t={t0 + 30}").json()
    assert at["path"] == a.name and abs(at["offset_s"] - 26) < 0.01 and at["stream"] == "main"
    at = client.get(f"/api/recordings/at?camera=door&t={t0 + 62}").json()
    assert at["path"] == a.name and abs(at["offset_s"] - 58) < 0.01
    at = client.get(f"/api/recordings/at?camera=door&t={t0 + 62}&pre_s=0").json()
    assert at["path"] == b.name and abs(at["offset_s"] - 2) < 0.01
    assert client.get(f"/api/recordings/at?camera=door&t={now + 3600}").status_code == 404
    v = client.get(at["url"])
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    rng = client.get(at["url"], headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100
    # a row pointing outside the recording root is never served
    site.store.add_recording("door", (tmp_path / "configs" / "s.json").as_posix(), t0, t0 + 1, stream="main")
    rid = next(r["id"] for r in site.store.list_recordings(stream="main") if r["path"].endswith("s.json"))
    assert client.get(f"/api/recordings/{rid}/video").status_code == 404
    site.store.delete_recording(rid)

    # retention: only the 10-day-old segment goes; the stray file and recent segments stay
    sw = sweep_main_recordings(site.store, root, keep_days=7, now=now)
    assert sw["deleted"] == 1 and not old.exists() and a.exists() and b.exists() and stray.exists()
    assert not any(r["path"].endswith(old.name) for r in site.store.list_recordings(stream="main"))
    assert sweep_main_recordings(site.store, root, keep_days=0, now=now)["deleted"] == 0   # 0 disables
    # a file deleted behind our back drops out of the index
    a.unlink()
    assert index_main_recordings(site.store, root)["removed"] == 1
    assert len(site.store.list_recordings(stream="analytics")) == 1   # analytics rows untouched


def test_recordings_table_migration(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE recordings(id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT, path TEXT, start REAL, end REAL)")
    con.execute("INSERT INTO recordings(camera,path,start,end) VALUES('cam1','recordings/cam1/a.mp4',1.0,61.0)")
    con.commit(); con.close()
    s = Store(db)
    rows = s.list_recordings()
    assert rows == [{"id": 1, "camera": "cam1", "path": "recordings/cam1/a.mp4", "start": 1.0, "end": 61.0,
                     "stream": "analytics", "size": None}]
    s.db.close()
    s = Store(db)   # idempotent
    s.add_recording("cam1", "recordings/cam1/b.mp4", 61.0, 121.0)
    assert [r["stream"] for r in s.list_recordings()] == ["analytics", "analytics"]


def test_loop_reconciles_and_indexes(tmp_path):
    site, fake = rs_site(tmp_path)
    rs = site.restream
    rs.cfg.update(sync_interval_s=0.05, index_interval_s=0.05)
    _seg(rs.cfg["record"]["root"], "door", time.time() - 90)
    rs.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not (site.store.list_recordings(stream="main") and "cam_door" in fake.paths):
            time.sleep(0.05)
        assert "cam_door" in fake.paths and site.store.list_recordings(stream="main")
        # a camera removed from the config disappears from MediaMTX on the next pass
        site.cfg["cameras"] = [c for c in site.cfg["cameras"] if c["id"] != "lobby"]
        rs.sync_soon()
        deadline = time.time() + 10
        while time.time() < deadline and "cam_lobby" in fake.paths:
            time.sleep(0.05)
        assert "cam_lobby" not in fake.paths
    finally:
        rs.close()
