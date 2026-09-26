"""Optional MediaMTX restreamer: smooth full-rate live video (WebRTC/WHEP) and main-stream recording.

Off by default (site config ``"restream": {"enabled": true, ...}``). Manual section 15 V-01/V-02 and section 03 R-01:
MediaMTX pulls each camera's MAIN stream once (``main_source``, default ``source``), forwards the camera's own encoded
frames to browsers over WebRTC without transcoding, and records fMP4 segments. Argus stays the only door:

* **Paths** ``cam_<camera id>`` are created/updated/deleted through MediaMTX's Control API (v3,
  ``/v3/config/paths/add|replace|delete/<name>``). Argus owns the ``cam_`` namespace on that MediaMTX and removes
  ``cam_*`` paths it did not ask for. Live-only paths use ``sourceOnDemand`` (the camera is pulled only while someone
  watches); recorded paths pull continuously.
* **Managed process**: if the API is not reachable and ``mediamtx_bin`` is set, Argus writes a minimal
  ``mediamtx.yml`` (WebRTC + API on localhost only, everything else off, no paths and so no credentials in the file)
  into ``<data>/mediamtx/`` and starts MediaMTX itself, restarting it if it exits.
* **WebRTC signalling** goes through ``POST /api/cameras/{cam}/whep`` (argonix/api_restream.py), which forwards the SDP
  offer to ``<webrtc>/cam_<id>/whep`` from localhost; media then flows peer-to-peer over ICE (UDP 8189 by default).
* **Recording**: segments land in ``<record.path>/cam_<id>/YYYY-MM-DD_HH-MM-SS-ffffff.mp4`` (MediaMTX's local time,
  so MediaMTX and Argus must share a clock and time zone: same host). ``index_main_recordings`` puts them in the
  recordings table as ``stream='main'``; ``sweep_main_recordings`` deletes segments older than ``keep_days``
  (MediaMTX's own deletion is off: ``recordDeleteAfter: 0s``).

Camera credentials go to MediaMTX (it has to pull the camera) but never into logs, API responses or error texts:
every string that leaves this module passes through ``cameras.redact``.
"""
from __future__ import annotations

import copy
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx

from .cameras import redact

PATH_PREFIX = "cam_"
SEGMENT_EXT = ".mp4"
# MediaMTX recordPath variables: %path %Y %m %d %H %M %S %f (µs); names are local time unless %z is present
RECORD_TEMPLATE = "%path/%Y-%m-%d_%H-%M-%S-%f"
SEG_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{1,6})([+-]\d{4})?\.mp4$")
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
RESTREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps")
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mediamtx_bin": "",
    "api": "http://127.0.0.1:9997",
    "webrtc": "http://127.0.0.1:8889",
    "user": "",                      # only for an external MediaMTX that requires credentials for api/read
    "password": "",
    "webrtc_udp_port": 8189,         # the one ICE media port browsers must reach (managed process only)
    "webrtc_udp_host": "127.0.0.1",  # where that port listens: localhost only by default (smooth view in a browser
                                     # on this machine); "" = every interface, so LAN browsers can play too
    "webrtc_additional_hosts": [],   # extra LAN/VPN IPs offered as ICE candidates (managed process only)
    "sync_interval_s": 30.0,         # reconcile paths (MediaMTX API changes live in memory only)
    "index_interval_s": 30.0,        # scan the recording directory into the recordings table
    "sweep_interval_s": 600.0,       # retention sweep
    "record": {"enabled": False, "path": "<data>/main_rec", "segment_s": 60, "part_s": 1, "keep_days": 7},
}


# ------------------------------------------------------------------------------------------------ configuration
def load_config(site_cfg: dict, data_dir: Path) -> dict:
    """The site's ``restream`` section merged over DEFAULTS, with the record path resolved (``<data>`` = the run's
    data directory; a relative path is relative to it)."""
    raw = site_cfg.get("restream") or {}
    rc = copy.deepcopy(DEFAULTS)
    rc.update({k: copy.deepcopy(v) for k, v in raw.items() if k != "record"})
    rc["record"].update(raw.get("record") or {})
    rec = rc["record"]
    p = str(rec.get("path") or "<data>/main_rec").replace("<data>", str(data_dir))
    root = Path(p)
    if not root.is_absolute():
        root = Path(data_dir) / root
    rec["root"] = root.resolve()
    rec["segment_s"] = max(1.0, float(rec.get("segment_s", 60)))
    rec["part_s"] = max(0.1, float(rec.get("part_s", 1)))
    rec["keep_days"] = max(0.0, float(rec.get("keep_days", 7)))
    rec["enabled"] = bool(rec.get("enabled")) and bool(rc.get("enabled"))
    rc["enabled"] = bool(rc.get("enabled"))
    for k in ("api", "webrtc"):
        rc[k] = str(rc[k]).rstrip("/")
    return rc


def path_name(cam_id: str) -> str:
    return PATH_PREFIX + cam_id


def main_source(cam: dict) -> str:
    return str(cam.get("main_source") or cam.get("source") or "")


def restreamable(src: str) -> tuple[bool, str]:
    """MediaMTX can pull RTSP(S)/RTMP(S) cameras and HLS playlists. Files and webcams stay analytics-only."""
    s = str(src).strip()
    if not s:
        return False, "no source"
    if "://" not in s:
        return False, "file or webcam source: only network cameras (RTSP/RTMP) can be restreamed"
    scheme = s.split("://", 1)[0].lower()
    if scheme in RESTREAM_SCHEMES:
        return True, ""
    if scheme in ("http", "https") and urlsplit(s).path.lower().endswith(".m3u8"):
        return True, ""
    return False, f"'{scheme}' sources cannot be restreamed (use rtsp/rtmp, or an HLS .m3u8 URL)"


def _dur(seconds: float) -> str:
    return f"{float(seconds):g}s"


def path_config(cam: dict, rc: dict) -> Optional[dict]:
    """MediaMTX path configuration for one camera, or None if it cannot / should not be restreamed."""
    if not cam.get("enabled", True):
        return None
    src = main_source(cam)
    ok, _ = restreamable(src)
    if not ok:
        return None
    rec = rc["record"]["enabled"]
    conf: dict[str, Any] = {"source": src, "sourceOnDemand": not rec, "record": rec}
    if src.lower().startswith("rtsp"):
        conf["rtspTransport"] = "tcp"
    if rec:
        conf.update({"recordPath": f"{rc['record']['root'].as_posix()}/{RECORD_TEMPLATE}", "recordFormat": "fmp4",
                     "recordPartDuration": _dur(rc["record"]["part_s"]),
                     "recordSegmentDuration": _dur(rc["record"]["segment_s"]),
                     "recordDeleteAfter": "0s"})  # Argus deletes (keep_days), not MediaMTX
    return conf


def desired_paths(site_cfg: dict, rc: dict) -> dict[str, dict]:
    out = {}
    for c in site_cfg.get("cameras", []):
        conf = path_config(c, rc)
        if conf is not None:
            out[path_name(c["id"])] = conf
    return out


_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DUR_UNIT = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def go_duration(v) -> Optional[float]:
    """Parse a Go duration string ("1m0s", "60s", "1.5h") to seconds; None if it is not one."""
    if not isinstance(v, str) or not v:
        return None
    parts = _DUR_RE.findall(v)
    if not parts or "".join(a + b for a, b in parts) != v:
        return None
    return sum(float(a) * _DUR_UNIT[b] for a, b in parts)


def conf_differs(current: dict, want: dict) -> bool:
    """True if any key we manage differs (durations compared by value: MediaMTX echoes "1m0s" for "60s")."""
    for k, v in want.items():
        cur = current.get(k)
        if cur == v:
            continue
        a, b = go_duration(cur), go_duration(v)
        if a is not None and b is not None and abs(a - b) < 1e-6:
            continue
        return True
    return False


def _hostport(url: str, default_port: int) -> str:
    u = urlsplit(url)
    return f"{u.hostname or '127.0.0.1'}:{u.port or default_port}"


def managed_yaml(rc: dict, log_file: Path) -> str:
    """Minimal MediaMTX (v1.21) configuration for the Argus-managed instance.

    Only the Control API and the WebRTC signalling listener are enabled, both bound to the address in the Argus
    config (localhost by default: Argus proxies WHEP, browsers never reach MediaMTX's HTTP side). The ICE/UDP media
    port, which browsers exchange media with directly, listens on `webrtc_udp_host`: 127.0.0.1 by default, offered as
    the only candidate, so nothing is bound to a network interface ("" restores every interface for LAN viewers).
    No paths are defined here: Argus adds them through the API, so camera credentials are never written here."""
    hosts = [str(h) for h in rc.get("webrtc_additional_hosts") or []]
    udp_host = str(rc.get("webrtc_udp_host", "127.0.0.1") or "")
    loopback = udp_host in ("127.0.0.1", "localhost", "::1")
    if loopback and "127.0.0.1" not in hosts:
        hosts.insert(0, "127.0.0.1")
    q = lambda s: '"' + str(s).replace("\\", "/").replace('"', "") + '"'   # noqa: E731
    lines = [
        "# Generated by Argus (argonix/restream.py) on every start. Do not edit: change the site config's",
        "# \"restream\" section instead. Paths (cameras) are added at runtime through the Control API.",
        "logLevel: info",
        "logDestinations: [file]",
        f"logFile: {q(log_file.as_posix())}",
        "readTimeout: 10s",
        "writeTimeout: 10s",
        "authMethod: internal",
        "authInternalUsers:",
        "  # Argus (API calls and the WHEP proxy) connects from localhost; nobody else can read or publish.",
        "  - user: any",
        "    pass:",
        "    ips: [\"127.0.0.1\", \"::1\"]",
        "    permissions:",
        "      - action: read",
        "      - action: api",
        "api: true",
        f"apiAddress: {_hostport(rc['api'], 9997)}",
        "metrics: false",
        "pprof: false",
        "playback: false",
        "rtsp: false",
        "rtmp: false",
        "hls: false",
        "srt: false",
        "moq: false",
        "webrtc: true",
        f"webrtcAddress: {_hostport(rc['webrtc'], 8889)}",
        "webrtcEncryption: false",
        f"webrtcLocalUDPAddress: \"{'127.0.0.1' if loopback else udp_host}:{int(rc.get('webrtc_udp_port') or 8189)}\"",
        "webrtcLocalTCPAddress: \"\"",
        f"webrtcIPsFromInterfaces: {'false' if loopback else 'true'}",
        f"webrtcAdditionalHosts: [{', '.join(q(h) for h in hosts)}]",
        "webrtcICEServers2: []",
        "pathDefaults:",
        "  recordDeleteAfter: 0s",
        "paths: {}",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ Control API client
class MediaMTXError(RuntimeError):
    pass


class MediaMTXClient:
    """Thin client for the MediaMTX v3 Control API. Error texts are redacted before they are raised."""

    def __init__(self, api: str, user: str = "", password: str = "", transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 5.0):
        self.api = api.rstrip("/")
        self.auth = (user, password) if user else None
        self.transport, self.timeout = transport, timeout

    def _req(self, method: str, path: str, json_body: Optional[dict] = None) -> httpx.Response:
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout, auth=self.auth) as c:
                return c.request(method, self.api + path, json=json_body)
        except httpx.HTTPError as e:
            raise MediaMTXError(redact(f"MediaMTX API unreachable at {self.api}: {type(e).__name__}: {e}")) from None

    @staticmethod
    def _err(r: httpx.Response, what: str) -> MediaMTXError:
        try:
            detail = r.json().get("error") or r.text
        except ValueError:
            detail = r.text
        return MediaMTXError(redact(f"{what}: HTTP {r.status_code} {str(detail)[:300]}"))

    def ping(self) -> bool:
        try:
            return self._req("GET", "/v3/paths/list?itemsPerPage=1").status_code == 200
        except MediaMTXError:
            return False

    def config_paths(self) -> dict[str, dict]:
        out, page = {}, 0
        while True:
            r = self._req("GET", f"/v3/config/paths/list?itemsPerPage=500&page={page}")
            if r.status_code != 200:
                raise self._err(r, "list path configs")
            d = r.json()
            for it in d.get("items") or []:
                out[it.get("name", "")] = it
            page += 1
            if page >= int(d.get("pageCount") or 1):
                return out

    def paths(self) -> dict[str, dict]:
        """Runtime state of every path (ready/online, tracks, readers)."""
        r = self._req("GET", "/v3/paths/list?itemsPerPage=500")
        if r.status_code != 200:
            raise self._err(r, "list paths")
        return {it.get("name", ""): it for it in r.json().get("items") or []}

    def add(self, name: str, conf: dict) -> None:
        r = self._req("POST", f"/v3/config/paths/add/{name}", conf)
        if r.status_code != 200:
            raise self._err(r, f"add path {name}")

    def replace(self, name: str, conf: dict) -> None:
        r = self._req("POST", f"/v3/config/paths/replace/{name}", conf)
        if r.status_code != 200:
            raise self._err(r, f"replace path {name}")

    def delete(self, name: str) -> None:
        r = self._req("DELETE", f"/v3/config/paths/delete/{name}")
        if r.status_code not in (200, 404):
            raise self._err(r, f"delete path {name}")


def reconcile(client: MediaMTXClient, desired: dict[str, dict]) -> dict:
    """Make MediaMTX's ``cam_*`` paths match ``desired``. Returns {added, updated, removed, errors} (names only)."""
    res: dict[str, list] = {"added": [], "updated": [], "removed": [], "errors": []}
    current = client.config_paths()
    for name, conf in desired.items():
        try:
            if name not in current:
                client.add(name, conf); res["added"].append(name)
            elif conf_differs(current[name], conf):
                client.replace(name, conf); res["updated"].append(name)
        except MediaMTXError as e:
            res["errors"].append(str(e))
    for name in current:
        if name.startswith(PATH_PREFIX) and name not in desired:
            try:
                client.delete(name); res["removed"].append(name)
            except MediaMTXError as e:
                res["errors"].append(str(e))
    return res


# ------------------------------------------------------------------------------------------------ recordings
def parse_segment_start(name: str) -> Optional[float]:
    """Epoch start time from a MediaMTX segment file name (local time, or the %z offset when present)."""
    m = SEG_RE.match(name)
    if not m:
        return None
    y, mo, d, hh, mi, ss, frac, tz = m.groups()
    us = int(frac.ljust(6, "0"))
    try:
        if tz:
            off = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
            dt = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss), us, tzinfo=timezone(timedelta(seconds=off)))
            return dt.timestamp()
        return datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss), us).timestamp()  # naive = local time
    except ValueError:
        return None


def scan_segments(root: Path) -> dict[str, list[tuple[float, Path]]]:
    """{camera id: [(start, file), ...] sorted by start} for ``<root>/cam_<id>/<segment>.mp4``."""
    out: dict[str, list] = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(PATH_PREFIX):
            continue
        segs = []
        for f in d.iterdir():
            t = parse_segment_start(f.name) if f.is_file() else None
            if t is not None:
                segs.append((t, f))
        if segs:
            out[d.name[len(PATH_PREFIX):]] = sorted(segs)
    return out


def index_main_recordings(store, root: Path) -> dict:
    """Upsert every main-stream segment under ``root`` into the recordings table (stream='main').

    A segment ends at its last write (file mtime), capped by the next segment's start. Rows whose file vanished are
    removed. Returns {indexed, updated, removed}."""
    stats = {"indexed": 0, "updated": 0, "removed": 0}
    seen: set[str] = set()
    for cam, segs in scan_segments(root).items():
        for i, (start, f) in enumerate(segs):
            try:
                st = f.stat()
            except OSError:
                continue
            end = max(start, st.st_mtime)
            if i + 1 < len(segs):
                end = min(end, segs[i + 1][0])
            key = f.resolve().as_posix()
            seen.add(key)
            r = store.upsert_recording(cam, key, start, end, stream="main", size=st.st_size)
            if r in stats:
                stats[r] += 1
    for rid, path in store.recording_paths("main"):
        if path not in seen and not Path(path).exists():
            store.delete_recording(rid)
            stats["removed"] += 1
    return stats


def sweep_main_recordings(store, root: Path, keep_days: float, now: Optional[float] = None) -> dict:
    """Delete main-stream segments (and their rows) whose last write is older than ``keep_days``.

    Only files named like MediaMTX segments inside ``<root>/cam_*`` are ever deleted. keep_days <= 0 disables."""
    stats = {"deleted": 0, "bytes": 0, "errors": 0}
    if keep_days <= 0:
        return stats
    cutoff = (now if now is not None else time.time()) - keep_days * 86400.0
    for cam, segs in scan_segments(root).items():
        for start, f in segs:
            try:
                st = f.stat()
                if max(start, st.st_mtime) >= cutoff:
                    continue
                f.unlink()
            except OSError:   # e.g. still open by MediaMTX on Windows: next sweep
                stats["errors"] += 1
                continue
            stats["deleted"] += 1
            stats["bytes"] += st.st_size
            store.delete_recording_path(f.resolve().as_posix(), stream="main")
    return stats


# ------------------------------------------------------------------------------------------------ WHEP proxy
class WhepResult:
    def __init__(self, status: int, body: bytes = b"", content_type: str = "", headers: Optional[dict] = None,
                 error: str = ""):
        self.status, self.body, self.content_type = status, body, content_type
        self.headers, self.error = headers or {}, error


PASS_HEADERS = ("etag", "link", "accept-patch")


def session_from_location(loc: str) -> Optional[str]:
    """MediaMTX answers ``Location: /<path>/whep/<secret>`` (absolute path). Return the secret."""
    if not loc:
        return None
    sid = loc.rstrip("/").rsplit("/", 1)[-1]
    return sid if SESSION_RE.match(sid) else None


# ------------------------------------------------------------------------------------------------ the service
class Restreamer:
    """Keeps MediaMTX in step with the site's cameras; indexes and prunes main-stream recordings; proxies WHEP."""

    def __init__(self, site, transport: Optional[httpx.BaseTransport] = None,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.site = site
        self.cfg = load_config(site.cfg, site.data)
        self.transport = transport
        self.popen = popen
        self.client = MediaMTXClient(self.cfg["api"], self.cfg["user"], self.cfg["password"], transport)
        self.mode = "stopped"      # stopped | external | managed | down
        self.error = ""
        self.last_sync: Optional[dict] = None
        self.last_sync_t = 0.0
        self.last_index: Optional[dict] = None
        self.last_sweep: Optional[dict] = None
        self.proc = None
        self.restarts = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.runtime_dir = Path(site.data) / "mediamtx"

    # ---- lifecycle -------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="restream")
        self._thread.start()

    def close(self) -> None:
        self._stop.set(); self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._stop_managed()
        self.mode = "stopped"

    def sync_soon(self) -> None:
        """Cameras changed: reconcile paths on the next loop turn (called by Site add/update/remove camera)."""
        self._wake.set()

    def _loop(self) -> None:
        last_index = last_sweep = 0.0
        while not self._stop.is_set():
            try:
                self.ensure_running()
                if self.mode in ("external", "managed"):
                    self.sync()
            except Exception as e:  # noqa: BLE001 - the restreamer must never take the site down
                self.error = redact(f"{type(e).__name__}: {e}")
            rec = self.cfg["record"]
            now = time.time()
            if rec["enabled"] or rec["root"].is_dir():
                try:
                    if now - last_index >= float(self.cfg["index_interval_s"]):
                        last_index = now
                        self.last_index = {"t": now, **index_main_recordings(self.site.store, rec["root"])}
                    if now - last_sweep >= float(self.cfg["sweep_interval_s"]):
                        last_sweep = now
                        self.last_sweep = {"t": now, **sweep_main_recordings(self.site.store, rec["root"],
                                                                             rec["keep_days"], now)}
                except Exception as e:  # noqa: BLE001
                    self.error = redact(f"recordings: {type(e).__name__}: {e}")
            self._wake.wait(min(float(self.cfg["sync_interval_s"]), float(self.cfg["index_interval_s"])))
            self._wake.clear()

    def ensure_running(self) -> str:
        """Use a reachable MediaMTX API (external or ours); else start the managed binary if configured."""
        if self.proc is not None and self.proc.poll() is not None:
            self.error = f"managed MediaMTX exited with code {self.proc.returncode}; see {self.log_file}"
            self.proc = None
            self.restarts += 1
        if self.client.ping():
            self.mode = "managed" if self.proc is not None else "external"
            return self.mode
        binary = str(self.cfg.get("mediamtx_bin") or "")
        if not binary:
            self.mode = "down"
            self.error = f"MediaMTX API not reachable at {self.cfg['api']} and no mediamtx_bin configured"
            return self.mode
        if self.proc is None:
            if not Path(binary).is_file():
                self.mode = "down"
                self.error = f"mediamtx_bin not found: {binary}"
                return self.mode
            self._start_managed(binary)
        deadline = time.time() + 15
        while time.time() < deadline and not self._stop.is_set():
            if self.proc is not None and self.proc.poll() is not None:
                break
            if self.client.ping():
                self.mode, self.error = "managed", ""
                return self.mode
            time.sleep(0.3)
        self.mode = "down"
        self.error = self.error or f"managed MediaMTX did not answer on {self.cfg['api']}; see {self.log_file}"
        return self.mode

    @property
    def log_file(self) -> Path:
        return self.runtime_dir / "mediamtx.log"

    def _start_managed(self, binary: str) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        yml = self.runtime_dir / "mediamtx.yml"
        yml.write_text(managed_yaml(self.cfg, self.log_file), encoding="utf-8")
        kw: dict[str, Any] = {"cwd": str(self.runtime_dir), "stdin": subprocess.DEVNULL,
                              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = self.popen([binary, str(yml)], **kw)
        print(f"restream: started managed MediaMTX (pid {getattr(self.proc, 'pid', '?')}), config {yml}", flush=True)

    def _stop_managed(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass

    # ---- paths -------------------------------------------------------------------------------------------------
    def desired(self) -> dict[str, dict]:
        with self.site.cfg_lock:
            return desired_paths(self.site.cfg, self.cfg)

    def sync(self) -> dict:
        with self._lock:
            res = reconcile(self.client, self.desired())
            self.last_sync, self.last_sync_t = res, time.time()
            self.error = res["errors"][0] if res["errors"] else ""
            if res["added"] or res["updated"] or res["removed"]:
                print("restream: paths " + ", ".join(f"{k} {v}" for k, v in res.items() if v and k != "errors"),
                      flush=True)
            return res

    def camera_info(self, cam: dict, runtime: Optional[dict] = None) -> dict:
        src = main_source(cam)
        ok, why = restreamable(src)
        if not cam.get("enabled", True):
            ok, why = False, "camera disabled"
        info: dict[str, Any] = {"path": path_name(cam["id"]), "available": ok and self.mode in ("external", "managed"),
                                "restreamable": ok, "reason": why, "main_source": redact(src) if src else "",
                                "main_is_separate": bool(cam.get("main_source")),
                                "recording": ok and self.cfg["record"]["enabled"]}
        if ok and not info["available"]:
            info["reason"] = "restreamer not running" + (f": {self.error}" if self.error else "")
        if runtime is not None:
            info.update(online=bool(runtime.get("online", runtime.get("ready", False))),
                        readers=len(runtime.get("readers") or []), codecs=_codecs(runtime))
        return info

    def status(self, with_runtime: bool = True) -> dict:
        rt: dict[str, dict] = {}
        rt_error = ""
        if with_runtime and self.mode in ("external", "managed"):
            try:
                rt = self.client.paths()
            except MediaMTXError as e:
                rt_error = str(e)
        rec = self.cfg["record"]
        with self.site.cfg_lock:
            cams = list(self.site.cfg.get("cameras", []))
        return {"enabled": True, "mode": self.mode, "error": redact(self.error or rt_error),
                "api": redact(self.cfg["api"]), "webrtc": redact(self.cfg["webrtc"]),
                "managed": {"binary": bool(self.cfg.get("mediamtx_bin")), "running": self.proc is not None,
                            "restarts": self.restarts, "log": str(self.log_file) if self.proc is not None else None},
                "record": {"enabled": rec["enabled"], "path": str(rec["root"]), "segment_s": rec["segment_s"],
                           "keep_days": rec["keep_days"]},
                "last_sync": self.last_sync, "last_sync_t": self.last_sync_t or None,
                "last_index": self.last_index, "last_sweep": self.last_sweep,
                "cameras": {c["id"]: self.camera_info(c, rt.get(path_name(c["id"])) if with_runtime else None)
                            for c in cams}}

    # ---- WHEP ----------------------------------------------------------------------------------------------------
    def _whep_url(self, cam_id: str, session: str = "") -> str:
        return f"{self.cfg['webrtc']}/{path_name(cam_id)}/whep" + (f"/{session}" if session else "")

    async def _forward(self, method: str, url: str, body: bytes, content_type: str,
                       extra: Optional[dict] = None) -> httpx.Response:
        headers = {"Content-Type": content_type} if content_type else {}
        headers.update(extra or {})
        auth = (self.cfg["user"], self.cfg["password"]) if self.cfg["user"] else None
        # Only the SDP and its content type are forwarded: no browser Origin/Sec-Fetch-*/cookies/X-Forwarded-For,
        # so MediaMTX sees a same-host request from Argus (its own origin check and IP rules then apply to us).
        async with httpx.AsyncClient(transport=self.transport, timeout=20.0, auth=auth) as c:
            return await c.request(method, url, content=body, headers=headers)

    async def whep_offer(self, cam_id: str, sdp: bytes) -> WhepResult:
        try:
            r = await self._forward("POST", self._whep_url(cam_id), sdp, "application/sdp")
        except httpx.HTTPError as e:
            return WhepResult(502, error=redact(f"restreamer unreachable: {type(e).__name__}"))
        if r.status_code not in (200, 201):
            return WhepResult(_map_status(r.status_code), error=_whep_error(r))
        headers = {k: r.headers[k] for k in PASS_HEADERS if k in r.headers}
        sid = session_from_location(r.headers.get("location", ""))
        if sid:
            headers["Location"] = f"/api/cameras/{cam_id}/whep/{sid}"
        return WhepResult(201, r.content, r.headers.get("content-type", "application/sdp"), headers)

    async def whep_patch(self, cam_id: str, session: str, body: bytes, content_type: str,
                         if_match: str = "") -> WhepResult:
        try:
            r = await self._forward("PATCH", self._whep_url(cam_id, session), body,
                                    content_type or "application/trickle-ice-sdpfrag",
                                    {"If-Match": if_match} if if_match else None)
        except httpx.HTTPError as e:
            return WhepResult(502, error=f"restreamer unreachable: {type(e).__name__}")
        if r.status_code >= 400:
            return WhepResult(_map_status(r.status_code), error=_whep_error(r))
        return WhepResult(r.status_code, r.content, r.headers.get("content-type", ""),
                          {k: r.headers[k] for k in PASS_HEADERS if k in r.headers})

    async def whep_delete(self, cam_id: str, session: str) -> WhepResult:
        try:
            r = await self._forward("DELETE", self._whep_url(cam_id, session), b"", "")
        except httpx.HTTPError as e:
            return WhepResult(502, error=f"restreamer unreachable: {type(e).__name__}")
        return WhepResult(200 if r.status_code < 400 else _map_status(r.status_code),
                          error="" if r.status_code < 400 else _whep_error(r))


def _codecs(runtime: dict) -> list[str]:
    out = []
    for t in runtime.get("tracks2") or []:
        c = t.get("codec") if isinstance(t, dict) else None
        if c:
            out.append(str(c))
    if not out:
        out = [str(t) for t in runtime.get("tracks") or [] if isinstance(t, str)]
    return out


def _map_status(code: int) -> int:
    # 404: path has no stream (camera offline / not configured); 400: bad offer (e.g. codec); others: upstream failure
    return code if code in (400, 404, 406, 409, 415) else 502


def _whep_error(r: httpx.Response) -> str:
    try:
        detail = r.json().get("error") or ""
    except ValueError:
        detail = r.text[:200]
    return redact(f"MediaMTX answered {r.status_code}" + (f": {detail}" if detail else ""))


def install(site, **kw) -> Optional[Restreamer]:
    """Create ``site.restream`` when the site config enables it (started/stopped with the site)."""
    rc = load_config(site.cfg, site.data)
    if not rc["enabled"]:
        site.restream = None
        return None
    site.restream = Restreamer(site, **kw)
    return site.restream
