"""End-to-end check of Argus's CCTV path against the RTSP replay harness (manual section 18).

Run it on the host where `tools/rtsp_replay.ps1 start` is publishing. It drives the same HTTP API the dashboard uses:
  1. ONVIF discovery (expects a clean, empty device list: the harness is not an ONVIF device) and vendor templates;
  2. POST /api/cameras/test for every source in the site file, plus negative probes (wrong password, unknown path,
     closed port) whose errors must not leak the password;
  3. POST /api/cameras with rtsp://viewer:<password>@.../<name>_sub sources (zones and masks included);
  4. watches /api/stats, /api/alerts and the MediaMTX control API for `--observe-s` seconds: per-stream analytics
     fps, source fps, detect ms, frames, health, alerts per rule, and the fault stream's drop -> camera-down ->
     reconnect timeline (reconnect latency = publisher back -> first new frame in Argus);
  5. optionally switches one camera to its MAIN stream and back (PUT /api/cameras/{id}) to compare decode cost;
  6. checks credential redaction in /api/cameras, /api/stats, /api/audit, test errors and the app log, that a
     redacted source echoed back keeps the stored password, and that the Cameras-page endpoints return the shapes
     argonix/static/js/cameras.js expects;
  7. samples CPU (Argus process tree) and GPU (nvidia-smi) throughout.

    python tools/rtsp_e2e.py --launch --port 8010 --site configs/rtsp_site.json --observe-s 240 --out e2e.json

`--launch` starts a fresh Argus instance bound to 127.0.0.1 from a copy of the site file with NO cameras
(configs/rtsp_site.local.json, git-ignored), so every camera goes in through the API. The viewer password comes from
the harness secrets file (default %USERPROFILE%/argonix_rtsp/secrets.json) or $RTSP_PASSWORD; `${RTSP_PASSWORD}` and
`${RTSP_HOST}` placeholders in the site file are substituted. The password is never printed or written to the
result file.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
MTX_API = "http://127.0.0.1:9997"


# ------------------------------------------------------------------------------------------------ setup helpers
def load_password(secrets: Path) -> tuple[str, str]:
    if os.environ.get("RTSP_PASSWORD"):
        return os.environ.get("RTSP_USER", "viewer"), os.environ["RTSP_PASSWORD"]
    d = json.loads(secrets.read_text(encoding="utf-8-sig"))
    return d.get("user", "viewer"), d["password"]


def substitute(obj, mapping: dict):
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("${" + k + "}", v)
        return obj
    if isinstance(obj, list):
        return [substitute(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, mapping) for k, v in obj.items()}
    return obj


def path_of(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


def mtx_paths(c: httpx.Client) -> dict:
    try:
        r = c.get(f"{MTX_API}/v3/paths/list", params={"itemsPerPage": 500}, timeout=5)
        return {i["name"]: i for i in r.json().get("items", [])}
    except Exception:  # noqa: BLE001 - the harness API is optional for most checks
        return {}


def parse_ts(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        s = re.sub(r"(\.\d{6})\d+", r"\1", s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


class Sampler(threading.Thread):
    """CPU of the Argus process tree (psutil) and GPU util/VRAM (nvidia-smi) every `period` seconds."""

    def __init__(self, pid: Optional[int], period: float = 2.0):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period
        self.samples: list[dict] = []
        self.stop_evt = threading.Event()
        self.label = "baseline"

    def _tree(self):
        import psutil
        try:
            p = psutil.Process(self.pid)
            return [p] + p.children(recursive=True)
        except Exception:  # noqa: BLE001
            return []

    def run(self) -> None:
        import psutil
        procs = {}
        psutil.cpu_percent(None)
        while not self.stop_evt.wait(self.period):
            s = {"t": time.time(), "label": self.label, "cpu_total_pct": psutil.cpu_percent(None)}
            if self.pid:
                tot, rss, pids = 0.0, 0, []
                for p in self._tree():
                    try:
                        q = procs.setdefault(p.pid, p)
                        tot += q.cpu_percent(None)
                        rss += q.memory_info().rss
                        pids.append(p.pid)
                    except Exception:  # noqa: BLE001
                        pass
                s.update(app_cpu_pct_of_one_core=round(tot, 1), app_rss_mb=round(rss / 2**20), pids=pids)
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                u, used, total = [float(x) for x in out.stdout.strip().splitlines()[0].split(",")]
                s.update(gpu_util_pct=u, gpu_mem_used_mb=used, gpu_mem_total_mb=total)
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                mine = 0.0
                for line in apps.stdout.strip().splitlines():
                    try:
                        pid, mem = [x.strip() for x in line.split(",")]
                        if int(pid) in s.get("pids", []):
                            mine += float(mem)
                    except ValueError:
                        pass
                s["app_vram_mb"] = mine or None
            except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi
                pass
            self.samples.append(s)

    def summary(self, label: str) -> dict:
        xs = [s for s in self.samples if s["label"] == label]
        out = {"samples": len(xs)}
        for k in ("cpu_total_pct", "app_cpu_pct_of_one_core", "app_rss_mb", "gpu_util_pct", "gpu_mem_used_mb", "app_vram_mb"):
            v = [s[k] for s in xs if s.get(k) is not None]
            if v:
                out[k] = {"median": round(statistics.median(v), 1), "max": round(max(v), 1)}
        return out


def launch(args, site: dict) -> tuple[subprocess.Popen, Path, Path]:
    derived = copy.deepcopy(site)
    derived["cameras"] = []
    derived["site"] = derived.get("site", "RTSP e2e") + " (rtsp_e2e: cameras added through the API)"
    cfg = ROOT / "configs" / "rtsp_site.local.json"
    for p in (cfg, cfg.with_name(cfg.name + ".bak")):
        p.unlink(missing_ok=True)
    cfg.write_text(json.dumps(derived, indent=2))
    data = ROOT / "data" / f"run_rtsp_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    logf = data.with_suffix(".log")
    data.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run.py", "--config", str(cfg), "--data", str(data), "--port", str(args.port),
           "--host", "127.0.0.1", "--open-demo"]  # fresh data dir: no accounts, so open demo (local only)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(logf, "w"), stderr=subprocess.STDOUT, creationflags=flags)
    return proc, cfg, logf


def stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_ready(c: httpx.Client, timeout_s: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if c.get("/api/me", timeout=3).status_code == 200:
                return time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Argus did not answer within {timeout_s}s")


# ------------------------------------------------------------------------------------------------ checks
def check(results: list, name: str, ok: bool, detail="") -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def frontend_contract(c: httpx.Client, cam_ids: list[str], checks: list) -> dict:
    """The shapes argonix/static/js/{state,cameras,system,live,zones}.js read (see those files)."""
    out = {}
    r = c.get("/api/cameras")
    cams = r.json()
    need = {"id", "name", "source", "enabled", "status", "health", "fps", "width", "height", "calibrated", "zones",
            "masks_px"}
    check(checks, "GET /api/cameras is a list with the fields cameras.js renders",
          r.status_code == 200 and isinstance(cams, list) and all(need <= set(x) for x in cams),
          f"{len(cams)} cameras")
    t = c.get("/api/cameras/templates").json()
    lst = t if isinstance(t, list) else t.get("templates")
    check(checks, "GET /api/cameras/templates carries a template list (cameras.js reads .templates)",
          isinstance(lst, list) and all({"vendor", "main", "sub"} <= set(x) for x in lst), f"{len(lst or [])} templates")
    unknown = {m for x in lst or [] for m in re.findall(r"\{(\w+)\}", x["main"] + x["sub"])} - \
        {"user", "password", "host", "port", "channel", "channel2"}
    check(checks, "template placeholders are all filled by cameras.js fillTemplate()", not unknown, str(sorted(unknown)))
    cfg = c.get("/api/config").json()
    check(checks, "GET /api/config lists the API-added cameras (state.js merges it)",
          set(cam_ids) <= {x["id"] for x in cfg.get("cameras", [])})
    st = c.get("/api/stats").json()
    keys = {"status", "fps", "detect_ms", "frames", "dropped", "events", "tracks"}
    check(checks, "GET /api/stats has per-camera status/fps/detect_ms/frames (system.js)",
          all(keys <= set(st["cameras"].get(i, {})) for i in cam_ids))
    for i in cam_ids[:2]:
        for ep, hdr in (("snapshot.jpg", "x-coordinate-space"), ("latest.jpg", None)):
            rr = c.get(f"/api/cameras/{i}/{ep}")
            check(checks, f"GET /api/cameras/{i}/{ep} is a JPEG" + (" in source px" if hdr else ""),
                  rr.status_code == 200 and rr.content[:2] == b"\xff\xd8" and (not hdr or rr.headers.get(hdr) == "source-px"))
    with c.stream("GET", f"/api/cameras/{cam_ids[0]}/mjpeg") as rr:
        first = b""
        for b in rr.iter_bytes():
            first += b
            if len(first) > 4096:
                break
    check(checks, "GET /api/cameras/{id}/mjpeg streams multipart JPEG (live.js)",
          b"--frame" in first and b"image/jpeg" in first)
    out["ws"] = ws_tick(c, cam_ids, checks)
    return out


def ws_tick(c: httpx.Client, cam_ids: list[str], checks: list) -> dict:
    try:
        import asyncio
        import websockets

        async def one():
            url = str(c.base_url).replace("http", "ws", 1).rstrip("/") + "/ws"
            async with websockets.connect(url, open_timeout=10) as ws:
                for _ in range(6):
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    if msg.get("type") == "tick":
                        return msg
        msg = asyncio.run(one())
        ok = msg and set(cam_ids) <= set(msg.get("stats", {}))
        check(checks, "WebSocket /ws tick carries stats for every camera (state/cameras.js status cells)", ok)
        return {"tracks": len(msg.get("tracks", [])) if msg else None}
    except Exception as e:  # noqa: BLE001
        check(checks, "WebSocket /ws tick", False, repr(e)[:200])
        return {}


def secret_leaks(text: str, password: str) -> list[str]:
    hits = []
    for m in re.finditer(re.escape(password), text):
        hits.append(text[max(0, m.start() - 60):m.end() + 20].replace(password, "<PASSWORD>"))
    return hits


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default=str(ROOT / "configs" / "rtsp_site.json"))
    ap.add_argument("--base-url", default=None, help="an already running Argus (default http://127.0.0.1:<port>)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--launch", action="store_true", help="start a fresh Argus instance (127.0.0.1) and stop it after")
    ap.add_argument("--secrets", default=str(Path.home() / "argonix_rtsp" / "secrets.json"))
    ap.add_argument("--rtsp-host", default="127.0.0.1:8554")
    ap.add_argument("--observe-s", type=float, default=240)
    ap.add_argument("--main-vs-sub-s", type=float, default=45, help="seconds per phase of the main/sub comparison (0 = skip)")
    ap.add_argument("--ready-timeout-s", type=float, default=300)
    ap.add_argument("--keep", action="store_true", help="leave the cameras (and a launched instance) running")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    user, password = load_password(Path(a.secrets))
    site_raw = json.loads(Path(a.site).read_text())
    mapping = {"RTSP_USER": user, "RTSP_PASSWORD": password, "RTSP_HOST": a.rtsp_host}
    cams = [substitute(cc, mapping) for cc in site_raw["cameras"]]
    fault_paths = set(site_raw.get("_e2e", {}).get("fault_paths", []))
    checks: list = []
    R: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(), "site": str(a.site),
               "observe_s": a.observe_s, "checks": checks}
    proc, logf = None, None
    sampler = Sampler(None)
    mtx = httpx.Client(timeout=5)
    if a.launch:
        sampler.start()  # baseline CPU/GPU before Argus starts
        time.sleep(10)
        proc, cfg_path, logf = launch(a, site_raw)
        sampler.pid = proc.pid
        R["launched"] = {"config": str(cfg_path.relative_to(ROOT)), "log": str(logf.relative_to(ROOT)), "pid": proc.pid}
    base = a.base_url or f"http://127.0.0.1:{a.port}"
    c = httpx.Client(base_url=base, timeout=httpx.Timeout(60.0, connect=10.0),
                     headers={"X-Argonix-CSRF": "1"})  # state-changing calls without Origin need the CSRF header
    try:
        R["startup_s"] = round(wait_ready(c, a.ready_timeout_s), 1)
        print(f"Argus ready at {base} after {R['startup_s']} s", flush=True)
        if not a.launch:
            sampler.start()
        sampler.label = "idle"

        # 1. discovery + templates -------------------------------------------------------------------------
        print("1. ONVIF discovery and templates", flush=True)
        t0 = time.time()
        r = c.get("/api/cameras/discover", params={"timeout_s": 3})
        d = r.json()
        R["discover"] = {"status": r.status_code, "devices": len(d.get("devices", [])), "elapsed_s": round(time.time() - t0, 2),
                         "note": d.get("note")}
        check(checks, "ONVIF discovery returns a clean empty list (no ONVIF devices on this host's LAN)",
              r.status_code == 200 and d.get("devices") == [] and bool(d.get("note")),
              f"{len(d.get('devices', []))} devices in {R['discover']['elapsed_s']} s")

        # 2. probes ------------------------------------------------------------------------------------------
        print("2. POST /api/cameras/test", flush=True)
        R["probes"] = {}
        for cc in cams:
            t0 = time.time()
            res = c.post("/api/cameras/test", json={"source": cc["source"], "timeout_s": 15}).json()
            # a fault stream is down for 20 s of every cycle by design: retry across one down window before failing
            while not res.get("ok") and path_of(cc["source"]) in fault_paths and time.time() - t0 < 45:
                time.sleep(3)
                res = c.post("/api/cameras/test", json={"source": cc["source"], "timeout_s": 15}).json()
            R["probes"][cc["id"]] = {k: res.get(k) for k in ("ok", "width", "height", "fps", "elapsed_s", "error")}
            R["probes"][cc["id"]]["snapshot_bytes"] = len(res.get("snapshot_jpeg_b64") or "") * 3 // 4
            check(checks, f"probe {cc['id']} ({path_of(cc['source'])})", res.get("ok") and res.get("snapshot_jpeg_b64"),
                  f"{res.get('width')}x{res.get('height')} @ {res.get('fps')} fps in {res.get('elapsed_s')} s")
            check(checks, f"probe {cc['id']} response hides the password",
                  not secret_leaks(json.dumps(res), password) and ":***@" in res.get("source", ""))
        first = cams[0]["source"]
        negatives = {
            "wrong_password": first.replace(f":{password}@", ":WrongPass123@"),
            "unknown_path": first.rsplit("/", 1)[0] + "/no_such_stream",
            "closed_port": re.sub(r"@([^/:]+):\d+/", r"@\1:8599/", first),
        }
        R["negative_probes"] = {}
        for k, src in negatives.items():
            res = c.post("/api/cameras/test", json={"source": src, "timeout_s": 10}).json()
            R["negative_probes"][k] = {"ok": res.get("ok"), "error": res.get("error"), "elapsed_s": res.get("elapsed_s")}
            check(checks, f"negative probe '{k}' fails cleanly without leaking credentials",
                  res.get("ok") is False and "WrongPass123" not in json.dumps(res) and not secret_leaks(json.dumps(res), password),
                  f"{res.get('error')} ({res.get('elapsed_s')} s)")

        # 3. add cameras through the API ----------------------------------------------------------------------
        print("3. POST /api/cameras", flush=True)
        ids = []
        for cc in cams:
            body = {k: cc[k] for k in ("id", "name", "source", "zones", "masks_px", "reconnect_min_s", "reconnect_max_s",
                                       "down_grace_s") if k in cc}
            r = c.post("/api/cameras", json=body, timeout=180)  # may wait for the detector prewarm to finish
            ok = r.status_code == 200
            check(checks, f"add camera {cc['id']}", ok, "" if ok else r.text[:200])
            if ok:
                ids.append(cc["id"])
                check(checks, f"add camera {cc['id']} response is redacted", not secret_leaks(r.text, password))
        t_added = time.time()

        # 4. wait for frames, then observe -------------------------------------------------------------------
        print(f"4. observing for {a.observe_s:.0f} s", flush=True)
        first_frame: dict = {}
        while time.time() - t_added < 90 and len(first_frame) < len(ids):
            st = c.get("/api/stats").json()["cameras"]
            for i in ids:
                if i not in first_frame and st.get(i, {}).get("frames", 0) > 0:
                    first_frame[i] = round(time.time() - t_added, 2)
            time.sleep(0.5)
        R["time_to_first_frame_s"] = first_frame
        check(checks, "every API-added camera produced analytics frames", len(first_frame) == len(ids),
              json.dumps(first_frame))
        sampler.label = "observe"
        fault_cams = [cc["id"] for cc in cams if path_of(cc["source"]) in fault_paths and cc["id"] in ids]
        timeline = defaultdict(list)
        prev_mtx = {p: None for p in fault_paths}
        prev_cam = {i: None for i in fault_cams}
        pub_up_at, pub_down_at = {}, {}
        per_cam = {i: [] for i in ids}
        stats0 = c.get("/api/stats").json()["cameras"]
        mtx0, t_obs0 = mtx_paths(mtx), time.time()
        while time.time() - t_obs0 < a.observe_s:
            now = time.time()
            st = c.get("/api/stats").json()["cameras"]
            for i in ids:
                s = st.get(i, {})
                per_cam[i].append({"t": now, "fps": s.get("fps"), "detect_ms": s.get("detect_ms"),
                                   "frames": s.get("frames"), "status": s.get("status"), "health": s.get("health"),
                                   "reconnects": s.get("reconnects"), "source_fps": s.get("source_fps")})
            paths = mtx_paths(mtx)
            for p in fault_paths:
                ready = bool(paths.get(p, {}).get("ready"))
                if prev_mtx[p] is not None and ready != prev_mtx[p]:
                    ev = "publisher_up" if ready else "publisher_down"
                    rt = parse_ts(paths.get(p, {}).get("readyTime")) if ready else None
                    tt = rt if (rt and abs(rt - now) < 5) else now
                    timeline[p].append({"event": ev, "t": round(tt, 2)})
                    (pub_up_at if ready else pub_down_at)[p] = tt
                prev_mtx[p] = ready
            for i in fault_cams:
                s = st.get(i, {})
                key = (s.get("status"), s.get("reconnects"), s.get("health"))
                if prev_cam[i] is not None and key != prev_cam[i]:
                    timeline[i].append({"event": "camera", "t": round(now, 2), "status": key[0], "reconnects": key[1],
                                        "health": key[2]})
                prev_cam[i] = key
            time.sleep(0.5)
        t_obs1 = time.time()
        stats1 = c.get("/api/stats").json()["cameras"]
        mtx1 = mtx_paths(mtx)
        sampler.label = "after"

        # per-stream numbers
        R["streams"] = {}
        for i in ids:
            xs = [x for x in per_cam[i] if x["status"] == "running"]
            src = next(cc["source"] for cc in cams if cc["id"] == i)
            p = path_of(src)
            df = (stats1.get(i, {}).get("frames", 0) or 0) - (stats0.get(i, {}).get("frames", 0) or 0)
            rx = None
            if p in mtx0 and p in mtx1:
                rx = round((mtx1[p]["bytesReceived"] - mtx0[p]["bytesReceived"]) * 8 / (t_obs1 - t_obs0) / 1000, 1)
            dms = [x["detect_ms"] for x in xs if x["detect_ms"]]
            R["streams"][i] = {
                "path": p, "probe": R["probes"].get(i, {}).get("width") and
                f"{R['probes'][i]['width']}x{R['probes'][i]['height']}",
                "analytics_fps_measured": round(df / (t_obs1 - t_obs0), 2),
                "fps_ema_median": round(statistics.median([x["fps"] for x in xs if x["fps"]]), 2) if xs else None,
                "source_fps": stats1.get(i, {}).get("source_fps"),
                "detect_ms_median": round(statistics.median(dms), 1) if dms else None,
                "detect_ms_p90": round(sorted(dms)[int(0.9 * (len(dms) - 1))], 1) if dms else None,
                "frames": stats1.get(i, {}).get("frames"), "dropped": stats1.get(i, {}).get("dropped"),
                "reconnects": stats1.get(i, {}).get("reconnects"), "health": stats1.get(i, {}).get("health"),
                "health_reasons": stats1.get(i, {}).get("health_reasons"),
                "monochrome": stats1.get(i, {}).get("monochrome"),
                "stream_kbps_published": rx,
                "running_fraction": round(len(xs) / max(1, len(per_cam[i])), 3)}

        # reconnect latency for the fault cameras
        R["fault"] = {"timeline": timeline, "cameras": {}}
        alerts = c.get("/api/alerts", params={"limit": 2000}).json()
        for i in fault_cams:
            p = path_of(next(cc["source"] for cc in cams if cc["id"] == i))
            ups = [e["t"] for e in timeline[p] if e["event"] == "publisher_up"]
            downs = [e["t"] for e in timeline[p] if e["event"] == "publisher_down"]
            cam_ev = timeline[i]
            lat_up, lat_down = [], []
            for tu in ups:  # first increase of `reconnects` after the publisher came back
                base_rc = next((e["reconnects"] for e in reversed(cam_ev) if e["t"] <= tu), None)
                nxt = next((e for e in cam_ev if e["t"] >= tu - 0.5 and e["reconnects"] is not None
                            and base_rc is not None and e["reconnects"] > base_rc), None)
                if nxt:
                    lat_up.append(round(nxt["t"] - tu, 2))
            for td in downs:
                nxt = next((e for e in cam_ev if e["t"] >= td - 0.5 and e["status"] == "reconnecting"), None)
                if nxt:
                    lat_down.append(round(nxt["t"] - td, 2))
            health_ev = [ev for al in alerts for ev in al["events"] if ev["camera_id"] == i and ev["rule"] == "camera_health"]
            conds = Counter((ev.get("derivation") or {}).get("condition") for ev in health_ev)
            R["fault"]["cameras"][i] = {"path": p, "publisher_up_events": len(ups), "publisher_down_events": len(downs),
                                        "reconnect_latency_s": lat_up, "down_detect_latency_s": lat_down,
                                        "health_alert_conditions": dict(conds),
                                        "cfg": {k: next(cc for cc in cams if cc["id"] == i).get(k)
                                                for k in ("reconnect_min_s", "reconnect_max_s", "down_grace_s")}}
            check(checks, f"{i}: auto-reconnected after the fault stream came back", bool(lat_up),
                  f"latencies {lat_up} s")
            check(checks, f"{i}: camera-down health alert raised during the outage", conds.get("down", 0) > 0,
                  json.dumps(dict(conds)))

        # 5. main vs sub ------------------------------------------------------------------------------------------
        mvs = [cc for cc in cams if cc.get("_e2e_main_vs_sub") and cc["id"] in ids]
        if mvs and a.main_vs_sub_s > 0:
            cc = mvs[0]
            i, sub = cc["id"], cc["source"]
            main_src = sub[:-4] if sub.endswith("_sub") else sub
            print(f"5. main vs sub decode cost on {i}", flush=True)
            phases = {}
            for label, src in (("main", main_src), ("sub", sub)):
                r = c.put(f"/api/cameras/{i}", json={"source": src})
                check(checks, f"PUT /api/cameras/{i} source -> {label} stream", r.status_code == 200 and
                      not secret_leaks(r.text, password))
                time.sleep(8)  # reconnect + warm-up
                sampler.label = f"mvs_{label}"
                f0, t0 = c.get("/api/stats").json()["cameras"][i]["frames"], time.time()
                time.sleep(a.main_vs_sub_s)
                s = c.get("/api/stats").json()["cameras"][i]
                view = next((x for x in c.get("/api/cameras").json() if x["id"] == i), {})
                phases[label] = {"analytics_fps": round((s["frames"] - f0) / (time.time() - t0), 2),
                                 "detect_ms": s.get("detect_ms"), "source_fps": s.get("source_fps"),
                                 "resolution": f"{view.get('width')}x{view.get('height')}",
                                 **sampler.summary(f"mvs_{label}")}
                sampler.label = "after"
            R["main_vs_sub"] = {"camera": i, **phases}

        # 6. redaction + stored-secret checks ---------------------------------------------------------------------
        print("6. credential redaction", flush=True)
        red = {}
        for ep in ("/api/cameras", "/api/stats", "/api/config", "/api/audit?limit=1000", "/api/alerts?limit=2000"):
            txt = c.get(ep).text
            red[ep] = len(secret_leaks(txt, password))
            check(checks, f"{ep} contains no plaintext password", red[ep] == 0,
                  f"{txt.count(':***@')} redacted URLs")
        audit = c.get("/api/audit", params={"limit": 1000}).json()
        acts = Counter(x.get("action") for x in audit)
        R["audit_actions"] = dict(acts)
        check(checks, "camera_test / camera_add / camera_discover were audited",
              acts.get("camera_test", 0) >= len(cams) and acts.get("camera_add", 0) >= len(ids) and acts.get("camera_discover", 0) >= 1,
              json.dumps(dict(acts)))
        listed = {x["id"]: x["source"] for x in c.get("/api/cameras").json()}
        i0 = ids[0]
        r = c.post("/api/cameras/test", json={"id": i0, "timeout_s": 15}).json()
        check(checks, "probe by id uses the stored (unredacted) source", r.get("ok") is True)
        r = c.put(f"/api/cameras/{i0}", json={"source": listed[i0]})
        r2 = c.post("/api/cameras/test", json={"id": i0, "timeout_s": 15}).json()
        check(checks, "PUT with the redacted source echoed back keeps the stored password",
              r.status_code == 200 and r2.get("ok") is True)
        if logf and logf.exists():
            leaks = secret_leaks(logf.read_text(errors="replace"), password)
            red["app_log"] = len(leaks)
            check(checks, "app log contains no plaintext password", not leaks, "; ".join(leaks[:2]))
        if a.launch:
            stored = (ROOT / "configs" / "rtsp_site.local.json").read_text()
            R["stored_config_holds_password"] = bool(secret_leaks(stored, password))
        R["redaction_leaks"] = red

        # 7. frontend contract --------------------------------------------------------------------------------------
        print("7. Cameras-page endpoint shapes", flush=True)
        R["frontend"] = frontend_contract(c, ids, checks)

        # alerts summary + evidence
        alerts = c.get("/api/alerts", params={"limit": 2000}).json()
        by = defaultdict(Counter)
        for al in alerts:
            for ev in al["events"]:
                by[ev["camera_id"]][ev["rule"] + (f":{(ev.get('derivation') or {}).get('condition')}"
                                                  if ev["rule"] == "camera_health" else "")] += 1
        R["events_by_camera"] = {k: dict(v) for k, v in by.items()}
        snap = next((ev["evidence"]["snapshot"] for al in alerts for ev in al["events"]
                     if (ev.get("evidence") or {}).get("snapshot")), None)
        if snap:
            rr = c.get(f"/media/{snap}")
            check(checks, "alert evidence snapshot is served", rr.status_code == 200 and rr.content[:2] == b"\xff\xd8", snap)
        R["cpu_gpu"] = {k: sampler.summary(k) for k in ("baseline", "idle", "observe")}
    except Exception:  # noqa: BLE001 - report what we have
        import traceback
        R["error"] = traceback.format_exc()[-2000:]
        print(R["error"], flush=True)
    finally:
        sampler.stop_evt.set()
        if not a.keep:
            for cc in cams:
                try:
                    c.delete(f"/api/cameras/{cc['id']}")
                except httpx.HTTPError:
                    pass
            if proc:
                stop_proc(proc)
    R["passed"] = sum(1 for x in checks if x["ok"])
    R["failed"] = [x["check"] for x in checks if not x["ok"]]
    txt = json.dumps(R, indent=2, default=str)
    assert password not in txt, "refusing to write a result that contains the password"
    if a.out:
        Path(a.out).write_text(txt)
    print(f"\n{R['passed']} checks passed, {len(R['failed'])} failed", flush=True)
    for f in R["failed"]:
        print(f"  FAILED: {f}")
    return 0 if not R["failed"] and "error" not in R else 1


if __name__ == "__main__":
    sys.exit(main())
