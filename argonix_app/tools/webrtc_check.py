"""Does the Live page's smooth (WebRTC) view really play? A headless-browser check for a running Argus.

    uv run python tools/webrtc_check.py --url http://127.0.0.1:8010 --camera cam_campus [--seconds 20] [--out r.json]

Starts Chrome or Edge headless with the DevTools protocol (no extra packages: httpx + websockets from the app
venv), opens #/live/<camera> (the single-camera view defaults to Smooth when restream is on), and samples every
<video> element: resolution, decoded frame count and playback time. PASS = a video whose decoded frames and time
keep advancing. Also saves a screenshot next to --out. The server must be in --open-demo mode (no sign-in).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]

SAMPLE = """(() => [...document.querySelectorAll('video')].map(v => {
  const q = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality() : {};
  return {w: v.videoWidth, h: v.videoHeight, t: v.currentTime, ready: v.readyState, paused: v.paused,
          stream: !!v.srcObject, frames: q.totalVideoFrames || 0, dropped: q.droppedVideoFrames || 0};
}))()"""
MODE_TEXT = """(() => { const t = document.body.innerText || '';
  return (t.match(/(Smooth[^\\n]{0,60}|WebRTC[^\\n]{0,60}|Analytics[^\\n]{0,60}|fell back[^\\n]{0,80})/g) || []).slice(0, 8);
})()"""


def find_browser(explicit: str = "") -> str:
    for b in ([explicit] if explicit else []) + BROWSERS:
        if b and (Path(b).exists() or shutil.which(b)):
            return b if Path(b).exists() else shutil.which(b)
    raise SystemExit("no Chrome/Edge found; pass --browser <path>")


async def run(a) -> dict:
    port = a.devtools_port
    prof = tempfile.mkdtemp(prefix="argonix_webrtc_")
    proc = subprocess.Popen([find_browser(a.browser), "--headless=new", f"--remote-debugging-port={port}",
                             "--remote-debugging-address=127.0.0.1", f"--user-data-dir={prof}", "--no-first-run",
                             "--autoplay-policy=no-user-gesture-required", "--mute-audio", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = {"url": a.url, "camera": a.camera, "samples": [], "mode_text": [], "pass": False}
    try:
        tabs = None
        for _ in range(100):
            try:
                tabs = httpx.get(f"http://127.0.0.1:{port}/json/list", timeout=2).json()
                break
            except Exception:  # noqa: BLE001 - browser still starting
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=64 * 2**20) as ws:
            ids = itertools.count(1)

            async def cmd(method, **params):
                i = next(ids)
                await ws.send(json.dumps({"id": i, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == i:
                        if "error" in m:
                            raise RuntimeError(f"{method}: {m['error']}")
                        return m.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            await cmd("Page.enable")
            await cmd("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
            await cmd("Page.navigate", url=f"{a.url.rstrip('/')}/#/live/{a.camera}")
            await asyncio.sleep(a.settle_s)
            t0 = time.time()
            while time.time() - t0 < a.seconds:
                result["samples"].append({"t": round(time.time() - t0, 1), "videos": await js(SAMPLE)})
                await asyncio.sleep(2)
            result["mode_text"] = await js(MODE_TEXT)
            if a.out:
                shot = await cmd("Page.captureScreenshot", format="png")
                Path(a.out).with_suffix(".png").write_bytes(base64.b64decode(shot["data"]))
    finally:
        proc.terminate()
    # PASS: some <video> with a MediaStream whose decoded frames and playback time both advanced
    first, last = result["samples"][0]["videos"] or [], result["samples"][-1]["videos"] or []
    for i, v in enumerate(last):
        v0 = first[i] if i < len(first) else {"frames": 0, "t": 0}
        if v["stream"] and v["w"] > 0 and v["frames"] > v0["frames"] + 10 and v["t"] > v0["t"] + 2:
            elapsed = result["samples"][-1]["t"] - result["samples"][0]["t"]
            result.update({"pass": True, "resolution": f"{v['w']}x{v['h']}",
                           "decoded_fps": round((v["frames"] - v0["frames"]) / max(elapsed, 1e-9), 1),
                           "dropped_frames": v["dropped"]})
            break
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=8.0, help="wait after loading before sampling")
    ap.add_argument("--browser", default="")
    ap.add_argument("--devtools-port", type=int, default=9335)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    r = asyncio.run(run(a))
    print(json.dumps({k: v for k, v in r.items() if k != "samples"}, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
    print("PASS" if r["pass"] else "FAIL", flush=True)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
