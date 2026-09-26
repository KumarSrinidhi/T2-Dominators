"""Record a captioned walkthrough video of a running Argus dashboard.

Drives the dashboard with Playwright (headless Chrome), captures every painted frame through the Chrome DevTools
screencast, draws captions and a visible cursor into the page, and encodes an MP4 with ffmpeg.

    uv run --no-project --with playwright python tools/record_demo.py --url http://127.0.0.1:8003

The instance must run with --open-demo (no sign-in, demo scenarios allowed). The scenario beat creates alerts that
are labelled DEMO in that instance's data dir. Needs Google Chrome installed and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

OVERLAY_JS = r"""
(() => {
  const boot = () => {
    if (window.__demo || !document.body) return;
    const st = document.createElement("style");
    st.textContent = `
      #__cap { position: fixed; left: 50%; bottom: 34px; transform: translateX(-50%) translateY(12px); z-index: 2147483000;
        max-width: 1280px; padding: 16px 28px 17px; border-radius: 14px; background: rgba(12,17,29,.90); color: #fff;
        font: 500 25px/1.3 "Segoe UI", Inter, system-ui, sans-serif; box-shadow: 0 12px 40px rgba(0,0,0,.35);
        opacity: 0; transition: opacity .35s, transform .35s; pointer-events: none; text-align: center; }
      #__cap.on { opacity: 1; transform: translateX(-50%) translateY(0); }
      #__cap b { display: block; font-weight: 700; font-size: 17px; letter-spacing: .12em; text-transform: uppercase;
        color: #7fb2ff; margin-bottom: 4px; }
      #__card { position: fixed; inset: 0; z-index: 2147483300; display: flex; flex-direction: column; align-items: center;
        justify-content: center; gap: 22px; background: radial-gradient(1200px 700px at 50% 40%, #16264a, #070b14);
        color: #fff; font-family: "Segoe UI", Inter, system-ui, sans-serif; opacity: 0; transition: opacity .6s;
        pointer-events: none; text-align: center; }
      #__card.on { opacity: 1; }
      #__card .k { font-size: 20px; letter-spacing: .3em; text-transform: uppercase; color: #7fb2ff; font-weight: 600; }
      #__card h1 { margin: 0; font-size: 84px; font-weight: 750; letter-spacing: -.02em; }
      #__card .s { font-size: 32px; color: #c9d6ee; max-width: 1300px; line-height: 1.35; }
      #__card ul { margin: 18px 0 0; padding: 0; list-style: none; font-size: 26px; color: #dfe7f7; line-height: 1.75; text-align: left; }
      #__card li::before { content: "▸ "; color: #7fb2ff; }
      #__card .f { position: absolute; bottom: 44px; font-size: 18px; color: #8594b3; }
      #__cur { position: fixed; left: 0; top: 0; width: 26px; height: 26px; z-index: 2147483200; pointer-events: none;
        transform: translate(-100px,-100px); transition: none; }
      #__cur svg { filter: drop-shadow(0 2px 3px rgba(0,0,0,.45)); }
      .__rip { position: fixed; z-index: 2147483150; width: 44px; height: 44px; margin: -22px 0 0 -22px; border-radius: 50%;
        border: 3px solid #3b82f6; pointer-events: none; animation: __rip .55s ease-out forwards; }
      @keyframes __rip { from { transform: scale(.3); opacity: 1; } to { transform: scale(1.5); opacity: 0; } }
      #__hl { position: fixed; z-index: 2147482900; border: 3px solid #f59e0b; border-radius: 12px; pointer-events: none;
        box-shadow: 0 0 0 9999px rgba(7,11,20,.28); opacity: 0; transition: opacity .35s, all .35s; }
      #__hl.on { opacity: 1; }`;
    document.head.appendChild(st);
    const cap = Object.assign(document.createElement("div"), { id: "__cap" });
    const card = Object.assign(document.createElement("div"), { id: "__card" });
    const hl = Object.assign(document.createElement("div"), { id: "__hl" });
    const cur = Object.assign(document.createElement("div"), { id: "__cur" });
    cur.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2l15 11.5-6.6.9 3.9 7.2-2.8 1.5-3.9-7.3L4 20z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    document.body.append(hl, cap, card, cur);
    document.addEventListener("mousemove", e => { cur.style.transform = `translate(${e.clientX - 3}px,${e.clientY - 2}px)`; }, true);
    document.addEventListener("mousedown", e => {
      const r = Object.assign(document.createElement("div"), { className: "__rip" });
      r.style.left = e.clientX + "px"; r.style.top = e.clientY + "px";
      document.body.appendChild(r); setTimeout(() => r.remove(), 700);
    }, true);
    const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    window.__demo = {
      cap(kicker, text) {
        cap.classList.remove("on");
        setTimeout(() => { cap.innerHTML = (kicker ? `<b>${esc(kicker)}</b>` : "") + esc(text); cap.classList.add("on"); }, kicker === null ? 0 : 180);
      },
      clear() { cap.classList.remove("on"); },
      card(kicker, title, sub, items, foot) {
        card.innerHTML = `<div class="k">${esc(kicker || "")}</div>` + (title ? `<h1>${esc(title)}</h1>` : "") +
          (sub ? `<div class="s">${esc(sub)}</div>` : "") +
          (items && items.length ? `<ul>${items.map(i => `<li>${esc(i)}</li>`).join("")}</ul>` : "") +
          (foot ? `<div class="f">${esc(foot)}</div>` : "");
        card.classList.add("on");
      },
      hideCard() { card.classList.remove("on"); },
      highlight(sel) {
        const el = typeof sel === "string" ? document.querySelector(sel) : sel;
        if (!el) { hl.classList.remove("on"); return; }
        const r = el.getBoundingClientRect(), p = 8;
        Object.assign(hl.style, { left: r.left - p + "px", top: r.top - p + "px", width: r.width + 2 * p + "px", height: r.height + 2 * p + "px" });
        hl.classList.add("on");
      },
      unhighlight() { hl.classList.remove("on"); },
    };
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
"""


class Director:
    """Small helpers around a Playwright page: captions, smooth cursor moves, clicks, scrolls."""

    def __init__(self, page: Page, base: str, pace: float):
        self.page, self.base, self.pace = page, base.rstrip("/"), pace
        self.paused = False       # while True, screencast frames are dropped (a cut in the video)
        self.cuts: set[int] = set()  # indexes of frames that end right before a cut
        self.frame_count = 0

    def pause(self):
        self.paused, self.paused_at = True, time.monotonic()

    def resume(self) -> float:
        """End a cut; returns how many seconds were cut."""
        cut = time.monotonic() - self.paused_at if self.paused else 0.0
        if self.paused and self.frame_count:
            self.cuts.add(self.frame_count - 1)
        self.paused = False
        return cut

    async def stats(self) -> dict:
        return await self.page.evaluate("fetch('/api/stats').then(r => r.json())")

    async def wait(self, s: float):
        await self.page.wait_for_timeout(int(s * 1000 * self.pace))

    async def cap(self, kicker: str, text: str, hold: float = 0):
        await self.page.evaluate("([k, t]) => window.__demo.cap(k, t)", [kicker, text])
        if hold:
            await self.wait(hold)

    async def card(self, kicker, title, sub="", items=(), foot="", hold=4.0):
        await self.page.evaluate("a => window.__demo.card(...a)", [kicker, title, sub, list(items), foot])
        await self.wait(hold)
        await self.page.evaluate("window.__demo.hideCard()")
        await self.wait(0.7)

    async def go(self, route: str, settle: float = 2.0):
        await self.page.evaluate("r => { location.hash = r; }", "#/" + route)
        await self.wait(settle)

    async def point(self, loc, steps: int = 28):
        for attempt in range(4):  # live pages re-render (the alert drawer refreshes on updates): re-resolve and retry
            try:
                await loc.scroll_into_view_if_needed(timeout=5000)
                box = await loc.bounding_box()
                if box:
                    break
            except Exception:
                if attempt == 3:
                    raise
            await self.page.wait_for_timeout(300)
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        await self.page.mouse.move(x, y, steps=steps)
        return x, y

    async def click(self, loc, after: float = 1.2):
        x, y = await self.point(loc)
        await self.wait(0.25)
        try:
            await loc.click(timeout=3000)  # re-resolves, so a list that re-rendered meanwhile still gets the right element
        except Exception:
            await self.page.mouse.click(x, y)  # styled controls (switches over hidden checkboxes)
        await self.wait(after)

    async def highlight(self, loc, hold: float = 2.5):
        handle = await loc.element_handle(timeout=5000)
        await self.page.evaluate("el => window.__demo.highlight(el)", handle)
        await self.wait(hold)
        await self.page.evaluate("window.__demo.unhighlight()")

    async def scroll(self, selector: str | None, dy: int, after: float = 1.6):
        await self.page.evaluate(
            "([s, dy]) => { const el = s ? document.querySelector(s) : null;"
            " const t = el && el.scrollHeight > el.clientHeight ? el : (document.querySelector('main') && document.querySelector('main').scrollHeight > document.querySelector('main').clientHeight ? document.querySelector('main') : document.scrollingElement);"
            " t.scrollBy({ top: dy, behavior: 'smooth' }); }", [selector, dy])
        await self.wait(after)

    async def scroll_top(self, selector: str | None = None):
        await self.page.evaluate(
            "s => { for (const el of [s && document.querySelector(s), document.querySelector('main'), document.scrollingElement]) if (el) el.scrollTo({ top: 0 }); }", selector)


async def beat(name: str, coro, d: "Director | None" = None):
    try:
        await coro
    except Exception as e:  # keep recording; a missing element only costs one beat
        if d:
            d.resume()
        print(f"[beat {name}] skipped part: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


async def storyboard(d: Director, run_scenario: bool):
    p = d.page

    # 0. Title
    await d.go("overview", 0.2)
    await d.card("Hackathon demo · problem statement #5", "Argus",
                 "Explainable multi-camera situational awareness for the CCTV a site already has",
                 foot="Real public footage · detection on an RTX 4050 GPU host over the local network", hold=5.5)

    # 1. Overview
    async def overview():
        await d.cap("Overview", "Live KPIs, cameras with open alerts first, and a short triage queue", 3.5)
        await d.highlight(p.locator(".kpi, .kpis > *").first, 2.2)
        await d.cap("Overview", "Every alert says what was seen, where, and for how long, never who someone is", 1.0)
        await d.scroll(None, 520, 3.0)
        await d.scroll(None, -520, 1.2)
    await beat("overview", overview())

    # 2. Live
    async def live():
        await d.click(p.locator("nav a[data-page=live], a[href='#/live']").first, 2.5)
        await d.cap("Live", "Two real clips as RTSP-style streams: OpenCV campus path and a Kaggle street clip", 4.5)
        await d.cap("Live", "Person boxes, track IDs and zones; black-and-white / IR input gets automatic low-light enhancement", 4.5)
        await d.click(p.get_by_role("button", name="Open Campus path (OpenCV vtest.avi)").first, 3.0)
        await d.cap("Camera view", "Per-camera view: zones, masks, fps and detection time, plus a tamper test", 4.5)
    await beat("live", live())

    # 3. An alert, end to end
    async def alert():
        await d.click(p.locator("nav a[data-page=review], a[href='#/review']").first, 2.5)
        await d.cap("Review", "Alerts ranked P1–P4 with the evidence snapshot attached", 3.0)
        await d.click(p.locator("button.fchip", has_text="Restricted-zone entry").first, 2.0)
        await d.click(p.locator(".cards .card").first, 2.5)
        await d.cap("Alert drawer", "Observation first: exactly what was seen, e.g. a foot-point inside ‘Lawn (keep off)’ for 2 s", 5.0)
        await d.scroll("#drawer", 420, 1.0)
        await d.cap("Alert drawer", "Derivation: rule, threshold, duration and the calibration error used to keep the boundary conservative", 5.0)
        await d.scroll("#drawer", 480, 1.0)
        await d.cap("Why this priority", "A factor breakdown. Learned models only corroborate: learned-only alerts are capped at P3", 5.0)
        explain = p.locator("#drawer button", has_text="Explain").first
        if await explain.count():
            await d.scroll_top("#drawer")
            await d.cap("Explain (assistive)", "Ask a vision-language model, Qwen2.5-VL-3B on the RTX 4050, to describe frames from the evidence clip", 1.0)
            vlm_text = "() => (document.querySelector('#drawer .vlm') || {}).innerText || ''"
            before = await p.evaluate(vlm_text)
            await d.click(explain, 2.5)
            d.pause()  # the real call takes ~30 s on LEGION; cut the wait and say so
            await p.wait_for_function("b => { const t = (document.querySelector('#drawer .vlm') || {}).innerText || ''; return t && t !== b; }",
                                      arg=before, timeout=150_000, polling=500)
            await d.page.wait_for_timeout(800)
            await d.page.evaluate("window.__demo.clear()")
            await d.page.evaluate("document.querySelector('#drawer .vlm').scrollIntoView({block: 'center'})")
            await d.page.wait_for_timeout(600)
            cut = d.resume()
            await d.cap("Explain (assistive)", f"{cut:.0f} s of waiting for the model cut. Labelled assistive: it never raises, scores or changes an alert", 1.0)
            await d.highlight(p.locator("#drawer .vlm").first, 6.5)
        seg = p.locator("#drawer button", has_text="Original").first
        if await seg.count():
            await d.click(seg, 1.5)
            await d.cap("Evidence", "‘Original’ is the frame exactly as analysed; the overlay view is stamped as software-drawn", 3.5)
            await d.click(p.locator("#drawer button", has_text="With overlay").first, 2.0)
        await d.scroll("#drawer", 600, 1.0)
        await d.cap("Evidence & workflow", "Hash-chained evidence export (optional anonymised copies), role-based workflow, SLA timer, legal hold", 5.0)
        await p.keyboard.press("Escape")
        await d.wait(1.0)
    await beat("alert", alert(), d)

    # 4. Scripted scenario → incident
    async def scenario():
        await d.click(p.locator("nav a[data-page=system], a[href='#/system']").first, 2.5)
        btn = p.get_by_role("button", name="Run demo scenario")
        await d.point(btn)
        await d.cap("Scripted scenario", "A guaranteed incident for the demo: restricted-zone entry, a forced door, then a covered camera", 3.5)
        await d.cap("Scripted scenario", "Everything it creates is badged DEMO and kept out of real alerts and precision stats", 1.0)
        await d.click(btn, 2.0)
        await d.go("live", 1.0)
        await d.cap("Scripted scenario", "t+0 s: someone enters the restricted zone…", 7.0)
        await d.cap("Scripted scenario", "t+8 s: the access-control system reports the door forced open. Sensor + video fuse into one P1 incident", 8.0)
        await d.cap("Scripted scenario", "t+15 s: the camera is covered, so it opens a camera-integrity incident and mutes that camera's alerts", 8.0)
        await d.click(p.locator("nav a[data-page=incidents], a[href='#/incidents']").first, 2.5)
        await d.cap("Incidents", "Related alerts grouped into incidents with a status lifecycle", 3.0)
        row = p.locator("tr.inc-row", has_text="forced").first  # the fused sensor + video P1 incident
        if not await row.count():
            row = p.locator("tr.inc-row", has_text="DEMO").first
        if not await row.count():
            row = p.locator("tr.inc-row").first
        await d.click(row, 3.0)
        await d.cap("Incident", "Incident graph: cameras, zones, tracks and the door sensor, linked in time", 5.0)
        replay = p.get_by_role("button", name="Load synchronised replay")
        if await replay.count():
            await d.click(replay, 1.0)
        else:
            await d.scroll(None, 600, 1.0)
        await d.cap("Incident", "Synchronised replay of every involved camera on one clock, with event markers", 5.5)
        await d.scroll(None, 700, 1.0)
        await d.cap("Incident", "Timeline split into observation and derivation; a self-contained HTML report with evidence hashes", 5.0)
        await d.scroll_top()
    if run_scenario:
        await beat("scenario", scenario())

    # 5. Map
    async def site_map():
        await d.click(p.locator("nav a[data-page=map], a[href='#/map']").first, 3.0)
        await d.cap("Site map", "Bird's-eye map from each camera's ground-plane homography: live tracks with position uncertainty", 4.5)
        hm = p.get_by_label("Occupancy heatmap")
        if await hm.count():
            await d.click(hm, 3.0)
        await d.cap("Site map", "Occupancy heatmap. The ground plane is 2D and every camera states its calibration error", 4.0)
    await beat("map", site_map())

    # 6. Analytics
    async def analytics():
        await d.click(p.locator("nav a[data-page=analytics], a[href='#/analytics']").first, 3.0)
        await d.cap("Analytics", "Counting lines, zone occupancy and dwell, density per m², queues and crowd flow", 4.0)
        await d.scroll(None, 650, 1.0)
        await d.cap("Analytics", "The congestion forecast is shown only when it beats ‘last value’ and ‘usual for this hour’ on held-out data", 5.0)
        await d.scroll(None, 700, 1.0)
        await d.cap("Analytics", "Anomaly indicators (IsolationForest, CUSUM) are labelled ‘corroborating only’", 3.5)
        await d.scroll_top()
    await beat("analytics", analytics())

    # 7. Journeys
    async def journeys():
        await d.click(p.locator("nav a[data-page=journeys], a[href='#/journeys']").first, 2.0)
        prev = p.get_by_role("button", name="Preview with sample data")
        if await prev.count():
            await d.click(prev, 2.5)
        await d.cap("Journeys (sample data)", "Cross-camera re-identification: one global ID, sightings per camera, a confidence per hop", 5.0)
        await d.cap("Journeys (sample data)", "Probabilistic and revisable: an operator can split a wrong link. Hints to verify, never an identity", 4.5)
    await beat("journeys", journeys())

    # 8. Investigate
    async def investigate():
        await d.click(p.locator("nav a[data-page=investigate], a[href='#/investigate']").first, 2.0)
        box = p.get_by_placeholder("Ask about", exact=False).first
        await d.click(box, 0.5)
        await p.keyboard.type("restricted-zone entries on the lawn in the last 24 hours", delay=45)
        await d.cap("Investigate", "Plain-language questions become an allow-listed structured query; the model never writes SQL", 1.5)
        await d.click(p.get_by_role("button", name="Ask").first, 3.5)
        await d.cap("Investigate", "The summary may only cite alert and incident IDs that really exist in the results", 4.0)
        await d.scroll(None, 500, 2.5)
        await d.scroll_top()
    await beat("investigate", investigate())

    # 9. System + handover
    async def system():
        await d.click(p.locator("nav a[data-page=system], a[href='#/system']").first, 2.5)
        await d.cap("System", "Camera health (tamper, blur, down), fps and detection time per camera, GPU inference node", 4.5)
        await d.scroll(None, 650, 1.0)
        await d.cap("System", "Per-rule precision from operator feedback (Wilson interval) and a hash-chained audit log check", 4.5)
        await d.scroll(None, 700, 2.0)
        await d.scroll_top()
        await d.click(p.locator("nav a[data-page=handover], a[href='#/handover']").first, 2.5)
        await d.cap("Shift handover", "What is still open at shift end: overdue alerts, camera problems, top zones, and a printable sheet", 4.5)
    await beat("system", system())

    # 10. Close
    await d.page.evaluate("window.__demo.clear()")
    measured = "RTX 4050 host: detection for every camera on one GPU over the local network"
    try:
        cams = [c for c in (await d.stats()).get("cameras", {}).values() if c.get("status") == "running"]
        fps = statistics.mean(c["fps"] for c in cams)
        rtt = statistics.median(c["rtt_ms"] for c in cams if c.get("rtt_ms"))
        measured = (f"RTX 4050 host over the LAN: {len(cams)} cameras at {fps:.1f} fps, {rtt:.0f} ms round trip "
                    "per frame (measured at the end of this recording)")
    except Exception:  # no remote node or stats: keep the unquantified line
        pass
    await d.card("Measured, not estimated", "",
                 "",
                 items=[measured,
                        "RTX 3050: batched TensorRT ~20 cameras at 10 fps",
                        "Re-ID on a rendered 3-camera scene: precision 0.61, recall 0.36",
                        "Deterministic rules alert; learned models corroborate; VLMs only explain"],
                 foot="Argus · explainable situational awareness", hold=7.0)
    await d.card("", "Argus", "Reports what was seen, with the evidence attached", hold=3.5)


async def record(args) -> None:
    out = Path(args.out)
    frames_dir = out.parent / (out.stem + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    frames: list[tuple[float, Path]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=True)
        ctx = await browser.new_context(viewport={"width": args.width, "height": args.height}, device_scale_factor=1,
                                         color_scheme=args.theme)
        await ctx.add_init_script(OVERLAY_JS)
        page = await ctx.new_page()
        await page.goto(args.url.rstrip("/") + "/#/overview", wait_until="domcontentloaded")
        await page.wait_for_function("!!window.__demo", timeout=15000)
        await page.wait_for_timeout(2500)
        await page.mouse.move(args.width / 2, args.height / 2)

        cdp = await ctx.new_cdp_session(page)
        d = Director(page, args.url, args.pace)

        async def on_frame(params):
            ts = params["metadata"].get("timestamp")
            if d.paused:
                try:
                    await cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
                except Exception:
                    pass
                return
            path = frames_dir / f"{len(frames):06d}.jpg"
            path.write_bytes(base64.b64decode(params["data"]))
            frames.append((ts, path))
            d.frame_count = len(frames)
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
            except Exception:
                pass

        cdp.on("Page.screencastFrame", lambda prm: asyncio.ensure_future(on_frame(prm)))
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": args.width,
                                                 "maxHeight": args.height, "everyNthFrame": 1})
        await storyboard(d, run_scenario=not args.no_scenario)
        await page.wait_for_timeout(500)
        await cdp.send("Page.stopScreencast")
        await browser.close()

    print(f"captured {len(frames)} frames")
    lines = []
    for i, (ts, path) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) and i not in d.cuts else ts + 1 / args.fps
        lines.append(f"file '{path.name}'\nduration {max(nxt - ts, 0.001):.4f}")
    lines.append(f"file '{frames[-1][1].name}'")
    concat = frames_dir / "frames.txt"
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", f"fps={args.fps},scale={args.width}:{args.height}:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-movflags", "+faststart", str(out)], check=True)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8003", help="running Argus instance started with --open-demo")
    ap.add_argument("--out", default="data/demo_video/argus_demo.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--pace", type=float, default=1.0, help="multiply every hold time (1.2 = slower)")
    ap.add_argument("--no-scenario", action="store_true", help="skip the scripted DEMO scenario beat")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
