// Synchronised multi-camera incident replay. One master clock (absolute epoch seconds, advanced by
// requestAnimationFrame at 0.5x/1x/2x) drives one <video> per camera: each player shows the segment that covers the
// master time, seeks to (master - segment start) scaled to the file's real duration, switches segments at their
// boundaries and is re-seeked when it drifts more than 250 ms. Gaps show a "no recording" placeholder. Data comes from
// GET /api/incidents/{id}/replay (argonix/replay.py). Every server string goes through text nodes.
import { api } from "./api.js";
import { segmented } from "./ui.js";
import { h, s, replace, enc, mediaUrl, fmtDur, icon, clamp, RULE } from "./util.js";

const DRIFT_S = 0.25;          // re-seek a playing video that drifts more than this from the master clock
const SEEK_COOLDOWN_MS = 400;  // ...but not more often than this (a seek itself takes time)
const HOLD_MAX_MS = 2500;      // the clock waits for a buffering player at most this long
const MIN_RATE = 0.0625;       // slowest playbackRate browsers accept
const ROLE = { primary: "incident camera", predicted_next: "predicted next · estimate", requested: "added" };
const clock = t => {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + Math.floor((t % 1) * 10);
};

/** createReplay(incidentId) → { el, destroy() }. Nothing loads until the operator presses "Load replay". */
export function createReplay(iid) {
  const el = h("div", { class: "rp" });
  let plan = null, tiles = [], t = 0, playing = false, speed = 1, raf = 0, lastTs = 0, holdSince = 0, dead = false;
  let pre = 10, post = 10;

  const playBtn = h("button", { type: "button", class: "primary", onclick: () => setPlaying(!playing) });
  const speedSeg = segmented([[0.5, "0.5×"], [1, "1×"], [2, "2×"]], speed, v => { speed = v; }, { label: "Playback speed" });
  const readout = h("span", { class: "rp-clock tnum", "aria-live": "off" });
  const status = h("span", { class: "xs mute" });
  const slider = h("input", { type: "range", class: "rp-slider", "aria-label": "Replay time", step: "0.1" });
  slider.addEventListener("input", () => seek(+slider.value));
  const lanes = h("div", { class: "rp-tl" });
  const grid = h("div", { class: "rp-grid" });
  let playhead = null, laneSvg = null;

  function intro(msg) {
    const winSel = h("select", { class: "sm", "aria-label": "Time around the incident" },
      [[10, "±10 s"], [30, "±30 s"], [60, "±60 s"]].map(([v, l]) => h("option", { value: v, selected: v === pre }, l)));
    winSel.addEventListener("change", () => { pre = post = +winSel.value; });
    replace(el, h("div", { class: "row" },
      h("button", { type: "button", onclick: () => load() }, icon("play"), "Load synchronised replay"),
      h("label", { class: "row small mute" }, "Window", winSel),
      msg ? h("span", { class: "small mute" }, msg) : null),
    h("div", { class: "note" }, "Plays every involved camera (and predicted next cameras) against one clock, from before the first event to after the last."));
  }

  async function load() {
    replace(el, h("div", { class: "small mute" }, "Loading replay…"));
    try { plan = await api(`/api/incidents/${enc(iid)}/replay?pre_s=${pre}&post_s=${post}`, { silent: true }); }
    catch (e) { if (!dead) intro(e?.kind === "forbidden" ? "Your role cannot view recordings (needs evidence.view)." : "Replay unavailable."); return; }
    if (dead) return;
    build();
  }

  function build() {
    stopTiles();
    t = plan.start;
    slider.min = String(plan.start); slider.max = String(plan.end); slider.value = String(t);
    tiles = plan.cameras.map(makeTile);
    replace(grid, tiles.map(x => x.el));
    paintLanes();
    const markers = plan.markers || [];
    const jump = dir => {
      const next = dir > 0 ? markers.find(m => m.t > t + 0.5) : [...markers].reverse().find(m => m.t < t - 2.5);
      if (next) seek(next.t - 2);
    };
    replace(el,
      h("div", { class: "rp-bar" }, playBtn, speedSeg,
        h("button", { type: "button", class: "sm", title: "Back to the start", onclick: () => seek(plan.start) }, icon("chevleft"), "Start"),
        h("button", { type: "button", class: "sm", title: "Previous event", onclick: () => jump(-1) }, "‹ event"),
        h("button", { type: "button", class: "sm", title: "Next event", onclick: () => jump(1) }, "event ›"),
        h("span", { class: "grow" }), readout,
        h("button", { type: "button", class: "sm ghost", title: "Reload the replay (new segments close every 60 s)", onclick: () => { setPlaying(false); load(); } }, icon("refresh"))),
      lanes, slider, h("div", { class: "row xs mute", style: { justifyContent: "space-between" } }, h("span", null, clock(plan.start)), status, h("span", null, clock(plan.end))),
      grid,
      h("div", { class: "note" }, plan.note, plan.clamped ? " The window was shortened to 2 h." : ""));
    setPlaying(false);
    render(true);
    cancelAnimationFrame(raf);
    lastTs = 0;
    raf = requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------------------------------------ tiles
  function makeTile(c) {
    const video = h("video", { muted: true, playsinline: true, preload: "auto", disablePictureInPicture: true });
    video.muted = true;
    const phText = h("div");
    const ph = h("div", { class: "rp-ph" }, h("div", null, icon("recordings"), phText));
    const src = h("span", { class: "xs mute" });
    const at = h("span", { class: "rp-at tnum" });
    const tile = { c, video, ph, phText, src, at, cur: undefined, ready: false, broken: false, lastSeek: 0,
      items: [...(c.segments || []), ...(c.clips || [])],
      el: h("div", { class: ["rp-tile", c.role !== "primary" && "rel"] },
        h("div", { class: "rp-th" }, h("b", null, c.name), h("span", { class: ["tag", c.role === "predicted_next" && "est"] }, ROLE[c.role] || c.role),
          c.prediction ? h("span", { class: "xs mute" }, `${Math.round((c.prediction.prob || 0) * 100)}%${c.prediction.eta_s != null ? ` · ~${Math.round(c.prediction.eta_s)} s` : ""}`) : null,
          h("span", { class: "grow" }), src),
        h("div", { class: "rp-v" }, video, ph, at)) };
    video.addEventListener("loadedmetadata", () => { tile.ready = true; sync(tile, true); });
    video.addEventListener("error", () => { if (tile.cur) { tile.broken = true; tile.ready = false; placeholder(tile, "Could not play this file"); } });
    return tile;
  }

  /** The item covering time x: a recording segment first, else an (annotated, approximate) evidence clip. */
  function pick(tile, x) {
    return tile.items.find(i => i.kind === "recording" && i.start <= x && x < i.end)
      || tile.items.find(i => i.kind === "evidence" && i.start <= x && x < i.end) || null;
  }

  function placeholder(tile, text) {
    tile.ph.hidden = false;
    if (tile.phText.textContent !== text) tile.phText.textContent = text;
  }

  function gapText(tile) {
    const ip = tile.c.in_progress;
    return ip && ip.start <= t && t <= ip.end ? `Segment still recording at ${clock(t)} (playable once it closes)` : `No recording at ${clock(t)}`;
  }

  function sync(tile, force = false) {
    const item = pick(tile, t);
    const v = tile.video;
    if (item !== tile.cur) {
      tile.cur = item; tile.ready = false; tile.broken = false;
      if (item) {
        v.src = mediaUrl(item.path);
        placeholder(tile, "Loading…");
        tile.src.textContent = item.kind === "recording" ? `recording · from ${clock(item.start)}` : `evidence clip (annotated, approx. timing) · ${RULE[item.rule] || item.rule || ""}`;
      } else {
        v.pause(); v.removeAttribute("src"); v.load();
        tile.src.textContent = "no recording";
      }
    }
    if (!item) { placeholder(tile, gapText(tile)); return; }
    if (!tile.ready || tile.broken) return;
    tile.ph.hidden = true;
    const span = item.end - item.start;
    const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : span;
    const scale = span > 0 ? dur / span : 1;          // file time per wall-clock second (dropped frames shorten a file)
    const expected = clamp((t - item.start) * scale, 0, Math.max(0, dur - 0.05));
    // A file much shorter than its wall-clock span (the pipeline fell behind and wrote few frames) needs a rate below
    // what browsers play: then the video stays paused and is stepped by seeking, like scrubbing.
    const rate = speed * scale;
    const stepped = rate < MIN_RATE;
    if (playing && !stepped) {
      if (Math.abs(v.playbackRate - Math.min(rate, 16)) > 1e-3) v.playbackRate = Math.min(rate, 16);
      if (v.paused && !v.ended) v.play().catch(() => {});
    } else if (!v.paused) v.pause();
    const drift = v.currentTime - expected;
    const now = performance.now();
    const tol = playing && !stepped ? DRIFT_S : Math.max(0.05, 0.5 * scale);   // stepped: about half a wall second
    if (!v.seeking && (force || (Math.abs(drift) > tol && now - tile.lastSeek > (playing && !stepped ? SEEK_COOLDOWN_MS : 0)))) {
      v.currentTime = expected;
      tile.lastSeek = now;
    }
  }

  function stopTiles() {
    for (const x of tiles) { x.video.pause(); x.video.removeAttribute("src"); x.video.load(); }
    tiles = [];
  }

  // ------------------------------------------------------------------------------------------------ clock
  function setPlaying(p) {
    if (p && plan && t >= plan.end - 0.05) t = plan.start;
    playing = !!p && !!plan;
    replace(playBtn, icon(playing ? "pause" : "play"), playing ? "Pause" : "Play");
    playBtn.setAttribute("aria-pressed", String(playing));
    render(true);
  }

  function seek(x) {
    if (!plan) return;
    t = clamp(x, plan.start, plan.end);
    render(true);
  }

  function buffering() {
    return tiles.some(x => x.cur && !x.broken && (!x.ready || x.video.readyState < 2));
  }

  function tick(ts) {
    if (dead) return;
    const dt = lastTs ? Math.min(0.25, (ts - lastTs) / 1000) : 0;
    lastTs = ts;
    if (playing) {
      const wait = buffering();
      if (wait && !holdSince) holdSince = ts;
      if (!wait) holdSince = 0;
      if (!wait || ts - holdSince > HOLD_MAX_MS) t += dt * speed;
      status.textContent = wait && ts - holdSince > 300 ? "buffering…" : "";
      if (t >= plan.end) { t = plan.end; setPlaying(false); }
      render(false);
    } else status.textContent = "";
    raf = requestAnimationFrame(tick);
  }

  function render(force) {
    if (!plan) return;
    if (document.activeElement !== slider) slider.value = String(t);
    readout.textContent = `${clock(t)}  (+${fmtDur(t - plan.start)} of ${fmtDur(plan.end - plan.start)})`;
    for (const x of tiles) { sync(x, force); x.at.textContent = clock(t); }
    if (playhead) {
      const X = laneX(t);
      playhead.setAttribute("x1", X); playhead.setAttribute("x2", X);
    }
  }

  // ------------------------------------------------------------------------------------------------ timeline
  const LW = 640, LL = 92, LR = 6;
  const laneX = x => LL + (x - plan.start) / Math.max(1e-6, plan.end - plan.start) * (LW - LL - LR);

  function paintLanes() {
    const lane = 20, top = 16, H = top + plan.cameras.length * lane + 4;
    const n = [s("rect", { x: 0, y: 0, width: LW, height: H, class: "tl-bg" })];
    plan.cameras.forEach((c, i) => {
      const y = top + i * lane;
      n.push(s("text", { x: 4, y: y + 14, class: "tl-txt" }, c.name.slice(0, 13)),
        s("rect", { x: LL, y: y + 3, width: LW - LL - LR, height: lane - 6, class: "rp-gap" }, s("title", null, "no recording")));
      for (const g of c.segments || []) n.push(s("rect", { x: laneX(g.start), y: y + 3, width: Math.max(1, laneX(g.end) - laneX(g.start)), height: lane - 6, class: "rp-rec" },
        s("title", null, `recording ${clock(g.start)} → ${clock(g.end)}`)));
      for (const g of c.clips || []) n.push(s("rect", { x: laneX(g.start), y: y + 5, width: Math.max(1, laneX(g.end) - laneX(g.start)), height: lane - 10, class: "rp-clip" },
        s("title", null, `evidence clip (approx.) ${clock(g.start)} → ${clock(g.end)}`)));
      if (c.in_progress) n.push(s("rect", { x: laneX(c.in_progress.start), y: y + 3, width: Math.max(1, laneX(c.in_progress.end) - laneX(c.in_progress.start)), height: lane - 6, class: "rp-prog" },
        s("title", null, "segment still recording")));
    });
    for (const m of plan.markers || []) {
      const x = laneX(m.t);
      const g = s("g", { class: "rp-mk" + (m.learned_model ? " inf" : ""), tabindex: 0, role: "button",
        "aria-label": `Jump to ${RULE[m.rule] || m.rule} at ${clock(m.t)}` },
        s("title", null, `${RULE[m.rule] || m.rule} · ${m.camera}${m.zone_name ? " · " + m.zone_name : ""} · ${clock(m.t)}\nclick: jump 2 s before`),
        s("line", { x1: x, x2: x, y1: 12, y2: H - 2 }), s("path", { d: `M${x - 5},2 L${x + 5},2 L${x},11 z` }));
      const go = e => { e.stopPropagation(); seek(m.t - 2); };
      g.addEventListener("click", go);
      g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(e); } });
      n.push(g);
    }
    playhead = s("line", { x1: LL, x2: LL, y1: 0, y2: H, class: "rp-head" });
    n.push(playhead);
    laneSvg = s("svg", { viewBox: `0 0 ${LW} ${H}`, class: "tl-svg rp-lanes", role: "img", "aria-label": "Recording coverage per camera and event markers" }, n);
    const scrub = e => {
      const r = laneSvg.getBoundingClientRect();
      const vx = (e.clientX - r.left) / r.width * LW;
      if (vx >= LL) seek(plan.start + (vx - LL) / (LW - LL - LR) * (plan.end - plan.start));
    };
    laneSvg.addEventListener("pointerdown", e => { scrub(e); laneSvg.setPointerCapture(e.pointerId); });
    laneSvg.addEventListener("pointermove", e => { if (e.buttons & 1) scrub(e); });
    replace(lanes, laneSvg, h("div", { class: "xs mute" }, "Solid: recording · thin: evidence clip (annotated, approximate) · dashed: still recording · triangles: events (click to jump)"));
  }

  intro();
  return {
    el,
    destroy() { dead = true; cancelAnimationFrame(raf); stopTiles(); },
  };
}
