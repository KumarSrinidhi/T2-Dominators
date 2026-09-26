// Live: camera wall (layout presets incl. focus + strip, fullscreen) and single-camera view. Streams pause when
// the tab is hidden, the page is left or a tile scrolls out of view (the MJPEG connection is dropped by swapping
// src to a 1x1 blank). In demo mode the wall switches to the focus layout and follows new alerts.
// #/live?incident=<id> is the incident focus: only the incident's primary camera (big), its related cameras and the
// handoff's predicted next camera (an estimate), opened from the P1/P2 toast or automatically (workflow.js setting).
// View modes (when the MediaMTX restreamer is enabled, argonix/restream.py): "Smooth" = WebRTC of the camera's main
// stream (full frame rate, no overlay) and "Analytics" = the annotated ~5 fps MJPEG. The single-camera view and the
// focus tile default to Smooth; any WebRTC failure falls back to Analytics automatically and says why.
import { api, safe } from "./api.js";
import { S, can, on, camById, camName, loadCameras, isMono, camState } from "./state.js";
import { autoOpen, setAutoOpen } from "./workflow.js";
import * as drawer from "./drawer.js";
import { toast, segmented, kv, pageHeader, emptyState, prioChip, estBadge, demoBadge, plainTitle } from "./ui.js";
import { h, icon, replace, enc, BLANK, fmtAgo, ls, RULE, num, fmtMs, isTyping } from "./util.js";
import { normRules } from "./state.js";
import { tileBadges } from "./handoff.js";
import { isDemo } from "./shell.js";
import { tilePaths } from "./forecast.js";
import { WhepPlayer, webrtcSupported } from "./webrtc.js";

const FAULTS = [["black", "Black"], ["cover", "Cover"], ["blur", "Blur"], ["shift", "Move"], ["freeze", "Freeze"]];
const ROTATE_MS = 8000, FOLLOW_MS = 20000;

/** One managed stream: <img> (analytics; mode "mjpeg" | "snap" = poll latest.jpg once per second) and, when the
 * restreamer is on, a <video> for WebRTC (vmode "webrtc"). `fallback` holds why WebRTC last failed. */
class Stream {
  constructor(camId, img, msg, video = null) {
    this.cam = camId; this.img = img; this.msg = msg; this.video = video; this.active = false; this.visible = true;
    this.mode = "mjpeg"; this.retry = null; this.poll = null;
    this.vmode = "analytics"; this.player = null; this.fallback = ""; this.live = false; this.onChange = () => {};
    img.addEventListener("error", () => {
      if (!this.active || this.img.src === BLANK) return;
      this.msg.textContent = "stream unavailable · retrying…"; this.msg.hidden = false;
      clearTimeout(this.retry);
      this.retry = setTimeout(() => this.apply(true), 5000);
    });
    img.addEventListener("load", () => { if (this.img.src !== BLANK) this.msg.hidden = true; });
  }
  setMode(m) { this.mode = m; this.apply(true); }
  /** "webrtc" | "analytics". Choosing a mode again clears an earlier WebRTC failure (i.e. retries). */
  setVMode(m) { this.vmode = m; this.fallback = ""; this.apply(true); this.onChange(); }
  get effective() { return this.video && this.vmode === "webrtc" && !this.fallback ? "webrtc" : "analytics"; }
  set(on) { this.active = on; this.apply(); }
  setVisible(v) { this.visible = v; this.apply(); }
  stopPlayer() {
    if (this.player) { this.player.stop(); this.player = null; }
    this.live = false;
    if (this.video) this.video.style.display = "none";
    this.img.style.display = "";
  }
  startPlayer() {
    this.img.src = BLANK; this.img.style.display = "none"; this.video.style.display = "";
    this.msg.textContent = "connecting (WebRTC)…"; this.msg.hidden = false;
    const p = this.player = new WhepPlayer(this.cam, this.video, {
      onPlaying: () => { if (this.player !== p) return; this.live = true; this.msg.hidden = true; this.onChange(); },
      onFail: reason => {
        if (this.player !== p) return;
        this.player = null; this.live = false; this.fallback = reason || "WebRTC failed";
        toast(`Smooth view unavailable for ${camById(this.cam)?.name || this.cam}: ${this.fallback}. Showing the analytics stream.`, "warn", { key: "webrtc:" + this.cam });
        this.apply(true); this.onChange();
      },
    });
    p.start();
  }
  apply(force = false) {
    const want = this.active && this.visible && !document.hidden && this.enabled !== false;
    const eff = this.effective;
    if (want && this.on && this.onMode === eff && !force) return;
    clearTimeout(this.retry); clearInterval(this.poll);
    if (!want) {
      this.stopPlayer();   // releases the WebRTC session as soon as the tile is hidden/off-screen
      if (this.img.src !== BLANK) this.img.src = BLANK;
      this.msg.textContent = this.enabled === false ? "camera disabled" : "paused";
      this.msg.hidden = false;
      this.on = false;
      return;
    }
    this.on = true; this.onMode = eff;
    if (eff === "webrtc") {
      if (!this.player) this.startPlayer();
      return;
    }
    this.stopPlayer();
    if (this.msg.textContent.includes("WebRTC")) this.msg.textContent = "connecting…";
    const base = `/api/cameras/${enc(this.cam)}/`;
    if (this.mode === "snap") {
      const load = () => {
        const pre = new Image();
        pre.onload = () => { if (this.on && this.mode === "snap") { this.img.src = pre.src; this.msg.hidden = true; } };
        pre.onerror = () => { this.msg.textContent = "no frame yet"; this.msg.hidden = false; };
        pre.src = base + "latest.jpg?t=" + Date.now();
      };
      load(); this.poll = setInterval(load, 1000);
    } else {
      this.img.src = base + "mjpeg?t=" + Date.now();
    }
  }
  destroy() { this.active = false; clearTimeout(this.retry); clearInterval(this.poll); this.stopPlayer(); this.img.src = BLANK; }
}

const SMOOTH_T = "Smooth (WebRTC, full-rate, no overlay)", ANALYTICS_T = "Analytics (overlay, ~5 fps)";
/** Overlay chip saying which view mode is active (only on cameras that can do WebRTC). */
function vmodeChip(st, fps) {
  if (st.effective === "webrtc") {
    return h("span", { class: ["vmode", "vm-rtc"], title: "WebRTC: the camera's main stream at full frame rate, straight from the restreamer. No analytics overlay: switch to Analytics to see boxes and zones." },
      st.live ? "Smooth · WebRTC · no overlay" : "Smooth · connecting…");
  }
  if (st.fallback) return h("span", { class: ["vmode", "vm-fail"], title: `WebRTC failed: ${st.fallback}` }, `Analytics · ${fps} fps · WebRTC failed`);
  return h("span", { class: "vmode", title: ANALYTICS_T }, `Analytics · ${fps} fps`);
}

function healthChip(s) {
  const st = s?.status && s.status !== "running" ? s.status : (s?.health || "WARMUP");
  return h("span", { class: ["hchip", "h-" + (s?.status === "error" ? "error" : st)] }, st);
}
const DOT = { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)", off: "var(--p4)", wait: "var(--p4)" };
function fullscreen(el) {
  if (document.fullscreenElement) { document.exitFullscreen?.().catch(() => {}); return; }
  el.requestFullscreen?.().catch(() => toast("Fullscreen is not available here.", "warn"));
}

export default function createLive() {
  const el = h("section", { class: "page" });
  let cols = +ls.get("live.cols", "2") || 2;
  let layout = ls.get("live.layout", "grid") === "focus" ? "focus" : "grid";
  // Each MJPEG stream holds one HTTP/1.1 connection (browsers allow ~6 per host), so default to snapshot
  // polling when there are many cameras unless the operator chose a mode.
  let chosenMode = ls.get("live.mode", null);
  let mode = chosenMode || "mjpeg";
  let streams = [];
  let tiles = new Map();       // camId -> {el, top, bot, stream, cam}
  let view = "grid", camId = null, single = null, shown = false;
  let grid = null, spot = null, spotAlert = null, spotUntil = 0, rotTimer = null, spotBanner = null;
  let focusInc = null;         // incident focus: {id, title, priority, cams: [{id, role, label, est}]}
  let rs = null;                  // /api/restream (null = off or not loaded yet)
  const vOverride = new Map();    // camId -> "webrtc" | "analytics" chosen on a grid tile (this page visit)
  let vmodeBox = null;            // single view: Smooth/Analytics toggle
  const io = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    for (const en of entries) streams.find(s => s.box === en.target)?.setVisible(en.isIntersecting);
  }, { rootMargin: "100px" }) : null;

  function stopAll() { streams.forEach(s => s.destroy()); io?.disconnect(); streams = []; tiles.clear(); single = null; vmodeBox = null; clearInterval(rotTimer); rotTimer = null; }

  const rtcOk = id => !!(rs?.cameras?.[id]?.available) && webrtcSupported();
  const fps = () => S.cfg.analytics_fps || 5;
  /** The view mode a camera should be in: operator choice, else Smooth for the focus target, else Analytics. */
  function wantedVMode(id) {
    if (!rtcOk(id)) return "analytics";
    if (view === "single") return ls.get("live.vmode", "webrtc") === "analytics" ? "analytics" : "webrtc";
    if (vOverride.has(id)) return vOverride.get(id);
    return grid?.classList.contains("focus") && spot === id ? "webrtc" : "analytics";
  }
  function applyVModes() {
    for (const [id, t] of tiles) {
      const m = wantedVMode(id);
      if (t.stream.vmode !== m) t.stream.setVMode(m);
      paintToggle(id);
    }
    paintVModeCtl();
  }
  function toggleVMode(id) {
    const t = tiles.get(id);
    if (!t) return;
    const next = t.stream.effective === "webrtc" ? "analytics" : "webrtc";
    if (view === "single") ls.set("live.vmode", next); else vOverride.set(id, next);
    t.stream.setVMode(next);
    paintToggle(id); paintVModeCtl();
  }
  function paintToggle(id) {
    const t = tiles.get(id);
    if (!t?.vbtn) return;
    t.vbtn.hidden = !rtcOk(id);
    const rtc = t.stream.effective === "webrtc";
    replace(t.vbtn, icon(rtc ? "layers" : "spark"));
    t.vbtn.title = rtc ? `Switch to ${ANALYTICS_T}` : `Switch to ${SMOOTH_T}`;
    t.vbtn.setAttribute("aria-label", t.vbtn.title);
  }
  function paintVModeCtl() {
    if (!vmodeBox || !single) return;
    const t = tiles.get(single.id);
    const info = rs?.cameras?.[single.id];
    if (!t || !rs?.enabled) { replace(vmodeBox); return; }
    if (!rtcOk(single.id)) {
      replace(vmodeBox, h("span", { class: "tag", title: info?.reason || (webrtcSupported() ? "" : "this browser has no WebRTC") }, "Analytics only"));
      return;
    }
    replace(vmodeBox, segmented([["webrtc", "Smooth", SMOOTH_T, "spark"], ["analytics", "Analytics", ANALYTICS_T, "layers"]],
      t.stream.effective, v => { if (v !== t.stream.effective) toggleVMode(single.id); }, { label: "View mode" }));
  }
  const loadRestream = safe(async () => {
    if (!S.cfg?.restream) { rs = null; return; }
    rs = await api("/api/restream", { silent: true });
    if (shown) { applyVModes(); updateTiles(); }
  });

  function mkStream(id, img, msg, video = null) {
    const st = new Stream(id, img, msg, video);
    st.onChange = () => { paintToggle(id); paintVModeCtl(); updateTiles(); };
    st.mode = mode;
    const c = camById(id);
    st.enabled = c?.enabled;
    st.box = img.parentElement || img;   // observe the tile: the <img> is display:none while WebRTC plays
    streams.push(st); io?.observe(st.box);
    return st;
  }

  const wallCams = () => focusInc ? focusInc.cams.map(x => camById(x.id)).filter(Boolean) : S.cams;
  const focusOn = () => (layout === "focus" || isDemo() || !!focusInc) && wallCams().length > 1;

  /** Incident focus from GET /api/incidents/{id}: primary = camera of the newest event, related = its other cameras,
   *  predicted next = the handoff estimates recorded on the incident (top 2 by probability, not already shown). */
  async function loadFocus(iid) {
    let inc;
    try { inc = await api(`/api/incidents/${enc(iid)}`, { silent: true }); }
    catch { toast(`Incident ${iid} is not available; showing all cameras.`, "warn", { key: "focus:" + iid }); return null; }
    const evs = inc.events || [];
    const primary = evs.length ? evs[evs.length - 1].camera_id : (inc.cameras || [])[0];
    const cams = [];
    const add = (id, role, label, est = false) => { if (id && camById(id) && !cams.some(c => c.id === id)) cams.push({ id, role, label, est }); };
    add(primary, "primary", "Primary");
    for (const c of inc.cameras || []) add(c, "related", "Related");
    const preds = Object.values(inc.predictions || {}).flatMap(r => (r.predictions || []).map(p => ({ ...p, from: r.camera })))
      .filter(p => p.camera && !p.leaves_coverage).sort((a, b) => (+b.prob || 0) - (+a.prob || 0));
    for (const p of preds.filter(p => !cams.some(c => c.id === p.camera)).slice(0, 2))
      add(p.camera, "predicted", `Predicted next · ${Math.round((+p.prob || 0) * 100)}%${p.eta_s != null ? ` · ~${Math.round(p.eta_s)} s` : ""}`, true);
    if (!cams.length) { toast(`Incident ${iid} has no camera on this wall; showing all cameras.`, "warn"); return null; }
    return { id: inc.id, title: inc.title, priority: inc.priority, phase: inc.phase, cams };
  }
  const roleOf = id => focusInc?.cams.find(c => c.id === id);
  function autoOpenBtn() {
    const b = h("button", { class: ["sm", autoOpen() && "on"], type: "button", "aria-pressed": String(autoOpen()),
      title: "Auto-open cameras on P1/P2: a new or escalated P1/P2 incident switches Live to its cameras (this browser only)" },
    icon("bell"), h("span", { class: "hide-sm" }, "Auto-open P1/P2"));
    b.addEventListener("click", () => { setAutoOpen(!autoOpen()); b.classList.toggle("on", autoOpen()); b.setAttribute("aria-pressed", String(autoOpen())); });
    return b;
  }

  // ---------------- grid ----------------
  function renderGrid() {
    stopAll();
    grid = h("div", { class: "grid", style: { "--cols": cols } });
    grid.style.setProperty("--cols", cols);
    const layoutSel = segmented([[1, "", "1 per row", "grid1"], [2, "", "2 × 2", "grid2"], [3, "", "3 × 3", "grid3"], [4, "", "4 × 4", "grid4"],
      ["focus", "", "Focus + strip (follows new alerts in demo mode)", "focus"]], layout === "focus" ? "focus" : cols, v => {
      if (v === "focus") { layout = "focus"; }
      else { layout = "grid"; cols = v; ls.set("live.cols", v); grid.style.setProperty("--cols", v); }
      ls.set("live.layout", layout === "focus" ? "focus" : null);
      applyLayout();
    }, { label: "Layout" });
    if (!chosenMode) mode = S.cams.length > 4 ? "snap" : "mjpeg";
    const modeSel = segmented([["mjpeg", "Stream", "Live MJPEG stream"], ["snap", "Low-bw", "Poll one snapshot per second (less bandwidth)"]], mode, v => {
      mode = chosenMode = v; ls.set("live.mode", v); streams.forEach(s => s.setMode(v));
    }, { label: "Stream mode" });
    spotBanner = h("div", { class: "spot-banner", hidden: true, role: "status" });
    replace(el,
      pageHeader({ title: "Live", subtitle: focusInc ? `Incident focus · ${focusInc.cams.length} of ${S.cams.length} cameras`
        : `${S.cams.length} camera${S.cams.length === 1 ? "" : "s"} · analytics at ${S.cfg.analytics_fps || 5} fps`,
        actions: [autoOpenBtn(), focusInc ? null : h("span", { class: "hide-sm" }, layoutSel), modeSel,
          h("button", { class: "icon", title: "Fullscreen wall", "aria-label": "Fullscreen camera wall", onclick: () => fullscreen(grid) }, icon("maximize"))] }),
      spotBanner, grid);
    if (!S.cams.length) {
      grid.replaceWith(emptyState({ icon: "cameras", title: "No cameras configured",
        text: "Connect an RTSP camera, an NVR channel, a video file or a webcam to start live analytics.",
        action: can("settings.edit") ? h("a", { class: "btnlink primary", href: "#/cameras" }, icon("plus"), "Add camera") : null }));
      return;
    }
    for (const c of wallCams()) {
      const img = h("img", { alt: `Live view of ${c.name}`, src: BLANK });
      const video = mkVideo(c);
      const msg = h("div", { class: "msg" }, "connecting…");
      const top = h("div", { class: "ov top" }), bot = h("div", { class: "ov bot" });
      const open = () => { location.hash = "#/live/" + enc(c.id); };
      const vbtn = h("button", { type: "button", hidden: true, onclick: e => { e.stopPropagation(); toggleVMode(c.id); } });
      const tile = h("div", { class: "tile", tabIndex: 0, role: "button", "aria-label": `Open ${c.name}`, dataset: { cam: c.id },
        onclick: open,
        onkeydown: e => { if (e.key === "Enter") open(); else if (e.key === "f" && !isTyping(e)) { e.preventDefault(); e.stopPropagation(); fullscreen(tile); } } },
      img, video, msg, top, bot,
      h("div", { class: "tile-acts" }, vbtn,
        h("button", { type: "button", title: "Fullscreen (f)", "aria-label": `Fullscreen ${c.name}`, onclick: e => { e.stopPropagation(); fullscreen(tile); } }, icon("maximize")),
        h("button", { type: "button", title: "Open camera", "aria-label": `Open ${c.name}`, onclick: e => { e.stopPropagation(); open(); } }, icon("external"))));
      if (roleOf(c.id)) tile.classList.add("frole-" + roleOf(c.id).role);
      grid.appendChild(tile);
      const st = mkStream(c.id, img, msg, video);
      tiles.set(c.id, { el: tile, top, bot, stream: st, cam: c, vbtn });
      st.vmode = wantedVMode(c.id);
      st.set(true);
      paintToggle(c.id);
    }
    applyLayout();
    updateTiles();
  }
  function mkVideo(c) {
    const v = h("video", { muted: true, autoplay: true, playsinline: true, "aria-label": `Smooth live view of ${c.name}`, style: { display: "none" } });
    v.muted = true;   // the attribute alone does not mute in every browser; audio is never played (manual section 15 V-11)
    return v;
  }

  // ---------------- focus layout + demo spotlight ----------------
  function applyLayout() {
    if (!grid) return;
    const f = focusOn();
    grid.classList.toggle("focus", f);
    if (!f) { setSpot(null); clearInterval(rotTimer); rotTimer = null; paintBanner(); return; }
    if (focusInc) setSpot(focusInc.cams[0].id);          // incident focus: the primary camera stays big, no rotation
    else if (!spot || !tiles.has(spot)) setSpot(S.cams[0]?.id);
    clearInterval(rotTimer);
    rotTimer = focusInc ? null : setInterval(rotate, ROTATE_MS);
    paintBanner();
  }
  function setSpot(id, alert = null) {
    spot = id; spotAlert = alert;
    for (const [cid, t] of tiles) t.el.classList.toggle("spot", cid === id && grid?.classList.contains("focus"));
    if (rs) applyVModes();   // the focus tile goes Smooth, the others back to Analytics (unless chosen per tile)
    paintBanner();
  }
  function rotate() {
    if (!shown || document.hidden || Date.now() < spotUntil) return;
    const ids = [...tiles.keys()];
    if (ids.length < 2) return;
    setSpot(ids[(ids.indexOf(spot) + 1) % ids.length]);
  }
  function paintBanner() {
    if (!spotBanner) return;
    const f = grid?.classList.contains("focus");
    spotBanner.hidden = !f && !focusInc;
    if (focusInc) {
      const other = focusInc.cams.slice(1);
      replace(spotBanner, icon("incidents"),
        h("span", { class: "ellipsis" }, "Incident focus: ", prioChip(focusInc.priority), " ", h("b", null, focusInc.title || focusInc.id),
          ` · primary ${camName(focusInc.cams[0].id)}`,
          other.length ? ` · ${other.map(c => `${c.est ? "predicted next (estimate)" : "related"} ${camName(c.id)}`).join(", ")}` : ""),
        h("span", { class: "grow" }),
        h("a", { class: "btnlink sm", href: "#/incidents/" + enc(focusInc.id) }, "Open incident"),
        h("a", { class: "btnlink sm", href: "#/live" }, "Show all cameras"));
      return;
    }
    if (!f) return;
    const c = camById(spot);
    const following = spotAlert && Date.now() < spotUntil;
    replace(spotBanner, icon(following ? "bell" : "focus"),
      following ? [h("span", null, "Following new alert on ", h("b", null, c?.name || spot), ": "), prioChip(spotAlert.priority), h("span", { class: "ellipsis" }, spotAlert.title)]
        : h("span", null, "Focus: ", h("b", null, c?.name || spot), isDemo() ? " · demo mode rotates every 8 s and jumps to new alerts" : " · rotates every 8 s"),
      h("span", { class: "cd" }, following ? `for ${Math.max(0, Math.ceil((spotUntil - Date.now()) / 1000))} s` : ""),
      following ? h("button", { class: "sm", onclick: () => { drawer.setNavigator(null); drawer.openAlert(spotAlert.id); } }, "Open alert") : null);
  }
  const offSpot = on("spotlight", a => {
    if (!shown || view !== "grid" || !focusOn() || focusInc) return;
    const cid = (a.cameras || []).find(id => tiles.has(id));
    if (!cid) return;
    spotUntil = Date.now() + FOLLOW_MS;
    setSpot(cid, a);
  });
  const offDemo = on("demo", () => { if (shown && view === "grid") applyLayout(); });

  function updateTiles() {
    for (const [id, t] of tiles) {
      const s = S.stats[id], n = S.tracks.filter(x => x.camera === id), alerting = n.some(x => x.alert), cs = camState(id);
      const role = view === "grid" ? roleOf(id) : null;
      replace(t.top, h("span", { class: ["dot", "vs-" + cs.state], title: cs.label }), h("span", { class: "name" }, t.cam.name),
        role ? h("span", { class: ["frole", role.role], title: role.est ? "Handoff estimate (probabilistic, research-grade); not an observation" : "Camera with events in this incident" }, role.label) : null,
        isMono(id) ? h("span", { class: "irb", title: "Monochrome / infrared picture" }, icon("ir"), "IR") : null,
        alerting ? h("span", { class: "ruleon" }, "Rule active") : healthChip(s));
      const chip = rtcOk(id) || t.stream.fallback ? vmodeChip(t.stream, fps()) : null;   // which view mode is on
      replace(t.bot, chip, s ? `${chip ? "analytics " : ""}${num(s.fps, 1)} fps · detect ${fmtMs(s.detect_ms)} · ${n.length} track${n.length === 1 ? "" : "s"}` : "waiting for stats…",
        s?.health_reasons?.length ? ` · ${s.health_reasons.join(", ")}` : "");
      t.el.classList.toggle("alerting", alerting);
      tileBadges(t.el, id);  // next-camera estimates for tracks leaving this view
      tilePaths(t.el, id);   // experimental trajectory forecasts (only when "forecast" is enabled)
    }
    if (spotAlert) paintBanner();
  }

  // ---------------- single camera ----------------
  function renderSingle(id) {
    stopAll();
    grid = null;
    const c = camById(id);
    if (!c) {
      replace(el, pageHeader({ title: "Unknown camera", crumbs: [{ label: "Live", href: "#/live" }] }),
        emptyState({ icon: "cameras", title: `No camera "${id}"`, text: "It may have been removed or renamed.", action: h("a", { class: "btnlink", href: "#/live" }, "Back to all cameras") }));
      return;
    }
    const img = h("img", { alt: `Live view of ${c.name}`, src: BLANK });
    const video = mkVideo(c);
    const msg = h("div", { class: "msg" }, "connecting…");
    const top = h("div", { class: "ov top" }), bot = h("div", { class: "ov bot" });
    vmodeBox = h("span", { class: "vmode-ctl" });
    const tile = h("div", { class: "tile", tabIndex: 0 }, img, video, msg, top, bot,
      h("div", { class: "tile-acts" }, h("button", { type: "button", title: "Fullscreen (f)", "aria-label": "Fullscreen", onclick: () => fullscreen(tile) }, icon("maximize"))));
    const statBox = h("div"), alertsBox = h("div", { class: "list" });
    const cal = c.calibration;
    const zones = (S.cfg.cameras.find(x => x.id === id)?.zones) || [];
    let tamper = null;
    if (can("settings.edit")) {
      const secs = h("select", { "aria-label": "Fault duration", class: "sm" }, [5, 10, 30].map(v => h("option", { value: v, selected: v === 10 }, `${v} s`)));
      tamper = h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("alert"), "Tamper test"), h("span", { class: "tag" }, "admin")),
        h("div", { class: "row" }, FAULTS.map(([f, l]) => h("button", { class: "sm", onclick: safe(async () => {
          await api(`/api/cameras/${enc(id)}/inject`, { method: "POST", body: { fault: f, seconds: +secs.value } });
          toast(`Injected "${l}" into ${c.name} for ${secs.value} s`, "ok");
        }) }, l)), secs),
        h("div", { class: "note" }, "Injects a synthetic fault into the analytics frames (demo/testing only). The camera-health rule should raise a \"Camera view problem\" alert."));
    }
    replace(el,
      pageHeader({ title: c.name, crumbs: [{ label: "Live", href: "#/live" }, { label: c.name }], back: "#/live",
        subtitle: h("span", { class: "mono" }, c.id),
        actions: [vmodeBox, (can("zone.edit") || can("settings.edit") || can("calibration.approve")) ? h("a", { class: "btnlink", href: `#/cameras/${enc(id)}` }, icon("sliders"), "Zones & calibration") : null,
          h("button", { onclick: () => fullscreen(tile) }, icon("maximize"), "Fullscreen")] }),
      h("div", { class: "cam-split" },
        h("div", null, tile),
        h("div", { class: "stack" },
          h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("signal"), "Status")), statBox),
          h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("target"), "Calibration & zones")),
            cal && cal.calib_error_m != null ? kv([["Ground error", [`≈ ${cal.calib_error_m} m `, estBadge()]],
              cal.measured_rms_m != null ? ["Measured RMS", `${cal.measured_rms_m} m on ${cal.check_points ?? 0} check point(s)`] : null,
              cal.fit && cal.fit.points > 4 ? ["Point fit", `${cal.fit.inliers} of ${cal.fit.points} points agree` +
                (cal.fit.outliers?.length ? ` (outliers: #${cal.fit.outliers.map(i => i + 1).join(", #")})` : "") +
                (cal.fit.loo_rmse_m != null ? `; leave-one-out error ${cal.fit.loo_rmse_m} m` : "")] : null].filter(Boolean)) : h("div", { class: "mute small" }, "Not calibrated: positions and speeds are unavailable; pixel zones still work."),
            h("div", { class: "list", style: { marginTop: "10px" } }, zones.length ? zones.map(z => h("div", { class: "item", style: { cursor: "default" } },
              h("div", { class: "row" }, h("b", null, z.name), h("span", { class: "grow" }), h("span", { class: "tag" }, `criticality ${z.criticality}`)),
              h("div", { class: "small mute" }, Object.keys(normRules(z.rules)).map(r => RULE[r] || r).join(", ") || "no rules"))) : h("div", { class: "mute small" }, "No zones.")),
            h("div", { class: "note" }, "Positions are a 2D ground-plane estimate from a homography, not a 3D model.")),
          tamper,
          h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("bell"), "Recent alerts")), alertsBox))));
    tile.addEventListener("keydown", e => { if (e.key === "f") { e.preventDefault(); fullscreen(tile); } });
    const st = mkStream(id, img, msg, video);
    single = { id, top, bot, statBox, alertsBox, cam: c, tile };
    tiles.set(id, { el: tile, top, bot, stream: st, cam: c });
    st.vmode = wantedVMode(id);
    st.set(true);
    paintVModeCtl();
    updateTiles(); updateSingle(); loadCamAlerts();
  }

  function updateSingle() {
    if (!single) return;
    const s = S.stats[single.id];
    replace(single.statBox, s ? kv([
      ["Worker", s.status + (s.error ? ` — ${s.error}` : "")],
      ...(rs?.enabled ? [["Live view", liveViewText(single.id)]] : []),
      ["View health", [h("span", { class: "h-" + (s.health || "WARMUP") }, s.health || "WARMUP"), (s.health_reasons || []).length ? ` (${s.health_reasons.join(", ")})` : ""]],
      ["Picture", isMono(single.id) ? "Monochrome / IR" : "Colour (or not reported)"],
      ["Analytics fps", num(s.fps, 2)], ["Detect time", fmtMs(s.detect_ms)],
      ["Frames / dropped", `${s.frames ?? "—"} / ${s.dropped ?? 0}`], ["Masked detections", s.masked ?? 0],
      ["Events", s.events ?? 0], ["Live tracks", S.tracks.filter(t => t.camera === single.id).length],
    ]) : h("div", { class: "mute small" }, S.ws === "live" ? "Waiting for stats…" : "Live connection down."));
  }

  function liveViewText(id) {
    const t = tiles.get(id), info = rs?.cameras?.[id];
    if (!t) return "—";
    if (t.stream.effective === "webrtc") return `Smooth: WebRTC, main stream${info?.codecs?.length ? " (" + info.codecs.join(", ") + ")" : ""}, full frame rate, no overlay`;
    if (t.stream.fallback) return `Analytics (~${fps()} fps). WebRTC failed: ${t.stream.fallback}`;
    return rtcOk(id) ? `Analytics: annotated, ~${fps()} fps` : `Analytics only${info?.reason ? " — " + info.reason : ""}`;
  }

  const loadCamAlerts = safe(async () => {
    if (!single) return;
    const id = single.id;
    const list = await api(`/api/alerts?limit=20&camera=${enc(id)}`, { silent: true });
    if (!single || single.id !== id) return;
    const ids = list.map(a => a.id);
    replace(single.alertsBox, list.length ? list.map(a => h("button", { class: "item", onclick: () => {
      drawer.setNavigator(navOver(ids, a.id));
      drawer.openAlert(a.id);
    } }, h("div", { class: "row", style: { flexWrap: "nowrap" } }, prioChip(a.priority), demoBadge(a), h("span", { class: "grow ellipsis b" }, plainTitle(a))),
      h("div", { class: "small mute", style: { marginTop: "3px" } }, `${fmtAgo(a.updated_at)} · ${a.status}`))) : h("div", { class: "mute small" }, "No alerts for this camera yet."));
  });

  function navOver(ids, start) {
    let i = ids.indexOf(start);
    return {
      next() { if (i < ids.length - 1) { i++; drawer.openAlert(ids[i]); return true; } return false; },
      prev() { if (i > 0) { i--; drawer.openAlert(ids[i]); return true; } return false; },
    };
  }

  const offCams = on("cameras", () => { if (shown) (view === "grid" ? renderGrid() : renderSingle(camId)); });

  return {
    el,
    async show(args, q) {
      shown = true;
      const next = args[0] ? "single" : "grid";
      const iid = next === "grid" ? (q?.get("incident") || null) : null;
      const refocus = iid !== (focusInc?.id ?? null);
      if (refocus) {
        focusInc = iid ? await loadFocus(iid) : null;
        if (iid && !focusInc) history.replaceState(null, "", "#/live");
        if (!shown) return;
      }
      if (refocus || next !== view || args[0] !== camId || !streams.length) {
        view = next; camId = args[0] || null;
        if (view === "grid") renderGrid(); else renderSingle(camId);
      } else { streams.forEach(s => s.set(true)); if (view === "grid") applyLayout(); }
      if (!S.cams.length) loadCameras().catch(() => {});
      loadRestream();
    },
    hide() { shown = false; streams.forEach(s => s.set(false)); clearInterval(rotTimer); rotTimer = null; },
    onVisibility() { streams.forEach(s => s.apply()); },
    onTick() { updateTiles(); updateSingle(); },
    onAlert(a) { if (single && (a.cameras || []).includes(single.id)) loadCamAlerts(); },
    onResync() { loadCamAlerts(); },
    destroy() { offCams(); offSpot(); offDemo(); stopAll(); },
  };
}

