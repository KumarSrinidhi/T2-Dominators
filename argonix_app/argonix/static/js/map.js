// Map: 2D site ground plane. Uses /api/map (background, FOVs, placements, heatmap) when available, otherwise
// falls back to drawing /api/config zones + live tracks in each camera's own ground frame.
import { optional } from "./api.js";
import { S, on, camColor, camName, loadMap, toSite, placement } from "./state.js";
import { h, s, replace, icon, clamp, cssVar, num, fmtDateTime, ls } from "./util.js";
import { segmented } from "./ui.js";
import { drawMap as drawHandoff, mapPanel as handoffPanel } from "./handoff.js";
import { pageHeader, estBadge, obsBadge } from "./ui.js";

// updated_at may be epoch seconds, epoch ms, an ISO string or an opaque version counter.
function fmtStamp(v) {
  if (typeof v === "number") return v > 1e12 ? fmtDateTime(v / 1000) : v > 1e9 ? fmtDateTime(v) : `version ${v}`;
  const t = Date.parse(v);
  return isNaN(t) ? String(v) : fmtDateTime(t / 1000);
}

export default function createMap() {
  const canvas = h("canvas", { role: "img", "aria-label": "Site map: 2D ground plane with zones, camera footprints and live tracks" });
  const legend = h("div", { class: "map-legend", "aria-hidden": "true" });
  const wrap = h("div", { class: "map-wrap" }, canvas, legend,
    h("div", { class: "map-ctl" },
      h("button", { title: "Zoom in", "aria-label": "Zoom in", onclick: () => zoomAt(1.4) }, icon("zoomin")),
      h("button", { title: "Zoom out", "aria-label": "Zoom out", onclick: () => zoomAt(1 / 1.4) }, icon("zoomout")),
      h("button", { title: "Fit to site (double-click)", "aria-label": "Fit", onclick: () => fit() }, icon("fit"))));
  const side = h("div", { class: "map-side" });
  const modeNote = h("div");
  const el = h("section", { class: "page" },
    pageHeader({ title: "Map", subtitle: "2D ground plane (not a 3D model) · wheel / pinch to zoom · drag to pan",
      actions: [estBadge("positions ± calibration error")] }),
    modeNote, h("div", { class: "map-split" }, wrap, side));

  function renderLegend() {
    const sym = (draw, label) => h("span", null, draw, label);
    const svg = (...kids) => s("svg", { class: "lg-sym", viewBox: "0 0 18 12" }, kids);
    replace(legend,
      sym(svg(s("circle", { cx: 9, cy: 6, r: 3.5, style: { fill: "var(--acc-text)" } })), "Person track (observed)"),
      sym(svg(s("circle", { cx: 9, cy: 6, r: 4.5, style: { fill: "none", stroke: "var(--p1)", strokeWidth: 2 } })), "Rule active"),
      layers.uncert ? sym(svg(s("circle", { cx: 9, cy: 6, r: 5.2, style: { fill: "var(--acc-soft)", stroke: "var(--acc-text)", strokeOpacity: .5 } })), "± calib. error (est.)") : null,
      layers.zones ? sym(svg(s("rect", { x: 2, y: 2, width: 14, height: 8, rx: 1.5, style: { fill: "color-mix(in srgb, var(--acc-text) 18%, transparent)", stroke: "var(--acc-text)", strokeWidth: 1.2 } })), "Zone (thicker = critical)") : null,
      layers.fov && S.map ? sym(svg(s("path", { d: "M2 10L9 2l7 8", style: { fill: "none", stroke: "var(--mute)", strokeWidth: 1.4, strokeDasharray: "3 2" } })), "Camera footprint") : null,
      hoAvail ? sym(svg(s("path", { d: "M2 6h11", style: { stroke: "var(--mute)", strokeWidth: 1.6, strokeDasharray: "3 2" } }), s("path", { d: "M12 3l4 3-4 3z", style: { fill: "var(--mute)" } })), "Next camera (est.)") : null);
  }
  let hoAvail = false;

  const layers = { background: true, fov: true, zones: true, tracks: true, uncert: true, heatmap: false };
  try { Object.assign(layers, JSON.parse(ls.get("map.layers", "{}"))); } catch { /* ignore */ }
  let heat = null, heatMissing = false, bg = null, bgUrl = null;
  // heatmap time-lapse (/api/map/heatmap/history): {data, idx, playing, timer, grid (Float32Array of the frame)}
  let tl = null, tlStep = +ls.get("map.tl.step", "3600") || 3600, tlMissing = false, tlLoading = false;
  const tlBox = h("div");
  let view = { s: 30, tx: 40, ty: 400 };   // screen = (tx + x*s, ty - y*s)
  let fitted = false, shown = false, raf = 0, dpr = 1, W = 0, H = 0, mapTimer = null;
  const trackList = h("div", { class: "list" });
  const camList = h("div", { class: "list" });
  const layerBox = h("div");

  // ---------- data ----------
  function bounds() {
    if (S.map?.bounds_m?.length === 4) return S.map.bounds_m.map(Number);
    const mw = +S.cfg.map?.width_m || 20, mh = +S.cfg.map?.height_m || 15;
    let b = [0, 0, mw, mh];
    for (const c of S.cfg.cameras || []) for (const z of c.zones || []) for (const [x, y] of z.polygon || []) {
      const [sx, sy] = toSite(c.id, x, y);
      b = [Math.min(b[0], sx), Math.min(b[1], sy), Math.max(b[2], sx), Math.max(b[3], sy)];
    }
    return b;
  }
  async function refreshMap() {
    await loadMap();
    const url = S.map?.background_url || (S.map ? "/api/map/background.png" : null);
    if (url && layers.background) {
      const u = url + (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(S.map.updated_at || "");
      if (u !== bgUrl) {
        bgUrl = u;
        const img = new Image();
        img.onload = () => { bg = img; draw(); };
        img.onerror = () => { bg = null; };
        img.src = u;
      }
    }
    renderSide();
    if (!fitted) fit();
    draw();
  }
  async function loadHeat() {
    if (!layers.heatmap || heatMissing) return;
    try {
      heat = await optional("/api/map/heatmap", "heatmap", { silent: true });
      if (heat === null) { heatMissing = true; layers.heatmap = false; renderSide(); }
    } catch { heat = null; }
    draw();
  }

  // ---------- view transform ----------
  const X = x => view.tx + x * view.s, Y = y => view.ty - y * view.s;
  function resize() {
    const r = wrap.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    if (!fitted) fit(); else draw();
  }
  function fit() {
    if (!W || !H) return;
    const [x0, y0, x1, y1] = bounds(), pad = 30;
    const sc = Math.min((W - 2 * pad) / Math.max(1e-3, x1 - x0), (H - 2 * pad) / Math.max(1e-3, y1 - y0));
    view.s = clamp(sc, 0.5, 2000);
    view.tx = (W - (x1 - x0) * view.s) / 2 - x0 * view.s;
    view.ty = H - (H - (y1 - y0) * view.s) / 2 + y0 * view.s;
    fitted = true; draw();
  }
  function zoomAt(f, mx = W / 2, my = H / 2) {
    const ns = clamp(view.s * f, 0.5, 2000); f = ns / view.s;
    view.tx = mx - (mx - view.tx) * f; view.ty = my - (my - view.ty) * f; view.s = ns; draw();
  }

  // ---------- interaction ----------
  const ptrs = new Map(); let pinch = null;
  canvas.addEventListener("wheel", e => { e.preventDefault(); const r = canvas.getBoundingClientRect(); zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  canvas.addEventListener("pointerdown", e => { canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); pinch = null; });
  canvas.addEventListener("pointermove", e => {
    if (!ptrs.has(e.pointerId)) return;
    const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 1) { view.tx += e.clientX - prev[0]; view.ty += e.clientY - prev[1]; draw(); }
    else if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]), r = canvas.getBoundingClientRect();
      if (pinch) zoomAt(d / pinch, (a[0] + b[0]) / 2 - r.left, (a[1] + b[1]) / 2 - r.top);
      pinch = d;
    }
  });
  const up = e => { ptrs.delete(e.pointerId); pinch = null; };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("dblclick", () => fit());

  // ---------- drawing ----------
  function draw() { if (!raf && shown) raf = requestAnimationFrame(paint); }
  function poly(g, pts) { g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))); g.closePath(); }
  function calErr(camId) {
    const m = S.map?.cameras?.find(c => c.id === camId);
    if (m?.calib_error_m != null) return +m.calib_error_m;
    return +(S.cfg.cameras.find(c => c.id === camId)?.calibration?.calib_error_m) || 0;
  }
  function paint() {
    raf = 0;
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const text = cssVar("--text"), mute = cssVar("--mute"), line = cssVar("--line"), err = cssVar("--p1");
    const [x0, y0, x1, y1] = bounds();
    g.font = "11px system-ui, sans-serif";
    // site extent
    if (layers.background && bg) {
      g.globalAlpha = 0.9;
      g.drawImage(bg, X(x0), Y(y1), (x1 - x0) * view.s, (y1 - y0) * view.s);
      g.globalAlpha = 1;
    }
    // grid
    const steps = [0.5, 1, 2, 5, 10, 20, 50, 100];
    const step = steps.find(st => st * view.s >= 40) || 100;
    g.strokeStyle = line; g.lineWidth = 1; g.fillStyle = mute;
    const gx0 = Math.floor(x0 / step) * step, gy0 = Math.floor(y0 / step) * step;
    for (let x = gx0; x <= x1 + 1e-6; x += step) { g.beginPath(); g.moveTo(X(x), Y(y0)); g.lineTo(X(x), Y(y1)); g.stroke(); g.fillText(`${+x.toFixed(1)} m`, X(x) + 2, Y(y0) + 12); }
    for (let y = gy0; y <= y1 + 1e-6; y += step) { g.beginPath(); g.moveTo(X(x0), Y(y)); g.lineTo(X(x1), Y(y)); g.stroke(); g.fillText(`${+y.toFixed(1)} m`, Math.max(2, X(x0) - 36), Y(y) + 4); }
    // heatmap time-lapse (one frame of the last 24 h), else the live decayed heatmap
    if (tl?.grid) {
      const d = tl.data, cell = +d.cell_m, [ox, oy] = d.origin_m, w = d.width, vmax = Math.max(1e-9, +d.max);
      for (let k = 0; k < tl.grid.length; k++) {
        const v = tl.grid[k];
        if (!v) continue;
        const f = clamp(v / vmax, 0, 1), r = Math.floor(k / w), c = k % w;
        g.fillStyle = heatColor(f);
        g.fillRect(X(ox + c * cell), Y(oy + (r + 1) * cell), cell * view.s + 0.5, cell * view.s + 0.5);
      }
      const fr = d.frames[tl.idx], label = `${fmtClock(fr.t_start)}–${fmtClock(fr.t_end)} · ${new Date(fr.t_start * 1000).toLocaleDateString()}`;
      g.font = "600 13px system-ui, sans-serif";
      const tw = g.measureText(label).width;
      g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(10, 10, tw + 20, 26);
      g.fillStyle = "#fff"; g.fillText(label, 20, 28);
      g.font = "11px system-ui, sans-serif";
    } else if (layers.heatmap && heat?.values) {
      const cell = +heat.cell_m || 1, [ox, oy] = heat.origin_m || [0, 0], w = +heat.width || 0, hh = +heat.height || 0;
      const vmax = +heat.max || heat.values.flat().reduce((m, v) => Math.max(m, +v || 0), 1e-9);
      for (let r = 0; r < hh; r++) for (let c = 0; c < w; c++) {
        const v = Array.isArray(heat.values[0]) ? heat.values[r]?.[c] : heat.values[r * w + c];
        if (!v) continue;
        const f = clamp(v / vmax, 0, 1);
        g.fillStyle = heatColor(f);
        g.fillRect(X(ox + c * cell), Y(oy + (r + 1) * cell), cell * view.s + 0.5, cell * view.s + 0.5);
      }
    }
    // camera FOVs (from /api/map, site frame)
    if (layers.fov && S.map?.cameras) for (const c of S.map.cameras) {
      if (!c.fov_polygon_m?.length) continue;
      const col = camColor(c.id);
      poly(g, c.fov_polygon_m);
      g.setLineDash([6, 4]); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
      g.fillStyle = col + "10"; g.fill();
      const [lx, ly] = c.fov_polygon_m[0];
      g.fillStyle = col;
      g.fillText(`${c.name || c.id}${c.calibrated === false ? " (uncalibrated)" : c.calib_error_m != null ? ` · ±${c.calib_error_m} m est.` : ""}`, X(lx) + 4, Y(ly) - 6);
    }
    // zones (camera ground frame -> site frame)
    if (layers.zones) for (const c of S.cfg.cameras || []) for (const z of c.zones || []) {
      if (!z.polygon?.length) continue;
      const col = camColor(c.id), pts = z.polygon.map(([x, y]) => toSite(c.id, x, y));
      poly(g, pts);
      g.fillStyle = col + "26"; g.fill(); g.strokeStyle = col; g.lineWidth = z.criticality >= 3 ? 2.5 : 1.5; g.stroke();
      g.fillStyle = text;
      g.fillText(`${z.name} (${c.id})`, X(pts[0][0]) + 4, Y(pts[0][1]) - 4);
      if (z.direction) {
        const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        const rot = placement(c.id).rot * Math.PI / 180, dx0 = +z.direction[0], dy0 = +z.direction[1];
        const n = Math.hypot(dx0, dy0) || 1, dx = (Math.cos(rot) * dx0 - Math.sin(rot) * dy0) / n, dy = (Math.sin(rot) * dx0 + Math.cos(rot) * dy0) / n;
        const L = 1.2;
        arrow(g, X(cx - dx * L), Y(cy - dy * L), X(cx + dx * L), Y(cy + dy * L), col);
      }
    }
    // live tracks
    if (layers.tracks) for (const t of S.tracks) {
      const [sx, sy] = toSite(t.camera, +t.x, +t.y), px = X(sx), py = Y(sy), col = camColor(t.camera);
      if (layers.uncert) {
        const r = calErr(t.camera) * view.s;
        if (r > 2) { g.beginPath(); g.arc(px, py, r, 0, 7); g.fillStyle = col + "22"; g.fill(); g.strokeStyle = col + "66"; g.lineWidth = 1; g.stroke(); }
      }
      g.beginPath(); g.arc(px, py, 6, 0, 7); g.fillStyle = col; g.fill();
      if (t.alert) { g.beginPath(); g.arc(px, py, 10, 0, 7); g.strokeStyle = err; g.lineWidth = 2.5; g.stroke(); }
      g.fillStyle = text; g.fillText(`#${t.track_id}`, px + 11, py + 4);
    }
    drawHandoff(g, X, Y, text);  // learned camera transitions + predicted next camera (estimates)
    // scale bar
    const sb = step * view.s;
    const sx0 = W - 16 - sb;
    g.strokeStyle = text; g.lineWidth = 2; g.beginPath(); g.moveTo(sx0, H - 14); g.lineTo(sx0 + sb, H - 14); g.stroke();
    g.fillStyle = text; g.fillText(`${step} m`, sx0, H - 20);
  }
  function heatColor(f) { return `hsla(${(1 - f) * 220}, 90%, 50%, ${0.15 + 0.55 * f})`; }
  const fmtClock = t => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // ---------- heatmap time-lapse ----------
  async function loadTimelapse() {
    if (tlLoading) return;
    tlLoading = true; renderTl();
    try {
      const d = await optional(`/api/map/heatmap/history?hours=24&step_s=${tlStep}`, "heat_history", { silent: true });
      if (d === null) { tlMissing = true; tl = null; }
      else {
        const last = d.frames.length - 1;
        const firstBusy = d.frames.findIndex(f => f.cells.length);
        stopTl();
        tl = { data: d, idx: firstBusy >= 0 ? firstBusy : last, playing: false, timer: 0, grid: null };
        setFrame(tl.idx);
      }
    } catch { /* toast already shown by api() unless silent: keep previous */ }
    tlLoading = false; renderTl(); draw();
  }
  function setFrame(i) {
    if (!tl) return;
    const d = tl.data;
    tl.idx = clamp(i, 0, d.frames.length - 1);
    const grid = new Float32Array(d.width * d.height);
    for (const [k, v] of d.frames[tl.idx].cells) grid[k] = v;
    tl.grid = grid;
    syncTl(); draw();
  }
  function stopTl() { if (tl?.timer) { clearInterval(tl.timer); tl.timer = 0; } if (tl) tl.playing = false; }
  function togglePlay() {
    if (!tl) return;
    if (tl.playing) { stopTl(); syncTl(); return; }
    if (tl.idx >= tl.data.frames.length - 1) setFrame(0);
    tl.playing = true;
    tl.timer = setInterval(() => {
      if (tl.idx >= tl.data.frames.length - 1) { stopTl(); syncTl(); return; }
      setFrame(tl.idx + 1);
    }, tlStep <= 900 ? 250 : 700);
    syncTl();
  }
  function closeTl() { stopTl(); tl = null; renderTl(); draw(); }
  const tlSlider = h("input", { type: "range", min: 0, max: 0, value: 0, step: 1, "aria-label": "Time-lapse position", style: { width: "100%" } });
  tlSlider.addEventListener("input", () => { stopTl(); setFrame(+tlSlider.value); });
  const tlPlay = h("button", { class: "primary", onclick: () => togglePlay() });
  const tlLabel = h("div", { class: "small tnum" });
  function syncTl() {
    if (!tl) return;
    const d = tl.data, fr = d.frames[tl.idx];
    tlSlider.max = String(d.frames.length - 1); tlSlider.value = String(tl.idx);
    replace(tlPlay, icon(tl.playing ? "pause" : "play"), tl.playing ? "Pause" : "Play");
    tlPlay.setAttribute("aria-label", tl.playing ? "Pause time-lapse" : "Play time-lapse");
    replace(tlLabel, h("b", null, `${fmtClock(fr.t_start)}–${fmtClock(fr.t_end)}`),
      h("span", { class: "mute" }, ` · ${num(fr.total, 0)} person-s in view${fr.cells.length ? "" : " (no one seen)"}`));
  }
  function renderTl() {
    const stepSeg = segmented([[900, "15 min"], [3600, "1 h"]], tlStep, v => { tlStep = v; ls.set("map.tl.step", String(v)); if (tl) loadTimelapse(); }, { label: "Time-lapse step" });
    if (tlMissing) { replace(tlBox, h("div", { class: "mute small" }, "Heatmap history is not available on this server.")); return; }
    if (!tl) {
      replace(tlBox, h("div", { class: "row", style: { gap: "8px" } }, stepSeg,
        h("button", { onclick: () => loadTimelapse(), disabled: tlLoading || !S.map }, icon("play"), tlLoading ? "Loading…" : "Last 24 h")),
        h("div", { class: "note" }, "Replays where people were seen over the last 24 hours, one frame per step."));
      return;
    }
    const d = tl.data, busy = d.frames.filter(f => f.cells.length).length;
    syncTl();
    replace(tlBox,
      h("div", { class: "row", style: { gap: "8px" } }, tlPlay, stepSeg, h("span", { class: "grow" }),
        h("button", { class: "ghost icon", title: "Reload", "aria-label": "Reload time-lapse", onclick: () => loadTimelapse() }, icon("refresh")),
        h("button", { class: "ghost icon", title: "Close time-lapse (back to live heatmap)", "aria-label": "Close time-lapse", onclick: () => closeTl() }, icon("x"))),
      tlSlider, tlLabel,
      h("div", { style: { marginTop: "8px" } },
        h("div", { style: { height: "10px", borderRadius: "5px", background: `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(heatColor).join(", ")})` } }),
        h("div", { class: "row small mute tnum", style: { justifyContent: "space-between" } }, h("span", null, "0"),
          h("span", null, `${num(d.max, 1)} person-s per ${d.cell_m} m cell`))),
      h("div", { class: "note" }, `${busy} of ${d.frames.length} frames have activity. Colour scale is shared by all frames. ${d.note}`));
  }

  function arrow(g, x0, y0, x1, y1, col) {
    const a = Math.atan2(y1 - y0, x1 - x0), L = 9;
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 3;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - L * Math.cos(a - 0.45), y1 - L * Math.sin(a - 0.45)); g.lineTo(x1 - L * Math.cos(a + 0.45), y1 - L * Math.sin(a + 0.45)); g.closePath(); g.fill();
  }

  // ---------- side panel ----------
  function layerToggle(key, label, disabled, hint) {
    const cb = h("input", { type: "checkbox", checked: layers[key] && !disabled, disabled, "aria-label": label });
    cb.addEventListener("change", () => {
      layers[key] = cb.checked; ls.set("map.layers", JSON.stringify(layers));
      if (key === "heatmap" && cb.checked) loadHeat();
      if (key === "background" && cb.checked) { bgUrl = null; refreshMap(); }
      renderLegend();
      draw();
    });
    return h("label", { class: ["layer", disabled && "off"], title: hint || "" },
      h("span", { class: "ln" }, label, disabled ? h("small", null, "Not available on this server") : hint ? h("small", null, hint) : null),
      h("span", { class: "switch" }, cb, h("span")));
  }
  function renderSide() {
    const hasMap = !!S.map;
    replace(modeNote, hasMap ? null : h("div", { class: "banner" }, icon("info"), h("div", null, "Site map service (/api/map) is not available on this server: showing zones and tracks in each camera's own ground frame, overlaid without placement. ",
      h("span", { class: "mute" }, "Background, camera footprints and heatmap appear once the backend provides them."))));
    replace(layerBox, h("div", { class: "panel-h" }, h("h3", null, icon("layers"), "Layers")),
      layerToggle("background", "Bird's-eye background", !hasMap, "Live frames warped onto the ground plane"),
      layerToggle("fov", "Camera footprints", !hasMap, "Usable ground area each camera sees"),
      layerToggle("zones", "Zones", false, "Rule zones projected to site metres"),
      layerToggle("tracks", "Live tracks", false, "Foot points of current person tracks"),
      layerToggle("uncert", "Position uncertainty", false, "Radius = the camera's estimated calibration error"),
      layerToggle("heatmap", "Occupancy heatmap", !hasMap || heatMissing, "Person-seconds per 0.5 m cell, 2 min half-life"),
      hasMap ? h("div", { style: { borderTop: "1px solid var(--line)", paddingTop: "8px", marginTop: "4px" } },
        h("div", { class: "small b", style: { marginBottom: "6px" } }, icon("clock"), " Heatmap time-lapse"), tlBox) : null);
    renderTl();
    replace(camList, (S.map?.cameras || S.cfg.cameras).map(c => {
      const cal = c.calibration || {};
      const errM = c.calib_error_m ?? cal.calib_error_m;
      return h("div", { class: "row", style: { gap: "8px", flexWrap: "nowrap", padding: "4px 0" } }, h("span", { class: "swatch", style: { background: camColor(c.id) } }),
        h("span", { class: "grow ellipsis small b", title: c.name || camName(c.id) }, c.name || camName(c.id)),
        c.calibrated === false ? h("span", { class: "tag warn" }, "uncalibrated") : errM != null ? estBadge(`± ${errM} m`) : h("span", { class: "tag" }, "no calibration"),
        c.method ? h("span", { class: "tag", title: "Calibration method" }, c.method) : null);
    }));
    const ho = handoffPanel(draw);
    hoAvail = !!ho;
    replace(side,
      h("div", { class: "panel" }, layerBox),
      h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("cameras"), "Cameras")), camList,
        S.map?.updated_at ? h("div", { class: "note" }, `Map updated ${fmtStamp(S.map.updated_at)}`) : null),
      h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon("user"), "Live tracks"), h("span", { class: "grow" }), obsBadge("observed")), trackList),
      ho,
      h("div", { class: "note" }, "2D ground-plane model from per-camera homographies (not a 3D digital twin). Each position carries its camera's estimated calibration error; zone tests use conservative margins. Dots from different cameras are not identity-linked."));
    renderTracks();
    renderLegend();
  }
  function renderTracks() {
    replace(trackList, S.tracks.length ? h("div", { class: "list" }, S.tracks.slice(0, 60).map(t => {
      const [sx, sy] = toSite(t.camera, +t.x, +t.y);
      return h("div", { class: "small", style: { borderLeft: `3px solid ${t.alert ? "var(--p1)" : camColor(t.camera)}`, paddingLeft: "8px" } },
        h("div", { class: "row", style: { gap: "6px" } }, h("b", null, `${t.cls} #${t.track_id}`),
          t.alert ? h("span", { class: "tag err" }, "rule active") : null),
        h("div", { class: "mute tnum" }, `${camName(t.camera)} · (${num(sx, 1)}, ${num(sy, 1)}) m · det. conf ${t.conf}`));
    })) : h("div", { class: "mute small" }, S.ws === "live" ? "No tracks right now." : "Live connection down."));
  }

  const ro = "ResizeObserver" in window ? new ResizeObserver(() => { if (shown) resize(); }) : null;
  ro?.observe(wrap);
  on("theme", () => draw());
  on("cameras", () => { if (shown) { renderSide(); draw(); } });

  let lastList = 0;
  return {
    el,
    async show() {
      shown = true;
      resize();
      renderSide();
      await refreshMap();
      if (layers.heatmap) loadHeat();
      clearInterval(mapTimer);
      mapTimer = setInterval(() => { if (!document.hidden) { refreshMap(); if (layers.heatmap) loadHeat(); } }, 30000);
    },
    hide() { shown = false; clearInterval(mapTimer); stopTl(); if (tl) syncTl(); },
    onTick() {
      draw();
      if (Date.now() - lastList > 1000) { lastList = Date.now(); renderTracks(); }
    },
  };
}
