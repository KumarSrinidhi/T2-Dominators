// Trajectory forecast overlay for Live tiles (EXPERIMENTAL). Draws each track's predicted path as a faint dashed
// line with spread circles that grow and fade with the horizon. These are PREDICTIONS from the person's own recent
// motion (no identity/appearance). Degrades silently when /api/forecast/* is absent (404: feature disabled).
// Only numbers from the server are used, as SVG attributes; no server strings are injected as HTML.
import { optional } from "./api.js";

const FS = { rows: [], missing: false, at: 0 };
const NS = "http://www.w3.org/2000/svg";
const TITLE = "Experimental forecast of where each person may be in the next few seconds, from their own recent motion only. A prediction, not an observation.";
let timer = null;

const wanted = () => !document.hidden && /^#\/live/.test(location.hash || "#/live");

async function poll() {
  if (FS.missing || !wanted()) return;
  try {
    const r = await optional("/api/forecast/tracks", "forecast", { silent: true });
    if (r === null) { FS.missing = true; return; }
    FS.rows = Array.isArray(r?.tracks) ? r.tracks : []; FS.at = Date.now();
  } catch { /* keep previous */ }
}

function ensurePolling() {
  if (timer || FS.missing) return;
  poll();
  timer = setInterval(() => { if (FS.missing) { clearInterval(timer); timer = null; } else poll(); }, 1000);
}

const fin = v => Number.isFinite(+v);
function svg(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

/** Live hook: keep an SVG overlay with the forecast paths of this camera's tracks inside `tileEl`. */
export function tilePaths(tileEl, camId) {
  ensurePolling();
  let ov = tileEl.querySelector(":scope > svg.fc-ov");
  const rows = FS.missing || Date.now() - FS.at > 5000 ? [] :
    FS.rows.filter(r => r.camera === camId && Array.isArray(r.path_px) && r.path_px.length && Array.isArray(r.frame_size));
  if (!rows.length) { if (ov) ov.remove(); tileEl.querySelector(":scope > .fc-tag")?.remove(); return; }
  const [W, H] = rows[0].frame_size.map(Number);
  if (!(W > 0 && H > 0)) return;
  if (!ov) {
    // same box as the <img> (object-fit: contain) so analytics-pixel coordinates line up with the video
    ov = svg("svg", { class: "fc-ov", preserveAspectRatio: "xMidYMid meet", "aria-hidden": "true" });
    Object.assign(ov.style, { position: "absolute", inset: "0", width: "100%", height: "100%", pointerEvents: "none" });
    tileEl.appendChild(ov);
    const tag = document.createElement("span");
    tag.className = "fc-tag"; tag.title = TITLE; tag.textContent = "paths: forecast (experimental)";
    Object.assign(tag.style, { position: "absolute", left: "8px", bottom: "30px", background: "#0008", color: "#cde",
      border: "1px dashed #9ab8", borderRadius: "5px", padding: "0 6px", fontSize: "10.5px", pointerEvents: "auto" });
    tileEl.appendChild(tag);
  }
  ov.setAttribute("viewBox", `0 0 ${W} ${H}`);
  while (ov.firstChild) ov.removeChild(ov.firstChild);
  const sw = Math.max(1.5, W / 400);
  for (const r of rows) {
    const pts = [r.current_px, ...r.path_px].filter(p => Array.isArray(p) && fin(p[0]) && fin(p[1]));
    if (pts.length < 2) continue;
    const col = r.method === "gru" ? "#7dd3fc" : "#e5e7eb";
    ov.appendChild(svg("polyline", { points: pts.map(p => `${+p[0]},${+p[1]}`).join(" "), fill: "none", stroke: col,
      "stroke-width": sw, "stroke-dasharray": `${sw * 3} ${sw * 2}`, "stroke-linecap": "round", opacity: 0.7 }));
    const n = r.path_px.length;
    for (let k = 3; k < n; k += 4) {  // spread circles at ~1.6 s, 3.2 s, 4.8 s: bigger and fainter further out
      const p = r.path_px[k], rad = +(r.sigma_px || [])[k];
      if (!p || !fin(p[0]) || !fin(p[1]) || !fin(rad)) continue;
      ov.appendChild(svg("circle", { cx: +p[0], cy: +p[1], r: Math.max(sw * 2, Math.min(rad, W / 3)), fill: col,
        "fill-opacity": 0.1 * (1 - k / (n + 2)), stroke: col, "stroke-opacity": 0.35 * (1 - k / (n + 2)),
        "stroke-width": sw * 0.6 }));
    }
  }
}
