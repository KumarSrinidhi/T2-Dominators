// DOM + formatting helpers. All server strings go through text nodes / properties (never innerHTML).

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const PROPS = new Set(["value", "checked", "disabled", "selected", "hidden", "indeterminate", "src", "href",
  "title", "type", "min", "max", "step", "placeholder", "tabIndex", "open", "muted", "loop", "autoplay", "controls"]);

function build(el, attrs, children) {
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.setAttribute("class", Array.isArray(v) ? v.filter(Boolean).join(" ") : v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (PROPS.has(k) && !(el instanceof SVGElement)) el[k] = v;
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false || c === "") continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** h("div", {class:"x", onclick: fn}, "text", child, [more]) — strings always become text nodes. */
export function h(tag, attrs, ...children) {
  return build(document.createElement(tag), attrs, children);
}
const SVGNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...children) {
  return build(document.createElementNS(SVGNS, tag), attrs, children);
}
export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function replace(el, ...children) { clear(el); return append(el, children); }

/** Static icon markup (constant strings only, never server data). */
const ICONS = {
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  back: '<path d="M15 18l-6-6 6-6"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  zoomin: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5M11 8v6M8 11h6"/>',
  zoomout: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5M8 11h6"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  refresh: '<path d="M20 11a8 8 0 10-2 6M20 5v6h-6"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
  edit: '<path d="M4 20h4L20 8l-4-4L4 16z"/>',
  play: '<path d="M7 4l13 8-13 8z"/>',
  pause: '<path d="M8 5v14M16 5v14"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5"/>',
  moon: '<path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/>',
  logout: '<path d="M15 4h4v16h-4M10 8l-4 4 4 4M6 12h11"/>',
  // ---- shell / navigation ----
  overview: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
  live: '<rect x="2" y="5" width="15" height="14" rx="2"/><path d="M17 10l5-3v10l-5-3"/>',
  review: '<path d="M12 2l8 3.5v6c0 5-3.4 9-8 10.5C7.4 20.5 4 16.5 4 11.5v-6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
  map: '<path d="M9 3L3 6v15l6-3 6 3 6-3V3l-6 3z"/><path d="M9 3v15M15 6v15"/>',
  recordings: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3.2" fill="currentColor"/>',
  cameras: '<path d="M3 7h4l2-3h6l2 3h4v13H3z"/><circle cx="12" cy="13" r="4"/>',
  system: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  incidents: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 17.5l9 5 9-5" opacity=".55"/>',
  analytics: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  journeys: '<circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="6" r="2.5"/><path d="M8.5 18H15a3 3 0 000-6H9a3 3 0 010-6h6.5"/>',
  investigate: '<circle cx="11" cy="11" r="6.5"/><path d="M20.5 20.5l-4.9-4.9M8.5 11h5M11 8.5v5"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0113 0"/><path d="M16 4.6a3.5 3.5 0 010 6.8M21.5 20a6.5 6.5 0 00-4-6"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1"/>',
  nodes: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8.5" y="8.5" width="7" height="7" rx="1"/><path d="M9 1.5v2.5M15 1.5v2.5M9 20v2.5M15 20v2.5M1.5 9H4M1.5 15H4M20 9h2.5M20 15h2.5"/>',
  // ---- ui ----
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/>',
  bell: '<path d="M6 8a6 6 0 1112 0c0 7 3 8.5 3 8.5H3S6 15 6 8z"/><path d="M10 20.5a2.2 2.2 0 004 0"/>',
  chevdown: '<path d="M6 9l6 6 6-6"/>',
  chevright: '<path d="M9 6l6 6-6 6"/>',
  chevleft: '<path d="M15 6l-6 6 6 6"/>',
  up: '<path d="M6 15l6-6 6 6"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  maximize: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  minimize: '<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/>',
  grid1: '<rect x="3" y="4" width="18" height="16" rx="2"/>',
  grid2: '<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/>',
  grid3: '<path d="M3 3h18v18H3zM9 3v18M15 3v18M3 9h18M3 15h18"/>',
  grid4: '<path d="M3 3h18v18H3zM7.5 3v18M12 3v18M16.5 3v18M3 7.5h18M3 12h18M3 16.5h18"/>',
  focus: '<rect x="3" y="3" width="13" height="12" rx="1.5"/><rect x="18" y="3" width="3" height="5" rx="1"/><rect x="18" y="10" width="3" height="5" rx="1"/><rect x="3" y="17" width="5" height="4" rx="1"/><rect x="10" y="17" width="5" height="4" rx="1"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="3.5" cy="6" r="1" fill="currentColor"/><circle cx="3.5" cy="12" r="1" fill="currentColor"/><circle cx="3.5" cy="18" r="1" fill="currentColor"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
  alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.01"/>',
  check: '<path d="M4 12.5l5 5L20 6.5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/>',
  more: '<circle cx="5" cy="12" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><circle cx="19" cy="12" r="1.3" fill="currentColor"/>',
  keyboard: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
  present: '<rect x="2" y="3" width="20" height="13" rx="2"/><path d="M8 21l4-5 4 5M12 7v5M9 10l3-3 3 3"/>',
  filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  pin: '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0114 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
  ir: '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 000 18z" fill="currentColor"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
  trendup: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  trenddown: '<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/>',
  building: '<path d="M4 21V5l8-3v19M12 8l8 3v10M2 21h20M7.5 8h1M7.5 12h1M7.5 16h1M15.5 14h1M15.5 17.5h1"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/>',
  wifioff: '<path d="M2 2l20 20M8.5 16.5a5 5 0 017 0M5 12.5a10 10 0 015.5-2.8M19 12.5a10 10 0 00-2.4-1.8M2 8.8A15 15 0 016.3 6M22 8.8a15 15 0 00-11-3.7M12 20h.01"/>',
  signal: '<path d="M2 20h.01M7 20v-4M12 20v-8M17 20V8M22 4v16"/>',
  sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  key: '<circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3L21 2M16 7l3 3M18 5l2 2"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 018 0v4"/>',
  door: '<path d="M3 21h18"/><rect x="6" y="3" width="12" height="18" rx="1"/><circle cx="14.5" cy="12" r="1" fill="currentColor"/>',
  copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 00-2-2H5a2 2 0 00-2 2v9a2 2 0 002 2h3"/>',
};
/** icon(name) → <svg class="i">. Unknown names fall back to a neutral dot so a typo never breaks a page. */
export function icon(name, cls = "") {
  const el = document.createElementNS(SVGNS, "svg");
  el.setAttribute("class", cls ? "i " + cls : "i"); el.setAttribute("viewBox", "0 0 24 24"); el.setAttribute("fill", "none");
  el.setAttribute("stroke", "currentColor"); el.setAttribute("stroke-width", "2");
  el.setAttribute("stroke-linecap", "round"); el.setAttribute("stroke-linejoin", "round");
  el.setAttribute("aria-hidden", "true"); el.setAttribute("focusable", "false");
  el.innerHTML = ICONS[name] || '<circle cx="12" cy="12" r="3.5"/>';
  return el;
}
export const hasIcon = name => Object.prototype.hasOwnProperty.call(ICONS, name);

// ---------- formatting ----------
export const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const num = (v, d = 2) => (typeof v === "number" && isFinite(v)) ? +v.toFixed(d) : "—";
export const fmtTime = t => t ? new Date(t * 1000).toLocaleTimeString() : "—";
export const fmtDateTime = t => t ? new Date(t * 1000).toLocaleString() : "—";
export function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const d = Math.floor(sec / 86400), hh = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60), ss = sec % 60;
  return d ? `${d}d ${hh}h` : hh ? `${hh}h ${m}m` : m ? `${m}m ${ss}s` : `${ss}s`;
}
export function fmtAgo(t) {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return fmtDateTime(t);
}
export const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const howLabel = how => `${DAYS[Math.floor(how / 24) % 7] || "?"} ${String(how % 24).padStart(2, "0")}:00`;

/** /media/<path> with each segment encoded. */
export function mediaUrl(p) {
  if (!p) return "";
  return "/media/" + String(p).split("/").map(encodeURIComponent).join("/");
}
export const enc = encodeURIComponent;

export const NOSNAP = "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 9"><rect width="16" height="9" fill="#111"/><text x="8" y="5.2" font-size="1.1" fill="#889" text-anchor="middle" font-family="sans-serif">no frame available</text></svg>');
export const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

// ---------- vocabulary (observable behaviour only; never labels a person) ----------
export const RULE = {
  intrusion: "Restricted-zone entry", loitering: "Extended dwell", running: "Fast movement",
  wrong_way: "Movement against permitted direction", crowding: "Occupancy above limit",
  camera_health: "Camera view problem", occupancy_deviation: "Unusual occupancy for this hour",
  counterflow: "Counter-flow in zone", density: "Crowd density above limit", queue_long: "Queue longer than limit",
  dispersal: "Rapid outward movement from a point (dispersal cue)",
  convergence: "Rapid inward movement to a point (convergence cue)",
  occupancy_shift: "Sustained occupancy shift for this hour", unattended_object: "Object left unattended",
  trajectory_anomaly: "Unusual trajectory (model)",
  line_cross: "Counting-line crossing", vehicle_in_zone: "Vehicle in pedestrian-only zone",
  vehicle_stopped: "Vehicle stopped in no-stopping zone",
  congestion_forecast: "Congestion expected (forecast)",
  door_forced: "Door reported forced open", door_held_open: "Door reported held open",
  access_denied_burst: "Repeated access denials at a door", alarm_input: "Alarm input reported",
  access_tamper: "Access-control device tamper reported", sensor_offline: "Access-control source silent",
};
export const RULE_PARAMS = {
  intrusion: { min_seconds: [1.0, "Min. seconds inside"] },
  loitering: { min_seconds: [20, "Min. seconds in zone"] },
  running: { speed_mps: [2.5, "Speed ≥ m/s"], min_seconds: [1.0, "For seconds"] },
  wrong_way: { min_seconds: [1.5, "For seconds"], cos_threshold: [-0.5, "cos ≤ (−1…0)"] },
  crowding: { max_count: [5, "People >"], min_seconds: [3.0, "For seconds"] },
  counterflow: { min_fraction: [0.3, "Share against ≥ (0–1)"], min_tracks: [4, "Moving people ≥"],
    min_against: [2, "Against ≥"], min_seconds: [3.0, "For seconds"] },
  density: { warn_ppm2: [2.0, "Warning ≥ people/m²"], critical_ppm2: [4.0, "Critical ≥ people/m²"],
    min_count: [3, "People ≥"], min_seconds: [5.0, "For seconds"] },
  queue_long: { max_length: [10, "Queue > people"], max_wait_s: [0, "or est. wait > s (0 = off)"], min_seconds: [10, "For seconds"] },
  dispersal: { min_people: [4, "Moving people ≥"], min_radial_mps: [1.5, "Outward ≥ m/s"], min_seconds: [1.0, "For seconds"] },
  convergence: { min_people: [4, "Moving people ≥"], min_radial_mps: [1.5, "Inward ≥ m/s"], min_seconds: [1.0, "For seconds"] },
  vehicle_in_zone: { min_seconds: [2.0, "Min. seconds inside"] },
  vehicle_stopped: { min_seconds: [60, "Stationary ≥ s"], window_s: [3, "Stillness window s"],
    still_m: [1.0, "Moved ≤ m (calibrated)"], still_box_frac: [0.15, "Moved ≤ box size (uncal., 0–1)"] },
};
/** Rules evaluated on vehicle tracks (need the site's `vehicles` detection switched on). */
export const VEHICLE_RULES = new Set(["vehicle_in_zone", "vehicle_stopped"]);
/** Rules that need a calibrated ground plane (metric speed, direction or area). */
export const METRIC_RULES = new Set(["running", "wrong_way", "counterflow", "density", "dispersal", "convergence"]);
export const STATUSES = ["new", "acknowledged", "escalated", "resolved", "dismissed"];
export const PRIORITIES = ["P1", "P2", "P3", "P4"];
export const DISP = {
  true_positive_actioned: "True positive — actioned", true_positive_no_action: "True positive — no action needed",
  false_positive_detection: "False positive — not a person", false_positive_rule: "False positive — rule not meaningful here",
  false_positive_calibration: "False positive — position/zone wrong", duplicate: "Duplicate", test_or_drill: "Test / drill",
  unclear: "Unclear — needs a second review",
};
export const HEALTH = ["OK", "WARMUP", "DEGRADED", "DOWN"];
/** Priority semantics used everywhere (chips, charts, legends). Bands come from the correlator's score. */
export const PRIO = {
  P1: { label: "Critical", hint: "Immediate operator attention" },
  P2: { label: "High", hint: "Review promptly" },
  P3: { label: "Medium", hint: "Review when possible" },
  P4: { label: "Low", hint: "Informational" },
};
export const STATUS_LABEL = { new: "New", acknowledged: "Acknowledged", escalated: "Escalated", resolved: "Resolved", dismissed: "Dismissed" };
export const OPEN_STATUSES = ["new", "acknowledged", "escalated"];
export function initials(name) {
  const w = String(name || "?").replace(/\(.*?\)/g, "").trim().split(/[\s._-]+/).filter(Boolean);
  return ((w[0]?.[0] || "?") + (w[1]?.[0] || "")).toUpperCase();
}
export function fmtCompact(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  const a = Math.abs(n);
  return a >= 1e6 ? (n / 1e6).toFixed(1) + "M" : a >= 1e4 ? Math.round(n / 1e3) + "k" : a >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n * 10) / 10);
}
export function fmtMs(ms) {
  if (typeof ms !== "number" || !isFinite(ms)) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}
export const CAMCOL = ["#3b82f6", "#a855f7", "#14b8a6", "#f59e0b", "#ec4899", "#84cc16", "#06b6d4", "#f43f5e"];

// ---------- misc ----------
export function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
export function throttle(fn, ms) {
  let last = 0, t = null;
  return (...a) => {
    const now = Date.now(), wait = last + ms - now;
    if (wait <= 0) { last = now; fn(...a); }
    else if (!t) t = setTimeout(() => { t = null; last = Date.now(); fn(...a); }, wait);
  };
}
export const ls = {
  get(k, d = null) { try { const v = localStorage.getItem("argonix." + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { if (v === null || v === undefined) localStorage.removeItem("argonix." + k); else localStorage.setItem("argonix." + k, v); } catch { /* storage unavailable */ } },
};
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export function slug(s) {
  return String(s || "").toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
}
export function isTyping(e) {
  const t = e.target;
  return t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
}
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
