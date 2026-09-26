// Analytics page: counting lines (in/out per minute, direction split), zone occupancy/dwell/direction, density per m²
// and queue length/wait (Little's law) with trend charts, crowd-flow events (counter-flow, dispersal, convergence),
// anomaly indicators (IsolationForest trajectory model + occupancy CUSUM; corroborating only), a zone congestion
// forecast (a labelled prediction, shown only when its backtest beats naive and seasonal-naive) and event hotspots.
//
// Endpoints (all GET, `alert.read`): /api/analytics/zones, /api/analytics/lines, /api/analytics/hotspots?window=,
// /api/analytics/anomaly, /api/analytics/forecast?horizon_min=, /api/baseline, /api/alerts?rule=counterflow|dispersal|convergence. The zones/lines routes
// 404 when flow analytics are off; every panel then explains how to enable its feature. In demo mode (or on request)
// the page falls back to clearly labelled sample data (fixtures.js) when the flow analytics are absent or empty.
import { api, optional } from "../api.js";
import { S, on, can, camName } from "../state.js";
import * as drawer from "../drawer.js";
import { h, s, icon, replace, enc, num, fmtAgo, fmtDur, RULE, OPEN_STATUSES } from "../util.js";
import { pageHeader, emptyState, skeleton, prioChip, statusPill, obsBadge, infBadge, estBadge, segmented } from "../ui.js";
import { sparkline, stackedBars, stackBar, meter, forecastChart } from "../charts.js";
import { hasEndpoint } from "../registry.js";
import { panelHead, sampleAllowed, sampleBanner, featureOff, code, colorOf } from "./common.js";
import { analyticsFixtures } from "./fixtures.js";

const FLOW_RULES = ["counterflow", "dispersal", "convergence"];
const DIRS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
const COL = { in: "var(--acc-text)", out: "var(--obs)", warn: "var(--warn)", err: "var(--err)", inf: "var(--inf)" };
const fin = v => typeof v === "number" && isFinite(v);
const hm = t => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function createAnalytics() {
  let shown = false, loading = false, timer = null, preview = false;
  let camFilter = "", win = "24h", heatCam = null, fcH = 30;
  let d = { zones: undefined, lines: undefined, hotspots: undefined, anomaly: undefined, baseline: undefined, flowAlerts: undefined, forecast: undefined };
  let sample = false;
  const names = new Map();
  const nm = id => names.get(id) || camName(id);

  // ------------------------------------------------------------------------------------------------ layout
  const subtitle = h("span");
  const camSel = h("select", { class: "sm", "aria-label": "Camera filter", onchange: () => { camFilter = camSel.value; renderAll(); } });
  const winSeg = segmented([["24h", "24 h", "Hotspots over the last 24 hours"], ["7d", "7 days", "Hotspots over the last 7 days"]], win,
    v => { win = v; loadHotspots(); }, { label: "Hotspot window" });
  const refreshBtn = h("button", { class: "sm", type: "button", title: "Refresh now", onclick: () => load(true) }, icon("refresh"), "Refresh");
  const bannerEl = h("div");
  const kpiEl = h("div", { class: "kpis" }, skeleton("kpi", 5));
  const linesBody = h("div", null, skeleton("card", 1));
  const dqBody = h("div", null, skeleton("card", 1));
  const zonesBody = h("div", null, skeleton("row", 3));
  const eventsBody = h("div", null, skeleton("row", 3));
  const anomBody = h("div", null, skeleton("row", 3));
  const hotBody = h("div", null, skeleton("card", 1));
  const fcBody = h("div", null, skeleton("card", 1));
  const fcSeg = segmented([[15, "15 min"], [30, "30 min"], [60, "60 min"]], fcH, v => { fcH = v; loadForecast(); }, { label: "Forecast horizon" });
  const reviewFlow = h("a", { class: "btnlink sm ghost", href: "#/review?rule=counterflow" }, "Review", icon("chevright"));
  // Line / density-queue / zone panels: hidden as a group when the flow analyzer is off (the banner explains it).
  const flowPanels = h("div", { class: "px-flow" },
    h("div", { class: "panel" }, panelHead("analytics", "Counting lines", "People crossing each line per minute, last 60 min, by direction"), linesBody),
    h("div", { class: "panel" }, panelHead("layers", "Density & queues", "Density per m² needs a calibrated camera; queue waits are Little's-law estimates"), dqBody),
    h("div", { class: "panel flush" }, panelHead("pin", "Zones · occupancy, dwell & direction", "Person tracks only; tracker ID switches split or merge stays"), zonesBody));

  const el = h("section", { class: "page px-page" },
    pageHeader({ title: "Analytics", subtitle,
      actions: [h("label", { class: "row small mute" }, "Camera", camSel), refreshBtn] }),
    bannerEl,
    kpiEl,
    flowPanels,
    h("div", { class: "panel" }, panelHead("clock", "Congestion forecast", "Zone occupancy / queue length ahead: a prediction, not an observation", fcSeg), fcBody),
    h("div", { class: "px-2col" },
      h("div", { class: "panel" }, panelHead("alert", "Crowd-flow events", "Counter-flow, dispersal and convergence cues (deterministic zone rules)", reviewFlow), eventsBody),
      h("div", { class: "panel" }, panelHead("spark", "Anomaly indicators", null, infBadge("corroborating only")), anomBody)),
    h("div", { class: "panel" }, panelHead("target", "Hotspots", "Where rules fired most: event frequency, not risk", winSeg), hotBody),
    h("div", { class: "provenance", style: { marginTop: "var(--s-4)" } }, obsBadge(), "counts, occupancy and crossings are observed person tracks.",
      estBadge("estimate"), "queue waits (Little's law) and density ranges carry stated uncertainty.",
      infBadge("Learned models"), "only corroborate; they never raise an alert on their own (capped at P3).",
      infBadge("Forecasts"), "are predictions from past occupancy, shown only when they beat simple references on this site's own history."));

  // ------------------------------------------------------------------------------------------------ data
  const offAt = new Map();                                   // path -> time it last answered "feature off" (404)
  async function get(path, feature, force = false) {
    const key = path.split("?")[0];
    if (hasEndpoint(key) === false) return null;             // route not served at all
    if (!force && Date.now() - (offAt.get(key) || 0) < 60000) return null;   // feature off: re-check once a minute
    try {
      const r = await optional(path, feature, { silent: true });
      if (r === null) offAt.set(key, Date.now()); else offAt.delete(key);
      return r;
    } catch (e) { return e?.kind === "forbidden" ? { forbidden: true } : { error: e?.message || "request failed" }; }
  }

  async function loadFlowAlerts() {
    const lists = await Promise.all(FLOW_RULES.map(r => api(`/api/alerts?rule=${enc(r)}&limit=60`, { silent: true }).catch(() => null)));
    if (lists.every(l => l === null)) return null;
    const seen = new Map();
    for (const l of lists) for (const a of l || []) seen.set(a.id, a);
    return [...seen.values()].sort((a, b) => b.updated_at - a.updated_at);
  }

  async function loadHotspots() {
    if (sample) { d.hotspots = analyticsFixtures().hotspots; d.hotspots.window = win; renderHot(); return; }
    replace(hotBody, skeleton("card", 1));
    d.hotspots = await get(`/api/analytics/hotspots?window=${enc(win)}`, "hotspots", true);
    if (shown) renderHot();
  }

  async function getForecast(force) {
    const r = await get(`/api/analytics/forecast?horizon_min=${fcH}`, "forecast", force);
    if (r && typeof r === "object") delete r.generated_at;   // keeps the change signature stable between polls
    return r;
  }

  async function loadForecast() {
    if (sample) { renderForecast(); return; }
    replace(fcBody, skeleton("card", 1));
    d.forecast = await getForecast(true);
    if (shown) renderForecast();
  }

  async function load(force = false) {
    if (loading && !force) return;
    loading = true;
    const [zones, lines, hotspots, anomaly, baseline, flowAlerts, forecast] = await Promise.all([
      get("/api/analytics/zones", "analytics", force), get("/api/analytics/lines", "analytics", force),
      get(`/api/analytics/hotspots?window=${enc(win)}`, "hotspots", force), get("/api/analytics/anomaly", "anomaly", force),
      get("/api/baseline", "baseline", force), loadFlowAlerts(), getForecast(force)]);
    loading = false;
    d = { zones, lines, hotspots, anomaly, baseline, flowAlerts, forecast };
    const nZones = (zones?.cameras || []).reduce((a, c) => a + (c.zones || []).length, 0);
    const nLines = (lines?.cameras || []).reduce((a, c) => a + (c.lines || []).length, 0);
    sample = !nZones && !nLines && sampleAllowed(preview);
    if (sample) {
      const f = analyticsFixtures();
      d = { zones: f.zones, lines: f.lines, hotspots: { ...f.hotspots, window: win }, anomaly: f.anomaly, baseline: f.baseline, flowAlerts: f.flowAlerts, forecast: null };
    }
    names.clear();
    for (const src of [d.zones, d.lines, d.hotspots]) for (const c of src?.cameras || []) if (c.name) names.set(c.camera, c.name);
    const sig = JSON.stringify([d, sample]);
    if (!force && sig === lastSig) return;                  // nothing changed: keep the DOM (open selects, hover)
    lastSig = sig;
    if (shown) renderAll();
  }
  let lastSig = "";

  // ------------------------------------------------------------------------------------------------ helpers
  const keep = cam => !camFilter || cam === camFilter;
  const camDot = id => h("span", { class: "px-dot", style: { background: colorOf(id) }, "aria-hidden": "true" });
  const lineCams = () => (d.lines?.cameras || []).filter(c => keep(c.camera) && (c.lines || []).length);
  const zoneRows = () => (d.zones?.cameras || []).filter(c => keep(c.camera)).flatMap(c => (c.zones || []).map(z => ({ cam: c, z })));
  const flowList = () => (Array.isArray(d.flowAlerts) ? d.flowAlerts : []).filter(a => !camFilter || (a.cameras || []).includes(camFilter));

  function axis(rows) {
    if (!rows?.length) return null;
    const pick = [0, Math.floor(rows.length / 2), rows.length - 1];
    return h("div", { class: "trend-axis", "aria-hidden": "true" }, pick.map(i => h("span", null, hm(rows[i].t))));
  }

  /** Line trend with an optional dashed threshold (same y scale). */
  function trend(rows, key, { color = "var(--acc-text)", thr = null, label, unit = "" } = {}) {
    const vals = (rows || []).map(r => (fin(r[key]) ? r[key] : null));
    const nums = vals.filter(v => v !== null);
    if (!nums.length) return h("div", { class: "px-nodata small mute" }, "No samples yet in this window.");
    const max = Math.max(1e-6, ...nums, fin(thr) ? thr : 0) * 1.12;
    const W = 240, H = 56;
    const svg = sparkline(vals, { width: W, height: H, min: 0, max, color, label, dot: true });
    if (fin(thr)) {
      const y = H - 2 - (thr / max) * (H - 4);
      svg.appendChild(s("line", { x1: 0, x2: W, y1: y, y2: y, "vector-effect": "non-scaling-stroke",
        style: { stroke: "var(--warn)", strokeWidth: 1, strokeDasharray: "4 3" } }));
    }
    return h("div", { class: "px-chart" },
      h("div", { class: "px-chart-top" }, h("span", { class: "small mute" }, label), h("span", { class: "grow" }),
        h("span", { class: "small tnum" }, `max ${num(Math.max(...nums), 2)}${unit}`),
        fin(thr) ? h("span", { class: "small c-warn" }, ` · limit ${thr}${unit}`) : null),
      svg, axis(rows));
  }

  /** 8-bin direction rose (share of track-seconds per heading) with dominant (solid) and permitted (dashed) arrows. */
  function rose(flow, size = 64) {
    const hist = flow?.hist || [];
    const tot = hist.reduce((a, v) => a + v, 0);
    const c = size / 2, R = size / 2 - 5, mx = Math.max(1e-9, ...hist);
    const pt = (deg, r) => [c + r * Math.cos(deg * Math.PI / 180), c - r * Math.sin(deg * Math.PI / 180)];
    const petals = hist.map((v, i) => {
      if (!v) return null;
      const a = i * 45, r = Math.max(2, (v / mx) * R);
      const [x1, y1] = pt(a - 17, r), [x2, y2] = pt(a + 17, r);
      return s("path", { d: `M${c},${c}L${x1},${y1}A${r},${r} 0 0 0 ${x2},${y2}Z`, style: { fill: "var(--acc-text)", opacity: 0.25 + 0.6 * v / mx } },
        s("title", null, `${DIRS[i]}: ${Math.round(v * 100)}% of track-seconds`));
    });
    const arrow = (deg, dashed) => { const [x, y] = pt(deg, R); return s("line", { x1: c, y1: c, x2: x, y2: y,
      style: { stroke: dashed ? "var(--ok)" : "var(--text)", strokeWidth: 1.6, strokeDasharray: dashed ? "3 2" : null, strokeLinecap: "round" } }); };
    return s("svg", { class: "px-rose", viewBox: `0 0 ${size} ${size}`, width: size, height: size, role: "img",
      "aria-label": tot ? `Direction histogram; dominant ${fin(flow.dominant_deg) ? DIRS[Math.round(flow.dominant_deg / 45) % 8] : "unclear"}` : "No direction data" },
      s("circle", { cx: c, cy: c, r: R, style: { fill: "none", stroke: "var(--line-strong)", strokeWidth: 1 } }),
      s("text", { x: c, y: 7, "text-anchor": "middle", style: { fill: "var(--faint)", fontSize: "7px" } }, "N"),
      petals, fin(flow?.permitted_deg) ? arrow(flow.permitted_deg, true) : null,
      fin(flow?.dominant_deg) && flow.concentration >= 0.3 ? arrow(flow.dominant_deg, false) : null);
  }

  // ------------------------------------------------------------------------------------------------ render
  function renderFilter() {
    const ids = new Set(S.cams.map(c => c.id));
    for (const src of [d.zones, d.lines, d.hotspots]) for (const c of src?.cameras || []) ids.add(c.camera);
    const cur = camFilter;
    replace(camSel, h("option", { value: "" }, "All cameras"), [...ids].map(id => h("option", { value: id, selected: id === cur }, nm(id))));
    camSel.value = ids.has(cur) ? cur : "";
    camFilter = camSel.value;
  }

  function renderBanner() {
    const parts = [];
    if (sample) parts.push(sampleBanner(preview ? () => { preview = false; load(true); } : null));
    else if (d.zones === null && d.lines === null) {
      parts.push(featureOff({ ic: "analytics", title: "Flow analytics are off on this server",
        text: "Line counts, zone occupancy, dwell, direction, density and queue statistics come from the flow analyzer.",
        steps: [["Set ", code('"analytics": {"enabled": true}'), " in the site config (it is on by default) and restart Argus."],
          ["Draw zones and counting lines on a camera (Cameras → Zones & masks)."],
          ["Calibrate the camera for density per m² and metric flow rules (Cameras → Calibration)."]],
        onPreview: () => { preview = true; load(true); } }));
    } else if (d.zones?.forbidden || d.lines?.forbidden) {
      parts.push(h("div", { class: "banner err" }, icon("alert"), "Your role cannot read analytics (needs alert.read)."));
    }
    const warns = [...new Set([...(d.zones?.cameras || []), ...(d.lines?.cameras || [])].filter(c => keep(c.camera)).flatMap(c => (c.warnings || []).map(w => `${nm(c.camera)}: ${w}`)))];
    if (!sample && warns.length) parts.push(h("div", { class: "banner" }, icon("alert"), h("div", null, h("b", null, "Configuration warnings. "), warns.join(" · "))));
    replace(bannerEl, parts);
  }

  function kpi({ ic, label, value, unit, sub, foot, title, href }) {
    return h(href ? "a" : "div", { class: "kpi", href, title: title || null },
      h("div", { class: "kpi-l" }, icon(ic), label),
      h("div", { class: "kpi-v" }, value, unit ? h("small", null, unit) : null),
      sub ? h("div", { class: "kpi-sub" }, sub) : null, foot ? h("div", { class: "kpi-foot" }, foot) : null);
  }

  function renderKpis() {
    const lc = lineCams().flatMap(c => c.lines);
    const tin = lc.reduce((a, l) => a + (l.counts_60m?.in || 0), 0), tout = lc.reduce((a, l) => a + (l.counts_60m?.out || 0), 0);
    const zr = zoneRows();
    const occ = zr.reduce((a, r) => a + (r.z.occupancy || 0), 0);
    const dens = zr.filter(r => r.z.density && fin(r.z.density.ppm2)).sort((a, b) => b.z.density.ppm2 - a.z.density.ppm2)[0];
    const qs = zr.filter(r => r.z.queue).sort((a, b) => (b.z.queue.length || 0) - (a.z.queue.length || 0));
    const q = qs[0];
    const day = Date.now() / 1000 - 86400;
    const fl = flowList().filter(a => a.created_at >= day);
    const lvlTag = lv => h("span", { class: ["tag", lv === "critical" ? "err" : lv === "warning" ? "warn" : "ok"] }, lv);
    const dash = "—";
    replace(kpiEl,
      kpi({ ic: "analytics", label: "Line crossings · 60 min", value: lc.length ? tin + tout : dash,
        sub: lc.length ? [h("span", null, `in ${tin} · out ${tout}`), obsBadge("observed")] : h("span", null, "No counting lines"),
        foot: lc.length ? stackBar([{ value: tin, color: COL.in, label: "in" }, { value: tout, color: COL.out, label: "out" }], { label: "Direction split, last 60 min" }) : null }),
      kpi({ ic: "user", label: "People in zones now", value: zr.length ? occ : dash,
        sub: zr.length ? [h("span", null, `${zr.length} zone${zr.length === 1 ? "" : "s"}`), obsBadge("observed")] : h("span", null, "No zones reporting"),
        title: "Person tracks currently inside analytics zones (a person inside two overlapping zones counts twice)." }),
      kpi({ ic: "layers", label: "Peak density now", value: dens ? num(dens.z.density.ppm2, 2) : dash, unit: dens ? "/m²" : null,
        sub: dens ? [h("span", null, `${dens.z.name} · ${nm(dens.cam.camera)}`), lvlTag(dens.z.density.level)] : h("span", null, "Needs a calibrated camera with a zone"),
        title: dens ? `Area ${num(dens.z.density.area_m2, 1)} m² on the calibrated ground plane (calibration error ${dens.z.density.calib_error_m ?? "?"} m)` : null }),
      kpi({ ic: "clock", label: "Longest queue now", value: q ? q.z.queue.length : dash, unit: q ? "people" : null,
        sub: q ? [h("span", null, q.z.name), fin(q.z.queue.wait_est_s) ? h("span", null, `~${fmtDur(q.z.queue.wait_est_s)} wait`) : h("span", null, "no wait estimate yet"),
          fin(q.z.queue.wait_est_s) ? estBadge("Little's law") : null] : h("span", null, "No queue zones (zone type \"queue\")") }),
      kpi({ ic: "alert", label: "Crowd-flow events · 24 h", value: Array.isArray(d.flowAlerts) ? fl.length : dash,
        sub: Array.isArray(d.flowAlerts) ? [h("span", null, FLOW_RULES.map(r => `${fl.filter(a => (a.events || []).some(e => e.rule === r)).length} ${r}`).join(" · "))] : h("span", null, "Alerts unavailable"),
        href: sample ? null : "#/review?rule=counterflow" }));
  }

  function renderLines() {
    if (d.lines === undefined) return;
    if (!d.lines || d.lines.error) {
      replace(linesBody, featureOff({ ic: "analytics", compact: true, title: d.lines?.error ? "Could not load counting lines" : "Counting lines unavailable",
        text: d.lines?.error || "The flow analyzer is off, so nothing is counted.", steps: d.lines?.error ? [] : [["Enable ", code('"analytics": {"enabled": true}'), " in the site config."]] }));
      return;
    }
    const cams = lineCams();
    if (!cams.length) {
      const first = S.cams[0];
      replace(linesBody, featureOff({ ic: "analytics", compact: true, title: "No counting lines yet",
        text: "A counting line counts people crossing it in each direction. It works without calibration.",
        steps: [["Open a camera's zone editor and choose the line tool; drag from one end of the line to the other."],
          ["\"In\" is the crossing towards the right-hand side when looking from the first point to the second; use \"Flip in/out\" to swap."]],
        action: first && can("zone.edit") ? h("a", { class: "btnlink sm primary", href: `#/cameras/${enc(first.id)}/zones` }, icon("plus"), "Draw a line") : null }));
      return;
    }
    replace(linesBody, h("div", { class: "px-lines" }, cams.flatMap(c => c.lines.map(l => {
      const tot = (l.counts_60m?.in || 0) + (l.counts_60m?.out || 0);
      const inPct = tot ? Math.round(l.counts_60m.in / tot * 100) : null;
      const rows = l.series || [];
      return h("div", { class: "px-card" },
        h("div", { class: "px-card-h" }, camDot(c.camera), h("b", null, l.name), h("span", { class: "small mute" }, nm(c.camera)),
          h("span", { class: "grow" }), l.space === "ground_m" ? h("span", { class: "tag" }, "ground m") : h("span", { class: "tag" }, "image px")),
        h("div", { class: "px-io-nums" },
          h("div", null, h("div", { class: "px-big tnum", style: { color: COL.in } }, l.counts_60m?.in ?? 0), h("div", { class: "xs mute" }, `${l.labels?.in || "in"} · 60 min`)),
          h("div", null, h("div", { class: "px-big tnum", style: { color: COL.out } }, l.counts_60m?.out ?? 0), h("div", { class: "xs mute" }, `${l.labels?.out || "out"} · 60 min`)),
          h("div", null, h("div", { class: "px-big tnum" }, (l.counts?.in || 0) + (l.counts?.out || 0)), h("div", { class: "xs mute" }, "total since start"))),
        stackBar([{ value: l.counts_60m?.in || 0, color: COL.in, label: l.labels?.in || "in" }, { value: l.counts_60m?.out || 0, color: COL.out, label: l.labels?.out || "out" }],
          { label: `Direction split: ${inPct ?? 0}% ${l.labels?.in || "in"}` }),
        h("div", { class: "legend", style: { margin: ".43rem 0" } },
          h("span", null, h("i", { style: { background: COL.in } }), `${l.labels?.in || "in"}${inPct !== null ? ` ${inPct}%` : ""}`),
          h("span", null, h("i", { style: { background: COL.out } }), `${l.labels?.out || "out"}${inPct !== null ? ` ${100 - inPct}%` : ""}`)),
        h("div", { class: "px-io" }, stackedBars(rows, ["in", "out"], { width: 240, height: 56, colors: COL, label: `Crossings per minute on ${l.name}`,
          titles: rows.map(r => `${hm(r.t)}: ${r.in} ${l.labels?.in || "in"}, ${r.out} ${l.labels?.out || "out"}`) })),
        axis(rows));
    }))),
    h("div", { class: "note" }, d.lines.direction_rule ? `Direction convention: ${d.lines.direction_rule}.` : "", " Counts are person tracks crossing the segment; tracker ID switches can double-count."));
  }

  function renderDQ() {
    if (d.zones === undefined) return;
    if (!d.zones || d.zones.error) {
      replace(dqBody, featureOff({ ic: "layers", compact: true, title: d.zones?.error ? "Could not load zone statistics" : "Zone statistics unavailable",
        text: d.zones?.error || "The flow analyzer is off.", steps: d.zones?.error ? [] : [["Enable ", code('"analytics": {"enabled": true}'), " in the site config."]] }));
      return;
    }
    const zr = zoneRows();
    const dens = zr.filter(r => r.z.density);
    const queues = zr.filter(r => r.z.queue);
    const uncal = zr.filter(r => !r.z.density).length;
    const cards = [];
    for (const { cam, z } of queues) {
      const q = z.queue;
      cards.push(h("div", { class: "px-card" },
        h("div", { class: "px-card-h" }, camDot(cam.camera), h("b", null, z.name), h("span", { class: "tag" }, "queue"), h("span", { class: "small mute" }, nm(cam.camera))),
        h("div", { class: "px-io-nums" },
          h("div", null, h("div", { class: "px-big tnum" }, q.length), h("div", { class: "xs mute" }, `in queue now · 60 s mean ${num(q.length_mean_60s, 1)}`)),
          h("div", { title: q.note || "" }, h("div", { class: "px-big tnum" }, fin(q.wait_est_s) ? fmtDur(q.wait_est_s) : "—"),
            h("div", { class: "xs mute row", style: { gap: ".3rem" } }, "est. wait", estBadge("Little's law"))),
          h("div", null, h("div", { class: "px-big tnum" }, fin(q.measured_wait_p50_s) ? fmtDur(q.measured_wait_p50_s) : "—"),
            h("div", { class: "xs mute row", style: { gap: ".3rem" } }, `measured median (n=${q.measured_waits_n || 0})`, obsBadge("measured")))),
        h("div", { class: "small mute" }, `Service ${fin(q.service_rate_per_min) ? `${num(q.service_rate_per_min, 1)} people/min` : "not observed yet"} · `,
          `${q.departures_window} departures in ${fmtDur(q.window_s)} · via ${q.method}`,
          q.steady_state === false ? h("span", { class: "c-warn" }, " · not in steady state: the Little's-law estimate is unreliable") : q.steady_state ? " · steady state" : ""),
        h("div", { class: "px-trends" },
          trend(z.series, "occ_mean", { label: "Queue length (mean per minute)", unit: "", color: "var(--acc-text)", thr: null }),
          trend(z.series, "wait_est_s", { label: "Estimated wait (s)", unit: " s", color: "var(--inf)" }))));
    }
    for (const { cam, z } of dens) {
      const dd = z.density;
      const lv = dd.level;
      cards.push(h("div", { class: "px-card" },
        h("div", { class: "px-card-h" }, camDot(cam.camera), h("b", null, z.name), h("span", { class: "small mute" }, nm(cam.camera)), h("span", { class: "grow" }),
          h("span", { class: ["tag", lv === "critical" ? "err" : lv === "warning" ? "warn" : "ok"] }, lv)),
        h("div", { class: "px-io-nums" },
          h("div", null, h("div", { class: "px-big tnum" }, num(dd.ppm2, 2), h("small", null, " /m²")), h("div", { class: "xs mute" }, "people per m² now")),
          h("div", null, h("div", { class: "px-big tnum" }, z.occupancy), h("div", { class: "xs mute" }, `people in ${num(dd.area_m2, 1)} m²`)),
          h("div", null, h("div", { class: "px-big tnum" }, dd.thresholds?.warn_ppm2 ?? "—", h("small", null, " /m²")),
            h("div", { class: "xs mute" }, dd.rule_enabled ? "warning limit (rule on)" : "warning limit (rule off)"))),
        h("div", { class: "px-trends" },
          trend(z.series, "density", { label: "Density (people/m², mean per minute)", unit: "", color: "var(--acc-text)", thr: dd.thresholds?.warn_ppm2 }),
          trend(z.series, "occ_mean", { label: "Occupancy (mean per minute)", unit: "", color: "var(--obs)" })),
        h("div", { class: "note" }, `Area on the calibrated ground plane; calibration error ${dd.calib_error_m ?? "unknown"} m widens the density range.`)));
    }
    if (!cards.length) {
      replace(dqBody, featureOff({ ic: "layers", compact: true, title: zr.length ? "No density or queue zones" : "No zones yet",
        text: zr.length ? `${uncal} zone${uncal === 1 ? " is" : "s are"} on uncalibrated cameras (counts only).` : "Density and queue statistics are computed per zone.",
        steps: [["Density per m²: calibrate the camera (Cameras → Calibration); every zone on it then reports people/m²."],
          ["Queues: set a zone's type to \"queue\" (or enable its queue_long rule); add a service line for measured waits."]] }));
      return;
    }
    replace(dqBody, h("div", { class: "px-cards" }, cards),
      uncal ? h("div", { class: "note" }, `${uncal} zone${uncal === 1 ? "" : "s"} on uncalibrated cameras report counts only (no area, density or m/s).`) : null);
  }

  function renderZones() {
    if (d.zones === undefined) return;
    const zr = zoneRows();
    if (!d.zones || d.zones.error || !zr.length) {
      replace(zonesBody, h("div", { style: { padding: "0 var(--s-4) var(--s-4)" } }, featureOff({ ic: "pin", compact: true,
        title: !d.zones ? "Zone statistics unavailable" : "No zones reporting",
        text: !d.zones ? "Enable the flow analyzer to see occupancy, dwell and direction per zone." : "Draw a zone on a camera (Cameras → Zones & masks) to see its occupancy, dwell times and dominant direction." })));
      return;
    }
    replace(zonesBody, h("div", { class: "table-wrap" }, h("table", null,
      h("thead", null, h("tr", null, ["Zone", "Now", "In / out · 60 min", "Dwell p50 / p90", "Dwell distribution", "Direction", "Density"].map(t => h("th", null, t)))),
      h("tbody", null, zr.map(({ cam, z }) => {
        const dw = z.dwell || {};
        const bins = (dw.counts || []).map((v, i) => ({ n: v, lab: i < (dw.bins_s || []).length - 1 ? `${dw.bins_s[i]}–${dw.bins_s[i + 1]} s` : `≥ ${dw.bins_s?.[i] ?? "?"} s` }));
        const fl = z.flow || {};
        const domIdx = fin(fl.dominant_deg) ? Math.round(fl.dominant_deg / 45) % 8 : null;
        return h("tr", null,
          h("td", null, h("div", { class: "row", style: { gap: ".43rem" } }, camDot(cam.camera), h("b", null, z.name), z.kind === "queue" ? h("span", { class: "tag" }, "queue") : null),
            h("div", { class: "xs mute" }, nm(cam.camera), z.rules?.length ? ` · rules: ${z.rules.map(r => RULE[r] || r).join(", ")}` : "")),
          h("td", { class: "tnum" }, h("b", null, z.occupancy ?? 0)),
          h("td", { class: "tnum" }, `${z.entries_60m ?? 0} / ${z.exits_60m ?? 0}`, h("div", { class: "xs mute" }, `total ${z.entries_total ?? 0} / ${z.exits_total ?? 0}`)),
          h("td", { class: "tnum" }, fin(dw.p50_s) ? `${fmtDur(dw.p50_s)} / ${fmtDur(dw.p90_s)}` : "—", h("div", { class: "xs mute" }, `${dw.n || 0} completed stays`)),
          h("td", null, bins.some(b => b.n) ? h("div", { class: "px-mini" }, stackedBars(bins.map(b => ({ n: b.n })), ["n"], { width: 96, height: 28, colors: { n: "var(--obs)" },
            label: "Dwell-time histogram", titles: bins.map(b => `${b.lab}: ${b.n}`) })) : h("span", { class: "mute" }, "—")),
          h("td", null, h("div", { class: "row", style: { gap: ".5rem", flexWrap: "nowrap" } }, rose(fl, 48),
            h("div", { class: "xs" }, domIdx !== null && fl.concentration >= 0.3 ? h("b", null, `→ ${DIRS[domIdx]}`) : h("span", { class: "mute" }, "mixed"),
              h("div", { class: "mute" }, `R ${num(fl.concentration, 2)}`), fin(fl.permitted_deg) ? h("div", { class: "c-ok" }, `permitted ${DIRS[Math.round(fl.permitted_deg / 45) % 8]}`) : null))),
          h("td", { class: "tnum" }, z.density ? [h("b", null, `${num(z.density.ppm2, 2)} /m²`), h("div", { class: "xs mute" }, `${num(z.density.area_m2, 1)} m²`)]
            : h("span", { class: "xs mute", title: z.density_note || "" }, "uncalibrated")));
      })))),
    h("div", { class: "note", style: { padding: "0 var(--s-4) var(--s-3)" } },
      "Direction: share of moving track-seconds per heading (", h("b", null, "N = up"), " on the map for calibrated cameras, on screen otherwise). ",
      "Solid arrow = learned dominant flow (shown when concentration R ≥ 0.3); dashed = configured permitted direction."));
  }

  const FC_TAG = { ok: ["inf", "forecast shown"], no_skill: ["", "insufficient skill"], insufficient_history: ["", "insufficient history"],
    no_data: ["", "no data yet"], stale: ["warn", "no recent data"] };
  const FC_ORDER = { ok: 0, no_skill: 1, insufficient_history: 2, stale: 3, no_data: 4 };

  function renderForecast() {
    if (d.forecast === undefined) return;
    const fc = d.forecast;
    if (sample) {
      replace(fcBody, featureOff({ ic: "clock", compact: true, title: "No forecast in sample mode",
        text: "A forecast is shown only for real zones with enough history: it must beat the last value and the usual-for-this-hour value in a backtest on this site's own data." }));
      return;
    }
    if (!fc || fc.error || fc.forbidden) {
      replace(fcBody, featureOff({ ic: "clock", compact: true, title: fc?.error ? "Could not load the forecast" : "Congestion forecast is off",
        text: fc?.error || "The zone congestion forecast reads the occupancy history that the seasonal baseline stores every minute.",
        steps: fc?.error ? [] : [["Set ", code('"congestion_forecast": {"enabled": true}'), " in the site config (on by default) and restart."]] }));
      return;
    }
    const zones = (fc.zones || []).filter(z => keep(z.camera))
      .sort((a, b) => (FC_ORDER[a.status] ?? 9) - (FC_ORDER[b.status] ?? 9) || String(a.name).localeCompare(String(b.name)));
    if (!zones.length) {
      replace(fcBody, emptyState({ icon: "clock", compact: true, title: "No zones to forecast", text: "Draw a zone on a camera; its occupancy history builds up minute by minute." }));
      return;
    }
    const nUseful = zones.filter(z => z.useful).length;
    replace(fcBody,
      h("div", { class: "small mute", style: { marginBottom: ".5rem" } },
        `${nUseful} of ${zones.length} zone${zones.length === 1 ? "" : "s"} with a forecast that beats both references at ${fc.horizon_min} min. `,
        fc.emit_events ? "Congestion notes are also recorded as low-priority informational events." : "Congestion notes appear here only (events are off)."),
      h("div", { class: "px-cards" }, zones.map(z => forecastCard(z, fc))),
      h("div", { class: "note" }, fc.note || "", " Model: ", fc.model || "—", "."));
  }

  function forecastCard(z, fc) {
    const [tcls, tlab] = FC_TAG[z.status] || ["", z.status];
    const bt = z.backtest?.horizons || [];
    const cong = z.congestion;
    const limit = z.limit?.value;
    const unit = z.kind === "queue" ? "in the queue" : "people";
    let note = null;
    if (cong?.observed_now_above) {
      note = h("div", { class: "px-fc-cong" }, icon("alert"), h("div", null, h("b", null, "Above the limit now "), obsBadge("observed"),
        h("div", { class: "xs mute" }, `${num(z.now, 1)} ${unit} (last minute's mean) vs the ${z.limit.label} of ${limit}. The deterministic rule reports this, not the forecast.`)));
    } else if (cong?.expected) {
      note = h("div", { class: "px-fc-cong" }, icon("clock"), h("div", null, h("b", null, `Congestion expected in ~${Math.max(1, Math.round(cong.in_min))} min `), infBadge("prediction"),
        h("div", { class: "xs mute" }, `The upper ${Math.round((fc.coverage || 0.8) * 100)}% bound (${num(cong.upper, 1)}) crosses the ${z.limit.label} of ${limit}`,
          cong.mean_in_min != null ? `; the expected value crosses it in ~${Math.max(1, Math.round(cong.mean_in_min))} min.` : "; the expected value stays below it.")));
    }
    const best = r => r.mae != null && r.mae < Math.min(r.mae_naive ?? Infinity, r.mae_seasonal ?? Infinity);
    return h("div", { class: "px-card" },
      h("div", { class: "px-card-h" }, camDot(z.camera), h("b", null, z.name), h("span", { class: "small mute" }, z.camera_name || nm(z.camera)),
        z.kind === "queue" ? h("span", { class: "tag" }, "queue") : null, h("span", { class: "grow" }), h("span", { class: ["tag", tcls] }, tlab)),
      h("div", { class: "xs mute" }, `${z.metric} · now ${fin(z.now) ? num(z.now, 1) : "—"}`,
        limit != null ? ` · ${z.limit.label} ${limit}` : " · no limit (enable crowding or queue_long on the zone)"),
      z.history?.length ? forecastChart({ history: z.history, forecast: z.forecast, asOf: z.as_of, limit, showForecast: !!z.useful,
        label: `${z.name}: observed ${z.metric} over the last hour${z.useful ? `, and the forecast for the next ${fc.horizon_min} min (prediction, not observation)` : ""}` })
        : h("div", { class: "px-nodata small mute" }, "No occupancy samples yet."),
      note,
      h("div", { class: "px-fc-msg" }, z.useful ? [infBadge("forecast"), " ", z.message] : z.message),
      bt.length ? h("details", null, h("summary", { class: "xs mute" }, `Backtest · ${z.backtest.n_origins} origins over ${z.backtest.backtest_days} days (MAE, lower is better)`),
        h("table", { class: "px-fc-bt" },
          h("thead", null, h("tr", null, ["Ahead", "Forecast", "Last value", "Usual for hour", "Held out", "In 80% band"].map(x => h("th", null, x)))),
          h("tbody", null, bt.map(r => h("tr", null,
            h("td", null, `${r.h_min} min`),
            h("td", { class: best(r) ? "win" : null }, num(r.mae, 2)),
            h("td", null, num(r.mae_naive, 2)), h("td", null, r.mae_seasonal != null ? num(r.mae_seasonal, 2) : "n/a"),
            h("td", null, String(r.n_eval)),
            h("td", null, r.interval?.coverage_eval != null ? `${Math.round(r.interval.coverage_eval * 100)}%` : "—"))))),
        h("div", { class: "xs mute" }, `${z.backtest.split}. Measured on this zone's own stored history.`)) : null);
  }

  function renderEvents() {
    if (d.flowAlerts === undefined) return;
    if (d.flowAlerts === null) { replace(eventsBody, emptyState({ icon: "alert", compact: true, title: "Alerts unavailable", text: "Could not read alerts from the server." })); return; }
    const list = flowList().slice(0, 12);
    reviewFlow.hidden = sample;
    if (!list.length) {
      replace(eventsBody, featureOff({ ic: "check", compact: true, title: "No crowd-flow events",
        text: "These rules only fire in zones where an operator switched them on, on calibrated cameras (metric speeds and directions).",
        steps: [["Enable counterflow, dispersal or convergence on a zone (Cameras → Zones & masks → rules)."],
          ["Counter-flow compares against the zone's permitted direction, or its learned dominant flow once enough tracks were seen."]] }));
      return;
    }
    const ids = list.filter(a => !a.sample).map(a => a.id);
    replace(eventsBody, h("div", { class: "feed" }, list.map(a => {
      const ev = [...(a.events || [])].reverse().find(e => FLOW_RULES.includes(e.rule)) || a.events?.[0] || {};
      const open = () => {
        if (a.sample) return;
        let k = ids.indexOf(a.id);
        drawer.setNavigator({ next: () => { if (k < ids.length - 1) { drawer.openAlert(ids[++k]); return true; } return false; },
          prev: () => { if (k > 0) { drawer.openAlert(ids[--k]); return true; } return false; } });
        drawer.openAlert(a.id);
      };
      return h(a.sample ? "div" : "button", a.sample ? { class: "feed-i", title: "Sample event (not a real alert)" } : { class: "feed-i", type: "button", onclick: open, title: "Open alert" },
        prioChip(a.priority),
        h("div", { style: { minWidth: 0 } },
          h("div", { class: "feed-t" }, RULE[ev.rule] || a.title),
          ev.observation ? h("div", { class: "feed-o" }, ev.observation) : null,
          h("div", { class: "feed-m" }, statusPill(a.status), h("span", null, `${nm(ev.camera_id || a.cameras?.[0])} · ${ev.zone_name || ev.zone_id || "—"}`),
            fin(ev.confidence) ? h("span", null, `· conf ${num(ev.confidence, 2)} (heuristic)`) : null)),
        h("div", { class: "feed-time" }, fmtAgo(a.updated_at)));
    })), h("div", { class: "note" }, obsBadge("motion cue"), " These describe how people moved, not why. ",
      `${list.filter(a => OPEN_STATUSES.includes(a.status)).length} of ${list.length} still open.`));
  }

  function renderAnomaly() {
    if (d.anomaly === undefined) return;
    const parts = [h("div", { class: "banner info" }, icon("info"), h("div", null,
      h("b", null, "Corroborating only. "), "These indicators never raise an alert on their own: model events are capped at P3 until a deterministic rule agrees, ",
      "and a CUSUM shift says occupancy changed, not why."))];
    const an = d.anomaly;
    parts.push(h("h4", null, "Unusual trajectories · IsolationForest ", infBadge("learned")));
    if (!an || an.error || an.forbidden) parts.push(emptyState({ icon: "spark", compact: true, title: "Anomaly model status unavailable", text: an?.error || "The endpoint is not available on this server." }));
    else if (!an.enabled) {
      parts.push(featureOff({ ic: "spark", compact: true, title: "Trajectory-anomaly model is off",
        text: "Off by default. When on, each camera learns its own typical trajectories (speed, straightness, dwell, turns) and scores new tracks against them.",
        steps: [["Set ", code('"anomaly": {"enabled": true}'), " in the site config and restart."],
          [`It needs about ${250} completed tracks per camera before it scores anything.`]] }));
    } else {
      const cams = (an.cameras || []).filter(c => keep(c.camera));
      if (!cams.length) parts.push(emptyState({ icon: "spark", compact: true, title: "No cameras scored yet", text: "The model starts collecting once person tracks complete." }));
      parts.push(h("div", { class: "px-anom" }, cams.map(c => {
        const top = (c.recent || []).slice().sort((a, b) => b.percentile - a.percentile).slice(0, 3);
        return h("div", { class: "px-anom-row" },
          h("div", { class: "row", style: { gap: ".43rem" } }, camDot(c.camera), h("b", null, nm(c.camera)), h("span", { class: "grow" }),
            c.trained ? h("span", { class: "tag ok" }, `trained v${c.version}`) : h("span", { class: "tag" }, c.fitting ? "fitting…" : "learning")),
          c.trained ? h("div", { class: "xs mute" }, `${c.scored} tracks scored · ${c.emitted} above the ${Math.round((an.params?.emit_percentile ?? 0.99) * 100)}th percentile · trained on ${c.n_train} (${c.speed_unit || "?"})`)
            : [h("div", { class: "xs mute" }, `${c.n_window} / ${c.min_train} tracks collected before the first fit`), meter(c.n_window / Math.max(1, c.min_train), { color: "var(--inf)", label: "Training progress" })],
          top.length ? h("div", { class: "px-anom-top" }, top.map(r => h("div", { class: "row small", style: { gap: ".43rem" } },
            h("span", { class: ["tag", r.percentile >= (an.params?.emit_percentile ?? 0.99) ? "warn" : ""] }, `p${num(r.percentile * 100, 1)}`),
            h("span", { class: "mute" }, `track #${r.track_id}`),
            r.top?.[0] ? h("span", { class: "xs mute" }, `${r.top[0].feature} ${r.top[0].value} (typical ${r.top[0].typical})`) : null,
            h("span", { class: "grow" }), h("span", { class: "xs mute" }, fmtAgo(r.t))))) : null);
      })));
    }
    const cu = d.baseline?.cusum;
    parts.push(h("h4", null, "Occupancy shift · CUSUM ", infBadge("statistical")));
    if (!d.baseline || d.baseline.error || d.baseline.forbidden) parts.push(emptyState({ icon: "analytics", compact: true, title: "Baseline unavailable", text: "The hour-of-week baseline endpoint did not answer." }));
    else if (!cu || cu.enabled === false) parts.push(featureOff({ ic: "analytics", compact: true, title: "CUSUM is off", text: "The two-sided CUSUM on occupancy residuals is disabled in the baseline config." }));
    else {
      const rows = (cu.series || []).filter(r => keep(r.camera));
      if (!rows.length) parts.push(emptyState({ icon: "clock", compact: true, title: "Still building the baseline",
        text: "CUSUM starts once each zone has enough samples for this hour of the week (the median/MAD baseline)." }));
      else parts.push(h("div", { class: "px-cusum" }, rows.map(r => {
        const hi = Math.max(r.s_pos, r.s_neg) / (cu.h || 8);
        return h("div", { class: "px-cusum-row", title: `S+ ${r.s_pos}, S− ${r.s_neg}, threshold h = ${cu.h}, k = ${cu.k}; ${r.evaluations} evaluations` },
          h("div", { class: "small" }, camDot(r.camera), " ", h("b", null, r.zone), h("span", { class: "xs mute" }, ` · ${nm(r.camera)}`)),
          h("div", { class: "px-cusum-m" }, h("span", { class: "xs mute" }, "↑"), meter(r.s_pos / (cu.h || 8), { color: r.s_pos >= cu.h ? COL.err : r.s_pos >= cu.h * 0.6 ? COL.warn : COL.inf, label: "Upward shift statistic" }),
            h("span", { class: "xs mute" }, "↓"), meter(r.s_neg / (cu.h || 8), { color: r.s_neg >= cu.h ? COL.err : r.s_neg >= cu.h * 0.6 ? COL.warn : COL.inf, label: "Downward shift statistic" })),
          h("span", { class: ["xs", "tnum", hi >= 0.6 ? "c-warn" : "mute"] }, `${Math.round(hi * 100)}% of h`));
      })), h("div", { class: "note" }, `A bar reaching 100% (h = ${cu.h}) records a sustained occupancy shift for that hour of the week.`));
    }
    replace(anomBody, parts);
  }

  function renderHot() {
    const hs = d.hotspots;
    if (hs === undefined) return;
    if (!hs || hs.error || hs.forbidden) {
      replace(hotBody, featureOff({ ic: "target", compact: true, title: "Hotspots unavailable",
        text: hs?.error || "This server has no hotspot endpoint (it ships with the flow analytics)." }));
      return;
    }
    const cams = (hs.cameras || []).filter(c => keep(c.camera) && (c.ground_cells?.length || c.image_cells?.length));
    const zones = (hs.zones || []).filter(z => keep(z.camera)).slice(0, 10);
    if (!hs.events || (!cams.length && !zones.length)) {
      replace(hotBody, emptyState({ icon: "target", compact: true, title: `No rule events in the last ${win === "7d" ? "7 days" : "24 hours"}`,
        text: "Hotspots build up from alert events as rules fire (dismissed alerts are excluded)." }));
      return;
    }
    if (!cams.some(c => c.camera === heatCam)) heatCam = cams[0]?.camera || null;
    const heatSel = h("select", { class: "sm", "aria-label": "Heatmap camera", onchange: () => { heatCam = heatSel.value; renderHot(); } },
      cams.map(c => h("option", { value: c.camera, selected: c.camera === heatCam }, `${nm(c.camera)} (${c.events})`)));
    const cur = cams.find(c => c.camera === heatCam);
    const maxZ = Math.max(1, ...zones.map(z => z.events));
    replace(hotBody, h("div", { class: "px-hot" },
      h("div", { class: "px-hot-map" },
        cur ? [h("div", { class: "row small" }, "Camera", heatSel, h("span", { class: "grow" }), heatLegend()), heatmap(cur),
          h("div", { class: "xs mute" }, cur.ground_cells?.length ? `${hs.cell_m} m cells in this camera's ground frame (x right, y up)` : `${hs.cell_px} px cells in image space (uncalibrated camera)`,
            cur.unplaced ? ` · ${cur.unplaced} event${cur.unplaced === 1 ? "" : "s"} without a position` : "")]
          : emptyState({ icon: "target", compact: true, title: "No positioned events", text: "Events without a ground or image position appear only in the table." })),
      h("div", { class: "px-hot-table" }, h("div", { class: "table-wrap" }, h("table", { class: "table compact" },
        h("thead", null, h("tr", null, ["Zone", "Events", "Alerts", "Rules", "Last"].map(t => h("th", null, t)))),
        h("tbody", null, zones.map(z => h("tr", null,
          h("td", null, h("div", { class: "row", style: { gap: ".43rem", flexWrap: "nowrap" } }, camDot(z.camera),
            sample ? h("b", null, z.zone_name) : h("a", { href: `#/review?camera=${enc(z.camera)}` }, h("b", null, z.zone_name))),
            h("div", { class: "xs mute" }, nm(z.camera)),
            h("div", { class: "hbar-track", style: { marginTop: ".25rem" } }, h("div", { style: { width: `${Math.max(3, z.events / maxZ * 100)}%`, background: "var(--p2)" } }))),
          h("td", { class: "tnum" }, z.events), h("td", { class: "tnum" }, z.alerts),
          h("td", null, Object.entries(z.by_rule || {}).sort((a, b) => b[1] - a[1]).map(([r, n]) => h("span", { class: "tag", style: { margin: "0 .2rem .2rem 0" }, title: RULE[r] || r }, `${r} ${n}`))),
          h("td", { class: "xs mute nowrap" }, fmtAgo(z.last)))))))),
      h("div", { class: "note" }, hs.note || "")));
  }

  function heatLegend() {
    return h("span", { class: "legend" }, h("span", null, "fewer"), h("i", { class: "px-heat-grad", style: { width: "4rem" } }), h("span", null, "more events"));
  }

  function heatmap(cam) {
    const ground = !!cam.ground_cells?.length;
    const cells = ground ? cam.ground_cells : cam.image_cells;
    const size = cells[0]?.size || 1;
    const xs = cells.map(c => c.x0), ys = cells.map(c => c.y0);
    let x0 = Math.min(...xs) - size, x1 = Math.max(...xs) + 2 * size, y0 = Math.min(...ys) - size, y1 = Math.max(...ys) + 2 * size;
    const padTo = (a, b, n) => { const need = n * size - (b - a); return need > 0 ? [a - Math.floor(need / size / 2) * size, b + Math.ceil(need / size / 2) * size] : [a, b]; };
    [x0, x1] = padTo(x0, x1, 8); [y0, y1] = padTo(y0, y1, 6);   // keep a few cells of context around sparse data
    const W = x1 - x0, H = y1 - y0;
    const mx = Math.max(1, ...cells.map(c => c.count));
    // ground: y up (flip); image: y down
    const Y = y => (ground ? y1 - y - size : y - y0);
    const grid = [];
    for (let gx = Math.ceil(x0 / size) * size; gx <= x1; gx += size) grid.push(s("line", { x1: gx - x0, x2: gx - x0, y1: 0, y2: H, "vector-effect": "non-scaling-stroke", style: { stroke: "var(--line)", strokeWidth: 0.5 } }));
    for (let gy = Math.ceil(y0 / size) * size; gy <= y1; gy += size) grid.push(s("line", { x1: 0, x2: W, y1: gy - y0, y2: gy - y0, "vector-effect": "non-scaling-stroke", style: { stroke: "var(--line)", strokeWidth: 0.5 } }));
    return s("svg", { class: "px-heat", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": `Event heatmap for ${nm(cam.camera)}: ${cells.length} cells, busiest ${mx} events` },
      s("rect", { x: 0, y: 0, width: W, height: H, style: { fill: "var(--sunken)" } }), grid,
      cells.map(c => { const f = c.count / mx;
        return s("rect", { x: c.x0 - x0, y: Y(c.y0), width: size, height: size, rx: size * 0.12,
          style: { fill: f > 0.66 ? "var(--p1)" : f > 0.33 ? "var(--p2)" : "var(--p3)", opacity: 0.35 + 0.65 * f } },
        s("title", null, `${c.count} event${c.count === 1 ? "" : "s"} · ${Object.entries(c.rules || {}).map(([r, n]) => `${RULE[r] || r} ${n}`).join(", ")}`)); }));
  }

  function renderAll() {
    if (!shown) return;
    const nCams = new Set([...(d.zones?.cameras || []), ...(d.lines?.cameras || [])].map(c => c.camera)).size;
    replace(subtitle, `${S.cfg.site || "Site"} · ${sample ? "sample data" : d.zones === null && d.lines === null ? "flow analytics off" : `${nCams} camera${nCams === 1 ? "" : "s"} reporting`} · updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
    flowPanels.hidden = !sample && d.zones === null && d.lines === null;
    renderFilter(); renderBanner(); renderKpis(); renderLines(); renderDQ(); renderZones(); renderForecast(); renderEvents(); renderAnomaly(); renderHot();
  }

  on("demo", () => { if (shown) load(true); });
  on("cameras", () => { if (shown) renderAll(); });

  return {
    el,
    show() {
      shown = true;
      if (d.zones !== undefined) renderAll();
      load(true);
      clearInterval(timer);
      timer = setInterval(() => { if (shown && !document.hidden) load(); }, 20000);
    },
    hide() { shown = false; clearInterval(timer); },
    onResync() { if (shown) load(true); },
    onVisibility(v) { if (v && shown) load(); },
  };
}
