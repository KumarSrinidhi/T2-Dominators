// Tiny dependency-free SVG charts. Colours are CSS custom properties (set through `style`, so theme switches
// repaint without re-rendering). All labels are text nodes; nothing here uses innerHTML.
import { h, s } from "./util.js";

const fin = v => typeof v === "number" && isFinite(v);

/**
 * sparkline(values, {width, height, color, fill, min, max, label, dot})
 * A single series line (optionally area-filled). Gaps (null) break the line.
 */
export function sparkline(values, o = {}) {
  const W = o.width || 120, H = o.height || 32, pad = 2;
  const vals = (values || []).map(v => (fin(v) ? v : null));
  const nums = vals.filter(v => v !== null);
  const min = o.min ?? Math.min(0, ...nums), max = o.max ?? Math.max(1e-9, ...nums);
  const n = Math.max(1, vals.length - 1);
  const X = i => pad + (i / n) * (W - 2 * pad);
  const Y = v => H - pad - ((v - min) / (max - min || 1)) * (H - 2 * pad);
  const color = o.color || "var(--acc)";
  let d = "", area = "", open = false, first = null, last = null;
  vals.forEach((v, i) => {
    if (v === null) { if (open && o.fill !== false) area += `L${X(last)},${H - pad}Z`; open = false; return; }
    if (!open) { d += `M${X(i)},${Y(v)}`; if (o.fill !== false) area += `M${X(i)},${H - pad}L${X(i)},${Y(v)}`; first = i; open = true; }
    else { d += `L${X(i)},${Y(v)}`; if (o.fill !== false) area += `L${X(i)},${Y(v)}`; }
    last = i;
  });
  if (open && o.fill !== false) area += `L${X(last)},${H - pad}Z`;
  const svg = s("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, width: W, height: H, preserveAspectRatio: "none",
    role: "img", "aria-label": o.label || "trend" },
  o.label ? s("title", null, o.label) : null,
  area ? s("path", { d: area, style: { fill: color, opacity: 0.14 } }) : null,
  d ? s("path", { d, style: { fill: "none", stroke: color, strokeWidth: o.strokeWidth || 1.75, strokeLinejoin: "round", strokeLinecap: "round" }, "vector-effect": "non-scaling-stroke" }) : null,
  o.dot !== false && last !== null ? s("circle", { cx: X(last), cy: Y(vals[last]), r: 2.4, style: { fill: color } }) : null);
  if (first === null) svg.classList.add("empty");
  return svg;
}

/**
 * stackedBars(rows, keys, {width, height, colors, label, max, onbar, titles})
 * rows: [{P1: 2, P2: 0, ...}, ...]; keys in bottom-to-top order; colors: {key: "var(--p1)"}.
 */
export function stackedBars(rows, keys, o = {}) {
  const W = o.width || 240, H = o.height || 48, n = Math.max(1, rows.length);
  const total = r => keys.reduce((a, k) => a + (+r[k] || 0), 0);
  const max = o.max || Math.max(1, ...rows.map(total));
  const bw = W / n, gap = Math.min(1.2, bw * 0.18);
  const nodes = [s("rect", { x: 0, y: H - 0.5, width: W, height: 0.5, style: { fill: "var(--line-strong)" } })];
  rows.forEach((r, i) => {
    let y = H;
    const g = s("g", { class: "bar", "data-i": i });
    if (o.titles?.[i]) g.appendChild(s("title", null, o.titles[i]));
    g.appendChild(s("rect", { x: i * bw, y: 0, width: bw, height: H, style: { fill: "transparent" } }));
    for (const k of keys) {
      const v = +r[k] || 0;
      if (!v) continue;
      const hh = Math.max(1.5, (v / max) * (H - 2));
      y -= hh;
      g.appendChild(s("rect", { x: i * bw + gap / 2, y, width: Math.max(0.5, bw - gap), height: hh, rx: Math.min(1.5, bw / 4),
        style: { fill: o.colors?.[k] || "var(--acc)" } }));
    }
    if (o.onbar) g.addEventListener("click", () => o.onbar(i, r));
    nodes.push(g);
  });
  return s("svg", { class: "bars", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": o.label || "bar chart" },
    o.label ? s("title", null, o.label) : null, nodes);
}

/** gauge(frac, {size, stroke, color, label, value}) → ring gauge with the value in the centre. */
export function gauge(frac, o = {}) {
  const S = o.size || 64, sw = o.stroke || 6, r = (S - sw) / 2, c = 2 * Math.PI * r;
  const f = fin(frac) ? Math.max(0, Math.min(1, frac)) : 0;
  const color = o.color || (f >= 0.9 ? "var(--err)" : f >= 0.75 ? "var(--warn)" : "var(--acc)");
  return h("div", { class: "gauge", style: { width: S + "px", height: S + "px" }, role: "img",
    "aria-label": `${o.label || "gauge"}: ${o.value ?? Math.round(f * 100) + "%"}` },
  s("svg", { viewBox: `0 0 ${S} ${S}`, width: S, height: S },
    s("circle", { cx: S / 2, cy: S / 2, r, style: { fill: "none", stroke: "var(--surface-3)", strokeWidth: sw } }),
    s("circle", { cx: S / 2, cy: S / 2, r, transform: `rotate(-90 ${S / 2} ${S / 2})`,
      style: { fill: "none", stroke: fin(frac) ? color : "transparent", strokeWidth: sw, strokeLinecap: "round",
        strokeDasharray: `${c * f} ${c}` } })),
  h("span", { class: "gauge-v" }, o.value ?? (fin(frac) ? Math.round(f * 100) + "%" : "—")));
}

/** meter(frac, {color, label}) → thin horizontal bar. */
export function meter(frac, o = {}) {
  const f = fin(frac) ? Math.max(0, Math.min(1, frac)) : 0;
  return h("div", { class: "meter", role: "meter", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(f * 100),
    "aria-label": o.label || "level" },
  h("div", { style: { width: (f * 100).toFixed(1) + "%", background: o.color || "var(--acc)" } }));
}

/** stackBar([{value, color, label}]) → one horizontal stacked bar (e.g. alerts by priority). */
export function stackBar(parts, o = {}) {
  const tot = parts.reduce((a, p) => a + (+p.value || 0), 0);
  return h("div", { class: "stackbar", role: "img", "aria-label": o.label || parts.map(p => `${p.label} ${p.value}`).join(", ") },
    tot ? parts.filter(p => +p.value > 0).map(p => h("span", { title: `${p.label}: ${p.value}`,
      style: { flexGrow: String(+p.value), background: p.color } })) : h("span", { class: "none" }));
}

/** hbars([{label, value, color?, sub?}], {max}) → labelled horizontal bars (top-N lists). */
export function hbars(items, o = {}) {
  const max = o.max || Math.max(1, ...items.map(i => +i.value || 0));
  return h("div", { class: "hbars" }, items.map(i => h("div", { class: "hbar", title: i.title || "" },
    h("div", { class: "hbar-top" }, h("span", { class: "hbar-l" }, i.label), h("span", { class: "hbar-v tnum" }, i.display ?? i.value)),
    h("div", { class: "hbar-track" }, h("div", { style: { width: `${Math.max(2, (+i.value || 0) / max * 100)}%`, background: i.color || "var(--acc)" } })),
    i.sub ? h("div", { class: "hbar-sub" }, i.sub) : null)));
}

/**
 * forecastChart({history: [{t, v}], forecast: [{t, mean, lo, hi}], asOf, limit, showForecast, label, width, height})
 * Observed series (solid) up to a "now" line, then the forecast (dashed mean, shaded prediction interval), with an
 * optional dashed limit. The forecast region carries its own label: it is a prediction, not an observation.
 */
export function forecastChart(o = {}) {
  const W = o.width || 480, H = o.height || 132, L = 30, R = 8, T = 18, B = 16;
  const hist = (o.history || []).filter(p => fin(p.v));
  const fc = o.showForecast ? (o.forecast || []).filter(p => fin(p.mean)) : [];
  const t0 = hist.length ? hist[0].t : (fc[0]?.t ?? 0);
  const tNow = fin(o.asOf) ? o.asOf : (hist.length ? hist[hist.length - 1].t + 60 : t0);
  const t1 = Math.max(tNow + 60, fc.length ? fc[fc.length - 1].t : tNow);
  const ys = [...hist.map(p => p.v), ...fc.flatMap(p => [p.mean, fin(p.hi) ? p.hi : p.mean]), fin(o.limit) ? o.limit : 0, 1];
  const top = Math.max(...ys) * 1.12;
  const X = t => L + (t - t0) / Math.max(1, t1 - t0) * (W - L - R);
  const Y = v => T + (1 - v / top) * (H - T - B);
  const gap = o.gapS || 90;   // a missing sample breaks the observed line instead of bridging it
  const path = (pts, breaks = false) => pts.map((p, i) => `${i && !(breaks && p[0] - pts[i - 1][0] > gap) ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const hm = t => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const nodes = [];
  for (const f of [0, 0.5, 1]) {
    const v = top / 1.12 * f;
    nodes.push(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), style: { stroke: "var(--line)", strokeWidth: 0.6 } }),
      s("text", { x: L - 4, y: Y(v) + 3, "text-anchor": "end", class: "fc-txt" }, String(Math.round(v * 10) / 10)));
  }
  const band = fc.filter(p => fin(p.lo) && fin(p.hi));
  if (band.length) {
    const d = path(band.map(p => [p.t, p.hi])) + path(band.map(p => [p.t, p.lo]).reverse()).replace(/^M/, "L") + "Z";
    nodes.push(s("path", { d, style: { fill: "var(--inf)", opacity: 0.16 } }, s("title", null, "80% prediction interval (from past forecast errors)")));
  }
  if (hist.length) nodes.push(s("path", { d: path(hist.map(p => [p.t, p.v]), true), style: { fill: "none", stroke: "var(--acc-text)", strokeWidth: 1.8, strokeLinejoin: "round" } },
    s("title", null, "Observed: mean per minute")));
  if (fc.length) {
    const join = hist.length ? [[hist[hist.length - 1].t, hist[hist.length - 1].v]] : [];
    nodes.push(s("path", { d: path(join.concat(fc.map(p => [p.t, p.mean]))), style: { fill: "none", stroke: "var(--inf)", strokeWidth: 1.8, strokeDasharray: "5 4" } },
      s("title", null, "Forecast mean: a prediction, not an observation")));
  }
  if (fin(o.limit)) nodes.push(s("line", { x1: L, x2: W - R, y1: Y(o.limit), y2: Y(o.limit), style: { stroke: "var(--warn)", strokeWidth: 1, strokeDasharray: "4 3" } },
    s("title", null, `limit ${o.limit}`)));
  nodes.push(s("line", { x1: X(tNow), x2: X(tNow), y1: T - 8, y2: H - B, style: { stroke: "var(--line-strong)", strokeWidth: 1, strokeDasharray: "2 2" } }),
    s("text", { x: X(tNow) - 3, y: T - 8, "text-anchor": "end", class: "fc-txt" }, "observed"),
    fc.length ? s("text", { x: X(tNow) + 3, y: T - 8, class: "fc-txt", style: { fill: "var(--inf)" } }, "forecast") : null,
    s("text", { x: L, y: H - 3, class: "fc-txt" }, hm(t0)),
    fc.length ? s("text", { x: X(tNow), y: H - 3, "text-anchor": "middle", class: "fc-txt" }, hm(tNow)) : null,
    s("text", { x: W - R, y: H - 3, "text-anchor": "end", class: "fc-txt" }, hm(t1)));
  const svg = s("svg", { class: "fc-chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "Occupancy with forecast" },
    o.label ? s("title", null, o.label) : null, nodes);
  return h("div", { class: "fc-wrap" }, svg, h("div", { class: "legend fc-legend" },
    h("span", null, h("i", { style: { background: "var(--acc-text)" } }), "observed"),
    fc.length ? [h("span", null, h("i", { class: "fc-dash" }), "forecast — prediction, not observation"),
      band.length ? h("span", null, h("i", { style: { background: "var(--inf)", opacity: 0.25 } }), "80% interval") : null] : null,
    fin(o.limit) ? h("span", null, h("i", { class: "fc-dash warn" }), "limit") : null));
}
