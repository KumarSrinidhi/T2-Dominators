// Incident evidence graph: a small force-directed layout in plain JS, drawn as SVG (text nodes only, XSS-safe).
// Nodes can be dragged; clicking a node calls onNode(node). Positions are deterministic for the same data.
import { h, s } from "./util.js";

const EDGE = {
  observed_in: { len: 70, k: 0.03, label: "observed in", dash: "", cls: "e-obs" },
  same_track: { len: 55, k: 0.05, label: "same track", dash: "", cls: "e-track" },
  corroborates: { len: 60, k: 0.04, label: "corroborates", dash: "", cls: "e-corr" },
  predicted_next: { len: 110, k: 0.02, label: "predicted next (estimate)", dash: "6 4", cls: "e-pred" },
  inhibited_by: { len: 100, k: 0.02, label: "inhibited by", dash: "3 3", cls: "e-inh" },
  evidence: { len: 35, k: 0.06, label: "evidence", dash: "1 3", cls: "e-evid" },
  reported_by: { len: 65, k: 0.04, label: "reported by (access control)", dash: "", cls: "e-rep" },
  covered_by: { len: 90, k: 0.02, label: "door in view of", dash: "5 3", cls: "e-cov" },
};
const KIND = {
  event: "Event (observation)", camera: "Camera", zone: "Zone", track: "Track (per-camera id)",
  global_track: "Global id (probabilistic)", evidence: "Evidence", incident: "Camera-integrity incident",
  coverage: "Leaves coverage (estimate)", sensor: "Access control (door controller / panel)",
};

function hash(str) { let x = 2166136261; for (let i = 0; i < str.length; i++) { x ^= str.charCodeAt(i); x = Math.imul(x, 16777619); } return (x >>> 0) / 4294967296; }

/** Run the layout. Mutates nodes (x, y). */
function layout(nodes, edges, iters = 320) {
  const W = 900, H = 520;
  const idx = new Map(nodes.map((n, i) => [n.id, i]));
  nodes.forEach(n => {
    if (n.x == null) {
      const a = hash(n.id) * Math.PI * 2, r = n.kind === "camera" ? 220 : n.kind === "event" ? 60 : 140 + 80 * hash(n.id + "r");
      n.x = W / 2 + Math.cos(a) * r; n.y = H / 2 + Math.sin(a) * r * 0.6;
    }
    n.vx = 0; n.vy = 0;
  });
  const E = edges.map(e => ({ a: idx.get(e.source), b: idx.get(e.target), p: EDGE[e.type] || EDGE.observed_in })).filter(e => e.a != null && e.b != null);
  const N = nodes.length;
  for (let it = 0; it < iters; it++) {
    const t = 1 - it / iters, rep = 2600;
    for (let i = 0; i < N; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < N; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
        if (d2 < 0.01) { dx = hash(a.id + b.id) - 0.5; dy = hash(b.id + a.id) - 0.5; d2 = 0.5; }
        if (d2 > 90000) continue;
        const f = rep / d2, d = Math.sqrt(d2);
        a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
      }
    }
    for (const e of E) {
      const a = nodes[e.a], b = nodes[e.b];
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(1, Math.hypot(dx, dy));
      const f = (d - e.p.len) * e.p.k;
      a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
    }
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.004; n.vy += (H / 2 - n.y) * 0.006;
      if (n.fixed) { n.vx = 0; n.vy = 0; continue; }
      const sp = Math.hypot(n.vx, n.vy), cap = 30 * t + 2;
      if (sp > cap) { n.vx *= cap / sp; n.vy *= cap / sp; }
      n.x += n.vx; n.y += n.vy; n.vx *= 0.55; n.vy *= 0.55;
    }
  }
}

function shape(n) {
  const c = "g-n k-" + n.kind + (n.layer === "inference" ? " inf" : "") + (n.modality === "access_control" ? " acs" : "") + (n.predicted_only ? " pred" : "");
  switch (n.kind) {
    case "camera": return s("rect", { class: c, x: -30, y: -11, width: 60, height: 22, rx: 5 });
    case "zone": return s("polygon", { class: c, points: "0,-11 16,0 0,11 -16,0" });
    case "event": return s("circle", { class: c, r: 8 });
    case "track": return s("circle", { class: c, r: 5 });
    case "global_track": return s("circle", { class: c, r: 7 });
    case "evidence": return s("rect", { class: c, x: -4, y: -4, width: 8, height: 8, rx: 1 });
    case "incident": return s("polygon", { class: c, points: "0,-12 12,0 0,12 -12,0" });
    case "sensor": return s("rect", { class: c, x: -9, y: -12, width: 18, height: 24, rx: 2 });  // a door
    default: return s("rect", { class: c, x: -34, y: -10, width: 68, height: 20, rx: 10 });
  }
}

function tip(n) {
  const lines = [`${KIND[n.kind] || n.kind}: ${n.label}`];
  if (n.kind === "event") {
    lines.push(`alert ${n.alert_id}`, n.zone ? `zone ${n.zone}` : "", `confidence ${n.confidence ?? "—"} (heuristic)`,
      n.t_end ? new Date(n.t_end * 1000).toLocaleString() : "", n.observation || "",
      n.layer === "inference" ? "learned-model inference (corroboration only)"
        : n.modality === "access_control" ? "access-control device report (not video)" : "observation", "click: open the alert");
  }
  if (n.note) lines.push(n.note);
  if (n.predicted_only) lines.push("only reached through a next-camera estimate");
  if (n.kind === "evidence") lines.push([n.snapshot && "snapshot", n.clip && "clip", n.crop && "crop"].filter(Boolean).join(" + "), "click: open");
  return lines.filter(Boolean).join("\n");
}

function short(label, n = 18) { label = String(label ?? ""); return label.length > n ? label.slice(0, n - 1) + "…" : label; }

/** Build the graph view. data = /api/incidents/{id}/graph. Returns {el}. */
export function forceGraph(data, { onNode } = {}) {
  const nodes = (data.nodes || []).map(n => ({ ...n }));
  const edges = data.edges || [];
  layout(nodes, edges);
  const byId = new Map(nodes.map(n => [n.id, n]));
  const svg = s("svg", { class: "graph", role: "img", "aria-label": `Evidence graph with ${nodes.length} nodes and ${edges.length} links` });
  const defs = s("defs", null, s("marker", { id: "arrowPred", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
    s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-pred" })));
  const gE = s("g"), gN = s("g");
  svg.append(defs, gE, gN);
  const lines = edges.map(e => {
    const p = EDGE[e.type] || EDGE.observed_in;
    const ln = s("line", { class: "g-e " + p.cls, "stroke-dasharray": p.dash || null, "marker-end": e.type === "predicted_next" ? "url(#arrowPred)" : null },
      s("title", null, `${p.label}` + (e.prob != null ? ` · p=${Math.round(e.prob * 100)}%` : "") + (e.eta_s != null ? ` · ETA ~${Math.round(e.eta_s)} s` : "") +
        (e.confidence != null ? ` · confidence ${e.confidence}` : "") + (e.estimate || e.probabilistic ? " (estimate, not identity)" : "")));
    gE.appendChild(ln);
    return { e, ln };
  });
  const gs = nodes.map(n => {
    const g = s("g", { class: "g-node", tabindex: 0, role: "button", "aria-label": `${KIND[n.kind] || n.kind} ${n.label}` },
      shape(n), s("title", null, tip(n)));
    if (["camera", "zone", "incident", "coverage", "global_track", "sensor"].includes(n.kind) || nodes.length < 40) {
      const dy = n.kind === "camera" || n.kind === "coverage" ? 4 : n.kind === "event" ? 20 : n.kind === "sensor" ? 24 : 18;
      g.appendChild(s("text", { class: "g-lbl" + (n.kind === "camera" || n.kind === "coverage" ? " in" : ""), y: dy, "text-anchor": "middle" },
        short(n.kind === "event" ? n.label.split(" ")[0] : n.label, n.kind === "camera" ? 11 : 16)));
    }
    gN.appendChild(g);
    return { n, g };
  });

  function paint() {
    for (const { e, ln } of lines) {
      const a = byId.get(e.source), b = byId.get(e.target);
      if (!a || !b) continue;
      let x2 = b.x, y2 = b.y;
      if (e.type === "predicted_next") { const d = Math.max(1, Math.hypot(b.x - a.x, b.y - a.y)); x2 -= (b.x - a.x) / d * 30; y2 -= (b.y - a.y) / d * 12; }
      ln.setAttribute("x1", a.x); ln.setAttribute("y1", a.y); ln.setAttribute("x2", x2); ln.setAttribute("y2", y2);
    }
    for (const { n, g } of gs) g.setAttribute("transform", `translate(${n.x.toFixed(1)},${n.y.toFixed(1)})`);
  }
  function fit() {
    if (!nodes.length) { svg.setAttribute("viewBox", "0 0 400 200"); return; }
    const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y), pad = 45;
    const x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad, w = Math.max(...xs) - x0 + pad, hh = Math.max(...ys) - y0 + pad;
    svg.setAttribute("viewBox", `${x0.toFixed(0)} ${y0.toFixed(0)} ${Math.max(200, w).toFixed(0)} ${Math.max(160, hh).toFixed(0)}`);
  }
  paint(); fit();

  // ---- drag (pointer) + click ----
  let drag = null;
  const toSvg = ev => { const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); };
  gs.forEach(({ n, g }) => {
    g.addEventListener("pointerdown", ev => { drag = { n, moved: false, x: ev.clientX, y: ev.clientY }; g.setPointerCapture(ev.pointerId); ev.preventDefault(); });
    g.addEventListener("pointermove", ev => {
      if (!drag || drag.n !== n) return;
      if (Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) > 3) drag.moved = true;
      if (!drag.moved) return;
      const p = toSvg(ev); n.x = p.x; n.y = p.y; n.fixed = true; paint();
    });
    g.addEventListener("pointerup", () => {
      const d = drag; drag = null;
      if (d && d.n === n && d.moved) { layout(nodes, edges, 40); paint(); }
      else onNode?.(n);
    });
    g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onNode?.(n); } });
  });

  const used = new Set(edges.map(e => e.type)), kinds = new Set(nodes.map(n => n.kind));
  const legend = h("div", { class: "g-legend small" },
    [...kinds].map(k => h("span", { class: "lg" }, legendNode(k), KIND[k] || k)),
    [...used].map(t => h("span", { class: "lg" },
      s("svg", { width: 26, height: 10, "aria-hidden": "true" },
        s("line", { x1: 1, y1: 5, x2: 25, y2: 5, class: "g-e " + (EDGE[t]?.cls || ""), "stroke-dasharray": EDGE[t]?.dash || null })),
      EDGE[t]?.label || t)));
  const reset = h("button", { class: "ghost small", onclick: () => { nodes.forEach(n => { n.fixed = false; n.x = null; }); layout(nodes, edges); paint(); fit(); } }, "Re-layout");
  const el = h("div", { class: "graph-box" }, svg,
    h("div", { class: "row", style: { justifyContent: "space-between", marginTop: "6px" } }, legend, reset));
  return { el };
}

function legendNode(kind) {
  return s("svg", { width: 18, height: 14, viewBox: "-36 -14 72 28", "aria-hidden": "true" }, shape({ kind, id: kind, label: "" }));
}
