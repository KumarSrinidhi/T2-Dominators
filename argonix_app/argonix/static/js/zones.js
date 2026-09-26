// Zone, mask and counting-line editor over the camera's raw snapshot (coordinates in SOURCE pixels).
// click = add point · drag = move point / polygon · double-click, Enter or click first point = close ·
// Backspace = undo point · Esc = cancel · Delete = remove selected · right-click a point = delete it.
// Counting lines and queue service lines are drawn by dragging from one end to the other.
import { api, ApiError } from "./api.js";
import { S, can, normRules, loadConfig } from "./state.js";
import { toast } from "./ui.js";
import { h, s, replace, enc, icon, RULE, RULE_PARAMS, METRIC_RULES, VEHICLE_RULES, slug, clamp, isTyping } from "./util.js";

const CRIT = { 1: ["Low", "#3b82f6"], 2: ["Medium", "#f59e0b"], 3: ["High", "#ef4444"] };
const MASK_COL = "#94a3b8";
const PRIVACY_COL = "#f8fafc";   // outline; privacy masks are filled black (that is what the server paints)
const LINE_COL = "#22d3ee";
const SERVICE_COL = "#a3e635";
const EDITABLE_RULES = ["intrusion", "loitering", "running", "wrong_way", "crowding",
  "counterflow", "density", "queue_long", "dispersal", "convergence", "vehicle_in_zone", "vehicle_stopped"];
const FRACTION_PARAMS = new Set(["min_fraction", "min_concentration", "still_box_frac"]);
const INT_PARAMS = new Set(["max_count", "min_tracks", "min_against", "min_count", "min_people"]);
// schedule windows: "<days> HH:MM-HH:MM" (the server does the full validation via /api/schedule/check)
const WIN_RE = /^\s*(?:[a-z*,\-\s]+?\s+)?\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$/i;
const SCHED_EXAMPLE = "mon-fri 19:00-07:00; sat,sun 00:00-24:00";

const clone = v => JSON.parse(JSON.stringify(v ?? null));

/** Normalise a config schedule (object, bare window string or list) to {windows: [...], tz, label} or null. */
function normSched(sc) {
  if (!sc) return null;
  if (typeof sc === "string" || Array.isArray(sc)) sc = { windows: sc };
  const w = typeof sc.windows === "string" ? sc.windows.split(";") : Array.isArray(sc.windows) ? sc.windows : [];
  return { windows: w.map(x => String(x).trim()).filter(Boolean), tz: sc.tz || "", label: sc.label || "" };
}
function schedPayload(sc) {
  if (!sc) return null;
  const o = { windows: sc.windows };
  if (sc.tz?.trim()) o.tz = sc.tz.trim();
  if (sc.label?.trim()) o.label = sc.label.trim();
  return o;
}
function schedErrors(sc, what) {
  if (!sc) return [];
  if (!sc.windows.length) return [`${what}: the schedule needs at least one window, e.g. "${SCHED_EXAMPLE}".`];
  return sc.windows.filter(w => !WIN_RE.test(w)).map(w => `${what}: schedule window "${w}" is not "<days> HH:MM-HH:MM".`);
}

function normZone(z, i) {
  return {
    id: String(z.id || `zone_${i + 1}`), name: String(z.name || z.id || `Zone ${i + 1}`),
    criticality: clamp(+z.criticality || 1, 1, 3),
    polygon_px: Array.isArray(z.polygon_px) ? z.polygon_px.map(p => [+p[0], +p[1]]) : null,
    polygon: Array.isArray(z.polygon) ? z.polygon : null,          // ground metres (legacy, geometry read-only here)
    direction: z.direction || null, direction_px: z.direction_px || null,
    site_zone: z.site_zone || "",
    rules: Object.fromEntries(Object.entries(normRules(z.rules)).map(([k, v]) => [k, { ...(v || {}) }])),
    type: z.type === "queue" || "queue_long" in normRules(z.rules) ? "queue" : "area",
    service_line_px: Array.isArray(z.service_line_px) ? z.service_line_px.map(p => [+p[0], +p[1]]) : null,
    service_line: Array.isArray(z.service_line) ? z.service_line : null,   // ground metres (config only)
    schedule: normSched(z.schedule),   // rules arm only inside these windows (per-rule schedules stay in rules[r].schedule)
  };
}

function normLine(l, i) {
  return {
    id: String(l.id || `line_${i + 1}`), name: String(l.name || l.id || `Line ${i + 1}`),
    points_px: Array.isArray(l.points_px) ? l.points_px.map(p => [+p[0], +p[1]]) : null,
    points: Array.isArray(l.points) ? l.points : null,                     // ground metres (config only)
    in_label: String(l.in_label || "in"), out_label: String(l.out_label || "out"),
    ...(l.band != null ? { band: +l.band } : {}),
    // opt-in line_cross events: {direction: in|out|both, min_gap_s, criticality, schedule}
    events: l.events ? { direction: l.events.direction || "both", min_gap_s: l.events.min_gap_s ?? 5,
      criticality: +l.events.criticality || 1, schedule: normSched(l.events.schedule) } : null,
  };
}

export function createZoneEditor(cam) {
  const ro = !can("zone.edit");
  let zones = (cam.zones || []).map(normZone);
  let masks = clone(cam.masks_px || []) || [];
  let privacy = clone(cam.privacy_masks_px || []) || [];   // blanked on the frame before anything else sees it
  let lines = (cam.lines || []).map(normLine);
  let orig = JSON.stringify([zones, masks, lines, privacy]);
  let zonesDirty = false, masksDirty = false, linesDirty = false, privacyDirty = false;
  const anyDirty = () => zonesDirty || masksDirty || linesDirty || privacyDirty;
  let sel = null;              // {k: "zone"|"mask"|"privacy"|"line", i}
  let drawing = null;          // {k, pts, hover}
  let arrow = null;            // {start, end} while drawing a direction
  let seg = null;              // {k: "line"|"service", start, end} while dragging a counting / service line
  let drag = null;
  let W = +cam.width || 1280, H = +cam.height || 720, scaleNote = null, imgUrl = null;

  const imgEl = s("image", { x: 0, y: 0, preserveAspectRatio: "none" });
  const bgRect = s("rect", { x: 0, y: 0, fill: "#111" });
  const overlay = s("g");
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Zone editor canvas" }, bgRect, imgEl, overlay);
  const box = h("div", { class: "canvas-box" }, svg);
  const hint = h("div", { class: "small mute", style: { margin: "6px 0" } });
  const status = h("span", { class: "dirty" });
  const listEl = h("div", { class: "stack ed-list" });
  const insp = h("div");
  const banner = h("div");
  const saveBtn = h("button", { class: "primary", disabled: true, onclick: () => save() }, "Save");
  const revertBtn = h("button", { disabled: true, onclick: () => revert() }, "Revert");
  const el = h("div", null, banner,
    h("div", { class: "ed-tools" },
      ro ? h("span", { class: "tag" }, "read-only (needs zone.edit)") : [
        h("button", { onclick: () => startDraw("zone") }, icon("plus"), "Zone"),
        h("button", { onclick: () => startDraw("mask") }, icon("plus"), "Mask"),
        h("button", { title: "Privacy mask: these pixels are painted black before detection, recording, evidence and live view", onclick: () => startDraw("privacy") }, icon("lock"), "Privacy mask"),
        h("button", { title: "Counting line: drag from one end to the other", onclick: () => startSeg("line") }, icon("plus"), "Line"),
        saveBtn, revertBtn],
      h("button", { title: "Reload snapshot", onclick: () => loadImage() }, icon("refresh"), "Snapshot"),
      status),
    h("div", { class: "editor" },
      h("div", null, box, hint,
        h("div", { class: "note" }, "Zones are drawn in source-image pixels; the server projects them onto the 2D ground plane with the camera's homography. " +
          "Intrusion tests use a conservative margin equal to the camera's estimated calibration error, so draw zones generously around the protected area.")),
      h("div", { class: "stack" }, h("div", { class: "panel" }, listEl), insp)));

  // ---------- image ----------
  function loadImage() {
    const t = Date.now();
    const img = new Image();
    img.onload = () => {
      W = +cam.width || img.naturalWidth; H = +cam.height || img.naturalHeight;
      if (!cam.width) { W = img.naturalWidth; H = img.naturalHeight; }
      imgUrl = img.src; scaleNote = null; render(); renderBanner();
    };
    img.onerror = () => {
      // Fallback: the annotated analytics frame (downscaled). Map it onto the source size when known.
      const f = new Image();
      f.onload = () => {
        W = +cam.width || f.naturalWidth; H = +cam.height || f.naturalHeight;
        imgUrl = f.src;
        scaleNote = cam.width ? "Raw snapshot endpoint unavailable: showing the annotated analytics frame stretched to the source resolution (overlays drawn by the server may be visible)."
          : "Raw snapshot endpoint unavailable and the source resolution is unknown: coordinates below are in analytics-frame pixels and may not match the server's source-pixel zones.";
        render(); renderBanner();
      };
      f.onerror = () => { imgUrl = null; scaleNote = "No frame available from this camera yet."; render(); renderBanner(); };
      f.src = `/api/cameras/${enc(cam.id)}/latest.jpg?t=${t}`;
    };
    img.src = `/api/cameras/${enc(cam.id)}/snapshot.jpg?t=${t}`;
  }
  function renderBanner() {
    const legacy = zones.filter(z => !z.polygon_px).length;
    replace(banner,
      scaleNote ? h("div", { class: "banner" }, scaleNote) : null,
      legacy ? h("div", { class: "banner" }, `${legacy} zone(s) are defined in ground metres in the site config and cannot be reshaped here; their rules can still be edited, or delete and redraw them in pixels.`) : null);
  }

  // ---------- geometry ----------
  const hr = () => 7 * W / Math.max(1, svg.getBoundingClientRect().width || W);   // handle radius in source px for ~7 screen px
  function pt(e) {
    const m = svg.getScreenCTM();
    if (!m) return [0, 0];
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return [Math.round(clamp(p.x, 0, W)), Math.round(clamp(p.y, 0, H))];
  }
  const polyOf = (k, i) => k === "zone" ? zones[i]?.polygon_px : k === "line" ? lines[i]?.points_px : k === "privacy" ? privacy[i] : masks[i];
  // "in" direction of a counting line A->B: n = (-dy, dx), i.e. the right-hand side looking from A to B on screen
  const inNormal = ([[x0, y0], [x1, y1]]) => { const L = Math.hypot(x1 - x0, y1 - y0) || 1; return [-(y1 - y0) / L, (x1 - x0) / L]; };
  function segNodes(pts, col, label, r, sw, fs, extra = {}) {
    const [[x0, y0], [x1, y1]] = pts, mx = (x0 + x1) / 2, my = (y0 + y1) / 2, [nx, ny] = inNormal(pts), L = r * 3;
    const out = [
      s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: "#000", "stroke-width": sw * 5, opacity: .45, "pointer-events": "none" }),
      s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: col, "stroke-width": sw * 2.4, "pointer-events": "none", ...(extra.dash ? { "stroke-dasharray": `${r} ${r / 2}` } : {}) }),
    ];
    if (extra.arrow !== false) {
      out.push(s("line", { x1: mx, y1: my, x2: mx + nx * L, y2: my + ny * L, stroke: col, "stroke-width": sw * 2, "marker-end": "url(#arrowhead)", "pointer-events": "none" }));
      if (extra.inLabel) out.push(s("text", { x: mx + nx * L * 1.7, y: my + ny * L * 1.7, fill: "#fff", "font-size": fs * 0.75, "text-anchor": "middle", "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke" }, extra.inLabel));
      if (extra.outLabel) out.push(s("text", { x: mx - nx * L * 1.2, y: my - ny * L * 1.2, fill: "#fff", "font-size": fs * 0.75, "text-anchor": "middle", "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke", opacity: .8 }, extra.outLabel));
    }
    if (label) out.push(s("text", { x: x0, y: y0 - r * 1.2, fill: col, "font-size": fs * 0.85, "font-weight": 700, "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke" }, label));
    if (extra.hit) out.push(s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: "transparent", "stroke-width": r * 2.2, "pointer-events": "stroke", ...extra.hit }));
    return out;
  }
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const centroid = pts => [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];

  // ---------- render ----------
  function render() {
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    for (const e of [imgEl, bgRect]) { e.setAttribute("width", W); e.setAttribute("height", H); }
    if (imgUrl) { if (imgEl.getAttribute("href") !== imgUrl) imgEl.setAttribute("href", imgUrl); imgEl.style.display = ""; }
    else imgEl.style.display = "none";
    const r = hr(), sw = r / 3.5, fs = Math.max(12, r * 1.9);
    const nodes = [
      s("defs", null,
        s("pattern", { id: "hatch", width: r * 1.5, height: r * 1.5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
          s("line", { x1: 0, y1: 0, x2: 0, y2: r * 1.5, stroke: MASK_COL, "stroke-width": sw * 1.5, opacity: 0.8 })),
        s("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" },
          s("path", { d: "M0 0L10 5L0 10z", fill: "#fff" }))),
    ];
    masks.forEach((m, i) => {
      if (!m?.length) return;
      const on = sel?.k === "mask" && sel.i === i;
      nodes.push(s("polygon", { points: m.map(p => p.join(",")).join(" "), fill: "url(#hatch)", stroke: MASK_COL,
        "stroke-width": on ? sw * 2 : sw, "stroke-dasharray": `${r} ${r / 2}`, "data-k": "mask", "data-i": i, style: ro ? "" : "cursor:move" }));
      const [cx, cy] = centroid(m);
      nodes.push(s("text", { x: cx, y: cy, fill: "#fff", "font-size": fs * 0.8, "text-anchor": "middle", "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke" }, `mask ${i + 1}`));
    });
    privacy.forEach((m, i) => {
      if (!m?.length) return;
      const on = sel?.k === "privacy" && sel.i === i;
      nodes.push(s("polygon", { points: m.map(p => p.join(",")).join(" "), fill: "#000", "fill-opacity": on ? 0.75 : 0.6, stroke: PRIVACY_COL,
        "stroke-width": on ? sw * 2 : sw, "stroke-dasharray": `${r / 2} ${r / 2}`, "data-k": "privacy", "data-i": i, style: ro ? "" : "cursor:move" }));
      const [cx, cy] = centroid(m);
      nodes.push(s("text", { x: cx, y: cy, fill: "#fff", "font-size": fs * 0.8, "text-anchor": "middle", "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke" }, `privacy ${i + 1}`));
    });
    zones.forEach((z, i) => {
      const pts = z.polygon_px;
      if (!pts?.length) return;
      const col = CRIT[z.criticality][1], on = sel?.k === "zone" && sel.i === i;
      nodes.push(s("polygon", { points: pts.map(p => p.join(",")).join(" "), fill: col, "fill-opacity": on ? 0.3 : 0.18, stroke: col,
        "stroke-width": on ? sw * 2.2 : sw * 1.3, "data-k": "zone", "data-i": i, style: ro ? "" : "cursor:move" }));
      const [cx, cy] = centroid(pts);
      nodes.push(s("text", { x: cx, y: cy, fill: "#fff", "font-size": fs, "font-weight": 700, "text-anchor": "middle", "pointer-events": "none", stroke: "#000", "stroke-width": sw, "paint-order": "stroke" }, z.name));
      if (z.direction_px?.length === 2) {
        const [[x0, y0], [x1, y1]] = z.direction_px;
        nodes.push(s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: "#000", "stroke-width": sw * 5, "pointer-events": "none", opacity: .5 }),
          s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: "#fff", "stroke-width": sw * 2.5, "marker-end": "url(#arrowhead)", "pointer-events": "none" }));
      }
      if (z.type === "queue" && z.service_line_px?.length === 2)
        nodes.push(...segNodes(z.service_line_px, SERVICE_COL, `${z.name}: service`, r, sw, fs, { dash: true, arrow: false }));
    });
    lines.forEach((l, i) => {
      if (l.points_px?.length !== 2) return;
      const on = sel?.k === "line" && sel.i === i;
      nodes.push(...segNodes(l.points_px, on ? "#fff" : LINE_COL, l.name, r, on ? sw * 1.4 : sw, fs,
        { inLabel: l.in_label, outLabel: l.out_label, hit: { "data-k": "line", "data-i": i, style: ro ? "" : "cursor:move" } }));
    });
    // handles for the selection
    if (sel && !drawing && !ro) {
      const pts = polyOf(sel.k, sel.i);
      if (pts) {
        if (sel.k !== "line") pts.forEach((p, j) => {
          const q = pts[(j + 1) % pts.length];
          nodes.push(s("circle", { cx: (p[0] + q[0]) / 2, cy: (p[1] + q[1]) / 2, r: r * 0.6, fill: "#fff8", stroke: "#000", "stroke-width": sw / 2,
            "data-role": "mid", "data-j": j, style: "cursor:copy" }, s("title", null, "Drag to insert a point")));
        });
        pts.forEach((p, j) => nodes.push(s("circle", { cx: p[0], cy: p[1], r, fill: "#fff", stroke: "#000", "stroke-width": sw,
          "data-role": "vtx", "data-j": j, style: "cursor:grab" }, s("title", null, "Drag to move · right-click to delete"))));
      }
    }
    if (drawing) {
      const pts = drawing.pts, col = drawing.k === "mask" ? MASK_COL : drawing.k === "privacy" ? PRIVACY_COL : "#22d3ee";
      const all = drawing.hover ? [...pts, drawing.hover] : pts;
      if (all.length > 1) nodes.push(s("polyline", { points: all.map(p => p.join(",")).join(" "), fill: "none", stroke: col, "stroke-width": sw * 1.5, "stroke-dasharray": `${r} ${r / 2}`, "pointer-events": "none" }));
      pts.forEach((p, j) => nodes.push(s("circle", { cx: p[0], cy: p[1], r: j === 0 && pts.length >= 3 ? r * 1.4 : r * 0.8, fill: j === 0 ? col : "#fff", stroke: "#000", "stroke-width": sw, "pointer-events": "none" })));
    }
    if (arrow?.end) nodes.push(s("line", { x1: arrow.start[0], y1: arrow.start[1], x2: arrow.end[0], y2: arrow.end[1], stroke: "#fff", "stroke-width": sw * 2.5, "marker-end": "url(#arrowhead)" }));
    if (seg?.start && seg?.end && dist(seg.start, seg.end) > 0)
      nodes.push(...segNodes([seg.start, seg.end], seg.k === "service" ? SERVICE_COL : LINE_COL, null, r, sw, fs, { dash: seg.k === "service", arrow: seg.k === "line", inLabel: "in", outLabel: "out" }));
    replace(overlay, nodes);
    box.classList.toggle("drawing", !!drawing || !!arrow || !!seg);
    hint.textContent = ro ? "Read-only view." : drawing ? `Drawing ${drawing.k}: click to add points (${drawing.pts.length}) · double-click, Enter or click the first point to close · Backspace undoes · Esc cancels`
      : arrow ? "Drag inside the image from where people should come from to where they should go · Esc cancels"
      : seg?.k === "line" ? "Drag from one end of the counting line to the other. People crossing towards the arrow (right-hand side looking from start to end) count as \"in\" · Esc cancels"
      : seg?.k === "service" ? "Drag the queue's service line (where people leave the queue, e.g. the counter) · Esc cancels"
      : sel?.k === "line" ? "Drag the line or its end points · Flip swaps in/out · Delete removes the line"
      : sel ? "Drag points or the shape · drag a small midpoint to insert a point · right-click a point to delete it · Delete removes the shape"
      : "Select a zone, mask or line, or add a new one.";
    const dirty = anyDirty();
    status.textContent = dirty ? "● unsaved changes" : "";
    saveBtn.disabled = !dirty; revertBtn.disabled = !dirty;
  }

  function renderList() {
    const item = (k, i, label, sub, col) => h("button", { class: ["item", sel?.k === k && sel.i === i && "on"], onclick: () => select(k, i) },
      h("div", { class: "row", style: { gap: "6px" } }, h("span", { class: "swatch", style: { background: col } }), h("b", null, label)),
      sub ? h("div", { class: "small mute" }, sub) : null);
    replace(listEl,
      h("h3", { style: { marginTop: 0 } }, `Zones (${zones.length})`),
      zones.length ? zones.map((z, i) => item("zone", i, z.name,
        (Object.keys(z.rules).map(r => RULE[r] || r).join(", ") || "no rules") + (z.schedule ? ` · scheduled (${z.schedule.label || "windows"})` : "")
          + (z.polygon_px ? "" : " · metres (config)"), CRIT[z.criticality][1]))
        : h("div", { class: "mute small" }, "No zones yet."),
      h("h3", null, `Detection masks (${masks.length})`),
      masks.length ? masks.map((m, i) => item("mask", i, `Mask ${i + 1}`, `${m.length} points`, MASK_COL))
        : h("div", { class: "mute small" }, "No masks. Masks hide static false positives (e.g. a bollard detected as a person)."),
      h("h3", null, `Privacy masks (${privacy.length})`),
      privacy.length ? privacy.map((m, i) => item("privacy", i, `Privacy mask ${i + 1}`, `${m.length} points · blanked before storage`, "#000"))
        : h("div", { class: "mute small" }, "None. A privacy mask blacks out e.g. a neighbour's window before the frame is analysed, recorded or shown."),
      h("h3", null, `Counting lines (${lines.length})`),
      lines.length ? lines.map((l, i) => item("line", i, l.name, (l.points_px ? `${l.in_label} / ${l.out_label}` : "metres (config)")
        + (l.events ? ` · events: ${l.events.direction}${l.events.schedule ? " (scheduled)" : ""}` : ""), LINE_COL))
        : h("div", { class: "mute small" }, "No lines. A line counts people crossing it in each direction (works without calibration)."));
  }

  function numInput(val, onchange, attrs = {}) {
    const inp = h("input", { type: "number", value: val, step: "any", disabled: ro, ...attrs });
    inp.addEventListener("input", () => { if (inp.value !== "" && isFinite(+inp.value)) onchange(+inp.value); });
    return inp;
  }

  // Schedule editor shared by zones and line events: get()/set(sched|null). The status line asks the server
  // (/api/schedule/check) whether the text is valid and whether it is active right now.
  let schedTimer = null;
  function scheduleBox(get, set, onToggle, help) {
    const sc = get();
    const cb = h("input", { type: "checkbox", checked: !!sc, disabled: ro });
    const winIn = h("input", { value: sc ? sc.windows.join("; ") : "", placeholder: SCHED_EXAMPLE, disabled: ro,
      "aria-label": "Schedule windows", class: "mono", style: { minWidth: "18rem" } });
    const tzIn = h("input", { value: sc?.tz || "", placeholder: "Asia/Kolkata (blank = server time)", disabled: ro, "aria-label": "Time zone" });
    const labIn = h("input", { value: sc?.label || "", placeholder: "after-hours", maxlength: 40, disabled: ro, "aria-label": "Schedule label" });
    const msg = h("div", { class: "small", style: { marginTop: "4px" } });
    const read = () => ({ windows: winIn.value.split(";").map(x => x.trim()).filter(Boolean), tz: tzIn.value, label: labIn.value });
    function check() {
      clearTimeout(schedTimer);
      const cur = get();
      if (!cur) { replace(msg, h("span", { class: "mute" }, "No schedule: the rules are always armed.")); return; }
      const local = schedErrors(cur, "Schedule");
      if (local.length) { replace(msg, h("span", { style: { color: "var(--warn)" } }, local[0], ` Example: ${SCHED_EXAMPLE}`)); return; }
      schedTimer = setTimeout(async () => {
        try {
          const r = await api("/api/schedule/check", { method: "POST", body: { schedule: schedPayload(cur) }, silent: true, feature: "schedule" });
          if (!r.ok) replace(msg, h("span", { style: { color: "var(--warn)" } }, r.error));
          else replace(msg, r.active
            ? h("span", { class: "h-OK" }, `Active now (${r.window}, ${r.local_time} ${r.tz})`)
            : h("span", { class: "mute" }, `Inactive now (${r.local_time} ${r.tz}): rules stay disarmed until a window opens.`),
            r.tz_resolution && r.tz_resolution.includes("no DST") ? h("div", { class: "mute" }, r.tz_resolution) : null);
        } catch { replace(msg, h("span", { class: "mute" }, "Schedule check unavailable; the server validates on save.")); }
      }, 350);
    }
    const onInput = () => { if (cb.checked) { set(read()); onToggle(false); check(); } };
    for (const el of [winIn, tzIn, labIn]) el.addEventListener("input", onInput);
    cb.addEventListener("change", () => {
      if (cb.checked && !winIn.value.trim()) winIn.value = SCHED_EXAMPLE;
      set(cb.checked ? read() : null); onToggle(true);
    });
    const fields = sc ? h("div", { class: "params" },
      h("label", null, "Windows (separate with ;)", winIn), h("label", null, "Time zone", tzIn), h("label", null, "Label", labIn)) : null;
    check();
    return h("div", { class: "rule-box" }, h("label", { class: "h" }, cb, "Schedule (arm only inside time windows)"), fields,
      sc ? msg : null,
      h("div", { class: "note" }, help || "Overnight windows wrap: \"mon-fri 19:00-07:00\" runs Mon 19:00 → Tue 07:00 … Fri 19:00 → Sat 07:00. Events name the active window."));
  }

  function renderInspector() {
    if (!sel) { replace(insp); return; }
    if (sel.k === "privacy") {
      replace(insp, h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, `Privacy mask ${sel.i + 1}`),
        h("div", { class: "small mute" }, "These pixels are painted black on every frame right after capture, so detection, recordings, evidence snapshots and clips, the live view, the VLM and exports never see them. Nothing inside can raise an alert. Earlier recordings are not rewritten."),
        ro ? null : h("div", { class: "row", style: { marginTop: "10px" } }, h("button", { class: "danger", onclick: () => removeSel() }, icon("trash"), "Delete privacy mask"))));
      return;
    }
    if (sel.k === "mask") {
      replace(insp, h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, `Mask ${sel.i + 1}`),
        h("div", { class: "small mute" }, "Detections whose foot point falls inside a mask are ignored (Frigate-style). Use sparingly: masked areas are blind to every rule."),
        ro ? null : h("div", { class: "row", style: { marginTop: "10px" } }, h("button", { class: "danger", onclick: () => removeSel() }, icon("trash"), "Delete mask"))));
      return;
    }
    if (sel.k === "line") {
      const l = lines[sel.i];
      const txt = (key, label, attrs = {}) => {
        const inp = h("input", { value: l[key], disabled: ro, "aria-label": label, ...attrs });
        inp.addEventListener("input", () => {
          l[key] = key === "id" ? (slug(inp.value) || inp.value) : inp.value;
          touchLines(); renderList();
        });
        return h("label", { class: "field" }, label, inp);
      };
      // opt-in line_cross events
      const evCb = h("input", { type: "checkbox", checked: !!l.events, disabled: ro });
      evCb.addEventListener("change", () => {
        l.events = evCb.checked ? { direction: "in", min_gap_s: 5, criticality: 1, schedule: null } : null;
        touchLines(); renderInspector();
      });
      let evBody = null;
      if (l.events) {
        const ev = l.events;
        const dirSel = h("select", { disabled: ro, "aria-label": "Event direction" },
          [["in", `In only (${l.in_label})`], ["out", `Out only (${l.out_label})`], ["both", "Both directions"]]
            .map(([v, t]) => h("option", { value: v, selected: ev.direction === v }, t)));
        dirSel.addEventListener("change", () => { ev.direction = dirSel.value; touchLines(); });
        const critSel = h("select", { disabled: ro, "aria-label": "Event criticality" },
          [1, 2, 3].map(v => h("option", { value: v, selected: ev.criticality === v }, `${v} · ${CRIT[v][0]}`)));
        critSel.addEventListener("change", () => { ev.criticality = +critSel.value; touchLines(); });
        evBody = [
          h("div", { class: "params" },
            h("label", null, "Direction", dirSel),
            h("label", null, "Min. gap between events (s)", numInput(ev.min_gap_s, v => { ev.min_gap_s = v; touchLines(); }, { min: 0, step: 1 })),
            h("label", null, "Criticality", critSel)),
          scheduleBox(() => ev.schedule, v => { ev.schedule = v; }, full => { touchLines(); if (full) renderInspector(); },
            "E.g. after-hours door entries: direction \"in\" + \"mon-fri 19:00-07:00; sat,sun 00:00-24:00\". Crossings outside the window are still counted, just not raised."),
        ];
      }
      const evBox = h("div", { class: "rule-box" },
        h("label", { class: "h" }, evCb, "Raise an event on each crossing (line_cross)"), evBody,
        h("div", { class: "note" }, "Crossings inside the min. gap are counted and reported with the next event, so a group walking through raises one alert, not one per person."));
      replace(insp, h("div", { class: "panel" },
        h("h3", { style: { marginTop: 0 } }, "Counting line"),
        h("div", { class: "fields" }, txt("name", "Name"), txt("id", "Id", { class: "mono" }),
          txt("in_label", "\"In\" label (towards the arrow)", { maxlength: 40 }), txt("out_label", "\"Out\" label", { maxlength: 40 })),
        l.points_px ? null : h("div", { class: "banner" }, "This line is defined in ground metres in the site config; it can be renamed here but not reshaped."),
        h("div", { class: "note" }, "Counts person tracks whose foot point crosses the segment (not its extension) by more than a small dead band, so standing on the line does not count. Tracker ID switches at the line can miss a crossing."),
        evBox,
        ro ? null : h("div", { class: "row", style: { marginTop: "10px" } },
          l.points_px ? h("button", { onclick: () => { l.points_px.reverse(); touchLines(); renderInspector(); } }, "Flip in/out") : null,
          h("button", { class: "danger", onclick: () => removeSel() }, icon("trash"), "Delete line"))));
      return;
    }
    const z = zones[sel.i];
    const nameIn = h("input", { value: z.name, disabled: ro, "aria-label": "Zone name" });
    nameIn.addEventListener("input", () => { z.name = nameIn.value; if (z._autoId) z.id = slug(z.name) || z.id; idIn.value = z.id; touchZones(false); renderList(); });
    const idIn = h("input", { value: z.id, disabled: ro, "aria-label": "Zone id", class: "mono" });
    idIn.addEventListener("input", () => { z.id = slug(idIn.value) || idIn.value; z._autoId = false; touchZones(false); });
    const crit = h("select", { disabled: ro, "aria-label": "Criticality" }, [1, 2, 3].map(v => h("option", { value: v, selected: z.criticality === v }, `${v} · ${CRIT[v][0]}`)));
    crit.addEventListener("change", () => { z.criticality = +crit.value; touchZones(false); renderList(); });
    const site = h("input", { value: z.site_zone, disabled: ro, placeholder: "optional", "aria-label": "Site zone" });
    site.addEventListener("input", () => { z.site_zone = site.value.trim(); touchZones(false); });
    const rules = EDITABLE_RULES.map(r => {
      const on = !!z.rules[r];
      const cb = h("input", { type: "checkbox", checked: on, disabled: ro });
      const params = h("div", { class: "params" });
      const fillParams = () => replace(params, z.rules[r] ? Object.entries(RULE_PARAMS[r]).map(([p, [def, label]]) =>
        h("label", null, label, numInput(z.rules[r][p] ?? def, v => { z.rules[r][p] = v; touchZones(false); },
          p === "cos_threshold" ? { min: -1, max: 0, step: 0.05 } : FRACTION_PARAMS.has(p) ? { min: 0, max: 1, step: 0.05 }
            : { min: 0, step: INT_PARAMS.has(p) ? 1 : 0.1 }))) : null);
      cb.addEventListener("change", () => {
        if (cb.checked) z.rules[r] = Object.fromEntries(Object.entries(RULE_PARAMS[r]).map(([p, [def]]) => [p, def]));
        else delete z.rules[r];
        if (r === "queue_long" && cb.checked) z.type = "queue";
        fillParams(); touchZones(false); renderList(); renderInspector();
      });
      fillParams();
      const needsCal = METRIC_RULES.has(r) && cam.calibrated === false && z.rules[r]
        ? h("div", { class: "small", style: { color: "var(--warn)", marginTop: "4px" } }, "Needs a calibrated camera (metric speed, direction or area): skipped until calibrated.")
        : VEHICLE_RULES.has(r) && z.rules[r] && cam.vehicles_enabled === false
          ? h("div", { class: "small", style: { color: "var(--warn)", marginTop: "4px" } }, "Needs vehicle detection: set \"vehicles\": {\"enabled\": true} in the site config (car, motorcycle, bus, truck).")
          : null;
      const ownSched = z.rules[r]?.schedule ? h("div", { class: "small mute", style: { marginTop: "4px" } },
        `Own schedule from config (replaces the zone schedule): ${(normSched(z.rules[r].schedule)?.windows || []).join("; ")}`) : null;
      const extra = r === "wrong_way" && z.rules[r] ? h("div", { class: "row small", style: { marginTop: "6px" } },
        z.direction_px ? h("span", { class: "h-OK" }, "direction drawn") : z.direction ? h("span", { class: "mute" }, `ground direction (${z.direction.join(", ")}) from config`)
          : h("span", { style: { color: "var(--warn)" } }, "no permitted direction yet"),
        ro ? null : h("button", { onclick: () => { arrow = { start: null, end: null }; render(); } }, z.direction_px ? "Redraw arrow" : "Draw arrow"),
        !ro && z.direction_px ? h("button", { class: "ghost", onclick: () => { z.direction_px = null; touchZones(); renderInspector(); } }, "Clear") : null) : null;
      return h("div", { class: "rule-box" }, h("label", { class: "h" }, cb, RULE[r] || r), params, extra, needsCal, ownSched);
    });
    const typeSel = h("select", { disabled: ro, "aria-label": "Zone type" },
      h("option", { value: "area", selected: z.type !== "queue" }, "Area"), h("option", { value: "queue", selected: z.type === "queue" }, "Queue"));
    typeSel.addEventListener("change", () => {
      z.type = typeSel.value;
      if (z.type !== "queue") delete z.rules.queue_long;
      touchZones(); renderInspector();
    });
    const queueBox = z.type === "queue" ? h("div", { class: "rule-box" },
      h("div", { class: "row small" },
        h("b", null, "Service line"),
        z.service_line_px ? h("span", { class: "h-OK" }, "drawn") : z.service_line ? h("span", { class: "mute" }, "ground metres (config)")
          : h("span", { class: "mute" }, "none: departures = people leaving the zone"),
        ro ? null : h("button", { onclick: () => startSeg("service") }, z.service_line_px ? "Redraw" : "Draw"),
        !ro && z.service_line_px ? h("button", { class: "ghost", onclick: () => { z.service_line_px = null; touchZones(); renderInspector(); } }, "Clear") : null),
      h("div", { class: "note" }, "Queue length = people in the zone. Wait time is a Little's-law estimate (mean length ÷ departure rate across the service line), valid only while the queue is steady; per-person entry→service times are shown as measured.")) : null;
    replace(insp, h("div", { class: "panel" },
      h("h3", { style: { marginTop: 0 } }, "Zone"),
      h("div", { class: "fields" },
        h("label", { class: "field" }, "Name", nameIn), h("label", { class: "field" }, "Id", idIn),
        h("label", { class: "field" }, "Criticality", crit),
        h("label", { class: "field", title: "Shared id for zones seen by several cameras (spatial corroboration only)" }, "Site zone", site),
        h("label", { class: "field", title: "Queue zones also report queue length, departures and wait estimates" }, "Type", typeSel)),
      queueBox,
      scheduleBox(() => z.schedule, v => { z.schedule = v; }, full => { touchZones(false); if (full) { renderList(); renderInspector(); } }),
      h("h4", null, "Rules (deterministic, observable behaviour only)"), rules,
      h("div", { class: "note" }, "Rules fire only after the condition holds for the minimum time. Thresholds are starting points, not tuned values."),
      ro ? null : h("div", { class: "row", style: { marginTop: "10px" } }, h("button", { class: "danger", onclick: () => removeSel() }, icon("trash"), "Delete zone"))));
  }

  // ---------- state changes ----------
  function touchZones(rerender = true) { zonesDirty = true; if (rerender) { render(); renderList(); } else render(); }
  function touchLines() { linesDirty = true; render(); }
  function touch(k) { if (k === "mask") masksDirty = true; else if (k === "privacy") privacyDirty = true; else if (k === "line") linesDirty = true; else zonesDirty = true; render(); }
  function select(k, i) { sel = k ? { k, i } : null; drawing = null; arrow = null; seg = null; render(); renderList(); renderInspector(); }
  function startDraw(k) { sel = null; arrow = null; seg = null; drawing = { k, pts: [], hover: null }; render(); renderList(); renderInspector(); }
  function startSeg(k) {  // k = "line" (new counting line) | "service" (service line of the selected queue zone)
    if (k === "line") { sel = null; renderList(); renderInspector(); }
    drawing = null; arrow = null; seg = { k, start: null, end: null }; render();
  }
  function finishSeg() {
    const { k, start, end } = seg;
    seg = null;
    if (!start || !end || dist(start, end) <= hr() * 2) { render(); return; }
    if (k === "service" && sel?.k === "zone") {
      zones[sel.i].service_line_px = [start, end]; touchZones(); renderInspector(); return;
    }
    if (k === "line") {
      let n = lines.length + 1, id = `line_${n}`;
      while (lines.some(l => l.id === id)) id = `line_${++n}`;
      lines.push({ id, name: `Line ${n}`, points_px: [start, end], points: null, in_label: "in", out_label: "out" });
      linesDirty = true; select("line", lines.length - 1); return;
    }
    render();
  }
  function finishDraw() {
    if (!drawing) return;
    const pts = drawing.pts;
    // drop trailing duplicates produced by the double-click's own clicks
    while (pts.length > 1 && dist(pts[pts.length - 1], pts[pts.length - 2]) < hr()) pts.pop();
    if (pts.length < 3) { toast("A polygon needs at least 3 points.", "warn"); return; }
    const k = drawing.k;
    drawing = null;
    if (k === "mask") { masks.push(pts); masksDirty = true; select("mask", masks.length - 1); }
    else if (k === "privacy") { privacy.push(pts); privacyDirty = true; select("privacy", privacy.length - 1); }
    else {
      let n = zones.length + 1, id = `zone_${n}`;
      while (zones.some(z => z.id === id)) id = `zone_${++n}`;
      zones.push({ id, name: `Zone ${n}`, criticality: 1, polygon_px: pts, polygon: null, direction: null, direction_px: null,
        site_zone: "", rules: { intrusion: { min_seconds: 1.0 } }, _autoId: true });
      zonesDirty = true; select("zone", zones.length - 1);
    }
  }
  function removeSel() {
    if (!sel) return;
    if (sel.k === "zone") { zones.splice(sel.i, 1); zonesDirty = true; }
    else if (sel.k === "line") { lines.splice(sel.i, 1); linesDirty = true; }
    else if (sel.k === "privacy") { privacy.splice(sel.i, 1); privacyDirty = true; }
    else { masks.splice(sel.i, 1); masksDirty = true; }
    select(null); renderBanner();
  }
  function revert() {
    [zones, masks, lines, privacy] = JSON.parse(orig);
    zonesDirty = masksDirty = linesDirty = privacyDirty = false; select(null); renderBanner();
  }

  // ---------- pointer ----------
  svg.addEventListener("pointerdown", e => {
    if (ro || e.button === 2) return;
    const p = pt(e);
    if (drawing) {
      if (drawing.pts.length >= 3 && dist(p, drawing.pts[0]) < hr() * 1.6) { finishDraw(); return; }
      drawing.pts.push(p); render(); return;
    }
    if (arrow) { arrow = { start: p, end: null }; svg.setPointerCapture(e.pointerId); return; }
    if (seg) { seg.start = p; seg.end = null; svg.setPointerCapture(e.pointerId); return; }
    const t = e.target, role = t.getAttribute?.("data-role");
    if (sel && role === "vtx") {
      drag = { type: "v", j: +t.getAttribute("data-j") };
    } else if (sel && role === "mid") {
      const pts = polyOf(sel.k, sel.i), j = +t.getAttribute("data-j") + 1;
      pts.splice(j, 0, p); drag = { type: "v", j, moved: true }; touch(sel.k);
    } else if (t.getAttribute?.("data-k")) {
      const k = t.getAttribute("data-k"), i = +t.getAttribute("data-i");
      if (!sel || sel.k !== k || sel.i !== i) select(k, i);
      const pts = polyOf(k, i);
      if (pts) drag = { type: "move", start: p, orig: clone(pts) };
    } else { if (sel) select(null); return; }
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", e => {
    if (drawing) { drawing.hover = pt(e); render(); return; }
    if (arrow?.start) { arrow.end = pt(e); render(); return; }
    if (seg?.start) { seg.end = pt(e); render(); return; }
    if (!drag || !sel) return;
    const p = pt(e), pts = polyOf(sel.k, sel.i);
    if (!pts) return;
    if (drag.type === "v") { pts[drag.j] = p; drag.moved = true; }
    else {
      const dx = p[0] - drag.start[0], dy = p[1] - drag.start[1];
      if (!drag.moved && Math.hypot(dx, dy) < hr() / 2) return;
      drag.moved = true;
      drag.orig.forEach((q, j) => { pts[j] = [Math.round(clamp(q[0] + dx, 0, W)), Math.round(clamp(q[1] + dy, 0, H))]; });
    }
    touch(sel.k);
  });
  const endDrag = () => {
    if (seg?.start) { finishSeg(); return; }
    if (arrow?.start) {
      if (arrow.end && dist(arrow.start, arrow.end) > hr() * 2 && sel?.k === "zone") {
        zones[sel.i].direction_px = [arrow.start, arrow.end];
        touchZones(); renderInspector();
      }
      arrow = null; render(); return;
    }
    if (drag) { drag = null; renderList(); }
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("pointerleave", () => { if (drawing) { drawing.hover = null; render(); } });
  svg.addEventListener("dblclick", e => { if (drawing) { e.preventDefault(); finishDraw(); } });
  svg.addEventListener("contextmenu", e => {
    if (ro || !sel || sel.k === "line" || e.target.getAttribute?.("data-role") !== "vtx") return;
    e.preventDefault();
    const pts = polyOf(sel.k, sel.i);
    if (pts.length <= 3) { toast("A polygon needs at least 3 points (delete the shape instead).", "warn"); return; }
    pts.splice(+e.target.getAttribute("data-j"), 1); touch(sel.k);
  });

  // ---------- save ----------
  function validate() {
    const errs = [], ids = new Set();
    for (const z of zones) {
      if (!z.name.trim()) errs.push("Every zone needs a name.");
      if (!z.id || ids.has(z.id)) errs.push(`Zone id "${z.id}" is empty or duplicated.`);
      ids.add(z.id);
      if (z.polygon_px && z.polygon_px.length < 3) errs.push(`Zone "${z.name}" needs at least 3 points.`);
      if (z.rules.wrong_way && !z.direction_px && !z.direction) errs.push(`Zone "${z.name}": wrong-way needs a permitted direction arrow.`);
      errs.push(...schedErrors(z.schedule, `Zone "${z.name}"`));
      for (const [r, p] of Object.entries(z.rules)) for (const [k, v] of Object.entries(p || {})) {
        if (k === "schedule") continue;  // per-rule schedule from the config: validated by the server
        if (!isFinite(+v) || (k !== "cos_threshold" && +v < 0)) errs.push(`Zone "${z.name}": ${RULE[r] || r} ${k} must be a non-negative number.`);
        else if (FRACTION_PARAMS.has(k) && +v > 1) errs.push(`Zone "${z.name}": ${RULE[r] || r} ${k} must be between 0 and 1.`);
      }
    }
    const lids = new Set();
    for (const l of lines) {
      if (!l.name.trim()) errs.push("Every counting line needs a name.");
      if (!/^[A-Za-z0-9_-]{1,40}$/.test(l.id) || lids.has(l.id)) errs.push(`Line id "${l.id}" is invalid or duplicated.`);
      lids.add(l.id);
      if (l.events) {
        if (!(+l.events.min_gap_s >= 0)) errs.push(`Line "${l.name}": the min. gap must be a non-negative number of seconds.`);
        errs.push(...schedErrors(l.events.schedule, `Line "${l.name}"`));
      }
    }
    return errs;
  }
  function payloadLines() {
    return lines.map(l => {
      const o = { id: l.id, name: l.name.trim(), in_label: l.in_label.trim() || "in", out_label: l.out_label.trim() || "out" };
      if (l.points_px) o.points_px = l.points_px; else if (l.points) o.points = l.points;
      if (l.band != null) o.band = l.band;
      if (l.events) {
        o.events = { direction: l.events.direction, min_gap_s: +l.events.min_gap_s, criticality: l.events.criticality };
        if (l.events.schedule) o.events.schedule = schedPayload(l.events.schedule);
      }
      return o;
    });
  }
  function payloadZones() {
    return zones.map(z => {
      const o = { id: z.id, name: z.name.trim(), criticality: z.criticality, rules: {} };
      for (const [r, p] of Object.entries(z.rules)) o.rules[r] = Object.fromEntries(Object.entries(p || {}).map(([k, v]) => [k, k === "schedule" ? v : +v]));
      if (z.schedule) o.schedule = schedPayload(z.schedule);
      if (z.polygon_px) o.polygon_px = z.polygon_px; else if (z.polygon) o.polygon = z.polygon;
      if (z.direction_px) o.direction_px = z.direction_px;
      if (z.direction && !z.direction_px) o.direction = z.direction;
      if (z.site_zone) o.site_zone = z.site_zone;
      if (z.type === "queue") {
        o.type = "queue";
        if (z.service_line_px) o.service_line_px = z.service_line_px;
        else if (z.service_line) o.service_line = z.service_line;
      }
      return o;
    });
  }
  async function save() {
    const errs = validate();
    if (errs.length) { toast(h("div", null, h("b", null, "Cannot save:"), errs.slice(0, 5).map(e => h("div", null, e))), "warn", { timeout: 9000 }); return; }
    saveBtn.disabled = true;
    try {
      if (zonesDirty) {
        await api(`/api/cameras/${enc(cam.id)}/zones`, { method: "PUT", body: { zones: payloadZones() }, feature: "zones" });
        zonesDirty = false;
      }
      if (masksDirty) {
        await api(`/api/cameras/${enc(cam.id)}/masks`, { method: "PUT", body: { masks_px: masks }, feature: "masks" });
        masksDirty = false;
      }
      if (linesDirty) {
        await api(`/api/cameras/${enc(cam.id)}/lines`, { method: "PUT", body: { lines: payloadLines() }, feature: "lines" });
        linesDirty = false;
      }
      if (privacyDirty) {
        await api(`/api/cameras/${enc(cam.id)}/privacy_masks`, { method: "PUT", body: { privacy_masks_px: privacy }, feature: "privacy_masks" });
        privacyDirty = false;
      }
      zones.forEach(z => { delete z._autoId; });
      orig = JSON.stringify([zones, masks, lines, privacy]);
      toast("Saved. The camera's analytics pick up the new zones/masks/lines on the server.", "ok");
      loadConfig().catch(() => {});
    } catch (e) {
      if (e instanceof ApiError && e.kind === "missing") toast("Zone/mask/line editing is not available on this server yet (endpoint missing). Your drawing is kept in this page until you leave.", "warn", { timeout: 10000 });
    } finally { render(); }
  }

  render(); renderList(); renderBanner(); loadImage();
  const onResize = () => render();
  window.addEventListener("resize", onResize);

  return {
    el,
    dirty: anyDirty,
    destroy() { window.removeEventListener("resize", onResize); },
    onKey(e) {
      if (ro) return false;
      if (drawing) {
        if (e.key === "Enter") { finishDraw(); return true; }
        if (e.key === "Escape") { drawing = null; render(); return true; }
        if (e.key === "Backspace" && !isTyping(e)) { drawing.pts.pop(); render(); return true; }
        return false;
      }
      if (arrow && e.key === "Escape") { arrow = null; render(); return true; }
      if (seg && e.key === "Escape") { seg = null; render(); return true; }
      if (isTyping(e)) return false;
      if (sel && (e.key === "Delete" || e.key === "Backspace")) { removeSel(); return true; }
      if (sel && e.key === "Escape") { select(null); return true; }
      return false;
    },
  };
}
