// Overview (default landing): KPI tiles, the most relevant live cameras, 24 h alert trend, hotspots, recent activity
// and system health. Uses only existing endpoints (/api/alerts, /api/stats, /api/vlm, optional /api/handoff/metrics,
// /api/incidents and /api/nodes when the backend provides them) and degrades gracefully when they are absent.
// Counts are measured on this deployment; nothing here is a capacity claim.
import { api, optional } from "./api.js";
import { S, on, camName, camById, isMono, camState } from "./state.js";
import { hasEndpoint, hasPage } from "./registry.js";
import * as drawer from "./drawer.js";
import { h, icon, replace, enc, fmtAgo, fmtMs, num, RULE, OPEN_STATUSES, PRIO, debounce, BLANK } from "./util.js";
import { pageHeader, emptyState, skeleton, prioChip, statusPill, obsBadge, infBadge, estBadge } from "./ui.js";
import { sparkline, stackedBars, gauge, stackBar, hbars } from "./charts.js";
import { statusSnapshot, camsSummary, isDemo } from "./shell.js";

const PCOL = { P1: "var(--p1)", P2: "var(--p2)", P3: "var(--p3)", P4: "var(--p4)" };
const PRIOS = ["P1", "P2", "P3", "P4"];
const HOUR = 3600;

// People-tracked history, sampled from the WebSocket ticks every 5 s from page load (client-side only).
const peopleHist = [];
let lastSample = 0;
on("tick", () => {
  const now = Date.now();
  if (now - lastSample < 5000) return;
  lastSample = now;
  peopleHist.push(S.tracks.filter(t => (t.cls || "person") === "person").length);
  if (peopleHist.length > 120) peopleHist.shift();
});

export default function createOverview() {
  const kpiEl = h("div", { class: "kpis" }, skeleton("kpi", 5));
  const camsBody = h("div", { class: "mini-grid" }, skeleton("tile", 4));
  const trendBody = h("div", null, skeleton("line", 4));
  const zonesBody = h("div", null, skeleton("line", 3));
  const feedBody = h("div", { class: "feed" }, skeleton("row", 5));
  const healthBody = h("div", null, skeleton("line", 4));
  const subtitle = h("span");
  const el = h("section", { class: "page" },
    pageHeader({ title: "Overview", subtitle,
      actions: [h("a", { class: "btnlink", href: "#/review?status=new" }, icon("review"), "Triage queue"),
        h("a", { class: "btnlink primary", href: "#/live" }, icon("live"), "Open Live wall")] }),
    kpiEl,
    h("div", { class: "ov-layout" },
      h("div", { class: "ov-col" },
        h("div", { class: "panel" }, panelHead("live", "Live · most relevant cameras", "Cameras with open alerts first, then by activity",
          h("a", { class: "btnlink sm ghost", href: "#/live" }, "All cameras", icon("chevright"))), camsBody),
        h("div", { class: "panel" }, panelHead("bell", "Recent activity", null,
          h("a", { class: "btnlink sm ghost", href: "#/review" }, "Review", icon("chevright"))), feedBody)),
      h("div", { class: "ov-col" },
        h("div", { class: "panel" }, panelHead("analytics", "Alert trend · 24 h", "Alerts created per hour, by priority"), trendBody),
        h("div", { class: "panel" }, panelHead("pin", "Hotspots · 24 h", "Zones with the most alert events"), zonesBody),
        h("div", { class: "panel" }, panelHead("system", "System health", "Measured on this deployment",
          h("a", { class: "btnlink sm ghost", href: "#/system" }, "Details", icon("chevright"))), healthBody))),
    h("div", { class: "provenance", style: { marginTop: "var(--s-4)" } }, obsBadge(), "counts come from deterministic rules on observed tracks.",
      infBadge("Learned models"), "only corroborate or explain; they never raise alerts on their own."));

  let shown = false, alerts = null, alertsAt = 0, newList = null, incidents = undefined, handoff = undefined, vlm = null, loading = false;
  const tiles = new Map();      // camId -> {el, img, top, bot, timer}
  let tileOrder = "";
  let pollTimer = null;

  function panelHead(ic, title, sub, action) {
    return h("div", { class: "panel-h" }, h("h3", null, icon(ic), title), sub ? h("span", { class: "sub hide-sm" }, sub) : null, h("span", { class: "grow" }), action || null);
  }

  // ------------------------------------------------------------------------------------------------ data
  const load = async (force = false) => {
    if (loading || (!force && Date.now() - alertsAt < 20000)) return;
    loading = true;
    try {
      alerts = await api("/api/alerts?limit=600", { silent: true });
      alertsAt = Date.now();
    } catch { alerts = alerts || []; }
    try {
      if (incidents === undefined || force) {
        if (hasPage("incidents") && hasEndpoint("/api/incidents")) {
          const r = await optional("/api/incidents?limit=200", "incidents", { silent: true });
          incidents = r === null ? null : (Array.isArray(r) ? r : Array.isArray(r?.incidents) ? r.incidents : Array.isArray(r?.items) ? r.items : null);
        } else incidents = null;
      }
    } catch { incidents = null; }
    try {
      if (handoff === undefined || force) handoff = await optional("/api/handoff/metrics", "handoff", { silent: true });
    } catch { handoff = null; }
    try { if (S.cfg?.explain_enabled) vlm = await api("/api/vlm", { silent: true }); } catch { vlm = null; }
    loading = false;
    if (shown) renderAll();
  };
  const loadSoon = debounce(() => load(true), 2500);

  function stats24(list) {
    const now = Date.now() / 1000, end = Math.ceil(now / HOUR) * HOUR, start = end - 24 * HOUR;
    const buckets = Array.from({ length: 24 }, () => ({ P1: 0, P2: 0, P3: 0, P4: 0 }));
    let last24 = 0, prev24 = 0, oldest = Infinity;
    const zones = new Map();
    for (const a of list) {
      oldest = Math.min(oldest, a.created_at);
      if (a.created_at >= now - 24 * HOUR) last24++;
      else if (a.created_at >= now - 48 * HOUR) prev24++;
      const i = Math.floor((a.created_at - start) / HOUR);
      if (i >= 0 && i < 24) buckets[i][a.priority] = (buckets[i][a.priority] || 0) + 1;
      if (a.created_at >= now - 24 * HOUR) for (const e of a.events || []) {
        const k = `${e.camera_id}\u0000${e.zone_name || e.zone_id || "?"}`;
        const z = zones.get(k) || { cam: e.camera_id, zone: e.zone_name || e.zone_id || "—", n: 0, rules: new Set(), top: "P4" };
        z.n++; z.rules.add(e.rule);
        if (PRIOS.indexOf(a.priority) < PRIOS.indexOf(z.top)) z.top = a.priority;
        zones.set(k, z);
      }
    }
    const capped = list.length >= 600 && oldest > now - 48 * HOUR;
    return { buckets, start, last24, prev24, zones: [...zones.values()].sort((a, b) => b.n - a.n), capped };
  }

  // ------------------------------------------------------------------------------------------------ KPIs
  function kpiTile({ href, ic, label, value, unit, sub, foot, cls, title }) {
    return h(href ? "a" : "div", { class: ["kpi", cls], href, title: title || null },
      h("div", { class: "kpi-l" }, icon(ic), label),
      h("div", { class: "kpi-v" }, value, unit ? h("small", null, unit) : null),
      sub ? h("div", { class: "kpi-sub" }, sub) : null,
      foot ? h("div", { class: "kpi-foot" }, foot) : null);
  }

  function renderKpis() {
    if (!alerts) return;
    const open = alerts.filter(a => OPEN_STATUSES.includes(a.status));
    const byP = Object.fromEntries(PRIOS.map(p => [p, open.filter(a => a.priority === p).length]));
    const newN = newList ? newList.length : open.filter(a => a.status === "new").length;
    const st = stats24(alerts);
    const cs = camsSummary();
    const people = S.tracks.filter(t => (t.cls || "person") === "person");
    const peopleCams = new Set(people.map(t => t.camera)).size;
    const delta = st.last24 - st.prev24;
    const pct = st.prev24 ? Math.round(delta / st.prev24 * 100) : null;
    const hourly = st.buckets.map(b => PRIOS.reduce((a, p) => a + b[p], 0));

    // incidents (endpoint) or correlated alerts (fallback, computed from alert factors / events)
    let incTile;
    if (Array.isArray(incidents)) {
      const openInc = incidents.filter(i => !/^(resolved|closed|dismissed)$/i.test(String(i.status || "open")));
      incTile = kpiTile({ href: "#/incidents", ic: "incidents", label: "Open incidents", value: openInc.length,
        sub: [h("span", null, `${incidents.length} total`), infBadge("correlated")] });
    } else {
      const multi = open.filter(a => {
        const f = a.factors;
        if (f && (f.distinct_rules > 1 || f.distinct_cameras > 1)) return true;
        const ev = a.events || [];
        return new Set(ev.map(e => e.rule)).size > 1 || (a.cameras || []).length > 1;
      });
      incTile = kpiTile({ href: "#/review", ic: "incidents", label: "Correlated alerts", value: multi.length,
        sub: [h("span", null, "≥ 2 behaviours or cameras agree"), infBadge("correlation")],
        title: "Open alerts where more than one behaviour or camera corroborates the same site zone / track (spatial only, no identity claim)." });
    }

    replace(kpiEl,
      kpiTile({ href: "#/review?status=new", ic: "alert", label: "Active alerts", value: open.length, unit: open.length ? `${newN} new` : null,
        cls: byP.P1 ? "alerting" : null,
        sub: h("div", { class: "legend" }, PRIOS.map(p => h("span", { title: PRIO[p].label }, h("i", { style: { background: PCOL[p] } }), `${p} ${byP[p]}`))),
        foot: stackBar(PRIOS.map(p => ({ value: byP[p], color: PCOL[p], label: `${p} ${PRIO[p].label}` })), { label: "Open alerts by priority" }) }),
      incTile,
      kpiTile({ href: "#/live", ic: "cameras", label: "Cameras online", value: cs.online, unit: `/ ${cs.total}`,
        cls: cs.down ? "alerting" : null,
        sub: [cs.degraded ? h("span", { class: "c-warn b" }, `${cs.degraded} degraded`) : null, cs.down ? h("span", { class: "c-err b" }, `${cs.down} down`) : null,
          !cs.degraded && !cs.down ? h("span", null, S.ws === "live" ? "All reporting views healthy" : "Waiting for live stats") : null],
        foot: h("div", { class: "cam-dots", "aria-hidden": "true" }, S.cams.map(c => {
          const k = camState(c.id).state;
          return h("i", { class: { ok: "", warn: "warn", err: "err", off: "off", wait: "off" }[k], title: `${c.name}: ${camState(c.id).label}` });
        })) }),
      kpiTile({ href: "#/map", ic: "user", label: "People tracked now", value: people.length,
        sub: [h("span", null, `on ${peopleCams} camera${peopleCams === 1 ? "" : "s"}`), obsBadge("observed")],
        foot: sparkline(peopleHist.length > 1 ? peopleHist : [people.length, people.length], { width: 200, height: 30, color: "var(--obs)",
          label: `People tracked, last ${Math.round(peopleHist.length * 5 / 60)} min (sampled every 5 s since this page opened)` }),
        title: "Live person tracks across all cameras. Tracks are per camera; the same person seen by two cameras counts twice (no identity linking)." }),
      kpiTile({ href: "#/review", ic: "analytics", label: "Alerts · last 24 h", value: st.last24,
        sub: [st.prev24 || st.last24 ? h("span", { class: ["delta", delta > 0 ? "up" : delta < 0 ? "down" : "flat"] },
          icon(delta > 0 ? "trendup" : delta < 0 ? "trenddown" : "chevright"), pct === null ? (delta > 0 ? `+${delta}` : `${delta}`) : `${delta > 0 ? "+" : ""}${pct}%`) : null,
        h("span", null, `vs ${st.prev24} in the previous 24 h`), st.capped ? estBadge("partial") : null],
        foot: sparkline(hourly, { width: 200, height: 30, color: "var(--acc-text)", label: "Alerts per hour, last 24 h" }) }));
  }

  // ------------------------------------------------------------------------------------------------ cameras
  function rankCams() {
    const open = (alerts || []).filter(a => OPEN_STATUSES.includes(a.status));
    const score = new Map();
    for (const c of S.cams) score.set(c.id, { id: c.id, best: 9, n: 0, last: 0, tracks: S.tracks.filter(t => t.camera === c.id).length });
    for (const a of open) for (const cid of a.cameras || []) {
      const s = score.get(cid);
      if (!s) continue;
      s.n++; s.best = Math.min(s.best, PRIOS.indexOf(a.priority)); s.last = Math.max(s.last, a.updated_at || 0);
    }
    return [...score.values()].sort((a, b) => (a.best - b.best) || (b.last - a.last) || (b.tracks - a.tracks)).slice(0, 4);
  }

  function makeTile(id) {
    const c = camById(id);
    const img = h("img", { alt: `Latest frame from ${c?.name || id}`, src: BLANK });
    const msg = h("div", { class: "msg" }, "loading frame…");
    const top = h("div", { class: "ov top" }), bot = h("div", { class: "ov bot" });
    const el = h("div", { class: "tile", tabIndex: 0, role: "link", "aria-label": `Open live view of ${c?.name || id}`,
      onclick: () => { location.hash = "#/live/" + enc(id); },
      onkeydown: e => { if (e.key === "Enter") location.hash = "#/live/" + enc(id); } }, img, msg, top, bot);
    const t = { id, el, img, msg, top, bot, timer: null, busy: false };
    const pull = () => {
      if (!shown || document.hidden || t.busy) return;
      if (camById(id)?.enabled === false) { msg.textContent = "camera disabled"; msg.hidden = false; return; }
      t.busy = true;
      const pre = new Image();
      pre.onload = () => { t.busy = false; img.src = pre.src; msg.hidden = true; };
      pre.onerror = () => { t.busy = false; msg.textContent = "no frame yet"; msg.hidden = false; };
      pre.src = `/api/cameras/${enc(id)}/latest.jpg?t=${Date.now()}`;
    };
    t.pull = pull;
    return t;
  }

  function renderCams() {
    if (!S.cams.length) {
      tiles.forEach(t => clearInterval(t.timer)); tiles.clear(); tileOrder = "";
      replace(camsBody, h("div", { style: { gridColumn: "1 / -1" } }, emptyState({ icon: "cameras", title: "No cameras yet",
        text: "Add an RTSP camera, NVR channel or video file to start detecting.", action: h("a", { class: "btnlink primary", href: "#/cameras" }, icon("plus"), "Add camera") })));
      return;
    }
    const ranked = rankCams();
    const order = ranked.map(r => r.id).join("|");
    if (order !== tileOrder) {
      tileOrder = order;
      const keep = new Set(ranked.map(r => r.id));
      for (const [id, t] of tiles) if (!keep.has(id)) { clearInterval(t.timer); tiles.delete(id); }
      replace(camsBody, ranked.map(r => {
        let t = tiles.get(r.id);
        if (!t) { t = makeTile(r.id); tiles.set(r.id, t); }
        return t.el;
      }));
      startTiles();
    }
    for (const r of ranked) {
      const t = tiles.get(r.id), c = camById(r.id), s = S.stats[r.id], cs = camState(r.id);
      if (!t) continue;
      const dotCol = { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)", off: "var(--p4)", wait: "var(--p4)" }[cs.state];
      replace(t.top, h("span", { class: ["dot", "vs-" + cs.state], title: cs.label }), h("span", { class: "name" }, c?.name || r.id),
        isMono(r.id) ? h("span", { class: "irb", title: "Monochrome / infrared picture" }, icon("ir"), "IR") : null,
        r.best < 9 ? h("span", { class: ["chip", PRIOS[r.best]], style: { marginLeft: "auto" } }, `${PRIOS[r.best]} · ${r.n}`) : h("span", { class: ["hchip", "h-" + (s?.health || "WARMUP")] }, s?.health || "WARMUP"));
      replace(t.bot, `${r.tracks} track${r.tracks === 1 ? "" : "s"}`, s ? ` · ${num(s.fps, 1)} fps` : "", h("span", { class: "grow" }), cs.state !== "ok" ? cs.label : "");
      t.el.classList.toggle("alerting", r.best === 0);
    }
  }
  function startTiles() {
    for (const t of tiles.values()) {
      clearInterval(t.timer);
      if (!shown) continue;
      t.pull();
      t.timer = setInterval(t.pull, isDemo() ? 1000 : 2000);
    }
  }
  function stopTiles() { for (const t of tiles.values()) { clearInterval(t.timer); t.timer = null; } }

  // ------------------------------------------------------------------------------------------------ trend + zones
  function renderTrend() {
    if (!alerts) return;
    const st = stats24(alerts);
    const total = st.buckets.reduce((a, b) => a + PRIOS.reduce((x, p) => x + b[p], 0), 0);
    const titles = st.buckets.map((b, i) => {
      const t0 = new Date((st.start + i * HOUR) * 1000);
      return `${t0.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}: ` + PRIOS.map(p => `${p} ${b[p]}`).join(" · ");
    });
    const axis = [0, 6, 12, 18, 24].map(i => new Date((st.start + i * HOUR) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    replace(trendBody,
      total ? [
        h("div", { class: "row", style: { marginBottom: ".5rem" } }, h("span", { class: "kpi-v", style: { fontSize: "var(--fs-xl)" } }, total, h("small", null, "alerts")),
          h("span", { class: "grow" }),
          h("div", { class: "legend" }, PRIOS.map(p => h("span", null, h("i", { style: { background: PCOL[p] } }), `${p} ${PRIO[p].label}`)))),
        stackedBars(st.buckets, ["P4", "P3", "P2", "P1"], { width: 240, height: 64, colors: PCOL, label: "Alerts per hour by priority, last 24 hours", titles,
          onbar: i => { const f = Math.round(st.start + i * HOUR); location.hash = `#/review?from=${f}&to=${f + HOUR}`; } }),
        h("div", { class: "trend-axis", "aria-hidden": "true" }, axis.map(a => h("span", null, a))),
        h("div", { class: "note" }, "Click a bar to review that hour.", st.capped ? " Based on the 600 most recently updated alerts." : "")]
        : emptyState({ icon: "check", title: "Quiet day", text: "No alerts were created in the last 24 hours.", compact: true }));
    const zs = st.zones.slice(0, 5);
    replace(zonesBody, zs.length ? hbars(zs.map(z => ({
      label: z.zone, value: z.n, display: `${z.n} event${z.n === 1 ? "" : "s"}`, color: PCOL[z.top],
      sub: `${camName(z.cam)} · ${[...z.rules].map(r => RULE[r] || r).join(", ")}`,
      title: `Highest priority among these alerts: ${z.top}` })))
      : emptyState({ icon: "pin", title: "No hotspots", text: "Zones with alert events in the last 24 h appear here.", compact: true }));
  }

  // ------------------------------------------------------------------------------------------------ feed
  function renderFeed() {
    if (!alerts) return;
    const list = [...alerts].sort((a, b) => b.updated_at - a.updated_at).slice(0, 7);
    const ids = list.map(a => a.id);
    replace(feedBody, list.length ? list.map((a, i) => {
      const ev = a.events?.[a.events.length - 1] || {};
      return h("button", { class: "feed-i", type: "button", onclick: () => {
        let k = i;
        drawer.setNavigator({ next: () => { if (k < ids.length - 1) { drawer.openAlert(ids[++k]); return true; } return false; },
          prev: () => { if (k > 0) { drawer.openAlert(ids[--k]); return true; } return false; } });
        drawer.openAlert(a.id);
      } },
      prioChip(a.priority),
      h("div", { style: { minWidth: 0 } },
        h("div", { class: "feed-t" }, a.title),
        ev.observation ? h("div", { class: "feed-o" }, ev.observation) : null,
        h("div", { class: "feed-m" }, statusPill(a.status), h("span", null, (a.cameras || []).map(camName).join(", ")),
          (a.events || []).length > 1 ? h("span", null, `· ${a.events.length} events`) : null)),
      h("div", { class: "feed-time" }, fmtAgo(a.updated_at)));
    }) : emptyState({ icon: "review", title: "No alerts yet", text: "Alerts appear here as soon as a rule fires on a camera.", compact: true }));
  }

  // ------------------------------------------------------------------------------------------------ health
  function renderHealth() {
    const snap = statusSnapshot();
    const st = snap.stats;
    const vals = Object.values(S.stats || {});
    const fps = vals.length ? vals.reduce((a, s) => a + (+s.fps || 0), 0) / vals.length : null;
    const det = vals.length ? vals.reduce((a, s) => a + (+s.detect_ms || 0), 0) / vals.length : null;
    const drop = vals.reduce((a, s) => a + (+s.dropped || 0), 0), frames = vals.reduce((a, s) => a + (+s.frames || 0), 0);
    const m = st?.model || {};
    const nodes = snap.nodes ? (Array.isArray(snap.nodes) ? snap.nodes : snap.nodes.nodes) : null;
    const vl = vlm?.models || [];
    const cell = (g, label, value, title) => h("div", { class: "hcell", title: title || null }, g, h("div", null, h("div", { class: "hl" }, label), h("div", { class: "hv" }, value)));
    const icCell = (ic, label, value, title) => cell(h("span", { class: "ki" }, icon(ic)), label, value, title);
    replace(healthBody,
      h("div", { class: "health-grid" },
        cell(gauge(st ? st.cpu_percent / 100 : NaN, { size: 44, stroke: 5, label: "CPU" }), "CPU (host)", st ? `${num(st.cpu_percent, 0)} %` : "—"),
        cell(gauge(st ? st.ram_percent / 100 : NaN, { size: 44, stroke: 5, label: "RAM" }), "Memory (host)", st ? `${num(st.ram_percent, 0)} %` : "—"),
        icCell("signal", "Analytics rate", fps != null ? `${num(fps, 1)} fps / camera` : "—", "Mean analytics frames per second per camera (target " + (S.cfg.analytics_fps || 5) + " fps)"),
        icCell("clock", "Detect time", fmtMs(det), `Mean detector+tracker time per frame${frames ? `; ${drop} of ${frames + drop} frames skipped to stay real-time` : ""}`),
        icCell(nodes ? "nodes" : "cpu", nodes ? "Compute nodes" : "Inference", nodes ? `${nodes.length} node${nodes.length === 1 ? "" : "s"}` : (m.device ? `${String(m.device).toUpperCase()} · ${m.runtime || "torch"}` : "—"),
          m.detector ? `Detector ${m.detector}, imgsz ${m.imgsz ?? "?"}, conf ≥ ${m.conf_threshold ?? "?"}` : null),
        icCell("spark", "VLM explainer", !S.cfg.explain_enabled ? "not configured" : vl.length ? `${vl.filter(x => x.loaded).length}/${vl.length} loaded` : "enabled",
          "On-request assistive descriptions only; never raises, scores or closes alerts."),
        handoff ? icCell("journeys", "Handoff model", handoff.nn?.top1 != null ? `top-1 ${num(handoff.nn.top1, 2)} vs ${num(handoff.baseline?.top1, 2)} base` : "learning",
          "Rolling next-camera accuracy against self-supervised labels (estimate, not ground truth)") : null),
      h("div", { class: "cam-health", role: "table", "aria-label": "Per-camera health" },
        S.cams.map(c => {
          const s = S.stats[c.id], cs = camState(c.id);
          return h("div", { class: "ch-row", role: "row" },
            h("span", { class: "nm", role: "cell" }, h("span", { class: "dot", style: { background: { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)", off: "var(--p4)", wait: "var(--p4)" }[cs.state] } }),
              h("span", { title: c.name }, c.name), isMono(c.id) ? h("span", { class: "tag", title: "Monochrome / infrared picture" }, icon("ir"), "IR") : null),
            h("span", { class: ["small", "h-" + (s?.health || "WARMUP")], role: "cell", title: cs.label }, s?.health || "—"),
            h("span", { class: "small tnum mute", role: "cell" }, s ? `${num(s.fps, 1)} fps` : "—"),
            h("span", { class: "small tnum mute x3", role: "cell" }, s ? fmtMs(s.detect_ms) : "—"));
        })));
  }

  function renderAll() {
    replace(subtitle, `${S.cfg.site || "Site"} · ${S.cams.length} camera${S.cams.length === 1 ? "" : "s"} · analytics ${S.cfg.analytics_fps || 5} fps`);
    renderKpis(); renderCams(); renderTrend(); renderFeed(); renderHealth();
  }

  // live refresh: KPIs + tiles + health on ticks (throttled), alerts on events
  let tickAt = 0;
  on("new-alerts", l => { newList = l; if (shown) renderKpis(); });
  on("cameras", () => { if (shown) { tileOrder = ""; renderAll(); } });
  on("alert-changed", () => { if (shown) loadSoon(); });
  on("demo", () => { if (shown) startTiles(); });

  return {
    el,
    show() {
      shown = true;
      replace(subtitle, `${S.cfg.site || "Site"} · ${S.cams.length} camera${S.cams.length === 1 ? "" : "s"}`);
      if (alerts) renderAll();
      load(!alerts || Date.now() - alertsAt > 20000);
      startTiles();
      clearInterval(pollTimer);
      pollTimer = setInterval(() => { if (shown && !document.hidden) { load(); renderHealth(); } }, 10000);
    },
    hide() { shown = false; stopTiles(); clearInterval(pollTimer); },
    onTick() {
      if (!shown || !alerts || Date.now() - tickAt < 1500) return;
      tickAt = Date.now();
      renderKpis(); renderCams(); renderHealth();
    },
    onAlert() { if (shown) loadSoon(); },
    onResync() { load(true); },
    onVisibility(v) { if (v && shown) startTiles(); else stopTiles(); },
  };
}
