// Incidents: list (lifecycle chips, assignee, merge), detail (evidence graph, timeline, related cameras, predicted
// next camera, alerts, push decisions, audit) and historical hotspots. Grouping is spatial/temporal only: it never
// claims identity. Every server string goes through text nodes (h/s helpers). Degrades when the API is absent.
import { api, optional, safe } from "./api.js";
import { S, can, camName, on } from "./state.js";
import { toast, openDialog, segmented, featureMissing } from "./ui.js";
import * as drawer from "./drawer.js";
import { forceGraph } from "./graph.js";
import { createReplay } from "./replay.js";
import { h, s, replace, enc, mediaUrl, fmtTime, fmtDateTime, fmtAgo, fmtDur, RULE, PRIORITIES, DAYS, howLabel, debounce } from "./util.js";

const PHASES = ["open", "active", "cooling", "closed"];
const ISTATUS = ["open", "acknowledged", "investigating", "resolved", "closed"];
const PHASE_HELP = {
  open: "Held: a burst below P2 settles for 10 s before any push decision",
  active: "Receiving events", cooling: "No new event for 60 s",
  closed: "Closed (a matching alert within 10 min reopens it)",
};
const DECISION = { would_push: "would push", budget_denied: "budget denied", suppressed: "suppressed", not_push_tier: "console only",
  webhook_sent: "webhook sent", webhook_failed: "webhook failed" };
const HO_NOTE = "Estimate only: next camera and ETA from exit motion + learned camera topology/travel times. No identity or appearance matching.";

export function phaseChip(ph) { return h("span", { class: ["ph", "ph-" + ph], title: PHASE_HELP[ph] || "" }, ph); }
const pct = p => `${Math.round((+p || 0) * 100)}%`;
const prioChip = p => h("span", { class: ["chip", p] }, p || "—");

export default function createIncidents() {
  const el = h("section", { class: "page" });
  const f = { phase: "", status: "", priority: "" };
  const hot = { by: "zone", metric: "incidents", days: 28, entity: "" };
  let view = null, shown = false, detailId = null, selected = new Set(), timer = null, parts = null, loading = false;

  const tabs = h("div", { class: "tabs" });
  function paintTabs() {
    replace(tabs, h("a", { href: "#/incidents", class: view !== "hotspots" ? "on" : "" }, "Incidents"),
      h("a", { href: "#/incidents/hotspots", class: view === "hotspots" ? "on" : "" }, "Hotspots"));
  }

  // =================================================================================================== list
  const listHead = h("div", { class: "page-head" });
  const filtersEl = h("div", { class: "filters" });
  const listBody = h("div");
  function chipGroup(label, key, options) {
    return h("div", { class: "fgroup", role: "group", "aria-label": label }, h("span", { class: "lbl" }, label),
      h("button", { class: ["fchip", !f[key] && "on"], onclick: () => setFilter(key, "") }, "All"),
      options.map(v => h("button", { class: ["fchip", f[key] === v && "on"], "aria-pressed": String(f[key] === v),
        onclick: () => setFilter(key, f[key] === v ? "" : v) }, v)));
  }
  function setFilter(k, v) { f[k] = v; syncHash(); paintFilters(); loadList(); }
  function paintFilters() {
    replace(filtersEl, chipGroup("Phase", "phase", PHASES), chipGroup("Status", "status", ISTATUS), chipGroup("Priority", "priority", PRIORITIES));
  }
  function syncHash() {
    const q = new URLSearchParams(Object.entries(f).filter(([, v]) => v));
    history.replaceState(null, "", "#/incidents" + (q.toString() ? "?" + q : ""));
  }

  const loadList = safe(async (quiet = false) => {
    if (loading) return;
    loading = true;
    try {
      const q = new URLSearchParams({ limit: 300, ...Object.fromEntries(Object.entries(f).filter(([, v]) => v)) });
      const rows = await optional("/api/incidents?" + q, "incidents", { silent: quiet });
      if (view !== "list") return;
      if (rows === null) { replace(listBody, featureMissing("Incidents")); return; }
      renderRows(rows);
    } finally { loading = false; }
  });

  function renderRows(rows) {
    selected = new Set([...selected].filter(id => rows.some(r => r.id === id && !r.merged_into)));
    const mergeBtn = can("incident.merge") ? h("button", { disabled: selected.size < 2, onclick: () => mergeDialog(rows.filter(r => selected.has(r.id))) },
      `Merge selected (${selected.size})`) : null;
    const tb = rows.map(r => {
      const cb = can("incident.merge") ? h("input", { type: "checkbox", checked: selected.has(r.id), disabled: !!r.merged_into,
        "aria-label": `Select ${r.id}`, onclick: e => e.stopPropagation(),
        onchange: e => { e.target.checked ? selected.add(r.id) : selected.delete(r.id); renderRows(rows); } }) : null;
      return h("tr", { class: ["inc-row", r.merged_into && "merged"], tabIndex: 0, onclick: () => { location.hash = "#/incidents/" + enc(r.id); },
        onkeydown: e => { if (e.key === "Enter") location.hash = "#/incidents/" + enc(r.id); } },
        cb ? h("td", null, cb) : null,
        h("td", null, prioChip(r.priority)),
        h("td", null, phaseChip(r.phase)),
        h("td", null, r.status),
        h("td", null, h("div", { class: "t" }, r.title),
          h("div", { class: "small mute" }, r.id,
            r.merged_into ? ` · merged into ${r.merged_into}` : "", r.split_from ? ` · split from ${r.split_from}` : "",
            r.reopen_count ? ` · reopened ${r.reopen_count}×` : "",
            r.suppressed || r.inhibited_by ? h("span", { class: "tag", title: "Camera-integrity incident open on this camera: muted, not pushed" }, " muted") : "")),
        h("td", null, (r.cameras || []).map(camName).join(", ")),
        h("td", { class: "nowrap" }, String(r.n_alerts ?? "—")),
        h("td", null, r.assignee || h("span", { class: "mute" }, "—")),
        h("td", { class: "nowrap", title: fmtDateTime(r.last_event_at) }, fmtAgo(r.last_event_at)));
    });
    replace(listBody,
      h("div", { class: "row", style: { marginBottom: "8px" } }, h("span", { class: "mute small" }, `${rows.length} incident(s)`),
        h("span", { class: "grow" }), mergeBtn),
      rows.length ? h("div", { class: "table-wrap panel", style: { padding: 0 } }, h("table", { class: "inc-table" },
        h("thead", null, h("tr", null, can("incident.merge") ? h("th", null, "") : null,
          ["Priority", "Phase", "Status", "Incident", "Cameras", "Alerts", "Assignee", "Last event"].map(x => h("th", null, x)))),
        h("tbody", null, tb)))
        : h("div", { class: "empty" }, "No incidents match these filters."),
      h("div", { class: "disclaimer" }, "An incident groups alerts that share a zone, a camera track or (for camera problems) a camera within a time window. ",
        "Grouping is spatial/temporal only: no identity claim, no assessment of intent. Phase is automatic; status and decisions rest with the operator."));
  }

  function renderListShell() {
    replace(listHead, h("h2", null, "Incidents"), h("span", { class: "grow" }),
      h("button", { onclick: () => loadList() }, "Refresh"));
    paintFilters();
    replace(el, listHead, tabs, filtersEl, listBody);
  }

  // =================================================================================================== dialogs
  function mergeDialog(rows, fixedTarget = null) {
    if (rows.length < (fixedTarget ? 1 : 2)) return;
    let target = fixedTarget || rows[0].id;
    const body = h("div", { class: "stack" },
      h("p", { style: { margin: 0 } }, fixedTarget ? `Merge the selected incident(s) into ${fixedTarget}. Their alerts move; they close as "merged".`
        : "Choose the incident that remains. The others close as \"merged\" and their alerts move into it."),
      fixedTarget ? null : h("div", { class: "stack" }, rows.map(r => h("label", { class: "row" },
        h("input", { type: "radio", name: "mtarget", checked: r.id === target, onchange: () => { target = r.id; } }),
        prioChip(r.priority), h("b", null, r.id), h("span", { class: "mute" }, r.title)))),
      h("div", { class: "small mute" }, "Audited. Merge only situations you have verified to be the same; this makes no identity claim."));
    openDialog({ title: "Merge incidents", body, actions: [{ label: "Cancel" },
      { label: "Merge", primary: true, onclick: async () => {
        const sources = rows.map(r => r.id).filter(id => id !== target);
        await api(`/api/incidents/${enc(target)}/merge`, { method: "POST", body: { sources } });
        toast(`Merged ${sources.length} incident(s) into ${target}`, "ok");
        selected.clear();
        if (view === "detail") renderDetail(detailId, true); else loadList();
      } }] });
  }

  async function mergeInto(inc) {
    const rows = (await api("/api/incidents?limit=200", { silent: true })).filter(r => r.id !== inc.id && !r.merged_into && r.kind === inc.kind);
    if (!rows.length) { toast("No other incident of the same kind to merge.", "info"); return; }
    const pick = new Set();
    const body = h("div", { class: "stack picklist" }, rows.map(r => h("label", { class: "row" },
      h("input", { type: "checkbox", onchange: e => { e.target.checked ? pick.add(r.id) : pick.delete(r.id); } }),
      prioChip(r.priority), phaseChip(r.phase), h("b", null, r.id), h("span", { class: "mute" }, r.title))));
    openDialog({ title: `Merge into ${inc.id}`, wide: true, body, actions: [{ label: "Cancel" }, { label: "Merge selected", primary: true, onclick: async () => {
      if (!pick.size) { toast("Select at least one incident.", "warn"); return false; }
      await api(`/api/incidents/${enc(inc.id)}/merge`, { method: "POST", body: { sources: [...pick] } });
      toast(`Merged ${pick.size} incident(s) into ${inc.id}`, "ok");
      renderDetail(inc.id, true);
    } }] });
  }

  function splitDialog(inc) {
    const pick = new Set();
    const det = new Map((inc.alerts_detail || []).map(a => [a.id, a]));
    const body = h("div", { class: "stack" },
      h("p", { style: { margin: 0 } }, "Move the selected alerts into a new incident (at least one alert must stay). Their future events follow them."),
      h("div", { class: "stack picklist" }, inc.alert_ids.map(id => {
        const a = det.get(id) || {};
        return h("label", { class: "row" }, h("input", { type: "checkbox", onchange: e => { e.target.checked ? pick.add(id) : pick.delete(id); } }),
          prioChip(a.priority), h("b", null, id), h("span", { class: "mute" }, a.title || ""));
      })));
    openDialog({ title: `Split ${inc.id}`, body, actions: [{ label: "Cancel" }, { label: "Split out", primary: true, onclick: async () => {
      if (!pick.size || pick.size >= inc.alert_ids.length) { toast("Select at least one alert and leave at least one.", "warn"); return false; }
      const n = await api(`/api/incidents/${enc(inc.id)}/split`, { method: "POST", body: { alert_ids: [...pick] } });
      toast(`New incident ${n.id} created from ${pick.size} alert(s)`, "ok");
      location.hash = "#/incidents/" + enc(n.id);
    } }] });
  }

  function statusDialog(inc, status) {
    const note = h("textarea", { rows: 2, placeholder: "Note (optional, audited)", "aria-label": "Note" });
    openDialog({ title: `${status === "resolved" ? "Resolve" : "Close"} ${inc.id}`,
      body: h("div", { class: "stack" }, h("p", { style: { margin: 0 } }, "Closes the incident. Alerts keep their own status and dispositions (set them in the alert drawer)."), note),
      actions: [{ label: "Cancel" }, { label: status === "resolved" ? "Resolve" : "Close", primary: true,
        onclick: () => patch(inc.id, { status, note: note.value }) }] });
  }

  const patch = safe(async (id, body) => {
    await api(`/api/incidents/${enc(id)}`, { method: "PATCH", body });
    toast(`Incident ${id} updated`, "ok", { timeout: 2500 });
    renderDetail(id, true);
  });

  // =================================================================================================== detail
  function dropReplay() { parts?.replayCtl?.destroy(); }

  function skeleton(id) {
    dropReplay();
    const replayCtl = can("evidence.view") ? createReplay(id) : null;
    parts = { id, head: h("div"), banner: h("div"), acts: h("div", { class: "actions row" }), graph: h("div"), graphSig: null,
      timeline: h("div"), side: h("div", { class: "stack" }), lower: h("div"), summary: null, summaryDirty: false,
      assign: null, assignDirty: false, replayCtl };
    replace(el, h("div", { class: "page-head" }, h("a", { href: "#/incidents", class: "btnlink" }, "← Incidents"), h("span", { class: "grow" }),
      h("button", { onclick: () => renderDetail(id) }, "Refresh")), tabs, parts.head, parts.banner, parts.acts,
      h("div", { class: "inc-grid" },
        h("div", { class: "stack", style: { minWidth: 0 } },
          h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Evidence graph ", h("span", { class: "tag" }, "typed links")),
            parts.graph),
          h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Timeline"), parts.timeline),
          h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Synchronised replay ", h("span", { class: "tag" }, "all cameras, one clock")),
            replayCtl ? replayCtl.el : h("div", { class: "small mute" }, "Your role cannot view recordings (needs evidence.view)."))),
        parts.side),
      parts.lower);
  }

  const renderDetail = safe(async (id, quiet = false) => {
    let inc;
    try { inc = await api(`/api/incidents/${enc(id)}`, { silent: quiet, feature: "incidents" }); }
    catch (e) {
      if (view !== "detail" || detailId !== id) return;
      if (e.kind === "missing") { replace(el, tabs, featureMissing("Incidents")); return; }
      if (e.kind === "notfound") {
        replace(el, h("div", { class: "page-head" }, h("a", { href: "#/incidents", class: "btnlink" }, "← Incidents")), tabs,
          h("div", { class: "empty" }, `Incident ${id} does not exist (it may belong to another site or database).`));
        return;
      }
      throw e;
    }
    if (view !== "detail" || detailId !== id) return;
    if (!parts || parts.id !== id) skeleton(id);
    paintHead(inc);
    paintActions(inc);
    paintSide(inc);
    paintTimeline(inc);
    paintLower(inc);
    const sig = `${inc.n_events}|${inc.alert_ids.length}|${Object.keys(inc.predictions || {}).length}|${Object.keys(inc.global_ids || {}).length}|${inc.merged_from?.length}`;
    if (sig !== parts.graphSig) {
      parts.graphSig = sig;
      try {
        const g = await api(`/api/incidents/${enc(id)}/graph`, { silent: true });
        if (parts?.id !== id) return;
        replace(parts.graph, g.nodes.length ? forceGraph(g, { onNode: n => nodeClick(n, inc) }).el : h("div", { class: "empty" }, "No events."),
          g.truncated ? h("div", { class: "small mute" }, `Showing the latest events of ${g.n_events}.`) : null,
          h("div", { class: "note" }, g.note));
      } catch { replace(parts.graph, h("div", { class: "empty" }, "Graph unavailable.")); }
    }
  });

  function nodeClick(n, inc) {
    if (n.kind === "event" && n.alert_id) openAlerts(inc, n.alert_id);
    else if (n.kind === "evidence" && can("evidence.view")) { const p = n.clip || n.snapshot || n.crop; if (p) window.open(mediaUrl(p), "_blank", "noopener"); }
    else if (n.kind === "incident" && n.incident_id) location.hash = "#/incidents/" + enc(n.incident_id);
    else if (n.kind === "camera" && n.camera_id) location.hash = "#/live/" + enc(n.camera_id);
    else if (n.kind === "sensor") location.hash = "#/system";  // Access-control sources card
  }

  function openAlerts(inc, aid) {
    const ids = inc.alert_ids.slice();
    let i = Math.max(0, ids.indexOf(aid));
    const go = d => { const j = i + d; if (j < 0 || j >= ids.length) return false; i = j; drawer.openAlert(ids[i]); return true; };
    drawer.setNavigator({ next: () => go(1), prev: () => go(-1) });
    drawer.openAlert(ids[i]);
  }

  function paintHead(inc) {
    replace(parts.head,
      h("div", { class: "row", style: { flexWrap: "nowrap" } }, prioChip(inc.priority), phaseChip(inc.phase),
        h("h2", { style: { margin: 0, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" } }, inc.title)),
      h("div", { class: "small mute", style: { margin: "4px 0 8px" } },
        `${inc.id} · ${inc.kind === "integrity" ? "camera integrity" : "behaviour"} · status `, h("b", { style: { color: "var(--text)" } }, inc.status),
        ` · opened ${fmtDateTime(inc.opened_at)} · last event ${fmtAgo(inc.last_event_at)}`,
        inc.closed_at ? ` · closed ${fmtTime(inc.closed_at)} (${inc.closed_by})` : "",
        ` · ${inc.alert_ids.length} alert(s), ${inc.n_events} event(s)`,
        inc.assignee ? ` · assigned to ${inc.assignee}` : "", inc.reopen_count ? ` · reopened ${inc.reopen_count}×` : ""));
    const banners = [];
    if (inc.merged_into) banners.push(h("div", { class: "banner" }, "This incident was merged into ",
      h("a", { href: "#/incidents/" + enc(inc.merged_into) }, inc.merged_into), "."));
    if (inc.inhibited_by) banners.push(h("div", { class: "banner" }, "Some events come from a camera with an open view problem (",
      h("a", { href: "#/incidents/" + enc(inc.inhibited_by) }, inc.inhibited_by), "): they are shown but muted (not pushed). Check the camera before relying on them."));
    if (inc.split_from) banners.push(h("div", { class: "small mute" }, "Split from ", h("a", { href: "#/incidents/" + enc(inc.split_from) }, inc.split_from), "."));
    replace(parts.banner, banners);
  }

  function paintActions(inc) {
    const a = [];
    const live = !inc.merged_into;
    if (live && can("alert.acknowledge") && inc.status === "open" && inc.phase !== "closed") a.push(h("button", { onclick: () => patch(inc.id, { status: "acknowledged" }) }, "Acknowledge"));
    if (live && can("alert.escalate") && inc.status !== "investigating" && inc.phase !== "closed") a.push(h("button", { onclick: () => patch(inc.id, { status: "investigating" }) }, "Investigating"));
    if (live && can("alert.resolve") && inc.phase !== "closed") {
      a.push(h("button", { onclick: () => statusDialog(inc, "resolved") }, "Resolve"), h("button", { class: "ghost", onclick: () => statusDialog(inc, "closed") }, "Close"));
    }
    if (live && can("alert.acknowledge") && inc.phase === "closed") a.push(h("button", { class: "ghost", onclick: () => patch(inc.id, { status: "open" }) }, "Re-open"));
    if (live && can("incident.assign")) {
      if (!parts.assign) {
        parts.assign = h("input", { placeholder: "Assignee", "aria-label": "Assignee", style: { width: "140px" } });
        parts.assign.addEventListener("input", () => { parts.assignDirty = true; });
      }
      if (!parts.assignDirty) parts.assign.value = inc.assignee || "";
      a.push(h("span", { class: "row", style: { gap: "4px" } }, parts.assign,
        h("button", { onclick: () => { parts.assignDirty = false; patch(inc.id, { assignee: parts.assign.value.trim() || null }); } }, "Assign"),
        !inc.assignee && S.me.username && !S.me.open_mode ? h("button", { class: "ghost", onclick: () => patch(inc.id, { assignee: S.me.username }) }, "Assign to me") : null));
    }
    if (live && can("incident.merge")) {
      a.push(h("button", { onclick: safe(() => mergeInto(inc)) }, "Merge into this…"));
      if (inc.alert_ids.length > 1) a.push(h("button", { onclick: () => splitDialog(inc) }, "Split…"));
    }
    const report = h("button", { type: "button",
      title: "Printable incident report in a new tab: priority factors, timeline, evidence, site map, dispositions, audit trail and "
        + "SHA-256 hashes. For a PDF use the browser's Print → Save as PDF (A4). Generating it is audited.",
      onclick: () => window.open(`/api/incidents/${enc(inc.id)}/report`, "_blank", "noopener") }, "Report");
    replace(parts.acts, report, a.length ? a : h("span", { class: "small mute" }, "Read-only: your role cannot change incidents."));
  }

  // Assistive multi-camera narrative (VLM, cite-or-refuse). Cited ids become links; the text is never evidence.
  const ID_SPLIT = /(\b[AI]\d{4,}-\d+\b)/g;
  function citeText(text, inc) {
    return String(text || "").split(ID_SPLIT).map((p, i) => {
      if (i % 2 === 0) return p;
      if (p.startsWith("I")) return h("a", { href: "#/incidents/" + enc(p) }, p);
      return h("a", { href: "#", title: "Open this alert", onclick: e => { e.preventDefault(); openAlerts(inc, p); } }, p);
    });
  }

  const summarise = safe(async inc => {
    parts.summarising = true;
    paintSide(inc);
    try {
      const r = await api(`/api/incidents/${enc(inc.id)}/summary`, { method: "POST" });
      if (r.engine === "template") toast(`Showing a summary built from the alert records${r.rejected ? ` (${r.rejected})` : ""}.`, "warn", { timeout: 6000 });
      else toast(`Assistive summary by ${r.model}`, "ok", { timeout: 2500 });
    } finally {
      if (parts?.id === inc.id) parts.summarising = false;
      renderDetail(inc.id, true);
    }
  });

  function assistiveView(inc) {
    const a = inc.assistive_summary;
    const btn = can("alert.acknowledge") && !inc.merged_into ? h("button", { type: "button", class: "sm", disabled: !!parts.summarising,
      title: "Ask the VLM for a short chronological narrative over this incident's alerts and up to 4 keyframes across cameras. "
        + "It must cite alert ids; otherwise a summary built from the alert records is shown. Audited.",
      onclick: () => summarise(inc) }, parts.summarising ? "Summarising…" : a ? "Re-summarise" : "Summarise") : null;
    return h("div", { style: { marginTop: "10px" } },
      h("div", { class: "row" }, h("span", { class: "tag est", title: "Assistive text: not evidence" }, "assistive"),
        h("b", { class: "small" }, "Multi-camera summary"), h("span", { class: "grow" }), btn),
      a ? [h("p", { class: "small", style: { margin: "6px 0 2px" } }, citeText(a.text, inc)),
        h("div", { class: "note" }, a.label, ` Generated ${fmtDateTime(a.generated_at)} by ${a.generated_by}`,
          a.keyframes?.length ? ` · ${a.keyframes.length} keyframe(s)` : "", a.rejected ? ` · ${a.rejected}` : "")]
        : h("div", { class: "small mute", style: { marginTop: "4px" } }, "No summary yet. The VLM describes the alerts in time order and must cite alert ids."));
  }

  function paintSide(inc) {
    if (!parts.summary) {
      parts.summary = h("textarea", { rows: 3, placeholder: "Operator summary (what was checked, what was done)", "aria-label": "Operator summary" });
      parts.summary.addEventListener("input", () => { parts.summaryDirty = true; });
    }
    if (!parts.summaryDirty) parts.summary.value = inc.summary || "";
    parts.summary.disabled = !can("feedback.write") || !!inc.merged_into;
    const camLinks = (inc.cameras || []).map(c => h("a", { class: "fchip", href: "#/live/" + enc(c), title: "Open live view" }, camName(c)));
    const zones = (inc.zones || []).map(z => h("span", { class: "tag" }, z));
    const rules = (inc.rules || []).map(r => h("span", { class: "tag" }, RULE[r] || r));
    replace(parts.side,
      h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Summary"),
        h("p", { class: "small", style: { marginTop: 0 } }, h("span", { class: "tag obs" }, "auto"), " ", inc.auto_summary),
        parts.summary,
        can("feedback.write") && !inc.merged_into ? h("div", { class: "row", style: { marginTop: "6px" } },
          h("button", { onclick: () => { parts.summaryDirty = false; patch(inc.id, { summary: parts.summary.value }); } }, "Save summary")) : null,
        assistiveView(inc)),
      h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Related cameras & zones"),
        h("div", { class: "row" }, camLinks.length ? camLinks : h("span", { class: "mute small" }, "—")),
        h("div", { class: "row", style: { marginTop: "6px" } }, zones), h("div", { class: "row", style: { marginTop: "6px" } }, rules)),
      h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Predicted next camera ", h("span", { class: "tag est" }, "estimate")),
        predictionsView(inc)),
      pushView(inc));
  }

  function predictionsView(inc) {
    const rows = Object.entries(inc.predictions || {});
    const gids = Object.entries(inc.global_ids || {});
    const out = [];
    if (!rows.length) out.push(h("div", { class: "small mute" }, "No next-camera estimate was recorded for this incident's tracks (handoff prediction is off, or no track was leaving a view)."));
    for (const [, r] of rows) {
      out.push(h("div", { class: "pred" },
        h("div", { class: "small" }, h("b", null, `#${r.track_id}`), ` on ${camName(r.camera)} · ${r.state === "exited" ? "left the view" : "leaving"} (${r.exit_side || "?"})`,
          r.updated ? h("span", { class: "mute" }, ` · ${fmtTime(r.updated)}`) : ""),
        (r.predictions || []).map(p => h("div", { class: "row small", style: { gap: "6px" } },
          h("span", null, "→ ", p.leaves_coverage ? "leaves coverage" : camName(p.camera)),
          h("div", { class: "progress", style: { width: "70px" } }, h("div", { style: { width: pct(p.prob) } })),
          h("span", { class: "mute" }, pct(p.prob), p.eta_s != null ? ` · ~${Math.round(p.eta_s)} s` : "")))));
    }
    if (gids.length) out.push(h("div", { class: "small", style: { marginTop: "8px" } }, h("span", { class: "tag est" }, "re-id, probabilistic"), " ",
      gids.map(([k, g]) => `${k.replace(":", " #")} ↔ global ${g.global_id}${g.confidence != null ? ` (${g.confidence})` : ""}`).join("; ")));
    out.push(h("div", { class: "note" }, HO_NOTE));
    return out;
  }

  function pushView(inc) {
    const rows = (inc.notifications || []).slice(0, 12);
    return h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Push decisions"),
      rows.length ? h("div", { class: "table-wrap" }, h("table", null, h("tbody", null, rows.map(n => h("tr", null,
        h("td", { class: "nowrap small" }, fmtTime(n.t)), h("td", null, prioChip(n.priority)),
        h("td", { class: "small" }, h("span", { class: ["dec", "dec-" + n.decision] }, DECISION[n.decision] || n.decision), ` (${n.trigger})`,
          h("div", { class: "mute" }, n.reason))))))) : h("div", { class: "small mute" }, "No push decision yet (held incidents decide after the 10 s hold-down)."),
      h("div", { class: "note" }, "Nothing is sent from the console: decisions are recorded as \"would push\" (P1/P2; P2 limited to 6 per operator-hour, burst 3; P1 exempt). A signed webhook is optional and off by default."));
  }

  function paintTimeline(inc) {
    const evs = (inc.events || []).slice().sort((a, b) => a.t_end - b.t_end);
    const t0 = Math.min(inc.opened_at, ...evs.map(e => e.t_start || e.t_end));
    const t1 = Math.max(inc.last_event_at, inc.closed_at || 0, ...inc.timeline.map(x => x.t), t0 + 5);
    const cams = inc.cameras.length ? inc.cameras : [...new Set(evs.map(e => e.camera_id))];
    const L = 120, W = 1000, R = 12, lane = 22, top = 26;
    const H = top + Math.max(1, cams.length) * lane + 22;
    const X = t => L + (t - t0) / (t1 - t0) * (W - L - R);
    const nodes = [s("rect", { x: 0, y: 0, width: W, height: H, class: "tl-bg" })];
    cams.forEach((c, i) => {
      const y = top + i * lane;
      nodes.push(s("line", { x1: L, x2: W - R, y1: y + lane / 2, y2: y + lane / 2, class: "tl-lane" }),
        s("text", { x: 6, y: y + lane / 2 + 4, class: "tl-txt" }, camName(c).slice(0, 18)));
    });
    for (const e of evs) {
      const i = Math.max(0, cams.indexOf(e.camera_id));
      const x0 = X(e.t_start || e.t_end), x1 = Math.max(x0 + 4, X(e.t_end));
      const r = s("rect", { x: x0, y: top + i * lane + 5, width: x1 - x0, height: lane - 10, rx: 2,
        class: "tl-ev" + (e.learned_model ? " inf" : "") + (e.inhibited_by ? " muted" : "") + (e.rule === "camera_health" ? " health" : ""), tabindex: 0 },
        s("title", null, `${RULE[e.rule] || e.rule} · ${camName(e.camera_id)} · ${e.zone_name || ""}\n${fmtTime(e.t_start)} → ${fmtTime(e.t_end)} · alert ${e.alert_id}` +
          (e.inhibited_by ? `\nmuted by ${e.inhibited_by}` : "") + "\nclick: open the alert"));
      r.addEventListener("click", () => openAlerts(inc, e.alert_id));
      nodes.push(r);
    }
    const marks = { active: "active", cooling: "cooling", closed: "closed", reopened: "reopened", merged: "merged", split: "split", push: "push" };
    for (const m of inc.timeline) {
      if (!marks[m.kind]) continue;
      const x = X(m.t);
      if (m.kind === "push") {
        nodes.push(s("path", { d: `M${x - 5},4 L${x + 5},4 L${x},12 z`, class: "tl-push dec-" + (m.decision || "") }, s("title", null, `${fmtTime(m.t)} ${m.text}`)));
      } else {
        nodes.push(s("line", { x1: x, x2: x, y1: 14, y2: H - 18, class: "tl-mark k-" + m.kind }, s("title", null, `${fmtTime(m.t)} ${m.text}`)),
          s("text", { x: x + 3, y: 22, class: "tl-txt small" }, m.kind));
      }
    }
    nodes.push(s("text", { x: L, y: H - 4, class: "tl-txt" }, fmtTime(t0)),
      s("text", { x: W - R, y: H - 4, class: "tl-txt", "text-anchor": "end" }, `${fmtTime(t1)} (${fmtDur(t1 - t0)})`));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "tl-svg", role: "img", "aria-label": `Incident timeline over ${fmtDur(t1 - t0)}` }, nodes);
    replace(parts.timeline, svg,
      h("div", { class: "small mute" }, "Bars: events per camera (click to open the alert). Lines: lifecycle changes. Triangles: push decisions."));
  }

  function paintLower(inc) {
    const alerts = inc.alerts_detail || [];
    const tl = inc.timeline.slice().reverse().slice(0, 80);
    replace(parts.lower,
      h("h3", null, `Alerts (${inc.alert_ids.length})`),
      alerts.length ? h("div", { class: "table-wrap panel", style: { padding: 0 } }, h("table", { class: "inc-table" },
        h("thead", null, h("tr", null, ["Priority", "Status", "Alert", "Cameras", "Events", "Updated"].map(x => h("th", null, x)))),
        h("tbody", null, alerts.slice().reverse().map(a => h("tr", { class: "inc-row", tabIndex: 0, onclick: () => openAlerts(inc, a.id),
          onkeydown: e => { if (e.key === "Enter") openAlerts(inc, a.id); } },
          h("td", null, prioChip(a.priority)), h("td", null, a.status, a.disposition ? h("div", { class: "small mute" }, a.disposition) : null),
          h("td", null, h("div", { class: "t" }, a.title), h("div", { class: "small mute" }, a.id, a.suppressed ? " · muted" : "")),
          h("td", null, (a.cameras || []).map(camName).join(", ")), h("td", null, String(a.n_events)),
          h("td", { class: "nowrap" }, fmtTime(a.updated_at)))))))
        : h("div", { class: "empty" }, inc.merged_into ? "Alerts moved to the incident this one was merged into." : "No alerts."),
      h("h3", null, "Lifecycle log"),
      h("div", { class: "audit-list" }, tl.map(x => h("div", null, fmtDateTime(x.t), " · ", h("b", null, x.kind), " · ", x.text, x.actor ? ` (${x.actor})` : ""))),
      h("div", { class: "disclaimer" }, inc.disclaimer || ""),
      h("h3", null, "Audit trail"), auditBox(inc.id));
  }

  function auditBox(id) {
    const box = h("div", { class: "audit-list" }, "Loading…");
    api(`/api/incidents/${enc(id)}/audit`, { silent: true }).then(list => {
      replace(box, list.length ? list.map(x => h("div", null, fmtDateTime(x.t), " · ", h("b", null, x.actor), " · ", x.action,
        x.details?.to !== undefined ? ` → ${x.details.to ?? "nobody"}` : "", x.details?.note ? ` — ${x.details.note}` : "",
        x.details?.sources ? ` — ${x.details.sources.join(", ")}` : "", x.details?.new_incident ? ` → ${x.details.new_incident}` : "")) : "No operator actions yet.");
    }).catch(() => replace(box, "Audit trail unavailable."));
    return box;
  }

  // =================================================================================================== hotspots
  async function renderHotspots() {
    const body = h("div", null, h("div", { class: "empty" }, "Loading…"));
    replace(el, h("div", { class: "page-head" }, h("h2", null, "Hotspots"), h("span", { class: "grow" }),
      segmented([["zone", "By zone"], ["camera", "By camera"]], hot.by, v => { hot.by = v; hot.entity = ""; renderHotspots(); }),
      segmented([["incidents", "Incidents"], ["alerts", "Alerts"]], hot.metric, v => { hot.metric = v; renderHotspots(); }),
      segmented([[7, "7 d"], [28, "28 d"], [90, "90 d"]], hot.days, v => { hot.days = v; renderHotspots(); })), tabs, body);
    let d;
    try { d = await api(`/api/hotspots?by=${hot.by}&metric=${hot.metric}&days=${hot.days}`, { feature: "hotspots" }); }
    catch (e) { if (view === "hotspots") replace(body, e.kind === "missing" ? featureMissing("Hotspots") : h("div", { class: "empty" }, "Hotspots unavailable.")); return; }
    if (view !== "hotspots") return;
    const ent = d.rows.find(r => r.key === hot.entity) || null;
    const counts = ent ? ent.counts : d.overall;
    const max = Math.max(1, ...counts);
    const sel = h("select", { "aria-label": "Zone or camera" }, h("option", { value: "" }, `All (${d.n_items} ${d.metric})`),
      d.rows.map(r => h("option", { value: r.key, selected: r.key === hot.entity }, `${r.label}${hot.by === "zone" ? ` (${r.camera})` : ""} — ${r.total}`)));
    sel.addEventListener("change", () => { hot.entity = sel.value; renderHotspots(); });
    const grid = h("div", { class: "heat", role: "table", "aria-label": "Frequency by weekday and hour" },
      h("div"), Array.from({ length: 24 }, (_, hh) => h("div", { class: "hh" }, hh % 3 ? "" : String(hh).padStart(2, "0"))),
      DAYS.map((dn, di) => [h("div", { class: "dn" }, dn), Array.from({ length: 24 }, (_, hh) => {
        const n = counts[di * 24 + hh] || 0;
        return h("div", { class: "c", title: `${dn} ${String(hh).padStart(2, "0")}:00–${String((hh + 1) % 24).padStart(2, "0")}:00: ${n}`,
          style: { background: n ? `color-mix(in srgb, var(--p2) ${Math.round(15 + 85 * n / max)}%, var(--panel2))` : "var(--panel2)" } }, n ? String(n) : "");
      })]));
    const strip = r => {
      const m = Math.max(1, ...r.counts);
      return s("svg", { viewBox: "0 0 168 10", class: "strip", preserveAspectRatio: "none", "aria-hidden": "true" },
        r.counts.map((n, i) => n ? s("rect", { x: i, y: 0, width: 1, height: 10, class: "strip-c", opacity: (0.2 + 0.8 * n / m).toFixed(2) }) : null));
    };
    const analytics = h("div");
    replace(body,
      h("div", { class: "inc-grid" },
        h("div", { class: "panel", style: { minWidth: 0 } }, h("div", { class: "row", style: { marginBottom: "8px" } }, h("b", null, "Hour of week"), sel),
          grid, h("div", { class: "note" }, d.note)),
        h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Top hotspots"),
          d.top.length ? h("ol", { class: "small", style: { paddingLeft: "18px", margin: 0 } }, d.top.map(t => h("li", null,
            h("b", null, t.label), hot.by === "zone" ? h("span", { class: "mute" }, ` (${t.camera})`) : "", ` · ${howLabel(t.how)} · ${t.count}×`,
            h("div", { class: "row", style: { gap: "4px" } }, (t.examples || []).map(id => idLink(id)))))) : h("div", { class: "mute small" }, "Nothing recorded in this period."))),
      h("div", { class: "panel", style: { marginTop: "12px" } }, h("h3", { style: { marginTop: 0 } }, `Per ${hot.by}`),
        d.rows.length ? h("div", { class: "table-wrap" }, h("table", null,
          h("thead", null, h("tr", null, [hot.by === "zone" ? "Zone" : "Camera", "Total", "Peak slot", "Mon → Sun (hourly)"].map(x => h("th", null, x)))),
          h("tbody", null, d.rows.slice(0, 50).map(r => h("tr", { class: "inc-row", onclick: () => { hot.entity = r.key; renderHotspots(); } },
            h("td", null, r.label, hot.by === "zone" ? h("div", { class: "small mute" }, r.camera) : null), h("td", null, String(r.total)),
            h("td", { class: "nowrap" }, howLabel(r.peak_how)), h("td", { style: { minWidth: "220px" } }, strip(r))))))) : h("div", { class: "mute small" }, "—")),
      analytics);
    // feature probe: an occupancy-analytics module may expose its own hotspot view
    try {
      const a = await optional(`/api/analytics/hotspots?days=${hot.days}`, "analytics-hotspots", { silent: true });
      if (a && view === "hotspots") replace(analytics, genericPanel("Occupancy analytics (/api/analytics/hotspots)", a));
    } catch { /* optional */ }
  }

  function genericPanel(title, data) {
    const rows = Array.isArray(data) ? data : Array.isArray(data?.rows) ? data.rows : Array.isArray(data?.hotspots) ? data.hotspots : null;
    let content;
    if (rows && rows.length && typeof rows[0] === "object") {
      const keys = Object.keys(rows[0]).filter(k => typeof rows[0][k] !== "object").slice(0, 7);
      content = h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null, keys.map(k => h("th", null, k)))),
        h("tbody", null, rows.slice(0, 20).map(r => h("tr", null, keys.map(k => h("td", null, String(r[k] ?? "—"))))))));
    } else content = h("pre", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, JSON.stringify(data, null, 1).slice(0, 3000));
    return h("div", { class: "panel", style: { marginTop: "12px" } }, h("h3", { style: { marginTop: 0 } }, title), content);
  }

  function idLink(id) {
    if (String(id).startsWith("I")) return h("a", { href: "#/incidents/" + enc(id), class: "small" }, id);
    return h("a", { href: "#", class: "small", onclick: e => { e.preventDefault(); drawer.setNavigator(null); drawer.openAlert(id); } }, id);
  }

  // =================================================================================================== wiring
  const refresh = debounce(() => {
    if (!shown || document.hidden) return;
    if (view === "list") loadList(true);
    else if (view === "detail" && detailId) renderDetail(detailId, true);
  }, 800);
  on("alert-changed", () => refresh());

  return {
    el,
    show(args, q) {
      shown = true;
      const next = args[0] === "hotspots" ? "hotspots" : args[0] ? "detail" : "list";
      const changed = next !== view || (next === "detail" && args[0] !== detailId);
      view = next;
      paintTabs();
      if (view === "hotspots") { if (changed) renderHotspots(); }
      else if (view === "detail") {
        if (changed) { dropReplay(); detailId = args[0]; parts = null; replace(el, tabs, h("div", { class: "empty" }, "Loading…")); }
        renderDetail(detailId);
      } else {
        for (const k of Object.keys(f)) f[k] = q.get(k) || "";
        renderListShell(); loadList();
      }
      clearInterval(timer);
      timer = setInterval(refresh, 5000);
    },
    hide() { shown = false; clearInterval(timer); timer = null; dropReplay(); if (parts) parts = null; },
    onAlert() { refresh(); },
    onResync() { refresh(); },
  };
}
