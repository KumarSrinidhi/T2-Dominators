// Shift handover: an automatic "unresolved at shift end" summary (open/overdue alerts and incidents, camera problems,
// top zones) for a selectable window, the operator's handover note (audited) and previous notes, plus a print view
// (/api/handover/print: self-contained HTML, the browser's Print → Save as PDF). Every server string is a text node.
import { api, optional, safe } from "./api.js";
import { can, on } from "./state.js";
import { toast, pageHeader, emptyState, prioChip, segmented } from "./ui.js";
import * as drawer from "./drawer.js";
import { h, icon, replace, enc, fmtDateTime, fmtAgo, debounce } from "./util.js";

const WINDOWS = [[4, "4 h"], [8, "8 h"], [12, "12 h"], [24, "24 h"]];

export default function createHandover() {
  let hours = 8, sinceIso = "", shown = false, timer = null, loading = false;
  const kpis = h("div", { class: "kpis" });
  const lines = h("div");
  const tables = h("div", { class: "stack" });
  const notesEl = h("div", { class: "stack" });
  const note = h("textarea", { rows: 6, maxLength: 4000, "aria-label": "Handover note",
    placeholder: "What the next shift needs to know: open situations, what was checked, what is pending, camera issues…" });
  const since = h("input", { type: "datetime-local", "aria-label": "Shift start", title: "Shift start (overrides the window)" });
  since.addEventListener("change", () => { sinceIso = since.value; load(); });
  const winSeg = h("span");
  const paintWin = () => replace(winSeg, segmented(WINDOWS, sinceIso ? null : hours, v => { hours = v; sinceIso = ""; since.value = ""; paintWin(); load(); },
    { label: "Shift window" }));
  paintWin();

  const query = () => {
    const t = sinceIso ? Date.parse(sinceIso) / 1000 : NaN;
    return Number.isFinite(t) ? `since=${t}` : `hours=${hours}`;
  };
  const printBtn = h("button", { type: "button", title: "Printable handover sheet in a new tab; use the browser's Print → Save as PDF (A4). Audited.",
    onclick: () => window.open(`/api/handover/print?${query()}`, "_blank", "noopener") }, icon("external"), "Print view");

  const el = h("section", { class: "page" },
    pageHeader({ title: "Shift handover", subtitle: "Unresolved at shift end, and a note for the next shift",
      actions: [winSeg, since, printBtn, h("button", { type: "button", onclick: () => load() }, "Refresh")] }),
    kpis,
    h("div", { class: "inc-grid" },
      h("div", { class: "stack", style: { minWidth: 0 } },
        h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Unresolved at shift end ", h("span", { class: "tag obs" }, "from the records")), lines),
        tables),
      h("div", { class: "stack" },
        h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Handover note"), note,
          h("div", { class: "row", style: { marginTop: "6px" } },
            h("span", { class: "small mute grow" }, "Saved with your name, the time and a snapshot of the open items. Audited."),
            h("button", { type: "button", class: "primary", disabled: !can("feedback.write"), onclick: () => save() }, "Save note"))),
        h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, "Previous notes"), notesEl))));

  const idLink = (id, kind) => kind === "incident" || String(id).startsWith("I")
    ? h("a", { href: "#/incidents/" + enc(id), class: "mono" }, id)
    : h("a", { href: "#", class: "mono", onclick: e => { e.preventDefault(); drawer.setNavigator(null); drawer.openAlert(id); } }, id);

  function table(head, rows) {
    return h("div", { class: "table-wrap" }, h("table", { class: "inc-table" },
      h("thead", null, h("tr", null, head.map(x => h("th", null, x)))), h("tbody", null, rows)));
  }
  function section(title, count, body, empty) {
    return h("div", { class: "panel" }, h("h3", { style: { marginTop: 0 } }, title, " ", h("span", { class: "tag" }, String(count))),
      h("div", null, count ? body : h("div", { class: "small mute" }, empty)));
  }
  const kpi = (label, v, sub, ic) => h("div", { class: "kpi" }, h("div", { class: "kpi-l" }, icon(ic), label), h("div", { class: "kpi-v" }, String(v)),
    h("div", { class: "kpi-sub" }, sub));

  function render(s) {
    const c = s.counts;
    replace(kpis,
      kpi("Alerts in shift", c.alerts, ["P1", "P2", "P3", "P4"].filter(p => c.by_priority[p]).map(p => `${p} ${c.by_priority[p]}`).join(" · ") || "none", "alert"),
      kpi("Still open", c.open_alerts + (s.incidents_enabled ? c.open_incidents : 0),
        `${c.open_alerts} alert(s)${s.incidents_enabled ? ` · ${c.open_incidents} incident(s)` : ""}`, "review"),
      kpi("Overdue", c.overdue, "not acknowledged within target", "clock"),
      kpi("Camera problems", c.camera_problems, c.camera_problems ? "check before relying on them" : "all views OK", "cameras"));
    replace(lines, s.lines.map(x => h("p", { class: "small", style: { margin: "0 0 4px" } }, x)),
      h("div", { class: "note" }, s.disclaimer, ` Window: ${fmtDateTime(s.since)} → ${fmtDateTime(s.until)}.`));
    const out = [];
    out.push(section("Overdue", s.overdue.length, table(["Item", "Priority", "Status", "Title", "Age", "Over target"],
      s.overdue.map(r => h("tr", null, h("td", null, idLink(r.id, r.kind)), h("td", null, prioChip(r.priority)), h("td", null, r.status),
        h("td", null, r.title), h("td", { class: "nowrap" }, `${r.age_min} min`), h("td", { class: "nowrap" }, `${r.overdue_by_min} min (target ${r.target_min})`)))),
      "Nothing is waiting for acknowledgement beyond its target."));
    out.push(section("Open alerts", s.open_alerts.length, table(["Alert", "Priority", "Status", "Title", "Cameras", "Created"],
      s.open_alerts.map(r => h("tr", null, h("td", null, idLink(r.id, "alert")), h("td", null, prioChip(r.priority)), h("td", null, r.status, r.suppressed ? h("div", { class: "small mute" }, "muted") : null),
        h("td", null, r.title), h("td", null, r.cameras.join(", ")), h("td", { class: "nowrap", title: fmtDateTime(r.created_at) }, fmtAgo(r.created_at))))),
      "No open alert in this window."));
    if (s.incidents_enabled) out.push(section("Open incidents", s.open_incidents.length, table(["Incident", "Priority", "Status", "Title", "Assignee", "Last event"],
      s.open_incidents.map(r => h("tr", null, h("td", null, idLink(r.id, "incident")), h("td", null, prioChip(r.priority)), h("td", null, `${r.status} / ${r.phase}`),
        h("td", null, r.title), h("td", null, r.assignee || h("span", { class: "mute" }, "—")), h("td", { class: "nowrap", title: fmtDateTime(r.last_event_at) }, fmtAgo(r.last_event_at))))),
      "No open incident in this window."));
    const camRows = [...s.camera_problems.map(p => [p, "now"]), ...(s.camera_problems_earlier || []).map(p => [p, "earlier in shift"])];
    const camPanel = section("Camera problems", s.camera_problems.length, null, "");
    replace(camPanel.lastChild, camRows.length ? table(["Camera", "When", "Health now", "Problem"],
      camRows.map(([p, when]) => h("tr", null, h("td", null, h("a", { href: "#/live/" + enc(p.id) }, p.name)), h("td", null, when),
        h("td", null, p.health), h("td", null, p.problem))))
      : h("div", { class: "small mute" }, `All ${s.cameras.length} camera view(s) are OK or warming up.`));
    out.push(camPanel);
    out.push(section("Top zones", s.top_zones.length, table(["Zone", "Camera", "Alerts", "Best priority", "Behaviours"],
      s.top_zones.map(z => h("tr", null, h("td", null, z.zone), h("td", null, z.camera), h("td", null, String(z.alerts)), h("td", null, prioChip(z.best_priority)),
        h("td", null, z.rules.join(", "))))), "No zone alert in this window."));
    replace(tables, out);
  }

  function renderNotes(list) {
    replace(notesEl, list.length ? list.map(n => h("div", { class: "rule-box" },
      h("div", { class: "small" }, h("b", null, n.author), h("span", { class: "mute" }, ` · ${fmtDateTime(n.t)}`)),
      h("div", { style: { whiteSpace: "pre-wrap", margin: "4px 0" } }, n.text),
      n.snapshot?.counts ? h("div", { class: "note" }, `At writing: ${n.snapshot.counts.open_alerts} open alert(s), `
        + `${n.snapshot.counts.open_incidents} open incident(s), ${n.snapshot.counts.overdue} overdue, ${n.snapshot.counts.camera_problems} camera problem(s).`) : null))
      : emptyState({ icon: "info", title: "No handover note yet", text: "Notes written here are kept with the author and time.", compact: true }));
  }

  const load = safe(async (quiet = false) => {
    if (loading) return;
    loading = true;
    try {
      const [s, list] = await Promise.all([
        optional(`/api/handover/summary?${query()}`, "handover", { silent: quiet }),
        optional("/api/handover?limit=30", "handover", { silent: true })]);
      if (s === null) { replace(tables, emptyState({ title: "Handover is not available on this server" })); return; }
      render(s);
      renderNotes(list || []);
    } finally { loading = false; }
  });

  const save = safe(async () => {
    const text = note.value.trim();
    if (!text) { toast("Write a note first.", "warn"); return; }
    const body = { text };
    const t = sinceIso ? Date.parse(sinceIso) / 1000 : NaN;
    if (Number.isFinite(t)) body.since = t; else body.hours = hours;
    await api("/api/handover", { method: "POST", body });
    note.value = "";
    toast("Handover note saved", "ok", { timeout: 2500 });
    load(true);
  });

  const refresh = debounce(() => { if (shown && !document.hidden) load(true); }, 1500);
  on("alert-changed", () => refresh());

  return {
    el,
    show() { shown = true; load(); clearInterval(timer); timer = setInterval(refresh, 30000); },
    hide() { shown = false; clearInterval(timer); timer = null; },
    onAlert() { refresh(); },
    onResync() { refresh(); },
  };
}
