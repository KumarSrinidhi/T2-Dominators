// Investigate: ask questions about past alerts/incidents in plain language. The server translates the question into
// an allow-listed structured query (LLM on the VLM server, or the offline keyword parser), validates it, runs it on
// the store and returns results + a summary that cites alert/incident ids. The query is shown as editable chips and
// can be re-run without the LLM. Every server string is rendered as text (no innerHTML).
import { api, safe } from "./api.js";
import { camName } from "./state.js";
import * as drawer from "./drawer.js";
import { h, replace, enc, fmtDateTime, RULE } from "./util.js";

const EXAMPLES = [
  "show crowding near the server wing yesterday evening",
  "which cameras had camera problems today?",
  "how many alerts in the last 24 hours?",
  "unacknowledged P1 or P2 alerts this week",
  "open incidents by zone",
  "restricted-zone entries by hour this week",
];
const PRESET_LABEL = { today: "today", yesterday: "yesterday", last_hour: "last hour", last_24h: "last 24 h", last_7d: "last 7 days",
  last_30d: "last 30 days", this_week: "this week", all: "any time" };
const AGG_LABEL = { list: "list", count: "count", by_zone: "by zone", by_camera: "by camera", by_hour: "by hour" };
const ID_RE = /\b[AI]\d{4,}-\d+\b/g;
// questions answered by the "what changed vs normal?" briefing instead of the alert/incident query
const CHANGED_RE = /\b(what('?s| has| have)? changed|changes?\b.*\bnormal|vs\.? normal|than usual|unusual|out of the ordinary|different from (normal|usual))/i;
const KIND_LABEL = { occupancy: "Occupancy", event_rate: "Event rate", dwell: "Dwell", camera_health: "Camera view" };

export default function createInvestigate() {
  let schema = null, busy = false;
  const turns = h("div", { class: "chat" });
  const input = h("input", { type: "text", placeholder: "Ask about past alerts or incidents, e.g. “crowding near the server wing yesterday evening”",
    "aria-label": "Question", maxLength: 500 });
  const askBtn = h("button", { class: "primary", type: "submit" }, "Ask");
  const engineInfo = h("span", { class: "small mute" });
  const form = h("form", { class: "ask row", onsubmit: e => { e.preventDefault(); ask(input.value); } }, input, askBtn);
  const hoursSel = h("select", { class: "fchip", "aria-label": "Compare the last" },
    [[1, "last hour"], [3, "last 3 h"], [8, "last 8 h"], [24, "last 24 h"]].map(([v, l]) => h("option", { value: v }, l)));
  const changedBtn = h("button", { class: "fchip", type: "button", title: "Compare with this site's own history at the same hours of the week",
    onclick: () => changed(+hoursSel.value) }, "What changed vs normal?");
  const examples = h("div", { class: "row small", style: { marginTop: "6px" } }, h("span", { class: "mute" }, "Try:"),
    changedBtn, hoursSel,
    EXAMPLES.map(q => h("button", { class: "fchip", type: "button", onclick: () => { input.value = q; ask(q); } }, q)));
  const el = h("section", { class: "page" },
    h("div", { class: "page-head" }, h("h2", null, "Investigate"), engineInfo),
    h("div", { class: "panel" }, form, examples,
      h("div", { class: "note" }, "Answers come from the alert/incident store through a validated, allow-listed query (the language model never writes SQL). ",
        "Summaries are assistive: verify them against the cited alerts' evidence. Nothing here identifies people.")),
    turns);

  async function loadSchema() {
    try {
      schema = await api("/api/investigate/schema", { silent: true, feature: "investigate" });
      engineInfo.textContent = schema.llm_configured ? "Translation: LLM on the VLM server (falls back to the keyword parser)"
        : "Translation: offline keyword parser (no LLM configured)";
    } catch (e) {
      if (e.kind === "missing") { replace(turns, h("div", { class: "banner" }, "Investigation is not available on this server.")); form.hidden = true; }
    }
  }

  const changed = safe(async (hours = 1, question = null) => {
    if (busy) return;
    busy = true; askBtn.disabled = true;
    const turn = newTurn(question || `What changed vs normal? (last ${hours} h)`);
    replace(turn.answer, h("div", { class: "row small mute" }, h("span", { class: "spinner" }), "Comparing with previous weeks…"));
    try {
      const r = await api(`/api/investigate/changes?hours=${enc(hours)}`);
      renderChanges(turn, r);
      if (question) input.value = "";
    } catch (e) {
      replace(turn.answer, h("div", { class: "banner err" }, `The comparison could not be made (${e.message}).`));
    } finally { busy = false; askBtn.disabled = false; }
  });

  function renderChanges(turn, r) {
    const w = r.window;
    const head = h("div", { class: "row small", style: { gap: "6px" } }, h("span", { class: "tag obs" }, "statistics"),
      h("span", { class: "mute" }, `${fmtDateTime(w.since)} → ${fmtDateTime(w.until)} vs the same hours of the week in previous weeks · |z| ≥ ${r.z_threshold}`));
    const list = r.changes.length ? h("div", { class: "table-wrap" }, h("table", null,
      h("thead", null, h("tr", null, ["What", "Where", "Now", "Usual", "Score", "Based on"].map(x => h("th", null, x)))),
      h("tbody", null, r.changes.map(c => h("tr", null,
        h("td", null, KIND_LABEL[c.kind] || c.kind, h("div", { class: "small mute" }, c.direction)),
        h("td", null, c.text),
        h("td", { class: "tnum" }, String(c.observed)), h("td", { class: "tnum" }, String(c.usual)),
        h("td", { class: "tnum" }, c.kind === "camera_health" ? "—" : (c.score > 0 ? "+" : "") + c.score),
        h("td", { class: "small mute" }, c.basis))))))
      : h("p", null, r.steady.length ? "Nothing stands out: every compared zone is within its usual range for these hours." : "Nothing to compare yet.");
    replace(turn.answer, head, list,
      r.more ? h("div", { class: "small mute" }, `${r.more} smaller change(s) not shown.`) : null,
      r.steady.length ? h("details", { class: "small" }, h("summary", null, `As usual (${r.steady.length})`), h("div", { class: "mute" }, r.steady.join(" · "))) : null,
      r.not_enough_history.length ? h("details", { class: "small" }, h("summary", null, `Not enough history yet (${r.not_enough_history.length})`),
        h("div", { class: "mute" }, r.not_enough_history.map(x => `${KIND_LABEL[x.kind] || x.kind}: ${x.zone} (${camName(x.camera)}), ${x.history} past samples`).join(" · "))) : null,
      h("div", { class: "disclaimer" }, r.disclaimer));
  }

  const ask = safe(async (question, query = null, card = null) => {
    question = String(question || "").trim();
    if (!query && !card && CHANGED_RE.test(question)) return changed(+hoursSel.value, question);
    if (busy || (!question && !query)) return;
    busy = true; askBtn.disabled = true;
    const turn = card || newTurn(question);
    const ans = turn.answer;
    replace(ans, h("div", { class: "row small mute" }, h("span", { class: "spinner" }), query ? "Running the edited query…" : "Translating and searching…"));
    try {
      const r = await api("/api/investigate", { method: "POST", body: query ? { query, question } : { question } });
      renderAnswer(turn, r);
      if (!card) input.value = "";
    } catch (e) {
      replace(ans, h("div", { class: "banner err" }, e.kind === "http" && e.status === 400 ? e.message : `The question could not be answered (${e.message}).`));
    } finally { busy = false; askBtn.disabled = false; }
  });

  function newTurn(question) {
    const answer = h("div", { class: "a" });
    const node = h("div", { class: "turn" }, h("div", { class: "q" }, question), answer);
    turns.prepend(node);
    return { node, answer, question };
  }

  // ---------------------------------------------------------------------------------------------- query chips
  function queryChips(turn, q0) {
    const q = JSON.parse(JSON.stringify(q0));
    const box = h("div", { class: "qchips" });
    const rerun = h("button", { class: "primary small", hidden: true, onclick: () => ask(turn.question, clean(q), turn) }, "Run edited query");
    const dirty = () => { rerun.hidden = false; paint(); };
    const vocab = schema?.vocab || { cameras: [], zones: [], rules: {} };
    const zoneName = id => (vocab.zones.find(z => z.id === id || z.site_zone === id) || {}).name || id;
    const chip = (label, onRemove, title) => h("span", { class: "qchip", title: title || "" }, label,
      onRemove ? h("button", { type: "button", class: "x", "aria-label": `Remove ${label}`, onclick: () => { onRemove(); dirty(); } }, "×") : null);
    const sel = (value, options, onChange, aria) => {
      const s = h("select", { class: "qchip", "aria-label": aria }, options.map(([v, l]) => h("option", { value: v, selected: v === value }, l)));
      s.addEventListener("change", () => { onChange(s.value); dirty(); });
      return s;
    };
    const listRemove = (k, v) => () => { q[k] = (q[k] || []).filter(x => x !== v); if (!q[k].length) delete q[k]; };
    function paint() {
      const t = q.time || {};
      const timeKind = t.preset || (t.last_hours != null ? "custom" : t.date ? "custom" : (t.from || t.to) ? "custom" : "all");
      const timeOpts = Object.entries(PRESET_LABEL);
      if (timeKind === "custom") timeOpts.unshift(["custom", t.date ? `on ${t.date}` : t.last_hours != null ? `last ${t.last_hours} h` : `${t.from || "…"} → ${t.to || "now"}`]);
      const add = h("select", { class: "qchip add", "aria-label": "Add a filter" }, h("option", { value: "" }, "+ filter"),
        h("optgroup", { label: "Camera" }, vocab.cameras.filter(c => !(q.cameras || []).includes(c.id)).map(c => h("option", { value: "cameras|" + c.id }, c.name))),
        h("optgroup", { label: "Zone" }, uniqZones(vocab).filter(z => !(q.zones || []).includes(z.key)).map(z => h("option", { value: "zones|" + z.key }, z.name))),
        h("optgroup", { label: "Behaviour" }, Object.entries(vocab.rules).filter(([k]) => !(q.rules || []).includes(k)).map(([k, l]) => h("option", { value: "rules|" + k }, l))),
        h("optgroup", { label: "Priority" }, ["P1", "P2", "P3", "P4"].filter(p => !(q.priorities || []).includes(p)).map(p => h("option", { value: "priorities|" + p }, p))),
        h("optgroup", { label: "Status" }, ((q.target === "incidents" ? schema?.incident_statuses : schema?.alert_statuses) || []).filter(x => !(q.status || []).includes(x))
          .map(x => h("option", { value: "status|" + x }, x))),
        h("optgroup", { label: "Confidence" }, [0.5, 0.7, 0.9].map(c => h("option", { value: "min_confidence|" + c }, `≥ ${c}`))));
      add.addEventListener("change", () => {
        const [k, v] = add.value.split("|");
        if (!k) return;
        if (k === "min_confidence") q.min_confidence = +v;
        else q[k] = [...(q[k] || []), v];
        dirty();
      });
      replace(box,
        sel(q.target || "alerts", [["alerts", "alerts"], ["incidents", "incidents"]], v => { q.target = v; delete q.status; }, "Search target"),
        sel(timeKind, timeOpts, v => { if (v === "custom") return; const pod = t.part_of_day; q.time = v === "all" ? {} : { preset: v }; if (pod && v !== "all" && !["last_hour", "last_24h"].includes(v)) q.time.part_of_day = pod; }, "Time range"),
        sel(t.part_of_day || "", [["", "any time of day"], ["morning", "morning"], ["afternoon", "afternoon"], ["evening", "evening"], ["night", "night"]],
          v => { q.time = { ...(q.time || {}) }; if (v) { delete q.time.last_hours; delete q.time.from; delete q.time.to; if (!q.time.preset && !q.time.date) q.time.preset = "today"; q.time.part_of_day = v; } else delete q.time.part_of_day; }, "Part of day"),
        (q.cameras || []).map(c => chip(`camera: ${camName(c)}`, listRemove("cameras", c))),
        (q.zones || []).map(z => chip(`zone: ${zoneName(z)}`, listRemove("zones", z))),
        (q.rules || []).map(r => chip(RULE[r] || vocab.rules[r] || r, listRemove("rules", r))),
        (q.priorities || []).map(p => chip(p, listRemove("priorities", p))),
        (q.status || []).map(st => chip(`status: ${st}`, listRemove("status", st))),
        q.min_confidence != null ? chip(`confidence ≥ ${q.min_confidence}`, () => { delete q.min_confidence; }, "heuristic confidence") : null,
        q.text_contains ? chip(`text: “${q.text_contains}”`, () => { delete q.text_contains; }) : null,
        sel(q.aggregation || "list", Object.entries(AGG_LABEL), v => { q.aggregation = v; }, "Result shape"),
        add, rerun);
    }
    paint();
    return box;
  }
  function uniqZones(v) {
    const m = new Map();
    for (const z of v.zones) { const key = z.site_zone || z.id; if (!m.has(key)) m.set(key, { key, name: `${z.name} (${camName(z.camera_id)})` }); }
    return [...m.values()];
  }
  function clean(q) {
    const out = { ...q };
    if (out.time && !Object.keys(out.time).length) delete out.time;
    return out;
  }

  // ---------------------------------------------------------------------------------------------- answer
  function idLink(id) {
    if (id.startsWith("I")) return h("a", { href: "#/incidents/" + enc(id) }, id);
    return h("a", { href: "#", onclick: e => { e.preventDefault(); drawer.setNavigator(null); drawer.openAlert(id); } }, id);
  }
  function linkify(text, allowed) {
    const out = [];
    let last = 0;
    for (const m of String(text).matchAll(ID_RE)) {
      out.push(text.slice(last, m.index));
      out.push(allowed.has(m[0]) ? idLink(m[0]) : m[0]);
      last = m.index + m[0].length;
    }
    out.push(text.slice(last));
    return out;
  }

  function resultsView(r) {
    const res = r.results;
    const head = h("div", { class: "small mute" }, `${res.total} ${res.target}`, res.windows?.length ? ` · ${res.windows.length === 1
      ? `${fmtDateTime(res.windows[0].from)} → ${fmtDateTime(res.windows[0].to)}` : `${res.windows.length} time windows`}` : " · any time",
      res.truncated ? " · only the most recent 5000 rows were searched" : "");
    if (res.groups) {
      return [head, h("div", { class: "table-wrap" }, h("table", null,
        h("thead", null, h("tr", null, ["Group", "Count", "Most recent ids"].map(x => h("th", null, x)))),
        h("tbody", null, res.groups.map(g => h("tr", null, h("td", null, g.label), h("td", null, String(g.count)),
          h("td", null, h("div", { class: "row", style: { gap: "6px" } }, g.ids.map(idLink))))))))];
    }
    if (res.aggregation === "count") {
      return [h("div", { class: "bignum" }, String(res.total)), head,
        res.items.length ? h("div", { class: "row small", style: { gap: "6px" } }, "e.g.", res.items.slice(0, 8).map(i => idLink(i.id))) : null];
    }
    if (!res.items.length) return [head];
    return [head, h("div", { class: "table-wrap" }, h("table", null,
      h("thead", null, h("tr", null, ["Id", "Priority", "Title", "Cameras", "Time", "Status", res.target === "alerts" ? "Incident" : "Alerts"].map(x => h("th", null, x)))),
      h("tbody", null, res.items.map(i => h("tr", null,
        h("td", { class: "nowrap" }, idLink(i.id)), h("td", null, h("span", { class: ["chip", i.priority] }, i.priority)),
        h("td", null, i.title, i.suppressed ? h("span", { class: "tag", title: "muted by a camera-integrity incident" }, " muted") : null,
          h("div", { class: "small mute" }, (i.rules || []).map(x => RULE[x] || x).join(", "), i.zones?.length ? ` · ${i.zones.join(", ")}` : "")),
        h("td", null, (i.cameras || []).map(camName).join(", ")),
        h("td", { class: "nowrap", title: fmtDateTime(i.t) }, fmtDateTime(i.t)),
        h("td", null, i.status, i.phase ? h("div", { class: "small mute" }, i.phase) : null),
        h("td", null, res.target === "alerts" ? (i.incident_id ? idLink(i.incident_id) : "—") : String(i.n_alerts ?? "—")))))))];
  }

  function renderAnswer(turn, r) {
    const allowed = new Set([...(r.results.items || []).map(i => i.id), ...(r.results.groups || []).flatMap(g => g.ids), ...(r.citations || []).map(c => c.id)]);
    const llm = r.summary_engine === "llm";
    const badges = h("div", { class: "row small", style: { gap: "6px" } },
      h("span", { class: ["tag", r.engine === "llm" ? "inf" : "obs"] },
        r.engine === "llm" ? "LLM translation" : r.engine === "rules" ? "keyword parser" : "edited query"),
      r.cached ? h("span", { class: "tag" }, "cached") : null,
      r.llm?.model ? h("span", { class: "mute" }, r.llm.model) : null,
      h("span", { class: "mute" }, [r.timings?.translate_ms != null ? `translate ${r.timings.translate_ms} ms` : "", r.timings?.summary_ms != null ? `summary ${r.timings.summary_ms} ms` : "",
        `total ${Math.round(r.timings?.total_ms ?? 0)} ms`].filter(Boolean).join(" · ")));
    replace(turn.answer,
      badges,
      h("div", { class: "small", style: { margin: "6px 0 2px" } }, h("b", null, "Structured query "), h("span", { class: "mute" }, "(edit the chips and re-run; values are validated on the server)")),
      queryChips(turn, r.query),
      r.warnings?.length ? h("ul", { class: "warn-list small" }, r.warnings.map(w => h("li", null, w))) : null,
      h("div", { class: ["summary", llm ? "llm" : "tpl"] },
        h("div", { class: "row small" }, h("span", { class: ["tag", llm ? "inf" : "obs"] }, llm ? "AI · inference" : "template"),
          h("b", null, llm ? "AI-generated summary — verify against evidence" : "Summary from the query results (template) — verify against evidence")),
        h("p", { style: { margin: "6px 0 4px" } }, linkify(r.summary, allowed)),
        r.citations?.length ? h("div", { class: "row small", style: { gap: "6px" } }, h("span", { class: "mute" }, "Cites:"), r.citations.map(c => idLink(c.id)))
          : h("div", { class: "small mute" }, "No records to cite.")),
      resultsView(r),
      h("div", { class: "disclaimer" }, r.disclaimer));
  }

  return {
    el,
    show() {
      if (!schema) loadSchema();
      setTimeout(() => input.focus({ preventScroll: true }), 0);
    },
  };
}
