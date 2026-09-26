// Review: 24 h timeline strip (10-min buckets, clickable), filter chips, keyed alert cards or a dense list,
// load-more, and keyboard triage (j/k or ↓/↑ to move, Enter to open).
import { api, safe } from "./api.js";
import { S, camName, on } from "./state.js";
import * as drawer from "./drawer.js";
import { h, s, icon, replace, mediaUrl, NOSNAP, fmtTime, fmtAgo, RULE, STATUSES, PRIORITIES, debounce, throttle, isTyping, ls, num } from "./util.js";
import { pageHeader, segmented, emptyState, skeleton, prioChip, statusPill, demoBadge, plainTitle } from "./ui.js";
import { slaChip, isFlood } from "./workflow.js";

const PAGE = 48;
const URGENT = new Set(["P1", "P2"]);
const BUCKET = 600, SPAN = 86400, NB = SPAN / BUCKET;
const TL_LIMIT = 1000;
const PCOL = { P1: "var(--p1)", P2: "var(--p2)", P3: "var(--p3)", P4: "var(--p4)" };

export default function createReview() {
  const f = { status: "", priority: "", review: "", camera: "", rule: "" };
  let win = null;          // [start, end] from the timeline
  let limit = PAGE, list = [], sel = -1, shown = false, loading = false, exhausted = false, loadedOnce = false;
  let mode = ls.get("review.view", "grid") === "list" ? "list" : "grid";
  const itemEls = new Map(); // id -> {el, sig}

  const tlSvg = s("svg", { viewBox: `0 0 ${NB} 50`, preserveAspectRatio: "none", role: "img", "aria-label": "Alerts per 10 minutes over the last 24 hours" });
  const tlAxis = h("div", { class: "axis", "aria-hidden": "true" });
  const tlInfo = h("span", { class: "small mute" });
  const winChip = h("span");
  const filtersEl = h("div", { class: "filters", role: "search", "aria-label": "Alert filters" });
  const results = h("div", { class: "cards", role: "list", "aria-label": "Alerts" });
  const moreBtn = h("button", { onclick: () => { limit += PAGE; load(); } }, "Load more");
  const countEl = h("span", { class: "tnum" });
  const viewSel = segmented([["grid", "", "Card grid", "grid2"], ["list", "", "Dense list", "list"]], mode, v => {
    mode = v; ls.set("review.view", v === "list" ? "list" : null); itemEls.clear(); replace(results); renderItems();
  }, { label: "View" });
  const el = h("section", { class: "page" },
    pageHeader({ title: "Review", subtitle: countEl, actions: [
      winChip,
      h("span", { class: "kbd-hint hide-md" }, h("span", null, h("kbd", null, "j"), " ", h("kbd", null, "k"), " move"), h("span", null, h("kbd", null, "Enter"), " open")),
      viewSel,
      h("button", { onclick: () => { load(); loadTimeline(true); }, title: "Refresh" }, icon("refresh"), h("span", { class: "hide-sm" }, "Refresh"))] }),
    h("div", { class: "timeline" },
      h("div", { class: "row", style: { marginBottom: "4px" } }, h("b", { class: "small" }, "Last 24 h"), tlInfo, h("span", { class: "grow" }),
        h("div", { class: "legend hide-sm" }, PRIORITIES.map(p => h("span", null, h("i", { style: { background: PCOL[p] } }), p)))),
      tlSvg, tlAxis),
    filtersEl, results, h("div", { class: "more" }, moreBtn));

  // ---------- filters ----------
  function chipGroup(label, key, options) {
    return h("div", { class: "fgroup", role: "group", "aria-label": label }, h("span", { class: "lbl" }, label),
      h("button", { class: ["fchip", !f[key] && "on"], "aria-pressed": String(!f[key]), onclick: () => setFilter(key, "") }, "All"),
      options.map(([v, l]) => h("button", { class: ["fchip", f[key] === v && "on"], "aria-pressed": String(f[key] === v),
        onclick: () => setFilter(key, f[key] === v ? "" : v) }, l)));
  }
  function renderFilters() {
    replace(filtersEl,
      chipGroup("Status", "status", STATUSES.map(v => [v, v[0].toUpperCase() + v.slice(1)])),
      chipGroup("Priority", "priority", [["P1,P2", "P1–P2"], ...PRIORITIES.map(v => [v, v])]),
      chipGroup("Review", "review", [["second", "Second review"]]),
      chipGroup("Camera", "camera", S.cams.map(c => [c.id, c.name])),
      chipGroup("Behaviour", "rule", Object.entries(RULE)));
    replace(winChip, win ? h("button", { class: "fchip on", title: "Clear time window", onclick: () => { win = null; syncHash(); renderFilters(); load(); drawTimeline(); } },
      icon("clock"), `${fmtTime(win[0])}–${fmtTime(win[1])}`, icon("x")) : null);
  }
  function setFilter(k, v) { f[k] = v; limit = PAGE; sel = -1; syncHash(); renderFilters(); load(); loadTimeline(true); }
  function query(extra = {}) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(f)) if (v) q.set(k, v);
    for (const [k, v] of Object.entries(extra)) q.set(k, v);
    return q.toString();
  }
  function syncHash() {
    const q = query(win ? { from: Math.round(win[0]), to: Math.round(win[1]) } : {});
    history.replaceState(null, "", "#/review" + (q ? "?" + q : ""));
  }

  // ---------- list ----------
  const load = safe(async () => {
    if (loading) { load.again = true; return; }
    loading = true;
    if (!loadedOnce) replace(results, skeleton(mode === "list" ? "row" : "card", mode === "list" ? 8 : 6));
    try {
      // time windows are filtered client-side (the API has no time filter), so fetch more rows then
      const n = win ? TL_LIMIT : limit + 1;
      let rows = await api("/api/alerts?" + query({ limit: n }), { silent: !shown });
      if (win) rows = rows.filter(a => a.created_at >= win[0] && a.created_at < win[1]);
      exhausted = rows.length <= limit;
      list = rows.slice(0, limit);
      // alert flood: unacknowledged P1/P2 first (stable), unless the operator already filters by priority
      if (isFlood() && !f.priority) {
        const urgent = a => a.status === "new" && URGENT.has(a.priority);
        list = [...list.filter(urgent), ...list.filter(a => !urgent(a))];
      }
      if (!loadedOnce) { loadedOnce = true; replace(results); itemEls.clear(); }
      renderItems();
    } finally {
      loading = false;
      if (load.again) { load.again = false; load(); }
    }
  });

  function sig(a) { return `${mode}|${a.status}|${a.updated_at}|${a.priority}|${a.events.length}|${a.disposition || ""}|${a.second_review ? 1 : 0}`; }
  // waiting time / "overdue" on unacknowledged alerts ticks without re-rendering the cards
  setInterval(() => {
    if (!shown || document.hidden) return;
    for (const c of itemEls.values()) if (c.slot && c.a?.status === "new") replace(c.slot, slaChip(c.a, { compact: true }));
  }, 5000);
  function thumb(a) {
    const last = a.events[a.events.length - 1] || {};
    const img = h("img", { loading: "lazy", alt: "", src: last.evidence?.snapshot ? mediaUrl(last.evidence.snapshot) : NOSNAP });
    img.addEventListener("error", () => { img.src = NOSNAP; }, { once: true });
    return img;
  }
  function onOpen(a) { return () => openAt(list.findIndex(x => x.id === a.id)); }
  function card(a) {
    const first = a.events[0] || {};
    const rules = [...new Set(a.events.map(e => RULE[e.rule] || e.rule))];
    return h("div", { class: ["card", a.status], tabIndex: 0, role: "listitem", "aria-label": `${a.priority} ${a.title}, ${a.status}`, dataset: { id: a.id },
      onclick: onOpen(a), onkeydown: e => { if (e.key === "Enter") { e.preventDefault(); onOpen(a)(); } } },
    thumb(a), prioChip(a.priority), h("span", { class: "st" }, a.status),
    h("div", { class: "b" }, h("div", { class: "t" }, demoBadge(a), plainTitle(a)),
      h("div", { class: "m" }, icon("cameras"), a.cameras.map(camName).join(", ")),
      h("div", { class: "m foot" }, h("span", { class: "sla-slot" }, slaChip(a, { compact: true })),
        a.second_review ? h("span", { class: "tag warn", title: "Flagged unclear: waiting for a second review" }, "2nd review") : null,h("span", { class: "tnum" }, fmtAgo(a.updated_at)), "·",
        h("span", null, `${a.events.length} event${a.events.length === 1 ? "" : "s"}${rules.length > 1 ? ` · ${rules.length} behaviours` : ""}`),
        h("span", { class: "grow" }), h("span", { class: "tnum", title: "Heuristic confidence of the first event (uncalibrated)" }, `conf ${num(first.confidence, 2)}`))));
  }
  function row(a) {
    const ev = a.events[a.events.length - 1] || {};
    return h("div", { class: ["rrow", a.status], tabIndex: 0, role: "listitem", "aria-label": `${a.priority} ${a.title}, ${a.status}`, dataset: { id: a.id },
      onclick: onOpen(a), onkeydown: e => { if (e.key === "Enter") { e.preventDefault(); onOpen(a)(); } } },
    h("span", { class: ["pbar", a.priority] }), thumb(a),
    h("div", { class: "rt" }, h("div", { class: "t" }, prioChip(a.priority), demoBadge(a), h("span", null, plainTitle(a))), h("div", { class: "o" }, ev.observation || "")),
    h("div", { class: "rc" }, a.cameras.map(camName).join(", "), h("div", { class: "mute" }, `${a.events.length} event${a.events.length === 1 ? "" : "s"} · conf ${num((a.events[0] || {}).confidence, 2)}`)),
    h("div", { class: "rs" }, statusPill(a.status), a.second_review ? h("span", { class: "tag warn", title: "Flagged unclear: waiting for a second review" }, "2nd review") : null, h("div", { class: "sla-slot" }, slaChip(a, { compact: true }))),
    h("div", { class: "rtime" }, fmtAgo(a.updated_at), h("div", null, fmtTime(a.updated_at))));
  }
  function renderItems() {
    results.className = mode === "list" ? "rlist" : "cards";
    const keep = new Set(list.map(a => a.id));
    for (const [id, c] of itemEls) if (!keep.has(id)) { c.el.remove(); itemEls.delete(id); }
    let head = results.querySelector(".rhead");
    if (mode === "list" && list.length && !head) {
      head = h("div", { class: "rhead", "aria-hidden": "true" }, h("span"), h("span", null, "Evidence"), h("span", null, "Alert · latest observation"),
        h("span", { class: "rc" }, "Cameras"), h("span", { class: "rs" }, "Status"), h("span", null, "Updated"));
      results.prepend(head);
    }
    if ((mode !== "list" || !list.length) && head) head.remove();
    let prev = mode === "list" && list.length ? head : null;
    for (const a of list) {
      let c = itemEls.get(a.id);
      if (!c || c.sig !== sig(a)) {
        const el2 = mode === "list" ? row(a) : card(a);
        if (c) c.el.replaceWith(el2);
        c = { el: el2, sig: sig(a), a, slot: el2.querySelector(".sla-slot") };
        itemEls.set(a.id, c);
      }
      const want = prev ? prev.nextSibling : results.firstChild;
      if (want !== c.el) results.insertBefore(c.el, want);
      prev = c.el;
    }
    const empty = results.querySelector(":scope > .empty");
    if (!list.length && !empty) results.appendChild(h("div", { style: { gridColumn: "1/-1" }, class: "empty-wrap" },
      emptyState({ icon: "filter", title: "No alerts match", text: win || Object.values(f).some(Boolean) ? "Try clearing a filter or the time window." : "Alerts appear here as soon as a rule fires.",
        action: win || Object.values(f).some(Boolean) ? h("button", { onclick: () => { for (const k of Object.keys(f)) f[k] = ""; win = null; syncHash(); renderFilters(); load(); drawTimeline(); } }, "Clear filters") : null })));
    if (list.length) results.querySelector(":scope > .empty-wrap")?.remove();
    moreBtn.hidden = exhausted || !!win;
    countEl.textContent = `${list.length}${exhausted ? "" : "+"} alert${list.length === 1 ? "" : "s"}${win ? " in the selected window" : ""}${isFlood() && !f.priority ? " · alert flood: new P1/P2 first" : ""}`;
    // keep selection pinned to the alert shown in the drawer
    const cur = drawer.currentId();
    if (cur) sel = list.findIndex(a => a.id === cur);
    markSel();
  }
  function markSel() {
    for (const [id, c] of itemEls) {
      const on = sel >= 0 && list[sel]?.id === id;
      c.el.classList.toggle("sel", on);
      if (on) c.el.setAttribute("aria-current", "true"); else c.el.removeAttribute("aria-current");
    }
  }
  function openAt(i, focus = true) {
    if (i < 0 || i >= list.length) return false;
    sel = i; markSel();
    const c = itemEls.get(list[i].id);
    c?.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (focus) {
      drawer.setNavigator({ next: () => move(1), prev: () => move(-1) });
      drawer.openAlert(list[i].id);
    } else c?.el.focus({ preventScroll: true });
    return true;
  }
  function move(d, open = true) {
    const i = sel < 0 ? 0 : sel + d;
    if (i >= list.length && !exhausted && !win) { limit += PAGE; load().then(() => openAt(i, open && drawer.isOpen())); return true; }
    return openAt(Math.max(0, Math.min(list.length - 1, i)), open);
  }

  // ---------- timeline ----------
  let tlRows = [], tlAt = 0;
  const loadTimeline = safe(async (force = false) => {
    if (!force && Date.now() - tlAt < 15000) return;
    tlAt = Date.now();
    tlRows = await api("/api/alerts?" + query({ limit: TL_LIMIT }), { silent: true });
    drawTimeline();
  });
  const loadTimelineThrottled = throttle(() => loadTimeline(true), 15000);

  function drawTimeline() {
    const now = Date.now() / 1000;
    const end = Math.ceil(now / BUCKET) * BUCKET, start = end - SPAN;
    const counts = Array.from({ length: NB }, () => ({ P1: 0, P2: 0, P3: 0, P4: 0, n: 0 }));
    let older = 0;
    for (const a of tlRows) {
      const i = Math.floor((a.created_at - start) / BUCKET);
      if (i < 0) { older++; continue; }
      if (i >= NB) continue;
      counts[i][a.priority] = (counts[i][a.priority] || 0) + 1; counts[i].n++;
    }
    const max = Math.max(1, ...counts.map(c => c.n));
    const nodes = [];
    for (let t = Math.ceil(start / 10800) * 10800; t <= end; t += 10800) {   // 3 h gridlines
      const x = (t - start) / BUCKET;
      nodes.push(s("rect", { x, y: 0, width: 0.12, height: 50, style: { fill: "var(--line)" } }));
    }
    nodes.push(s("rect", { x: 0, y: 49.4, width: NB, height: 0.6, style: { fill: "var(--line-strong)" } }));
    counts.forEach((c, i) => {
      const t0 = start + i * BUCKET, selected = win && Math.abs(win[0] - t0) < 1;
      const g = s("g", { class: "tl-bucket", "data-i": i },
        s("rect", { x: i, y: 0, width: 1, height: 50, style: { fill: selected ? "var(--acc)" : "transparent", opacity: selected ? 0.18 : 0 } }),
        s("title", null, `${fmtTime(t0)}–${fmtTime(t0 + BUCKET)}: ${c.n} alert(s)` + (c.n ? ` (P1 ${c.P1}, P2 ${c.P2}, P3 ${c.P3}, P4 ${c.P4})` : "")));
      let y = 49.4;
      for (const p of ["P4", "P3", "P2", "P1"]) {
        if (!c[p]) continue;
        const hgt = Math.max(2, (c[p] / max) * 46);
        y -= hgt;
        g.appendChild(s("rect", { x: i + 0.12, y, width: 0.76, height: hgt, style: { fill: PCOL[p] } }));
      }
      nodes.push(g);
    });
    replace(tlSvg, nodes);
    // hour labels every 3 h
    const labels = [];
    for (let t = Math.ceil(start / 10800) * 10800; t <= end; t += 10800) {
      const pct = (t - start) / SPAN * 100;
      if (pct < 3 || pct > 97) continue;
      labels.push(h("span", { style: { left: pct + "%" } }, new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })));
    }
    replace(tlAxis, labels);
    const total = counts.reduce((a, c) => a + c.n, 0);
    tlInfo.textContent = ` · ${total} alert${total === 1 ? "" : "s"}` + (tlRows.length >= TL_LIMIT && !older ? ` (from the ${TL_LIMIT} most recent; older ones may be missing)` : "") + " · click a bar to filter by 10-min window";
  }
  tlSvg.addEventListener("click", e => {
    const g = e.target.closest?.("g.tl-bucket");
    const rect = tlSvg.getBoundingClientRect();
    const i = g ? +g.dataset.i : Math.floor((e.clientX - rect.left) / rect.width * NB);
    const end = Math.ceil(Date.now() / 1000 / BUCKET) * BUCKET, t0 = end - SPAN + i * BUCKET;
    win = win && Math.abs(win[0] - t0) < 1 ? null : [t0, t0 + BUCKET];
    sel = -1; syncHash(); renderFilters(); drawTimeline(); load();
  });

  const refresh = debounce(() => { if (shown) { load(); loadTimelineThrottled(); } }, 700);
  on("alert-changed", () => refresh());
  moreBtn.hidden = true;

  return {
    el,
    show(args, q) {
      shown = true;
      for (const k of Object.keys(f)) f[k] = q.get(k) || "";
      const v = q.get("view");   // the flood banner's "compact" link opens the dense list
      if ((v === "list" || v === "grid") && v !== mode) { mode = v; ls.set("review.view", v === "list" ? "list" : null); itemEls.clear(); replace(results); viewSel.set?.(v); }
      const from = +q.get("from"), to = +q.get("to");
      win = from && to ? [from, to] : null;
      renderFilters(); load(); loadTimeline(true);
    },
    hide() { shown = false; },
    onAlert() { refresh(); },
    onResync() { refresh(); },
    onKey(e) {
      if (isTyping(e)) return false;
      if (e.key === "j") return move(1, drawer.isOpen());
      if (e.key === "k") return move(-1, drawer.isOpen());
      const inList = results.contains(document.activeElement);
      if (inList && e.key === "ArrowDown") return move(1, drawer.isOpen());
      if (inList && e.key === "ArrowUp") return move(-1, drawer.isOpen());
      if (e.key === "Enter" && sel >= 0 && !drawer.isOpen() && !inList) return openAt(sel);
      return false;
    },
  };
}

