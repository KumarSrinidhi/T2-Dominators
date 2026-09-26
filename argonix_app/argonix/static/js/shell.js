// Application shell: grouped sidebar (from the page registry), mobile tab bar, breadcrumb, site selector,
// command palette (Ctrl+K), notification bell, user menu (theme, demo mode), and the status bar.
// Everything is built with h(); server strings only ever reach text nodes.
import { api } from "./api.js";
import { S, on, emit, camName, can } from "./state.js";
import { visiblePages, getPage, onPagesChanged, GROUPS, hasEndpoint } from "./registry.js";
import { $, h, icon, replace, ls, fmtAgo, initials, enc, fmtMs, num, throttle, isTyping } from "./util.js";
import { popover, closePopover, prioChip, emptyState, openDialog, segmented, toast } from "./ui.js";
import * as drawer from "./drawer.js";
import { slaChip, autoOpen, setAutoOpen } from "./workflow.js";

let active = null;                 // current page id
const badges = new Map();          // page id -> count
const late = new Map();            // page id -> alerts past their acknowledge SLA
let newAlerts = [];                // latest status=new alerts (from app.js badge poll)

// ================================================================================================ theme
function systemTheme() { return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
export function themeMode() { return document.documentElement.dataset.theme || "system"; }
export function effectiveTheme() { return document.documentElement.dataset.theme || systemTheme(); }
export function setTheme(mode) {
  if (mode === "light" || mode === "dark") { document.documentElement.dataset.theme = mode; ls.set("theme", mode); }
  else { delete document.documentElement.dataset.theme; ls.set("theme", null); }
  paintThemeIcon(); emit("theme", effectiveTheme());
}
export function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  setTheme(next === systemTheme() ? "system" : next);
}
function paintThemeIcon() {
  const b = $("#themeBtn");
  if (!b) return;
  replace(b, icon(effectiveTheme() === "dark" ? "sun" : "moon"));
  b.title = `Switch to ${effectiveTheme() === "dark" ? "light" : "dark"} theme (t)`;
}

// ================================================================================================ demo mode
export function isDemo() { return document.documentElement.classList.contains("demo"); }
export function setDemo(onOff, { quiet = false } = {}) {
  document.documentElement.classList.toggle("demo", !!onOff);
  ls.set("demo", onOff ? "1" : null);
  const pill = $("#demoPill");
  if (pill) pill.hidden = !onOff;
  emit("demo", !!onOff);
  window.dispatchEvent(new Event("resize"));
  if (!quiet) toast(onOff ? "Demo mode on: larger type, and the Live wall follows new alerts." : "Demo mode off.", "info", { timeout: 3500 });
}

// ================================================================================================ sidebar
function pageIcon(p) {
  if (p.icon instanceof Element) { const c = p.icon.cloneNode(true); c.classList?.add("i"); c.setAttribute("aria-hidden", "true"); return c; }
  return icon(p.icon);
}

export function renderNav() {
  const host = $("#navGroups");
  if (!host) return;
  const vis = visiblePages();
  const groups = [...GROUPS];
  for (const p of vis) if (!groups.some(g => g.id === p.group)) groups.push({ id: p.group, label: p.group });
  let idx = 0;
  const nodes = [];
  for (const g of groups) {
    const items = vis.filter(p => p.group === g.id);
    if (!items.length) continue;
    nodes.push(h("div", { class: "nav-group", role: "group", "aria-label": g.label },
      h("div", { class: "nav-group-label" }, g.label),
      items.map(p => {
        const n = idx++, b = badges.get(p.id);
        return h("a", { class: ["nav-item", p.id === active && "on"], href: "#/" + p.id, dataset: { page: p.id, tip: p.label },
          "aria-current": p.id === active ? "page" : null },
        pageIcon(p), h("span", { class: "lbl" }, p.label),
        b ? h("span", { class: "badge", title: `${b} awaiting triage` }, b > 999 ? "999+" : String(b)) : null,
        late.get(p.id) ? h("span", { class: "badge late", title: `${late.get(p.id)} past the acknowledge SLA` }, icon("clock"), String(late.get(p.id))) : null,
        n < 9 ? h("kbd", { class: "kbd", "aria-hidden": "true" }, String(n + 1)) : null);
      })));
  }
  replace(host, nodes);
  renderTabbar(vis);
}

/** Pages in keyboard-shortcut order (1–9). */
export function navOrder() { return visiblePages().map(p => p.id); }

function renderTabbar(vis) {
  const bar = $("#tabbar");
  if (!bar) return;
  const pref = ["overview", "live", "review", "map"];
  const items = pref.map(id => vis.find(p => p.id === id)).filter(Boolean).slice(0, 4);
  replace(bar, items.map(p => {
    const b = badges.get(p.id);
    return h("a", { class: ["tab", p.id === active && "on"], href: "#/" + p.id, "aria-current": p.id === active ? "page" : null },
      pageIcon(p), h("span", null, p.label), b ? h("span", { class: "badge" }, b > 99 ? "99+" : String(b)) : null);
  }), h("button", { class: ["tab", !items.some(p => p.id === active) && "on"], type: "button", "aria-label": "More pages", onclick: () => toggleMobileNav(true) },
    icon("more"), h("span", null, "More")));
}

export function toggleMobileNav(open) {
  const app = $("#app");
  const want = open ?? !app.classList.contains("nav-open");
  app.classList.toggle("nav-open", want);
  $("#navScrim").hidden = !want;
  if (want) $("#sidebar .nav-item.on, #sidebar .nav-item")?.focus({ preventScroll: true });
}

export function setActive(id) {
  active = id;
  const p = getPage(id);
  const t = $("#pageTitle");
  if (t) t.textContent = p?.label || id;
  for (const a of document.querySelectorAll("#navGroups .nav-item, #tabbar .tab")) {
    const on = a.getAttribute("href") === "#/" + id;
    a.classList.toggle("on", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  const more = $("#tabbar button.tab");
  if (more) more.classList.toggle("on", !$("#tabbar a.tab.on"));
  toggleMobileNav(false);
}

export function setBadge(id, n) {
  const v = +n || 0;
  if ((badges.get(id) || 0) === v) return;
  if (v) badges.set(id, v); else badges.delete(id);
  renderNav();
}

/** Second badge: how many of the page's items are past their acknowledge SLA. */
export function setLate(id, n) {
  const v = +n || 0;
  if ((late.get(id) || 0) === v) return;
  if (v) late.set(id, v); else late.delete(id);
  renderNav();
}

// ================================================================================================ alert flood banner
let floodEl = null, floodDismissed = false;
/** st = {n, threshold, window_s, flood}. Dismissing hides it until the flood ends; a new flood shows it again. */
export function setFlood(st) {
  if (!st?.flood) { floodDismissed = false; if (floodEl) floodEl.hidden = true; return; }
  if (!floodEl) {
    floodEl = h("div", { class: "flood-banner", role: "status", "aria-live": "polite", hidden: true });
    $("#pages")?.before(floodEl);
  }
  floodEl.hidden = floodDismissed;
  if (floodDismissed) return;
  const mins = Math.round(st.window_s / 60);
  replace(floodEl, icon("bell"),
    h("span", { class: "grow" }, h("b", null, "Alert flood: "), `${st.n} new in ${mins} min — showing P1/P2 first`,
      h("span", { class: "mute hide-sm" }, ` (more than ${st.threshold} unacknowledged alerts in ${mins} min)`)),
    h("a", { class: "btnlink sm primary", href: "#/review?status=new&priority=P1%2CP2&view=list",
      title: "Review: dense list, new P1 and P2 alerts only" }, icon("filter"), "Compact · P1–P2 only"),
    h("button", { class: "ghost icon sm", type: "button", title: "Dismiss until this flood ends", "aria-label": "Dismiss alert-flood banner",
      onclick: () => { floodDismissed = true; floodEl.hidden = true; } }, icon("x")));
}

function autoOpenSwitch() {
  const sw = h("input", { type: "checkbox", checked: autoOpen(), "aria-label": "Auto-open cameras on P1/P2 incidents" });
  sw.addEventListener("change", () => { setAutoOpen(sw.checked); emit("autoopen", sw.checked); });
  return h("label", { class: "menu-i sw" }, icon("live"), h("span", { class: "grow" }, "Auto-open cameras on P1/P2",
    h("span", { class: "menu-sub" }, "New or escalated P1/P2 incident: Live focuses its cameras (this browser)")),
  h("span", { class: "switch" }, sw, h("span")));
}

// ================================================================================================ site selector
function siteMenu(anchor) {
  popover(anchor, close => h("div", { class: "menu" },
    h("div", { class: "menu-h" }, "Sites"),
    h("button", { class: "menu-i on", type: "button", onclick: () => { close(); location.hash = "#/overview"; } },
      icon("building"), h("span", { class: "grow" }, S.cfg.site || "This site"), icon("check")),
    h("div", { class: "menu-sep" }),
    h("div", { class: "menu-note" }, "Multi-site federation is on the roadmap: each site runs its own Argus core; this menu will switch between them.")),
  { align: "start", label: "Site selector", className: "w-280" });
}

// ================================================================================================ notifications
function lastSeen() { return +ls.get("notif.seen", "0") || 0; }
export function setNewAlerts(list) {
  newAlerts = Array.isArray(list) ? list : [];
  const unread = newAlerts.filter(a => (a.updated_at || 0) > lastSeen()).length;
  const b = $("#bellCount");
  if (b) { b.hidden = !unread; b.textContent = unread > 99 ? "99+" : String(unread); }
  const btn = $("#bellBtn");
  if (btn) btn.setAttribute("aria-label", unread ? `Notifications, ${unread} unread` : "Notifications");
}
function bellMenu(anchor) {
  const seen = lastSeen();
  const items = newAlerts.slice(0, 8);
  popover(anchor, close => h("div", { class: "menu notif" },
    h("div", { class: "menu-h row" }, h("span", { class: "grow" }, "Notifications"),
      h("span", { class: "mute small" }, `${newAlerts.length >= 500 ? "500+" : newAlerts.length} awaiting triage`)),
    items.length ? h("div", { class: "notif-list" }, items.map(a => h("button", { class: ["notif-i", (a.updated_at || 0) > seen && "unread"], type: "button",
      onclick: () => { close(); drawer.setNavigator(null); drawer.openAlert(a.id); } },
    prioChip(a.priority),
    h("span", { class: "notif-b" }, h("span", { class: "notif-t" }, a.title),
      h("span", { class: "notif-m" }, `${(a.cameras || []).map(camName).join(", ")} · ${fmtAgo(a.updated_at)} `, slaChip(a, { compact: true }))))))
      : emptyState({ icon: "check", title: "All caught up", text: "No alerts are waiting for triage.", compact: true }),
    h("div", { class: "menu-sep" }),
    h("div", { class: "row menu-foot" }, h("a", { href: "#/review?status=new", class: "btnlink sm", onclick: () => close() }, "Open review queue"),
      h("span", { class: "grow" }))),
  { align: "end", label: "Notifications", className: "w-360",
    onclose: () => { ls.set("notif.seen", String(Math.max(Date.now() / 1000, ...newAlerts.map(a => a.updated_at || 0)))); setNewAlerts(newAlerts); } });
}

// ================================================================================================ user menu
export function renderUser(opts) {
  const { operator } = opts;
  const box = $("#userbox");
  if (!box) return;
  const name = S.me.open_mode ? (operator() === "operator" ? "Operator" : operator()) : S.me.username;
  const btn = h("button", { class: "avatar-btn", type: "button", "aria-haspopup": "dialog", "aria-expanded": "false",
    title: `${name} · account & display`, "aria-label": "Account and display menu" },
  h("span", { class: "avatar", "aria-hidden": "true" }, initials(name)),
  h("span", { class: "avatar-name hide-md" }, name), icon("chevdown", "hide-md"));
  btn.addEventListener("click", () => userMenu(btn, opts));
  replace(box, btn);
}
function userMenu(anchor, { logout, setOperator, operator, changePassword, signOutEverywhere }) {
  popover(anchor, close => {
    const demoSw = h("input", { type: "checkbox", checked: isDemo(), "aria-label": "Demo mode" });
    demoSw.addEventListener("change", () => setDemo(demoSw.checked));
    let opInput = null;
    if (S.me.open_mode) {
      opInput = h("input", { placeholder: "Your name (for the audit trail)", "aria-label": "Operator name", maxlength: 48,
        value: operator() === "operator" ? "" : operator() });
      opInput.addEventListener("change", () => { setOperator(opInput.value.trim()); });
      opInput.addEventListener("keydown", e => { if (e.key === "Enter") { setOperator(opInput.value.trim()); close(); } });
    }
    return h("div", { class: "menu" },
      h("div", { class: "menu-id" },
        h("span", { class: "avatar lg", "aria-hidden": "true" }, initials(S.me.open_mode ? operator() : S.me.username)),
        h("div", null, h("div", { class: "menu-name" }, S.me.open_mode ? "Demo access" : S.me.username),
          h("div", { class: "row", style: { gap: "4px", marginTop: "2px" } },
            S.me.open_mode ? h("span", { class: "tag warn" }, "authentication off") : S.me.roles.map(r => h("span", { class: "role" }, r))))),
      opInput ? h("label", { class: "field menu-pad" }, "Operator name", opInput,
        h("span", { class: "small mute" }, "Recorded with every action (unauthenticated in demo mode).")) : null,
      h("div", { class: "menu-sep" }),
      h("label", { class: "menu-i sw" }, icon("present"), h("span", { class: "grow" }, "Demo mode",
        h("span", { class: "menu-sub" }, "Presenter layout: larger type, Live follows new alerts")),
      h("span", { class: "switch" }, demoSw, h("span"))),
      autoOpenSwitch(),
      h("div", { class: "menu-i static" }, icon(effectiveTheme() === "dark" ? "moon" : "sun"), h("span", { class: "grow" }, "Theme"),
        segmented([["system", "Auto", "Follow the OS"], ["light", "Light"], ["dark", "Dark"]], themeMode(), v => setTheme(v), { label: "Theme" })),
      h("button", { class: "menu-i", type: "button", onclick: () => { close(); shortcutsHelp(); } }, icon("keyboard"), h("span", { class: "grow" }, "Keyboard shortcuts"), h("kbd", null, "?")),
      h("button", { class: "menu-i", type: "button", onclick: () => { close(); openPalette(); } }, icon("search"), h("span", { class: "grow" }, "Command palette"), h("kbd", null, "Ctrl K")),
      !S.me.open_mode ? [h("div", { class: "menu-sep" }),
        h("button", { class: "menu-i", type: "button", onclick: () => { close(); location.hash = "#/account"; } }, icon("key"), h("span", { class: "grow" }, "Account & API tokens")),
        S.me.auth !== "token" && changePassword ? h("button", { class: "menu-i", type: "button", onclick: () => { close(); changePassword(); } }, icon("lock"), h("span", { class: "grow" }, "Change password…")) : null,
        can("user.admin") ? h("button", { class: "menu-i", type: "button", onclick: () => { close(); location.hash = "#/users"; } }, icon("users"), h("span", { class: "grow" }, "Manage users")) : null,
        S.me.auth !== "token" && signOutEverywhere ? h("button", { class: "menu-i", type: "button", onclick: () => { close(); signOutEverywhere(); } }, icon("logout"), h("span", { class: "grow" }, "Sign out all sessions…")) : null,
        h("button", { class: "menu-i danger", type: "button", onclick: () => { close(); logout(); } }, icon("logout"), h("span", { class: "grow" }, "Sign out"))] : null);
  }, { align: "end", label: "Account", className: "w-320" });
}

// ================================================================================================ shortcuts help
export function shortcutsHelp() {
  const order = navOrder().slice(0, 9).map(id => getPage(id)?.label || id);
  const rows = [["Ctrl K  or  /", "Command palette: jump to a page, camera or alert"],
    ["1 – " + Math.max(1, order.length), order.join(" · ")],
    ["Esc", "Close drawer / dialog / palette; cancel drawing"],
    ["j / k  ·  ↓ / ↑", "Next / previous alert (Review, also inside the drawer)"], ["Enter", "Open selected alert (Review)"],
    ["f", "Fullscreen the focused camera (Live)"], ["[", "Collapse / expand sidebar"], ["t", "Toggle light / dark theme"],
    ["d", "Toggle demo (presenter) mode"], ["?", "This help"],
    ["Zone editor", "click = add point · drag = move point · double-click / Enter = close polygon · Backspace = undo point · Delete = remove selected"]];
  openDialog({ title: "Keyboard shortcuts", body: h("table", { class: "table compact" }, h("tbody", null,
    rows.map(([k, v]) => h("tr", null, h("td", { class: "nowrap" }, h("kbd", null, k)), h("td", null, v))))) });
}

// ================================================================================================ command palette
let pal = null;
let recent = { at: 0, list: [] };
async function recentAlerts() {
  if (Date.now() - recent.at < 30000) return recent.list;
  try { recent = { at: Date.now(), list: await api("/api/alerts?limit=300", { silent: true }) }; } catch { /* keep old */ }
  return recent.list;
}
function score(text, q) {
  text = String(text || "").toLowerCase();
  if (!q) return 1;
  const i = text.indexOf(q);
  if (i >= 0) return 100 - Math.min(60, i) + (i === 0 ? 30 : 0);
  let ti = 0, hits = 0;                      // subsequence fallback
  for (const c of q) { const k = text.indexOf(c, ti); if (k < 0) return 0; hits += k === ti ? 2 : 1; ti = k + 1; }
  return hits;
}
function commands() {
  const out = [];
  for (const p of visiblePages()) out.push({ group: "Pages", label: p.label, sub: "#/" + p.id, icon: p.icon, keys: [p.id, ...(p.keywords || [])], run: () => { location.hash = "#/" + p.id; } });
  for (const c of S.cams) {
    out.push({ group: "Cameras", label: c.name, sub: `Live view · ${c.id}`, icon: "live", keys: [c.id, "camera"], run: () => { location.hash = "#/live/" + enc(c.id); } });
    if (can("zone.edit") || can("settings.edit") || can("calibration.approve"))
      out.push({ group: "Cameras", label: `${c.name}: zones & calibration`, sub: c.id, icon: "cameras", keys: [c.id, "zones", "calibrate"], run: () => { location.hash = "#/cameras/" + enc(c.id); } });
  }
  out.push({ group: "Actions", label: "Toggle light / dark theme", icon: "moon", keys: ["theme", "dark", "light"], run: toggleTheme });
  out.push({ group: "Actions", label: isDemo() ? "Turn demo mode off" : "Turn demo (presenter) mode on", icon: "present", keys: ["demo", "presenter"], run: () => setDemo(!isDemo()) });
  out.push({ group: "Actions", label: "Keyboard shortcuts", icon: "keyboard", keys: ["help", "keys"], run: shortcutsHelp });
  out.push({ group: "Actions", label: "Show new alerts in Review", icon: "review", keys: ["triage", "queue", "new"], run: () => { location.hash = "#/review?status=new"; } });
  return out;
}
export function openPalette(initial = "") {
  if (pal) { pal.input.focus(); return; }
  closePopover();
  const input = h("input", { class: "pal-input", type: "text", placeholder: "Jump to a page, camera or alert id…", "aria-label": "Command palette search",
    autocomplete: "off", spellcheck: "false", role: "combobox", "aria-expanded": "true", "aria-controls": "palList", "aria-autocomplete": "list", value: initial });
  const list = h("div", { class: "pal-list", id: "palList", role: "listbox", "aria-label": "Results" });
  const wrap = h("div", { class: "pal-wrap" },
    h("div", { class: "pal-scrim", onclick: () => close() }),
    h("div", { class: "palette", role: "dialog", "aria-modal": "true", "aria-label": "Command palette" },
      h("div", { class: "pal-top" }, icon("search"), input, h("kbd", null, "Esc")),
      list,
      h("div", { class: "pal-foot" }, h("span", null, h("kbd", null, "↑"), h("kbd", null, "↓"), " navigate"), h("span", null, h("kbd", null, "Enter"), " open"),
        h("span", { class: "grow" }), h("span", { class: "mute" }, "Alert ids look like A1790301878-496"))));
  let results = [], sel = 0, alerts = recent.list, prevFocus = document.activeElement;
  const close = () => { wrap.remove(); pal = null; document.removeEventListener("keydown", onKey, true); prevFocus?.focus?.({ preventScroll: true }); };
  const run = r => { close(); try { r.run(); } catch (e) { console.error(e); } };
  function compute() {
    const q = input.value.trim().toLowerCase();
    const cmds = commands().map(c => ({ ...c, sc: Math.max(score(c.label, q), ...c.keys.map(k => score(k, q) * 0.8)) })).filter(c => c.sc > 0);
    const al = (q ? alerts : alerts.slice(0, 5)).map(a => ({
      group: "Alerts", label: a.title, sub: `${a.id} · ${(a.cameras || []).map(camName).join(", ")} · ${a.status} · ${fmtAgo(a.updated_at)}`,
      prio: a.priority, keys: [a.id],
      sc: q ? Math.max(score(a.id, q) * 1.2, score(a.title, q) * 0.7, score((a.zones || []).join(" "), q) * 0.6, score((a.cameras || []).map(camName).join(" "), q) * 0.5) : 1,
      run: () => { drawer.setNavigator(null); drawer.openAlert(a.id); },
    })).filter(a => a.sc > 0).sort((a, b) => b.sc - a.sc).slice(0, 6);
    if (/^a\d{6,}(-\d+)?$/i.test(q) && !al.some(a => a.keys[0].toLowerCase() === q))
      al.unshift({ group: "Alerts", label: `Open alert ${input.value.trim()}`, sub: "Look up by id", icon: "review", run: () => { drawer.setNavigator(null); drawer.openAlert(input.value.trim()); } });
    const grouped = {};
    for (const c of cmds.sort((a, b) => b.sc - a.sc)) (grouped[c.group] ||= []).push(c);
    results = [...(grouped.Pages || []).slice(0, q ? 5 : 12), ...(grouped.Cameras || []).slice(0, q ? 6 : 4), ...al, ...(grouped.Actions || []).slice(0, q ? 4 : 4)];
    sel = Math.min(sel, Math.max(0, results.length - 1));
    render();
  }
  function render() {
    let g = null;
    const nodes = [];
    results.forEach((r, i) => {
      if (r.group !== g) { g = r.group; nodes.push(h("div", { class: "pal-g", role: "presentation" }, g)); }
      nodes.push(h("div", { class: ["pal-i", i === sel && "on"], role: "option", id: "pal-" + i, "aria-selected": String(i === sel),
        onmousemove: () => { if (sel !== i) { sel = i; mark(); } }, onclick: () => run(r) },
      r.prio ? prioChip(r.prio) : icon(r.icon || "chevright"),
      h("span", { class: "pal-l" }, h("span", { class: "pal-t" }, r.label), r.sub ? h("span", { class: "pal-s" }, r.sub) : null),
      i === sel ? h("kbd", { class: "pal-enter" }, "↵") : null));
    });
    replace(list, nodes.length ? nodes : emptyState({ icon: "search", title: "No matches", text: "Try a camera name, a page, or an alert id.", compact: true }));
    input.setAttribute("aria-activedescendant", results.length ? "pal-" + sel : "");
  }
  function mark() {
    list.querySelectorAll(".pal-i").forEach((el, i) => { el.classList.toggle("on", i === sel); el.setAttribute("aria-selected", String(i === sel)); });
    input.setAttribute("aria-activedescendant", "pal-" + sel);
    list.querySelector(".pal-i.on")?.scrollIntoView({ block: "nearest" });
  }
  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); return; }
    if (e.key === "ArrowDown" || (e.key === "n" && e.ctrlKey)) { e.preventDefault(); sel = Math.min(results.length - 1, sel + 1); mark(); return; }
    if (e.key === "ArrowUp" || (e.key === "p" && e.ctrlKey)) { e.preventDefault(); sel = Math.max(0, sel - 1); mark(); return; }
    if (e.key === "Enter") { e.preventDefault(); if (results[sel]) run(results[sel]); return; }
    if (e.key === "Tab") { e.preventDefault(); input.focus(); }
  }
  input.addEventListener("input", () => { sel = 0; compute(); });
  document.addEventListener("keydown", onKey, true);
  document.body.appendChild(wrap);
  pal = { input, close };
  input.focus();
  compute();
  recentAlerts().then(l => { if (pal) { alerts = l; compute(); } });
}
export function isPaletteOpen() { return !!pal; }

// ================================================================================================ status bar
const SB = { stats: null, statsAt: 0, nodes: null, nodesMissing: false, nodesAt: 0 };
export function statusSnapshot() { return SB; }

function camsSummary() {
  let online = 0, degraded = 0, down = 0;
  for (const c of S.cams) {
    if (c.enabled === false) continue;
    const st = S.stats[c.id];
    if (!st) continue;
    if (st.status !== "running" || st.health === "DOWN") down++;
    else if (st.health === "DEGRADED") degraded++;
    else online++;
  }
  return { online, degraded, down, total: S.cams.filter(c => c.enabled !== false).length };
}
export { camsSummary };

async function pollSlow() {
  if (document.hidden) return;
  if (Date.now() - SB.statsAt > 20000) {
    SB.statsAt = Date.now();
    try { SB.stats = await api("/api/stats", { silent: true }); } catch { /* keep */ }
  }
  if (!SB.nodesMissing && hasEndpoint("/api/nodes") && Date.now() - SB.nodesAt > 15000) {
    SB.nodesAt = Date.now();
    try { SB.nodes = await api("/api/nodes", { silent: true }); }
    catch (e) { if (e.kind === "missing" || e.kind === "notfound") SB.nodesMissing = true; }
  }
  if (hasEndpoint("/api/nodes") === false) SB.nodesMissing = true;
  renderStatus();
}

function nodesSummary() {
  const raw = SB.nodes;
  const list = Array.isArray(raw) ? raw : Array.isArray(raw?.nodes) ? raw.nodes : null;
  if (!list) return null;
  const up = list.filter(n => n && (n.online === true || n.healthy === true || /^(ok|online|up|ready|healthy|running)$/i.test(String(n.status || n.state || "")))).length;
  const gpu = list.map(n => n?.gpu?.name || n?.gpu_name || (Array.isArray(n?.gpus) && n.gpus[0]?.name) || "").find(Boolean);
  return { up, total: list.length, gpu };
}

const renderStatus = throttle(() => {
  const bar = $("#statusbar");
  if (!bar || bar.hidden) return;
  const cs = camsSummary();
  const people = S.tracks.filter(t => (t.cls || "person") === "person").length;
  const vals = Object.values(S.stats || {});
  const fps = vals.length ? vals.reduce((a, s) => a + (+s.fps || 0), 0) / vals.length : null;
  const det = vals.length ? vals.reduce((a, s) => a + (+s.detect_ms || 0), 0) / vals.length : null;
  const m = SB.stats?.model || {};
  const ns = nodesSummary();
  const wsText = { live: "Live", connecting: "Connecting…", retry: "Reconnecting…", offline: "Offline", auth: "Sign-in required" }[S.ws] || S.ws;
  replace(bar,
    h("span", { class: "sb-i", "data-state": S.ws, title: "Real-time link to the Argus core (WebSocket)" }, h("span", { class: "dot" }), wsText),
    h("span", { class: "sb-i", title: "Cameras online / degraded / down (from worker health)" }, icon("cameras"),
      `${cs.online}/${cs.total} online`, cs.degraded ? h("b", { class: "c-warn" }, ` · ${cs.degraded} degraded`) : null, cs.down ? h("b", { class: "c-err" }, ` · ${cs.down} down`) : null),
    h("span", { class: "sb-i hide-md", title: "Person tracks currently live across all cameras" }, icon("user"), `${people} tracked`),
    h("span", { class: "sb-i hide-md", title: "Mean analytics rate and detection time per frame (measured on this host)" }, icon("signal"),
      fps != null ? `${num(fps, 1)} fps · detect ${fmtMs(det)}` : "waiting for stats"),
    h("span", { class: "grow" }),
    ns ? h("span", { class: "sb-i", title: "Compute nodes reporting to this core" }, icon("nodes"), `Nodes ${ns.up}/${ns.total} online`, ns.gpu ? h("span", { class: "mute" }, ` · ${ns.gpu}`) : null)
      : h("span", { class: "sb-i hide-md", title: "Inference device of this Argus core" }, icon("cpu"),
        m.device ? `${String(m.device).toUpperCase()} · ${m.runtime || m.framework || "detector"}` : "Compute: this host"),
    h("span", { class: "sb-i tnum", id: "sbClock", title: "Local time" }, icon("clock"), new Date().toLocaleTimeString()),
    h("span", { class: "sb-i hide-md", title: "Signed-in operator" }, icon("user"), S.me.open_mode ? (ls.get("operator", "") || "operator") + " (demo)" : S.me.username));
}, 1000);

// ================================================================================================ init
export function initShell({ logout, onToggleSidebar } = {}) {
  paintThemeIcon();
  $("#themeBtn")?.addEventListener("click", toggleTheme);
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { paintThemeIcon(); emit("theme", effectiveTheme()); });
  if (ls.get("demo") === "1") setDemo(true, { quiet: true });
  $("#demoPill")?.addEventListener("click", () => setDemo(false));
  $("#searchBtn")?.addEventListener("click", () => openPalette());
  $("#bellBtn")?.addEventListener("click", e => bellMenu(e.currentTarget));
  $("#siteBtn")?.addEventListener("click", e => siteMenu(e.currentTarget));
  $("#navScrim")?.addEventListener("click", () => toggleMobileNav(false));
  $("#navClose")?.addEventListener("click", () => toggleMobileNav(false));
  $("#collapseBtn")?.addEventListener("click", () => {
    if (matchMedia("(max-width: 720px)").matches) { toggleMobileNav(); return; }
    onToggleSidebar?.();
  });
  onPagesChanged(throttle(renderNav, 50));
  on("tick", renderStatus);
  on("ws", renderStatus);
  on("cameras", renderStatus);
  setInterval(() => {
    if (document.hidden) return;
    const c = $("#sbClock");
    if (c) replace(c, icon("clock"), new Date().toLocaleTimeString());
  }, 1000);
  setInterval(pollSlow, 5000);
  pollSlow();
  renderNav();
  renderStatus();
  void logout;
}

/** Global keys owned by the shell. Returns true when handled. */
export function handleKey(e) {
  if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "k" || e.key === "K")) { e.preventDefault(); pal ? pal.close() : openPalette(); return true; }
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  if (pal) return false;
  if (isTyping(e)) return false;
  if (e.key === "/") { e.preventDefault(); openPalette(); return true; }
  if (e.key === "d") { setDemo(!isDemo()); return true; }
  return false;
}
