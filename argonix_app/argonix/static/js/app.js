// App shell: session, router, WebSocket (auto-reconnect), shortcuts, sidebar, badge.
// Pages plug in through registerPage() (registry.js, re-exported here). The legacy PAGES object below still works:
// adding `foo` to PAGES (plus an optional <a class="nav-item" data-page="foo"> in index.html) registers it too.
import { api, optional, ApiError, onAuthRequired, operatorName } from "./api.js";
import { S, can, emit, on, loadConfig, loadMap, loadVlms, camName } from "./state.js";
import { toast, isDialogOpen } from "./ui.js";
import { $, h, replace, ls, throttle, isTyping, hasIcon, enc, icon } from "./util.js";
import { floodState, overdueCount, autoOpen } from "./workflow.js";
import * as drawer from "./drawer.js";
import * as account from "./account.js";
import users from "./users.js";
import live from "./live.js";
import review from "./review.js";
import map from "./map.js";
import recordings from "./recordings.js";
import cameras from "./cameras.js";
import system from "./system.js";
import incidents from "./incidents.js";
import investigate from "./investigate.js";
import handover from "./handover.js";

const PAGES = { live, review, map, recordings, cameras, system };
const ORDER = ["live", "review", "map", "recordings", "cameras", "system"];
const TITLES = { live: "Live", review: "Review", map: "Map", recordings: "Recordings", cameras: "Cameras", system: "System" };

import overview from "./overview.js";
import "./pages/index.js";   // Analytics + Journeys pages (registerPage, lazy-loaded)
import { registerPage, getPage, hasPage, permitted, probeEndpoints } from "./registry.js";
import * as shell from "./shell.js";
export { registerPage } from "./registry.js";

registerPage({ id: "overview", label: "Overview", icon: "overview", group: "monitor", order: 10, module: overview,
  keywords: ["home", "dashboard", "kpi"] });
// Settings -> Users (admins, only when sign-in is on) and the per-user Account page (user menu; not in the sidebar).
registerPage({ id: "users", label: "Users", icon: "users", group: "configure", order: 120, module: users,
  perm: () => can("user.admin") && !S.me.open_mode, keywords: ["accounts", "roles", "tokens", "admin"] });
registerPage({ id: "account", label: "Account", icon: "user", group: "configure", hidden: true, module: account.createAccountPage,
  keywords: ["password", "token", "api", "sign out"] });

registerPage({ id: "incidents", module: incidents, perm: "incident.read", probe: "/api/incidents",
  keywords: ["case", "graph", "timeline", "correlation"] });
registerPage({ id: "investigate", module: investigate, perm: "alert.read", probe: "/api/investigate",
  keywords: ["ask", "question", "search", "natural language", "chat"] });
registerPage({ id: "handover", label: "Handover", icon: "clock", group: "monitor", order: 85, module: handover,
  perm: "alert.read", probe: "/api/handover", keywords: ["shift", "handoff note", "unresolved", "overdue", "print"] });

const DEFAULT_PAGE = "overview";
const inst = {};           // created page instances
let current = null;        // {name, page}
let booted = false;

// ---------- global safety net: no unhandled rejections reach the console as errors ----------
window.addEventListener("unhandledrejection", e => {
  e.preventDefault();
  const r = e.reason;
  if (r instanceof ApiError) return;      // already surfaced by the api wrapper
  console.error("Unhandled:", r);
  toast(`Unexpected error: ${r?.message || r}`, "error", { key: "unhandled" });
});

/** Pages added the legacy way (PAGES entry + optional static nav anchor) join the registry as fallbacks. */
function adoptLegacyPages() {
  const anchors = new Map([...document.querySelectorAll("#sidebar > a.nav-item[data-page]")].map(a => [a.dataset.page, a]));
  for (const [id, factory] of Object.entries(PAGES)) {
    if (typeof factory !== "function") continue;
    const a = anchors.get(id);
    registerPage({ id, module: factory, _fallback: true,
      label: TITLES[id] || a?.querySelector(".lbl")?.textContent?.trim(),
      icon: hasIcon(id) ? id : (a?.querySelector("svg") || undefined) });
  }
}

// ---------- sidebar ----------
const appEl = $("#app");
if (ls.get("sidebar") === "collapsed") appEl.classList.add("collapsed");
function toggleSidebar() {
  appEl.classList.toggle("collapsed");
  ls.set("sidebar", appEl.classList.contains("collapsed") ? "collapsed" : null);
  $("#collapseBtn").setAttribute("aria-expanded", String(!appEl.classList.contains("collapsed")));
  window.dispatchEvent(new Event("resize"));
}

// ---------- sign-in / setup screens (rendered by account.js into #login) ----------
let loginShown = false;
function showLogin(reason) {
  if (loginShown) return;
  loginShown = true;
  ws.stop("auth");
  drawer.closeDrawer();
  if (isDialogOpen()) $("#dialog").close();
  account.showLoginScreen(reason);
}
onAuthRequired(showLogin);

const logout = () => account.logout();

function renderUserbox() {
  const off = $("#authOff");
  off.hidden = !S.me.open_mode;
  if (S.me.open_mode) {
    const canSetup = !!S.auth?.setup_required;
    off.title = "Open demo mode (started with --open-demo): anyone who can reach this server has full access and actions are unauthenticated."
      + (canSetup ? " Click to create the first admin account and turn sign-in on." : "");
    off.onclick = canSetup ? () => account.showSetupScreen(S.auth, { onCancel: () => {} }) : null;
    off.classList.toggle("click", canSetup);
  }
  shell.renderUser({
    logout,
    changePassword: account.changePasswordDialog,
    signOutEverywhere: account.signOutEverywhere,
    operator: operatorName,
    setOperator: v => { ls.set("operator", (v || "").slice(0, 48) || null); renderUserbox(); },
  });
}

// ---------- WebSocket with backoff + watchdog ----------
const ws = (() => {
  let sock = null, attempt = 0, timer = null, watchdog = null, lastMsg = 0, stopped = false, retryAt = 0, countdown = null;
  function setState(st, text) {
    S.ws = st;
    const c = $("#conn");
    c.dataset.state = st;
    c.title = `Live connection (WebSocket): ${text}`;
    $("#connText").textContent = text;
    emit("ws", st);
  }
  function connect() {
    if (stopped) return;
    clearTimeout(timer);
    setState(attempt ? "retry" : "connecting", attempt ? "reconnecting…" : "connecting…");
    try {
      sock = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    } catch { schedule(); return; }
    sock.onopen = () => {
      const wasRetry = attempt > 0;
      attempt = 0; lastMsg = Date.now();
      setState("live", "live");
      if (wasRetry) emit("resync");
      clearInterval(watchdog);
      watchdog = setInterval(() => { if (Date.now() - lastMsg > 10000 && sock?.readyState === 1) sock.close(); }, 3000);
    };
    sock.onmessage = ev => {
      lastMsg = Date.now();
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === "tick") {
        S.tracks = Array.isArray(m.tracks) ? m.tracks : [];
        S.stats = m.stats || {};
        emit("tick");
      } else if (m.type === "alert" && m.alert) emit("alert", m.alert);
      else if (m.type === "incident" && m.incident) emit("incident", m.incident);
    };
    sock.onclose = ev => {
      clearInterval(watchdog);
      sock = null;
      if (stopped) return;
      if (ev.code === 4401) {           // server refused: session missing/expired
        setState("auth", "sign-in required");
        api("/api/me", { silent: true }).catch(() => {});  // triggers the login overlay on 401
        schedule(15000);
        return;
      }
      schedule();
    };
    sock.onerror = () => { /* onclose follows */ };
  }
  function schedule(min = 0) {
    attempt++;
    const delay = Math.max(min, Math.min(30000, 1000 * 2 ** Math.min(attempt - 1, 5)) * (0.8 + Math.random() * 0.4));
    retryAt = Date.now() + delay;
    clearInterval(countdown);
    const tick = () => {
      const s = Math.max(0, Math.ceil((retryAt - Date.now()) / 1000));
      if (S.ws !== "auth") setState(navigator.onLine === false ? "offline" : "retry", `${navigator.onLine === false ? "offline" : "disconnected"} · retry in ${s}s`);
    };
    tick(); countdown = setInterval(tick, 1000);
    timer = setTimeout(() => { clearInterval(countdown); connect(); }, delay);
  }
  $("#conn").addEventListener("click", () => { if (!sock && !stopped) { clearTimeout(timer); clearInterval(countdown); connect(); } });
  window.addEventListener("online", () => { if (!sock && !stopped) { clearTimeout(timer); clearInterval(countdown); connect(); } });
  return {
    start() { stopped = false; connect(); },
    stop(state) { stopped = true; clearTimeout(timer); clearInterval(watchdog); clearInterval(countdown); sock?.close(); if (state) setState(state, state === "auth" ? "sign-in required" : state); },
  };
})();

// ---------- alerts: badge + bell + toasts ----------
const seen = new Set();
let newCount = 0;
function paintTitle() {
  const p = current ? getPage(current.name)?.label : null;
  document.title = (newCount ? `(${newCount >= 500 ? "500+" : newCount}) ` : "") + "Argus" + (p ? ` · ${p}` : "");
}
let newList = [], newAt = 0;
/** Acknowledge-SLA badge and alert-flood banner: time-dependent, so recomputed every few seconds from the last list. */
function paintQueue() {
  shell.setLate("review", overdueCount(newList));
  S.flood = floodState(newList);
  shell.setFlood(S.flood);
}
const refreshBadge = throttle(async () => {
  try {
    const list = await api("/api/alerts?status=new&limit=500", { silent: true });
    newList = list; newAt = Date.now();
    newCount = list.length;
    shell.setBadge("review", list.length >= 500 ? 500 : list.length);
    shell.setNewAlerts(list);
    paintQueue();
    paintTitle();
    emit("new-alerts", list);
  } catch { /* polling; ignore */ }
}, 3000);
setInterval(() => {
  if (!booted || document.hidden) return;
  if (Date.now() - newAt > 30000) refreshBadge(); else paintQueue();
}, 5000);

on("alert", a => {
  refreshBadge();
  drawer.onAlertUpdate(a);
  if (!seen.has(a.id) && a.status === "new") {
    seen.add(a.id);
    if (shell.isDemo()) emit("spotlight", a);
    const ev = a.events?.[a.events.length - 1];
    toast(h("div", null,
      h("div", { class: "row", style: { gap: "6px" } }, h("span", { class: ["chip", a.priority] }, a.priority), h("b", null, a.title)),
      h("div", { class: "small mute" }, `${(a.cameras || []).map(camName).join(", ")} · ${new Date(a.updated_at * 1000).toLocaleTimeString()}`,
        ev ? ` · conf ${ev.confidence} (heuristic)` : "")),
      a.priority === "P1" ? "error" : a.priority === "P2" ? "warn" : "info",
      { key: "alert:" + a.id, onclick: () => { drawer.setNavigator(null); drawer.openAlert(a.id); }, timeout: 8000, icon: "bell" });
  }
});
on("alert-changed", () => refreshBadge());
on("resync", () => refreshBadge());

// ---------- incidents: P1/P2 created or escalated -> toast with "Open cameras" (+ optional auto-open) ----------
// Incidents arrive on the WebSocket ({type: "incident"}); while it is down, /api/incidents is polled instead.
const RANK = { P1: 0, P2: 1, P3: 2, P4: 3 };
const incSeen = new Map();   // id -> {r: priority rank, sla: overdue already seen}
const RECENT_S = 300;
function openIncidentCameras(iid) { location.hash = "#/live?incident=" + enc(iid); }
function onIncident(inc) {
  if (!inc?.id || inc.merged_into) return;
  const prev = incSeen.get(inc.id);
  const r = RANK[inc.priority] ?? 9, slaOver = inc.sla?.state === "overdue";
  incSeen.delete(inc.id);
  incSeen.set(inc.id, { r, sla: slaOver });
  if (incSeen.size > 1000) incSeen.delete(incSeen.keys().next().value);
  if (inc.phase === "closed" || inc.suppressed || r > 1) return;
  const now = Date.now() / 1000;
  let why = null;
  if (!prev) {   // first sight: only when it opened (or reached this priority) recently, not on every reconnect
    if (now - (+inc.opened_at || 0) < RECENT_S) why = "new";
    else if (now - (+inc.prio_since?.[inc.priority] || 0) < RECENT_S) why = "escalated";
  } else if (r < prev.r) why = "escalated";
  else if (slaOver && !prev.sla) why = "sla";
  if (why) attention(inc, why);
}
function attention(inc, why) {
  const cams = (inc.cameras || []).map(camName).join(", ");
  const text = { new: "New incident", escalated: `Escalated to ${inc.priority}`, sla: "Not acknowledged within the SLA" }[why];
  const t = toast(h("div", null,
    h("div", { class: "row", style: { gap: "6px" } }, h("span", { class: ["chip", inc.priority] }, inc.priority), h("b", null, inc.title || inc.id)),
    h("div", { class: "small mute" }, `${text} · ${cams || "no camera"}`),
    h("div", { class: "row", style: { gap: "6px", marginTop: "6px" } },
      h("button", { class: "sm primary", type: "button", onclick: e => { e.stopPropagation(); e.currentTarget.closest(".toast")?.remove(); openIncidentCameras(inc.id); } },
        icon("live"), "Open cameras"),
      h("a", { class: "btnlink sm", href: "#/incidents/" + enc(inc.id), onclick: e => { e.stopPropagation(); e.currentTarget.closest(".toast")?.remove(); } }, "Incident"))),
  inc.priority === "P1" ? "error" : "warn", { key: `inc:${inc.id}:${why}:${inc.priority}`, timeout: 15000, icon: "incidents", dedupeMs: 60000 });
  // auto-open never interrupts typing or an open dialog; the toast above is still shown
  if (t && autoOpen() && !isDialogOpen() && !isTyping({ target: document.activeElement })) openIncidentCameras(inc.id);
}
on("incident", onIncident);
setInterval(async () => {   // polling fallback while the live connection is down
  if (!booted || S.ws === "live" || document.hidden || !can("incident.read") || !hasPage("incidents")) return;
  try { (await api("/api/incidents?limit=30", { silent: true })).reverse().forEach(onIncident); } catch { /* ignore */ }
}, 10000);

// ---------- router ----------
function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  let name = hasPage(parts[0]) ? parts[0] : (hasPage(DEFAULT_PAGE) ? DEFAULT_PAGE : "live");
  if (!permitted(getPage(name))) {
    toast(`Your role cannot open "${getPage(name)?.label || name}".`, "warn", { key: "perm:" + name });
    name = DEFAULT_PAGE;
  }
  return { name, args: name === parts[0] ? parts.slice(1) : [], query: new URLSearchParams(qs || "") };
}

async function instantiate(name) {
  if (inst[name]) return inst[name];
  const p = getPage(name);
  let factory = p.factory;
  if (!factory && p.load) {
    const m = await p.load();
    factory = typeof m === "function" ? m : m?.default;
    registerPage({ id: name, module: factory });
  }
  inst[name] = factory();
  inst[name].el.classList.add("page");
  inst[name].el.dataset.page = name;
  $("#pages").appendChild(inst[name].el);
  return inst[name];
}

let lastHash = location.hash, bypassGuard = false;
async function route() {
  if (!booted) return;
  const { name, args, query } = parseHash();
  if (current?.page.canLeave && !bypassGuard) {
    const ok = current.page.canLeave(name, args);   // true | false | Promise<boolean>
    if (ok !== true) {
      const target = location.hash;
      history.replaceState(null, "", lastHash || "#/" + current.name);   // revert without re-routing
      if (ok && await ok) { bypassGuard = true; location.hash = target; }
      return;
    }
  }
  bypassGuard = false;
  lastHash = location.hash;
  if (!current || current.name !== name) {
    if (current) { current.page.hide?.(); current.page.el.hidden = true; }
    let page;
    try { page = await instantiate(name); }
    catch (e) { console.error(e); toast(`Could not load page "${name}": ${e.message}`, "error"); return; }
    current = { name, page };
    current.page.el.hidden = false;
    $("#pages").scrollTop = 0;
  }
  shell.setActive(name);
  paintTitle();
  try { await current.page.show?.(args, query); }
  catch (e) { if (!(e instanceof ApiError)) { console.error(e); toast(`Page error: ${e.message}`, "error"); } }
}
window.addEventListener("hashchange", route);
window.addEventListener("beforeunload", e => {
  if (current?.page.dirty?.()) { e.preventDefault(); e.returnValue = ""; }
});

// Page broadcast hooks
on("tick", () => current?.page.onTick?.());
on("alert", a => current?.page.onAlert?.(a));
on("incident", inc => current?.page.onIncident?.(inc));
on("resync", () => current?.page.onResync?.());
document.addEventListener("visibilitychange", () => current?.page.onVisibility?.(!document.hidden));

// ---------- keyboard ----------
document.addEventListener("keydown", e => {
  if (!booted || account.screenShown()) return;
  if (isDialogOpen()) return;                                   // <dialog> handles Esc itself
  if (shell.handleKey(e)) return;                               // Ctrl+K, "/", "d"
  if (shell.isPaletteOpen() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (current?.page.onKey?.(e)) { e.preventDefault(); return; }  // page-specific keys (editor, review) first
  if (e.key === "Escape") {
    if (drawer.isOpen()) { drawer.closeDrawer(); e.preventDefault(); return; }
    if ($("#app").classList.contains("nav-open")) { shell.toggleMobileNav(false); e.preventDefault(); return; }
    if (isTyping(e)) { e.target.blur(); return; }
  }
  if (isTyping(e)) return;
  if (/^[1-9]$/.test(e.key)) {
    const id = shell.navOrder()[+e.key - 1];
    if (id) { location.hash = "#/" + id; e.preventDefault(); }
    return;
  }
  if (drawer.isOpen() && (e.key === "j" || e.key === "k")) { drawer.navigate(e.key === "j" ? 1 : -1); e.preventDefault(); return; }
  if (e.key === "?") { shell.shortcutsHelp(); e.preventDefault(); return; }
  if (e.key === "[") { toggleSidebar(); e.preventDefault(); return; }
  if (e.key === "t") { shell.toggleTheme(); e.preventDefault(); }
});
$(".side-foot")?.addEventListener("click", () => shell.shortcutsHelp());

// ---------- boot ----------
async function boot() {
  const bootMsg = $("#boot");
  if (bootMsg) bootMsg.dataset.state = "loading";
  shell.initShell({ logout, onToggleSidebar: toggleSidebar });
  adoptLegacyPages();
  const probe = probeEndpoints();
  const unreachable = () => replace($("#pages"), h("div", { class: "banner err" }, "Cannot reach the Argus server. ",
    h("button", { onclick: () => location.reload() }, "Retry")));
  try {
    S.auth = await optional("/api/auth/status", "authstatus", { silent: true });   // null on older servers
  } catch (e) {
    if (e.kind === "network") { unreachable(); return; }
    S.auth = null;
  }
  if (S.auth?.setup_required && !S.auth.open_demo) { account.showSetupScreen(S.auth); return; }
  try {
    S.me = await api("/api/me", { silent: true });
  } catch (e) {
    if (e.kind !== "auth") unreachable();
    return;
  }
  if (S.me.must_change) { account.showForcedChangeScreen(S.me); return; }
  renderUserbox();
  account.startIdleWatch();
  document.body.classList.toggle("can-settings", can("settings.edit"));
  try { await loadConfig(); }
  catch (e) {
    if (e.kind === "forbidden") replace($("#pages"), h("div", { class: "banner err" }, "Your role cannot read alerts or configuration."));
    return;
  }
  await probe;
  $("#siteName").textContent = S.cfg.site || "Argus";
  $("#siteName").title = S.cfg.site || "";
  shell.renderNav();
  booted = true;
  $("#boot")?.remove();
  ws.start();
  refreshBadge();
  loadVlms();
  loadMap();
  if (!location.hash || !hasPage(location.hash.replace(/^#\/?/, "").split(/[/?]/)[0])) history.replaceState(null, "", "#/" + DEFAULT_PAGE);
  route();
}
boot();
