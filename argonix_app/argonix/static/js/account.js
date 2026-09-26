// Accounts UI: sign-in / first-run setup / forced password change screens, user-menu actions (change password,
// sign out, sign out everywhere), the idle-timeout watcher, and the Account page (profile + API tokens).
// Server strings only ever reach the DOM as text nodes (h()); never innerHTML.
import { api, optional } from "./api.js";
import { S } from "./state.js";
import { toast, openDialog, confirmDialog, featureMissing, kv } from "./ui.js";
import { $, h, replace, icon, enc, fmtDateTime, fmtAgo, fmtDur } from "./util.js";

const FLASH_KEY = "argonix.flash";
const FLASH = {
  logout: "You have signed out.",
  all: "Signed out of all sessions.",
};
const minPw = () => S.auth?.password_min_length || 12;

function setFlash(kind) { try { sessionStorage.setItem(FLASH_KEY, kind); } catch { /* storage unavailable */ } }
function takeFlash() {
  try { const v = sessionStorage.getItem(FLASH_KEY); sessionStorage.removeItem(FLASH_KEY); return v; } catch { return null; }
}

/** Human text for the server's 401 reason. */
export function reasonText(reason) {
  const r = String(reason || "");
  if (r.includes("idle")) return "Signed out due to inactivity.";
  if (r.includes("maximum")) return "Your session reached its maximum length. Please sign in again.";
  if (r.includes("revoked")) return "You were signed out: your password was changed or all sessions were signed out.";
  if (r.includes("disabled")) return "This account is disabled. Ask an administrator.";
  return "";
}

/** POST without the api() wrapper (these screens run before/without a session; they show errors inline). */
export async function post(path, body = {}) {
  let r;
  try {
    r = await fetch(path, { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Argonix-CSRF": "1" }, body: JSON.stringify(body) });
  } catch { return { ok: false, status: 0, detail: "Server unreachable.", data: null }; }
  let data = null, detail = "";
  try { data = await r.json(); detail = typeof data?.detail === "string" ? data.detail : ""; } catch { /* no body */ }
  return { ok: r.ok, status: r.status, detail: detail || (r.ok ? "" : `HTTP ${r.status}`), data };
}

// ---------------------------------------------------------------------------------------------------------------
// password inputs with live hints (the server's policy is authoritative; this only gives early feedback)
function passwordFields(getUsername, { label = "New password" } = {}) {
  const pw = h("input", { type: "password", autocomplete: "new-password", required: true, "aria-label": label, placeholder: label });
  const pw2 = h("input", { type: "password", autocomplete: "new-password", required: true, "aria-label": "Repeat password", placeholder: "Repeat password" });
  const show = h("input", { type: "checkbox" });
  show.addEventListener("change", () => { pw.type = pw2.type = show.checked ? "text" : "password"; });
  const hints = h("ul", { class: "pw-hints", "aria-live": "polite" });
  const checks = () => {
    const v = pw.value, u = String(getUsername() || "").trim().toLowerCase();
    return [[v.length >= minPw(), `At least ${minPw()} characters`],
      [!(u.length >= 3 && v.toLowerCase().includes(u)), "Does not contain the username"],
      [!!v && v === pw2.value, "Both entries match"]];
  };
  const update = () => replace(hints, checks().map(([ok, t]) => h("li", { class: ok ? "ok" : "" }, ok ? "✓ " : "○ ", t)),
    h("li", null, "Common or predictable passwords (e.g. Summer2024!, P@ssw0rd…) are refused by the server."));
  pw.addEventListener("input", update); pw2.addEventListener("input", update);
  update();
  return {
    pw, pw2, update,
    el: h("div", { class: "stack", style: { gap: "6px" } }, pw, pw2,
      h("label", { class: "check small mute" }, show, "Show passwords"), hints),
    problem() { const bad = checks().find(([ok]) => !ok); return bad ? bad[1].replace(/^Both/, "The two entries must") : ""; },
  };
}

/** Random temporary password (unambiguous characters, 4 groups of 5): ~99 bits from crypto.getRandomValues. */
export function randomPassword() {
  const A = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789";
  const out = [];
  const buf = new Uint32Array(20);
  crypto.getRandomValues(buf);
  for (let i = 0; i < 20; i++) out.push(A[buf[i] % A.length]);
  return [0, 5, 10, 15].map(i => out.slice(i, i + 5).join("")).join("-");
}

// ---------------------------------------------------------------------------------------------------------------
// full-screen auth screens (#login overlay)
function screen(form) {
  const box = $("#login");
  replace(box, form);
  box.hidden = false;
  setTimeout(() => form.querySelector("input:not([type=checkbox])")?.focus(), 0);
}
export function hideScreen() { const box = $("#login"); box.hidden = true; replace(box); }
export function screenShown() { return !$("#login").hidden; }

export function showLoginScreen(reason) {
  if (reason && String(reason).includes("setup required") && S.auth) { showSetupScreen(S.auth); return; }
  const user = h("input", { autocomplete: "username", placeholder: "Username", required: true, "aria-label": "Username", maxLength: 128 });
  const pass = h("input", { type: "password", autocomplete: "current-password", placeholder: "Password", required: true, "aria-label": "Password" });
  const err = h("div", { class: "small err-text", role: "alert" });
  const btn = h("button", { class: "primary", type: "submit" }, "Sign in");
  const msg = FLASH[takeFlash()] || reasonText(reason);
  const form = h("form", { autocomplete: "on", "aria-label": "Sign in" },
    h("h2", null, "Argus sign in"),
    msg ? h("div", { class: "banner", role: "status" }, msg) : null,
    user, pass, btn, err);
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    err.textContent = ""; btn.disabled = true;
    const r = await post("/api/login", { username: user.value, password: pass.value });
    if (r.ok) { location.reload(); return; }
    err.textContent = r.status === 401 ? "Sign-in failed: wrong username or password, or too many attempts (then wait 15 minutes)."
      : r.status === 0 ? "Server unreachable." : `Sign-in failed: ${r.detail}`;
    btn.disabled = false; pass.value = ""; pass.focus();
  });
  screen(form);
}

export function showSetupScreen(status, { onCancel = null } = {}) {
  const user = h("input", { autocomplete: "username", placeholder: "Admin username (e.g. alice)", required: true, maxLength: 48, "aria-label": "Admin username" });
  const f = passwordFields(() => user.value);
  user.addEventListener("input", f.update);
  const code = status?.setup_code_required
    ? h("input", { placeholder: "Setup code  XXXX-XXXX-XXXX", autocomplete: "one-time-code", required: true, spellcheck: false, "aria-label": "Setup code" })
    : null;
  const sup = h("input", { type: "checkbox", checked: true });
  const err = h("div", { class: "small err-text", role: "alert" });
  const btn = h("button", { class: "primary", type: "submit" }, "Create admin account");
  const form = h("form", { class: "wide", "aria-label": "First-run setup" },
    h("h2", null, "Create the first admin account"),
    h("p", { class: "lead" }, "No accounts exist on this Argus server yet. The account you create now is the administrator, and sign-in is required from then on."),
    status?.open_demo ? h("div", { class: "banner" }, "The server is in open demo mode (no sign-in). Creating this account turns sign-in on immediately.") : null,
    h("label", { class: "field" }, "Username", user),
    h("div", { class: "field" }, "Password", f.el),
    code ? h("label", { class: "field" }, "One-time setup code", code,
      h("span", { class: "small mute" }, "Printed in the server console at start-up and saved on the server in data/run/setup_code.txt. ",
        "It is not needed when you open the dashboard on the server itself (http://localhost).")) : null,
    h("label", { class: "check" }, sup, h("span", null, "Also give this account supervisor rights (live video, alert handling, evidence export).",
      h("span", { class: "small mute", style: { display: "block" } }, "Handy for a one-person demo. For real use, keep admin separate and create supervisor/operator accounts (separation of duties)."))),
    h("div", { class: "row" }, btn, onCancel ? h("button", { type: "button", onclick: () => { hideScreen(); onCancel(); } }, "Back to demo") : null),
    err);
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    err.textContent = "";
    const bad = f.problem();
    if (bad) { err.textContent = bad + "."; return; }
    btn.disabled = true;
    const r = await post("/api/setup", { username: user.value.trim(), password: f.pw.value, setup_code: code ? code.value : "",
      roles: sup.checked ? ["admin", "supervisor"] : ["admin"] });
    if (r.ok) { location.reload(); return; }
    err.textContent = r.status === 409 ? "Setup was already completed. Reloading…" : r.detail;
    if (r.status === 409) setTimeout(() => location.reload(), 1500);
    btn.disabled = false;
  });
  screen(form);
}

export function showForcedChangeScreen(me) {
  const cur = h("input", { type: "password", autocomplete: "current-password", required: true, placeholder: "Temporary password", "aria-label": "Temporary password" });
  const f = passwordFields(() => me.username);
  const err = h("div", { class: "small err-text", role: "alert" });
  const btn = h("button", { class: "primary", type: "submit" }, "Set password and continue");
  const form = h("form", { class: "wide", "aria-label": "Set a new password" },
    h("h2", null, "Set a new password"),
    h("p", { class: "lead" }, `Signed in as ${me.username}. An administrator gave you a temporary password; choose your own to continue.`),
    h("label", { class: "field" }, "Temporary password", cur),
    h("div", { class: "field" }, "New password", f.el),
    h("div", { class: "row" }, btn, h("button", { type: "button", class: "ghost", onclick: logout }, "Sign out")),
    err);
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    err.textContent = "";
    const bad = f.problem();
    if (bad) { err.textContent = bad + "."; return; }
    btn.disabled = true;
    const r = await post("/api/me/password", { current: cur.value, new: f.pw.value });
    if (r.ok) { location.reload(); return; }
    err.textContent = r.detail;
    btn.disabled = false;
  });
  screen(form);
}

// ---------------------------------------------------------------------------------------------------------------
// user-menu actions
export async function logout() {
  setFlash("logout");
  await post("/api/logout");
  location.reload();
}

export async function signOutEverywhere() {
  const ok = await confirmDialog("Sign out all sessions?",
    "Every browser signed in as you, including this one, is signed out. API tokens keep working (revoke them on the Account page).",
    { danger: true, okLabel: "Sign out everywhere" });
  if (!ok) return;
  const r = await post("/api/me/sessions/revoke");
  if (!r.ok) { toast(`Could not sign out sessions: ${r.detail}`, "error"); return; }
  setFlash("all");
  location.reload();
}

export function changePasswordDialog() {
  const cur = h("input", { type: "password", autocomplete: "current-password", placeholder: "Current password", "aria-label": "Current password" });
  const f = passwordFields(() => S.me.username);
  const err = h("div", { class: "small err-text", role: "alert" });
  openDialog({
    title: "Change password",
    body: h("div", { class: "stack" }, h("label", { class: "field" }, "Current password", cur), h("div", { class: "field" }, "New password", f.el),
      h("div", { class: "note" }, "Your other signed-in browsers are signed out; this one stays signed in."), err),
    actions: [{ label: "Cancel" }, { label: "Change password", primary: true, onclick: async () => {
      err.textContent = "";
      if (!cur.value) { err.textContent = "Enter your current password."; return false; }
      const bad = f.problem();
      if (bad) { err.textContent = bad + "."; return false; }
      const r = await post("/api/me/password", { current: cur.value, new: f.pw.value });
      if (!r.ok) { err.textContent = r.detail; return false; }
      toast("Password changed. Your other sessions were signed out.", "ok");
      return true;
    } }],
  });
  setTimeout(() => cur.focus(), 0);
}

// ---------------------------------------------------------------------------------------------------------------
// idle watcher: user input -> activity ping (slides the server's idle timer); warns a minute before expiry
export function startIdleWatch() {
  if (S.me?.auth !== "session" || !S.me.session?.idle_timeout_s) return;
  let deadline = 0, lastActivity = Date.now(), lastPing = Date.now(), busy = false, warn = null;
  const rebase = s => { if (s && s.idle_expires_at && s.now) deadline = Date.now() + (s.idle_expires_at - s.now) * 1000; };
  rebase(S.me.session);
  const mark = () => { lastActivity = Date.now(); };
  for (const ev of ["pointerdown", "keydown", "wheel", "touchstart"]) addEventListener(ev, mark, { passive: true, capture: true });
  let moved = 0;
  addEventListener("pointermove", () => { const n = Date.now(); if (n - moved > 5000) { moved = n; mark(); } }, { passive: true });
  async function ping() {
    if (busy || screenShown()) return;
    busy = true;
    try {
      const r = await api("/api/session/ping", { method: "POST", silent: true });
      lastPing = Date.now(); rebase(r.session);
      warn?.remove(); warn = null;
    } catch { /* 401 -> api() shows the sign-in screen with the reason */ }
    finally { busy = false; }
  }
  setInterval(async () => {
    if (screenShown()) return;
    const now = Date.now();
    if (lastActivity > lastPing && now - lastPing > 30000) { ping(); return; }
    const left = (deadline - now) / 1000;
    if (left <= 0) {                                  // the server decides; a 401 shows "Signed out due to inactivity"
      try { rebase((await api("/api/me", { silent: true })).session); } catch { /* handled */ }
      return;
    }
    if (left <= 60 && !warn?.isConnected) {
      warn = toast(h("span", null, "You will be signed out soon due to inactivity. ", h("b", null, "Click to stay signed in.")),
        "warn", { key: "idle-warn", timeout: 0, dedupeMs: 0, onclick: () => { mark(); ping(); } });
    }
  }, 5000);
}

// ---------------------------------------------------------------------------------------------------------------
// API tokens panel (own tokens; admins may pass {all: true})
export function createTokensPanel({ all = false } = {}) {
  const body = h("tbody");
  const msg = h("div");
  const el = h("div", { class: "panel" });
  const cols = all ? ["User", "Name", "Token", "Roles", "Created", "Last used", "Expires", ""] : ["Name", "Token", "Roles", "Created", "Last used", "Expires", ""];

  function example(tok) {
    const origin = location.origin;
    return [`curl -H "Authorization: Bearer ${tok}" ${origin}/api/alerts`,
      `curl -u "token:${tok}" ${origin}/api/alerts          # clients that only support Basic auth`].join("\n");
  }

  function showSecret(dlg, t) {
    const code = h("code", { class: "mono secret", tabIndex: 0 }, t.token);
    const copy = h("button", { type: "button", onclick: async () => {
      try { await navigator.clipboard.writeText(t.token); toast("Token copied.", "ok"); }
      catch { const sel = getSelection(); sel.removeAllRanges(); const rg = document.createRange(); rg.selectNodeContents(code); sel.addRange(rg); toast("Press Ctrl+C to copy.", "info"); }
    } }, icon("copy"), "Copy");
    replace(dlg.el.querySelector(".dlg-b"), h("div", { class: "stack" },
      h("div", { class: "banner" }, "Copy this token now. It is shown only once; the server stores only a SHA-256 hash of it."),
      h("div", { class: "row", style: { flexWrap: "nowrap", alignItems: "stretch" } }, code, copy),
      h("div", { class: "small mute" }, `Name: ${t.name} · roles: ${(t.roles || []).join(", ")} · expires: ${t.expires_at ? fmtDateTime(t.expires_at) : "never"}`),
      h("pre", { class: "mono small curl" }, example(t.token))));
    dlg.setActions([{ label: "Done", primary: true }]);
  }

  function newToken() {
    const name = h("input", { placeholder: "e.g. NVR webhook", maxLength: 64, "aria-label": "Token name" });
    const boxes = (S.me.roles || []).map(r => [r, h("input", { type: "checkbox", checked: true })]);
    const days = h("input", { type: "number", min: 1, max: 3650, placeholder: "never", "aria-label": "Expires after days" });
    const err = h("div", { class: "small err-text", role: "alert" });
    const dlg = openDialog({
      title: "New API token",
      body: h("div", { class: "stack" },
        h("label", { class: "field" }, "Name (what uses it)", name),
        h("div", { class: "field" }, "Roles (a subset of yours; the token never gets more than you currently hold)",
          h("div", { class: "row" }, boxes.map(([r, b]) => h("label", { class: "check" }, b, r)))),
        h("label", { class: "field" }, "Expires after (days, empty = never)", days),
        h("div", { class: "note" }, "Tokens authenticate scripts and integrations (NVR/VMS webhooks, exports) on /api/ and /media/. They cannot manage users, passwords or other tokens."),
        err),
      actions: [{ label: "Cancel" }, { label: "Create token", primary: true, onclick: async () => {
        err.textContent = "";
        const roles = boxes.filter(([, b]) => b.checked).map(([r]) => r);
        if (!name.value.trim()) { err.textContent = "Give the token a name."; return false; }
        if (!roles.length) { err.textContent = "Pick at least one role."; return false; }
        try {
          const t = await api("/api/tokens", { method: "POST", silent: true,
            body: { name: name.value.trim(), roles, expires_days: days.value ? +days.value : null } });
          showSecret(dlg, t);
          load();
        } catch (e) { err.textContent = e.detail || e.message; }
        return false;
      } }],
    });
    setTimeout(() => name.focus(), 0);
  }

  async function revoke(t) {
    const ok = await confirmDialog("Revoke token?", `"${t.name}"${all ? ` (${t.user})` : ""} stops working immediately. Anything using it will get 401 errors.`,
      { danger: true, okLabel: "Revoke" });
    if (!ok) return;
    try { await api(`/api/tokens/${enc(t.id)}`, { method: "DELETE" }); toast("Token revoked.", "ok"); }
    catch { /* toasted */ }
    load();
  }

  async function load() {
    let list;
    try { list = await optional("/api/tokens" + (all ? "?all=true" : ""), "tokens", { silent: true }); }
    catch (e) { replace(msg, h("div", { class: "banner err" }, `Tokens unavailable (${e.message}).`)); return; }
    if (list === null) { replace(el, featureMissing("API tokens")); return; }
    replace(msg);
    replace(body, list.length ? list.map(t => h("tr", null,
      all ? h("td", null, t.user) : null,
      h("td", null, t.name), h("td", { class: "mono small nowrap" }, t.prefix || t.id),
      h("td", null, (t.roles || []).map(r => h("span", { class: "role" }, r))),
      h("td", { class: "small nowrap" }, fmtDateTime(t.created_at)),
      h("td", { class: "small nowrap" }, t.last_used ? fmtAgo(t.last_used) : "never"),
      h("td", { class: "small nowrap" }, t.expires_at ? fmtDateTime(t.expires_at) : "never"),
      h("td", null, h("button", { class: "danger", type: "button", onclick: () => revoke(t) }, "Revoke"))))
      : h("tr", null, h("td", { colspan: cols.length, class: "mute" }, all ? "No API tokens on this server." : "You have no API tokens.")));
  }

  replace(el,
    h("div", { class: "row" }, h("h3", { style: { margin: 0 } }, all ? "All API tokens" : "API tokens"), h("span", { class: "grow" }),
      all ? null : h("button", { class: "primary", type: "button", onclick: newToken }, icon("plus"), "New token")),
    msg,
    h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null, cols.map(c => h("th", null, c)))), body)),
    all ? null : h("div", { class: "note" }, "Scripts can also use HTTP Basic auth with your username and password, e.g. ",
      h("code", null, `curl -u ${S.me.username} ${location.origin}/api/alerts`), ". Prefer a token: it is revocable and can carry fewer roles."));
  return { el, load };
}

// ---------------------------------------------------------------------------------------------------------------
// Account page (#/account)
export function createAccountPage() {
  const el = h("section", { class: "page" });
  let tokens = null;

  async function render() {
    const head = h("div", { class: "page-head" }, h("h2", null, "Account"));
    if (S.me.open_mode) {
      replace(el, head, h("div", { class: "banner" }, "Open demo mode: there are no accounts and no sign-in. ",
        S.auth?.setup_required ? h("button", { type: "button", onclick: () => showSetupScreen(S.auth, { onCancel: () => {} }) }, "Create the first admin account") : null));
      return;
    }
    let me = S.me;
    try { me = await api("/api/me", { silent: true }); } catch { /* keep cached */ }
    const s = me.session || {};
    const via = { session: "dashboard session", basic: "HTTP Basic", token: "API token" }[me.auth] || me.auth;
    tokens = me.auth === "token" ? null : createTokensPanel();
    replace(el, head,
      h("div", { class: "panel" },
        h("h3", { style: { marginTop: 0 } }, "Profile"),
        kv([["Username", me.username], ["Roles", h("span", { class: "row", style: { gap: "4px" } }, (me.roles || []).map(r => h("span", { class: "role" }, r)))],
          ["Signed in via", via],
          me.last_login ? ["Last sign-in", `${fmtDateTime(me.last_login)} (${fmtAgo(me.last_login)})`] : null,
          s.expires_at ? ["Session ends", `${fmtDateTime(s.expires_at)} at the latest`] : null,
          s.idle_timeout_s ? ["Idle timeout", `signed out after ${s.idle_timeout_s % 60 ? fmtDur(s.idle_timeout_s) : `${s.idle_timeout_s / 60} min`} without activity`] : null]),
        h("div", { class: "row", style: { marginTop: "12px" } },
          h("button", { type: "button", onclick: changePasswordDialog }, icon("lock"), "Change password"),
          h("button", { type: "button", class: "danger", onclick: signOutEverywhere }, "Sign out all sessions"),
          h("button", { type: "button", onclick: logout }, icon("logout"), "Sign out"))),
      tokens ? tokens.el : null);
    tokens?.load();
  }
  return { el, show() { render(); } };
}
