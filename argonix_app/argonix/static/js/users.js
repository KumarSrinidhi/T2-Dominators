// Settings -> Users (permission user.admin): list, create, edit roles, disable/enable, reset password, sign out,
// delete; role reference table; every API token on the server. The server enforces every rule (self-protection,
// last active admin, password policy); this page only surfaces its messages.
import { api, optional } from "./api.js";
import { S, can } from "./state.js";
import { toast, openDialog, confirmDialog, featureMissing } from "./ui.js";
import { h, replace, icon, enc, fmtDateTime, fmtAgo } from "./util.js";
import { createTokensPanel, randomPassword, showSetupScreen } from "./account.js";

const ROLE_HELP = {
  viewer: "Watch live video, alerts and evidence.",
  operator: "Viewer + acknowledge, escalate, resolve and dismiss alerts.",
  supervisor: "Operator + export/hold evidence, approve calibration, assign/merge incidents.",
  admin: "Cameras, zones, rules, settings and users. Cannot handle alerts (separation of duties).",
  auditor: "Read the audit log and alerts; cannot change anything.",
  integration: "Machine account for access-control panels: may only post sensor events (API token).",
};
const roleNames = () => Object.keys(S.auth?.roles || ROLE_HELP);

function roleBoxes(selected) {
  const boxes = roleNames().map(r => [r, h("input", { type: "checkbox", checked: selected.includes(r) })]);
  return {
    boxes,
    el: h("div", { class: "stack", style: { gap: "4px" } }, boxes.map(([r, b]) =>
      h("label", { class: "check" }, b, h("span", null, h("b", null, r), h("span", { class: "small mute" }, ` · ${ROLE_HELP[r] || ""}`))))),
    value: () => boxes.filter(([, b]) => b.checked).map(([r]) => r),
  };
}

function tempPasswordField() {
  const inp = h("input", { value: randomPassword(), autocomplete: "off", spellcheck: false, class: "mono", "aria-label": "Temporary password",
    style: { width: "100%", minWidth: "0" } });
  const gen = h("button", { type: "button", title: "Generate another", onclick: () => { inp.value = randomPassword(); } }, icon("refresh"), "Generate");
  const copy = h("button", { type: "button", title: "Copy", onclick: async () => {
    try { await navigator.clipboard.writeText(inp.value); toast("Copied.", "ok"); } catch { inp.select(); }
  } }, icon("copy"));
  return { inp, el: h("div", { class: "row", style: { flexWrap: "nowrap" } }, h("span", { class: "grow", style: { display: "flex" } }, inp), gen, copy) };
}

export default function createUsers() {
  const el = h("section", { class: "page" });
  let users = [], tokens = null;

  const head = () => h("div", { class: "page-head" }, h("h2", null, "Users"), h("span", { class: "tag" }, "Settings"),
    h("span", { class: "mute small" }, users.length ? `${users.length} account${users.length === 1 ? "" : "s"}` : ""), h("span", { class: "grow" }),
    h("button", { class: "primary", type: "button", onclick: addUser }, icon("plus"), "Add user"),
    h("button", { type: "button", onclick: () => load() }, icon("refresh"), "Refresh"));

  function status(u) {
    if (u.disabled) return h("span", { class: "h-DOWN" }, "disabled");
    if (u.must_change) return h("span", { class: "h-DEGRADED" }, "must change password");
    return h("span", { class: "h-OK" }, "active");
  }

  function render() {
    const me = S.me.username;
    const rows = users.map(u => h("tr", null,
      h("td", null, h("b", null, u.username), u.username === me ? h("span", { class: "tag", style: { marginLeft: "6px" } }, "you") : null),
      h("td", null, h("span", { class: "row", style: { gap: "4px" } }, u.roles.map(r => h("span", { class: "role" }, r)))),
      h("td", null, status(u)),
      h("td", { class: "small nowrap", title: u.last_login ? fmtDateTime(u.last_login) : "" }, u.last_login ? fmtAgo(u.last_login) : "never"),
      h("td", { class: "small nowrap hide-sm" }, u.created_at ? fmtDateTime(u.created_at) : "—"),
      h("td", { class: "small" }, u.tokens || 0),
      h("td", null, h("div", { class: "row", style: { gap: "4px", minWidth: "220px" } },
        h("button", { type: "button", onclick: () => editRoles(u), title: "Edit roles" }, icon("edit"), "Roles"),
        h("button", { type: "button", onclick: () => resetPassword(u), disabled: u.username === me, title: u.username === me ? "Use Change password in your user menu" : "Set a temporary password" }, icon("key"), "Reset"),
        h("button", { type: "button", onclick: () => setDisabled(u, !u.disabled), disabled: u.username === me }, u.disabled ? "Enable" : "Disable"),
        h("button", { type: "button", onclick: () => signOut(u), title: "Sign this user out of every browser" }, "Sign out"),
        h("button", { type: "button", class: "icon danger", title: "Delete", "aria-label": `Delete ${u.username}`, disabled: u.username === me, onclick: () => del(u) }, icon("trash"))))));
    tokens = createTokensPanel({ all: true });
    replace(el, head(),
      h("div", { class: "table-wrap panel", style: { padding: 0 } }, h("table", null,
        h("thead", null, h("tr", null, ["User", "Roles", "Status", "Last sign-in", "Created", "Tokens", ""].map((t, i) => h("th", { class: i === 4 ? "hide-sm" : "" }, t)))),
        h("tbody", null, rows.length ? rows : h("tr", null, h("td", { colspan: 7, class: "mute" }, "No users."))))),
      h("div", { class: "note" }, "New users and password resets get a temporary password and must choose their own at first sign-in. ",
        "Disabling, resetting or \"Sign out\" ends that user's sessions immediately. The last active admin cannot be removed, and nobody can disable or delete themselves. Every change is written to the audit log (passwords never are)."),
      h("h3", null, "Roles"),
      h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null, ["Role", "What it is for", "Permissions"].map(t => h("th", null, t)))),
        h("tbody", null, roleNames().map(r => h("tr", null, h("td", null, h("b", null, r)), h("td", null, ROLE_HELP[r] || ""),
          h("td", { class: "small mono" }, (S.auth?.roles?.[r] || []).join(", "))))))),
      h("div", { style: { marginTop: "18px" } }, tokens.el));
    tokens.load();
  }

  async function load() {
    const top = h("div", { class: "page-head" }, h("h2", null, "Users"));
    if (S.me.open_mode) {
      replace(el, top, h("div", { class: "banner" }, "Open demo mode: there are no accounts yet. ",
        S.auth?.setup_required ? h("button", { type: "button", onclick: () => showSetupScreen(S.auth, { onCancel: () => {} }) }, "Create the first admin account") : null));
      return;
    }
    if (!can("user.admin")) { replace(el, top, h("div", { class: "banner" }, "Managing users needs the user.admin permission (admin role).")); return; }
    let list;
    try { list = await optional("/api/users", "users", { silent: true }); }
    catch (e) { replace(el, top, h("div", { class: "banner err" }, `Users unavailable (${e.message}).`)); return; }
    if (list === null) { replace(el, top, featureMissing("User management")); return; }
    users = list;
    render();
  }

  // ---- actions ----
  async function call(path, method, body, okMsg) {
    try {
      const r = await api(path, { method, body, silent: true });
      if (okMsg) toast(okMsg, "ok");
      await load();
      return r;
    } catch (e) {
      toast(e.detail || e.message, "error");
      return null;
    }
  }

  function addUser() {
    const name = h("input", { placeholder: "username", maxLength: 48, autocomplete: "off", spellcheck: false, "aria-label": "Username" });
    const roles = roleBoxes(["operator"]);
    const pw = tempPasswordField();
    const must = h("input", { type: "checkbox", checked: true });
    const err = h("div", { class: "small err-text", role: "alert" });
    openDialog({
      title: "Add user",
      body: h("div", { class: "stack" },
        h("label", { class: "field" }, "Username (letters, digits, . _ @ -)", name),
        h("div", { class: "field" }, "Roles", roles.el),
        h("div", { class: "field" }, "Temporary password (give it to the user through a separate channel)", pw.el),
        h("label", { class: "check" }, must, "Require a new password at first sign-in"),
        err),
      actions: [{ label: "Cancel" }, { label: "Create user", primary: true, onclick: async () => {
        err.textContent = "";
        if (!roles.value().length) { err.textContent = "Pick at least one role."; return false; }
        try {
          await api("/api/users", { method: "POST", silent: true,
            body: { username: name.value.trim(), roles: roles.value(), password: pw.inp.value, must_change: must.checked } });
        } catch (e) { err.textContent = e.detail || e.message; return false; }
        toast(`User ${name.value.trim()} created.`, "ok");
        load();
        return true;
      } }],
    });
    setTimeout(() => name.focus(), 0);
  }

  function editRoles(u) {
    const roles = roleBoxes(u.roles);
    const err = h("div", { class: "small err-text", role: "alert" });
    openDialog({
      title: `Roles for ${u.username}`,
      body: h("div", { class: "stack" }, roles.el,
        h("div", { class: "note" }, "Takes effect on the user's next request; their API tokens are narrowed to the roles they still hold."), err),
      actions: [{ label: "Cancel" }, { label: "Save", primary: true, onclick: async () => {
        err.textContent = "";
        if (!roles.value().length) { err.textContent = "Pick at least one role."; return false; }
        try { await api(`/api/users/${enc(u.username)}`, { method: "PUT", silent: true, body: { roles: roles.value() } }); }
        catch (e) { err.textContent = e.detail || e.message; return false; }
        toast("Roles updated.", "ok");
        load();
        return true;
      } }],
    });
  }

  function resetPassword(u) {
    const pw = tempPasswordField();
    const must = h("input", { type: "checkbox", checked: true });
    const err = h("div", { class: "small err-text", role: "alert" });
    openDialog({
      title: `Reset password for ${u.username}`,
      body: h("div", { class: "stack" },
        h("div", { class: "field" }, "Temporary password", pw.el),
        h("label", { class: "check" }, must, "Require a new password at next sign-in"),
        h("div", { class: "note" }, `${u.username} is signed out of every browser. Their API tokens keep working (revoke them below if needed).`),
        err),
      actions: [{ label: "Cancel" }, { label: "Reset password", primary: true, onclick: async () => {
        err.textContent = "";
        try { await api(`/api/users/${enc(u.username)}/reset`, { method: "POST", silent: true, body: { password: pw.inp.value, must_change: must.checked } }); }
        catch (e) { err.textContent = e.detail || e.message; return false; }
        toast(`Password for ${u.username} reset. Hand over the temporary password securely.`, "ok");
        load();
        return true;
      } }],
    });
  }

  async function setDisabled(u, disabled) {
    if (disabled && !await confirmDialog(`Disable ${u.username}?`,
      "They are signed out immediately, cannot sign in, and their API tokens stop working until re-enabled.", { danger: true, okLabel: "Disable" })) return;
    call(`/api/users/${enc(u.username)}`, "PUT", { disabled }, disabled ? `${u.username} disabled.` : `${u.username} enabled.`);
  }

  async function signOut(u) {
    if (!await confirmDialog(`Sign out ${u.username}?`, "Ends every dashboard session of this user. They can sign in again.", { okLabel: "Sign out" })) return;
    call(`/api/users/${enc(u.username)}`, "PUT", { revoke_sessions: true }, `${u.username} signed out everywhere.`);
  }

  async function del(u) {
    if (!await confirmDialog(`Delete ${u.username}?`, "The account and its API tokens are removed permanently. Audit log entries are kept.",
      { danger: true, okLabel: "Delete user" })) return;
    call(`/api/users/${enc(u.username)}`, "DELETE", undefined, `${u.username} deleted.`);
  }

  return { el, show() { load(); } };
}
