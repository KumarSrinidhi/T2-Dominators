// System page cards: "Run demo scenario" (argonix/scenario.py) and phone-push status + test (argonix/notify.py).
import { api, optional, safe } from "./api.js";
import { can } from "./state.js";
import { toast, progress, featureMissing, kv } from "./ui.js";
import { h, icon, replace, num, fmtDateTime } from "./util.js";

const STEP_STATE = { pending: ["mute", "pending"], done: ["h-OK", "done"], failed: ["h-error", "failed"], skipped: ["mute", "skipped"] };

export function createDemoCard() {
  const el = h("div");
  let data = null, missing = false, timer = null, pick = null;

  async function refresh() {
    if (!can("settings.edit")) { replace(el, h("div", { class: "mute small" }, "Needs the settings.edit permission.")); return; }
    if (missing) return;
    try {
      data = await optional("/api/demo/scenarios", "demo", { silent: true });
      if (data === null) { missing = true; replace(el, featureMissing("Demo scenarios")); return; }
    } catch { return; }
    render();
    clearTimeout(timer);
    if (data.running) timer = setTimeout(refresh, 700);   // follow progress while a scenario plays
  }

  const run = safe(async () => {
    const name = pick?.value;
    if (!name) return;
    await api(`/api/demo/scenarios/${encodeURIComponent(name)}/run`, { method: "POST", feature: "demo" });
    toast("Demo scenario started. Its alerts are labelled DEMO.", "ok");
    refresh();
  });
  const stop = safe(async () => {
    await api("/api/demo/stop", { method: "POST", feature: "demo" });
    toast("Demo scenario stopped; any injected camera fault was cleared.", "info");
    refresh();
  });

  function render() {
    const valid = (data.scenarios || []).filter(s => !s.error);
    const r = data.run;
    const prev = pick?.value;
    pick = h("select", { "aria-label": "Scenario", disabled: data.running || !valid.length },
      valid.map(s => h("option", { value: s.name, selected: s.name === (prev || r?.name) }, s.title || s.name)));
    const cur = valid.find(s => s.name === pick.value) || valid[0];
    pick.addEventListener("change", render);
    replace(el,
      !data.enabled ? h("div", { class: "banner" }, icon("info"), h("div", null, data.hint || "Demo scenarios are off.")) : null,
      h("div", { class: "row", style: { gap: "8px" } }, pick,
        data.running ? h("button", { class: "danger", onclick: stop }, icon("stop"), "Stop")
          : h("button", { class: "primary", disabled: !data.enabled || !valid.length, onclick: run }, icon("play"), "Run demo scenario")),
      cur ? h("div", { class: "small mute", style: { margin: "6px 0" } }, cur.description) : h("div", { class: "mute small" }, "No scenario files in configs/scenarios/."),
      r ? h("div", { class: "stack", style: { marginTop: "8px" } },
        h("div", { class: "row small", style: { gap: "8px" } }, h("span", { class: "tag demo" }, "DEMO"), h("b", null, r.title),
          h("span", { class: "grow" }), h("span", { class: "tnum mute" }, `${num(r.elapsed_s, 0)} / ${num(r.duration_s, 0)} s · ${r.state}`)),
        progress(r.progress),
        h("div", { class: "table-wrap" }, h("table", null, h("tbody", null, r.steps.map(s => h("tr", null,
          h("td", { class: "tnum nowrap" }, `t+${num(s.at, 0)} s`), h("td", null, s.label, s.detail ? h("div", { class: "small mute" }, s.detail) : null),
          h("td", { class: STEP_STATE[s.state]?.[0] || "" }, STEP_STATE[s.state]?.[1] || s.state)))))),
        r.alerts?.length ? h("div", { class: "small mute" }, `Alerts: ${r.alerts.join(", ")} (see Review)`) : null) : null,
      (data.scenarios || []).filter(s => s.error).map(s => h("div", { class: "small h-error" }, `${s.file}: ${s.error}`)),
      h("div", { class: "note" }, data.note || ""));
  }

  return { el, refresh, destroy() { clearTimeout(timer); } };
}

export function createNotifyCard() {
  const el = h("div");
  let missing = false;

  const test = safe(async () => {
    const r = await api("/api/notify/test", { method: "POST", body: {}, feature: "notify" });
    for (const x of r.results) toast(`${x.channel}: ${x.ok ? "test sent" : "failed"} (${x.detail})`, x.ok ? "ok" : "warn", { timeout: 8000 });
    refresh();
  });

  async function refresh() {
    if (!can("settings.edit")) { replace(el, h("div", { class: "mute small" }, "Needs the settings.edit permission.")); return; }
    if (missing) return;
    let st;
    try {
      st = await optional("/api/notify", "notify", { silent: true });
      if (st === null) { missing = true; replace(el, featureMissing("Phone push")); return; }
    } catch { return; }
    const anyOn = st.channels.some(c => c.enabled);
    replace(el,
      h("div", { class: "table-wrap" }, h("table", null,
        h("thead", null, h("tr", null, ["Channel", "State", "Min. priority", "Sent / failed", "Last"].map(t => h("th", null, t)))),
        h("tbody", null, st.channels.map(c => h("tr", null,
          h("td", null, h("b", null, c.kind === "telegram" ? "Telegram" : "HTTP (ntfy-style)"),
            h("div", { class: "small mute" }, c.kind === "telegram" ? `chat ${c.chat_id || "—"} · token ${c.token_env || "—"} ${c.token_set ? "(set)" : "(not set)"}` : `host ${c.host || "—"}`)),
          h("td", { class: !c.enabled ? "mute" : c.ready ? "h-OK" : "h-error" }, !c.enabled ? "off" : c.ready ? "ready" : "misconfigured",
            c.enabled && c.problem ? h("div", { class: "small" }, c.problem) : null),
          h("td", null, c.min_priority), h("td", { class: "tnum" }, `${c.sent} / ${c.failed}`),
          h("td", { class: "small" }, c.last ? [c.last.ok ? "ok" : "failed", h("div", { class: "mute" }, `${fmtDateTime(c.last.t)} · ${c.last.detail}`)] : "—")))))),
      h("div", { class: "row", style: { marginTop: "8px" } },
        h("button", { disabled: !anyOn, onclick: test, title: anyOn ? "Send a test message to every enabled channel" : "Enable a channel in the site config (\"notify\")" }, icon("bell"), "Send test notification")),
      !anyOn ? h("div", { class: "note" }, "Off. Set \"notify\": {\"telegram\": {\"enabled\": true, \"token_env\": \"ARGONIX_TG_TOKEN\", \"chat_id\": \"…\"}} in the site config and export the bot token in that environment variable.") : null,
      kv([["Console link", st.console_url || "not set (notify.console_url)"], ["Demo incidents", st.include_demo ? "pushed, labelled [DEMO]" : "not pushed"]]),
      h("div", { class: "note" }, st.note));
  }

  return { el, refresh };
}
