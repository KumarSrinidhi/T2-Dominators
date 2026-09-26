// System: process stats, per-camera pipeline table, rule precision, baselines, VLM status, audit log (auditor).
import { api } from "./api.js";
import { S, can, camName } from "./state.js";
import * as drawer from "./drawer.js";
import { h, icon, replace, fmtDur, fmtDateTime, fmtAgo, fmtMs, RULE, howLabel, num } from "./util.js";
import { kv, pageHeader, estBadge, acsBadge, skeleton } from "./ui.js";
import { gauge } from "./charts.js";
import { createHandoffCard } from "./handoff.js";
import { createNodesCard } from "./nodes.js";
import { createDemoCard, createNotifyCard } from "./demo.js";

export default function createSystem() {
  const statsEl = h("div", { class: "sys-kpis" }, skeleton("kpi", 4));
  const camBody = h("tbody");
  const precBody = h("tbody");
  const baseEl = h("div");
  const vlmEl = h("div");
  const auditEl = h("div");
  const modelEl = h("div");
  const retEl = h("div", { class: "mute small" }, "Loading…");
  const errEl = h("div");
  const handoff = createHandoffCard();
  const nodes = createNodesCard();
  const sensorsEl = h("div");
  const demo = createDemoCard();
  const notify = createNotifyCard();
  const card = (ic, title, sub, ...body) => h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h3", null, icon(ic), title), sub ? h("span", { class: "sub" }, sub) : null), body);
  const el = h("section", { class: "page" },
    pageHeader({ title: "System", subtitle: "Pipelines, feedback loop, baselines and models · measured on this deployment" }),
    errEl, statsEl,
    card("cameras", "Camera pipelines", null,
      h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null,
        ["Camera", "Worker", "View health", "Analytics fps", "Detect", "Frames", "Dropped", "Masked", "Events", "Tracks", "Calib. error (est.)"].map(t => h("th", null, t)))), camBody)),
      h("div", { class: "note" }, "Dropped = frames skipped to stay real-time when detection falls behind. Measured on this machine; not a capacity claim.")),
    sensorsEl,
    card("nodes", "Compute nodes", "distributed inference", nodes.el),
    h("div", { class: "grid-cards", style: { marginTop: "var(--s-4)" } },
      card("play", "Demo scenario", "scripted · labelled DEMO", demo.el),
      card("bell", "Phone push (Telegram / ntfy)", "urgent incidents only", notify.el),
      card("check", "Rule precision from operator feedback", "feedback loop",
        h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null,
          ["Rule", "Alerts", "Labelled", "TP / FP", "Precision", "95% CI (Wilson)"].map(t => h("th", null, t)))), precBody)),
        h("div", { class: "note" }, "Computed only from operator dispositions (duplicates and drills excluded). Fewer than 50 labels per rule is not enough to claim a tuned precision.")),
      card("journeys", "Next-camera prediction (handoff)", "estimate", handoff.el),
      card("analytics", "Occupancy baselines (hour-of-week)", "corroboration only", baseEl),
      card("spark", "Explainer (VLM)", "assistive, on request", vlmEl),
      card("cpu", "Model versions", null, modelEl),
      card("lock", "Retention & legal hold", "evidence lifecycle", retEl)),
    auditEl);

  let timer = null, slowAt = 0, shown = false;

  async function tick() {
    try {
      const s = await api("/api/stats", { silent: true });
      replace(errEl);
      const camsN = Object.keys(s.cameras || {}).length;
      const kpi = (g, k, v, title) => h("div", { class: "sys-kpi", title: title || null }, g, h("div", { style: { minWidth: 0 } }, h("div", { class: "kl" }, k), h("div", { class: "kv2" }, v)));
      const ic = n => h("span", { class: "ki" }, icon(n));
      replace(statsEl,
        kpi(ic("clock"), "Uptime", fmtDur(s.uptime_s)),
        kpi(gauge(s.cpu_percent / 100, { size: 46, stroke: 5, label: "CPU" }), "CPU (host)", `${num(s.cpu_percent, 0)} %`),
        kpi(gauge(s.ram_percent / 100, { size: 46, stroke: 5, label: "RAM" }), "Memory (host)", `${num(s.ram_percent, 0)} %`),
        kpi(ic("cpu"), "Process RSS", `${s.process_rss_mb} MB`),
        kpi(ic("cameras"), "Cameras", camsN),
        kpi(ic("signal"), "Live link", S.ws, "WebSocket state"));
      replace(camBody, Object.entries(s.cameras || {}).map(([id, c]) => {
        const cal = S.cfg.cameras.find(x => x.id === id)?.calibration;
        return h("tr", null, h("td", null, h("b", null, camName(id)), h("div", { class: "small mute mono" }, id)),
          h("td", { class: c.status === "error" ? "h-error" : "" }, c.status, c.error ? h("div", { class: "small" }, c.error) : null),
          h("td", { class: "h-" + (c.health || "WARMUP") }, c.health || "WARMUP", (c.health_reasons || []).length ? h("div", { class: "small" }, c.health_reasons.join(", ")) : null),
          h("td", { class: "tnum" }, num(c.fps, 2)), h("td", { class: "tnum" }, fmtMs(c.detect_ms)), h("td", { class: "tnum" }, c.frames ?? "—"), h("td", { class: "tnum" }, c.dropped ?? 0),
          h("td", { class: "tnum" }, c.masked ?? 0), h("td", { class: "tnum" }, c.events ?? 0), h("td", { class: "tnum" }, c.tracks ?? 0),
          h("td", null, cal ? estBadge(`± ${cal.calib_error_m} m`) : "—", cal ? h("div", { class: "small mute" }, `RMS ${cal.measured_rms_m} m / ${cal.check_points} pt`) : null));
      }));
      const m = s.model || {};
      replace(modelEl, kv(Object.entries(m).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)])));
    } catch (e) {
      replace(errEl, h("div", { class: "banner err" }, `Stats unavailable (${e.message}).`));
    }
    nodes.refresh();
    sensors();
    if (Date.now() - slowAt > 15000) { slowAt = Date.now(); slow(); }
  }

  // Access-control sources (door controllers / panels): hidden when none is configured (API answers 404).
  const SENS_STATE = { ok: ["OK", "online"], offline: ["DOWN", "offline (silent past heartbeat)"], waiting: ["WARMUP", "waiting for first message"] };
  const openAlert = id => { drawer.setNavigator(null); drawer.openAlert(id); };
  async function sensors() {
    let d;
    try { d = await api("/api/sensors", { silent: true }); } catch { replace(sensorsEl); return; }
    replace(sensorsEl, card("door", "Access-control sources", "non-video · door controllers and panels",
      (d.warnings || []).length ? h("div", { class: "banner", style: { margin: "0 0 8px" } }, h("b", null, "Configuration: "), d.warnings.join(" · ")) : null,
      h("div", { class: "table-wrap" }, h("table", null, h("thead", null, h("tr", null,
        ["Source", "Status", "Last message", "Last event", "Reports", "Alerts", "Rejected", "Signing", "Doors → zone"].map(t => h("th", null, t)))),
        h("tbody", null, d.sources.map(src => {
          const [cls, label] = SENS_STATE[src.state] || ["WARMUP", src.state];
          const le = src.last_event;
          return h("tr", null,
            h("td", null, acsBadge(src.name), h("div", { class: "small mute mono" }, `${src.id} · ${src.kind}`)),
            h("td", { class: "h-" + cls }, label, src.heartbeat_s ? h("div", { class: "small" }, `heartbeat ${src.heartbeat_s} s`) : h("div", { class: "small mute" }, "no heartbeat check")),
            h("td", { class: "nowrap", title: src.last_seen ? fmtDateTime(src.last_seen) : "" }, src.last_seen ? fmtAgo(src.last_seen) : "never"),
            h("td", null, le ? [h("code", null, le.type), le.door ? ` · ${le.door}` : "", h("div", { class: "small mute" }, fmtDateTime(le.t)),
              le.alert_id ? h("a", { href: "#", onclick: e => { e.preventDefault(); openAlert(le.alert_id); } }, `alert ${le.alert_id}`) : null] : "—"),
            h("td", null, h("div", { class: "sens-counts" }, Object.entries(src.counts || {}).map(([k, v]) => h("span", { class: "tag" }, `${k.replace(/_/g, " ")} ${v}`)))),
            h("td", { class: "tnum" }, src.alerts),
            h("td", { class: "tnum" + (src.rejected ? " h-DEGRADED" : ""), title: src.last_rejected ? `last: ${src.last_rejected.reason} (${fmtDateTime(src.last_rejected.t)})` : "" }, src.rejected),
            h("td", { class: "small" }, src.hmac),
            h("td", { class: "small" }, (src.doors || []).map(x => h("div", null, h("b", null, x.id), x.name ? ` ${x.name}` : "", " → ", x.site_zone || "—", x.camera ? h("span", { class: "mute" }, ` (${camName(x.camera)})`) : null))));
        })))),
      h("div", { class: "note" }, `${d.note} A report that shares a site zone with a video alert is grouped into the same alert and incident ("sensor + video" corroboration). Rejected = failed signature, replayed id, clock skew or invalid body.`)));
  }

  async function slow() {
    handoff.refresh();
    demo.refresh();
    notify.refresh();
    try {
      const f = await api("/api/feedback/stats", { silent: true });
      replace(precBody, f.rules.length ? f.rules.map(r => h("tr", null, h("td", null, RULE[r.rule] || r.rule), h("td", null, r.alerts), h("td", null, r.labelled),
        h("td", null, `${r.tp} / ${r.fp}`), h("td", null, r.precision ?? "—"),
        h("td", null, r.labelled ? (r.precision_ci95 || []).join(" – ") : "—", r.labelled && !r.enough_data ? h("span", { class: "mute" }, " (n<50)") : null)))
        : h("tr", null, h("td", { colspan: 6, class: "mute" }, "No alerts yet.")));
    } catch { /* keep previous */ }
    try {
      const bl = await api("/api/baseline", { silent: true });
      replace(baseEl, bl.buckets.length ? h("div", { class: "table-wrap" }, h("table", null,
        h("thead", null, h("tr", null, ["Camera", "Zone", "Hour of week", "Samples", "Mean occupancy", "State"].map(t => h("th", null, t)))),
        h("tbody", null, bl.buckets.slice(0, 200).map(b => h("tr", null, h("td", null, camName(b.camera)), h("td", null, b.zone),
          h("td", null, `${howLabel(b.hour_of_week)} (${b.hour_of_week})`), h("td", null, b.samples), h("td", null, b.mean),
          h("td", { class: b.ready ? "h-OK" : "mute" }, b.ready ? "active" : `learning (${b.samples}/${bl.min_samples})`)))))) : h("div", { class: "mute small" }, "No samples yet."),
        h("div", { class: "note" }, `Deviation fires when the modified z-score reaches ${bl.z_threshold} over ${bl.bucket_s} s buckets, and stays silent until ${bl.min_samples} samples exist for that hour of the week. It is corroboration only, never a primary alarm.`));
    } catch { /* keep previous */ }
    try {
      const v = await api("/api/vlm", { silent: true });
      replace(vlmEl, !v.enabled ? h("div", { class: "mute small" }, "Not configured (start with --vlm-url to enable the on-request \"Explain\" button).")
        : [v.models?.length ? h("div", { class: "table-wrap" }, h("table", null,
            h("thead", null, h("tr", null, ["Model", "Params", "Mode", "Available", "Loaded"].map(t => h("th", null, t)))),
            h("tbody", null, v.models.map(m => h("tr", null, h("td", { class: "mono" }, m.id), h("td", null, m.params_b != null ? `${m.params_b} B` : "—"),
              h("td", null, m.mode || "—"), h("td", null, m.available === false ? "no" : "yes"), h("td", null, m.loaded ? "yes" : "no")))))) : h("div", { class: "mute small" }, "No model list reported."),
          v.health ? h("pre", { class: "mono mute", style: { whiteSpace: "pre-wrap" } }, JSON.stringify(v.health, null, 2)) : null,
          h("div", { class: "note" }, "Assistive descriptions only, generated on operator request. Never used to trigger, score or close alerts.")]);
    } catch { /* keep previous */ }
    try {
      const r = await api("/api/retention", { silent: true });
      const p = r.policy || {}, l = r.last_sweep;
      const lim = (v, unit) => v ? `${v} ${unit}` : "no limit";
      const mb = b => b == null ? "—" : `${num(b / 1e6, 1)} MB`;
      replace(retEl, kv([
        ["Recordings kept", lim(p.recordings_days, "days")], ["Evidence kept", lim(p.evidence_days, "days")],
        ["Disk cap (recordings + evidence)", lim(p.max_disk_gb, "GB")],
        ["Closed alerts kept", lim(p.alerts_days, "days")], ["Sweep", r.running ? `every ${p.interval_min} min` : "not running (offline run or disabled)"],
        ["Alerts on legal hold", String(r.held_alerts ?? 0)],
        ["Last sweep", !l ? "none yet" : l.error ? `failed: ${l.error}` : [fmtAgo(l.t), h("div", { class: "small mute" },
          `deleted ${l.recordings ?? 0} recording(s), ${l.evidence ?? 0} evidence file(s), ${l.alerts ?? 0} closed alert(s), ${mb(l.bytes)} freed; `,
          `${l.kept_on_hold ?? 0} file(s) kept for legal hold; now ${mb(l.disk_bytes)} on disk`)]]]),
        h("div", { class: "note" }, "Oldest files go first when over the disk cap. Evidence of alerts on legal hold is never deleted; the audit log is never deleted. Every sweep is audited (retention_swept)."));
    } catch { replace(retEl, h("div", { class: "mute small" }, "Retention status unavailable.")); }
    if (can("audit.read")) {
      try {
        const [au, chain] = await Promise.all([api("/api/audit?limit=200", { silent: true }), api("/api/audit/verify", { silent: true }).catch(() => null)]);
        const chainTag = !chain ? null : chain.ok
          ? h("span", { class: "tag ok", title: `SHA-256 hash chain recomputed over ${chain.rows} row(s); head ${String(chain.head || "").slice(0, 16)}…` }, icon("check"), "chain intact ✓")
          : h("span", { class: "tag err", title: "A row was edited, inserted or deleted at or before this entry" }, icon("alert"), `chain broken at #${chain.first_bad_seq}`);
        replace(auditEl, h("div", { class: "panel", style: { marginTop: "var(--s-4)" } },
          h("div", { class: "panel-h" }, h("h3", null, icon("review"), "Audit log"), h("span", { class: "tag" }, "auditor"), chainTag, h("span", { class: "sub" }, "every view, export and status change · tamper-evident")),
          h("div", { class: "table-wrap", style: { maxHeight: "360px", overflow: "auto" } }, h("table", null,
            h("thead", null, h("tr", null, ["#", "Time", "Actor", "Action", "Alert", "Details"].map(t => h("th", null, t)))),
            h("tbody", null, au.length ? au.map(x => h("tr", null, h("td", { class: "tnum small mute" }, x.seq ?? ""), h("td", { class: "nowrap small" }, fmtDateTime(x.t)), h("td", null, x.actor), h("td", null, x.action),
              h("td", null, x.alert_id ? h("a", { href: "#", onclick: e => { e.preventDefault(); drawer.setNavigator(null); drawer.openAlert(x.alert_id); } }, x.alert_id) : ""),
              h("td", { class: "small mute" }, x.details && Object.keys(x.details).length ? JSON.stringify(x.details) : ""))) : h("tr", null, h("td", { colspan: 6, class: "mute" }, "Empty.")))))));
      } catch { /* keep previous */ }
    } else replace(auditEl);
  }

  return {
    el,
    show() { shown = true; slowAt = 0; tick(); clearInterval(timer); timer = setInterval(() => { if (shown && !document.hidden) tick(); }, 3000); },
    hide() { shown = false; clearInterval(timer); demo.destroy(); },
  };
}
