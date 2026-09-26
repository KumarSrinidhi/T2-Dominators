// Operator workflow on the client: the server's alert state machine (GET /api/config -> workflow), acknowledge SLA
// waiting time / overdue, alert-flood detection, and the per-browser "auto-open cameras on P1/P2" setting.
// The server enforces every transition (409 on an illegal one); this module only decides which buttons to offer.
import { S, can } from "./state.js";
import { h, icon, ls, fmtDur } from "./util.js";

const FALLBACK = {
  transitions: { new: ["acknowledged", "escalated", "dismissed"], acknowledged: ["escalated", "resolved", "dismissed"],
    escalated: ["acknowledged", "resolved"], resolved: ["new", "acknowledged"], dismissed: ["new", "acknowledged"] },
  reopen_perms: ["incident.assign", "incident.merge", "user.admin"],
  sla_s: { P1: 60, P2: 300, P3: 1800, P4: null },
  flood: { threshold: 10, window_s: 600 },
  unclear: "unclear",
};
export const CLOSED = ["resolved", "dismissed"];

export function wf() { return S.cfg?.workflow || FALLBACK; }

/** Can the signed-in user move an alert from `cur` to `to`? (mirrors argonix.workflow.plan) */
export function allowed(cur, to) {
  const t = wf().transitions[cur || "new"] || [];
  if (!t.includes(to)) return false;
  if (CLOSED.includes(cur)) return canReopen();
  const perm = { acknowledged: "alert.acknowledge", escalated: "alert.escalate", resolved: "alert.resolve", dismissed: "alert.dismiss", new: "alert.acknowledge" }[to];
  return can(perm);
}
export function canReopen() { return wf().reopen_perms.some(p => can(p)); }
/** "Unclear" keeps (or makes) the alert acknowledged and flags it for a second review. */
export function canFlagUnclear(a) {
  if (a.second_review || CLOSED.includes(a.status)) return false;
  return can("alert.acknowledge") && (a.status === "acknowledged" || allowed(a.status, "acknowledged"));
}

// ------------------------------------------------------------------------------------------------ acknowledge SLA
export function slaFor(priority) { const v = wf().sla_s?.[priority]; return v ? +v : null; }

/** For an unacknowledged (status "new") alert: {wait, sla, overdue, left}; null otherwise. */
export function waitInfo(a, now = Date.now() / 1000) {
  if (!a || a.status !== "new") return null;
  const since = +(a.reopened_at ?? a.created_at ?? now);
  const wait = Math.max(0, now - since), sla = slaFor(a.priority);
  return { wait, sla, overdue: sla != null && wait > sla, left: sla != null ? sla - wait : null };
}

export function overdueCount(list, now = Date.now() / 1000) {
  let n = 0;
  for (const a of list || []) if (waitInfo(a, now)?.overdue) n++;
  return n;
}

/** Small chip: "waiting 2m 10s" (with the SLA) or "overdue +1m 5s". Null for acknowledged alerts. */
export function slaChip(a, { compact = false } = {}) {
  const w = waitInfo(a);
  if (!w) return null;
  const title = w.sla != null ? `Unacknowledged for ${fmtDur(w.wait)}; ${a.priority} acknowledge SLA is ${fmtDur(w.sla)}`
    : `Unacknowledged for ${fmtDur(w.wait)} (no SLA for ${a.priority})`;
  if (w.overdue) return h("span", { class: "sla over", title }, icon("clock"), compact ? `+${fmtDur(w.wait - w.sla)}` : `overdue +${fmtDur(w.wait - w.sla)}`);
  return h("span", { class: ["sla", w.sla != null && w.left < w.sla * 0.25 && "soon"], title }, icon("clock"),
    compact ? fmtDur(w.wait) : `waiting ${fmtDur(w.wait)}${w.sla != null ? ` / ${fmtDur(w.sla)}` : ""}`);
}

// ------------------------------------------------------------------------------------------------ alert flood
/** {n, threshold, window_s, flood} from the list of status=new alerts. */
export function floodState(newAlerts, now = Date.now() / 1000) {
  const f = wf().flood || FALLBACK.flood;
  const win = +f.window_s || 600, thr = +f.threshold || 10;
  const n = (newAlerts || []).filter(a => +a.created_at >= now - win).length;
  return { n, threshold: thr, window_s: win, flood: n > thr };
}
export function isFlood() { return !!S.flood?.flood; }

// ------------------------------------------------------------------------------------------------ auto-open cameras
export function autoOpen() { return ls.get("autoopen.p12", "0") === "1"; }
export function setAutoOpen(v) { ls.set("autoopen.p12", v ? "1" : null); }
