// Alert detail drawer. Order: evidence first (clip/snapshot + crops), summary, "why this priority" (the
// correlator's factor breakdown), then each event split into OBSERVATION (what was seen) and INFERENCE (how the
// rule derived it), optional VLM text (assistive inference), note, disclaimer and audit trail.
// Status buttons follow the server's state machine (workflow.js): only legal next steps are offered.
import { api, safe, ApiError } from "./api.js";
import { S, can, camName, emit, loadVlms, isSensorEvent } from "./state.js";
import { toast, obsBadge, infBadge, estBadge, acsBadge, prioChip, statusPill, openDialog, segmented, demoBadge, plainTitle } from "./ui.js";
import { allowed, canFlagUnclear, slaChip, CLOSED } from "./workflow.js";
import { $, h, icon, replace, fmtTime, fmtDateTime, fmtAgo, fmtDur, mediaUrl, enc, NOSNAP, RULE, DISP, num, PRIO } from "./util.js";

const drawer = () => $("#drawer");
let cur = null;          // alert id shown
let alertData = null;
let viewedIds = new Set();
let nav = null;          // {prev(), next()} supplied by the page that opened the drawer
let parts = null;        // persistent sub-containers for the open alert
let busy = false;
let returnFocus = null;

export function isOpen() { return drawer().classList.contains("on"); }
export function currentId() { return cur; }
export function setNavigator(n) { nav = n; }

export function closeDrawer() {
  const d = drawer();
  d.classList.remove("on");
  d.setAttribute("aria-hidden", "true");
  document.body.classList.remove("drawer-open");
  const v = d.querySelector("video");
  if (v) v.pause();
  cur = null; alertData = null; parts = null;
  emit("drawer", null);
  if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
  returnFocus = null;
}

export function navigate(delta) {
  if (!nav) return false;
  return delta > 0 ? nav.next() : nav.prev();
}

/** Open (or refresh) an alert in the drawer. */
export async function openAlert(id, { quiet = false } = {}) {
  let a;
  try { a = await api("/api/alerts/" + enc(id), { silent: quiet }); }
  catch (e) {
    if (!quiet && (e.kind === "missing" || e.kind === "notfound")) toast(`Alert ${id} no longer exists.`, "warn");
    return;
  }
  const d = drawer();
  if (cur !== id || !parts) {
    cur = id;
    parts = { head: h("div", { class: "dhead" }), media: h("div", { class: "media" }), mediaCap: h("div", { class: "media-cap" }),
      crops: h("div", { class: "crops" }), note: null, noteDirty: false, body: h("div"), audit: h("div", { class: "audit-list" }), clip: null };
    replace(d, parts.head, h("div", { class: "dbody" }, parts.media, parts.mediaCap, parts.crops, parts.body));
    d.scrollTop = 0;
  }
  alertData = a;
  render(a);
  if (!d.classList.contains("on")) {
    if (!returnFocus) returnFocus = document.activeElement;
    d.classList.add("on");
    d.setAttribute("aria-hidden", "false");
    document.body.classList.add("drawer-open");
    d.querySelector(".dhead button.close")?.focus({ preventScroll: true });
  }
  emit("drawer", id);
  if (!viewedIds.has(id) && can("evidence.view")) {
    viewedIds.add(id);
    api(`/api/alerts/${enc(id)}/viewed`, { method: "POST", body: {}, silent: true }).catch(() => {});
  }
  loadAudit(id);
}

async function loadAudit(id) {
  try {
    const list = await api(`/api/alerts/${enc(id)}/audit`, { silent: true });
    if (cur !== id || !parts) return;
    replace(parts.audit, list.length ? list.map(x => h("div", null,
      h("span", { class: "tnum" }, fmtDateTime(x.t)), " · ", h("b", null, x.actor), " · ", x.action,
      x.details?.from ? ` (from ${x.details.from}${x.details.reopen ? ", re-opened" : ""})` : "",
      x.details?.disposition ? ` (${DISP[x.details.disposition] || x.details.disposition})` : "",
      x.details?.reason ? ` — reason: ${x.details.reason}${x.details.to_role || x.details.to_user ? ` → ${[x.details.to_role, x.details.to_user].filter(Boolean).join(" / ")}` : ""}` : "",
      x.details?.note ? ` — ${x.details.note}` : "",
      x.details?.model ? ` — model ${x.details.model}` : "")) : h("div", null, "No audit entries yet."));
  } catch { /* audit is secondary; ignore */ }
}

function section(title, badge, ...rest) {
  return h("section", { class: "dsec" }, h("div", { class: "dsec-h" }, h("h3", null, title), badge || null, h("span", { class: "grow" })), rest);
}

function render(a) {
  const last = a.events[a.events.length - 1] || {};
  // ---- header (sticky: title + all action buttons, so toasts at the bottom never cover them) ----
  const noteVal = parts.note && parts.noteDirty ? parts.note.value : (a.operator_note || "");
  const disp = h("select", { "aria-label": "Disposition (final verdict for Resolve / Dismiss)", class: "sm" }, h("option", { value: "" }, "Disposition…"),
    Object.entries(DISP).filter(([k]) => k !== "unclear").map(([k, v]) => h("option", { value: k, selected: (parts.disp ?? a.disposition) === k }, v)));
  disp.addEventListener("change", () => { parts.disp = disp.value; });
  const err = h("span", { class: "small", role: "alert", style: { color: "var(--err)" } });
  const setStatus = safe(async (status, needDisp, extra = {}) => {
    if (busy) return;
    err.textContent = "";
    const body = { status, note: parts.note?.value || "", ...extra };
    if (needDisp) {
      if (!disp.value) { err.textContent = "Choose a disposition before closing the alert."; disp.focus(); return; }
      body.disposition = disp.value;
    }
    busy = true;
    let refresh = false;
    try {
      await api(`/api/alerts/${enc(a.id)}/status`, { method: "POST", body, silent: true });
      parts.noteDirty = false; parts.disp = undefined;
      toast(`Alert ${a.id} → ${body.disposition === "unclear" ? "flagged for second review" : status}`, "ok", { timeout: 2500 });
      refresh = true;
    } catch (e) {
      if (!(e instanceof ApiError)) throw e;
      err.textContent = e.detail || e.message;     // e.g. 409: someone else moved it first, or the step is not legal
      refresh = e.status === 409;
    } finally { busy = false; }
    if (refresh) {
      emit("alert-changed", a.id);
      const msg = err.textContent;
      await openAlert(a.id, { quiet: true });
      if (msg && parts?.errEl) parts.errEl.textContent = msg;
    }
  });
  parts.errEl = err;
  const reopen = CLOSED.includes(a.status);
  const acts = [];
  if (!reopen && allowed(a.status, "acknowledged")) acts.push(h("button", { class: "sm", onclick: () => setStatus("acknowledged") }, icon("check"), "Acknowledge"));
  if (allowed(a.status, "escalated")) acts.push(h("button", { class: "sm", title: "Escalate with a reason (optionally to a role or person)",
    onclick: () => escalateDialog(a, x => setStatus("escalated", false, x)) }, icon("alert"), "Escalate…"));
  if (canFlagUnclear(a)) acts.push(h("button", { class: "sm", title: "Not sure: keep it acknowledged and flag it for a second review",
    onclick: () => setStatus("acknowledged", false, { disposition: "unclear" }) }, icon("info"), "Unclear"));
  const canRes = allowed(a.status, "resolved"), canDis = allowed(a.status, "dismissed");
  if (canRes || canDis) acts.push(disp);
  if (canRes) acts.push(h("button", { class: "sm primary", onclick: () => setStatus("resolved", true) }, "Resolve"));
  if (canDis) acts.push(h("button", { class: "sm", onclick: () => setStatus("dismissed", true) }, "Dismiss"));
  if (reopen && allowed(a.status, "new"))
    acts.push(h("button", { class: "sm ghost", title: "Re-open into the triage queue (supervisor / admin)", onclick: () => setStatus("new") }, "Re-open"));
  if (can("evidence.export")) {
    const exp = h("a", { class: "btnlink sm", title: "Evidence bundle: alert JSON, audit trail, snapshots (original + with overlay) and clips, hash-chained" }, icon("download"), "Export");
    const setHref = () => {
      exp.href = `/api/alerts/${enc(a.id)}/export${parts.anon ? "?anonymise=1" : ""}`;
      exp.download = `argus_${a.id}${parts.anon ? "_anonymised" : ""}.zip`;
    };
    const chk = h("input", { type: "checkbox", checked: !!parts.anon, onchange: () => { parts.anon = chk.checked; setHref(); } });
    setHref();
    acts.push(exp, h("label", { class: "small", title: "Also add face/head-blurred copies, labelled 'anonymised copy (derived)'. The originals stay in the bundle as the evidence record." },
      chk, " anonymised copies"));
  }
  if (can("evidence.hold")) {
    const held = !!a.legal_hold;
    acts.push(h("button", { class: "sm" + (held ? " ghost" : ""), title: held ? "Release the legal hold: retention may then delete this alert and its evidence"
      : "Legal hold: retention never deletes this alert or its evidence while held", onclick: safe(async () => {
      if (busy) return;
      busy = true;
      try {
        await api(`/api/alerts/${enc(a.id)}/hold`, held ? { method: "DELETE" } : { method: "POST", body: { reason: parts.note?.value || "" } });
        toast(held ? `Legal hold released on ${a.id}` : `Alert ${a.id} placed on legal hold`, "ok", { timeout: 2500 });
        emit("alert-changed", a.id);
        await openAlert(a.id, { quiet: true });
      } finally { busy = false; }
    }) }, icon("lock"), held ? "Release hold" : "Legal hold"));
  }
  if (S.cfg?.explain_enabled && can("alert.acknowledge")) {
    const sel = h("select", { "aria-label": "VLM model", class: "sm" }, S.vlms.length ? S.vlms.map(m => h("option", {
      value: m.id, disabled: m.available === false, selected: parts.vlm ? parts.vlm === m.id : !!m.loaded,
    }, `${String(m.id).split("/").pop()} (${m.params_b ?? "?"}B, ${m.mode || "won't fit here"})`)) : h("option", { value: "" }, "default model"));
    sel.addEventListener("change", () => { parts.vlm = sel.value; });
    const btn = h("button", { class: "sm", title: "Ask the vision-language model for an assistive description (never used for scoring)" }, icon("spark"), "Explain");
    btn.addEventListener("click", safe(async () => {
      if (busy) return;
      busy = true; btn.disabled = true; replace(btn, h("span", { class: "spinner" }), "Asking VLM (first use loads the model)…");
      try {
        await api(`/api/alerts/${enc(a.id)}/explain`, { method: "POST", body: sel.value ? { model: sel.value } : {} });
        loadVlms();
        await openAlert(a.id, { quiet: true });
      } finally { busy = false; btn.disabled = false; replace(btn, icon("spark"), "Explain"); }
    }));
    acts.push(sel, btn);
  }
  const navBtns = nav ? [
    h("button", { class: "ghost icon", title: "Previous alert (k)", "aria-label": "Previous alert", onclick: () => navigate(-1) }, icon("up")),
    h("button", { class: "ghost icon", title: "Next alert (j)", "aria-label": "Next alert", onclick: () => navigate(1) }, icon("down"))] : [];
  replace(parts.head,
    h("div", { class: "row", style: { flexWrap: "nowrap" } },
      prioChip(a.priority, { long: true }), demoBadge(a),
      h("h2", { class: "grow", style: { margin: 0, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: a.title }, plainTitle(a)),
      navBtns,
      h("button", { class: "ghost icon close", title: "Close (Esc)", "aria-label": "Close alert details", onclick: closeDrawer }, icon("x"))),
    h("div", { class: "dmeta" },
      statusPill(a.status),
      slaChip(a),
      a.second_review ? h("span", { class: "tag warn", title: `Flagged unclear by ${a.second_review_flagged_by || "an operator"}: waiting for a second review` }, "second review") : null,
      a.disposition && a.disposition !== "unclear" ? h("span", null, DISP[a.disposition] || a.disposition) : null,
      a.legal_hold ? h("span", { class: "tag warn", title: `Legal hold by ${a.legal_hold.actor} since ${fmtDateTime(a.legal_hold.t)}${a.legal_hold.reason ? " — " + a.legal_hold.reason : ""}. Retention will not delete it.` },
        icon("lock"), "Legal hold") : null,
      h("span", { class: "mono" }, a.id),
      h("span", null, "score ", h("b", { class: "tnum" }, num(a.score, 2))),
      h("span", null, (a.cameras || []).map(camName).join(", ")),
      h("span", { class: "tnum" }, `${fmtTime(a.created_at)} – ${fmtTime(a.updated_at)}`),
      a.incident_id ? h("a", { href: "#/incidents/" + enc(a.incident_id), title: "Open the incident this alert belongs to",
        onclick: () => closeDrawer() }, `incident ${a.incident_id}`) : null),
    a.status === "escalated" && a.escalation ? h("div", { class: "banner err", style: { margin: "6px 0 0" } }, icon("alert"),
      h("span", null, h("b", null, "Escalated"),
        a.escalation.to_role || a.escalation.to_user ? ` to ${[a.escalation.to_role, a.escalation.to_user].filter(Boolean).join(" / ")}` : "",
        ` by ${a.escalation.by || "?"} at ${fmtTime(a.escalation.t)}: `, a.escalation.reason)) : null,
    a.suppressed ? h("div", { class: "banner", style: { margin: "6px 0 0" } }, h("b", null, "Muted: "),
      a.suppressed_reason || "a camera-integrity incident is open on this camera; the alert is visible but not pushed.") : null,
    acts.length ? h("div", { class: "actions" }, acts, err) : h("div", { class: "small mute", style: { marginTop: "6px" } },
      reopen ? `This alert is ${a.status}. Re-opening it needs a supervisor or admin.` : "Read-only: your role cannot change alert status."));

  // ---- evidence: only replaced when the clip changes (keeps playback position across live updates) ----
  const clip = last.evidence?.clip ? mediaUrl(last.evidence.clip) : null;
  const poster = last.evidence?.snapshot ? mediaUrl(last.evidence.snapshot) : NOSNAP;
  const raw = last.evidence?.snapshot_raw ? mediaUrl(last.evidence.snapshot_raw) : null;
  const view = raw && parts.view === "original" ? "original" : "overlay";
  if (parts.clip !== (view === "original" ? raw : (clip || poster))) {
    parts.clip = view === "original" ? raw : (clip || poster);
    if (view === "original" && can("evidence.view")) {
      replace(parts.media, h("img", { src: raw, alt: "Original evidence frame (no software overlay)" }));
    } else if (clip && can("evidence.view")) {
      const v = h("video", { controls: true, autoplay: true, muted: true, loop: true, playsinline: true, "aria-label": "Evidence clip" });
      const myParts = parts;
      let tries = 0;
      const note = h("div", { class: "small mute", role: "status" });
      v.poster = poster; v.src = clip;
      // Clips are written a few seconds after the event: retry a few times before giving up.
      v.addEventListener("error", () => {
        if (parts !== myParts) return;
        if (tries++ < 6) {
          note.textContent = "Clip is still being written… retrying.";
          setTimeout(() => { if (parts === myParts) v.src = clip + (clip.includes("?") ? "&" : "?") + "r=" + tries; }, 3000);
        } else note.textContent = "Clip not playable (showing the snapshot poster).";
      });
      v.addEventListener("loadeddata", () => { note.textContent = ""; });
      replace(parts.media, v, note);
    } else replace(parts.media, h("img", { src: can("evidence.view") ? poster : NOSNAP, alt: "Evidence snapshot" }));
  }
  const ft = last.evidence?.frame_time;
  const toggle = raw && can("evidence.view") ? segmented([["original", "Original", "The frame as analysed, nothing drawn on it"],
    ["overlay", "With overlay", "Clip and snapshot with boxes, zones and labels drawn by Argus (software overlay)"]], view,
  v => { parts.view = v; render(alertData); }, { label: "Evidence view" }) : null;
  const sensorCam = isSensorEvent(last) ? last.evidence?.camera : null;  // camera that views the reporting door
  replace(parts.mediaCap, obsBadge("Evidence"),
    sensorCam ? h("span", null, `${camName(sensorCam)} · view when the access-control report arrived · ${ft ? fmtDateTime(ft) : "time not recorded"}`)
      : isSensorEvent(last) ? h("span", null, last.derivation?.covering_camera
        ? `${camName(last.derivation.covering_camera)} views this door · no frame was available when the report arrived`
        : `${camName(last.camera_id)} · no camera mapped to this door`)
      : h("span", null, `${camName(last.camera_id)} · frame ${ft ? fmtDateTime(ft) : "time not recorded"}`),
    h("span", { class: "grow" }),
    toggle,
    // main-stream recording (MediaMTX restreamer): the Recordings page finds the segment and seeks to 4 s before the frame
    can("evidence.view") && S.cfg?.restream && ft && last.camera_id
      ? h("a", { href: `#/recordings?cam=${enc(last.camera_id)}&t=${enc(ft)}&stream=main`, onclick: () => closeDrawer(),
        title: "Open the camera's full-resolution main-stream recording at this moment (4 s before the evidence frame)" }, "Main-stream recording", icon("play"))
      : null,
    can("evidence.view") && last.evidence?.snapshot ? h("a", { href: view === "original" ? raw : poster, target: "_blank", rel: "noopener" },
      view === "original" ? "Original" : "Snapshot", icon("external")) : null);
  const crops = can("evidence.view") ? a.events.filter(e => e.evidence?.crop).slice(-8) : [];
  replace(parts.crops, crops.map(e => h("a", { href: mediaUrl(e.evidence.crop), target: "_blank", rel: "noopener", title: `${RULE[e.rule] || e.rule} · ${fmtTime(e.t_end)} (open full size)` },
    h("img", { src: mediaUrl(e.evidence.crop), alt: `Evidence crop: ${RULE[e.rule] || e.rule}`, loading: "lazy" }))));
  parts.crops.hidden = !crops.length;

  // ---- body ----
  if (!parts.note) {
    parts.note = h("textarea", { rows: 2, placeholder: "Operator note (saved with the next status action)", "aria-label": "Operator note" });
    parts.note.addEventListener("input", () => { parts.noteDirty = true; });
  }
  if (!parts.noteDirty) parts.note.value = noteVal;
  parts.note.disabled = !can("alert.acknowledge") && !can("alert.resolve");

  const zones = [...new Set(a.events.map(e => e.zone_name).filter(Boolean))];
  const summary = h("div", { class: "sumgrid" },
    [["Priority", `${a.priority} · ${PRIO[a.priority]?.label || ""}`], ["Cameras", (a.cameras || []).map(camName).join(", ") || "—"],
      ["Zones", zones.join(", ") || "—"], ["Events", `${a.events.length} · ${[...new Set(a.events.map(e => RULE[e.rule] || e.rule))].length} behaviour(s)`],
      ["First seen", `${fmtTime(a.created_at)}`], ["Last update", fmtAgo(a.updated_at)]]
      .map(([k, v]) => h("div", null, h("div", { class: "sl" }, k), h("div", { class: "sv", title: v }, v))));

  replace(parts.body,
    summary,
    priorityWhy(a),
    healthRecovery(a),
    section(`Events (${a.events.length})`, null, h("div", { class: "provenance" }, obsBadge(), "what the detector and tracker saw ·", infBadge(), "how a deterministic rule derived the alert"),
      a.events.slice().reverse().map(e => eventCard(e, a))),
    a.explanation ? h("div", { class: "vlm" },
      h("div", { class: "row" }, infBadge("VLM · assistive inference"), h("b", null, "Description (not evidence)")),
      h("p", { style: { margin: "6px 0 0" } }, a.explanation),
      h("div", { class: "small mute" }, "Generated on operator request from the evidence (frames sampled from the clip, or the snapshot; see the bracket). Never used to trigger, score or close alerts.")) : null,
    section("Operator note", null, parts.note),
    h("div", { class: "disclaimer" }, a.disclaimer || "Deterministic-rule alert on observed position/motion only; confidence is heuristic and uncalibrated. Decision rests with the operator.",
      " It does not assess intent or identity. Cross-camera grouping is spatial only (no identity claim)."),
    section("Audit trail", null, parts.audit));
}

/** Escalation needs a reason; the target role / person is optional. */
function escalateDialog(a, submit) {
  const reason = h("textarea", { rows: 3, maxlength: 500, placeholder: "Why does this need escalation? (required)", "aria-label": "Escalation reason" });
  const role = h("select", { "aria-label": "Escalate to role" }, h("option", { value: "" }, "Any role"),
    ["supervisor", "operator", "admin"].map(r => h("option", { value: r, selected: r === "supervisor" }, r)));
  const user = h("input", { maxlength: 64, placeholder: "Username (optional)", "aria-label": "Escalate to user" });
  const msg = h("div", { class: "small", role: "alert", style: { color: "var(--err)" } });
  openDialog({ title: `Escalate alert ${a.id}`,
    body: h("div", { class: "stack" },
      h("div", { class: "small mute" }, a.title),
      h("label", { class: "field" }, "Reason", reason),
      h("div", { class: "row" }, h("label", { class: "field grow" }, "To role", role), h("label", { class: "field grow" }, "To person", user)),
      h("div", { class: "note" }, "Recorded on the alert and in the audit trail. Nothing is sent from the console."), msg),
    actions: [{ label: "Cancel" }, { label: "Escalate", primary: true, onclick: () => {
      if (!reason.value.trim()) { msg.textContent = "A reason is required."; reason.focus(); return false; }
      submit({ reason: reason.value.trim(), ...(role.value ? { to_role: role.value } : {}), ...(user.value.trim() ? { to_user: user.value.trim() } : {}) });
      return true;
    } }] });
  setTimeout(() => reason.focus(), 30);
}

/** "Why was I shown this?": score = severity × criticality × confidence × corroboration, against the P-bands. */
function priorityWhy(al) {
  const f = al.factors;
  if (!f || !f.top_event) return null;
  const term = (v, l, cls) => h("span", { class: ["term", cls] }, h("b", null, v), h("span", null, l));
  const op = s => h("span", { class: "op", "aria-hidden": "true" }, s);
  const b = f.bands || {};
  const max = Math.max((+b.P1 || 6) * 1.5, (+f.score || 0) * 1.1, 1);
  const seg = [["P4", 0, +b.P3 || 0], ["P3", +b.P3 || 0, +b.P2 || 0], ["P2", +b.P2 || 0, +b.P1 || 0], ["P1", +b.P1 || 0, max]];
  return h("details", { class: "why", open: true },
    h("summary", null, infBadge("Inference"), `Why ${al.priority}?`, h("span", { class: "mute small", style: { fontWeight: 500 } }, `score ${num(f.score, 2)} from the correlator`)),
    h("div", { class: "why-b" },
      h("div", { class: "eq", role: "math", "aria-label": `severity ${f.severity} times zone criticality ${f.zone_criticality} times confidence ${f.confidence} equals ${f.base_score}, times corroboration ${f.corroboration_multiplier} equals ${f.score}` },
        term(f.severity, "severity"), op("×"), term(f.zone_criticality, "zone crit."), op("×"), term(num(f.confidence, 2), "confidence"),
        op("="), term(num(f.base_score, 2), "base"), op("×"), term(`×${num(f.corroboration_multiplier, 2)}`, "corroboration"), op("="), term(num(f.score, 2), "score", "res")),
      h("div", { class: "bandbar", role: "img", "aria-label": `Priority bands: P3 from ${b.P3}, P2 from ${b.P2}, P1 from ${b.P1}; this alert scores ${f.score}` },
        seg.map(([p, a0, a1]) => h("div", { class: p, style: { width: `${Math.max(0, (a1 - a0) / max * 100)}%` }, title: `${p} ≥ ${a0}` }, p)),
        h("span", { class: "mk", style: { left: `calc(${Math.min(100, (+f.score || 0) / max * 100)}% - 1px)` }, title: `score ${f.score}` })),
      h("div", { class: "small mute" },
        `Top behaviour: ${RULE[f.top_event] || f.top_event}. Corroboration: ${f.distinct_rules} behaviour(s) on ${f.distinct_cameras} camera(s)`,
        f.cross_modal ? [" across ", h("b", null, f.cross_modal), " (two independent modalities agree; +0.25 per extra modality)"] : "", ". ",
        `Bands: P1 ≥ ${b.P1}, P2 ≥ ${b.P2}, P3 ≥ ${b.P3}. Confidence is heuristic (uncalibrated).`),
      trackRecord(f.track_record),
      f.demoted ? h("div", { class: "cap" }, icon("info"), `Lowered ${f.demoted.from} → ${f.demoted.to}: ${f.demoted.reason}.`) : null,
      f.cap ? h("div", { class: "cap" }, icon("info"), f.cap) : null));
}

/** Operator track record for the top behaviour on this camera: reviewed decisions, not a model. */
function trackRecord(tr) {
  if (!tr) return null;
  const rule = RULE[tr.rule] || tr.rule;
  const site = tr.site_reviewed ? ` Site-wide: ${tr.site_confirmed} of ${tr.site_reviewed}.` : "";
  if (!tr.reviewed) return h("div", { class: "small mute" }, h("b", null, "Track record: "),
    `no reviewed '${rule}' alerts on ${camName(tr.camera)} in the last ${tr.window_days} days yet.${site}`);
  return h("div", { class: "small mute" }, h("b", null, "Track record: "),
    `operators confirmed ${tr.confirmed} of ${tr.reviewed} reviewed '${rule}' alerts on ${camName(tr.camera)} in the last ${tr.window_days} days `,
    `(precision ${num(tr.precision, 2)}, 95% CI ${num(tr.ci95[0], 2)}–${num(tr.ci95[1], 2)}).${site}`);
}

/** Camera view conditions that cleared after this alert was raised (argonix Site.on_health_recovered). */
function healthRecovery(a) {
  const rec = a.health_recovery;
  if (!rec || !rec.length) return null;
  return h("div", { class: "recovered", role: "status" }, icon("info"),
    rec.map(r => `View condition '${r.condition}' cleared at ${fmtTime(r.recovered_at)} after ${fmtDur(r.outage_s)}.`).join(" "));
}

function thresholds(t) {
  if (!t || typeof t !== "object") return t == null ? "—" : String(t);
  const ent = Object.entries(t);
  return ent.length ? ent.map(([k, v]) => `${k} = ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ") : "—";
}

function sensorCard(e) {
  const d = e.derivation || {}, ev = e.evidence || {};
  const kv = pairs => h("div", { class: "kv" }, pairs.filter(Boolean).flatMap(([k, v]) => [h("div", null, k), h("div", null, v)]));
  const det = d.details && Object.keys(d.details).length ? Object.entries(d.details).map(([k, v]) => `${k} = ${v}`).join(", ") : null;
  return h("div", { class: "ev acs" },
    h("div", { class: "ev-head" }, acsBadge(), h("b", null, RULE[e.rule] || e.rule), h("code", { class: "mute" }, e.rule),
      h("span", { class: "grow" }), h("span", { class: "mute small tnum" }, e.t_start !== e.t_end ? `${fmtTime(e.t_start)} → ${fmtTime(e.t_end)}` : fmtTime(e.t_end))),
    h("div", { class: "ev-cols" },
      h("div", { class: "o" },
        h("h4", null, obsBadge("Device report"), "What was reported"),
        h("p", { style: { margin: "0 0 8px" } }, e.observation),
        kv([["Source", `${d.source_kind || "access-control device"} ${d.source_id || e.camera_id.replace(/^sensor:/, "")}`],
          d.door ? ["Door", d.door] : null,
          ["Zone", [e.zone_name, e.site_zone ? ` · site zone ${e.site_zone}` : ""]],
          ["Event type", h("code", null, d.event_type || "—")],
          ["Time source", d.clock === "panel" ? "panel clock (within the accepted skew)" : "arrival time at Argus"],
          det ? ["Details", h("span", { class: "mono" }, det)] : null,
          (d.card_hashes || (d.card_hash ? [d.card_hash] : [])).length ? ["Credentials", h("span", { class: "mono" }, (d.card_hashes || [d.card_hash]).join(", "), h("div", { class: "mute small" }, "keyed hashes; raw card ids are never stored"))] : null,
          ["Camera view", ev.camera ? [camName(ev.camera), ev.snapshot && can("evidence.view") ? [" · ",
            h("a", { href: mediaUrl(ev.snapshot), target: "_blank", rel: "noopener" }, "snapshot"), " · ",
            h("a", { href: mediaUrl(ev.clip), target: "_blank", rel: "noopener" }, "clip")] : ""]
            : d.covering_camera ? `${camName(d.covering_camera)} (no frame available at arrival)` : d.door ? "no camera mapped to this door" : "—"]])),
      h("div", { class: "d" },
        h("h4", null, infBadge(), "Why it fired"),
        kv([["Rule", h("code", null, d.rule || e.rule)],
          d.threshold ? ["Thresholds", thresholds(d.threshold)] : null,
          ["Confidence", [`${e.confidence} `, estBadge("heuristic"), h("div", { class: "mute small" }, d.confidence_method || "method not recorded")]],
          ["Severity × criticality", `${e.severity} × ${e.criticality}`],
          ["Reported by", d.reported_by || "access-control device"],
          ["Authenticated", d.auth === "hmac" ? "HMAC-signed request (per-source secret)" : d.auth ? `API token (${String(d.auth).replace(/^token:/, "")})` : "—"],
          ["Modality", "access control (deterministic device report, not a learned model)"]]))));
}

function eventCard(e, a) {
  if (isSensorEvent(e)) return sensorCard(e);
  const d = e.derivation || {}, ev = e.evidence || {}, cal = ev.calibration || {};
  const calErr = d.calib_error_m ?? cal.calib_error_m;
  const model = ev.model || {};
  const kv = pairs => h("div", { class: "kv" }, pairs.filter(Boolean).flatMap(([k, v]) => [h("div", null, k), h("div", null, v)]));
  return h("div", { class: "ev" },
    h("div", { class: "ev-head" }, h("b", null, RULE[e.rule] || e.rule), h("code", { class: "mute" }, e.rule),
      h("span", { class: "grow" }), h("span", { class: "mute small tnum" }, `${fmtTime(e.t_start)} → ${fmtTime(e.t_end)}`)),
    h("div", { class: "ev-cols" },
      h("div", { class: "o" },
        h("h4", null, obsBadge(), "What was seen"),
        h("p", { style: { margin: "0 0 8px" } }, e.observation),
        ev.crop && can("evidence.view") ? h("a", { href: mediaUrl(ev.crop), target: "_blank", rel: "noopener", title: "Evidence crop (open full size)" },
          h("img", { src: mediaUrl(ev.crop), alt: "Evidence crop", style: { maxWidth: "100%", maxHeight: "180px", display: "block", margin: "0 0 8px", borderRadius: "6px" } })) : null,
        kv([["Camera", `${camName(e.camera_id)} (${e.camera_id})`],
          ["Zone", [e.zone_name, e.site_zone ? ` · site zone ${e.site_zone}` : ""]],
          ["Tracks", (e.track_ids || []).length ? e.track_ids.map(t => "#" + t).join(", ") : "—"],
          ["Duration", d.duration_s != null ? `${d.duration_s} s` : "—"],
          ["Ground position", e.ground ? [`(${num(e.ground[0])}, ${num(e.ground[1])}) m `, estBadge(`± ${calErr ?? "?"} m`)] : "—"],
          ["Evidence", ev.snapshot && can("evidence.view")
            ? [ev.snapshot_raw ? [h("a", { href: mediaUrl(ev.snapshot_raw), target: "_blank", rel: "noopener", title: "Frame as analysed, no overlay" }, "original"), " · "] : null,
              h("a", { href: mediaUrl(ev.snapshot), target: "_blank", rel: "noopener", title: "With software overlay" }, "snapshot"), " · ",
              h("a", { href: mediaUrl(ev.clip), target: "_blank", rel: "noopener" }, "clip")]
            : "none (no frame)"]])),
      h("div", { class: "d" },
        h("h4", null, infBadge(), "Why it fired"),
        kv([["Rule", [h("code", null, d.rule || e.rule), d.condition ? ` · condition ${d.condition}` : ""]],
          ["Thresholds", thresholds(d.threshold)],
          ["Confidence", [`${e.confidence} `, estBadge("heuristic"), h("div", { class: "mute small" }, d.confidence_method || "method not recorded")]],
          ["Severity × criticality", `${e.severity} × ${e.criticality}`],
          d.metrics ? ["Image metrics", h("span", { class: "mono" }, thresholds(d.metrics))] : null,
          d.hour_of_week != null ? ["Baseline", `hour-of-week ${d.hour_of_week}, value ${d.value ?? "?"} (corroboration only)`] : null,
          ["Model", [model.detector, model.tracker, a.model?.rules].filter(Boolean).join(" · ") || "—"],
          ["Calibration", [calErr != null ? `error ≈ ${calErr} m ` : "—", calErr != null ? estBadge() : null,
            cal.measured_rms_m != null ? h("div", { class: "mute small" }, `measured RMS ${cal.measured_rms_m} m on ${cal.check_points ?? 0} check point(s)`) : null]]]))));
}

/** Called by the app on WebSocket alert updates. */
export function onAlertUpdate(a) {
  if (cur && a.id === cur && isOpen() && !busy) openAlert(cur, { quiet: true });
}
