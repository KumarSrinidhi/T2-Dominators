// Cameras (settings): list + status, add-camera wizard (ONVIF discovery / vendor templates / test), edit,
// enable/disable, delete, and the per-camera detail page (zone & mask editor, auto-calibration, placement).
import { api, optional, ApiError } from "./api.js";
import { S, can, on, camById, loadCameras, loadConfig } from "./state.js";
import { toast, openDialog, confirmDialog, segmented, featureMissing, progress, pageHeader, emptyState, estBadge } from "./ui.js";
import { h, replace, enc, icon, slug, num, BLANK } from "./util.js";
import { isMono } from "./state.js";
import { createZoneEditor } from "./zones.js";
import { createCalibPanel } from "./calib.js";

// Used only when GET /api/cameras/templates is missing. Common, publicly documented RTSP path patterns.
const FALLBACK_TEMPLATES = [
  { vendor: "Generic RTSP (enter path manually)", main: "rtsp://{user}:{password}@{host}:554/", sub: "rtsp://{user}:{password}@{host}:554/", notes: "Check the camera manual for the exact RTSP path." },
  { vendor: "Hikvision", main: "rtsp://{user}:{password}@{host}:554/Streaming/Channels/{channel}01", sub: "rtsp://{user}:{password}@{host}:554/Streaming/Channels/{channel}02", notes: "Channel 1 → 101 (main) / 102 (sub)." },
  { vendor: "Dahua / Amcrest", main: "rtsp://{user}:{password}@{host}:554/cam/realmonitor?channel={channel}&subtype=0", sub: "rtsp://{user}:{password}@{host}:554/cam/realmonitor?channel={channel}&subtype=1", notes: "subtype=1 is the sub-stream." },
  { vendor: "Reolink", main: "rtsp://{user}:{password}@{host}:554/h264Preview_0{channel}_main", sub: "rtsp://{user}:{password}@{host}:554/h264Preview_0{channel}_sub", notes: "Enable RTSP in the Reolink network settings first." },
  { vendor: "Axis", main: "rtsp://{user}:{password}@{host}/axis-media/media.amp", sub: "rtsp://{user}:{password}@{host}/axis-media/media.amp?resolution=640x360", notes: "Sub-stream via a resolution parameter." },
];

// Server templates (GET /api/cameras/templates) use {user} {password} {host} {port} {channel} {channel2}.
function fillTemplate(tpl, v, mask = false, defPort = 554) {
  let out = String(tpl || "");
  if (!v.user && !v.password) out = out.replace("{user}:{password}@", "");
  const ch = Math.max(1, Math.round(+v.channel || 1));
  return out.replaceAll("{host}", v.host.trim()).replaceAll("{port}", String(+v.port || defPort || 554))
    .replaceAll("{channel2}", String(ch).padStart(2, "0")).replaceAll("{channel}", String(ch))
    .replaceAll("{user}", encodeURIComponent(v.user || ""))
    .replaceAll("{password}", mask ? (v.password ? "••••" : "") : encodeURIComponent(v.password || ""));
}
function hostFrom(dev) {
  if (typeof dev.host === "string" && /^[\w.:-]+$/.test(dev.host)) return dev.host;
  for (const x of dev.xaddrs || []) { try { return new URL(x).hostname; } catch { /* next */ } }
  const a = String(dev.address || "");
  return /^[\w.:-]+$/.test(a) ? a.replace(/:\d+$/, "") : "";
}
const b64ok = s => typeof s === "string" && s.length > 0 && /^[A-Za-z0-9+/=\s]+$/.test(s);

function statusCell(c) {
  const st = S.stats[c.id];
  if (c.enabled === false) return h("span", { class: "mute" }, "disabled");
  const status = st?.status || c.status || "unknown";
  const health = st?.health || c.health;
  return h("span", null, h("span", { class: "dot", style: { background: status === "running" ? (health === "DEGRADED" ? "var(--warn)" : "var(--ok)") : status === "error" ? "var(--err)" : "var(--p4)", marginRight: "6px" } }),
    status, health ? h("span", { class: "h-" + health }, ` · ${health}`) : null,
    st?.fps != null ? h("span", { class: "mute" }, ` · ${num(st.fps, 1)} fps`) : c.fps ? h("span", { class: "mute" }, ` · ${num(+c.fps, 1)} fps`) : null);
}

// ---------------- add-camera wizard ----------------
async function addCameraWizard() {
  const v = { method: "template", host: "", port: "", user: "", password: "", channel: 1, stream: "sub", manual: "", tpl: 0, name: "", id: "", enabled: true };
  let templates = FALLBACK_TEMPLATES, builtIn = true, disclaimer = "", step = 1, test = null, devices = null, discovering = false;
  try {
    // the server answers {disclaimer, verified, templates: [...]}; older servers may return a bare array
    const t = await optional("/api/cameras/templates", "templates", { silent: true });
    const list = Array.isArray(t) ? t : Array.isArray(t?.templates) ? t.templates : null;
    if (list && list.length) { templates = list; builtIn = false; disclaimer = typeof t?.disclaimer === "string" ? t.disclaimer : ""; }
  } catch { /* keep fallback */ }

  const body = h("div");
  const source = () => v.method === "manual" ? v.manual.trim() : fillTemplate(templates[v.tpl]?.[v.stream], v, false, templates[v.tpl]?.port);
  const sourceMasked = () => v.method === "manual" ? v.manual.trim().replace(/(\/\/[^:/@]+:)[^@]*@/, "$1••••@") : fillTemplate(templates[v.tpl]?.[v.stream], v, true, templates[v.tpl]?.port);
  const dlg = openDialog({ title: "Add camera", body, wide: true });

  const input = (key, attrs = {}) => {
    const i = h("input", { value: v[key], ...attrs });
    i.addEventListener("input", () => { v[key] = attrs.type === "number" ? +i.value : i.value; test = null; preview(); renderActions(); });
    return i;
  };
  const previewEl = h("code", { class: "mono", style: { wordBreak: "break-all" } });
  const preview = () => { previewEl.textContent = sourceMasked() || "—"; };

  function stepper() {
    return [h("div", { class: "wizard-steps", role: "list", "aria-label": `Step ${step} of 3` }, ["Source", "Test connection", "Name & save"].map((l, i) =>
      h("span", { class: [i + 1 === step && "on", i + 1 < step && "done"], role: "listitem", "aria-current": i + 1 === step ? "step" : null }, l))),
    h("div", { class: "wizard-progress" }, progress((step - 1) / 2))];
  }

  function renderDiscover() {
    if (discovering) return h("div", { class: "row small mute" }, h("span", { class: "spinner" }), "Scanning the local network (ONVIF WS-Discovery)…");
    if (devices === null) return h("button", { onclick: discover }, "Scan network");
    if (devices === "missing") return featureMissing("ONVIF discovery");
    if (!devices.length) return h("div", null, h("div", { class: "mute small" }, "No ONVIF devices answered. Cameras on other subnets/VLANs are not discoverable; use a template instead."), h("button", { onclick: discover }, "Scan again"));
    return h("div", { class: "list picklist" }, devices.map(d => h("button", { class: "item", onclick: () => {
      v.host = hostFrom(d); v.name = v.name || d.name || ""; v.method = "template";
      const hw = String(d.hardware || d.name || "").toLowerCase();
      const i = templates.findIndex(t => hw && String(t.vendor).toLowerCase().split(/[^a-z]+/).some(w => w.length > 2 && hw.includes(w)));
      if (i >= 0) v.tpl = i;
      render();
    } },
      h("b", null, d.name || d.hardware || "ONVIF device"), h("div", { class: "small mute" }, [hostFrom(d) || d.address, d.hardware].filter(Boolean).join(" · ")))));
  }
  async function discover() {
    discovering = true; render();
    try {
      const r = await optional("/api/cameras/discover", "discover");
      // the server answers {devices: [...], note}; older servers may return a bare array
      devices = r === null ? "missing" : Array.isArray(r) ? r : Array.isArray(r?.devices) ? r.devices : [];
    } catch { devices = []; }
    discovering = false; render();
  }

  function renderStep1() {
    const tplSel = h("select", null, templates.map((t, i) => h("option", { value: i, selected: i === v.tpl }, t.vendor)));
    tplSel.addEventListener("change", () => { v.tpl = +tplSel.value; test = null; render(); });
    const t = templates[v.tpl] || {};
    return [
      h("div", { class: "row", style: { marginBottom: "12px" } }, segmented([["discover", "Discover (ONVIF)"], ["template", "Vendor template"], ["manual", "Manual URL"]], v.method, m => { v.method = m; test = null; render(); })),
      v.method === "discover" ? renderDiscover() : null,
      v.method === "template" ? h("div", { class: "stack" },
        h("div", { class: "fields" },
          h("label", { class: "field" }, "Vendor", tplSel),
          h("label", { class: "field" }, "Host / IP", input("host", { placeholder: "192.168.1.64", autocomplete: "off" })),
          h("label", { class: "field" }, "Port", input("port", { type: "number", min: 1, max: 65535, step: 1, placeholder: String(t.port || 554) })),
          h("label", { class: "field" }, "Channel", input("channel", { type: "number", min: 1, step: 1 })),
          h("label", { class: "field" }, "Username", input("user", { autocomplete: "off" })),
          h("label", { class: "field" }, "Password", input("password", { type: "password", autocomplete: "new-password" }))),
        t.notes ? h("div", { class: "small mute" }, t.notes) : null,
        disclaimer ? h("div", { class: "small mute" }, disclaimer) : null,
        builtIn ? h("div", { class: "small mute" }, "Using built-in templates (server template list not available).") : null,
        h("div", { class: "row" }, h("b", { class: "small" }, "Stream"),
          segmented([["sub", "Sub-stream (recommended)"], ["main", "Main stream"]], v.stream, s2 => { v.stream = s2; test = null; preview(); })),
        h("div", { class: "note", style: { marginTop: 0 } }, "Analytics run at about 5 fps on a low-resolution frame, so the camera's sub-stream (e.g. 640×360) is enough and costs far less decode CPU/GPU and bandwidth. " +
          "The main stream is only worth it for evidence quality. With a sub-stream selected, the template's main-stream URL is saved as the camera's " +
          "main stream, which the optional restreamer uses for smooth WebRTC live view and full-resolution recording.")) : null,
      v.method === "manual" ? h("div", { class: "stack" },
        h("label", { class: "field" }, "Source", input("manual", { placeholder: "rtsp://user:pass@host:554/path  ·  /path/to/file.mp4  ·  0 (webcam)", autocomplete: "off" })),
        h("div", { class: "note", style: { marginTop: 0 } }, "Prefer the camera's sub-stream URL: analytics run at ~5 fps on a small frame.")) : null,
      v.method !== "discover" ? h("div", { class: "small", style: { marginTop: "12px" } }, h("span", { class: "mute" }, "Source URL: "), previewEl) : null,
    ];
  }

  function renderStep2() {
    const btn = h("button", { class: "primary", onclick: runTest }, "Test connection");
    let res = null;
    if (test === "running") res = h("div", { class: "row small mute" }, h("span", { class: "spinner" }), "Connecting and grabbing a frame…");
    else if (test === "missing") res = featureMissing("Connection testing");
    else if (test) res = test.ok
      ? h("div", { class: "stack" }, h("div", { class: "h-OK" }, `Connected · ${test.width ?? "?"}×${test.height ?? "?"}${test.fps ? ` @ ${num(+test.fps, 1)} fps` : ""}`),
        b64ok(test.snapshot_jpeg_b64) ? h("img", { class: "snap", alt: "Test snapshot", src: "data:image/jpeg;base64," + test.snapshot_jpeg_b64.replace(/\s/g, "") }) : null)
      : h("div", { class: "banner err" }, "Connection failed: ", test.error || "unknown error");
    return [h("div", { class: "small", style: { marginBottom: "10px" } }, h("span", { class: "mute" }, "Source: "), h("code", { class: "mono", style: { wordBreak: "break-all" } }, sourceMasked())),
      h("div", { class: "row" }, btn), h("div", { style: { marginTop: "12px" } }, res)];
  }
  async function runTest() {
    test = "running"; render();
    try {
      const r = await api("/api/cameras/test", { method: "POST", body: { source: source() }, feature: "camtest" });
      test = r && typeof r === "object" ? r : { ok: false, error: "unexpected response" };
    } catch (e) { test = e.kind === "missing" ? "missing" : { ok: false, error: e.message }; }
    render();
  }

  function renderStep3() {
    const auto = slug(v.name) || "";
    return [h("div", { class: "fields" },
      h("label", { class: "field" }, "Name", input("name", { placeholder: "Front entrance" })),
      h("label", { class: "field" }, "Id (optional)", input("id", { placeholder: auto || "auto", class: "mono" }))),
    h("label", { class: "row", style: { marginTop: "10px" } }, (() => { const c = h("input", { type: "checkbox", checked: v.enabled }); c.addEventListener("change", () => { v.enabled = c.checked; }); return c; })(), "Enabled (start analytics immediately)"),
    test && test !== "missing" && test !== "running" && !test.ok ? h("div", { class: "banner", style: { marginTop: "10px" } }, "The connection test failed; you can still save and fix the source later.") : null,
    h("div", { class: "note" }, "Next: draw zones on the snapshot and calibrate the ground plane. Until calibrated, positions and speeds are not meaningful.")];
  }

  async function save() {
    if (!v.name.trim()) { toast("Give the camera a name.", "warn"); return false; }
    const bodyObj = { name: v.name.trim(), source: source(), enabled: v.enabled };
    const tpl = templates[v.tpl];
    if (v.method === "template" && v.stream === "sub" && tpl?.main && tpl.main !== tpl.sub) {
      bodyObj.main_source = fillTemplate(tpl.main, v, false, tpl.port);   // restreamer: WebRTC live + main-stream recording
    }
    const id = slug(v.id || "");
    if (id) bodyObj.id = id;
    try {
      const r = await api("/api/cameras", { method: "POST", body: bodyObj, feature: "cameras-write" });
      toast(`Camera "${bodyObj.name}" added.`, "ok");
      await loadCameras(); loadConfig().catch(() => {});
      const newId = r?.id || id;
      if (newId && can("zone.edit")) location.hash = `#/cameras/${enc(newId)}/zones`;
      return true;
    } catch (e) {
      if (e instanceof ApiError && e.kind === "missing") toast("Adding cameras is not available on this server.", "warn");
      return false;
    }
  }

  function render() {
    replace(body, stepper(), step === 1 ? renderStep1() : step === 2 ? renderStep2() : renderStep3());
    preview();
    renderActions();
  }
  function renderActions() {
    const valid1 = v.method === "manual" ? !!v.manual.trim() : v.method === "template" ? !!v.host.trim() : false;
    dlg.setActions([
      { label: "Cancel" },
      step > 1 ? { label: "Back", onclick: () => { step--; render(); return false; } } : null,
      step < 3 ? { label: "Next", primary: true, disabled: step === 1 && !valid1, onclick: () => { step++; render(); return false; } } : null,
      step === 3 ? { label: "Save camera", primary: true, onclick: save } : null,
    ].filter(Boolean));
  }
  render();
}

// ---------------- edit ----------------
function editCamera(c) {
  const name = h("input", { value: c.name });
  const src = h("input", { placeholder: "Leave blank to keep the current source", autocomplete: "off" });
  const en = h("input", { type: "checkbox", checked: c.enabled !== false });
  const main = h("input", { placeholder: c.main_source ? "Leave blank to keep the current main stream" : "rtsp://user:pass@host:554/main-stream-path (optional)", autocomplete: "off" });
  const clearMain = h("input", { type: "checkbox" });
  openDialog({
    title: `Edit ${c.name}`,
    body: h("div", { class: "stack" },
      h("label", { class: "field" }, "Name", name),
      h("label", { class: "field" }, "Source", src, h("span", { class: "small mute" }, "Current (password redacted): ", h("code", { class: "mono", style: { wordBreak: "break-all" } }, c.source || "—"))),
      h("label", { class: "field" }, "Main stream (optional)", main,
        h("span", { class: "small mute" }, "Full-resolution RTSP URL for smooth WebRTC live view and main-stream recording through the restreamer (off unless configured). Blank = the source above. ",
          c.main_source ? ["Current: ", h("code", { class: "mono", style: { wordBreak: "break-all" } }, c.main_source)] : null)),
      c.main_source ? h("label", { class: "row" }, clearMain, "Remove the main-stream URL (use the source)") : null,
      h("label", { class: "row" }, en, "Enabled")),
    actions: [{ label: "Cancel" }, { label: "Save", primary: true, onclick: async () => {
      const body = {};
      if (name.value.trim() && name.value.trim() !== c.name) body.name = name.value.trim();
      if (src.value.trim()) body.source = src.value.trim();
      if (clearMain.checked) body.main_source = "";
      else if (main.value.trim()) body.main_source = main.value.trim();
      if (en.checked !== (c.enabled !== false)) body.enabled = en.checked;
      if (!Object.keys(body).length) return true;
      try {
        await api(`/api/cameras/${enc(c.id)}`, { method: "PUT", body, feature: "cameras-write" });
        toast("Camera updated.", "ok"); await loadCameras(); loadConfig().catch(() => {});
        return true;
      } catch (e) { if (e.kind === "missing") toast("Editing cameras is not available on this server.", "warn"); return false; }
    } }],
  });
}

async function setEnabled(c, on, sw) {
  try {
    await api(`/api/cameras/${enc(c.id)}`, { method: "PUT", body: { enabled: on }, feature: "cameras-write" });
    toast(`${c.name} ${on ? "enabled" : "disabled"}.`, "ok");
    await loadCameras();
  } catch (e) {
    sw.checked = !on;
    if (e.kind === "missing") toast("Enabling/disabling cameras is not available on this server.", "warn");
  }
}

async function deleteCamera(c) {
  if (!await confirmDialog(`Delete ${c.name}?`, `This removes camera "${c.id}" and its zones from the site configuration. Existing alerts and recordings are kept.`, { danger: true, okLabel: "Delete" })) return;
  try {
    await api(`/api/cameras/${enc(c.id)}`, { method: "DELETE", feature: "cameras-write" });
    toast(`Deleted ${c.name}.`, "ok"); await loadCameras(); loadConfig().catch(() => {});
  } catch (e) { if (e.kind === "missing") toast("Deleting cameras is not available on this server.", "warn"); }
}

// ---------------- page ----------------
export default function createCameras() {
  const el = h("section", { class: "page" });
  let detail = null;   // {id, tab, comp}
  let shown = false;

  function renderList() {
    detail?.comp?.destroy?.(); detail = null;
    const edit = can("settings.edit");
    const rows = S.cams.map(c => {
      const sw = h("input", { type: "checkbox", checked: c.enabled !== false, disabled: !edit || !S.camsApi, "aria-label": `Enable ${c.name}` });
      sw.addEventListener("change", () => setEnabled(c, sw.checked, sw));
      const zonesN = (c.zones || []).length, masksN = (c.masks_px || []).length;
      const thumb = h("img", { class: "cam-thumb", alt: "", src: c.enabled === false ? BLANK : `/api/cameras/${enc(c.id)}/latest.jpg?t=${Math.floor(Date.now() / 30000)}`, loading: "lazy" });
      thumb.addEventListener("error", () => { thumb.src = BLANK; }, { once: true });
      return h("tr", null,
        h("td", null, h("div", { class: "cam-name" }, thumb, h("div", null, h("a", { href: `#/cameras/${enc(c.id)}` }, h("b", null, c.name)),
          h("div", { class: "small mute mono" }, c.id, isMono(c.id) ? [" · ", h("span", { class: "tag", title: "Monochrome / infrared picture" }, icon("ir"), "IR")] : null)))),
        h("td", null, statusCell(c)),
        h("td", { class: "hide-sm" }, c.source ? h("code", { class: "mono src", title: c.source }, c.source) : h("span", { class: "mute" }, "—")),
        h("td", { class: "nowrap tnum" }, c.width ? `${c.width}×${c.height}` : "—"),
        h("td", null, c.calibrated === false ? h("span", { class: "tag warn" }, "not calibrated") : c.calibration ? estBadge(`± ${c.calibration.calib_error_m} m`) : (c.calibrated ? "yes" : "—")),
        h("td", { class: "tnum" }, `${zonesN} / ${masksN}`),
        h("td", null, h("label", { class: "switch", title: S.camsApi ? "" : "Camera management API not available" }, sw, h("span"))),
        h("td", { class: "nowrap" }, h("div", { class: "row", style: { flexWrap: "nowrap", gap: "6px" } },
          h("a", { class: "btnlink sm", href: `#/cameras/${enc(c.id)}/zones`, title: "Zones & masks" }, "Zones"),
          h("a", { class: "btnlink sm", href: `#/cameras/${enc(c.id)}/calibration`, title: "Calibration & placement" }, "Calibrate"),
          edit && S.camsApi ? [h("button", { class: "sm icon ghost", title: "Edit", "aria-label": `Edit ${c.name}`, onclick: () => editCamera(c) }, icon("edit")),
            h("button", { class: "sm icon ghost danger", title: "Delete", "aria-label": `Delete ${c.name}`, onclick: () => deleteCamera(c) }, icon("trash"))] : null)));
    });
    replace(el,
      pageHeader({ title: "Cameras", subtitle: `${S.cams.length} configured`,
        actions: edit && S.camsApi ? h("button", { class: "primary", onclick: () => addCameraWizard() }, icon("plus"), "Add camera") : null }),
      S.camsApi ? null : featureMissing("Camera management (add / edit / delete)"),
      !edit ? h("div", { class: "banner info" }, icon("info"), "Read-only: changing cameras needs the settings.edit permission (admin).") : null,
      rows.length ? h("div", { class: "table-wrap panel flush" }, h("table", { class: "cam-table" },
        h("thead", null, h("tr", null, ["Camera", "Status", "Source", "Resolution", "Calibration", "Zones / masks", "Enabled", ""].map((t, i) => h("th", { class: i === 2 ? "hide-sm" : "" }, t)))),
        h("tbody", null, rows)))
        : emptyState({ icon: "cameras", title: "No cameras yet", text: "Add an RTSP camera, an NVR channel, a video file or a webcam. Use the camera's sub-stream for analytics.",
          action: edit && S.camsApi ? h("button", { class: "primary", onclick: () => addCameraWizard() }, icon("plus"), "Add camera") : null }),
      h("div", { class: "note" }, "Sources are shown with passwords redacted by the server. Calibration error is an estimate; see each camera's calibration tab."));
  }

  function renderDetail(id, tab) {
    const c = camById(id);
    if (!c) { replace(el, h("div", { class: "banner err" }, `Unknown camera "${id}". `, h("a", { href: "#/cameras" }, "Back to cameras"))); detail = null; return; }
    tab = tab === "calibration" ? "calibration" : "zones";
    if (detail?.id === id && detail.tab === tab && detail.comp) return;   // keep editor state
    detail?.comp?.destroy?.();
    const comp = tab === "zones" ? createZoneEditor(c) : createCalibPanel(c);
    detail = { id, tab, comp };
    replace(el,
      pageHeader({ title: c.name, crumbs: [{ label: "Cameras", href: "#/cameras" }, { label: c.name }], back: "#/cameras",
        subtitle: h("span", { class: "mono" }, c.id), actions: h("a", { class: "btnlink", href: `#/live/${enc(id)}` }, icon("live"), "Live view") }),
      h("div", { class: "tabs" },
        h("a", { href: `#/cameras/${enc(id)}/zones`, class: tab === "zones" ? "on" : "" }, "Zones & masks"),
        h("a", { href: `#/cameras/${enc(id)}/calibration`, class: tab === "calibration" ? "on" : "" }, "Calibration & placement")),
      !S.camsApi && tab === "zones" ? h("div", { class: "banner" }, "Camera management API not available: zones below come from the site config (ground metres) and saving will not work until the backend provides it.") : null,
      comp.el);
  }

  on("cameras", () => { if (shown && !detail) renderList(); });
  on("tick", () => { if (shown && !detail) { /* cheap status refresh */ el.querySelectorAll("tbody tr").length && renderStatusOnly(); } });
  function renderStatusOnly() {
    const rows = el.querySelectorAll(".cam-table tbody tr");
    S.cams.forEach((c, i) => { const td = rows[i]?.children[1]; if (td) replace(td, statusCell(c)); });
  }

  return {
    el,
    async show(args) {
      shown = true;
      if (args[0]) renderDetail(args[0], args[1]);
      else { await loadCameras().catch(() => {}); if (shown && !args[0]) renderList(); }
    },
    hide() { shown = false; detail?.comp?.destroy?.(); detail = null; },
    onKey(e) { return detail?.comp?.onKey?.(e) || false; },
    onVisibility(v) { detail?.comp?.onVisibility?.(v); },
    dirty() { return !!detail?.comp?.dirty?.(); },
    canLeave(name, args) {
      if (!detail?.comp?.dirty?.()) return true;
      if (name === "cameras" && args[0] === detail.id && (args[1] || "zones") === detail.tab) return true;
      return confirmDialog("Discard changes?", "You have unsaved zone/mask changes on this camera.", { danger: true, okLabel: "Discard" });
    },
  };
}
