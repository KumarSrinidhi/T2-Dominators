// Toasts and modal dialogs.
import { $, h, icon, replace, append } from "./util.js";

const recent = new Map(); // dedupe key -> time
const MAX_TOASTS = 5;

/**
 * toast("text" | Node, kind = info|ok|warn|error, {key, timeout, onclick})
 * Toasts stack bottom-left (away from the right-hand alert drawer and its buttons).
 */
export function toast(msg, kind = "info", opts = {}) {
  const box = $("#toasts");
  if (!box) return null;
  const key = opts.key ?? (typeof msg === "string" ? kind + msg : null);
  const now = Date.now();
  if (key && recent.has(key) && now - recent.get(key) < (opts.dedupeMs ?? 5000)) return null;
  if (key) recent.set(key, now);
  const ic = { error: "alert", warn: "alert", ok: "check", info: "info" }[kind] || "info";
  const el = h("div", { class: ["toast", kind, opts.onclick && "click"], role: kind === "error" ? "alert" : "status" },
    h("span", { class: "toast-ic" }, icon(opts.icon || ic)),
    h("div", { class: "grow" }, msg),
    h("button", { class: "x", title: "Dismiss", "aria-label": "Dismiss", onclick: e => { e.stopPropagation(); el.remove(); } }, icon("x")));
  if (opts.onclick) el.addEventListener("click", () => { opts.onclick(); el.remove(); });
  box.prepend(el);
  while (box.children.length > MAX_TOASTS) box.lastElementChild.remove();
  const timeout = opts.timeout ?? (kind === "error" ? 9000 : 6000);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

/**
 * Modal dialog built on <dialog>. body: Node; actions: [{label, primary, danger, onclick -> false keeps open}].
 * Returns {close, el, body, footer}.
 */
export function openDialog({ title, body, actions = [], wide = false, onclose }) {
  const dlg = $("#dialog");
  if (dlg.open) dlg.close();
  const footer = h("div", { class: "dlg-f" });
  const handle = { el: dlg, footer, close: () => { if (dlg.open) dlg.close(); } };
  const setActions = acts => {
    replace(footer, acts.map(a => h("button", {
      class: [a.primary && "primary", a.danger && "danger"], disabled: a.disabled, type: "button",
      onclick: async () => {
        try { if ((await a.onclick?.()) !== false) handle.close(); }
        catch (e) { if (!e?.kind) toast(String(e?.message || e), "error"); }
      },
    }, a.label)));
    footer.hidden = !acts.length;
  };
  handle.setActions = setActions;
  replace(dlg,
    h("div", { class: "dlg-h" }, h("h2", { class: "grow" }, title),
      h("button", { class: "ghost icon", type: "button", "aria-label": "Close", onclick: () => handle.close() }, icon("x"))),
    h("div", { class: "dlg-b" }, body), footer);
  setActions(actions);
  dlg.style.width = wide ? "min(900px, calc(100vw - 24px))" : "";
  dlg.onclose = () => { onclose?.(); };
  dlg.showModal();
  return handle;
}

export function confirmDialog(title, message, { danger = false, okLabel = "Confirm" } = {}) {
  return new Promise(resolve => {
    let done = false;
    openDialog({
      title, body: h("p", { style: "margin:0" }, message),
      actions: [{ label: "Cancel", onclick: () => { done = true; resolve(false); } },
        { label: okLabel, primary: !danger, danger, onclick: () => { done = true; resolve(true); } }],
      onclose: () => { if (!done) resolve(false); },
    });
  });
}

export function isDialogOpen() { return !!$("#dialog")?.open; }

/** Segmented control. options: [[value, label, title?, iconName?], ...]; label may be "" for icon-only. */
export function segmented(options, value, onchange, { label: groupLabel } = {}) {
  const wrap = h("div", { class: "seg", role: "group", "aria-label": groupLabel || null });
  const render = v => {
    replace(wrap, options.map(([val, label, title, ic]) => h("button", {
      class: [val === v && "on", ic && !label && "icon"], type: "button", title: title || "", "aria-pressed": String(val === v),
      "aria-label": !label ? (title || String(val)) : null,
      onclick: () => { render(val); onchange(val); },
    }, ic ? icon(ic) : null, label)));
  };
  render(value);
  wrap.set = render;
  return wrap;
}

// ---------------------------------------------------------------------------------------------------------------
// Design-system helpers (see app.css). Every string goes through text nodes.

/** Page header: breadcrumb trail + title + optional subtitle and right-aligned actions. */
export function pageHeader({ title, crumbs = [], subtitle, actions, back } = {}) {
  return h("div", { class: "page-head" },
    h("div", { class: "ph-main" },
      crumbs.length ? h("nav", { class: "crumbs-sm", "aria-label": "Breadcrumb" },
        crumbs.map((c, i) => [i ? h("span", { class: "sep", "aria-hidden": "true" }, "/") : null,
          c.href ? h("a", { href: c.href }, c.label) : h("span", null, c.label)])) : null,
      h("div", { class: "ph-title" }, back ? h("a", { class: "btnlink icon ghost", href: back, "aria-label": "Back", title: "Back" }, icon("chevleft")) : null,
        h("h2", null, title), subtitle ? h("span", { class: "ph-sub" }, subtitle) : null)),
    actions ? h("div", { class: "ph-actions" }, actions) : null);
}

/** Empty state with icon, title, explanatory text and an optional action node. */
export function emptyState({ icon: ic = "info", title, text, action, compact = false } = {}) {
  return h("div", { class: ["empty", compact && "compact"], role: "status" },
    h("div", { class: "empty-ic" }, icon(ic)),
    title ? h("div", { class: "empty-t" }, title) : null,
    text ? h("div", { class: "empty-x" }, text) : null,
    action ? h("div", { class: "empty-a" }, action) : null);
}

/** Loading skeletons: kind = "line" | "tile" | "card" | "row" | "kpi". */
export function skeleton(kind = "line", n = 1) {
  return Array.from({ length: n }, () => h("div", { class: ["skel", "skel-" + kind], "aria-hidden": "true" }));
}

export const obsBadge = (text = "Observation") => h("span", { class: "tag obs", title: "Observed by sensors/detector: what was seen" }, icon("eye"), text);
export const infBadge = (text = "Inference") => h("span", { class: "tag inf", title: "Derived by rules or a model: an interpretation, not a fact" }, icon("spark"), text);
export const acsBadge = (text = "Access control") => h("span", { class: "tag acs", title: "Reported by an access-control device (door controller / alarm panel), not by video" }, icon("door"), text);
export const estBadge =(text = "estimate") => h("span", { class: "tag est", title: "Estimated value with stated uncertainty" }, text);

// Alerts scripted by a demo scenario (server: argonix/scenario.py) carry demo=true and a "[DEMO] " title prefix.
export const demoBadge = a => a?.demo ? h("span", { class: "tag demo", title: a.demo_note || "Scripted by a demo scenario: not observed" }, "DEMO") : null;
export const plainTitle = a => a?.demo && typeof a.title === "string" ? a.title.replace(/^\[DEMO\] /, "") : a?.title;

const PRIO_LABEL = { P1: "Critical", P2: "High", P3: "Medium", P4: "Low" };
export function prioChip(p, { long = false } = {}) {
  return h("span", { class: ["chip", p], title: `${p} · ${PRIO_LABEL[p] || ""} priority` }, long ? `${p} · ${PRIO_LABEL[p] || ""}` : p);
}
export function statusPill(st) {
  return h("span", { class: ["pill", "st-" + st] }, h("span", { class: "pd", "aria-hidden": "true" }), st);
}

/**
 * Lightweight popover anchored to an element. Closes on outside click, Esc, or close(). Returns {el, close}.
 * content: Node or (close) => Node.
 */
let openPop = null;
export function popover(anchor, content, { align = "end", className = "", onclose, label } = {}) {
  if (openPop) openPop.close();
  const el = h("div", { class: ["popover", className], role: "dialog", "aria-label": label || null });
  const handle = { el, close: () => {} };
  const onDoc = e => { if (!el.contains(e.target) && !anchor.contains(e.target)) handle.close(); };
  const onKey = e => { if (e.key === "Escape") { e.stopPropagation(); handle.close(); anchor.focus?.(); } };
  const place = () => {
    const r = anchor.getBoundingClientRect(), vw = window.innerWidth, w = el.offsetWidth;
    let left = align === "start" ? r.left : r.right - w;
    left = Math.max(8, Math.min(vw - w - 8, left));
    el.style.left = left + "px"; el.style.top = (r.bottom + 6) + "px";
  };
  handle.close = () => {
    if (!el.isConnected) return;
    el.remove(); document.removeEventListener("pointerdown", onDoc, true); document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", place);
    anchor.setAttribute("aria-expanded", "false");
    if (openPop === handle) openPop = null;
    onclose?.();
  };
  append(el, [typeof content === "function" ? content(handle.close) : content]);
  document.body.appendChild(el);
  anchor.setAttribute("aria-expanded", "true");
  place();
  setTimeout(() => { document.addEventListener("pointerdown", onDoc, true); }, 0);
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", place);
  openPop = handle;
  (el.querySelector("input, button, [tabindex='0'], a[href]"))?.focus({ preventScroll: true });
  return handle;
}
export function closePopover() { openPop?.close(); }

export function progress(frac) {
  const f = Math.max(0, Math.min(1, +frac || 0));
  return h("div", { class: "progress", role: "progressbar", "aria-valuenow": Math.round(f * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
    h("div", { style: { width: (f * 100).toFixed(1) + "%" } }));
}

export function kv(pairs) {
  return h("div", { class: "kv" }, pairs.filter(Boolean).flatMap(([k, v]) => [h("div", null, k), h("div", null, v)]));
}

export function featureMissing(what) {
  return h("div", { class: "banner" }, icon("info"), h("div", null, `${what} is not available on this server (endpoint missing). `,
    h("span", { class: "mute" }, "This part of the UI activates automatically once the backend provides it.")));
}
