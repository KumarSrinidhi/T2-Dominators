// Shared bits for the Analytics and Journeys pages: panel headers, the "sample data" banner, feature-off empty states
// and the rule for when sample data may replace a missing/empty feature (demo mode, or an explicit preview).
import { h, icon, CAMCOL } from "../util.js";
import { emptyState } from "../ui.js";
import { isDemo } from "../shell.js";
import { S, camColor } from "../state.js";

/** Camera colour: the site's own palette slot, or the next free slot for cameras not in the config (sample data). */
const extra = new Map();
export function colorOf(id) {
  if (S.cams.some(c => c.id === id)) return camColor(id);
  if (!extra.has(id)) extra.set(id, extra.size);
  return CAMCOL[(S.cams.length + extra.get(id)) % CAMCOL.length];
}

export function panelHead(ic, title, sub, ...extra) {
  return h("div", { class: "panel-h" }, h("h3", null, icon(ic), title), sub ? h("span", { class: "sub" }, sub) : null,
    h("span", { class: "grow" }), extra);
}

/** Sample data may stand in for a feature that is absent or empty: in demo mode, or after "Preview with sample data". */
export function sampleAllowed(preview) { return isDemo() || !!preview; }

export function sampleBanner(onExit) {
  return h("div", { class: "banner info px-sample", role: "note" }, icon("present"),
    h("div", { class: "grow" }, h("b", null, "Sample data. "),
      "The backend feature is off or has no data yet, so this page shows illustrative values generated in the browser. ",
      "Nothing here was measured on this site."),
    onExit ? h("button", { class: "sm", type: "button", onclick: onExit }, "Hide sample") : null);
}

/**
 * Empty state for a feature that is disabled or has no data. `steps` explain how to turn it on; `onPreview` adds a
 * "Preview with sample data" button (omitted in demo mode, where samples show automatically).
 */
export function featureOff({ ic = "info", title, text, steps = [], action, onPreview, compact = false }) {
  return emptyState({ icon: ic, title, compact,
    text: h("div", null, text ? h("div", null, text) : null,
      steps.length ? h("ol", { class: "px-steps" }, steps.map(s => h("li", null, s))) : null),
    action: [action || null,
      onPreview && !isDemo() ? h("button", { class: "sm", type: "button", onclick: onPreview }, icon("present"), "Preview with sample data") : null] });
}

export const code = t => h("code", null, t);
