// Page registry: the shell's plug-in API. A page is a factory returning {el, show(args, query), hide(), ...}
// (the same contract live.js / review.js use). Registering a page adds a route (#/<id>) and a sidebar item that
// is shown only when the user has `perm` and, if `probe` is given, the backend actually serves that endpoint.
//
//   import { registerPage } from "./registry.js";          // or from "./app.js" (re-exported)
//   import incidents from "./incidents.js";
//   registerPage({ id: "incidents", label: "Incidents", icon: "incidents", group: "monitor",
//                  perm: "alert.read", probe: "/api/incidents", module: incidents });
//
// Fields: id (route, [a-z0-9_-]), label, icon (a util.js icon name or an <svg> element), group ("monitor" |
// "configure" | any custom label), perm (permission string, array = any-of, or () => boolean), probe (API path or
// array of paths that must exist; checked against /openapi.json), module (page factory, or the module namespace
// with a default export), load (() => import("./x.js") for lazy pages), order (sort key), keywords (palette search).
import { S, can } from "./state.js";

export const GROUPS = [
  { id: "monitor", label: "Monitor" },
  { id: "configure", label: "Configure" },
];

// Defaults for known page ids (so a one-line `PAGES` addition still gets a good label, icon and position).
const KNOWN = {
  overview:    { label: "Overview",    icon: "overview",    group: "monitor",   order: 10 },
  live:        { label: "Live",        icon: "live",        group: "monitor",   order: 20 },
  map:         { label: "Map",         icon: "map",         group: "monitor",   order: 30 },
  review:      { label: "Review",      icon: "review",      group: "monitor",   order: 40 },
  incidents:   { label: "Incidents",   icon: "incidents",   group: "monitor",   order: 50 },
  analytics:   { label: "Analytics",   icon: "analytics",   group: "monitor",   order: 60 },
  journeys:    { label: "Journeys",    icon: "journeys",    group: "monitor",   order: 70 },
  investigate: { label: "Investigate", icon: "investigate", group: "monitor",   order: 80 },
  recordings:  { label: "Recordings",  icon: "recordings",  group: "monitor",   order: 90 },
  cameras:     { label: "Cameras",     icon: "cameras",     group: "configure", order: 110 },
  users:       { label: "Users",       icon: "users",       group: "configure", order: 120 },
  settings:    { label: "Settings",    icon: "settings",    group: "configure", order: 125 },
  nodes:       { label: "Compute",     icon: "nodes",       group: "configure", order: 128 },
  system:      { label: "System",      icon: "system",      group: "configure", order: 130 },
};

const pages = new Map();
const listeners = new Set();
let seq = 0;
let endpoints = null;       // Set of API paths from /openapi.json, or null when unknown (then probes pass)

const titleCase = id => String(id).replace(/[_-]+/g, " ").replace(/\b\w/g, c => c.toUpperCase());

function notify() { for (const fn of listeners) { try { fn(); } catch (e) { console.error(e); } } }

/** Register (or update) a page. Later calls for the same id merge over earlier ones. Returns the entry. */
export function registerPage(def = {}) {
  const id = String(def.id || "");
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(id)) throw new Error(`registerPage: invalid id "${id}"`);
  const prev = pages.get(id) || { id, seq: seq++, explicit: new Set() };
  const k = KNOWN[id] || {};
  let factory = def.module;
  if (factory && typeof factory === "object" && typeof factory.default === "function") factory = factory.default;
  const e = { ...prev };
  const set = (key, val, explicit = true) => {
    if (val === undefined || val === null || val === "") return;
    if (!explicit && prev.explicit.has(key)) return;        // never let a fallback override an explicit value
    e[key] = val;
    if (explicit) e.explicit = new Set([...(e.explicit || []), key]);
  };
  const explicit = def._fallback !== true;
  set("label", def.label, explicit); set("icon", def.icon, explicit); set("group", def.group, explicit);
  set("perm", def.perm, explicit); set("probe", def.probe, explicit); set("order", def.order, explicit);
  set("keywords", def.keywords, explicit); set("hidden", def.hidden, explicit);
  if (typeof factory === "function") e.factory = factory;
  if (typeof def.load === "function") e.load = def.load;
  e.label ??= k.label || titleCase(id);
  e.icon ??= k.icon || "overview";
  e.group ??= k.group || "monitor";
  e.order ??= k.order ?? 400 + e.seq;
  pages.set(id, e);
  notify();
  return e;
}

export function getPage(id) { return pages.get(id) || null; }
export function hasPage(id) { const p = pages.get(id); return !!(p && (p.factory || p.load)); }

/** All routable pages, sorted by group then order. */
export function listPages() {
  const gi = g => { const i = GROUPS.findIndex(x => x.id === g); return i < 0 ? GROUPS.length : i; };
  return [...pages.values()].filter(p => p.factory || p.load)
    .sort((a, b) => gi(a.group) - gi(b.group) || a.order - b.order || a.seq - b.seq);
}

export function permitted(p) {
  if (!p) return false;
  const perm = p.perm;
  if (!perm) return true;
  if (typeof perm === "function") { try { return !!perm(S.me); } catch { return false; } }
  if (Array.isArray(perm)) return perm.some(x => can(x));
  return can(perm);
}

export function probed(p) {
  if (!p?.probe) return true;
  if (typeof p.probe === "function") { try { return !!p.probe(); } catch { return false; } }
  if (!endpoints) return true;
  return [].concat(p.probe).every(hasEndpoint);
}

/** Visible in navigation: routable, permitted, backend present, not hidden. */
export function visiblePages() { return listPages().filter(p => !p.hidden && permitted(p) && probed(p)); }

export function onPagesChanged(fn) { listeners.add(fn); return () => listeners.delete(fn); }

// ---------------------------------------------------------------------------------------------------------------
// Endpoint discovery (feature probe without 404 noise): FastAPI publishes its routes at /openapi.json.
/** Load the route table once. Resolves to the Set of paths, or null when the schema is unavailable. */
export async function probeEndpoints() {
  try {
    const r = await fetch("/openapi.json", { credentials: "same-origin" });
    if (!r.ok) { endpoints = null; return null; }
    const j = await r.json();
    endpoints = new Set(Object.keys(j?.paths || {}));
  } catch { endpoints = null; }
  notify();
  return endpoints;
}

/** true / false when the route table is known; null when it is not (callers should then try the call). */
export function hasEndpoint(path) {
  if (!endpoints) return null;
  if (endpoints.has(path)) return true;
  // match templated routes: /api/cameras/{cid}/zones  vs  /api/cameras/cam1/zones
  const parts = path.split("?")[0].split("/");
  for (const ep of endpoints) {
    const q = ep.split("/");
    if (q.length !== parts.length) continue;
    if (q.every((seg, i) => seg === parts[i] || /^\{.+\}$/.test(seg))) return true;
  }
  return false;
}
