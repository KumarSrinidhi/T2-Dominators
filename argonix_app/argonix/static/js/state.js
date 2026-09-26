// Shared client state + a tiny event bus.
import { api, optional, missing } from "./api.js";
import { CAMCOL, RULE } from "./util.js";

export const S = {
  me: { username: "", roles: [], permissions: [], open_mode: false },
  auth: null,           // GET /api/auth/status (null on servers without it)
  cfg: { cameras: [], map: {}, site: "Argus" },
  cams: [],             // merged camera list: /api/config (+ /api/cameras when available)
  camsApi: false,       // true when GET /api/cameras exists
  stats: {},            // per-camera stats from the WebSocket tick
  tracks: [],           // live tracks (each camera's own ground frame)
  ws: "connecting",
  vlms: [],
  map: null,            // /api/map payload or null when unavailable
};

const bus = new EventTarget();
export function on(name, fn) {
  const l = e => fn(e.detail);
  bus.addEventListener(name, l);
  return () => bus.removeEventListener(name, l);
}
export function emit(name, detail) { bus.dispatchEvent(new CustomEvent(name, { detail })); }

export const can = p => !!S.me?.permissions?.includes(p);

export function camById(id) { return S.cams.find(c => c.id === id); }
export function camName(id) {
  // non-video sources (access control) put "sensor:<source id>" where a camera id would be
  if (typeof id === "string" && id.startsWith("sensor:")) return `Access control · ${id.slice(7)}`;
  return camById(id)?.name || id;
}
export const isSensorEvent = e => (e?.derivation?.modality || e?.modality) === "access_control" || String(e?.camera_id || "").startsWith("sensor:");
export function camColor(id) {
  const i = S.cams.findIndex(c => c.id === id);
  return CAMCOL[(i < 0 ? 0 : i) % CAMCOL.length];
}

/** Monochrome / IR picture, when the worker reports it (stats.monochrome or stats.health_metrics.monochrome). */
export function isMono(camId) {
  const s = S.stats[camId];
  return !!(s && (s.monochrome === true || s.health_metrics?.monochrome === true || s.color_mode === "mono" || s.ir === true));
}

/** Coarse camera state for dots/badges: ok | warn | err | off | wait, with a short label. */
export function camState(camId) {
  const c = camById(camId), s = S.stats[camId];
  if (c?.enabled === false) return { state: "off", label: "Disabled" };
  if (!s) return { state: "wait", label: "No stats yet" };
  if (s.status === "error") return { state: "err", label: "Worker error" };
  if (s.status && s.status !== "running") return { state: s.status === "reconnecting" ? "warn" : "wait", label: s.status[0].toUpperCase() + s.status.slice(1) };
  if (s.health === "DOWN") return { state: "err", label: "View down" };
  if (s.health === "DEGRADED") return { state: "warn", label: "Degraded" + ((s.health_reasons || []).length ? `: ${s.health_reasons.join(", ")}` : "") };
  if (s.health === "WARMUP") return { state: "wait", label: "Warming up" };
  return { state: "ok", label: "Online" };
}

/** Normalise a zone's rules to {rule: params}. /api/config returns a list of rule names. */
export function normRules(r) {
  if (Array.isArray(r)) return Object.fromEntries(r.map(k => [k, {}]));
  return r && typeof r === "object" ? r : {};
}

export async function loadConfig() {
  S.cfg = await api("/api/config");
  Object.assign(RULE, S.cfg.rule_labels || {});  // labels for analyzer-registered event types
  await loadCameras();
  return S.cfg;
}

/** Merge the static config view with the camera-management API when it exists. */
export async function loadCameras() {
  let managed = null;
  try { managed = await optional("/api/cameras", "cameras", { silent: true }); }
  catch { managed = null; }
  S.camsApi = Array.isArray(managed);
  const cfgCams = S.cfg?.cameras || [];
  const byId = new Map(cfgCams.map(c => [c.id, { ...c, enabled: true, fromConfig: true }]));
  if (S.camsApi) {
    for (const m of managed) {
      const base = byId.get(m.id) || { calibration: null, zones: [] };
      byId.set(m.id, { ...base, ...m, cfgZones: base.zones, zones: Array.isArray(m.zones) ? m.zones : base.zones, fromConfig: false });
    }
    // cameras deleted via the API disappear from the managed list
    for (const id of [...byId.keys()]) if (!managed.some(m => m.id === id)) byId.delete(id);
  }
  S.cams = [...byId.values()];
  emit("cameras", S.cams);
  return S.cams;
}

export async function loadMap() {
  try { S.map = await optional("/api/map", "map", { silent: true }); }
  catch { S.map = null; }
  emit("map", S.map);
  return S.map;
}

export async function loadVlms() {
  if (!S.cfg?.explain_enabled) { S.vlms = []; return; }
  try { S.vlms = (await api("/api/vlm", { silent: true })).models || []; } catch { S.vlms = []; }
}

export function featureMissing(key) { return missing.has(key); }

/** Camera placement for site-frame transforms (identity when /api/map is unavailable). */
export function placement(camId) {
  const c = S.map?.cameras?.find(x => x.id === camId);
  const p = c?.placement || {};
  return { offset: p.offset_m || [0, 0], rot: +p.rotation_deg || 0 };
}
export function toSite(camId, x, y) {
  const { offset, rot } = placement(camId);
  const r = rot * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
  return [c * x - s * y + (+offset[0] || 0), s * x + c * y + (+offset[1] || 0)];
}
