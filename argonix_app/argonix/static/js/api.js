// Single fetch wrapper: 401 -> login overlay (with the server's reason, e.g. idle timeout), 403 -> "not permitted" toast, 404/405 on an unknown route ->
// "feature missing" (caller degrades), network errors -> one de-duplicated toast. Always throws ApiError on failure.
import { toast } from "./ui.js";
import { ls } from "./util.js";

export class ApiError extends Error {
  constructor(kind, status, message, detail = "") {
    super(message);
    this.kind = kind;       // network | auth | forbidden | missing | notfound | http | abort
    this.status = status;
    this.detail = detail;
  }
}

/** Feature keys whose endpoints returned 404/405 on this server. */
export const missing = new Set();
let authHandler = () => {};
export function onAuthRequired(fn) { authHandler = fn; }

export function operatorName() { return ls.get("operator", "") || "operator"; }

function routeMissing(status, detail) {
  // FastAPI's generic unmatched-route responses. Handler-raised 404s carry a specific detail (e.g. "unknown camera").
  return status === 405 || (status === 404 && (detail === "Not Found" || detail === ""));
}

/**
 * api(path, {method, body, raw, silent, feature, signal})
 *  - silent: suppress toasts (background polling); errors are still thrown.
 *  - feature: key recorded in `missing` when the route does not exist.
 */
export async function api(path, opts = {}) {
  const { method = "GET", body, raw = false, silent = false, feature = null, signal } = opts;
  // X-Argonix-CSRF: the server refuses cookie-authenticated state changes without Origin/Referer unless this is set.
  const headers = { "X-Operator": operatorName(), "X-Argonix-CSRF": "1" };
  const init = { method, headers, credentials: "same-origin", signal };
  if (body !== undefined) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  let r;
  try {
    r = await fetch(path, init);
  } catch (e) {
    if (e?.name === "AbortError") throw new ApiError("abort", 0, "aborted");
    if (!silent) toast("Network error: the Argus server is unreachable.", "error", { key: "net", dedupeMs: 10000 });
    throw new ApiError("network", 0, "network error");
  }
  if (r.ok) {
    if (feature) missing.delete(feature);
    if (raw) return r;
    const ct = r.headers.get("content-type") || "";
    return ct.includes("json") ? r.json() : r.text();
  }
  let detail = "";
  try {
    const t = await r.text();
    try {
      const j = JSON.parse(t);
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { detail = t.slice(0, 300); }
  } catch { /* body unreadable */ }
  if (r.status === 401) {
    authHandler(detail);
    throw new ApiError("auth", 401, "login required", detail);
  }
  if (r.status === 403) {
    if (!silent) toast(`Not permitted: ${detail || "your role does not allow this action"}.`, "warn", { key: "403:" + detail });
    throw new ApiError("forbidden", 403, detail || "forbidden", detail);
  }
  if (routeMissing(r.status, detail) && feature) {
    missing.add(feature);
    throw new ApiError("missing", r.status, "not available on this server", detail);
  }
  const msg = detail || r.statusText || `HTTP ${r.status}`;
  if (!silent) toast(`${method} ${path.split("?")[0]} failed (${r.status}): ${msg}`, "error");
  throw new ApiError(r.status === 404 ? "notfound" : "http", r.status, msg, detail);
}

/** Like api() but resolves to null when the endpoint does not exist (feature probe). Other errors still throw. */
export async function optional(path, feature, opts = {}) {
  try { return await api(path, { ...opts, feature }); }
  catch (e) { if (e.kind === "missing") return null; throw e; }
}

/** Wrap an async UI handler so failures never become unhandled rejections (ApiErrors are already toasted). */
export function safe(fn) {
  return async (...a) => {
    try { return await fn(...a); }
    catch (e) {
      if (e instanceof ApiError) { if (e.kind === "missing") toast("Not available on this server.", "warn", { key: "missing" }); return undefined; }
      console.error(e);
      toast(String(e?.message || e), "error");
      return undefined;
    }
  };
}
