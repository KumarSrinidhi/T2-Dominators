// System page "Compute nodes" card: the inference node (remote GPU node or in-process detector) and the VLM node,
// from GET /api/nodes. Degrades to a note when the endpoint is missing, and shows each node's own error when it is
// unreachable. All server strings go through text nodes.
import { optional } from "./api.js";
import { camName } from "./state.js";
import { h, replace, num } from "./util.js";
import { kv } from "./ui.js";

const STATE_CLASS = { up: "h-OK", fallback: "h-DEGRADED", connecting: "h-WARMUP", configured: "h-WARMUP" };
const stateCls = s => STATE_CLASS[s] || "h-DOWN";
const ms = v => (v == null ? "—" : `${num(v, 1)} ms`);

function inferenceRows(n) {
  const m = n.model || {};
  const rows = [["Model", `${m.detector ?? "?"} @${m.imgsz ?? "?"} · ${m.node_runtime || m.runtime || "?"} · ${m.device ?? "?"}${m.fp16 ? " fp16" : ""}`]];
  if (n.location === "remote") {
    const g = n.gpu || {};
    rows.push(
      ["GPU", g.available ? `${g.name ?? "GPU"} · ${num(g.util_pct, 0)} % util · ${g.mem_used_mb ?? "?"} / ${g.mem_total_mb ?? "?"} MiB (all tenants)` : "not reported"],
      ["Node latency", n.latency_ms ? `p50 ${ms(n.latency_ms.p50)} · p95 ${ms(n.latency_ms.p95)} (frame received → result, last 60 s)` : "—"],
      ["Load", `${num(n.fps_in, 1)} fps in · mean batch ${num(n.mean_batch, 2)} · queue ${n.queue_depth ?? "—"} · ${n.clients ?? 0} camera stream(s)`],
      ["Probe", `${ms(n.probe_ms)} from this server${n.uptime_s != null ? ` · node up ${Math.round(n.uptime_s / 60)} min` : ""}`]);
  }
  return rows;
}

function cameraTable(n) {
  const cams = n.cameras || [];
  if (n.location !== "remote" || !cams.length) return null;
  return h("div", { class: "table-wrap" }, h("table", null,
    h("thead", null, h("tr", null, ["Camera", "Link", "RTT (EMA / p95)", "Node ms", "KB / frame", "Dropped", "Timeouts", "Fallback frames"].map(t => h("th", null, t)))),
    h("tbody", null, cams.map(c => h("tr", null,
      h("td", null, camName(c.camera)),
      h("td", { class: stateCls(c.remote_state) }, c.remote_state || "—"),
      h("td", null, `${ms(c.rtt_ms)} / ${ms(c.rtt_p95_ms)}`), h("td", null, ms(c.node_ms)), h("td", null, num(c.kb_per_frame, 1)),
      h("td", null, c.dropped_remote ?? 0), h("td", null, c.remote_timeouts ?? 0), h("td", null, c.fallback_frames ?? 0))))));
}

function nodeCard(n) {
  const title = n.kind === "vlm" ? "VLM explainer" : n.location === "remote" ? "Inference node" : "Inference (in-process)";
  const head = h("div", { class: "row", style: { gap: "8px", alignItems: "baseline", flexWrap: "wrap" } },
    h("b", null, title), n.name ? h("span", { class: "mono small" }, n.name) : null,
    h("span", { class: "tag " + stateCls(n.state) }, n.state || "?"),
    n.url ? h("span", { class: "mono small mute" }, n.url) : null);
  let body;
  if (n.error) body = h("div", { class: "small h-DOWN" }, n.error);
  if (n.kind === "inference" && n.state === "up") body = [kv(inferenceRows(n)), cameraTable(n)];
  else if (n.kind === "vlm" && n.health) body = h("pre", { class: "mono mute small", style: { whiteSpace: "pre-wrap", margin: "4px 0 0" } }, JSON.stringify(n.health, null, 2));
  return h("div", { class: "stat", style: { marginBottom: "8px" } }, head, body);
}

export function createNodesCard() {
  const el = h("div", null, h("div", { class: "mute small" }, "Loading…"));
  let last = 0, busy = false;

  async function refresh(force = false) {
    if (busy || (!force && Date.now() - last < 5000)) return;
    busy = true; last = Date.now();
    try {
      const r = await optional("/api/nodes", "nodes", { silent: true });
      if (r === null) { replace(el, h("div", { class: "mute small" }, "Not available on this server.")); return; }
      const nodes = r.nodes || [];
      replace(el, nodes.length ? nodes.map(nodeCard) : h("div", { class: "mute small" }, "No compute nodes reported."),
        h("div", { class: "note" }, "Remote inference sends each camera's analytics frames (JPEG) to the GPU node and gets detections + track IDs back. RTT is measured per frame on this server; a node that stays down hands cameras to the local CPU fallback if one is configured."));
    } catch (e) {
      replace(el, h("div", { class: "banner err" }, `Node status unavailable (${e.message}).`));
    } finally { busy = false; }
  }

  return { el, refresh };
}
