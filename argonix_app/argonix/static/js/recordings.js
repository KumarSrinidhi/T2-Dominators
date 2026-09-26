// Recordings: segment list per camera and stream with inline player.
//   analytics = the annotated analytics-resolution stream Argus writes itself (60 s segments)
//   main      = the camera's full-resolution main stream recorded by the MediaMTX restreamer (fMP4), when enabled
// Deep link (used by the alert drawer's "open main-stream recording at this time"):
//   #/recordings?cam=<camera id>&t=<epoch s>[&stream=main]  -> finds the segment and seeks to t minus 4 s pre-roll
import { api, safe } from "./api.js";
import { S, camName, can } from "./state.js";
import { h, replace, enc, fmtDateTime, fmtDur, icon } from "./util.js";
import { pageHeader, emptyState, segmented, toast } from "./ui.js";

const PAGE = 50;
const STREAM_LABEL = { main: "Main stream", analytics: "Analytics" };
const STREAM_TITLE = {
  main: "The camera's own full-resolution stream, recorded by the MediaMTX restreamer (no overlay)",
  analytics: "Annotated analytics-resolution stream (boxes and zones burned in), ~5 fps",
};

function fmtSize(b) {
  if (!b && b !== 0) return "";
  return b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(1)} GB` : b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} kB`;
}

export default function createRecordings() {
  let cam = "", stream = "", rows = [], limit = PAGE, playing = null, seekTo = null;
  const camSel = h("select", { "aria-label": "Camera" });
  camSel.addEventListener("change", () => { cam = camSel.value; limit = PAGE; load(); });
  const streamSel = segmented([["", "All", "Both streams"], ["main", "Main", STREAM_TITLE.main], ["analytics", "Analytics", STREAM_TITLE.analytics]],
    stream, v => { stream = v; limit = PAGE; load(); }, { label: "Stream" });
  const player = h("video", { controls: true, hidden: true, playsinline: true, muted: true, style: { width: "100%", maxWidth: "960px", background: "var(--video)", borderRadius: "8px", display: "block" } });
  player.addEventListener("loadedmetadata", () => {
    if (seekTo != null) { try { player.currentTime = seekTo; } catch { /* not seekable yet */ } seekTo = null; }
  });
  const nowPlaying = h("div", { class: "small mute", style: { margin: "6px 0 12px" } });
  const tbody = h("tbody");
  const more = h("button", { onclick: () => { limit += PAGE; render(); } }, "Load more");
  const note = h("div", { class: "note" });
  const el = h("section", { class: "page" },
    pageHeader({ title: "Recordings", subtitle: "Analytics-stream segments and, when the restreamer records, the cameras' main streams",
      actions: [streamSel, camSel, h("button", { onclick: () => load() }, icon("refresh"), "Refresh")] }),
    player, nowPlaying,
    h("div", { class: "table-wrap panel flush" }, h("table", null, h("thead", null, h("tr", null, ["Camera", "Stream", "Start", "Duration", ""].map(t => h("th", null, t)))), tbody)),
    h("div", { class: "more" }, more),
    note);

  function fillCams() {
    replace(camSel, h("option", { value: "" }, "All cameras"), S.cams.map(c => h("option", { value: c.id, selected: c.id === cam }, c.name)));
  }
  function play(r, offset = 0) {
    playing = r.url;
    seekTo = offset > 0 ? offset : null;
    player.hidden = false; player.src = r.url;
    player.play().catch(() => {});
    nowPlaying.textContent = `${camName(r.camera)} · ${STREAM_LABEL[r.stream] || r.stream} · ${fmtDateTime(r.start)} · ${fmtDur(r.end - r.start)}`
      + (offset > 0 ? ` · from ${fmtDateTime(r.start + offset)}` : "");
    render();
    player.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  function render() {
    const vis = rows.slice(0, limit);
    let day = "";
    const out = [];
    for (const r of vis) {
      const d = new Date(r.start * 1000).toLocaleDateString();
      if (d !== day) { day = d; out.push(h("tr", null, h("td", { colspan: 5, class: "small mute", style: { background: "var(--panel2)" } }, d))); }
      out.push(h("tr", { style: playing === r.url ? { background: "color-mix(in srgb, var(--acc) 12%, transparent)" } : null },
        h("td", null, camName(r.camera)),
        h("td", null, h("span", { class: "tag", title: STREAM_TITLE[r.stream] || "" }, STREAM_LABEL[r.stream] || r.stream),
          r.size ? h("span", { class: "small mute" }, " " + fmtSize(r.size)) : null),
        h("td", null, new Date(r.start * 1000).toLocaleTimeString()), h("td", null, fmtDur(r.end - r.start)),
        h("td", { class: "row", style: { justifyContent: "flex-end" } },
          h("button", { class: "sm", onclick: () => play(r) }, icon("play"), "Play"),
          h("a", { class: "btnlink sm icon", href: r.url, download: "", title: "Download segment", "aria-label": "Download segment" }, icon("download")))));
    }
    replace(tbody, out.length ? out : h("tr", null, h("td", { colspan: 5 }, emptyState({ icon: "recordings", title: "No segments yet",
      text: stream === "main" ? "Main-stream recording needs the restreamer (config \"restream\" with record.enabled). Segments are indexed about every 30 s."
        : "Segments close every 60 s; the first one appears a minute after a camera starts.", compact: true }))));
    more.hidden = rows.length <= limit;
    const nMain = rows.filter(r => r.stream === "main").length;
    note.textContent = "Analytics segments are the annotated analytics-resolution stream, closed every 60 s. "
      + (nMain ? "Main-stream segments are the camera's own full-resolution video recorded by MediaMTX (fMP4); the newest one may still be growing. "
        : "Main-stream recording is off unless the restreamer is enabled with recording (see README, \"Smooth live view and main-stream recording\"). ")
      + "Times are this server's clock.";
  }
  const load = safe(async () => {
    const q = new URLSearchParams();
    if (cam) q.set("camera", cam);
    if (stream) q.set("stream", stream);
    rows = await api("/api/recordings" + (q.toString() ? "?" + q : ""));
    render();
  });
  const openAt = safe(async (camId, t, st) => {
    let r;
    try {
      r = await api(`/api/recordings/at?camera=${enc(camId)}&t=${enc(t)}&stream=${enc(st)}`, { silent: true });
    } catch (e) {
      toast(`No ${STREAM_LABEL[st]?.toLowerCase() || st} recording of ${camName(camId)} covers ${fmtDateTime(+t)}.`, "warn");
      return;
    }
    cam = camId; stream = st; streamSel.set(st); fillCams();
    await load();
    play(r, r.offset_s || 0);
  });

  return {
    el,
    show(args, query) {
      if (!can("evidence.view")) { replace(el, pageHeader({ title: "Recordings" }), emptyState({ icon: "recordings", title: "Not permitted", text: "Your role cannot view recordings (needs evidence.view)." })); return; }
      fillCams();
      const t = query?.get?.("t"), qc = query?.get?.("cam");
      if (t && qc) openAt(qc, t, query.get("stream") === "analytics" ? "analytics" : "main");
      else load();
    },
    hide() { player.pause(); },
  };
}
