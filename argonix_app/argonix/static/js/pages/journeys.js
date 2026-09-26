// Journeys page: cross-camera global IDs (G-n) from the Re-ID engine. Each ID lists its per-camera sightings on a
// timeline, the link confidence of every hop with the evidence behind it (topology prior, travel time, appearance,
// direction, top alternatives), a click-through to the evidence crops and to the alerts of that camera/time, and the
// journey's path on the site map when cameras are placed/calibrated.
//
// Global IDs are opaque, probabilistic and revisable estimates (AGENTS.md: cross-camera identity is research-grade):
// the UI says so everywhere a link is shown. Nothing here names or labels a person.
//
// Endpoints: GET /api/reid/tracks (alert.read), GET /api/reid/journey/{gid} (evidence.view), GET /api/reid/metrics
// (alert.read), POST /api/reid/unlink (alert.acknowledge), GET /api/map (+ background.png), /media/<crop>. The reid
// routes 404 when "reid" is not enabled: the page then explains how to enable it, and in demo mode (or on request)
// shows labelled sample data (fixtures.js).
import { api, optional } from "../api.js";
import { S, on, can, camName } from "../state.js";
import { h, s, icon, replace, enc, num, fmtAgo, fmtDur, mediaUrl, NOSNAP } from "../util.js";
import { pageHeader, emptyState, skeleton, infBadge, obsBadge, estBadge, segmented, openDialog, confirmDialog, toast, kv } from "../ui.js";
import { meter } from "../charts.js";
import { hasEndpoint } from "../registry.js";
import { panelHead, sampleAllowed, sampleBanner, featureOff, code, colorOf } from "./common.js";
import { journeyFixtures } from "./fixtures.js";

const hms = t => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const fin = v => typeof v === "number" && isFinite(v);
const pct = v => (fin(v) ? `${Math.round(v * 100)}%` : "—");
const KIND = { handoff: "camera handoff", reentry: "re-entry, same camera", fragment: "tracker fragment, same camera" };

export default function createJourneys() {
  let shown = false, loading = false, timer = null, preview = false, sample = false;
  let scope = "recent", query = "", selected = null;
  let ids = undefined, metrics = null, journey = null, journeyErr = null, fx = null;
  const names = new Map();
  const nm = id => names.get(id) || camName(id);

  const subtitle = h("span");
  const scopeSeg = segmented([["active", "Active", "Only IDs with a sighting still in progress"], ["recent", "Recent", "Active plus IDs seen in the last 15 minutes"],
    ["hour", "Last hour", "IDs seen in the last hour (retention permitting)"]], scope, v => { scope = v; load(true); }, { label: "Time scope" });
  const search = h("input", { type: "search", placeholder: "G-number or camera", "aria-label": "Filter global IDs", class: "sm px-search",
    oninput: () => { query = search.value.trim().toLowerCase(); renderList(); } });
  const refreshBtn = h("button", { class: "sm", type: "button", title: "Refresh now", onclick: () => load(true) }, icon("refresh"), "Refresh");
  const bannerEl = h("div");
  const noteEl = h("div");
  const listBody = h("div", { class: "px-jlist", role: "listbox", "aria-label": "Global IDs" }, skeleton("row", 5));
  const detailBody = h("div", null, skeleton("card", 1));
  const mainEl = h("div", { class: "px-jgrid" },
    h("div", { class: "panel flush px-jcol" }, panelHead("journeys", "Global IDs", null, h("span", { class: "xs mute", id: "pxJCount" })),
      h("div", { class: "px-jtools" }, search), listBody),
    h("div", { class: "px-jdetail" }, detailBody));
  const el = h("section", { class: "page px-page" },
    pageHeader({ title: "Journeys", subtitle, actions: [scopeSeg, refreshBtn] }),
    bannerEl, noteEl, mainEl);

  // ------------------------------------------------------------------------------------------------ data
  async function load(force = false) {
    if (loading && !force) return;
    loading = true;
    const since = scope === "hour" ? 3600 : 900;
    let r = null, err = null;
    // Re-ID off answers 404: re-check only once a minute (or on Refresh) instead of every poll.
    const recheck = force || (Array.isArray(ids) && !sample) || Date.now() - offAt > 60000;
    if (hasEndpoint("/api/reid/tracks") !== false && recheck) {
      try {
        r = await optional(`/api/reid/tracks?limit=300${scope === "active" ? "&active=true" : `&since_s=${since}`}`, "reid", { silent: true });
        offAt = r === null ? Date.now() : 0;
      } catch (e) { err = e; }
    }
    try { metrics = r ? await optional("/api/reid/metrics", "reid", { silent: true }) : null; } catch { metrics = null; }
    loading = false;
    const real = r?.identities || [];
    sample = !real.length && !err && sampleAllowed(preview);
    if (sample) {
      fx = fx || journeyFixtures();
      ids = scope === "active" ? fx.identities.filter(i => i.active) : fx.identities;
      metrics = fx.metrics;
      for (const c of fx.map.cameras) names.set(c.id, c.name);
    } else {
      fx = null;
      ids = err ? { error: err } : r === null ? null : real;
    }
    if (Array.isArray(ids) && ids.length && (selected === null || !ids.some(i => i.gid === selected))) {
      if (selected === null) selected = ids[0].gid;
    }
    const sig = JSON.stringify([ids, metrics, sample, scope]);
    if (shown && (force || sig !== listSig)) { listSig = sig; renderTop(); renderList(); }   // unchanged: keep the DOM
    if (shown) await loadJourney();
  }
  let listSig = "", detailSig = "", offAt = 0;

  async function loadJourney() {
    if (selected === null || !Array.isArray(ids)) { journey = null; renderDetail(); return; }
    if (sample) { journey = fx?.journeys[selected] || null; journeyErr = null; renderDetail(); return; }
    try {
      journey = await api(`/api/reid/journey/${enc(selected)}`, { silent: true });
      journeyErr = null;
      if (journey.merged_into && journey.merged_into !== selected) selected = journey.merged_into;
    } catch (e) {
      journeyErr = e;
      journey = null;
    }
    renderDetail();
  }

  // ------------------------------------------------------------------------------------------------ helpers
  const band = c => {
    const thr = metrics?.thresholds?.link_threshold ?? 0.6;
    if (!fin(c)) return { cls: "", text: "no score" };
    return c >= 0.85 ? { cls: "ok", text: "high" } : c >= thr ? { cls: "warn", text: "medium" } : { cls: "err", text: "low" };
  };
  const confTag = (c, by) => {
    const b = band(c);
    return h("span", { class: ["tag", b.cls], title: `Link confidence ${pct(c)} (${b.text}): model estimate P(this sighting continues the previous one). ` +
      `Decided by ${by || "auto"}; revisable as more evidence arrives.` }, by === "operator" ? icon("check") : null, pct(c));
  };
  const camChip = id => { const c = colorOf(id);
    return h("span", { class: "px-camchip", style: { borderColor: c, background: `color-mix(in srgb, ${c} 14%, transparent)` } }, nm(id)); };
  const chain = cams => h("div", { class: "px-chain" }, cams.map((c, i) => [i ? h("span", { class: "px-arrow", "aria-hidden": "true" }, "→") : null, camChip(c)]));

  // ------------------------------------------------------------------------------------------------ render
  function renderTop() {
    const parts = [];
    if (sample) parts.push(sampleBanner(preview ? () => { preview = false; selected = null; load(true); } : null));
    replace(bannerEl, parts);
    const m = metrics;
    replace(noteEl, ids === null || ids?.error ? null : h("div", { class: "banner info px-research" }, icon("info"), h("div", { class: "grow" },
      h("b", null, "Research-grade, probabilistic and revisable. "),
      "A global ID is an opaque number for one journey through the cameras, estimated from appearance, camera topology and travel time. ",
      "It is not a name or an identity, can be wrong, and links are re-scored, split, re-linked or merged as evidence arrives. ",
      m ? h("span", { class: "mute" }, `${m.identities ?? "?"} IDs in memory · ${m.multi_camera_identities ?? "?"} span cameras · mean auto-link confidence ${pct(m.auto?.mean_link_confidence)}`,
        fin(m.operator?.auto_link_precision_from_feedback) ? ` · operator-confirmed precision ${pct(m.operator.auto_link_precision_from_feedback)} (n=${(m.operator.auto_links_confirmed || 0) + (m.operator.auto_links_rejected || 0)})` : "",
        m.retention_s ? ` · purged after ${fmtDur(m.retention_s)}` : "") : null)));
    const n = Array.isArray(ids) ? ids.length : 0;
    replace(subtitle, `${S.cfg.site || "Site"} · ${sample ? "sample data" : ids === null ? "Re-ID off" : `${n} global ID${n === 1 ? "" : "s"}`}${m?.topology_source ? ` · topology: ${m.topology_source}` : ""}`);
    mainEl.hidden = ids === null || !!ids?.error;
    if (ids === null) {
      replace(bannerEl, featureOff({ ic: "journeys", title: "Cross-camera Re-ID is off on this server",
        text: "Journeys link person tracks across cameras into probabilistic global IDs (appearance embeddings + learned camera topology and travel times).",
        steps: [["Set ", code('"reid": {"enabled": true}'), " in the site config and restart Argus. The OSNet Re-ID weights are loaded from data/models (GPU recommended, CPU works)."],
          ["Enable the handoff module for learned topology and travel-time gates; without it a uniform travel-time window is used."],
          ["Optional: place and calibrate the cameras so journeys can be drawn on the site map."]],
        onPreview: () => { preview = true; load(true); } }));
    } else if (ids?.error) {
      replace(bannerEl, h("div", { class: "banner err" }, icon("alert"), ids.error.kind === "forbidden" ? "Your role cannot read global IDs (needs alert.read)." : `Could not load global IDs: ${ids.error.message}`));
    }
  }

  function renderList() {
    const cnt = el.querySelector("#pxJCount");
    if (!Array.isArray(ids)) return;
    const list = ids.filter(i => !query || i.label.toLowerCase().includes(query) || String(i.gid) === query.replace(/^g-?/, "")
      || i.cameras.some(c => nm(c).toLowerCase().includes(query) || c.toLowerCase().includes(query)));
    if (cnt) cnt.textContent = `${list.length} shown`;
    if (!list.length) {
      replace(listBody, emptyState({ icon: "journeys", compact: true, title: ids.length ? "No match" : scope === "active" ? "Nobody tracked right now" : "No journeys yet",
        text: ids.length ? "No global ID matches this filter." : "Global IDs appear as soon as person tracks are seen. Widen the time scope to see ended journeys." }));
      return;
    }
    replace(listBody, list.map(i => {
      const on = i.gid === selected;
      const dur = Math.max(0, i.last_t - i.first_t);
      return h("button", { class: ["px-jitem", on && "on"], type: "button", role: "option", "aria-selected": String(on),
        onclick: () => { selected = i.gid; history.replaceState(null, "", `#/journeys/${i.gid}`); renderList(); journey = null; detailSig = ""; replace(detailBody, skeleton("card", 1)); loadJourney(); } },
        h("div", { class: "row", style: { gap: ".43rem" } }, h("b", { class: "px-gid" }, i.label),
          i.active ? h("span", { class: "pill st-new" }, h("span", { class: "pd", "aria-hidden": "true" }), "active") : h("span", { class: "xs mute" }, "ended"),
          i.operator_verified ? h("span", { class: "tag ok", title: "An operator confirmed or corrected at least one link" }, icon("check"), "verified") : null,
          h("span", { class: "grow" }), h("span", { class: "xs mute" }, fmtAgo(i.last_t))),
        chain(i.cameras),
        h("div", { class: "row xs mute", style: { gap: ".5rem" } },
          `${i.n_segments} sighting${i.n_segments === 1 ? "" : "s"} · ${fmtDur(dur)}`,
          h("span", { class: "grow" }),
          i.n_segments > 1 ? h("span", null, "weakest link ", confTag(i.min_link_confidence)) : h("span", null, "single sighting")));
    }));
  }

  function renderDetail() {
    const sig = JSON.stringify([selected, journey, journeyErr?.message || null, sample, Array.isArray(ids) ? ids.length : String(ids), !!S.map]);
    if (sig === detailSig) return;                           // unchanged: keep scroll/hover/open dialogs intact
    detailSig = sig;
    if (!Array.isArray(ids)) { replace(detailBody); return; }
    if (!ids.length) { replace(detailBody, h("div", { class: "panel" }, emptyState({ icon: "journeys", title: "Select a global ID", text: "Journeys appear here once people are tracked." }))); return; }
    const idn = ids.find(i => i.gid === selected);
    const j = journey;
    if (!j && journeyErr) {
      const why = journeyErr.kind === "forbidden" ? "Your role cannot view journey evidence (needs evidence.view). The summary below comes from the ID list."
        : journeyErr.status === 404 ? "This global ID expired or was merged; refresh the list." : `Could not load the journey: ${journeyErr.message}`;
      replace(detailBody, h("div", { class: "panel" }, h("div", { class: "banner" }, icon("info"), why), idn ? summary(idn, idn.segments, false) : null));
      return;
    }
    if (!j) { replace(detailBody, h("div", { class: "panel" }, skeleton("card", 1))); return; }
    replace(detailBody, summary(j, j.segments, true));
  }

  function summary(j, segs, full) {
    segs = [...(segs || [])].sort((a, b) => a.first_t - b.first_t);
    const camSeq = [];
    for (const sg of segs) if (camSeq[camSeq.length - 1] !== sg.camera) camSeq.push(sg.camera);
    const confs = segs.slice(1).map(sg => sg.link?.fused_confidence ?? sg.link_confidence).filter(fin);
    return h("div", { class: "px-jd" },
      h("div", { class: "panel" },
        h("div", { class: "px-jd-h" },
          h("div", null, h("div", { class: "row", style: { gap: ".5rem" } }, h("h3", { class: "px-gid lg" }, j.label || `G-${j.gid}`),
            j.active ? h("span", { class: "pill st-new" }, h("span", { class: "pd", "aria-hidden": "true" }), "active") : h("span", { class: "pill st-resolved" }, h("span", { class: "pd", "aria-hidden": "true" }), "ended"),
            estBadge("estimate"), infBadge("Re-ID")),
            h("div", { class: "small mute" }, `${hms(j.first_t)} → ${hms(j.last_t)} · ${fmtDur(j.last_t - j.first_t)} · ${segs.length} sighting${segs.length === 1 ? "" : "s"} on ${new Set(segs.map(x => x.camera)).size} camera${new Set(segs.map(x => x.camera)).size === 1 ? "" : "s"}`,
              j.merged_into ? ` · merged from G-${j.requested}` : "")),
          h("span", { class: "grow" }),
          confs.length ? h("div", { class: "px-jd-conf", title: "The chain is only as reliable as its weakest link." },
            h("div", { class: "xs mute" }, "weakest link"), confTag(Math.min(...confs))) : null),
        chain(camSeq),
        timeline(segs)),
      mapPanel(segs),
      h("div", { class: "panel" }, panelHead("eye", "Sightings & links", "Each hop shows why it was linked and what else it could have been"),
        h("ol", { class: "px-hops" }, segs.map((sg, i) => hop(sg, segs[i - 1], i, full, j)))));
  }

  // ---- timeline: one lane per camera, a bar per sighting, hop confidence at each linked bar -----------------------
  function timeline(segs) {
    if (!segs.length) return null;
    const lanes = [];
    for (const sg of segs) if (!lanes.includes(sg.camera)) lanes.push(sg.camera);
    const t0 = Math.min(...segs.map(x => x.first_t)), t1 = Math.max(...segs.map(x => x.last_t));
    const span = Math.max(10, t1 - t0), pad = span * 0.03;
    const X = t => ((t - t0 + pad) / (span + 2 * pad)) * 100;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => t0 + f * span);
    return h("div", { class: "px-tl", role: "img", "aria-label": `Timeline of ${segs.length} sightings across ${lanes.length} cameras` },
      lanes.map(c => h("div", { class: "px-tl-lane" },
        h("div", { class: "px-tl-l", title: c }, h("span", { class: "px-dot", style: { background: colorOf(c) } }), nm(c)),
        h("div", { class: "px-tl-track" }, segs.filter(x => x.camera === c).map(x => {
          const i = segs.indexOf(x);
          const conf = x.link?.fused_confidence ?? x.link_confidence;
          return h("div", { class: ["px-tl-bar", x.active && "live"], style: { left: `${X(x.first_t)}%`, width: `${Math.max(1.2, X(x.last_t) - X(x.first_t))}%`, background: colorOf(c) },
            title: `#${i + 1} · ${nm(c)} track ${x.track_id} · ${hms(x.first_t)}–${hms(x.last_t)} (${fmtDur(x.last_t - x.first_t)})${i ? ` · link ${pct(conf)}` : ""}` },
          h("span", { class: "px-tl-n" }, i + 1),
          i ? h("span", { class: ["px-tl-c", band(conf).cls] }, pct(conf)) : null);
        })))),
      h("div", { class: "px-tl-axis", "aria-hidden": "true" }, h("div"), h("div", { class: "px-tl-ticks" }, ticks.map(t => h("span", { style: { left: `${X(t)}%` } }, hms(t))))));
  }

  // ---- one hop: the sighting, its evidence crop, and (from the 2nd on) the link evidence --------------------------
  function hop(sg, prev, i, full, j) {
    const link = sg.link;
    const conf = link?.fused_confidence ?? sg.link_confidence;
    const by = link?.decided_by || sg.decided_by;
    const img = sg.crop ? h("img", { src: mediaUrl(sg.crop), alt: `Evidence crop: ${nm(sg.camera)} track ${sg.track_id}`, loading: "lazy",
      onerror: e => { e.target.src = NOSNAP; } }) : null;
    const evidenceBtn = h("button", { class: "px-crop", type: "button", disabled: !img || null,
      title: img ? "Open the evidence crop" : sample ? "Sample data: no crop" : full ? "No crop saved for this sighting (crops need save_crops and expire with the embedding TTL)" : "Needs evidence.view",
      onclick: () => img && openCrop(sg, j) }, img || h("span", { class: "px-crop-none" }, icon("user"), h("span", { class: "xs" }, "no crop")));
    const alertsHref = `#/review?camera=${enc(sg.camera)}&from=${Math.floor(sg.first_t - 5)}&to=${Math.ceil(sg.last_t + 5)}`;
    return h("li", { class: "px-hop" },
      i ? linkCard(sg, prev, link, conf, by) : h("div", { class: "px-link first" }, h("span", { class: "xs mute" }, "First sighting of this global ID")),
      h("div", { class: "px-sight" },
        h("span", { class: "px-sight-n", style: { background: colorOf(sg.camera) } }, i + 1),
        evidenceBtn,
        h("div", { class: "grow", style: { minWidth: 0 } },
          h("div", { class: "row", style: { gap: ".43rem" } }, h("b", null, nm(sg.camera)), h("span", { class: "xs mute" }, `track ${sg.track_id}`),
            sg.active ? h("span", { class: "pill st-new" }, h("span", { class: "pd", "aria-hidden": "true" }), "in view") : null, obsBadge("observed")),
          h("div", { class: "small mute" }, `${hms(sg.first_t)} → ${sg.active ? "now" : hms(sg.last_t)} · ${fmtDur(sg.last_t - sg.first_t)}`,
            full && (sg.entry || sg.exit) ? ` · entered ${sg.entry || "mid-image"}, ${sg.exit === "open" ? "still in view" : `left ${sg.exit || "?"}`}` : ""),
          h("div", { class: "row", style: { gap: ".5rem", marginTop: ".3rem" } },
            sample ? null : h("a", { class: "btnlink sm ghost", href: alertsHref, title: "Alerts on this camera during this sighting" }, icon("review"), "Alerts in this window"),
            sample ? null : h("a", { class: "btnlink sm ghost", href: `#/live/${enc(sg.camera)}` }, icon("live"), "Camera")))));
  }

  function linkCard(sg, prev, link, conf, by) {
    const b = band(conf);
    const ev = link || {};
    const exp = ev.expected_dt_s;
    const app = ev.appearance;
    const alts = (sg.alternatives || []).filter(a => !(a.segment && a.segment === prev?.id));
    const canUnlink = !sample && can("alert.acknowledge") && prev;
    return h("div", { class: ["px-link", b.cls] },
      h("div", { class: "px-link-h" },
        h("span", { class: "px-link-line", "aria-hidden": "true" }),
        h("b", null, pct(conf)), h("span", { class: "small" }, `link confidence (${b.text})`),
        h("span", { class: "xs mute" }, KIND[ev.kind] || (prev && prev.camera === sg.camera ? "same camera" : "camera handoff"),
          fin(ev.dt_s) ? ` · Δt ${fmtDur(ev.dt_s)}` : prev ? ` · Δt ${fmtDur(sg.first_t - prev.last_t)}` : ""),
        h("span", { class: "grow" }),
        by === "operator" ? h("span", { class: "tag ok", title: "Decided by an operator: locked against automatic changes" }, icon("check"), "operator") : h("span", { class: "tag est", title: "Automatic, probabilistic decision; re-scored when this sighting ends" }, "auto · revisable"),
        canUnlink ? h("button", { class: "sm ghost", type: "button", title: "Cut this link: this and later sightings get a new global ID (audited)", onclick: () => unlink(sg) }, "Split here") : null),
      link ? h("div", { class: "px-evid" },
        ev.kind === "handoff" ? evRow("Topology", fin(ev.topology_prob) ? `P(${nm(ev.from_camera)} → ${nm(ev.to_camera)}) = ${num(ev.topology_prob, 2)}` : "uniform prior",
          ev.topology_source || null) : null,
        exp ? evRow("Travel time", h("span", null, `${fmtDur(ev.dt_s)} observed · typical ${fmtDur(exp.p10)}–${fmtDur(exp.p90)} (median ${fmtDur(exp.p50)}, n=${exp.n})`,
          dtGauge(ev.dt_s, exp)), fin(ev.travel_time_llr) ? `log-likelihood ratio ${num(ev.travel_time_llr, 2)}` : null) : null,
        app ? evRow("Appearance", h("span", null, `similarity ${num(app.similarity, 2)} (${app.mode})`, meter(Math.max(0, app.similarity), { color: "var(--inf)", label: "Appearance similarity" })),
          `LLR ${num(app.llr, 2)} · crops ${(app.n_crops || []).join(" / ")}`) : evRow("Appearance", h("span", { class: "mute" }, "no embedding compared (topology/time only)")),
        ev.direction ? evRow("Direction", `exit ${ev.direction.exit || "?"} → entry ${ev.direction.entry || "?"}`,
          [ev.direction.exit_motion_ok === false || ev.direction.entry_motion_ok === false ? "motion inconsistent with the edge" : "motion consistent", fin(ev.direction.llr) ? `LLR ${num(ev.direction.llr, 2)}` : ""].filter(Boolean).join(" · ")) : null,
        ev.kind === "fragment" && fin(ev.gap_distance) ? evRow("Gap", `${num(ev.gap_distance, 3)} (normalised image distance)`, "a tracker lost and re-found the same person") : null,
        alts.length ? evRow("Alternatives", h("div", { class: "px-alts" }, alts.map(a => a.new_identity
          ? h("span", { class: "tag est", title: "Probability that this sighting is a new, unrelated person" }, `new person ${pct(a.confidence)}`)
          : h("span", { class: "tag", title: `Segment ${a.segment}` }, `${a.gid ? `G-${a.gid}` : "unassigned"} · ${nm(a.camera)} ${pct(a.confidence)}`))), "top candidates when decided") : null)
        : h("div", { class: "xs mute" }, "Link evidence needs evidence.view."));
  }

  function evRow(label, value, sub) {
    return h("div", { class: "px-ev" }, h("div", { class: "xs mute" }, label), h("div", { class: "small" }, value, sub ? h("div", { class: "xs mute" }, sub) : null));
  }

  /** Observed Δt on the learned p10–p90 travel-time band. */
  function dtGauge(dt, exp) {
    const lo = 0, hi = Math.max(exp.p90 * 1.4, dt * 1.1, 1);
    const X = v => `${Math.max(0, Math.min(100, (v - lo) / (hi - lo) * 100))}%`;
    return h("div", { class: "px-dt", role: "img", "aria-label": `Observed ${Math.round(dt)} s against typical ${Math.round(exp.p10)} to ${Math.round(exp.p90)} s` },
      h("div", { class: "px-dt-band", style: { left: X(exp.p10), width: `calc(${X(exp.p90)} - ${X(exp.p10)})` } }),
      h("div", { class: "px-dt-med", style: { left: X(exp.p50) } }),
      h("div", { class: "px-dt-obs", style: { left: X(dt) } }));
  }

  // ---- site map: FOVs of placed cameras, the journey path through them ------------------------------------------
  function mapPanel(segs) {
    const real = S.map?.cameras?.some(c => (c.fov_polygon_m || []).length || c.camera_xy_m) ? S.map : null;
    const mp = sample && (!real || !segs.some(sg => real.cameras.some(c => c.id === sg.camera && (c.fov_polygon_m || []).length))) ? fx?.map : real;
    const anchor = id => {
      const c = mp?.cameras?.find(x => x.id === id);
      if (!c) return null;
      const p = c.fov_polygon_m || [];
      if (p.length >= 3) return [p.reduce((a, q) => a + q[0], 0) / p.length, p.reduce((a, q) => a + q[1], 0) / p.length];
      return Array.isArray(c.camera_xy_m) ? c.camera_xy_m : null;
    };
    const stops = segs.map((sg, i) => ({ sg, i, p: anchor(sg.camera) })).filter(x => x.p);
    const head = panelHead("map", "Path on the site map", "Sightings at each camera's field-of-view centre; straight lines are hops, not walked routes");
    if (!mp || !stops.length) {
      return h("div", { class: "panel" }, head, emptyState({ icon: "map", compact: true, title: "No map position for these cameras",
        text: "Place and calibrate the cameras (Cameras → Calibration & placement) to draw journeys on the site map.",
        action: can("zone.edit") && segs[0] ? h("a", { class: "btnlink sm", href: `#/cameras/${enc(segs[0].camera)}/calibration` }, icon("sliders"), "Calibrate") : null }));
    }
    const b = mp.bounds_m || (() => {
      const all = mp.cameras.flatMap(c => [...(c.fov_polygon_m || []), ...(c.camera_xy_m ? [c.camera_xy_m] : [])]);
      const xs = all.map(q => q[0]), ys = all.map(q => q[1]);
      return [Math.min(...xs) - 2, Math.min(...ys) - 2, Math.max(...xs) + 2, Math.max(...ys) + 2];
    })();
    const [x0, y0, x1, y1] = b, W = Math.max(1, x1 - x0), H = Math.max(1, y1 - y0), U = Math.max(W, H);
    const P = ([x, y]) => [x - x0, y1 - y];
    const onPath = new Set(segs.map(sg => sg.camera));
    const bg = !mp.sample && mp.background_url && can("evidence.view")
      ? s("image", { href: mp.background_url, x: 0, y: 0, width: W, height: H, preserveAspectRatio: "none", style: { opacity: 0.55 } }) : null;
    const fovs = mp.cameras.filter(c => (c.fov_polygon_m || []).length >= 3).map(c => {
      const hl = onPath.has(c.id);
      return s("g", null, s("polygon", { points: c.fov_polygon_m.map(q => P(q).join(",")).join(" "), "vector-effect": "non-scaling-stroke",
        style: { fill: hl ? colorOf(c.id) : "var(--surface-3)", fillOpacity: hl ? 0.18 : 0.5, stroke: hl ? colorOf(c.id) : "var(--line-strong)", strokeWidth: 1 } }),
      s("text", { x: P(anchorOf(c))[0], y: P(anchorOf(c))[1] - U * 0.045, "text-anchor": "middle", style: { fontSize: `${U * 0.026}px`, fill: hl ? "var(--text)" : "var(--mute)", fontWeight: hl ? 600 : 400 } }, nm(c.id)));
    });
    const path = [];
    for (let k = 1; k < stops.length; k++) {
      const a = P(stops[k - 1].p), c = P(stops[k].p);
      const conf = stops[k].sg.link?.fused_confidence ?? stops[k].sg.link_confidence;
      const col = { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)", "": "var(--mute)" }[band(conf).cls];
      if (a[0] === c[0] && a[1] === c[1]) {  // same camera again: small loop
        const r = U * 0.03;
        path.push(s("circle", { cx: a[0] + r, cy: a[1] - r, r, "vector-effect": "non-scaling-stroke", style: { fill: "none", stroke: col, strokeWidth: 2, strokeDasharray: "3 3" } }));
        continue;
      }
      const dx = c[0] - a[0], dy = c[1] - a[1], L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
      const m = [a[0] + dx * 0.55, a[1] + dy * 0.55], ah = U * 0.022;
      path.push(s("line", { x1: a[0], y1: a[1], x2: c[0], y2: c[1], "vector-effect": "non-scaling-stroke", style: { stroke: col, strokeWidth: 2.5, strokeLinecap: "round",
        strokeDasharray: (stops[k].sg.link?.decided_by || stops[k].sg.decided_by) === "operator" ? null : "6 4" } }),
      s("polygon", { points: [[m[0] + ux * ah, m[1] + uy * ah], [m[0] - ux * ah - uy * ah * 0.7, m[1] - uy * ah + ux * ah * 0.7], [m[0] - ux * ah + uy * ah * 0.7, m[1] - uy * ah - ux * ah * 0.7]].map(q => q.join(",")).join(" "),
        style: { fill: col } }),
      s("text", { x: m[0] - uy * U * 0.035, y: m[1] + ux * U * 0.035, "text-anchor": "middle", style: { fontSize: `${U * 0.024}px`, fill: col, fontWeight: 700, paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: U * 0.006 } }, pct(conf)));
    }
    const dots = stops.map(({ sg, i, p }) => { const q = P(p);
      return s("g", null, s("circle", { cx: q[0], cy: q[1], r: U * 0.018, style: { fill: colorOf(sg.camera), stroke: "var(--surface)", strokeWidth: U * 0.004 } }),
        s("text", { x: q[0], y: q[1] + U * 0.008, "text-anchor": "middle", style: { fontSize: `${U * 0.02}px`, fill: "#fff", fontWeight: 700 } }, String(i + 1)),
        s("title", null, `#${i + 1} ${nm(sg.camera)} · ${hms(sg.first_t)}`)); });
    const missingCams = [...onPath].filter(c => !anchor(c));
    return h("div", { class: "panel" }, head,
      s("svg", { class: "px-map", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", role: "img", "aria-label": `Journey path across ${stops.length} sightings` },
        s("rect", { x: 0, y: 0, width: W, height: H, style: { fill: "var(--sunken)" } }), bg, fovs, path, dots),
      h("div", { class: "legend", style: { marginTop: ".43rem" } },
        h("span", null, h("i", { style: { background: "var(--ok)" } }), "high confidence"), h("span", null, h("i", { style: { background: "var(--warn)" } }), "medium"),
        h("span", null, h("i", { style: { background: "var(--err)" } }), "low"), h("span", null, "dashed = automatic (revisable) · solid = operator-confirmed")),
      h("div", { class: "note" }, mp.sample ? "Sample layout: not a real site." : (mp.model || "2D ground-plane map"),
        missingCams.length ? ` · not placed: ${missingCams.map(nm).join(", ")}` : ""));
  }
  function anchorOf(c) {
    const p = c.fov_polygon_m || [];
    return p.length >= 3 ? [p.reduce((a, q) => a + q[0], 0) / p.length, p.reduce((a, q) => a + q[1], 0) / p.length] : (c.camera_xy_m || [0, 0]);
  }

  // ---- actions ---------------------------------------------------------------------------------------------------
  function openCrop(sg, j) {
    openDialog({ title: `${j.label || `G-${j.gid}`} · ${nm(sg.camera)} track ${sg.track_id}`, wide: true,
      body: h("div", { class: "px-crop-dlg" },
        h("img", { src: mediaUrl(sg.crop), alt: `Evidence crop from ${nm(sg.camera)}`, onerror: e => { e.target.src = NOSNAP; } }),
        h("div", null,
          kv([["Camera", nm(sg.camera)], ["Track", String(sg.track_id)], ["Seen", `${hms(sg.first_t)} → ${sg.active ? "now" : hms(sg.last_t)}`],
            ["Link", sg.link ? `${pct(sg.link.fused_confidence)} · ${sg.link.decided_by || "auto"}` : "first sighting"], ["Crops embedded", String(sg.n_crops ?? "—")]]),
          h("p", { class: "note" }, "Best-quality person crop used for the appearance embedding. Viewing journey evidence is recorded in the audit trail. ",
            "The crop supports review; it is not an identification."),
          h("a", { class: "btnlink sm", href: `#/review?camera=${enc(sg.camera)}&from=${Math.floor(sg.first_t - 5)}&to=${Math.ceil(sg.last_t + 5)}` }, icon("review"), "Alerts in this window"))) });
  }

  async function unlink(sg) {
    const ok = await confirmDialog("Split this journey?",
      `Cut the link into this sighting (${nm(sg.camera)}, track ${sg.track_id}). It and any later sightings get a new global ID. The correction is locked against automatic changes and recorded in the audit trail.`,
      { okLabel: "Split here" });
    if (!ok) return;
    try {
      await api("/api/reid/unlink", { method: "POST", body: { segment: sg.id } });
      toast("Link removed. The sighting now starts a new global ID.", "ok");
      await load(true);
    } catch { /* api() already showed the error */ }
  }

  on("demo", () => { if (shown) { selected = null; load(true); } });
  on("map", () => { if (shown && journey) { detailSig = ""; renderDetail(); } });

  return {
    el,
    show(args) {
      shown = true;
      const g = parseInt(String(args?.[0] || "").replace(/^g-?/i, ""), 10);
      if (Number.isFinite(g)) selected = g;
      if (ids !== undefined) { renderTop(); renderList(); renderDetail(); }
      load(true);
      clearInterval(timer);
      timer = setInterval(() => { if (shown && !document.hidden) load(); }, 8000);
    },
    hide() { shown = false; clearInterval(timer); },
    onResync() { if (shown) load(true); },
    onVisibility(v) { if (v && shown) load(); },
  };
}
