// Sample data for the Analytics and Journeys pages. Used only when the real endpoint is absent or empty AND the
// dashboard is in demo (presenter) mode, or when the operator explicitly asks for a preview. Every screen that shows
// it carries a "Sample data" banner: these numbers are illustrative, never measured. Shapes mirror the real payloads
// (argonix/api_analytics.py, flow.py, anomaly.py, baseline.py, api_reid.py, global_ids.py) so the same renderers run.
import { S } from "../state.js";

// Deterministic pseudo-random numbers so the sample charts do not jump on every refresh.
function rng(seed) {
  let x = seed >>> 0 || 1;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
}
const r2 = v => Math.round(v * 100) / 100;

/** Up to three cameras: the site's real ids/names when there are any (so links and names resolve), else stand-ins. */
function demoCams() {
  const real = (S.cams || []).slice(0, 3).map(c => ({ id: c.id, name: c.name || c.id }));
  const stand = [{ id: "demo_entrance", name: "Entrance (sample)" }, { id: "demo_concourse", name: "Concourse (sample)" },
    { id: "demo_platform", name: "Platform (sample)" }];
  return real.length >= 3 ? real : [...real, ...stand.slice(real.length)];
}

function minuteSeries(n, fn) {
  const now = Math.floor(Date.now() / 60000) * 60;
  return Array.from({ length: n }, (_, i) => ({ t: now - (n - 1 - i) * 60, ...fn(i) }));
}

// --------------------------------------------------------------------------------------------------- analytics
export function analyticsFixtures() {
  const [c1, c2, c3] = demoCams();
  const R = rng(20260925);
  const wave = (i, base, amp, per = 18) => Math.max(0, base + amp * Math.sin(i / per * Math.PI * 2) + (R() - 0.5) * amp * 0.8);
  const now = Date.now() / 1000;

  const lineSeries = (bi, bo) => minuteSeries(60, i => ({ in: Math.round(wave(i, bi, bi * 0.6)), out: Math.round(wave(i + 7, bo, bo * 0.6)) }));
  const mkLine = (id, name, bi, bo, labels = { in: "in", out: "out" }) => {
    const series = lineSeries(bi, bo);
    const c60 = { in: series.reduce((a, r) => a + r.in, 0), out: series.reduce((a, r) => a + r.out, 0) };
    return { id, name, space: "image_px", points: [[120, 400], [620, 410]], labels, counts: { in: c60.in * 9 + 41, out: c60.out * 9 + 17 },
      counts_60m: c60, series };
  };
  const lines = { bucket_s: 60, direction_rule: "in = crossing along n = (-dy, dx) of A->B (image lines: towards the right-hand side looking from A to B)",
    cameras: [
      { camera: c1.id, name: c1.name, t: now, warnings: [], lines: [mkLine("gate_a", "Gate A", 6, 4, { in: "entering", out: "leaving" }), mkLine("gate_b", "Gate B", 2.5, 3)] },
      { camera: c2.id, name: c2.name, t: now, warnings: [], lines: [mkLine("stairs", "Stairs", 4, 5, { in: "up", out: "down" })] },
    ] };

  const zoneSeries = (occ, dens, wait) => minuteSeries(60, i => {
    const o = wave(i, occ, occ * 0.5, 22);
    return { occ_mean: r2(o), occ_max: Math.ceil(o * 1.3), entries: Math.round(wave(i, occ / 2, occ / 3)), exits: Math.round(wave(i + 3, occ / 2, occ / 3)),
      density: dens ? r2(o / dens) : null, wait_est_s: wait ? Math.round(wave(i, wait, wait * 0.35, 25)) : null };
  });
  const flowHist = peak => { const h = [0.04, 0.05, 0.06, 0.04, 0.05, 0.04, 0.05, 0.03]; h[peak] += 0.64; return h.map(r2); };
  const zones = { bucket_s: 60, series_minutes: 60, note: "Sample data (demo mode): illustrative values, not measured.",
    cameras: [
      { camera: c1.id, name: c1.name, space: "ground_m", calibrated: true, calib_error_m: 0.3, t: now, warnings: [], zones: [
        { id: "lobby", name: "Lobby", kind: "area", space: "ground_m", criticality: 1, rules: ["counterflow", "density"], occupancy: 9,
          entries_total: 1840, exits_total: 1822, entries_60m: 212, exits_60m: 205,
          dwell: { n: 205, p50_s: 21.4, p90_s: 74.0, bins_s: [0, 5, 15, 30, 60, 120, 300, 600], counts: [12, 51, 63, 41, 24, 9, 4, 1] },
          flow: { bins: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"], hist: flowHist(2), track_seconds: 5120, dominant_deg: 88.5, concentration: 0.71, permitted_deg: 90 },
          density: { area_m2: 46.5, ppm2: 0.19, level: "normal", thresholds: { warn_ppm2: 2, critical_ppm2: 4 }, rule_enabled: true, calib_error_m: 0.3 },
          queue: null, series: zoneSeries(8, 46.5, 0) },
        { id: "ticket_queue", name: "Ticket queue", kind: "queue", space: "ground_m", criticality: 1, rules: ["queue_long"], occupancy: 7,
          entries_total: 640, exits_total: 633, entries_60m: 58, exits_60m: 56,
          dwell: { n: 56, p50_s: 82.0, p90_s: 151.0, bins_s: [0, 5, 15, 30, 60, 120, 300, 600], counts: [0, 2, 5, 9, 18, 16, 5, 1] },
          flow: { bins: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"], hist: flowHist(0), track_seconds: 2210, dominant_deg: 2.1, concentration: 0.62, permitted_deg: null },
          density: { area_m2: 12.0, ppm2: 0.58, level: "normal", thresholds: { warn_ppm2: 2, critical_ppm2: 4 }, rule_enabled: false, calib_error_m: 0.3 },
          queue: { length: 7, length_mean_60s: 6.4, length_mean_window: 6.1, window_s: 300, departures_window: 22, service_rate_per_min: 4.4,
            wait_est_s: 83.2, steady_state: true, measured_wait_p50_s: 79.5, measured_waits_n: 31, method: "service-line crossings",
            note: "Estimate: Little's law W = L / lambda (mean queue length over the window / departure rate). Valid only in steady state; measured waits are per-track entry->service times." },
          series: zoneSeries(6, 12, 80) }] },
      { camera: c2.id, name: c2.name, space: "ground_m", calibrated: true, calib_error_m: 0.45, t: now, warnings: [], zones: [
        { id: "concourse", name: "Concourse", kind: "area", space: "ground_m", criticality: 2, rules: ["counterflow", "dispersal", "convergence", "density"], occupancy: 23,
          entries_total: 3120, exits_total: 3101, entries_60m: 402, exits_60m: 398,
          dwell: { n: 398, p50_s: 12.8, p90_s: 40.2, bins_s: [0, 5, 15, 30, 60, 120, 300, 600], counts: [48, 170, 112, 44, 16, 6, 2, 0] },
          flow: { bins: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"], hist: flowHist(4), track_seconds: 11800, dominant_deg: 181.0, concentration: 0.55, permitted_deg: null },
          density: { area_m2: 120.0, ppm2: 0.21, level: "normal", thresholds: { warn_ppm2: 2, critical_ppm2: 4 }, rule_enabled: true, calib_error_m: 0.45 },
          queue: null, series: zoneSeries(22, 120, 0) }] },
      { camera: c3.id, name: c3.name, space: "image_px", calibrated: false, calib_error_m: null, t: now, warnings: [], zones: [
        { id: "platform_edge", name: "Platform edge", kind: "area", space: "image_px", criticality: 3, rules: [], occupancy: 4,
          entries_total: 410, exits_total: 407, entries_60m: 39, exits_60m: 38,
          dwell: { n: 38, p50_s: 35.0, p90_s: 118.0, bins_s: [0, 5, 15, 30, 60, 120, 300, 600], counts: [1, 4, 9, 11, 8, 4, 1, 0] },
          flow: { bins: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"], hist: flowHist(6), track_seconds: 980, dominant_deg: 268.0, concentration: 0.48, permitted_deg: null },
          density: null, density_note: "uncalibrated camera: counts only (no metric area, speed or m/s thresholds)",
          queue: null, series: zoneSeries(4, 0, 0) }] },
    ] };

  const ev = (rule, cam, zone, ago, obs, prio, conf, deriv = {}) => ({
    id: `sample-${rule}-${ago}`, title: { counterflow: "Counter-flow in zone", dispersal: "Rapid outward movement from a point (dispersal cue)",
      convergence: "Rapid inward movement to a point (convergence cue)", density: "Crowd density above limit", queue_long: "Queue longer than limit" }[rule],
    priority: prio, status: ago < 1800 ? "new" : "resolved", created_at: now - ago, updated_at: now - ago + 20, cameras: [cam.id], sample: true,
    events: [{ rule, camera_id: cam.id, zone_id: zone.toLowerCase().replace(/\W+/g, "_"), zone_name: zone, t_start: now - ago - 8, t_end: now - ago,
      observation: obs, confidence: conf, derivation: { rule, ...deriv } }] });
  const flowAlerts = [
    ev("counterflow", c2, "Concourse", 540, "5 of 12 moving person tracks in 'Concourse' moving against the learned dominant flow (towards E instead of W); peak 42% for 6.2 s", "P3", 0.71, { fraction_against: 0.42, moving_tracks: 12 }),
    ev("dispersal", c2, "Concourse", 3900, "6 of 8 moving person tracks in 'Concourse' moving rapidly outward from a common point (median radial speed 2.1 m/s, up from 0.9 m/s); a motion cue only, the cause is unknown", "P2", 0.66, { median_radial_speed_mps: 2.1 }),
    ev("queue_long", c1, "Ticket queue", 7300, "Queue in 'Ticket queue': 11.2 people (60 s mean, peak 13) for 48 s, estimated wait 152 s (Little's law, estimate)", "P3", 0.62),
    ev("counterflow", c1, "Lobby", 15800, "3 of 7 moving person tracks in 'Lobby' moving against the permitted direction (configured) (towards S instead of N); peak 43% for 3.4 s", "P3", 0.68, { fraction_against: 0.43, moving_tracks: 7 }),
    ev("convergence", c2, "Concourse", 40200, "5 of 7 moving person tracks in 'Concourse' moving rapidly inward towards a common point (median radial speed 1.7 m/s, up from 0.8 m/s); a motion cue only, the cause is unknown", "P3", 0.6, { median_radial_speed_mps: 1.7 }),
  ];

  const cells = (n, size, ox, oy, R2) => Array.from({ length: n }, () => {
    const x = Math.floor(R2() * 8) * size + ox, y = Math.floor(R2() * 6) * size + oy;
    return { x0: x, y0: y, size, count: 1 + Math.floor(R2() * R2() * 14), rules: { counterflow: 1 } };
  }).reduce((acc, c) => { const k = `${c.x0},${c.y0}`; const o = acc.m.get(k); if (o) o.count += c.count; else { acc.m.set(k, c); acc.l.push(c); } return acc; },
    { m: new Map(), l: [] }).l.sort((a, b) => b.count - a.count);
  const R3 = rng(42);
  const hotspots = { window: "24h", since: now - 86400, until: now, cell_m: 1, cell_px: 48, include_dismissed: false, events: 57,
    note: "Event frequency, not risk: a hotspot shows where rules fired most (true or false positives). Ground cells use each camera's own ground frame; image cells are for uncalibrated cameras.",
    cameras: [
      { camera: c1.id, name: c1.name, events: 21, unplaced: 1, ground_cells: cells(22, 1, -4, 2, R3), image_cells: [] },
      { camera: c2.id, name: c2.name, events: 29, unplaced: 0, ground_cells: cells(30, 1, -3, 4, R3), image_cells: [] },
      { camera: c3.id, name: c3.name, events: 7, unplaced: 0, ground_cells: [], image_cells: cells(9, 48, 96, 144, R3) }],
    zones: [
      { camera: c2.id, zone_id: "concourse", zone_name: "Concourse", events: 24, alerts: 9, by_rule: { counterflow: 15, dispersal: 3, convergence: 2, density: 4 }, last: now - 540 },
      { camera: c1.id, zone_id: "ticket_queue", zone_name: "Ticket queue", events: 13, alerts: 6, by_rule: { queue_long: 13 }, last: now - 7300 },
      { camera: c1.id, zone_id: "lobby", zone_name: "Lobby", events: 8, alerts: 5, by_rule: { counterflow: 6, loitering: 2 }, last: now - 15800 },
      { camera: c3.id, zone_id: "platform_edge", zone_name: "Platform edge", events: 7, alerts: 4, by_rule: { intrusion: 7 }, last: now - 21000 },
      { camera: c2.id, zone_id: "exit_ramp", zone_name: "Exit ramp", events: 5, alerts: 2, by_rule: { running: 5 }, last: now - 50000 }] };

  const feats = ["speed_mean", "speed_p90", "speed_std", "straightness", "log_dwell_s", "dir_changes_per_min"];
  const recent = (seed) => { const Rr = rng(seed); return Array.from({ length: 8 }, (_, i) => {
    const p = i === 1 ? 0.994 : 0.4 + Rr() * 0.55;
    return { track_id: 300 + i * 7, t: now - i * 95, percentile: r2(p * 100) / 100, score: r2(0.4 + p * 0.2),
      top: [{ feature: feats[Math.floor(Rr() * feats.length)], value: r2(Rr() * 3), typical: r2(0.9 + Rr()), robust_z: r2((Rr() - 0.3) * 6) }] }; }); };
  const anomaly = { enabled: true, model: "iforest-trajectory", features: feats,
    params: { n_trees: 100, max_samples: 256, calib_fraction: 0.25, emit_percentile: 0.99, refit_every: 500, refit_s: 3600, window: 2000 },
    note: "Corroboration only (AGENTS.md): events are INFO level and capped at P3 unless a deterministic rule agrees. Percentiles are relative to this camera's recent traffic.",
    cameras: [
      { camera: c1.id, trained: true, version: 4, fitted_at: now - 2400, n_window: 1600, n_train: 1200, n_calib: 400, min_train: 250, fitting: false,
        speed_unit: "m/s", scored: 812, emitted: 6, recent: recent(7), typical: { speed_mean: 1.21, speed_p90: 1.62, speed_std: 0.21, straightness: 0.91, log_dwell_s: 2.9, dir_changes_per_min: 1.4 } },
      { camera: c2.id, trained: true, version: 2, fitted_at: now - 5100, n_window: 900, n_train: 680, n_calib: 220, min_train: 250, fitting: false,
        speed_unit: "m/s", scored: 1530, emitted: 11, recent: recent(9), typical: { speed_mean: 1.34, speed_p90: 1.8, speed_std: 0.25, straightness: 0.88, log_dwell_s: 2.4, dir_changes_per_min: 1.9 } },
      { camera: c3.id, trained: false, version: 0, fitted_at: null, n_window: 120, n_train: 0, n_calib: 0, min_train: 250, fitting: false,
        speed_unit: "body-heights/s", scored: 0, emitted: 0, recent: [], typical: null }] };

  const baseline = { cusum: { enabled: true, k: 0.5, h: 8, cooldown_s: 1800, series: [
    { camera: c1.id, zone: "lobby", s_pos: 2.4, s_neg: 0.3, evaluations: 410 },
    { camera: c1.id, zone: "ticket_queue", s_pos: 6.1, s_neg: 0.0, evaluations: 410 },
    { camera: c2.id, zone: "concourse", s_pos: 0.8, s_neg: 3.7, evaluations: 395 },
    { camera: c3.id, zone: "platform_edge", s_pos: 0.0, s_neg: 1.2, evaluations: 120 }] } };

  return { zones, lines, flowAlerts, hotspots, anomaly, baseline };
}

// --------------------------------------------------------------------------------------------------- journeys
export function journeyFixtures() {
  const cams = demoCams();
  const now = Date.now() / 1000;
  const R = rng(1717);
  const mkId = (spec, gid) => {
    let t = now - spec.start;
    const segs = spec.hops.map((h, i) => {
      if (i) t += h.dt;
      const dur = h.dur;
      const s = { id: `${cams[h.cam].id}:${100 + gid * 3 + i}:${Math.round(t)}`, camera: cams[h.cam].id, track_id: 100 + gid * 3 + i,
        first_t: t, last_t: t + dur, active: spec.active && i === spec.hops.length - 1, gid,
        link_confidence: i ? h.conf : null, decided_by: i ? (h.by || "auto") : null, locked: h.by === "operator" };
      t += dur;
      return s;
    });
    return { segs, spec };
  };
  const specs = [
    { gid: 17, start: 420, active: true, hops: [{ cam: 0, dur: 38 }, { cam: 1, dt: 21, dur: 64, conf: 0.86 }, { cam: 2, dt: 34, dur: 0, conf: 0.64 }] },
    { gid: 12, start: 1500, active: false, hops: [{ cam: 1, dur: 52 }, { cam: 0, dt: 26, dur: 41, conf: 0.91, by: "operator" }] },
    { gid: 9, start: 2600, active: false, hops: [{ cam: 2, dur: 30 }, { cam: 1, dt: 40, dur: 75, conf: 0.62 }, { cam: 1, dt: 6, dur: 22, conf: 0.97, kind: "fragment" }, { cam: 0, dt: 19, dur: 28, conf: 0.73 }] },
    { gid: 21, start: 200, active: true, hops: [{ cam: 1, dur: 0 }] },
  ];
  const built = specs.map(sp => mkId(sp, sp.gid));
  // live segments keep growing until "now"
  for (const b of built) for (const s of b.segs) if (s.active) s.last_t = now - 2;

  const identities = built.map(({ segs, spec }) => {
    const camsSeq = [];
    for (const s of segs) if (camsSeq[camsSeq.length - 1] !== s.camera) camsSeq.push(s.camera);
    const confs = segs.map(s => s.link_confidence).filter(v => v != null);
    const cur = segs.find(s => s.active);
    return { gid: spec.gid, label: `G-${spec.gid}`, active: spec.active, first_t: segs[0].first_t, last_t: Math.max(...segs.map(s => s.last_t)),
      cameras: camsSeq, n_segments: segs.length, n_cameras: new Set(camsSeq).size, min_link_confidence: confs.length ? Math.min(...confs) : null,
      operator_verified: segs.some(s => s.locked), current: cur ? { camera: cur.camera, track_id: cur.track_id, bbox: [0.4, 0.3, 0.5, 0.8] } : null,
      segments: segs };
  });

  const journeys = {};
  for (const { segs, spec } of built) {
    const full = segs.map((s, i) => {
      const hop = spec.hops[i];
      const prev = segs[i - 1];
      const kind = hop.kind || (prev && prev.camera === s.camera ? "reentry" : "handoff");
      const sim = r2(0.55 + R() * 0.3);
      const link = prev ? { kind, from_segment: prev.id, to_segment: s.id, from_camera: prev.camera, to_camera: s.camera, dt_s: hop.dt,
        method: "bayes-odds-v1", mode: "fused", arrival_rate_per_min: r2(1 + R() * 3),
        topology_prob: kind === "handoff" ? r2(0.45 + R() * 0.4) : null,
        expected_dt_s: kind === "handoff" ? { p10: Math.round(hop.dt * 0.6), p50: Math.round(hop.dt * 1.05), p90: Math.round(hop.dt * 1.8), n: 40 + Math.floor(R() * 90), source: "learned" } : null,
        travel_time_llr: kind === "handoff" ? r2(0.4 + R()) : null,
        direction: kind === "handoff" ? { exit: "right", entry: "left", exit_motion_ok: true, entry_motion_ok: true, learned_entry_llr: 0.6, llr: 0.6 } : null,
        appearance: { similarity: sim, mode: "color", llr: r2((sim - 0.5) * 8), n_crops: [4, 5], different_person_model: "site negatives (sample)" },
        prior_odds: r2(0.5 + R()), odds: r2(2 + R() * 6), fused_confidence: hop.conf, decided_by: hop.by || "auto", t: s.first_t + 2 } : null;
      const alts = prev ? [
        { segment: prev.id, gid: spec.gid, camera: prev.camera, track_id: prev.track_id, confidence: hop.conf },
        { segment: `${cams[(spec.hops[i].cam + 1) % cams.length].id}:77:${Math.round(s.first_t - 30)}`, gid: 4, camera: cams[(spec.hops[i].cam + 1) % cams.length].id, track_id: 77, confidence: r2((1 - hop.conf) * 0.6) },
        { new_identity: true, confidence: r2((1 - hop.conf) * 0.4) }] : [];
      return { ...s, entry: i ? "left" : "top", exit: s.active ? "open" : "right", n_crops: 4, crop: null, link, alternatives: alts, decided: true, bbox: [0.4, 0.3, 0.5, 0.8] };
    });
    const idn = identities.find(x => x.gid === spec.gid);
    journeys[spec.gid] = { gid: spec.gid, label: `G-${spec.gid}`, requested: spec.gid, merged_into: null, active: spec.active,
      first_t: idn.first_t, last_t: idn.last_t, segments: full, info: { created_t: idn.first_t }, events: [], kind: "estimate",
      note: "Sample data (demo mode): illustrative, not measured." };
  }

  const metrics = { identities: identities.length, identities_active: identities.filter(i => i.active).length,
    multi_camera_identities: identities.filter(i => i.n_cameras > 1).length, segments_total: 11,
    auto: { links: 6, new: 9, merges: 1, splits: 1, relinks: 0, displaced: 0, mean_link_confidence: 0.79 },
    operator: { links: 1, unlinks: 0, confirms: 1, auto_links_confirmed: 3, auto_links_rejected: 1, auto_link_precision_from_feedback: 0.75 },
    thresholds: { link_threshold: 0.6, unlink_threshold: 0.35, relink_margin: 0.15, top_k: 3 }, retention_s: 3600,
    topology_source: "handoff (learned)", method: "bayes-odds-v1", embedder: { model: "osnet_x0_25 (sample)", device: "cuda" }, warnings: [] };

  // A plausible site map for the sample cameras (used only when the real /api/map has no placed cameras).
  const map = { bounds_m: [-2, -2, 42, 22], sample: true, model: "sample layout (not a real site)", cameras: cams.map((c, i) => {
    const ox = 4 + i * 13, oy = 6 + (i % 2) * 5;
    return { id: c.id, name: c.name, calibrated: true, camera_xy_m: [ox + 4, oy - 5],
      fov_polygon_m: [[ox, oy], [ox + 9, oy], [ox + 10, oy + 9], [ox - 1, oy + 9]] };
  }) };

  return { identities, journeys, metrics, map };
}
