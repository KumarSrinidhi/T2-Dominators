"""Synchronised multi-camera incident replay: which media covers [incident start - pre, end + post] on each camera.

For every camera involved in an incident (the cameras its events came from) and every related camera (a predicted next
camera from the handoff estimate, or one the operator asked for), the plan lists
  * recording segments: the analytics-resolution MP4 segments (unannotated, `segment_s` long) overlapping the window,
    with their absolute start/end (the capture times of their first and last frames);
  * evidence clips: the annotated alert clips (boxes burned in) of the incident's alerts. Their span is approximate:
    `clip_pre_s` before to `clip_post_s` after the event frame, fewer frames if the camera had just started;
  * gaps: the parts of the window that no playable file covers, and the still-open segment ("in progress": its MP4
    is finalised when the segment closes).
The dashboard plays every camera against one master clock; this module only selects media (no video decoding).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

MAX_WINDOW_S = 2 * 3600.0
NOTE = ("Recordings are the analytics-resolution stream (no boxes); evidence clips are annotated and their timing is "
        "approximate (about ±1 frame interval, fewer pre-event frames if the camera had just started). Each player "
        "seeks to the master time minus the segment's start; a gap means no stored video for that camera and time.")


def overlapping(items: Iterable[dict], start: float, end: float) -> list[dict]:
    """Items with [start, end] overlapping the window (touching ends excluded), sorted by start."""
    return sorted((i for i in items if i["start"] < end and i["end"] > start), key=lambda i: (i["start"], i["end"]))


def gaps(covered: Iterable[tuple[float, float]], start: float, end: float, min_gap_s: float = 0.5) -> list[dict]:
    """Parts of [start, end] not covered by any interval (gaps shorter than `min_gap_s` are frame jitter)."""
    out, cur = [], start
    for a, b in sorted(covered):
        if b <= cur:
            continue
        if a - cur >= min_gap_s:
            out.append({"start": cur, "end": min(a, end)})
        cur = max(cur, b)
        if cur >= end:
            break
    if end - cur >= min_gap_s:
        out.append({"start": cur, "end": end})
    return [g for g in out if g["end"] > g["start"]]


def window(inc: dict, pre_s: float, post_s: float) -> tuple[float, float, bool]:
    """[first event - pre, last event + post] (clamped to MAX_WINDOW_S; True if clamped)."""
    evs = inc.get("events") or []
    t0 = min([float(inc["opened_at"])] + [float(e.get("t_start") or e.get("t_end")) for e in evs
                                            if e.get("t_start") or e.get("t_end")])
    t1 = max([float(inc.get("last_event_at") or t0)] + [float(e["t_end"]) for e in evs if e.get("t_end")])
    start, end = t0 - pre_s, t1 + post_s
    if end - start > MAX_WINDOW_S:
        return start, start + MAX_WINDOW_S, True
    return start, end, False


def related_cameras(inc: dict) -> dict[str, dict]:
    """Predicted next cameras of the incident's tracks (handoff estimate): camera -> best {prob, eta_s, from}."""
    out: dict[str, dict] = {}
    for key, row in (inc.get("predictions") or {}).items():
        for p in row.get("predictions") or []:
            cam = p.get("camera")
            if not cam or p.get("leaves_coverage"):
                continue
            if cam not in out or (p.get("prob") or 0) > (out[cam].get("prob") or 0):
                out[cam] = {"prob": p.get("prob"), "eta_s": p.get("eta_s"), "from": key}
    return out


def build_plan(inc: dict, recordings: Callable[[str, float, float], list[dict]], alerts: list[dict],
               exists: Callable[[str], bool], names: dict, pre_s: float = 10.0, post_s: float = 10.0,
               clip_pre_s: float = 4.0, clip_post_s: float = 4.0, extra: Iterable[str] = (),
               known: Optional[set] = None, open_segments: Optional[dict] = None) -> dict:
    """The replay plan for one incident.

    recordings(camera, start, end) -> closed segments; alerts: the incident's alert bodies (for evidence clips and
    their frame times); exists(rel_path) -> file present; open_segments: camera -> {"start", "end"} of the segment
    still being written; known: camera ids that exist on this site (unknown related/extra cameras are dropped)."""
    start, end, clamped = window(inc, pre_s, post_s)
    primary = list(inc.get("cameras") or [])
    for e in inc.get("events") or []:
        if e.get("camera_id") and e["camera_id"] not in primary:
            primary.append(e["camera_id"])
    rel = related_cameras(inc)
    cams: list[tuple[str, str, Optional[dict]]] = [(c, "primary", None) for c in primary]
    for c, info in rel.items():
        if c not in primary and (known is None or c in known):
            cams.append((c, "predicted_next", info))
    for c in extra:
        if c and c not in {x[0] for x in cams} and (known is None or c in known):
            cams.append((c, "requested", None))

    clips_by_cam: dict[str, list[dict]] = {}
    seen = set()
    for a in alerts:
        for ev in a.get("events") or []:
            evd = ev.get("evidence") or {}
            clip, ft = evd.get("clip"), evd.get("frame_time")
            if not clip or ft is None or clip in seen:
                continue
            seen.add(clip)
            clips_by_cam.setdefault(ev.get("camera_id"), []).append(
                {"kind": "evidence", "path": clip, "start": float(ft) - clip_pre_s, "end": float(ft) + clip_post_s,
                 "event_t": float(ft), "alert_id": a.get("id"), "rule": ev.get("rule"), "approx": True})

    out_cams = []
    for cam, role, info in cams:
        segs, missing = [], 0
        for r in overlapping(recordings(cam, start, end), start, end):
            if exists(r["path"]):
                segs.append({"kind": "recording", "path": r["path"], "start": r["start"], "end": r["end"]})
            else:
                missing += 1
        clips = []
        for c in overlapping(clips_by_cam.get(cam, []), start, end):
            if exists(c["path"]):
                clips.append(c)
            else:
                missing += 1
        live = (open_segments or {}).get(cam)
        pending = ({"kind": "in_progress", "start": max(start, live["start"]), "end": min(end, live["end"])}
                   if live and live["start"] < end and live["end"] > start else None)
        entry = {"camera": cam, "name": names.get(cam, cam), "role": role, "segments": segs, "clips": clips,
                 "in_progress": pending, "missing_files": missing,
                 "gaps": gaps([(s["start"], s["end"]) for s in segs + clips], start, end),
                 "coverage_s": round(sum(min(end, s["end"]) - max(start, s["start"]) for s in segs), 2)}
        if info:
            entry["prediction"] = {**info, "estimate": True}
        out_cams.append(entry)

    markers = sorted(({"t": float(e.get("t_start") or e.get("t_end")), "t_end": float(e.get("t_end") or e.get("t_start")),
                       "camera": e.get("camera_id"), "rule": e.get("rule"), "alert_id": e.get("alert_id"),
                       "zone_name": e.get("zone_name"), "learned_model": bool(e.get("learned_model"))}
                      for e in inc.get("events") or [] if e.get("t_start") or e.get("t_end")), key=lambda m: m["t"])
    return {"incident_id": inc["id"], "start": start, "end": end, "clamped": clamped, "pre_s": pre_s, "post_s": post_s,
            "cameras": out_cams, "markers": markers, "note": NOTE}


def plan_for_site(site, inc: dict, pre_s: float, post_s: float, extra: Iterable[str] = ()) -> dict:
    """build_plan wired to a running Site: its store, data directory, workers and alert bodies."""
    root = Path(site.data).resolve()

    def exists(rel: str) -> bool:
        f = (root / rel).resolve()
        return root in f.parents and f.is_file() and f.stat().st_size > 0

    names = {c["id"]: c.get("name", c["id"]) for c in site.cfg.get("cameras", [])}
    open_segments = {}
    for w in site.workers:
        rec, st, last = getattr(w, "_rec", None), getattr(w, "_rec_start", 0.0), getattr(w, "_last_t", None)
        if rec is not None and last:
            open_segments[w.id] = {"start": float(st), "end": float(last)}
    alerts = [a for a in (site.store.get_alert(aid) for aid in (inc.get("alert_ids") or [])[-200:]) if a]
    return build_plan(inc, site.store.recordings_between, alerts, exists, names, pre_s, post_s,
                      float(site.cfg.get("clip_pre_s", 4)), float(site.cfg.get("clip_post_s", 4)), extra,
                      known=set(names), open_segments=open_segments)
