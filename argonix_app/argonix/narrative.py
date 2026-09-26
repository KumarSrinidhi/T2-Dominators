"""Multi-camera incident summary: a short chronological narrative written by the VLM, cite-or-refuse.

The model gets the incident's alert records (ids, times, cameras, zones, rule labels, the rules' own observations)
and up to 4 keyframes (evidence snapshots spread over the cameras, time order) through the VLM server's /explain
(multi-image) endpoint; without keyframes or a vision server, the text model (/chat) gets the records only.

The narrative is kept only if it passes the same check as the investigation summaries (argonix.investigate):
it cites at least one id, every cited id belongs to this incident (its alerts, or the incident itself), it is not
too long and it uses no person-labelling words. Otherwise, and whenever no model is configured or the call fails,
a deterministic narrative is built from the alert records. Either way the text is stored on the incident as an
ASSISTIVE summary (labelled with its engine/model): it is never evidence and never changes priority or status.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .investigate import check_summary
from .rules import LABEL

PROMPT_VERSION = "incident-narrative-v1"
MAX_ALERTS = 25          # records given to the model
MAX_FRAMES = 4           # keyframes sent to the VLM
MAX_TEMPLATE_ALERTS = 8  # alerts named in the deterministic narrative
PRIO_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
SYSTEM = ("You write a short chronological narrative of one security-camera incident for an operator. Use only the "
          "alert records given (and the images, if any: evidence frames with detection boxes drawn in). Write 2 to 5 "
          "sentences in time order and cite the alert id in square brackets after each fact, e.g. [A123-4]. Cite only "
          "ids from the records. Describe observable events only (positions, movement, counts, camera and zone): never "
          "describe or judge people (no words like suspicious, criminal, intruder or dangerous), never guess identity "
          "or intent, and do not claim that detections on different cameras are the same person.")


def _hms(t: Optional[float]) -> str:
    return time.strftime("%H:%M:%S", time.localtime(float(t))) if t else "?"


def alert_records(store, inc: dict, cam_names: Optional[dict] = None) -> list[dict]:
    """The incident's alerts in time order, reduced to what the narrative may use."""
    cam_names = cam_names or {}
    rows = []
    for aid in inc.get("alert_ids", [])[-200:]:
        a = store.get_alert(aid)
        if not a:
            continue
        evs = a.get("events") or []
        obs = list(dict.fromkeys(str(e.get("observation") or "").strip() for e in evs if e.get("observation")))
        snaps = [(float(e.get("t_end") or 0), e.get("camera_id"), (e.get("evidence") or {}).get("snapshot"))
                 for e in evs if (e.get("evidence") or {}).get("snapshot")]
        rows.append({"id": aid, "priority": a.get("priority"),
                     "t0": min([float(e.get("t_start") or e.get("t_end") or 0) for e in evs] or [a.get("created_at") or 0]),
                     "t1": max([float(e.get("t_end") or 0) for e in evs] or [a.get("updated_at") or 0]),
                     "cameras": [cam_names.get(c, c) for c in a.get("cameras") or []],
                     "camera_ids": list(a.get("cameras") or []),
                     "zones": list(a.get("zones") or []),
                     "rules": sorted({LABEL.get(e.get("rule"), e.get("rule")) for e in evs}),
                     "observations": [o[:160] for o in obs[:3]], "snapshots": snaps})
    rows.sort(key=lambda r: r["t0"])
    return rows


def pick_keyframes(rows: list[dict], data_dir: Path, n: int = MAX_FRAMES) -> list[dict]:
    """Up to n evidence snapshots, spread over cameras first (round robin, best priority then newest per camera),
    returned in time order."""
    per_cam: dict[str, list] = {}
    for r in rows:
        for t, cam, rel in r["snapshots"]:
            if rel and (Path(data_dir) / rel).is_file():
                per_cam.setdefault(cam, []).append((PRIO_RANK.get(r["priority"], 9), -t, t, rel, r["id"], cam))
    for c in per_cam.values():
        c.sort()
    picked, used = [], set()
    while len(picked) < n and any(per_cam.values()):
        for cam in sorted(per_cam):
            while per_cam[cam] and per_cam[cam][0][3] in used:
                per_cam[cam].pop(0)
            if per_cam[cam] and len(picked) < n:
                _, _, t, rel, aid, c = per_cam[cam].pop(0)
                used.add(rel)
                picked.append({"alert_id": aid, "camera": c, "t": t, "path": rel})
    return sorted(picked, key=lambda f: f["t"])


def _jpeg(path: Path, max_side: int = 640) -> Optional[bytes]:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    except Exception:  # noqa: BLE001
        return None


def records_text(inc: dict, rows: list[dict], frames: list[dict], cam_names: dict) -> str:
    shown = rows[:MAX_ALERTS]
    lines = [f"[{r['id']}] {_hms(r['t0'])}-{_hms(r['t1'])} | {r['priority']} | camera {', '.join(r['cameras']) or '?'}"
             f" | zone {', '.join(r['zones']) or '?'} | {' + '.join(r['rules'])} | observed: "
             f"{'; '.join(r['observations']) or '-'}" for r in shown]
    head = (f"Incident {inc['id']} on {', '.join(cam_names.get(c, c) for c in inc.get('cameras', [])) or '?'}: "
            f"{len(rows)} alert(s){f' (first {len(shown)} listed)' if len(rows) > len(shown) else ''}.")
    img = [f"Image {i + 1}: evidence frame of [{f['alert_id']}] from camera {cam_names.get(f['camera'], f['camera'])} "
           f"at {_hms(f['t'])}." for i, f in enumerate(frames)]
    return "\n".join([head, "Alert records (time order):", *lines, *img])


def template_narrative(rows: list[dict]) -> tuple[str, list[str]]:
    """Deterministic chronological narrative from the alert records (cites every alert it names)."""
    if not rows:
        return "This incident has no alert records to summarise.", []
    shown = rows[:MAX_TEMPLATE_ALERTS]
    bits = [f"at {_hms(r['t0'])} {' + '.join(r['rules']) or 'an alert'} in {', '.join(r['zones']) or 'the view'} on "
            f"{', '.join(r['cameras']) or '?'} ({r['priority']}) [{r['id']}]" for r in shown]
    more = f"; and {len(rows) - len(shown)} later alert(s)" if len(rows) > len(shown) else ""
    cams = sorted({c for r in rows for c in r["cameras"]})
    text = (f"{len(rows)} alert(s) on {len(cams)} camera(s) between {_hms(rows[0]['t0'])} and "
            f"{_hms(max(r['t1'] for r in rows))}: " + "; ".join(bits) + more + ".")
    return text, [r["id"] for r in shown]


def call_model(explainer: Any, llm: Any, inc: dict, rows: list[dict], frames: list[dict], data_dir: Path,
               cam_names: dict) -> tuple[Optional[str], dict]:
    """(text, meta) from the VLM (keyframes + records) or the text model (records only); (None, meta) if neither."""
    images = [b for b in (_jpeg(Path(data_dir) / f["path"]) for f in frames) if b]
    if images and explainer is not None and callable(getattr(explainer, "describe", None)):
        prompt = SYSTEM + "\n\n" + records_text(inc, rows, frames[:len(images)], cam_names)
        t0 = time.perf_counter()
        d = explainer.describe(images, prompt, max_new_tokens=260)
        return str(d.get("text") or "").strip(), {
            "engine": "vlm", "endpoint": "explain", "model": d.get("model") or "vlm", "device": d.get("device"),
            "n_images": int(d.get("n_images") or len(images)), "latency_s": round(time.perf_counter() - t0, 2)}
    if llm is not None:
        t0 = time.perf_counter()
        r = llm.chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": records_text(inc, rows, [], cam_names)}], max_new_tokens=260)
        return str(r.get("text") or "").strip(), {"engine": "llm", "endpoint": "chat", "model": r.get("model") or "llm",
                                                  "n_images": 0, "latency_s": round(time.perf_counter() - t0, 2)}
    return None, {"engine": None}


def summarize_incident(site, inc: dict, explainer: Any = None, llm: Any = None, actor: str = "operator",
                       now: Optional[float] = None) -> dict:
    """Build the assistive summary record for one incident (the caller stores it and audits)."""
    now = time.time() if now is None else float(now)
    cam_names = {w.id: w.name for w in getattr(site, "workers", [])}
    rows = alert_records(site.store, inc, cam_names)
    frames = pick_keyframes(rows, Path(site.data))
    allowed = {r["id"] for r in rows} | set(inc.get("alert_ids", [])) | {inc["id"]}
    rec: dict = {"prompt_version": PROMPT_VERSION, "generated_at": now, "generated_by": actor,
                 "keyframes": [{k: f[k] for k in ("alert_id", "camera", "t")} for f in frames],
                 "n_alerts": len(rows), "rejected": None, "model": None}
    text, meta = None, {"engine": None}
    if rows:
        try:
            text, meta = call_model(explainer, llm, inc, rows, frames, Path(site.data), cam_names)
        except Exception as e:  # noqa: BLE001 - unreachable/slow model -> deterministic narrative
            rec["rejected"] = f"model unavailable ({type(e).__name__}: {str(e)[:160]})"
    if text is not None:
        ok, why, cited = check_summary(text, allowed)
        if ok:
            rec.update(text=text, citations=cited, engine=meta["engine"], model=meta.get("model"),
                       n_images=meta.get("n_images", 0), latency_s=meta.get("latency_s"),
                       label=f"Assistive summary — model {meta.get('model')}, not evidence. Verify each cited "
                             "alert's evidence.")
            return rec
        rec["rejected"] = f"model output rejected ({why})"
        rec["model_attempted"] = meta.get("model")
    elif meta.get("engine") is None and rec["rejected"] is None and rows:
        rec["rejected"] = "no VLM or text model is configured"
    t, cited = template_narrative(rows)
    rec.update(text=t, citations=cited, engine="template", n_images=0, keyframes=[],
               label="Deterministic summary built from the alert records (no model), not evidence.")
    return rec
