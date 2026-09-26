"""Shift handover: operator notes plus an automatic "unresolved at shift end" summary.

The summary is deterministic, built from the alert/incident store and the cameras' current state:
  * alerts updated since the shift start that are still open (new / acknowledged / escalated);
  * incidents active in the shift whose status is not resolved/closed (merged-away incidents excluded);
  * overdue items: alerts still "new" and incidents still "open" (nobody acknowledged them) for longer than the
    acknowledgement target of their priority (config "handover": {"ack_sla_min": {...}}; defaults below);
  * camera problems now (health DEGRADED/DOWN, stream error/reconnecting) and camera-integrity alerts in the shift;
  * top zones by alert count in the shift.
Handover notes are stored in the site database (SQLite, table handover_notes) with author and time, each with a
snapshot of the summary counts it was written against, and every note is audited ("handover_note").
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from typing import Optional

from .rules import LABEL

OPEN_ALERT = ("new", "acknowledged", "escalated")
OPEN_INCIDENT = ("open", "acknowledged", "investigating")
ACK_SLA_MIN = {"P1": 5.0, "P2": 15.0, "P3": 60.0, "P4": 240.0}
PRIO_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
MAX_ITEMS = 200
MAX_NOTE = 4000
DISCLAIMER = ("Generated from the alert and incident records and the cameras' current state. It lists what is still "
              "open; it does not assess people or intent. Overdue = not acknowledged within the priority's target.")


class HandoverNotes:
    """handover_notes table in the site's SQLite database."""

    def __init__(self, store):
        self.store = store
        with store.lock:
            store.db.execute("CREATE TABLE IF NOT EXISTS handover_notes(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                             "t REAL, author TEXT, shift_start REAL, text TEXT, snapshot TEXT)")
            store.db.commit()

    def add(self, author: str, text: str, shift_start: float, snapshot: dict, t: Optional[float] = None) -> dict:
        t = time.time() if t is None else float(t)
        with self.store.lock:
            cur = self.store.db.execute(
                "INSERT INTO handover_notes(t,author,shift_start,text,snapshot) VALUES(?,?,?,?,?)",
                (t, author, float(shift_start), text, json.dumps(snapshot)))
            self.store.db.commit()
            nid = cur.lastrowid
        return {"id": nid, "t": t, "author": author, "shift_start": float(shift_start), "text": text,
                "snapshot": snapshot}

    def list(self, limit: int = 50) -> list[dict]:
        with self.store.lock:
            rows = self.store.db.execute("SELECT id,t,author,shift_start,text,snapshot FROM handover_notes "
                                         "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"id": i, "t": t, "author": a, "shift_start": s, "text": x, "snapshot": json.loads(sn or "{}")}
                for i, t, a, s, x, sn in rows]


def _age_min(now: float, t) -> float:
    return round(max(0.0, now - float(t or now)) / 60.0, 1)


def shift_summary(site, since: float, now: Optional[float] = None, cfg: Optional[dict] = None) -> dict:
    """Unresolved-at-shift-end summary for the window [since, now]."""
    now = time.time() if now is None else float(now)
    cfg = {**(site.cfg.get("handover") or {}), **(cfg or {})}
    sla = {**ACK_SLA_MIN, **(cfg.get("ack_sla_min") or {})}
    names = {w.id: w.name for w in getattr(site, "workers", [])}
    store = site.store

    alerts = store.list_alerts(since=since, limit=5000)
    open_alerts, overdue = [], []
    by_prio, by_status = Counter(), Counter()
    zones: dict[tuple, dict] = {}
    integrity = defaultdict(int)
    for a in alerts:
        by_prio[a.get("priority")] += 1
        by_status[a.get("status")] += 1
        rules = sorted({e.get("rule") for e in a.get("events") or []})
        for e in a.get("events") or []:
            if e.get("rule") == "camera_health":
                integrity[e.get("camera_id")] += 1
                continue
            k = (e.get("camera_id"), e.get("zone_id"))
            z = zones.setdefault(k, {"camera_id": k[0], "camera": names.get(k[0], k[0]), "zone_id": k[1],
                                     "zone": e.get("zone_name") or k[1], "alerts": set(), "best": "P4",
                                     "rules": Counter()})
            z["alerts"].add(a["id"])
            z["rules"][e.get("rule")] += 1
            if PRIO_RANK.get(a.get("priority"), 9) < PRIO_RANK.get(z["best"], 9):
                z["best"] = a.get("priority")
        if a.get("status") not in OPEN_ALERT:
            continue
        row = {"id": a["id"], "title": a.get("title"), "priority": a.get("priority"), "status": a.get("status"),
               "created_at": a.get("created_at"), "updated_at": a.get("updated_at"),
               "cameras": [names.get(c, c) for c in a.get("cameras") or []], "zones": a.get("zones") or [],
               "incident_id": a.get("incident_id"), "rules": rules, "age_min": _age_min(now, a.get("created_at")),
               "suppressed": bool(a.get("suppressed"))}
        target = float(sla.get(a.get("priority"), ACK_SLA_MIN["P4"]))
        if a.get("status") == "new" and row["age_min"] > target:
            row["overdue_by_min"] = round(row["age_min"] - target, 1)
            overdue.append({**row, "kind": "alert", "target_min": target})
        open_alerts.append(row)
    open_alerts.sort(key=lambda r: (PRIO_RANK.get(r["priority"], 9), -float(r["updated_at"] or 0)))

    open_incidents = []
    if getattr(site, "incidents", None) is not None:
        for i in store.list_incidents(since=since, limit=2000):
            if i.get("merged_into") or i.get("status") not in OPEN_INCIDENT:
                continue
            row = {"id": i["id"], "title": i.get("title"), "priority": i.get("priority"), "status": i.get("status"),
                   "phase": i.get("phase"), "opened_at": i.get("opened_at"), "last_event_at": i.get("last_event_at"),
                   "assignee": i.get("assignee"), "n_alerts": len(i.get("alert_ids") or []),
                   "cameras": [names.get(c, c) for c in i.get("cameras") or []], "zones": i.get("zones") or [],
                   "age_min": _age_min(now, i.get("opened_at"))}
            target = float(sla.get(i.get("priority"), ACK_SLA_MIN["P4"]))
            if i.get("status") == "open" and row["age_min"] > target:
                row["overdue_by_min"] = round(row["age_min"] - target, 1)
                overdue.append({**row, "kind": "incident", "target_min": target})
            open_incidents.append(row)
        open_incidents.sort(key=lambda r: (PRIO_RANK.get(r["priority"], 9), -float(r["last_event_at"] or 0)))
    overdue.sort(key=lambda r: (PRIO_RANK.get(r["priority"], 9), -r["overdue_by_min"]))

    cameras, problems = [], []
    for w in getattr(site, "workers", []):
        st = getattr(w, "stats", {}) or {}
        hm = getattr(w, "health", None)
        health = getattr(hm, "state", None) or st.get("health") or "UNKNOWN"
        reasons = sorted(getattr(hm, "active", None) or st.get("health_reasons") or [])
        status = st.get("status", "?")
        row = {"id": w.id, "name": w.name, "health": health, "reasons": reasons, "status": status,
               "error": str(st.get("error") or "")[:200], "fps": st.get("fps"),
               "integrity_alerts": integrity.get(w.id, 0), "calibrated": getattr(w, "H", None) is not None}
        cameras.append(row)
        why = []
        if health in ("DEGRADED", "DOWN"):
            why.append(f"view {health.lower()}" + (f" ({', '.join(reasons)})" if reasons else ""))
        if status in ("error", "reconnecting"):
            why.append(f"stream {status}" + (f": {row['error']}" if row["error"] else ""))
        if why:
            problems.append({**row, "problem": "; ".join(why)})
    # cameras that had view problems during the shift but are fine now: history, not an open problem
    earlier = [{**c, "problem": f"{integrity[c['id']]} camera-integrity alert(s) in the shift; view is "
                                f"{str(c['health']).lower()} now"}
               for c in cameras if integrity.get(c["id"]) and not any(p["id"] == c["id"] for p in problems)]

    top_zones = sorted(({"camera_id": z["camera_id"], "camera": z["camera"], "zone_id": z["zone_id"], "zone": z["zone"],
                         "alerts": len(z["alerts"]), "best_priority": z["best"],
                         "rules": [LABEL.get(r, r) for r, _ in z["rules"].most_common(3)]} for z in zones.values()),
                       key=lambda z: (-z["alerts"], PRIO_RANK.get(z["best_priority"], 9)))[:5]

    hours = (now - since) / 3600.0
    lines = [f"Shift window {time.strftime('%Y-%m-%d %H:%M', time.localtime(since))} to "
             f"{time.strftime('%H:%M', time.localtime(now))} ({hours:.1f} h): {len(alerts)} alert(s) "
             f"({', '.join(f'{p} {by_prio[p]}' for p in ('P1', 'P2', 'P3', 'P4') if by_prio[p]) or 'none'})."]
    lines.append(f"Still open: {len(open_alerts)} alert(s)"
                 + (f" ({', '.join(f'{s} {n}' for s, n in Counter(r['status'] for r in open_alerts).items())})"
                    if open_alerts else "")
                 + (f" and {len(open_incidents)} incident(s)." if getattr(site, "incidents", None) is not None else "."))
    lines.append(f"Overdue (not acknowledged within target): {len(overdue)}." if overdue
                 else "Nothing is overdue for acknowledgement.")
    lines.append(f"Camera problems now: {'; '.join(p['name'] + ' — ' + p['problem'] for p in problems)}." if problems
                 else "No camera problem is open now.")
    if earlier:
        lines.append("Earlier in the shift: " + "; ".join(f"{c['name']} — {c['problem']}" for c in earlier) + ".")
    if top_zones:
        lines.append("Busiest zones: " + "; ".join(f"{z['zone']} on {z['camera']} ({z['alerts']})" for z in top_zones[:3])
                     + ".")
    return {"since": since, "until": now, "hours": round(hours, 2), "generated_at": now,
            "counts": {"alerts": len(alerts), "by_priority": dict(by_prio), "by_status": dict(by_status),
                       "open_alerts": len(open_alerts), "open_incidents": len(open_incidents), "overdue": len(overdue),
                       "camera_problems": len(problems)},
            "open_alerts": open_alerts[:MAX_ITEMS], "open_incidents": open_incidents[:MAX_ITEMS],
            "overdue": overdue[:MAX_ITEMS], "cameras": cameras, "camera_problems": problems,
            "camera_problems_earlier": earlier, "top_zones": top_zones,
            "ack_sla_min": sla, "lines": lines, "incidents_enabled": getattr(site, "incidents", None) is not None,
            "disclaimer": DISCLAIMER}


def snapshot(summary: dict) -> dict:
    """What a note records about the shift it was written in (counts and the open/overdue ids)."""
    return {"since": summary["since"], "until": summary["until"], "counts": summary["counts"],
            "open_alert_ids": [r["id"] for r in summary["open_alerts"]][:100],
            "open_incident_ids": [r["id"] for r in summary["open_incidents"]][:100],
            "overdue_ids": [r["id"] for r in summary["overdue"]][:100],
            "camera_problems": [p["id"] for p in summary["camera_problems"]]}


def print_html(site, summary: dict, notes: list[dict], actor: str) -> tuple[str, str]:
    """Printable handover sheet (same self-contained page and integrity footer as the incident report)."""
    from .report import chip, esc, fmt_t, page, tz_label
    site_name = site.cfg.get("site") if isinstance(site.cfg.get("site"), str) else (site.cfg.get("site") or {}).get("name")
    o = [f"<h1>Shift handover</h1><p class=\"sub\">{esc(site_name)} · {fmt_t(summary['since'])} → "
         f"{fmt_t(summary['until'])} ({tz_label()})</p>",
         "<div class=\"meta\">" + "".join(f"<div><b>{k}</b>{v}</div>" for k, v in [
             ("Alerts in shift", summary["counts"]["alerts"]), ("Open alerts", summary["counts"]["open_alerts"]),
             ("Open incidents", summary["counts"]["open_incidents"] if summary["incidents_enabled"] else "n/a"),
             ("Overdue", summary["counts"]["overdue"]), ("Camera problems", summary["counts"]["camera_problems"]),
             ("Printed by", esc(actor))]) + "</div>",
         "<h2>Unresolved at shift end</h2>", "".join(f"<p>{esc(x)}</p>" for x in summary["lines"])]

    def table(head, rows):
        return ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
    if summary["overdue"]:
        o.append("<h3>Overdue</h3>" + table(["Item", "Priority", "Status", "Title", "Age", "Over target by"], [
            [f"<span class=\"mono\">{esc(r['id'])}</span>", chip(r["priority"]), esc(r["status"]), esc(r["title"]),
             f"{r['age_min']:g} min", f"{r['overdue_by_min']:g} min (target {r['target_min']:g})"]
            for r in summary["overdue"]]))
    o.append("<h3>Open alerts</h3>" + (table(["Alert", "Priority", "Status", "Title", "Cameras", "Created"], [
        [f"<span class=\"mono\">{esc(r['id'])}</span>", chip(r["priority"]), esc(r["status"]), esc(r["title"]),
         esc(", ".join(r["cameras"])), fmt_t(r["created_at"])] for r in summary["open_alerts"]])
        if summary["open_alerts"] else "<p class=\"mute\">None.</p>"))
    if summary["incidents_enabled"]:
        o.append("<h3>Open incidents</h3>" + (table(["Incident", "Priority", "Status / phase", "Title", "Assignee",
                                                     "Last event"], [
            [f"<span class=\"mono\">{esc(r['id'])}</span>", chip(r["priority"]), f"{esc(r['status'])} / {esc(r['phase'])}",
             esc(r["title"]), esc(r["assignee"]), fmt_t(r["last_event_at"])] for r in summary["open_incidents"]])
            if summary["open_incidents"] else "<p class=\"mute\">None.</p>"))
    o.append("<h3>Cameras</h3>" + table(["Camera", "Health", "Stream", "Integrity alerts in shift", "Problem"], [
        [esc(c["name"]), esc(c["health"]), esc(c["status"]), c["integrity_alerts"],
         esc(next((p["problem"] for p in summary["camera_problems"] + summary["camera_problems_earlier"]
                   if p["id"] == c["id"]), ""))]
        for c in summary["cameras"]]))
    if summary["top_zones"]:
        o.append("<h3>Top zones</h3>" + table(["Zone", "Camera", "Alerts", "Best priority", "Behaviours"], [
            [esc(z["zone"]), esc(z["camera"]), z["alerts"], chip(z["best_priority"]), esc(", ".join(z["rules"]))]
            for z in summary["top_zones"]]))
    o.append("<h2>Handover notes</h2>")
    if notes:
        o.extend(f"<div class=\"box op\"><b>{esc(n['author'])}</b> · {fmt_t(n['t'])}<br>"
                 f"{esc(n['text']).replace(chr(10), '<br>')}</div>" for n in notes)
    else:
        o.append("<p class=\"mute\">No note written for this shift.</p>")
    o.append(f"<p class=\"note\">{esc(summary['disclaimer'])} Acknowledgement targets (min): "
             f"{esc(', '.join(f'{k} {v:g}' for k, v in summary['ack_sla_min'].items()))}.</p>")
    return page("Shift handover — Argus", "".join(o), "Shift handover.")
