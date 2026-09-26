"""Incident report, multi-camera incident summary and shift handover API.

  GET  /api/incidents/{id}/report     printable self-contained HTML (incident.read; audited incident_report_generated)
  POST /api/incidents/{id}/summary    assistive VLM narrative, cite-or-refuse (alert.acknowledge, the permission of
                                      the per-alert VLM explain; audited incident_summary_generated)
  GET  /api/handover                  handover notes (alert.read)
  POST /api/handover                  write a note {text, since?} (feedback.write; audited handover_note)
  GET  /api/handover/summary          unresolved-at-shift-end summary; since=<epoch s> or hours=<h> (default 8)
  GET  /api/handover/print            printable handover sheet (alert.read; audited handover_printed)
"""
from __future__ import annotations

import re
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Response

from . import handover as ho
from .vlm import VlmBusy, vlm_slot
from .narrative import summarize_incident
from .report import CSP, incident_report

HTML_HEADERS = {"content-security-policy": CSP, "cache-control": "no-store", "x-content-type-options": "nosniff"}


def _llm(site):
    from .api_incidents import _llm as incident_llm
    return incident_llm(site)


def reports_router(site, need, explainer=None) -> APIRouter:
    r = APIRouter()
    notes = ho.HandoverNotes(site.store)

    def incident(iid: str) -> dict:
        m = getattr(site, "incidents", None)
        if m is None:
            raise HTTPException(404)
        inc = m.get(iid)
        if inc is None:
            raise HTTPException(404, f"unknown incident {iid}")
        return inc

    def html(doc: str, digest: str, filename: str, download: bool) -> Response:
        headers = {**HTML_HEADERS, "x-report-sha256": digest}
        if download:
            headers["content-disposition"] = f'attachment; filename="{filename}"'
        return Response(doc, media_type="text/html; charset=utf-8", headers=headers)

    # ---- incident report -----------------------------------------------------------------------------------------
    @r.get("/api/incidents/{iid}/report")
    def report(iid: str, download: bool = False, p=Depends(need("incident.read"))):
        inc = incident(iid)
        doc, meta = incident_report(site, inc, p.username)
        site.store.audit(p.username, "incident_report_generated", iid,
                         {"sha256": meta["sha256"], "report_version": meta["report_version"], "alerts": meta["alerts"],
                          "evidence_files": len(meta["evidence"])})
        return html(doc, meta["sha256"], f"argonix_incident_{re.sub(r'[^A-Za-z0-9_-]', '_', iid)}.html", download)

    # ---- assistive multi-camera summary ----------------------------------------------------------------------------
    @r.post("/api/incidents/{iid}/summary")
    def summary(iid: str, p=Depends(need("alert.acknowledge"))):
        inc = incident(iid)
        if inc.get("merged_into"):
            raise HTTPException(400, f"incident was merged into {inc['merged_into']}")
        if not p.can("incident.read"):
            raise HTTPException(403, "permission 'incident.read' required")
        try:
            with vlm_slot(site):
                rec = summarize_incident(site, inc, explainer, _llm(site), p.username)
        except VlmBusy as e:
            raise HTTPException(429, str(e))
        site.incidents.set_assistive_summary(iid, rec, p.username)
        site.store.audit(p.username, "incident_summary_generated", iid,
                         {"engine": rec["engine"], "model": rec.get("model") or rec.get("model_attempted"),
                          "citations": rec.get("citations", [])[:20], "rejected": rec.get("rejected"),
                          "keyframes": [k["alert_id"] for k in rec.get("keyframes", [])]})
        return {"incident_id": iid, **rec}

    # ---- shift handover --------------------------------------------------------------------------------------------
    def window(since: float, hours: float) -> tuple[float, float]:
        now = time.time()
        if since and since > 0:
            if since >= now:
                raise HTTPException(400, "since must be in the past (epoch seconds)")
            return float(since), now
        if not 0 < hours <= 24 * 14:
            raise HTTPException(400, "hours must be in (0, 336]")
        return now - hours * 3600.0, now

    @r.get("/api/handover")
    def list_notes(limit: int = 50, p=Depends(need("alert.read"))):
        return notes.list(max(1, min(int(limit), 500)))

    @r.post("/api/handover")
    def add_note(body: dict = Body(...), p=Depends(need("feedback.write"))):
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "body must be {\"text\": \"...\"} (optional \"since\" epoch s or \"hours\")")
        if len(text) > ho.MAX_NOTE:
            raise HTTPException(400, f"note longer than {ho.MAX_NOTE} characters")
        try:
            since, now = window(float(body.get("since") or 0), float(body.get("hours") or 8))
        except (TypeError, ValueError):
            raise HTTPException(400, "since/hours must be numbers")
        snap = ho.snapshot(ho.shift_summary(site, since, now))
        n = notes.add(p.username, text.strip(), since, snap, now)
        site.store.audit(p.username, "handover_note", "", {"note_id": n["id"], "chars": len(n["text"]),
                                                           "since": since, "counts": snap["counts"]})
        return n

    @r.get("/api/handover/summary")
    def handover_summary(since: float = 0, hours: float = 8, p=Depends(need("alert.read"))):
        s, now = window(since, hours)
        return ho.shift_summary(site, s, now)

    @r.get("/api/handover/print")
    def handover_print(since: float = 0, hours: float = 8, download: bool = False, p=Depends(need("alert.read"))):
        s, now = window(since, hours)
        summ = ho.shift_summary(site, s, now)
        shift_notes = [n for n in notes.list(100) if float(n["t"]) >= s]
        doc, digest = ho.print_html(site, summ, shift_notes, p.username)
        site.store.audit(p.username, "handover_printed", "", {"sha256": digest, "since": s})
        return html(doc, digest, f"argonix_handover_{time.strftime('%Y%m%d_%H%M', time.localtime(now))}.html",
                    download)

    return r

