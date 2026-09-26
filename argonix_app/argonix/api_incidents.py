"""Incidents, incident graph, synchronised replay plan, hotspots and natural-language investigation API.

Incident endpoints 404 with a plain "Not Found" (the UI then treats the feature as absent) when the incident manager
is disabled ("incidents": {"enabled": false}). Every mutating call and every investigation is audited.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from .incidents import compact, hotspots as compute_hotspots
from .investigate import Investigator, QueryError, build_vocab, schema
from .replay import plan_for_site

INC_STATUS_PERM = {"open": "alert.acknowledge", "acknowledged": "alert.acknowledge", "investigating": "alert.escalate",
                   "resolved": "alert.resolve", "closed": "alert.resolve"}
PATCH_KEYS = {"status", "assignee", "summary", "note"}


def _llm(site):
    """The text model for investigation: site.llm (set by run.py --vlm-url) or config investigate.llm_url."""
    llm = getattr(site, "llm", None)
    if llm is None:
        url = (site.cfg.get("investigate") or {}).get("llm_url")
        if url:
            from .vlm import ChatClient
            llm = site.llm = ChatClient(url, float((site.cfg.get("investigate") or {}).get("timeout_s", 45)))
    return llm


def incidents_router(site, need) -> APIRouter:
    r = APIRouter()
    inv = Investigator(site.store, lambda: build_vocab(site), lambda: _llm(site), site.cfg.get("investigate") or {})
    site.investigator = inv

    def mgr():
        m = getattr(site, "incidents", None)
        if m is None:
            raise HTTPException(404)
        return m

    def names() -> dict:
        return {w.id: w.name for w in site.workers}

    # ---- incidents -------------------------------------------------------------------------------------------
    @r.get("/api/incidents")
    def list_incidents(phase: str = "", status: str = "", priority: str = "", camera: str = "", alert_id: str = "",
                       limit: int = 100, p=Depends(need("incident.read"))):
        m = mgr()
        m.tick()
        if alert_id:
            inc = m.get(m.incident_for_alert(alert_id) or "")
            return [compact(inc)] if inc else []
        rows = site.store.list_incidents(phase, status, priority, camera, limit=max(1, min(int(limit), 1000)))
        return [compact(i) for i in rows]

    @r.get("/api/incidents/{iid}")
    def get_incident(iid: str, p=Depends(need("incident.read"))):
        m = mgr()
        m.tick()
        inc = m.get(iid)
        if inc is None:
            raise HTTPException(404, f"unknown incident {iid}")
        detail = []
        for aid in inc["alert_ids"][-100:]:
            a = site.store.get_alert(aid)
            if not a:
                continue
            last = (a.get("events") or [{}])[-1]
            detail.append({"id": a["id"], "title": a.get("title"), "priority": a.get("priority"),
                           "status": a.get("status"), "cameras": a.get("cameras"), "zones": a.get("zones"),
                           "created_at": a.get("created_at"), "updated_at": a.get("updated_at"),
                           "disposition": a.get("disposition"), "suppressed": bool(a.get("suppressed")),
                           "inhibited_by": a.get("inhibited_by"), "n_events": len(a.get("events") or []),
                           "snapshot": (last.get("evidence") or {}).get("snapshot")})
        inc["alerts_detail"] = detail
        inc["notifications"] = site.store.list_notifications(iid, 50)
        return inc

    @r.patch("/api/incidents/{iid}")
    def patch_incident(iid: str, body: dict = Body(...), p=Depends(need("incident.read"))):
        m = mgr()
        if not isinstance(body, dict) or not body or set(body) - PATCH_KEYS:
            raise HTTPException(400, f"body may only contain {sorted(PATCH_KEYS)}")
        if "status" in body:
            perm = INC_STATUS_PERM.get(body["status"])
            if perm is None:
                raise HTTPException(400, f"status must be one of {sorted(INC_STATUS_PERM)}")
            if not p.can(perm):
                raise HTTPException(403, f"permission '{perm}' required")
        if "assignee" in body and not p.can("incident.assign"):
            raise HTTPException(403, "permission 'incident.assign' required")
        if "summary" in body and not p.can("feedback.write"):
            raise HTTPException(403, "permission 'feedback.write' required")
        if set(body) == {"note"}:
            raise HTTPException(400, "a note is saved together with a status change")
        try:
            return m.update(iid, body, p.username)
        except KeyError:
            raise HTTPException(404, f"unknown incident {iid}")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.post("/api/incidents/{iid}/merge")
    def merge(iid: str, body: dict = Body(...), p=Depends(need("incident.merge"))):
        m = mgr()
        src = body.get("sources") if isinstance(body, dict) else None
        if not isinstance(src, list) or not src or len(src) > 50 or not all(isinstance(s, str) for s in src):
            raise HTTPException(400, "body must be {\"sources\": [incident ids]}")
        try:
            return m.merge(iid, src, p.username)
        except KeyError as e:
            raise HTTPException(404, f"unknown incident {e.args[0]}")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.post("/api/incidents/{iid}/split")
    def split(iid: str, body: dict = Body(...), p=Depends(need("incident.merge"))):
        m = mgr()
        ids = body.get("alert_ids") if isinstance(body, dict) else None
        if not isinstance(ids, list) or not ids or not all(isinstance(s, str) for s in ids):
            raise HTTPException(400, "body must be {\"alert_ids\": [alert ids]}")
        try:
            return m.split(iid, ids, p.username)
        except KeyError:
            raise HTTPException(404, f"unknown incident {iid}")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.get("/api/incidents/{iid}/graph")
    def graph(iid: str, max_events: int = 60, p=Depends(need("incident.read"))):
        g = mgr().graph(iid, max(1, min(int(max_events), 200)))
        if g is None:
            raise HTTPException(404, f"unknown incident {iid}")
        return g

    @r.get("/api/incidents/{iid}/replay")
    def replay(iid: str, pre_s: float = 10.0, post_s: float = 10.0, cameras: str = "",
               p=Depends(need("evidence.view"))):
        """Media for a synchronised multi-camera replay of [incident start - pre_s, end + post_s]: per camera (the
        incident's cameras, predicted next cameras, and any in `cameras`, comma-separated) the recording segments and
        evidence clips with their absolute start times, plus the gaps and the incident's event markers."""
        m = mgr()
        if not (0 <= pre_s <= 600 and 0 <= post_s <= 600):
            raise HTTPException(400, "pre_s and post_s must be 0-600 s")
        inc = m.get(iid)
        if inc is None:
            raise HTTPException(404, f"unknown incident {iid}")
        extra = [c.strip() for c in cameras.split(",") if c.strip()][:16]
        return plan_for_site(site, inc, float(pre_s), float(post_s), extra)

    @r.get("/api/incidents/{iid}/audit")
    def incident_audit(iid: str, p=Depends(need("incident.read"))):
        mgr()
        return site.store.list_audit(iid)

    @r.get("/api/notifications")
    def notifications(incident_id: str = "", limit: int = 100, p=Depends(need("incident.read"))):
        mgr()
        return site.store.list_notifications(incident_id, max(1, min(int(limit), 1000)))

    # ---- hotspots -----------------------------------------------------------------------------------------------
    @r.get("/api/hotspots")
    def hotspots(days: float = 28, by: str = "zone", metric: str = "incidents", p=Depends(need("alert.read"))):
        if by not in ("zone", "camera") or metric not in ("incidents", "alerts") or not 0 < days <= 365:
            raise HTTPException(400, "by must be zone|camera, metric incidents|alerts, days in (0, 365]")
        if metric == "incidents" and getattr(site, "incidents", None) is None:
            metric = "alerts"
        m = getattr(site, "incidents", None)
        now = m.now() if m is not None else time.time()
        out = compute_hotspots(site.store, now, days, by, metric, names())
        try:  # where in each view the rules fired (same implementation as /api/analytics/hotspots)
            from .api_analytics import spatial_hotspots
            out["spatial"] = spatial_hotspots(site, now - days * 86400.0, now)
        except Exception as e:  # noqa: BLE001 - the spatial view is optional; the heatmap above must still render
            out["spatial_error"] = repr(e)[:200]
        return out

    # ---- natural-language investigation --------------------------------------------------------------------------
    @r.get("/api/investigate/schema")
    def investigate_schema(p=Depends(need("alert.read"))):
        d = schema(build_vocab(site))
        d["llm_configured"] = _llm(site) is not None
        return d

    @r.get("/api/investigate/changes")
    def investigate_changes(hours: float = 1.0, p=Depends(need("alert.read"))):
        """"What changed vs normal?": the last `hours` (0.25-24) vs this site's history at the same hours of the
        week (argonix/changes.py). Descriptive statistics only; raises no alert."""
        from .changes import what_changed
        return what_changed(site, hours=hours)

    @r.post("/api/investigate")
    def investigate(body: dict = Body(...), p=Depends(need("alert.read"))):
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        question: Optional[str] = body.get("question")
        query = body.get("query")
        if question is not None and not isinstance(question, str):
            raise HTTPException(400, "question must be a string")
        if query is not None and not isinstance(query, dict):
            raise HTTPException(400, "query must be an object")
        summarize = body.get("summarize", "auto")
        if summarize not in ("auto", "template"):
            raise HTTPException(400, "summarize must be auto|template")
        if query is not None and query.get("target") == "incidents" and getattr(site, "incidents", None) is None:
            raise HTTPException(400, "incidents are disabled on this site")
        try:
            return inv.ask(question or "", p.username, query, summarize)
        except QueryError as e:
            site.store.audit(p.username, "investigate_rejected", "", {"question": (question or "")[:300],
                                                                      "error": str(e)[:300]})
            raise HTTPException(400, f"invalid query: {e}")

    return r
