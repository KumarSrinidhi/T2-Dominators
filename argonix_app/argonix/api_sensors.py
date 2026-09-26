"""Access-control sensor API: signed/token ingestion, per-source status, recent reports.

POST /api/sensors/events is the only Argus route a panel talks to. It never uses the session cookie: the request is
either signed (Standard Webhooks HMAC, per-source secret) or carries an API token with `sensor.ingest`. All routes 404
when no sensor source is configured (site config "sensors").
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .sensors import MAX_BODY, RULES, IngestError

SIGNED_HEADERS = ("webhook-id", "webhook-timestamp", "webhook-signature")


def sensors_router(site, need, principal) -> APIRouter:
    r = APIRouter()

    def hub():
        h = getattr(site, "sensors", None)
        if h is None:
            raise HTTPException(404, 'no sensor sources configured (site config "sensors")')
        return h

    @r.post("/api/sensors/events", status_code=202)
    async def ingest(request: Request):
        h = hub()
        n = request.headers.get("content-length", "")
        if n.isdigit() and int(n) > MAX_BODY:
            raise HTTPException(413, f"body larger than {MAX_BODY} bytes")
        body = await request.body()
        try:
            if any(k in request.headers for k in SIGNED_HEADERS):
                return await run_in_threadpool(h.ingest_signed, request.headers, body)
            if not request.headers.get("authorization"):
                raise HTTPException(401, "sign the request (webhook-id, webhook-timestamp, webhook-signature) or send "
                                         "an API token whose role grants 'sensor.ingest'")
            p = principal(request, "sensor")  # Basic / Bearer; raises 401 on bad credentials
            if not p.can("sensor.ingest"):
                raise HTTPException(403, "permission 'sensor.ingest' required")
            return await run_in_threadpool(h.ingest_token, p.username, body)
        except IngestError as e:
            raise HTTPException(e.status, str(e))

    @r.get("/api/sensors")
    def sensors_status(p=Depends(need("alert.read"))):
        h = hub()
        return {"sources": h.status(), "warnings": list(h.warnings),
                "rules": {k: {"label": lbl, "severity": sev} for k, (lbl, sev) in RULES.items()},
                "note": ("Panel-reported states (deterministic source, not verified by video). Card ids are stored "
                         "only as keyed hashes.")}

    @r.get("/api/sensors/events")
    def sensor_events(source: str = "", limit: int = 100, p=Depends(need("alert.read"))):
        return hub().recent(source, limit)

    return r
