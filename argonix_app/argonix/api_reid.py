"""Multi-camera Re-ID / global-ID API. Global IDs are opaque, probabilistic estimates (never names or identities).

Endpoints 404 (plain "Not Found", so the UI treats the feature as absent) when "reid" is not enabled.
  GET  /api/reid/tracks            alert.read         recent/active global IDs with per-camera segments + confidence
  GET  /api/reid/journey/{gid}     evidence.view      ordered segments with crops and the evidence of every link
  POST /api/reid/link              alert.acknowledge  {"a": seg, "b": seg} or {"gid_a": 3, "gid_b": 7} (audited)
  POST /api/reid/unlink            alert.acknowledge  {"segment": seg} - cut the link into that segment (audited)
  GET  /api/reid/metrics           alert.read
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from .global_ids import NOTE


def reid_router(site, need) -> APIRouter:
    r = APIRouter()

    def analyzer():
        an = getattr(site, "reid", None)
        if an is None:
            raise HTTPException(404)
        return an

    def run(fn, *a):
        try:
            return fn(*a)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0]) if e.args else "not found")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/api/reid/tracks")
    def tracks(active: bool = False, since_s: Optional[float] = None, limit: int = 200,
               p=Depends(need("alert.read"))):
        eng = analyzer().engine
        return {"identities": eng.identities(active_only=active, since_s=since_s, limit=max(1, min(limit, 1000))),
                "note": NOTE, "kind": "estimate"}

    viewed: dict = {}

    @r.get("/api/reid/journey/{gid}")
    def journey(gid: int, p=Depends(need("evidence.view"))):
        j = analyzer().engine.journey(gid)
        if j is None:
            raise HTTPException(404, "unknown or expired global ID")
        now = time.time()
        if now - viewed.get((p.username, j["gid"]), 0.0) > 300:  # person crops across cameras: audit (de-duplicated)
            viewed[(p.username, j["gid"])] = now
            site.store.audit(p.username, "reid_journey_viewed", "", {"gid": j["gid"]})
        return j

    @r.post("/api/reid/link")
    def link(body: dict = Body(...), p=Depends(need("alert.acknowledge"))):
        eng = analyzer().engine
        if body.get("gid_a") is not None and body.get("gid_b") is not None:
            try:
                ga, gb = int(body["gid_a"]), int(body["gid_b"])
            except (TypeError, ValueError):
                raise HTTPException(400, "gid_a and gid_b must be integers")
            res = run(eng.operator_merge, ga, gb, p.username)
            site.store.audit(p.username, "reid_merge", "", {"gids": [ga, gb], "result": res["gid"],
                                                             "note": str(body.get("note", ""))[:500]})
            return res
        a, b = body.get("a"), body.get("b")
        if not (isinstance(a, str) and isinstance(b, str)):
            raise HTTPException(400, 'body must be {"a": segment_id, "b": segment_id} or {"gid_a": n, "gid_b": m}')
        res = run(eng.operator_link, a, b, p.username)
        site.store.audit(p.username, "reid_link", "", {**res, "note": str(body.get("note", ""))[:500]})
        return res

    @r.post("/api/reid/unlink")
    def unlink(body: dict = Body(...), p=Depends(need("alert.acknowledge"))):
        seg = body.get("segment")
        if not isinstance(seg, str):
            raise HTTPException(400, 'body must be {"segment": segment_id}')
        res = run(analyzer().engine.operator_unlink, seg, p.username)
        site.store.audit(p.username, "reid_unlink", "", {**res, "note": str(body.get("note", ""))[:500]})
        return res

    @r.get("/api/reid/metrics")
    def metrics(p=Depends(need("alert.read"))):
        an = analyzer()
        return {**an.engine.metrics(), **an.status()}

    return r
