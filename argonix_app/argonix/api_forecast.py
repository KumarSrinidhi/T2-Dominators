"""Experimental analytics API: trajectory forecasts and fall-detector status.

Endpoints 404 (plain "Not Found", so the UI treats the feature as absent) when the feature is not enabled.
Forecasts are labelled predictions; fall events flow through the normal alerts API.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .forecast import NOTE


def forecast_router(site, need) -> APIRouter:
    r = APIRouter()

    def fc():
        an = getattr(site, "forecast", None)
        if an is None:
            raise HTTPException(404)
        return an

    @r.get("/api/forecast/tracks")
    def tracks(camera: str = "", p=Depends(need("alert.read"))):
        an = fc()
        return {"tracks": an.tracks(camera), "model": an.fc.info, "note": NOTE, "experimental": True}

    @r.get("/api/forecast/status")
    def status(p=Depends(need("alert.read"))):
        an = fc()
        return {**an.status(), "warnings": list(getattr(an, "warnings", []))}

    @r.get("/api/fall/status")
    def fall_status(p=Depends(need("alert.read"))):
        an = getattr(site, "fall", None)
        if an is None:
            raise HTTPException(404)
        return {**an.status(), "warnings": list((getattr(site, "fall_status", None) or {}).get("warnings", []))}

    return r
