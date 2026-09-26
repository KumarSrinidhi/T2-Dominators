"""Restream API: WebRTC (WHEP) signalling proxy to MediaMTX, restreamer status, and recordings of both streams.

The browser never talks to MediaMTX's HTTP side: ``POST /api/cameras/{cam}/whep`` is authenticated and authorised
here (``evidence.view``; the CSRF middleware covers POST/PATCH/DELETE), then forwarded from localhost to
``<restream.webrtc>/cam_<cam>/whep``. Only the SDP answer and the WHEP headers (Location rewritten to this API, ETag,
Link, Accept-Patch) come back. Media then flows browser <-> MediaMTX over ICE/UDP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from .auth import Principal
from .restream import SESSION_RE, load_config, path_name

MAX_SDP = 64 * 1024
STREAMS = ("analytics", "main")


def recording_view(r: dict) -> dict:
    """A recordings row for the API: playable URL per stream, no absolute server paths for main segments."""
    out = dict(r)
    if r.get("stream") == "main":
        out["url"] = f"/api/recordings/{r['id']}/video"
        out["path"] = Path(str(r["path"])).name
    else:
        out["stream"] = r.get("stream") or "analytics"
        out["url"] = "/media/" + str(r["path"]).lstrip("/")
    return out


def restream_router(site, need: Callable[[str], Callable]) -> APIRouter:
    r = APIRouter()

    def service():
        rs = getattr(site, "restream", None)
        if rs is None:
            raise HTTPException(503, "restream (WebRTC live view) is not enabled on this server")
        return rs

    def camera(cam: str) -> dict:
        c = site.camera_config(cam)
        if c is None:
            raise HTTPException(404, f"unknown camera {cam}")
        return c

    def check_session(sid: str) -> str:
        if not SESSION_RE.match(sid):
            raise HTTPException(400, "invalid WHEP session id")
        return sid

    def reply(res) -> Response:
        if res.error:
            raise HTTPException(res.status, res.error)
        return Response(res.body, status_code=res.status, media_type=res.content_type or None, headers=res.headers)

    @r.get("/api/restream")
    def status(p: Principal = Depends(need("alert.read"))):
        rs = getattr(site, "restream", None)
        if rs is None:
            return {"enabled": False, "mode": "off", "cameras": {}}
        return rs.status()

    @r.post("/api/cameras/{cam}/whep")
    async def whep_offer(cam: str, request: Request, p: Principal = Depends(need("evidence.view"))):
        rs = service()
        info = rs.camera_info(camera(cam))
        if not info["available"]:
            raise HTTPException(409, f"WebRTC not available for {cam}: {info['reason']}")
        if (request.headers.get("content-type") or "").split(";")[0].strip().lower() != "application/sdp":
            raise HTTPException(415, "WHEP offers must be Content-Type: application/sdp")
        body = await request.body()
        if not body or len(body) > MAX_SDP:
            raise HTTPException(400, "empty or oversized SDP offer")
        res = await rs.whep_offer(cam, body)
        site.store.audit(p.username, "live_view_webrtc", "", {"camera": cam, "path": path_name(cam),
                                                              "ok": not res.error, "error": res.error})
        return reply(res)

    @r.patch("/api/cameras/{cam}/whep/{sid}")
    async def whep_patch(cam: str, sid: str, request: Request, p: Principal = Depends(need("evidence.view"))):
        rs = service()
        camera(cam)
        body = await request.body()
        if len(body) > MAX_SDP:
            raise HTTPException(400, "oversized ICE fragment")
        return reply(await rs.whep_patch(cam, check_session(sid), body, request.headers.get("content-type", ""),
                                         request.headers.get("if-match", "")))

    @r.delete("/api/cameras/{cam}/whep/{sid}")
    async def whep_delete(cam: str, sid: str, p: Principal = Depends(need("evidence.view"))):
        rs = service()
        camera(cam)
        res = await rs.whep_delete(cam, check_session(sid))
        if res.error and res.status != 404:
            raise HTTPException(res.status, res.error)
        return {"ok": True}

    # ---- recordings (both streams) ------------------------------------------------------------------------------
    @r.get("/api/recordings")
    def recordings(camera: str = "", stream: str = "", limit: int = 0, p: Principal = Depends(need("evidence.view"))):
        if stream and stream not in STREAMS:
            raise HTTPException(400, f"stream must be one of {list(STREAMS)}")
        return [recording_view(x) for x in site.store.list_recordings(camera, stream, max(0, min(int(limit), 100000)))]

    @r.get("/api/recordings/at")
    def recording_at(camera: str, t: float, stream: str = "main", pre_s: float = 4.0,
                     p: Principal = Depends(need("evidence.view"))):
        """The segment of ``camera`` covering ``t - pre_s`` (falls back to ``t``) and the offset to seek to."""
        if stream not in STREAMS:
            raise HTTPException(400, f"stream must be one of {list(STREAMS)}")
        want = t - max(0.0, float(pre_s))
        rec = site.store.recording_at(camera, want, stream) or site.store.recording_at(camera, t, stream)
        if rec is None:
            raise HTTPException(404, f"no {stream}-stream recording of {camera} covers that time")
        return {**recording_view(rec), "offset_s": round(max(0.0, want - rec["start"]), 2), "t": t}

    @r.get("/api/recordings/{rid}/video")
    def recording_video(rid: int, p: Principal = Depends(need("evidence.view"))):
        rec = site.store.get_recording(rid)
        if rec is None:
            raise HTTPException(404)
        if rec["stream"] != "main":   # analytics segments live under the data dir: /media/ serves them
            root, f = site.data.resolve(), (site.data / rec["path"]).resolve()
        else:
            root = load_config(site.cfg, site.data)["record"]["root"]
            f = Path(rec["path"]).resolve()
        if root not in f.parents or f.suffix.lower() != ".mp4" or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, media_type="video/mp4")

    return r
