"""FastAPI operator API + WebSocket + static dashboard, with authentication and role-based access control.

Auth (argonix.auth / api_auth / security): session cookies for the dashboard, HTTP Basic and API tokens for script
clients on /api/ and /media/, first-run setup, optional explicit open demo mode, CSRF defence and security headers."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import psutil
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .api_analytics import analytics_router
from .api_auth import auth_router
from .api_calib import calib_router
from .api_cameras import camera_router
from .api_handoff import handoff_router
from .api_incidents import incidents_router
from .api_nodes import nodes_router
from .api_reid import reid_router
from .api_restream import restream_router
from .auth import IDLE_S, LOGIN_FAILED, ROLES, SESSION_S, STATUS_PERM, TOKEN_PREFIX, Auth, Principal
from .vlm import VlmBusy, vlm_slot
from .feedback import DISPOSITIONS, export_bundle, precision_by_rule, verify_bundle
from .incidents import compact
from .pipeline import Site
from .rules import LABEL
from .security import (COOKIE, UNSAFE, SecurityMiddleware, client_ip, origin_allowed, set_session_cookie)
from .workflow import TransitionError, plan as plan_transition, public_config as workflow_config

STATIC = Path(__file__).parent / "static"
# Routes a principal flagged "must change password" may still use.
MUST_CHANGE_OK = {"/api/me", "/api/me/password", "/api/logout", "/api/session/ping", "/api/auth/status"}
BASIC_CHALLENGE = {"WWW-Authenticate": 'Basic realm="Argus API", charset="UTF-8"'}


def create_app(site: Site, explainer=None, auth: Optional[Auth] = None, open_demo: Optional[bool] = None,
               auth_options: Optional[dict] = None) -> FastAPI:
    """`open_demo`: None = site config `auth.open_demo` (default off). `auth_options` override the site config's
    `auth` section (idle_timeout_min, session_max_h, basic_auth, localhost_setup_without_code, trusted_origins,
    secure_cookie)."""
    app = FastAPI(title="Argus")
    started = time.time()
    acfg = {**(site.cfg.get("auth") or {}), **(auth_options or {})}
    if auth is None:
        auth = Auth(site.config_path.resolve().parent / "users.json", site.data / "secret.key",
                    idle_timeout_s=float(acfg.get("idle_timeout_min", IDLE_S / 60)) * 60,
                    session_max_s=float(acfg.get("session_max_h", SESSION_S / 3600)) * 3600,
                    basic_enabled=bool(acfg.get("basic_auth", True)),
                    localhost_setup=bool(acfg.get("localhost_setup_without_code", True)))
    auth.open_demo = bool(acfg.get("open_demo", False)) if open_demo is None else bool(open_demo)
    app.state.auth = auth
    app.add_middleware(SecurityMiddleware, trusted_origins=acfg.get("trusted_origins", []),
                       secure_cookie=acfg.get("secure_cookie", "auto"))
    if auth.setup_required:
        auth.ensure_setup_code(announce=True)
        if auth.open_demo:
            print("OPEN DEMO MODE: no sign-in; anyone who can reach this server has full access. "
                  "Creating the first admin account (dashboard or CLI) turns sign-in on.", flush=True)

    def header_principal(authz: str, path: str, ip: str) -> Principal:
        """HTTP Basic or Bearer (API token) for script clients. Never for the HTML page."""
        if not (path.startswith("/api/") or path.startswith("/media/") or path == "/ws"):
            raise HTTPException(401, "the Authorization header is only accepted on /api/ and /media/ routes")
        scheme, _, cred = authz.strip().partition(" ")
        cred = cred.strip()
        if scheme.lower() == "basic":
            try:
                user, sep, pw = base64.b64decode(cred, validate=True).decode("utf-8").partition(":")
            except (ValueError, UnicodeDecodeError):
                user, sep, pw = "", "", ""
            if not sep:
                raise HTTPException(401, "malformed Basic credentials", headers=BASIC_CHALLENGE)
            if pw.startswith(TOKEN_PREFIX) and user in ("", "token"):   # clients that can only do Basic
                scheme, cred = "bearer", pw
            else:
                if not auth.basic_enabled:
                    raise HTTPException(401, "HTTP Basic auth is disabled on this server; use an API token")
                was_blocked = auth.brake.blocked(user, ip)
                p = auth.basic(user, pw, ip) if auth.enabled else None
                if p is None:
                    if not was_blocked:
                        site.store.audit(user[:64], "login_failed", "", {"ip": ip, "via": "basic"})
                    raise HTTPException(401, LOGIN_FAILED, headers=BASIC_CHALLENGE)
                return p
        if scheme.lower() == "bearer":
            p = auth.bearer(cred, ip)
            if p is None:
                raise HTTPException(401, "invalid, expired or revoked API token",
                                    headers={"WWW-Authenticate": 'Bearer realm="Argus API"'})
            return p
        raise HTTPException(401, "unsupported Authorization scheme (use Basic or Bearer)")

    def principal(request: Request, x_operator: str = Header(default="operator")) -> Principal:
        authz = request.headers.get("authorization")
        if authz:
            p = header_principal(authz, request.url.path, client_ip(request))
        else:
            # Only state-changing requests (user actions) and the dashboard's activity ping slide the idle timer;
            # background polling (GET) never keeps a session alive.
            p, reason, refreshed = auth.from_cookie(request.cookies.get(COOKIE), activity=request.method in UNSAFE)
            if p is None:
                raise HTTPException(401, reason)
            if refreshed:
                set_session_cookie(request, refreshed, auth.session_max_age(refreshed))
        if p.must_change and request.url.path not in MUST_CHANGE_OK:
            raise HTTPException(403, "password change required: set a new password first")
        if p.open_mode:  # demo mode: name comes from the UI's operator box, unauthenticated
            p = Principal(f"{x_operator[:48]} (unauthenticated)", p.roles, open_mode=True, via="open")
        return p

    def need(perm: str):
        def dep(p: Principal = Depends(principal)) -> Principal:
            if not p.can(perm):
                raise HTTPException(403, f"permission '{perm}' required")
            return p
        return dep

    def worker(cam: str):
        w = next((w for w in site.workers if w.id == cam), None)
        if not w:
            raise HTTPException(404, f"unknown camera {cam}")
        return w

    # Who looked at what: live views and evidence files go into the hash-chained audit log (privacy: access to
    # footage is itself recorded). One row per user and camera/file per `access_log_s` (default 10 min), so the
    # once-a-second snapshot polling of the Live wall does not flood the chain. Site config "privacy":
    # {"access_log": false} turns it off.
    pcfg = site.cfg.get("privacy") or {}
    access_log_on = bool(pcfg.get("access_log", True))
    access_log_s = float(pcfg.get("access_log_s", 600))
    access_seen: dict[tuple, float] = {}

    def log_access(p: Principal, action: str, key: str, details: dict) -> None:
        if not access_log_on:
            return
        now = time.time()
        k = (p.username, action, key)
        if now - access_seen.get(k, -1e18) < access_log_s:
            return
        access_seen[k] = now
        if len(access_seen) > 5000:  # forget entries older than the window
            for kk in [kk for kk, v in access_seen.items() if now - v >= access_log_s]:
                access_seen.pop(kk, None)
        site.store.audit(p.username, action, "", details)

    app.include_router(auth_router(site, auth, principal))
    app.include_router(calib_router(site, need))
    app.include_router(camera_router(site, need))
    app.include_router(handoff_router(site, need))
    app.include_router(incidents_router(site, need))
    app.include_router(analytics_router(site, need))
    app.include_router(reid_router(site, need))
    app.include_router(restream_router(site, need))  # WebRTC (WHEP) proxy, restreamer status, recordings of both streams
    app.include_router(nodes_router(site, need, explainer))
    from .api_forecast import forecast_router  # experimental: trajectory forecasts + fall-detector status
    app.include_router(forecast_router(site, need))
    from .api_reports import reports_router  # incident report, assistive incident summary, shift handover
    app.include_router(reports_router(site, need, explainer))
    from .api_sensors import sensors_router  # access-control / door events (signed webhook or sensor.ingest token)
    app.include_router(sensors_router(site, need, principal))
    from .api_demo import demo_router  # demo scenarios + push-notification status/test
    app.include_router(demo_router(site, need, lambda: auth.open_demo))

    # ---- dashboard shell (no auth: it only loads the JS, which then asks /api/auth/status and /api/me) -----------
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ---- config / live ---------------------------------------------------------------------
    @app.get("/api/config")
    def config(p: Principal = Depends(need("alert.read"))):
        return {"site": site.cfg.get("site"), "map": site.cfg.get("map", {}),
                "analytics_fps": site.cfg.get("analytics_fps", 5), "explain_enabled": explainer is not None,
                "auth_enabled": auth.enabled, "rule_labels": dict(LABEL), "workflow": workflow_config(site.cfg),
                "restream": getattr(site, "restream", None) is not None,
                "cameras": [{"id": w.id, "name": w.name, "calibration": w.calib_info,
                             "zones": [{"id": z.id, "name": z.name, "polygon": z.polygon, "rules": list(z.rules),
                                        "criticality": z.criticality, "direction": z.direction} for z in w.zones]}
                            for w in site.workers]}

    @app.get("/api/cameras/{cam}/latest.jpg")
    def latest(cam: str, p: Principal = Depends(need("evidence.view"))):
        w = worker(cam)
        if not w.latest_jpeg:
            raise HTTPException(503, "no frame yet")
        log_access(p, "live_view", cam, {"camera": cam, "via": "snapshot"})
        return Response(w.latest_jpeg, media_type="image/jpeg")

    @app.get("/api/cameras/{cam}/mjpeg")
    async def mjpeg(cam: str, p: Principal = Depends(need("evidence.view"))):
        w = worker(cam)
        period = 1.0 / float(site.cfg.get("analytics_fps", 5))
        log_access(p, "live_view", cam, {"camera": cam, "via": "mjpeg"})

        async def gen():
            last, checked = None, time.monotonic()
            while True:
                if time.monotonic() - checked > 5:   # end the stream once the session is revoked/expired
                    checked = time.monotonic()
                    if auth.revalidate(p):
                        return
                j = w.latest_jpeg
                if j is not None and j is not last:
                    last = j
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                await asyncio.sleep(period / 2)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/cameras/{cam}/inject")
    def inject(cam: str, body: dict = Body(...), p: Principal = Depends(need("settings.edit"))):
        """Demo/test tamper injection: black | blur | cover | shift | freeze for N seconds."""
        try:
            worker(cam).inject(body.get("fault", ""), float(body.get("seconds", 10)))
        except ValueError as e:
            raise HTTPException(400, str(e))
        site.store.audit(p.username, f"tamper_test:{body['fault']}", "", {"camera": cam})
        return {"camera": cam, "fault": body["fault"], "seconds": float(body.get("seconds", 10))}

    @app.get("/media/{path:path}")
    def media(path: str, p: Principal = Depends(need("evidence.view"))):
        root = site.data.resolve()
        f = (root / path).resolve()
        if root not in f.parents or not f.is_file() or f.suffix.lower() not in {".jpg", ".mp4"}:
            raise HTTPException(404)
        rel = f.relative_to(root).as_posix()
        log_access(p, "media_read", rel, {"path": rel})
        return FileResponse(f)

    # ---- alerts ----------------------------------------------------------------------------
    @app.get("/api/alerts")
    def alerts(status: str = "", priority: str = "", camera: str = "", rule: str = "", limit: int = 200,
               review: str = "", p: Principal = Depends(need("alert.read"))):
        """status / priority accept comma lists ("P1,P2"); review=second: only alerts flagged "unclear"."""
        return site.store.list_alerts(status, priority, camera, rule, limit=limit, second_review=review == "second")

    @app.get("/api/alerts/{aid}")
    def alert(aid: str, p: Principal = Depends(need("alert.read"))):
        a = site.store.get_alert(aid)
        if not a:
            raise HTTPException(404)
        return a

    @app.post("/api/alerts/{aid}/status")
    def set_status(aid: str, body: dict = Body(...), p: Principal = Depends(principal)):
        """Operator status change through the workflow state machine (argonix.workflow): 409 on an illegal
        transition, 403 without the permission, 400 on a missing disposition / escalation reason."""
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        with site.lock:
            a = site.store.get_alert(aid)
            if not a:
                raise HTTPException(404)
            try:
                ch = plan_transition(a, body, p.can, STATUS_PERM, ROLES, p.username)
            except TransitionError as e:
                raise HTTPException(e.code, e.msg)
            d = site.store.set_status(aid, ch["to"], ch["note"], ch["disposition"], p.username, ch["fields"],
                                      ch["details"])
            for oa in site.correlator.open:
                if oa.id == aid:
                    oa.status = ch["to"]
            if ch["disposition"] and getattr(site, "track_record", None) is not None:
                site.track_record.invalidate()  # the next alert's "Why P?" counts this decision
        return d

    @app.get("/api/vlm")
    def vlm_info(p: Principal = Depends(need("alert.read"))):
        if explainer is None or not hasattr(explainer, "models"):
            return {"enabled": explainer is not None, "models": [], "health": None}
        try:
            return {"enabled": True, "models": explainer.models(), "health": explainer.health()}
        except Exception as e:
            return {"enabled": True, "models": [], "health": {"error": repr(e)}}

    @app.post("/api/alerts/{aid}/explain")
    def explain(aid: str, body: dict = Body(default={}), p: Principal = Depends(need("alert.acknowledge"))):
        if explainer is None:
            raise HTTPException(503, "VLM explainer not configured")
        a = site.store.get_alert(aid)
        if not a:
            raise HTTPException(404)
        model = (body or {}).get("model")
        try:
            with vlm_slot(site):
                text = explainer(a, site.data, model) if model else explainer(a, site.data)
        except VlmBusy as e:
            raise HTTPException(429, str(e))
        except Exception as e:
            raise HTTPException(502, f"VLM failed: {e}")
        with site.lock:
            a = site.store.get_alert(aid)
            a["explanation"] = text
            site.store.upsert_alert(a)
        site.store.audit(p.username, "vlm_explain", aid, {"model": model})
        return a

    @app.post("/api/alerts/{aid}/viewed")
    def viewed(aid: str, p: Principal = Depends(need("evidence.view"))):
        site.store.audit(p.username, "evidence_viewed", aid)
        return {"ok": True}

    @app.get("/api/alerts/{aid}/audit")
    def alert_audit(aid: str, p: Principal = Depends(need("alert.read"))):
        return site.store.list_audit(aid)

    @app.get("/api/audit")
    def audit_log(limit: int = 200, p: Principal = Depends(need("audit.read"))):
        return site.store.list_audit(limit=limit)

    @app.get("/api/audit/verify")
    def audit_verify(p: Principal = Depends(need("audit.read"))):
        """Recompute the audit hash chain: {ok, rows, first_bad_seq, head}."""
        return site.store.verify_audit()

    @app.get("/api/alerts/{aid}/export")
    def export(aid: str, anonymise: int = 0, p: Principal = Depends(need("evidence.export"))):
        a = site.store.get_alert(aid)
        if not a:
            raise HTTPException(404)
        site.store.audit(p.username, "evidence_exported", aid, {"anonymised_copies": bool(anonymise)})
        fm = (site.cfg.get("privacy") or {}).get("face_model")
        fm = (Path(fm) if Path(fm).is_absolute() else site.config_path.resolve().parent.parent / fm) if fm else None
        data = export_bundle(a, site.store.list_audit(aid), site.data, anonymise=bool(anonymise),
                             audit_chain=site.store.audit_head(), face_model=fm, key=auth.secret)
        suffix = "_anonymised" if anonymise else ""
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="argus_{aid}{suffix}.zip"'})

    @app.post("/api/evidence/verify")
    async def evidence_verify(request: Request, p: Principal = Depends(need("evidence.export"))):
        """Check an exported bundle (raw zip body): hashes and chain, and the signature against this server's key."""
        body = await request.body()
        if len(body) > 512 * 2**20:
            raise HTTPException(413, "bundle too large")
        try:
            chain_ok = verify_bundle(body)
            signed_ok = verify_bundle(body, key=auth.secret) if chain_ok else False
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                m = json.loads(z.read("manifest.json"))
        except (zipfile.BadZipFile, KeyError, ValueError):
            raise HTTPException(400, "not an Argus evidence bundle")
        site.store.audit(p.username, "evidence_verified", str(m.get("alert_id", "")),
                         {"chain_ok": chain_ok, "signature_ok": signed_ok})
        return {"alert_id": m.get("alert_id"), "chain_ok": chain_ok, "signed": "signature" in m,
                "signature_ok": signed_ok, "missing": m.get("missing", [])}

    # ---- legal hold + retention (argonix/retention.py) ----------------------------------------------------------
    @app.post("/api/alerts/{aid}/hold")
    def hold(aid: str, body: dict = Body(default={}), p: Principal = Depends(need("evidence.hold"))):
        reason = str((body or {}).get("reason", ""))[:500]
        with site.lock:
            d = site.store.set_hold(aid, p.username, reason)
        if not d:
            raise HTTPException(404)
        site.store.audit(p.username, "legal_hold_set", aid, {"reason": reason})
        return d

    @app.delete("/api/alerts/{aid}/hold")
    def release_hold(aid: str, p: Principal = Depends(need("evidence.hold"))):
        with site.lock:
            prev = site.store.get_alert(aid)
            d = site.store.release_hold(aid)
        if not d:
            raise HTTPException(404)
        site.store.audit(p.username, "legal_hold_released", aid,
                         {"held_since": (prev.get("legal_hold") or {}).get("t"),
                          "held_by": (prev.get("legal_hold") or {}).get("actor")})
        return d

    @app.get("/api/retention")
    def retention(p: Principal = Depends(need("alert.read"))):
        return site.retention.status()

    @app.post("/api/retention/sweep")
    def retention_sweep(p: Principal = Depends(need("settings.edit"))):
        return site.retention.run_once(actor=p.username)

    # ---- insights / system -----------------------------------------------------------------
    @app.get("/api/feedback/stats")
    def feedback_stats(p: Principal = Depends(need("alert.read"))):
        return {"dispositions": DISPOSITIONS, "rules": precision_by_rule(site.store.list_alerts(limit=100000))}

    @app.get("/api/baseline")
    def baseline(p: Principal = Depends(need("alert.read"))):
        b = site.baseline
        return {"min_samples": b.min_samples, "z_threshold": b.z, "bucket_s": b.bucket_s, "buckets": b.summary(),
                "cusum": b.summary_cusum(), "rate_bucket_s": b.rate_bucket_s, "event_rates": b.summary_rates(),
                "dwell": b.summary_dwell()}

    @app.get("/api/stats")
    def stats(p: Principal = Depends(need("alert.read"))):
        proc = psutil.Process()
        return {"uptime_s": round(time.time() - started), "cpu_percent": psutil.cpu_percent(None),
                "ram_percent": psutil.virtual_memory().percent,
                "process_rss_mb": round(proc.memory_info().rss / 2**20),
                "model": site.correlator.model_info,
                "cameras": {w.id: {**w.stats, "tracks": len(w.tracks)} for w in site.workers},
                "poll_sensors": {ps.id: ps.status for ps in getattr(site, "poll_sensors", [])}}

    trusted = frozenset(o.rstrip("/") for o in acfg.get("trusted_origins", []))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        origin = sock.headers.get("origin")
        if origin is not None and not origin_allowed(origin, sock.headers.get("host", ""), trusted):
            await sock.close(code=4403)   # cross-site WebSocket hijacking guard
            return
        authz = sock.headers.get("authorization")
        try:
            p = header_principal(authz, "/ws", client_ip(sock)) if authz else auth.from_cookie(sock.cookies.get(COOKIE))[0]
        except HTTPException:
            p = None
        if p is None or p.must_change or not p.can("alert.read"):
            await sock.close(code=4401)
            return
        await sock.accept()
        seq = site.store.seq
        inc_seq = getattr(site.store, "inc_seq", 0)
        want_incidents = p.can("incident.read") and getattr(site, "incidents", None) is not None
        n = 0
        try:
            while True:
                n += 1
                if n % 10 == 0 and auth.revalidate(p):   # every ~5 s: revoked / idle / disabled -> close
                    await sock.close(code=4401)
                    return
                for _ in range(20):  # oldest change first, in pages: a burst of >50 used to skip the older ones
                    new = site.store.alerts_since(seq, 50)
                    for a in new:
                        seq = a["seq"]
                        await sock.send_json({"type": "alert", "alert": a})
                    if len(new) < 50:
                        break
                if want_incidents and site.store.inc_seq > inc_seq:   # incident created / changed (compact view)
                    for inc in site.store.incidents_since(inc_seq, 50):
                        inc_seq = max(inc_seq, inc["seq"])
                        await sock.send_json({"type": "incident", "incident": compact(inc)})
                await sock.send_json({"type": "tick", "t": time.time(), "tracks": site.tracks(),
                                      "stats": {w.id: w.stats for w in site.workers}})
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
