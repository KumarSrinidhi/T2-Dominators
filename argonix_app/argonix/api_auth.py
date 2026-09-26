"""Accounts API: first-run setup, sign-in/out, self-service password + sessions, user admin, API tokens.

Every change is written to the audit log; passwords, token secrets and setup codes never are."""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from .auth import LOGIN_FAILED, MIN_PASSWORD, ROLE_PERMISSIONS, Auth, AuthError, Principal
from .security import COOKIE, client_ip, is_loopback_request, set_session_cookie


def auth_router(site, auth: Auth, principal: Callable) -> APIRouter:
    r = APIRouter()
    audit = site.store.audit

    def err(e: AuthError):
        return HTTPException(e.status, e.detail)

    def account(p: Principal = Depends(principal)) -> Principal:
        """A real signed-in person (session or Basic): not open demo mode, not an API token."""
        if p.open_mode:
            raise HTTPException(403, "not available in open demo mode: create the first admin account to turn sign-in on")
        if p.via == "token":
            raise HTTPException(403, "API tokens cannot manage accounts, passwords or tokens")
        return p

    def admin(p: Principal = Depends(account)) -> Principal:
        if not p.can("user.admin"):
            raise HTTPException(403, "permission 'user.admin' required")
        return p

    def issue(request: Request, token: str) -> None:
        set_session_cookie(request, token, auth.session_max_age(token))

    # ---- public ------------------------------------------------------------------------------------------------
    @r.get("/api/auth/status")
    def status(request: Request):
        setup = auth.setup_required
        return {"setup_required": setup, "open_demo": bool(auth.open_demo and setup),
                "setup_code_required": setup and not (auth.localhost_setup and is_loopback_request(request)),
                "password_min_length": MIN_PASSWORD, "basic_auth": auth.basic_enabled,
                "idle_timeout_s": auth.idle_timeout_s, "session_max_s": auth.session_max_s,
                "https": request.url.scheme == "https",
                "roles": {k: sorted(v) for k, v in ROLE_PERMISSIONS.items()}}

    @r.post("/api/setup")
    def setup(request: Request, body: dict = Body(...)):
        ip, local = client_ip(request), is_loopback_request(request)
        username = str(body.get("username", "")).strip()
        try:
            token = auth.setup(username, str(body.get("password", "")), body.get("roles") or ["admin"],
                               str(body.get("setup_code", "")), local, ip)
        except AuthError as e:
            if e.status in (403, 429):
                audit("(setup)", "setup_failed", "", {"ip": ip, "reason": e.detail})
            raise err(e)
        issue(request, token)
        rec = auth.users.get(username) or {}
        audit(username, "setup_completed", "", {"ip": ip, "roles": rec.get("roles", []),
                                                "via": "loopback" if (local and auth.localhost_setup) else "setup code"})
        return {"ok": True, "username": username}

    @r.post("/api/login")
    def login(request: Request, body: dict = Body(...)):
        user, ip = str(body.get("username", ""))[:128], client_ip(request)
        token = auth.login(user, str(body.get("password", "")), ip) if auth.enabled else None
        if not token:
            audit(user[:64], "login_failed", "", {"ip": ip})
            raise HTTPException(401, LOGIN_FAILED)
        issue(request, token)
        audit(user, "login", "", {"ip": ip})
        return {"ok": True, "must_change": bool((auth.users.get(user) or {}).get("must_change"))}

    @r.post("/api/logout")
    def logout(request: Request):
        """Always clears the cookie; revokes the session server-side when it is still valid."""
        p, _, _ = auth.from_cookie(request.cookies.get(COOKIE))
        if p is not None and not p.open_mode:
            auth.logout(p)
            audit(p.username, "logout", "", {"ip": client_ip(request)})
        set_session_cookie(request, "", 0)
        return {"ok": True}

    # ---- self-service --------------------------------------------------------------------------------------------
    @r.get("/api/me")
    def me(p: Principal = Depends(principal)):
        rec = None if p.open_mode else auth.users.get(p.username)
        return {"username": p.username, "roles": list(p.roles), "permissions": sorted(p.perms),
                "open_mode": p.open_mode, "auth": p.via, "must_change": p.must_change,
                "last_login": (rec or {}).get("last_login"), "token_id": p.token_id or None,
                "session": auth.session_info(p)}

    @r.post("/api/session/ping")
    def ping(p: Principal = Depends(principal)):
        """User activity heartbeat from the dashboard: slides the idle timer (background polling never does)."""
        return {"ok": True, "session": auth.session_info(p)}

    @r.post("/api/me/password")
    def change_password(request: Request, body: dict = Body(...), p: Principal = Depends(account)):
        ip = client_ip(request)
        try:
            token = auth.change_password(p.username, str(body.get("current", "")), str(body.get("new", "")), ip)
        except AuthError as e:
            if e.status in (403, 429):
                audit(p.username, "password_change_failed", "", {"ip": ip})
            raise err(e)
        if p.via == "session":
            issue(request, token)
        audit(p.username, "password_changed", "", {"ip": ip, "other_sessions_signed_out": True})
        return {"ok": True}

    @r.post("/api/me/sessions/revoke")
    def revoke_sessions(request: Request, p: Principal = Depends(account)):
        """Sign out everywhere (this browser included) by bumping the account's session epoch."""
        auth.revoke_all(p.username)
        set_session_cookie(request, "", 0)
        audit(p.username, "sessions_revoked", "", {"ip": client_ip(request), "scope": "all"})
        return {"ok": True}

    # ---- user administration -------------------------------------------------------------------------------------
    @r.get("/api/users")
    def list_users(p: Principal = Depends(admin)):
        return auth.list_users()

    @r.post("/api/users")
    def create_user(body: dict = Body(...), p: Principal = Depends(admin)):
        name = str(body.get("username", "")).strip()
        try:
            u = auth.create_user(name, body.get("roles"), str(body.get("password", "")),
                                 bool(body.get("must_change", True)))
        except AuthError as e:
            raise err(e)
        audit(p.username, "user_created", "", {"user": name, "roles": u["roles"], "must_change": u["must_change"]})
        return u

    @r.put("/api/users/{name}")
    def update_user(name: str, body: dict = Body(...), p: Principal = Depends(admin)):
        try:
            u, changes = auth.update_user(p.username, name, body.get("roles"), body.get("disabled"))
            if body.get("revoke_sessions"):
                auth.revoke_all(name)
                changes["sessions_revoked"] = True
        except AuthError as e:
            raise err(e)
        if changes:
            audit(p.username, "user_updated", "", {"user": name, **changes})
        return u

    @r.delete("/api/users/{name}")
    def delete_user(name: str, p: Principal = Depends(admin)):
        try:
            auth.delete_user(p.username, name)
        except AuthError as e:
            raise err(e)
        audit(p.username, "user_deleted", "", {"user": name})
        return {"ok": True}

    @r.post("/api/users/{name}/reset")
    def reset_password(name: str, body: dict = Body(...), p: Principal = Depends(admin)):
        try:
            u = auth.reset_password(p.username, name, str(body.get("password", "")), bool(body.get("must_change", True)))
        except AuthError as e:
            raise err(e)
        audit(p.username, "user_password_reset", "", {"user": name, "must_change": u["must_change"],
                                                      "sessions_revoked": True})
        return u

    # ---- API tokens ------------------------------------------------------------------------------------------------
    @r.get("/api/tokens")
    def list_tokens(all: bool = False, p: Principal = Depends(account)):
        if all:
            if not p.can("user.admin"):
                raise HTTPException(403, "permission 'user.admin' required to list everyone's tokens")
            return auth.list_tokens()
        return auth.list_tokens(p.username)

    @r.post("/api/tokens")
    def create_token(body: dict = Body(...), p: Principal = Depends(account)):
        try:
            info, secret = auth.create_token(p.username, body.get("name"), body.get("roles"), body.get("expires_days"))
        except AuthError as e:
            raise err(e)
        audit(p.username, "token_created", "", {"token_id": info["id"], "name": info["name"], "roles": info["roles"],
                                                "expires_at": info["expires_at"]})
        return {**info, "token": secret, "note": "Shown once. Only a SHA-256 hash is stored on the server."}

    @r.delete("/api/tokens/{tid}")
    def revoke_token(tid: str, p: Principal = Depends(account)):
        try:
            info = auth.revoke_token(tid, owner=None if p.can("user.admin") else p.username)
        except AuthError as e:
            raise err(e)
        audit(p.username, "token_revoked", "", {"token_id": tid, "user": info["user"], "name": info["name"]})
        return {"ok": True}

    return r
