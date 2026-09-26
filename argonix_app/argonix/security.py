"""HTTP hardening shared by every route: CSRF defence for state-changing requests, security headers, and delivery of
the (sliding) session cookie. Pure ASGI so streaming responses (MJPEG) pass through untouched."""
from __future__ import annotations

import ipaddress
import re
from typing import Iterable
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, Response

COOKIE = "argonix_session"
CSRF_HEADER = "x-argonix-csrf"          # sent by the dashboard on every request; cross-site pages cannot set it
UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HOST_RE = re.compile(r"^(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")
_COOKIE_RE = re.compile(r"(?:^|;)\s*" + COOKIE + "=")
_DEFAULT_PORT = {"http": ":80", "https": ":443", "ws": ":80", "wss": ":443"}


def _netloc(netloc: str, scheme: str) -> str:
    n = netloc.lower()
    d = _DEFAULT_PORT.get(scheme)
    return n[: -len(d)] if d and n.endswith(d) else n


def origin_allowed(origin: str | None, host: str, trusted: Iterable[str] = ()) -> bool:
    """True when `origin` (scheme://host[:port]) names this server (the request's Host header) or a trusted origin.

    The scheme is not compared, so a TLS-terminating reverse proxy that forwards Host unchanged still works."""
    if not origin or origin == "null":
        return False
    if origin.rstrip("/") in trusted:
        return True
    try:
        u = urlsplit(origin)
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.netloc:
        return False
    o, h = _netloc(u.netloc, u.scheme), (host or "").lower()
    return bool(h) and (o == h or o == _netloc(h, u.scheme))


def csrf_problem(method: str, headers: Headers, trusted: Iterable[str] = ()) -> str:
    """'' when a request may proceed. Only state-changing methods are checked. A foreign Origin/Referer is always
    refused (browsers can attach cached Basic credentials too); when neither is sent (scripts, curl), cookie-carrying
    requests must add the X-Argonix-CSRF header, which a cross-site page cannot set without a CORS preflight."""
    if method not in UNSAFE:
        return ""
    host = headers.get("host", "")
    origin = headers.get("origin")
    if origin is not None:
        return "" if origin_allowed(origin, host, trusted) else "cross-origin request blocked (CSRF protection)"
    referer = headers.get("referer")
    if referer:
        try:
            u = urlsplit(referer)
            ref = f"{u.scheme}://{u.netloc}"
        except ValueError:
            ref = ""
        return "" if origin_allowed(ref, host, trusted) else "cross-site request blocked (CSRF protection)"
    if (_COOKIE_RE.search(headers.get("cookie", "")) and not headers.get("authorization")
            and not headers.get(CSRF_HEADER)):
        return "request without Origin/Referer needs the X-Argonix-CSRF header (CSRF protection)"
    return ""


def security_headers(path: str, host: str) -> dict[str, str]:
    ws = f" ws://{host} wss://{host}" if _HOST_RE.match(host or "") else ""
    h = {
        "content-security-policy": (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
            f"media-src 'self' blob: data:; connect-src 'self'{ws}; font-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"),
        "x-frame-options": "DENY",
        "x-content-type-options": "nosniff",
        "referrer-policy": "same-origin",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-resource-policy": "same-origin",
        "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    }
    if path.startswith("/api/") or path.startswith("/media/") or path == "/":
        h["cache-control"] = "no-store"
    return h


def cookie_header(value: str, max_age: int, secure: bool) -> str:
    r = Response()
    r.set_cookie(COOKIE, value, max_age=max(0, int(max_age)), httponly=True, samesite="strict", secure=secure, path="/")
    return r.headers["set-cookie"]


def client_ip(conn: HTTPConnection) -> str:
    return conn.client.host if conn.client else ""


def is_loopback_request(conn: HTTPConnection) -> bool:
    """Request made on this machine to a loopback name (defeats DNS rebinding and forwarded proxy traffic)."""
    try:
        if not ipaddress.ip_address(client_ip(conn)).is_loopback:
            return False
    except ValueError:
        return False
    if any(h in conn.headers for h in ("x-forwarded-for", "forwarded", "x-real-ip")):
        return False
    host = (urlsplit("//" + conn.headers.get("host", "")).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def set_session_cookie(conn: HTTPConnection, value: str, max_age: int) -> None:
    """Queue a Set-Cookie for the response (delivered by SecurityMiddleware, also on streamed/file responses)."""
    holder = conn.scope.get("argonix")
    if holder is not None:
        holder["cookie"] = (value, max_age)


class SecurityMiddleware:
    def __init__(self, app, trusted_origins: Iterable[str] = (), secure_cookie="auto"):
        self.app = app
        self.trusted = frozenset(o.rstrip("/") for o in trusted_origins)
        self.secure_cookie = secure_cookie

    def _secure(self, scope) -> bool:
        if self.secure_cookie in (True, False):
            return self.secure_cookie
        return scope.get("scheme") == "https"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        extra = security_headers(scope.get("path", ""), headers.get("host", ""))
        problem = csrf_problem(scope["method"], headers, self.trusted)
        if problem:
            resp = JSONResponse({"detail": problem}, status_code=403)
            resp.headers.update(extra)
            await resp(scope, receive, send)
            return
        holder: dict = {}
        scope["argonix"] = holder

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                mh = MutableHeaders(scope=message)
                for k, v in extra.items():
                    if k not in mh:
                        mh[k] = v
                if "cookie" in holder:
                    value, max_age = holder["cookie"]
                    mh.append("set-cookie", cookie_header(value, max_age, self._secure(scope)))
            await send(message)

        await self.app(scope, receive, send_wrapper)
