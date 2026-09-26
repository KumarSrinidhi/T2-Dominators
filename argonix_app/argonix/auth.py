"""Authentication + role-based access (roles/permissions from manual §13 D-13.7 ROLE_PERMISSIONS).

Users live in a git-ignored JSON file (configs/users.json) with PBKDF2-SHA256 hashes. Browser sessions are
HMAC-signed cookies that carry a per-user *session epoch*: bumping the epoch ("sign out all sessions", a password
change or reset) invalidates every older cookie. Sessions expire after an idle timeout (default 30 min, slid only by
user activity) and an absolute maximum (default 12 h).

No users yet -> FIRST-RUN SETUP: the dashboard asks for the first admin account. `POST /api/setup` needs a one-time
setup code that is printed to the console and written to <data>/setup_code.txt (loopback requests may skip it).
OPEN DEMO MODE (every request is an anonymous admin-equivalent) only runs when explicitly enabled (`--open-demo` /
`"auth": {"open_demo": true}`) and ends as soon as the first account exists.

Script clients use HTTP Basic auth or per-user API tokens (hashed at rest, shown once, revocable, role-scoped).

Manage users from the CLI (break-glass; the dashboard's Settings -> Users page does the same):
    uv run python -m argonix.auth add <username> <role> [--temp] [--file configs/users.json]
    uv run python -m argonix.auth passwd|disable|enable|remove <username>
    uv run python -m argonix.auth list
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("argonix.auth")

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"alert.read", "incident.read", "evidence.view"}),
    "operator": frozenset({"alert.read", "incident.read", "evidence.view", "alert.acknowledge",
                           "alert.escalate", "alert.resolve", "alert.dismiss", "feedback.write"}),
    "supervisor": frozenset({"alert.read", "incident.read", "evidence.view", "alert.acknowledge",
                             "alert.escalate", "alert.resolve", "alert.dismiss", "feedback.write",
                             "evidence.export", "evidence.hold", "calibration.approve", "incident.assign",
                             "health.rereference", "incident.merge"}),
    "admin": frozenset({"alert.read", "incident.read", "zone.edit", "rule.edit", "settings.edit", "user.admin",
                        "calibration.approve", "sensor.ingest"}),
    "auditor": frozenset({"audit.read", "alert.read", "incident.read"}),
    # machine accounts for access-control panels / door controllers: may only post sensor events (POST /api/sensors/events)
    "integration": frozenset({"sensor.ingest"}),
}
ROLES = tuple(ROLE_PERMISSIONS)
ALL_PERMS = frozenset().union(*ROLE_PERMISSIONS.values())
STATUS_PERM = {"acknowledged": "alert.acknowledge", "escalated": "alert.escalate", "resolved": "alert.resolve",
               "dismissed": "alert.dismiss", "new": "alert.acknowledge"}

ITER = 310_000                 # PBKDF2-SHA256 iterations for new hashes (records keep their own count)
SESSION_S = 12 * 3600          # absolute session maximum
IDLE_S = 30 * 60               # idle timeout (slid by user activity only, never by background polling)
SLIDE_EVERY_S = 60             # re-issue the sliding cookie at most this often
MIN_PASSWORD, MAX_PASSWORD = 12, 1024
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,47}$")
TOKEN_PREFIX = "axt_"
MAX_TOKENS_PER_USER = 20
BASIC_CACHE_S = 300            # a verified Basic credential is trusted this long (avoids PBKDF2 per request)

# Brute-force brake (manual §22 §8.2: fixed window per (username, source IP), plus per-user and per-IP ceilings).
FAIL_WINDOW_S = 15 * 60
FAIL_PER_PAIR, FAIL_PER_USER, FAIL_PER_IP = 5, 20, 30
LOGIN_FAILED = "invalid credentials or too many attempts"   # one message for every failure: never reveals if a user exists

SETUP_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"            # no 0/O/1/I/L
SETUP_CODE_FILE = "setup_code.txt"

# Small, deliberately boring deny-list. Rules below also catch "<common word><digits/symbols>" and l33t variants.
COMMON_PASSWORDS = frozenset("""
123456789012 1234567890123 12345678901234 123456123456 111111111111 000000000000 123123123123 121212121212
password1234 password12345 password123456 passwordpassword password1234! password123! p@ssw0rd1234 passw0rd1234
qwertyuiop12 qwertyuiop123 qwerty123456 qwertyqwerty 1qaz2wsx3edc 1q2w3e4r5t6y q1w2e3r4t5y6 zaq12wsxcde3 asdfghjkl123
iloveyou1234 iloveyou123! letmein12345 welcome12345 welcome123! administrator admin1234567 adminadmin12 changeme1234
trustno11234 football1234 baseball1234 sunshine1234 princess1234 starwars1234 superman1234 monkey123456 dragon123456
abc123abc123 abcdefghijkl aaaaaaaaaaaa correcthorsebatterystaple correct-horse-battery-staple
""".split())
COMMON_WORDS = frozenset("""
password passwd passw0rd qwerty qwertz azerty asdf zxcv letmein welcome admin administrator root argonix iloveyou love
monkey dragon football baseball soccer hockey sunshine princess master shadow superman batman trustno changeme secret
login default guest test hello freedom whatever starwars pokemon computer internet summer winter spring autumn camera
security operator supervisor viewer auditor cctv qwertyuiop asdfghjkl zxcvbnm abc abcd
""".split())
LEET = str.maketrans({"@": "a", "4": "a", "$": "s", "5": "s", "0": "o", "1": "l", "!": "i", "3": "e", "7": "t", "+": "t"})
SEQUENCES = ("01234567890123456789", "abcdefghijklmnopqrstuvwxyz", "qwertyuiopasdfghjklzxcvbnm", "1qaz2wsx3edc4rfv5tgb")


class AuthError(Exception):
    """Raised by account operations; server.py maps it to an HTTP error with the same status and message."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class Principal:
    username: str
    roles: tuple[str, ...]
    open_mode: bool = False
    via: str = "session"               # session | basic | token | open
    must_change: bool = False
    sid: str = ""
    token_id: str = ""
    session: Optional[dict] = None     # verified cookie payload (session principals)

    @property
    def perms(self) -> frozenset[str]:
        if self.open_mode:
            return ALL_PERMS
        return frozenset().union(*(ROLE_PERMISSIONS.get(r, frozenset()) for r in self.roles))

    def can(self, perm: str) -> bool:
        return perm in self.perms


# ---------------------------------------------------------------------------------------------------------------
# passwords
def hash_password(pw: str, salt: Optional[bytes] = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITER)
    return {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(dk).decode(), "iter": ITER}


def verify_password(pw: str, rec: dict) -> bool:
    try:
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), base64.b64decode(rec["salt"]), int(rec.get("iter", ITER)))
        return hmac.compare_digest(dk, base64.b64decode(rec["hash"]))
    except (KeyError, ValueError, TypeError):
        return False


def _predictable(pw: str) -> bool:
    low = pw.lower()
    if low in COMMON_PASSWORDS or len(set(low)) < 5:
        return True
    alnum = re.sub(r"[^a-z0-9]", "", low)
    if alnum in COMMON_PASSWORDS:
        return True
    if len(alnum) >= 6 and any(alnum in s or alnum in s[::-1] for s in SEQUENCES):
        return True
    for unit in range(1, 5):   # "abcabcabcabc", "121212121212"
        if alnum and len(alnum) >= 2 * unit and alnum == (alnum[:unit] * (len(alnum) // unit + 1))[:len(alnum)]:
            return True
    for variant in (re.sub(r"[^a-z]", "", low), re.sub(r"[^a-z]", "", low.translate(LEET))):
        rest = variant
        for w in sorted(COMMON_WORDS, key=len, reverse=True):
            rest = rest.replace(w, "")
        if variant and len(rest) < 4:   # only common words plus digits/symbols ("Summer2024!!", "P@ssw0rd-2024")
            return True
    return False


def password_problems(pw: str, username: str = "") -> list[str]:
    """Server-side password policy. Returns human-readable problems (empty list = acceptable)."""
    if not isinstance(pw, str):
        return ["password must be a string"]
    probs = []
    if len(pw) < MIN_PASSWORD:
        probs.append(f"must be at least {MIN_PASSWORD} characters")
    if len(pw) > MAX_PASSWORD:
        probs.append(f"must be at most {MAX_PASSWORD} characters")
    if username and len(username) >= 3 and username.lower() in pw.lower():
        probs.append("must not contain the username")
    if len(pw) >= MIN_PASSWORD and _predictable(pw):
        probs.append("is too common or predictable (avoid common words, keyboard runs and repeats)")
    return probs


def check_password(pw: str, username: str = "") -> None:
    probs = password_problems(pw, username)
    if probs:
        raise AuthError(400, "password " + "; ".join(probs))


def valid_username(name: str) -> bool:
    return isinstance(name, str) and bool(USERNAME_RE.match(name))


def _clean_roles(roles, allowed: Optional[set] = None) -> list[str]:
    if isinstance(roles, str):
        roles = [roles]
    if not isinstance(roles, (list, tuple)) or not roles:
        raise AuthError(400, f"roles must be a non-empty list of {list(ROLES)}")
    out = sorted({str(r) for r in roles})
    bad = [r for r in out if r not in ROLE_PERMISSIONS]
    if bad:
        raise AuthError(400, f"unknown role(s) {bad}; valid roles are {list(ROLES)}")
    if allowed is not None and not set(out) <= allowed:
        raise AuthError(403, "a token can only carry roles you hold yourself")
    return out


def _write_private(path: Path, text: str) -> None:
    """Atomic write with owner-only permissions (0600 where the OS supports it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _normalise(rec: dict) -> dict:
    rec.setdefault("roles", [])
    rec.setdefault("disabled", False)
    rec.setdefault("must_change", False)
    rec.setdefault("epoch", 0)
    rec.setdefault("tokens", [])
    return rec


class UserStore:
    """The users JSON file, cached in memory. Reloaded when the file changes on disk (e.g. the CLI edited it)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: dict[str, dict] = {}
        self._mtime: Optional[float] = None
        self._checked = 0.0
        self.reload()

    def reload(self) -> None:
        with self.lock:
            try:
                st = self.path.stat()
            except FileNotFoundError:
                self.data, self._mtime = {}, None
                return
            raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self.data = {u: _normalise(r) for u, r in raw.items() if isinstance(r, dict)}
            self._mtime = st.st_mtime_ns

    def maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._checked < 1.0:
            return
        self._checked = now
        try:
            m = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            m = None
        if m != self._mtime:
            self.reload()

    def save(self) -> None:
        with self.lock:
            _write_private(self.path, json.dumps(self.data, indent=2, sort_keys=True))
            self._mtime = self.path.stat().st_mtime_ns

    def get(self, name: str) -> Optional[dict]:
        return self.data.get(name) if isinstance(name, str) else None


class Brake:
    """In-memory fixed-window failure counters (single process; reset on restart; see manual §22 §5.8)."""

    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.fails: dict[tuple, list[float]] = {}
        self.lock = threading.Lock()

    def _recent(self, key: tuple) -> list[float]:
        now = self.clock()
        v = [t for t in self.fails.get(key, []) if now - t < FAIL_WINDOW_S]
        if v:
            self.fails[key] = v
        else:
            self.fails.pop(key, None)
        return v

    def blocked(self, user: str, ip: str) -> bool:
        u = (user or "").lower()
        with self.lock:
            return (len(self._recent(("p", u, ip))) >= FAIL_PER_PAIR or len(self._recent(("u", u))) >= FAIL_PER_USER
                    or (bool(ip) and len(self._recent(("i", ip))) >= FAIL_PER_IP))

    def fail(self, user: str, ip: str) -> None:
        u, now = (user or "").lower(), self.clock()
        with self.lock:
            if len(self.fails) > 20000:
                self.fails = {k: v for k, v in self.fails.items() if v and now - v[-1] < FAIL_WINDOW_S}
            for key in (("p", u, ip), ("u", u), ("i", ip)):
                self.fails.setdefault(key, []).append(now)

    def blocked_ip(self, ip: str) -> bool:
        with self.lock:
            return len(self._recent(("i", ip))) >= FAIL_PER_IP

    def fail_ip(self, ip: str) -> None:
        with self.lock:
            self.fails.setdefault(("i", ip), []).append(self.clock())

    def success(self, user: str, ip: str) -> None:
        u = (user or "").lower()
        with self.lock:
            self.fails.pop(("p", u, ip), None)
            self.fails.pop(("u", u), None)

    def clear_user(self, user: str) -> None:
        u = (user or "").lower()
        with self.lock:
            for k in [k for k in self.fails if len(k) > 1 and k[1] == u and k[0] in ("p", "u")]:
                self.fails.pop(k, None)


# ---------------------------------------------------------------------------------------------------------------
class Auth:
    def __init__(self, users_file: Path, secret_file: Path, *, idle_timeout_s: float = IDLE_S,
                 session_max_s: float = SESSION_S, setup_code_file: Optional[Path] = None, open_demo: bool = False,
                 basic_enabled: bool = True, localhost_setup: bool = True, clock: Callable[[], float] = time.time):
        self.users = UserStore(users_file)
        secret_file = Path(secret_file)
        if not secret_file.exists():
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        self.secret = secret_file.read_bytes()
        self.setup_code_file = Path(setup_code_file) if setup_code_file else secret_file.parent / SETUP_CODE_FILE
        self.idle_timeout_s = float(idle_timeout_s)
        self.session_max_s = float(session_max_s)
        self.open_demo = bool(open_demo)
        self.basic_enabled = bool(basic_enabled)
        self.localhost_setup = bool(localhost_setup)
        self.now = clock
        self.brake = Brake(lambda: self.now())
        self._setup_code: Optional[str] = None
        self._activity: dict[str, float] = {}      # sid -> last user activity (authoritative while the process lives)
        self._revoked: dict[str, float] = {}       # sid -> expiry (explicit logouts)
        self._basic_cache: dict[str, tuple[str, str, float]] = {}
        self._dummy: Optional[dict] = None
        self.lock = threading.RLock()

    # ---- state -----------------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        self.users.maybe_reload()
        return bool(self.users.data)

    @property
    def setup_required(self) -> bool:
        return not self.enabled

    def _mac(self, domain: str, body: str) -> str:
        return hmac.new(self.secret, f"{domain}|{body}".encode(), hashlib.sha256).hexdigest()

    def _verify_user_pw(self, rec: Optional[dict], pw: str) -> bool:
        """Constant-work check: unknown users still cost one PBKDF2 so timing does not reveal existence."""
        if rec is None:
            if self._dummy is None or int(self._dummy["iter"]) != ITER:
                self._dummy = hash_password(secrets.token_urlsafe(16))
            verify_password(pw, self._dummy)
            return False
        return verify_password(pw, rec)

    # ---- first-run setup -------------------------------------------------------------------------------------
    def ensure_setup_code(self, announce: bool = True) -> Optional[str]:
        if not self.setup_required:
            return None
        with self.lock:
            if self._setup_code is None:
                raw = "".join(secrets.choice(SETUP_ALPHABET) for _ in range(12))
                self._setup_code = "-".join(raw[i:i + 4] for i in range(0, 12, 4))
                _write_private(self.setup_code_file, self._setup_code + "\n")
                if announce:
                    msg = (f"\n{'=' * 72}\n  Argus first-run setup: no user accounts exist yet.\n"
                           f"  Open the dashboard and create the first admin account.\n"
                           f"  One-time setup code: {self._setup_code}\n"
                           f"  (also written to {self.setup_code_file}; not needed from this machine's own browser)\n"
                           f"{'=' * 72}\n")
                    print(msg, flush=True)
                    log.warning("first-run setup pending; setup code written to %s", self.setup_code_file)
            return self._setup_code

    def check_setup_code(self, code: str) -> bool:
        want = self._setup_code or ""
        got = re.sub(r"[^A-Z0-9]", "", str(code or "").upper())
        return bool(want) and hmac.compare_digest(got, want.replace("-", ""))

    def setup(self, username: str, password: str, roles: list[str], code: str, local: bool, ip: str) -> str:
        """Create the first admin. Allowed only while no users exist, with the setup code unless `local`."""
        if not self.setup_required:
            raise AuthError(409, "setup already completed; sign in instead")
        self.ensure_setup_code()
        if not (local and self.localhost_setup):
            if self.brake.blocked("__setup__", ip):
                raise AuthError(429, "too many wrong setup codes; wait 15 minutes")
            if not self.check_setup_code(code):
                self.brake.fail("__setup__", ip)
                raise AuthError(403, "wrong or missing setup code (see the server console or setup_code.txt)")
        if not valid_username(username):
            raise AuthError(400, "username: 1-48 characters, letters, digits and . _ @ - (must start with a letter or digit)")
        roles = _clean_roles(sorted(set([roles] if isinstance(roles, str) else (roles or [])) | {"admin"}))
        check_password(password, username)
        with self.users.lock:
            if self.users.data:
                raise AuthError(409, "setup already completed; sign in instead")
            self._put_user(username, password, roles, must_change=False)
            self.users.data[username]["last_login"] = self.now()
            self.users.save()
            self._setup_code = None
        try:
            self.setup_code_file.unlink()
        except FileNotFoundError:
            pass
        return self.issue_session(username)

    # ---- sessions --------------------------------------------------------------------------------------------
    def _sign(self, payload: dict) -> str:
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
        return f"{body}.{self._mac('session', body)}"

    def _decode(self, token: Optional[str]) -> Optional[dict]:
        if not token or not isinstance(token, str) or "." not in token or len(token) > 4096:
            return None
        body, mac = token.rsplit(".", 1)
        if not hmac.compare_digest(mac, self._mac("session", body)):
            return None
        try:
            p = json.loads(base64.urlsafe_b64decode(body.encode()))
        except ValueError:
            return None
        if not isinstance(p, dict) or not all(k in p for k in ("u", "sid", "e", "iat", "lat")):
            return None   # includes cookies from before session epochs existed
        return p

    def issue_session(self, username: str) -> str:
        rec = self.users.get(username)
        now = int(self.now())
        sid = secrets.token_hex(12)
        self._activity[sid] = now
        return self._sign({"u": username, "sid": sid, "e": int(rec.get("epoch", 0)) if rec else 0, "iat": now, "lat": now})

    def session_max_age(self, payload_or_token) -> int:
        p = self._decode(payload_or_token) if isinstance(payload_or_token, str) else payload_or_token
        return max(0, int(p["iat"] + self.session_max_s - self.now())) if p else 0

    def check_session(self, p: dict) -> str:
        """'' when the session payload is still valid, else the reason it is not."""
        rec = self.users.get(p.get("u"))
        now = self.now()
        if rec is None:
            return "login required"
        if rec.get("disabled"):
            return "account disabled"
        if int(p.get("e", -1)) != int(rec.get("epoch", 0)) or p.get("sid") in self._revoked:
            return "session revoked"
        if now - float(p["iat"]) > self.session_max_s:
            return "session expired (maximum session length reached)"
        if now - max(float(p["lat"]), self._activity.get(p["sid"], 0.0)) > self.idle_timeout_s:
            return "session expired (idle)"
        return ""

    def from_cookie(self, token: Optional[str], activity: bool = False) -> tuple[Optional[Principal], str, Optional[str]]:
        """-> (principal, reason-if-none, refreshed-cookie-or-None). `activity` slides the idle timer."""
        if not self.enabled:
            if self.open_demo:
                return Principal("open-demo", ("admin",), open_mode=True, via="open"), "", None
            return None, "setup required", None
        p = self._decode(token)
        if p is None:
            return None, "login required", None
        reason = self.check_session(p)
        if reason:
            return None, reason, None
        rec = self.users.get(p["u"])
        refreshed = None
        if activity:
            now = self.now()
            self._activity[p["sid"]] = now
            if now - float(p["lat"]) >= SLIDE_EVERY_S:
                p = {**p, "lat": int(now)}
                refreshed = self._sign(p)
            if len(self._activity) > 5000:
                self._prune()
        return (Principal(p["u"], tuple(rec.get("roles", ())), via="session", must_change=bool(rec.get("must_change")),
                          sid=p["sid"], session=p), "", refreshed)

    def _prune(self) -> None:
        now = self.now()
        with self.lock:
            self._activity = {k: v for k, v in self._activity.items() if now - v <= self.idle_timeout_s}
            self._revoked = {k: v for k, v in self._revoked.items() if v > now}

    def session_info(self, p: Principal) -> dict:
        if p.via != "session" or not p.session:
            return {}
        last = max(float(p.session["lat"]), self._activity.get(p.sid, 0.0))
        return {"issued_at": p.session["iat"], "expires_at": p.session["iat"] + self.session_max_s,
                "idle_timeout_s": self.idle_timeout_s, "idle_expires_at": last + self.idle_timeout_s,
                "now": self.now()}   # lets the client compute deadlines without trusting its own clock

    def logout(self, p: Principal) -> None:
        if p.via == "session" and p.session:
            with self.lock:
                self._revoked[p.sid] = float(p.session["iat"]) + self.session_max_s
                self._activity.pop(p.sid, None)

    def revoke_all(self, username: str) -> None:
        """Bump the user's session epoch: every existing cookie for this user stops working."""
        with self.users.lock:
            rec = self.users.get(username)
            if rec is None:
                raise AuthError(404, "unknown user")
            rec["epoch"] = int(rec.get("epoch", 0)) + 1
            self.users.save()

    def revalidate(self, p: Principal) -> str:
        """For long-lived connections (WebSocket, MJPEG): '' while the principal is still valid."""
        if p.open_mode:
            return "" if (self.open_demo and not self.enabled) else "sign-in required"
        rec = self.users.get(p.username)
        if rec is None or rec.get("disabled"):
            return "account disabled"
        if p.via == "session":
            return self.check_session(p.session or {})
        if p.via == "token":
            t = next((t for t in rec.get("tokens", []) if t["id"] == p.token_id), None)
            if t is None or (t.get("expires_at") and t["expires_at"] < self.now()):
                return "token revoked"
        return ""

    # ---- password login --------------------------------------------------------------------------------------
    def login(self, username: str, password: str, ip: str = "") -> Optional[str]:
        """Session cookie value on success; None on any failure (wrong user/password, disabled, locked out)."""
        self.users.maybe_reload()
        if self.brake.blocked(username, ip):
            return None
        rec = self.users.get(username)
        ok = self._verify_user_pw(rec, password)
        if not ok or rec.get("disabled"):
            self.brake.fail(username, ip)
            return None
        self.brake.success(username, ip)
        with self.users.lock:
            rec["last_login"] = self.now()
            if int(rec.get("iter", ITER)) < ITER:   # transparent upgrade to the current work factor
                rec.update(hash_password(password))
            self.users.save()
        return self.issue_session(username)

    def basic(self, username: str, password: str, ip: str = "") -> Optional[Principal]:
        self.users.maybe_reload()
        rec = self.users.get(username)
        key = self._mac("basic", f"{username}\0{password}")
        hit = self._basic_cache.get(key)
        now = self.now()
        if not (hit and rec is not None and hit[0] == username and hit[1] == rec.get("hash") and hit[2] > now):
            if self.brake.blocked(username, ip):
                return None
            if not self._verify_user_pw(rec, password) or rec.get("disabled"):
                self.brake.fail(username, ip)
                return None
            self.brake.success(username, ip)
            with self.lock:
                if len(self._basic_cache) > 1000:
                    self._basic_cache = {k: v for k, v in self._basic_cache.items() if v[2] > now}
                self._basic_cache[key] = (username, rec.get("hash"), now + BASIC_CACHE_S)
            with self.users.lock:
                rec["last_login"] = now
                self.users.save()
        if rec.get("disabled"):
            return None
        return Principal(username, tuple(rec.get("roles", ())), via="basic", must_change=bool(rec.get("must_change")))

    # ---- API tokens ------------------------------------------------------------------------------------------
    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def bearer(self, token: str, ip: str = "") -> Optional[Principal]:
        self.users.maybe_reload()
        # Tokens carry 256 random bits, so guessing is hopeless; the brake is per source IP only (a global counter
        # would let anyone lock every integration out by sending garbage tokens).
        if self.brake.blocked_ip(ip):
            return None
        m = re.fullmatch(TOKEN_PREFIX + r"([0-9a-f]{10})_([A-Za-z0-9_-]{20,100})", token or "")
        if not m:
            self.brake.fail_ip(ip)
            return None
        tid, h = m.group(1), self._token_hash(token)
        for username, rec in self.users.data.items():
            for t in rec.get("tokens", []):
                if t["id"] == tid and hmac.compare_digest(t["hash"], h):
                    now = self.now()
                    if rec.get("disabled") or (t.get("expires_at") and t["expires_at"] < now):
                        return None
                    if now - float(t.get("last_used") or 0) > 300:   # persist usage at most every 5 min
                        with self.users.lock:
                            t["last_used"] = now
                            self.users.save()
                    roles = tuple(r for r in t.get("roles", []) if r in rec.get("roles", []))
                    return Principal(username, roles, via="token", token_id=tid)
        self.brake.fail_ip(ip)
        return None

    def list_tokens(self, username: Optional[str] = None) -> list[dict]:
        out = []
        for u, rec in sorted(self.users.data.items()):
            if username is not None and u != username:
                continue
            for t in rec.get("tokens", []):
                out.append({"id": t["id"], "user": u, "name": t.get("name", ""), "roles": t.get("roles", []),
                            "created_at": t.get("created_at"), "expires_at": t.get("expires_at"),
                            "last_used": t.get("last_used"), "prefix": f"{TOKEN_PREFIX}{t['id']}_…"})
        return out

    def create_token(self, username: str, name: str, roles, expires_days=None) -> tuple[dict, str]:
        rec = self.users.get(username)
        if rec is None:
            raise AuthError(404, "unknown user")
        name = str(name or "").strip()
        if not name or len(name) > 64:
            raise AuthError(400, "token name: 1-64 characters")
        roles = _clean_roles(roles if roles else rec.get("roles", []), allowed=set(rec.get("roles", [])))
        exp = None
        if expires_days not in (None, "", 0):
            try:
                d = float(expires_days)
            except (TypeError, ValueError):
                raise AuthError(400, "expires_days must be a number")
            if not 0 < d <= 3650:
                raise AuthError(400, "expires_days must be between 0 and 3650 (empty = never)")
            exp = self.now() + d * 86400
        with self.users.lock:
            if len(rec.get("tokens", [])) >= MAX_TOKENS_PER_USER:
                raise AuthError(409, f"at most {MAX_TOKENS_PER_USER} tokens per user; revoke an old one first")
            tid = secrets.token_hex(5)
            secret = f"{TOKEN_PREFIX}{tid}_{secrets.token_urlsafe(32)}"
            rec.setdefault("tokens", []).append({"id": tid, "name": name, "hash": self._token_hash(secret),
                                                 "roles": roles, "created_at": self.now(), "expires_at": exp,
                                                 "last_used": None})
            self.users.save()
        return next(t for t in self.list_tokens(username) if t["id"] == tid), secret

    def revoke_token(self, tid: str, owner: Optional[str] = None) -> dict:
        """Remove token `tid`. `owner` restricts the search to one user's tokens (non-admin callers)."""
        with self.users.lock:
            for u, rec in self.users.data.items():
                if owner is not None and u != owner:
                    continue
                for t in rec.get("tokens", []):
                    if t["id"] == tid:
                        rec["tokens"].remove(t)
                        self.users.save()
                        return {"id": tid, "user": u, "name": t.get("name", "")}
        raise AuthError(404, "unknown token")

    # ---- user administration -----------------------------------------------------------------------------------
    def _put_user(self, username: str, password: str, roles: list[str], must_change: bool) -> None:
        now = self.now()
        self.users.data[username] = _normalise({**hash_password(password), "roles": roles, "must_change": must_change,
                                                "created_at": now, "password_changed_at": now, "last_login": None})

    def _active_admins(self, exclude: str = "") -> int:
        return sum(1 for u, r in self.users.data.items()
                   if u != exclude and "admin" in r.get("roles", []) and not r.get("disabled"))

    @staticmethod
    def public(username: str, rec: dict) -> dict:
        return {"username": username, "roles": list(rec.get("roles", [])), "disabled": bool(rec.get("disabled")),
                "must_change": bool(rec.get("must_change")), "created_at": rec.get("created_at"),
                "last_login": rec.get("last_login"), "password_changed_at": rec.get("password_changed_at"),
                "tokens": len(rec.get("tokens", []))}

    def list_users(self) -> list[dict]:
        self.users.maybe_reload()
        return [self.public(u, r) for u, r in sorted(self.users.data.items(), key=lambda kv: kv[0].lower())]

    def create_user(self, username: str, roles, password: str, must_change: bool = True) -> dict:
        if not valid_username(username):
            raise AuthError(400, "username: 1-48 characters, letters, digits and . _ @ - (must start with a letter or digit)")
        roles = _clean_roles(roles)
        check_password(password, username)
        with self.users.lock:
            if any(u.lower() == username.lower() for u in self.users.data):
                raise AuthError(409, "a user with that name already exists")
            self._put_user(username, password, roles, must_change=bool(must_change))
            self.users.save()
            return self.public(username, self.users.data[username])

    def update_user(self, actor: str, username: str, roles=None, disabled=None) -> tuple[dict, dict]:
        """-> (public record, changes). Guards: never disable yourself; never leave zero active admins."""
        with self.users.lock:
            rec = self.users.get(username)
            if rec is None:
                raise AuthError(404, "unknown user")
            changes: dict = {}
            new_roles = _clean_roles(roles) if roles is not None else list(rec.get("roles", []))
            new_disabled = bool(disabled) if disabled is not None else bool(rec.get("disabled"))
            if new_disabled and username == actor:
                raise AuthError(409, "you cannot disable your own account")
            if username == actor and "admin" in rec.get("roles", []) and "admin" not in new_roles:
                raise AuthError(409, "you cannot remove your own admin role (ask another admin)")
            losing_admin = ("admin" in rec.get("roles", []) and not rec.get("disabled")) and \
                           ("admin" not in new_roles or new_disabled)
            if losing_admin and self._active_admins(exclude=username) == 0:
                raise AuthError(409, "this is the last active admin; add another admin first")
            if new_roles != list(rec.get("roles", [])):
                changes["roles"] = {"from": list(rec.get("roles", [])), "to": new_roles}
                rec["roles"] = new_roles
            if new_disabled != bool(rec.get("disabled")):
                changes["disabled"] = new_disabled
                rec["disabled"] = new_disabled
            if changes:
                self.users.save()
            return self.public(username, rec), changes

    def delete_user(self, actor: str, username: str) -> None:
        with self.users.lock:
            rec = self.users.get(username)
            if rec is None:
                raise AuthError(404, "unknown user")
            if username == actor:
                raise AuthError(409, "you cannot delete your own account")
            if "admin" in rec.get("roles", []) and not rec.get("disabled") and self._active_admins(exclude=username) == 0:
                raise AuthError(409, "this is the last active admin; add another admin first")
            del self.users.data[username]
            self.users.save()

    def reset_password(self, actor: str, username: str, password: str, must_change: bool = True) -> dict:
        if username == actor:
            raise AuthError(409, "use 'Change password' for your own account (it asks for your current password)")
        check_password(password, username)
        with self.users.lock:
            rec = self.users.get(username)
            if rec is None:
                raise AuthError(404, "unknown user")
            rec.update(hash_password(password))
            rec["must_change"] = bool(must_change)
            rec["password_changed_at"] = self.now()
            rec["epoch"] = int(rec.get("epoch", 0)) + 1   # signs the user out everywhere
            self.users.save()
        self.brake.clear_user(username)
        return self.public(username, rec)

    def change_password(self, username: str, current: str, new: str, ip: str = "") -> str:
        """Self-service change. Verifies `current`; signs out every other session; returns a fresh session."""
        rec = self.users.get(username)
        if rec is None:
            raise AuthError(404, "unknown user")
        if self.brake.blocked(username, ip):
            raise AuthError(429, "too many failed attempts; wait 15 minutes")
        if not verify_password(str(current or ""), rec):
            self.brake.fail(username, ip)
            raise AuthError(403, "current password is wrong")
        if hmac.compare_digest(str(new or ""), str(current or "")):
            raise AuthError(400, "the new password must differ from the current one")
        check_password(new, username)
        with self.users.lock:
            rec.update(hash_password(new))
            rec["must_change"] = False
            rec["password_changed_at"] = self.now()
            rec["epoch"] = int(rec.get("epoch", 0)) + 1
            self.users.save()
        self.brake.success(username, ip)
        return self.issue_session(username)


# ---------------------------------------------------------------------------------------------------------------
def _ask_password(username: str) -> str:
    pw = getpass.getpass(f"Password for {username}: ")
    if pw != getpass.getpass("Repeat: "):
        raise SystemExit("passwords do not match")
    probs = password_problems(pw, username)
    if probs:
        raise SystemExit("password " + "; ".join(probs))
    return pw


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Manage Argus users (break-glass CLI; the dashboard does the same).")
    ap.add_argument("cmd", choices=["add", "remove", "list", "passwd", "disable", "enable", "signout"])
    ap.add_argument("username", nargs="?")
    ap.add_argument("role", nargs="?", choices=sorted(ROLE_PERMISSIONS))
    ap.add_argument("--temp", action="store_true", help="force a password change at next sign-in")
    ap.add_argument("--file", default=str(Path(__file__).resolve().parents[1] / "configs" / "users.json"))
    a = ap.parse_args(argv)
    store = UserStore(Path(a.file))
    users = store.data
    if a.cmd == "list":
        for u, r in sorted(users.items()):
            flags = [f for f, on in (("disabled", r.get("disabled")), ("must-change", r.get("must_change"))) if on]
            print(u, r.get("roles"), " ".join(flags), f"tokens={len(r.get('tokens', []))}")
        return
    if not a.username:
        ap.error("username required")
    rec = users.get(a.username)
    if a.cmd == "add":
        if not a.role:
            ap.error("role required")
        if not valid_username(a.username):
            raise SystemExit("username: 1-48 characters, letters, digits and . _ @ -")
        if rec is None:
            now = time.time()
            users[a.username] = _normalise({**hash_password(_ask_password(a.username)), "roles": [a.role],
                                            "must_change": a.temp, "created_at": now, "password_changed_at": now,
                                            "last_login": None})
        else:
            rec["roles"] = sorted(set(rec.get("roles", [])) | {a.role})
            print(f"{a.username} exists; added role {a.role} (password unchanged)")
    elif rec is None:
        raise SystemExit(f"unknown user {a.username}")
    elif a.cmd == "remove":
        if "admin" in rec.get("roles", []) and not any("admin" in r.get("roles", []) and not r.get("disabled")
                                                        for u, r in users.items() if u != a.username):
            print("warning: removing the last active admin; the dashboard will need another admin created via this CLI")
        users.pop(a.username)
    elif a.cmd == "passwd":
        rec.update(hash_password(_ask_password(a.username)))
        rec.update(must_change=a.temp, password_changed_at=time.time(), epoch=int(rec.get("epoch", 0)) + 1)
    elif a.cmd in ("disable", "enable"):
        rec["disabled"] = a.cmd == "disable"
    elif a.cmd == "signout":
        rec["epoch"] = int(rec.get("epoch", 0)) + 1
    store.save()
    print(f"saved {store.path} ({len(users)} users). A running server picks the change up within a second.")


if __name__ == "__main__":
    main()
