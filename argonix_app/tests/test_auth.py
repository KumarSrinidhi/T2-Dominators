"""Accounts & access: RBAC (manual §13), first-run setup, password policy, user admin, sessions (epoch, idle, max),
HTTP Basic + API tokens, CSRF/Origin checks, security headers, explicit open demo mode.

All usernames/passwords here are throwaway test fixtures."""
import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import argonix.auth as auth_mod
from argonix.auth import LOGIN_FAILED, Auth, AuthError, hash_password, password_problems
from argonix.server import create_app
from tests.test_server import make_site

PW = "correct-horse-battery"          # test fixture only
NEW_PW = "violet-otter-jumps-42"      # test fixture only
CSRF = {"X-Argonix-CSRF": "1"}        # what the dashboard sends on every request


@pytest.fixture(autouse=True)
def fast_hashes(monkeypatch):
    """Cheap PBKDF2 in tests only (production default stays at auth.ITER)."""
    monkeypatch.setattr(auth_mod, "ITER", 2000)


def write_users(tmp_path, pairs):
    users = {name: {**hash_password(PW), "roles": [role]} for name, role in pairs}
    (tmp_path / "users.json").write_text(json.dumps(users))


STAFF = [("vic", "viewer"), ("olga", "operator"), ("sam", "supervisor"), ("ada", "admin"), ("aud", "auditor")]


def client_with_users(tmp_path, run=True, **auth_kw):
    site = make_site(tmp_path)
    if run:
        site.start(); site.wait(timeout=120); site.stop()
    write_users(tmp_path, STAFF)
    auth = Auth(tmp_path / "users.json", tmp_path / "secret.key", **auth_kw)
    return site, create_app(site, auth=auth)


def login(app, user, pw=PW, **kw):
    c = TestClient(app, headers=CSRF, **kw)
    r = c.post("/api/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return c


def audit_text(site):
    return json.dumps(site.store.list_audit(limit=10000))


# ---------------------------------------------------------------------------------------------------------------
def test_rbac_matrix(tmp_path):
    site, app = client_with_users(tmp_path)
    anon = TestClient(app, headers=CSRF)
    assert anon.get("/api/alerts").status_code == 401
    assert anon.post("/api/login", json={"username": "vic", "password": "wrong"}).status_code == 401
    aid = login(app, "vic").get("/api/alerts").json()[0]["id"]
    snap = login(app, "vic").get(f"/api/alerts/{aid}").json()["events"][0]["evidence"]["snapshot"]

    vic, olga, sam, ada, aud = (login(app, u) for u in ("vic", "olga", "sam", "ada", "aud"))
    assert vic.get(f"/media/{snap}").status_code == 200
    assert vic.post(f"/api/alerts/{aid}/status", json={"status": "acknowledged"}).status_code == 403
    assert olga.post(f"/api/alerts/{aid}/status", json={"status": "acknowledged"}).status_code == 200
    assert olga.get(f"/api/alerts/{aid}/export").status_code == 403
    assert sam.get(f"/api/alerts/{aid}/export").status_code == 200
    assert olga.post("/api/cameras/cam1/inject", json={"fault": "blur"}).status_code == 403
    assert ada.post("/api/cameras/cam1/inject", json={"fault": "blur", "seconds": 1}).status_code == 200
    assert ada.post(f"/api/alerts/{aid}/status", json={"status": "acknowledged"}).status_code == 403  # separation of duties
    assert olga.get("/api/audit").status_code == 403
    log = aud.get("/api/audit").json()
    assert any(e["actor"] == "olga" and e["action"] == "status:acknowledged" for e in log)
    assert any(e["action"] == "login_failed" for e in log)
    assert vic.get("/api/me").json()["roles"] == ["viewer"]
    # account management is admin-only
    for c in (vic, olga, sam, aud):
        assert c.get("/api/users").status_code == 403
    assert ada.get("/api/users").status_code == 200


def test_media_path_traversal_blocked(tmp_path):
    site, app = client_with_users(tmp_path)
    vic = login(app, "vic")
    for bad in ["../users.json", "..%2Fusers.json", "argonix.db", "secret.key", "setup_code.txt"]:
        assert vic.get(f"/media/{bad}").status_code == 404


def test_bruteforce_brake_and_uniform_errors(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    c = TestClient(app, headers=CSRF)
    unknown = c.post("/api/login", json={"username": "nobody", "password": "whatever-it-is"})
    wrong = c.post("/api/login", json={"username": "vic", "password": "not-the-password"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": LOGIN_FAILED}   # never reveals whether a user exists
    for _ in range(5):
        c.post("/api/login", json={"username": "olga", "password": "nope"})
    locked = c.post("/api/login", json={"username": "olga", "password": PW})
    assert locked.status_code == 401 and locked.json() == {"detail": LOGIN_FAILED}
    # the brake is per (user, source address): another address can still sign in as olga
    assert TestClient(app, headers=CSRF, client=("10.0.0.9", 4000)).post(
        "/api/login", json={"username": "olga", "password": PW}).status_code == 200


# ---- first-run setup / open demo ----------------------------------------------------------------------------------
def test_no_users_means_setup_not_open_mode(tmp_path):
    site = make_site(tmp_path)
    c = TestClient(create_app(site), headers=CSRF)
    st = c.get("/api/auth/status").json()
    assert st["setup_required"] is True and st["open_demo"] is False and st["setup_code_required"] is True
    r = c.get("/api/me")
    assert r.status_code == 401 and r.json()["detail"] == "setup required"
    assert c.get("/api/alerts").status_code == 401
    assert c.get("/api/users").status_code == 401


def test_open_demo_flag_and_config(tmp_path):
    site = make_site(tmp_path)
    c = TestClient(create_app(site, open_demo=True), headers=CSRF)
    me = c.get("/api/me").json()
    assert me["open_mode"] is True and me["auth"] == "open" and "evidence.export" in me["permissions"]
    assert c.get("/api/auth/status").json()["open_demo"] is True
    # account management and tokens are refused in open mode (no bypass of the setup code)
    assert c.post("/api/users", json={"username": "x", "roles": ["admin"], "password": NEW_PW}).status_code == 403
    assert c.post("/api/tokens", json={"name": "x"}).status_code == 403
    site.cfg["auth"] = {"open_demo": True}
    assert TestClient(create_app(site)).get("/api/me").json()["open_mode"] is True


def test_first_run_setup_needs_code_from_lan(tmp_path):
    site = make_site(tmp_path)
    app = create_app(site, open_demo=True)
    code_file = site.data / "setup_code.txt"
    code = code_file.read_text().strip()
    lan = TestClient(app, headers=CSRF, client=("192.168.1.50", 5555))
    body = {"username": "owner", "password": NEW_PW, "roles": ["admin", "supervisor"]}
    assert lan.post("/api/setup", json=body).status_code == 403                       # no code
    assert lan.post("/api/setup", json={**body, "setup_code": "AAAA-BBBB-CCCC"}).status_code == 403
    weak = lan.post("/api/setup", json={**body, "password": "password1234", "setup_code": code})
    assert weak.status_code == 400 and "common" in weak.json()["detail"]
    r = lan.post("/api/setup", json={**body, "setup_code": code.lower().replace("-", " ")})
    assert r.status_code == 200, r.text
    assert "argonix_session" in r.headers["set-cookie"]
    me = lan.get("/api/me").json()
    assert me["username"] == "owner" and set(me["roles"]) == {"admin", "supervisor"} and me["open_mode"] is False
    assert not code_file.exists()
    # open demo mode ends immediately: anonymous requests now need a session
    anon = TestClient(app, headers=CSRF)
    assert anon.get("/api/alerts").status_code == 401
    assert anon.get("/api/auth/status").json()["setup_required"] is False
    again = anon.post("/api/setup", json={**body, "username": "intruder", "setup_code": code})
    assert again.status_code == 409
    text = audit_text(site)
    assert "setup_completed" in text and "setup_failed" in text
    assert NEW_PW not in text and code not in text and code.replace("-", "") not in text


def test_setup_from_loopback_skips_code_but_not_via_rebinding(tmp_path):
    site = make_site(tmp_path)
    app = create_app(site)
    body = {"username": "owner", "password": NEW_PW}
    rebind = TestClient(app, headers=CSRF, client=("127.0.0.1", 5000), base_url="http://evil.example:8020")
    assert rebind.get("/api/auth/status").json()["setup_code_required"] is True
    assert rebind.post("/api/setup", json=body).status_code == 403
    proxied = TestClient(app, headers={**CSRF, "X-Forwarded-For": "203.0.113.9"}, client=("127.0.0.1", 5000),
                         base_url="http://127.0.0.1:8020")
    assert proxied.post("/api/setup", json=body).status_code == 403
    local = TestClient(app, headers=CSRF, client=("127.0.0.1", 5000), base_url="http://localhost:8020")
    assert local.get("/api/auth/status").json()["setup_code_required"] is False
    assert local.post("/api/setup", json=body).status_code == 200
    assert local.get("/api/me").json()["roles"] == ["admin"]


# ---- password policy ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pw", ["short-1", "password1234", "Summer2024!!", "P@ssw0rd-2024", "aaaaaaaaaaaaaa",
                                "123456789012", "qwertyuiop12", "abcabcabcabc", "Welcome12345", "adminadmin12"])
def test_password_policy_rejects(pw):
    assert password_problems(pw, "someone")


@pytest.mark.parametrize("pw", [PW, NEW_PW, "Xk9#mQ2$vL7p", "tram stop under the old oak"])
def test_password_policy_accepts(pw):
    assert password_problems(pw, "someone") == []


def test_password_policy_username():
    assert any("username" in p for p in password_problems("olga-rides-bikes-99", "olga"))
    assert password_problems("olga-rides-bikes-99", "sam") == []


# ---- user administration ------------------------------------------------------------------------------------------
def test_user_crud_guards_and_audit(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    ada = login(app, "ada")
    weak = ada.post("/api/users", json={"username": "nina", "roles": ["operator"], "password": "nina-password-1"})
    assert weak.status_code == 400
    r = ada.post("/api/users", json={"username": "nina", "roles": ["operator"], "password": NEW_PW})
    assert r.status_code == 200 and r.json()["must_change"] is True and r.json()["roles"] == ["operator"]
    assert ada.post("/api/users", json={"username": "NINA", "roles": ["viewer"], "password": NEW_PW}).status_code == 409
    assert ada.post("/api/users", json={"username": "x y", "roles": ["viewer"], "password": NEW_PW}).status_code == 400
    assert ada.post("/api/users", json={"username": "zed", "roles": ["root"], "password": NEW_PW}).status_code == 400
    listed = {u["username"]: u for u in ada.get("/api/users").json()}
    assert listed["nina"]["last_login"] is None and "hash" not in listed["nina"] and "salt" not in listed["nina"]

    # forced password change on first sign-in
    nina = login(app, "nina", NEW_PW)
    assert nina.get("/api/me").json()["must_change"] is True
    blocked = nina.get("/api/alerts")
    assert blocked.status_code == 403 and "password change" in blocked.json()["detail"]
    assert nina.post("/api/me/password", json={"current": "wrong-one-entirely", "new": PW + "!"}).status_code == 403
    assert nina.post("/api/me/password", json={"current": NEW_PW, "new": "password12345"}).status_code == 400
    assert nina.post("/api/me/password", json={"current": NEW_PW, "new": "amber-kettle-sings-7"}).status_code == 200
    assert nina.get("/api/alerts").status_code == 200        # new cookie issued for this browser
    assert {u["username"]: u for u in ada.get("/api/users").json()}["nina"]["last_login"]

    # roles / disable / enable
    assert ada.put("/api/users/nina", json={"roles": ["viewer"]}).json()["roles"] == ["viewer"]
    assert nina.get("/api/me").json()["roles"] == ["viewer"]
    assert ada.put("/api/users/nina", json={"disabled": True}).status_code == 200
    r = nina.get("/api/me")
    assert r.status_code == 401 and r.json()["detail"] == "account disabled"
    assert TestClient(app, headers=CSRF).post("/api/login", json={"username": "nina", "password": "amber-kettle-sings-7"}).json() == {"detail": LOGIN_FAILED}
    assert ada.put("/api/users/nina", json={"disabled": False}).status_code == 200

    # reset: temp password, other sessions revoked, forced change again
    nina2 = login(app, "nina", "amber-kettle-sings-7")
    assert ada.post("/api/users/nina/reset", json={"password": "short"}).status_code == 400
    assert ada.post("/api/users/nina/reset", json={"password": "maple-river-stone-5"}).status_code == 200
    assert nina2.get("/api/me").json()["detail"] == "session revoked"
    assert login(app, "nina", "maple-river-stone-5").get("/api/me").json()["must_change"] is True

    # self-protection and last-admin protection
    assert ada.delete("/api/users/ada").status_code == 409
    assert ada.put("/api/users/ada", json={"disabled": True}).status_code == 409
    assert ada.put("/api/users/ada", json={"roles": ["viewer"]}).status_code == 409
    assert ada.post("/api/users/ada/reset", json={"password": NEW_PW}).status_code == 409
    auth = app.state.auth
    with pytest.raises(AuthError) as e:
        auth.update_user("cli", "ada", roles=["viewer"])
    assert e.value.status == 409 and "last active admin" in e.value.detail
    with pytest.raises(AuthError):
        auth.delete_user("cli", "ada")
    assert ada.post("/api/users", json={"username": "ben", "roles": ["admin"], "password": NEW_PW}).status_code == 200
    assert ada.put("/api/users/ben", json={"disabled": True}).status_code == 200
    with pytest.raises(AuthError):
        auth.delete_user("cli", "ada")                        # ben is disabled, so ada is still the last active admin
    assert ada.delete("/api/users/nina").status_code == 200
    assert ada.delete("/api/users/nina").status_code == 404
    assert login(app, "vic").delete("/api/users/sam").status_code == 403

    actions = [e["action"] for e in site.store.list_audit(limit=1000)]
    for a in ("user_created", "user_updated", "user_password_reset", "user_deleted", "password_changed"):
        assert a in actions
    text = audit_text(site)
    for secret in (NEW_PW, "amber-kettle-sings-7", "maple-river-stone-5", PW):
        assert secret not in text


# ---- sessions -------------------------------------------------------------------------------------------------------
def test_session_epoch_logout_and_password_change(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    a, b = login(app, "olga"), login(app, "olga")
    assert a.get("/api/me").status_code == b.get("/api/me").status_code == 200
    # logout revokes that cookie server-side, even if someone kept a copy
    kept = b.cookies.get("argonix_session")
    assert b.post("/api/logout").status_code == 200
    thief = TestClient(app, headers=CSRF, cookies={"argonix_session": kept})
    assert thief.get("/api/me").json()["detail"] == "session revoked"
    assert a.get("/api/me").status_code == 200
    # password change keeps this browser signed in and signs everything else out
    c = login(app, "olga")
    assert a.post("/api/me/password", json={"current": PW, "new": NEW_PW}).status_code == 200
    assert a.get("/api/me").status_code == 200
    assert c.get("/api/me").json()["detail"] == "session revoked"
    # sign out all sessions (including this one)
    d = login(app, "olga", NEW_PW)
    a_cookie = a.cookies.get("argonix_session")
    assert a.post("/api/me/sessions/revoke").status_code == 200
    assert a.get("/api/me").status_code == 401                       # cookie cleared in this browser
    assert d.get("/api/me").json()["detail"] == "session revoked"
    assert TestClient(app, cookies={"argonix_session": a_cookie}).get("/api/me").json()["detail"] == "session revoked"
    assert "sessions_revoked" in [e["action"] for e in site.store.list_audit(limit=100)]
    # tampered cookie
    evil = TestClient(app, cookies={"argonix_session": kept[:-4] + "0000"})
    assert evil.get("/api/me").json()["detail"] == "login required"


def test_idle_timeout_and_absolute_maximum(tmp_path):
    t = [1_000_000.0]
    site, app = client_with_users(tmp_path, run=False, clock=lambda: t[0], idle_timeout_s=60, session_max_s=300)
    c = login(app, "vic")
    info = c.get("/api/me").json()["session"]
    assert info["idle_timeout_s"] == 60 and info["expires_at"] == t[0] + 300
    t[0] += 50
    assert c.get("/api/me").status_code == 200          # reads (background polling) do not slide the idle timer
    t[0] += 20
    r = c.get("/api/me")
    assert r.status_code == 401 and r.json()["detail"] == "session expired (idle)"
    c = login(app, "vic")
    for _ in range(5):                                  # user activity keeps the session alive ...
        t[0] += 50
        assert c.post("/api/session/ping").status_code == 200
    t[0] += 51                                          # ... until the absolute maximum
    r = c.get("/api/me")
    assert r.status_code == 401 and "maximum" in r.json()["detail"]
    # long-lived connections (WebSocket, MJPEG) are re-validated with the same rules
    c = login(app, "vic")
    p, _, _ = app.state.auth.from_cookie(c.cookies.get("argonix_session"))
    assert p is not None and app.state.auth.revalidate(p) == ""
    t[0] += 61
    assert "idle" in app.state.auth.revalidate(p)


def test_sliding_cookie_is_reissued_on_activity(tmp_path):
    t = [2_000_000.0]
    site, app = client_with_users(tmp_path, run=False, clock=lambda: t[0])
    c = login(app, "vic")
    first = c.cookies.get("argonix_session")
    t[0] += 120
    r = c.post("/api/session/ping")
    assert r.status_code == 200 and "argonix_session=" in r.headers.get("set-cookie", "")
    assert c.cookies.get("argonix_session") != first
    t[0] += 120
    assert "set-cookie" not in c.get("/api/me").headers        # GETs never refresh


# ---- HTTP Basic + API tokens --------------------------------------------------------------------------------------
def basic(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def test_http_basic_auth(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    c = TestClient(app)
    assert c.get("/api/alerts", auth=("vic", PW)).status_code == 200
    assert c.post("/api/session/ping", headers=basic("olga", PW)).status_code == 200    # no CSRF header needed
    bad = c.get("/api/alerts", headers=basic("vic", "wrong-password"))
    assert bad.status_code == 401 and bad.headers["www-authenticate"].startswith("Basic")
    # cookie-less browser requests get no challenge (no native password pop-up), and never for the page itself
    assert "www-authenticate" not in c.get("/api/alerts").headers
    page = c.get("/", headers=basic("vic", "wrong-password"))
    assert page.status_code == 200 and "www-authenticate" not in page.headers
    assert c.get("/static/js/app.js", headers=basic("vic", PW)).status_code == 200
    # same brute-force brake as the login form
    for _ in range(5):
        c.get("/api/alerts", headers=basic("aud", "nope-nope-nope"))
    assert c.get("/api/alerts", headers=basic("aud", PW)).status_code == 401
    # permissions are the user's own
    assert c.get("/api/users", headers=basic("vic", PW)).status_code == 403
    assert c.get("/api/users", headers=basic("ada", PW)).status_code == 200
    assert c.get("/api/me", headers=basic("sam", PW)).json()["auth"] == "basic"
    # a user who must change their password cannot use Basic for anything else
    login(app, "ada").post("/api/users", json={"username": "tmp1", "roles": ["viewer"], "password": NEW_PW})
    r = c.get("/api/alerts", headers=basic("tmp1", NEW_PW))
    assert r.status_code == 403 and "password change" in r.json()["detail"]
    assert c.get("/api/alerts", headers=basic("nobody", PW)).json() == {"detail": LOGIN_FAILED}
    assert c.get("/api/alerts", headers={"Authorization": "Basic !!!notbase64"}).status_code == 401
    assert c.get("/api/alerts", headers={"Authorization": "Digest abc"}).status_code == 401


def test_http_basic_can_be_disabled(tmp_path):
    site = make_site(tmp_path)
    write_users(site.config_path.parent, STAFF)      # the default Auth reads users.json next to the site config
    site.cfg["auth"] = {"basic_auth": False}
    r = TestClient(create_app(site)).get("/api/alerts", auth=("vic", PW))
    assert r.status_code == 401 and "disabled" in r.json()["detail"]


def test_api_tokens(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    sam = login(app, "sam")
    assert sam.post("/api/tokens", json={"name": "nvr", "roles": ["viewer"]}).status_code == 403   # not held
    assert sam.post("/api/tokens", json={"name": "", "roles": ["supervisor"]}).status_code == 400
    r = sam.post("/api/tokens", json={"name": "nvr webhook", "roles": ["supervisor"], "expires_days": 30})
    assert r.status_code == 200
    tok = r.json()["token"]
    assert tok.startswith("axt_") and r.json()["expires_at"]
    stored = (tmp_path / "users.json").read_text()
    assert tok not in stored and hashlib.sha256(tok.encode()).hexdigest() in stored     # hashed at rest
    listed = sam.get("/api/tokens").json()
    assert len(listed) == 1 and "hash" not in listed[0] and "token" not in listed[0]

    c = TestClient(app)
    bearer = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/alerts", headers=bearer).status_code == 200
    assert c.get("/api/me", headers=bearer).json()["auth"] == "token"
    assert c.get("/api/alerts", auth=("token", tok)).status_code == 200         # Basic-only clients
    # tokens cannot manage accounts or mint more tokens
    assert c.post("/api/tokens", headers=bearer, json={"name": "x"}).status_code == 403
    assert c.post("/api/me/password", headers=bearer, json={"current": PW, "new": NEW_PW}).status_code == 403
    assert c.get("/api/users", headers=bearer).status_code == 403
    # scope follows the owner's current roles
    ada = login(app, "ada")
    assert ada.put("/api/users/sam", json={"roles": ["viewer"]}).status_code == 200
    assert c.get("/api/me", headers=bearer).json()["roles"] == []
    assert c.get("/api/alerts", headers=bearer).status_code == 403
    assert ada.put("/api/users/sam", json={"roles": ["supervisor"]}).status_code == 200
    # admin can list and revoke anyone's tokens; others only their own
    assert sam.get("/api/tokens?all=true").status_code == 403
    assert [t["user"] for t in ada.get("/api/tokens?all=true").json()] == ["sam"]
    tid = listed[0]["id"]
    assert login(app, "olga").delete(f"/api/tokens/{tid}").status_code == 404
    assert sam.delete(f"/api/tokens/{tid}").status_code == 200
    r = c.get("/api/alerts", headers=bearer)
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
    # expiry + disabled owner
    tok2 = sam.post("/api/tokens", json={"name": "short", "expires_days": 1}).json()["token"]
    assert c.get("/api/alerts", headers={"Authorization": f"Bearer {tok2}"}).status_code == 200
    app.state.auth.users.data["sam"]["tokens"][0]["expires_at"] = 1.0
    assert c.get("/api/alerts", headers={"Authorization": f"Bearer {tok2}"}).status_code == 401
    assert c.get("/api/alerts", headers={"Authorization": "Bearer axt_0000000000_" + "x" * 43}).status_code == 401
    actions = [e["action"] for e in site.store.list_audit(limit=100)]
    assert "token_created" in actions and "token_revoked" in actions
    assert tok not in audit_text(site) and tok2 not in audit_text(site)


# ---- CSRF / headers / cookies ---------------------------------------------------------------------------------------
def test_csrf_origin_and_header_checks(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    c = login(app, "olga")
    ok = c.post("/api/session/ping", headers={"Origin": "http://testserver"})
    assert ok.status_code == 200
    for h in ({"Origin": "http://evil.example"}, {"Origin": "null"}, {"Origin": "http://testserver:9999"},
              {"Referer": "http://evil.example/page"}):
        r = c.post("/api/session/ping", headers=h)
        assert r.status_code == 403 and "CSRF" in r.json()["detail"], h
    no_header = TestClient(app, cookies={"argonix_session": c.cookies.get("argonix_session")})
    assert no_header.post("/api/session/ping").status_code == 403           # cookie, no Origin/Referer, no header
    assert no_header.get("/api/me").status_code == 200                       # reads are not affected
    assert no_header.post("/api/session/ping", headers={"Referer": "http://testserver/#/live"}).status_code == 200
    # login CSRF and cross-origin Basic are refused too; plain scripts (no Origin) are fine
    assert TestClient(app).post("/api/login", json={"username": "vic", "password": PW},
                                headers={"Origin": "http://evil.example"}).status_code == 403
    assert TestClient(app).post("/api/session/ping", headers={**basic("vic", PW), "Origin": "http://evil.example"}).status_code == 403
    # trusted origins (e.g. a reverse proxy on another name)
    site.cfg["auth"] = {"trusted_origins": ["https://argonix.example.org"]}
    app2 = create_app(site, auth=app.state.auth)
    c2 = login(app2, "olga")
    assert c2.post("/api/session/ping", headers={"Origin": "https://argonix.example.org"}).status_code == 200


def test_websocket_origin_and_auth(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    c = login(app, "vic")
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws", headers={"Origin": "http://evil.example"}) as s:
            s.receive_json()
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws") as s:
            s.receive_json()
    with c.websocket_connect("/ws", headers={"Origin": "http://testserver"}) as s:
        assert s.receive_json()["type"] == "tick"


def test_security_headers_and_cookie_flags(tmp_path):
    site, app = client_with_users(tmp_path, run=False)
    c = TestClient(app, headers=CSRF)
    for path in ("/", "/static/js/app.js", "/api/auth/status", "/api/alerts"):
        h = c.get(path).headers
        csp = h["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
        assert "script-src 'self';" in csp                                   # no inline/eval scripts
        assert "img-src 'self' blob: data:" in csp and "media-src 'self' blob: data:" in csp
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
        assert h["referrer-policy"] == "same-origin"
    assert c.get("/api/auth/status").headers["cache-control"] == "no-store"
    html = c.get("/").text
    assert "<script>" not in html and "onclick=" not in html                 # CSP-compatible page
    plain = c.post("/api/login", json={"username": "vic", "password": PW}).headers["set-cookie"].lower()
    assert "httponly" in plain and "samesite=strict" in plain and "secure" not in plain
    tls = TestClient(app, base_url="https://testserver", headers=CSRF)
    sc = tls.post("/api/login", json={"username": "vic", "password": PW}).headers["set-cookie"].lower()
    assert "; secure" in sc


def test_cli_manages_users_file(tmp_path, monkeypatch, capsys):
    f = tmp_path / "users.json"
    answers = iter([NEW_PW, NEW_PW])
    monkeypatch.setattr(auth_mod.getpass, "getpass", lambda prompt="": next(answers))
    auth_mod.main(["add", "cli_admin", "admin", "--temp", "--file", str(f)])
    rec = json.loads(f.read_text())["cli_admin"]
    assert rec["roles"] == ["admin"] and rec["must_change"] is True and NEW_PW not in f.read_text()
    weak = iter(["password1234", "password1234"])
    monkeypatch.setattr(auth_mod.getpass, "getpass", lambda prompt="": next(weak))
    with pytest.raises(SystemExit):
        auth_mod.main(["add", "other", "viewer", "--file", str(f)])
    auth_mod.main(["disable", "cli_admin", "--file", str(f)])
    assert json.loads(f.read_text())["cli_admin"]["disabled"] is True
    auth_mod.main(["list", "--file", str(f)])
    assert "cli_admin" in capsys.readouterr().out
