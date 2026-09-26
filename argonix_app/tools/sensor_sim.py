"""Access-control panel simulator: signs (Standard Webhooks HMAC-SHA256) and posts door events to a running Argus.

Standard library only, so it runs anywhere a real panel would sit. The secret comes from an environment variable (the
same one the source's `secret_env` names in the site config), never from the command line.

    python tools/sensor_sim.py new-secret                       # print a fresh whsec_ secret + how to set it
    python tools/sensor_sim.py forced --door D1                 # one "door forced" report
    python tools/sensor_sim.py held --door D1 --held-s 45       # door held open
    python tools/sensor_sim.py denials --door D1 --count 3      # access-denied burst (3 in a few seconds)
    python tools/sensor_sim.py demo                             # forced + held open + denial burst, spaced out
    python tools/sensor_sim.py heartbeat --every 10             # keep the source "online" (Ctrl+C: it goes silent)

Options: --url (default http://127.0.0.1:8000), --source (door-ctrl-1), --secret-env (ARGONIX_SENSOR_SECRET_DOOR1),
--token-env NAME to authenticate with an API token (role "integration") instead of signing, --insecure for a
self-signed HTTPS certificate, --replay to send the last request twice (the second must be refused with 409).
Card ids are synthetic ("SIM-0001"...); Argus hashes them on arrival.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid


def sign(secret: str, msg_id: str, ts: int, body: str) -> str:
    key = base64.b64decode(secret[len("whsec_"):]) if secret.startswith("whsec_") else secret.encode()
    return "v1," + base64.b64encode(hmac.new(key, f"{msg_id}.{ts}.{body}".encode(), hashlib.sha256).digest()).decode()


class Panel:
    def __init__(self, a: argparse.Namespace):
        self.url = a.url.rstrip("/") + "/api/sensors/events"
        self.source = a.source
        self.secret = os.environ.get(a.secret_env) if not a.token_env else None
        self.token = os.environ.get(a.token_env) if a.token_env else None
        if not (self.secret or self.token):
            sys.exit(f"set {a.token_env or a.secret_env} first (python tools/sensor_sim.py new-secret)")
        self.ctx = ssl._create_unverified_context() if a.insecure else None  # noqa: S323 - opt-in for self-signed
        self.replay = a.replay

    def post(self, typ: str, door: str | None = None, **extra) -> dict:
        body = {"source_id": self.source, "type": typ, "ts": round(time.time(), 3)}
        if door:
            body["door"] = door
        body.update({k: v for k, v in extra.items() if v is not None})
        raw = json.dumps(body)
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        else:
            msg_id, ts = f"sim_{uuid.uuid4().hex}", int(time.time())
            headers.update({"webhook-id": msg_id, "webhook-timestamp": str(ts),
                            "webhook-signature": sign(self.secret, msg_id, ts, raw)})
        out = self._send(raw, headers)
        print(f"{typ:15s} {door or '-':4s} -> {out}")
        if self.replay and not self.token:
            print(f"{'(replay)':15s} {door or '-':4s} -> {self._send(raw, headers)}")
        return out

    def _send(self, raw: str, headers: dict) -> dict:
        req = urllib.request.Request(self.url, raw.encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10, context=self.ctx) as r:
                return {"http": r.status, **json.loads(r.read() or b"{}")}
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except ValueError:
                detail = e.reason
            return {"http": e.code, "error": detail}
        except urllib.error.URLError as e:
            sys.exit(f"cannot reach {self.url}: {e.reason}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=["new-secret", "forced", "held", "denials", "granted", "alarm", "tamper",
                                         "heartbeat", "demo"])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default="door-ctrl-1")
    ap.add_argument("--door", default="D1")
    ap.add_argument("--secret-env", default="ARGONIX_SENSOR_SECRET_DOOR1")
    ap.add_argument("--token-env", default=None, help="env var holding an API token (instead of HMAC signing)")
    ap.add_argument("--count", type=int, default=3, help="denials: how many")
    ap.add_argument("--held-s", type=float, default=45.0)
    ap.add_argument("--every", type=float, default=10.0, help="heartbeat: interval in seconds")
    ap.add_argument("--insecure", action="store_true", help="accept a self-signed HTTPS certificate")
    ap.add_argument("--replay", action="store_true", help="send each request twice (the copy must get 409)")
    a = ap.parse_args(argv)

    if a.scenario == "new-secret":
        s = "whsec_" + base64.b64encode(secrets.token_bytes(32)).decode()
        print(s)
        print(f"\nPowerShell:  $env:{a.secret_env} = \"{s}\"\nbash:        export {a.secret_env}='{s}'\n"
              "Set it in the shell that runs Argus AND in the one that runs this simulator.", file=sys.stderr)
        return
    p = Panel(a)
    if a.scenario == "forced":
        p.post("door_forced", a.door)
    elif a.scenario == "held":
        p.post("door_held_open", a.door, details={"held_s": a.held_s})
    elif a.scenario == "granted":
        p.post("access_granted", a.door, card_id="SIM-0001")
    elif a.scenario == "alarm":
        p.post("alarm_input", a.door, details={"input": "IN1"})
    elif a.scenario == "tamper":
        p.post("tamper", a.door)
    elif a.scenario == "denials":
        for i in range(a.count):
            p.post("access_denied", a.door, card_id=f"SIM-{1000 + i % 2}")
            time.sleep(1.0)
    elif a.scenario == "heartbeat":
        try:
            while True:
                p.post("heartbeat")
                time.sleep(a.every)
        except KeyboardInterrupt:
            print("stopped: the source goes silent (sensor_offline after its heartbeat_s)")
    elif a.scenario == "demo":
        p.post("door_forced", a.door)
        time.sleep(5)
        p.post("door_held_open", a.door, details={"held_s": a.held_s})
        time.sleep(5)
        for i in range(a.count):
            p.post("access_denied", a.door, card_id=f"SIM-{1000 + i % 2}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
