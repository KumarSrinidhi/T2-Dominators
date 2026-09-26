"""Phone / chat push for urgent incidents: Telegram Bot API and a generic ntfy-style HTTP POST.

Hooked into IncidentManager._decide: a channel sends only when the incident manager decided "would_push" (P1 exempt
from the budget, P2 through the token bucket, suppressed/inhibited incidents never), and only at or above the
channel's `min_priority`. Sending happens on a background thread with retries and backoff, so the pipeline never
waits on the network. Tokens are read from environment variables named in the config and never logged: errors are
scrubbed of the token and the target URL before they reach the audit log or the API.

    "notify": {
      "console_url": "https://argonix.lan:8000",
      "telegram": {"enabled": true, "token_env": "ARGONIX_TG_TOKEN", "chat_id": "123456789",
                   "min_priority": "P1", "send_snapshot": false},
      "http": {"enabled": true, "url": "https://ntfy.sh/<your-private-topic>", "min_priority": "P1",
               "token_env": "", "headers": {}}
    }
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from .rules import LABEL

PRIO = ("P1", "P2", "P3", "P4")
TELEGRAM_API = "https://api.telegram.org"
NTFY_PRIORITY = {"P1": "5", "P2": "4", "P3": "3", "P4": "2"}
DISCLAIMER = "Observed event; a person decides what to do."


def at_least(prio: str, min_prio: str) -> bool:
    return prio in PRIO and PRIO.index(prio) <= PRIO.index(min_prio if min_prio in PRIO else "P1")


class Channel:
    kind = "channel"

    def __init__(self, cfg: dict, env=os.environ):
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled"))
        self.min_priority = str(self.cfg.get("min_priority", "P1"))
        if self.min_priority not in PRIO:
            raise ValueError(f"notify.{self.kind}.min_priority must be one of {PRIO}")
        self.token_env = str(self.cfg.get("token_env") or "")
        self.token = env.get(self.token_env, "") if self.token_env else ""
        self.sent = self.failed = self.dropped = 0
        self.last: Optional[dict] = None

    def scrub(self, text: str) -> str:
        """Remove secrets (token, full target URL) from an error string."""
        text = str(text)
        for secret in (self.token, *self._secrets()):
            if secret:
                text = text.replace(secret, "***")
        return text[:300]

    def _secrets(self) -> tuple:
        return ()

    def problem(self) -> str:
        return ""

    def public(self) -> dict:
        return {"kind": self.kind, "enabled": self.enabled, "min_priority": self.min_priority,
                "token_env": self.token_env or None, "token_set": bool(self.token), "ready": not self.problem(),
                "problem": self.problem() or None, "sent": self.sent, "failed": self.failed, "dropped": self.dropped,
                "last": self.last}

    def send(self, client: httpx.Client, msg: dict) -> tuple[bool, str, bool]:
        """One attempt: (ok, detail, retryable)."""
        raise NotImplementedError


class TelegramChannel(Channel):
    kind = "telegram"

    def __init__(self, cfg: dict, env=os.environ):
        super().__init__(cfg, env)
        self.chat_id = str(self.cfg.get("chat_id") or "")
        self.send_snapshot = bool(self.cfg.get("send_snapshot", False))
        self.api = str(self.cfg.get("api_base") or TELEGRAM_API).rstrip("/")

    def problem(self) -> str:
        if not self.token:
            return f"environment variable {self.token_env or '(token_env not set)'} is empty"
        if not self.chat_id:
            return "chat_id is not set"
        return ""

    def public(self) -> dict:
        c = self.chat_id
        return {**super().public(), "chat_id": (c[:3] + "…" + c[-2:]) if len(c) > 6 else ("set" if c else None),
                "send_snapshot": self.send_snapshot}

    def send(self, client, msg):
        base = f"{self.api}/bot{self.token}"
        photo = msg.get("photo") if self.send_snapshot else None
        if photo:
            r = client.post(f"{base}/sendPhoto", data={"chat_id": self.chat_id, "caption": msg["text"][:1024]},
                            files={"photo": ("snapshot.jpg", photo, "image/jpeg")})
        else:
            r = client.post(f"{base}/sendMessage", json={"chat_id": self.chat_id, "text": msg["text"],
                                                          "disable_web_page_preview": True})
        if r.status_code == 200:
            return True, "HTTP 200" + (" (with snapshot)" if photo else ""), False
        desc = ""
        try:
            desc = str(r.json().get("description", ""))
        except ValueError:
            pass
        return False, self.scrub(f"HTTP {r.status_code} {desc}".strip()), r.status_code == 429 or r.status_code >= 500


class HttpChannel(Channel):
    """ntfy-style: POST the message text to a URL with Title / Priority / Tags / Click headers."""
    kind = "http"

    def __init__(self, cfg: dict, env=os.environ):
        super().__init__(cfg, env)
        self.url = str(self.cfg.get("url") or "")
        self.headers = {str(k): str(v) for k, v in (self.cfg.get("headers") or {}).items()}

    def _secrets(self):
        p = urlparse(self.url)
        return (self.url, p.path) if p.path not in ("", "/") else (self.url,)

    def problem(self) -> str:
        if not self.url.startswith(("http://", "https://")):
            return "url must start with http:// or https://"
        if self.token_env and not self.token:
            return f"environment variable {self.token_env} is empty"
        return ""

    def public(self) -> dict:
        p = urlparse(self.url)
        return {**super().public(), "host": p.hostname or None}

    def send(self, client, msg):
        h = {"Title": msg["title"].encode("ascii", "replace").decode(), "Priority": NTFY_PRIORITY.get(msg["priority"], "3"),
             "Tags": "rotating_light" if msg["priority"] == "P1" else "warning", **self.headers}
        if msg.get("url"):
            h["Click"] = msg["url"]
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        r = client.post(self.url, content=msg["text"].encode("utf-8"), headers=h)
        if 200 <= r.status_code < 300:
            return True, f"HTTP {r.status_code}", False
        return False, f"HTTP {r.status_code}", r.status_code == 429 or r.status_code >= 500


class Notifier:
    """Formats incident pushes and delivers them to the enabled channels on one background thread."""

    def __init__(self, cfg: Optional[dict] = None, *, audit: Optional[Callable[[str, str, str, dict], None]] = None,
                 record: Optional[Callable[..., None]] = None, data_dir: Optional[Path] = None,
                 camera_names: Optional[Callable[[], dict]] = None, env=os.environ,
                 transport: Optional[httpx.BaseTransport] = None, retries: int = 3, backoff_s: float = 1.0,
                 timeout_s: float = 10.0):
        cfg = dict(cfg or {})
        self.console_url = str(cfg.get("console_url") or "").rstrip("/")
        self.include_demo = bool(cfg.get("include_demo", True))
        self.channels: list[Channel] = [TelegramChannel(cfg.get("telegram") or {}, env),
                                        HttpChannel(cfg.get("http") or {}, env)]
        self.audit = audit or (lambda actor, action, alert_id, details: None)
        self.record = record
        self.data_dir = Path(data_dir) if data_dir else None
        self.camera_names = camera_names or (lambda: {})
        self.retries, self.backoff_s = max(1, int(retries)), float(backoff_s)
        self.client = httpx.Client(transport=transport, timeout=timeout_s)
        self.q: queue.Queue = queue.Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    @property
    def active(self) -> list[Channel]:
        return [c for c in self.channels if c.enabled]

    # ---- message -------------------------------------------------------------------------------------------------
    def message(self, inc: dict, rec: dict) -> dict:
        names = {}
        try:
            names = dict(self.camera_names() or {})
        except Exception:  # noqa: BLE001
            pass
        prio = rec.get("priority") or inc.get("priority") or "P4"
        evs = inc.get("events") or []
        ev = next((e for e in reversed(evs) if e.get("alert_id") == rec.get("alert_id")), evs[-1] if evs else {})
        rule = ev.get("rule") or (inc.get("rules") or ["?"])[0]
        label = LABEL.get(rule, rule)
        cams = inc.get("cameras") or [ev.get("camera_id", "?")]
        cam_txt = ", ".join(f"{names.get(c, c)} ({c})" if names.get(c, c) != c else c for c in cams)
        zone = ev.get("zone_name") or ", ".join(inc.get("zones") or []) or "—"
        t = float(ev.get("t_end") or rec.get("t") or time.time())
        demo = bool(inc.get("demo"))
        url = f"{self.console_url}/#/incidents/{inc['id']}" if self.console_url else None
        title = f"{'[DEMO] ' if demo else ''}{prio} {label}"
        lines = [title, f"Camera: {cam_txt}", f"Zone: {zone}",
                 f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}",
                 f"Incident {inc['id']}" + (f" · alert {rec['alert_id']}" if rec.get("alert_id") else "")
                 + (f" · {rec['trigger']}" if rec.get("trigger") and rec["trigger"] != "new" else "")]
        if url:
            lines.append(url)
        if demo:
            lines.append("DEMO scenario: scripted, not observed.")
        lines.append(DISCLAIMER)
        snap = (ev.get("evidence") or {}).get("snapshot")
        return {"title": title, "text": "\n".join(lines), "priority": prio, "url": url, "snapshot": snap,
                "incident_id": inc["id"], "alert_id": rec.get("alert_id", ""), "demo": demo}

    def _photo(self, rel: Optional[str]) -> Optional[bytes]:
        if not rel or self.data_dir is None:
            return None
        root = self.data_dir.resolve()
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file() or f.suffix.lower() != ".jpg":
            return None
        return f.read_bytes()

    # ---- hook (called under the incident lock: must not block) ------------------------------------------------------
    def __call__(self, inc: dict, rec: dict) -> Optional[str]:
        if inc.get("demo") and not self.include_demo:
            return None
        targets = [c for c in self.active if at_least(rec.get("priority", "P4"), c.min_priority)]
        if not targets:
            return None
        msg = self.message(inc, rec)
        out = []
        for c in targets:
            if c.problem():
                out.append(f"{c.kind}: not sent ({c.problem()})")
                continue
            try:
                self.q.put_nowait((c, msg, "system"))
                out.append(f"{c.kind}: queued")
            except queue.Full:
                c.dropped += 1
                out.append(f"{c.kind}: dropped (queue full)")
        self._ensure_thread()
        return "; ".join(out)

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._closed:
            self._thread = threading.Thread(target=self._run, daemon=True, name="notify")
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            ch, msg, actor = item
            self.deliver(ch, msg, actor)

    def deliver(self, ch: Channel, msg: dict, actor: str = "system", retries: Optional[int] = None) -> dict:
        """Send with retries and exponential backoff; audit and record the outcome (never the token)."""
        m = dict(msg)
        if isinstance(ch, TelegramChannel) and ch.send_snapshot:
            try:
                m["photo"] = self._photo(msg.get("snapshot"))
            except OSError:
                m["photo"] = None
        ok, detail, attempts = False, "not sent", 0
        for attempt in range(retries or self.retries):
            attempts = attempt + 1
            try:
                ok, detail, retry = ch.send(self.client, m)
            except Exception as e:  # noqa: BLE001 - network errors are retried, then recorded (scrubbed)
                ok, detail, retry = False, ch.scrub(f"{type(e).__name__}: {e}"), True
            if ok or not retry or attempts >= (retries or self.retries):
                break
            time.sleep(min(30.0, self.backoff_s * 2 ** attempt))
        if ok:
            ch.sent += 1
        else:
            ch.failed += 1
        res = {"channel": ch.kind, "ok": ok, "detail": detail, "attempts": attempts, "t": round(time.time(), 3),
               "incident_id": msg.get("incident_id", ""), "priority": msg.get("priority"), "test": bool(msg.get("test"))}
        ch.last = res
        self.audit(actor, "notify_sent" if ok else "notify_failed", msg.get("alert_id", ""),
                   {k: v for k, v in res.items() if k != "t"})
        if self.record is not None and not msg.get("test"):
            try:
                self.record(time.time(), msg.get("incident_id", ""), msg.get("alert_id", ""), msg.get("priority", ""),
                            ch.kind, f"{ch.kind}_{'sent' if ok else 'failed'}", detail)
            except Exception:  # noqa: BLE001
                pass
        return res

    # ---- operator actions ----------------------------------------------------------------------------------------
    def test(self, actor: str, channel: str = "") -> list[dict]:
        """Send a test message now (synchronously, one attempt per channel) and return the results."""
        chans = [c for c in self.channels if (c.kind == channel if channel else c.enabled)]
        if not chans:
            raise ValueError("no notification channel is enabled (config \"notify\")" if not channel
                             else f"unknown channel {channel!r}")
        msg = {"title": "Argus test notification", "priority": "P4", "url": self.console_url or None,
               "text": "Argus test notification\nIf you can read this, phone push works.\n"
                       f"Sent {time.strftime('%Y-%m-%d %H:%M:%S')} by {actor}.",
               "incident_id": "", "alert_id": "", "test": True, "snapshot": None}
        out = []
        for c in chans:
            if c.problem():
                res = {"channel": c.kind, "ok": False, "detail": f"not configured: {c.problem()}", "attempts": 0}
                self.audit(actor, "notify_test_failed", "", res)
                out.append(res)
            else:
                out.append(self.deliver(c, msg, actor, retries=1))
        return out

    def status(self) -> dict:
        return {"channels": [c.public() for c in self.channels], "queued": self.q.qsize(),
                "console_url": self.console_url or None, "include_demo": self.include_demo,
                "note": "Sends only incidents the push policy marked 'would push' (P1 always, P2 within the push "
                        "budget, never muted ones), at or above each channel's min_priority. Tokens come from "
                        "environment variables and are never logged."}

    def close(self) -> None:
        self._closed = True
        if self._thread is not None:
            self.q.put(None)
            self._thread.join(5)
        self.client.close()


def install(site) -> Optional[Notifier]:
    """Attach a Notifier to the site and its incident manager (config "notify"; channels off by default)."""
    cfg = dict(site.cfg.get("notify") or {})
    store = site.store
    try:
        n = Notifier(cfg, audit=lambda actor, action, aid, details: store.audit(actor, action, aid, details),
                     record=store.add_notification, data_dir=site.data,
                     camera_names=lambda: {w.id: w.name for w in site.workers})
    except ValueError as e:
        raise ValueError(f"site config: {e}")
    site.notifier = n
    inc = getattr(site, "incidents", None)
    if inc is not None and n.active:
        inc.notifier = n
    return n
