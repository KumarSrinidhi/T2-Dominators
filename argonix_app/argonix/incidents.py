"""Incidents: group correlated alerts across time, cameras and zones into one operator-facing situation.

The Correlator (argonix.correlate) merges events into *alerts* inside a short window (15 s). An *incident* is a
longer session over alerts: every alert belongs to exactly one incident, and an incident collects the alerts that
share a zone key (site zone, or camera+zone), a camera track, or (for camera-integrity problems) a camera.

Lifecycle (manual §10 C-04; times are event time = max(wall clock, newest event time)):
  OPEN     first alert. Incidents below P2 are held for `hold_s` (10 s) so a burst settles before any push decision.
  ACTIVE   the push decision is made on activation; stays ACTIVE while events keep arriving.
  COOLING  `cool_after_s` (60 s) without a new event. Camera-integrity incidents stay ACTIVE while the camera's
           health is still DEGRADED/DOWN (the condition persists without new transition events).
  CLOSED   `close_after_s` (240 s) more; or the operator resolves/closes it, merges it away, or it rolls over after
           `max_duration_s`. A matching alert within `reopen_window_s` (600 s) of a quiescence/operator close reopens
           the same incident (reopen_count + 1).
Operators merge incidents, split alerts out into a new incident, assign, change status and write a summary.

Cascade inhibition (manual §10 C-06, Alertmanager semantics): while a camera-integrity incident (camera_health) is
open on a camera, new rule alerts from that camera are annotated `inhibited_by` / `suppressed` (still stored and
visible, never pushed). The integrity alert itself is never inhibited.

Notification budget (manual §10 C-11): only P1/P2 may interrupt. P2 draws from an event-time token bucket of 6
pushes per operator-hour (burst 3); P1 is exempt. Decisions are stored as "would_push" / "budget_denied" /
"suppressed" / "not_push_tier" rows (nothing is sent), plus an optional signed webhook (Standard Webhooks v1,
HMAC-SHA256), disabled by default. Grouping makes no identity claim and never assesses intent.

Acknowledge SLA (argonix.workflow): a P1/P2 incident whose alerts are all still "new" (and whose own status is still
"open") past its priority's `sla_s` (P1 60 s, P2 300 s by default; measured from when the incident reached that
priority) gets one "sla_overdue" escalation: a notification row (exempt from the push budget, still muted by
inhibition), the webhook again if configured, a timeline entry and an audit record. Once per incident.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import itertools
import json
import queue
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

from .models import SENSOR_PREFIX, alert_is_demo, is_demo
from .rules import LABEL
from .workflow import sla_config

PHASES = ("open", "active", "cooling", "closed")
STATUSES = ("open", "acknowledged", "investigating", "resolved", "closed")
PRIO = ("P1", "P2", "P3", "P4")
INTEGRITY_RULES = frozenset({"camera_health", "sensor_offline"})
NON_MUTING_INTEGRITY = frozenset({"sensor_offline"})  # a silent door controller says nothing about camera evidence
MAX_EVENTS = 300         # event refs kept per incident (newest); n_events counts all
MAX_TIMELINE = 400
EDGE_TYPES = ("observed_in", "same_track", "corroborates", "predicted_next", "inhibited_by", "evidence",
              "reported_by", "covered_by")
DISCLAIMER = ("An incident groups alerts that share a zone, a camera track or (for camera problems) a camera within "
              "a time window. Grouping is deterministic and spatial/temporal only: it makes no identity claim and "
              "does not assess intent. Phase changes are automatic; status and every decision rest with the operator.")

DEFAULTS: dict = {
    "enabled": True,
    "hold_s": 10.0,             # OPEN hold-down for incidents below P2
    "cool_after_s": 60.0,       # ACTIVE -> COOLING after this long without a new event
    "close_after_s": 240.0,     # COOLING -> CLOSED after this long more
    "reopen_window_s": 600.0,   # a matching alert this soon after a close reopens the same incident
    "max_duration_s": 3600.0,   # rollover: a very long incident is closed and a new one continues
    "push_tiers": ["P1", "P2"],
    "budget_exempt": ["P1"],
    "push_budget_per_operator_hour": 6.0,
    "push_burst": 3.0,
    "operators_on_duty": 1,
    "renotify_on_reopen": True,
    "inhibit": [{"source_rules": ["camera_health"], "target_rules": "*", "equal": "camera"}],
    "webhook": {"url": "", "secret": "", "timeout_s": 5.0, "retries": 3, "console_url": ""},
    "sla_s": None,              # per-priority acknowledge SLA; None = the site config "sla_s" / workflow defaults
    "sla_tiers": ["P1", "P2"],  # tiers whose missed SLA escalates (notification + webhook)
}


def rank(p: Optional[str]) -> int:
    return PRIO.index(p) if p in PRIO else len(PRIO)


def best(ps) -> Optional[str]:
    ps = [p for p in ps if p in PRIO]
    return min(ps, key=rank) if ps else None


def is_integrity(rule: str) -> bool:
    return rule in INTEGRITY_RULES


def zone_key(ev: dict) -> str:
    return ev.get("site_zone") or f"{ev['camera_id']}:{ev['zone_id']}"


# ---------------------------------------------------------------------------------------------------------------
# Push budget and webhook
# ---------------------------------------------------------------------------------------------------------------
class TokenBucket:
    """Event-time token bucket: capacity `burst`, refill `rate_per_hour`. Deterministic on replay."""

    def __init__(self, rate_per_hour: float, burst: float):
        self.cap = float(burst)
        self.rate = float(rate_per_hour) / 3600.0
        self.tokens = float(burst)
        self.t: Optional[float] = None

    def _refill(self, t: float) -> None:
        if self.t is None:
            self.t = t
        elif t > self.t:
            self.tokens = min(self.cap, self.tokens + (t - self.t) * self.rate)
            self.t = t

    def try_take(self, t: float) -> bool:
        self._refill(t)
        if self.tokens >= 1.0 - 1e-9:
            self.tokens -= 1.0
            return True
        return False


def sign_webhook(secret: str, msg_id: str, ts: int, body: str) -> str:
    """Standard Webhooks v1 signature: base64(HMAC-SHA256(key, "{id}.{timestamp}.{body}")), key = base64 part of a
    'whsec_' secret (a secret without the prefix is used as raw UTF-8 bytes)."""
    key = base64.b64decode(secret[len("whsec_"):]) if secret.startswith("whsec_") else secret.encode()
    mac = hmac.new(key, f"{msg_id}.{ts}.{body}".encode(), hashlib.sha256).digest()
    return "v1," + base64.b64encode(mac).decode()


def webhook_headers(secret: str, msg_id: str, ts: int, body: str) -> dict:
    h = {"content-type": "application/json", "webhook-id": msg_id, "webhook-timestamp": str(ts)}
    if secret:
        h["webhook-signature"] = sign_webhook(secret, msg_id, ts, body)
    return h


class WebhookSender:
    """Background sender (never blocks the pipeline). Retries with backoff; HTTP 410 disables the endpoint."""

    def __init__(self, url: str, secret: str = "", timeout_s: float = 5.0, retries: int = 3,
                 on_result: Optional[Callable[[dict, str, str], None]] = None, post: Optional[Callable] = None):
        self.url, self.secret, self.timeout, self.retries = url, secret, float(timeout_s), int(retries)
        self.on_result = on_result
        self._post = post
        self.disabled = False
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.thread = threading.Thread(target=self._run, daemon=True, name="incident-webhook")
        self.thread.start()

    def enqueue(self, msg_id: str, payload: dict) -> bool:
        if self.disabled:
            return False
        try:
            self.q.put_nowait((msg_id, payload))
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(5)

    def send(self, msg_id: str, payload: dict) -> tuple[str, str]:
        import httpx
        post = self._post or httpx.post
        body = json.dumps(payload, separators=(",", ":"))
        last = "not sent"
        for attempt in range(max(1, self.retries)):
            ts = int(time.time())
            try:
                r = post(self.url, content=body, headers=webhook_headers(self.secret, msg_id, ts, body),
                         timeout=self.timeout)
                if 200 <= r.status_code < 300:
                    return "sent", f"HTTP {r.status_code}"
                last = f"HTTP {r.status_code}"
                if r.status_code == 410:
                    self.disabled = True
                    return "failed", "HTTP 410 Gone: endpoint disabled"
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return "failed", last
            except Exception as e:  # noqa: BLE001 - network errors are retried and then recorded
                last = f"{type(e).__name__}: {e}"[:200]
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
        return "failed", last

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            msg_id, payload = item
            status, detail = self.send(msg_id, payload)
            if self.on_result:
                try:
                    self.on_result(payload, status, detail)
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------------------------------------------
# Incident manager
# ---------------------------------------------------------------------------------------------------------------
class IncidentManager:
    """Groups alerts into incidents. `on_alert(d)` is called by Site.on_event with the alert dict before it is
    stored (under the site lock); it annotates d with `incident_id` (and `inhibited_by`/`suppressed`)."""

    def __init__(self, store, cfg: Optional[dict] = None, clock: Callable[[], float] = time.time, *,
                 alert_lock=None, health: Optional[Callable[[str], Optional[str]]] = None,
                 predictions: Optional[Callable[[], list]] = None,
                 reid: Optional[Callable[[str, int], Optional[dict]]] = None,
                 camera_names: Optional[Callable[[], dict]] = None, webhook_post: Optional[Callable] = None):
        cfg = dict(cfg or {})
        self.cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "webhook"}}
        self.cfg["webhook"] = {**DEFAULTS["webhook"], **(cfg.get("webhook") or {})}
        self.store = store
        self.clock = clock
        self.alert_lock = alert_lock if alert_lock is not None else threading.RLock()
        self.lock = threading.RLock()
        self.health, self.predictions, self.reid = health, predictions, reid
        self.camera_names = camera_names or (lambda: {})
        self.watermark = 0.0
        self.live: dict[str, dict] = {}
        self.alert_map: dict[str, str] = {}
        k = max(1, int(self.cfg["operators_on_duty"]))
        self.bucket = TokenBucket(float(self.cfg["push_budget_per_operator_hour"]) * k, float(self.cfg["push_burst"]) * k)
        self.sla = sla_config(self.cfg)
        with store.lock:
            n = store.db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        self._ids = itertools.count(int(n) + 1)
        self._pred_cache: tuple[float, dict] = (0.0, {})
        self._last_pred_refresh = 0.0
        wh = self.cfg["webhook"]
        self.webhook = (WebhookSender(wh["url"], wh.get("secret") or "", wh.get("timeout_s", 5.0),
                                      wh.get("retries", 3), self._webhook_result, webhook_post)
                        if wh.get("url") else None)
        # optional phone/chat push (argonix/notify.py): called with (incident, decision record) on "would_push"
        self.notifier: Optional[Callable[[dict, dict], Optional[str]]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warm_start()

    # ---- lifecycle of the manager ------------------------------------------------------------------------------
    def _warm_start(self) -> None:
        """Reload live (and recently closed, still reopenable) incidents after a restart."""
        now = self.clock()
        for inc in self.store.list_incidents(live_or_closed_after=now - float(self.cfg["reopen_window_s"]),
                                             limit=500):
            self.live[inc["id"]] = inc
            for aid in inc.get("alert_ids", []):
                self.alert_map[aid] = inc["id"]
            self.watermark = max(self.watermark, float(inc.get("last_event_at") or 0))

    def start(self, interval_s: float = 1.0) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, args=(interval_s,), daemon=True, name="incidents")
            self._thread.start()

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the lifecycle thread must survive a bad record
                pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None
        if self.webhook is not None:
            self.webhook.close()

    # ---- time ----------------------------------------------------------------------------------------------------
    def now(self, t: Optional[float] = None) -> float:
        if t is not None:
            self.watermark = max(self.watermark, float(t))
        return max(float(self.clock()), self.watermark)

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _tl(self, inc: dict, t: float, kind: str, text: str, actor: Optional[str] = None, **extra) -> None:
        e = {"t": round(float(t), 3), "kind": kind, "text": text}
        if actor:
            e["actor"] = actor
        e.update(extra)
        inc["timeline"].append(e)
        if len(inc["timeline"]) > MAX_TIMELINE:
            inc["timeline"] = inc["timeline"][:20] + inc["timeline"][-(MAX_TIMELINE - 20):]

    def _save(self, inc: dict) -> None:
        self.store.upsert_incident(inc)

    def _get(self, iid: Optional[str]) -> Optional[dict]:
        if not iid:
            return None
        inc = self.live.get(iid)
        if inc is None:
            inc = self.store.get_incident(iid)
        return inc

    def _follow(self, inc: Optional[dict]) -> Optional[dict]:
        seen = set()
        while inc is not None and inc.get("merged_into") and inc["id"] not in seen:
            seen.add(inc["id"])
            inc = self._get(inc["merged_into"])
        return inc

    def _hold(self, inc: dict) -> dict:
        """Keep an incident being modified in memory (so later lookups see the same object)."""
        self.live[inc["id"]] = inc
        return inc

    def _reopenable(self, inc: dict, now: float) -> bool:
        return (inc["phase"] == "closed" and not inc.get("merged_into")
                and inc.get("closed_by") in ("quiescence", "operator")
                and now - float(inc.get("closed_at") or 0) <= float(self.cfg["reopen_window_s"]))

    def _keys(self, ev: dict, kind: str) -> set[str]:
        cam = ev["camera_id"]
        if kind == "integrity":
            ks = {f"camera:{cam}"}
        else:
            ks = {f"zone:{zone_key(ev)}"}
            ks |= {f"track:{cam}:{t}" for t in (ev.get("track_ids") or [])[:20] if t is not None}
        # demo-scenario events (scripted, not observed) never share an incident with observed ones
        return {f"demo:{k}" for k in ks} if is_demo(ev) else ks

    def _names(self) -> dict:
        try:
            return dict(self.camera_names() or {})
        except Exception:  # noqa: BLE001
            return {}

    def _blank(self, kind: str, t: float, priority: str) -> dict:
        iid = f"I{int(t)}-{next(self._ids)}"
        return {"id": iid, "kind": kind, "keys": [], "keyless": False, "phase": "open", "status": "open",
                "priority": priority, "score": 0.0, "title": "", "summary": "", "auto_summary": "",
                "assignee": None, "cameras": [], "zones": [], "zone_keys": [], "rules": [], "tracks": [],
                "alert_ids": [], "alerts": {}, "events": [], "n_events": 0,
                "opened_at": t, "activated_at": None, "last_event_at": t, "quiet_since": t, "cooling_at": None,
                "closed_at": None, "closed_by": None, "updated_at": t, "reopen_count": 0,
                "merged_into": None, "merged_from": [], "merged_alert_ids": [], "split_from": None,
                "inhibited_by": None, "suppressed": False, "predictions": {}, "global_ids": {},
                "timeline": [], "pushes": [], "last_decided_priority": None, "highest_pushed": None,
                "disclaimer": DISCLAIMER}

    # ---- main entry: an alert was created or updated ---------------------------------------------------------------
    def on_alert(self, d: dict) -> Optional[dict]:
        if not d.get("events"):
            return None
        ev = d["events"][-1]
        with self.lock:
            now = self.now(float(ev.get("t_end") or 0))
            self._tick_locked(now)
            kind = "integrity" if is_integrity(ev["rule"]) else "behaviour"
            inc, reason = self._find(d, ev, kind, now)
            reopened = False
            if inc is None:
                inc = self._hold(self._blank(kind, float(ev.get("t_end") or now), d.get("priority", "P4")))
                held = rank(d.get("priority")) > rank("P2")
                self._tl(inc, inc["opened_at"], "opened",
                         f"Opened by alert {d['id']}" + (f" (held {self.cfg['hold_s']:g} s: below P2, burst settles "
                                                         "before any push decision)" if held else ""))
                reason = "first alert"
            elif inc["phase"] == "closed":
                self._reopen(inc, now, f"matching alert {d['id']} ({reason})")
                reopened = True
            inhibitor = self._inhibitor(ev, kind)
            self._join(inc, d, ev, now, reason, inhibitor)
            inc.setdefault("prio_since", {}).setdefault(inc["priority"], float(ev.get("t_end") or now))
            self._apply_alert_flags(d, inc)
            self._advance(inc, now)
            if reopened and self.cfg["renotify_on_reopen"] and inc["phase"] == "active":
                self._decide(inc, now, "reopened", d["id"])
            elif inc["phase"] in ("active", "cooling") and inc.get("last_decided_priority") and \
                    rank(inc["priority"]) < rank(inc["last_decided_priority"]):
                self._decide(inc, now, "escalated", d["id"])
            inc["updated_at"] = max(float(inc["updated_at"]), now)
            self._save(inc)
            return inc

    def _find(self, d: dict, ev: dict, kind: str, now: float) -> tuple[Optional[dict], str]:
        iid = self.alert_map.get(d["id"]) or self.store.incident_for_alert(d["id"])
        if iid:
            inc = self._follow(self._get(iid))
            if inc is not None and (inc["phase"] != "closed" or self._reopenable(inc, now)):
                self._hold(inc)
                return inc, "same alert"
        keys = self._keys(ev, kind)
        cands = []
        for inc in self.live.values():
            if inc["kind"] != kind or inc.get("merged_into") or inc.get("keyless"):
                continue
            shared = keys & set(inc.get("keys") or [])
            if not shared:
                continue
            if inc["phase"] != "closed" or self._reopenable(inc, now):
                cands.append((inc["phase"] != "closed", float(inc.get("last_event_at") or 0), inc, sorted(shared)))
        if not cands:
            return None, ""
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        _, _, inc, shared = cands[0]
        k = shared[0]
        why = ("same camera " + k.split(":", 1)[1] if k.startswith("camera:") else
               "same zone " + k.split(":", 1)[1] if k.startswith("zone:") else
               "shared track " + k.split(":", 1)[1].replace(":", " #"))
        return inc, why

    def _inhibitor(self, ev: dict, kind: str) -> Optional[dict]:
        if kind == "integrity":
            return None
        for rule in self.cfg.get("inhibit") or []:
            src = set(rule.get("source_rules") or [])
            tgt = rule.get("target_rules", "*")
            if ev["rule"] in src or (tgt != "*" and ev["rule"] not in set(tgt)):
                continue
            equal = rule.get("equal", "camera")
            found = []
            for inc in self.live.values():
                if inc["phase"] == "closed" or inc.get("merged_into") or not (set(inc["rules"]) & src):
                    continue
                if inc.get("demo") and not is_demo(ev):
                    continue
                if equal == "camera" and ev["camera_id"] in inc["cameras"]:
                    found.append(inc)
                elif equal == "zone" and zone_key(ev) in inc.get("zone_keys", []):
                    found.append(inc)
            if found:
                return max(found, key=lambda i: float(i.get("last_event_at") or 0))
        return None

    def _join(self, inc: dict, d: dict, ev: dict, now: float, reason: str, inhibitor: Optional[dict]) -> None:
        aid = d["id"]
        new_alert = aid not in inc["alerts"]
        summ = inc["alerts"].setdefault(aid, {})
        summ.update(priority=d.get("priority"), score=d.get("score"), title=d.get("title"),
                    created_at=d.get("created_at"), updated_at=d.get("updated_at"), cameras=list(d.get("cameras", [])),
                    zones=list(d.get("zones", [])), rules=sorted({e["rule"] for e in d["events"]}))
        if new_alert:
            inc["alert_ids"].append(aid)
            self.alert_map[aid] = inc["id"]
            self.store.link_alert(aid, inc["id"])
            self._tl(inc, ev.get("t_end") or now, "alert", f"Alert {aid} joined ({reason}): {d.get('title', '')}",
                     alert_id=aid)
        cam_names = self._names()
        if inhibitor is not None:
            src_cam = ev["camera_id"]
            summ["inhibited_by"] = inhibitor["id"]
            summ["suppressed_reason"] = (f"Muted (not pushed) while incident {inhibitor['id']} (camera view problem on "
                                         f"{cam_names.get(src_cam, src_cam)}) is open; evidence from this camera may "
                                         "be unreliable.")
            if inc.get("inhibited_by") != inhibitor["id"]:
                inc["inhibited_by"] = inhibitor["id"]
                self._tl(inc, ev.get("t_end") or now, "inhibited",
                         f"Events from {cam_names.get(src_cam, src_cam)} are muted by camera-integrity incident "
                         f"{inhibitor['id']}", source=inhibitor["id"])
        ref = {"alert_id": aid, "rule": ev["rule"], "camera_id": ev["camera_id"], "zone_id": ev.get("zone_id"),
               "zone_name": ev.get("zone_name"), "site_zone": ev.get("site_zone"),
               "track_ids": list(ev.get("track_ids") or [])[:50], "t_start": ev.get("t_start"), "t_end": ev.get("t_end"),
               "confidence": ev.get("confidence"), "severity": ev.get("severity"),
               "observation": str(ev.get("observation", ""))[:300],
               "learned_model": bool((ev.get("derivation") or {}).get("learned_model")),
               **({"demo": True} if is_demo(ev) else {}),
               "evidence": {k: (ev.get("evidence") or {}).get(k) for k in ("snapshot", "clip", "crop")
                            if (ev.get("evidence") or {}).get(k)}}
        if inhibitor is not None:
            ref["inhibited_by"] = inhibitor["id"]
        deriv = ev.get("derivation") or {}
        if deriv.get("modality") and deriv["modality"] != "video":  # non-video source (access control)
            ref["modality"] = deriv["modality"]
            ref["sensor"] = {"source_id": deriv.get("source_id"), "door": deriv.get("door"),
                             "event_type": deriv.get("event_type"),
                             "camera": (ev.get("evidence") or {}).get("camera") or deriv.get("covering_camera")}
        sig =(aid, ref["rule"], ref["camera_id"], ref["zone_id"], ref["t_start"], tuple(ref["track_ids"]))
        if not any((e["alert_id"], e["rule"], e["camera_id"], e["zone_id"], e["t_start"], tuple(e["track_ids"])) == sig
                   for e in inc["events"][-30:]):
            inc["events"].append(ref)
            inc["n_events"] = int(inc.get("n_events", 0)) + 1
            if len(inc["events"]) > MAX_EVENTS:
                inc["events"] = inc["events"][-MAX_EVENTS:]
        t_end = float(ev.get("t_end") or now)
        inc["last_event_at"] = max(float(inc["last_event_at"]), t_end)
        inc["quiet_since"] = max(float(inc["quiet_since"]), t_end)
        if inc["phase"] == "cooling":
            inc["phase"], inc["cooling_at"] = "active", None
            self._tl(inc, t_end, "active", "Active again: new event")
        self._snapshot_tracks(inc, ev)
        self._recompute(inc)

    def _apply_alert_flags(self, d: dict, inc: dict) -> None:
        d["incident_id"] = inc["id"]
        summ = inc["alerts"].get(d["id"], {})
        if summ.get("inhibited_by"):
            d["inhibited_by"] = summ["inhibited_by"]
            d["suppressed"] = True
            d["suppressed_reason"] = summ.get("suppressed_reason", "")

    def _snapshot_tracks(self, inc: dict, ev: Optional[dict] = None, keys: Optional[list] = None) -> bool:
        """Copy handoff predictions / re-id global ids for the incident's tracks (they are transient upstream)."""
        if keys is None:
            keys = [f"{ev['camera_id']}:{t}" for t in (ev.get("track_ids") or [])[:10]] if ev else []
        changed = False
        if self.predictions is not None and keys:
            idx = self._pred_index()
            for k in keys:
                row = idx.get(k)
                if row and row.get("predictions") and inc["predictions"].get(k, {}).get("updated") != row.get("updated"):
                    inc["predictions"][k] = {kk: row.get(kk) for kk in ("camera", "track_id", "state", "exit_side",
                                                                        "predictions", "t_exit", "updated", "source",
                                                                        "model_version", "kind")}
                    changed = True
        if self.reid is not None:
            for k in keys:
                cam, _, tid = k.partition(":")
                try:
                    g = self.reid(cam, int(tid))
                except Exception:  # noqa: BLE001 - the optional re-id module must never break incidents
                    g = None
                if g and g.get("global_id") is not None and inc["global_ids"].get(k) != g:
                    inc["global_ids"][k] = {"global_id": str(g["global_id"]), "confidence": g.get("confidence")}
                    changed = True
        return changed

    def _pred_index(self) -> dict:
        t = time.monotonic()
        if t - self._pred_cache[0] > 0.5:
            try:
                rows = self.predictions() or []
            except Exception:  # noqa: BLE001
                rows = []
            self._pred_cache = (t, {f"{r.get('camera')}:{r.get('track_id')}": r for r in rows})
        return self._pred_cache[1]

    def _recompute(self, inc: dict) -> None:
        evs = inc["events"]
        alerts = inc["alerts"]
        inc["cameras"] = sorted({c for a in alerts.values() for c in a.get("cameras", [])} |
                                {e["camera_id"] for e in evs})
        inc["zones"] = sorted({e["zone_name"] for e in evs if e.get("zone_name")})
        inc["zone_keys"] = sorted({zone_key(e) for e in evs if e.get("zone_id")})
        inc["rules"] = sorted({r for a in alerts.values() for r in a.get("rules", [])} | {e["rule"] for e in evs})
        inc["tracks"] = sorted({f"{e['camera_id']}:{t}" for e in evs for t in e.get("track_ids", [])[:20]})[:200]
        if not inc.get("keyless"):
            ks = set()
            for e in evs:
                ks |= self._keys(e, inc["kind"])
            inc["keys"] = sorted(ks)
        inc["priority"] = best(a.get("priority") for a in alerts.values()) or inc.get("priority") or "P4"
        inc["score"] = max([float(a.get("score") or 0) for a in alerts.values()] or [0.0])
        if any(e.get("demo") for e in evs):
            inc["demo"] = True
        names = self._names()
        cams = [names.get(c) or (f"access control {c[len(SENSOR_PREFIX):]}" if c.startswith(SENSOR_PREFIX) else c)
                for c in inc["cameras"]]
        labels = [LABEL.get(r, r) for r in inc["rules"]]
        if inc["kind"] == "integrity":
            head = " + ".join(dict.fromkeys(l for r, l in zip(inc["rules"], labels) if is_integrity(r)))
            inc["title"] = f"{head or LABEL.get('camera_health', 'Camera view problem')} — {', '.join(cams) or '?'}"
        else:
            beh = [l for r, l in zip(inc["rules"], labels) if not is_integrity(r)]
            head = " + ".join(beh[:3]) + (f" +{len(beh) - 3}" if len(beh) > 3 else "")
            z = inc["zones"]
            inc["title"] = f"{head or 'Alert'} — {z[0] if z else ', '.join(cams)}" + \
                           (f" (+{len(z) - 1} zone{'s' if len(z) > 2 else ''})" if len(z) > 1 else "")
        if inc.get("demo"):
            inc["title"] = "[DEMO] " + inc["title"]
        t0, t1 = inc["opened_at"], inc["last_event_at"]
        inc["auto_summary"] = ("DEMO scenario (scripted, not observed). " if inc.get("demo") else "") + (
            f"{len(inc['alert_ids'])} alert(s) with {inc['n_events']} event(s) on {', '.join(cams) or '—'}"
            f"{' in ' + ', '.join(inc['zones'][:4]) if inc['zones'] else ''}, "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))} to {time.strftime('%H:%M:%S', time.localtime(t1))}. "
            f"Behaviours: {', '.join(labels) or '—'}. Highest priority {inc['priority']}."
            + (" Some events come from a camera with an open view problem (muted)." if inc.get("inhibited_by") else "")
            + (f" Reopened {inc['reopen_count']}×." if inc.get("reopen_count") else ""))

    # ---- phases --------------------------------------------------------------------------------------------------
    def _camera_degraded(self, cams: list) -> bool:
        if self.health is None:
            return False
        for c in cams:
            try:
                if self.health(c) in ("DEGRADED", "DOWN"):
                    return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def _advance(self, inc: dict, now: float) -> bool:
        if inc["phase"] == "closed":
            return False
        changed = False
        if inc["kind"] == "integrity" and self._camera_degraded(inc["cameras"]):
            if inc["phase"] == "cooling":
                inc["phase"], inc["cooling_at"] = "active", None
                self._tl(inc, now, "active", "Active again: camera condition persists")
                changed = True
            inc["quiet_since"] = max(float(inc["quiet_since"]), now)   # the condition persists: not quiet
        if inc["phase"] == "open":
            urgent = rank(inc["priority"]) <= rank("P2")
            t_hold = float(inc["opened_at"]) + float(self.cfg["hold_s"])
            if urgent or now >= t_hold:
                t_act = now if urgent else t_hold
                inc["phase"], inc["activated_at"] = "active", t_act
                self._tl(inc, t_act, "active", "Active: " + ("priority P2 or higher, no hold-down" if urgent
                                                             else f"hold-down of {self.cfg['hold_s']:g} s elapsed"))
                self._decide(inc, t_act, "new", inc["alert_ids"][-1] if inc["alert_ids"] else "")
                changed = True
        q = float(inc["quiet_since"])
        cool, close = float(self.cfg["cool_after_s"]), float(self.cfg["close_after_s"])
        if inc["phase"] == "active" and now - q >= cool:
            inc["phase"], inc["cooling_at"] = "cooling", q + cool
            self._tl(inc, q + cool, "cooling", f"Cooling: no new event for {cool:g} s")
            changed = True
        if inc["phase"] == "cooling" and now - q >= cool + close:
            self._close(inc, q + cool + close, "quiescence", f"Closed: no new event for {cool + close:g} s")
            changed = True
        if inc["phase"] != "closed" and now - float(inc["opened_at"]) >= float(self.cfg["max_duration_s"]):
            self._close(inc, now, "rollover", f"Closed (rollover after {self.cfg['max_duration_s']:g} s); "
                                              "further alerts open a new incident")
            changed = True
        return changed

    def _close(self, inc: dict, t: float, by: str, text: str, actor: Optional[str] = None) -> None:
        inc["phase"], inc["closed_at"], inc["closed_by"] = "closed", t, by
        if inc["status"] not in ("resolved", "closed"):
            inc["status"] = "closed"
        inc["updated_at"] = max(float(inc["updated_at"]), t)
        self._tl(inc, t, "closed", text, actor)

    def _reopen(self, inc: dict, now: float, why: str, actor: Optional[str] = None) -> None:
        inc["phase"], inc["closed_at"], inc["closed_by"], inc["cooling_at"] = "active", None, None, None
        inc["reopen_count"] = int(inc.get("reopen_count", 0)) + 1
        inc["status"] = "open"
        inc["quiet_since"] = max(float(inc["quiet_since"]), now)
        self._tl(inc, now, "reopened", f"Reopened: {why}", actor)

    def tick(self, now: Optional[float] = None) -> None:
        with self.lock:
            self._tick_locked(self.now() if now is None else self.now(now))

    def _tick_locked(self, now: float) -> None:
        refresh = self.predictions is not None or self.reid is not None
        refresh = refresh and time.monotonic() - self._last_pred_refresh > 2.0
        if refresh:
            self._last_pred_refresh = time.monotonic()
        window = float(self.cfg["reopen_window_s"])
        for iid, inc in list(self.live.items()):
            changed = self._advance(inc, now)
            changed = self._check_sla(inc, now) or changed
            if refresh and inc["phase"] != "closed" and inc["tracks"]:
                changed = self._snapshot_tracks(inc, keys=inc["tracks"][-40:]) or changed
            if changed:
                inc["updated_at"] = max(float(inc["updated_at"]), now)
                self._save(inc)
            if inc["phase"] == "closed" and (inc.get("merged_into") or inc.get("closed_by") == "rollover"
                                             or now - float(inc.get("closed_at") or 0) > window):
                self.live.pop(iid, None)

    # ---- push decisions ------------------------------------------------------------------------------------------
    def _suppressor(self, inc: dict) -> Optional[str]:
        """A live integrity incident mutes this one if every camera of this incident has one open."""
        if inc["kind"] == "integrity" or not inc["cameras"]:
            return None
        src = None
        for cam in inc["cameras"]:
            hits = [i for i in self.live.values() if i["kind"] == "integrity" and i["phase"] != "closed"
                    and not i.get("merged_into") and cam in i["cameras"] and (inc.get("demo") or not i.get("demo"))
                    and not set(i["rules"]) <= NON_MUTING_INTEGRITY]
            if not hits:
                return None
            src = src or hits[0]["id"]
        return src

    def _decide(self, inc: dict, t: float, trigger: str, alert_id: str) -> dict:
        prio = inc["priority"]
        last = inc.get("last_decided_priority")
        inc["last_decided_priority"] = prio if last is None or rank(prio) < rank(last) else last
        sup = self._suppressor(inc)
        inc["suppressed"] = bool(sup)
        if sup:
            decision, reason = "suppressed", f"inhibited by camera-integrity incident {sup}; visible, not pushed"
        elif prio not in self.cfg["push_tiers"]:
            decision, reason = "not_push_tier", f"{prio} is shown in the console only (push tiers: " \
                                                f"{', '.join(self.cfg['push_tiers'])})"
        elif prio in self.cfg["budget_exempt"]:
            decision, reason = "would_push", f"{prio} is exempt from the push budget"
        elif self.bucket.try_take(t):
            decision, reason = "would_push", f"within push budget ({self.bucket.tokens:.2f} token(s) left)"
        else:
            decision, reason = "budget_denied", (
                f"push budget exhausted ({self.cfg['push_budget_per_operator_hour']:g} per operator-hour, burst "
                f"{self.cfg['push_burst']:g}); still visible in the console")
        rec = {"t": round(float(t), 3), "trigger": trigger, "priority": prio, "decision": decision, "reason": reason,
               "alert_id": alert_id}
        inc["pushes"] = (inc.get("pushes") or [])[-49:] + [rec]
        self._tl(inc, t, "push", f"Push decision ({trigger}, {prio}): {decision.replace('_', ' ')} — {reason}",
                 decision=decision)
        webhook = notify = None
        if decision == "would_push":
            inc["highest_pushed"] = best([prio, inc.get("highest_pushed")])
            if self.webhook is not None:
                webhook = self._send_webhook(inc, rec)
            if self.notifier is not None:
                try:
                    notify = self.notifier(inc, rec)
                except Exception as e:  # noqa: BLE001 - a push channel must never break incident bookkeeping
                    notify = f"error: {type(e).__name__}"
        details = {k: v for k, v in (("webhook", webhook), ("notify", notify)) if v}
        self.store.add_notification(t, inc["id"], alert_id, prio, trigger, decision, reason, details)
        return rec

    # ---- acknowledge SLA ---------------------------------------------------------------------------------------------
    def sla_deadline(self, inc: dict) -> Optional[tuple[float, float]]:
        """(deadline, sla seconds) for an incident's current priority, or None when that tier has no SLA."""
        prio = inc.get("priority")
        lim = self.sla.get(prio) if prio in (self.cfg.get("sla_tiers") or []) else None
        if not lim:
            return None
        t0 = float((inc.get("prio_since") or {}).get(prio) or inc["opened_at"])
        return t0 + float(lim), float(lim)

    def _acknowledgement(self, inc: dict) -> Optional[dict]:
        """How the incident was acknowledged (its own status, or any alert moved out of "new"), or None."""
        if inc.get("status") not in (None, "open"):
            return {"via": f"incident status {inc['status']}"}
        for aid in inc.get("alert_ids", [])[-50:]:
            a = self.store.get_alert(aid)
            if a is not None and (a.get("status") or "new") != "new":
                return {"via": f"alert {aid} {a['status']}", "by": a.get("acknowledged_by"),
                        "t": a.get("acknowledged_at")}
        return None

    def _check_sla(self, inc: dict, now: float) -> bool:
        if inc["phase"] == "closed" or inc.get("merged_into") or (inc.get("sla") or {}).get("state"):
            return False
        dl = self.sla_deadline(inc)
        if dl is None or now < dl[0]:
            return False
        deadline, lim = dl
        prio = inc["priority"]
        ack = self._acknowledgement(inc)
        if ack is not None:
            inc["sla"] = {"state": "acknowledged", "priority": prio, "sla_s": lim, "deadline": deadline, **ack}
            return True
        waited = now - (deadline - lim)
        sup = self._suppressor(inc)
        if sup:
            decision, reason = "suppressed", f"inhibited by camera-integrity incident {sup}; visible, not pushed"
        else:
            decision, reason = "would_push", (f"not acknowledged within the {prio} SLA of {lim:g} s (waiting "
                                              f"{waited:.0f} s); SLA escalations are exempt from the push budget")
        aid = inc["alert_ids"][-1] if inc.get("alert_ids") else ""
        rec = {"t": round(float(now), 3), "trigger": "sla_overdue", "priority": prio, "decision": decision,
               "reason": reason, "alert_id": aid}
        inc["pushes"] = (inc.get("pushes") or [])[-49:] + [rec]
        inc["sla"] = {"state": "overdue", "priority": prio, "sla_s": lim, "deadline": deadline, "t": rec["t"],
                      "decision": decision}
        self._tl(inc, now, "sla", f"Acknowledge SLA missed ({prio}, {lim:g} s, waiting {waited:.0f} s): escalation "
                                  f"{decision.replace('_', ' ')}", decision=decision)
        extra = {"sla_s": lim, "waiting_s": round(waited, 1)}
        webhook = None
        if decision == "would_push" and self.webhook is not None:
            webhook = self._send_webhook(inc, rec, extra)
        self.store.add_notification(now, inc["id"], aid, prio, "sla_overdue", decision, reason,
                                    {**extra, "webhook": webhook} if webhook else extra)
        self._audit("system", "incident_sla_overdue", inc["id"], {"priority": prio, **extra, "decision": decision,
                                                                  "webhook": webhook})
        return True

    def _send_webhook(self, inc: dict, rec: dict, extra: Optional[dict] = None) -> str:
        msg_id = f"notify:{inc['id']}:{rec['trigger']}:{len(inc['pushes'])}"
        base = (self.cfg["webhook"].get("console_url") or "").rstrip("/")
        payload = {"type": "argonix.incident.push",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec["t"])),
                   "data": {"incident_id": inc["id"], "alert_id": rec["alert_id"], "trigger": rec["trigger"],
                            "priority": rec["priority"], "title": inc["title"], "camera_ids": inc["cameras"],
                            "zones": inc["zones"], "requires_human_decision": True,
                            "url": f"{base}/#/incidents/{inc['id']}" if base else None, **(extra or {})}}
        return "queued" if self.webhook.enqueue(msg_id, payload) else "dropped (queue full or endpoint disabled)"

    def _webhook_result(self, payload: dict, status: str, detail: str) -> None:
        d = payload.get("data", {})
        self.store.add_notification(time.time(), d.get("incident_id", ""), d.get("alert_id", ""),
                                    d.get("priority", ""), "webhook", f"webhook_{status}", detail)

    # ---- operator actions ----------------------------------------------------------------------------------------
    def _load(self, iid: str) -> dict:
        inc = self._get(iid)
        if inc is None:
            raise KeyError(iid)
        return self._hold(inc)

    def _audit(self, actor: str, action: str, iid: str, details: dict) -> None:
        self.store.audit(actor, action, iid, details)

    def update(self, iid: str, patch: dict, actor: str) -> dict:
        """Operator PATCH: status, assignee, summary (validated by the API for permissions)."""
        with self.lock:
            inc = self._load(iid)
            now = self.now()
            if "status" in patch and patch["status"] != inc["status"]:
                st = patch["status"]
                if st not in STATUSES:
                    raise ValueError(f"status must be one of {list(STATUSES)}")
                if inc.get("merged_into"):
                    raise ValueError(f"incident was merged into {inc['merged_into']}")
                prev = inc["status"]
                if st in ("resolved", "closed") and inc["phase"] != "closed":
                    inc["status"] = st
                    self._close(inc, now, "operator", f"Closed: {st} by {actor}", actor)
                elif st == "open" and inc["phase"] == "closed":
                    self._reopen(inc, now, f"by {actor}", actor)
                    inc["activated_at"] = inc.get("activated_at") or now
                else:
                    inc["status"] = st
                    self._tl(inc, now, "status", f"Status {prev} → {st}", actor)
                self._audit(actor, f"incident_status:{st}", iid, {"from": prev, "note": str(patch.get("note", ""))[:500]})
            if "assignee" in patch:
                a = patch["assignee"]
                a = (str(a).strip()[:64] or None) if a is not None else None
                if a != inc.get("assignee"):
                    self._tl(inc, now, "assigned", f"Assigned to {a}" if a else "Unassigned", actor)
                    self._audit(actor, "incident_assign", iid, {"from": inc.get("assignee"), "to": a})
                    inc["assignee"] = a
            if "summary" in patch:
                s = str(patch["summary"] or "")[:2000]
                if s != inc.get("summary"):
                    inc["summary"] = s
                    self._tl(inc, now, "summary", "Operator summary updated", actor)
                    self._audit(actor, "incident_summary", iid, {"chars": len(s)})
            inc["updated_at"] = max(float(inc["updated_at"]), now)
            self._save(inc)
            return inc

    def set_assistive_summary(self, iid: str, rec: dict, actor: str) -> dict:
        """Store a generated narrative (argonix.narrative) as the incident's assistive summary. It is labelled and
        kept apart from the operator summary; it never changes priority, phase or status."""
        with self.lock:
            inc = self._load(iid)
            now = self.now()
            inc["assistive_summary"] = dict(rec)
            what = f"model {rec.get('model')}" if rec.get("engine") in ("vlm", "llm") else "deterministic template"
            self._tl(inc, now, "assistive_summary", f"Assistive summary generated ({what})", actor)
            inc["updated_at"] = max(float(inc["updated_at"]), now)
            self._save(inc)
            return inc

    def _move_alert_bodies(self, alert_ids: list, iid: str) -> None:
        for aid in alert_ids:
            a = self.store.get_alert(aid)
            if a is not None and a.get("incident_id") != iid:
                a["incident_id"] = iid
                self.store.upsert_alert(a)

    def merge(self, target_id: str, source_ids: list, actor: str) -> dict:
        """Merge source incidents into the target. Sources close (closed_by=merge, merged_into=target)."""
        with self.alert_lock, self.lock:
            tgt = self._load(target_id)
            if tgt.get("merged_into"):
                raise ValueError(f"target {target_id} was itself merged into {tgt['merged_into']}")
            srcs = []
            for sid in dict.fromkeys(str(s) for s in source_ids):
                if sid == target_id:
                    raise ValueError("cannot merge an incident into itself")
                s = self._load(sid)
                if s.get("merged_into"):
                    raise ValueError(f"{sid} was already merged into {s['merged_into']}")
                if s["kind"] != tgt["kind"]:
                    raise ValueError("cannot merge a camera-integrity incident with a behaviour incident")
                srcs.append(s)
            if not srcs:
                raise ValueError("no source incidents given")
            now = self.now()
            was_closed = tgt["phase"] == "closed"
            any_live = any(s["phase"] != "closed" for s in srcs)
            for s in srcs:
                moved = list(s["alert_ids"])
                for aid in moved:
                    tgt["alerts"][aid] = s["alerts"].get(aid, {})
                    if aid not in tgt["alert_ids"]:
                        tgt["alert_ids"].append(aid)
                    self.alert_map[aid] = tgt["id"]
                    self.store.link_alert(aid, tgt["id"])
                tgt["events"] = sorted(tgt["events"] + s["events"], key=lambda e: float(e.get("t_end") or 0))[-MAX_EVENTS:]
                tgt["n_events"] = int(tgt.get("n_events", 0)) + int(s.get("n_events", 0))
                tgt["predictions"].update(s.get("predictions") or {})
                tgt["global_ids"].update(s.get("global_ids") or {})
                for k in ("last_event_at", "quiet_since"):
                    tgt[k] = max(float(tgt[k]), float(s[k]))
                tgt["opened_at"] = min(float(tgt["opened_at"]), float(s["opened_at"]))
                tgt["merged_from"] = list(tgt.get("merged_from") or []) + [s["id"]]
                tgt["inhibited_by"] = tgt.get("inhibited_by") or s.get("inhibited_by")
                if s.get("assignee") and not tgt.get("assignee"):
                    tgt["assignee"] = s["assignee"]
                self._tl(tgt, now, "merged", f"Merged {s['id']} into this incident ({len(moved)} alert(s))", actor,
                         source=s["id"])
                s["merged_alert_ids"] = list(s.get("merged_alert_ids") or []) + moved
                s["alert_ids"], s["alerts"], s["events"] = [], {}, []
                s["merged_into"] = tgt["id"]
                if s["phase"] != "closed":
                    self._close(s, now, "merge", f"Merged into {tgt['id']} by {actor}", actor)
                else:
                    self._tl(s, now, "merged", f"Merged into {tgt['id']} by {actor}", actor)
                s["status"] = "closed"
                s["updated_at"] = now
                self._save(s)
                self._move_alert_bodies(moved, tgt["id"])
            if was_closed and any_live:
                self._reopen(tgt, now, "a live incident was merged into it", actor)
            self._recompute(tgt)
            self._advance(tgt, now)
            tgt["updated_at"] = now
            self._save(tgt)
            self._audit(actor, "incident_merge", tgt["id"], {"sources": [s["id"] for s in srcs]})
            return tgt

    def split(self, iid: str, alert_ids: list, actor: str) -> dict:
        """Move the given alerts out of an incident into a new incident. Returns the new incident."""
        with self.alert_lock, self.lock:
            src = self._load(iid)
            if src.get("merged_into"):
                raise ValueError(f"incident was merged into {src['merged_into']}")
            ids = list(dict.fromkeys(str(a) for a in alert_ids))
            unknown = [a for a in ids if a not in src["alert_ids"]]
            if not ids or unknown:
                raise ValueError(f"alerts not in {iid}: {unknown}" if unknown else "no alerts given")
            if len(ids) >= len(src["alert_ids"]):
                raise ValueError("leave at least one alert in the original incident")
            now = self.now()
            moved_events = [e for e in src["events"] if e["alert_id"] in ids]
            t0 = min([float(src["alerts"][a].get("created_at") or now) for a in ids] + [now])
            new = self._hold(self._blank(src["kind"], t0, "P4"))
            new.update(keyless=True, split_from=src["id"], status="open", assignee=src.get("assignee"),
                       inhibited_by=src.get("inhibited_by"))
            for aid in ids:
                new["alerts"][aid] = src["alerts"].pop(aid)
                new["alert_ids"].append(aid)
                self.alert_map[aid] = new["id"]
                self.store.link_alert(aid, new["id"])
            src["alert_ids"] = [a for a in src["alert_ids"] if a not in ids]
            src["events"] = [e for e in src["events"] if e["alert_id"] not in ids]
            src["n_events"] = max(len(src["events"]), int(src.get("n_events", 0)) - len(moved_events))
            new["events"], new["n_events"] = moved_events, len(moved_events)
            tr = {f"{e['camera_id']}:{t}" for e in moved_events for t in e.get("track_ids", [])}
            new["predictions"] = {k: v for k, v in (src.get("predictions") or {}).items() if k in tr}
            new["global_ids"] = {k: v for k, v in (src.get("global_ids") or {}).items() if k in tr}
            last = max([float(e.get("t_end") or t0) for e in moved_events] + [t0])
            new["last_event_at"] = new["quiet_since"] = last
            if src["phase"] == "closed":
                new.update(phase="closed", closed_at=src["closed_at"], closed_by="operator", status="closed")
            else:
                new.update(phase="active", activated_at=now)
            self._tl(new, now, "split", f"Split from {src['id']} by {actor} ({len(ids)} alert(s))", actor,
                     source=src["id"])
            self._tl(src, now, "split", f"{len(ids)} alert(s) split out into {new['id']} by {actor}", actor,
                     target=new["id"])
            self._recompute(new)
            self._recompute(src)
            new["last_decided_priority"] = new["priority"]   # an operator split never pages by itself
            src["last_event_at"] = max([float(e.get("t_end") or 0) for e in src["events"]] + [float(src["opened_at"])])
            self._advance(new, now)
            new["updated_at"] = src["updated_at"] = now
            self._save(src)
            self._save(new)
            self._move_alert_bodies(ids, new["id"])
            self._audit(actor, "incident_split", src["id"], {"new_incident": new["id"], "alert_ids": ids})
            return new

    # ---- reads ---------------------------------------------------------------------------------------------------
    def get(self, iid: str) -> Optional[dict]:
        with self.lock:
            inc = self._get(iid)
            return json.loads(json.dumps(inc)) if inc else None

    def incident_for_alert(self, alert_id: str) -> Optional[str]:
        return self.alert_map.get(alert_id) or self.store.incident_for_alert(alert_id)

    def graph(self, iid: str, max_events: int = 60) -> Optional[dict]:
        with self.lock:
            inc = self._get(iid)
            if inc is None:
                return None
            if inc["phase"] != "closed" and (self.predictions is not None or self.reid is not None):
                if self._snapshot_tracks(inc, keys=inc["tracks"][-40:]):
                    self._save(inc)
            inc = json.loads(json.dumps(inc))
            inhibitors = {}
            for e in inc["events"]:
                sid = e.get("inhibited_by")
                if sid and sid not in inhibitors:
                    s = self._get(sid)
                    inhibitors[sid] = {"title": s["title"], "phase": s["phase"], "priority": s["priority"]} if s else {}
        return build_graph(inc, self._names(), inhibitors, max_events)


# ---------------------------------------------------------------------------------------------------------------
# Graph + compact views + hotspots (pure functions, used by the API)
# ---------------------------------------------------------------------------------------------------------------
def build_graph(inc: dict, cam_names: Optional[dict] = None, inhibitors: Optional[dict] = None,
                max_events: int = 60) -> dict:
    """Typed evidence graph of one incident. Nodes: event, track, global_track (probabilistic re-id, only if a re-id
    module supplied ids), camera, zone, predicted camera / leaves-coverage (handoff estimate), evidence, incident
    (inhibiting camera-integrity incidents), sensor (access-control device; reported_by from its events, covered_by to
    the camera that views its door). Edge types: EDGE_TYPES."""
    cam_names = cam_names or {}
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    eset: set = set()

    def node(nid: str, kind: str, label: str, **attrs) -> str:
        if nid not in nodes:
            nodes[nid] = {"id": nid, "kind": kind, "label": label, **attrs}
        return nid

    def edge(a: str, b: str, typ: str, **attrs) -> None:
        k = (a, b, typ)
        if a != b and k not in eset:
            eset.add(k)
            edges.append({"source": a, "target": b, "type": typ, **attrs})

    evs = sorted(inc["events"], key=lambda e: float(e.get("t_end") or 0))
    truncated = len(evs) > max_events
    evs = evs[-max_events:]
    by_alert: dict[str, list] = defaultdict(list)
    for i, e in enumerate(evs):
        eid = f"ev:{i}"
        cam = e["camera_id"]
        node(eid, "event", LABEL.get(e["rule"], e["rule"]), rule=e["rule"], camera_id=cam, alert_id=e["alert_id"],
             zone=e.get("zone_name"), t_start=e.get("t_start"), t_end=e.get("t_end"), confidence=e.get("confidence"),
             observation=e.get("observation"), learned_model=e.get("learned_model", False),
             layer="inference" if e.get("learned_model") else "observation", modality=e.get("modality") or "video")
        is_sensor = bool(e.get("modality")) or cam.startswith(SENSOR_PREFIX)
        if is_sensor:  # access-control device: a sensor node, not a camera
            sens = e.get("sensor") or {}
            if cam.startswith(SENSOR_PREFIX):
                src, covering = sens.get("source_id") or cam[len(SENSOR_PREFIX):], sens.get("camera")
            else:  # a door report filed under a camera (scripted demo door event without the sensor module)
                src, covering = sens.get("source_id") or f"door near {cam_names.get(cam, cam)}", sens.get("camera") or cam
            c = node(f"sensor:{src}", "sensor", src, source_id=src,
                     note="non-video source (door controller / alarm panel); panel-reported state, not verified by video")
            edge(eid, c, "reported_by")
            if covering:
                cc = node(f"cam:{covering}", "camera", cam_names.get(covering, covering), camera_id=covering)
                edge(c, cc, "covered_by")
        else:
            c = node(f"cam:{cam}", "camera", cam_names.get(cam, cam), camera_id=cam)
            edge(eid, c, "observed_in")
        if e.get("zone_id"):
            z = node(f"zone:{zone_key(e)}", "zone", e.get("zone_name") or e["zone_id"], camera_id=cam,
                     site_zone=e.get("site_zone"))
            edge(eid, z, "observed_in")
            if not is_sensor:  # a door's zone links to cameras through the video events that share it
                edge(z, c, "observed_in")
        for t in (e.get("track_ids") or [])[:12]:
            tk = node(f"track:{cam}:{t}", "track", f"{cam_names.get(cam, cam)} #{t}", camera_id=cam, track_id=t,
                      note="per-camera tracker id (no identity)")
            edge(eid, tk, "same_track")
            edge(tk, c, "observed_in")
        ev_items = e.get("evidence") or {}
        if ev_items:
            ek = node(f"evid:{i}", "evidence", "evidence", **{k: v for k, v in ev_items.items()})
            edge(eid, ek, "evidence")
        sid = e.get("inhibited_by")
        if sid:
            info = (inhibitors or {}).get(sid, {})
            s = node(f"inc:{sid}", "incident", info.get("title") or sid, incident_id=sid, phase=info.get("phase"))
            edge(eid, s, "inhibited_by")
        by_alert[e["alert_id"]].append((eid, e))
    for items in by_alert.values():  # the correlator's corroboration: distinct rule/camera inside one alert
        reps = {}
        for eid, e in items:
            reps.setdefault((e["rule"], e["camera_id"]), eid)
        rl = list(reps.values())
        for a in range(len(rl)):
            for b in range(a + 1, len(rl)):
                edge(rl[a], rl[b], "corroborates")
    for key, g in (inc.get("global_ids") or {}).items():
        tk = f"track:{key}"
        if tk in nodes and g.get("global_id") is not None:
            gn = node(f"gid:{g['global_id']}", "global_track", f"global #{g['global_id']}",
                      note="probabilistic cross-camera association (revisable, research-grade)")
            edge(tk, gn, "same_track", probabilistic=True, confidence=g.get("confidence"))
    for key, row in (inc.get("predictions") or {}).items():
        tk = f"track:{key}"
        if tk not in nodes:
            continue
        for p in (row.get("predictions") or [])[:3]:
            if p.get("leaves_coverage"):
                dest = node("cov:outside", "coverage", "leaves coverage", estimate=True)
            else:
                cam = p.get("camera")
                if not cam:
                    continue
                dest = node(f"cam:{cam}", "camera", cam_names.get(cam, cam), camera_id=cam, predicted_only=True)
            edge(tk, dest, "predicted_next", prob=p.get("prob"), eta_s=p.get("eta_s"), estimate=True,
                 method=row.get("source"), state=row.get("state"))
    for n in nodes.values():  # a camera reached only through a prediction keeps the predicted_only flag
        if n["kind"] == "camera" and n.get("predicted_only") and any(
                e["target"] == n["id"] and e["type"] == "observed_in" for e in edges):
            n.pop("predicted_only", None)
    return {"incident_id": inc["id"], "nodes": list(nodes.values()), "edges": edges, "edge_types": list(EDGE_TYPES),
            "truncated": truncated, "n_events": inc.get("n_events", len(inc["events"])),
            "note": ("Edges are typed evidence links. same_track links events to one per-camera tracker id; "
                     "predicted_next and global_track edges are estimates (probabilistic, revisable) and never "
                     "identity claims.")}


COMPACT_DROP = ("events", "timeline", "alerts", "predictions", "global_ids", "pushes", "keys", "tracks")


def compact(inc: dict) -> dict:
    d = {k: v for k, v in inc.items() if k not in COMPACT_DROP}
    d["n_alerts"] = len(inc.get("alert_ids", []))
    d["n_predictions"] = len(inc.get("predictions") or {})
    return d


def hotspots(store, now: float, days: float = 28.0, by: str = "zone", metric: str = "incidents",
             cam_names: Optional[dict] = None, top_n: int = 10) -> dict:
    """Incident (or alert) frequency by local hour-of-week (Mon 00:00 = 0 … Sun 23:00 = 167) per zone or camera."""
    cam_names = cam_names or {}
    since = now - float(days) * 86400.0
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * 168)
    labels: dict[str, dict] = {}
    overall = [0] * 168
    n_items = 0

    def entities(evs: list, cams: list) -> list[tuple[str, str, str]]:
        if by == "camera":
            return [(c, cam_names.get(c, c), c) for c in sorted(set(cams) | {e["camera_id"] for e in evs})]
        out = {}
        for e in evs:
            k = zone_key(e)
            cam = e["camera_id"]
            out[k] = (k, e.get("zone_name") or e.get("zone_id") or k, cam)
        return list(out.values())

    if metric == "alerts":
        items = [(float(a["created_at"]), a.get("events") or [], a.get("cameras") or [], a["id"])
                 for a in store.list_alerts(since=since, limit=100000)
                 if float(a["created_at"]) >= since and not alert_is_demo(a)]
    else:
        items = [(float(i["opened_at"]), i.get("events") or [], i.get("cameras") or [], i["id"])
                 for i in store.list_incidents(since=since, limit=100000)
                 if float(i["opened_at"]) >= since and not i.get("merged_into") and not i.get("demo")]
    examples: dict[tuple, list] = defaultdict(list)
    for t, evs, cams, iid in items:
        lt = time.localtime(t)
        how = lt.tm_wday * 24 + lt.tm_hour
        overall[how] += 1
        n_items += 1
        for key, label, cam in entities(evs, cams):
            counts[key][how] += 1
            labels[key] = {"label": label, "camera_id": cam, "camera": cam_names.get(cam, cam)}
            if len(examples[(key, how)]) < 5:
                examples[(key, how)].append(iid)
    rows = [{"key": k, **labels[k], "total": sum(c), "counts": c,
             "peak_how": max(range(168), key=lambda h: c[h])} for k, c in counts.items()]
    rows.sort(key=lambda r: -r["total"])
    cells = sorted(((c[h], k, h) for k, c in counts.items() for h in range(168) if c[h]), key=lambda x: (-x[0], x[2]))
    top = [{"key": k, **labels[k], "how": h, "count": n, "examples": examples[(k, h)]} for n, k, h in cells[:top_n]]
    return {"since": since, "until": now, "days": days, "by": by, "metric": metric, "n_items": n_items,
            "overall": overall, "rows": rows, "top": top, "source": f"{metric} store",
            "note": (f"Counts of {metric} opened per local hour-of-week slot over the last {days:g} days. An item that "
                     f"spans several {by}s counts once for each. Frequencies of past alerts, not a forecast; small "
                     "counts are noisy.")}


# ---------------------------------------------------------------------------------------------------------------
# Site wiring
# ---------------------------------------------------------------------------------------------------------------
def _reid_lookup(site, cam: str, tid: int) -> Optional[dict]:
    """Probe an optional re-id module (site.reid) for a global id. Tolerates several method names/return shapes."""
    r = getattr(site, "reid", None)
    if r is None:
        return None
    for obj in (r, getattr(r, "engine", None)):
        if obj is None:
            continue
        for name in ("global_id", "global_id_for", "lookup", "resolve"):
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            v = fn(cam, tid)
            if v is None:
                return None
            if isinstance(v, dict):
                gid, conf = v.get("global_id", v.get("id")), v.get("confidence", v.get("score"))
            elif isinstance(v, (tuple, list)) and len(v) == 2:
                gid, conf = v
            else:
                gid, conf = v, None
            if gid is None:
                return None
            return {"global_id": str(gid), "confidence": None if conf is None else round(float(conf), 3)}
    return None


def install(site) -> Optional[IncidentManager]:
    """Attach an IncidentManager to a Site (config key "incidents", on by default)."""
    cfg = dict(site.cfg.get("incidents") or {})
    if cfg.get("enabled", True) is False:
        site.incidents = None
        return None

    def health(cam: str) -> Optional[str]:
        if cam.startswith(SENSOR_PREFIX):  # access-control source: DOWN while silent past its heartbeat
            hub = getattr(site, "sensors", None)
            return hub.health(cam[len(SENSOR_PREFIX):]) if hub is not None else None
        w = site.worker(cam)
        return w.health.state if w is not None else None

    def predictions() -> list:
        an = getattr(site, "handoff", None)
        return an.engine.predictions_list() if an is not None else []

    if cfg.get("sla_s") is None and isinstance(site.cfg.get("sla_s"), dict):
        cfg["sla_s"] = site.cfg["sla_s"]   # top-level site config key, shared with the dashboard
    mgr = IncidentManager(site.store, cfg, alert_lock=site.lock, health=health, predictions=predictions,
                          reid=lambda cam, tid: _reid_lookup(site, cam, tid),
                          camera_names=lambda: {w.id: w.name for w in site.workers})
    site.incidents = mgr
    return mgr
