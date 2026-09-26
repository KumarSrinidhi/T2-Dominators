"""Access-control / door-sensor events: the first non-video data stream, fused with video (manual §28 S-06/S-12/S-13).

A door controller, access panel or alarm panel POSTs one JSON event per state change to `/api/sensors/events`:

    {"source_id": "door-ctrl-1", "type": "door_forced", "ts": 1790000000.0, "door": "D1",
     "card_id": "optional, hashed on arrival", "details": {"held_s": 45}}

Authentication (either):
  * Standard Webhooks v1 HMAC-SHA256 signature (`webhook-id`, `webhook-timestamp`, `webhook-signature` headers) with a
    per-source secret read from the environment variable named by the source's `secret_env`. Requests more than 5 min
    away from server time, and re-used webhook-ids (replays), are rejected before the body is interpreted.
  * an API token (`Authorization: Bearer ...`) whose roles grant `sensor.ingest` (role "integration" or "admin").

Each report maps to an Argus Event with rule ids registered like any video rule (door_forced, door_held_open,
access_denied_burst for >= 3 denials at one door in 60 s, alarm_input, access_tamper). The door mapping in the site
config gives the event the site zone / camera zone of the camera that views the door, so the existing correlator puts
a door report and a video rule in the same site zone into ONE alert, and explain_priority reports "sensor + video"
corroboration. A panel report is a deterministic source (not a learned model), stated as panel-reported and not
verified by video. access_granted and heartbeat messages are recorded (last seen, counts) but raise nothing.

Sensor health (S-13): a source configured with `heartbeat_s` that stays silent that long raises `sensor_offline`
(an integrity event, like camera_health). Timestamps further than `max_clock_skew_s` from server time are rejected
rather than silently re-timed, so a drifting panel clock is visible instead of mis-correlated.

Privacy: card_id is replaced on arrival by a keyed hash (HMAC-SHA256 with a per-site random key in
`<data>/sensor_card.key`); the raw value is never stored or logged. `details` must not carry credentials.
Observations are factual device reports; nothing here says who a person is or what they intend.
"""
from __future__ import annotations

import itertools
import hmac
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .baseline import RATE_EXCLUDE
from .incidents import sign_webhook
from .models import SENSOR_PREFIX, Event
from .rules import SEVERITY, register_rule

TYPES = ("door_forced", "door_held_open", "access_denied", "access_granted", "alarm_input", "tamper", "heartbeat")
DOOR_REQUIRED = frozenset({"door_forced", "door_held_open", "access_denied", "access_granted"})
DIRECT_RULE = {"door_forced": "door_forced", "door_held_open": "door_held_open", "alarm_input": "alarm_input",
               "tamper": "access_tamper"}
RULES = {  # rule id: (label, severity). Labels describe what the device reported, never a person.
    "door_forced": ("Door reported forced open", 3),
    "door_held_open": ("Door reported held open", 2),
    "access_denied_burst": ("Repeated access denials at a door", 2),
    "alarm_input": ("Alarm input reported", 3),
    "access_tamper": ("Access-control device tamper reported", 2),
    "sensor_offline": ("Access-control source silent", 2),
}
for _name, (_label, _sev) in RULES.items():
    register_rule(_name, _label, _sev)
RATE_EXCLUDE.add("sensor_offline")  # device health, not zone activity

MODALITY = "access_control"
CONFIDENCE = 0.9
CONFIDENCE_METHOD = "fixed 0.9 for a panel-reported state change (deterministic source; not verified by video)"
MAX_BODY = 16 * 1024
SIG_TOLERANCE_S = 300.0
REPLAY_KEEP_S = 2 * SIG_TOLERANCE_S
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MSG_ID_RE = re.compile(r"^[\x21-\x7e]{1,128}$")
BODY_KEYS = frozenset({"source_id", "type", "ts", "door", "card_id", "details"})
CREDENTIAL_KEYS = re.compile(r"card|badge|credential|pin|token|secret|password", re.I)
DEFAULTS = {"max_clock_skew_s": SIG_TOLERANCE_S, "access_denied_burst": {"count": 3, "window_s": 60.0}}

_EV_SEQ = itertools.count(1)  # unique evidence file names, even for events in the same millisecond


class IngestError(Exception):
    """Rejected sensor request; `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def verify_signature(secret: str, headers: Mapping[str, str], body: bytes, now: float,
                     tolerance_s: float = SIG_TOLERANCE_S) -> str:
    """Standard Webhooks v1: base64(HMAC-SHA256(key, "{id}.{timestamp}.{body}")) in `webhook-signature` (a
    space-separated list of "v1,<sig>"). Returns the webhook-id; raises IngestError(401) otherwise."""
    msg_id = headers.get("webhook-id") or ""
    ts = headers.get("webhook-timestamp") or ""
    sigs = headers.get("webhook-signature") or ""
    if not (msg_id and ts and sigs):
        raise IngestError(401, "signed requests need webhook-id, webhook-timestamp and webhook-signature headers")
    if not MSG_ID_RE.match(msg_id):
        raise IngestError(401, "webhook-id must be 1-128 printable ASCII characters")
    try:
        ts_i = int(ts)
    except ValueError:
        raise IngestError(401, "webhook-timestamp must be integer Unix seconds")
    if abs(now - ts_i) > tolerance_s:
        raise IngestError(401, f"webhook-timestamp is more than {tolerance_s:g} s from server time (clock skew or replay)")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise IngestError(401, "body is not UTF-8")
    expected = sign_webhook(secret, msg_id, ts_i, text)
    if not any(hmac.compare_digest(expected, s) for s in sigs.split()):
        raise IngestError(401, "signature mismatch")
    return msg_id


def parse_ts(v: Any) -> float:
    """Epoch seconds (number) or ISO-8601 with an explicit offset ("Z" allowed)."""
    if isinstance(v, bool):
        raise IngestError(422, "ts must be epoch seconds or ISO-8601 with a timezone")
    if isinstance(v, (int, float)):
        t = float(v)
    elif isinstance(v, str) and len(v) <= 40:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise IngestError(422, "ts is not valid ISO-8601")
        if d.tzinfo is None:
            raise IngestError(422, "ts needs a timezone offset (e.g. 2026-09-25T10:00:00Z)")
        t = d.timestamp()
    else:
        raise IngestError(422, "ts must be epoch seconds or ISO-8601 with a timezone")
    if not math.isfinite(t):
        raise IngestError(422, "ts must be finite")
    return t


def load_pepper(path: Path) -> bytes:
    """Per-site random key for card-id hashing (created on first use, owner-only where the OS supports it)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    return path.read_bytes()


@dataclass
class Source:
    id: str
    name: str
    kind: str
    doors: dict
    secret_env: Optional[str]
    secret: Optional[str]
    heartbeat_s: Optional[float]
    criticality: int
    burst_count: int
    burst_window_s: float
    last_seen: Optional[float] = None
    last_event: Optional[dict] = None
    counts: dict = field(default_factory=dict)
    alerts: int = 0
    rejected: int = 0
    last_rejected: Optional[dict] = None
    state: str = "waiting"          # waiting (nothing yet) | ok | offline
    offline_since: Optional[float] = None
    recovered_at: Optional[float] = None
    denials: dict = field(default_factory=dict)  # door -> deque[(t, card_hash)]

    def hmac_status(self) -> str:
        if not self.secret_env:
            return "not configured (API token only)"
        return "configured" if self.secret else f"missing: set environment variable {self.secret_env}"


def _source(d: dict, defaults: dict) -> Source:
    sid = str(d.get("id", ""))
    if not ID_RE.match(sid):
        raise ValueError(f"sensor source id {sid!r}: 1-64 letters, digits, '_', '.', ':' or '-'")
    doors = d.get("doors") or {}
    if not isinstance(doors, dict) or not all(ID_RE.match(str(k)) and isinstance(v, dict) for k, v in doors.items()):
        raise ValueError(f"sensor source {sid}: doors must map door ids to mapping objects")
    secret_env = d.get("secret_env") or None
    secret = os.environ.get(secret_env) if secret_env else None
    if secret and secret.startswith("whsec_"):
        sign_webhook(secret, "check", 0, "")  # raises on a malformed whsec_ secret: fail at start, not per request
    hb = d.get("heartbeat_s")
    if hb is not None and not (isinstance(hb, (int, float)) and hb > 0):
        raise ValueError(f"sensor source {sid}: heartbeat_s must be a positive number")
    burst = {**defaults["access_denied_burst"], **(d.get("access_denied_burst") or {})}
    return Source(sid, str(d.get("name") or sid), str(d.get("kind") or "door controller"), dict(doors), secret_env,
                  secret, float(hb) if hb else None, int(d.get("criticality", 2)), max(1, int(burst["count"])),
                  float(burst["window_s"]))


class SensorHub:
    """Validates, authenticates, records and maps sensor reports to Events; tracks per-source health."""

    def __init__(self, cfg: dict, emit: Callable[[Event], Optional[dict]], store=None,
                 zone_lookup: Optional[Callable[[str, str], Optional[dict]]] = None,
                 evidence: Optional[Callable[[str, str], Optional[dict]]] = None,
                 clock: Callable[[], float] = time.time, pepper: Optional[bytes] = None):
        self.cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "sources"}}
        self.cfg["access_denied_burst"] = {**DEFAULTS["access_denied_burst"], **(cfg.get("access_denied_burst") or {})}
        self.sources: dict[str, Source] = {}
        for d in cfg.get("sources") or []:
            s = _source(d, self.cfg)
            if s.id in self.sources:
                raise ValueError(f"duplicate sensor source id {s.id}")
            self.sources[s.id] = s
        self.emit, self.store, self.zone_lookup, self.evidence = emit, store, zone_lookup, evidence
        self.clock = clock
        self.pepper = pepper or secrets.token_bytes(32)
        self.skew = float(self.cfg["max_clock_skew_s"])
        self.lock = threading.RLock()
        self.started_at = clock()
        self._seen: dict[tuple[str, str], float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.warnings: list[str] = []
        if store is not None:
            self._init_db()

    # ---- persistence ---------------------------------------------------------------------------------------------
    def _init_db(self) -> None:
        with self.store.lock:
            self.store.db.executescript("""
            CREATE TABLE IF NOT EXISTS sensor_events(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, arrival REAL,
                source_id TEXT, door TEXT, type TEXT, card_hash TEXT, rule TEXT, alert_id TEXT, via TEXT, msg_id TEXT,
                details TEXT);
            CREATE INDEX IF NOT EXISTS ix_sensor_events_t ON sensor_events(arrival);
            CREATE TABLE IF NOT EXISTS sensor_msgs(source_id TEXT, msg_id TEXT, t REAL, PRIMARY KEY(source_id, msg_id));
            """)
            rows = self.store.db.execute(
                "SELECT source_id, type, COUNT(*), MAX(arrival) FROM sensor_events GROUP BY source_id, type").fetchall()
            n_alerts = dict(self.store.db.execute("SELECT source_id, COUNT(*) FROM sensor_events WHERE alert_id "
                                                  "IS NOT NULL GROUP BY source_id").fetchall())
        for sid, typ, n, last in rows:  # warm start: counts and last-seen survive a restart
            s = self.sources.get(sid)
            if s is not None:
                s.counts[typ] = int(n)
                s.last_seen = max(s.last_seen or 0.0, float(last or 0.0)) or None
                s.alerts = int(n_alerts.get(sid, 0))

    def _claim(self, sid: str, msg_id: str, now: float) -> bool:
        """True the first time (source, webhook-id) is seen within the replay window."""
        if self.store is not None:
            with self.store.lock:
                self.store.db.execute("DELETE FROM sensor_msgs WHERE t < ?", (now - REPLAY_KEEP_S,))
                cur = self.store.db.execute("INSERT OR IGNORE INTO sensor_msgs VALUES(?,?,?)", (sid, msg_id, now))
                self.store.db.commit()
                return cur.rowcount == 1
        with self.lock:
            self._seen = {k: t for k, t in self._seen.items() if t >= now - REPLAY_KEEP_S}
            if (sid, msg_id) in self._seen:
                return False
            self._seen[(sid, msg_id)] = now
            return True

    def _record(self, r: dict) -> None:
        if self.store is None:
            return
        with self.store.lock:
            self.store.db.execute("INSERT INTO sensor_events(t, arrival, source_id, door, type, card_hash, rule, "
                                  "alert_id, via, msg_id, details) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                  (r["t"], r["arrival"], r["source_id"], r.get("door"), r["type"], r.get("card_hash"),
                                   r.get("rule"), r.get("alert_id"), r["via"], r.get("msg_id"),
                                   json.dumps(r.get("details") or {})))
            self.store.db.commit()

    def recent(self, source_id: str = "", limit: int = 100) -> list[dict]:
        if self.store is None:
            return []
        q, a = "SELECT t, arrival, source_id, door, type, card_hash, rule, alert_id, via, details FROM sensor_events", []
        if source_id:
            q += " WHERE source_id = ?"
            a.append(source_id)
        q += " ORDER BY id DESC LIMIT ?"
        a.append(max(1, min(int(limit), 1000)))
        with self.store.lock:
            rows = self.store.db.execute(q, a).fetchall()
        keys = ("t", "arrival", "source_id", "door", "type", "card_hash", "rule", "alert_id", "via", "details")
        out = [dict(zip(keys, r)) for r in rows]
        for r in out:
            r["details"] = json.loads(r["details"] or "{}")
        return out

    # ---- ingestion -----------------------------------------------------------------------------------------------
    @staticmethod
    def _parse(body: bytes) -> dict:
        if len(body) > MAX_BODY:
            raise IngestError(413, f"body larger than {MAX_BODY} bytes")
        try:
            p = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise IngestError(400, "body is not valid JSON")
        if not isinstance(p, dict):
            raise IngestError(422, "body must be a JSON object")
        return p

    def _reject(self, s: Optional[Source], now: float, err: IngestError) -> IngestError:
        if s is not None:
            with self.lock:
                s.rejected += 1
                s.last_rejected = {"t": now, "reason": str(err)[:200], "status": err.status}
        return err

    def ingest_signed(self, headers: Mapping[str, str], body: bytes) -> dict:
        """HMAC path: authenticate against the named source's secret, refuse replays, then validate and map."""
        now = self.clock()
        p = self._parse(body)
        sid = p.get("source_id")
        s = self.sources.get(sid) if isinstance(sid, str) else None
        if s is None:
            raise IngestError(401, "unknown source_id")
        if not s.secret:
            raise self._reject(s, now, IngestError(401, f"signed ingestion is disabled for {s.id}: {s.hmac_status()}"))
        try:
            msg_id = verify_signature(s.secret, headers, body, now)
        except IngestError as e:
            raise self._reject(s, now, e)
        if not self._claim(s.id, msg_id, now):
            raise self._reject(s, now, IngestError(409, "replayed webhook-id (already processed)"))
        return self._ingest(s, p, now, via="hmac", msg_id=msg_id)

    def ingest_token(self, username: str, body: bytes) -> dict:
        """API-token path (the caller already checked the token's `sensor.ingest` permission)."""
        now = self.clock()
        p = self._parse(body)
        sid = p.get("source_id")
        s = self.sources.get(sid) if isinstance(sid, str) else None
        if s is None:
            raise IngestError(422, "unknown source_id (configure it under sensors.sources)")
        return self._ingest(s, p, now, via=f"token:{username}")

    def _validate(self, s: Source, p: dict, now: float) -> dict:
        extra = sorted(set(p) - BODY_KEYS)
        if extra:
            raise IngestError(422, f"unknown field(s) {extra}; allowed: {sorted(BODY_KEYS)}")
        typ = p.get("type")
        if typ not in TYPES:
            raise IngestError(422, f"type must be one of {list(TYPES)}")
        t = now if p.get("ts") is None else parse_ts(p["ts"])
        if abs(t - now) > self.skew:
            raise IngestError(422, f"ts is {t - now:+.0f} s from server time (limit ±{self.skew:g} s): check the "
                                   "panel clock; events are not re-timed")
        door = p.get("door")
        if door is not None:
            if not isinstance(door, str) or door not in s.doors:
                raise IngestError(422, f"unknown door for {s.id}; configured: {sorted(s.doors)}")
        elif typ in DOOR_REQUIRED:
            raise IngestError(422, f"door is required for {typ}")
        card = p.get("card_id")
        card_hash = None
        if card is not None:
            if isinstance(card, bool) or not isinstance(card, (str, int)) or not 0 < len(str(card)) <= 128:
                raise IngestError(422, "card_id must be a string or integer of 1-128 characters")
            card_hash = "card:" + hmac.new(self.pepper, str(card).encode(), hashlib.sha256).hexdigest()[:16]
        details = p.get("details")
        if details is None:
            details = {}
        if not isinstance(details, dict) or len(details) > 16:
            raise IngestError(422, "details must be an object with at most 16 keys")
        for k, v in details.items():
            if not ID_RE.match(k) or len(k) > 40:
                raise IngestError(422, f"details key {k[:40]!r}: 1-40 letters, digits, '_', '.', ':' or '-'")
            if CREDENTIAL_KEYS.search(k):
                raise IngestError(422, f"details key {k!r} looks like a credential; send it as card_id (hashed)")
            ok = v is None or isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(v)) or \
                (isinstance(v, str) and len(v) <= 200)
            if not ok:
                raise IngestError(422, f"details[{k!r}] must be a number, boolean, null or a string <= 200 characters")
        return {"type": typ, "t": t, "clock": "panel" if p.get("ts") is not None else "arrival", "door": door,
                "card_hash": card_hash, "details": details}

    def _ingest(self, s: Source, p: dict, now: float, via: str, msg_id: Optional[str] = None) -> dict:
        try:
            v = self._validate(s, p, now)
        except IngestError as e:
            raise self._reject(s, now, e)
        with self.lock:
            s.last_seen = now
            s.counts[v["type"]] = s.counts.get(v["type"], 0) + 1
            if s.state == "offline":
                s.recovered_at = now
            s.state, s.offline_since = "ok", None
            ev = self._map(s, v, via)
        d = self.emit(ev) if ev is not None else None
        aid = (d or {}).get("id")
        with self.lock:
            if ev is not None:
                s.alerts += 1
            s.last_event = {"type": v["type"], "door": v["door"], "t": v["t"], "arrival": now,
                            "rule": ev.rule if ev else None, "alert_id": aid}
        self._record({"t": v["t"], "arrival": now, "source_id": s.id, "door": v["door"], "type": v["type"],
                      "card_hash": v["card_hash"], "rule": ev.rule if ev else None, "alert_id": aid, "via": via,
                      "msg_id": msg_id, "details": v["details"]})
        out = {"status": "accepted" if ev else "recorded", "source_id": s.id, "type": v["type"],
               "rule": ev.rule if ev else None, "alert_id": aid, "incident_id": (d or {}).get("incident_id")}
        if v["type"] == "access_denied" and ev is None:
            n = len(s.denials.get(v["door"], ()))
            out["note"] = f"{n} of {s.burst_count} denials within {s.burst_window_s:g} s at this door"
        return out

    # ---- mapping -------------------------------------------------------------------------------------------------
    def _door_ctx(self, s: Source, door: Optional[str]) -> dict:
        m = s.doors.get(door) if door else None
        if m is None:  # panel-level report (tamper / alarm input without a door)
            return {"label": s.name, "camera": None, "zone_id": "_panel", "zone_name": f"{s.name} (panel)",
                    "site_zone": None, "criticality": s.criticality}
        cam, zone = m.get("camera"), m.get("zone")
        z = (self.zone_lookup(cam, zone) if cam and zone and self.zone_lookup else None) or {}
        # Same site zone as the camera zone that views the door: the correlator's grouping key. Without a site zone,
        # the camera's own "<camera>:<zone>" key is used, which is exactly what that camera's video events carry.
        site_zone = m.get("site_zone") or z.get("site_zone") or (f"{cam}:{zone}" if cam and zone else None)
        label = f"{door} ({m['name']})" if m.get("name") else door
        return {"label": label, "camera": cam, "zone_id": zone or f"door:{door}",
                "zone_name": z.get("name") or m.get("name") or f"Door {door}", "site_zone": site_zone,
                "criticality": int(m.get("criticality") or z.get("criticality") or s.criticality)}

    def _map(self, s: Source, v: dict, via: str) -> Optional[Event]:
        typ, t, door = v["type"], v["t"], v["door"]
        ctx = self._door_ctx(s, door)
        who = f"{s.kind} {s.id}"
        det = v["details"]
        deriv: dict[str, Any] = {}
        t0 = t
        if typ == "access_denied":
            q = s.denials.setdefault(door, deque())
            q.append((t, v["card_hash"]))
            while q and t - q[0][0] > s.burst_window_s:
                q.popleft()
            if len(q) < s.burst_count:
                return None
            rule, t0, n = "access_denied_burst", q[0][0], len(q)
            creds = sorted({c for _, c in q if c})
            obs = (f"{n} access denials at door {ctx['label']} within {t - t0:.0f} s ({who})"
                   + (f"; {len(creds)} distinct credential(s)" if creds else ""))
            deriv = {"threshold": {"count": s.burst_count, "window_s": s.burst_window_s}, "count": n,
                     "card_hashes": creds}
            q.clear()  # one alert per burst; the next burst starts counting from zero
        elif typ in DIRECT_RULE:
            rule = DIRECT_RULE[typ]
            if typ == "door_forced":
                obs = f"Door {ctx['label']} reported forced open ({who})"
            elif typ == "door_held_open":
                held = det.get("held_s")
                dur = f" for {held:.0f} s" if isinstance(held, (int, float)) and not isinstance(held, bool) else ""
                obs = f"Door {ctx['label']} reported held open{dur} ({who})"
            elif typ == "alarm_input":
                inp = det.get("input")
                obs = f"Alarm input{f' {inp}' if isinstance(inp, str) else ''} reported active at {ctx['label']} ({who})"
            else:
                obs = f"Tamper reported by access-control device at {ctx['label']} ({who})"
        else:  # access_granted, heartbeat: recorded only
            return None
        stem = f"{ctx['camera']}_{rule}_{int(t * 1000)}_{next(_EV_SEQ)}"
        evidence: dict[str, Any] = {"source": MODALITY, "source_id": s.id, "door": door, "frame_time": None}
        if ctx["camera"] and self.evidence is not None:
            cam_ev = self.evidence(ctx["camera"], stem)
            if cam_ev:
                evidence.update(cam_ev, note="camera view when the sensor report arrived (not what triggered it)")
        deriv = {"rule": rule, "modality": MODALITY, "source_id": s.id, "source_kind": s.kind, "door": door,
                 "event_type": typ, "covering_camera": ctx["camera"], "auth": via, "clock": v["clock"],
                 "reported_by": f"{who} (panel-reported state, not verified by video)",
                 "confidence_method": CONFIDENCE_METHOD, "duration_s": round(t - t0, 2), "details": det, **deriv}
        if v["card_hash"] and rule != "access_denied_burst":
            deriv["card_hash"] = v["card_hash"]  # keyed hash only; the raw card id never leaves _validate
        return Event(rule, SENSOR_PREFIX + s.id, ctx["zone_id"], ctx["zone_name"], [], t0, t, obs, deriv,
                     CONFIDENCE, SEVERITY[rule], ctx["criticality"], ctx["site_zone"], None, evidence)

    # ---- health --------------------------------------------------------------------------------------------------
    def check(self, now: Optional[float] = None) -> list[Event]:
        """Raise sensor_offline for sources silent past heartbeat_s (once per outage). Called at 1 Hz."""
        now = self.clock() if now is None else now
        out = []
        with self.lock:
            for s in self.sources.values():
                if not s.heartbeat_s or s.state == "offline":
                    continue
                ref = max(s.last_seen or 0.0, self.started_at)  # a restart gives every source a fresh grace period
                silent = now - ref
                if silent <= s.heartbeat_s:
                    continue
                s.state, s.offline_since = "offline", now
                deriv = {"rule": "sensor_offline", "modality": MODALITY, "source_id": s.id, "source_kind": s.kind,
                         "event_type": "heartbeat_missed", "threshold": {"heartbeat_s": s.heartbeat_s},
                         "silent_s": round(silent, 1), "last_seen": s.last_seen, "duration_s": 0,
                         "confidence_method": "deterministic: time since the last message from this source"}
                obs = (f"No message from access-control source {s.id} for {silent:.0f} s "
                       f"(heartbeat expected every {s.heartbeat_s:g} s)")
                out.append(Event("sensor_offline", SENSOR_PREFIX + s.id, "_source", f"{s.name} (access-control source)",
                                 [], now, now, obs, deriv, 0.9, SEVERITY["sensor_offline"], s.criticality, None, None,
                                 {"source": MODALITY, "source_id": s.id, "snapshot": None, "clip": None,
                                  "frame_time": None}))
        for e in out:
            self.emit(e)
        return out

    def health(self, source_id: str) -> Optional[str]:
        """Camera-style health state for the incident manager: OK / DOWN / WARMUP (nothing received yet)."""
        s = self.sources.get(source_id)
        if s is None:
            return None
        return {"ok": "OK", "offline": "DOWN"}.get(s.state, "WARMUP")

    def status(self, now: Optional[float] = None) -> list[dict]:
        now = self.clock() if now is None else now
        with self.lock:
            return [{"id": s.id, "name": s.name, "kind": s.kind, "state": s.state, "health": self.health(s.id),
                     "last_seen": s.last_seen, "silent_s": round(now - s.last_seen, 1) if s.last_seen else None,
                     "heartbeat_s": s.heartbeat_s, "offline_since": s.offline_since, "recovered_at": s.recovered_at,
                     "last_event": dict(s.last_event) if s.last_event else None, "counts": dict(s.counts),
                     "alerts": s.alerts, "rejected": s.rejected,
                     "last_rejected": dict(s.last_rejected) if s.last_rejected else None, "hmac": s.hmac_status(),
                     "access_denied_burst": {"count": s.burst_count, "window_s": s.burst_window_s},
                     "doors": [{"id": k, "name": m.get("name"), "camera": m.get("camera"), "zone": m.get("zone"),
                                "site_zone": self._door_ctx(s, k)["site_zone"]} for k, m in s.doors.items()]}
                    for s in self.sources.values()]

    # ---- lifecycle -----------------------------------------------------------------------------------------------
    def start(self, interval_s: float = 1.0) -> None:
        if self._thread is not None or not any(s.heartbeat_s for s in self.sources.values()):
            return
        self.started_at = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(interval_s,), daemon=True, name="sensor-health")
        self._thread.start()

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            try:
                self.check()
            except Exception:  # noqa: BLE001 - health bookkeeping must never kill the thread
                pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def _zone_lookup(site) -> Callable[[str, str], Optional[dict]]:
    def lookup(cam: str, zone: str) -> Optional[dict]:
        w = site.worker(cam)
        for z in (list(w.zones) + list(w.px_zones)) if w is not None else []:
            if z.id == zone:
                return {"name": z.name, "site_zone": z.site_zone, "criticality": z.criticality}
        for z in (site.camera_config(cam) or {}).get("zones", []):  # camera disabled: fall back to its config
            if z.get("id") == zone:
                return {"name": z.get("name", zone), "site_zone": z.get("site_zone"),
                        "criticality": int(z.get("criticality", 1))}
        return None
    return lookup


def install(site) -> Optional[SensorHub]:
    """Attach a SensorHub to a Site (config key "sensors"; off when absent or without sources)."""
    cfg = dict(site.cfg.get("sensors") or {})
    if cfg.get("enabled") is False or not cfg.get("sources"):
        site.sensors = None
        return None

    def evidence(cam: str, stem: str) -> Optional[dict]:
        w = site.worker(cam)
        return w.request_evidence(stem) if w is not None else None

    hub = SensorHub(cfg, emit=site.on_event, store=site.store, zone_lookup=_zone_lookup(site), evidence=evidence,
                    pepper=load_pepper(site.data / "sensor_card.key"))
    known = {c["id"] for c in site.cfg.get("cameras", [])}
    for s in hub.sources.values():
        for door, m in s.doors.items():
            if m.get("camera") and m["camera"] not in known:
                hub.warnings.append(f"{s.id}/{door}: camera {m['camera']} is not in this site")
            elif m.get("camera") and m.get("zone") and hub.zone_lookup(m["camera"], m["zone"]) is None:
                hub.warnings.append(f"{s.id}/{door}: camera {m['camera']} has no zone {m['zone']}")
            elif not (m.get("site_zone") or (m.get("camera") and m.get("zone"))):
                hub.warnings.append(f"{s.id}/{door}: no site_zone or camera+zone; its events cannot group with video")
        if s.secret_env and not s.secret:
            hub.warnings.append(f"{s.id}: environment variable {s.secret_env} is not set; signed ingestion disabled")
    for w in hub.warnings:
        print(f"sensors: {w}", flush=True)
    site.sensors = hub
    return hub
