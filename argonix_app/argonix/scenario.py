"""Demo scenarios: play a scripted, clearly labelled DEMO sequence through the real alert path.

A scenario is a JSON file in `configs/scenarios/` (or the site config's `demo.scenarios_dir`):

    {"name": "door_intrusion_tamper", "title": "...", "description": "...",
     "steps": [{"at": 0,  "action": "event",   "rule": "intrusion", "camera": "cam1", "zone": "server_door"},
               {"at": 8,  "action": "sensor",  "type": "door_forced", "camera": "cam1", "zone": "server_door"},
               {"at": 15, "action": "tamper",  "camera": "cam1", "fault": "cover", "seconds": 15},
               {"at": 30, "action": "recover", "camera": "cam1"}]}

Actions:
  event    a synthetic Event handed to Site.on_event (correlation, incidents, push decisions, notifications all run
           for real). derivation.demo_scenario = true, the observation starts with "[DEMO]" and the evidence
           snapshot is the camera's current view stamped "DEMO".
  sensor   an access-control report. If the other sensor module is installed (site.sensors with a configured
           `source_id`/`door`), the report goes through its ingestion with details.demo_scenario = true; otherwise an
           equivalent `door_forced`-style Event is emitted directly (rule registered here if nobody else did).
  tamper   the existing tamper injection (CameraWorker.inject: cover | freeze | black | blur | shift); the camera's
           health monitor then raises a real camera_health event, labelled DEMO while the injection is active.
  recover  ends the injection early.
  note     a timeline marker only.
`camera` may be omitted (first running camera) and `zone` may be omitted (that camera's first zone).

Demo events never correlate with observed ones, never share an incident with them, never inhibit them, and are
excluded from rule precision, baselines and hotspots (see models.is_demo).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .health import FAULTS
from .models import Event
from .rules import LABEL, SEVERITY, register_rule

ACTIONS = ("event", "sensor", "tamper", "recover", "note")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_DURATION_S = 600.0
DEMO_CONFIDENCE_METHOD = "scripted demo scenario: not observed, confidence is a fixed placeholder"
SENSOR_LABELS = {"door_forced": ("Door reported forced open", 3), "door_held_open": ("Door reported held open", 2),
                 "alarm_input": ("Alarm input reported", 3)}


def _ensure_rule(rule: str) -> None:
    """Sensor rules may be registered by the sensor module; register a fallback label only if nobody did."""
    if rule not in LABEL and rule in SENSOR_LABELS:
        register_rule(rule, *SENSOR_LABELS[rule])


def load_scenario(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate(d, Path(path).stem)


def validate(d: dict, default_name: str = "") -> dict:
    if not isinstance(d, dict):
        raise ValueError("scenario must be a JSON object")
    name = str(d.get("name") or default_name)
    if not NAME_RE.match(name):
        raise ValueError(f"scenario name {name!r}: 1-64 letters, digits, '_' or '-'")
    steps = d.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"scenario {name}: 'steps' must be a non-empty list")
    out = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise ValueError(f"scenario {name} step {i}: must be an object")
        act = s.get("action")
        if act not in ACTIONS:
            raise ValueError(f"scenario {name} step {i}: action must be one of {ACTIONS}")
        try:
            at = float(s.get("at", 0))
        except (TypeError, ValueError):
            raise ValueError(f"scenario {name} step {i}: 'at' must be seconds")
        if not 0 <= at <= MAX_DURATION_S:
            raise ValueError(f"scenario {name} step {i}: 'at' must be 0-{MAX_DURATION_S:g} s")
        if act == "event" and not s.get("rule"):
            raise ValueError(f"scenario {name} step {i}: an event needs a 'rule'")
        if act == "tamper" and s.get("fault", "cover") not in FAULTS:
            raise ValueError(f"scenario {name} step {i}: fault must be one of {FAULTS}")
        out.append({**s, "at": at})
    out.sort(key=lambda s: s["at"])
    return {"name": name, "title": str(d.get("title") or name), "description": str(d.get("description") or ""),
            "steps": out, "duration_s": max(s["at"] for s in out)}


def describe(step: dict) -> str:
    if step.get("label"):
        return str(step["label"])
    act = step["action"]
    cam = step.get("camera") or "first camera"
    if act == "event":
        return f"{LABEL.get(step['rule'], step['rule'])} on {cam}" + (f" / {step['zone']}" if step.get("zone") else "")
    if act == "sensor":
        return f"Access-control report '{step.get('type', 'door_forced')}' near {cam}"
    if act == "tamper":
        return f"Tamper injection '{step.get('fault', 'cover')}' on {cam} for {float(step.get('seconds', 15)):g} s"
    if act == "recover":
        return f"Camera {cam} recovers (injection ends)"
    return str(step.get("text") or "note")


class ScenarioEngine:
    """Runs one scenario at a time on a background thread. `speed` > 1 compresses time (tests)."""

    def __init__(self, site, scenarios_dir: Path, audit: Optional[Callable[[str, str, dict], None]] = None,
                 speed: float = 1.0, clock: Callable[[], float] = time.time):
        self.site, self.dir, self.speed, self.clock = site, Path(scenarios_dir), max(1e-3, float(speed)), clock
        self.audit = audit or (lambda actor, action, details: None)
        self.lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._run: Optional[dict] = None       # status of the current / last run
        self._tampered: set[str] = set()       # cameras with an injection started by the current run

    # ---- catalogue -------------------------------------------------------------------------------------------
    def scenarios(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")) if self.dir.is_dir() else []:
            try:
                sc = load_scenario(p)
                out.append({"name": sc["name"], "title": sc["title"], "description": sc["description"],
                            "duration_s": sc["duration_s"], "file": p.name,
                            "steps": [{"at": s["at"], "action": s["action"], "label": describe(s)} for s in sc["steps"]]})
            except (ValueError, OSError) as e:
                out.append({"name": p.stem, "file": p.name, "error": str(e)[:300]})
        return out

    def get(self, name: str) -> dict:
        if not NAME_RE.match(name or ""):
            raise KeyError(name)
        for p in sorted(self.dir.glob("*.json")) if self.dir.is_dir() else []:
            try:
                sc = load_scenario(p)
            except (ValueError, OSError):
                continue
            if sc["name"] == name:
                return sc
        raise KeyError(name)

    # ---- run control -----------------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        with self.lock:
            r = json.loads(json.dumps(self._run)) if self._run else None
        if r is None:
            return {"running": False, "run": None}
        if r["state"] == "running":
            r["elapsed_s"] = round(min(r["duration_s"], (self.clock() - r["started_at"]) * self.speed), 1)
        r["progress"] = round(r["elapsed_s"] / r["duration_s"], 3) if r["duration_s"] else 1.0
        return {"running": self.running, "run": r}

    def start(self, name: str, actor: str) -> dict:
        sc = self.get(name)
        with self.lock:
            if self.running:
                raise RuntimeError(f"scenario '{self._run['name']}' is already running")
            self._stop.clear()
            self._tampered = set()
            self._run = {"name": sc["name"], "title": sc["title"], "actor": actor, "state": "running",
                         "started_at": self.clock(), "ended_at": None, "elapsed_s": 0.0,
                         "duration_s": sc["duration_s"], "alerts": [],
                         "steps": [{"at": s["at"], "action": s["action"], "label": describe(s), "state": "pending",
                                    "detail": ""} for s in sc["steps"]]}
            self._thread = threading.Thread(target=self._loop, args=(sc,), daemon=True, name="demo-scenario")
            self._thread.start()
        self.audit(actor, "demo_scenario_run", {"scenario": sc["name"], "steps": len(sc["steps"])})
        return self.status()

    def stop(self, actor: str) -> dict:
        was = self.running
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10)
        if was:
            self.audit(actor, "demo_scenario_stopped", {"scenario": (self._run or {}).get("name")})
        return self.status()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)

    # ---- execution -------------------------------------------------------------------------------------------
    def _set(self, i: int, state: str, detail: str = "") -> None:
        with self.lock:
            st = self._run["steps"][i]
            st["state"], st["detail"] = state, detail[:300]

    def _loop(self, sc: dict) -> None:
        t0 = self.clock()
        final = "done"
        try:
            for i, step in enumerate(sc["steps"]):
                wait = step["at"] / self.speed - (self.clock() - t0)
                if wait > 0 and self._stop.wait(wait):
                    final = "stopped"
                    break
                if self._stop.is_set():
                    final = "stopped"
                    break
                try:
                    self._set(i, "done", self._do(sc, step) or "")
                except Exception as e:  # noqa: BLE001 - one bad step must not end the demo
                    self._set(i, "failed", f"{type(e).__name__}: {e}")
        finally:
            if final == "stopped":
                for w in self._tampered:  # never leave a camera "covered" after a stopped demo
                    wk = self.site.worker(w)
                    if wk is not None:
                        wk.clear_fault()
            with self.lock:
                r = self._run
                r["state"], r["ended_at"] = final, self.clock()
                r["elapsed_s"] = round(min(r["duration_s"], (r["ended_at"] - r["started_at"]) * self.speed), 1)
                for st in r["steps"]:
                    if st["state"] == "pending":
                        st["state"] = "skipped"

    def _worker(self, step: dict):
        cam = step.get("camera")
        ws = list(getattr(self.site, "workers", []))
        w = next((w for w in ws if w.id == cam), None) if cam else (ws[0] if ws else None)
        if w is None:
            raise ValueError(f"unknown camera {cam!r}" if cam else "no camera is running")
        return w

    @staticmethod
    def _zone(w, zid: Optional[str]):
        zones = list(getattr(getattr(w, "engine", None), "zones", []) or [])
        if zid:
            z = next((z for z in zones if z.id == zid), None)
            if z is None:
                raise ValueError(f"camera {w.id} has no zone {zid!r}")
            return z
        return zones[0] if zones else None

    def _do(self, sc: dict, step: dict) -> str:
        act = step["action"]
        if act == "note":
            return str(step.get("text") or "")
        w = self._worker(step)
        if act == "tamper":
            fault, secs = step.get("fault", "cover"), float(step.get("seconds", 15))
            w.demo_tag = (sc["name"], time.time() + secs + float(step.get("label_grace_s", 15)))
            w.inject(fault, secs)
            self._tampered.add(w.id)
            return f"{fault} injected on {w.id} for {secs:g} s (health monitor raises the event once it persists)"
        if act == "recover":
            w.clear_fault()
            return f"injection on {w.id} ended; the health monitor clears the condition on its own"
        if act == "sensor":
            return self._sensor(sc, step, w)
        return self._event(sc, step, w, step["rule"])

    def _event(self, sc: dict, step: dict, w, rule: str, extra: Optional[dict] = None) -> str:
        z = self._zone(w, step.get("zone"))
        now = time.time()
        zid, zname = (z.id, z.name) if z is not None else ("_view", f"{w.name} (whole view)")
        crit = int(step.get("criticality") or (z.criticality if z is not None else 2))
        ground = None
        if z is not None and getattr(w, "H", None) is not None and z.polygon:
            ground = (float(np.mean([p[0] for p in z.polygon])), float(np.mean([p[1] for p in z.polygon])))
        text = step.get("observation") or f"{LABEL.get(rule, rule)} in {zname} (scripted)"
        der = {"rule": rule, "demo_scenario": True, "scenario": sc["name"], "step_at_s": step["at"],
               "confidence_method": DEMO_CONFIDENCE_METHOD, "threshold": {"demo": "scripted, no threshold"},
               "duration_s": 0, **(extra or {})}
        e = Event(rule, w.id, zid, zname, [], now, now, f"[DEMO] {text}", der,
                  float(step.get("confidence", 0.9)), int(step.get("severity") or SEVERITY.get(rule, 2)), crit,
                  getattr(z, "site_zone", None), ground, self._evidence(w, rule, now))
        self.site.on_event(e)
        aid = self._alert_for(w.id, now)
        if aid:
            with self.lock:
                if aid not in self._run["alerts"]:
                    self._run["alerts"].append(aid)
        return f"{LABEL.get(rule, rule)} on {w.id}/{zid}" + (f" -> alert {aid}" if aid else "")

    def _sensor(self, sc: dict, step: dict, w) -> str:
        typ = step.get("type", "door_forced")
        hub = getattr(self.site, "sensors", None)
        src, door = step.get("source_id"), step.get("door")
        ingest = getattr(hub, "ingest_token", None)
        if callable(ingest) and src and src in (getattr(hub, "sources", None) or {}):
            body = {"source_id": src, "type": typ, "door": door,
                    "details": {"demo_scenario": True, "scenario": sc["name"]}}
            try:
                r = ingest(f"demo:{sc['name']}", json.dumps(body).encode())
                if r.get("alert_id"):
                    with self.lock:
                        self._run["alerts"].append(r["alert_id"])
                return f"sent through the sensor module ({src}/{door}): {r.get('status')}"
            except Exception as e:  # noqa: BLE001 - fall back to a direct demo event
                fallback = f"sensor module refused ({type(e).__name__}: {e}); "
        else:
            fallback = "sensor module not configured for this step; "
        _ensure_rule(typ)
        return fallback + self._event(sc, {**step, "observation": step.get("observation") or
                                           f"Door reported {typ.replace('door_', '').replace('_', ' ')} by an "
                                           "access-control panel (simulated)"}, w, typ,
                                      {"modality": "access_control (simulated)", "event_type": typ})

    def _alert_for(self, cam: str, t: float) -> Optional[str]:
        corr = getattr(self.site, "correlator", None)
        for a in reversed(getattr(corr, "open", []) or []):
            if any(e.camera_id == cam and e.derivation.get("demo_scenario") and abs(e.t_end - t) < 1 for e in a.events):
                return a.id
        return None

    def _evidence(self, w, rule: str, t: float) -> dict:
        """The camera's current view stamped DEMO, so the drawer shows context (not what 'triggered' the event)."""
        jpg = getattr(w, "latest_jpeg", None)
        data = getattr(self.site, "data", None)
        if not jpg or data is None:
            return {"snapshot": None, "clip": None, "frame_time": None, "demo": True}
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"snapshot": None, "clip": None, "frame_time": None, "demo": True}
        hh, ww = img.shape[:2]
        cv2.rectangle(img, (0, 0), (ww - 1, hh - 1), (0, 165, 255), max(4, ww // 160))
        label = "DEMO SCENARIO - scripted event, not observed"
        fs = max(0.5, ww / 1400)
        cv2.rectangle(img, (0, 0), (int(ww * 0.75), int(38 * fs)), (0, 0, 0), -1)
        cv2.putText(img, label, (8, int(27 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), 2, cv2.LINE_AA)
        rel = f"evidence/demo_{w.id}_{rule}_{int(t * 1000)}.jpg"
        (Path(data) / "evidence").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(data) / rel), img)
        return {"snapshot": rel, "clip": None, "frame_time": t, "demo": True,
                "note": "camera view when the demo step ran, stamped DEMO"}
