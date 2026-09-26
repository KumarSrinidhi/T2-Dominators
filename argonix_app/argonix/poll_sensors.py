"""Motion sensors polled over HTTP, e.g. the PIR on an ESP32-CAM (firmware/esp32cam_argonix, docs/ESP32_CAM.md).

Argus PULLS the sensor state (GET <url> -> {"motion": bool, "count": int, ...}), so nothing has to reach the
dashboard from the sensor's network. A rising edge (motion false -> true, or the device's trigger count going up)
becomes one `sensor_motion` event:

  * It is a deterministic observation ("PIR 'esp32_pir' triggered"), not an inference, and says nothing about who.
  * On its own it is low priority (severity 1). Linked to a camera zone with a `site_zone`, it merges with that
    zone's video events in the correlator, and the extra distinct rule raises the alert's score.
  * The linked camera's latest annotated frame is saved as the event's snapshot, so the operator can look.

This is separate from argonix/sensors.py (signed access-control webhooks that devices PUSH): config key
"poll_sensors", not "sensors".

  "poll_sensors": [{"id": "esp32_pir", "url": "http://192.168.1.50/pir", "camera": "esp32cam", "zone": "entrance",
                    "zone_name": "Entrance", "site_zone": "entrance", "criticality": 1, "poll_s": 0.5}]

Contributed on branch binay/webcam-capture (2026-09-25), ported onto the current pipeline.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request
from typing import Callable, Optional
from urllib.parse import urlsplit

from .cameras import redact
from .models import Event
from .rules import register_rule

RULE = "sensor_motion"
register_rule(RULE, "Motion sensor triggered", 1)
_EV_SEQ = itertools.count(1)  # unique evidence file names


def _http_get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - scheme checked in validate_poll_sensor
        return json.loads(r.read(4096).decode("utf-8"))


def validate_poll_sensor(cfg: dict) -> dict:
    sid = str(cfg.get("id", ""))
    if not sid or not all(ch.isalnum() or ch in "_-" for ch in sid) or len(sid) > 40:
        raise ValueError("poll sensor id must be 1-40 characters of letters, digits, '_' or '-'")
    url = str(cfg.get("url", ""))
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"poll sensor '{sid}': url must be http(s)://host/... (got {redact(url)!r})")
    poll = float(cfg.get("poll_s", 0.5))
    if not 0.1 <= poll <= 60:
        raise ValueError(f"poll sensor '{sid}': poll_s must be 0.1-60")
    crit = int(cfg.get("criticality", 1))
    if not 1 <= crit <= 3:
        raise ValueError(f"poll sensor '{sid}': criticality must be 1-3")
    return {**cfg, "id": sid, "url": url, "poll_s": poll, "criticality": crit}


class HttpPollSensor(threading.Thread):
    def __init__(self, cfg: dict, site, fetch: Optional[Callable[[str, float], dict]] = None):
        cfg = validate_poll_sensor(cfg)
        super().__init__(daemon=True, name=f"poll-sensor-{cfg['id']}")
        self.cfg, self.site, self.id = cfg, site, cfg["id"]
        self.fetch = fetch or _http_get_json
        self.stop_evt = threading.Event()
        self.status = {"type": "http_poll", "source": redact(cfg["url"]), "online": False, "motion": None,
                       "count": None, "triggers": 0, "last_ok": None, "error": ""}
        self._motion: Optional[bool] = None
        self._count: Optional[int] = None

    # ---- one poll (also called directly by tests) --------------------------------------------------------------
    def poll_once(self, now: Optional[float] = None) -> Optional[Event]:
        now = time.time() if now is None else now
        try:
            d = self.fetch(self.cfg["url"], float(self.cfg.get("timeout_s", 2.0)))
            motion, count = bool(d.get("motion")), d.get("count")
            count = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else None
        except Exception as e:  # noqa: BLE001 - device offline, Wi-Fi drop, bad JSON: report it, keep polling
            self.status.update(online=False, error=redact(repr(e))[:200])
            return None
        rising = (self._motion is False and motion) or (
            self._count is not None and count is not None and count > self._count)
        first = self._motion is None
        self._motion, self._count = motion, count
        self.status.update(online=True, error="", motion=motion, count=count, last_ok=now)
        if first or not rising:  # the first reading only sets the baseline
            return None
        self.status["triggers"] += 1
        return self._event(now, count)

    def _event(self, t: float, count: Optional[int]) -> Event:
        c = self.cfg
        cam = c.get("camera") or self.id
        zone = c.get("zone") or "_sensor"
        e = Event(RULE, cam, zone, c.get("zone_name") or f"{self.id} (sensor)", [], t, t,
                  f"Motion sensor '{self.id}' triggered" + (f" (PIR, trigger #{count})" if count is not None else ""),
                  {"rule": RULE, "sensor": self.id, "kind": c.get("kind", "pir"), "source": redact(c["url"]),
                   "device_count": count, "modality": "motion_sensor", "source_id": self.id, "event_type": "motion",
                   "confidence_method": "fixed for a binary motion sensor (uncalibrated)",
                   "note": "detects a warm moving body in the sensor's field; says nothing about identity"},
                  0.9, 1, c["criticality"], c.get("site_zone"))
        e.evidence = self._snapshot(cam, t)
        return e

    def _snapshot(self, cam: str, t: float) -> dict:
        w = self.site.worker(cam) if hasattr(self.site, "worker") else None
        jpeg = getattr(w, "latest_jpeg", None) if w else None
        if not jpeg:
            return {"snapshot": None, "clip": None, "frame_time": t, "sensor": self.id}
        rel = f"evidence/{cam}_{RULE}_{self.id}_{int(t * 1000)}_{next(_EV_SEQ)}.jpg"
        (self.site.data / "evidence").mkdir(parents=True, exist_ok=True)
        (self.site.data / rel).write_bytes(jpeg)
        return {"snapshot": rel, "clip": None, "frame_time": t, "sensor": self.id, "camera": cam}

    # ---- thread -------------------------------------------------------------------------------------------------
    def run(self) -> None:
        while not self.stop_evt.is_set():
            e = self.poll_once()
            if e is not None:
                try:
                    self.site.on_event(e)
                except Exception as err:  # noqa: BLE001 - never let one bad event stop the poller
                    self.status["error"] = redact(repr(err))[:200]
            self.stop_evt.wait(self.cfg["poll_s"])

    def stop(self) -> None:
        self.stop_evt.set()


def install(site) -> list[HttpPollSensor]:
    """Build the site's poll sensors from config "poll_sensors" (started and stopped with the site)."""
    site.poll_sensors = [HttpPollSensor(s, site) for s in site.cfg.get("poll_sensors", [])]
    return site.poll_sensors
