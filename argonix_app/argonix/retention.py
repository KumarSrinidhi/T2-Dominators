"""Retention sweep: delete old recording segments, evidence files and (optionally) old closed alerts.

Site config (all optional; 0 or null disables a limit):

    "retention": {"recordings_days": 7, "evidence_days": 30, "max_disk_gb": 20, "alerts_days": 90,
                  "interval_min": 10, "enabled": true}

Order per sweep: age limits first, then, while recordings + evidence together exceed `max_disk_gb`, the oldest file
(by modification time, recordings and evidence alike) goes first. Never deleted: the evidence files of an alert on
legal hold, recording segments of a held alert's cameras that overlap its time window (+/- 60 s), open alerts, and
files modified in the last `min_age_s` (a segment or clip still being written). Audit rows are never deleted.
Each sweep is logged to the audit table as `retention_swept` (actor `system`) with counts and bytes.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

DEFAULTS = {"recordings_days": 7, "evidence_days": 30, "max_disk_gb": 20, "alerts_days": 90, "interval_min": 10,
            "enabled": True, "min_age_s": 300}
HOLD_MARGIN_S = 60.0
DAY = 86400.0


def policy(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("retention") or {})}


def _protected(store, data: Path) -> tuple[set[Path], list[tuple[set, float, float]]]:
    """Evidence files referenced by held alerts, and (cameras, t0, t1) windows whose recordings are protected."""
    files, windows = set(), []
    for a in store.held_alerts():
        for e in a.get("events", []):
            for v in (e.get("evidence") or {}).values():
                if isinstance(v, str) and "/" in v:
                    files.add((data / v).resolve())
        windows.append((set(a.get("cameras") or []), float(a.get("created_at") or 0) - HOLD_MARGIN_S,
                        float(a.get("updated_at") or 0) + HOLD_MARGIN_S))
    return files, windows


def sweep(store, data_dir: Path, pol: dict, now: Optional[float] = None) -> dict:
    now = time.time() if now is None else float(now)
    data = Path(data_dir).resolve()
    held_files, windows = _protected(store, data)
    rec_rows = {(data / r["path"]).resolve(): r for r in store.list_recordings()}

    def rec_held(p: Path) -> bool:
        r = rec_rows.get(p)
        return bool(r) and any(r["camera"] in cams and r["start"] <= t1 and (r["end"] or r["start"]) >= t0
                               for cams, t0, t1 in windows)

    files = []  # (mtime, size, path, kind)
    for kind in ("recordings", "evidence"):
        root = data / kind
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    files.append((st.st_mtime, st.st_size, p.resolve(), kind))
    files.sort(key=lambda f: f[0])
    min_age = float(pol.get("min_age_s") or 0)
    out = {"recordings": 0, "evidence": 0, "bytes": 0, "alerts": 0, "kept_on_hold": 0, "errors": 0}
    deleted_recs: list[str] = []

    def keep(p: Path, kind: str) -> bool:
        return p in held_files if kind == "evidence" else rec_held(p)

    def delete(f) -> bool:
        _, size, p, kind = f
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            out["errors"] += 1
            return False
        out[kind] += 1
        out["bytes"] += size
        if kind == "recordings" and p in rec_rows:
            deleted_recs.append(rec_rows[p]["path"])
        return True

    held_seen: set[Path] = set()
    remaining = []
    for f in files:
        mtime, _, p, kind = f
        days = float(pol.get(f"{kind}_days") or 0)
        if days and mtime < now - days * DAY and now - mtime >= min_age:
            if keep(p, kind):
                held_seen.add(p)
            elif delete(f):
                continue
        remaining.append(f)

    cap = float(pol.get("max_disk_gb") or 0) * 1e9
    total = sum(f[1] for f in remaining)
    out["disk_bytes_before_cap"] = total
    if cap and total > cap:
        for f in remaining:
            if total <= cap:
                break
            mtime, size, p, kind = f
            if now - mtime < min_age:
                continue
            if keep(p, kind):
                held_seen.add(p)
                continue
            if delete(f):
                total -= size
        out["over_cap"] = total > cap
    out["disk_bytes"] = total
    out["kept_on_hold"] = len(held_seen)
    rdays = float(pol.get("recordings_days") or 0)
    if rdays:  # index rows whose file is already gone (deleted by hand, crash) once they are past retention
        deleted_recs += [r["path"] for p, r in rec_rows.items()
                         if (r["end"] or r["start"]) < now - rdays * DAY and not p.exists() and r["path"] not in deleted_recs]
    if deleted_recs:
        store.delete_recordings(deleted_recs)
    adays = float(pol.get("alerts_days") or 0)
    if adays:
        out["alerts"] = store.delete_closed_alerts(now - adays * DAY)
    for kind in ("recordings", "evidence"):  # drop empty per-camera folders
        root = data / kind
        if root.is_dir():
            for d in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
                try:
                    d.rmdir()
                except OSError:
                    pass
    out["t"] = now
    return out


class RetentionJob:
    """Background sweeper for a Site. First sweep shortly after start, then every `interval_min`."""

    def __init__(self, site, first_delay_s: float = 30.0):
        self.site = site
        self.first_delay = float(first_delay_s)
        self.last: Optional[dict] = None
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None

    @property
    def policy(self) -> dict:
        return policy(self.site.cfg)

    def run_once(self, actor: str = "system", now: Optional[float] = None) -> dict:
        with self.lock:
            res = sweep(self.site.store, self.site.data, self.policy, now)
            pol = {k: v for k, v in self.policy.items() if k != "enabled"}
            self.site.store.audit(actor, "retention_swept", "", {**res, "policy": pol})
            self.last = res
            return res

    def start(self) -> None:
        if not self.policy.get("enabled", True) or (self.thread and self.thread.is_alive()):
            return
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="retention")
        self.thread.start()

    def _loop(self) -> None:
        wait = self.first_delay
        while not self.stop_evt.wait(wait):
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001 - a failed sweep must never take the site down
                self.last = {"t": time.time(), "error": repr(e)[:300]}
            wait = max(10.0, float(self.policy.get("interval_min") or 10) * 60)

    def close(self) -> None:
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

    def status(self) -> dict:
        last = self.last
        if last is None:  # after a restart: the newest logged sweep
            prev = self.site.store.last_audit("retention_swept")
            last = {**prev["details"], "t": prev["t"]} if prev else None
        return {"policy": self.policy, "running": bool(self.thread and self.thread.is_alive()), "last_sweep": last,
                "held_alerts": len(self.site.store.held_alerts())}
