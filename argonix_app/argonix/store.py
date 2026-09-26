"""SQLite store for alerts and recordings (Postgres/Timescale deferred until measured need)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

GENESIS = "0" * 64


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.seq = 0
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY, seq INTEGER, priority TEXT, status TEXT,
                created_at REAL, updated_at REAL, cameras TEXT, rules TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS recordings(id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT,
                path TEXT, start REAL, end REAL);
            CREATE INDEX IF NOT EXISTS ix_alert_seq ON alerts(seq);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, actor TEXT, action TEXT,
                alert_id TEXT, details TEXT, prev_hash TEXT, hash TEXT);
            CREATE INDEX IF NOT EXISTS ix_alert_updated ON alerts(updated_at);
            CREATE TABLE IF NOT EXISTS incidents(id TEXT PRIMARY KEY, seq INTEGER, phase TEXT, status TEXT,
                priority TEXT, opened_at REAL, updated_at REAL, closed_at REAL, cameras TEXT, rules TEXT,
                assignee TEXT, body TEXT);
            CREATE INDEX IF NOT EXISTS ix_inc_updated ON incidents(updated_at);
            CREATE TABLE IF NOT EXISTS incident_alerts(alert_id TEXT PRIMARY KEY, incident_id TEXT);
            CREATE INDEX IF NOT EXISTS ix_inc_alerts ON incident_alerts(incident_id);
            CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, incident_id TEXT,
                alert_id TEXT, priority TEXT, trigger TEXT, decision TEXT, reason TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS nl_cache(key TEXT PRIMARY KEY, t REAL, question TEXT, query TEXT, meta TEXT);
            CREATE TABLE IF NOT EXISTS holds(alert_id TEXT PRIMARY KEY, t REAL, actor TEXT, reason TEXT);
            """)
            self._migrate_recordings()
            self.seq = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM alerts").fetchone()[0]
            self.inc_seq = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM incidents").fetchone()[0]
            self._migrate_audit_chain()

    # ---- tamper-evident audit chain ----------------------------------------------------------------------------
    # hash_n = sha256(hash_{n-1} + canonical JSON of row n), hash_0's predecessor = 64 zeros. Editing, deleting or
    # re-ordering any row breaks every later link; verify_audit() finds the first broken row. Deleting rows from the
    # *end* is only detectable against an externally kept head (export bundles carry it: audit_head()).
    @staticmethod
    def _row_hash(prev: str, seq: int, t: float, actor: str, action: str, alert_id: str, details: str) -> str:
        row = json.dumps({"seq": seq, "t": t, "actor": actor, "action": action, "alert_id": alert_id or "",
                          "details": details or "{}"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256((prev + row).encode("utf-8")).hexdigest()

    def _migrate_audit_chain(self) -> None:
        """Add prev_hash/hash to audit tables created by older versions and chain rows that have no hash yet (only
        the unchained tail after the last hashed row: a NULL hash in the middle stays visible to verify_audit)."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(audit)")}
        for col in ("prev_hash", "hash"):
            if col not in cols:
                self.db.execute(f"ALTER TABLE audit ADD COLUMN {col} TEXT")
        last = self.db.execute("SELECT id, hash FROM audit WHERE hash IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
        prev, after = (last[1], last[0]) if last else (GENESIS, 0)
        rows = self.db.execute("SELECT id,t,actor,action,alert_id,details FROM audit WHERE id > ? ORDER BY id",
                               (after,)).fetchall()
        for rid, t, actor, action, alert_id, details in rows:
            h = self._row_hash(prev, rid, t, actor, action, alert_id, details)
            self.db.execute("UPDATE audit SET prev_hash=?, hash=? WHERE id=?", (prev, h, rid))
            prev = h
        self.db.commit()
        self.audit_chained_on_start = len(rows)
        self._audit_head = prev

    def upsert_alert(self, d: dict[str, Any]) -> int:
        with self.lock:
            self.seq += 1
            rules = ",".join(sorted({e["rule"] for e in d["events"]}))
            self.db.execute("INSERT OR REPLACE INTO alerts VALUES(?,?,?,?,?,?,?,?,?)",
                            (d["id"], self.seq, d["priority"], d["status"], d["created_at"], d["updated_at"],
                             ",".join(d["cameras"]), rules, json.dumps(d)))
            self.db.commit()
            return self.seq

    def get_alert(self, aid: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT body FROM alerts WHERE id=?", (aid,)).fetchone()
        return json.loads(r[0]) if r else None

    def list_alerts(self, status: str = "", priority: str = "", camera: str = "", rule: str = "",
                    since_seq: int = 0, limit: int = 200, since: float = 0.0, until: float = 0.0,
                    updated_since: float = 0.0, second_review: bool = False) -> list[dict]:
        """Newest first. `since`/`until` (epoch s) keep alerts whose [created_at, updated_at] overlaps the range.
        `updated_since` is an alias of `since` (kept for the analytics API). `status` / `priority` accept a
        comma-separated list ("P1,P2"). `second_review`: only alerts flagged "unclear" for a second review."""
        q, a = "SELECT body, seq FROM alerts WHERE seq > ?", [since_seq]
        since = max(float(since or 0.0), float(updated_since or 0.0))
        for col, val in (("status", status), ("priority", priority)):
            vals = [v for v in str(val or "").split(",") if v]
            if vals:
                q += f" AND {col} IN ({','.join('?' * len(vals))})"; a.extend(vals)
        if second_review:
            q += " AND json_extract(body, '$.second_review') = 1"
        if camera:
            q += " AND (',' || cameras || ',') LIKE ?"; a.append(f"%,{camera},%")
        if rule:
            q += " AND (',' || rules || ',') LIKE ?"; a.append(f"%,{rule},%")
        if since:
            q += " AND updated_at >= ?"; a.append(float(since))
        if until:
            q += " AND created_at <= ?"; a.append(float(until))
        q += " ORDER BY updated_at DESC LIMIT ?"; a.append(limit)
        with self.lock:
            rows = self.db.execute(q, a).fetchall()
        out = []
        for body, seq in rows:
            d = json.loads(body); d["seq"] = seq; out.append(d)
        return out

    def set_status(self, aid: str, status: str, note: str = "", disposition: str = "",
                   actor: str = "operator", fields: Optional[dict] = None,
                   details: Optional[dict] = None) -> Optional[dict]:
        """`fields`: extra operator fields (argonix.workflow.plan); `details`: extra audit details."""
        d = self.get_alert(aid)
        if not d:
            return None
        d.update(fields or {})
        d["status"] = status
        if note:
            d["operator_note"] = note
        if disposition:
            d["disposition"] = disposition
        self.upsert_alert(d)
        self.audit(actor, f"status:{status}", aid, {"note": note, "disposition": disposition, **(details or {})})
        return d

    def audit(self, actor: str, action: str, alert_id: str = "", details: Optional[dict] = None) -> None:
        with self.lock:
            t, det, prev = time.time(), json.dumps(details or {}), self._audit_head
            cur = self.db.execute("INSERT INTO audit(t,actor,action,alert_id,details,prev_hash) VALUES(?,?,?,?,?,?)",
                                  (t, actor, action, alert_id, det, prev))
            h = self._row_hash(prev, cur.lastrowid, t, actor, action, alert_id, det)
            self.db.execute("UPDATE audit SET hash=? WHERE id=?", (h, cur.lastrowid))
            self.db.commit()
            self._audit_head = h

    def list_audit(self, alert_id: str = "", limit: int = 500) -> list[dict]:
        q = "SELECT id,t,actor,action,alert_id,details FROM audit" + (" WHERE alert_id=?" if alert_id else "")
        q += " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, ((alert_id,) if alert_id else ()) + (limit,)).fetchall()
        return [{"seq": s, "t": t, "actor": a, "action": ac, "alert_id": al, "details": json.loads(d)}
                for s, t, a, ac, al, d in rows]

    def last_audit(self, action: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT id,t,actor,alert_id,details FROM audit WHERE action=? ORDER BY id DESC LIMIT 1",
                                (action,)).fetchone()
        return {"seq": r[0], "t": r[1], "actor": r[2], "action": action, "alert_id": r[3],
                "details": json.loads(r[4])} if r else None

    def audit_head(self) -> dict:
        """Current chain head (seq + hash of the newest audit row): anchor it outside the DB to detect truncation."""
        with self.lock:
            r = self.db.execute("SELECT id, hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return {"seq": r[0] if r else 0, "hash": r[1] if r else GENESIS, "algorithm": "sha256",
                "rule": "hash_n = sha256(hash_{n-1} + canonical_json(row_n)), hash_0 predecessor = 64 zeros"}

    def verify_audit(self) -> dict:
        """Recompute the whole chain. ok=False with first_bad_seq = the first row whose link does not match
        (edited, inserted or deleted rows before it)."""
        with self.lock:
            rows = self.db.execute("SELECT id,t,actor,action,alert_id,details,prev_hash,hash FROM audit "
                                   "ORDER BY id").fetchall()
        prev, bad = GENESIS, None
        for rid, t, actor, action, alert_id, details, ph, h in rows:
            if ph != prev or h != self._row_hash(prev, rid, t, actor, action, alert_id, details):
                bad = rid
                break
            prev = h
        return {"ok": bad is None, "rows": len(rows), "first_bad_seq": bad,
                "head": rows[-1][7] if rows and bad is None else None}

    # ---- legal hold ------------------------------------------------------------------------------------------
    def set_hold(self, aid: str, actor: str, reason: str = "") -> Optional[dict]:
        """Put an alert on legal hold: retention never deletes it or its evidence until released."""
        d = self.get_alert(aid)
        if not d:
            return None
        hold = {"t": time.time(), "actor": actor, "reason": reason[:500]}
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO holds VALUES(?,?,?,?)", (aid, hold["t"], actor, hold["reason"]))
            self.db.commit()
        d["legal_hold"] = hold
        self.upsert_alert(d)
        return d

    def release_hold(self, aid: str) -> Optional[dict]:
        d = self.get_alert(aid)
        if not d:
            return None
        with self.lock:
            self.db.execute("DELETE FROM holds WHERE alert_id=?", (aid,))
            self.db.commit()
        d.pop("legal_hold", None)
        self.upsert_alert(d)
        return d

    def held_alerts(self) -> list[dict]:
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT alert_id FROM holds")]
        return [a for a in (self.get_alert(i) for i in ids) if a]

    def delete_closed_alerts(self, before: float, statuses: tuple = ("resolved", "dismissed")) -> int:
        """Retention: drop closed alerts last updated before `before`, never ones on hold. The audit trail stays."""
        q = (f"SELECT id FROM alerts WHERE updated_at < ? AND status IN ({','.join('?' * len(statuses))}) "
             "AND id NOT IN (SELECT alert_id FROM holds)")
        with self.lock:
            ids = [r[0] for r in self.db.execute(q, (float(before), *statuses))]
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                self.db.execute(f"DELETE FROM alerts WHERE id IN ({marks})", chunk)
                self.db.execute(f"DELETE FROM incident_alerts WHERE alert_id IN ({marks})", chunk)
            self.db.commit()
        return len(ids)

    # ---- incidents (argonix.incidents) -----------------------------------------------------------------------
    def upsert_incident(self, d: dict[str, Any]) -> int:
        with self.lock:
            self.inc_seq += 1
            d["seq"] = self.inc_seq
            self.db.execute("INSERT OR REPLACE INTO incidents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (d["id"], self.inc_seq, d["phase"], d["status"], d["priority"], d["opened_at"],
                             d["updated_at"], d.get("closed_at"), ",".join(d.get("cameras", [])),
                             ",".join(d.get("rules", [])), d.get("assignee") or "", json.dumps(d)))
            self.db.commit()
            return self.inc_seq

    def get_incident(self, iid: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT body FROM incidents WHERE id=?", (iid,)).fetchone()
        return json.loads(r[0]) if r else None

    def list_incidents(self, phase: str = "", status: str = "", priority: str = "", camera: str = "",
                       since: float = 0.0, until: float = 0.0, live_or_closed_after: float = 0.0,
                       limit: int = 200) -> list[dict]:
        """Newest activity first. since/until: incidents whose [opened_at, updated_at] overlaps the range.
        live_or_closed_after: only incidents not closed, or closed at/after that time (manager warm start)."""
        q, a = "SELECT body FROM incidents WHERE 1=1", []
        for col, val in (("phase", phase), ("status", status), ("priority", priority)):
            if val:
                q += f" AND {col}=?"; a.append(val)
        if camera:
            q += " AND (',' || cameras || ',') LIKE ?"; a.append(f"%,{camera},%")
        if since:
            q += " AND updated_at >= ?"; a.append(float(since))
        if until:
            q += " AND opened_at <= ?"; a.append(float(until))
        if live_or_closed_after:
            q += " AND (phase != 'closed' OR closed_at >= ?)"; a.append(float(live_or_closed_after))
        q += " ORDER BY updated_at DESC LIMIT ?"; a.append(int(limit))
        with self.lock:
            rows = self.db.execute(q, a).fetchall()
        return [json.loads(r[0]) for r in rows]

    def alerts_since(self, seq: int, limit: int = 50) -> list[dict]:
        """Alerts written after `seq`, oldest change first, so a caller can page through a burst without gaps."""
        with self.lock:
            rows = self.db.execute("SELECT body, seq FROM alerts WHERE seq > ? ORDER BY seq LIMIT ?",
                                   (int(seq), int(limit))).fetchall()
        out = []
        for body, sq in rows:
            d = json.loads(body); d["seq"] = sq; out.append(d)
        return out

    def incidents_since(self, seq: int, limit: int = 50) -> list[dict]:
        """Incidents written after `seq` (oldest change first; each body carries its seq). Used by the WebSocket."""
        with self.lock:
            rows = self.db.execute("SELECT body, seq FROM incidents WHERE seq > ? ORDER BY seq LIMIT ?",
                                   (int(seq), int(limit))).fetchall()
        out = []
        for body, sq in rows:
            d = json.loads(body); d["seq"] = sq; out.append(d)
        return out

    def link_alert(self, alert_id: str, incident_id: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO incident_alerts VALUES(?,?)", (alert_id, incident_id))
            self.db.commit()

    def incident_for_alert(self, alert_id: str) -> Optional[str]:
        with self.lock:
            r = self.db.execute("SELECT incident_id FROM incident_alerts WHERE alert_id=?", (alert_id,)).fetchone()
        return r[0] if r else None

    def add_notification(self, t: float, incident_id: str, alert_id: str, priority: str, trigger: str,
                         decision: str, reason: str, details: Optional[dict] = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO notifications(t,incident_id,alert_id,priority,trigger,decision,reason,details)"
                            " VALUES(?,?,?,?,?,?,?,?)", (t, incident_id, alert_id, priority, trigger, decision,
                                                         reason, json.dumps(details or {})))
            self.db.commit()

    def list_notifications(self, incident_id: str = "", limit: int = 200) -> list[dict]:
        q = "SELECT t,incident_id,alert_id,priority,trigger,decision,reason,details FROM notifications"
        q += (" WHERE incident_id=?" if incident_id else "") + " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, ((incident_id,) if incident_id else ()) + (int(limit),)).fetchall()
        keys = ("t", "incident_id", "alert_id", "priority", "trigger", "decision", "reason")
        return [{**dict(zip(keys, r[:7])), "details": json.loads(r[7] or "{}")} for r in rows]

    def cache_get(self, key: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT query, meta FROM nl_cache WHERE key=?", (key,)).fetchone()
        return {"query": json.loads(r[0]), "meta": json.loads(r[1] or "{}")} if r else None

    def cache_put(self, key: str, question: str, query: dict, meta: Optional[dict] = None) -> None:
        import time
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO nl_cache VALUES(?,?,?,?,?)",
                            (key, time.time(), question[:500], json.dumps(query), json.dumps(meta or {})))
            self.db.commit()

    # ---- recordings: stream='analytics' (pipeline H264Writer) | 'main' (MediaMTX restream, argonix/restream.py) ----
    REC_COLS = ("id", "camera", "path", "start", "end", "stream", "size")

    def _migrate_recordings(self) -> None:
        """Add the stream/size columns to databases created before main-stream recording (idempotent and additive:
        existing rows become stream='analytics')."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(recordings)").fetchall()}
        if "stream" not in cols:
            self.db.execute("ALTER TABLE recordings ADD COLUMN stream TEXT NOT NULL DEFAULT 'analytics'")
        if "size" not in cols:
            self.db.execute("ALTER TABLE recordings ADD COLUMN size INTEGER")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_rec_stream_path ON recordings(stream, path)")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_rec_cam_start ON recordings(camera, start)")
        self.db.commit()

    def add_recording(self, camera: str, path: str, start: float, end: float, stream: str = "analytics",
                      size: Optional[int] = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO recordings(camera,path,start,end,stream,size) VALUES(?,?,?,?,?,?)",
                            (camera, path, start, end, stream, size))
            self.db.commit()

    def recordings_between(self, camera: str, start: float, end: float) -> list[dict]:
        """A camera's closed segments overlapping [start, end], oldest first."""
        with self.lock:
            rows = self.db.execute("SELECT camera,path,start,end FROM recordings WHERE camera=? AND start < ? AND end > ? "
                                   "ORDER BY start", (camera, float(end), float(start))).fetchall()
        return [dict(zip(("camera", "path", "start", "end"), r)) for r in rows]

    def upsert_recording(self, camera: str, path: str, start: float, end: float, stream: str = "main",
                         size: Optional[int] = None) -> Optional[str]:
        """Insert or refresh a recording identified by (stream, path). Returns 'indexed', 'updated' or None."""
        with self.lock:
            r = self.db.execute("SELECT id, end, size FROM recordings WHERE stream=? AND path=?", (stream, path)).fetchone()
            if r is None:
                self.db.execute("INSERT INTO recordings(camera,path,start,end,stream,size) VALUES(?,?,?,?,?,?)",
                                (camera, path, start, end, stream, size))
                self.db.commit()
                return "indexed"
            if abs((r[1] or 0) - end) > 1e-3 or r[2] != size:
                self.db.execute("UPDATE recordings SET start=?, end=?, size=? WHERE id=?", (start, end, size, r[0]))
                self.db.commit()
                return "updated"
            return None

    def recording_paths(self, stream: str) -> list[tuple[int, str]]:
        with self.lock:
            return [(r[0], r[1]) for r in self.db.execute("SELECT id, path FROM recordings WHERE stream=?", (stream,))]

    def delete_recording(self, rid: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM recordings WHERE id=?", (int(rid),))
            self.db.commit()

    def delete_recording_path(self, path: str, stream: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM recordings WHERE stream=? AND path=?", (stream, path))
            self.db.commit()

    def get_recording(self, rid: int) -> Optional[dict]:
        with self.lock:
            r = self.db.execute(f"SELECT {','.join(self.REC_COLS)} FROM recordings WHERE id=?", (int(rid),)).fetchone()
        return dict(zip(self.REC_COLS, r)) if r else None

    def list_recordings(self, camera: str = "", stream: str = "", limit: int = 0) -> list[dict]:
        """Newest first. ``stream``: '' = both, 'analytics' or 'main'."""
        q, a = f"SELECT {','.join(self.REC_COLS)} FROM recordings WHERE 1=1", []
        if camera:
            q += " AND camera=?"; a.append(camera)
        if stream:
            q += " AND stream=?"; a.append(stream)
        q += " ORDER BY start DESC"
        if limit:
            q += " LIMIT ?"; a.append(int(limit))
        with self.lock:
            rows = self.db.execute(q, a).fetchall()
        return [dict(zip(self.REC_COLS, r)) for r in rows]

    def recording_at(self, camera: str, t: float, stream: str = "main", slack_s: float = 2.0) -> Optional[dict]:
        """The segment covering time ``t``, or one starting at most ``slack_s`` after it (segments are cut on
        keyframes, so a moment can fall in a small gap between two)."""
        with self.lock:
            r = self.db.execute(f"SELECT {','.join(self.REC_COLS)} FROM recordings WHERE camera=? AND stream=? "
                                "AND start <= ? AND end >= ? ORDER BY CASE WHEN start <= ? THEN 0 ELSE 1 END, "
                                "CASE WHEN start <= ? THEN -start ELSE start END LIMIT 1",
                                (camera, stream, t + slack_s, t, t, t)).fetchone()
        return dict(zip(self.REC_COLS, r)) if r else None

    def delete_recordings(self, paths: list[str]) -> None:
        with self.lock:
            self.db.executemany("DELETE FROM recordings WHERE path=?", [(p,) for p in paths])
            self.db.commit()
