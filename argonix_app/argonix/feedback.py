"""Operator feedback: disposition taxonomy, per-rule precision with Wilson intervals, evidence export."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Optional

from .models import alert_is_demo

DISPOSITIONS = {
    "true_positive_actioned": True,
    "true_positive_no_action": True,
    "false_positive_detection": False,    # detector saw something that is not a person
    "false_positive_rule": False,         # person was real, rule/threshold not meaningful here
    "false_positive_calibration": False,  # position/zone wrong due to calibration
    "duplicate": None,                    # excluded from precision
    "test_or_drill": None,
    "unclear": None,                      # not decided: stays acknowledged, flagged for a second review
}
CLOSED = {"resolved", "dismissed"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def precision_by_rule(alerts: list[dict]) -> list[dict]:
    """Per rule: TP/FP from operator dispositions. Tiny samples are reported with wide intervals, not hidden.
    Demo-scenario alerts (scripted, not observed) are excluded."""
    stats: dict[str, dict] = {}
    for a in alerts:
        if alert_is_demo(a):
            continue
        rules = {e["rule"] for e in a["events"]}
        disp = a.get("disposition")
        for r in rules:
            s = stats.setdefault(r, {"rule": r, "alerts": 0, "labelled": 0, "tp": 0, "fp": 0, "by_disposition": {}})
            s["alerts"] += 1
            if disp:
                s["by_disposition"][disp] = s["by_disposition"].get(disp, 0) + 1
                tp = DISPOSITIONS.get(disp)
                if tp is not None:
                    s["labelled"] += 1
                    s["tp" if tp else "fp"] += 1
    for s in stats.values():
        n = s["labelled"]
        s["precision"] = round(s["tp"] / n, 3) if n else None
        s["precision_ci95"] = wilson(s["tp"], n)
        s["enough_data"] = n >= 50  # below this, calibration/tuning claims are not supported
    return sorted(stats.values(), key=lambda s: s["rule"])


EVIDENCE_KINDS = {"snapshot": "annotated snapshot (software overlay)", "snapshot_raw": "original frame (no overlay)",
                  "clip": "clip (software overlay)", "crop": "crop"}
DERIVED = "anonymised copy (derived)"


SIGN_DOMAIN = "argonix-evidence-export-v1"


def sign_head(key: bytes, head: str) -> dict:
    """HMAC-SHA256 over the manifest chain head, with a domain tag so it can never be mistaken for a session MAC."""
    return {"alg": "HMAC-SHA256", "domain": SIGN_DOMAIN, "key_id": hashlib.sha256(key).hexdigest()[:16],
            "value": hmac.new(key, f"{SIGN_DOMAIN}|{head}".encode(), hashlib.sha256).hexdigest()}


def export_bundle(alert: dict, audit: list[dict], data_dir: Path, anonymise: bool = False,
                  audit_chain: Optional[dict] = None, face_model: Optional[Path] = None,
                  key: Optional[bytes] = None) -> bytes:
    """Zip with alert JSON, evidence files, audit trail and a SHA-256 hash-chained manifest.

    `anonymise`: also add face/head-blurred copies of the snapshots (and of frames sampled from the clips when a face
    detector is available) under anonymised/, labelled as derived copies; the originals stay in the bundle as the
    evidence record. `audit_chain`: the audit log's chain head at export time (Store.audit_head()), so a later
    truncation of the audit table can be detected against this bundle. `key`: HMAC-sign the chain head (the chain
    alone only shows accidental damage: whoever edits a file can recompute it; the signature needs the server key).
    Referenced evidence that is missing or outside `data_dir` is listed under "missing", never silently dropped."""
    buf = io.BytesIO()
    entries, prev = [], "0" * 64
    files = [("alert.json", json.dumps(alert, indent=2).encode(), {"kind": "alert record"}),
             ("audit.json", json.dumps(audit, indent=2).encode(), {"kind": "audit trail (this alert)"})]
    seen: set[str] = set()
    missing: list[dict] = []
    root = Path(data_dir).resolve()
    anon_src = []  # (key, name, data, person_boxes, abs path)
    for i, e in enumerate(alert["events"]):
        ev = e.get("evidence") or {}
        for k, kind in EVIDENCE_KINDS.items():
            rel = ev.get(k)
            if not rel:
                continue
            f = (root / rel).resolve()
            if root not in f.parents or not f.is_file():  # not written (yet), deleted, or outside the data folder
                missing.append({"event": i, "rule": e.get("rule"), "kind": kind, "path": str(rel)})
                continue
            if f.name in seen:  # several events can point at one file: store it once
                continue
            name = f"evidence/{f.name}"
            seen.add(f.name)
            data = f.read_bytes()
            files.append((name, data, {"kind": kind, "camera": e.get("camera_id"), "frame_time": ev.get("frame_time")}))
            anon_src.append((k, name, data, ev.get("person_boxes"), f))
    anon_info = None
    if anonymise:
        from .privacy import anonymise_clip_frames, anonymise_jpeg, get_detector
        det = get_detector(face_model)
        anon_info = {"face_detector": det.method, "head_region_from_person_boxes": True, "notes": []}
        if not det.available:
            anon_info["notes"].append("no face detector in this OpenCV build and no local YuNet model: only the head "
                                      "region of recorded person boxes is blurred; clip frames are not anonymised")
        for k, name, data, boxes, path in anon_src:
            if k in ("snapshot", "snapshot_raw"):
                out, info = anonymise_jpeg(data, boxes, det)
                if out is not None:
                    files.append((f"anonymised/{Path(name).name}", out,
                                  {"kind": DERIVED, "derived_from": name, "anonymisation": info}))
            elif k == "clip":
                for i, (out, info) in enumerate(anonymise_clip_frames(path, 6, det)):
                    files.append((f"anonymised/{Path(name).stem}_f{i}.jpg", out,
                                  {"kind": DERIVED, "derived_from": name, "anonymisation": info}))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data, meta in files:
            h = hashlib.sha256(data).hexdigest()
            link = hashlib.sha256((prev + h + name).encode()).hexdigest()
            entries.append({"file": name, "sha256": h, "bytes": len(data), "chain": link, **meta})
            prev = link
            z.writestr(name, data)
        manifest = {"alert_id": alert["id"], "exported_at": time.time(), "algorithm": "sha256",
                    "chain_rule": "chain_i = sha256(chain_{i-1} + sha256(file_i) + name_i), chain_0 = 64 zeros",
                    "entries": entries, "head": prev,
                    "note": "Files of kind 'anonymised copy (derived)' are blurred derivatives made at export time; "
                            "the evidence record is the original files they were derived from."}
        if anon_info is not None:
            manifest["anonymisation"] = anon_info
        if audit_chain is not None:
            manifest["audit_chain"] = audit_chain
        manifest["missing"] = missing
        if key:
            manifest["signature"] = sign_head(key, prev)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue()


def verify_bundle(data: bytes, key: Optional[bytes] = None) -> bool:
    """True when every file matches its hash, the chain links up to the head and no unlisted file was added.
    With `key`, the manifest's signature over the head must also be valid (so a recomputed chain is caught)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        m = json.loads(z.read("manifest.json"))
        if set(z.namelist()) - {"manifest.json"} != {e["file"] for e in m["entries"]}:
            return False
        prev = "0" * 64
        for e in m["entries"]:
            h = hashlib.sha256(z.read(e["file"])).hexdigest()
            if h != e["sha256"]:
                return False
            prev = hashlib.sha256((prev + h + e["file"]).encode()).hexdigest()
            if prev != e["chain"]:
                return False
        if prev != m["head"]:
            return False
        if key is not None:
            sig = m.get("signature") or {}
            return hmac.compare_digest(str(sig.get("value", "")), sign_head(key, prev)["value"])
        return True
