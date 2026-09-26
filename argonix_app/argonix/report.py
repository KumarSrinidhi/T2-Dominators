"""One-click printable incident report (and the shift-handover print view).

`incident_report()` renders ONE self-contained HTML page: inline CSS, evidence thumbnails and the site map embedded
as data: URIs, print CSS for A4 (the browser's "Print -> Save as PDF" gives the PDF; no PDF library is needed).

Content follows AGENTS.md: observation (what a rule saw) and derivation (why it fired) are separate columns,
confidences are labelled heuristic/uncalibrated, VLM text is labelled assistive and never evidence, and nothing
labels a person. Evidence snapshots are annotated frames (Argus draws boxes and zones into them).

Integrity: the report body sits between BODY_START and BODY_END. Its SHA-256 (UTF-8 bytes from the first character
of BODY_START to the last character of BODY_END) is printed in the footer, returned in the X-Report-SHA256 header
and written to the audit log, so a saved copy can be checked with `verify_report()`. The body lists the SHA-256 of
every evidence file it refers to, as stored on disk (thumbnails are re-encoded for size; the hashes are of the
originals, which the evidence export bundle contains byte for byte).
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any, Optional

from .correlate import explain_priority
from .models import Event
from .rules import LABEL

REPORT_VERSION = "incident-report-v1"
BODY_START = "<!--ARGONIX-REPORT-BODY-START-->"
BODY_END = "<!--ARGONIX-REPORT-BODY-END-->"
MAX_ALERTS = 200
MAX_THUMBS = 12
MAX_TIMELINE_ROWS = 400
PRIO_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

PRINT_JS = "document.getElementById('ax-print').addEventListener('click',function(){window.print();});"
PRINT_JS_HASH = base64.b64encode(hashlib.sha256(PRINT_JS.encode()).digest()).decode()
# The report is served with its own CSP: no network access at all, only the inline print button script (by hash).
CSP = (f"default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{PRINT_JS_HASH}'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

CSS = """
:root{--ink:#16181d;--mute:#5b6270;--line:#d7dbe2;--soft:#f3f5f8;--p1:#c62828;--p2:#e65100;--p3:#b58900;--p4:#546e7a;
--obs:#1b5e20;--inf:#6a1b9a;--asst:#0d47a1}
*{box-sizing:border-box}
html{background:#e9ecf1}
body{margin:0 auto;max-width:210mm;background:#fff;color:var(--ink);font:10.5pt/1.45 "Segoe UI",system-ui,-apple-system,
Roboto,Arial,sans-serif;padding:14mm 14mm 10mm}
h1{font-size:19pt;margin:0 0 2px}h2{font-size:12.5pt;margin:18px 0 6px;padding-bottom:3px;border-bottom:2px solid var(--ink)}
h3{font-size:11pt;margin:12px 0 4px}
p{margin:4px 0}
.sub{color:var(--mute);margin:0 0 10px}
.toolbar{position:sticky;top:0;display:flex;gap:10px;align-items:center;background:#fffbe6;border:1px solid #f0d27a;
padding:6px 10px;margin:-8mm -6mm 10px;border-radius:6px;font-size:9.5pt}
.toolbar button{font:inherit;white-space:nowrap;padding:4px 12px;border-radius:5px;border:1px solid #888;background:#fff;cursor:pointer}
.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 14px;border:1px solid var(--line);border-radius:6px;
padding:8px 10px;background:var(--soft)}
.meta div{min-width:0}.meta b{display:block;font-size:8pt;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;margin:4px 0;font-size:9pt}
th,td{border:1px solid var(--line);padding:3px 5px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
table.tl{table-layout:fixed}
th{background:var(--soft);font-weight:600}
thead{display:table-header-group}
tr{break-inside:avoid;page-break-inside:avoid}
.mono{font-family:Consolas,"Cascadia Mono",Menlo,monospace;font-size:8pt;overflow-wrap:anywhere}
.chip{display:inline-block;padding:0 6px;border-radius:9px;color:#fff;font-weight:700;font-size:8.5pt}
.P1{background:var(--p1)}.P2{background:var(--p2)}.P3{background:var(--p3)}.P4{background:var(--p4)}
.tag{display:inline-block;padding:0 5px;border-radius:4px;font-size:7.5pt;font-weight:700;letter-spacing:.03em;
text-transform:uppercase;border:1px solid currentColor}
.nowrap{white-space:nowrap}
.obs{color:var(--obs)}.inf{color:var(--inf)}.asst{color:var(--asst)}.mute{color:var(--mute)}
.box{border:1px solid var(--line);border-left:4px solid var(--mute);border-radius:4px;padding:6px 9px;margin:6px 0;
break-inside:avoid}
.box.asst{border-left-color:var(--asst);color:var(--ink)}.box.op{border-left-color:var(--obs)}
.note{font-size:8.5pt;color:var(--mute)}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
figure{margin:0;border:1px solid var(--line);border-radius:4px;padding:4px;break-inside:avoid;page-break-inside:avoid}
figure img{width:100%;display:block;background:#111}
figcaption{font-size:8pt;color:var(--mute);margin-top:3px;overflow-wrap:anywhere}
.map{text-align:center;break-inside:avoid}.map img{max-width:100%;max-height:150mm;border:1px solid var(--line)}
footer.integrity{margin-top:14px;border-top:2px solid var(--ink);padding-top:6px;font-size:8.5pt;color:var(--mute)}
footer.integrity .mono{color:var(--ink);font-size:8.5pt}
.kv td:first-child{width:34%;color:var(--mute)}
@page{size:A4;margin:12mm 11mm 14mm}
@media print{html{background:#fff}body{max-width:none;padding:0;font-size:9.5pt}.toolbar{display:none}
h2{break-after:avoid;page-break-after:avoid}h3{break-after:avoid}.section{break-inside:auto}
a{color:inherit;text-decoration:none}}
"""


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------
def esc(v: Any) -> str:
    return html.escape("—" if v is None or v == "" else str(v), quote=True)


def fmt_t(t: Optional[float], with_date: bool = True) -> str:
    if not t:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S", time.localtime(float(t)))


def tz_label() -> str:
    return f"local time, UTC{time.strftime('%z')}"


def chip(p: Optional[str]) -> str:
    return f'<span class="chip {esc(p) if p in PRIO_RANK else "P4"}">{esc(p)}</span>'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def thumb_data_uri(path: Path, max_side: int = 520, quality: int = 80) -> Optional[str]:
    """A downscaled JPEG of an evidence image as a data: URI (None if unreadable)."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception:  # noqa: BLE001 - a broken evidence file must not break the report
        return None


def to_event(d: dict) -> Event:
    names = {f.name for f in dataclasses.fields(Event)}
    kw = {k: v for k, v in d.items() if k in names}
    kw.setdefault("derivation", {})
    kw.setdefault("criticality", 1)
    return Event(**kw)


def priority_factors(alert: dict) -> tuple[dict, str]:
    """explain_priority() recomputed from the alert's stored events; the stored factors if that is impossible."""
    try:
        _, _, f = explain_priority([to_event(e) for e in alert.get("events") or []])
        return f, "recomputed from the stored events with the correlator's explain_priority()"
    except Exception:  # noqa: BLE001 - old/partial alert records: show what was stored at alert time
        return dict(alert.get("factors") or {}), "as stored with the alert"


def deciding_alert(alerts: list[dict]) -> Optional[dict]:
    """The alert that sets the incident priority (best priority, then highest score)."""
    if not alerts:
        return None
    return min(alerts, key=lambda a: (PRIO_RANK.get(a.get("priority"), 9), -float(a.get("score") or 0)))


def derivation_text(d: dict) -> str:
    d = {k: v for k, v in (d or {}).items() if k not in ("rule",)}
    if not d:
        return "—"
    parts = []
    for k, v in d.items():
        s = json.dumps(v, separators=(",", ":"), default=str) if isinstance(v, (dict, list)) else str(v)
        parts.append(f"{k}={s}")
    out = "; ".join(parts)
    return out if len(out) <= 360 else out[:357] + "…"


def confidence_text(e: dict) -> str:
    c = e.get("confidence")
    c = f"{float(c):.2f}" if isinstance(c, (int, float)) else "—"
    kind = "learned model, uncalibrated" if (e.get("derivation") or {}).get("learned_model") else "heuristic, uncalibrated"
    return f"{c} <span class=\"note\">({kind})</span>"


def page(title: str, body: str, subtitle: str = "") -> tuple[str, str]:
    """Wrap a body in the self-contained page; returns (html, sha256 of the marked body)."""
    marked = BODY_START + body + BODY_END
    digest = hashlib.sha256(marked.encode("utf-8")).hexdigest()
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
            "<div class=\"toolbar\"><button id=\"ax-print\" type=\"button\">Print / Save as PDF</button>"
            "<span>Use the browser's print dialog and choose <b>Save as PDF</b> (A4). This bar is not printed."
            f"{(' ' + esc(subtitle)) if subtitle else ''}</span></div>"
            f"{marked}"
            "<footer class=\"integrity\"><div><b>Integrity.</b> SHA-256 of this report body (UTF-8, from the "
            f"<span class=\"mono\">{esc(BODY_START)}</span> marker through the "
            f"<span class=\"mono\">{esc(BODY_END)}</span> marker, inclusive):</div>"
            f"<div class=\"mono\">{digest}</div>"
            "<div>The same value is recorded in the Argus audit log for this report. Per-file evidence hashes are "
            "listed in the body above.</div></footer>"
            f"<script>{PRINT_JS}</script></body></html>"), digest


def verify_report(doc: str) -> Optional[bool]:
    """True if the footer hash matches the body of a saved report; None if the markers are missing."""
    a, b = doc.find(BODY_START), doc.find(BODY_END)
    if a < 0 or b < 0:
        return None
    marked = doc[a:b + len(BODY_END)]
    digest = hashlib.sha256(marked.encode("utf-8")).hexdigest()
    return digest in doc[b + len(BODY_END):]


# ---------------------------------------------------------------------------------------------------------------
# site map with the incident marked
# ---------------------------------------------------------------------------------------------------------------
def incident_map(site, alerts: list[dict]) -> tuple[Optional[str], str]:
    """(data URI of the site map with the incident's zones and event positions drawn, note)."""
    smap = getattr(site, "site_map", None)
    if smap is None:
        return None, "The bird's-eye site map is not available on this server."
    try:
        import cv2
        import numpy as np

        from .birdseye import site_xy
        lay = smap.layout()
        if not any(v.get("fov_site") for v in lay.values()):
            return None, "No camera has a ground-plane calibration, so there is no site map."
        png, b = smap.background_png()
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        ppm = smap.ppm(b)

        def px(x: float, y: float) -> tuple[int, int]:
            return int(round((x - b[0]) * ppm)), int(round((b[3] - y) * ppm))

        zone_ids = {(e.get("camera_id"), e.get("zone_id")) for a in alerts for e in a.get("events") or []}
        overlay = img.copy()
        drawn_zones, drawn_pts = 0, 0
        for w in getattr(site, "workers", []):
            placement = (lay.get(w.id) or {}).get("placement")
            if not placement:
                continue
            for z in getattr(w, "zones", []):
                if (w.id, z.id) not in zone_ids or not z.polygon:
                    continue
                poly = np.array([px(*site_xy(w, tuple(p), placement)) for p in z.polygon], np.int32)
                cv2.fillPoly(overlay, [poly], (40, 40, 220))
                cv2.polylines(img, [poly], True, (40, 40, 255), 2, cv2.LINE_AA)
                c = poly.mean(axis=0).astype(int)
                cv2.putText(img, z.name[:28], (int(c[0]) - 30, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(img, z.name[:28], (int(c[0]) - 30, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                drawn_zones += 1
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, dst=img)
        for a in alerts:
            for e in a.get("events") or []:
                g = e.get("ground")
                w = site.worker(e.get("camera_id")) if hasattr(site, "worker") else None
                placement = (lay.get(e.get("camera_id")) or {}).get("placement")
                if not g or w is None or getattr(w, "H", None) is None or not placement:
                    continue
                p = px(*site_xy(w, (float(g[0]), float(g[1])), placement))
                cv2.circle(img, p, 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, p, 4, (0, 220, 255), -1, cv2.LINE_AA)
                drawn_pts += 1
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return None, "The site map could not be encoded."
        note = (f"Red: the incident's zones ({drawn_zones}); yellow dots: event positions ({drawn_pts}) as ground "
                "foot points. 2D ground-plane (homography) approximation with each camera's calibration error; "
                "not a 3D model. Grid: 1 m / 5 m.")
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), note
    except Exception as e:  # noqa: BLE001 - the map is optional; the report must still render
        return None, f"The site map could not be drawn ({type(e).__name__})."


# ---------------------------------------------------------------------------------------------------------------
# incident report
# ---------------------------------------------------------------------------------------------------------------
def incident_report(site, inc: dict, actor: str, now: Optional[float] = None) -> tuple[str, dict]:
    """(html, meta) for one incident. meta: sha256, evidence hashes, counts (for the audit record)."""
    now = time.time() if now is None else float(now)
    store, data_dir = site.store, Path(site.data)
    names = {w.id: w.name for w in getattr(site, "workers", [])}
    cam = lambda c: names.get(c, c)  # noqa: E731
    alerts = [a for a in (store.get_alert(aid) for aid in inc.get("alert_ids", [])[-MAX_ALERTS:]) if a]
    alerts.sort(key=lambda a: float(a.get("created_at") or 0))
    site_name = site.cfg.get("site") if isinstance(site.cfg.get("site"), str) else (site.cfg.get("site") or {}).get("name")
    out: list[str] = []

    # ---- header -------------------------------------------------------------------------------------------------
    out.append(f"<h1>Incident report — {esc(inc['id'])}</h1><p class=\"sub\">{esc(inc.get('title'))}</p>")
    closed = f"{fmt_t(inc.get('closed_at'))} ({esc(inc.get('closed_by'))})" if inc.get("closed_at") else "—"
    meta = [("Site", esc(site_name)), ("Incident", esc(inc["id"])),
            ("Kind", "camera integrity" if inc.get("kind") == "integrity" else "behaviour"),
            ("Status / phase", f"{esc(inc.get('status'))} / {esc(inc.get('phase'))}"),
            ("Priority", chip(inc.get("priority"))), ("Assignee", esc(inc.get("assignee"))),
            ("Opened", fmt_t(inc.get("opened_at"))), ("Last event", fmt_t(inc.get("last_event_at"))),
            ("Closed", closed),
            ("Alerts / events", f"{len(inc.get('alert_ids', []))} / {esc(inc.get('n_events'))}"),
            ("Generated by", esc(actor)), ("Generated at", f"{fmt_t(now)} <span class=\"note\">({tz_label()})</span>")]
    if inc.get("merged_into"):
        meta.append(("Merged into", esc(inc["merged_into"])))
    if inc.get("merged_from"):
        meta.append(("Merged from", esc(", ".join(inc["merged_from"]))))
    if inc.get("split_from"):
        meta.append(("Split from", esc(inc["split_from"])))
    out.append("<div class=\"meta\">" + "".join(f"<div><b>{k}</b>{v}</div>" for k, v in meta) + "</div>")

    # ---- summary ------------------------------------------------------------------------------------------------
    out.append("<h2>Summary</h2>")
    out.append(f"<div class=\"box\"><span class=\"tag obs\">automatic</span> {esc(inc.get('auto_summary'))}</div>")
    if inc.get("summary"):
        out.append(f"<div class=\"box op\"><span class=\"tag obs\">operator</span> {esc(inc['summary'])}</div>")
    asst = inc.get("assistive_summary") or {}
    if asst.get("text"):
        out.append(f"<div class=\"box asst\"><span class=\"tag asst\">assistive</span> {esc(asst['text'])}"
                   f"<div class=\"note\">{esc(asst.get('label'))} Generated {fmt_t(asst.get('generated_at'))} by "
                   f"{esc(asst.get('generated_by'))}.</div></div>")
    out.append(f"<p class=\"note\">{esc(inc.get('disclaimer'))}</p>")

    # ---- why this priority --------------------------------------------------------------------------------------
    out.append("<h2>Why this priority</h2>")
    top = deciding_alert(alerts)
    if top is None:
        out.append("<p class=\"mute\">No alert records are available for this incident.</p>")
    else:
        f, src = priority_factors(top)
        bands = f.get("bands") or {}
        rows = [("Deciding alert", f"{esc(top['id'])} {chip(top.get('priority'))} — {esc(top.get('title'))}"),
                ("Top event (rule)", esc(LABEL.get(f.get("top_event"), f.get("top_event")))),
                ("Severity", esc(f.get("severity"))), ("Zone criticality", esc(f.get("zone_criticality"))),
                ("Confidence", f"{esc(f.get('confidence'))} <span class=\"note\">(heuristic, uncalibrated)</span>"),
                ("Base score = severity × criticality × confidence", esc(f.get("base_score"))),
                ("Distinct rules / cameras", f"{esc(f.get('distinct_rules'))} / {esc(f.get('distinct_cameras'))}"),
                ("Corroboration multiplier = 1 + 0.25 × (rules + cameras − 2)", esc(f.get("corroboration_multiplier"))),
                ("Score", esc(f.get("score"))),
                ("Priority bands (score ≥)", esc(", ".join(f"{p} ≥ {s:g}" for p, s in bands.items()) if bands else None)),
                ("Learned-model inferences only", "yes" if f.get("learned_model_only") else "no"),
                ("Cap", esc(f.get("cap") or "none"))]
        out.append("<table class=\"kv\"><tbody>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
                   + "</tbody></table>")
        out.append(f"<p class=\"note\">Factors {esc(src)}. The incident takes the best priority of its alerts. "
                   "Learned-model-only alerts are capped at P3 until a deterministic rule agrees.</p>")
        if len(alerts) > 1:
            out.append("<table><thead><tr><th>Alert</th><th>Priority</th><th>Score</th><th>Title</th></tr></thead><tbody>"
                       + "".join(f"<tr><td class=\"mono\">{esc(a['id'])}</td><td>{chip(a.get('priority'))}</td>"
                                 f"<td>{esc(a.get('score'))}</td><td>{esc(a.get('title'))}</td></tr>" for a in alerts)
                       + "</tbody></table>")

    # ---- timeline -----------------------------------------------------------------------------------------------
    out.append("<h2>Timeline of alerts</h2>")
    evs = sorted(((a, e) for a in alerts for e in a.get("events") or []),
                 key=lambda ae: float(ae[1].get("t_start") or ae[1].get("t_end") or 0))
    shown = evs[-MAX_TIMELINE_ROWS:]
    if not shown:
        out.append("<p class=\"mute\">No events.</p>")
    else:
        out.append("<table class=\"tl\"><colgroup><col style=\"width:11%\"><col style=\"width:13%\">"
                   "<col style=\"width:11%\"><col style=\"width:13%\"><col style=\"width:20%\"><col style=\"width:23%\">"
                   "<col style=\"width:9%\"></colgroup><thead><tr><th>Time</th><th>Alert</th><th>Rule</th><th>Camera / zone</th>"
                   "<th><span class=\"tag obs\">observation</span> what was seen</th>"
                   "<th><span class=\"tag inf\">derivation</span> why it fired</th><th>Confidence</th></tr></thead><tbody>")
        for a, e in shown:
            learned = bool((e.get("derivation") or {}).get("learned_model"))
            rule = esc(LABEL.get(e.get("rule"), e.get("rule"))) + (" <span class=\"tag inf\">learned</span>" if learned else "")
            out.append(f"<tr><td>{fmt_t(e.get('t_start'), False)}<br>–{fmt_t(e.get('t_end'), False)}</td>"
                       f"<td class=\"mono\">{esc(a['id'])}</td><td>{rule}</td>"
                       f"<td>{esc(cam(e.get('camera_id')))}<br><span class=\"mute\">{esc(e.get('zone_name'))}</span></td>"
                       f"<td>{esc(e.get('observation'))}</td><td class=\"note\">{esc(derivation_text(e.get('derivation')))}</td>"
                       f"<td>{confidence_text(e)}</td></tr>")
        out.append("</tbody></table>")
        if len(evs) > len(shown):
            out.append(f"<p class=\"note\">Showing the latest {len(shown)} of {len(evs)} events.</p>")
        out.append(f"<p class=\"note\">Times in {tz_label()} on {fmt_t(shown[0][1].get('t_start'))[:10]}. "
                   "Observation = what the detector/tracker measured; derivation = the rule parameters and "
                   "thresholds that turned it into an event. Confidence values are heuristic and uncalibrated.</p>")
    lifecycle = inc.get("timeline") or []
    if lifecycle:
        out.append("<h3>Lifecycle log</h3><table><thead><tr><th>Time</th><th>Change</th><th>Details</th>"
                   "<th>By</th></tr></thead><tbody>"
                   + "".join(f"<tr><td class=\"nowrap\">{fmt_t(x.get('t'))}</td><td>{esc(x.get('kind'))}</td>"
                             f"<td>{esc(x.get('text'))}</td><td>{esc(x.get('actor') or 'system')}</td></tr>"
                             for x in lifecycle[-120:])
                   + "</tbody></table>")

    # ---- evidence -----------------------------------------------------------------------------------------------
    out.append("<h2>Evidence</h2>")
    files: list[dict] = []
    seen: set[str] = set()
    for a, e in evs:
        for kind in ("snapshot", "clip", "crop"):
            rel = (e.get("evidence") or {}).get(kind)
            if not rel or rel in seen:
                continue
            seen.add(rel)
            p = (data_dir / rel).resolve()
            ok = p.is_file() and data_dir.resolve() in p.parents
            files.append({"alert_id": a["id"], "camera": e.get("camera_id"), "kind": kind, "file": rel,
                          "t": e.get("t_end"), "exists": ok, "bytes": p.stat().st_size if ok else None,
                          "sha256": sha256_file(p) if ok else None, "path": p if ok else None})
    snaps = [f for f in files if f["kind"] in ("snapshot", "crop") and f["exists"]]
    if len(snaps) > MAX_THUMBS:  # spread the thumbnails over the incident (first, last and evenly between)
        idx = sorted({round(i * (len(snaps) - 1) / (MAX_THUMBS - 1)) for i in range(MAX_THUMBS)})
        snaps = [snaps[i] for i in idx]
    figs = []
    for f in snaps:
        uri = thumb_data_uri(f["path"])
        if uri:
            figs.append(f"<figure><img alt=\"evidence frame {esc(f['alert_id'])}\" src=\"{uri}\"><figcaption>"
                        f"<b>{esc(f['alert_id'])}</b> · {esc(cam(f['camera']))} · {fmt_t(f['t'], False)} · "
                        f"{esc(Path(f['file']).name)}</figcaption></figure>")
    if figs:
        out.append("<div class=\"thumbs\">" + "".join(figs) + "</div>")
        out.append("<p class=\"note\">Snapshots are annotated frames: Argus draws the detection boxes, track ids and "
                   "zones into them. Thumbnails are downscaled copies; the originals and their hashes are listed under "
                   "Integrity.</p>")
    else:
        out.append("<p class=\"mute\">No evidence snapshot is stored for this incident (recording may be off).</p>")

    # ---- site map -----------------------------------------------------------------------------------------------
    out.append("<h2>Site map</h2>")
    uri, note = incident_map(site, alerts)
    if uri:
        out.append(f"<div class=\"map\"><img alt=\"site map with the incident marked\" src=\"{uri}\"></div>")
    out.append(f"<p class=\"note\">{esc(note)}</p>")

    # ---- cameras ------------------------------------------------------------------------------------------------
    out.append("<h2>Involved cameras</h2><table><thead><tr><th>Camera</th><th>Name</th><th>Health now</th>"
               "<th>Calibration</th><th>Events in incident</th></tr></thead><tbody>")
    per_cam: dict[str, int] = {}
    for _, e in evs:
        per_cam[e.get("camera_id")] = per_cam.get(e.get("camera_id"), 0) + 1
    for c in inc.get("cameras") or sorted(per_cam):
        w = site.worker(c) if hasattr(site, "worker") else None
        if w is not None:
            hs = getattr(getattr(w, "health", None), "state", None)
            reasons = sorted(getattr(getattr(w, "health", None), "active", []) or [])
            health = f"{esc(hs)}{(' (' + esc(', '.join(reasons)) + ')') if reasons else ''}"
            ci = getattr(w, "calib_info", {}) or {}
            err = ci.get("calib_error_m")
            calib = ("ground plane" + (f", ±{float(err):.2f} m" if isinstance(err, (int, float)) else "")
                     if getattr(w, "H", None) is not None else "uncalibrated (pixel space)")
        else:
            health, calib = "not configured now", "—"
        out.append(f"<tr><td class=\"mono\">{esc(c)}</td><td>{esc(cam(c))}</td><td>{health}</td><td>{esc(calib)}</td>"
                   f"<td>{per_cam.get(c, 0)}</td></tr>")
    out.append("</tbody></table>")

    # ---- dispositions -------------------------------------------------------------------------------------------
    out.append("<h2>Dispositions</h2>")
    if alerts:
        out.append("<table><thead><tr><th>Alert</th><th>Status</th><th>Disposition</th><th>Operator note</th></tr>"
                   "</thead><tbody>" + "".join(
                       f"<tr><td class=\"mono\">{esc(a['id'])}</td><td>{esc(a.get('status'))}</td>"
                       f"<td>{esc(a.get('disposition'))}</td><td>{esc(a.get('operator_note'))}</td></tr>" for a in alerts)
                   + "</tbody></table>")
    else:
        out.append("<p class=\"mute\">—</p>")
    pushes = store.list_notifications(inc["id"], 30) if hasattr(store, "list_notifications") else []
    if pushes:
        out.append("<h3>Push decisions</h3><table><thead><tr><th>Time</th><th>Priority</th><th>Decision</th>"
                   "<th>Reason</th></tr></thead><tbody>" + "".join(
                       f"<tr><td class=\"nowrap\">{fmt_t(n.get('t'))}</td><td>{chip(n.get('priority'))}</td>"
                       f"<td>{esc(str(n.get('decision', '')).replace('_', ' '))} ({esc(n.get('trigger'))})</td>"
                       f"<td>{esc(n.get('reason'))}</td></tr>" for n in reversed(pushes))
                   + "</tbody></table><p class=\"note\">Nothing is sent from the console; decisions are recorded.</p>")

    # ---- audit trail --------------------------------------------------------------------------------------------
    out.append("<h2>Audit trail</h2>")
    audit = list(store.list_audit(inc["id"], 300))
    for a in alerts:
        audit += store.list_audit(a["id"], 100)
    audit.sort(key=lambda x: float(x.get("t") or 0))
    if audit:
        out.append("<table><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Subject</th><th>Details</th></tr>"
                   "</thead><tbody>" + "".join(
                       f"<tr><td class=\"nowrap\">{fmt_t(x.get('t'))}</td><td>{esc(x.get('actor'))}</td>"
                       f"<td>{esc(x.get('action'))}</td><td class=\"mono\">{esc(x.get('alert_id'))}</td>"
                       f"<td class=\"note\">{esc(derivation_text(x.get('details') or {}))}</td></tr>" for x in audit[-400:])
                   + "</tbody></table>")
    else:
        out.append("<p class=\"mute\">No operator action has been recorded yet.</p>")

    # ---- assistive text -----------------------------------------------------------------------------------------
    expl = [a for a in alerts if a.get("explanation")]
    if expl:
        out.append("<h2>Assistive descriptions (VLM)</h2>")
        out.extend(f"<div class=\"box asst\"><span class=\"tag asst\">assistive</span> <b class=\"mono\">"
                   f"{esc(a['id'])}</b> {esc(a['explanation'])}</div>" for a in expl)
        out.append("<p class=\"note\">Written by a vision-language model on operator request. Assistive only: it can "
                   "be wrong, it is not evidence and it never triggered, scored or closed an alert.</p>")

    # ---- integrity ----------------------------------------------------------------------------------------------
    out.append("<h2>Integrity</h2>")
    if files:
        out.append("<table><thead><tr><th>Evidence file</th><th>Alert</th><th>Bytes</th><th>SHA-256</th></tr></thead>"
                   "<tbody>" + "".join(
                       f"<tr><td class=\"mono\">{esc(f['file'])}</td><td class=\"mono\">{esc(f['alert_id'])}</td>"
                       f"<td>{esc(f['bytes'])}</td><td class=\"mono\">{esc(f['sha256'] or 'file missing')}</td></tr>"
                       for f in files) + "</tbody></table>")
    else:
        out.append("<p class=\"mute\">No evidence files are referenced.</p>")
    out.append(f"<p class=\"note\">Report format {REPORT_VERSION}. Hashes are of the evidence files as stored on the "
               "Argus server at generation time; the per-alert evidence export (hash-chained zip) contains the same "
               "files. Generating this report is audited.</p>")

    doc, digest = page(f"Incident {inc['id']} — Argus report", "".join(out), f"Incident {inc['id']}.")
    return doc, {"sha256": digest, "report_version": REPORT_VERSION, "alerts": len(alerts), "events": len(evs),
                 "evidence": [{"file": f["file"], "sha256": f["sha256"]} for f in files]}
