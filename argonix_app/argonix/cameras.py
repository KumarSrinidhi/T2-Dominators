"""Connecting existing CCTV: ONVIF WS-Discovery, vendor RTSP URL templates, source probing and credential redaction.

Pure stdlib + OpenCV. Nothing here talks to a camera unless an operator asks (discover/probe), and nothing here
returns or logs a plaintext password: use `redact()` on every source string that leaves the process.
"""
from __future__ import annotations

import base64
import os
import re
import select
import socket
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote, urlsplit

import cv2

# ---------------------------------------------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------------------------------------------
# scheme://user:password@  -> scheme://user:***@   (greedy up to the last '@' before whitespace/?/#, so a
# password containing '@' or '/' is still fully hidden; over-redaction of odd URLs is the safe failure mode)
_URL_CRED = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)([^\s:/@?#]*):([^\s?#]*)@")
_QUERY_CRED = re.compile(r"(?i)([?&;](?:password|passwd|pwd|pass|token|auth|key|secret)=)[^&#\s]*")
MASK = "***"


def redact(source) -> str:
    """Hide credentials in a source URL (or any text containing URLs): rtsp://user:***@host/..."""
    s = str(source)
    s = _URL_CRED.sub(lambda m: f"{m.group(1)}{m.group(2)}:{MASK}@", s)
    return _QUERY_CRED.sub(lambda m: m.group(1) + MASK, s)


def is_redacted(source) -> bool:
    return MASK in str(source)


def unredact(new: str, old: Optional[str]) -> str:
    """A client that echoes back a redacted source means "unchanged": restore the stored secret.
    Raises ValueError if a masked source does not correspond to the stored one."""
    if not is_redacted(new):
        return new
    if old is not None and redact(old) == new:
        return old
    raise ValueError("source contains a masked password; re-enter the full URL including the password")


def is_live(source) -> bool:
    s = str(source)
    return s.isdigit() or ("://" in s and not s.lower().startswith("file://"))


_ALLOWED_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https")


def validate_source(source) -> str:
    """Accept webcam indices, network streams with a known scheme, or an existing local video file.

    Anything else is rejected, so an API caller cannot hand FFmpeg exotic protocol strings (concat:, subfile:, ...).
    """
    s = str(source).strip()
    if not s:
        raise ValueError("source is empty")
    if s.isdigit():
        return s
    if "://" in s:
        scheme = s.split("://", 1)[0].lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"unsupported source scheme '{scheme}' (use one of {', '.join(_ALLOWED_SCHEMES)})")
        if not urlsplit(s).hostname:
            raise ValueError("source URL has no host")
        return s
    if Path(s).is_file():
        return s
    raise ValueError("source must be an rtsp/http URL, a webcam index, or an existing video file path")


# ---------------------------------------------------------------------------------------------------------------
# Opening streams
# ---------------------------------------------------------------------------------------------------------------
OPEN_TIMEOUT_MS = 8000
READ_TIMEOUT_MS = 8000


def apply_capture_settings(cap: cv2.VideoCapture, capture: Optional[dict]) -> None:
    """Ask a USB/UVC webcam for a mode before the first read, e.g. {"width": 1280, "height": 720, "fps": 30,
    "fourcc": "MJPG"}. Without this OpenCV opens most UVC cameras at 640x480 (an Arducam 16MP gave 640x480 @ 20 fps
    instead of 1280x720 @ 30). The camera may pick the nearest mode it supports; the worker reports what it got.
    Changing the mode after frames have been read crashed OpenCV's MSMF backend on Windows, so only call it here."""
    if not capture:
        return
    fourcc = str(capture.get("fourcc") or "")
    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    for key, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH), ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                      ("fps", cv2.CAP_PROP_FPS)):
        if capture.get(key):
            cap.set(prop, float(capture[key]))


def open_capture(source, capture: Optional[dict] = None) -> cv2.VideoCapture:
    """Open a file / webcam / network stream. RTSP goes over TCP (fewer smeared frames on lossy links) and network
    opens/reads get finite timeouts so a dead camera cannot hang a thread forever. `capture` (webcams only) requests a
    resolution / fps / pixel format, see apply_capture_settings."""
    s = str(source)
    if s.isdigit():
        cap = cv2.VideoCapture(int(s))
        if cap.isOpened():
            apply_capture_settings(cap, capture)
        return cap
    if "://" in s and s.lower().startswith(("rtsp", "rtmp", "http")):
        if s.lower().startswith("rtsp"):
            # must be set before the FFmpeg backend opens the stream; process-wide, and TCP is what we want anyway
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        try:
            return cv2.VideoCapture(s, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
        except (TypeError, cv2.error):  # very old OpenCV without the params overload
            return cv2.VideoCapture(s, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(s)


def probe_source(source, timeout_s: float = 10.0) -> dict:
    """Try to open a source and grab one frame, in a worker thread with a hard timeout.

    Returns {ok, width, height, fps, error, snapshot_jpeg_b64, source (redacted), elapsed_s}. The error text is
    redacted. On timeout the helper thread is abandoned (OpenCV's own open/read timeouts end it later).
    """
    out = {"ok": False, "width": None, "height": None, "fps": None, "error": "", "snapshot_jpeg_b64": None,
           "source": redact(source), "elapsed_s": None}
    try:
        src = validate_source(source)
    except ValueError as e:
        out["error"] = str(e)
        return out
    res: dict = {}
    t0 = time.time()

    def work():
        cap = None
        try:
            cap = open_capture(src)
            if not cap.isOpened():
                res["error"] = "could not open source (wrong URL/path, credentials, port, or camera offline)"
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                res["error"] = "opened but no frame received (stream path or codec may be wrong)"
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            h, w = frame.shape[:2]
            snap = frame
            if w > 640:
                snap = cv2.resize(frame, (640, int(h * 640 / w)))
            jpg = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            res.update(ok=True, width=int(w), height=int(h), fps=round(fps, 2) if 0 < fps <= 240 else None,
                       snapshot_jpeg_b64=base64.b64encode(jpg).decode())
        except Exception as e:  # noqa: BLE001 - report any backend failure to the operator
            res["error"] = f"{type(e).__name__}: {e}"
        finally:
            if cap is not None:
                cap.release()

    th = threading.Thread(target=work, daemon=True, name="probe-source")
    th.start()
    th.join(timeout_s)
    out["elapsed_s"] = round(time.time() - t0, 2)
    if th.is_alive():
        out["error"] = f"timed out after {timeout_s:.0f}s (host unreachable, firewall, or wrong port?)"
        return out
    out.update({k: v for k, v in res.items() if k != "error"})
    out["error"] = redact(res.get("error", "").replace(src, redact(src)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Vendor RTSP templates
# ---------------------------------------------------------------------------------------------------------------
TEMPLATE_DISCLAIMER = ("Common factory-default RTSP paths collected from vendor documentation and community notes. "
                       "They are starting points to VERIFY with 'Test', not guarantees: firmware, model line, NVR vs "
                       "camera, and site configuration all change them. Prefer the sub-stream for analytics.")

# placeholders: {user} {password} {host} {port} {channel} {channel2} (two-digit channel)
TEMPLATES: list[dict] = [
    {"id": "hikvision", "vendor": "Hikvision (and many OEM rebrands)", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{channel}01",
     "sub": "rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{channel}02",
     "notes": "Channel 1 gives .../101 (main) and .../102 (sub). On an NVR, camera N is N01/N02."},
    {"id": "dahua", "vendor": "Dahua / Amcrest / Lorex (Dahua OEM)", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype=0",
     "sub": "rtsp://{user}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype=1",
     "notes": "subtype=0 main, 1 sub (2 = third stream on some models). NVR channels are 1-based."},
    {"id": "axis", "vendor": "Axis", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/axis-media/media.amp",
     "sub": "rtsp://{user}:{password}@{host}:{port}/axis-media/media.amp?resolution=640x360",
     "notes": "Axis has no fixed sub-stream; request a lower resolution or a stream profile "
              "(?streamprofile=<name>). Multi-channel encoders add ?camera=N."},
    {"id": "reolink", "vendor": "Reolink", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/h264Preview_{channel2}_main",
     "sub": "rtsp://{user}:{password}@{host}:{port}/h264Preview_{channel2}_sub",
     "notes": "RTSP may need enabling in the Reolink app. Some H.265 models use /Preview_01_main instead."},
    {"id": "uniview", "vendor": "Uniview (UNV)", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/unicast/c{channel}/s0/live",
     "sub": "rtsp://{user}:{password}@{host}:{port}/unicast/c{channel}/s1/live",
     "notes": "s0 main, s1 sub. Older firmware: /media/video1 (main), /media/video2 (sub)."},
    {"id": "tapo", "vendor": "TP-Link Tapo / VIGI", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/stream1",
     "sub": "rtsp://{user}:{password}@{host}:{port}/stream2",
     "notes": "Tapo needs a separate 'camera account' created in the Tapo app; it is not your TP-Link login."},
    {"id": "onvif", "vendor": "Generic ONVIF (other brands)", "port": 554,
     "main": "rtsp://{user}:{password}@{host}:{port}/",
     "sub": "rtsp://{user}:{password}@{host}:{port}/",
     "notes": "No universal path. Use 'Discover' to find the device, then read the stream URI from the camera's "
              "web UI, the vendor manual, or ONVIF Device Manager (Media > GetStreamUri), and paste it here."},
]


def templates() -> dict:
    return {"disclaimer": TEMPLATE_DISCLAIMER, "verified": False, "templates": TEMPLATES}


def build_url(template_id: str, host: str, user: str = "admin", password: str = "", channel: int = 1,
              stream: str = "sub", port: Optional[int] = None) -> str:
    """Fill a vendor template. Credentials are percent-encoded so ':' '@' '/' in passwords survive."""
    t = next((t for t in TEMPLATES if t["id"] == template_id), None)
    if t is None:
        raise ValueError(f"unknown template '{template_id}'")
    if stream not in ("main", "sub"):
        raise ValueError("stream must be 'main' or 'sub'")
    url = t[stream].format(user=quote(user, safe=""), password=quote(password, safe=""), host=host,
                           port=int(port or t["port"]), channel=int(channel), channel2=f"{int(channel):02d}")
    if not password:
        url = url.replace(":@", "@")
    if not user and not password:
        url = url.replace("://@", "://")
    return url


# ---------------------------------------------------------------------------------------------------------------
# ONVIF WS-Discovery
# ---------------------------------------------------------------------------------------------------------------
WSD_ADDR = ("239.255.255.250", 3702)
PROBE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header>
<w:MessageID>uuid:{msg_id}</w:MessageID>
<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
</e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""

_VENDOR_HINTS = [("hikvision", "hikvision"), ("hikvison", "hikvision"), ("dahua", "dahua"), ("amcrest", "dahua"),
                 ("lorex", "dahua"), ("axis", "axis"), ("reolink", "reolink"), ("uniview", "uniview"),
                 ("unv", "uniview"), ("tapo", "tapo"), ("tp-link", "tapo"), ("vigi", "tapo")]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag.split(":")[-1]


def _scope_values(scopes: list[str], key: str) -> list[str]:
    pre = f"onvif://www.onvif.org/{key}/"
    return [unquote(s[len(pre):]) for s in scopes if s.lower().startswith(pre)]


def parse_probe_matches(data: bytes | str) -> list[dict]:
    """Parse a WS-Discovery ProbeMatches SOAP message into device dicts (namespace-prefix agnostic)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    if len(data) > 65536:
        return []
    if "<!DOCTYPE" in data.upper() or "<!ENTITY" in data.upper():
        return []  # SOAP replies never carry a DTD; refusing one rules out entity-expansion tricks from the network
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    out = []
    for pm in root.iter():
        if _local(pm.tag) != "ProbeMatch":
            continue
        f = {"endpoint": "", "types": [], "scopes": [], "xaddrs": [], "metadata_version": None}
        for el in pm.iter():
            name, text = _local(el.tag), (el.text or "").strip()
            if name == "Address" and not f["endpoint"]:
                f["endpoint"] = text
            elif name == "Types":
                f["types"] = text.split()
            elif name == "Scopes":
                f["scopes"] = text.split()
            elif name == "XAddrs":
                f["xaddrs"] = text.split()
            elif name == "MetadataVersion":
                f["metadata_version"] = text
        hosts = [urlsplit(x).hostname for x in f["xaddrs"]]
        names = _scope_values(f["scopes"], "name")
        hardware = _scope_values(f["scopes"], "hardware")
        blob = " ".join(names + hardware + f["scopes"]).lower()
        f.update(host=next((h for h in hosts if h), None), name=names[0] if names else None,
                 hardware=hardware[0] if hardware else None,
                 location=(_scope_values(f["scopes"], "location") or [None])[0],
                 vendor_guess=next((v for k, v in _VENDOR_HINTS if k in blob), None))
        out.append(f)
    return out


def _interface_ips() -> list[str]:
    ips = []
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                    ips.append(a.address)
    except Exception:  # noqa: BLE001 - discovery must never crash the API
        pass
    return ips or [""]


def discover(timeout_s: float = 3.0) -> list[dict]:
    """Multicast an ONVIF Probe for NetworkVideoTransmitter on every IPv4 interface and collect ProbeMatches.

    Only finds ONVIF devices on the same L2 segment/VLAN that have discovery enabled; routers do not forward it.
    """
    msg = PROBE_TEMPLATE.format(msg_id=uuid.uuid4()).encode()
    socks = []
    for ip in _interface_ips():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            if ip:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                s.bind((ip, 0))
            s.setblocking(False)
            s.sendto(msg, WSD_ADDR)
            socks.append(s)
        except OSError:
            continue
    found: dict[str, dict] = {}
    deadline = time.time() + timeout_s
    try:
        while socks and time.time() < deadline:
            ready, _, _ = select.select(socks, [], [], max(0.0, deadline - time.time()))
            for s in ready:
                try:
                    data, addr = s.recvfrom(65535)
                except OSError:
                    continue
                for dev in parse_probe_matches(data):
                    dev["responder"] = addr[0]
                    dev["host"] = dev["host"] or addr[0]
                    found.setdefault(dev["endpoint"] or " ".join(dev["xaddrs"]) or addr[0], dev)
    finally:
        for s in socks:
            s.close()
    return list(found.values())
