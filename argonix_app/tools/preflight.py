"""Pre-demo checklist: PASS / WARN / FAIL for everything a site config needs before `run.py` starts.

    uv run python tools/preflight.py --config configs/real_site.json --data data/run_demo --port 8003
    uv run python tools/preflight.py --url http://127.0.0.1:18000 --vlm-url http://127.0.0.1:18100   # running services
    (on a CUDA host: uv run --no-sync python tools/preflight.py ...)

Checks: Python and torch versions, CUDA, the detector weights the config will load, every camera source (files are
opened and one frame is read; RTSP/HTTP streams are probed with a short timeout; webcams are listed, not opened),
the data directory (writable, free disk), the port, optional Re-ID weights, and TensorRT/engines (info only).
With --url it asks a running app for /api/auth/status; with --vlm-url it asks the VLM server for /health.
Exit code 1 when any check FAILs, else 0. WARN means the demo can run but something is degraded. Stream
credentials are never printed (URLs are redacted).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
PY_MIN, PY_MAX = (3, 11), (3, 13)  # pyproject: requires-python >=3.11,<3.14
STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https")
OFFICIAL_WEIGHTS = re.compile(r"^yolo(v?\d+|11|26)[nslmx](-seg|-pose|-cls|-obb)?\.pt$")
# where each generated clip folder comes from (a missing clip then gets a concrete fix)
GENERATORS = {
    "data/synth/": "run.py generates it on the first start with configs/demo_site.json (or tools/make_synthetic.py)",
    "data/synth_handoff/": "uv run python tools/make_synthetic_handoff.py",
    "data/reid_eval/scene/": "uv run python tools/make_reid_demo.py",
    "data/real/": "uv run python tools/setup_assets.py (the street clip: --only street)",
}


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


# ---------------------------------------------------------------------------------------------- pure helpers
def check_python(version: tuple) -> Check:
    v = tuple(version[:2])
    s = ".".join(map(str, version[:3]))
    if PY_MIN <= v <= PY_MAX:
        return Check(PASS, "python", s)
    return Check(FAIL, "python", f"{s}: this project needs Python {PY_MIN[0]}.{PY_MIN[1]}-{PY_MAX[0]}.{PY_MAX[1]}")


def check_torch(version: Optional[str], cuda_available: bool, device: str, nvidia_gpu: bool) -> Check:
    """version None = torch not importable. device = the detector's configured device ("auto", "cpu", "0", "cuda:0")."""
    if version is None:
        return Check(FAIL, "torch", "not importable: run `uv sync --extra cpu` (or --extra cu126 / cu128 on a GPU host)")
    wants_cuda = device.isdigit() or device.startswith("cuda")
    if cuda_available:
        return Check(PASS, "torch", f"{version}, CUDA available")
    if wants_cuda:
        return Check(FAIL, "torch", f"{version}: the config asks for device '{device}' but CUDA is not available")
    if nvidia_gpu and version.endswith("+cpu"):
        return Check(WARN, "torch", f"{version} (CPU build) on a machine with an NVIDIA GPU: a plain `uv run` may have "
                                    "re-synced the CPU wheel. Run `uv sync --extra cu126` (or cu128) and then "
                                    "`uv run --no-sync ...`")
    return Check(INFO, "torch", f"{version}, CPU only (fine for 2 cameras at about 3-5 fps analytics)")


def classify_source(src) -> str:
    """webcam | stream | placeholder | file | bad_scheme"""
    s = str(src).strip()
    if "${" in s:
        return "placeholder"
    if s.isdigit():
        return "webcam"
    if "://" in s:
        return "stream" if s.split("://", 1)[0].lower() in STREAM_SCHEMES else "bad_scheme"
    return "file"


def resolve_path(p: str, base: Path) -> Path:
    """The same rule Site uses: a relative path is taken from the project root (the config folder's parent)."""
    q = Path(p)
    return q if q.is_absolute() else base / q


def generator_hint(rel: str) -> str:
    r = rel.replace("\\", "/")
    for prefix, hint in GENERATORS.items():
        if r.startswith(prefix):
            return hint
    return "fix the camera's source in the config"


def check_weights(weights, base: Path, cuda: bool, remote: bool = False) -> Check:
    """What the detector backend will load (see argonix/detect.py resolve_weights)."""
    if remote:
        return Check(INFO, "detector weights", "remote inference node: weights live on the node (--weights there)")
    w = str(weights or "auto")
    models = base / "data" / "models"
    if w == "auto":
        stem = "yolo26s" if cuda else "yolo26n"
        p = models / f"{stem}.pt"
        if p.exists():
            return Check(PASS, "detector weights", f"auto -> {p.relative_to(base)}")
        return Check(WARN, "detector weights", f"auto -> {stem}.pt is not in data/models: Ultralytics downloads it "
                                               "at startup (needs internet)")
    p = resolve_path(w, base)
    if p.suffix == ".engine":
        if p.exists():
            return Check(PASS if cuda else WARN, "detector weights",
                         f"{w}" + ("" if cuda else ": TensorRT engine but no CUDA here; the sibling .pt is used"))
        stem = re.sub(r"-\d+-b\d+$", "", p.stem)
        pt = p.with_name(stem + ".pt")
        if pt.exists() or (models / f"{stem}.pt").exists():
            return Check(WARN, "detector weights", f"{w} missing: falls back to {stem}.pt (fp16 on CUDA). "
                                                   "Engines are GPU-specific; rebuild with tools/bench_gpu.py --export")
        return Check(FAIL, "detector weights", f"{w} and its fallback {stem}.pt are both missing")
    if p.exists():
        return Check(PASS, "detector weights", w)
    if OFFICIAL_WEIGHTS.match(p.name):
        return Check(WARN, "detector weights", f"{w} missing: Ultralytics downloads {p.name} by name at startup "
                                               "(needs internet)")
    return Check(FAIL, "detector weights", f"{w} not found")


def check_disk(free_bytes: int, min_free_gb: float) -> Check:
    gb = free_bytes / 2**30
    if gb < min(1.0, min_free_gb):
        return Check(FAIL, "free disk", f"{gb:.1f} GiB free: evidence clips and recordings need space")
    if gb < min_free_gb:
        return Check(WARN, "free disk", f"{gb:.1f} GiB free (below {min_free_gb:g} GiB)")
    return Check(PASS, "free disk", f"{gb:.1f} GiB free")


def port_status(host: str, port: int) -> tuple[bool, str]:
    """(bindable, reason). On Windows, WinError 10013 usually means the port is inside a range reserved by
    Hyper-V/WSL/Docker (`netsh int ipv4 show excludedportrange protocol=tcp`), not that another process has it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: otherwise a second bind can succeed
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind((host, port))
        return True, "free"
    except PermissionError:
        return False, ("not allowed (on Windows: probably a reserved range, see "
                       "`netsh int ipv4 show excludedportrange protocol=tcp`)")
    except OSError as e:
        if getattr(e, "winerror", None) == 10013:
            return False, ("not allowed (on Windows: probably a reserved range, see "
                           "`netsh int ipv4 show excludedportrange protocol=tcp`)")
        return False, "in use: stop the other process or pick another --port"
    finally:
        s.close()


def port_free(host: str, port: int) -> bool:
    return port_status(host, port)[0]


def base_url(url: str) -> str:
    """http://host:port[/path] -> http://host:port (the VLM flag is sometimes given with /explain)."""
    u = urlsplit(url if "://" in url else "http://" + url)
    return f"{u.scheme}://{u.netloc}"


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == FAIL for c in checks) else 0


def summary_line(checks: list[Check]) -> str:
    n = {k: sum(1 for c in checks if c.status == k) for k in (PASS, WARN, FAIL, INFO)}
    verdict = "NOT READY" if n[FAIL] else ("READY with warnings" if n[WARN] else "READY")
    return f"{verdict}: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[INFO]} info"


# ---------------------------------------------------------------------------------------------- probes (I/O)
def check_file_source(cam_id: str, src: str, base: Path) -> Check:
    p = resolve_path(src, base)
    name = f"camera {cam_id}"
    if not p.exists():
        return Check(FAIL, name, f"{src} missing: {generator_hint(src)}")
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            return Check(FAIL, name, f"{src} exists but OpenCV cannot open it")
        ok, frame = cap.read()
        if not ok or frame is None:
            return Check(FAIL, name, f"{src} opened but no frame could be read")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return Check(PASS, name, f"{src} ({frame.shape[1]}x{frame.shape[0]}, {fps:.0f} fps)")
    finally:
        cap.release()


def check_stream_source(cam_id: str, src: str, timeout_s: float) -> Check:
    from argonix.cameras import probe_source, redact
    r = probe_source(src, timeout_s=timeout_s)
    if r.get("ok"):
        return Check(PASS, f"camera {cam_id}", f"{redact(src)} ({r.get('width')}x{r.get('height')}, "
                                               f"{r.get('fps') or 0:.0f} fps, {r.get('elapsed_s')} s)")
    return Check(FAIL, f"camera {cam_id}", f"{redact(src)}: {r.get('error') or 'no frame'} (timeout {timeout_s:g} s)")


def check_camera(c: dict, base: Path, timeout_s: float) -> Check:
    cid, src = c.get("id", "?"), str(c.get("source", ""))
    kind = classify_source(src)
    if kind == "placeholder":
        return Check(FAIL, f"camera {cid}", "source still has a ${...} placeholder: copy the config to "
                                            "configs/<name>.local.json and fill it in")
    if kind == "bad_scheme":
        return Check(FAIL, f"camera {cid}", f"unsupported source scheme (use one of {', '.join(STREAM_SCHEMES)})")
    if kind == "webcam":
        return Check(INFO, f"camera {cid}", f"webcam index {src} (not opened by the preflight)")
    if kind == "stream":
        return check_stream_source(cid, src, timeout_s)
    return check_file_source(cid, src, base)


def check_data_dir(data: Path) -> Check:
    try:
        data.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".preflight-", dir=str(data))
        os.close(fd)
        os.remove(tmp)
        return Check(PASS, "data dir", f"{data} writable")
    except OSError as e:
        return Check(FAIL, "data dir", f"{data} not writable: {e}")


def http_json(url: str, timeout_s: float) -> tuple[Optional[int], Optional[dict], str]:
    import httpx
    try:
        r = httpx.get(url, timeout=timeout_s)
        try:
            body = r.json()
        except ValueError:
            body = None
        return r.status_code, body, ""
    except Exception as e:  # noqa: BLE001 - unreachable, refused, timeout
        return None, None, f"{type(e).__name__}: {e}"[:200]


def check_app(url: str, timeout_s: float, get: Callable = http_json) -> Check:
    code, body, err = get(base_url(url) + "/api/auth/status", timeout_s)
    if code != 200 or not isinstance(body, dict):
        return Check(FAIL, "app", f"{base_url(url)}: " + (err or f"HTTP {code}") +
                     " (is it running? through an ssh -L tunnel, is the tunnel up?)")
    if body.get("open_demo"):
        return Check(WARN, "app", f"{base_url(url)} up in OPEN DEMO mode: no sign-in, everyone on the port is admin")
    if body.get("setup_required"):
        return Check(WARN, "app", f"{base_url(url)} up; first-run setup pending: the setup code is in the app log "
                                  "and in <data>/setup_code.txt")
    return Check(PASS, "app", f"{base_url(url)} up, sign-in enabled")


def check_vlm(url: str, timeout_s: float, get: Callable = http_json) -> Check:
    code, body, err = get(base_url(url) + "/health", timeout_s)
    if code != 200:
        return Check(WARN, "vlm", f"{base_url(url)}: " + (err or f"HTTP {code}") +
                     ": the Explain button and LLM investigation are unavailable (the keyword parser still works)")
    body = body if isinstance(body, dict) else {}
    if body.get("error"):
        return Check(WARN, "vlm", f"{base_url(url)} up but reports an error: {str(body['error'])[:160]}")
    if body.get("loading"):
        return Check(WARN, "vlm", f"{base_url(url)} up, model still loading")
    loaded = body.get("loaded")
    if not loaded:
        return Check(WARN, "vlm", f"{base_url(url)} up but no model is loaded yet (the first Explain loads it)")
    return Check(PASS, "vlm", f"{base_url(url)} up ({str(loaded)[:60]}, {body.get('mode') or '?'}, "
                              f"{body.get('device') or '?'})")


def nvidia_gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def torch_info() -> tuple[Optional[str], bool]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None, False
    try:
        return torch.__version__, bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return torch.__version__, False


def check_tensorrt(base: Path) -> Check:
    engines = sorted(p.name for p in (base / "data" / "models").glob("*.engine")) if (base / "data" / "models").exists() else []
    try:
        import tensorrt  # noqa: F401
        trt = f"tensorrt {getattr(tensorrt, '__version__', '?')} importable"
    except Exception:  # noqa: BLE001
        trt = "tensorrt not installed (optional; the .pt runs in fp16 on CUDA)"
    eng = f"engines: {', '.join(engines)}" if engines else "no engines in data/models"
    return Check(INFO, "tensorrt", f"{trt}; {eng}. Engines are GPU-specific: rebuild them on each GPU")


def check_reid(cfg: dict, base: Path) -> Optional[Check]:
    if not (cfg.get("reid") or {}).get("enabled"):
        return None
    d = base / "data" / "models" / "reid"
    found = sorted(p.name for p in d.glob("*.pth")) if d.exists() else []
    if found:
        return Check(PASS, "re-id weights", ", ".join(found))
    return Check(WARN, "re-id weights", "none in data/models/reid/: the weak hand-crafted embedder is used (labelled "
                                        "as such on Journeys)")


def check_accounts(cfg_path: Path, open_demo: bool) -> Check:
    users = cfg_path.resolve().parent / "users.json"
    try:
        raw = json.loads(users.read_text(encoding="utf-8") or "{}") if users.exists() else {}
        n = sum(1 for r in raw.values() if isinstance(r, dict))  # {username: record} (argonix/auth.py)
    except (OSError, ValueError, AttributeError):
        n = 0
    if n:
        return Check(INFO, "accounts", f"{n} account(s) in {users.name}")
    if open_demo:
        return Check(INFO, "accounts", "none yet; --open-demo: no sign-in until the first admin is created")
    return Check(INFO, "accounts", "none yet: the first start prints a one-time setup code (also written to "
                                   "<data>/setup_code.txt)")


# ---------------------------------------------------------------------------------------------- main
def run_checks(a) -> list[Check]:
    checks = [check_python(sys.version_info)]
    cfg_path = Path(a.config)
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        checks.append(Check(FAIL, "config", f"{cfg_path}: {e}"))
        return checks
    base = cfg_path.resolve().parent.parent
    cams = cfg.get("cameras") or []
    checks.append(Check(PASS if cams else FAIL, "config", f"{cfg_path} ({len(cams)} camera(s))"))
    det = cfg.get("detector") or {}
    device = str(det.get("device", "auto"))
    ver, cuda = torch_info()
    checks.append(check_torch(ver, cuda, device, nvidia_gpu_present()))
    try:
        import ultralytics
        checks.append(Check(PASS, "ultralytics", ultralytics.__version__))
    except Exception as e:  # noqa: BLE001
        checks.append(Check(FAIL, "ultralytics", f"not importable ({type(e).__name__}): run `uv sync`"))
    use_cuda = cuda and device != "cpu"
    checks.append(check_weights(det.get("weights", "auto"), base, use_cuda, remote=bool(det.get("remote"))))
    for c in cams:
        checks.append(check_camera(c, base, a.rtsp_timeout))
    data = Path(a.data)
    checks.append(check_data_dir(data))
    try:
        checks.append(check_disk(shutil.disk_usage(data if data.exists() else Path.cwd()).free, a.min_free_gb))
    except OSError as e:
        checks.append(Check(WARN, "free disk", str(e)))
    if a.url:
        checks.append(Check(INFO, "port", "not checked (--url checks the running app instead)"))
    else:
        ok, why = port_status(a.host, a.port)
        checks.append(Check(PASS if ok else FAIL, "port", f"{a.host}:{a.port} {why}"))
    checks.append(check_accounts(cfg_path, a.open_demo))
    r = check_reid(cfg, base)
    if r:
        checks.append(r)
    if use_cuda or a.tensorrt:
        checks.append(check_tensorrt(base))
    if a.url:
        checks.append(check_app(a.url, a.http_timeout))
    if a.vlm_url:
        checks.append(check_vlm(a.vlm_url, a.http_timeout))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "demo_site.json"))
    ap.add_argument("--data", default="data/run", help="the --data dir run.py will use")
    ap.add_argument("--host", default="0.0.0.0", help="the bind address run.py will use (for the port check)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open-demo", action="store_true", help="you will start run.py with --open-demo")
    ap.add_argument("--url", default="", help="a running app to check, e.g. http://127.0.0.1:18000")
    ap.add_argument("--vlm-url", default="", help="a running VLM server to check, e.g. http://127.0.0.1:8100")
    ap.add_argument("--rtsp-timeout", type=float, default=5.0, help="seconds per network stream probe")
    ap.add_argument("--http-timeout", type=float, default=5.0)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--tensorrt", action="store_true", help="also report TensorRT/engines without CUDA")
    ap.add_argument("--json", action="store_true", help="print the checks as JSON")
    a = ap.parse_args(argv)
    t0 = time.time()
    checks = run_checks(a)
    if a.json:
        print(json.dumps({"checks": [asdict(c) for c in checks], "summary": summary_line(checks),
                          "exit_code": exit_code(checks)}, indent=1))
    else:
        for c in checks:
            print(f"{c.status:4s}  {c.name:22s} {c.detail}")
        print(f"\n{summary_line(checks)} ({time.time() - t0:.1f} s)")
    return exit_code(checks)


if __name__ == "__main__":
    raise SystemExit(main())
