"""Run the Argus site + dashboard.

    uv run python run.py [--config configs/demo_site.json] [--port 8000] [--open-demo]
                         [--tls-cert cert.pem --tls-key key.pem] [--idle-timeout-min 30]

First start without accounts: the dashboard asks for the first admin account (a one-time setup code is printed
below and written to <data>/setup_code.txt; the browser on this machine does not need it). --open-demo skips sign-in
entirely until the first account exists (synthetic demo only; clearly labelled in the UI) and then listens on
127.0.0.1 only, unless --host names a network address.
"""
import argparse
import os
from pathlib import Path

import uvicorn

from argonix.hosttune import fine_timers, prefer_performance_cores
from argonix.pipeline import Site
from argonix.server import create_app

ROOT = Path(__file__).parent

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def bind_host(requested, open_now: bool) -> tuple[str, str]:
    """The address to listen on, and a warning to print (or "").

    Open demo (no sign-in) stays on this computer unless a network address is asked for explicitly; a server that
    asks for sign-in listens on every interface by default."""
    if requested is None:
        return ("127.0.0.1" if open_now else "0.0.0.0"), ""
    if open_now and requested not in LOCAL_HOSTS:
        return requested, (f"WARNING: open demo on {requested}: anyone who can reach this port gets full access "
                           "(live video, cameras, evidence) until the first admin account is created. "
                           "Trusted networks only.")
    return requested, ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "demo_site.json"))
    ap.add_argument("--data", default=str(ROOT / "data" / "run"))
    ap.add_argument("--host", default=None,
                    help="bind address: 0.0.0.0 (all networks) by default, 127.0.0.1 with --open-demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vlm-url", default="", help="optional VLM server, e.g. http://gpu-host:8100 (explain + /chat)")
    ap.add_argument("--open-demo", action="store_true",
                    help="no sign-in while no accounts exist (anyone on the network gets full access). Demo only.")
    ap.add_argument("--idle-timeout-min", type=float, default=None, help="sign out after N idle minutes (default 30)")
    ap.add_argument("--session-max-h", type=float, default=None, help="absolute session length in hours (default 12)")
    ap.add_argument("--infer-node", default=os.environ.get("ARGONIX_INFER_URL", ""),
                    help="run detection on a GPU inference node (argonix.infer_server), e.g. ws://127.0.0.1:8200 through an "
                         "SSH tunnel; falls back to the local CPU if the node is down (env ARGONIX_INFER_URL)")
    ap.add_argument("--infer-token-file", default=os.environ.get("ARGONIX_INFER_TOKEN_FILE", "data/infer_token"),
                    help="bearer token copied from the node's data/infer_token")
    ap.add_argument("--tls-cert", default="", help="PEM certificate: serve HTTPS (cookies then get the Secure flag)")
    ap.add_argument("--tls-key", default="", help="PEM private key for --tls-cert")
    a = ap.parse_args()
    fine_timers()  # Windows: 1 ms waits instead of 15.6 ms ticks (argonix/hosttune.py has the measurements)
    pcores = prefer_performance_cores()  # Windows hybrid CPU: keep off the efficiency cores
    if pcores:
        print(f"host: pinned to performance cores {pcores[0]}-{pcores[-1]} (ARGONIX_ANY_CORES=1 to disable)", flush=True)
    if bool(a.tls_cert) != bool(a.tls_key):
        ap.error("--tls-cert and --tls-key go together")
    if not (ROOT / "data" / "synth" / "cam1.mp4").exists() and "demo_site" in a.config:
        from tools.make_synthetic import main
        main()
    opts = {k: v for k, v in (("idle_timeout_min", a.idle_timeout_min), ("session_max_h", a.session_max_h)) if v is not None}
    overrides = None
    if a.infer_node:  # this run only; the site config file is not changed
        overrides = {"remote": {"url": a.infer_node, "token_file": a.infer_token_file, "fallback": "local"}}
        print(f"detector: GPU inference node {a.infer_node} (CPU fallback if it is down)", flush=True)
    site = Site(Path(a.config), data_dir=Path(a.data), realtime=True, detector_overrides=overrides)
    explainer = None
    if a.vlm_url:
        from argonix.vlm import RemoteExplainer
        explainer = RemoteExplainer(a.vlm_url, privacy=site.cfg.get("privacy"))  # privacy.blur_faces_for_vlm
    if a.vlm_url:  # the same server's /chat (resident model, text only) translates investigation questions
        from argonix.vlm import ChatClient
        site.llm = ChatClient(a.vlm_url)
    site.start()
    app = create_app(site, explainer, open_demo=True if a.open_demo else None, auth_options=opts)
    auth = app.state.auth
    a.host, warning = bind_host(a.host, auth.open_demo and not auth.enabled)  # open demo: no account exists yet
    if warning:
        print(warning, flush=True)
    scheme = "https" if a.tls_cert else "http"
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    print(f"Argus dashboard: {scheme}://{shown}:{a.port}" + (" (all interfaces)" if shown != a.host else ""))
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning",
                    ssl_certfile=a.tls_cert or None, ssl_keyfile=a.tls_key or None)
    finally:
        site.stop()
