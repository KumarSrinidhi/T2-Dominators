# Argus app: full reference

Explainable multi-camera situational awareness. This is the complete reference for the runnable app: every feature, config key, API and measurement. For an overview, screenshots and a quick start, see the [repository README](../README.md).

```
video/RTSP/webcam ─► YOLO26n (model-agnostic interface) ─► ByteTrack ─► foot-point homography (2D ground plane)
   ─► zones ─► deterministic rules ─► correlation + P1–P4 priority ─► evidence (snapshot + clip)
   ─► SQLite ─► FastAPI/WebSocket ─► Frigate-style dashboard (Live · Review · Map · Recordings · System)
```

**Contents**
- Getting started: [Run](#run) · [Accounts & access](#accounts--access) · [Connecting existing CCTV](#connecting-existing-cctv) · [Troubleshooting](#troubleshooting)
- Core pipeline: [Automatic calibration & floor map](#automatic-calibration--floor-map) · [Behaviours (rules-v1)](#behaviours-rules-v1) · [Incidents & incident graph](#incidents--incident-graph) · [Crowd flow, queues & analytics](#crowd-flow-queues--analytics) · [Natural-language investigation](#natural-language-investigation)
- Across cameras (research-grade): [Next-camera prediction](#next-camera-prediction-handoff) · [Multi-camera Re-ID and global IDs](#multi-camera-re-id-and-global-ids-research-grade)
- Experimental, off by default: [Trajectory forecasting](#trajectory-forecasting-experimental-research-grade) · [Fall detection](#fall-detection-person-on-the-ground-experimental) · [Weapon-like objects and possible altercation](#weapon-like-objects-and-possible-altercation-experimental)
- Evidence and scope: [Measured, not claimed](#measured-not-claimed) · [Deferred](#deferred-per-agentsmd) · [Real footage](#real-footage-public-datasets)
- GPU and deployment: [Optional VLM explainer](#optional-vlm-explainer-gpu-box) · [GPU: one batched detector](#gpu-one-batched-detector-for-all-cameras) · [Distributed GPU inference nodes](#distributed-gpu-inference-nodes)
- Model cards: [`docs/model-cards/`](docs/model-cards/) · Demo walkthrough: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)

## Screenshots

Taken 2026-09-25 on the laptop CPU (`configs/real_site.json`, `--open-demo`): the two public clips at about 2 fps analytics, after one run of the scripted demo scenario. Alerts and incidents marked **DEMO** come from that script, not from the footage.

| | |
|---|---|
| ![Overview](docs/screenshots/overview-desktop-light.png) **Overview**: KPIs, most relevant cameras, alert trend, hotspots | ![Live](docs/screenshots/live-desktop-dark.png) **Live**: tracks and zones on both cameras, health per tile |
| ![Alert drawer](docs/screenshots/drawer-desktop-light.png) **Alert drawer**: observation vs derivation, "Why P2?", legal workflow steps, Original / With overlay | ![Incident](docs/screenshots/incident-desktop-light.png) **Incident**: typed evidence graph (door report + restricted-zone entry → one P1) |
| ![Map](docs/screenshots/map-desktop-dark.png) **Map**: bird's-eye ground plane with calibration error, zones, heatmap time-lapse | ![Analytics](docs/screenshots/analytics-desktop-light.png) **Analytics**: occupancy, density per m², queues, counting lines |
| ![System](docs/screenshots/system-desktop-dark.png) **System**: per-camera pipelines, compute nodes, demo scenario, phone push | ![Review](docs/screenshots/review-desktop-light.png) **Review**: filterable alert cards with SLA ages |
| ![Handover](docs/screenshots/handover-desktop-light.png) **Handover**: what is open at shift end | ![Phone](docs/screenshots/live-phone-dark.png) **Phone**: Live on a 390 px screen |

## Run

```bash
uv sync --extra cpu             # or --extra cu126 / cu128 on an NVIDIA GPU host
uv run python tools/setup_assets.py   # weights + clips git doesn't carry (see "Files git doesn't carry" below)
uv run python run.py            # generates synthetic demo clips on first run, then open http://127.0.0.1:8000
uv run python run.py --open-demo   # synthetic demo without sign-in (see "Accounts & access")
uv run pytest -q                # geometry, rules, correlation, full pipeline on video, API, RBAC, accounts, camera management
uv run pytest -q -m "not slow"  # skip the tests that run YOLO over whole videos (what CI runs)
uv run python tools/preflight.py --config configs/real_site.json --port 8001   # PASS/WARN/FAIL checklist before a demo
```

On a CUDA host use `uv run --no-sync` for all of these (see [Troubleshooting](#troubleshooting)). CI
(`.github/workflows/ci.yml`) installs CPU torch, runs the tests not marked `slow`, and a ruff check for syntax errors
and undefined names.

### Files git doesn't carry

Weights and clips live in the git-ignored `data/`. After cloning, `tools/setup_assets.py` fetches or renders them. It is idempotent: it skips files that are already there, resumes partial downloads and checks pinned SHA-256 hashes.

```bash
uv run python tools/setup_assets.py                  # core (default)
uv run python tools/setup_assets.py --all            # every group
uv run python tools/setup_assets.py --only reid,street
uv run python tools/setup_assets.py --check          # list what is present or missing; exit 1 if something is missing
```

| Group | Files | Used by |
|---|---|---|
| `core` | `data/models/yolo26n.pt`, `data/real/vtest.avi`, `data/synth/cam1.mp4` + `cam2.mp4` | `configs/demo_site.json`, `configs/real_site.json` (`cam_campus`), the tests |
| `models` | `data/models/yolo26s.pt`, `data/models/yolo26n-pose.pt` | `real_site_gpu*.json`, fall detection, the altercation analyzer |
| `street` | `data/real/social-distance/pedestrian.mp4` (Kaggle, read straight out of the dataset zip) | `cam_street` in `configs/real_site.json`. Needs `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY` |
| `reid` | `data/models/reid/osnet_ain_x1_0_msmt17_cosine.pth` (torchreid model zoo), `data/reid_eval/scene/` | `configs/reid_demo.json` |
| `handoff` | `data/synth_handoff/camA–C.mp4` | `configs/handoff_demo.json` |

Not fetched on purpose: the fine-tuned weapon and altercation weights (their training data has an unknown licence, so they aren't redistributed: `tools/train_weapons.py`, `tools/train_altercation.py`), TensorRT engines (GPU-specific: `tools/bench_gpu.py --export`), the VLM weights (`tools/vlm_server.py` downloads them on its host) and the evaluation datasets (`tools/datasets.py`). The `reid` and `handoff` scenes are rendered locally (laptop CPU: about 15–20 min and about 1 min). `assets/forecast/argonix_forecast_gru.pt` is committed. On a CUDA host run the script with `uv run --no-sync`.

On the first start the dashboard asks you to create the first admin account. See [Accounts & access](#accounts--access).

Use your own cameras by copying `configs/demo_site.json` and setting each camera's `source` to an RTSP URL, a file, or `0` for a webcam. Then set its `calibration` (4 image points ↔ 4 ground points in metres, plus optional held-out `check_points`) and its `zones`:
`uv run python run.py --config configs/my_site.json`. You can also add cameras from the dashboard API (see below).

## Accounts & access

Accounts live in `configs/users.json` (git-ignored, written atomically with owner-only permissions). Passwords are stored as PBKDF2-SHA256 hashes (310,000 iterations, per-user salt). The code is in `argonix/auth.py` (accounts, sessions, tokens), `argonix/api_auth.py` (routes) and `argonix/security.py` (CSRF, headers, cookies).

### First run

1. Start the server. While no accounts exist, it prints a **one-time setup code** and writes it to `<data>/setup_code.txt` (for example `data/run/setup_code.txt`, git-ignored, 0600):
   ```
   Argus first-run setup: no user accounts exist yet.
   One-time setup code: ABCD-EFGH-JKLM
   ```
2. Open the dashboard. It shows **Create the first admin account**: a username, a password (policy below) and the setup code.
   - A browser on the server itself (`http://localhost:8000` or `http://127.0.0.1:8000`) does not need the code.
   - Anyone else on the LAN does. This stops another machine from claiming the empty server first.
   - The loopback exemption is refused when the `Host` header is not a loopback name (DNS rebinding) or when `X-Forwarded-For`/`Forwarded` is present. Behind a reverse proxy on the same machine, set `"localhost_setup_without_code": false` anyway.
3. The option "Also give this account supervisor rights" is on by default. It lets a one-person demo watch live video and handle alerts, which the `admin` role alone cannot do (separation of duties). Turn it off for a real deployment and create separate supervisor/operator accounts.
4. Setup is accepted only while no accounts exist. After that, `POST /api/setup` returns 409, the code file is deleted, and every request needs a session, Basic credentials or a token.

`--open-demo` (or `"auth": {"open_demo": true}` in the site config) keeps the old behaviour for the synthetic demo: no sign-in while no accounts exist. Everyone who can reach the port is an unauthenticated admin-equivalent, so in this mode the server listens on **127.0.0.1 only** unless you pass `--host` (a network address then prints a warning). The top bar shows a red **OPEN DEMO · no sign-in** badge; click it to create the first admin, which turns sign-in on immediately. Account and token endpoints are refused in this mode, so the setup code cannot be bypassed. Without the flag, a server with no accounts is never open.

### Roles

Permissions come from manual §13 D-13.7 (`ROLE_PERMISSIONS` in `argonix/auth.py`). The admin role deliberately cannot handle alerts.

| Role | For | Permissions |
|---|---|---|
| `viewer` | Watching | `alert.read`, `incident.read`, `evidence.view` |
| `operator` | Alert triage | viewer + `alert.acknowledge/escalate/resolve/dismiss`, `feedback.write` |
| `supervisor` | Case handling | operator + `evidence.export`, `evidence.hold`, `calibration.approve`, `incident.assign`, `incident.merge`, `health.rereference` |
| `admin` | Configuration | `alert.read`, `incident.read`, `zone.edit`, `rule.edit`, `settings.edit`, `user.admin`, `calibration.approve` |
| `auditor` | Oversight | `audit.read`, `alert.read`, `incident.read` |

### Managing users

In the dashboard, **Settings → Users** (key `7`, needs `user.admin`) lists accounts with their roles, status, last sign-in and token count. From there you can:
- **Add user**: username, roles, and a generated temporary password. The user must choose their own password at first sign-in, and until they do, every other API call returns 403.
- **Roles**: edit a user's roles.
- **Disable/enable**: disabling ends the user's sessions and stops their tokens at once.
- **Reset password**: sets a new temporary password and signs the user out everywhere.
- **Sign out**: ends all of that user's sessions.
- **Delete**: removes the account.

Nobody can disable, demote or delete themselves, and the last active admin cannot be removed. Every change goes to the audit log (`user_created`, `user_updated`, `user_password_reset`, `user_deleted`, `password_changed`, `sessions_revoked`, `token_created`, `token_revoked`, `setup_completed`, `login`, `login_failed`, `logout`). Passwords, tokens and setup codes are never logged.

The CLI does the same offline (break-glass). A running server picks up file changes within about a second.

```bash
uv run python -m argonix.auth add alice admin            # prompts for the password (same policy)
uv run python -m argonix.auth add bob operator --temp     # --temp: must change at first sign-in
uv run python -m argonix.auth passwd bob [--temp]         # new password + sign out everywhere
uv run python -m argonix.auth disable|enable|signout|remove bob
uv run python -m argonix.auth list
```

**Password policy (server-side):**
- at least 12 characters (at most 1024)
- must not contain the username
- must not be a common or predictable password: a small deny-list, common words (including l33t spellings) padded with digits/symbols such as `Summer2024!!` or `P@ssw0rd-2024`, keyboard runs, sequences or repeats

The dashboard shows live hints, but the server decides.

### Sessions

- The dashboard session is an HttpOnly, `SameSite=Strict` cookie, HMAC-signed with `<data>/secret.key`. It gets the `Secure` flag when the request arrived over HTTPS. `"secure_cookie": true` forces the flag, for example behind a TLS proxy.
- **Idle timeout:** 30 min by default (`--idle-timeout-min`, or `"idle_timeout_min"` in the config).
  - Only user activity resets the timer: clicks or keys, which the dashboard reports via `POST /api/session/ping`, and state-changing requests. Background polling, live video and the WebSocket never keep a session alive.
  - A minute before expiry a toast offers "stay signed in". After expiry the sign-in page says **Signed out due to inactivity**.
  - Open WebSockets and MJPEG streams are re-checked about every 5 s and closed when the session ends.
- **Absolute maximum:** 12 h (`--session-max-h`, `"session_max_h"`).
- **Session epoch:** every cookie carries the account's epoch. **Sign out all sessions** (user menu), a password change, an admin reset, "Sign out" on the Users page or `argonix.auth signout` bumps it, and all older cookies stop working. A password change keeps the current browser signed in. **Sign out** also revokes that one cookie on the server.
- **Brute-force brake:** 5 failures per (username, source IP) in 15 minutes lock that pair for the rest of the window. There are also ceilings of 20 failures per username and 30 per IP. The same brake covers the login form, HTTP Basic and the current password in *Change password*.
  - Every failure gets the same 401 `invalid credentials or too many attempts`. Unknown users still cost a password hash, so neither the message nor the timing reveals whether an account exists.

### Scripts and integrations: HTTP Basic and API tokens

On `/api/...` and `/media/...` (never on the HTML page), clients can send `Authorization: Basic` with a username and password. The users, permissions and brake are the same as the dashboard's. A successful check is cached for 5 minutes, so scripts do not pay for PBKDF2 on every request. Disable Basic with `"basic_auth": false`.

```bash
curl -u alice http://argonix.local:8000/api/alerts?status=new            # prompts for the password
curl -u sam -o clip.zip http://argonix.local:8000/api/alerts/<id>/export     # sam = a supervisor (evidence.export)
```

**API tokens** are better for integrations such as NVR/VMS webhooks and exports.
- Create one under **Account & API tokens** in the user menu, or with `POST /api/tokens {"name": "nvr", "roles": ["viewer"], "expires_days": 90}`.
- The token (`axt_<id>_<secret>`) is **shown once**. The server stores only its SHA-256.
- A token carries a subset of its owner's roles. It never has more than the owner currently holds, and it stops working when the owner is disabled or deleted.
- Revoke it with `DELETE /api/tokens/{id}`. Admins can list and revoke everyone's tokens (`GET /api/tokens?all=true`).
- Tokens cannot manage users, passwords or other tokens.

```bash
curl -H "Authorization: Bearer axt_0123456789_…" http://argonix.local:8000/api/alerts
curl -u "token:axt_0123456789_…" http://argonix.local:8000/api/alerts      # clients that only speak Basic
```

A failed Basic or token check returns a `WWW-Authenticate` challenge. Plain cookie-less 401s never do, so browsers do not pop up a native password dialog.

### Hardening

- **CSRF:** any `POST/PUT/PATCH/DELETE` whose `Origin` (or, failing that, `Referer`) is not this server is refused with 403. That includes the login and setup forms and Basic-authenticated calls. When neither header is sent, a request that carries the session cookie must also send `X-Argonix-CSRF: 1` (the dashboard always does; cross-site pages cannot). Add other allowed origins, for example a reverse-proxy name, with `"trusted_origins": ["https://argonix.example.org"]`. The WebSocket checks `Origin` too.
- **Security headers on every response:**
  - `Content-Security-Policy`: only `self`, plus `blob:`/`data:` for images and video. No inline or eval scripts, no framing, `object-src 'none'`.
  - `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `Cross-Origin-Opener-Policy`/`Cross-Origin-Resource-Policy: same-origin` and a restrictive `Permissions-Policy`.
  - `Cache-Control: no-store` on the API and media.
- **XSS:** the UI builds all DOM from text nodes (`h()` in `static/js/util.js`) and never puts server strings into `innerHTML`.
- **TLS (optional):** `uv run python run.py --tls-cert data/tls/cert.pem --tls-key data/tls/key.pem`. Get a certificate in one of two ways:
  - Self-signed: `uv run --no-project --with cryptography python tools/make_cert.py [--name argonix.lan --ip 192.168.1.20]`. It covers localhost and this machine's names/IPs, and browsers show a warning you accept once.
  - Locally trusted: [mkcert](https://github.com/FiloSottile/mkcert), with `mkcert -install && mkcert -cert-file data/tls/cert.pem -key-file data/tls/key.pem localhost 127.0.0.1 <lan-ip>`.
  - Use `--host 127.0.0.1` to listen on this machine only.

**Config** (all optional, in the site config):
```json
"auth": {"open_demo": false, "idle_timeout_min": 30, "session_max_h": 12, "basic_auth": true,
         "localhost_setup_without_code": true, "trusted_origins": [], "secure_cookie": "auto"}
```

**Limitations:**
- Session revocation on logout and the brute-force counters are in memory, so both reset on restart. Epoch-based revocation (sign out everywhere, password change or reset, disable) is persisted.
- The design is single-process. Manual §22 plans Argon2id, server-side session rows, MFA for admins and OIDC for production; none of these is implemented here.
- No HSTS, because it would lock browsers out of self-signed LAN setups.
- `style-src` allows `'unsafe-inline'`, because the UI sets element styles.
- There is no password-reset-by-email: an admin, or the CLI on the server, resets passwords.

## Connecting existing CCTV

Argus reads the streams your cameras or NVR already publish. It does not replace the recorder.

- **Find the RTSP URL.** `GET /api/cameras/templates` lists common default paths for Hikvision, Dahua/Amcrest, Axis, Reolink, Uniview, TP-Link Tapo and generic ONVIF. These are factory defaults to check, not guarantees. Firmware, model line and site setup change them. Always run `POST /api/cameras/test {"source": "rtsp://..."}` first: it opens the stream over TCP with a hard timeout and returns resolution, fps and a snapshot.
- **Use the sub-stream for analytics** (Hikvision `.../Channels/102`, Dahua `subtype=1`, Reolink `_sub`, Tapo `stream2`). Detection runs on about 960 px at 5 fps, so decoding a 4K main stream only wastes CPU. Keep recording the main stream on your NVR.
- **NVR channels work the same way.** Point the URL at the NVR's IP and choose the channel. For example, Hikvision NVR camera 3 sub-stream is `/Streaming/Channels/302`, and Dahua is `channel=3&subtype=1`.
- **ONVIF discovery.** `GET /api/cameras/discover` sends a WS-Discovery probe (UDP multicast 239.255.255.250:3702) and lists the devices that reply, with their IP, name, hardware and a vendor guess. It only reaches devices on the same LAN segment or VLAN that have ONVIF enabled. An empty list does not mean there are no cameras. Discovery gives you the device address, not the stream URL. Get that from a template, the camera's web UI or ONVIF Device Manager.
- **Credentials.** Create a dedicated low-privilege "viewer" account on the camera or NVR for Argus. The site config JSON only ever stores the redacted URL (`rtsp://user:***@host/...`). The full URL, with the password, goes to `configs/<site>.secrets.json` beside it, which git ignores (owner-only where the OS allows) and which is read back at start-up. Keep that file private and out of backups you share. `${VAR}` placeholders are left as written. The API and audit log only ever show `rtsp://user:***@host/...`. If a client sends a redacted URL back unchanged, the stored password is kept. Characters like `@ : /` in a password must be percent-encoded, which `cameras.build_url()` does for you.
- **Manage cameras** with `GET/POST /api/cameras`, `PUT/DELETE /api/cameras/{id}`, `PUT /api/cameras/{id}/zones` and `PUT /api/cameras/{id}/masks`. Every change is written atomically to the site config (the previous version is kept as `<config>.bak`) and restarts only that camera. Mutating calls need `settings.edit` or `zone.edit` and are audited.
- **Processing errors don't kill a camera.** In live operation a camera whose frame processing raises (a GPU batch timeout, a bad frame) restarts after `restart_min_s` (2 s), doubling up to `restart_max_s` (60 s); the System page shows the error and `restarts`. A recording failure (ffmpeg exited, disk full) only pauses recording for `record_retry_s` (30 s, site config) and sets `recording_error`; analytics carry on.
- **Live streams** are read by a latest-frame grabber, so analytics never fall behind a buffered stream. A stream with no frames for `down_grace_s` (default 2 s) raises a camera-down health alert. The grabber reconnects with exponential backoff (1 s up to 30 s).
- **Calibration is optional.** An uncalibrated camera still runs detection, tracking, health checks and recording. Zones drawn in image pixels (`polygon_px`) still drive `intrusion`, `loitering` and `crowding`, using image-space point-in-polygon with a small pixel margin (`zone_margin_px`, default 8). Those alerts are labelled `space: image_px` and have no ground position. `running`, `wrong_way`, map positions and zones given in ground metres need calibration and are skipped until the camera is calibrated. The skipped items are listed in the camera's `calib_info`.

### USB webcams and the ESP32-CAM node

**Webcam mode.** A camera whose `source` is a device index (`"2"`) can ask for a capture mode before the first frame: `"capture": {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}` (also settable through `POST`/`PUT /api/cameras`). Without it OpenCV opens most UVC cameras at 640×480; an Arducam 16MP went from 640×480 at 20 fps to 1280×720 at 30. The camera may pick the nearest mode it supports. Example: `configs/arducam.json`.

**ESP32-CAM with a PIR sensor** (`firmware/esp32cam_argonix`, wiring and flashing in [`docs/ESP32_CAM.md`](docs/ESP32_CAM.md)). The board serves MJPEG on `:81/stream` (added as an ordinary `http://` camera) and its PIR state on `/pir`. Argus **polls** the PIR (`argonix/poll_sensors.py`, config key `poll_sensors`), so nothing on the sensor's network connects in:
- A rising edge, or the device's trigger count going up between polls, becomes a `sensor_motion` event (severity 1) with the linked camera's current frame as its snapshot. It is a deterministic observation that says nothing about who.
- Mapped to a zone's `site_zone`, it corroborates video events there ("sensor + video"); alone it stays low priority.
- An offline board shows in `/api/stats` → `poll_sensors`, and polling carries on.
- `tools/sim_esp32cam.py` imitates the board on 127.0.0.1 (a replayed clip plus `/pir/on`, `/pir/off`); `tests/test_poll_sensors.py` runs the real pipeline against it. Example site: `configs/arducam_esp32.json`.

This is separate from the signed door-controller webhooks below, which devices push. Contributed by Binay (branch `binay/webcam-capture`); tested live on an Arducam, and the ESP32 firmware was compiled but the full board setup has only been exercised through the simulator here.

## Automatic calibration & floor map

You no longer have to click four ground points per camera. Instead, Argus can **estimate** the ground plane from the people walking through the view, and then build a bird's-eye floor map automatically. The estimate is a 2D ground-plane homography, not a 3D model. It always comes with a stated error, and **an operator must approve it before it is applied**.

**How it works** (`argonix/autocalib.py`)
1. `POST /api/cameras/{id}/autocalibrate/start` begins collecting samples. Every 0.3 s it reads the camera's tracked `person` boxes and keeps those with confidence ≥ 0.5 that are not cut off by the image edge and have a plausible shape. For each one it takes the foot (bottom-centre) and the head (top-centre) in source pixels. Each track contributes at most one sample per second and at most 20 samples in total, so one person standing still cannot dominate.
2. It fits once there are ≥ 150 samples from ≥ 8 tracks and the foot points span ≥ 30% of the image height. The model is a pinhole camera with zero roll and the principal point at the image centre. The unknowns are the focal length *f*, the tilt *θ* and the camera height *h*. For each sample the foot is projected onto the ground, a 1.70 m person (sd 0.09 m) is stood there, and the predicted head row is compared with the observed one. The fit is a Huber-robust coarse grid followed by Levenberg–Marquardt, in numpy only. A weak prior on *f* (median 0.8 × image width) keeps the fit stable. If you know the lens's horizontal FOV, pass `{"hfov_deg": …}` to `start` and it will pin *f*.
3. The uncertainty is the larger of two estimates: a bootstrap over tracks, and a Laplace/Fisher estimate that allows for all samples of one person sharing one height. Either one is converted to a ground-position error at the median foot distance, plus 3% for the height prior. The result is `calib_error_m`, which is never below 0.5 m. Rules already use `calib_error_m` as the zone margin.
4. `GET …/autocalibrate` returns the state (`idle | collecting | ready | failed`), the counts, the frame coverage, the estimate and the reason. The estimate uses the normal config format: `image_points` and `ground_points`, in camera-centred metres with x to the right and y forward. It also includes `method: "auto-pedestrian"`, `params` (f, tilt, height, horizon row), the uncertainty, `rms_head_err_px`, and a walking-speed diagnostic that is not used in the fit.
5. `POST …/autocalibrate/apply` (`calibration.approve`) writes the estimate to the camera config through `site.update_camera_config` and records the change in the audit log. Zones drawn in pixels (`polygon_px`) are re-projected automatically. Zones given in metres are not, and the response warns about them.

**When it fails, and why that is on purpose.** Each failure gives a reason. It fails when:
- there are too few samples or tracks;
- nobody walks nearer or farther from the camera, so the foot points have too little vertical spread;
- the horizon falls below the feet;
- the height, tilt or *f* is implausible (height must be 1.5–30 m, tilt 5–85°, *f* 0.3–5× the image width);
- fewer than 60% of samples fit;
- or the error would be larger than max(3 m, 35% of the distance).

Box heights alone pin down the horizon and the lateral scale well. They only weakly pin down *f*, and so the depth scale, when the camera tilts only a little. In that case the fit refuses and asks for the lens HFOV. This happened on `vtest.avi`, below.

**Assumptions.** People are about 1.70 m tall. The ground is flat and there is one ground plane per camera. The camera has no roll and no lens distortion (fisheye and wide-angle lenses break the model). The principal point is at the image centre. Children, crowds of seated people, stairs and slopes all bias the result.

**Measured on synthetic ground truth** (`tests/test_autocalib.py`: 30 simulated walkers, 1.70 ± 0.09 m, ±1.5 px box noise, 1280×720):

| Camera | Estimate | Ground error at 10 m | Stated `calib_error_m` |
|---|---|---|---|
| f 700 px, tilt 35°, h 8 m, no lens info | f 682, tilt 35.7°, h 7.98 m | 0.29 m (0.36 m at the 19 m median distance) | 3.8 m (conservative; mostly the uncertainty in *f*) |
| f 1400 px, tilt 12°, h 3.5 m, no lens info | refused: "supply the lens HFOV" | – | – |
| same, with HFOV given | tilt 11.4°, h 3.37 m | 0.05 m (1.1 m at the 27 m median distance) | 3.1 m |

These are simulations, not field accuracy.

**On `data/real/vtest.avi`** (YOLO26n + ByteTrack at 5 fps, 426 samples from 77 tracks, foot rows spanning 63% of the frame):
- **With no lens info the fit is refused.** The best fit was f ≈ 1036 px, tilt 17.7°, h 6.1 m, but its error was about 16 m at 21 m.
- **With `hfov_deg: 50`** (a guess) the estimate is f 848 px, tilt 20.7°, h 5.8 m, horizon at row −33, `calib_error_m` 2.05 m.
- **Compared with the hand calibration** in `configs/real_site.json`:
  - Horizon: the two agree (row −33 vs −36).
  - Lateral widths: within 10% (6.7 vs 6.1 m at row 520, 14.2 vs 12.8 m at row 185).
  - Depth: the auto estimate is **~1.4–1.7× longer** (15.6 m vs 9.2 m between rows 520 and 185 on the centre column).
- **Walking speeds, as a cross-check.** The median net speed of moving tracks is 0.88 m/s with the hand calibration and 1.0–1.2 m/s with the auto estimates. Typical walking is about 1.2–1.4 m/s. This suggests the hand calibration's depth is too short, which the "scale may be ~20% short" note above also hints at. There is no ground truth, so neither calibration is verified.

**Floor map** (`argonix/birdseye.py`)
- **Per-camera view.** `GET /api/cameras/{id}/birdseye.png` warps the live frame onto the ground: north-up, 20 px/m by default, clipped below the horizon and to 40 m. Anything that is not on the ground (walls, people, vehicles) gets smeared along the viewing ray.
- **Site map.** `GET /api/map` returns the site bounds, px/m, each camera's placement, its field-of-view polygon in site metres, its calibration status, error and method, and a reason for any uncalibrated camera. `GET /api/map/background.png` is the composite of all the cameras with a 1 m / 5 m grid and field-of-view outlines. It is cached and rebuilt at most every 10 s.
- **Placement.** Each camera can have a `placement: {offset_m: [x, y], rotation_deg}` (counter-clockwise). Set it with `PUT /api/cameras/{id}/placement` (`zone.edit`). Cameras without one keep their own frame if their footprints don't overlap. Otherwise they are laid out side by side along +x.
- **Track coordinates.** `birdseye.site_xy(worker, ground_xy)` converts per-camera ground coordinates, such as the WebSocket track list, into site coordinates.
- **Heatmap.** `GET /api/map/heatmap` is an occupancy grid of person foot points. Cells are 0.5 m, row 0 is the lowest y, and values are person-seconds with a 120 s half-life decay. A person seen by overlapping cameras is counted once per camera.

## Behaviours (rules-v1)

All rules are scoped to a zone. A rule fires only after its condition has held for `min_seconds`, and a short grace period absorbs tracker flicker.

| Rule | Condition | Default |
|---|---|---|
| `intrusion` | person foot-point inside zone by ≥ calibration error (conservative boundary) | 1 s |
| `loitering` | person inside zone; optionally longer than is usual *for this zone at this hour* (`baseline_quantile`, see [Decision support](#decision-support-what-changed-track-record-dwell-baseline-recovery-calibration-access-log)) | 20 s |
| `running` | ground speed ≥ `speed_mps` | 2.5 m/s for 1 s |
| `wrong_way` | velocity against the zone's permitted `direction` | cos ≤ −0.5 for 1.5 s |
| `crowding` | person count > `max_count` | 5 for 3 s |
| `line_cross` | person track crosses a counting line in the configured direction (opt-in per line, `events`) | at most one per `min_gap_s` = 5 s per line |
| `vehicle_in_zone` | vehicle (car/motorcycle/bus/truck) box bottom-centre inside a pedestrian-only zone by ≥ calibration error (needs `vehicles.enabled`) | 2 s |
| `vehicle_stopped` | vehicle in a no-stopping zone that moved ≤ `still_m` (calibrated) or ≤ `still_box_frac` × its box size (uncalibrated) over `window_s` (needs `vehicles.enabled`) | 1.0 m / 0.15 per 3 s, for 60 s |

Any zone rule (and a line's events) can carry a **schedule**; outside its windows the rule does not arm (see [Schedules, line events and vehicles](#schedules-line-events-and-vehicles)).

Every alert stores these fields:
- **observation**: what was seen
- **derivation**: rule, thresholds, duration
- heuristic **confidence** and how it was computed
- camera, zone and site zone
- ground position ± calibration error
- detector, tracker and rules versions
- snapshot and clip

Alerts on the same site zone or the same track within 15 s are merged. Each extra rule or camera that agrees raises the score. Cross-camera merging is spatial only and makes no identity claim.

Operators can acknowledge, escalate, or dismiss an alert and add a note.

## Incidents & incident graph

The correlator merges events into **alerts** inside 15 s. An **incident** is the longer, operator-facing situation
built on top of them (`argonix/incidents.py`). Every alert belongs to exactly one incident. An incident collects
the alerts that share a zone key (the site zone, or camera + zone) or a per-camera track. For camera problems, it
collects the alerts from the same camera. Grouping is deterministic and uses only space and time. It makes no
identity claim and never assesses intent.

Incidents are on by default. `"incidents": {"enabled": false}` turns them off; the incident endpoints then return 404
and the console hides the page.

**Lifecycle.** Phase changes are automatic and run on event time.

| Phase | When |
|---|---|
| `open` | The first alert opens it. An incident below P2 is held for `hold_s` (10 s) so a burst settles before any push decision. |
| `active` | Immediately for P1/P2, otherwise after the hold. The push decision is made here. It stays active while events keep arriving. |
| `cooling` | No new event for `cool_after_s` (60 s). A camera-problem incident stays active while that camera's health is still DEGRADED or DOWN. |
| `closed` | `close_after_s` (240 s) more without an event; or an operator resolves or closes it, or merges it away; or it rolls over after `max_duration_s` (1 h). |

A matching alert within `reopen_window_s` (600 s) of a quiet or operator close **reopens** the same incident and
counts it in `reopen_count`. A rollover close is never reopened; the next alert opens a new incident.

**Operator actions.** The operator's *status* (`open`, `acknowledged`, `investigating`, `resolved`, `closed`) is
separate from the automatic phase. Each change is audited and shown on the incident's timeline.

| Action | Permission |
|---|---|
| Status `open` or `acknowledged` | `alert.acknowledge` |
| Status `investigating` | `alert.escalate` |
| Status `resolved` or `closed` (also closes the phase) | `alert.resolve` |
| Assign | `incident.assign` |
| Write a summary | `feedback.write` |
| Merge incidents, split alerts out | `incident.merge` |

- **Merge** moves the alerts of one or more source incidents into a target of the same kind. The sources close with `merged_into` set. A camera-problem incident cannot be merged with a behaviour incident.
- **Split** moves chosen alerts into a new incident. At least one alert must stay behind, and the new incident never sends a push by itself.

**Inhibition** (Alertmanager-style). While a camera-problem incident (`camera_health`) is open on a camera, new rule
alerts from that camera are still stored and shown. They are flagged `suppressed` with `inhibited_by`, and they are
never pushed, because evidence from a covered, blurred or moved camera may be unreliable. The camera-problem alert
itself is never inhibited. The rule is configurable with `"inhibit": [{"source_rules": [...], "target_rules": "*",
"equal": "camera" | "zone"}]`.

**Push budget.** Only `push_tiers` (P1, P2) may interrupt anyone.
- P1 is exempt from the budget.
- P2 draws from an event-time token bucket: 6 pushes per operator-hour, burst 3. Both scale with `operators_on_duty`.
- Every decision is stored as a notification row: `would_push`, `budget_denied`, `suppressed` or `not_push_tier`, with the reason. A reopened incident, or one that escalates to a higher priority, gets a new decision.
- **Nothing is sent anywhere** unless a webhook is configured. There is no SMS, email or app push.
- The token bucket lives in memory and refills on restart. Incidents themselves are in SQLite, and live or still-reopenable incidents are reloaded at start-up.

**Webhook** (off by default). Each `would_push` decision is POSTed as `argonix.incident.push`. The payload carries:
- the incident and alert ids, the trigger and the priority;
- the title, cameras and zones;
- `requires_human_decision: true`;
- a console link when `console_url` is set.

Delivery works as follows:
- The headers follow Standard Webhooks v1: `webhook-id`, `webhook-timestamp`, and `webhook-signature` = `v1,` + base64(HMAC-SHA256 of `{id}.{timestamp}.{body}`). A `whsec_` secret is base64-decoded first.
- A background thread sends the requests, so the pipeline never waits. Network errors, 5xx and 429 responses are retried with backoff. HTTP 410 disables the endpoint.
- Every delivery result is stored as a notification row (`webhook_sent` or `webhook_failed`).

```json
"incidents": {"hold_s": 10, "cool_after_s": 60, "close_after_s": 240, "reopen_window_s": 600, "max_duration_s": 3600,
              "push_tiers": ["P1", "P2"], "budget_exempt": ["P1"], "push_budget_per_operator_hour": 6, "push_burst": 3,
              "operators_on_duty": 1,
              "webhook": {"url": "https://example.org/hook", "secret": "whsec_…", "retries": 3, "console_url": "https://argonix.lan:8000"}}
```

**Evidence graph.** `GET /api/incidents/{id}/graph` returns a typed graph of the newest 60 events (`max_events`, at
most 200). `static/js/graph.js` draws it as a force-directed SVG: the layout is deterministic and nodes can be
dragged. Clicking a node opens the alert, the evidence clip, the linked incident or the camera.

| Node | Meaning |
|---|---|
| event | One rule event. It sits on the `observation` layer, or on the `inference` layer when a learned model raised it. |
| camera, zone | Where the event was seen. |
| track | A per-camera tracker id. It is not an identity. |
| evidence | The snapshot and clip. |
| incident | The camera-problem incident that muted this event. |
| global track | A Re-ID global id. It appears only when Re-ID is on, and it is marked probabilistic. |
| predicted camera / leaves coverage | The handoff estimate. It is marked as an estimate. |

The edge types are `observed_in`, `same_track`, `corroborates` (distinct rules or cameras inside one alert),
`predicted_next`, `inhibited_by` and `evidence`.

**Hotspots.** `GET /api/hotspots?days=28&by=zone|camera&metric=incidents|alerts` counts incidents (or alerts) per
local hour-of-week slot (Mon 00:00 … Sun 23:00, 168 slots) for each zone or camera. It lists the top cells with example
ids, and attaches the spatial grid of where the rules fired (see [Crowd flow, queues & analytics](#crowd-flow-queues--analytics)).
These are frequencies of past alerts, not a forecast, and small counts are noisy.

**API** (all audited when they change something)

| Endpoint | Permission |
|---|---|
| `GET /api/incidents?phase=&status=&priority=&camera=&alert_id=&limit=` | `incident.read` |
| `GET /api/incidents/{id}` (with alert details and push decisions) | `incident.read` |
| `PATCH /api/incidents/{id}` `{"status", "assignee", "summary", "note"}` | see the table above |
| `POST /api/incidents/{id}/merge` `{"sources": [ids]}` | `incident.merge` |
| `POST /api/incidents/{id}/split` `{"alert_ids": [ids]}` | `incident.merge` |
| `GET /api/incidents/{id}/graph` · `/audit` | `incident.read` |
| `GET /api/notifications?incident_id=` | `incident.read` |
| `GET /api/hotspots` | `alert.read` |

**UI.** The **Incidents** page has two tabs.
- **Incidents:** the list, with lifecycle chips, filters and merge. Each incident opens a detail view with the evidence graph, the timeline, the push decisions, the related and predicted-next cameras, the split dialog and the audit trail.
- **Hotspots:** an hour-of-week heatmap.

The lifecycle, graph, merge/split, inhibition, budget and webhook signing are covered by `tests/test_incidents.py`
and `tests/test_incidents_api.py`. They are unit tests on scripted alerts, not a field evaluation.

## Crowd flow, queues & analytics

`argonix/flow.py` computes crowd statistics from person tracks only.
- **Calibrated cameras:** foot points on the ground plane, in metres.
- **Uncalibrated cameras:** source-image pixels. They give counts only: no area, no m/s thresholds, and speeds in body heights per second.

The statistics are passive. Events come only from zone rules that an operator switched on, and each event states
what was measured, against which reference, with which thresholds.

The flow analyzer is on by default (`"analytics": {"enabled": true, "half_life_s": 300, "queue_window_s": 300}`), and
`argonix/analytics_install.py` installs it. The unattended-object and anomaly analyzers are off by default.

**Counting lines.** Each camera can have `lines`. Edit them with the line editor or with `PUT /api/cameras/{id}/lines`
(`zone.edit`, audited, at most 32 per camera).
- **Format:** `{"id", "name", "points_px": [[x0, y0], [x1, y1]]}` in image pixels, which works without calibration; or `"points"` in ground metres, which needs calibration. `in_label`, `out_label` and `band` are optional.
- **Direction:** "in" is the right-hand side when looking from A to B on screen.
- **Crossing:** counted when a foot point moves from more than `band` (6 px or 0.15 m) on one side to more than `band` on the other side, through the segment itself (not its extension).
- `GET /api/analytics/lines` returns the totals, the last 60 minutes and per-minute series.

**Zones.** `GET /api/analytics/zones` returns, for each zone:
- occupancy, entries and exits;
- a dwell histogram (0, 5, 15, 30, 60, 120, 300, 600+ s);
- an 8-way direction histogram and the dominant flow direction with its concentration R (decayed with a 300 s half-life);
- density in people/m² (calibrated cameras only);
- 60 minutes of per-minute series.

Completed stays also feed the per-zone dwell baseline.

**Queues.** A zone becomes a queue with `"type": "queue"` (or when it has the `queue_long` rule).
- **Departures:** with a service line (`service_line_px` or `service_line`), a crossing from the queue side is a departure. Without one, a zone exit after at least 2 s counts.
- **Reported values:** length, 60 s mean length, departures, service rate per minute, and a steady-state check (the two halves of the window within 30 %).
- **Wait estimate** by **Little's law**, W = L / λ (mean length over `queue_window_s` ÷ departure rate). It needs at least 30 s of data and 3 departures. It is labelled an estimate, valid only in steady state.
- **Measured wait:** the median of per-track entry-to-service times, shown next to the estimate.

**Crowd rules.** They are opt-in per zone, for example `"rules": {"counterflow": {}, "density": {"warn_ppm2": 1.5}}`.
Each fires once per episode and re-arms after 60 s, with a 2 s grace for tracker flicker.

| Rule | Condition (defaults) | Needs calibration |
|---|---|---|
| `counterflow` | At least 4 tracks moving (≥ 0.4 m/s), and at least 2 of them, and at least 30 %, moving against the reference (cos ≤ −0.5), for 3 s. The reference is the zone's configured `direction`, or the learned dominant flow (concentration R ≥ 0.5 over ≥ 30 track-seconds). Learning is frozen while the rule fires. | yes |
| `density` | At least 3 people and ≥ 2.0 people/m² (warning) or ≥ 4.0 (critical, higher severity), for 5 s. The derivation gives a density range from `calib_error_m` × the zone perimeter. | yes |
| `queue_long` | 60 s mean length > `max_length` (10), or Little's-law wait > `max_wait_s` (0 = off; skipped when the queue is known not to be steady), for 10 s | no |
| `dispersal` / `convergence` | At least 4 moving people, of whom at least 60 % move radially outward from (or inward to) their centroid, with a median radial speed ≥ 1.5 m/s and a median speed ≥ 2× the speed 3–8 s earlier; 1 s. The text says it is "a motion cue only, the cause is unknown". | yes |

**Unattended object** (`argonix/objects.py`, off by default; `"objects": {"enabled": true}`).
- **Detection:** enabling it adds COCO backpack, handbag and suitcase to the detector. Objects get their own tracker and never reach the person rules.
- **States:** each object moves through *moving → stationary → attended/unattended*. The associated person is the nearest person at drop time (within 2 m). The object is attended while that person stays within 3 m, while the object is inside a person box, or once another person has stayed within 1 m for 3 s.
- **Event:** after 10 s stationary and 30 s continuously unattended, one `unattended_object` event fires. The object then re-arms after 300 s.
- **Wording:** the event reports the object, how long it has been still, and the last associated person track as a pointer to review. It is not an accusation: people put bags down for many reasons, and a missed detection or an ID switch can make an attended bag look unattended.
- **Distances:** in ground metres when the camera is calibrated. Otherwise they are approximated from the person's box height (1.70 m), and the event is labelled that way.
- `GET /api/analytics/objects` returns the state. With a remote inference node, start the node with `--extra-classes 24,26,28`.

**Learned trajectory anomaly: corroboration only** (`argonix/anomaly.py`, off by default; `"anomaly": {"enabled": true}`).
- **Model:** a numpy Isolation Forest per camera (100 trees, 256 samples each).
- **Features per track:** mean, p90 and std of speed; straightness; log dwell; heading changes per minute; fraction of time stopped; hour of day; weekend.
- **Training data:** a rolling window of tracks that were not part of any fired rule. The analyzer stays silent until it has 250 of them.
- **Score:** turned into a percentile against held-out recent tracks. An event needs the 99th percentile. The percentile is relative to this camera's recent traffic and is **not a probability**.
- **Event:** `trajectory_anomaly` has severity 1 and sets `learned_model`, so the correlator caps it at **P3** until a deterministic rule agrees. The event lists which features were unusual, with their typical values, and never who or why.
- **Persistence:** the training window is kept under `<data>/anomaly/`, and the forest is refitted from it at start-up (no pickles).
- `GET /api/analytics/anomaly` returns the status.

**Occupancy baselines and CUSUM** (`argonix/baseline.py`, always on; config `"baseline"`).
- **Reference:** per camera, zone and hour of the week, the median and MAD of 1-minute mean occupancy from earlier weeks. The most recent hour is excluded, so a shift in progress cannot become its own reference.
- **Point test:** once 8 samples exist for that slot, a modified z ≥ 3.5 raises `occupancy_deviation`.
- **CUSUM:** a two-sided CUSUM (k 0.5, h 8) raises `occupancy_shift` for a sustained small shift that never crosses the point threshold.
- **Caveats:** both are low-severity corroboration signals. Consecutive minutes are autocorrelated, so false alarms are more frequent than the textbook CUSUM tables suggest. `GET /api/baseline` returns the baselines.

**Spatial hotspots.** `GET /api/analytics/hotspots?window=24h|7d&cell_m=1&cell_px=48` shows where rules fired.
- **Grid:** per camera, 1 m ground cells (or 48 px image cells for uncalibrated cameras), plus per-zone counts.
- **Dismissed alerts** are left out unless `include_dismissed=true`.
- **Meaning:** this is event frequency, not risk. It counts true and false positives alike.

**UI.** The **Analytics** page shows:
- counting lines, zone occupancy, dwell and direction;
- density, and queue length and wait with trend charts;
- crowd-flow events;
- the anomaly indicators, labelled *corroborating only*;
- the hotspots.

When the flow analytics are off or still empty, the page can show sample data instead: in demo mode, or when you ask
for it. The sample data is clearly labelled.

**What has and has not been measured.** `tests/test_analytics.py` checks every rule, the queue maths, the Isolation
Forest and the CUSUM on **synthetic tracks with known answers**. No crowd, queue or counting accuracy has been
measured on real footage, because there is no ground truth for it here.
- Counts are person *tracks*, so tracker ID switches split or merge people.
- There is no density-map crowd model. In dense crowds the detector misses people, so density is likely underestimated.

## Natural-language investigation

The **Investigate** page answers questions about past alerts and incidents, such as "intrusions on the lawn in the last
hour" or "which cameras had camera problems today?" (`argonix/investigate.py`, `static/js/investigate.js`).

**How a question is answered**
1. **Translate.** The question (at most 500 characters, control characters stripped) becomes a structured query with **allow-listed fields only**:

   | Field | Allowed values |
   |---|---|
   | `target` | `alerts` or `incidents` |
   | `time` | a preset (`today`, `yesterday`, `last_hour`, `last_24h`, `last_7d`, `last_30d`, `this_week`, `all`), `last_hours` (up to 90 days), a `date`, a `part_of_day` (morning/afternoon/evening/night), or `from`/`to` |
   | `cameras`, `zones`, `rules`, `priorities`, `status` | the site's own vocabulary |
   | `min_confidence`, `text_contains`, `limit` | values |
   | `aggregation` | `list`, `count`, `by_zone`, `by_camera` or `by_hour` |

   Two engines can translate:
   - **LLM:** the text model on the VLM server (`/chat`, the same resident model as `/explain`, so no extra VRAM). It is enabled by `run.py --vlm-url` or `"investigate": {"llm_url": …}`.
   - **Rules:** a deterministic keyword and date parser. It works offline, and it is also the fallback.
2. **Validate.** The query is checked strictly against the site's camera ids, zone ids, rule ids, priorities and statuses. An unknown field or value is rejected, and the keyword parser is used instead, with a warning.
   - The model **never writes SQL and never sees the database**.
   - An unreachable or slow LLM backs off for 60 s.
   - Valid translations are cached in SQLite, keyed by prompt version, model, vocabulary and question. A translation that turned a relative day ("yesterday") into a fixed date is not cached.
3. **Execute.** The query runs in Python over rows fetched with parameterised store calls (up to 5,000 rows; the answer says when it was truncated).
4. **Summarise: cite or refuse.** A template summary always cites alert and incident ids from the results. An LLM-written summary replaces it only when all of these hold:
   - it is at most 1,200 characters;
   - it cites at least one id, and every id it cites is in the results;
   - it uses no person-labelling words ("suspicious", "criminal", "dangerous", "intruder" …).

   Otherwise the template summary is shown, with the reason. `"summarize": "template"` forces the template.

**API** (`alert.read`; every question is audited as `investigate` or `investigate_rejected`)
- `POST /api/investigate` `{"question": "…"}`, or `{"query": {…}}` to run a structured query directly. The UI shows the query as editable chips and re-runs it without the LLM.
- The answer carries:
  - `engine` (`llm`, `rules` or `structured`) and `cached`;
  - the results and the summary with its `citations`;
  - `warnings` and `timings`;
  - a plain-language description of the query and a disclaimer.
- `GET /api/investigate/schema` returns the allow-list and the site vocabulary.

The summary is assistive text, not evidence: open the cited alerts. `tests/test_investigate.py` covers the parser,
the validator and the citation guard on fixed examples. The LLM's translation accuracy has not been measured.

## Next-camera prediction (handoff)

When a person track leaves a camera's view, Argus estimates **which camera it will appear in next**, or whether it leaves coverage, and **roughly when**. The estimate is learned from how tracks move between cameras. It uses **no appearance or identity matching**. A cross-camera link is only a guess gated by topology and travel time. It is probabilistic and can be revised, and for non-overlapping cameras this is research-grade (see AGENTS.md). The output is labelled as an estimate with probabilities, and it raises **no alerts**. The informational `handoff_expected` event type exists but is off by default (`"emit_events": false`).

Enable it per site with `"handoff": {"enabled": true, ...}` (see `configs/handoff_demo.json`). `argonix/handoff.py` registers itself as a `Site.analyzers` entry.

**Method** (`argonix/handoff.py`)
1. **Track lifecycle per camera.**
   - An **exit** is a person track that goes unseen for more than `gap_s` (1.5 s). It must have last been seen within `border_margin` (8 %) of an image border, or be extrapolated past the border within 0.6 s, or have been inside a configured `exit_zones_px` polygon (for example a doorway in the back wall).
   - An **entry** is a new track whose first position is near a border or inside an exit zone.
   - A track lost in the middle of the image is neither, because it is most likely an occlusion or an ID switch.
2. **Features at exit** (46 values plus a camera one-hot):
   - the last 8 normalised foot positions and their velocities
   - exit side (one-hot) and an exit-zone flag
   - speed, and heading as sin/cos
   - ground velocity in m/s when the camera is calibrated
   - log dwell time
   - hour of day as sin/cos
3. **HandoffNet.** An MLP (in → 64 → 32 → K cameras + "leaves coverage", 5.6 k parameters, CPU only) with a travel-time head that predicts a log-normal μ/σ of transit seconds for each destination. Softmax confidence is temperature-scaled on a held-out split.
4. **Self-supervised labels.** After `max_dt_s`, each exit is associated with the entries on other cameras inside the travel-time gate. The gate uses the learned p05–p95 range of the edge, or `[min_dt_s, max_dt_s]` until that is known.
   - Each candidate is scored as P(dest | camera, exit side) × travel-time density ÷ (the destination's rate of unrelated entries × the number of exits competing for that entry). This is compared against P(leaves coverage).
   - Several candidates share the weight; the label is a soft distribution, never a certainty. With no candidate the label is "leaves coverage".
   - Association waits on a per-camera time watermark, so cameras processed at different speeds do not matter.
   - The learned **topology** (weighted transition counts and travel-time p10/p50/p90) is a by-product.
5. **Cold start and online learning.**
   - `tools/sim_handoff.py` simulates a floor plan: portals (edges and doors), corridors with lengths and leak probabilities, route choice, walkers at 1.0–1.6 m/s with pauses. It projects the walkers through each camera's homography, adding partial visibility at borders, jitter, dropouts and ID switches, and runs them through the **same** lifecycle code.
   - The initial model is trained on the simulator's ground truth.
   - After that it is fine-tuned from the replay buffer every `retrain_every` online labels. The synthetic samples' weight fades as real labels accumulate.
   - Model, buffer and topology persist under `<data>/handoff/`.
6. **Served ETA.** By default (`"eta_source": "auto"`) the ETA is the per-edge empirical median (p10–p90) once an edge has at least 8 samples, and the NN head is used otherwise. The NN head generalised worse than the edge medians on rendered video (see below).
7. **Scoring.** The prediction made at exit is scored when its label resolves: rolling top-1, log-loss and ETA error. The **baseline** is scored in the same way: the most frequent destination per (camera, exit side), trained on the same data. Rolling metrics are scored against the self-supervised labels, not ground truth.

**API.** All endpoints need `alert.read` except retrain; each returns 404 when handoff is disabled.
- `GET /api/handoff/predictions`: top-3 `{camera, prob, eta_s}` for tracks leaving a view, and for recently exited tracks.
- `GET /api/handoff/topology`
- `GET /api/handoff/metrics`
- `POST /api/handoff/retrain`: needs `settings.edit` and is audited.

**UI.**
- **Live:** tile badges such as "#12 → East corridor 71% · ~6 s est."
- **Map:** learned transition arrows (width = count) and dashed predicted-destination arrows.
- **System:** a "Next-camera prediction" card with metrics, the topology table and a Retrain button.

**Demo.** Render the clips, watch them live, or score them against ground truth:

```bash
uv run python tools/make_synthetic_handoff.py                      # 3 cameras, 480 s, 37 walkers, truth.json
uv run python run.py --config configs/handoff_demo.json            # watch it live
uv run python tools/make_synthetic_handoff.py --eval --imgsz 320   # full pipeline, faster than real time, vs truth
uv run python tools/sim_handoff.py                                 # NN vs baseline on held-out simulation only
```

**Measured.** Laptop CPU (8 threads, no GPU), 2026-09-25. These are small samples on synthetic data.

| Setting | n | NN top-1 | Baseline top-1 | NN log-loss | Baseline log-loss | ETA MAE |
|---|---|---|---|---|---|---|
| Held-out simulation (train 2 h, test 30 min simulated) | 375 | **0.960** | 0.877 | 0.154 | 0.279 | NN head 2.0 s · edge median 2.2 s |
| Rendered video, synthetic cold start + online fine-tune | 62 matched exits | **0.855** (95 % CI 0.75–0.92) | 0.806 (0.69–0.89) | 0.452 | 0.463 | NN head 5.2 s · edge median 2.1 s (served) |
| Rendered video, online labels only (no cold start) | 29 with a model / 62 | 0.41 (12/29) | 0.39 (12/29 paired; 0.39 on all 62) | 2.92 | 1.29 | 7.2 s |

- **Held-out simulation.** The NN's gain is concentrated where the exit side alone is ambiguous. camA's right edge has two corridors, to camB (near) and camC (far): there the NN scores 0.91 and the baseline 0.53. Elsewhere the two are equal, or the NN is slightly worse (camB left: 0.85 vs 0.89).
- **Rendered video**, 480 s, 3 cameras, YOLO26n imgsz 320 + ByteTrack at 5 fps, processed at **1.22× real time**:
  - The lifecycle found 62 of 75 true exits (recall 0.83).
  - On paired exits the NN was right 53 times and the baseline 50 times. **That difference is not statistically significant.**
  - The NN wins on camA's right edge (16/19 vs 10/19) and loses on camB left (6/9 vs 7/9) and camC bottom (3/5 vs 5/5).
  - The self-supervised association labels matched the truth on 82 % of exits (mean weight on the true class 0.76).
- **Online only:** 8 minutes of traffic is far too little. The labels matched the truth on only 47 % of exits, because the uniform travel-time prior is diffuse and loses to "leaves coverage". The NN, trained after 30 labels, was no better than the baseline.

**Limitations.**
- No identity is used, so labels are ambiguous by construction whenever several people move at once.
- The cold start is **optimistic** here, because the simulator is the same floor plan that generated the video. For a real site you must describe its topology (`sim_topology`: portals, corridors, routes), and the simulator will only approximate it.
- A real site needs **weeks of real traffic** before the online topology and fine-tuning mean much. Rolling accuracy is measured against self-supervised labels, not truth.
- The NN travel-time head did not transfer from simulation to rendered video, which is why ETAs come from the per-edge distributions by default.
- Non-overlapping multi-camera association is research-grade. Use these outputs as operator hints only, never as evidence about a person.

## Multi-camera Re-ID and global IDs (research-grade)

**Off by default.** Enable it with `"reid": {"enabled": true}` (defaults in `argonix/global_ids.py` `DEFAULTS`). The Journeys page and `/api/reid/*` show the results.

A global ID is an **opaque number for one journey through the cameras**. It is not a name or an identity. There is no face recognition anywhere. IDs are **probabilistic and revisable**, and they are purged after `retention_s` (1 h). Embeddings are dropped 5 min after a segment ends. Nothing here raises an alert or labels anyone.

**Method** (`argonix/reid.py`, `argonix/global_ids.py`)
- **Appearance.** Good crops only: not truncated, detector confidence ≥ 0.5, minimum height, plausible aspect ratio. They are embedded by a pretrained OSNet person Re-ID network (torchreid weights trained on MSMT17, which is non-commercial research use; fp16 batched on GPU). One shared embedding service serves all cameras.
- **Association.** When a track appears, every candidate predecessor gets Bayesian odds against "new person". The odds combine:
  - the topology prior (learned by the handoff module);
  - travel time against the edge's learned band;
  - the appearance likelihood ratio (cosine similarity);
  - direction consistency.
  The track joins a global ID only when P ≥ `link_threshold` (0.6). The top alternatives and their evidence are stored.
- **Revisable.** When a segment ends, the link is re-scored with all its crops. It can be split, re-linked or merged. Operator link, unlink and merge corrections are locked, persisted and audited.
- **Grey and IR.** When either side is monochrome, embeddings are compared on luminance only.
- Weights go in `data/models/reid/` (`osnet_ain_x1_0_msmt17_cosine.pth` preferred). Without them, a weak hand-crafted HSV/HOG embedder is used and labelled as such.

**Measured** (`tools/eval_reid.py`, RTX 4050 Laptop, 2026-09-25; VRAM peak for Re-ID 51 MiB)

| Evaluation | Result |
|---|---|
| Real footage, single camera (vtest.avi): same-track vs different-track similarity, ROC AUC, colour / grey crops | OSNet-AIN 0.891 / 0.885 · OSNet-x0.25 0.878 / 0.864 · hand-crafted 0.782 / 0.738 (317 same pairs, 12,678 different) |
| Simulated trajectories, synthetic appearance vectors: cross-camera links, fused | separability 2.0: P 1.00, R 1.00 · 1.5: P 0.96, R 0.62 · 1.0: P 0.43, R 0.35 |
| Rendered 3-camera video (real pedestrian crops pasted per visit, with colour/brightness/blur changes; 31 true links), full pipeline, colour footage | fused P 0.61, R 0.36 (IDF1 0.67) · topology only P 0.80, R 0.26 · appearance only 0 correct links |
| Same, grey footage | fused P 0.50, R 0.13 (IDF1 0.60) · topology only P 0.67, R 0.19 |

**Demo.** `configs/reid_demo.json` watches the rendered 3-camera scene from the table above, looping. The scene has
real pedestrian crops from `vtest.avi`, a different real frame of the same person on each camera visit, and a
colour, brightness and blur change per camera. The config turns on handoff (demo floor-plan topology, simulator
cold start) and Re-ID (`"reid": {"enabled": true, "model": "auto", "color_mode": "auto", "max_dt_s": 40}`).

```bash
uv run python tools/make_reid_demo.py     # renders data/reid_eval/scene/ once (CPU, needs data/real/vtest.avi), rewrites the config
uv run python run.py --config configs/reid_demo.json --data data/run_reid --port 8004 --open-demo
```

Rendering is slow on a CPU. On the laptop, shared with other jobs, a 60 s test scene took 7 minutes; the default
scene is 480 s. `tools/eval_reid.py --video` writes the same folder, so if a GPU host has already run the evaluation,
copy its `data/reid_eval/scene/` (and `bank/`) instead. That is the exact scene the table above was measured on. The
tool reuses an existing scene unless `--force` is given.

Open **Journeys**. Global IDs appear once walkers have crossed between cameras. Without OSNet weights in
`data/models/reid/`, the weak hand-crafted embedder is used. `GET /api/reid/metrics` reports which one (`embedder`,
with a warning), and `tools/preflight.py` warns about it too. On this scene the measured fused precision was 0.61
(table above), so expect wrong links. They are a chance to show the operator's **Split here** correction.

**What this means.** The single-camera AUC shows that OSNet separates people far better than the hand-crafted features. On the rendered multi-camera video, though, **appearance alone linked nothing correctly**. Fusing it with topology raised recall (0.26 → 0.36) at a cost in precision (0.80 → 0.61). The confidence intervals are wide: 31 true links, Wilson 95 % for precision is 0.36–0.78. The synthetic per-visit colour changes and small crops are harsher than many real sites, but that is still no evidence Re-ID works on a real site. Treat global IDs as operator hints to verify on the evidence crops, never as a finding about a person.

## Trajectory forecasting (experimental, research-grade)

**Off by default.** When it is on, Argus predicts where each tracked person may be over the next 4.8 s. The prediction uses only that person's own recent motion. It uses no identity, appearance or interaction with other people. It is a **prediction, not an observation**, and the UI and API label it as one.

Enable it with `"forecast": {"enabled": true}`; `configs/real_site_experimental.json` has a full example. `argonix/forecast.py` registers itself as a `Site.analyzers` entry through `argonix/experimental.py`.

**Method**
- **Sampling.** Each person track is resampled every 0.4 s, 8 samples back (3.2 s), using a local straight-line fit over ±0.2 s. That smoothing removes detector jitter without lagging at the newest sample.
- **Coordinates.** Calibrated cameras use ground metres. Uncalibrated cameras use image pixels, converted to rough metres by assuming the person's box is 1.7 m tall (`approx_scale: true`). Tracks with less than 2.8 s of history use constant velocity.
- **Model.** A GRU with 25.5 k parameters works in a heading-aligned frame. It predicts a *residual on top of constant velocity* plus a per-step spread (σ), trained with an L2 + Gaussian-NLL loss. Inference on the laptop CPU (1 thread) takes 1.6 ms for 1 track and 2.3 ms for 20 tracks per update. It runs at 2.5 Hz.
- **Without a checkpoint** the forecaster falls back to constant velocity (`"method": "cv"`). Its σ then comes from the constant-velocity error measured on ETH/UCY.
- **Training.** `tools/train_forecast.py` downloads ETH (BIWI eth, hotel) and UCY (univ, zara1, zara2): about 10 MB of world-metre trajectories, using the leave-one-scene-out splits redistributed in the Trajectron++ repository (MIT-licensed repository; the data is for research use per its authors). It trains on CPU in about 18 minutes for all five folds plus the final model. If the download fails, it trains on **synthetic walkers** and says so in the checkpoint metadata. The final all-scenes model is bundled at `assets/forecast/argonix_forecast_gru.pt` (108 KB). A checkpoint in `data/models/argonix_forecast_gru.pt` takes precedence.
- **Predicted zone entry** (`"zone_entry": {"enabled": true}`, off by default). It applies to a person *outside* a zone that has the intrusion rule. The event fires when their forecast enters the zone within 3.2 s with a Monte-Carlo probability of at least 0.5, on 2 consecutive updates. The event `predicted_zone_entry` has severity 1 and sets `learned_model`. The correlator therefore **caps it at P3** until the deterministic intrusion rule fires in the same zone. It shares the zone's `site_zone`, so a real entry corroborates it.

**API.** Both endpoints need `alert.read` and return 404 when forecasting is off.
- `GET /api/forecast/tracks?camera=`: per track, `path` and `sigma` in the rule engine's units, plus `path_px`/`sigma_px` in analytics pixels and the `method` used.
- `GET /api/forecast/status`: model, the LOSO metrics stored in the checkpoint, and warnings.

**UI.** Live tiles show each track's forecast as a faint dashed path. Spread circles at 1.6, 3.2 and 4.8 s grow and fade with the horizon; paths are blue for the GRU and grey for constant velocity. A "paths: forecast (experimental)" tag marks the tile.

**Measured: ETH/UCY, leave-one-scene-out.** Standard protocol: 8 observed → 12 predicted steps at 0.4 s, every window at stride 1, ADE/FDE in metres. Each fold picks its epoch on its val split. Measured 2026-09-25 on this laptop's CPU (AMD Family 23 Model 160, torch 2.14 CPU, 4 threads).

| Held-out scene | windows | GRU ADE / FDE | Constant velocity ADE / FDE | GRU 3.2 s ADE / FDE | CV 3.2 s ADE / FDE |
|---|---|---|---|---|---|
| ETH | 364 | **1.014 / 2.074** | 1.075 / 2.282 | 0.645 / 1.229 | 0.662 / 1.302 |
| Hotel | 1,197 | 0.346 / 0.662 | **0.319 / 0.614** | 0.234 / 0.424 | 0.215 / 0.390 |
| Univ | 24,334 | **0.514 / 1.131** | 0.524 / 1.165 | 0.299 / 0.639 | 0.303 / 0.651 |
| Zara1 | 2,356 | 0.437 / 0.982 | **0.427 / 0.952** | 0.252 / 0.539 | 0.248 / 0.529 |
| Zara2 | 5,910 | 0.321 / 0.715 | 0.324 / 0.724 | 0.185 / 0.398 | 0.186 / 0.402 |
| **Average of the 5 folds** | | **0.526 / 1.113** | 0.534 / 1.147 | 0.323 / 0.646 | 0.323 / 0.655 |

**Honest reading.** The small GRU is **roughly tied with constant velocity**: 1.5 % better ADE and 3 % better FDE on average. It wins on ETH and Univ, is level on Zara2, and loses on Hotel and Zara1. The 3.2 s averages are equal.

The predicted spread is **not well calibrated**. The ideal 2D coverage is 39 % of true positions within 1σ. The GRU covers 29 % on ETH (too narrow) and 53–77 % on the other scenes (too wide).

Do not treat these forecasts as more than a smooth extrapolation with a rough uncertainty band.

**Measured: transfer to Argus's own tracks.** `tools/eval_forecast_real.py` runs YOLO26n at imgsz 480 + ByteTrack at 5 fps and scores the forecast against the tracker's later ground positions, which include detector jitter, ID switches and homography error. See the table below.

| Camera (calibration error) | windows | GRU ADE / FDE | CV ADE / FDE | "Stays put" ADE / FDE |
|---|---|---|---|---|
| `cam_campus`, OpenCV vtest (±1.0 m) | 262 | **0.935 / 1.829** | 0.999 / 2.034 | 1.726 / 3.023 |
| `cam_street`, Kaggle clip (±1.5 m) | 9 | 0.190 / 0.889 | 0.191 / 0.879 | 0.143 / 0.829 |

On the campus clip the GRU is 6 % better on ADE and 10 % better on FDE than constant velocity: one clip, one hand calibration. The street clip gives only 9 windows (short, fragmented tracks), which is too few to read anything into.

Commands:

```bash
uv run python tools/train_forecast.py                       # ETH/UCY LOSO folds + final model (~18 min on CPU)
uv run python tools/eval_forecast_real.py                   # GRU vs CV on our own tracker output
uv run python run.py --config configs/real_site_experimental.json --data data/run_exp --port 8003
```

**Limitations.**
- Single-mode: it cannot represent "turns left *or* right".
- It uses no scene context and no other people.
- ETH/UCY are overhead, 25 fps annotations; our tracks are 5 fps detector foot points.
- Uncalibrated cameras use a body-height scale that ignores perspective along the path.
- Predicted-zone-entry is an early hint for an operator and never a primary alarm.

## Fall detection: person on the ground (experimental)

**Off by default.** The project manual advises against fall detection for a hackathon demo. It is built here on request, disabled unless configured, and labelled experimental everywhere.

The event is **"Person on the ground / possible fall"** (`person_down`, severity 3). Its text describes **posture only**, for example "posture changed from upright to horizontal within 0.8 s … and has stayed horizontal near the ground for 3.2 s". It never makes claims about injury, health, cause or intent, and it says that an operator must review the clip.

Enable it with `"fall": {"enabled": true, ...}` (see `configs/real_site_experimental.json`). It is implemented in `argonix/fall.py`.

**Method: transparent per-track state machine**
- **Inputs.** The tracker box, plus YOLO26n-pose keypoints matched to the track by IoU. Pose runs at 5 fps, and only while someone's box is wider than 0.6 × its height or a track is already "down". When the altercation analyzer is on, fall detection **shares its pose model** through a per-frame cache, so the model runs once per frame.
- **Upright:** box w/h ≤ 0.75, and the torso is within 35° of vertical when keypoints are available. The median upright box height is the person's reference height.
- **Lying:** all of the following hold.
  - box w/h ≥ 1.0
  - box height ≤ 0.65 × reference height
  - with keypoints: torso ≥ 55° from vertical, and hips within 0.3 reference heights of the box bottom

  These conditions rule out sitting on the floor, where the torso is upright, and bending over, where the hips stay high. A box touching the image edge is ignored, because truncation is not posture.
- **Event.** A rapid descent must come first: the box top drops at least 0.35 reference heights within 2 s of the lying onset. The lying posture must then persist for at least 3 s, with up to 1 s of flicker tolerated. It fires once per episode, with a 60 s re-arm per track. Getting up for 1 s resets it.
- **Without a rapid descent.** `report_without_descent: true` also reports lying down slowly, worded "no rapid descent was observed". This is off by default.
- **Missing pose model.** If the pose model cannot load, the detector runs **boxes-only** and a warning is shown at `GET /api/fall/status`. Boxes-only mode cannot tell a long deep bend from lying down.
- **Optional classifier hook** (`"classifier": "<checkpoint>"`, a small GRU over the shared `track_features`). **None is trained or shipped.** When the classifier agrees with the rule, confidence is raised. An event from the classifier *alone* sets `learned_model`, so the correlator caps it at P3.

**Measured** (`tools/eval_fall.py`, YOLO26n imgsz 480 + ByteTrack + YOLO26n-pose, 5 fps, laptop CPU, 2026-09-25). This is a small sanity check, not a benchmark.

| Footage | Duration | Expected | Events |
|---|---|---|---|
| CAVIAR `Rest_FallOnFloor`, `Rest_SlumpOnFloor`, `Rest_WiggleOnFloor` (384×288 lobby camera) | 129 s | a person ends up on the floor (3 clips) | **0 of 3 detected** |
| Normal walking footage: CAVIAR Walk1–3 and Browse1, OpenCV vtest, the Kaggle street clip | 4.6 min | no fall | 0 false alarms |
| Synthetic box + keypoint sequences (`tests/test_fall.py`) | | fall vs sitting on a chair or the floor, bending, slow lying down, getting up, image-edge truncation | behaves as designed |

**Why 0 of 3 on CAVIAR.** A frame-by-frame dump shows that YOLO26n barely detects the small (~40 px) walking person in these 384×288 clips, even at imgsz 800 and confidence 0.12. The person on the floor then appears as a **new track with no upright history**. Seen from the high wide-angle camera, the lying body's box stays at w/h ≈ 0.9–1.1. So the precondition, a tracked upright person who then goes horizontal, never occurs.

On this footage the real limit is the detector's recall and the viewpoint, not the thresholds. The thresholds were not tuned to these clips. Treat the feature as **unvalidated on real falls**. The zero false alarms come from only 4.6 minutes of ordinary walking.

```bash
uv run python tools/eval_fall.py --download                 # fetch the 3 CAVIAR "on floor" clips (~37 MB), then run
```

**Limitations.**
- Everything upstream matters. A person the detector misses, or whose track ID changes while falling, has no upright reference, so nothing fires.
- High, wide-angle cameras foreshorten a lying body, so the box may not become wider than it is tall.
- Occlusion by furniture hides the posture.
- It has not been validated on greyscale/IR footage beyond the pipeline's generic greyscale conditioning.
- Verify every event on the clip.

## Weapon-like objects and possible altercation (experimental)

**Both off by default.** Enable them through `"threats"` (see `configs/real_site_gpu_threats.json` and `argonix/threat_install.py`). Both are learned models, so the correlator caps them at **P3** until a deterministic rule agrees.

The events describe the **object** ("possible weapon-like object: handgun, near track 12") or the **interaction** ("possible physical altercation between tracks 4 and 9"). They never label a person. Every event must be verified on the clip.

**Weapon-like objects** (`argonix/threats.py`)
- Two models run at 3 fps on frames and on person crops, and are merged by NMS:
  - the COCO YOLO26n classes knife, bat and scissors;
  - a fine-tuned YOLO26n for knife and handgun (`tools/train_weapons.py`).
- An event needs *k* of *n* detections (3 of 5) at confidence ≥ 0.5, with a person nearby.
- **Training data:** Kaggle `raghavnanjappan/weapon-dataset-for-yolov5`, 4,154 images. Exact and near-duplicates (dHash) are grouped so they never straddle splits: 3,197 train, 325 val, 324 test. The licence is "unknown", so this is personal demo use only; don't redistribute the weights.
- **Training:** 50 epochs at 640 px, batch 16, 95 min on the RTX 4050 Laptop (2026-09-25). The weights are `data/models/argonix_weapons_yolo26n.pt` (not in git; `v1-9ae9b3aa`).

**Measured** (`tools/eval_threats.py`, RTX 4050 Laptop, 2026-09-25)

| Test | Fine-tuned YOLO26n | COCO YOLO26n (zero training) |
|---|---|---|
| Held-out test split, handgun (250 boxes): AP50 · precision/recall at conf 0.5 | **0.906** · 0.94 / 0.78 | 0.000 |
| Held-out test split, knife (124 boxes): AP50 · precision/recall at conf 0.5 | **0.605** · 0.70 / 0.42 | 0.082 |
| Ultralytics `val` on the test split: mAP50 / mAP50-95 | 0.753 / 0.503 | — |
| Synthetic CCTV views (`simuletic/cctv-weapon-dataset`, 130 weapons, class-agnostic): AP50 · P/R | 0.058 · 0.20 / 0.08 | 0.004 |
| Normal footage with no weapons, full runtime path, both models as configured: events/hour | RWF-2000 val NonFight, 200 clips (16.7 min): **137/h**, 29 of 200 clips had an event · campus + street clips: 49–131/h (1–2 min each, very wide intervals) | |

**What this means.** On images like its training data, the fine-tuned model finds handguns well and knives moderately, where the COCO model finds almost nothing. It does **not** generalise to CCTV-style views: recall is 8 % on the synthetic CCTV set. On ordinary surveillance footage it raises roughly **one false event every 30 seconds**. It is useful only as a demo of the corroboration-only design, not as a detector, and it must stay off by default.

**Against a third-party pretrained model** (`tools/compare_pretrained.py`, RTX 4050 Laptop, 2026-09-25). The comparison model is a public YOLOv8n threat detector (Hugging Face `Subh775`, classes Gun / explosion / grenade / knife; "Gun" is scored as handgun although it also covers rifles). Its training images are unknown and may overlap our test split, which would favour it.

| Test | Ours (fine-tuned YOLO26n) | Third-party YOLOv8n |
|---|---|---|
| Held-out test split, handgun: AP50 · P/R at conf 0.5 | **0.906** · 0.94 / 0.78 | 0.685 · 0.85 / 0.57 |
| Held-out test split, knife: AP50 · P/R at conf 0.5 | **0.605** · 0.70 / 0.42 | 0.525 · 0.66 / 0.54 |
| Synthetic CCTV views, class-agnostic: AP50 · P/R | 0.058 · 0.20 / 0.08 | 0.055 · 0.60 / 0.02 |
| RWF-2000 val NonFight, 200 clips, full runtime path (with the COCO model): events/hour (95 % upper) | 151 (196), 31 clips | **94** (130), 22 clips |

Neither model is usable on CCTV-style footage. The third-party model is quieter on normal footage but weaker on the test split. The campus/street clips gave identical counts with both models (4 and 1 events), so those events come from the COCO branch.

**Possible altercation** (`argonix/altercation.py`): pose (YOLO26n-pose) for pairs of nearby people. A transparent heuristic score is optionally corroborated by a small pair-GRU (37k parameters) trained with multiple-instance learning on RWF-2000 pose windows (`tools/train_altercation.py`; 60 epochs, best at 34 by val AUC). The weights are `data/models/argonix_altercation_gru.pt` (not in git; `v1-9ddf25dd`).

**Measured** (RTX 4050 Laptop, 2026-09-25). Clip level: a clip counts as a fight if any pair window scores above the threshold chosen on val. 1,353 train clips gave 22,944 windows, but many clips produced no pair window at all (198 of 400 test clips, 756 of 1,000 hockey clips); those count as "no fight".

| Test (clip level) | Pair-GRU | Heuristic only |
|---|---|---|
| RWF-2000 val, 247 clips: ROC AUC · P/R · FPR | 0.611 · 0.87 / 0.36 · 6 % | 0.584 · 0.71 / 0.38 · 17 % |
| RWF-2000 test, 400 clips: ROC AUC · P/R · FPR | **0.681** · 0.74 / 0.53 · 19 % | 0.647 · 0.67 / 0.53 · 26 % |
| Hockey Fight (out of domain), 1,000 clips: ROC AUC · P/R · FPR | 0.656 · 0.85 / 0.33 · 6 % | 0.646 · 0.85 / 0.31 · 6 % |

| Full runtime path (`tools/eval_threats.py normal --only altercation`, k-of-n gating) | Pair-GRU | Heuristic only |
|---|---|---|
| RWF-2000 val NonFight, 200 clips (16.7 min): events/hour (95 % upper), clips with an event | 21.6 (42.6), 5 | **0** (10.7), 0 |
| RWF-2000 val Fight, 200 clips: clips with an event | 11 of 200 | 1 of 200 |
| Campus + street clips (3 min in total) | 0 events | 0 events |

**What this means.** The GRU is only slightly better than the heuristic (AUC 0.68 vs 0.65 on RWF test) and both are far from a useful fight detector. In the runtime path the heuristic almost never fires, and the GRU catches 11 of 200 fight clips at the cost of about one false event every 3 minutes on non-fight footage. Keep it off by default; it is a demo of learned corroboration capped at P3.

**Against a third-party frame classifier** (`tools/compare_pretrained.py vit`, RTX 4050 Laptop, 2026-09-26). The comparison model is `jaranohaal/vit-base-violence-detection` (ViT-B/16 trained on Kaggle "Real Life Violence Situations", so RWF-2000 and Hockey are out-of-sample for it). It scores frames sampled at 2 fps; the clip score is the maximum (or mean) P(violent).
- **Loading.** The published checkpoint stores timm-format weights under a transformers config, so `from_pretrained` silently skipped every weight. The first run therefore scored a random network, and its results are withdrawn. The script now converts the weights and refuses to run unless every tensor loads.
- **Which output is "violent".** The config names its outputs only `LABEL_0` / `LABEL_1`. The direction was chosen on 102 RWF-2000 **train** clips, which none of the scored sets use: output 1 gives AUC 0.61 there (output 0 gives 0.39).

| Test | ViT, clip = max · mean ROC AUC | At P ≥ 0.5 (max): recall / false-positive rate | Our pair-GRU (above) |
|---|---|---|---|
| RWF-2000 val, 352 clips | 0.63 · 0.59 | 0.89 / 0.77 | 0.61 (val) · 0.68 (test) |
| Hockey Fight, 1,000 clips | 0.48 · 0.59 | 0.99 / 0.97 | 0.66 |
| Normal footage (5 clips, 3 min) | 123 of 371 frames ≥ 0.5, about 315 onsets per hour | | 21.6 events/h in the runtime path |
| Speed | 2.4 ms per frame | | |

**What this means.** The ViT ranks fight clips about as well as our pose model on RWF-2000 and worse on Hockey, and at its natural threshold it flags almost everything, including a third of the frames of ordinary street footage. Neither is usable as a fight detector on CCTV-style video. Ours has the advantage of working per pair of people, so its events can say *which* people are involved.

## Schedules, line events and vehicles

All three are deterministic rules (no learned model) and are off until configured.

**Schedules** (`argonix/schedule.py`). Add `"schedule"` to a zone to arm all its rules only inside time windows, or to one rule's parameters to override the zone for that rule:

```json
"schedule": {"tz": "Asia/Kolkata", "windows": ["mon-fri 19:00-07:00", "sat,sun 00:00-24:00"], "label": "after-hours"}
```

- A window is `<days> HH:MM-HH:MM`. Days can be `mon`…`sun`, lists (`sat,sun`), ranges (`mon-fri`, or wrapping `fri-mon`), `weekdays`, `weekends` or `daily`. Leaving the days out means daily. `24:00` is allowed as an end time.
- **Overnight windows wrap.** `mon-fri 19:00-07:00` runs from Mon 19:00 to Tue 07:00, and so on to Fri 19:00 to Sat 07:00. Mon 03:00 is not inside it, because that time belongs to a Sunday window. The start time is inside the window and the end time is not.
- `tz` takes an IANA name, `UTC`, or a fixed offset such as `+05:30`. If it is left out, the server's local time is used. On Windows without the `tzdata` package, the no-DST zones in `schedule.FIXED_OFFSETS` (Asia/Kolkata, Asia/Dubai, Asia/Singapore, …) fall back to their fixed offset. Any other IANA name is then a configuration error.
- A person already inside when the window opens starts counting `min_seconds` from the opening time. When the window closes, open episodes end.
- **Events say which window was active.** The derivation carries a `schedule` field (window, label, tz, local time and source: zone, rule or line), and the observation ends with, for example, `[after-hours window mon-fri 19:00-07:00]`.
- **Validation.** Schedules are checked when the config loads and in `PUT /zones` and `/lines`, and a bad one fails with a message naming the zone, rule or line. `POST /api/schedule/check {"schedule": …}` returns validity and whether the schedule is active now.
- **Editor.** The zone editor has a schedule box (windows, time zone and label) that shows "Active now" or "Inactive now" from the server clock. `/api/analytics/zones` reports each zone's schedule status.

**Line-crossing events** (`argonix/flow.py`). Counting lines count crossings in both directions by default. Adding `"events"` to a line also raises a `line_cross` event (severity 1) for each crossing, for example "person track #12 (cam1) crossed line 'Door 1' inbound ('entering') [after-hours window …]":

```json
{"id": "door1", "name": "Door 1", "points_px": [[..], [..]], "in_label": "entering",
 "events": {"direction": "in", "min_gap_s": 5, "criticality": 2, "schedule": {…}}}
```

- `direction` is `in`, `out` or `both`.
- **Gap.** At most one event is raised per `min_gap_s` per line. Matching crossings inside the gap are still counted, and are reported in the next event as `crossings_suppressed_by_gap`. A group walking through a door therefore raises one alert.
- **Schedule.** Outside the line's schedule, crossings are counted but not raised.
- **Editor.** The line editor has an "events" box with direction, gap, criticality and schedule.
- Only person tracks are counted. The event's `zone_id` is `line:<id>`.

**Vehicles** (opt-in: `"vehicles": {"enabled": true, "classes": [2, 3, 5, 7], "detector_conf": 0.35}`)

- **Detection.** The detector adds COCO car, motorcycle, bus and truck to `detector.extra_classes`. They are tracked by the second (object) tracker, which is shared with the unattended-object bags, with IDs offset by 1,000,000. Person tracking, person counts, crowding, flow, lines, baselines and the heatmap stay person-only.
- **Rules.** `vehicle_in_zone` fires when a vehicle is in a pedestrian-only zone. `vehicle_stopped` fires when a vehicle is stationary in a no-stopping zone. Both work on uncalibrated cameras:
  - calibrated: stillness is ground displacement in metres;
  - uncalibrated: stillness is the box-centre displacement divided by the box's larger side.
- **Positions.** A vehicle's position is the bottom centre of its box, which is only approximate on the ground plane.
- **Annotated frames.** Vehicles are drawn in blue-cyan, distinct from people (green) and alerting tracks (red).
- **Remote inference node.** The node's class list is fixed when it starts. Start it with `--extra-classes 2,3,5,7` (add `24,26,28` if the site also uses unattended objects), or it returns no vehicles. The site logs a warning at start-up when the node lacks the classes. The local CPU fallback detects them itself.
- **Confidence threshold.** Objects and vehicles share one extra-class confidence threshold (the lowest `detector_conf` requested).
- **Not measured.** No vehicle-specific evaluation has been run. The COCO detector's accuracy on CCTV vehicles in this deployment has not been measured.

Tests: `tests/test_rule_additions.py` covers schedule parsing and wrap-around, rule gating, line events with direction, gap and schedule, the vehicle rules in metres and pixels, and API validation.

## Operator workflow: alert states, acknowledge SLA, flood banner, auto-open cameras

**Alert state machine** (`argonix/workflow.py`). `POST /api/alerts/{id}/status` accepts only these steps:

| From | To |
|---|---|
| new | acknowledged · escalated · dismissed (with a disposition) |
| acknowledged | escalated · resolved / dismissed (with a disposition) |
| escalated | acknowledged · resolved (with a disposition) |
| resolved, dismissed | new or acknowledged: **reopen**, needs `incident.assign`, `incident.merge` or `user.admin` (supervisor or admin) |

- An illegal step returns **409** with the legal next steps; a missing permission returns 403. A missing disposition or escalation reason returns 400.
- **Escalating needs a `reason`.** `to_role` (a role name) and `to_user` are optional. The escalation is stored on the alert and in the audit trail. Nothing is sent from the console.
- **`unclear` disposition:** `{"status": "acknowledged", "disposition": "unclear"}` keeps the alert acknowledged and sets `second_review`. Review → *Second review* lists these alerts (`GET /api/alerts?review=second`). Resolving or dismissing with a final disposition clears the flag and records who did the second review. `unclear` never counts toward rule precision.
- The drawer offers only the legal buttons for your role; Escalate opens a reason dialog. When an alert is re-correlated, the pipeline keeps every operator field (status, note, disposition, escalation, flags).

**Acknowledge SLA.** Set it in the site config: `"sla_s": {"P1": 60, "P2": 300, "P3": 1800, "P4": null}` (these are the defaults).
- Review cards, the drawer and the bell show how long a new alert has waited, or "overdue +…". The Review nav item gets a red clock badge counting alerts past their SLA.
- In `IncidentManager.tick`, a P1/P2 incident whose alerts are all still `new` past the SLA gets **one** `sla_overdue` escalation:
  - a notification row, exempt from the push budget (camera-integrity inhibition still mutes it);
  - the signed webhook again, if configured;
  - a timeline entry and an audit record (`incident_sla_overdue`).
- The clock starts when the incident reached its current priority. Any alert moved out of `new`, or an incident status other than `open`, counts as acknowledged (`incident.sla.state = "acknowledged"`).

**Alert flood banner.** When more than 10 new alerts arrive within 10 minutes (`"alert_flood": {"threshold": 10, "window_s": 600}`), the shell shows a dismissible banner, for example "Alert flood: 14 new in 10 min — showing P1/P2 first". Its one-click **Compact · P1–P2 only** opens Review as a dense list of new P1/P2 alerts. During a flood, Review puts new P1/P2 alerts first. `priority` and `status` filters accept comma lists (`?priority=P1,P2`).

**Auto-open cameras on P1/P2.** The WebSocket now carries `{"type": "incident"}` messages with the compact incident view. While the connection is down, the dashboard polls `/api/incidents` instead.
- A new or escalated P1/P2 incident, or one that missed its SLA, raises a toast with **Open cameras**. Suppressed and closed incidents raise no toast.
- With **Auto-open cameras on P1/P2** switched on (user menu or the Live toolbar; stored per browser), Live switches to `#/live?incident=<id>` by itself. That view is a focus layout with the incident's primary camera (the newest event's camera) large, its related cameras beside it, and up to two **predicted next** cameras from the incident's handoff estimates. The predicted cameras are dashed and labelled as estimates.
- Auto-open never fires while a dialog is open or you are typing.

## Evidence integrity, retention and privacy

**Clean evidence frames.** Each event saves two snapshots at the same (analytics) resolution:
- `evidence/<stem>_raw.jpg` (`evidence.snapshot_raw`) is the frame as analysed (after B/W conditioning; `evidence.enhanced` says whether CLAHE was on) with nothing drawn on it.
- `evidence/<stem>.jpg` (`evidence.snapshot`) and the clip carry boxes, zones and labels, stamped **"software overlay"** in the bottom-left corner.

The alert drawer switches between **Original** and **With overlay**. Both go into the export bundle, and the manifest labels each file with its `kind`. Recording segments were already overlay-free. `evidence.person_boxes` keeps the person boxes of that frame; the anonymiser uses them.

**Export bundle** (`GET /api/alerts/{id}/export[?anonymise=1]`, `evidence.export`).
- The SHA-256 hash-chained manifest (`verify_bundle`) now also rejects files that are not listed in it.
- It carries `audit_chain`: the audit log's head at export time, so a later truncation of the log can be detected against the bundle.
- The chain alone only shows accidental damage: whoever edits a file can recompute it. So the chain head is also **HMAC-SHA256-signed with the server's key** (`manifest.signature`). `POST /api/evidence/verify` (raw zip body, `evidence.export`, audited) returns `chain_ok` and `signature_ok`; only the server that exported a bundle (or anyone given its `secret.key`) can check the signature.
- Evidence the alert references but that is missing (a clip not written yet, a deleted file) or that resolves outside the data folder is listed under `manifest.missing` instead of being silently left out.
- `?anonymise=1` (the "anonymised copies" checkbox in the drawer) adds blurred copies under `anonymised/`, of kind **"anonymised copy (derived)"**, each with `derived_from`. The originals stay in the bundle as the evidence record.

**What gets blurred** (`argonix/privacy.py`):
- the top 22 % of every recorded person box (the head region);
- faces found by OpenCV's Haar frontal-face cascade.

The OpenCV 5 wheel in `uv.lock` ships **no Haar cascades**. In that case a local YuNet model (`"privacy": {"face_model": "data/models/face_detection_yunet_2023mar.onnx"}`) is used if you supply one; nothing is downloaded. With neither detector, only head regions are blurred, clip frames are not anonymised, and the manifest says so. This is best-effort, not a guarantee.

`"privacy": {"blur_faces_for_vlm": true}` applies the same blurring to every image before it is sent to the VLM server. The explanation then ends with "faces blurred before sending".

**Retention** (`argonix/retention.py`, a background thread in live runs). Config and defaults:
```json
"retention": {"recordings_days": 7, "evidence_days": 30, "max_disk_gb": 20, "alerts_days": 90, "interval_min": 10}
```
- Age limits apply first. Then, while recordings and evidence together exceed the cap, the oldest file goes first. `0` disables a limit.
- Closed (resolved/dismissed) alerts older than `alerts_days` are deleted. Open alerts are never deleted, and neither is the audit log.
- Files modified in the last 5 min are never touched (segments and clips still being written).
- Every sweep is audited as `retention_swept`, with counts and bytes. The System page shows the policy and the last sweep (`GET /api/retention`). An admin can force a sweep with `POST /api/retention/sweep`.

**Legal hold.** `POST /api/alerts/{id}/hold` places a hold and `DELETE` releases it. Both need `evidence.hold` (supervisor) and both are audited.
- A held alert, its evidence files, and the recording segments of its cameras overlapping its time window (±60 s) are never deleted.
- The drawer shows a "Legal hold" badge and a hold/release button.

**Tamper-evident audit log.** Every audit row stores `prev_hash` and `hash = sha256(prev_hash + canonical JSON of the row)`.
- Older databases are migrated on start: the columns are added and the existing rows are chained once.
- `GET /api/audit/verify` (`audit.read`) recomputes the chain and returns `{ok, rows, first_bad_seq, head}`. The System page's audit card shows "chain intact ✓" or "chain broken at #n".
- Editing, inserting or deleting any row is detected at that row. Deleting only the newest rows is detectable only against a head kept elsewhere, such as the head in an export bundle. The chain gives no protection against someone who rewrites the whole database *and* every copy of the head.

## Synchronised incident replay

The incident view has a **Synchronised replay** panel. It plays every camera involved in the incident, plus any predicted next camera (tagged as an estimate), against **one master clock**. The window runs from before the first event to after the last (±10/30/60 s).

- **Data:** `GET /api/incidents/{id}/replay?pre_s=10&post_s=10[&cameras=a,b]` (needs `evidence.view`; `argonix/replay.py`). For each camera it returns:
  - the 60 s recording segments overlapping the window, with each segment's absolute start and end;
  - the incident's evidence clips (annotated; timing approximate, about 4 s before to 4 s after the event frame);
  - the gaps with no stored video;
  - the segment still being written (playable once it closes).
  The response also includes the incident's event markers.
- **Player** (`static/js/replay.js`):
  - A `requestAnimationFrame` clock runs at 0.5×, 1× or 2×.
  - Each `<video>` seeks to *(master time − segment start)*, scaled to the file's real duration: a segment the pipeline wrote with dropped frames is shorter than its wall-clock span.
  - Players switch segments at the boundaries and re-seek when they drift by more than 250 ms.
  - A file that would need a rate below the browser minimum is stepped by seeking instead.
  - The clock waits up to 2.5 s for a buffering player.
- **Timeline:** coverage lanes per camera. Event markers (click to jump 2 s before the event), a scrubber, and absolute time labels on each tile. Gaps show "No recording at …".
- Recordings are the unannotated analytics-resolution stream. Main-stream recording is not part of this feature.

## Zone congestion forecast (a prediction, not an observation)

The Analytics page's **Congestion forecast** panel shows each zone's observed occupancy for the last hour, followed by a forecast for the next 15, 30 or 60 minutes. For a queue zone the occupancy is the queue length. The forecast is a dashed line with a shaded 80% interval, labelled "forecast — prediction, not observation". The API is `GET /api/analytics/forecast?zone=cam:zone&horizon_min=30`; without `zone` it returns every zone (`argonix/congestion.py`).

**Model** (statistical; nothing is trained or stored). It uses the 1-minute occupancy that the seasonal baseline already keeps in `baseline.db`.
- **Seasonal-naive:** the hour-of-week median from earlier data, interpolated between hour centres, plus a damped Holt smoothing of the last 60 min of the deviation from it.
- **Weighting:** the deviation's weight per horizon is fitted on the **older half** of a rolling-origin backtest over the last 7 days.
- **Interval:** the 80% interval comes from the older half's errors. Its coverage is measured on the **newer half**.

**Skill gate.** The forecast is shown only when, on the held-out newer half (≥ 30 origins), its MAE beats **both** the last value (naive) and the usual-for-this-hour value (seasonal-naive) by at least 5%. Otherwise the panel says "insufficient history" or "insufficient skill" and draws no band. The seasonal reference needs roughly a week of history before each hour of the week, so a fresh demo site shows "insufficient history". The backtest table is shown per zone, next to the chart.

**Congestion note.** When the zone has a limit (`crowding.max_count`, or `queue_long.max_length`) and the upper 80% bound crosses it, the panel shows "congestion expected in ~N min". If the zone is already above the limit, it says so as an observation; that case is the deterministic rule's job.
- **Optional event:** `"congestion_forecast": {"enabled": true, "emit_events": true}` also records a `congestion_forecast` event.
  - It has severity 1 and is marked `prediction` / `learned_model`, so the correlator caps it at P3.
  - It fires once per zone per 30 min.
  - Events are **off by default**. `enabled: false` removes the API.

**Measured** (synthetic only, `tests/test_congestion.py`; no real-site history exists yet). The test series has 21 days of minute data: a daily cycle, an evening peak, a persistent AR(1) deviation and noise. At 30 min on held-out origins:

| Series | Forecast MAE | Last value | Usual for hour | Result |
|---|---|---|---|---|
| Daily cycle + AR(1) deviation + noise | 2.81 | 3.20 | 3.11 | Shown |
| White noise | 1.65 | 2.20 | 1.63 | Gated: no skill |

The 80% interval covered 71–81% on held-out origins. These are synthetic numbers and say nothing about a real site.

## Incident report, assistive incident summary and shift handover

**Incident report** (`argonix/report.py`; **Report** button on an incident): `GET /api/incidents/{id}/report` returns one self-contained HTML page. CSS is inline, and the evidence thumbnails and site map are embedded as `data:` URIs, so the page can be saved or mailed as a single file. For a PDF, use the browser's **Print → Save as PDF**: the print CSS lays it out on A4 and hides the toolbar. No PDF library is involved. The report contains:
- a header (site, incident, status/phase, priority, times, who generated it and when) and the automatic, operator and assistive summaries, each labelled;
- **Why this priority**: the correlator's `explain_priority()` factors, recomputed from the deciding alert's stored events;
- a timeline with **observation** (what was seen) and **derivation** (why the rule fired) in separate columns, and confidences labelled heuristic/uncalibrated (or learned model);
- evidence thumbnails (annotated frames);
- the bird's-eye site map with the incident's zones and event foot points drawn in (2D ground-plane approximation);
- involved cameras with their current health and calibration;
- dispositions and push decisions, and the merged audit trail of the incident and its alerts;
- any VLM text, labelled assistive;
- an **Integrity** section with the SHA-256 of every evidence file as stored.

The footer carries the SHA-256 of the report body, over the UTF-8 bytes between the `ARGONIX-REPORT-BODY-START` and `…-END` markers inclusive. The same value is sent in `X-Report-SHA256` and recorded in the audit entry `incident_report_generated`. `argonix.report.verify_report(html)` checks a saved copy. Permission: `incident.read`. The page is served with its own strict CSP (no network access; one hashed inline script for the print button).

**Assistive multi-camera summary** (`argonix/narrative.py`; **Summarise** button in the incident's Summary panel): `POST /api/incidents/{id}/summary`.
- **Input:** the incident's alert records (ids, times, cameras, zones, rule labels, observations) plus up to 4 keyframes. The keyframes are evidence snapshots spread round-robin over cameras and sent in time order to the VLM server's multi-image `/explain`. Without snapshots or a vision server, the text model gets the records only (`/chat`).
- **Output:** a short chronological narrative that must cite alert ids like `[A…]`.
- **Validation:** the investigation's cite-or-refuse check (`investigate.check_summary`). The text needs at least one citation, every cited id must belong to this incident, and person-labelling words are refused. If the check fails, the model is not configured or the call fails, a deterministic narrative is built from the alert records instead, and the reason is stored.
- **Storage:** the result goes on the incident as `assistive_summary` and is labelled "Assistive summary — model X, not evidence". It never changes priority, phase or status. Cited ids render as links.
- **Permission and audit:** `alert.acknowledge`, the same as the per-alert VLM explain (there is no separate `alert.explain` permission). Audited as `incident_summary_generated`, with engine, model, citations, rejection reason and keyframes.

**Shift handover** (`argonix/handover.py`; **Handover** page):
- **Summary:** `GET /api/handover/summary?hours=8` (or `since=<epoch s>`) lists what is unresolved at shift end:
  - open alerts (new, acknowledged or escalated) and incidents (open, acknowledged or investigating) in the window;
  - **overdue** items still unacknowledged past their priority's target (P1 5 min, P2 15, P3 60, P4 240; override with config `"handover": {"ack_sla_min": {...}}`);
  - camera problems now (DEGRADED/DOWN, stream error or reconnecting) and camera-integrity alerts in the shift;
  - the top zones.
- **Notes:** `POST /api/handover {text}` (permission `feedback.write`, audited as `handover_note`) stores a note with its author, the time and a snapshot of the open and overdue ids. `GET /api/handover` lists the notes (`alert.read`).
- **Print view:** `GET /api/handover/print` is a printable sheet with the same integrity footer, audited as `handover_printed`.

Notes live in the site's SQLite database (table `handover_notes`). Tests: `tests/test_reports.py`. The report, summary validation and handover were tested with a fake VLM on synthetic events. No VLM narrative quality has been measured yet.

## Smooth live view (WebRTC) and main-stream recording (optional)

**Off by default.** Without it, live view is the annotated analytics MJPEG (~5 fps) and recordings are the analytics-resolution stream. With `"restream"` enabled, a [MediaMTX](https://github.com/bluenviron/mediamtx) instance (MIT) managed by Argus (`argonix/restream.py`) pulls each camera's **main stream** once and adds two features:
- **Smooth live view**: WebRTC at the camera's full frame rate. The camera's own H.264 is forwarded without transcoding.
- **Main-stream recording**: fragmented-MP4 segments. This follows manual §15 V-01/V-02/V-03 and §03 R-01.

"restream": {"enabled": true, "mediamtx_bin": "C:/…/mediamtx.exe", "api": "http://127.0.0.1:9997",
             "webrtc": "http://127.0.0.1:8889", "webrtc_udp_port": 8189, "webrtc_udp_host": "127.0.0.1", "webrtc_additional_hosts": [],
             "record": {"enabled": true, "path": "<data>/main_rec", "segment_s": 60, "keep_days": 7}}
Per camera, `"main_source"` is the full-resolution RTSP URL. It defaults to `source`. The add-camera wizard fills it from the vendor template when you pick the sub-stream, and the camera edit dialog can set or clear it. Like `source`, it is always redacted in the API.

**How it works**
- **One path per camera.** For every enabled network camera (RTSP/RTMP, or an HLS `.m3u8`), Argus creates the MediaMTX path `cam_<id>` through MediaMTX's Control API (`/v3/config/paths/add|replace|delete`).
  - Paths are reconciled every 30 s and right after a camera is added, edited or removed. API changes live only in MediaMTX's memory, so a MediaMTX restart is healed on the next pass.
  - Argus owns the `cam_` namespace on that MediaMTX and deletes `cam_*` paths it did not create.
  - Live-only paths use `sourceOnDemand`: the camera is pulled only while someone watches. Recorded paths pull continuously.
  - File and webcam cameras stay analytics-only.
- **Managed or external MediaMTX.** If the API at `api` answers, that MediaMTX is used as it is ("external"). Otherwise, if `mediamtx_bin` is set, Argus writes `<data>/mediamtx/mediamtx.yml` and starts the binary itself, restarting it if it exits ("managed").
  - In the managed config only the Control API and WHEP signalling are enabled, on the addresses above (localhost by default), plus the ICE/UDP media port.
  - RTSP, RTMP, HLS, SRT, MoQ and playback are off. Only localhost may read or use the API.
  - The file defines no paths, so **camera passwords are never written to disk by the restreamer**. MediaMTX gets them over the localhost API.
- **The browser never talks to MediaMTX over HTTP.** `POST /api/cameras/{cam}/whep` needs a session with `evidence.view`, plus the CSRF header or a same-origin `Origin`. Argus forwards only the SDP offer to `<webrtc>/cam_<id>/whep` from localhost, and returns the answer with `Location` rewritten to `/api/cameras/{cam}/whep/<session>` (PATCH for trickle ICE, DELETE on close), `ETag`, `Link` and `Accept-Patch`.
  - Media then flows **peer-to-peer over ICE**, with UDP `webrtc_udp_port` (default 8189) from the browser to the MediaMTX host. That is fine on a LAN.
  - Remote viewers would need TURN (manual §15 V-07), which is not implemented.
  - Each session start is audited (`live_view_webrtc`).
- **Live page**
  - Each tile and the single-camera view have a toggle between **Smooth (WebRTC, full-rate, no overlay)** and **Analytics (overlay, ~5 fps)**.
  - The single-camera view and the focus tile default to Smooth. Other tiles stay on Analytics unless you switch them, because each WebRTC viewer is its own decoder.
  - A chip on every tile says which mode is on. Any WebRTC failure falls back to Analytics automatically and says why: no stream, a network or UDP block, no decoded frame within 10 s, or a lost connection. Choosing Smooth again retries.
  - There is **no client-side overlay** on the WebRTC video: no endpoint serves pixel boxes cheaply enough. Use Analytics for boxes and zones.
  - Sessions close as soon as a tile scrolls out of view, the tab is hidden or the page is left.
- **Recording**
  - MediaMTX writes `<record.path>/cam_<id>/YYYY-MM-DD_HH-MM-SS-ffffff.mp4` (fMP4, `recordSegmentDuration` = `segment_s`, 1 s parts). MediaMTX's own deletion is off.
  - Every 30 s Argus indexes the segments into the `recordings` table with `stream = 'main'`. The column is added by an additive migration; old rows become `analytics`.
  - Every 10 min it deletes segments older than `keep_days`, touching only files that match the segment name pattern inside `cam_*` directories.
  - The Recordings page filters All / Main / Analytics. `GET /api/recordings?stream=main|analytics` and `/api/recordings/{id}/video` (HTTP Range) serve the files.
  - The alert drawer's **Main-stream recording** link opens `#/recordings?cam=…&t=…`, which finds the covering segment (`/api/recordings/at`) and seeks to 4 s before the evidence frame.
  - Segment names are MediaMTX's **local time**. MediaMTX and Argus must therefore run on the same host, or share a clock and time zone.

**Camera requirements (manual §15 V-03/V-04).** The main stream must be **H.264 without B-frames**, with GOP ≤ 2 s. WebRTC cannot play B-frames, and MediaMTX ends the session, so the tile falls back to Analytics. H.265 plays only in browsers with hardware HEVC decode. There is no transcoding, and audio is never played.

**How this was verified (laptop, no MediaMTX binary; 2026-09-25)**
- `tests/test_restream.py` covers:
  - path config generation;
  - Control API reconcile (with MediaMTX's `1m0s` duration echo);
  - credential redaction in status, camera API, audit and MediaMTX error texts;
  - the WHEP proxy: header rewrite, no browser headers forwarded, 401/403/CSRF, and 404/409/415/502 mapping;
  - the managed-process fallback, the segment indexer, `recording_at` offsets, HTTP Range, retention, and the table migration.
- MediaMTX is an `httpx.MockTransport` stand-in written from MediaMTX's OpenAPI spec (v1.21 schema).
- A dashboard smoke test ran against a stub MediaMTX (WHEP answering 404) and ffmpeg-made fMP4 segments. It showed the Smooth → Analytics fallback with the reason, the per-tile toggle, and the Recordings deep link seeking to the right second.
- **Real playback, measured 2026-09-26 on the RTX 4050 host** (`tools/jobs/webrtc_job.ps1` there; the check is `tools/webrtc_check.py`, a headless Chrome that opens `#/live/<camera>` and samples the `<video>` element): real MediaMTX 1.x managed by Argus, the RTSP replay harness as cameras, detection through the local inference node.

  | Camera (main stream) | Smooth view | Decoded | Dropped |
  |---|---|---|---|
  | `cam_campus` (vtest, 768×576, 10 fps source) | PASS, WebRTC · no overlay | 10.0 fps | 1 frame |
  | `cam_street_bw` (854×480, 24 fps source) | PASS | 24.1 fps | 5 frames |

  Both play at the source frame rate. Main-stream recording wrote and indexed 4 segments in the first 90 s. Every listener was on 127.0.0.1 (ICE/UDP 8189, WHEP 8889, API 9987, the app on 8010).
- **Localhost only by default.** `webrtc_udp_host` (default `127.0.0.1`) is where the ICE/UDP media port listens, and 127.0.0.1 is then the only candidate MediaMTX offers. Smooth view therefore works in a browser **on the same machine** (or through a remote desktop to it). Set `"webrtc_udp_host": ""` to listen on every interface so LAN browsers can play; that opens a UDP port to the network, so only do it on a trusted network. The Analytics view (MJPEG) works everywhere through the SSH tunnel.

**End-to-end on the Windows GPU host (replay harness already on :8554).** The restreamer is a **second** MediaMTX that Argus starts itself (API :9987, WHEP :8889, UDP :8189), so the harness (API :9997) is untouched.
```powershell
cd $env:USERPROFILE\argonix\argonix_app
powershell -NoProfile -ExecutionPolicy Bypass -File tools\rtsp_replay.ps1 status      # vtest, street_bw (+ _sub) ready
$s = Get-Content "$env:USERPROFILE\argonix_rtsp\secrets.json" -Raw | ConvertFrom-Json
$bin = "$env:USERPROFILE\argonix_rtsp\mediamtx\mediamtx.exe"
(Get-Content configs\restream_site.json -Raw).Replace('${RTSP_USER}', $s.user).Replace('${RTSP_PASSWORD}', $s.password).Replace('${RTSP_HOST}', '127.0.0.1:8554').Replace('${MEDIAMTX_BIN}', $bin.Replace('\','/')) | Set-Content -Encoding utf8 configs\restream_site.local.json
uv run --no-sync python run.py --config configs\restream_site.local.json --data data\run_restream --port 8010 --host 127.0.0.1 --open-demo
Then:
1. The app log shows `restream: started managed MediaMTX` and `restream: paths added ['cam_campus', 'cam_street_bw']`. `Invoke-RestMethod http://127.0.0.1:8010/api/restream` shows `mode: managed`, each camera with `available: true` and `online: true`, `codecs: [H264]`, and no password anywhere.
2. Open `http://127.0.0.1:8010/#/live/cam_campus` **in a browser on the GPU host itself**. That needs no firewall change, because ICE uses loopback/LAN host candidates. The chip should read "Smooth · WebRTC · no overlay" with fluid 25 fps video; toggle to Analytics and back.
   - From the laptop, the page loads through `ssh -N -L 18010:127.0.0.1:8010 Luikz@192.168.137.135`, but WebRTC media cannot go through an SSH (TCP) tunnel, so tiles fall back to Analytics with the reason shown. Smooth view from the laptop would need `"webrtc_udp_host": ""` plus a firewall rule for UDP 8189 on the host, which this setup deliberately avoids.
3. After about 1–2 minutes, `data\run_restream\main_rec\cam_campus\*.mp4` appears. The Recordings page → Main lists 60 s segments that play. An alert's drawer link "Main-stream recording" opens the segment at the alert time.
4. Failure paths:
   - Stop the harness (`rtsp_replay.ps1 stop`): the tile falls back to Analytics with the reason. After `start`, choosing Smooth works again.
   - Kill `mediamtx.exe` (the managed one): within 30 s Argus restarts it and re-adds the paths.
5. Retention: set `"keep_days": 0.001` in the local config and restart. Segments older than about 90 s are deleted on the first sweep, and their rows disappear from Recordings.

## Access control: door events fused with video

Argus's first non-video stream (`argonix/sensors.py`, manual §28 S-06/S-12/S-13). A door controller, access panel or alarm panel POSTs one JSON report per state change to `POST /api/sensors/events`:

{"source_id": "door-ctrl-1", "type": "door_forced", "ts": "2026-09-25T10:00:00Z", "door": "D1",
 "card_id": "optional", "details": {"held_s": 45}}

- **Types:** `door_forced`, `door_held_open`, `access_denied`, `access_granted`, `alarm_input`, `tamper`, and `heartbeat` (keep-alive). `ts` is epoch seconds or ISO-8601 with an offset; if omitted, the arrival time is used.
- **Authentication** (either):
  - a Standard Webhooks HMAC-SHA256 signature (`webhook-id`, `webhook-timestamp`, `webhook-signature`) with a per-source secret taken from the environment variable in `secret_env`. Requests more than 5 min from server time are refused (401), and so is a re-used `webhook-id` (409);
  - an API token with the new `sensor.ingest` permission: role `integration` (a machine account that can only post sensor events) or `admin`.
- **Strict validation.** Unknown fields, unknown source or door, a `ts` more than `max_clock_skew_s` (300 s) from server time, nested `details` and credential-like `details` keys are refused (422). Events are never silently re-timed.
- **Privacy.** `card_id` is replaced on arrival by a keyed hash (`card:` + 16 hex digits; HMAC-SHA256 with the random per-site key in `<data>/sensor_card.key`). The raw value is never stored.

**Configuration** (site config):

"sensors": {"sources": [{"id": "door-ctrl-1", "secret_env": "ARGONIX_SENSOR_SECRET_DOOR1", "heartbeat_s": 60,
  "doors": {"D1": {"name": "Server room door", "camera": "cam1", "zone": "server_door"},
            "D2": {"site_zone": "lobby"}}}]}

A door maps to the camera zone that views it; the report takes that zone's `site_zone`, or the camera's own `<camera>:<zone>` key when the zone has none. Optional: `access_denied_burst: {"count": 3, "window_s": 60}` and `criticality`.

**Rules.** Each report becomes an ordinary Argus event, registered like the video rules:

| Rule | From | Severity |
|---|---|---|
| `door_forced` | `door_forced` | 3 |
| `door_held_open` | `door_held_open` | 2 |
| `access_denied_burst` | 3 or more `access_denied` at one door within 60 s | 2 |
| `alarm_input` | `alarm_input` | 3 |
| `access_tamper` | `tamper` | 2 |
| `sensor_offline` | a source with `heartbeat_s` silent that long (integrity, like `camera_health`) | 2 |

`access_granted` and `heartbeat` are recorded (counts, last seen) but raise nothing. Observations are factual device reports, for example "Door D1 (Server room door) reported forced open (door controller door-ctrl-1)". A panel report is a deterministic source, not a learned model: it is stated as panel-reported and not verified by video, with a fixed confidence of 0.9.

**Fusion.** Door reports carry the site zone of the camera that views the door, so the existing correlator groups a door report and a video rule in the same site zone into **one** alert and incident. `explain_priority` adds +0.25 per extra modality and reports `cross_modal: "sensor + video"`. Example: `door_forced` (3 × criticality 3 × 0.9 = 8.1) plus `intrusion` on cam1 gives a corroboration multiplier of 1.5 (two rules, one camera, two modalities).
- Integrity handling: camera-integrity muting never silences a door report, and a silent sensor never mutes camera alerts. `sensor_offline` opens a separate integrity incident, which stays active while the source is silent.
- The alert drawer shows door events with an "Access control" badge, the source, door, event type, time source and authentication. Its media is the covering camera's view when the report arrived: a snapshot, plus a clip with the usual pre/post seconds.
- The incident graph adds a **sensor** node, with `reported_by` edges from its events and a `covered_by` edge to the camera that views the door.
- The System page lists each source: status, last message, last event, report counts, alerts, rejected requests and signing status.
- `GET /api/sensors` (status) and `GET /api/sensors/events` (recent reports) need `alert.read`.

**Demo beat** (`configs/demo_site.json` maps `door-ctrl-1` D1 to cam1's "Server room door" zone and D2 to cam2's lobby):

uv run python tools/sensor_sim.py new-secret              # prints a whsec_ secret
$env:ARGONIX_SENSOR_SECRET_DOOR1 = "whsec_..."            # in BOTH shells (app and simulator)
uv run python run.py --open-demo
uv run python tools/sensor_sim.py forced --door D1        # while the synthetic person walks into the server-door zone

1. Trigger "door forced" while someone walks into the server-room door zone.
2. The result is one alert and one incident, titled for example "Door reported forced open + Restricted-zone entry". "Why P1?" shows "sensor + video"; the incident graph shows the door controller node next to cam1.
3. Other scenarios:
   - `held` for a door held open;
   - `denials` for three denials in 3 s, which gives `access_denied_burst`;
   - `--replay` to show the 409;
   - `heartbeat --every 10` then Ctrl+C for `sensor_offline` (set `heartbeat_s` first).

**Limits:**
- Webhook ingestion only. ONVIF Profile A/C, OSDP and MQTT are documented in manual §28 and not implemented.
- Replay protection keeps webhook-ids for 10 min in SQLite, twice the 5 min timestamp tolerance.
- There is no per-source rate limit yet.
- Nothing about door events has been measured on a real panel: the numbers above are configuration, not results.

## Demo scenarios, phone push, privacy masks, heatmap time-lapse

### Demo scenarios (clearly labelled DEMO)

**System → Demo scenario → Run demo scenario** plays a scripted sequence from `configs/scenarios/*.json` through the real alert path. Correlation, incidents, push decisions and notifications all run for real. The code is in `argonix/scenario.py`.

The bundled `intrusion_door_tamper.json` uses the first camera and its first zone:

| Time | Step |
| t+0 s | Restricted-zone entry |
| t+8 s | "Door forced" access-control report |
| t+15 s | Camera cover, using the existing tamper injection |
| t+30 s | Recovery |

On the synthetic demo site, the entry and the door report group into one **P1** alert and incident.

Steps are `event`, `sensor`, `tamper`, `recover` and `note`, each at an `at` offset in seconds. `camera` and `zone` are optional and default to the first camera and its first zone.

- **Labelling.** Every scripted event carries `derivation.demo_scenario = true`. Its observation starts with `[DEMO]`. Its evidence snapshot is the camera's current view, stamped "DEMO SCENARIO – scripted event, not observed".
  - The alert and incident get `demo: true` and a `[DEMO]` title prefix.
  - The dashboard shows an orange **DEMO** badge in Review, in the alert drawer and in the Live alert list.
  - Health events raised while a scripted tamper is active are labelled the same way.
- **Isolation.** Demo events never correlate with observed events. They never share an incident with observed ones, and a demo camera problem never mutes observed alerts. Demo alerts are excluded from:
  - rule precision (`/api/feedback/stats`);
  - the event-rate baseline;
  - both hotspot views.
- **Door sensor.** If the access-control sensor module is installed (`site.sensors` with the step's `source_id` and `door` configured), the `sensor` step goes through its ingestion with `details.demo_scenario = true`. Otherwise an equivalent `door_forced` event is emitted directly. The rule is registered locally only if nobody else registered it.
- **Stop.** Stopping a scenario clears any camera fault it injected.
- **When it runs.** Scenarios run only in `--open-demo` mode or with `"demo": {"scenarios": true}` in the site config; otherwise the API returns 409. `"demo": {"scenarios_dir": "..."}` points at another folder.
- **API** (needs `settings.edit`; each run and stop is audited): `GET /api/demo/scenarios`, `POST /api/demo/scenarios/{name}/run`, `POST /api/demo/stop`, plus `GET /api/demo/status` (needs `alert.read`).
- **Slow machines.** On a slow CPU, the cover is detected only if the camera still processes frames for the health monitor's `persist_s` (3 s). On this laptop, with other jobs running, analytics fell to 0.25 fps, and the 15 s cover raised no health event during a manual run.

### Phone push: Telegram and ntfy (`argonix/notify.py`)

Push rides on the incident push policy. A channel sends only when `IncidentManager` decides "would push":
- P1 always;
- P2 only within the push budget;
- never suppressed or inhibited incidents;
- and only at or above the channel's `min_priority`.

"notify": {"console_url": "https://argonix.lan:8000",
           "telegram": {"enabled": true, "token_env": "ARGONIX_TG_TOKEN", "chat_id": "123456789",
                        "min_priority": "P1", "send_snapshot": false},
           "http": {"enabled": false, "url": "https://ntfy.sh/<private-topic>", "min_priority": "P1", "token_env": ""}}

- **Message.** Priority, rule label, camera, zone, local time, incident and alert IDs, and the incident link (when `console_url` is set). It ends with "Observed event; a person decides what to do." Demo incidents are prefixed `[DEMO]`; set `include_demo: false` to skip them.
- **Snapshot.** `send_snapshot` uses Telegram `sendPhoto` with the evidence snapshot.
- **HTTP channel.** The `http` channel POSTs the text with ntfy headers: `Title`, `Priority` (5 for P1, 4 for P2), `Tags` and `Click`.
- **Delivery.** A background queue sends with up to 3 attempts and exponential backoff. Only 429 and 5xx responses are retried. The pipeline never waits on the network.
- **Audit.** Every result is audited (`notify_sent` / `notify_failed`) and stored as a notification row (`telegram_sent`, …).
- **Secrets.** The bot token comes only from the named environment variable. It and the ntfy URL path are scrubbed from every error and status and are never logged.
- **Testing.** **System → Phone push → Send test notification** (`POST /api/notify/test`, `settings.edit`, audited) sends one attempt per enabled channel and shows the result. `GET /api/notify` shows the channel state without secrets.
- **Not tested live.** Tests use a mocked HTTP transport. No message was sent to a real Telegram bot from this machine.

### Privacy masks (blanked before storage)

A privacy mask is a per-camera `"privacy_masks_px": [[[x, y], ...]]`, in source pixels. It is separate from detection masks.

- **What it blanks.** The polygon is filled black on each frame right after grab and conditioning, before anything else sees the frame. It is therefore blank for detection, recording, evidence snapshots and clips, the MJPEG/live view, the editor snapshot, the bird's-eye map, the VLM and exports. Nothing inside it can raise an alert.
- **Editing.** In the zone editor, **Privacy mask** draws a polygon shown as semi-transparent black. `PUT /api/cameras/{id}/privacy_masks` validates it (3 or more points, at most 64 polygons, `zone.edit`, audited).
- **Old footage.** Recordings made before the mask was added are not rewritten.

### Heatmap time-lapse (Map page)

- **Recording.** The site-map heatmap thread also accumulates **undecayed** person-seconds per 0.5 m cell in 15-minute buckets (`map.heat_bucket_s`). It persists them to `<data>/heatmap.db`: SQLite, zlib float32 grids, kept for 7 days (`map.heat_retention_days`). The running bucket is flushed every 60 s and on shutdown.
- **API.** `GET /api/map/heatmap/history?hours=24&step_s=3600` (`alert.read`; hours ≤ 168, step 60 s–1 day, rounded to a bucket multiple) returns one sparse frame per step. The frames are re-gridded onto the current map bounds.
- **Viewer.** On the Map page, **Heatmap time-lapse → Last 24 h** gives a 15 min / 1 h step, play/pause, a time slider, and a time label on the map. The legend uses one colour scale shared by all frames.
- **Limits.** Nothing is recorded while the server is stopped. Cameras with overlapping views count a person once per camera.

Tests: `tests/test_demo_features.py` covers:
- scenario runs (correlation, P1, DEMO labelling and exclusions, isolation from observed alerts, stop, API gating);
- Telegram and ntfy delivery with a mocked transport (message content, min priority, retries, snapshot, token never leaked, test button);
- privacy-mask pixels in saved snapshots, recordings, live frames and the editor snapshot;
- heatmap history bucketing, persistence across a restart, and the API.

## Decision support: what changed, track record, dwell baseline, recovery, calibration, access log

Six additions aimed at the last clause of the problem statement: *faster and more informed decisions by authorised personnel*. None of them raises an alert on its own. Tests: `tests/test_decision_support.py`.

**"What changed vs normal?"** (`argonix/changes.py`, `GET /api/investigate/changes?hours=1`, `alert.read`). Investigate has a **What changed vs normal?** button with a window selector (1, 3, 8 or 24 h). Typed questions such as "what changed?" or "anything unusual?" are answered the same way. It compares the window with this site's own history **at the same hours of the week in previous weeks**:

| Comparison | Now | Usual | Test |
|---|---|---|---|
| Occupancy per zone | mean people count in the window | per-minute samples from previous weeks | modified z (median/MAD), the same test as the occupancy baseline |
| Event rate per zone | rule and sensor events in the window (demo, health and baseline events excluded) | hourly event counts from previous weeks | Poisson-style z, and at least 2 events apart |
| Dwell per zone | median completed stay in the window (3 or more stays) | earlier stays at the same hours (10 or more) | modified z |
| Camera view | cameras degraded right now | "ok" | always listed first |

Results are ranked by |z| (threshold 2). Zones without enough history are listed under **Not enough history yet**, never guessed. Zones within their usual range are listed under **As usual**. Hours of the week are local time, counted in 5-minute steps, so half-hour time zones (IST) split correctly.

**Operator track record in "Why P?"** (`argonix/track_record.py`). Every alert's priority breakdown shows how often operators confirmed past alerts from its top rule on its camera. It counts the last 30 days of resolved or dismissed alerts with a disposition, gives a Wilson 95 % interval, and shows the site-wide figure alongside. `duplicate`, `test_or_drill`, `unclear` and demo alerts don't count. The cache refreshes every 60 s, and immediately when an operator records a disposition.

Optional **feedback demotion**, off by default:
```json
"feedback_demotion": {"enabled": true, "min_reviewed": 20, "max_precision_upper": 0.2}
```
A *single, uncorroborated* alert drops one band (P2 → P3) when its rule on that camera has at least `min_reviewed` decisions and the upper end of the 95 % interval is below `max_precision_upper`. "Single, uncorroborated" means one rule, one camera and one modality. The breakdown says why ("Lowered P2 → P3: operators confirmed 1 of the last 20 …"). Alerts that another rule, camera or door sensor corroborates are never demoted, and neither are demo alerts or P4 alerts. A noisy rule therefore stops paging people, but can't hide a situation something else confirms.

**Baseline-aware loitering.** A loitering rule can learn what "long" means for its zone:
```json
"rules": {"loitering": {"min_seconds": 20, "baseline_quantile": 0.95, "baseline_factor": 1.5, "baseline_min_stays": 30}}
```
Once `baseline_min_stays` completed stays have been recorded in this zone in this hour of the week (the flow analytics record them; they are on by default), the threshold becomes max(`min_seconds`, factor × p95 of those stays). A bus-stop bench then alerts on stays that are long *for a bus stop at 8 am*, and a corridor alerts on its configured minimum. The alert says so, for example "remained in 'Waiting area' for 91.0 s (usual here at this hour: p95 60 s over 50 stays)". The derivation keeps `configured_min_seconds` and the baseline it used. Before enough history exists, the rule uses `min_seconds` and the derivation says "not ready (5 of 30 stays …)".

**Camera recovery.** When a view condition clears (covered, dark, blurred, frozen, down), the newest camera-health alert that reported it gets a `health_recovery` entry, for example "View condition 'cover' cleared at 14:02:10 after 42 s". The drawer shows it, it goes into the audit log as `camera_recovered`, and no new alert is raised. An operator can see at a glance whether a tamper alert is still ongoing.

**Robust N-point calibration** (`geometry.fit_homography`). `calibration.image_points` / `ground_points` accept **4 or more** pairs:
- 4 points: the exact solve, as before.
- 5 or more: RANSAC over 4-point subsets (exhaustive up to 2,000 subsets), then a least-squares refit on the inliers. Points more than `robust_threshold_m` (default 0.5 m) off are reported as outliers, usually a mistyped or mis-clicked point.
- With 5 or more inliers, the **leave-one-out error** is computed: refit without each point and measure that point.

The stated calibration error is never below the configured `calib_error_m`, the RMS on held-out check points, or the leave-one-out error. Live → camera → *Calibration & zones* shows "6 of 6 points agree; leave-one-out error 0.21 m".

**Footage access log.** Opening a live view (MJPEG stream or snapshot polling) and reading any evidence file under `/media/` are written to the hash-chained audit log as `live_view` and `media_read`. There is one row per user and camera or file per 10 minutes (`"privacy": {"access_log_s": 600}`), so snapshot polling doesn't flood the chain. `"privacy": {"access_log": false}` turns it off.

## Measured, not claimed

- Laptop CPU, 2 cameras, YOLO26n at imgsz 640: about 0.7–1.4 s per detection when both cameras run at once. When the pipeline falls behind real time it drops frames (see `dropped` in System) instead of lagging. The default is now imgsz 480.
- At 5 fps analytics, very fast motion can break track continuity. Running detection depends on keeping the same track ID.
- The synthetic clips use pasted person crops on a rendered floor, with a known homography. They test the pipeline and are **not** an accuracy benchmark.

### Accuracy runs

Two measurements against public ground truth. Both run the production pipeline (`Site` → detector → ByteTrack → zones → rules), faster than real time, and write committed artifacts to `eval_runs/<date>-<label>/` (JSON with the hardware, detector and exact command, plus a markdown table). The metric code is `tools/eval_metrics.py`, with known-answer tests in `tests/test_eval_metrics.py`.

**1. Rule precision/recall on CAVIAR** (`--only caviar` → `caviar_rules.{json,md}`)
- **Data.** CAVIAR first set: 27 clips from one wide-angle lobby camera, 384×288 at 25 fps, with CVML per-frame labels. Acknowledgement: EC Funded CAVIAR project/IST 2001 37540.
- **No calibration.** CAVIAR has none, so the default `--caviar-space px` uses a whole-view **image-pixel** zone and pixel thresholds:
  - `loitering`: 20 s;
  - `crowding`: more than 1 person for 3 s;
  - `running`: at least 1.5 body-heights/s for 1 s. This is the foot-point speed divided by the track's median box height, an uncalibrated proxy (about 2.5 m/s for a 1.7 m person). It is opt-in on uncalibrated cameras with `"running": {"speed_bh_s": ...}`.
  - `--caviar-space auto` uses the pedestrian self-calibration instead, and needs `--hfov`.
- **Ground-truth episodes.** Each label below must hold for the stated time, with gaps of 0.5 s or less merged:

  | Behaviour | CVML label | Minimum |
  |---|---|---|
  | `running` | object movement "running" | 1 s |
  | `fighting` | group context/situation "fighting" | 1 s |
  | `meeting` | group context "meeting" | 3 s |
  | `browsing_or_inactive_20s` | object browsing/inactive | 20 s |
  | `left_object` | left object | — (no Argus rule, reported as unsupported) |

- **Pairs scored.** loitering vs browsing/inactive, running vs running, crowding vs meeting (a *proxy*: two people in view is not a meeting), altercation vs fighting (only with `--threats`), and running vs fighting (secondary).
- **Matching rule.** An event is a true positive if both conditions hold:
  - its interval overlaps an episode of the target behaviour in the same clip, within ±2 s (`--tol-s`);
  - in a shared frame of that overlap, the centre of one of the event's track boxes is within 30 px (`--prox-px`) of the episode's box (a group box or a member's box).

  Precision = true events / events. Recall = episodes hit / episodes. Both are many-to-one. Each comes with a Wilson 95 % interval.

**2. Tracking accuracy on MOT17** (`--only mot17` → `mot17_tracking.{json,md}`)
- **Data.** The 7 MOT17 train sequences (FRCNN copies; MOTChallenge; non-commercial research use; not redistributed). Only these sequences are read out of the 5.5 GB `MOT17.zip`, through HTTP Range requests (`datasets.RemoteZip`); `--mot-max-frames N` takes only the first N frames.
- **Tracker.** Our detector (conf 0.3) + ByteTrack. No public detections are used.
- **Two settings:**
  - `5fps`: the app's analytics rate; only processed frames are scored, and the GT is subsampled the same way;
  - `full`: every frame.
- **Metrics.** MOTA/MOTP (CLEAR-MOT), IDF1 and HOTA, implemented following TrackEval:
  - IoU 0.5;
  - MOTP is the mean IoU of matches;
  - HOTA is averaged over α = 0.05–0.95;
  - "combined" pools counts across sequences.
- **MOT17 conventions.** GT is class 1 (pedestrian) with the consider flag set. Tracker boxes matched to distractor classes 2/7/8/12 are dropped. There is no visibility threshold, as in the official evaluation; `--mot-min-vis` changes that, and the artifact then says so.
- **Frames.** They are re-encoded to H.264 CRF 18, one frame per image, because the pipeline reads video files.

**Results** (RTX 4050 Laptop, Windows, 2026-09-26; YOLO26s + ByteTrack, the production path; artifacts in `eval_runs/2026-09-26-legion/` and `eval_runs/2026-09-26-legion-imgsz960/`). `eval_runs/2026-09-25-laptop-smoke/` is only a CPU smoke test, not a result.

*CAVIAR rules* (27 clips, 17 min, whole-view pixel zone, no calibration; Wilson 95 % intervals in the artifacts):

| Ground truth → rule | Episodes | imgsz 480: events · P · R | imgsz 960: events · P · R |
|---|---|---|---|
| browsing/inactive ≥ 20 s → `loitering` | 4 | 0 · n/a · 0.00 | 17 · 0.06 · 0.25 |
| running → `running` | 8 | 0 · n/a · 0.00 | 3 · 0.00 · 0.00 |
| meeting → `crowding` (proxy: > 1 person in view) | 10 | 1 · 0.00 · 0.00 | 20 · 0.25 · 0.50 |
| person detections per frame | | 0.16 | 0.80 |

*MOT17 tracking* (7 train sequences, private detector, TrackEval-style metrics; combined = pooled):

| Setting | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Detection recall / precision |
|---|---|---|---|---|---|---|---|
| 5 fps (the app's analytics rate) | 0.286 | 0.221 | 0.377 | 0.226 | 0.314 | 487 | 0.28 / 0.90 |
| every frame | 0.342 | 0.258 | 0.456 | 0.280 | 0.386 | 453 | 0.32 / 0.90 |

Per sequence, HOTA ranges from 0.13 (MOT17-13, 5 fps) to 0.50 (MOT17-11, every frame).

**What this means.** Both results are limited by the detector's **recall on small people**, not by the rules or the tracker's association:
- CAVIAR people are 20–60 px tall in 384×288 frames. At imgsz 480 YOLO26s finds 0.16 people per frame and the rules have almost nothing to work with. At imgsz 960 it finds 5× more; crowding then catches half the meetings, but loitering's precision is low (people browsing in the lobby for a long time are not labelled "inactive" in the ground truth).
- On MOT17 precision is 0.90 but recall is 0.32: the app downscales 1080p to 960 px wide and runs the detector at 480, so distant pedestrians vanish. Published private-detector trackers reach HOTA ≈ 0.6 on MOT17 with detectors trained on it; ours is a general COCO model at a laptop-friendly resolution.
- Running the analytics at 5 fps costs about 0.06 HOTA against every frame.

**Fix: lower confidence cut, larger input** (sweep 2026-09-26, same data and hardware, MOT17 at the app's 5 fps; artifacts `eval_runs/2026-09-26-legion-i*-c0.1/`). Two changes, both now the defaults:
- **Confidence cut 0.3 → 0.1.** The detector used to drop every box below 0.3 before ByteTrack saw it, but ByteTrack's second association stage exists to use exactly those low-score boxes (occluded and small people) to keep existing tracks alive. New tracks still need 0.25, so this adds continuity, not false tracks.
- **960 px input on the GPU** (the inference node and `real_site_gpu.json`; a TensorRT engine is built for it). The laptop-CPU configs stay at 480 px for speed.

| MOT17, 5 fps, combined | HOTA | IDF1 | MOTA | Detection recall | Precision | CAVIAR detections / frame |
|---|---|---|---|---|---|---|
| 480 px, conf 0.3 (old default) | 0.286 | 0.314 | 0.226 | 0.28 | 0.90 | 0.16 |
| 480 px, conf 0.1 | 0.297 | 0.332 | 0.237 | 0.30 | 0.89 | 0.41 |
| 640 px, conf 0.1 | 0.322 | 0.369 | 0.275 | 0.35 | 0.89 | 0.93 |
| **960 px, conf 0.1 (new GPU default)** | **0.372** | **0.444** | **0.354** | **0.45** | **0.89** | 1.28 |
| **+ track stitching (default on)** | **0.380** | **0.460** | **0.360** | 0.45 | 0.89 | |

Recall rose by 61 % and HOTA by 30 % at the same precision. **Track stitching** (`argonix/stitch.py`, on by default) then re-links a person the tracker lost for up to 5 frames (occlusion, crossing, fast movement) when they reappear where they were heading at a similar size: ID switches fell from 834 to 718 (−14 %) and IDF1 rose from 0.444 to 0.460. A looser setting removed more switches but joined different people (IDF1 0.412), so the strict one is the default. IDs can still change after long occlusions (over about 1 s at 5 fps). The hardest sequences are still weak (MOT17-13, a moving car camera with tiny pedestrians: recall 0.21), because the app first scales every frame to 960 px wide; tiling the full-resolution frame would be the next step. On CAVIAR at 960 px, crowding caught 40 % of meetings (precision 0.16) and loitering 1 of 4 inactivity episodes (precision 0.04): with enough detections, the remaining CAVIAR errors are about the rules and the whole-view pixel zone, not the detector.

One observation from the smoke run, not a measurement: on the CAVIAR clip, YOLO26n at imgsz 480 found almost no people (0.05 person detections per frame), which agrees with the fall-detection finding above. That is why the GPU commands include an imgsz 960 run.

**Metric cross-check.** On random noisy tracks, `tests/test_eval_metrics.py` checks our metrics against py-motmetrics 1.4 when that package is installed:
- FP, FN and IDF1 agree exactly.
- IDSW, and therefore MOTA, can differ slightly. We follow TrackEval, which keeps only the last frame's correspondences; py-motmetrics keeps the last-ever one. On 1 of 10 random sequences this changed IDSW by 2.
- On the smoke run's real tracker output (MOT17-09, both settings, after our MOT17 preprocessing), MOTA, IDF1, IDSW, FP, FN and MOTP all agreed exactly with py-motmetrics.
- For your own check: `--save-tracks <dir>` writes our tracks in MOTChallenge format, so TrackEval or py-motmetrics can score them independently.

**Reproduce on the GPU host.** Windows, PowerShell, from `%USERPROFILE%\argonix\argonix_app`. Datasets go under `%USERPROFILE%\argonix_datasets`. The detector defaults to `data/models/yolo26s.pt`. `--imgsz 480` matches `configs/real_site_gpu.json`.

```powershell
$env:PYTHONUTF8 = "1"
$R = "$env:USERPROFILE\argonix_datasets"
# CAVIAR: fetch 27 clips + GT (~380 MB), convert, score the rules
.venv\Scripts\python.exe tools\eval_datasets.py --only caviar --fetch --root $R --device cuda:0 --imgsz 480 --label legion
# sensitivity: CAVIAR people are ~20-60 px tall in 384x288 frames; upscaling helps the detector (report both)
.venv\Scripts\python.exe tools\eval_datasets.py --only caviar --root $R --device cuda:0 --imgsz 960 --label legion-imgsz960
# optional: add the altercation analyzer (needs data/models/yolo26n-pose.pt)
.venv\Scripts\python.exe tools\eval_datasets.py --only caviar --threats --root $R --device cuda:0 --imgsz 480 --label legion-threats
# MOT17: fetch the 7 FRCNN train sequences (~0.9 GB of JPEGs out of the 5.5 GB zip), convert, score 5 fps + full rate
.venv\Scripts\python.exe tools\eval_datasets.py --only mot17 --fetch --root $R --device cuda:0 --imgsz 480 --label legion
```

**Expected runtime.** These are *estimates, not measurements*:
- CAVIAR: about 5–10 min after the download (27 short clips batched on one GPU model).
- MOT17: about 15–30 min after the download (5,316 frames at full rate plus the 5 fps pass; H.264 conversion of the 1080p frames; per-sequence model start-up).

**Outputs.** `eval_runs\<date>-legion\caviar_rules.{json,md}` and `eval_runs\<date>-legion\mot17_tracking.{json,md}`. Copy them back and commit them.

**Without internet on the host.** Fetch on the laptop with the same `--fetch` flags (`--only caviar,mot17`), then send the set: `tools/datasets.py ship --only caviar,mot17 --tar -`, piped over ssh.

### RTSP end-to-end on the GPU host

`tools/rtsp_e2e.py --launch` starts a fresh Argus on the RTX 4050 host and drives it like an installer would: ONVIF discovery, stream probes, five cameras added through `POST /api/cameras` from a local MediaMTX replay harness (sub-streams, a black-and-white stream, and a fault stream that drops for 20 s and freezes for 40 s every 110 s), 240 s of observation, a main/sub-stream switch through `PUT`, credential-redaction checks on every endpoint and the log, and the dashboard's endpoint shapes. Run 2026-09-26 on the latest code: **52 of 52 checks passed** (`data/train_logs/rtsp_e2e2.json` on the host).

| What | Result |
|---|---|
| App ready after launch | 3.7 s; first analytics frame per camera 3.6–12.2 s after it was added (the shared detector warm-up had finished in the background) |
| Fault stream, default reconnect (1–30 s backoff) | camera-down alert during every outage, auto-reconnect 13.6–13.7 s after the stream returned |
| Fault stream, fast reconnect (0.5–4 s) | same alerts, reconnect in 5.9–6.1 s |
| Frozen picture (40 s per cycle) | `frozen` health events raised on both fault cameras |
| Analytics rate, 5 cameras at 5 fps | 4.7 fps per camera, detect 135–140 ms median per call |
| Stored site config | holds **no** camera password (the secrets file does); no plaintext password in any API response, the audit log or the app log |

The first two runs failed at the camera-add step (the first camera on an empty GPU site blocked on model warm-up for over 60 s) and at the main/sub switch (editing a camera on a batched GPU site failed); both are fixed and covered by tests.

## Deferred (per AGENTS.md)

Deferred: Postgres/Timescale/pgvector, message buses, learned-anomaly alarms (learned anomaly scores only corroborate), and the MapLibre basemap (a canvas ground plane is used for now). The VLM is optional: `--vlm-url` enables an "Explain" button. Its output is labelled assistive and is never used to trigger alerts.

## Optional VLM explainer (GPU box)

```bash
scp tools/vlm_server.py tools/remote_setup.sh root@<box>:argonix_vlm/ && ssh root@<box> "cd argonix_vlm && ./remote_setup.sh"
ssh root@<box> "cd argonix_vlm && HF_HOME=~/argonix_vlm/hf nohup .venv/bin/python vlm_server.py > vlm.log 2>&1 &"
ssh -N -L 8100:127.0.0.1:8100 root@<box>          # tunnel; the service only listens on 127.0.0.1
uv run python run.py --vlm-url http://127.0.0.1:8100/explain
```

At most `vlm_max_concurrent` (site config, default 2) Explain or Summarise calls run at once; a further click gets **429 "the VLM is busy"** instead of tying up an API thread for as long as the model takes.

With CUDA available the server loads Qwen2.5-VL-3B-Instruct in bf16. Without it (for example, when the container's GPU devices are cgroup-blocked), it falls back to SmolVLM-500M-Instruct on CPU. Its output is an assistive description only.

## Real footage (public datasets)

```bash
uv run python run.py --config configs/real_site.json --data data/run_real --port 8001
uv run python tools/eval_real.py configs/real_site.json     # calibration sanity + events per rule
```

Sources (downloaded into the git-ignored `data/real/` by `tools/setup_assets.py`; the street clip with `--only street`):
- `vtest.avi` from the official OpenCV repository (`samples/data`).
- `pedestrian.mp4` from the Kaggle dataset `jayitabhattacharyya/social-distance`.
- `smeschke/pedestrian-dataset` (low-angle clips, unused because they are poor for ground-plane mapping).

Neither used clip has survey data. `cam_campus` scale is inferred from person height (1.7 m prior, ±1.0 m). `cam_street` scale comes from a 3.6 m lane width (±1.5 m). Zones on the campus are drawn in image pixels (`polygon_px`).

Measured with `tools/eval_real.py`, one pass, laptop CPU, both cameras at once:

| Camera | Tracks (≥10 samples) | Median / p90 walking speed | Detect ms | Health |
|---|---|---|---|---|
| cam_campus | 43 | 0.93 / 1.19 m/s (includes people standing; scale may be ~20% short) | ~200 | OK |
| cam_street | 9 | 1.46 / 1.58 m/s (normal walking range) | ~200 | OK |

Observed on review:
- Lawn-entry alerts show people who really are walking on the lawn (true positives by visual check).
- A striped bollard on the street was detected as `person` in 44 of 49 sampled frames. It is now handled with a Frigate-style detection mask (`masks_px`).
- None of these are accuracy benchmarks: there are no ground-truth labels for these rules.

## GPU: one batched detector for all cameras

On CUDA the default backend is `argonix/inference.py`: **one** model on the GPU serves every camera. Each camera's `track(frame)` enqueues the frame. A single inference thread collects frames from all cameras into one batch: up to `max_batch` frames, waiting at most `max_wait_ms` after the oldest frame, and not waiting at all once every recently active camera has a frame queued. It runs one `predict` in fp16 (`quantize=16`, `classes=[0]`, fixed square input). Then each camera's **own** ByteTrack updates. Track IDs are numbered separately per camera, so a track never jumps between cameras. `CameraWorker` is unchanged.

| `detector` key | Default | Meaning |
|---|---|---|
| `batched` | `true` on CUDA, `false` on CPU | shared batched service vs the per-camera `UltralyticsTracker` (CPU and tests) |
| `weights` | `"auto"` | `auto` = yolo26s on CUDA with ≥ 4 GiB VRAM, else yolo26n. On CUDA with `tensorrt` importable, a matching exported engine in `data/models` is preferred. A configured `.engine` that is missing or unusable falls back to the sibling `.pt` (fp16). |
| `max_batch` / `max_wait_ms` | 16 / 8 | batch limits (an engine's own max batch caps `max_batch`) |

Startup runs every batch size from 1 to `max_batch` once, **on the inference thread**. This takes about 25 s for yolo26s `.pt`. It is needed because on the test GPU the first `.pt` call at each new batch size cost about 1.5 s, and that cost is per thread (warming up on another thread did not help). Before this warm-up, those stalls hit live cameras. `Site.stop()` does not close the shared service yet: its thread is a daemon and an `atexit` hook closes it.

TensorRT is optional. It is not in `pyproject.toml`, so a later `uv sync` removes it:

```bash
uv sync --extra cu128
uv pip install tensorrt-cu12 onnx onnxslim "nvidia-modelopt[onnx]"   # TensorRT 11 needs ModelOpt for FP16 export
uv run --no-sync python tools/bench_gpu.py --export --models yolo26s --imgsz 480 --batch 16   # -> data/models/yolo26s-480-b16.engine
uv run --no-sync python run.py --config configs/real_site_gpu.json
```

**On CUDA hosts always use `uv run --no-sync` (or `uv run --extra cu126`/`--extra cu128`) after `uv sync --extra cu126`/`cu128`.** A plain `uv run` re-syncs the default environment, which has no CUDA extra: on Windows it replaces the CUDA torch with PyPI's CPU-only wheel (seen on the RTX 4050 host: `2.14.0+cu126` became `2.14.0+cpu`), and it also removes TensorRT. `tools/win_services.ps1` and the bootstrap scripts already use `--no-sync`.

Each engine export takes about 2 minutes and about 1.8 GiB of VRAM. The yolo26s @640 batch-16 export was OOM-killed on the 14 GB host: ModelOpt's CPU calibration pass ran out of RAM. Batch 8 worked (`yolo26s-640-b8.engine`).

### Measured: RTX 3050 6 GB Laptop GPU

Hardware and conditions:
- GPU: RTX 3050 6 GB Laptop, driver 595.91. CPU: i5-12450HX (12 threads). Software: torch 2.11 + CUDA 12.8, ultralytics 8.4.161, TensorRT 11.3.
- The GPU was **shared**: a VLM server (about 2.8 GiB, idle) and a running Argus instance were resident throughout. Background GPU use just before each case was 2–28%.
- Streams: the two real clips (768×576, 854×480) replayed as N streams, all through the production code path with tracking included. Each case ran in a fresh process after warm-up.
- VRAM is this process only (nvidia-smi). GPU util is the mean over the run.
- Command: `tools/bench_gpu.py [--paced] --out results.md`.

**Saturation:** every stream calls `track()` back to back. The table gives aggregate FPS for all streams, with p95 per-call latency in parentheses.

| Mode | Model @ imgsz | N = 1 | N = 4 | N = 16 | GPU util @16 | VRAM @16 |
|---|---|---|---|---|---|---|
| (a) per-camera .pt fp16 (old path) | yolo26n @480 | 163 (7 ms) | 186 (37 ms) | 132 (131 ms) | 39% | 1366 MiB |
| | yolo26n @640 | 143 (10 ms) | 207 (31 ms) | 143 (125 ms) | 48% | 1462 MiB |
| | yolo26s @480 | 159 (8 ms) | 197 (33 ms) | 137 (133 ms) | 54% | 1762 MiB |
| | yolo26s @640 | 134 (10 ms) | 172 (34 ms) | 131 (138 ms) | 67% | 1808 MiB |
| (b) batched .pt fp16 | yolo26n @480 | 149 (8 ms) | 315 (15 ms) | 352 (75 ms) | 59% | 504 MiB |
| | yolo26n @640 | 155 (8 ms) | 228 (20 ms) | 241 (115 ms) | 73% | 784 MiB |
| | yolo26s @480 | 155 (8 ms) | 217 (21 ms) | 230 (118 ms) | 77% | 982 MiB |
| | yolo26s @640 | 117 (10 ms) | 135 (43 ms) | 142 (195 ms) | 84% | 1564 MiB |
| (c) batched TensorRT fp16 | yolo26n @480 | 252 (5 ms) | 383 (13 ms) | 450 (58 ms) | 47% | 750 MiB |
| | yolo26n @640 | 194 (6 ms) | 297 (16 ms) | 339 (80 ms) | 61% | 1340 MiB |
| | yolo26s @480 | 187 (10 ms) | 333 (15 ms) | 387 (70 ms) | 65% | 1076 MiB |
| | yolo26s @640 (max batch 8) | 162 (7 ms) | 208 (22 ms) | 236 (71 ms) | 71% | 1100 MiB |

**Paced:** every camera submits at a fixed analytics rate. A camera count counts as sustained only if every camera stays at ≥ 95% of the target rate and p95 latency stays within one frame interval. Each camera thread also does what `CameraWorker` does on the CPU:
- decodes a 1920×1080 25 fps H.264 clip, every frame
- resizes it to 960 px wide
- JPEG-encodes it

| Mode | Model @ imgsz | Rate | Max cameras sustained | First failing N |
|---|---|---|---|---|
| (a) per-camera .pt | yolo26s @480 | 10 fps | **10** (p95 32 ms) | 12 (9.5 fps, p95 110 ms) |
| (b) batched .pt | yolo26s @480 | 10 fps | **16** (p95 62 ms) | 20 (8.1 fps, 17% dropped) |
| (c) batched TensorRT | yolo26s @480 | 10 fps | **20** (p95 41 ms) | 24 (8.1 fps, 16% dropped) |
| (c) batched TensorRT | yolo26s @480 | 15 fps | **16** (p95 48 ms) | 20 (11.1 fps, 25% dropped) |
| (c) without decode/JPEG | yolo26s @480 | 10 fps | 28 (p95 37 ms) | 32 (p95 110 ms) |

What these numbers mean:
- At 16 streams, batching gives 1.1–2.7× the saturated throughput of one model per camera. The gain is largest for yolo26n, where per-call overhead dominates. yolo26s @640 is compute-bound (84% util) and batching barely helps it.
- The per-camera path gets *slower* past 2–4 streams: every thread runs its own model, and they contend for the GIL and the GPU.
- TensorRT adds another 1.3–1.7× on top of batched `.pt`.
- At 20+ decoding cameras, the limit is the **host CPU/GIL, not the GPU**. GPU util was about 40% when the 24-camera run failed, and the same engine sustained 28 cameras once decode and JPEG were removed.
- Decode is CPU-side, because OpenCV pip wheels have no NVDEC. Measured with `--decode`, one 1080p25 H.264 stream costs 6.2 ms CPU per frame, which is **15% of a core**, because `CameraWorker` decodes every source frame. The 854×480 clip costs 1.6 ms per frame (4% of a core).
- The next levers are NVDEC (PyAV/FFmpeg with `-hwaccel cuda`) and `grab()` for skipped frames. Neither is implemented.
- Two ByteTrack caveats:
  - ByteTrack's `track_buffer` (30) is counted in frames, so at 10 fps a lost track survives 3 s instead of 6 s at 5 fps.
  - Batched mode letterboxes to a square input, which is up to 33% more pixels than `rect` for 4:3 video.

**Recommended for this laptop** (`configs/real_site_gpu.json`): `analytics_fps` 10, batched yolo26s @480 as a TensorRT engine (falls back to `.pt` fp16), `max_batch` 16, `max_wait_ms` 8.
- Measured capacity with these settings: **20 cameras at 10 fps**, or 16 at 15 fps, using about 1.1 GiB of VRAM next to the other GPU tenants.
- The streams were synthetic replicas, and the pipeline's rules, recording and dashboard were not included. Treat these numbers as an upper bound for this one machine, not a general capacity claim.
- The real 2-camera site (`tools/eval_real.py configs/real_site_gpu.json`, TensorRT) measured 6–10 ms per detection call, compared with about 200 ms on the laptop CPU.

### Measured: RTX 4050 6 GB Laptop GPU (Windows host) — host-limited, partly fixed

Hardware and conditions: RTX 4050 Laptop (driver 566.24) in a Lenovo Legion, i7-14650HX (24 threads), Windows 11, torch 2.14 + CUDA 12.6, TensorRT 11.3. App and VLM stopped, GPU otherwise idle. Same `tools/bench_gpu.py` harness as above, 2026-09-25. The laptop was first measured in its Quiet power mode (GPU held at 615 MHz); these numbers are from **Performance mode**, and they were only about 5 % better (`data/train_logs/bench_rtx4050*.md` on the host).

| Saturation (aggregate FPS, p50 per call) | N = 1 | N = 4 | N = 16 | GPU util @16 |
|---|---|---|---|---|
| batched .pt yolo26s @480 | 16 (58 ms) | 36 (109 ms) | 93 (158 ms) | 28 % |
| batched TensorRT yolo26n @480 | 34 (23 ms) | 48 (77 ms) | 79 (176 ms) | 28 % |
| batched TensorRT yolo26s @480 | 39 (23 ms) | 64 (61 ms) | 96 (155 ms) | 28 % |

**Paced at 10 fps with decode + JPEG** (as above): 1 camera sustained with every model; 2 cameras ran at 10 fps with 0 % drops on TensorRT but p95 was 104 ms, just over the 100 ms frame interval, so they don't count as sustained.

**What this means.** On this host the GPU is mostly idle (28 %) and one TensorRT call takes 23 ms, where the older RTX 3050 on Linux took 5 ms. The limit is host-side per-frame work (pre/post-processing, tracking, thread hand-offs), not the GPU. One cause is measured: Windows rounds thread waits up to its 15.6 ms timer tick (a `Condition.wait(0.008)` took 15 ms). `run.py` now requests 1 ms ticks, which gave +5–14 % throughput and 20–40 % lower p95 in a 4-stream TensorRT case, so it is not the main cost. The rest has not been profiled yet; CPU thread oversubscription (24 threads for torch and OpenCV) and Python 3.11 on this host are the next suspects. 
**Second cause, fixed: efficiency cores.** The i7-14650HX has 8 performance and 8 efficiency cores, and Windows was running the process on the efficiency cores. Kernel launches and post-processing are CPU-bound, so one frame took 8.0 ms of TensorRT inference (45 ms with fp16 `.pt`). Pinned to the performance cores it took 4.9 ms (26 ms). `argonix/hosttune.py` now does this at start-up on any Windows hybrid CPU (it reads each core's efficiency class from Windows; `ARGONIX_ANY_CORES=1` turns it off). Re-measured with both fixes, 2026-09-26:

| yolo26s @480 | Saturated, N = 1 | N = 4 | N = 16 | Paced 10 fps with decode |
|---|---|---|---|---|
| batched .pt | 28 fps (was 16) | 70 (36) | 134 (93) | **2** sustained; 4 at 9.9 fps, 0.9 % dropped, p95 108 ms |
| batched TensorRT | 69 fps (was 39) | 124 (64) | 149 (96) | **2** sustained; 4 at 9.7 fps, 0.3 % dropped, p95 107 ms |

**What this means for the demo host:** 2 cameras at 10 fps strictly, about 4 in practice (they keep 97–99 % of the rate, but p95 latency is just over one frame interval). GPU use is still only 20–35 %, so more can be won on the CPU side (decode, per-camera threads under the GIL). Don't quote the RTX 3050 capacity for this machine.

## Distributed GPU inference nodes

The camera workers (decode, rules, recording, dashboard) can run on one machine while detection and tracking run on a
GPU machine elsewhere. The node is `argonix/infer_server.py`: **one** shared `BatchedInference` (see above), served
over WebSockets. The client is `argonix/remote_inference.py`, a normal tracker backend, so `CameraWorker` and the
rules do not change.

**Start a node** (on the GPU host):

```bash
uv run --no-sync python -m argonix.infer_server --host 127.0.0.1 --port 8200 --weights auto --imgsz 480 --max-batch 16
```

On a Windows host, use `win_services.ps1`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\win_services.ps1 start-infer [-InferHost <ip>] [-InferPort 8200] [-Weights auto] [-Imgsz 480] [-MaxBatch 16] [-ExtraClasses 24,26,28]
```

It binds **127.0.0.1 only**, never a network interface (the app and the VLM that `win_services.ps1` starts are
local-only too). It survives the SSH session (see [Troubleshooting](#troubleshooting)). `stop-infer` stops it, and
`status` probes its `/health`.

**Use it from the camera machine** (this is how the laptop runs against the RTX 4050 host). Open an SSH tunnel, copy the
token once, and start the site with `--infer-node` (or set `ARGONIX_INFER_URL`); the site config is not changed:

```bash
ssh -N -L 127.0.0.1:8200:127.0.0.1:8200 Luikz@192.168.137.135          # keep this running
scp Luikz@192.168.137.135:argonix/argonix_app/data/infer_token data/infer_token
uv run python run.py --config configs/real_site.json --infer-node ws://127.0.0.1:8200
```

Measured 2026-09-26, laptop → RTX 4050 host over the Wi-Fi hotspot and the SSH tunnel, 2 cameras at 5 fps: both held
5.0 fps with a round trip of 80–95 ms per frame and no CPU fallback (the laptop's own CPU needs about 200 ms per
frame). The System page's Compute nodes card shows the node as up with its GPU.

**Security.**
- **Token:** on first start the node creates a random bearer token in `data/infer_token` (git-ignored; 0600 where the OS supports it). Copy it to each client, or set `ARGONIX_INFER_TOKEN`.
- **Endpoints:** `GET /health` is an unauthenticated liveness probe. `/v1/info`, `/metrics` and `WS /v1/stream` need `Authorization: Bearer <token>`, and a WebSocket without it is refused before the handshake completes.
- **Transport:** frames travel **unencrypted**, so the node listens on 127.0.0.1 and clients come in through an SSH tunnel. Don't bind it to a network interface.

**Node behaviour.**
- **Tracking:** one ByteTrack per (client, camera). A reconnect within `--tracker-ttl-s` (30 s) resumes the same tracker, so track IDs survive a network blip.
- **Backpressure:** at most `--max-inflight` (2) frames per camera are decoding or waiting for the GPU. A newer frame replaces the one waiting ("superseded"), and a frame that waited longer than `--max-age-ms` (1000) is dropped ("stale"). Every reply carries the frame's sequence number, so a client never waits for a frame the node discarded.
- **Object classes:** unattended-object analysis needs the node started with `--extra-classes 24,26,28`. The site logs a warning when the node does not detect the classes it asked for.

**Client config** (site JSON):

```json
"detector": {"remote": {"url": "ws://100.x.y.z:8200", "token_file": "data/infer_token", "fallback": "local"}}
```

- **One WebSocket per camera.** Each frame is downscaled to the node's input size, JPEG-encoded (quality 80) and sent. The call waits for the reply, so the boxes always belong to the frame being analysed (no pipelining, no reused boxes).
- **Adaptive:** every 12 results the median round trip is compared with `target_rtt_ms` (250). The frame size and JPEG quality step down on a slow link and up again on a fast one, never below `min_side` (256). A step down that does not cut the round trip by 5 % is undone, because the link is latency-bound and detection quality would be lost for nothing. The reply timeout follows the measured round trip (3 × p95, between 1 and 5 s).
- **Self-clocked:** one frame per camera is in flight, so on a slow link a camera's analytics rate falls to about 1 / round trip instead of queueing stale frames. `uplink_kbps` sets an optional upload budget shared by all cameras.
- **Node down:** frames return no detections until `fallback_after_s` (10 s) has passed. After that, with `"fallback": "local"`, a CPU YOLO26n at imgsz 320 takes over until the node is back. Track IDs are remapped, so a switch of tracker never reuses an ID for a different person. Each alert's evidence names the model that actually produced the detections.

**Compute nodes card.** The System page lists the inference node and the VLM node from `GET /api/nodes` (`alert.read`):
- **Remote inference node:** its state (up, down, unauthorized, error), GPU name, utilisation and memory, queue depth, mean batch, latency and input fps.
- **Per camera:** round trip, drops, timeouts, fallback frames and KB per frame.
- **In-process detector:** its model, when there is no remote node.

Each probe has a 3 s timeout, and one dead node never breaks the page. The token never reaches the browser.

**Benchmark.** Run this from the camera machine:

```bash
uv run python tools/bench_remote.py --url ws://<gpu-host>:8200 --token-file data/infer_token --out bench_remote.md
```

It measures five phases:
- WebSocket link latency;
- JPEG quality × frame size;
- 1…N cameras back to back;
- N cameras at a paced fps;
- the same cameras on this machine's CPU, for comparison.

**No remote-inference numbers are committed yet.** Results depend on the network: label any you quote with both
machines and the link used.

## Troubleshooting

Run `uv run python tools/preflight.py --config <your config> --data <run dir> --port <port>` first. On a GPU host, run
it with `uv run --no-sync`. It prints PASS, WARN or FAIL for each item below that it can check, and exits non-zero on
any FAIL. Add `--url http://127.0.0.1:<port>` and `--vlm-url http://127.0.0.1:8100` to check services that are already
running.

| Symptom | Cause and fix |
|---|---|
| The dashboard asks for a **setup code** | No account exists yet. The code is printed at start-up (in `app.log` when `win_services.ps1` started the app) and written to `<data>/setup_code.txt`. A browser on the server itself does not need it. For a local demo, `--open-demo` skips sign-in until the first admin is created. |
| The dashboard works on the host but **not from another machine** | Windows Firewall blocks inbound connections to the port. Use a tunnel: `ssh -N -L 18000:127.0.0.1:8000 user@host`, then open http://127.0.0.1:18000. Or, as an administrator on the host, allow the port: `New-NetFirewallRule -DisplayName "Argus 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`. |
| **CUDA torch turned into `+cpu`**, or TensorRT vanished | A plain `uv run` or `uv sync` re-synced the default environment. Run `uv sync --extra cu126` (driver ≥ 527; `cu128` for newer drivers), reinstall TensorRT if you use it, and from then on use **`uv run --no-sync`**. |
| Services **die when the SSH session closes** (Windows host) | Windows OpenSSH kills every child process on disconnect. Start them with `tools\win_services.ps1 start` (or `start-app`, `start-vlm`, `start-infer`). It launches them through `Win32_Process`, so they survive. They do not survive a reboot. |
| `running scripts is disabled on this system` | The PowerShell execution policy blocks the script. Run it as `powershell -NoProfile -ExecutionPolicy Bypass -File tools\win_services.ps1 <action>`. |
| A **`.engine` fails to load**, or the log says it fell back to `.pt` | TensorRT engines are specific to the GPU and the TensorRT version. The app then runs the sibling `.pt` in fp16. Rebuild the engine on that GPU with `tools/bench_gpu.py --export` (see [GPU](#gpu-one-batched-detector-for-all-cameras)). |
| **WinError 10013** when binding a port | On Windows this usually means the port is inside a range reserved by Hyper-V, WSL or Docker, not that another program has it. On the development laptop 8031–8130 was reserved. Check with `netsh int ipv4 show excludedportrange protocol=tcp` and pick another port. |
| A second `uv sync` **hangs** | Another `uv sync` holds uv's lock. Wait for it to finish. |
| **VLM model downloads hang** | `huggingface_hub` downloads (including the xet backend) hung on the test hosts. `tools/vlm_server.py` uses its own parallel ranged downloader. At run time set `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_XET=1`, which `win_services.ps1` does. |
| **No alerts** appear | First check that Live shows boxes. Then check that the zones have rules and that the rule's `min_seconds` has passed. `running`, `wrong_way` and the crowd rules need a calibrated camera (see the camera's `calib_info`). A rule that fired re-arms only after its re-arm time. |
