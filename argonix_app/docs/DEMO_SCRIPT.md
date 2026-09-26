# Argus demo script (9–12 minutes)

A walkthrough for showing Argus to friends and faculty. Every number below is labelled with where it was
measured. Do not present estimates as demonstrated capability (AGENTS.md).

## 0. Before the audience arrives (10 min)

Pick one setup.

**A. GPU host (preferred): LEGION, RTX 4050**

```powershell
ssh -o UserKnownHostsFile=$env:TEMP\argonix_known_hosts Luikz@192.168.137.135 "powershell -File %USERPROFILE%\argonix\argonix_app\tools\win_services.ps1 status"
ssh -N -L 18000:127.0.0.1:8000 -L 18100:127.0.0.1:8100 -L 8200:127.0.0.1:8200 -o UserKnownHostsFile=$env:TEMP\argonix_known_hosts Luikz@192.168.137.135
```

Open http://127.0.0.1:18000 and sign in. Everything on LEGION listens on 127.0.0.1 only, so the tunnel is the way in.
Port 8200 is LEGION's GPU inference node, which setup B uses.
If a service is down, run `win_services.ps1 start` on the host (it launches detached; plain `ssh ... python` dies on
disconnect).

**B. This laptop only (CPU, no sign-in)**

```bash
cd argonix_app
uv run python run.py --config configs/real_site.json --data data/run_demo --port 8003 --open-demo --infer-node ws://127.0.0.1:8200
```

With the tunnel from setup A open (and `data/infer_token` copied from LEGION), detection runs on LEGION's GPU: both
cameras at the full 5 fps, about 90 ms per frame round trip. Without the tunnel it falls back to the laptop CPU after
10 s, which runs 2 cameras at about 3 fps analytics. That is enough for every page. The "Explain" button needs the VLM
on the GPU host.

**Preflight** (`tools/preflight.py`). It prints PASS/WARN/FAIL for:
- Python and torch versions, and CUDA;
- the detector weights the config will load;
- every camera source (clips are opened, RTSP is probed with a short timeout);
- the data dir and free disk, and the port;
- Re-ID weights and TensorRT engines;
- the running app and VLM, when you pass `--url` / `--vlm-url`.

It exits non-zero on any FAIL: fix those before the audience arrives.

```bash
# setup B, before starting the app
uv run python tools/preflight.py --config configs/real_site.json --data data/run_demo --port 8003 --open-demo
```

```powershell
# setup A, on LEGION against the running services (the port check is skipped with --url)
ssh -o UserKnownHostsFile=$env:TEMP\argonix_known_hosts Luikz@192.168.137.135 "cd %USERPROFILE%\argonix\argonix_app && uv run --no-sync python tools/preflight.py --config configs/real_site_gpu.json --data data/run_real_site --url http://127.0.0.1:8000 --vlm-url http://127.0.0.1:8100"
```

**Checklist**
- [ ] Overview shows 2/2 cameras online and a detection time in the status bar.
- [ ] Live shows boxes on both clips.
- [ ] Review has at least a few alerts (intrusion on the campus lawn appears within about a minute).
- [ ] System → **Demo scenario** lists `intrusion_door_tamper`. Scenarios run only with `--open-demo` or
      `"demo": {"scenarios": true}` in the config (`real_site_gpu.json` has it); otherwise the Run button gets a 409.
- [ ] System → audit card says the chain is intact.
- [ ] Theme: light for a projector, dark for a laptop screen (moon icon, top right).
- [ ] Close other heavy apps; the CPU path drops frames rather than lagging (see System → dropped).

## 1. The problem (30 s)

"Most CCTV is watched by nobody, and systems that do raise alerts cannot say *why*. Argus plugs into the cameras a
site already has (RTSP/ONVIF), reports **what was seen** with the evidence attached, and ranks alerts so an operator
looks at the right one first. It never labels a person as dangerous; it reports observable events."

## 2. Overview → Live (1 min)

- **Overview**: KPIs, the cameras with open alerts first, a short triage queue.
- **Live**: two real public clips (OpenCV `vtest.avi` campus path, Kaggle street clip). Point out the boxes, track
  IDs and zones.
- Mention black-and-white/IR: the pipeline detects monochrome input and applies CLAHE low-light enhancement
  automatically.

## 3. An alert, end to end (2 min): the core of the demo

Open **Review**, pick an `intrusion` alert on the lawn, and open the drawer.

1. **Observation**: "person foot-point inside *Lawn (keep off)* for 1.2 s". That is what was seen.
2. **Derivation**: rule, threshold and duration, plus the calibration error used to keep the boundary conservative.
3. **Why this priority**: the factor breakdown (severity, zone weight, corroboration, duration). Learned models can
   only *corroborate*: an alert raised only by a learned model is capped at P3 until a deterministic rule agrees.
4. **Evidence**: snapshot and clip. Switch between **Original** (the frame exactly as analysed, nothing drawn on it)
   and **With overlay** (boxes and zones, stamped "software overlay"). Export the hash-chained evidence zip; tick
   *anonymised copies* to add blurred copies (heads and faces), labelled as derived.
5. **Explain** (GPU host): the VLM writes a description of the snapshot. It is labelled assistive and never triggers
   or changes an alert.
6. **Workflow**: the drawer offers only the legal next steps for your role. **Escalate** asks for a reason. The chip
   shows how long the alert has waited against its acknowledge SLA (P1 60 s, P2 5 min). Acknowledge, then resolve
   or dismiss with a disposition. Dispositions feed per-rule precision (Wilson interval) on System.
7. **Legal hold** (supervisor): hold the alert. Retention will never delete it, its evidence, or the overlapping
   recordings. Both the hold and the release are audited.

## 3b. A guaranteed incident: the scripted scenario (1.5 min)

Use this when live footage is quiet, or as the finale. System → **Demo scenario** → **Run demo scenario**
(`intrusion_door_tamper`, 30 s). Say up front that it is scripted: every alert and incident it creates carries an
orange **DEMO** badge and a `[DEMO]` prefix, and it never mixes with observed alerts or precision stats.

1. t+0 s, restricted-zone entry; t+8 s, a "door forced" access-control report. They correlate into **one P1
   incident** ("Why P1?" shows sensor + video). A toast offers **Open cameras**. With *Auto-open cameras on P1/P2*
   switched on (user menu or Live toolbar), Live jumps to the incident's cameras by itself.
2. t+15 s, the camera is covered: a camera-integrity incident opens and mutes (but still shows) that camera's alerts.
3. t+30 s, recovery.
4. Open the incident:
   - **Synchronised replay** plays every involved camera on one clock, with event markers on the timeline;
   - **Summarise** (GPU host) writes an assistive narrative that must cite real alert ids, or it is refused;
   - **Report** opens a self-contained HTML report with the "why this priority" factors, the timeline split into
     observation and derivation, evidence hashes and a SHA-256 footer.

On a slow CPU the 15 s cover may raise no health event (analytics must keep running for 3 s during the cover). Use
setup A for this beat when you can.

**Real door event** (optional, `configs/demo_site.json` only): with `ARGONIX_SENSOR_SECRET_DOOR1` set in both shells,
`uv run python tools/sensor_sim.py forced --door D1` while the synthetic person walks into the server-door zone gives
the same fusion from a signed webhook instead of a script.

## 4. Site picture (1.5 min)

- **Map**: bird's-eye site map from the homography, tracks as dots, the heatmap toggle. Say that the ground plane is
  2D with a stated calibration error. **Heatmap time-lapse → Last 24 h** replays occupancy in 15 min or 1 h steps
  (it only has data from while the server was running).
- **Analytics**: counting lines, zone occupancy and dwell, density per m², queue length and wait (Little's-law estimate
  next to the measured median), crowd-flow events (counter-flow, dispersal, convergence), and the hotspot heatmap.
  The anomaly indicators (IsolationForest, occupancy CUSUM) are labelled *corroborating only*.
  **Congestion forecast** shows a zone's next 15–60 min only when it beat both "last value" and "usual for this hour"
  on held-out history. A fresh demo instance has too little history, so it will say so: that gate is the point.
- **Incidents**: related alerts grouped into one incident with a graph (cameras, zones, tracks, time). Open one and
  walk the timeline.

## 5. Across cameras (1.5 min)

- **Journeys** (needs Re-ID on; otherwise use *Preview with sample data*, which is labelled as sample data): one
  global ID with sightings per camera. Every hop shows its link confidence and evidence (topology, travel time against
  the learned band, appearance). Say that this is **probabilistic and revisable**: the operator can split a wrong
  link.
- **Journeys with real pedestrians** (a second instance, set up before the demo). The three cameras show real people
  cut out of `vtest.avi`, walking a simulated floor plan, with a different real frame of each person on every camera
  and a colour/blur change per camera:

  ```bash
  uv run python tools/make_reid_demo.py     # once; renders data/reid_eval/scene/ on the CPU
  uv run python run.py --config configs/reid_demo.json --data data/run_reid --port 8004 --open-demo
  ```

  Rendering the 480 s scene on the laptop CPU takes long (a 60 s test took 7 min under load). Copy LEGION's
  `data/reid_eval/scene/` instead: it is the scene the numbers below were measured on. Run this in PowerShell from
  `argonix_app`, then run the tool, which reuses the scene and writes the config:
  `mkdir -Force data/reid_eval; scp -r -o UserKnownHostsFile=$env:TEMP\argonix_known_hosts Luikz@192.168.137.135:argonix/argonix_app/data/reid_eval/scene data/reid_eval/`.
  In a check on the laptop, with other heavy jobs running, the three cameras ran at well under 1 fps. So start this
  instance early, or run it on the GPU host.

  Open http://127.0.0.1:8004/#/journeys after a minute or two, once walkers have crossed cameras. Measured on this
  scene (README, RTX 4050): fused precision 0.61 and recall 0.36 on 31 true links. Some links **will** be wrong: pick
  one and use **Split here** to show the operator correction. Say that global IDs are hints to verify on the crops,
  never an identity.
- **Next-camera prediction** (`configs/handoff_demo.json`): when a track leaves a camera, it predicts where and roughly
  when the person reappears, with no appearance matching. Measured on the synthetic handoff scene: 96 % vs 88 % for the
  baseline in simulation, and 86 % vs 81 % on rendered video, which is **not** statistically significant. Say that.

## 6. Ask the system (1 min)

**Investigate**: type "intrusions on the lawn in the last hour" or "P2 alerts on the street camera today". The
question becomes an **allow-listed structured query** (cameras, zones, rules, priorities, time), checked against the
site's own vocabulary. The model never writes SQL and never sees the database. Without the GPU host, a deterministic
keyword parser does the translation. The summary cites alert and incident ids from the results. An LLM-written
summary is kept only when every id it cites is real; otherwise the template summary is used.

## 7. Operations (1 min)

- **Cameras**: add an RTSP camera (vendor templates, *Test* opens the stream and returns resolution, fps and a
  snapshot), ONVIF discovery, draw zones and masks. A **Privacy mask** is blanked on every frame before anything is
  stored, analysed or shown. Zones and counting lines can carry schedules ("after-hours" windows).
- **System**: camera health (tamper, blur, down), fps and detection time per camera, compute nodes, per-rule
  precision, access-control sources, the retention policy and last sweep, and the audit chain check (editing any
  audit row is detected at that row).
- **Handover**: what is still open at shift end (open and overdue alerts, camera problems, top zones), notes, and a
  printable sheet.
- Distributed GPU: cameras can send frames to a GPU inference node (`infer_server.py`); one batched detector serves
  all cameras.

**Tamper beat (1 min).** You need a role with `settings.edit`: the admin, or anyone in `--open-demo`.
1. Open **Live → a camera** (for example the campus clip). That camera's page has a **Tamper test** panel.
2. Choose **30 s** and click **Cover** (or **Black**). Argus's own analytics frames become a flat grey (or black)
   picture, so the live view and the recording show the fault. The camera itself is untouched.
3. After about 3 s (the persistence window), **System → View health** for that camera shows **DEGRADED** (`covered`
   or `dark`). **Review** gets a *Camera view problem* alert, which opens a camera-integrity incident. While that
   incident is open, new rule alerts from this camera are still shown but **muted** (not pushed): the evidence from a
   covered camera is not trustworthy. Show that banner on **Incidents**.
4. When the 30 s end, the next health check (about 1 s later) clears the condition, and the camera returns to **OK**.
   The incident cools after 60 s and closes 240 s later.

Don't use **Freeze** on stage. A frozen picture is only flagged after the camera's `health.freeze_s` (30 s by
default, 600 s in the handoff and Re-ID demo configs), which is as long as or longer than the panel's longest
injection.
**DOWN** means frames stopped arriving, and an injection cannot cause it. To show DOWN, stop a real stream: for
example, stop one MediaMTX publisher in the RTSP replay setup. The camera goes DOWN after `down_grace_s` (2 s), and
the grabber then reconnects with backoff.

## 8. Experimental features (only if asked)

All off by default and labelled experimental in the UI and API.
- **Trajectory forecasting**: 4.8 s ahead. On ETH/UCY (leave-one-scene-out) it is about tied with constant velocity
  (ADE 0.53 m vs 0.53 m; FDE 1.11 m vs 1.15 m). It is only clearly better on the campus clip (FDE 1.83 m vs 2.03 m).
- **Person on the ground / possible fall**: posture only, never a claim about injury. On the public CAVIAR clips it
  found 0 of 3 real falls, because the small detector misses the tiny people in them, and it raised no false alarms.
- **Weapon-like objects**: the fine-tuned model scores handgun AP50 0.91 on its own test split, but finds only 8 % of
  weapons in CCTV-style views and raises 137–151 false events per hour on normal footage. It is a demo of
  corroboration capped at P3, not a detector.
- **Possible altercation**: the pose pair-GRU reaches ROC AUC 0.68 on RWF-2000 test (heuristic 0.65) and caught 11
  of 200 fight clips in the runtime path, with about 22 false events per hour on non-fight clips.
- Never present any of these as reliable detection.

## Numbers you may quote (with where they were measured)

| Claim | Measured on |
|---|---|
| 6–14 ms per detection call; about 20 cameras at 10 fps with batched TensorRT YOLO26s | RTX 3050 laptop (Linux) |
| **Not** LEGION: on the RTX 4050 Windows host, 2 cameras at 10 fps strictly, about 4 in practice (host-limited, GPU mostly idle) | RTX 4050 laptop (Windows) |
| MOT17 tracking: HOTA 0.34, detection precision 0.90 but recall 0.32 (small distant people) | RTX 4050, 7 MOT17 train sequences |
| About 200 ms per frame for 2 cameras at imgsz 480 | This laptop, CPU only |
| Qwen2.5-VL-3B 4-bit: about 2 s per explanation | RTX 3050 |
| Self-calibration: 0.29 m ground error at 10 m | Synthetic scene only |
| Median walking speed 0.93 m/s (campus) and 1.46 m/s (street) | Real clips, hand calibration |
| Re-ID single-camera ROC AUC 0.89 (OSNet-AIN); cross-camera fused precision 0.61, recall 0.36 on 31 links | RTX 4050, `vtest.avi` and the rendered 3-camera scene |
| Weapons handgun AP50 0.91 on the test split; 8 % recall on CCTV-style views | RTX 4050, Kaggle weapon dataset / synthetic CCTV set |
| Altercation ROC AUC 0.68 vs heuristic 0.65 | RTX 4050, RWF-2000 test (400 clips) |

## If something breaks

| Symptom | Fix |
|---|---|
| Dashboard asks for a setup code | It is printed in `app.log` on the host, or use `--open-demo` locally |
| Explain button missing or failing | VLM down: `win_services.ps1 start-vlm` on LEGION, or skip section 3.5 |
| Live tiles frozen | System page → camera health; restart the app. Clips loop automatically |
| Journeys empty | Re-ID is off on this config: use *Preview with sample data* and say so, or switch to the `reid_demo` instance (section 5) |
| LEGION unreachable | Switch to setup B on this laptop |
| Run demo scenario fails with 409 | The instance runs without `--open-demo` and its config lacks `"demo": {"scenarios": true}`. Restart with one of them |
| No alerts firing | Check that Live shows boxes. If it does not, the detector is not running: see System → Detect and `app.log`. The campus lawn intrusion needs about a minute after start. Rules re-arm only after their re-arm time, so a clip loop does not repeat every alert at once. For a guaranteed alert right now, run the demo scenario (section 3b) or the tamper beat (section 7) |
| Projector washes out the dashboard | Light theme (moon icon, top right), browser zoom 125 %, F11 for fullscreen. `f` makes a Live tile fullscreen |
| SSH tunnel or hotspot dropped (page stops updating) | Re-run the `ssh -N -L 18000:… -L 18100:…` command from section 0. If the hotspot itself is down, reconnect LEGION to it, or switch to setup B |
| Laptop on battery | Plug in and set Windows power mode to *Best performance*. On battery the CPU may throttle, and the CPU path then drops more frames (System → Dropped) |
| GPU out of memory on LEGION (CUDA OOM in `app.log` or `vlm.log`) | The 6 GB card is shared by the detector and the 4-bit VLM (`tools/weapons_job.ps1` pauses the VLM while it trains for this reason). Stop anything else on the GPU (`nvidia-smi`), then `win_services.ps1 stop-vlm`, `start-app`, `start-vlm`. If it recurs, run without the VLM and skip section 3.5 |
| Port already in use, or WinError 10013 | Another instance is running (pick another `--port`), or the port is in a Windows reserved range (`netsh int ipv4 show excludedportrange protocol=tcp`). The preflight reports which |
