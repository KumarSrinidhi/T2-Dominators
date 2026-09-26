# Argus: documentation vs implementation audit (2026-09-25)

> **Update 2026-09-26.** Since this audit, the following have been built and are in the READMEs:
> - Tier A: community files, the Re-ID demo config, retention, clean evidence frames, the hash-chained audit log, README sections, preflight and troubleshooting.
> - Tier B: line events and schedules, **baseline-aware dwell**, the workflow rules including `unclear`, auto-open cameras, **"what changed vs normal?"**, face blur and the **live-view/evidence access log**, CAVIAR/MOT17 evaluation runs, CI, the signed door-sensor stream, **robust N-point calibration** (RANSAC + leave-one-out error).
> - Tier C: WebRTC live view, main-stream recording, synchronised replay, the congestion forecast, vehicle rules, and **camera recovery events**.
>
> Priority calibration from dispositions is covered by the operator **track record** in "Why P?" and the optional **feedback demotion**. That is Wilson intervals per rule and camera, not Platt scaling. Still open from Tier C 23: per-zone coverage loss, adaptive day/night reference, 401/403 lockout backoff and NTP clock offset. The deck (M1) is the user's own. The tables below are kept as written on 2026-09-25.

This audit checked every `.md` document in the repo root and `manual/sections/00a…35`, plus 90 (build plan) and 91
(decisions), against the code on `main`. Nine read-only agents did the checking, and I merged and de-duplicated their
findings. "Done" means the feature is implemented **and** wired in (config, API or UI).

Some items are marked **diverged by user decision**; they are not gaps:
- SQLite only (no Postgres/Timescale/pgvector);
- no message bus and no Docker/Kubernetes;
- vanilla-JS console instead of React;
- licences later;
- fall detection built anyway, off by default.

## 1. What is done

| Area | Done |
|---|---|
| Cameras | RTSP/ONVIF discovery, 7 vendor templates, stream test, CRUD, RTSP over TCP, reconnect with backoff, latest-frame grabbing, B/W–IR conditioning, MediaMTX replay harness |
| Detection and tracking | Model-agnostic backend (local, batched GPU, remote node), YOLO26 + ByteTrack, TensorRT/fp16, one batcher for all cameras, CPU fallback |
| Geometry | Ground-plane homography with a stated error, pedestrian self-calibration, bird's-eye site map and heatmap, pixel-space zones for uncalibrated cameras |
| Rules | Intrusion, dwell ("loitering"), running, wrong-way, crowding, counter-flow, dispersal/convergence, density per m², queue (Little's law), unattended object, camera health/tamper |
| Baselines and anomaly | Hour-of-week median/MAD, CUSUM, IsolationForest (corroboration only, capped at P3), hotspots |
| Correlation | 15 s alert grouping, incidents (lifecycle, merge/split, reopen), incident graph, "why this priority" factors, learned-only P3 cap, push budget, signed webhooks, tamper inhibition |
| Operator | Review drawer (observation vs derivation), dispositions with per-rule Wilson precision, hash-chained evidence export, audit table, Journeys corrections |
| Console | Overview, Live wall, Review, Incidents, Analytics, Journeys, Map, Investigate, Recordings, Cameras, System, Users/Account, zone/mask/line editor, demo mode, themes |
| Security | Accounts, 5 roles/permissions, sessions (idle/max), Basic + API tokens, CSRF, security headers/CSP, first-run setup, login throttling, TLS option, credential redaction |
| Research layer | Re-ID (OSNet) + probabilistic, revisable global IDs; next-camera prediction and topology; trajectory forecast (GRU vs CV); fall (off); weapon/altercation (off, P3 cap); VLM explain; NL investigation (allow-listed queries, cited ids) |
| Distributed | GPU inference node (`infer_server`), remote client, compute nodes card |
| Measured (labelled) | GPU scaling (RTX 3050), handoff, forecast (ETH/UCY), fall (CAVIAR), **weapons and Re-ID (RTX 4050, today)** |
| Ops | Bootstrap and service scripts, backup (Windows + Linux), demo script, 300 tests green |

## 2. What is left, ranked by demo value vs effort

### Tier A: high value, small effort (hours)
1. **Round-1 deck.** The scored deliverable; missing. Build it from manual §27 slide copy, `DEMO_SCRIPT.md` and the measured numbers.
2. **Open-source community files.** 15 rubric points. Add CONTRIBUTING, CODE_OF_CONDUCT, SECURITY and model cards (paste-ready drafts in manual §27). LICENSE waits, per the user.
3. **Re-ID demo config.** No config enables `reid`, so Journeys shows only sample data. Add `configs/reid_demo.json` built on the rendered 3-camera scene from `eval_reid`.
4. **Retention sweep.** Recordings, evidence and alerts are never deleted (about 450 MB in `data/run_real/evidence` already). Add a size/age cap that skips legal holds, and wire the orphaned `evidence.hold` permission.
5. **Clean evidence frames.** Snapshots and clips have boxes burned in. Save the raw frame next to the annotated one, labelled "software overlay".
6. **Hash-chained audit log.** Add `prev_hash`/`hash` columns, `/api/audit/verify` and a "chain intact" badge.
7. **README sections for Analytics, Incidents, Investigate and compute nodes.** They are built but undocumented. Retake the screenshots on the new UI.
8. **Demo safety.** Add a tamper beat to the demo script, a `tools/preflight.py` PASS/FAIL check, a recorded 2–3 min fallback video, and a troubleshooting table in the README.

### Tier B: medium effort (half a day each)
9. **Rule additions:** line-crossing events per direction; zone/rule **schedules** (after-hours); baseline-aware dwell (loitering threshold from the zone's p95 dwell).
10. **Workflow rules on the server:** reject new → resolved; require a reason on escalation; add an `unclear` disposition; acknowledge-SLA ages and a flood banner in the shell; wall-clock escalation of unacknowledged P1/P2 incidents.
11. **Auto-open the recommended cameras** (primary, related, predicted next) on a P1/P2 incident, and add a "what changed vs normal?" intent to Investigate (spec demo steps 8 and 10).
12. **Privacy:** face/person blur on export and before VLM calls; log live-view and `/media` reads.
13. **Evaluation runs:** CAVIAR rule precision/recall (`eval_datasets.py` exists, never run), MOT17 IDF1/HOTA, end-to-end latency p50/p95. Commit the result JSONs.
14. **Minimal CI:** a GitHub Actions workflow running pytest with a `slow` marker, plus ruff; `weights_only=True` for the three checkpoint loads.
15. **Non-video data stream:** an HMAC-signed `POST /api/sensors/events` (door forced/held) that is correlated into incidents with video. This covers "multiple data streams" in the problem statement.
16. **Robust N-point calibration:** MAGSAC plus a leave-one-out RMSE grade, to replace the exact 4-point solve.
17. **Priority calibration from dispositions:** Platt scaling per rule, shown as "calibrated p".

### Tier C: larger (a day or more)
18. **Smooth live video:** WebRTC/WHEP via MediaMTX; today it is 5 fps MJPEG of annotated frames.
19. **Main-stream recording:** stream copy instead of re-encoded analytics frames.
20. **Synchronised multi-camera replay** in the incident view.
21. **Zone congestion forecast:** "next 5 min", with MAE against naive.
22. **Vehicle analytics:** vehicle-in-pedestrian zone, vehicle stopped.
23. **Camera health depth:** per-zone coverage loss, recovery events, adaptive day/night baseline, 401/403 lockout backoff, NTP clock offset.

### Deferred or diverged (not planned)
Postgres/Timescale/pgvector, message bus, Docker/Compose/Kubernetes, React/MapLibre, Prometheus/Grafana/tracing, MFA/OIDC/mTLS, DPIA and compliance paperwork, BACnet/OSDP/audio, density-map crowd models, SigLIP semantic search, RFC 3161/BagIt evidence.

## 3. Per-milestone status (manual §90)

| Milestone | Status |
|---|---|
| M0 bootstrap | Partial: no CI or lint; contracts diverged |
| M1 deck | **Not started** |
| M2 data layer | Diverged (SQLite); audit not chained |
| M3 ingest/record/health | Mostly: recording is analytics frames; no retention |
| M4 detect/track | Mostly: tracker IDF1 never measured |
| M5 geometry | Mostly: no robust N-point fit |
| M6 rules | Mostly: no schedules or line-crossing events; precision gate not enforced |
| M7 API/console | Done (diverged stack) |
| M8 correlation/alerting | Mostly: no calibration from feedback |
| M9 security/privacy/ops | Partial: retention, anonymisation and CI missing |
| M10 evaluation/rehearsal | Partial: tools exist; few committed runs |
| M11 research stretch | Done (beyond scope); all corroboration-only |
