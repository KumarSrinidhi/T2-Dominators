export const meta = {
  name: 'argonix-implementation-manual',
  description: 'Research and write a verified top-to-bottom implementation manual for Argus (foundation, 27 topic sections, critique, gap-fill, harmonize, build plan)',
  phases: [
    { title: 'Resume scan', detail: 'read durable state in manual/state; skip steps already done' },
    { title: 'Foundation', detail: 'pinned stack matrices + architecture/contracts, each verified' },
    { title: 'Research', detail: 'one deep-research writer per topic, then an adversarial fact-checker' },
    { title: 'Critique', detail: 'completeness + consistency critic across all sections' },
    { title: 'Fill', detail: 'write missing sections, deepen thin ones' },
    { title: 'Harmonize', detail: 'consolidate versions, apply cross-section fixes' },
    { title: 'Synthesize', detail: 'master build plan + decision register' },
    { title: 'Assemble', detail: 'join all sections into IMPLEMENTATION_MANUAL.md' },
  ],
}

const ROOT = 'C:\\Users\\Ivermectin\\Desktop\\Argonix'
const SEC = ROOT + '\\manual\\sections'
const BASH_SEC = '/c/Users/Ivermectin/Desktop/Argonix/manual/sections'
const DATE = (args && args.date) || '2026-09-23'
// FAST mode (user deadline): non-foundation writers and all later steps use the faster model; separate
// fact-check, critique, gap-fill and deepen passes are skipped. Foundation sections keep full depth.
const FAST = !!(args && args.fast)
const FASTOPTS = (heavy) => (FAST ? { model: 'sonnet', effort: heavy ? 'medium' : 'low' } : {})
const LENGTH_RULE = FAST
  ? 'LENGTH (deadline mode): target 5,000-8,000 words, hard cap 10,000. Prioritize what an implementer needs most: exact versions, install commands, config keys with defaults, the core reference code, integration contracts, tests/acceptance criteria, pitfalls. Live-verify the 15-20 facts most likely to be wrong or stale (versions, licenses, API signatures, CLI flags) and do not re-check stable facts. If the file already exists and is long, only finish its missing format parts, concisely. Work efficiently: finish within about 30 minutes.'
  : 'Typical length 6,000-14,000 words; go longer if the topic needs it.'
const base = (p) => String(p).split(/[\\/]/).pop()
const STATE = ROOT + '\\manual\\state'
const BASH_STATE = '/c/Users/Ivermectin/Desktop/Argonix/manual/state'

function durableNote(key, label) {
  return `DURABLE STATE + CONTINUOUS LOGGING (mandatory: this is how the run survives crashes and machine restarts):
- As your very first action, run with the Bash tool: mkdir -p ${BASH_STATE} && echo "$(date -Iseconds) | START | ${label}" >> ${BASH_STATE}/progress.log
- After each major milestone (research finished, each part of a file written/appended, each batch of checks), append a line the same way: echo "$(date -Iseconds) | PROGRESS | ${label} | <short note>" >> ${BASH_STATE}/progress.log
- Immediately BEFORE returning your structured result, write that exact result object as valid JSON to ${STATE}\\${key}.json with the Write tool, then run: echo "$(date -Iseconds) | DONE | ${label}" >> ${BASH_STATE}/progress.log
- On an unrecoverable problem, log: echo "$(date -Iseconds) | FAIL | ${label} | <reason>" >> ${BASH_STATE}/progress.log and still return the best result you can.`
}

const CORPUS = `Existing corpus in ${ROOT} (read only the parts relevant to your topic; build on it and go deeper rather than re-deriving what it already settles):
- AGENTS.md (constraints; already in your context)
- INTELLIGENT_SITUATIONAL_AWARENESS_DEEP_RESEARCH_VERIFICATION.md (highest authority: claim verdicts, capacity model sections 6-7, realistic MVP section 31, experiments section 33 + Appendix E)
- intelligent_threat_detection_situational_awareness_project.md (original spec, features F1-F42)
- INTELLIGENT_SITUATIONAL_AWARENESS_EXTREME_RESEARCH.md (81-section design rationale; DDL sketches section 32; event schemas section 33; APIs section 35)
- RESEARCH_BRIEF_TRACKS.md and RESEARCH_EXTENSION_2026-09-22.md (source-linked facts, live refresh 2026-09-22)
- verification1.md (older audit; superseded on conflict)
- Problem_statemen.txt (hackathon brief + Round 1 rubric)
Facts to respect: the MinIO community repo was archived 2026-04-25 (evaluate alternatives); the MOTChallenge evaluation service is closed (static archive); DOI 10.1007/s11263-020-01393-0 is the CORRECT DOI of Dendorfer et al. MOTChallenge IJCV 2021 (HOTA is 10.1007/s11263-020-01375-2), do not "correct" it.`

function preamble(s) {
  const foundationNote = s.foundation
    ? 'You are writing a FOUNDATION section: every other section will conform to it, so be decisive, precise and complete.'
    : `FIRST read the foundation sections and conform to them (package names, module layout, data contracts, pinned versions, license posture): ${SEC}\\00a-stack-ml-video.md, ${SEC}\\00b-stack-platform.md, ${SEC}\\01-architecture-contracts.md. If you believe a foundation choice is wrong, or a component/version you need is missing there, still write your section coherently, add an explicit "Deviations from foundation" note, and report it in conflicts_with_foundation.`
  return `You are writing one section of the Argus IMPLEMENTATION MANUAL: the single, self-contained, top-to-bottom reference the team will build from. Today is ${DATE}. The user's explicit goal: during implementation nobody should need to stop and research anything. So every version, install command, API call, config key, default parameter, algorithm detail, license constraint and pitfall needed for this topic must be in your section, verified against live primary sources.

Project context: Argus is an intelligent threat-detection / situational-awareness system for a hackathon (Problem_statemen.txt): multi-camera video analytics (plus other real-time data streams) producing explainable, evidence-backed, prioritized alerts for authorized human operators. Realistic MVP (AGENTS.md): 2-4 cameras, RTSP main-stream recording + 5-fps sub-stream analytics, detector behind a model-agnostic interface, ByteTrack/BoT-SORT, foot-point homography + zones, deterministic rules, event correlation, evidence clips, calibrated explainable alerts, MapLibre/WebSocket operator UI, PostgreSQL + TimescaleDB + pgvector + object storage. Research layers (Re-ID, VLM, forecasting, learned anomaly) come later but must also be documented so they can be built without new research. The development machine is Windows 11 (the team may use WSL2 + Docker Desktop); the production target is Ubuntu 24.04 LTS with an NVIDIA GPU. Also document CPU-only / non-NVIDIA fallbacks where relevant, because hackathon hardware is uncertain. The Python package / project name is "argonix".

${CORPUS}

YOUR SECTION: ${s.id} - ${s.title}
Write it to this exact file: ${SEC}\\${s.file}
RESUME RULE: if that file already exists, it is partial output from an earlier run of this same task that was interrupted by a machine restart. Read it fully, keep what is correct (spot-check its key facts), and CONTINUE and complete it from where it stops (append with the heredoc method below); fix anything wrong with Edit. Rewrite from scratch only if it is badly broken. If it does not exist, create it.
${foundationNote}
Corpus pointers for this topic: ${s.corpus}

SCOPE: cover all of the following, plus anything else an implementer of this topic would otherwise have to go and research:
${s.brief}

RESEARCH METHOD (mandatory):
- Load web tools with ToolSearch (query "select:WebSearch,WebFetch") and do real, extensive live research. Prefer primary sources: official docs, official GitHub repos (README, releases, and source files for exact API signatures), standards bodies, arXiv/DOI papers. Fast version/license checks: PyPI JSON (https://pypi.org/pypi/<pkg>/json), npm (https://registry.npmjs.org/<pkg>/latest), GitHub API (https://api.github.com/repos/<owner>/<repo> and .../releases/latest). You may use the Bash or PowerShell tools for curl / Invoke-RestMethod lookups.
- Verify every version number, release date, license (SPDX), CLI flag, config key, function/class signature, default value and URL you write. Your training knowledge may be stale; the live source wins.
- Tag claims inline where it matters: [VERIFIED-DOC] official doc, [VERIFIED-SRC] read in source code, [PAPER], [BENCHMARK] vendor/third-party numbers, [ESTIMATE] engineering estimate, [DESIGN] our recommendation, [UNVERIFIED] could not confirm (say exactly what to check). Never fabricate: an honest [UNVERIFIED] beats a confident guess.
- Respect every AGENTS.md constraint (observation vs inference; no appearance-based threat labels; identity is probabilistic; homography is 2D/2.5D; rules + baselines are the primary alert layer; VLMs explain only; label estimates; no DukeMTMC; check licenses; no premature infrastructure).

SECTION FORMAT (markdown):
1. First line: "# ${s.id} — ${s.title}".
2. A "Quick reference" block at the top: the decisions, pinned versions, and the handful of commands/snippets an implementer needs most.
3. Then, as applicable: Purpose & scope; Decisions (each with rationale, rejected alternatives, and when to revisit); Pinned dependencies table (component | version | license | install | source URL); Setup / installation (Linux and Windows/WSL2 where they differ); Concepts & algorithms (with the math, default parameters and why); Reference implementation (real, complete, typed Python 3 / TypeScript / SQL / YAML: importable modules, not pseudo-code; conform to the foundation contracts; mark anything you could not test); Configuration (every key, default, unit, meaning); Integration points (inputs/outputs by contract name, events emitted); Testing & acceptance criteria (concrete, measurable); Performance notes (numbers labeled by evidence type); Failure modes & pitfalls (symptom -> cause -> fix); Security & privacy notes; Deferred / research items and the triggers for adopting them; Sources (numbered: title, URL, accessed ${DATE}).
Be exhaustive and dense: tables, commands, code, parameters, no marketing filler. ${LENGTH_RULE} Cross-reference other sections by id (e.g. "see §07"). Section ids in this manual: 00a stack ML/video, 00b stack platform, 01 architecture & contracts, 02 ingestion, 03 recording & evidence, 04 camera health, 05 detection, 06 tracking, 07 geometry & calibration, 08 rules engine, 09 baselines & statistical anomaly, 10 correlation & alerting, 11 data layer, 12 messaging & frame transport, 13 backend API, 14 frontend console, 15 live video delivery, 16 operator workflow & HITL, 17 cameras/site/network, 18 datasets & simulation, 19 evaluation, 20 performance & scaling, 21 observability & deployment, 22 security, 23 privacy & compliance, 24 Re-ID & topology, 25 VLM & NL search, 26 crowd/forecast/learned anomaly/pose, 27 pitch/market/open source, 28 non-video data streams.
Writing large files: create the file with the Write tool for the first part, then append further parts with the Bash tool using a quoted heredoc, e.g. cat >> ${BASH_SEC}/${s.file} <<'EOF_ARGONIX' (newline, content, newline, EOF_ARGONIX). The quoted delimiter stops anything expanding. Do not modify any other file in the repository.

When finished, return the structured result: file path, title, a 120-200 word summary, key_decisions (short imperative statements), pinned_versions (every component + version you pinned, with license and source URL), conflicts_with_foundation, open_questions (things only the team/site can decide, or that you could not verify), approx_words.

${durableNote('write-' + s.id, 'write:' + s.id)}`
}

function verifyPrompt(s, res) {
  return `You are an adversarial fact-checker for the Argus implementation manual. Today is ${DATE}. Section file: ${SEC}\\${s.file} (${s.id} - ${s.title}). The team will build directly from this text without re-researching, so every wrong version, flag, API signature, license or URL will cost them hours. Assume errors exist and hunt for them.

Steps:
1. Read the whole section file. Also skim ${SEC}\\00a-stack-ml-video.md, ${SEC}\\00b-stack-platform.md and ${SEC}\\01-architecture-contracts.md (${s.foundation ? 'the other foundation sections, if present, for consistency' : 'the foundation this section must conform to'}).
2. Load WebSearch/WebFetch via ToolSearch ("select:WebSearch,WebFetch"). Pick at least 30 of the highest-risk checkable claims (pinned versions and release dates; licenses of code AND model weights / datasets; CLI flags; config keys; library function/class names and parameters; default values; legal dates; benchmark numbers; URLs) and verify each against a live primary source (PyPI/npm/GitHub API, official docs, source code, papers). Bash/PowerShell curl or Invoke-RestMethod lookups are fine.
3. Review every code block for API misuse, wrong imports, syntax errors, wrong units, blocking calls inside async code, resource leaks, and mismatches with the contracts in 01-architecture-contracts.md. Fix them.
4. Fix errors IN PLACE with the Edit tool (correct the text, or downgrade to [UNVERIFIED] with what to check). Do not rewrite correct material and do not shorten the section. Do not edit any other file.
5. Append a final subsection "## Verification log (${DATE})" listing each checked claim -> outcome (confirmed / corrected / unverified) with its source URL. If such a log already exists from an interrupted earlier run, keep it, re-check nothing already confirmed there, and continue it.
Writer's own summary, for context: ${res ? JSON.stringify({ summary: res.summary, open_questions: res.open_questions }) : `read it from ${STATE}\\write-${s.id}.json if that file exists`}
Return the structured result.

${durableNote('verify-' + s.id, 'verify:' + s.id)}`
}

const SECTION_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    title: { type: 'string' },
    summary: { type: 'string' },
    key_decisions: { type: 'array', items: { type: 'string' } },
    pinned_versions: { type: 'array', items: { type: 'object', properties: { component: { type: 'string' }, version: { type: 'string' }, license: { type: 'string' }, source: { type: 'string' } }, required: ['component', 'version'] } },
    conflicts_with_foundation: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
    approx_words: { type: 'number' },
  },
  required: ['file', 'title', 'summary', 'key_decisions', 'pinned_versions', 'conflicts_with_foundation', 'open_questions', 'approx_words'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    claims_checked: { type: 'number' },
    corrections: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, correction: { type: 'string' }, source: { type: 'string' } }, required: ['claim', 'correction'] } },
    code_issues_fixed: { type: 'array', items: { type: 'string' } },
    still_unverified: { type: 'array', items: { type: 'string' } },
  },
  required: ['file', 'claims_checked', 'corrections', 'code_issues_fixed', 'still_unverified'],
}
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, file: { type: 'string' }, title: { type: 'string' }, brief: { type: 'string' } }, required: ['id', 'file', 'title', 'brief'] } },
    thin: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, missing: { type: 'string' } }, required: ['file', 'missing'] } },
    conflicts: { type: 'array', items: { type: 'object', properties: { description: { type: 'string' }, files: { type: 'array', items: { type: 'string' } }, resolution: { type: 'string' } }, required: ['description', 'files', 'resolution'] } },
  },
  required: ['gaps', 'thin', 'conflicts'],
}
const CONSOL_SCHEMA = {
  type: 'object',
  properties: {
    updated_files: { type: 'array', items: { type: 'string' } },
    canonical: { type: 'array', items: { type: 'object', properties: { component: { type: 'string' }, version: { type: 'string' } }, required: ['component', 'version'] } },
    fixes: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, fix: { type: 'string' } }, required: ['file', 'fix'] } },
  },
  required: ['updated_files', 'canonical', 'fixes'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: { file: { type: 'string' }, applied: { type: 'array', items: { type: 'string' } }, skipped: { type: 'array', items: { type: 'string' } } },
  required: ['file', 'applied', 'skipped'],
}
const SYN_SCHEMA = { type: 'object', properties: { file: { type: 'string' }, summary: { type: 'string' } }, required: ['file', 'summary'] }

// ---------------- FOUNDATION ----------------
const FOUNDATION = [
  {
    id: '00a', file: '00a-stack-ml-video.md', foundation: true,
    title: 'Pinned stack I: GPU, ML, video & model licenses',
    corpus: 'Verification report sections 5-9, 23-24, 29; RESEARCH_EXTENSION Track 2, 3, 14; AGENTS.md known corrections (DeepStream 9.1, YOLO27, Frigate 0.18).',
    brief: `- The single source of truth for versions, compatibility and licenses of the GPU / ML / video half of the stack. For every component: pinned version (latest stable as of ${DATE}, verified live), release date, license (SPDX, for code AND for pretrained weights), install command (Linux; Windows/WSL2 where different), compatibility constraints, source URL.
- OS and drivers: Ubuntu 24.04 LTS; NVIDIA driver branch recommendation; CUDA toolkit major/minor; cuDNN; TensorRT 10.x (and TensorRT for RTX / TensorRT-RTX if relevant to consumer or Windows GPUs); NVIDIA Container Toolkit; WSL2 GPU support (what works: CUDA, TensorRT; does NVDEC/NVENC work in WSL2? verify); Windows native options.
- Python version choice (3.12 vs 3.13 vs 3.14): decide based on verified wheel availability for ALL key libs: torch, torchvision, tensorrt, onnxruntime-gpu, opencv-python-headless, PyAV, numpy 2.x, scipy, shapely, scikit-learn, PyNvVideoCodec, ultralytics, supervision.
- PyTorch + torchvision: version, CUDA variants and exact wheel index URLs; ONNX + onnxruntime / onnxruntime-gpu (which CUDA/cuDNN each build needs), ONNX Runtime execution providers (CUDA, TensorRT, OpenVINO, DirectML/WinML on Windows, CoreML on macOS: current status); OpenVINO for Intel CPU/iGPU fallback; NVIDIA ModelOpt for INT8/FP8.
- Video: FFmpeg version and which builds include NVDEC/cuvid/NVENC (BtbN static builds, jellyfin-ffmpeg, distro packages; license GPL vs LGPL builds and why it matters); PyAV version (bundled FFmpeg version in wheels; hardware-acceleration API support); PyNvVideoCodec version and license; GStreamer version + nvcodec plugin; OpenCV version (opencv-python-headless vs contrib; FFmpeg backend; license).
- Vision libraries: Ultralytics (current YOLO family status: YOLO26 / YOLO27 release state; AGPL-3.0 + enterprise license terms); RF-DETR (license per model size; verify); D-FINE; DEIM / DEIMv2; RT-DETRv2; YOLOX; supervision (MIT); roboflow trackers package; BoxMOT (AGPL-3.0); torchreid; fast-reid; SAHI; rtmlib / RTMPose.
- Scale-path components: DeepStream 9.1 (+ pyservicemaker, license/EULA), Triton Inference Server (release naming, license), NVIDIA DALI, Jetson JetPack versions (Orin Nano Super / NX / AGX / Thor), Savant framework status.
- A compatibility matrix table (driver <-> CUDA <-> cuDNN <-> TensorRT <-> torch <-> onnxruntime-gpu <-> Python) and a "known incompatibilities / gotchas" list.
- License posture: define two stack profiles, (A) permissive-only (Apache/MIT/BSD) and (B) AGPL-accepting, explain AGPL-3.0 obligations for network services and what triggers them, weights vs code licensing, dataset-derived weight restrictions (COCO, Objects365), and recommend the default for a hackathon project that wants to be open source.
- A requirements snippet (pyproject dependency block with pins and extras for cpu / cuda / openvino) the team can paste.`,
  },
  {
    id: '00b', file: '00b-stack-platform.md', foundation: true,
    title: 'Pinned stack II: data, backend, frontend, infrastructure & licenses',
    corpus: 'Verification report sections 21-24, 29, 38; RESEARCH_EXTENSION Tracks 2, 3, 12-14; EXTREME sections 32-37, 44, 64-65.',
    brief: `- The single source of truth for the platform half of the stack. For every component: pinned version (latest stable as of ${DATE}, verified live), release date, license (SPDX), install command / docker image tag, compatibility constraints, source URL.
- Data: PostgreSQL major version (17 vs 18: decide based on TimescaleDB + pgvector + pgvectorscale support); TimescaleDB version and the Apache-2.0 vs Timescale License (TSL) split: which features (compression / columnstore, continuous aggregates, retention policies, jobs) are TSL, what "community edition" permits, which docker images ship which (timescale/timescaledb vs timescale/timescaledb-ha; do they include pgvector, pgvectorscale, timescaledb-toolkit?); pgvector version; pgvectorscale; PostGIS (needed or not? license); Python drivers: psycopg 3 (+ psycopg_pool), asyncpg, SQLAlchemy 2.x, Alembic; pgBackRest.
- Messaging: Redis 8.x (license terms), Valkey (BSD) as alternative, redis-py version; NATS server + nats-py; orjson / msgspec / msgpack.
- Object storage after MinIO archival: compare AIStor Free (exact license terms), SeaweedFS, Garage, Ceph RGW, Zenko CloudServer, plain filesystem; S3 Object Lock / WORM support in each; boto3 / aioboto3 / obstore; recommend MVP and production choices.
- Backend: Python web stack (FastAPI, Starlette, Pydantic v2, pydantic-settings, uvicorn, httpx, websockets), auth libraries (PyJWT / joserfc / Authlib; argon2-cffi), OIDC providers (Keycloak, Authentik, Zitadel: licenses), structlog, orjson, tenacity, APScheduler or alternative, typer (CLI).
- Frontend: Node.js LTS, pnpm, TypeScript, React, Vite, router choice, TanStack Query, Zustand, Tailwind CSS v4, shadcn/ui + Radix, MapLibre GL JS, @vis.gl/react-maplibre (or react-map-gl), deck.gl, terra-draw (drawing polygons), hls.js, charting lib (ECharts / uPlot), Vitest, Playwright, openapi-typescript / hey-api; all versions & licenses.
- Video servers: go2rtc and MediaMTX (versions, licenses, feature highlights); coturn if needed.
- Observability & ops: Prometheus, prometheus-client (Python), Grafana (license AGPL-3.0 implications), Loki, Tempo or Jaeger, OpenTelemetry Python SDK + instrumentations, DCGM exporter, node_exporter, postgres_exporter; Docker Engine / Compose v2 versions; Caddy / nginx; step-ca; SOPS + age; OpenBao / Vault licenses; Trivy, Syft, Grype, OSV-Scanner, pip-audit, cosign.
- Dev tooling: uv (version, workspace support), ruff, mypy or pyright, pytest + pytest-asyncio + hypothesis, pre-commit, GitHub Actions runners.
- A whole-platform license matrix table and an "AGPL / source-available exposure" note (Grafana, Loki, Redis tri-license, Garage, TimescaleDB TSL) with what each means for an open-source hackathon project.
- A docker-compose image list with exact tags, and the package.json dependency block with pins the team can paste.`,
  },
  {
    id: '01', file: '01-architecture-contracts.md', foundation: true,
    title: 'Architecture, repository layout, process model & data contracts',
    corpus: 'EXTREME sections 9, 10, 13.4, 33, 42-43, 62-64; verification report sections 22, 27, 30-32, 39; AGENTS.md "Realistic MVP" and staged scaling.',
    brief: `- The implementation architecture every other section conforms to. Modular monolith first (per AGENTS.md), with seams that allow the staged split: 1-4 cameras monolith -> 5-20 split services with Redis Streams -> 20-50 sub-streams + GPU batching -> 50-100 pool/scheduler/object storage -> 100-200 edge decode + regional core.
- Process model on one machine, concretely: per-camera ingest workers (processes vs threads; GIL behavior of PyAV decode: verify whether it releases the GIL), inference process with cross-camera batching, analytics process (tracking, geometry, rules, correlation), API process, recorder (external: MediaMTX/go2rtc/ffmpeg), DB. Frame transport between processes at MVP (multiprocessing.shared_memory ring buffers vs queues; CUDA IPC later). Diagram (mermaid) of processes, queues and data flow.
- Repository layout: a monorepo with a uv workspace (verify uv workspace semantics), Python package "argonix" with subpackages (ingest, health, detect, track, geometry, rules, baselines, correlate, alerting, evidence, storage, bus, api, vlm, reid, sim, eval, common), a frontend app, deploy/ (compose, configs), configs/ (site profiles), docs/ (this manual, ADRs), tests/. Give the full directory tree.
- Core data contracts as complete Pydantic v2 models (actual importable code) with field names, types, units and coordinate conventions: Envelope (schema_version, message id, idempotency key, trace/span ids, timestamps: capture_ts derived from PTS, pts, ingest_ts, processed_ts), CameraConfig / StreamConfig, FrameRef (shared-memory slot reference), Detection (bbox xyxy in source-pixel coordinates, class, score, model_id + model_version), TrackUpdate, GroundPoint (world x,y in meters, ENU frame, with covariance / uncertainty), ZoneEvent, RuleEvent (explicitly labeled OBSERVATION), Inference records (explicitly labeled INFERENCE, with model/version and derivation), Incident, Alert (evidence refs, derivation, confidence + calibration status, model/version list, affected cameras/zones, observation vs inference items), Feedback / Disposition, HealthStatus, EvidenceManifest ref, AuditEvent.
- Conventions: time (UTC everywhere, monotonic clocks for intervals, PTS -> wall-clock mapping, clock-drift fields), IDs (UUIDv7 / ULID: verify stdlib support by Python version or pick a library), coordinates (pixel origin top-left, bbox xyxy, world ENU meters, map CRS), units (SI), naming, enums.
- Plugin interfaces as typing.Protocol / ABC with complete signatures: FrameSource, Detector, Tracker, Projector (homography), Rule, Correlator, Scorer, EvidenceStore, Bus, Sink. Model-agnostic detector interface requirements (per AGENTS.md).
- Config system: YAML site profiles validated with pydantic-settings (layering: defaults -> site file -> env -> secrets), hot reload rules, config versioning stamped into events.
- Cross-cutting: logging (structlog JSON), tracing ids propagation, error handling / retry policy, graceful shutdown, backpressure policy (drop analytics frames before alerts, never drop alerts), idempotency, schema evolution rules.
- Testing conventions (pytest, pytest-asyncio, hypothesis, golden files, scenario tests), code quality (ruff config, type checking strictness), ADR template and the initial ADR list.
- Provide the actual Python code for argonix/common/contracts.py and argonix/common/interfaces.py (complete and importable), plus pyproject.toml skeletons for the workspace.`,
  },
]

// ---------------- TOPIC TRACKS ----------------
const TRACKS = [
  {
    id: '02', file: '02-ingestion.md', title: 'Video ingestion: RTSP/ONVIF, decoding, sampling & timestamps',
    corpus: 'Verification sections 5-6; EXTREME section 11; RESEARCH_BRIEF section 7; spec F1.',
    brief: `- RTSP details that matter for implementation: RTP over TCP (interleaved) vs UDP, RTSPS/SRTP, digest auth, session timeouts and keepalives, camera connection limits, typical camera failure behaviors.
- ONVIF: Profile S/T/M operations needed (GetProfiles, GetStreamUri, GetSnapshotUri, video encoder configuration for sub-streams: resolution, fps, GOP, bitrate), WS-Discovery; Python libraries (onvif-zeep, onvif-zeep-async, others: maintenance status, license) with working example code to discover cameras and fetch main/sub stream URIs; ONVIF event subscription (PullPoint) for motion/tamper.
- Verified RTSP URL patterns and main/sub-stream selection for Hikvision, Dahua, Axis, Uniview, Hanwha, Reolink, TP-Link Tapo, Amcrest, CP Plus; how to URL-encode credentials.
- Decode paths compared, with a recommendation per platform (NVIDIA Linux, Windows/WSL2, Intel, macOS, CPU-only): PyAV (version, hardware acceleration API: how to request cuda / vaapi / qsv / d3d11va / videotoolbox, whether frames come back in system memory), OpenCV VideoCapture (OPENCV_FFMPEG_CAPTURE_OPTIONS, CAP_PROP_HW_ACCELERATION), ffmpeg subprocess rawvideo pipe, GStreamer (rtspsrc, nvh264dec, appsink via PyGObject), PyNvVideoCodec (demux + NVDEC + DLPack zero-copy), DeepStream later.
- FFmpeg/PyAV option reference with exact names and units: rtsp_transport, timeout (microseconds; stimeout was removed, verify which FFmpeg version), rw_timeout, fflags nobuffer, flags low_delay, max_delay, reorder_queue_size, buffer_size, analyzeduration, probesize, threads, skip_frame (nokey / nonref) for cheap decoding.
- Sampling to 5 fps from a 15-25 fps sub-stream: decode-all vs skip non-reference frames; PTS-based time sampling; keep real dt for the tracker.
- Timestamps: PTS -> wall clock mapping, RTCP sender reports / NTP (does FFmpeg expose start_time_realtime for RTSP? verify), arrival time, drift estimation, camera NTP configuration.
- Robustness: reconnect with exponential backoff + jitter, watchdog for hung reads (PyAV blocking reads: timeouts / interrupt), per-camera process isolation, bounded ring buffers with drop-oldest, memory bounds, graceful shutdown.
- Restreaming proxy in front of cameras (go2rtc or MediaMTX) so recorder, analytics and browser share one camera connection; config examples for both; which one to choose.
- Complete reference implementation: CameraWorker (PyAV, reconnect, sampling, health hooks, output FrameRef to shared memory per §01 contracts) + supervisor managing N cameras.
- Test harness: MediaMTX + "ffmpeg -re -stream_loop -1" publishing test files as RTSP; fault injection (stop the stream, network impairment with tc netem on Linux / clumsy on Windows).
- Performance: CPU decode cost for 1080p vs 720p/D1 sub-streams, NVDEC figures from the corpus, how to measure.
- Pitfalls: H.265 streams, B-frames, corrupted packets / smearing on UDP, timestamp jumps and wraparound, camera reboots, special characters in credentials, per-camera connection limits.`,
  },
  {
    id: '03', file: '03-recording-evidence.md', title: 'Recording, evidence clips, evidence packages & object storage',
    corpus: 'Verification sections 7, 21, 26, 28; EXTREME sections 56-57, 59; RESEARCH_EXTENSION S1/T2.6 (MinIO archived); spec F32, F36.',
    brief: `- Continuous main-stream recording without re-encoding: ffmpeg segment muxer (exact command: -c copy -f segment -segment_time, -reset_timestamps, -strftime, segment_format options), versus MediaMTX built-in recording (record, recordPath, recordFormat fmp4, recordSegmentDuration, recordDeleteAfter, playback server: verify keys) versus go2rtc; recommend one.
- Container choice (fragmented MP4 vs MPEG-TS vs MKV) for crash resilience, browser playback, H.265; faststart / fragmented flags.
- Segment index in the database (camera, start, end, path, size, sha256) and how to build it (watch folder, MediaMTX hooks such as runOnRecordSegmentComplete: verify).
- Evidence clip extraction with pre/post-roll: segment selection, concat demuxer, -ss/-to with stream copy (keyframe snapping) vs re-encode for frame-exact cuts (NVENC / libx264 / openh264 licensing), browser-friendly H.264 MP4 output; snapshots (clean + annotated with boxes/track ids); trajectory GeoJSON.
- Evidence package: manifest JSON schema (event ids, derivation, confidence, model/version, cameras/zones, observation vs inference items, time ranges, file list with SHA-256), manifest hash, hash chaining across packages, optional RFC 3161 trusted timestamps (tools, free TSA options), chain-of-custody log, export bundle (ZIP) for authorities; digital-evidence handling guidance (SWGDE, ACPO/NPCC principles, ISO/IEC 27037) summarized with sources.
- Object storage after MinIO archival: AIStor Free (exact license terms), SeaweedFS, Garage, Ceph RGW, plain filesystem: S3 API coverage, Object Lock (governance/compliance modes, retention, legal hold) support, operational complexity, licenses; boto3 / aioboto3 usage for put/get/presigned URLs/object lock; recommend MVP (filesystem) and production choices.
- Retention: per-class retention (raw 7-30 days, evidence longer), deletion jobs, legal-hold override, storage sizing formulas (corpus numbers, labeled estimates), disk/filesystem choices, write throughput.
- Complete reference code: RecorderSupervisor, SegmentIndexer, ClipExtractor, SnapshotRenderer, EvidencePackager, RetentionJob; tests and acceptance criteria (e.g. clip boundaries within one GOP, hashes verifiable, playback in Chrome/Firefox/Edge).`,
  },
  {
    id: '04', file: '04-camera-health.md', title: 'Camera health, tamper & degradation detection',
    corpus: 'Spec F2; verification section 5; EXTREME sections 11.3, 43.',
    brief: `- Stream-level metrics: connection state, actual vs expected fps, bitrate, frame age, decode errors, reconnect counts, timestamp drift, resolution changes.
- Image-level checks with exact algorithms, default thresholds and compute cost: black / overexposed (luma mean and percentiles), frozen frame (frame differencing on downscaled grayscale, SSIM, perceptual hash; replay/loop detection), blur/defocus (variance of Laplacian, Tenengrad), noise, occlusion/obstruction (spray paint, covering), camera moved / scene changed (ORB features + homography deviation against a reference frame), IR night-mode switching, glare, color cast.
- Survey the camera-tamper-detection literature (classic e.g. Ribnick et al. 2006, Saglam & Temizel 2009, plus recent deep-learning approaches) and give recommended methods with sources.
- Per-camera adaptive thresholds (day/night baselines), scheduling (1 Hz on low-res frames or on keyframes only), hysteresis, health state machine OK / DEGRADED / DOWN, coverage impact mapping (zone -> cameras) so the operator sees blind zones; health events are OBSERVATIONS.
- Camera-reported ONVIF events (tamper, video loss) integration.
- Complete reference code (numpy/OpenCV) with benchmarks of cost per check; validation via synthetic injection (black, blur, freeze, shift, occlusion overlays) and any public tamper datasets.`,
  },
  {
    id: '05', file: '05-detection.md', title: 'Object detection: models, licenses, export, runtimes & tuning',
    corpus: 'Verification sections 8, 24, 29; RESEARCH_BRIEF section 1; spec F3; EXTREME section 13, 39.',
    brief: `- Detector interface and model registry conventions (conform to §01); model_id/model_version stamping (engine hash).
- Candidate models current as of ${DATE} with official COCO AP, latency (T4/TensorRT or official numbers), parameter counts, and licenses of code AND weights: Ultralytics YOLO26 / YOLO11 (AGPL-3.0; check YOLO27 status), RF-DETR (license per size; verify), D-FINE, DEIM and DEIMv2, RT-DETRv2 (and any v3/v4), YOLOX, PP-YOLOE, LW-DETR, YOLOv9/v10 licenses. Recommend a default for license profile A (permissive) and profile B (AGPL-accepting).
- Relevant COCO classes and ids (person, bicycle, car, motorcycle, bus, truck, backpack, handbag, suitcase, etc.); airport/site classes missing from COCO (trolley/cart, service vehicle, stroller, wheelchair) and a fine-tuning plan: datasets and their licenses (COCO, Objects365, OpenImages, CrowdHuman), annotation, training commands per framework, expected data volume.
- Export per model family: ONNX export commands (opset, dynamic batch), TensorRT engine build with trtexec (fp16, int8 with calibration, min/opt/max shapes, memPoolSize, timing cache; engine portability constraints), ModelOpt quantization, ONNX Runtime (CUDA EP, TensorRT EP options, OpenVINO EP, DirectML/WinML on Windows, CoreML on macOS), OpenVINO conversion for Intel.
- Complete pre/post-processing code: letterbox, normalization, batching several cameras, YOLO-style decode + NMS (torchvision batched_nms or cv2.dnn.NMSBoxes) vs DETR-style top-k without NMS; mapping boxes back to source pixels; class-specific confidence thresholds.
- Accuracy knobs: input size, per-class thresholds, SAHI tiling for small/distant objects, ROI masks, class-aware NMS; night/IR and wide-angle domain shift; crowded scenes.
- Throughput figures (official or [BENCHMARK]) and the measurement procedure; CPU-only / OpenVINO expected fps.
- AGPL-3.0 implications for Ultralytics in a networked service; how to keep the detector swappable.`,
  },
  {
    id: '06', file: '06-tracking.md', title: 'Per-camera multi-object tracking',
    corpus: 'Verification section 9; RESEARCH_EXTENSION T1.3-T1.5; EXTREME section 14; spec F4.',
    brief: `- Algorithms: SORT, ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, Hybrid-SORT, and newer 2024-2026 methods (MambaMOT etc.); when each fits a 5-fps fixed-camera sub-stream.
- Implementations and licenses with exact APIs and parameter names: supervision ByteTrack (MIT; verify class and parameter names in the current version), roboflow "trackers" package (verify contents and license), BoxMOT (AGPL-3.0; list trackers and API), original ByteTrack repo (MIT), OC-SORT (MIT), norfair (BSD-3). Recommend a permissive default.
- Internals needed to adapt them: ByteTrack two-stage association and thresholds; Kalman state (xyah vs xywh) and noise; delta-t-aware process model for variable fps (F and Q as functions of dt), track buffers expressed in seconds; camera motion compensation (only for moving cameras); appearance features optional.
- The 5-fps problem: IoU association degrades at low fps; mitigations (motion model, lower IoU threshold, higher fps in crowded cameras, GIoU/DIoU costs) with evidence.
- Complete reference implementation: per-camera tracker wrapper producing TrackUpdate contracts, variable-dt handling, lifecycle states, foot-point smoothing, stationary-object handling for abandoned-object logic (detection-level persistence), class flicker handling.
- Tuning procedure with TrackEval on own annotated clips (data format, commands, HOTA/IDF1/MOTA/IDSW), default parameters for our setting, acceptance thresholds (IDF1 >= 60 target from corpus).`,
  },
  {
    id: '07', file: '07-geometry-calibration.md', title: 'Geometry: calibration, homography, uncertainty, zones & map coordinates',
    corpus: 'Verification section 13; RESEARCH_BRIEF section 3; EXTREME sections 16-17, 21.2, 30; spec F8-F10.',
    brief: `- Lens distortion: when it matters; cv2.calibrateCamera with checkerboard / ChArUco (current cv2.aruco API since OpenCV 4.7), fisheye module; cv2.undistortPoints before projection.
- Homography: collecting correspondences (surveyed points, CAD/floor-plan, tape measure; a web point-pairing tool), cv2.findHomography methods (RANSAC, USAC_MAGSAC, LMEDS) and thresholds; point count/spread guidance (8-12 well spread); reprojection error in pixels AND world meters; leave-one-out validation; the OpenCV RANSAC issue noted in the corpus.
- Uncertainty: first-order propagation through the homography Jacobian and Monte Carlo; per-camera error maps vs distance; conservative zone buffers derived from the error (AGENTS.md: report calibration error, conservative boundaries).
- Foot point: bottom-center, truncated boxes at frame edges, occlusion, vehicles' ground contact; smoothing.
- World frame and map: local ENU meters, georeferencing a floor plan (image corners), MapLibre options (image source with coordinates, or a local metric frame mapped to lon/lat near 0,0; distortion analysis), GeoJSON, pyproj when real geo coordinates exist, multiple floors.
- Zone geometry: shapely 2.x (prepared geometries, STRtree), point-in-polygon, inset buffers, directed line crossing (cross-product sign), hysteresis, pixel masks vs world polygons, overlapping-FOV duplicate suppression by world-distance gating (MVP-level).
- Speed/heading estimation with smoothing (Savitzky-Golay, Kalman in world coordinates) and reported uncertainty.
- Calibration drift detection against a reference image (ORB/SIFT; SIFT is in main OpenCV since the patent expired), PTZ cameras excluded; mention DeepStream AutoMagicCalib as scale option.
- Complete reference code: calibration module (fit, validate, serialize), Projector with uncertainty, ZoneEngine geometry, calibration-tool data model; tests with synthetic cameras of known H.`,
  },
  {
    id: '08', file: '08-rules-engine.md', title: 'Behavior rules catalog & rule engine',
    corpus: 'Verification sections 15, 30-31; EXTREME section 18, 25.3; spec F14-F20.',
    brief: `- Rule catalog with precise definitions, parameters, default values, state machines and emitted OBSERVATION events (with evidence fields): restricted-zone entry (min dwell, penetration depth), directional line crossing / counting, loitering / dwell (zone-specific, baseline-aware), wrong-way / counter-flow (heading vs zone flow vector, temporal + distance confirmation), occupancy threshold, crowd density in persons/m2 (Fruin Level-of-Service thresholds for walkways and queues: verify values and source), queue length and wait-time estimate (Little's law, service rate), unattended/abandoned object (object-person association + state machine; reference datasets PETS 2006, AVSS 2007, ABODA), running / sudden speed change, stationary vehicle / no-parking, vehicle in pedestrian zone, tailgating (door sensor + count), camera-coverage loss. Defer fall/fight detection (pose; see §26).
- Event naming that is observable, never judgmental (no "suspicious person").
- Engine architecture: declarative YAML rules validated by Pydantic; stateful evaluators keyed by (rule, zone, track); event-time processing with watermarks; debouncing, hysteresis, cooldowns; schedules (active hours); zone sensitivity; rule versioning stamped into events; hot reload.
- Survey rule/expression libraries (CEL via cel-python / celpy, json-logic, durable_rules, business-rules, experta): maintenance, license; recommend typed Python evaluators + optional safe expression language for conditions.
- Complete reference code for the engine and every MVP rule, plus a synthetic-trajectory test harness; per-rule evaluation protocol (event precision/recall vs annotated intervals).`,
  },
  {
    id: '09', file: '09-baselines-anomaly.md', title: 'Seasonal baselines & statistical anomaly detection (primary layer)',
    corpus: 'Verification sections 16, 20; RESEARCH_BRIEF section 5; EXTREME sections 23, 28.3; spec F21, F25.',
    brief: `- What to baseline: zone occupancy, flow counts by direction, dwell-time distributions, queue length, event rates per rule, camera health metrics.
- Seasonal baselines: hour-of-week buckets, robust median/MAD and quantiles, EWMA and exponential decay for drift, minimum sample counts, cold-start (global priors, partial pooling across similar zones, schedule-driven), holiday / special-event calendars, optional flight-schedule covariates.
- Deviation statistics with formulas and defaults: modified z-score (0.6745 constant), quantile bands, CUSUM and Page-Hinkley for level shifts, STL decomposition (statsmodels), controlling alert volume (multiple testing, alert budget).
- Storage and computation: TimescaleDB continuous aggregates with time_bucket and refresh policies; percentile approximations (timescaledb_toolkit uddsketch/percentile_agg: availability and license) vs Python-side computation; incremental updates.
- Isolation Forest (scikit-learn) and PyOD as CORROBORATION only: features, training cadence, contamination, score-to-percentile calibration; never a standalone alarm (AGENTS.md).
- Streaming libraries: river (HalfSpaceTrees etc.), statsforecast for baseline bands: versions and licenses.
- Complete reference code and SQL; backtesting protocol on recorded weeks; false-alarm budgeting.`,
  },
  {
    id: '10', file: '10-correlation-alerting.md', title: 'Event correlation, incidents, prioritization & calibration',
    corpus: 'Verification sections 17-18, 27; RESEARCH_BRIEF section 5-6; EXTREME sections 25-27, 53-54; spec F22-F24.',
    brief: `- CEP design in Python: event-time windows (tumbling, sliding, session), correlation keys (zone, camera neighborhood via topology graph, local track id), watermarks and late events, correlation rules (spatial proximity, temporal order, shared entities, type relationships), cascade suppression, dedup and idempotency keys.
- Incident lifecycle state machine (open -> active -> cooling -> closed; merge/split; reopen), operator snooze, cooldowns, rate limits.
- Survey implementation options with current status and license: hand-rolled asyncio (likely MVP), Bytewax, faust-streaming, Quix Streams, PyFlink, Esper; recommend.
- Priority scoring: transparent factor model (rule severity, zone sensitivity, confidence, corroboration count, duration, rate of change, camera-health context) -> raw score -> calibrated probability via isotonic regression or Platt scaling (scikit-learn IsotonicRegression / CalibratedClassifierCV; minimum sample sizes), cold start with expert weights and conservative tiers; ECE, Brier score, reliability diagrams; tier thresholds from precision targets; alert budget per operator-hour.
- Explanation payload schema: factor contributions, evidence links, rule versions, model versions, observation vs inference labels, calibration status.
- Notification channels (UI, webhook, email/SMS via Apprise: license) and escalation policies.
- Complete reference code: CorrelationEngine, IncidentManager, PriorityScorer, Calibrator, with tests; metrics (alerts/hour, precision, time-to-acknowledge, events-per-incident).`,
  },
  {
    id: '11', file: '11-data-layer.md', title: 'Data layer: PostgreSQL, TimescaleDB, pgvector, schema & migrations',
    corpus: 'Verification section 21; EXTREME sections 10, 32; spec section 46-47.',
    brief: `- Versions and docker image choice (timescale/timescaledb-ha vs timescale/timescaledb; included extensions) and license implications (TSL features); timescaledb-tune and ingest-oriented PostgreSQL settings.
- Complete DDL conforming to §01 contracts: sites, cameras, streams, calibrations (versioned), zones (versioned; geometry as JSONB/GeoJSON vs PostGIS: decide), rules (versioned), model registry, tracks, track_points (hypertable), zone_events, events (hypertable), incidents, incident_events, alerts, alert_feedback, evidence_items, recording_segments, camera_health (hypertable), baselines, audit_log (append-only, hash chain), users/roles/zone scopes, embeddings (pgvector, later), settings.
- Hypertable chunk intervals; compression / columnstore settings with the CURRENT TimescaleDB API (verify: compress_segmentby/orderby vs Hypercore columnstore syntax and add_compression_policy vs add_columnstore_policy); retention policies; continuous aggregates for baselines and dashboards.
- Indexing (btree, BRIN, GIN on JSONB, partial), JSONB usage rules.
- Ingest: batched COPY with psycopg 3 (cursor.copy) or asyncpg copy_records_to_table; batching latency trade-offs; idempotent upserts (ON CONFLICT); expected write rates (corpus estimates).
- SQLAlchemy 2.0 async + Alembic migrations including TimescaleDB-specific operations; naming conventions.
- Row-level security for zone-scoped operators; roles and least privilege.
- pgvector HNSW parameters (m, ef_construction, ef_search), halfvec, iterative index scans; pgvectorscale when.
- Backups (pgBackRest or pg_dump + WAL archiving), restore drills, upgrades, monitoring (pg_stat_statements, TimescaleDB job stats).
- Complete SQL files + Python repository layer code.`,
  },
  {
    id: '12', file: '12-messaging-frames.md', title: 'Messaging, event bus & frame transport',
    corpus: 'Verification section 22; EXTREME sections 9.3, 33-34, 43; AGENTS.md staged scaling.',
    brief: `- MVP: in-process asyncio queues + multiprocessing; frame transport via multiprocessing.shared_memory ring buffers with numpy views (slot protocol, sequence numbers, readers/writers, cleanup on crash, Windows vs Linux differences), CUDA IPC (torch.multiprocessing) later; never push raw frames through Redis/NATS.
- Stage 2 Redis Streams: XADD with MAXLEN ~, XREADGROUP, XACK, XPENDING, XAUTOCLAIM, XTRIM, consumer groups, at-least-once, persistence (AOF), memory sizing; redis-py asyncio code; key design; Redis 8 license vs Valkey.
- NATS JetStream: streams, durable pull consumers, ack policies, dedup window (Nats-Msg-Id), subject hierarchy, KV; nats-py code; when to switch.
- Serialization: JSON (orjson) vs msgpack vs msgspec vs protobuf; schema evolution; size budgets.
- Idempotency, dead-letter queues, retries, poison messages, per-key ordering, backpressure and load shedding (drop analytics frames first, never drop alerts).
- A Bus interface with InProcess, RedisStreams and NATS implementations (complete code), plus a throughput/latency benchmark script.`,
  },
  {
    id: '13', file: '13-backend-api.md', title: 'Backend API, realtime, authentication, authorization & audit',
    corpus: 'EXTREME sections 35-36, 50; verification sections 20, 28; RESEARCH_BRIEF section 7, 12.',
    brief: `- FastAPI application structure (lifespan, dependency injection, settings), REST resources (sites, cameras, zones, rules, incidents, alerts, feedback, evidence, search, health, admin), pagination/filtering, RFC 9457 problem details, OpenAPI.
- WebSocket hub: connection manager, per-client subscriptions/filters, fan-out across uvicorn workers (Redis pub/sub), heartbeats, bounded per-client queues and drop policy, resume via last-event-id, auth on WebSocket without putting tokens in query strings (subprotocol or first message); SSE alternative (sse-starlette).
- Authentication: local accounts (argon2) + JWT vs OIDC with Keycloak / Authentik / Zitadel; MFA; token lifetimes; CSRF for cookies.
- Authorization: roles (viewer, operator, supervisor, admin, auditor), zone-scoped permissions enforced in the API AND via PostgreSQL row-level security, evidence-access logging, break-glass access.
- Audit logging of every operator action and every evidence/video access; tamper-evident hash chain.
- Media serving: presigned URLs, HTTP range requests for MP4, authenticated proxying to go2rtc/MediaMTX (MediaMTX external HTTP auth / JWT: verify keys).
- Rate limiting, CORS, security headers, input validation; TypeScript client generation from OpenAPI.
- Testing (httpx AsyncClient, WebSocket tests, contract tests); deployment behind Caddy with TLS.
- Complete reference code skeleton (app factory, routers, WS hub, auth, RBAC dependency, audit middleware).`,
  },
  {
    id: '14', file: '14-frontend-console.md', title: 'Operator console frontend',
    corpus: 'EXTREME sections 30, 37; spec section 53; RESEARCH_BRIEF sections 6, 11; verification section 35.',
    brief: `- Stack with versions and licenses (React, TypeScript, Vite, router, TanStack Query, Zustand, Tailwind v4, shadcn/ui + Radix, MapLibre GL JS + React wrapper, deck.gl for dense layers, terra-draw for drawing, hls.js, charts).
- Screens and components: operations overview (map + prioritized alert queue + camera tiles + timeline), alert/evidence panel (clip player with overlays, snapshot, trajectory on map, rule and factor explanation, OBSERVATION vs INFERENCE badges, confidence display, model versions), incident timeline and synchronized multi-camera replay, camera wall, zone/rule editors (polygons on camera frame and on the map), calibration tool (point pairing, error display), camera health dashboard, search, admin/users, audit views, feedback / disposition capture.
- Realtime data flow: WebSocket client with reconnect/resume, normalized store, throttled rendering (requestAnimationFrame batching), MapLibre GeoJSON source setData limits and feature-state, memory limits.
- Map specifics: indoor floor plan as image source or vector (GeoJSON from CAD via QGIS/ogr2ogr), local coordinates, camera FOV cones, zones, track trails, heatmaps.
- Video overlays: drawing boxes on video via canvas synced with requestVideoFrameCallback; metadata/video latency alignment for WebRTC and HLS.
- Accessibility (WCAG 2.2 AA: contrast in dark theme, severity not by color alone, keyboard operation, focus management, reduced motion) and control-room UX principles (alarm-management guidance adapted: ISA-18.2 / EEMUA 191).
- Testing (Vitest, Testing Library, Playwright), performance budgets.
- Complete reference code: project structure, MapView, AlertQueue, EvidencePanel, VideoWithOverlay, useLiveSocket hook, API client wiring.`,
  },
  {
    id: '15', file: '15-video-delivery.md', title: 'Live video delivery to the browser & recorded playback',
    corpus: 'EXTREME section 36.2; RESEARCH_BRIEF section 7; spec F36.',
    brief: `- go2rtc vs MediaMTX in depth (versions, licenses, protocols: RTSP, WebRTC/WHEP, HLS/LL-HLS, MSE, recording, playback APIs, auth hooks, API for dynamic paths, resource use); choose one for the MVP and justify.
- Topology: cameras -> restreamer -> (recorder, analytics, browser); main stream for focus view, sub-stream for tiles.
- Codec support in browsers as of ${DATE} (H.264 everywhere; H.265 in WebRTC/MSE per browser: verify), when transcoding is needed and its cost (NVENC/QSV).
- Browser clients: WHEP client libraries or the restreamer's bundled player, hls.js for recordings/fallback; latency per protocol; NAT/TURN (coturn) only for remote access.
- Authentication integration with the backend (tokens, signed URLs), viewer scaling limits, CPU/GPU cost.
- Recorded playback: MediaMTX playback server / segment listing API (verify), seeking, synchronized multi-camera replay approach, clip download.
- Complete configs (mediamtx.yml and/or go2rtc.yaml) and a React player component; test procedure (latency measurement with an on-screen clock).`,
  },
  {
    id: '16', file: '16-operator-hitl.md', title: 'Operator workflow, human factors, human-in-the-loop & feedback loop',
    corpus: 'RESEARCH_BRIEF section 6; verification section 18, 28; EXTREME sections 4.4, 52-54; spec F38-F39.',
    brief: `- Evidence base with primary sources: vigilance decrement and multi-monitor CCTV operator performance (verify the origin and validity of the commonly quoted "attention drops after 20 minutes" claim), alarm-fatigue literature (clinical, SOC, industrial), alarm-management standards (ISA-18.2 / IEC 62682, EEMUA 191 benchmark alarm rates), control-room design ISO 11064, UK Home Office CCTV Operational Requirements Manual (CAST; Rotakin), SINTEF CCTV human-factors requirements, arXiv 2307.07023 from the corpus, automation bias and its mitigations.
- Alert workflow design: lifecycle states, disposition taxonomy (true event / false alarm / needs review / wrong location / wrong object / duplicate), evidence required before action buttons, escalation, shift handover notes, time-to-acknowledge SLAs, operator workload budget (alerts per operator-hour, screens per operator).
- Explanation presentation: OBSERVATION vs INFERENCE badges, verbal + numeric confidence, uncertainty display research, model/version display.
- Feedback data model and quality; using feedback for calibration, threshold tuning, rule suppression and active learning; guarding against feedback loops and bias.
- Operator training material outline and SOP templates; demo user-study protocol (tasks, SUS, NASA-TLX, ethics).`,
  },
  {
    id: '17', file: '17-cameras-site-network.md', title: 'Cameras, site survey, placement, network & time sync',
    corpus: 'RESEARCH_BRIEF section 7; EXTREME sections 11, 16.4; verification section 13.',
    brief: `- Pixel density requirements for analytics tasks: IEC/EN 62676-4 DORI values (Detect/Observe/Recognize/Identify px/m: verify current edition and numbers), UK CAST Operational Requirements percentages; formulas to compute horizontal FOV, pixel density at distance, lens choice.
- Mounting height and tilt for homography accuracy vs occlusion, overlap zones for later cross-camera handoff, avoiding backlight; WDR, low light, IR effects on detection; shutter speed and motion blur.
- Encoder settings for analytics: H.264 sub-stream, GOP = 1-2 x fps, CBR vs VBR, "smart codec" features (H.264+/H.265+, Zipstream) and their effect on analytics; per-vendor sub-stream configuration.
- Network: PoE budgets (802.3af/at/bt), VLAN segmentation, bandwidth planning (corpus numbers), multicast, switch sizing; time sync with NTP/chrony and PTP (IEEE 1588): why it matters for correlation.
- Camera hardening: firmware, unique credentials, disable UPnP/P2P cloud, HTTPS, ONVIF user levels; ONVIF conformant products database; NDAA / procurement restrictions on some vendors (verify).
- Hackathon/demo hardware options with prices where verifiable: low-cost RTSP/ONVIF cameras (e.g. TP-Link Tapo C200/C210 RTSP support, Reolink, CP Plus, Hikvision), phone-as-IP-camera apps, USB webcams published via ffmpeg/MediaMTX, laptops with NVIDIA GPUs, mini-PCs, Jetson Orin Nano Super dev kit.
- Site survey checklist and a calibration-friendly placement checklist.`,
  },
  {
    id: '18', file: '18-datasets-simulation.md', title: 'Datasets, annotation, replay harness & simulation',
    corpus: 'Verification sections 25-26; RESEARCH_BRIEF sections 2, 4; RESEARCH_EXTENSION Track 1, 14; EXTREME sections 47, 55.',
    brief: `- Datasets table (task, size, license and whether commercial / demo use is allowed, download URL, format, relevance): COCO, Objects365, CrowdHuman, MOT17/MOT20, DanceTrack, WILDTRACK, MultiviewX, MMPTRACK, MEVA, VIRAT, PETS 2006/2009, AVSS 2007, ABODA, CAVIAR, UCF-Crime, ShanghaiTech, XD-Violence, UCSD Ped, CityFlow and AI City tracks, AI City synthetic multi-camera people data, MTA, JTA, Market-1501, MSMT17 (research-only), crowd sets (JHU-Crowd++, NWPU, UCF-QNRF, ShanghaiTech A/B). No DukeMTMC.
- Royalty-free CCTV-like footage for demos (Pexels/Pixabay license terms) and recording your own footage with consent (consent-form items).
- Annotation tools: CVAT, Label Studio, X-AnyLabeling (licenses, setup); annotation guidelines for boxes, track ids, and temporal event intervals; formats (COCO JSON, MOT txt, CVAT XML) and converters.
- Replay harness: MediaMTX + ffmpeg looping several files as synchronized RTSP cameras; converting image-sequence datasets (e.g. WILDTRACK frames) to video; clock alignment; fault injection.
- Metadata-level simulator for scale tests (20-200 virtual cameras): agent-based pedestrian simulation options (JuPedSim, Vadere, PedSim, Menge, social-force implementations: licenses, Python APIs), generating trajectories -> synthetic TrackUpdates through the real pipeline; scenario injection (crowd surge, counter-flow, abandoned object, camera failure); ground truth.
- Synthetic video options (NVIDIA Isaac Sim / Omniverse Replicator, CARLA, Unreal): what they give, cost, license: research tier.
- Scenario test-suite format (YAML) with expected events and tolerances; complete simulator and harness code.`,
  },
  {
    id: '19', file: '19-evaluation.md', title: 'Evaluation & benchmarking methodology',
    corpus: 'Verification sections 26, 33, Appendix E; spec section 58; EXTREME sections 70-72.',
    brief: `- Metric definitions and tools with versions and exact usage: detection (pycocotools / faster-coco-eval, per-class and per-zone), tracking (TrackEval: HOTA, IDF1, MOTA, IDSW: data layout and commands; py-motmetrics), event detection (interval matching with temporal IoU, precision/recall/F1, false alarms per camera-hour, detection delay), calibration (ECE, Brier, reliability diagrams), geometry (reprojection error, world error vs surveyed points), counting (MAE/RMSE per zone), forecasting (MAE, MASE, interval coverage), system (end-to-end and per-stage latency percentiles, fps, drops, GPU utilization), operator (time-to-acknowledge, SUS, NASA-TLX).
- Turn experiments E1-E7 from the verification report into step-by-step procedures and scripts (tools: py-spy, Nsight Systems, nvidia-smi dmon, DCGM, k6/Locust for API/WebSocket load).
- Data splits (temporal, camera, identity disjoint); statistics (bootstrap confidence intervals, paired tests, Diebold-Mariano); reporting templates; a "claims gate" table stating what may be claimed after which experiment.
- Complete reference code: event matcher, ECE/reliability, latency collector, report generator.`,
  },
  {
    id: '20', file: '20-performance-scaling.md', title: 'Performance engineering, GPU pipeline, scheduling & scale path',
    corpus: 'Verification sections 6-7, 32, Appendix D; RESEARCH_BRIEF sections 8-9; RESEARCH_EXTENSION Track 11; EXTREME sections 42, 58-61; spec F40.',
    brief: `- Where time goes per frame (decode, NV12->RGB, resize/letterbox, host-to-device copy, inference, NMS, tracking, geometry, DB writes) and how to profile each.
- MVP optimizations: CPU decode of sub-streams vs NVDEC; GPU preprocessing with torch (pinned memory, non_blocking copies, CUDA streams); cross-camera batching with a max-wait; TensorRT dynamic-batch engines; async inference loop design.
- Adaptive FPS scheduler: states, global GPU budget (token bucket), priority queue, starvation floors, hysteresis, dt-correct tracking; complete code.
- Scale-path technology: PyNvVideoCodec zero-copy decode, NVIDIA DALI video, Triton Inference Server (dynamic batching, model analyzer), DeepStream 9.1 (nvstreammux, nvinfer, nvtracker NvDCF, pyservicemaker, licensing), Savant (status), Jetson (JetPack, decode limits, power modes).
- Capacity planning worksheet from the corpus model (all labeled [ESTIMATE]) and the measurement procedure to replace estimates; GPU memory planning (detector + VLM later).
- Multi-node: edge decode + regional core, metadata-only uplink.
- Windows/WSL2 performance specifics and CPU-only fallback performance (OpenVINO, ONNX Runtime CPU) with expected fps labeled by evidence type.`,
  },
  {
    id: '21', file: '21-observability-deployment.md', title: 'Observability, deployment, environments & operations',
    corpus: 'EXTREME sections 44, 65; verification section 22; RESEARCH_EXTENSION Track 13.',
    brief: `- Metrics: prometheus-client usage and naming, the full metric list per module (fps, decode errors, queue depths, inference latency histograms, event and alert rates, DB write latency, WebSocket clients), GPU metrics (DCGM exporter vs alternatives; consumer GPU support), node/postgres/redis exporters, Grafana dashboard provisioning, Alertmanager rules for pipeline health.
- Tracing: OpenTelemetry Python SDK + instrumentations (FastAPI, asyncpg/psycopg, redis), OTLP to Tempo or Jaeger, sampling strategy for per-frame spans; structured logging (structlog) to Loki or files; PII in logs.
- Deployment: complete docker-compose.yml for the MVP (all services, GPU reservation syntax, healthchecks, volumes, networks, secrets, restart policies), NVIDIA Container Toolkit install + CDI, base images (nvidia/cuda, NGC TensorRT containers and their licenses), multi-stage builds with uv, image sizes.
- Windows 11 dev: WSL2 + Docker Desktop GPU support and limitations, reaching LAN RTSP cameras from WSL2 (mirrored networking mode), file-system performance tips; Ubuntu 24.04 host setup (driver install, toolkit), systemd alternative.
- Reverse proxy (Caddy) with TLS, secrets management, configuration management, backups/restore, upgrades/rollback.
- CI/CD with GitHub Actions: lint, type-check, tests, image builds, SBOM, GPU tests on a self-hosted runner.
- Runbooks for common incidents (camera down, GPU OOM, disk full, DB bloat, clock drift).`,
  },
  {
    id: '22', file: '22-security.md', title: 'Security architecture & threat model',
    corpus: 'EXTREME section 50; RESEARCH_BRIEF section 7; verification section 27.',
    brief: `- STRIDE threat model per component; camera-network attacks (default credentials, RTSP sniffing, ONVIF and firmware vulnerabilities, rogue devices), video injection / replay / loop attacks and detection, adversarial patches against person detectors (e.g. Thys et al. 2019 and later) and mitigations, zone spoofing, insider misuse, ransomware on recorders.
- Controls with concrete configuration: network segmentation, RTSPS/SRTP where supported, digest auth, credential vault (Docker secrets, SOPS + age, OpenBao/Vault), internal TLS with step-ca, PostgreSQL sslmode verify-full, Redis ACL + TLS, NATS auth, container hardening (non-root, read-only rootfs, capabilities, seccomp), supply chain (uv lock with hashes, pip-audit, OSV-Scanner, Trivy, Syft SBOM, cosign signing, model file hashes, safetensors vs pickle, torch.load weights_only), SAST (bandit, semgrep), OWASP ASVS / API Security Top 10 (2023) checklist, web security headers and CSP.
- LLM/VLM security: OWASP Top 10 for LLM Applications 2025, prompt injection via text visible in video frames, output handling, tool allow-listing.
- Tamper-evident audit logs, security testing checklist, incident response plan; brief mapping to IEC 62443, NIST SP 800-53 Rev. 5, ISO/IEC 27001:2022 and ONVIF security guidance.`,
  },
  {
    id: '23', file: '23-privacy-compliance.md', title: 'Privacy, ethics, legal compliance & anonymization',
    corpus: 'RESEARCH_BRIEF section 12; RESEARCH_EXTENSION Track 10 (EU AI Act + Omnibus); verification section 28; EXTREME sections 51, 79.',
    brief: `- Legal landscape, verified live: EU AI Act (prohibited practices incl. untargeted facial-image scraping from CCTV and real-time remote biometric identification for law enforcement; high-risk categories in Annex III (biometrics, critical infrastructure); provider/deployer obligations (risk management, data governance, Art. 12 logging, transparency, Art. 14 human oversight, accuracy/robustness, Art. 27 fundamental-rights impact assessment); timeline including the Digital Omnibus changes: verify exact status and dates, correcting the corpus if needed); GDPR (lawful basis, Art. 9 special-category biometric data: is a Re-ID embedding biometric data? analysis with sources; Art. 35 DPIA; EDPB Guidelines 3/2019 on video devices); UK ICO video surveillance guidance and the Surveillance Camera Code of Practice; India DPDP Act 2023 and DPDP Rules 2025 (notification date, phased timelines, notice/consent, legitimate uses, security safeguards, breach notification, retention: verify); US (Illinois BIPA, CCPA/CPRA); aviation security context (ICAO Annex 17, EU Regulation 2015/1998) at a high level.
- Privacy-by-design features and implementations: no face recognition / biometric identification by default; data minimization; person/face anonymization in UI and exports (tools/models and their licenses, e.g. deface, EgoBlur, YOLO face models; MIRAGE from the corpus); retention enforcement; purpose limitation; access logging; subject-access-request redaction workflow; signage.
- Templates: DPIA outline for this system, AI Act technical documentation outline (Annex IV), FRIA outline, model card and datasheet templates.
- Bias and fairness testing methodology for detection/tracking across demographics and conditions (datasets with demographic annotations and their licenses, e.g. FACET), documentation.
- Ethics guardrails: observation vs inference, no profiling, human oversight, contestability.`,
  },
  {
    id: '24', file: '24-reid-topology.md', title: 'Research layer A: Re-ID, cross-camera association & camera topology',
    corpus: 'Verification sections 10-12, 27; RESEARCH_BRIEF section 2; RESEARCH_EXTENSION Track 4; EXTREME section 15-16.5; spec F5-F7, F28.',
    brief: `- Governance first: when to enable, privacy implications (biometric-adjacent), operator correction, confidence propagation (AGENTS.md: probabilistic, revisable identity).
- Models and libraries with licenses of code AND weights: OSNet (torchreid), fast-reid, CLIP-ReID, TransReID, SOLIDER, KPR, BoxMOT ReID weights, newer 2025-2026 options; weights trained on research-only datasets and what that means; ONNX/TensorRT export; vehicle Re-ID options.
- Embedding pipeline: crop quality gating (size, blur, occlusion, pose), when to extract (track birth, best frames, periodic), aggregation, normalization; storage in pgvector (HNSW settings), gallery TTL and deletion.
- Association algorithm: candidate generation with topology + travel-time + direction gates, a scoring function combining appearance with spatio-temporal likelihood, windowed Hungarian assignment, thresholds calibrated on site data, identity states (tentative / confirmed / merged / split), operator correction flow; overlapping-FOV fusion in world coordinates vs non-overlapping handoff.
- Topology: manual graph + learned transit-time distributions (histograms / log-normal fits), bootstrap to break the circular dependency (verification section 12), next-camera prediction (smoothed Markov) with honest metrics.
- Evaluation protocol for handoff accuracy (experiments E3/E4), datasets usable for it (MSMT17, WILDTRACK, MMPTRACK, MEVA, AI City synthetic), metrics.
- Complete reference code skeleton.`,
  },
  {
    id: '25', file: '25-vlm-nl-search.md', title: 'Research layer B: VLM event explanation & natural-language investigation',
    corpus: 'Verification sections 19-20; RESEARCH_EXTENSION Track 6 and T5.3; RESEARCH_BRIEF section 14; EXTREME section 38; spec F30-F31.',
    brief: `- Open-weight VLM options as of ${DATE} (Qwen3-VL sizes, any newer Qwen VL releases such as Qwen3.5 if they exist, InternVL3.5, Gemma 3/3n, MiniCPM-V, SmolVLM2, others): licenses, video input support, context length, VRAM at FP16/FP8/INT4, quality evidence.
- Serving: vLLM (version, video input API, structured outputs / guided decoding config), SGLang, llama.cpp (multimodal support), Ollama (does it support video input? verify); OpenAI-compatible APIs; GPU memory co-existence with the detector.
- Prior art: Frigate GenAI review summaries (docs) as the reference pattern.
- Design: triggered only by events, never on the alert critical path; frame selection around the event (keyframes, <= N frames), Set-of-Mark style overlays of track ids to ground references, prompt templates, JSON-schema output (observed actions, objects, uncertainty, explicit "cannot determine"), post-validation against tracks, hallucination checks, labeling as INFERENCE, timeouts, caching.
- NL investigation: tool calling with allow-listed parameterized queries (function schemas), local LLM options, answers must cite event ids, refusal when no evidence, audit logging; semantic search over events/frames with embeddings (SigLIP 2, Qwen3-VL-Embedding, CLIP) stored in pgvector; no face search.
- Evaluation (hallucination rate on labeled clips, JSON validity, latency); safety (prompt injection via scene text; Video-SafetyBench from the corpus).
- Complete reference code: VLM client, prompts, schema, validator, NL tool registry.`,
  },
  {
    id: '26', file: '26-crowd-forecast-anomaly-pose.md', title: 'Research layer C: crowd counting, forecasting, learned anomaly & pose',
    corpus: 'Verification sections 14, 16, 23; RESEARCH_EXTENSION T1.7, T1.9, Track 7-8; EXTREME sections 19-20, 29; spec F11-F13, F21, F27, F29.',
    brief: `- Crowd counting regimes (detection counts vs density maps; switch-over criteria), models with code/weights availability and licenses (CSRNet, DM-Count, P2PNet, CLTR, APGCC, PET, ZIP and other 2025-2026 methods), dataset licenses, perspective correction and per-m2 density via homography, fusion with detection counts, validation.
- Queue analytics algorithms and wait-time estimation.
- Short-term forecasting: seasonal naive, EWMA / Holt-Winters, AutoETS / AutoARIMA (statsforecast), LightGBM with calendar features (mlforecast), conformal prediction intervals; evaluation with MASE; baseline-first rule (Zeng et al.).
- Learned anomaly as CORROBORATION only: trajectory-feature Isolation Forest, pose-based anomaly (e.g. STG-NF), feature-based VAD with pretrained video encoders (licenses), CLIP-based (VadCLIP), training-free VLM approaches (ASK-Hint); per-scene calibration, data requirements, why benchmark AUC is not operational performance (Micro-AUROC critique in the corpus).
- Pose and actions: RTMPose via rtmlib (license), YOLO-pose (AGPL), ViTPose; skeleton action recognition (MMAction2 PoseC3D); fall detection datasets and realistic reliability; recommendation on deferral.
- Reference code for queue analytics, forecasting baselines and an Isolation Forest corroboration scorer.`,
  },
  {
    id: '27', file: '27-pitch-market-opensource.md', title: 'Hackathon pitch, market & user need, competition, open-source & community plan',
    corpus: 'Problem_statemen.txt (Round 1 rubric); spec sections 54-55, 60; verification sections 34-36; verification1.md section 20; EXTREME sections 6, 7, 69, 73.',
    brief: `- Round 1 rubric mapping (Problem-Solution Fit 15, Market & User Need 15, Innovation & Creativity 20, Technical Feasibility & Execution 15, Potential Impact 10, Scalability & Future Scope 10, Community Engagement & Open Source Contribution 15) and what strong answers look like for each. Note: the Canva template link cannot be accessed; structure the outline so it can be poured into any template.
- Slide-by-slide outline (about 10-14 slides) with draft content, the evidence to show, and visuals/diagrams to create; a 2-3 minute demo video plan; speaker notes.
- Market & user need with sources: market sizes for video surveillance, video analytics / AI video analytics, VSaaS, airport security (cite several firms, give ranges, label as vendor-report estimates); drivers (camera-to-operator ratios, alert fatigue, labor cost); user personas (control-room operator, security supervisor, airport/transport operations manager, DPO/compliance officer, IT/OT admin) with jobs-to-be-done and evidenced pain points; adjacent markets (rail/metro, stadiums, campuses, hospitals, industrial sites) from the corpus's domain profiles.
- Competitive landscape: commercial (Genetec, Milestone XProtect + BriefCam, Motorola Avigilon, Axis, Hanwha, Verkada, IronYun Vaidio, Ipsotek, Agent Vi, Irisity, Hikvision/Dahua analytics and procurement bans) and open source (Frigate, Viseron, Shinobi, ZoneMinder, Kerberos.io, DeepStream reference apps, Savant): features, gaps, pricing where public; positioning and defensible differentiation (explainable evidence-first calibrated alerts, open and vendor-neutral, privacy by design, measured claims).
- Innovation claims: which are defensible (per verification1 section 20 and the verification report) and which must be avoided or labeled; no "N citation errors fixed" claims.
- Impact metrics and how to measure them in the demo; scalability story (staged, measured).
- Community & open-source plan: license choice for Argus (Apache-2.0 vs AGPL consequences given dependencies), repo hygiene (README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY.md, governance, DCO), docs site (MkDocs Material), good-first-issues, upstream contributions (e.g. supervision, MediaMTX, pgvector docs), an open scenario/benchmark suite contribution (the non-overlap benchmark gap), responsible-AI docs (model cards), community channels, engagement metrics.
- Business model options (open core, services, managed), risks & mitigations slide, ethics slide.
- Offline hackathon demo script: hardware list, network plan, fallback to recorded feeds, rehearsal checklist.`,
  },
  {
    id: '28', file: '28-nonvideo-streams.md', title: 'Non-video real-time data streams & multi-source fusion',
    corpus: 'Problem_statemen.txt ("multiple real-time data streams"); EXTREME sections 12, 22-24; RESEARCH_BRIEF section 5.',
    brief: `- Why the problem statement's "multiple real-time data streams" requires more than video, and what to integrate first for the MVP vs later.
- MQTT (Mosquitto: license, version; paho-mqtt / aiomqtt clients), topic design, QoS, TLS/auth.
- Access control integration: ONVIF Profile A/C, OSDP, vendor APIs; door-forced / door-held-open events; privacy of badge data.
- ONVIF Profile M analytics metadata and events from cameras with on-board analytics.
- Fire/building systems: BACnet (BAC0 / bacpypes3), Modbus (pymodbus); alarm panels; licenses.
- Audio events (glass break, alarms, gunshot-like impulsive sounds) with open models (YAMNet, PANNs, BEATs: licenses) and privacy constraints on audio recording.
- Context feeds: weather APIs, flight schedule APIs (options and licensing), calendars.
- Sensor reading envelope conforming to §01, validation, clock handling, sensor health, and fusion into correlation (§10): confirmation ladders, weighted/Bayesian/Dempster-Shafer fusion with calibration.
- Simulators for each stream for the demo; complete reference code for an MQTT gateway and a sensor-to-event adapter.`,
  },
]

// ---------------- RUN (durable: state in manual/state, log in manual/state/progress.log) ----------------
const STATE_SCHEMA = {
  type: 'object',
  properties: {
    done: { type: 'array', items: { type: 'string' } },
    invalid: { type: 'array', items: { type: 'string' } },
    critic_json: { type: 'string' },
    consolidate_json: { type: 'string' },
  },
  required: ['done', 'invalid', 'critic_json', 'consolidate_json'],
}

phase('Resume scan')
const scan = await agent(`You list the durable state of the Argus manual workflow. Do exactly this:
1. Bash: mkdir -p ${BASH_STATE} && touch ${BASH_STATE}/progress.log && echo "$(date -Iseconds) | RUN-START | workflow (re)launched; scanning state" >> ${BASH_STATE}/progress.log
2. For every *.json file in ${STATE}: check it parses as valid JSON and is a non-empty object (use node, python, or PowerShell ConvertFrom-Json, whichever works).
3. Return: done = base names without ".json" of the valid files (e.g. "write-05", "verify-05", "critic", "deepen-07-geometry-calibration"); invalid = base names of files that fail to parse; critic_json = the full raw text of critic.json if it is valid, else ""; consolidate_json = the full raw text of consolidate.json if it is valid, else "".
Do not modify any file other than appending to progress.log.`, { label: 'resume-scan', phase: 'Resume scan', schema: STATE_SCHEMA, effort: 'low' })

if (!scan) {
  log('HALT: the resume scan itself failed (usage limit or outage?). Nothing else was started. Relaunch the same script later.')
  return { stopped_at: 'resume-scan', failures: ['resume-scan'], message: 'Resume scan failed; relaunch later.' }
}
const done = new Set(scan.done)
log(`Resume scan: ${done.size} completed steps found on disk (${[...done].join(', ')})${scan.invalid.length ? '; invalid state files to redo: ' + scan.invalid.join(', ') : ''}`)

// Halt switch: after HALT_AFTER agent failures in a row (typically a usage limit or outage), stop launching
// agents instead of burning through every remaining step. Everything finished is already on disk.
// Transient failures (e.g. intermittent TLS interception: SELF_SIGNED_CERT_IN_CHAIN) arrive in small bursts,
// so each step gets up to MAX_ATTEMPTS tries and the halt threshold is higher than a typical burst.
const failures = []
const succeeded = new Set()
const MAX_ATTEMPTS = 3
const HALT_AFTER = 6
let consecutiveFailures = 0
let halted = false
const pendingFailures = () => [...new Set(failures)].filter((k) => !succeeded.has(k))
async function runAgent(key, prompt, opts) {
  if (done.has(key) || succeeded.has(key)) { log(`skip ${key}: already done`); return { cached: true } }
  if (halted) { failures.push(key); return null }
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (halted) break
    let r = null
    try {
      r = await agent(prompt, { ...opts, label: attempt === 1 ? opts.label : `${opts.label}:retry${attempt - 1}` })
    } catch (e) {
      log(`${key} attempt ${attempt} threw: ${String(e).slice(0, 200)}`)
    }
    if (r) { consecutiveFailures = 0; succeeded.add(key); return r }
    consecutiveFailures++
    if (consecutiveFailures >= HALT_AFTER && !halted) {
      halted = true
      log(`HALT: ${consecutiveFailures} agent failures in a row (likely a usage limit or outage). No new agents will start; finished work is on disk. Relaunch the same script later to continue.`)
    }
    log(`${key} attempt ${attempt}/${MAX_ATTEMPTS} returned nothing${attempt < MAX_ATTEMPTS && !halted ? '; retrying (the agent continues its partial file)' : '; giving up for this run (relaunch the workflow to retry)'}`)
  }
  failures.push(key)
  return null
}
function stopEarly(stage) {
  const pf = pendingFailures()
  log(`Stopping before ${stage}: ${pf.length} step(s) not finished (${pf.join(', ')}). Later phases need complete inputs. Relaunch the same script to continue.`)
  return { stopped_at: stage, halted, failures: pf, done_on_disk_at_start: [...done], message: 'Relaunch the same script; finished steps are skipped automatically.' }
}

// DEEP mode: sections first written in deadline mode (09-28, ~8k words) get an Opus expansion pass to the
// full depth of 02-08 before their independent fact-check.
const DEEP = !!(args && args.deep)
const EXPAND_IDS = new Set(['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'])
function expandPrompt(s) {
  return `You are bringing one section of the Argus IMPLEMENTATION MANUAL up to FULL DEPTH. Today is ${DATE}. File: ${SEC}\\${s.file} (${s.id} - ${s.title}).
This section was written earlier in a deadline mode (capped around 8,000 words, fewer live checks). The bar is the full-depth sections 02-08 (23,000-36,000 words each): read one of them (e.g. ${SEC}\\06-tracking.md or ${SEC}\\07-geometry-calibration.md) to calibrate the expected depth and style, plus the foundation sections ${SEC}\\00a-stack-ml-video.md, 00b-stack-platform.md and 01-architecture-contracts.md for conformance.
The user's goal: during implementation nobody should need to stop and research anything for this topic.

Original scope brief for this section:
${s.brief}

Do this:
1. Read the whole existing file. Keep everything that is correct.
2. Expand every part of the section format to full depth: complete Quick reference; decisions with rationale, rejected alternatives and revisit triggers; a complete pinned-dependencies table; setup for Linux and Windows/WSL2; concepts and algorithms with math and default parameters; COMPLETE reference implementation (typed, importable modules plus tests, conforming to the §01 contracts; run them in a scratch directory if Python/Node are available and say so); every configuration key with default, unit and meaning; integration points; measurable acceptance tests; performance notes labeled by evidence type; failure modes (symptom -> cause -> fix); security and privacy notes; deferred items with adoption triggers; sources.
3. Research every addition live (ToolSearch "select:WebSearch,WebFetch"; PyPI/npm/GitHub APIs via curl or Invoke-RestMethod are fine). Prefer primary sources. Tag claims ([VERIFIED-DOC], [VERIFIED-SRC], [PAPER], [BENCHMARK], [ESTIMATE], [DESIGN], [UNVERIFIED]). Try to resolve existing [UNVERIFIED] tags.
4. Target 15,000-25,000 words in total. If the file is already at that depth and complete (e.g. section 11), only fill genuine gaps.
Insert material in the right places with Edit, or append with a Bash quoted heredoc: cat >> ${BASH_SEC}/${s.file} <<'EOF_ARGONIX'. Only edit this one file. Respect every AGENTS.md constraint.
Return the full structured section result (file, title, updated summary, key_decisions, pinned_versions, conflicts_with_foundation, open_questions, approx_words).

${durableNote('expand-' + s.id, 'expand:' + s.id)}`
}

function sectionStages(ph) {
  return [
    (s) => runAgent('write-' + s.id, preamble(s), { label: `write:${s.id}`, phase: ph, schema: SECTION_SCHEMA, effort: 'high', ...(s.foundation ? {} : FASTOPTS(true)) }),
    (res, s) => {
      if (res === null || !(DEEP && EXPAND_IDS.has(s.id))) return res
      return runAgent('expand-' + s.id, expandPrompt(s), { label: `expand:${s.id}`, phase: ph, schema: SECTION_SCHEMA, effort: 'high' })
        .then((x) => (x === null ? null : x.cached ? res : x))
    },
    (res, s) => {
      if (res === null) { log(`skip verify:${s.id} because write:${s.id} failed`); return { section: s, res: null, verify: null } }
      if (FAST && !s.foundation && !done.has('verify-' + s.id)) {
        return { section: s, res, verify: { skipped: true, claims_checked: 0, corrections: [], code_issues_fixed: [], still_unverified: [] } }
      }
      return runAgent('verify-' + s.id, verifyPrompt(s, res.cached ? null : res), { label: `verify:${s.id}`, phase: ph, schema: VERIFY_SCHEMA, effort: 'high' })
        .then((v) => ({ section: s, res, verify: v }))
    },
  ]
}

phase('Foundation')
log(`Foundation: ${FOUNDATION.length} sections, each fact-checked`)
const foundationRes = await pipeline(FOUNDATION, ...sectionStages('Foundation'))

phase('Research')
log(`Research: ${TRACKS.length} topic sections, each written by a deep-research agent then adversarially fact-checked`)
const trackRes = await pipeline(TRACKS, ...sectionStages('Research'))

let firstPass = [...foundationRes, ...trackRes].filter(Boolean)
const sectionIncomplete = (s) => {
  const r = firstPass.find((x) => x.section.id === s.id)
  return !r || r.res === null || r.verify === null
}
// Retry rounds: re-run every section that still has a failed step (finished steps are skipped).
for (let round = 1; round <= 2 && !halted; round++) {
  const bad = [...FOUNDATION, ...TRACKS].filter(sectionIncomplete)
  if (!bad.length) break
  log(`Retry round ${round}: re-running unfinished sections ${bad.map((s) => s.id).join(', ')}`)
  const again = (await pipeline(bad, ...sectionStages('Research'))).filter(Boolean)
  firstPass = firstPass.filter((x) => !bad.find((b) => b.id === x.section.id)).concat(again)
}
const failedIds = [...FOUNDATION, ...TRACKS].filter(sectionIncomplete).map((s) => s.id)
if (failedIds.length || pendingFailures().length) return stopEarly('Critique')

phase('Critique')
if (FAST) log('FAST mode: skipping critique, gap-fill and deepen passes')
let critic = FAST ? { gaps: [], thin: [], conflicts: [] } : null
if (done.has('critic') && scan && scan.critic_json) {
  try { critic = JSON.parse(scan.critic_json); log('critic: loaded from disk') } catch (e) { critic = null }
}
if (!critic) {
  done.delete('critic')
  critic = await runAgent('critic', `You are the completeness and consistency critic for the Argus implementation manual (section files in ${SEC}). Today is ${DATE}. The user's requirement: ONE top-to-bottom manual so that NO research is needed during implementation.
Writer results (summaries, key decisions, open questions, pinned versions) are in ${STATE}\\write-*.json and ${STATE}\\expand-*.json (an expand result supersedes the write result for the same section); fact-checker results in ${STATE}\\verify-*.json. Read them all, then the section files as needed (Grep/Read).
Sections that failed or are incomplete in this run: ${JSON.stringify(failedIds)}.

Tasks (do not edit any section file):
A. Gaps. Walk through actually building the MVP end to end, day by day (set up the dev machine on Windows 11 / WSL2 and the Ubuntu server, get frames from a camera, detect, track, calibrate, define zones, run rules, store in the DB, serve the API, show live video and alerts in the UI, extract evidence, score and calibrate alerts, capture feedback, test, deploy, run the hackathon demo), then the later research layers and the pitch. What would the implementer still have to research? Propose up to 8 NEW sections (ids "29", "30", ...; file names like "29-some-topic.md"; each with a detailed bullet-list brief). Only topics genuinely missing.
B. Thin sections: shallow coverage relative to need (TODO/TBD, "..." inside code, missing commands/defaults/config keys, unverified core facts, low word counts, failed or partial sections). For each: file name and a precise list of what to add. Max 14 entries.
C. Conflicts across sections: versions (grep PostgreSQL, TimescaleDB, pgvector, Python, CUDA, TensorRT, torch, onnxruntime, PyAV, FFmpeg, FastAPI, Pydantic, React, Vite, MapLibre, MediaMTX, go2rtc, Redis, NATS), package/module names, contract field names, ports, file paths, docker image tags, defaults, contradictory decisions. For each: description, files involved (file names), and the resolution (prefer the foundation sections unless factually wrong).
Return the structured result.

${durableNote('critic', 'critic')}`, { label: 'critic', phase: 'Critique', schema: CRITIC_SCHEMA, effort: 'high' })
}
if (!critic || critic.cached) return stopEarly('Fill (critic result unavailable)')

const gaps = (critic.gaps || []).slice(0, 8).map((g) => ({
  id: String(g.id),
  file: base(g.file).endsWith('.md') ? base(g.file) : base(g.file) + '.md',
  title: g.title,
  corpus: 'Consult whichever corpus documents are relevant (grep them).',
  brief: g.brief,
}))
if ((critic.gaps || []).length > 8) log(`Critic proposed ${critic.gaps.length} gap sections; running the first 8; dropped: ${critic.gaps.slice(8).map((g) => g.title).join('; ')}`)

const thinMap = {}
for (const t of critic.thin || []) {
  const f = base(t.file)
  thinMap[f] = thinMap[f] ? thinMap[f] + '\n' + t.missing : t.missing
}
let thinList = Object.entries(thinMap).map(([file, missingText]) => ({ file, missing: missingText }))
if (thinList.length > 14) { log(`Thin sections capped at 14 of ${thinList.length}; dropped: ${thinList.slice(14).map((t) => t.file).join(', ')}`); thinList = thinList.slice(0, 14) }
log(`Critic: ${gaps.length} new sections, ${thinList.length} thin sections to deepen, ${(critic.conflicts || []).length} conflicts`)
const keyOf = (file) => file.replace(/\.md$/, '')

phase('Fill')
const [gapRes, deepenRes] = await Promise.all([
  pipeline(gaps, ...sectionStages('Fill')),
  parallel(thinList.map((t) => () => runAgent('deepen-' + keyOf(t.file), `You are deepening one section of the Argus implementation manual. Today is ${DATE}. File: ${SEC}\\${t.file}. A completeness critic found it too thin for the goal (implementers must never need to research mid-build). Missing content to add:
${t.missing}

If the file does not exist or is only partial (its writer failed), write/complete it following the section format used by the other files in ${SEC}. Read the file first, plus the foundation sections (${SEC}\\00a-stack-ml-video.md, 00b-stack-platform.md, 01-architecture-contracts.md) for conformance. Research every addition live (ToolSearch "select:WebSearch,WebFetch"; PyPI/npm/GitHub APIs via curl or Invoke-RestMethod are fine), prefer primary sources, and tag claims ([VERIFIED-DOC], [VERIFIED-SRC], [PAPER], [BENCHMARK], [ESTIMATE], [DESIGN], [UNVERIFIED]). Insert new material in the right places with Edit, or append with a Bash quoted heredoc: cat >> ${BASH_SEC}/${t.file} <<'EOF_ARGONIX'. Code must be complete and conform to the §01 contracts. Add new sources to the Sources list and entries to the "Verification log" subsection for facts you checked. Do not remove correct material and do not edit other files.
Return: file, applied (what you added), skipped (what you could not add and why).

${durableNote('deepen-' + keyOf(t.file), 'deepen:' + t.file)}`, { label: `deepen:${t.file.slice(0, 3)}`, phase: 'Fill', schema: FIX_SCHEMA, effort: 'high' }))),
])

if (pendingFailures().length) return stopEarly('Harmonize')

phase('Harmonize')
let consol = null
if (done.has('consolidate') && scan && scan.consolidate_json) {
  try { consol = JSON.parse(scan.consolidate_json); log('consolidate: loaded from disk') } catch (e) { consol = null }
}
if (!consol) {
  done.delete('consolidate')
  consol = await runAgent('consolidate', `You are consolidating the pinned versions of the Argus implementation manual into the two stack sections so they become the complete single source of truth. Today is ${DATE}. Files to edit: ${SEC}\\00a-stack-ml-video.md (GPU/ML/video) and ${SEC}\\00b-stack-platform.md (data/backend/frontend/infra).
The pins reported by every section writer are in the "pinned_versions" arrays of ${STATE}\\write-*.json and ${STATE}\\expand-*.json (expand supersedes write for the same section); also grep the section files in ${SEC} for version strings, docker tags and install commands.
Tasks:
1. For each component decide the canonical version. Where sections disagree, or a pin looks older than the latest stable release, verify live (ToolSearch "select:WebSearch,WebFetch", PyPI/npm/GitHub releases, official docs; curl is fine) and pick the right one, respecting compatibility constraints (driver/CUDA/TensorRT/torch/onnxruntime/Python; PostgreSQL/TimescaleDB/pgvector).
2. Edit 00a and 00b (only these two files) so every component used anywhere in the manual appears with version, license, install command and source; add a "Consolidated from sections" note listing which sections use which component.
3. Return canonical versions and the list of fixes needed in OTHER section files (file name + exact fix, e.g. "replace TimescaleDB 2.x.y with 2.a.b in the Pinned dependencies table and the docker-compose image tag").

${durableNote('consolidate', 'consolidate')}`, { label: 'consolidate-versions', phase: 'Harmonize', schema: CONSOL_SCHEMA, effort: 'high', ...FASTOPTS(true) })
}
if (!consol || consol.cached) return stopEarly('fixes (consolidation result unavailable)')

const fixesByFile = {}
for (const f of consol.fixes || []) {
  const k = base(f.file)
  ;(fixesByFile[k] = fixesByFile[k] || []).push('[version] ' + f.fix)
}
for (const c of critic.conflicts || []) {
  for (const f of c.files || []) {
    const k = base(f)
    ;(fixesByFile[k] = fixesByFile[k] || []).push(`[conflict] ${c.description} => RESOLUTION: ${c.resolution}`)
  }
}
const fixTargets = Object.entries(fixesByFile).filter(([file]) => /^\d/.test(file))
log(`Harmonize: applying fixes to ${fixTargets.length} files`)
const fixRes = await parallel(fixTargets.map(([file, fixes]) => () => runAgent('fix-' + keyOf(file), `You are applying consistency fixes to one section of the Argus implementation manual. Today is ${DATE}. File: ${SEC}\\${file}.
Canonical versions live in ${SEC}\\00a-stack-ml-video.md and ${SEC}\\00b-stack-platform.md; contracts in ${SEC}\\01-architecture-contracts.md.
Fixes to apply (some may already be applied or may not apply to this file; check first):
${fixes.map((x, i) => `${i + 1}. ${x}`).join('\n')}
Apply each with Edit, keeping the section coherent (update tables, code, commands and prose that depend on the changed fact). If a fix looks wrong, verify live before applying and explain in skipped. Only edit this one file. Add a line per applied fix to its "Verification log" subsection.
Return: file, applied, skipped.

${durableNote('fix-' + keyOf(file), 'fix:' + file)}`, { label: `fix:${file.slice(0, 3)}`, phase: 'Harmonize', schema: FIX_SCHEMA, ...FASTOPTS(false) })))

if (pendingFailures().length) return stopEarly('Synthesize')

phase('Synthesize')
const synth = await parallel([
  () => runAgent('synth-90', `Write the MASTER BUILD PLAN for the Argus implementation manual to ${SEC}\\90-build-plan.md. Today is ${DATE}. All other sections are in ${SEC}; writer summaries, key decisions and open questions are in ${STATE}\\write-*.json and ${STATE}\\expand-*.json (expand supersedes write). Read what you need (Grep headings, Read relevant parts).
Content: "# 90 — Master build plan" then: day-0 environment setup checklist (Windows 11 + WSL2 dev and Ubuntu 24.04 GPU server, with exact commands or pointers to the exact subsections); milestones M0..Mn from an empty repo to the MVP demo and then to the research layers, following AGENTS.md (measure before scaling; defer Re-ID/VLM/forecasting/learned anomaly until the measurement and feedback loop works); for each milestone: goal, tasks (checkbox list with section references), dependencies, deliverables, acceptance tests with measurable thresholds, demo-able outcome, risks; a critical path; a hackathon timeline mapping (Round 1 deck first, then the offline hackathon build) with a suggested split of work for a team of 3-5 people; a definition of done; a risk register; and an "open decisions the team must make" list consolidated from the sections. Be concrete and complete. If the file already exists (from an earlier, shallower pass or an interrupted run), update and extend it so it reflects every section as it is NOW (several sections were expanded and fact-checked since), keeping what is still correct. Do not edit other files. Return file + summary.

${durableNote('synth-90', 'synth:90')}`, { label: 'build-plan', phase: 'Synthesize', schema: SYN_SCHEMA, effort: 'high', ...FASTOPTS(true) }),
  () => runAgent('synth-91', `Write the DECISION REGISTER and GLOSSARY for the Argus implementation manual to ${SEC}\\91-decision-register.md. Today is ${DATE}. All sections are in ${SEC}; writer key decisions and open questions are in ${STATE}\\write-*.json and ${STATE}\\expand-*.json (expand supersedes write). Read the files as needed (Grep for "Decision", "ADR", "Quick reference").
Content: "# 91 — Decision register, open questions & glossary" then: (1) an ADR-style register table of every significant decision across sections (id, decision, rationale, rejected alternatives, revisit trigger, section reference), grouped by area; flag decisions that contradict each other; (2) an open-questions register (question, owner role, needed by milestone, section); (3) a claims register: what Argus may and may not claim publicly (per AGENTS.md and the verification report), with the evidence needed to upgrade a claim; (4) a glossary of all technical terms and acronyms used in the manual (one-line definitions). If the file already exists (from an earlier, shallower pass or an interrupted run), update and extend it so it reflects every section as it is NOW (several sections were expanded and fact-checked since), keeping what is still correct. Do not edit other files. Return file + summary.

${durableNote('synth-91', 'synth:91')}`, { label: 'decision-register', phase: 'Synthesize', schema: SYN_SCHEMA, effort: 'high', ...FASTOPTS(true) }),
])

if (pendingFailures().length) return stopEarly('Assemble')

phase('Assemble')
const assembled = await runAgent('assemble', `Assemble the Argus implementation manual into ONE file: ${ROOT}\\IMPLEMENTATION_MANUAL.md. Today is ${DATE}. Source files: every *.md in ${SEC}, in lexical file-name order (00a, 00b, 01 ... 28, 29+, 90, 91). Use a script (bash, node or python) so the section text is copied byte-for-byte; do not retype content.
Structure: (1) a front page: title "Argus — Implementation Manual", generation date, one paragraph on purpose (single top-to-bottom reference so no research is needed during implementation), how to use it (start with §90 build plan, pinned versions in §00a/§00b, contracts in §01), the evidence-tag legend ([VERIFIED-DOC], [VERIFIED-SRC], [PAPER], [BENCHMARK], [ESTIMATE], [DESIGN], [UNVERIFIED]); (2) a status table per section: file, words, fact-checker claims checked / corrections / still unverified (from ${STATE}\\verify-*.json when present), and flags for sections whose writer never completed (no write-*.json state file); sections that have a write-*.json but no verify-*.json were checked only by their own writer (deadline mode skipped the independent fact-check), so mark them "writer-verified only" in the table and explain this on the front page; (3) a table of contents of every H1 and H2 heading with working GitHub-style anchor links; (4) the sections, each separated by a horizontal rule. Afterwards check the output (file exists, all sections present, heading count matches the sources) and fix any problem.
Put the list of any sections missing a DONE state on the front page under "Known gaps" so the team knows what to rerun.
Return file + summary (sizes, section count, known gaps).

${durableNote('assemble', 'assemble')}`, { label: 'assemble', phase: 'Assemble', schema: SYN_SCHEMA, ...FASTOPTS(false) })

log(pendingFailures().length ? `Finished with ${pendingFailures().length} failed steps: ${pendingFailures().join(', ')}. Relaunch the same script to retry only those.` : 'Finished: all steps completed.')

return {
  failures,
  skipped_as_done: [...done],
  sections: [...firstPass, ...(gapRes || []).filter(Boolean)].map((r) => ({ id: r.section.id, file: r.section.file, write: r.res ? (r.res.cached ? 'cached' : 'ok') : 'FAILED', verify: r.verify ? (r.verify.cached ? 'cached' : `ok (${r.verify.claims_checked} checked, ${r.verify.corrections.length} corrected, ${r.verify.still_unverified.length} unverified)`) : 'FAILED/skipped' })),
  critic: { gaps: gaps.map((g) => g.id + ' ' + g.title), thin: thinList.map((t) => t.file), conflicts: (critic.conflicts || []).length },
  deepen: (deepenRes || []).map((d, i) => ({ file: thinList[i] && thinList[i].file, status: d ? (d.cached ? 'cached' : `ok (${d.applied.length} added)`) : 'FAILED' })),
  fixes: (fixRes || []).map((f, i) => ({ file: fixTargets[i] && fixTargets[i][0], status: f ? (f.cached ? 'cached' : `ok (${f.applied.length} applied)`) : 'FAILED' })),
  synth: (synth || []).map((x) => (x ? (x.cached ? 'cached' : x.file) : 'FAILED')),
  assembled: assembled ? (assembled.cached ? 'cached' : assembled) : 'FAILED',
}
