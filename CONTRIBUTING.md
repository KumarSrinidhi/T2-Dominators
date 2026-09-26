# Contributing to Argus

Thanks for considering a contribution. Argus is an explainable multi-camera situational-awareness demo: deterministic rules and baselines raise alerts, learned models only corroborate, and every alert cites its rule, camera, zone and evidence clip.

The runnable app lives in [`argonix_app/`](argonix_app/). The design manual (`IMPLEMENTATION_MANUAL.md`, `manual/`) and the research documents at the repo root are reference material.

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.13. ffmpeg is optional (clip export, RTSP replay).

```bash
cd argonix_app
uv sync --extra cpu          # or --extra cu126 on an NVIDIA GPU with a CUDA 12.6+ driver
uv run python run.py --open-demo
```

`--open-demo` skips login for local demos. Without it, the first run prints a setup code to create the admin account.

## Before you open a pull request

Run the same checks as CI (`.github/workflows/ci.yml`) from `argonix_app/`:

```bash
uvx ruff check . --select E9,F63,F7,F82
uv run pytest -q -p no:warnings -m "not slow"
```

Run the full suite (`uv run pytest -q -p no:warnings`, about 5 minutes) when you touch the pipeline, detection or tracking.

- **Tests.** New rules, analyzers and API endpoints come with tests in `argonix_app/tests/`. API tests must either use `open_demo=True` or send the `X-Argonix-CSRF: 1` header.
- **Config keys.** Document a new config key in the README section for that feature, with its default.
- **Dashboard.** The dashboard is vanilla ES modules with no build step (`argonix/static/js/`). Register new pages with `registerPage()` in `registry.js`.
- **Numbers.** Every accuracy or performance number in docs or UI text states what was measured, on which data and on which hardware. Don't present an estimate as a demonstrated result.

## What we will and won't merge

These come from [`AGENTS.md`](AGENTS.md), which is the project's hard-rule list. We will not merge a change that:

- labels a person as dangerous, suspicious or criminal, or infers intent, from appearance or anything else. Events describe what was observed ("person in restricted zone for 42 s"), not who someone is;
- lets a learned model (anomaly score, classifier, VLM) raise an alert above **P3** on its own. The correlator caps learned-only alerts until a deterministic rule agrees;
- presents cross-camera identity links as certain. They are probabilistic, revisable and research-grade;
- adds face recognition, or trains on or ships anything derived from **DukeMTMC**;
- adds infrastructure the project deliberately avoids (Postgres, a message bus, Kubernetes) without measurements showing SQLite and the current process model are the bottleneck.

## Reporting bugs and proposing features

Open a GitHub issue with the steps to reproduce, the config you ran, and the relevant part of the app log. For a security problem, **don't** open a public issue: see [SECURITY.md](SECURITY.md).

## Code of conduct

Everyone taking part in this project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

The project licence has not been chosen yet. Until a `LICENSE` file is added, contributions are accepted on the understanding that they will be released under the licence the maintainers pick. Third-party model weights and datasets keep their own licences; see [`argonix_app/docs/model-cards/`](argonix_app/docs/model-cards/).
