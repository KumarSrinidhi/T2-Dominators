## What and why

<!-- What this changes, and the problem it solves. Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran, and for UI changes a screenshot. Numbers in docs or UI say what data and hardware they come from. -->

- [ ] `uvx ruff check argonix_app --select E9,F63,F7,F82`
- [ ] `uv run pytest -q -p no:warnings -m "not slow"` (from `argonix_app/`)
- [ ] New rules, analyzers or endpoints have tests
- [ ] New config keys are documented in `argonix_app/README.md` with their defaults

## Project rules

- [ ] Events describe what was observed; nothing labels a person as dangerous or suspicious
- [ ] Learned models only corroborate (learned-only alerts stay capped at P3)
