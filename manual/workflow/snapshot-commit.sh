#!/usr/bin/env bash
# Periodically commits and pushes the manual workflow's progress to GitHub (origin/main).
# Usage: bash snapshot-commit.sh [interval_seconds]   (default 1800 = 30 min)
# Runs until the deep pass is assembled (manual/state/assemble.json exists), then pushes a final snapshot and exits.
# Logs to manual/state/git-snapshots.log (NOT progress.log, whose last-line age drives the auto-relaunch check).
REPO=/c/Users/Ivermectin/Desktop/Argonix
GITLOG=$REPO/manual/state/git-snapshots.log
INTERVAL=${1:-1800}
cd "$REPO" || exit 1

note() { echo "$(date -Iseconds) | $*" >> "$GITLOG"; echo "$(date -Iseconds) | $*"; }

snapshot() {
  git add -A
  if git diff --cached --quiet; then
    note "no changes; nothing to commit"
    return 0
  fi
  local steps words msg
  steps=$(ls manual/state/*.json 2>/dev/null | wc -l)
  words=$(cat manual/sections/*.md 2>/dev/null | wc -w)
  msg="Manual deep pass snapshot $(date '+%Y-%m-%d %H:%M') ($steps steps done, $words words in sections)"
  if ! git commit -q -m "$msg" \
      -m "Automatic snapshot of the Argus implementation-manual research workflow (manual/sections, manual/state)."; then
    note "commit FAILED"
    return 1
  fi
  if git push -q origin main; then
    note "pushed: $msg"
  elif git pull -q --rebase --autostash origin main && git push -q origin main; then
    note "pushed after rebase: $msg"
  else
    note "push FAILED (commit kept locally; will retry next cycle): $msg"
  fi
}

while true; do
  snapshot
  if [ -f "$REPO/manual/state/assemble.json" ]; then
    note "deep pass assembled; final snapshot done; stopping"
    exit 0
  fi
  sleep "$INTERVAL"
done
