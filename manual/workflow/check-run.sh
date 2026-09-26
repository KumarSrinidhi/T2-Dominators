#!/usr/bin/env bash
# Prints the state of the Argus manual workflow for the auto-relaunch job:
#   COMPLETE          - assemble.json exists (the manual was re-assembled after the deep pass)
#   STALE <minutes>   - no log line for >= 45 min: the run halted (usage limit) or died -> relaunch
#   ACTIVE <minutes>  - a run is still writing log lines -> do nothing
STATE=/c/Users/Ivermectin/Desktop/Argonix/manual/state
LOG=$STATE/progress.log
if [ -f "$STATE/assemble.json" ]; then
  echo "COMPLETE"
  exit 0
fi
last=$(grep -E '^[0-9]{4}-' "$LOG" | tail -n 1 | cut -d' ' -f1)
now=$(date +%s)
then_s=$(date -d "$last" +%s)
age=$(( (now - then_s) / 60 ))
if [ "$age" -ge 45 ]; then
  echo "STALE $age"
else
  echo "ACTIVE $age"
fi
echo "done steps: $(ls "$STATE"/*.json 2>/dev/null | wc -l)"
echo "last line: $(grep -E '^[0-9]{4}-' "$LOG" | tail -n 1 | cut -c1-200)"
