#!/usr/bin/env bash
# Pull everything that is NOT in git from a Windows GPU host (e.g. LEGION) to this machine. Windows counterpart of
# backup_gpu_host.sh; run from Git Bash on the laptop:
#   bash argonix_app/tools/backup_windows_host.sh [user@host] [dest_dir]
# Backs up: data/run_* (SQLite via the online backup API, evidence, recordings, secret.key), configs/ (users.json and
# camera credentials are kept OUT of git), trained weights data/models/argonix_* + training logs, and once: VLM weights
# not already in dest/vlm_models. Old snapshots are not deleted automatically (prune by hand).
set -euo pipefail
HOST=${1:-Luikz@192.168.137.135}
DEST=${2:-/c/Users/Ivermectin/ArgonixBackups}
KH="$(cygpath -u "$TEMP" 2>/dev/null || echo /tmp)/argonix_known_hosts"
O=(-o BatchMode=yes -o ConnectTimeout=15 -o UserKnownHostsFile="$KH" -o LogLevel=ERROR)
R() { ssh "${O[@]}" "$HOST" "$@" < /dev/null; }
PS() { ssh "${O[@]}" "$HOST" "powershell -NoProfile -Command -"; }   # stdin script -> remote PowerShell
STAMP=$(date +%Y%m%d-%H%M%S)
SNAP="$DEST/snapshots/$STAMP"
mkdir -p "$SNAP" "$DEST/vlm_models"
HOME_W=$(R "echo %USERPROFILE%" | tr -d '\r' | sed 's#\\#/#g')
APP="$HOME_W/argonix/argonix_app"

# 1) consistent SQLite copies next to each live DB (<db>.bak), using the app's own venv Python
PS <<EOF
\$null = 0
Set-Location "$APP"
\$py = ".venv\\Scripts\\python.exe"
Get-ChildItem data -Directory -Filter "run_*" | ForEach-Object { Get-ChildItem \$_.FullName -Filter *.db } | ForEach-Object { & \$py -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[1]+'.bak'); s.backup(d); d.close(); s.close()" \$_.FullName }
EOF

# 2) stream run data + configs + trained weights as one tar (Windows bsdtar; live .db files excluded, .bak kept)
# (`dir` exits 1 when nothing matches; `|| true` keeps pipefail from aborting)
ITEMS=$( (R "cd /d \"$APP\" && dir /b /ad data\\run_* 2>nul" || true) | tr -d '\r' | sed 's#^#data/#' | tr '\n' ' ')
EXTRA=$( (R "cd /d \"$APP\" && dir /b data\\models\\argonix_* 2>nul" || true) | tr -d '\r' | sed 's#^#data/models/#' | tr '\n' ' ')
[ -n "$(R "if exist \"$APP/data/train_logs\" echo y" | tr -d '\r')" ] && EXTRA="$EXTRA data/train_logs"
echo "== run data: ${ITEMS:-none}; extra: ${EXTRA:-none}"
# shellcheck disable=SC2086
R "cd /d \"$APP\" && tar -c -f - --exclude *.db --exclude *.db-journal --exclude *.db-wal --exclude *.db-shm $ITEMS $EXTRA configs" \
  | tar -x -f - -C "$SNAP"
find "$SNAP/data" -name "*.db.bak" 2>/dev/null | while read -r f; do mv "$f" "${f%.bak}"; done
PS <<EOF
\$null = 0
Get-ChildItem "$APP/data" -Directory -Filter "run_*" | ForEach-Object { Get-ChildItem \$_.FullName -Filter *.db.bak } | Remove-Item -Force
EOF

# 3) VLM weights: only models not already backed up (marker file .complete)
for m in $( (R "dir /b /ad \"$HOME_W/argonix_vlm/models\" 2>nul" || true) | tr -d '\r'); do
  if [ ! -f "$DEST/vlm_models/$m/.complete" ]; then
    echo "== VLM weights $m (large, LAN speed) ..."
    R "tar -c -f - -C \"$HOME_W/argonix_vlm/models\" $m" | tar -x -f - -C "$DEST/vlm_models"
    touch "$DEST/vlm_models/$m/.complete"
  fi
done

echo "$STAMP" > "$DEST/LATEST_SNAPSHOT.txt"   # not "LATEST": a "latest" dir may exist and Windows names are case-insensitive
echo "backup ok: $SNAP ($(du -sh "$SNAP" | cut -f1)); snapshots kept: $(ls -1 "$DEST/snapshots" | wc -l); models: $(ls "$DEST/vlm_models" | tr '\n' ' ')"
