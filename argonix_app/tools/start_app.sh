#!/usr/bin/env bash
# (Re)start the Argus site + dashboard in the background, tracked by PID file.
# Usage: ./tools/start_app.sh [config] [port] [data_dir]
#   e.g. ./tools/start_app.sh configs/real_site_gpu.json 8000 data/run_real_site
cd "$(dirname "$0")/.."
CFG=${1:-configs/real_site.json}
PORT=${2:-8000}
DATA=${3:-data/run_$(basename "$CFG" .json)}
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -f app.pid ] && kill "$(cat app.pid)" 2>/dev/null && sleep 2
VLM_ARGS=()
if python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8100/health', timeout=3)" 2>/dev/null; then
  VLM_ARGS=(--vlm-url http://127.0.0.1:8100)
fi
setsid nohup "$UV" run python run.py --config "$CFG" --data "$DATA" --port "$PORT" \
  "${VLM_ARGS[@]}" > app.log 2>&1 < /dev/null &
echo $! > app.pid
echo "started pid $(cat app.pid) on port $PORT (vlm: ${VLM_ARGS[*]:-off})"
