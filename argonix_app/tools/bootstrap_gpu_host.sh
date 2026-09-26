#!/usr/bin/env bash
# Stand up Argus on a NEW Linux GPU host from this machine: code (git HEAD) + models/videos + latest backup.
#   bash argonix_app/tools/bootstrap_gpu_host.sh user@new-host [backup_dir]
# Requirements on the host: NVIDIA driver (nvidia-smi works), python3, ssh key login. No sudo/curl needed.
# Env overrides: TORCH_INDEX (default cu130), TORCH_SPEC for the VLM env; SKIP_VLM=1 to skip the VLM.
set -euo pipefail
HOST=${1:?usage: bootstrap_gpu_host.sh user@host [backup_dir]}
BK=${2:-/c/Users/Ivermectin/ArgonixBackups}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)            # repo root
KH="$(cygpath -u "$TEMP" 2>/dev/null || echo /tmp)/argonix_known_hosts"
O=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR)
R() { ssh "${O[@]}" "$HOST" "$@" < /dev/null; }
LATEST=$( [ -e "$BK/latest" ] && readlink -f "$BK/latest" || echo "$BK/snapshots/$(cat "$BK/LATEST" 2>/dev/null)" )

echo "== host check"; R 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 --version'
echo "== uv"; R 'mkdir -p ~/.local/bin ~/argonix ~/argonix_vlm && [ -x ~/.local/bin/uv ] || (cd /tmp && python3 -c "import urllib.request as u; u.urlretrieve(\"https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz\", \"uv.tgz\")" && tar -xzf uv.tgz && cp uv-x86_64-unknown-linux-gnu/uv* ~/.local/bin/)'
echo "== code (git HEAD)"; (cd "$ROOT" && git archive HEAD argonix_app) | ssh "${O[@]}" "$HOST" "tar -x -C ~/argonix"
echo "== models + public videos"; (cd "$ROOT/argonix_app" && tar -c data/models data/real 2>/dev/null) | ssh "${O[@]}" "$HOST" "tar -x -C ~/argonix/argonix_app"
if [ -d "$LATEST" ]; then
  echo "== restore run data + configs from $LATEST"
  (cd "$LATEST" && tar -c .) | ssh "${O[@]}" "$HOST" "tar -x -C ~/argonix/argonix_app"
fi
echo "== app env (CUDA torch)"; R 'cd ~/argonix/argonix_app && ~/.local/bin/uv sync -q --extra cu128 && ~/.local/bin/uv run --no-sync python -c "import torch; print(\"cuda\", torch.cuda.is_available())"'
# TensorRT is not in the lock (uv sync would remove it): install after sync; engines are GPU-specific, so rebuild
# them on the new host with tools/bench_gpu.py --export (the detector falls back to .pt fp16 until then).
echo "== TensorRT (optional)"; R 'cd ~/argonix/argonix_app && UV_HTTP_TIMEOUT=600 ~/.local/bin/uv pip install -q tensorrt-cu12 onnx onnxslim "nvidia-modelopt[onnx]" || echo "TensorRT install failed: using .pt fp16"'
if [ "${SKIP_VLM:-0}" != 1 ]; then
  echo "== VLM env + weights"
  scp -q "${O[@]}" "$ROOT"/argonix_app/tools/{vlm_server.py,start_vlm.sh,remote_setup.sh} "$HOST":argonix_vlm/
  if [ -d "$BK/vlm_models" ]; then (cd "$BK/vlm_models" && tar -c .) | ssh "${O[@]}" "$HOST" "mkdir -p ~/argonix_vlm/models && tar -x -C ~/argonix_vlm/models"; fi
  R "cd ~/argonix_vlm && sed -i 's/\r$//' *.sh vlm_server.py && chmod +x *.sh && TORCH_INDEX=${TORCH_INDEX:-cu130} TORCH_SPEC='${TORCH_SPEC:-torch torchvision}' ./remote_setup.sh | tail -2 && HF_HUB_OFFLINE=1 ./start_vlm.sh"
  sleep 60
fi
echo "== start app"; R 'cd ~/argonix/argonix_app && chmod +x tools/*.sh && ./tools/start_app.sh configs/real_site_gpu.json 8000 data/run_real_site'
echo "done: http://${HOST#*@}:8000"
