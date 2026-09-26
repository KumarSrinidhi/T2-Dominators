#!/usr/bin/env bash
# Stand up / update Argus on a WINDOWS GPU host over OpenSSH (remote shell = cmd.exe), from this machine.
#   bash argonix_app/tools/bootstrap_windows_host.sh user@host [step...]
# Steps (default: all): code data restore env vlmenv
#   code    - git HEAD of argonix_app  -> %USERPROFILE%\argonix\argonix_app
#   data    - local data/models, data/real, data/synth
#   restore - newest backup snapshot (run data + configs) and VLM weights from ArgonixBackups
#   env     - uv sync --extra ${TORCH_EXTRA:-cu126} and a CUDA check
#   vlmenv  - %USERPROFILE%\argonix_vlm venv (torch ${TORCH_EXTRA}, transformers, bitsandbytes) + server files
# Needs on the host: NVIDIA driver, uv, Windows tar (built in). Services are started by tools/win_services.sh.
set -euo pipefail
HOST=${1:?usage: bootstrap_windows_host.sh user@host [steps]}; shift || true
STEPS=${*:-code data restore env vlmenv}
BK=${BK:-/c/Users/Ivermectin/ArgonixBackups}
EXTRA=${TORCH_EXTRA:-cu126}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
KH="$(cygpath -u "$TEMP" 2>/dev/null || echo /tmp)/argonix_known_hosts"
O=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR)
R() { ssh "${O[@]}" "$HOST" "$@" < /dev/null; }
PUT() { ssh "${O[@]}" "$HOST" "tar -x -f - -C $1"; }   # stdin tar stream -> remote dir (forward slashes)
HOME_W=$(R "echo %USERPROFILE%" | tr -d '\r' | sed 's#\\#/#g')
APP="$HOME_W/argonix/argonix_app"; VLM="$HOME_W/argonix_vlm"
has() { [[ " $STEPS " == *" $1 "* ]]; }

R "if not exist \"$HOME_W/argonix\" mkdir \"$HOME_W/argonix\"" || true
R "if not exist \"$VLM\" mkdir \"$VLM\"" || true
if has code; then echo "== code"; (cd "$ROOT" && git archive HEAD argonix_app) | PUT "$HOME_W/argonix"; fi
if has data; then echo "== data"; (cd "$ROOT/argonix_app" && tar -c data/models data/real data/synth) | PUT "$APP"; fi
if has restore; then
  LATEST=$(ls -1d "$BK"/snapshots/*/ 2>/dev/null | sort | tail -1)
  if [ -n "$LATEST" ]; then echo "== restore $LATEST"; (cd "$LATEST" && tar -c data configs) | PUT "$APP"; fi
  if [ -d "$BK/vlm_models" ]; then
    R "if not exist \"$VLM/models\" mkdir \"$VLM/models\"" || true
    for m in $(ls "$BK/vlm_models"); do
      if ! R "if exist \"$VLM/models/$m/.complete\" (echo yes)" | grep -q yes; then
        echo "== VLM weights $m"; (cd "$BK/vlm_models" && tar -c "$m") | PUT "$VLM/models"
      fi
    done
  fi
fi
if has env; then
  echo "== app env ($EXTRA)"
  R "cd /d \"$APP\" && uv sync -q --extra $EXTRA && uv run --no-sync python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())\""
fi
if has vlmenv; then
  echo "== VLM env"
  (cd "$ROOT/argonix_app/tools" && tar -c vlm_server.py) | PUT "$VLM"
  R "cd /d \"$VLM\" && (if not exist .venv uv venv -q -p 3.13 .venv) && uv pip install -q --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/$EXTRA && uv pip install -q --python .venv/Scripts/python.exe \"transformers>=4.49\" accelerate pillow fastapi \"uvicorn>=0.30\" num2words bitsandbytes && .venv\\Scripts\\python.exe -c \"import torch,transformers;print('vlm torch',torch.__version__,torch.cuda.is_available(),'transformers',transformers.__version__)\""
fi
echo "done ($STEPS) on $HOST: app at $APP, vlm at $VLM"
