# Runs detached on a Windows GPU host (launch via Win32_Process, see win_services.ps1): prepare -> train -> eval the
# weapon detector, pausing the VLM to free VRAM. Status + logs in data\train_logs. Works offline (datasets already in
# %USERPROFILE%\argonix_datasets, base weights in data\models).
$ErrorActionPreference = "Continue"
$App = Join-Path $env:USERPROFILE "argonix\argonix_app"
$Ds = Join-Path $env:USERPROFILE "argonix_datasets"
Set-Location $App
$py = Join-Path $App ".venv\Scripts\python.exe"
$L = Join-Path $App "data\train_logs"
New-Item -ItemType Directory -Force $L | Out-Null
$env:YOLO_OFFLINE = "1"; $env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"
"stage vlm-stop $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
& powershell -NoProfile -ExecutionPolicy Bypass -File tools\win_services.ps1 stop-vlm *> "$L\vlm_stop.log"
$vp = Join-Path $env:USERPROFILE "argonix_vlm\vlm.pid"
if (Test-Path $vp) { $p = [int](Get-Content $vp); Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*vlm_server.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"stage prepare $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
if (-not (Test-Path data\weapons_ds\weapons.yaml)) { & $py tools\train_weapons.py prepare --datasets $Ds --out data\weapons_ds *> "$L\weapons_prepare.log" }
"stage train rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
# 6 dataloader workers: Windows spawns them slowly (about 1 min before the first batch), be patient
& $py tools\train_weapons.py train --data data\weapons_ds\weapons.yaml --epochs 50 --batch 16 --workers 6 --patience 15 *> "$L\weapons_train.log"
"stage eval rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
& $py tools\eval_threats.py weapons --datasets $Ds --device 0 *> "$L\weapons_eval.log"
"stage eval-normal rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
& $py tools\eval_threats.py normal --datasets $Ds --device 0 --only objects *> "$L\weapons_eval_normal.log"
"stage vlm-start rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
& powershell -NoProfile -ExecutionPolicy Bypass -File tools\win_services.ps1 start-vlm *> "$L\vlm_start.log"
"done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$L\weapons_status.txt"
