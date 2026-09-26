# Start/stop/status for Argus on a Windows GPU host. Processes are created via Win32_Process so they survive
# the SSH session that launched them (OpenSSH kills its job object on disconnect). Not persistent across reboots.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win_services.ps1 <action>
#     actions: start | stop | status | start-app | start-vlm | stop-app | stop-vlm | start-infer | stop-infer
#     [-Config configs/real_site_gpu.json] [-Data data/run_real_site] [-Port 8000]
#     inference node: [-InferDir <project dir, default: the app>] [-InferHost <ip, default 127.0.0.1>]
#                      [-InferPort 8200] [-Weights auto] [-Imgsz 960] [-MaxBatch 16] [-Conf 0.1]
# The app detects through the local inference node (one detector on the GPU, shared with remote clients; CPU
# fallback while the node is down). Start the node first: start-infer, then start.
# Everything listens on 127.0.0.1 only (app, VLM, inference node): nothing is bound to a network interface. Clients
# reach them through an SSH tunnel, e.g. ssh -N -L 8200:127.0.0.1:8200 -L 18000:127.0.0.1:8000 user@gpu-host
#                      [-ExtraClasses 24,26,28  (object classes for sites with unattended-object analysis)]
# The inference node's bearer token is created on first start in <InferDir>\data\infer_token; copy it to clients.
param([string]$Action = "status", [string]$Config = "configs/real_site_gpu.json",
      [string]$Data = "data/run_real_site", [int]$Port = 8000,
      [string]$InferDir = "", [string]$InferHost = "", [int]$InferPort = 8200, [string]$Weights = "auto",
      [int]$Imgsz = 960, [int]$MaxBatch = 16, [double]$Conf = 0.1, [string]$ExtraClasses = "")
$ErrorActionPreference = "Stop"
$App = Join-Path $env:USERPROFILE "argonix\argonix_app"
$Vlm = Join-Path $env:USERPROFILE "argonix_vlm"
if (-not $InferDir) { $InferDir = $App }
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) { $Uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }

function Stop-Tree($pidfile) {
  if (Test-Path $pidfile) {
    $p = (Get-Content $pidfile | Select-Object -First 1).Trim()
    # a stale pid file (process already gone) must not abort: taskkill's stderr is a terminating error in PS 5.1
    # under $ErrorActionPreference = "Stop"
    if ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue)) {
      try { & taskkill.exe /T /F /PID $p 2>&1 | Out-Null } catch { }
    }
    Remove-Item $pidfile -ErrorAction SilentlyContinue
  }
}
function Start-Detached($cmd, $cwd, $pidfile) {
  Stop-Tree $pidfile; Start-Sleep -Seconds 2
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $cwd }
  if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed ($($r.ReturnValue)): $cmd" }
  Set-Content -Path $pidfile -Value $r.ProcessId
  "started pid $($r.ProcessId) in $cwd"
}
function Probe($url) {
  try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 $url).Content } catch { "DOWN ($($_.Exception.Message))" }
}
function Infer-Host {
  if ($InferHost) { return $InferHost }
  return "127.0.0.1"  # never a network interface by default: clients use an SSH tunnel (frames are not encrypted)
}
function Start-Vlm {
  $py = Join-Path $Vlm ".venv\Scripts\python.exe"
  Start-Detached "cmd.exe /c `"set HF_HUB_OFFLINE=1&& set HF_HUB_DISABLE_XET=1&& `"$py`" vlm_server.py --port 8100 > vlm.log 2>&1`"" $Vlm (Join-Path $Vlm "vlm.pid")
}
function Start-App {
  Start-Detached "cmd.exe /c `"`"$Uv`" run --no-sync python run.py --config $Config --data $Data --host 127.0.0.1 --port $Port --vlm-url http://127.0.0.1:8100 --infer-node ws://127.0.0.1:$InferPort > app.log 2>&1`"" $App (Join-Path $App "app.pid")
}
function Start-Infer {
  $bind = Infer-Host
  # the project's own venv when it exists (no implicit `uv run` sync at service start), else uv run
  $venvPy = Join-Path $InferDir ".venv\Scripts\python.exe"
  $py = if (Test-Path $venvPy) { "`"$venvPy`"" } else { "`"$Uv`" run --no-sync python" }
  Start-Detached "cmd.exe /c `"$py -m argonix.infer_server --host $bind --port $InferPort --weights $Weights --imgsz $Imgsz --conf $Conf --max-batch $MaxBatch$(if ($ExtraClasses) { " --extra-classes $ExtraClasses" }) > infer.log 2>&1`"" $InferDir (Join-Path $InferDir "infer.pid")
  "inference node: ws://${bind}:$InferPort (log $InferDir\infer.log, token $InferDir\data\infer_token)"
}
switch ($Action) {
  "start"       { Start-Vlm; Start-App }
  "start-vlm"   { Start-Vlm }
  "stop-vlm"    { Stop-Tree (Join-Path $Vlm "vlm.pid"); "vlm stopped" }
  "stop-app"    { Stop-Tree (Join-Path $App "app.pid"); "app stopped" }
  "start-app"   { Start-App }
  "start-infer" { Start-Infer }
  "stop-infer"  { Stop-Tree (Join-Path $InferDir "infer.pid"); "inference node stopped" }
  "stop"        { Stop-Tree (Join-Path $App "app.pid"); Stop-Tree (Join-Path $Vlm "vlm.pid"); "stopped (app, vlm; the inference node has its own stop-infer)" }
  "status"      {
    "app  : " + (Probe "http://127.0.0.1:$Port/api/auth/status")   # public; /api/me needs a session
    "vlm  : " + (Probe "http://127.0.0.1:8100/health")
    "infer: " + (Probe "http://$(Infer-Host):$InferPort/health")
    & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
  }
  default { throw "unknown action $Action" }
}
