# Re-ID development helper for the Windows GPU host (dev copy in %USERPROFILE%\argonix_reid_dev; production dirs are
# only READ from). Long runs are detached via Win32_Process so they survive the SSH session.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\reid_host.ps1 <setup|env|run|status|log> [-PyArgs "..."]
param([string]$Action = "status", [string]$PyArgs = "", [string]$Log = "eval_reid.log")
$ErrorActionPreference = "Stop"
$Dev = Join-Path $env:USERPROFILE "argonix_reid_dev\argonix_app"
$Prod = Join-Path $env:USERPROFILE "argonix\argonix_app"
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) { $Uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
Set-Location $Dev
switch ($Action) {
  "setup" {  # copy the detector weights and real clips that the production copy already has (no network)
    New-Item -ItemType Directory -Force data\models, data\real\social-distance | Out-Null
    Copy-Item -Force (Join-Path $Prod "data\models\yolo26n.pt") data\models\
    Copy-Item -Force (Join-Path $Prod "data\real\vtest.avi") data\real\
    Copy-Item -Force (Join-Path $Prod "data\real\social-distance\pedestrian.mp4") data\real\social-distance\
    Get-ChildItem -Recurse data | Select-Object FullName, Length | Format-Table -AutoSize | Out-String -Width 200
  }
  "env" {  # offline: the uv cache already holds torch cu126; never download here (slow link)
    & $Uv sync --offline --extra cu126 2>&1 | Select-Object -Last 15
    & .venv\Scripts\python.exe -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  }
  "run" {
    $cmd = "cmd.exe /c `".venv\Scripts\python.exe $PyArgs > $Log 2>&1`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Dev }
    if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed ($($r.ReturnValue))" }
    Set-Content -Path "$Log.pid" -Value $r.ProcessId
    "started pid $($r.ProcessId): $PyArgs"
  }
  "status" {
    & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    Get-ChildItem *.log.pid -ErrorAction SilentlyContinue | ForEach-Object {
      $p = (Get-Content $_ | Select-Object -First 1).Trim()
      $alive = [bool](Get-Process -Id $p -ErrorAction SilentlyContinue)
      "$($_.Name): pid $p alive=$alive"
    }
  }
  "log" { if (Test-Path $Log) { Get-Content $Log -Tail 40 } else { "no $Log" } }
  default { throw "unknown action $Action" }
}
