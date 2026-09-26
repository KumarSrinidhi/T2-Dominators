# RTSP replay harness (manual section 18) for a Windows host: MediaMTX + ffmpeg publish dataset clips as looping RTSP
# "cameras" -- <name> (main stream, original resolution) and <name>_sub (downscaled sub-stream) -- plus one stream
# with deliberate faults (periodic drop/restart and a frozen picture) to exercise Argus's reconnect and health
# checks. Everything binds to 127.0.0.1 by default; no firewall or system settings are touched.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\rtsp_replay.ps1 <install|start|stop|status|restart|url>
#     [-Streams tools\rtsp_streams.json] [-Datasets %USERPROFILE%\argonix_datasets] [-Runtime %USERPROFILE%\argonix_rtsp]
#     [-Address 127.0.0.1:8554] [-NoAuth] [-Ffmpeg <ffmpeg.exe>]
#
# install : download the pinned MediaMTX release (official GitHub, bluenviron/mediamtx, MIT) and verify its SHA-256.
# start   : generate the viewer password on first use (<Runtime>\secrets.json, never committed), render
#           tools\mediamtx.yml into <Runtime>, start MediaMTX and one publisher worker per stream.
# url     : print the viewer URL template (password masked unless -ShowPassword).
# Processes are created via Win32_Process so they outlive the SSH session that started them (same pattern as
# win_services.ps1); they do not survive a reboot. Publisher workers restart ffmpeg if it exits.
param([string]$Action = "status", [string]$Streams = "", [string]$Datasets = "", [string]$Runtime = "",
      [string]$Address = "127.0.0.1:8554", [switch]$NoAuth, [string]$Ffmpeg = "", [switch]$ShowPassword,
      [string]$MediamtxVersion = "v1.21.1",
      # worker arguments (internal: used by `start` to launch one publisher per stream)
      [string]$Name = "", [string]$File = "", [int]$SubWidth = 640, [string]$Main = "copy", [switch]$Gray,
      [double]$UpS = 0, [double]$DownS = 20, [double]$FreezeAtS = 0, [double]$FreezeS = 0, [double]$Fps = 25)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Split-Path -Parent $Here
if (-not $Streams) { $Streams = Join-Path $Here "rtsp_streams.json" }
if (-not $Datasets) { $Datasets = Join-Path $env:USERPROFILE "argonix_datasets" }
if (-not $Runtime) { $Runtime = Join-Path $env:USERPROFILE "argonix_rtsp" }
$PidDir = Join-Path $Runtime "pids"; $LogDir = Join-Path $Runtime "logs"
$MtxDir = Join-Path $Runtime "mediamtx"; $MtxExe = Join-Path $MtxDir "mediamtx.exe"
$Secrets = Join-Path $Runtime "secrets.json"
$ApiBase = "http://127.0.0.1:9997"
New-Item -ItemType Directory -Force $Runtime, $PidDir, $LogDir | Out-Null
$Port = [int]($Address.Split(":")[-1])
$PubBase = "rtsp://127.0.0.1:$Port"   # publishers are accepted from localhost only (see tools\mediamtx.yml)

function Find-Ffmpeg {
  if ($Ffmpeg) { return $Ffmpeg }
  $c = Get-Command ffmpeg -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  $py = Join-Path $AppRoot ".venv\Scripts\python.exe"
  if (Test-Path $py) {
    $p = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    if ($p) { return $p.Trim() }
  }
  throw "ffmpeg not found: pass -Ffmpeg <path> (official BtbN/gyan build) or install imageio-ffmpeg in the app venv"
}

function Stop-Tree($pidfile) {
  if (Test-Path $pidfile) {
    $p = (Get-Content $pidfile | Select-Object -First 1)
    # via cmd so taskkill's stderr (e.g. a child that already exited) cannot become a terminating PowerShell error
    if ($p) { & cmd.exe /c "taskkill /T /F /PID $($p.Trim()) >nul 2>&1" }
    Remove-Item $pidfile -ErrorAction SilentlyContinue
  }
}

function Start-Detached($name, $cmd, $cwd) {
  $pidfile = Join-Path $PidDir "$name.pid"
  Stop-Tree $pidfile
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $cwd }
  if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed ($($r.ReturnValue)) for $name" }
  Set-Content -Path $pidfile -Value $r.ProcessId
  "started $name (pid $($r.ProcessId))"
}

function Install-Mediamtx {
  if (Test-Path $MtxExe) { "mediamtx already installed: $MtxExe"; return }
  $zip = "mediamtx_$($MediamtxVersion)_windows_amd64.zip"
  $base = "https://github.com/bluenviron/mediamtx/releases/download/$MediamtxVersion"
  $tmp = Join-Path $Runtime $zip
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $ProgressPreference = "SilentlyContinue"
  Invoke-WebRequest -UseBasicParsing "$base/$zip" -OutFile $tmp
  $sums = (Invoke-WebRequest -UseBasicParsing "$base/checksums.sha256").Content
  if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
  $line = $sums -split "`n" | Where-Object { $_ -match [regex]::Escape($zip) } | Select-Object -First 1
  $want = if ($line) { ($line.Trim() -split "\s+")[0].ToLower() } else { "" }
  $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
  if (-not $want -or $got -ne $want) { Remove-Item $tmp; throw "checksum mismatch for $zip (got $got, want '$want')" }
  Expand-Archive -Force $tmp $MtxDir
  Remove-Item $tmp
  "installed mediamtx $MediamtxVersion (sha256 $got) -> $MtxExe"
}

function Get-Secrets {
  if (Test-Path $Secrets) { return (Get-Content $Secrets -Raw | ConvertFrom-Json) }
  $chars = [char[]]"ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
  $bytes = New-Object byte[] 24
  (New-Object System.Security.Cryptography.RNGCryptoServiceProvider).GetBytes($bytes)
  $pw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
  $s = [pscustomobject]@{ user = "viewer"; password = $pw; created = (Get-Date).ToString("s");
                          note = "RTSP replay harness viewer account (tools/rtsp_replay.ps1). Local test secret; do not commit." }
  [IO.File]::WriteAllText($Secrets, ($s | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
  "generated a new viewer password in $Secrets" | Write-Host
  return $s
}

function Write-Config($s) {
  $t = [IO.File]::ReadAllText((Join-Path $Here "mediamtx.yml"))
  $anon = "# anonymous read disabled (viewer account required)"
  if ($NoAuth) { $anon = "- user: any`n    pass:`n    ips: []`n    permissions:`n      - action: read" }
  $t = $t.Replace("__VIEWER_PASS__", $s.password).Replace("__RTSP_ADDRESS__", $Address)
  $t = $t.Replace("__LOG_FILE__", (Join-Path $LogDir "mediamtx.log").Replace("\", "/"))
  $t = $t.Replace("# __ANON_READ__ (rtsp_replay.ps1 -NoAuth turns this into anonymous read access)", $anon)
  $out = Join-Path $Runtime "mediamtx.yml"
  [IO.File]::WriteAllText($out, $t, (New-Object Text.UTF8Encoding $false))
  return $out
}

function Resolve-Clip($f) {
  if ([IO.Path]::IsPathRooted($f)) { return $f }
  foreach ($b in @($Datasets, $AppRoot)) { $p = Join-Path $b $f; if (Test-Path $p) { return (Resolve-Path $p).Path } }
  throw "clip not found: $f (looked in $Datasets and $AppRoot)"
}

function Wait-Port($port, $seconds) {
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $port); $c.Close(); return $true } catch { Start-Sleep -Milliseconds 300 } finally { $c.Dispose() }
  }
  return $false
}

function Start-All {
  if (-not (Test-Path $MtxExe)) { Install-Mediamtx }
  $s = Get-Secrets
  $cfg = Write-Config $s
  $ff = Find-Ffmpeg
  Start-Detached "mediamtx" "`"$MtxExe`" `"$cfg`"" $MtxDir
  if (-not (Wait-Port $Port 15)) { throw "MediaMTX did not open port $Port; see $LogDir\mediamtx.log" }
  $spec = Get-Content $Streams -Raw | ConvertFrom-Json
  $ps1 = $PSCommandPath
  if (-not $ps1) { $ps1 = Join-Path $Here "rtsp_replay.ps1" }
  $common = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$ps1`" worker -Runtime `"$Runtime`" -Address $Address -Ffmpeg `"$ff`""
  foreach ($st in $spec.streams) {
    $clip = Resolve-Clip $st.file
    $sw = if ($st.sub_width) { [int]$st.sub_width } else { 640 }
    $mm = if ($st.main) { $st.main } else { "copy" }
    $g = if ($st.gray) { " -Gray" } else { "" }
    Start-Detached "pub_$($st.name)" "$common -Name $($st.name) -File `"$clip`" -SubWidth $sw -Main $mm$g" $Runtime
  }
  if ($spec.fault) {
    $f = $spec.fault
    $clip = Resolve-Clip $f.file
    Start-Detached "pub_$($f.name)" ("$common -Name $($f.name) -File `"$clip`" -SubWidth $([int]$f.sub_width) -Main none " +
      "-UpS $($f.up_s) -DownS $($f.down_s) -FreezeAtS $($f.freeze_at_s) -FreezeS $($f.freeze_s) -Fps $($f.fps)") $Runtime
  }
  "RTSP replay running on rtsp://$Address  (read with user '$($s.user)', password in $Secrets)"
}

# One publisher: loops a clip forever (UpS = 0) or, for the fault stream, publishes for UpS seconds (with a frozen
# stretch from FreezeAtS for FreezeS seconds), disappears for DownS seconds, and starts again.
function Run-Worker {
  $ErrorActionPreference = "Continue"
  $ev = Join-Path $LogDir "$Name.events.log"
  # -Gray: a black-and-white / IR-night camera -- luma only, sent as ordinary yuv420p H.264 with neutral chroma
  $gf = if ($Gray) { ",format=gray,format=yuv420p" } else { "" }
  if ($Gray -and $Main -eq "copy") { $Main = "encode" }
  $x264 = "-c:v libx264 -preset veryfast -tune zerolatency -profile:v main -pix_fmt yuv420p -bf 0 " +
          "-force_key_frames expr:gte(t,n_forced*2) -b:v 800k -maxrate 1200k -bufsize 2400k"
  while ($true) {
    if ($UpS -gt 0) {
      # freezeframes needs the replacement frame on a second input: split the stream and freeze frames a..b to a-1
      # (1:1 frame mapping, so -re pacing and timestamps are unchanged; the picture just stops changing)
      $fc = "[0:v]scale=$($SubWidth):-2$gf[v]"
      if ($FreezeS -gt 0) {
        $a = [int]($FreezeAtS * $Fps); $b = [int](($FreezeAtS + $FreezeS) * $Fps)
        $fc = "[0:v]scale=$($SubWidth):-2$gf,split[s0][s1];[s0][s1]freezeframes=first=$($a):last=$($b):replace=$($a - 1)[v]"
      }
      $ffargs = "-hide_banner -loglevel warning -nostdin -re -stream_loop -1 -i `"$File`" -an -t $UpS -filter_complex $fc " +
                "-map [v] $x264 -f rtsp -rtsp_transport tcp $PubBase/$($Name)_sub"
    } else {
      $mainOut = ""
      if ($Main -eq "copy") { $mainOut = "-map 0:v:0 -c:v copy -f rtsp -rtsp_transport tcp $PubBase/$Name " }
      elseif ($Main -eq "encode") {
        $mvf = if ($Gray) { "-vf format=gray,format=yuv420p " } else { "" }
        $mainOut = "-map 0:v:0 $mvf$x264 -b:v 4M -maxrate 6M -bufsize 8M -f rtsp -rtsp_transport tcp $PubBase/$Name "
      }
      $ffargs = "-hide_banner -loglevel warning -nostdin -re -stream_loop -1 -i `"$File`" -an $mainOut" +
                "-map 0:v:0 -vf scale=$($SubWidth):-2$gf $x264 -f rtsp -rtsp_transport tcp $PubBase/$($Name)_sub"
    }
    $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    Add-Content $ev "up $t0"
    $p = Start-Process -FilePath $Ffmpeg -ArgumentList $ffargs -NoNewWindow -PassThru -Wait `
         -RedirectStandardError (Join-Path $LogDir "$Name.ffmpeg.log") -RedirectStandardOutput (Join-Path $LogDir "$Name.out.log")
    $t1 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    Add-Content $ev "down $t1 exit=$($p.ExitCode)"
    if ($UpS -gt 0) { Start-Sleep -Milliseconds ([int]($DownS * 1000)) } else { Start-Sleep -Seconds 2 }
  }
}

function Show-Status {
  foreach ($f in Get-ChildItem $PidDir -Filter *.pid -ErrorAction SilentlyContinue) {
    $p = (Get-Content $f.FullName | Select-Object -First 1).Trim()
    $alive = [bool](Get-Process -Id $p -ErrorAction SilentlyContinue)
    "{0,-24} pid {1,-7} {2}" -f $f.BaseName, $p, $(if ($alive) { "running" } else { "DEAD" })
  }
  try {
    $r = Invoke-RestMethod -TimeoutSec 5 "$ApiBase/v3/paths/list?itemsPerPage=200"
    foreach ($i in $r.items) {
      "{0,-26} ready={1,-5} readers={2} tracks={3} rx={4:N1} MB" -f $i.name, $i.ready, @($i.readers).Count,
        ($i.tracks -join ","), ($i.bytesReceived / 1MB)
    }
  } catch { "mediamtx api: DOWN ($($_.Exception.Message))" }
}

switch ($Action) {
  "install" { Install-Mediamtx }
  "start"   { Start-All }
  "restart" { foreach ($f in Get-ChildItem $PidDir -Filter *.pid) { Stop-Tree $f.FullName }; Start-Sleep 2; Start-All }
  "stop"    {
    foreach ($f in Get-ChildItem $PidDir -Filter "pub_*.pid") { Stop-Tree $f.FullName }
    Stop-Tree (Join-Path $PidDir "mediamtx.pid"); "stopped"
  }
  "status"  { Show-Status }
  "url"     {
    $s = Get-Secrets
    $pw = if ($ShowPassword) { $s.password } else { "***" }
    "rtsp://$($s.user):$pw@$Address/<name>_sub"
  }
  "worker"  { Run-Worker }
  default   { throw "unknown action $Action" }
}
