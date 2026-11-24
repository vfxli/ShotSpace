# ShotSpace installer bootstrap (Windows): fetch uv and Python into .tools, then hand over to install/wizard.py.
# Called by setup.bat.  Nothing outside the ShotSpace folder is written: uv's cache, its Pythons and every temporary
# file live under .tools; Pythons are installed with --no-registry --no-bin (no registry keys, nothing on PATH).
# "Continue": in Windows PowerShell 5.1 a native program's stderr (uv prints its progress there) would otherwise
# stop the script; every native call is checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root ".tools"
$UvDir = Join-Path $Tools "uv"
$Uv = Join-Path $UvDir "uv.exe"
$Tmp = Join-Path $Tools "tmp"

$UvVersion = "0.13.7"
$UvWheelPath = "packages/8e/4f/041b59d2a38807a3e31ef41064cf3e45d2752fe0409d89040d52778049ae/uv-0.13.7-py3-none-win_amd64.whl"
$UvWheelSha256 = "2835e2ab397d0db86323b9975dce75ecd76aa9cfa0d6a2aea84734aae7f8c9b3"
$UvWheelHosts = @(
  "https://files.pythonhosted.org",
  "https://mirrors.aliyun.com/pypi",
  "https://pypi.tuna.tsinghua.edu.cn",
  "https://mirrors.ustc.edu.cn/pypi"
)
$PythonMirrors = @("", "https://registry.npmmirror.com/-/binary/python-build-standalone")

New-Item -ItemType Directory -Force -Path $UvDir, $Tmp -ErrorAction Stop | Out-Null
$env:UV_CACHE_DIR = Join-Path $Tools "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_MANAGED_PYTHON = "1"
$env:UV_NO_CONFIG = "1"
$env:TEMP = $Tmp
$env:TMP = $Tmp

function Get-UvVersion {
  if (-not (Test-Path $Uv)) { return "" }
  try { return ((& $Uv --version) -split " ")[1] } catch { return "" }
}

if ((Get-UvVersion) -ne $UvVersion) {
  Write-Host "==> uv $UvVersion"
  $whl = Join-Path $Tmp "uv-$UvVersion.zip"
  $ok = $false
  # fastest source first: time a tiny piece of the file from each
  $curl = Join-Path $env:SystemRoot "System32\curl.exe"
  $hasCurl = Test-Path $curl
  $ranked = $UvWheelHosts | ForEach-Object {
    $t = 999
    if ($hasCurl) {
      $r = & $curl -s -o NUL -r 0-65535 -w "%{http_code} %{time_total}" --connect-timeout 8 --max-time 15 "$_/$UvWheelPath"
      if ($r -match "^2\d\d (\S+)") { $t = [double]$Matches[1] }
    } else {
      $sw = [Diagnostics.Stopwatch]::StartNew()
      try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 15 -Uri "$_/$UvWheelPath" -Headers @{ Range = "bytes=0-65535" } -ErrorAction Stop | Out-Null; $t = $sw.Elapsed.TotalSeconds } catch { }
    }
    [pscustomobject]@{ Host = $_; T = $t }
  } | Sort-Object T | ForEach-Object { $_.Host }
  foreach ($h in $ranked) {
    Write-Host "    $h"
    try {
      if ($hasCurl) {
        # a source slower than 100 KB/s for 20 s is dropped for the next one
        & $curl -fL --retry 2 --connect-timeout 15 --speed-limit 102400 --speed-time 20 --max-time 900 -s -o $whl "$h/$UvWheelPath"
        if ($LASTEXITCODE -ne 0) { throw "curl exit $LASTEXITCODE" }
      } else {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -Uri "$h/$UvWheelPath" -OutFile $whl -ErrorAction Stop
      }
      if ((Get-FileHash -Algorithm SHA256 $whl).Hash.ToLower() -eq $UvWheelSha256) { $ok = $true; break }
      Write-Host "    checksum mismatch, trying the next source"
    } catch { Write-Host "    failed: $($_.Exception.Message)" }
  }
  if (-not $ok) { Write-Host "Could not download uv from any source. / 所有下载源都失败了，请检查网络后重新运行 setup.bat"; exit 1 }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [IO.Compression.ZipFile]::OpenRead($whl)
  try {
    $entry = $zip.Entries | Where-Object { $_.FullName -eq "uv-$UvVersion.data/scripts/uv.exe" }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Uv, $true)
  } finally { $zip.Dispose() }
  Remove-Item -Force $whl
}

& $Uv python find 3.12 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "==> Python 3.12"
  $ok = $false
  # order: as --region says, else the source that answers first
  $region = ""
  for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--region" -and $i + 1 -lt $args.Count) { $region = $args[$i + 1] }
    elseif ("$($args[$i])" -like "--region=*") { $region = "$($args[$i])".Substring(9) }
  }
  function Time-Url($u) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 15 -Uri $u -ErrorAction Stop | Out-Null; return $sw.Elapsed.TotalSeconds } catch { return 999 }
  }
  $cnFirst = $region -eq "cn"
  if ($region -ne "cn" -and $region -ne "global") {
    $cnFirst = (Time-Url "$($PythonMirrors[1])/") -lt (Time-Url "https://github.com/astral-sh/python-build-standalone/releases")
  }
  if ($cnFirst) { $PythonMirrors = @($PythonMirrors[1], "") }
  foreach ($m in $PythonMirrors) {
    if ($m) { Write-Host "    $m"; & $Uv python install 3.12 --no-bin --no-registry --mirror $m }
    else { Write-Host "    github.com"; & $Uv python install 3.12 --no-bin --no-registry }
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  }
  if (-not $ok) { Write-Host "Could not download Python. / Python 下载失败，请检查网络后重新运行 setup.bat"; exit 1 }
}
$Py = (& $Uv python find 3.12).Trim()

& $Py -I (Join-Path $Root "install\wizard.py") --uv $Uv @args
exit $LASTEXITCODE
