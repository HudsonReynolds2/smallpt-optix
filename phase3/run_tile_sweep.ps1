# =============================================================================
# Phase 3 tile-size sweep.
#
# Sweeps --tile-size at a fixed (resolution, spp) and writes a single
# tile_sweep.csv that build_tile_sweep_chart.py reads.
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase3
#   powershell -ExecutionPolicy Bypass -File .\run_tile_sweep.ps1
#
#   # Override defaults:
#   powershell -ExecutionPolicy Bypass -File .\run_tile_sweep.ps1 `
#       -Width 4096 -Height 3072 -Spp 256 `
#       -TileSizes "0,128,256,512,1024,2048,4096" -Tag deck_4k
#
# Notes:
#   - tile-size 0 means single full-image launch (phase 2 launch model).
#   - The largest tile sizes that exceed the image size will collapse to
#     a single launch (tile clamps inside the binary).
#   - Output: results\phase3_optix_triangles\tile_sweep\<timestamp>_<tag>\
# =============================================================================

[CmdletBinding()]
param(
    [int]$Width  = 4096,
    [int]$Height = 3072,
    [int]$Spp    = 256,
    [int]$MaxDepth = 20,
    [string]$TileSizes = "0,128,256,512,1024,2048,4096",
    [string]$Tag = "tile_sweep",
    [switch]$SkipPng
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\optix_smallpt.exe"
$PtxPath     = Join-Path $ProjectRoot "build\shaders.ptx"

$RunStamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$RunFolder = "${RunStamp}_$Tag"
$RunDir    = Join-Path $ProjectRoot "results\phase3_optix_triangles\tile_sweep\$RunFolder"
$RendersDir = Join-Path $RunDir "renders"
$LogsDir    = Join-Path $RunDir "logs"
$CsvPath    = Join-Path $RunDir "tile_sweep.csv"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "Build with --tile-size support first." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null

Write-Host "=== Tile-size sweep ===" -ForegroundColor Cyan
Write-Host "Resolution: ${Width}x${Height}"
Write-Host "SPP:        $Spp"
Write-Host "MaxDepth:   $MaxDepth"
Write-Host "TileSizes:  $TileSizes"
Write-Host "Output:     $RunDir"
Write-Host ""

"tile_size,resolution,spp,tile_w,tile_h,n_tiles,time_ms,mrays_sec,peak_mb,run_timestamp" |
    Out-File -FilePath $CsvPath -Encoding ascii

$tileList = $TileSizes -split ',' | ForEach-Object { [int]$_.Trim() }

foreach ($ts in $tileList) {
    $name = "${Width}x${Height}_${Spp}spp_t${ts}"
    Write-Host "--- tile-size=$ts ---" -ForegroundColor Yellow

    $ppmPath = Join-Path $RendersDir "$name.ppm"
    $pngPath = Join-Path $RendersDir "$name.png"
    $logPath = Join-Path $LogsDir    "$name.log"

    $exeArgs = @(
        "--width",      $Width,
        "--height",     $Height,
        "--spp",        $Spp,
        "--max-depth",  $MaxDepth,
        "--tile-size",  $ts,
        "--output",     $ppmPath,
        "--ptx",        $PtxPath
    )

    $stdoutTmp = "$logPath.stdout.tmp"
    $stderrTmp = "$logPath.stderr.tmp"

    $cfgStart = Get-Date
    $proc = Start-Process -FilePath $ExePath -ArgumentList $exeArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutTmp `
        -RedirectStandardError  $stderrTmp
    $cfgElapsed = (Get-Date) - $cfgStart

    if (Test-Path $stderrTmp) { Get-Content $stderrTmp | Add-Content -Path $logPath }
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | Add-Content -Path $logPath }
    if (Test-Path $stderrTmp) { Get-Content $stderrTmp | ForEach-Object { Write-Host "  [err] $_" } }
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | ForEach-Object { Write-Host "  $_" } }
    Remove-Item $stdoutTmp,$stderrTmp -ErrorAction SilentlyContinue

    if ($proc.ExitCode -ne 0) {
        Write-Host "  FAILED: exit $($proc.ExitCode)" -ForegroundColor Red
        # Record the failure as a row with peak_mb=-1 so the chart can
        # mark it as "OOM / failed" rather than be silently missing.
        "$ts,${Width}x${Height},$Spp,0,0,0,0,0,-1,$RunStamp" |
            Out-File -FilePath $CsvPath -Append -Encoding ascii
        continue
    }

    $sweepLine = Select-String -Path $logPath -Pattern "^TILE_SWEEP: " | Select-Object -Last 1
    if ($sweepLine) {
        $payload = $sweepLine.Line.Substring(12).Trim()
        # payload = res,spp,tile_w,tile_h,n_tiles,time_ms,mrays_sec,peak_mb
        "$ts,$payload,$RunStamp" | Out-File -FilePath $CsvPath -Append -Encoding ascii
        Write-Host "  Logged" -ForegroundColor Green
    } else {
        Write-Host "  WARN: no TILE_SWEEP line found" -ForegroundColor Yellow
    }

    if (-not $SkipPng -and (Test-Path $ppmPath)) {
        $pyCmd = "from PIL import Image; Image.open(r'$ppmPath').save(r'$pngPath')"
        $pyCmdEscaped = $pyCmd.Replace('"', '\"')
        & cmd.exe /c "python -c `"$pyCmdEscaped`" >NUL 2>NUL"
    }

    Write-Host ("  Wall time: {0:N1}s" -f $cfgElapsed.TotalSeconds)
    Write-Host ""
}

Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "CSV: $CsvPath"
Write-Host ""
Get-Content $CsvPath | ForEach-Object { Write-Host "  $_" }
