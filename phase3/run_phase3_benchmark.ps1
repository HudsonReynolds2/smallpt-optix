# =============================================================================
# Phase 3 benchmark grid runner
#
# Adapted from run_phase2_benchmark.ps1. Key differences vs phase 2:
#   - Points at phase3\build\ for exe + PTX
#   - Results land in results\phase3_optix_triangles\runs\<timestamp>\
#   - CSV phase tag is "optix_phase3" (matches the CSV line the exe emits)
#   - Default config grid is the same quick+full sets; 4K high-SPP is separate
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase3
#   powershell -ExecutionPolicy Bypass -File .\run_phase3_benchmark.ps1
#
# Options:
#   -SkipPng       Don't convert PPM -> PNG (faster if you just want timings)
#   -ConfigFilter  Regex; only run configs whose name matches (e.g. "1024x768")
#   -Tag           Optional label appended to the run folder name
#   -Full          Run the full 9-config grid instead of the 3-config quick set
#   -Mem           Run only the 4K config (memory stress test)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$SkipPng,
    [string]$ConfigFilter = "",
    [string]$Tag = "",
    [switch]$Full,
    [switch]$Mem
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\optix_smallpt.exe"
$PtxPath     = Join-Path $ProjectRoot "build\shaders.ptx"
$SceneHPath  = Join-Path $ProjectRoot "src\scene.h"
$RunsRoot    = Join-Path $ProjectRoot "results\phase3_optix_triangles\runs"

$RunStamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$RunFolder = if ($Tag -ne "") { "${RunStamp}_$Tag" } else { $RunStamp }
$RunDir    = Join-Path $RunsRoot $RunFolder
$RendersDir = Join-Path $RunDir "renders"
$LogsDir    = Join-Path $RunDir "logs"
$CsvPath    = Join-Path $RunDir "timings.csv"
$RunInfoPath = Join-Path $RunDir "run_info.txt"

# -----------------------------------------------------------------------------
# Config sets
# -----------------------------------------------------------------------------

# Quick: 3 configs, finishes in < 60 s. Default.
$QuickConfigs = @(
    @{ Name = "512x384_64spp";    Width = 512;  Height = 384;  Spp = 64  },
    @{ Name = "1024x768_256spp";  Width = 1024; Height = 768;  Spp = 256 },
    @{ Name = "2048x1536_256spp"; Width = 2048; Height = 1536; Spp = 256 }
)

# Full: same 9-config grid as phase 2 for apples-to-apples comparison.
$FullConfigs = @(
    @{ Name = "512x384_256spp";    Width = 512;  Height = 384;  Spp = 256  },
    @{ Name = "512x384_1024spp";   Width = 512;  Height = 384;  Spp = 1024 },
    @{ Name = "1024x768_256spp";   Width = 1024; Height = 768;  Spp = 256  },
    @{ Name = "1024x768_1024spp";  Width = 1024; Height = 768;  Spp = 1024 },
    @{ Name = "1024x768_4096spp";  Width = 1024; Height = 768;  Spp = 4096 },
    @{ Name = "2048x1536_256spp";  Width = 2048; Height = 1536; Spp = 256  },
    @{ Name = "2048x1536_1024spp"; Width = 2048; Height = 1536; Spp = 1024 },
    @{ Name = "4096x3072_256spp";  Width = 4096; Height = 3072; Spp = 256  },
    @{ Name = "4096x3072_1024spp"; Width = 4096; Height = 3072; Spp = 1024 }
)

# Mem: single 4K high-SPP run to verify tile launches keep memory bounded.
$MemConfigs = @(
    @{ Name = "4096x3072_1024spp"; Width = 4096; Height = 3072; Spp = 1024 }
)

# Pick which set to run
$Configs = if ($Mem)  { $MemConfigs  }
      elseif ($Full)  { $FullConfigs }
      else            { $QuickConfigs }

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
Write-Host "=== Phase 3 benchmark runner ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Run folder:   $RunDir"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "Build first:  .\select_scene.ps1 -Name default" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $PtxPath)) {
    Write-Host "ERROR: PTX not found at $PtxPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $SceneHPath)) {
    Write-Host "ERROR: scene.h not found at $SceneHPath" -ForegroundColor Red
    Write-Host "Run .\select_scene.ps1 -Name default first." -ForegroundColor Red
    exit 1
}

$sceneNameLine = Select-String -Path $SceneHPath -Pattern '^//\s*SCENE_NAME:\s*(.*)$' | Select-Object -First 1
$ActiveScene = if ($sceneNameLine) { $sceneNameLine.Matches[0].Groups[1].Value.Trim() } else { "(unknown)" }
Write-Host "Active scene: $ActiveScene"

if (-not $SkipPng) {
    & cmd.exe /c "python -c `"from PIL import Image`" >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: python+PIL not found, using -SkipPng automatically" -ForegroundColor Yellow
        $SkipPng = $true
    }
}

# -----------------------------------------------------------------------------
# Output dirs + metadata
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null

Copy-Item -Path $SceneHPath -Destination (Join-Path $RunDir "scene.h.snapshot") -Force

$gitHash = ""
try { $gitHash = & git -C $ProjectRoot rev-parse --short HEAD 2>$null } catch {}

@(
    "Run timestamp: $RunStamp"
    "Active scene:  $ActiveScene"
    "Tag:           $Tag"
    "Exe:           $ExePath"
    "PTX:           $PtxPath"
    "Git hash:      $gitHash"
    "Config set:    $(if ($Mem) {'mem'} elseif ($Full) {'full'} else {'quick'})"
    "Configs:       $($Configs.Count)"
) | Out-File -FilePath $RunInfoPath -Encoding ascii

"phase,scene,resolution,spp,time_ms,mrays_sec,notes,run_timestamp" | Out-File -FilePath $CsvPath -Encoding ascii

if ($ConfigFilter -ne "") {
    $Configs = $Configs | Where-Object { $_.Name -match $ConfigFilter }
    Write-Host "Filter '$ConfigFilter' matched $($Configs.Count) config(s)"
}

Write-Host "Config set:     $(if ($Mem) {'mem'} elseif ($Full) {'full'} else {'quick'}) ($($Configs.Count) configs)"
Write-Host ""

# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------
$grandStart = Get-Date
$failures   = @()

foreach ($cfg in $Configs) {
    $name   = $cfg.Name
    $width  = $cfg.Width
    $height = $cfg.Height
    $spp    = $cfg.Spp

    $ppmPath = Join-Path $RendersDir "$name.ppm"
    $pngPath = Join-Path $RendersDir "$name.png"
    $logPath = Join-Path $LogsDir    "$name.log"

    Write-Host ("--- $name ({0}x{1}, $spp spp) ---" -f $width, $height) -ForegroundColor Yellow

    $cfgStart = Get-Date

    $exeArgs = @(
        "--width",  $width,
        "--height", $height,
        "--spp",    $spp,
        "--output", $ppmPath,
        "--ptx",    $PtxPath
    )

    @(
        "# Phase 3 benchmark: $name"
        "# Resolution: ${width}x${height}"
        "# SPP: $spp"
        "# Run: $RunStamp"
        "# Scene: $ActiveScene"
        "# Git: $gitHash"
        "# Command: $ExePath $($exeArgs -join ' ')"
        "# ---"
    ) | Out-File -FilePath $logPath -Encoding ascii

    $stdoutTmp = "$logPath.stdout.tmp"
    $stderrTmp = "$logPath.stderr.tmp"

    $proc = Start-Process -FilePath $ExePath -ArgumentList $exeArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutTmp `
        -RedirectStandardError  $stderrTmp

    Add-Content -Path $logPath -Value "## stderr (OptiX log + diagnostics)"
    if (Test-Path $stderrTmp) { Get-Content $stderrTmp | Add-Content -Path $logPath }
    Add-Content -Path $logPath -Value "## stdout (program output)"
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | Add-Content -Path $logPath }

    if (Test-Path $stderrTmp) { Get-Content $stderrTmp | ForEach-Object { Write-Host "  [err] $_" } }
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | ForEach-Object { Write-Host "  $_" } }

    Remove-Item $stdoutTmp,$stderrTmp -ErrorAction SilentlyContinue

    if ($proc.ExitCode -ne 0) {
        Write-Host "  FAILED: exit code $($proc.ExitCode) (see $logPath)" -ForegroundColor Red
        $failures += @{ Config = $name; Reason = "exit code $($proc.ExitCode)" }
        continue
    }

    # Parse "CSV: optix_phase3,1024x768,256,499.99,471.86,sm_86"
    $csvLine = Select-String -Path $logPath -Pattern "^CSV: " | Select-Object -Last 1
    if ($csvLine) {
        $payload = $csvLine.Line.Substring(5).Trim()
        $parts = $payload -split ','
        if ($parts.Count -ge 5) {
            $phase = $parts[0]
            $rest  = $parts[1..($parts.Count-1)] -join ','
            "$phase,$ActiveScene,$rest,$RunStamp" | Out-File -FilePath $CsvPath -Append -Encoding ascii
            Write-Host "  Logged to timings.csv" -ForegroundColor Green
        } else {
            "$payload,$RunStamp" | Out-File -FilePath $CsvPath -Append -Encoding ascii
            Write-Host "  Logged to timings.csv (unparsed)" -ForegroundColor Green
        }
    } else {
        Write-Host "  WARN: no CSV line found in exe output" -ForegroundColor Yellow
        $failures += @{ Config = $name; Reason = "no CSV line in output" }
    }

    if (-not $SkipPng) {
        if (Test-Path $ppmPath) {
            $pyCmd = "from PIL import Image; Image.open(r'$ppmPath').save(r'$pngPath')"
            $pyCmdEscaped = $pyCmd.Replace('"', '\"')
            & cmd.exe /c "python -c `"$pyCmdEscaped`" >NUL 2>NUL"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  PNG: $pngPath" -ForegroundColor Green
            } else {
                Write-Host "  PNG conversion failed (exit $LASTEXITCODE)" -ForegroundColor Yellow
                $failures += @{ Config = $name; Reason = "PNG conversion exit $LASTEXITCODE" }
            }
        } else {
            Write-Host "  WARN: PPM not produced at $ppmPath" -ForegroundColor Yellow
            $failures += @{ Config = $name; Reason = "PPM not produced" }
        }
    }

    $cfgElapsed = (Get-Date) - $cfgStart
    Write-Host ("  Wall time: {0:N1}s" -f $cfgElapsed.TotalSeconds)
    Write-Host ""
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
$grandElapsed = (Get-Date) - $grandStart
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host ("Total wall time: {0:N1}s" -f $grandElapsed.TotalSeconds)
Write-Host "Run folder: $RunDir"

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures ($($failures.Count)):" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red
    }
    exit 1
}

Write-Host ""
Write-Host "Timings:" -ForegroundColor Cyan
Get-Content $CsvPath | ForEach-Object { Write-Host "  $_" }
