# =============================================================================
# Phase 1 benchmark grid runner.
#
# Adapted from run_phase3_benchmark.ps1 for the phase 1 cu-smallpt baseline.
# Differences vs the phase 3 runner:
#   - Exe is build\Release\smallpt.exe (no PTX, no scene.h selection).
#   - Phase 1 CLI is just --width / --height / --spp / --output, so the
#     ablation / max-depth / tile-size machinery from phase 3 is dropped.
#   - Output schema is the same so the deck chart builder in phase 3 can
#     read this file's timings.csv as --phase1-csv directly.
#   - Memory logging via nvidia-smi kept so we have peak VRAM numbers
#     for the 4K configs (useful as a phase-1-vs-phase-3 contrast).
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase1
#   powershell -ExecutionPolicy Bypass -File .\run_phase1_benchmark.ps1
#
# Common invocations:
#   (no flags)   the 3-config quick set (~30s on a 3080 Ti)
#   -Full        the full 9-config grid for the deck (long: phase 1 is slow)
#   -Mem         single 4K_1024spp run with memory logging
#   -ConfigFilter <regex>  run only configs whose name matches the regex
#   -Tag <name>            append a tag to the run folder name
#   -SkipPng               skip PPM->PNG conversion (no Pillow needed)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$SkipPng,
    [string]$ConfigFilter = "",
    [string]$Tag = "",
    [switch]$Full,
    [switch]$Mem,
    [switch]$MemLog,
    [switch]$NoMemLog
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\smallpt.exe"
$RunsRoot    = Join-Path $ProjectRoot "results\runs"

$RunStamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$RunFolder = if ($Tag -ne "") { "${RunStamp}_$Tag" } else { $RunStamp }
$RunDir    = Join-Path $RunsRoot $RunFolder
$RendersDir  = Join-Path $RunDir "renders"
$LogsDir     = Join-Path $RunDir "logs"
$MemLogDir   = Join-Path $RunDir "memlog"
$CsvPath     = Join-Path $RunDir "timings.csv"
$RunInfoPath = Join-Path $RunDir "run_info.txt"

# -----------------------------------------------------------------------------
# Config sets (aligned with phase 3 so apples-to-apples comparisons line up)
# -----------------------------------------------------------------------------

# Quick: 3 configs. Default. Note phase 1 is much slower than phase 3, so even
# the "quick" set takes a couple of minutes at the 1024x768x256spp config.
$QuickConfigs = @(
    @{ Name = "512x384_64spp";    Width = 512;  Height = 384;  Spp = 64  },
    @{ Name = "1024x768_256spp";  Width = 1024; Height = 768;  Spp = 256 },
    @{ Name = "2048x1536_256spp"; Width = 2048; Height = 1536; Spp = 256 }
)

# Full: same 9-config grid as phase 2/3 for apples-to-apples comparison.
# WARNING: phase 1 has no BVH; the 4K x 1024spp config can take many
# minutes. Run -Full overnight if you can.
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

# Mem: single 4K high-SPP run for memory profiling.
$MemConfigs = @(
    @{ Name = "4096x3072_1024spp"; Width = 4096; Height = 3072; Spp = 1024 }
)

# Pick which set to run
if ($Mem) {
    $Configs = $MemConfigs
} elseif ($Full) {
    $Configs = $FullConfigs
} else {
    $Configs = $QuickConfigs
}

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
Write-Host "=== Phase 1 benchmark runner ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Run folder:   $RunDir"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "Build first:" -ForegroundColor Red
    Write-Host "  cmake -S . -B build -G `"Visual Studio 17 2022`" -A x64" -ForegroundColor Red
    Write-Host "  cmake --build build --config Release" -ForegroundColor Red
    exit 1
}

if (-not $SkipPng) {
    & cmd.exe /c "python -c `"from PIL import Image`" >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: python+PIL not found, using -SkipPng automatically" -ForegroundColor Yellow
        $SkipPng = $true
    }
}

# Detect whether nvidia-smi is on PATH (for the memory logger)
$NvidiaSmiAvailable = $false
try {
    & cmd.exe /c "where nvidia-smi >NUL 2>NUL"
    if ($LASTEXITCODE -eq 0) { $NvidiaSmiAvailable = $true }
} catch { $NvidiaSmiAvailable = $false }

# -----------------------------------------------------------------------------
# Output dirs + metadata
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null
New-Item -ItemType Directory -Force -Path $MemLogDir  | Out-Null

$gitHash = ""
try { $gitHash = & git -C $ProjectRoot rev-parse --short HEAD 2>$null } catch {}

$gpuName = "(unknown)"
if ($NvidiaSmiAvailable) {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
    } catch {}
}

@(
    "Run timestamp:   $RunStamp"
    "Tag:             $Tag"
    "Exe:             $ExePath"
    "Git hash:        $gitHash"
    "GPU:             $gpuName"
    "Config set:      $(if ($Mem){'mem'} elseif ($Full){'full'} else {'quick'})"
    "Configs:         $($Configs.Count)"
    "MemLog:          forced=$($MemLog), suppressed=$($NoMemLog), nvsmi=$NvidiaSmiAvailable"
) | Out-File -FilePath $RunInfoPath -Encoding ascii

# Header matches phase 3's timings.csv schema so build_charts.py can read it
# as --phase1-csv without any conversion.
"phase,scene,resolution,spp,time_ms,mrays_sec,notes,run_timestamp" | Out-File -FilePath $CsvPath -Encoding ascii

if ($ConfigFilter -ne "") {
    $Configs = $Configs | Where-Object { $_.Name -match $ConfigFilter }
    Write-Host "Filter '$ConfigFilter' matched $($Configs.Count) config(s)"
}

Write-Host "Config set:     $(if ($Mem){'mem'} elseif ($Full){'full'} else {'quick'}) ($($Configs.Count) configs)"
Write-Host ""

# -----------------------------------------------------------------------------
# Memory logger helper (background nvidia-smi sampler)
# -----------------------------------------------------------------------------
function Should-LogMem {
    param([string]$ConfigName)
    if ($NoMemLog)                { return $false }
    if (-not $NvidiaSmiAvailable) { return $false }
    if ($MemLog)                  { return $true }
    # Auto-on for any 4K config
    if ($ConfigName -match '4096') { return $true }
    return $false
}

function Start-MemLogger {
    param([string]$ConfigName, [string]$LogDir)
    $logPath = Join-Path $LogDir "${ConfigName}_nvsmi.csv"
    $errPath = Join-Path $LogDir "${ConfigName}_nvsmi.err"
    # Sample every 100ms. -lms is the correct flag for sub-second polling
    # (--loop-ms=100 is silently rejected by some nvidia-smi versions).
    $nvsmiArgs = @(
        "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-lms", "100"
    )
    $proc = Start-Process -FilePath "nvidia-smi" -ArgumentList $nvsmiArgs `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError  $errPath
    Start-Sleep -Milliseconds 250  # let nvidia-smi spin up + emit first sample
    return @{ Process = $proc; LogPath = $logPath; ErrPath = $errPath }
}

function Stop-MemLogger {
    param($Logger)
    if ($null -eq $Logger) { return 0 }
    try {
        if (-not $Logger.Process.HasExited) {
            Stop-Process -Id $Logger.Process.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {}
    if (Test-Path $Logger.ErrPath) {
        $errText = (Get-Content $Logger.ErrPath -Raw).Trim()
        if ($errText -ne "") {
            Write-Host "  nvsmi stderr:" -ForegroundColor Yellow
            $errText -split "`n" | Select-Object -First 5 | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Yellow
            }
        }
    }
    if (Test-Path $Logger.LogPath) {
        $maxMb = 0
        Get-Content $Logger.LogPath | ForEach-Object {
            $cols = $_ -split ',' | ForEach-Object { $_.Trim() }
            if ($cols.Count -ge 2) {
                $mb = 0
                if ([int]::TryParse($cols[1], [ref]$mb)) {
                    if ($mb -gt $maxMb) { $maxMb = $mb }
                }
            }
        }
        return $maxMb
    }
    return 0
}

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
        "--output", $ppmPath
    )

    @(
        "# Phase 1 benchmark: $name"
        "# Resolution: ${width}x${height}"
        "# SPP: $spp"
        "# Run: $RunStamp"
        "# GPU: $gpuName"
        "# Git: $gitHash"
        "# Command: $ExePath $($exeArgs -join ' ')"
        "# ---"
    ) | Out-File -FilePath $logPath -Encoding ascii

    # Start memory logger if this config warrants it
    $memLogger = $null
    if (Should-LogMem $name) {
        Write-Host "  nvsmi memlog enabled" -ForegroundColor DarkGray
        $memLogger = Start-MemLogger -ConfigName $name -LogDir $MemLogDir
    }

    $stdoutTmp = "$logPath.stdout.tmp"
    $stderrTmp = "$logPath.stderr.tmp"

    $proc = Start-Process -FilePath $ExePath -ArgumentList $exeArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutTmp `
        -RedirectStandardError  $stderrTmp

    # Stop memory logger before parsing logs
    $nvsmiPeakMb = 0
    if ($memLogger) {
        $nvsmiPeakMb = Stop-MemLogger $memLogger
        if ($nvsmiPeakMb -gt 0) {
            Write-Host ("  nvsmi peak: $nvsmiPeakMb MB") -ForegroundColor DarkGray
        }
    }

    Add-Content -Path $logPath -Value "## stderr"
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

    # Parse "CSV: cuda_baseline,1024x768,40,1048.43,30.00,sm_86"
    # Phase 1 uses 6 fields: phase,resolution,spp,time_ms,mrays_sec,notes
    $csvLine = Select-String -Path $logPath -Pattern "^CSV: " | Select-Object -Last 1
    if ($csvLine) {
        $payload = $csvLine.Line.Substring(5).Trim()
        $parts = $payload -split ','
        if ($parts.Count -ge 6) {
            $phase     = $parts[0]
            $res       = $parts[1]
            $sppEx     = $parts[2]
            $timeEx    = $parts[3]
            $mraysEx   = $parts[4]
            $notesEx   = $parts[5]
            # Match phase 3 timings.csv schema: phase,scene,res,spp,time,mrays,notes,run_timestamp
            "$phase,phase1_baseline,$res,$sppEx,$timeEx,$mraysEx,$notesEx,$RunStamp" |
                Out-File -FilePath $CsvPath -Append -Encoding ascii
            Write-Host "  Logged to timings.csv" -ForegroundColor Green
        } else {
            "$payload,$RunStamp" | Out-File -FilePath $CsvPath -Append -Encoding ascii
        }
    } else {
        Write-Host "  WARN: no CSV line found in exe output" -ForegroundColor Yellow
        $failures += @{ Config = $name; Reason = "no CSV line in output" }
    }

    if ($nvsmiPeakMb -gt 0) {
        Write-Host ("  Peak (nvsmi): {0} MB" -f $nvsmiPeakMb) -ForegroundColor DarkGray
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
