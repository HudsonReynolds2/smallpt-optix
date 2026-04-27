# =============================================================================
# Phase 3 benchmark grid runner (v3, deck-prep edition)
#
# Adapted from run_phase2_benchmark.ps1. Differences from the v2 phase 3 runner:
#   - -MaxDepth N         : pass --max-depth to the exe (uses default 20 if unset)
#   - -Ablation           : sweep depth in {2,4,6,8,12,20} at the headline
#                            config (1024x768x1024spp). Writes ablation.csv.
#   - -AblationDepths     : override the depth sweep list (comma-separated)
#   - -AblationConfig     : override which (W,H,SPP) the sweep runs at
#   - -MemLog             : enable nvidia-smi memory logger for 4K configs
#                            (auto-on for any 4096-named config; -MemLog forces on
#                             everywhere, -NoMemLog forces off)
#   - PHASE3_EXT line is parsed if present and adds depth + peak_mb to timings
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase3
#   powershell -ExecutionPolicy Bypass -File .\run_phase3_benchmark.ps1
#
# Common invocations:
#   -Full        run the 9-config grid for the deck
#   -Mem         single 4K_1024spp run with memory logging
#   -Ablation    sweep --max-depth at 1024x768x1024spp
# =============================================================================

[CmdletBinding()]
param(
    [switch]$SkipPng,
    [string]$ConfigFilter = "",
    [string]$Tag = "",
    [switch]$Full,
    [switch]$Mem,
    [switch]$Ablation,
    [int]$MaxDepth = 0,
    [string]$AblationDepths = "2,4,6,8,12,20",
    [string]$AblationConfig = "1024x768_1024spp",
    [switch]$MemLog,
    [switch]$NoMemLog,
    [int]$TileSize = -1
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
$MemLogDir  = Join-Path $RunDir "memlog"
$CsvPath    = Join-Path $RunDir "timings.csv"
$ExtCsvPath = Join-Path $RunDir "timings_ext.csv"
$AblationCsvPath = Join-Path $RunDir "ablation.csv"
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

# Ablation: parse the configs. The depth sweep itself is per-depth.
function Parse-AblationConfig {
    param([string]$Spec)
    # "1024x768_1024spp" -> Width=1024 Height=768 Spp=1024
    if ($Spec -match '^(\d+)x(\d+)_(\d+)spp$') {
        return @{
            Name   = $Spec
            Width  = [int]$Matches[1]
            Height = [int]$Matches[2]
            Spp    = [int]$Matches[3]
        }
    }
    Write-Host "ERROR: bad -AblationConfig '$Spec' (expected like '1024x768_1024spp')" -ForegroundColor Red
    exit 1
}

# Pick which set to run
if ($Ablation) {
    # Ablation is special: same config repeated with varying --max-depth.
    $AblConfig = Parse-AblationConfig $AblationConfig
    $depths    = $AblationDepths -split ',' | ForEach-Object { [int]$_.Trim() }
    $Configs = @()
    foreach ($d in $depths) {
        $cfgCopy = @{
            Name      = "$($AblConfig.Name)_d$d"
            Width     = $AblConfig.Width
            Height    = $AblConfig.Height
            Spp       = $AblConfig.Spp
            MaxDepth  = $d
        }
        $Configs += $cfgCopy
    }
} elseif ($Mem) {
    $Configs = $MemConfigs
} elseif ($Full) {
    $Configs = $FullConfigs
} else {
    $Configs = $QuickConfigs
}

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
Write-Host "=== Phase 3 benchmark runner (v3) ===" -ForegroundColor Cyan
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
    # Find a working Python+Pillow combo for PNG conversion.
    #
    # Scoop installs Python into a VERSIONED subfolder, e.g.:
    #   %USERPROFILE%\scoop\apps\python313\3.13.2\python.exe
    # The \current\ junction that older Scoop versions created is often
    # absent on fresh installs, so we glob for versioned subfolders instead.
    $script:PyLauncher = $null

    $pyCandidates = @()

    # 1. Walk every python* app in Scoop and find python.exe in any subfolder
    #    (covers both \current\ and versioned \X.Y.Z\ layouts).
    $scoopApps = "$env:USERPROFILE\scoop\apps"
    foreach ($appDir in @(Get-ChildItem -Path $scoopApps -Directory -Filter "python*" -ErrorAction SilentlyContinue)) {
        foreach ($pyExe in @(Get-ChildItem -Path $appDir.FullName -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue)) {
            if ($null -eq $pyExe) { continue }
            # Skip the shim (it lives in \scoop\shims, not \scoop\apps)
            $pyCandidates += "`"$($pyExe.FullName)`""
        }
    }

    # 2. Windows py launcher in C:\Windows (rare in SSH sessions).
    $py = "$env:WINDIR\py.exe"
    if (Test-Path -LiteralPath $py) { $pyCandidates += "`"$py`" -3" }

    # 3. Bare names as last resort (may hit the Microsoft Store stub).
    $pyCandidates += "py -3"
    $pyCandidates += "python3"
    $pyCandidates += "python"

    Write-Host "Probing Python launchers:"
    foreach ($cand in $pyCandidates) {
        & cmd.exe /c "$cand -c `"from PIL import Image`" > NUL 2> NUL"
        $rc = $LASTEXITCODE
        if ($rc -eq 0) {
            Write-Host "  [OK]    $cand" -ForegroundColor Green
            $script:PyLauncher = $cand
            break
        } else {
            Write-Host "  [fail $rc] $cand" -ForegroundColor DarkGray
        }
    }

    if ($null -eq $script:PyLauncher) {
        Write-Host "WARN: no working python+Pillow found, using -SkipPng automatically" -ForegroundColor Yellow
        Write-Host "      Your Python is at: $scoopApps\python313\<version>\python.exe" -ForegroundColor Yellow
        Write-Host "      Install Pillow with: <python> -m pip install pillow" -ForegroundColor Yellow
        $SkipPng = $true
    } else {
        Write-Host "Python+PIL:     $script:PyLauncher" -ForegroundColor DarkGray
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

# Emit the ppm->png helper script once. Per-image conversion just calls
# `python helper.py <ppm> <png>`, which means paths with spaces, Unicode,
# or apostrophes can't break us via quote-escaping (the previous version
# embedded paths into a `python -c "..."` command line through cmd.exe and
# was vulnerable to all of those). Argv-based is bulletproof.
$PpmToPngScript = Join-Path $RunDir "ppm_to_png.py"
if (-not $SkipPng) {
    @"
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
Image.open(src).save(dst)
"@ | Out-File -FilePath $PpmToPngScript -Encoding ascii
}

Copy-Item -Path $SceneHPath -Destination (Join-Path $RunDir "scene.h.snapshot") -Force

$gitHash = ""
try { $gitHash = & git -C $ProjectRoot rev-parse --short HEAD 2>$null } catch {}

@(
    "Run timestamp:   $RunStamp"
    "Active scene:    $ActiveScene"
    "Tag:             $Tag"
    "Exe:             $ExePath"
    "PTX:             $PtxPath"
    "Git hash:        $gitHash"
    "Config set:      $(if ($Ablation){'ablation'} elseif ($Mem){'mem'} elseif ($Full){'full'} else {'quick'})"
    "Configs:         $($Configs.Count)"
    "MaxDepth:        $(if ($MaxDepth -gt 0) { $MaxDepth } else { 'default (20)' })"
    "TileSize:        $(if ($TileSize -ge 0) { $TileSize } else { 'default (compile-time TILE_W/TILE_H)' })"
    "MemLog:          forced=$($MemLog), suppressed=$($NoMemLog), nvsmi=$NvidiaSmiAvailable"
) | Out-File -FilePath $RunInfoPath -Encoding ascii

"phase,scene,resolution,spp,time_ms,mrays_sec,notes,run_timestamp" | Out-File -FilePath $CsvPath -Encoding ascii
"phase,scene,resolution,spp,max_depth,time_ms,mrays_sec,peak_mb,run_timestamp" | Out-File -FilePath $ExtCsvPath -Encoding ascii
if ($Ablation) {
    "max_depth,resolution,spp,time_ms,mrays_sec,peak_mb,run_timestamp" | Out-File -FilePath $AblationCsvPath -Encoding ascii
}

if ($ConfigFilter -ne "") {
    $Configs = $Configs | Where-Object { $_.Name -match $ConfigFilter }
    Write-Host "Filter '$ConfigFilter' matched $($Configs.Count) config(s)"
}

Write-Host "Config set:     $(if ($Ablation){'ablation'} elseif ($Mem){'mem'} elseif ($Full){'full'} else {'quick'}) ($($Configs.Count) configs)"
Write-Host ""

# -----------------------------------------------------------------------------
# Memory logger helper (background nvidia-smi sampler)
# -----------------------------------------------------------------------------
function Should-LogMem {
    param([string]$ConfigName)
    if ($NoMemLog)              { return $false }
    if (-not $NvidiaSmiAvailable) { return $false }
    if ($MemLog)                { return $true }
    # Auto-on for any 4K config
    if ($ConfigName -match '4096') { return $true }
    return $false
}

function Start-MemLogger {
    param([string]$ConfigName, [string]$LogDir)
    $logPath = Join-Path $LogDir "${ConfigName}_nvsmi.csv"
    $errPath = Join-Path $LogDir "${ConfigName}_nvsmi.err"
    $nvsmiArgs = @(
        "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-lms", "100"
    )

    try {
        $proc = Start-Process -FilePath "nvidia-smi" -ArgumentList $nvsmiArgs `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError  $errPath `
            -ErrorAction Stop

        if ($null -eq $proc) {
            Write-Host "  nvsmi memlog launch returned no process" -ForegroundColor Yellow
            return $null
        }

        Start-Sleep -Milliseconds 250

        try {
            if ($proc.HasExited -and $proc.ExitCode -ne 0) {
                Write-Host "  nvsmi memlog exited immediately (code $($proc.ExitCode))" -ForegroundColor Yellow
                return $null
            }
        } catch {
            Write-Host "  nvsmi memlog status check failed: $($_.Exception.Message)" -ForegroundColor Yellow
            return $null
        }

        return @{ Process = $proc; LogPath = $logPath; ErrPath = $errPath }
    } catch {
        Write-Host "  nvsmi memlog launch failed: $($_.Exception.Message)" -ForegroundColor Yellow
        return $null
    }
}

function Stop-MemLogger {
    param($Logger)
    if ($null -eq $Logger) { return 0 }
    if ($null -eq $Logger.Process) { return 0 }
    try {
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
    } catch {
        Write-Host "  nvsmi memlog teardown failed: $($_.Exception.Message)" -ForegroundColor Yellow
        return 0
    }
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

    # Per-config max-depth (ablation sets this; otherwise use the global -MaxDepth or default)
    $cfgMaxDepth = if ($cfg.ContainsKey('MaxDepth')) { $cfg.MaxDepth } elseif ($MaxDepth -gt 0) { $MaxDepth } else { 0 }

    $ppmPath = Join-Path $RendersDir "$name.ppm"
    $pngPath = Join-Path $RendersDir "$name.png"
    $logPath = Join-Path $LogsDir    "$name.log"

    $depthDisplay = if ($cfgMaxDepth -gt 0) { "depth=$cfgMaxDepth" } else { "depth=default" }
    Write-Host ("--- $name ({0}x{1}, $spp spp, $depthDisplay) ---" -f $width, $height) -ForegroundColor Yellow

    $cfgStart = Get-Date

    $exeArgs = @(
        "--width",  $width,
        "--height", $height,
        "--spp",    $spp,
        "--output", $ppmPath,
        "--ptx",    $PtxPath
    )
    if ($cfgMaxDepth -gt 0) {
        $exeArgs += @("--max-depth", $cfgMaxDepth)
    }
    if ($TileSize -ge 0) {
        $exeArgs += @("--tile-size", $TileSize)
    }

    @(
        "# Phase 3 benchmark: $name"
        "# Resolution: ${width}x${height}"
        "# SPP: $spp"
        "# MaxDepth: $cfgMaxDepth"
        "# Run: $RunStamp"
        "# Scene: $ActiveScene"
        "# Git: $gitHash"
        "# Command: $ExePath $($exeArgs -join ' ')"
        "# ---"
    ) | Out-File -FilePath $logPath -Encoding ascii

    # Start memory logger if this config warrants it
    $memLogger = $null
    $extPeakMb = 0
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

    # Parse "CSV: optix_phase3,1024x768,256,499.99,471.86,sm_86"  (legacy line)
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
        }
    } else {
        Write-Host "  WARN: no CSV line found in exe output" -ForegroundColor Yellow
        $failures += @{ Config = $name; Reason = "no CSV line in output" }
    }

    # Parse "PHASE3_EXT: optix_phase3,1024x768,256,20,499.99,471.86,1234"  (new ext line)
    $extLine = Select-String -Path $logPath -Pattern "^PHASE3_EXT: " | Select-Object -Last 1
    if ($extLine) {
        $payload = $extLine.Line.Substring(12).Trim()
        $parts = $payload -split ','
        if ($parts.Count -ge 7) {
            $phase    = $parts[0]
            $res      = $parts[1]
            $sppEx    = $parts[2]
            $depthEx  = $parts[3]
            $timeEx   = $parts[4]
            $mraysEx  = $parts[5]
            $peakEx   = $parts[6]
            $extPeakMb = [int]$peakEx
            "$phase,$ActiveScene,$res,$sppEx,$depthEx,$timeEx,$mraysEx,$peakEx,$RunStamp" |
                Out-File -FilePath $ExtCsvPath -Append -Encoding ascii

            # If this is an ablation run, also append to ablation.csv
            if ($Ablation) {
                "$depthEx,$res,$sppEx,$timeEx,$mraysEx,$peakEx,$RunStamp" |
                    Out-File -FilePath $AblationCsvPath -Append -Encoding ascii
            }
        }
    }

    $reportedPeak = [Math]::Max($extPeakMb, $nvsmiPeakMb)
    if ($reportedPeak -gt 0) {
        Write-Host ("  Peak: {0} MB (exe={1}, nvsmi={2})" -f $reportedPeak, $extPeakMb, $nvsmiPeakMb) -ForegroundColor DarkGray
    }

    if (-not $SkipPng) {
        if (Test-Path $ppmPath) {
            $line  = "$($script:PyLauncher) `"$PpmToPngScript`" `"$ppmPath`" `"$pngPath`""
            $pyOut = & cmd.exe /c "$line 2>&1"
            if ($LASTEXITCODE -eq 0 -and (Test-Path $pngPath)) {
                Write-Host "  PNG: $pngPath" -ForegroundColor Green
            } else {
                $reason = if ($LASTEXITCODE -ne 0) { "exit $LASTEXITCODE" } else { "exit 0 but file not on disk" }
                Write-Host "  PNG conversion failed ($reason)" -ForegroundColor Yellow
                Write-Host "    cmdline: $line" -ForegroundColor Yellow
                if ($pyOut) {
                    $pyOut | Select-Object -First 5 | ForEach-Object {
                        Write-Host "    $_" -ForegroundColor Yellow
                    }
                }
                $failures += @{ Config = $name; Reason = "PNG conversion: $reason" }
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

if (Test-Path $ExtCsvPath) {
    Write-Host ""
    Write-Host "Extended timings (with depth + peak_mb):" -ForegroundColor Cyan
    Get-Content $ExtCsvPath | ForEach-Object { Write-Host "  $_" }
}

if ($Ablation) {
    Write-Host ""
    Write-Host "Ablation:" -ForegroundColor Cyan
    Get-Content $AblationCsvPath | ForEach-Object { Write-Host "  $_" }
}