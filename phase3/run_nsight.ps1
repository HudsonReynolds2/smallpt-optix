# =============================================================================
# Nsight Compute runner for phase 2/3 A/B comparison.
#
# Captures the metrics and full-set reports needed for the deck.
#
#   .\run_nsight.ps1 -Variant phase2_baseline -ExePath ..\phase2\build\Release\optix_smallpt.exe -PtxPath ..\phase2\build\shaders.ptx
#   .\run_nsight.ps1 -Variant phase3_tess_tile -ExePath .\build\Release\optix_smallpt.exe -PtxPath .\build\shaders.ptx
#
# Output goes to results\nsight\<timestamp>_<variant>\ by default.
# A PPM and (if Pillow is available) a PNG are written under
# results\nsight\<...>\renders\ for every profiled config so a broken
# render cannot pass silently.
#
# =============================================================================
# OPTIX + NSIGHT COMPUTE INTEGRATION (READ THIS FIRST)
# =============================================================================
# By default, OptiX 8 launches its raygen kernel through a code path that
# Nsight Compute does NOT intercept as a discrete kernel. Symptom:
#
#     ==WARNING== No kernels were profiled.
#     Available Kernels:
#       1. NVIDIA internal (optixAccelBuild)
#       2. optixAccelBuild
#
# Only the BVH-build kernels show up; the actual raygen is invisible to ncu.
#
# Fix: set OPTIX_FORCE_DEPRECATED_LAUNCHER=1 in the environment when
# launching the application under ncu. This forces OptiX to use its older
# launch path which ncu *does* see. Side effect: Shader Execution Reordering
# (SER) is disabled while this is set -- but SER is Ada+ only and the
# RTX 3080 Ti is Ampere, so this changes nothing on this machine. For Ada+
# users this matters and SER metrics need a different capture strategy.
#
# Source: NVIDIA OptiX engineer "droettger" on the NVIDIA dev forums.
#   https://forums.developer.nvidia.com/t/need-help-profiling-an-optix-application/265266
# Cross-referenced with the Nsight Compute OptiX support page:
#   https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/library-support-optix.html
#
# This script sets OPTIX_FORCE_DEPRECATED_LAUNCHER=1 only for the duration
# of each ncu invocation, via Start-Process -Environment. It does NOT
# pollute the parent shell or persist after the script exits.
# =============================================================================
#
# REQUIREMENTS:
#   - ncu.exe must be on PATH. With CUDA Toolkit 12.8 the default install
#     puts it at: "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.x.y\ncu.exe"
#     Either add that to PATH or pass -NcuPath.
#   - First-time use needs GPU performance counters unlocked:
#       NVIDIA Control Panel -> Manage 3D settings -> Developer ->
#       Manage GPU Performance Counters -> "Allow access to all users"
#     If you see ERR_NVGPUCTRPERM, that's what's missing. Reboot if the
#     setting doesn't take effect.
#   - The exe must already be built. This script does NOT rebuild.
#   - Python with Pillow on PATH for PPM->PNG conversion (or pass -SkipPng).
#
# MODES:
#   quick (default) - one targeted-metrics capture on 1024x768_64spp.
#                     ~30-90s wall time. Output: <config>_metrics.csv.
#   full            - --set full capture on 3 configs.
#                     ~5-15 min wall time per config. Output: <config>_full.ncu-rep
#                     plus a parsed summary CSV. Open .ncu-rep in the Nsight
#                     Compute GUI for full section view.
#
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Variant,

    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [Parameter(Mandatory=$true)]
    [string]$PtxPath,

    [ValidateSet("quick","full")]
    [string]$Mode = "quick",

    [string]$RunDir = "",

    [string]$NcuPath = "ncu",

    # Regex passed to ncu --kernel-name. Default matches any kernel whose
    # name contains "raygen" (case insensitive). The OptiX docs say
    # kernel names start with "raygen__" -- we match the substring so
    # any user-defined raygen entry is captured.
    [string]$KernelNameRegex = "regex:(?i)raygen",

    [switch]$SkipPng,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# -----------------------------------------------------------------------------
# Resolve paths
# -----------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not [System.IO.Path]::IsPathRooted($ExePath)) {
    $ExePath = Join-Path $ScriptDir $ExePath
}
if (-not [System.IO.Path]::IsPathRooted($PtxPath)) {
    $PtxPath = Join-Path $ScriptDir $PtxPath
}
$ExePath = [System.IO.Path]::GetFullPath($ExePath)
$PtxPath = [System.IO.Path]::GetFullPath($PtxPath)

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found: $ExePath" -ForegroundColor Red
    exit 2
}
if (-not (Test-Path $PtxPath)) {
    Write-Host "ERROR: PTX not found: $PtxPath" -ForegroundColor Red
    exit 2
}

try {
    $ncuVer = & $NcuPath --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
} catch {
    Write-Host "ERROR: cannot invoke ncu at '$NcuPath'." -ForegroundColor Red
    Write-Host "  Add Nsight Compute to PATH or pass -NcuPath 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.x.y\ncu.exe'" -ForegroundColor Red
    Write-Host "  Detail: $_" -ForegroundColor Red
    exit 2
}
Write-Host "ncu: $($ncuVer | Select-Object -First 1)" -ForegroundColor DarkGray

# Pillow availability check (only matters if PNG conversion is wanted).
$PngEnabled = -not $SkipPng
if ($PngEnabled) {
    & cmd.exe /c "python -c `"from PIL import Image`" >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: python+Pillow not available -- skipping PNG conversion (PPMs will still be written)." -ForegroundColor Yellow
        Write-Host "      Install with: pip install pillow  -- or pass -SkipPng to silence." -ForegroundColor Yellow
        $PngEnabled = $false
    }
}

# -----------------------------------------------------------------------------
# Set the OptiX deprecated-launcher env var for this PowerShell session only.
# Start-Process inherits the parent environment, so we need it set here.
# We restore the prior value (or remove the var) at script exit.
# -----------------------------------------------------------------------------
$EnvVarName = "OPTIX_FORCE_DEPRECATED_LAUNCHER"
$priorEnv = [System.Environment]::GetEnvironmentVariable($EnvVarName, "Process")
[System.Environment]::SetEnvironmentVariable($EnvVarName, "1", "Process")
Write-Host "Env: $EnvVarName=1 (process-scoped, restored on exit)" -ForegroundColor DarkGray

# -----------------------------------------------------------------------------
# Run folder
# -----------------------------------------------------------------------------
if ($RunDir -eq "") {
    $RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $ScriptDir "results\nsight\${RunStamp}_${Variant}"
}
$NsightDir  = Join-Path $RunDir "nsight"
$RendersDir = Join-Path $RunDir "renders"
New-Item -ItemType Directory -Force -Path $NsightDir  | Out-Null
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null

Write-Host "=== Nsight Compute capture ===" -ForegroundColor Cyan
Write-Host "Variant:     $Variant"
Write-Host "Mode:        $Mode"
Write-Host "Exe:         $ExePath"
Write-Host "PTX:         $PtxPath"
Write-Host "Kernel:      $KernelNameRegex"
Write-Host "Output:      $NsightDir"
Write-Host "Renders:     $RendersDir"
Write-Host ""

# Snapshot scene.h and git hash if available
$SceneSnapshot = Join-Path (Split-Path -Parent $ExePath) "..\..\src\scene.h"
$SceneSnapshot = [System.IO.Path]::GetFullPath($SceneSnapshot)
if (Test-Path $SceneSnapshot) {
    Copy-Item $SceneSnapshot (Join-Path $NsightDir "scene.h.snapshot") -Force
}
try {
    Push-Location $ScriptDir
    $gitHash = (& git rev-parse HEAD 2>$null) -join ""
    if ($LASTEXITCODE -eq 0 -and $gitHash) {
        $gitHash | Out-File -FilePath (Join-Path $NsightDir "git_hash.txt") -Encoding ascii
    }
} catch {
    # git not installed or not a repo, fine
} finally {
    Pop-Location
}

@(
    "Nsight capture timestamp:          $(Get-Date -Format 'o')"
    "Variant:                           $Variant"
    "Mode:                              $Mode"
    "Exe:                               $ExePath"
    "PTX:                               $PtxPath"
    "Kernel filter:                     $KernelNameRegex"
    "OPTIX_FORCE_DEPRECATED_LAUNCHER:   1"
    "Host:                              $env:COMPUTERNAME"
    "User:                              $env:USERNAME"
) | Out-File -FilePath (Join-Path $NsightDir "info.txt") -Encoding ascii

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
if ($Mode -eq "quick") {
    $Configs = @(
        @{ Name = "1024x768_64spp"; Width = 1024; Height = 768; Spp = 64 }
    )
} else {
    $Configs = @(
        @{ Name = "512x384_64spp";   Width = 512;  Height = 384;  Spp = 64 },
        @{ Name = "1024x768_64spp";  Width = 1024; Height = 768;  Spp = 64 },
        @{ Name = "2048x1536_32spp"; Width = 2048; Height = 1536; Spp = 32 }
    )
}

$Metrics = @(
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors.sum",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active"
) -join ","

# -----------------------------------------------------------------------------
# Helper: convert PPM -> PNG using Pillow
# -----------------------------------------------------------------------------
function Convert-PpmToPng {
    param([string]$PpmPath, [string]$PngPath)
    if (-not (Test-Path $PpmPath)) { return $false }
    $pyCmd = "from PIL import Image; Image.open(r'$PpmPath').save(r'$PngPath')"
    $pyCmdEscaped = $pyCmd.Replace('"', '\"')
    & cmd.exe /c "python -c `"$pyCmdEscaped`" >NUL 2>NUL"
    return ($LASTEXITCODE -eq 0)
}

# -----------------------------------------------------------------------------
# Capture loop
# -----------------------------------------------------------------------------
$grandStart = Get-Date
$failures = @()

try {

foreach ($cfg in $Configs) {
    $name   = $cfg.Name
    $width  = $cfg.Width
    $height = $cfg.Height
    $spp    = $cfg.Spp

    Write-Host "--- $name ($width x $height, $spp spp) ---" -ForegroundColor Yellow

    $ppmPath = Join-Path $RendersDir "${name}.ppm"
    $pngPath = Join-Path $RendersDir "${name}.png"

    $appArgs = @(
        "--width",  $width,
        "--height", $height,
        "--spp",    $spp,
        "--output", $ppmPath,
        "--ptx",    $PtxPath
    )

    if ($Mode -eq "quick") {
        $outCsv = Join-Path $NsightDir "${name}_metrics.csv"
        $ncuArgs = @(
            "--metrics", $Metrics,
            "--kernel-name", $KernelNameRegex,
            "--csv",
            "--target-processes", "all",
            "--",
            $ExePath
        ) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outCsv.stderr.tmp"
        $stdoutTmp = "$outCsv.stdout.tmp"
        # $env:VAR = "1" propagates to child processes in PS 5.1 (unlike
        # [System.Environment]::SetEnvironmentVariable which only affects the
        # current process object but is NOT inherited by Start-Process children).
        # -Environment was added in PS 7.3 and is not available here.
        $env:OPTIX_FORCE_DEPRECATED_LAUNCHER = "1"
        $proc = Start-Process -FilePath $NcuPath -ArgumentList $ncuArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutTmp `
            -RedirectStandardError $stderrTmp

        if (Test-Path $stdoutTmp) { Move-Item -Force $stdoutTmp $outCsv }
        if (Test-Path $stderrTmp) { Move-Item -Force $stderrTmp "$outCsv.log" }

        $cfgElapsed = (Get-Date) - $cfgStart
        Write-Host ("  Wall time: {0:N1}s" -f $cfgElapsed.TotalSeconds)

        if ($proc.ExitCode -ne 0) {
            Write-Host "  FAILED: ncu exit $($proc.ExitCode). See $outCsv.log" -ForegroundColor Red
            if (Test-Path "$outCsv.log") {
                Write-Host "  --- ncu stderr (last 30 lines) ---" -ForegroundColor Red
                Get-Content "$outCsv.log" -Tail 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            }
            $failures += @{ Config = $name; Reason = "ncu exit $($proc.ExitCode)" }
            continue
        }

        if (-not (Test-Path $outCsv) -or (Get-Item $outCsv).Length -eq 0) {
            Write-Host "  FAILED: ncu produced no CSV output" -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "empty CSV" }
            continue
        }

        # Detect "no kernels were profiled" up front (its own failure mode)
        $csvRaw = Get-Content $outCsv -Raw
        if ($csvRaw -match 'No kernels were profiled') {
            Write-Host "  FAILED: ncu reports 'No kernels were profiled'." -ForegroundColor Red
            Write-Host "         OPTIX_FORCE_DEPRECATED_LAUNCHER may not have been picked up." -ForegroundColor Red
            Write-Host "         Verify in info.txt that the env var was set; if so, this may need" -ForegroundColor Red
            Write-Host "         OptixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE in main.cpp." -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "ncu: no kernels profiled" }
            continue
        }

        # Parse Kernel Name column
        $csvLines = Get-Content $outCsv
        $headerIdx = -1
        for ($i = 0; $i -lt $csvLines.Count; $i++) {
            if ($csvLines[$i] -match '^"ID"') { $headerIdx = $i; break }
        }
        if ($headerIdx -lt 0) {
            Write-Host "  FAILED: no CSV header row found in output" -ForegroundColor Red
            Write-Host "  First 10 lines:" -ForegroundColor Red
            $csvLines | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            $failures += @{ Config = $name; Reason = "malformed CSV (no header)" }
            continue
        }
        $kernelNames = @{}
        for ($i = $headerIdx + 1; $i -lt $csvLines.Count; $i++) {
            $line = $csvLines[$i]
            if ($line -match '^"\d+"') {
                $fields = $line.TrimStart('"').Split('","', [StringSplitOptions]::None)
                if ($fields.Count -ge 5) {
                    $kn = $fields[4]
                    if (-not $kernelNames.ContainsKey($kn)) { $kernelNames[$kn] = 0 }
                    $kernelNames[$kn]++
                }
            }
        }

        if ($kernelNames.Count -eq 0) {
            Write-Host "  FAILED: CSV has header but no data rows" -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "no data rows" }
            continue
        }

        Write-Host "  Kernels profiled:" -ForegroundColor DarkGray
        foreach ($kn in $kernelNames.Keys) {
            Write-Host ("    {0,-50}  {1} metrics" -f $kn, $kernelNames[$kn]) -ForegroundColor DarkGray
        }

        $hasRaygen = $false
        foreach ($kn in $kernelNames.Keys) {
            if ($kn -match '(?i)raygen') { $hasRaygen = $true; break }
        }
        if (-not $hasRaygen) {
            Write-Host "  FAILED: no raygen kernel in profiled set." -ForegroundColor Red
            Write-Host "         The kernel-name filter '$KernelNameRegex' didn't match anything." -ForegroundColor Red
            Write-Host "         Try -KernelNameRegex 'regex:.*' to capture all kernels and inspect names." -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "no raygen kernel profiled" }
            continue
        }

        $occLine = $csvLines | Where-Object { $_ -match 'warps_active' } | Select-Object -First 1
        if ($occLine) {
            $occFields = $occLine.TrimStart('"').Split('","', [StringSplitOptions]::None)
            if ($occFields.Count -ge 1) {
                $occVal = $occFields[$occFields.Count - 1].TrimEnd('"')
                if ($occVal -eq 'n/a' -or $occVal -eq '') {
                    Write-Host "  WARN: warps_active value is '$occVal'." -ForegroundColor Yellow
                } else {
                    Write-Host "  Sanity: warps_active = $occVal" -ForegroundColor DarkGray
                }
            }
        }
        Write-Host "  OK: $outCsv" -ForegroundColor Green

    } else {
        # full mode
        $outRep = Join-Path $NsightDir "${name}_full.ncu-rep"
        $ncuArgs = @(
            "--set", "full",
            "--kernel-name", $KernelNameRegex,
            "--target-processes", "all",
            "--export", $outRep,
            "--force-overwrite",
            "--",
            $ExePath
        ) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outRep.stderr.tmp"
        $stdoutTmp = "$outRep.stdout.tmp"
        # Same $env: approach as quick mode — PS 5.1 compatible.
        $env:OPTIX_FORCE_DEPRECATED_LAUNCHER = "1"
        $proc = Start-Process -FilePath $NcuPath -ArgumentList $ncuArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutTmp `
            -RedirectStandardError $stderrTmp

        if (Test-Path $stdoutTmp) { Move-Item -Force $stdoutTmp "$outRep.stdout.log" }
        if (Test-Path $stderrTmp) { Move-Item -Force $stderrTmp "$outRep.stderr.log" }

        $cfgElapsed = (Get-Date) - $cfgStart
        Write-Host ("  Wall time: {0:N1}s" -f $cfgElapsed.TotalSeconds)

        # Detect 'no kernels profiled' first, regardless of exit code.
        if (Test-Path "$outRep.stdout.log") {
            $stdoutContent = Get-Content "$outRep.stdout.log" -Raw
            if ($stdoutContent -match 'No kernels were profiled') {
                Write-Host "  FAILED: ncu reports 'No kernels were profiled'." -ForegroundColor Red
                Write-Host "         OPTIX_FORCE_DEPRECATED_LAUNCHER may not have been picked up." -ForegroundColor Red
                $failures += @{ Config = $name; Reason = "ncu: no kernels profiled" }
                continue
            }
        }

        if ($proc.ExitCode -ne 0) {
            Write-Host "  FAILED: ncu exit $($proc.ExitCode)." -ForegroundColor Red
            if (Test-Path "$outRep.stderr.log") {
                Write-Host "  --- ncu stderr (last 30 lines) ---" -ForegroundColor Red
                Get-Content "$outRep.stderr.log" -Tail 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            }
            $failures += @{ Config = $name; Reason = "ncu exit $($proc.ExitCode)" }
            continue
        }

        if (-not (Test-Path $outRep)) {
            Write-Host "  FAILED: .ncu-rep not produced" -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "no .ncu-rep" }
            continue
        }

        $sizeMB = (Get-Item $outRep).Length / 1MB
        Write-Host ("  Report: $outRep ({0:N1} MB)" -f $sizeMB) -ForegroundColor DarkGray

        if ($sizeMB -lt 5.0) {
            Write-Host "  WARN: report unusually small ($sizeMB MB). The kernel-name filter may not have matched raygen." -ForegroundColor Yellow
            Write-Host "        Quick check: run with -Mode quick first; it lists which kernels were profiled." -ForegroundColor Yellow
        }

        # Export a CSV summary for diffing.
        $outSummary = Join-Path $NsightDir "${name}_full_summary.csv"
        $sumArgs = @(
            "--import", $outRep,
            "--csv",
            "--page", "details",
            "--print-units", "base"
        )
        Write-Host "  Exporting summary CSV..." -ForegroundColor DarkGray
        $sumStdout = "$outSummary.tmp"
        $sumProc = Start-Process -FilePath $NcuPath -ArgumentList $sumArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $sumStdout `
            -RedirectStandardError "$outSummary.log"
        if (Test-Path $sumStdout) { Move-Item -Force $sumStdout $outSummary }
        if ($sumProc.ExitCode -ne 0) {
            Write-Host "  WARN: summary export failed (exit $($sumProc.ExitCode))." -ForegroundColor Yellow
        }

        if (Test-Path $outSummary) {
            $sumContent = Get-Content $outSummary -Raw
            if ($sumContent -notmatch '(?i)raygen') {
                Write-Host "  FAILED: summary CSV has no raygen mention." -ForegroundColor Red
                $failures += @{ Config = $name; Reason = "raygen absent from full report" }
                continue
            }
        }

        Write-Host "  OK: $outRep" -ForegroundColor Green
    }

    # Visual artifact check
    if (-not (Test-Path $ppmPath)) {
        Write-Host "  WARN: no PPM at $ppmPath -- the exe ran under the profiler but produced no image." -ForegroundColor Yellow
        $failures += @{ Config = $name; Reason = "no PPM written" }
    } else {
        $ppmBytes = (Get-Item $ppmPath).Length
        if ($ppmBytes -lt 1024) {
            Write-Host ("  WARN: PPM is suspiciously small ({0} bytes). Render may be corrupt." -f $ppmBytes) -ForegroundColor Yellow
        } else {
            Write-Host ("  PPM: $ppmPath ({0:N1} MB)" -f ($ppmBytes / 1MB)) -ForegroundColor DarkGray
        }

        if ($PngEnabled) {
            if (Convert-PpmToPng $ppmPath $pngPath) {
                Write-Host "  PNG: $pngPath" -ForegroundColor Green
            } else {
                Write-Host "  WARN: PNG conversion failed (PPM is still on disk)" -ForegroundColor Yellow
            }
        }
    }
}

} finally {
    # Restore the env var (both $env: and the process-level variable).
    if ($null -eq $priorEnv) {
        Remove-Item Env:\OPTIX_FORCE_DEPRECATED_LAUNCHER -ErrorAction SilentlyContinue
        [System.Environment]::SetEnvironmentVariable($EnvVarName, $null, "Process")
    } else {
        $env:OPTIX_FORCE_DEPRECATED_LAUNCHER = $priorEnv
        [System.Environment]::SetEnvironmentVariable($EnvVarName, $priorEnv, "Process")
    }
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
$grandElapsed = (Get-Date) - $grandStart
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host ("Total wall time: {0:N1}s" -f $grandElapsed.TotalSeconds)
Write-Host "Output:  $NsightDir"
Write-Host "Renders: $RendersDir"

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures ($($failures.Count)):" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red
    }
    exit 1
}

exit 0
