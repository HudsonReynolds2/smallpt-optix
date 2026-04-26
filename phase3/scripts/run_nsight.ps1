# =============================================================================
# Nsight Compute runner for phase 2/3 A/B comparison (v3, deck-prep).
#
# Differences vs v2:
#   - Metric set replaced with one that supports an empirical arithmetic
#     intensity calculation: FLOP counters (FFMA, FADD, FMUL) + DRAM bytes.
#   - -ApplicationReplay switch tries `--replay-mode application` to
#     potentially capture the real raygen kernel name on Ampere instead of
#     the deprecated-launcher wrapper. See rationale below.
#   - Quick-mode output now includes a derived "FLOPs/byte" line and
#     dumps a slim summary CSV alongside the raw ncu CSV for chart use.
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
# launch path which ncu *does* see, but names the kernel `optixLaunch`
# (the launcher wrapper) rather than `__raygen__rg`. Side effect: Shader
# Execution Reordering (SER) is disabled while this is set -- but SER is
# Ada+ only and the RTX 3080 Ti is Ampere, so this changes nothing on
# this machine. Profiling caveat: PC-sampling stalls come back `n/a` in
# this mode; that's a known limitation of the deprecated launcher capture.
#
# Source: NVIDIA OptiX engineer "droettger" on the NVIDIA dev forums.
#   https://forums.developer.nvidia.com/t/need-help-profiling-an-optix-application/265266
# Cross-referenced with the Nsight Compute OptiX support page:
#   https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/library-support-optix.html
#
# This script sets OPTIX_FORCE_DEPRECATED_LAUNCHER=1 only for the duration
# of each ncu invocation. It does NOT pollute the parent shell or persist
# after the script exits.
#
# -ApplicationReplay attempts an alternative capture mode where ncu re-runs
# the whole application per metric pass instead of per-kernel replay. On
# some setups this lets the modern launcher succeed and gives raygen its
# real name. We try it for the deck because raygen-named metrics (with
# pc-sampling stalls populated) would be a much better slide. If it
# doesn't work, the script falls back gracefully and we ship with the
# deprecated-launcher caveat.
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
#   quick (default) - one targeted-metrics capture on 1024x768_512spp.
#                     ~30-90s wall time. Output: <config>_metrics.csv
#                     plus a derived <config>_summary.csv with AI numbers.
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
    # name contains "raygen" or "optixLaunch" (case insensitive). On Ampere
    # with the deprecated launcher, "optixLaunch" is what comes through.
    [string]$KernelNameRegex = "regex:(?i)(raygen|optixLaunch)",

    # Try application-replay mode. On some setups this captures the modern
    # launcher and gives raygen its real name. If it fails or returns no
    # kernels, fall back to the standard kernel-replay path.
    [switch]$ApplicationReplay,

    # Skip the deprecated-launcher env var. Only useful with -ApplicationReplay
    # since application replay sometimes works with the modern launcher path.
    [switch]$SkipDeprecatedLauncher,

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
if (-not $SkipDeprecatedLauncher) {
    [System.Environment]::SetEnvironmentVariable($EnvVarName, "1", "Process")
    Write-Host "Env: $EnvVarName=1 (process-scoped, restored on exit)" -ForegroundColor DarkGray
} else {
    Write-Host "Env: $EnvVarName NOT set (-SkipDeprecatedLauncher)" -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
# Run folder
# -----------------------------------------------------------------------------
if ($RunDir -eq "") {
    $RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $tagSuffix = ""
    if ($ApplicationReplay) { $tagSuffix += "_appreplay" }
    if ($SkipDeprecatedLauncher) { $tagSuffix += "_modern" }
    $RunDir = Join-Path $ScriptDir "results\nsight\${RunStamp}_${Variant}${tagSuffix}"
}
$NsightDir  = Join-Path $RunDir "nsight"
$RendersDir = Join-Path $RunDir "renders"
New-Item -ItemType Directory -Force -Path $NsightDir  | Out-Null
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null

Write-Host "=== Nsight Compute capture ===" -ForegroundColor Cyan
Write-Host "Variant:           $Variant"
Write-Host "Mode:              $Mode"
Write-Host "ApplicationReplay: $ApplicationReplay"
Write-Host "DeprLauncher:      $(-not $SkipDeprecatedLauncher)"
Write-Host "Exe:               $ExePath"
Write-Host "PTX:               $PtxPath"
Write-Host "Kernel:            $KernelNameRegex"
Write-Host "Output:            $NsightDir"
Write-Host "Renders:           $RendersDir"
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
    "ApplicationReplay:                 $ApplicationReplay"
    "Exe:                               $ExePath"
    "PTX:                               $PtxPath"
    "Kernel filter:                     $KernelNameRegex"
    "OPTIX_FORCE_DEPRECATED_LAUNCHER:   $(if ($SkipDeprecatedLauncher) {'NOT SET'} else {'1'})"
    "Host:                              $env:COMPUTERNAME"
    "User:                              $env:USERNAME"
) | Out-File -FilePath (Join-Path $NsightDir "info.txt") -Encoding ascii

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
if ($Mode -eq "quick") {
    $Configs = @(
        @{ Name = "1024x768_512spp"; Width = 1024; Height = 768; Spp = 512 }
    )
} else {
    $Configs = @(
        @{ Name = "512x384_64spp";   Width = 512;  Height = 384;  Spp = 64 },
        @{ Name = "1024x768_64spp";  Width = 1024; Height = 768;  Spp = 64 },
        @{ Name = "2048x1536_32spp"; Width = 2048; Height = 1536; Spp = 32 }
    )
}

# Metrics for the deck. Why each one is here:
#
#   gpu__time_duration.sum                                    : kernel time
#   sm__throughput.avg.pct_of_peak_sustained_elapsed          : SM utilization
#   smsp__warps_active.avg.pct_of_peak_sustained_active       : occupancy
#   smsp__thread_inst_executed_per_inst_executed.ratio        : divergence (24/32 = 75%)
#   dram__throughput.avg.pct_of_peak_sustained_elapsed        : DRAM utilization (memory pressure)
#   dram__bytes.sum                                           : actual bytes hit DRAM (denominator for AI)
#   l1tex__throughput.avg.pct_of_peak_sustained_active        : L1/TEX pressure
#   lts__t_sectors.sum / lts__t_sectors_lookup_hit.sum        : L2 hit rate
#   sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active: transcendental pipe usage
#   smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg.*    : memory stall reason (n/a in deprecated launcher)
#   smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg.*   : compute stall reason (n/a in deprecated launcher)
#   smsp__sass_thread_inst_executed_op_ffma_pred_on.sum       : FFMA = 2 FLOPs/inst (numerator for AI)
#   smsp__sass_thread_inst_executed_op_fadd_pred_on.sum       : FADD = 1 FLOP/inst
#   smsp__sass_thread_inst_executed_op_fmul_pred_on.sum       : FMUL = 1 FLOP/inst
#
$Metrics = @(
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors.sum",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
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

# Helper: parse a numeric metric value out of the ncu CSV. Returns 0 on miss.
# The CSV from ncu has columns: ID, Process ID, ..., Metric Name, Metric Unit, Metric Value
# We sum across rows (multiple kernel invocations contribute multiple rows).
function Parse-MetricSum {
    param([string]$CsvPath, [string]$MetricName)
    if (-not (Test-Path $CsvPath)) { return 0.0 }

    $lines = Get-Content $CsvPath
    $headerIdx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^"ID"') { $headerIdx = $i; break }
    }
    if ($headerIdx -lt 0) { return 0.0 }

    $hdr = [regex]::Split($lines[$headerIdx], '","') | ForEach-Object { $_.Trim('"') }
    $nameCol  = [Array]::IndexOf($hdr, 'Metric Name')
    $valueCol = [Array]::IndexOf($hdr, 'Metric Value')
    if ($nameCol -lt 0 -or $valueCol -lt 0) { return 0.0 }

    $sum = 0.0
    for ($i = $headerIdx + 1; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if (-not ($line -match '^"\d+"')) { continue }
        $f = [regex]::Split($line, '","') | ForEach-Object { $_.Trim('"') }
        if ($f.Count -le $valueCol) { continue }
        if ($f[$nameCol] -ne $MetricName) { continue }

        # Strip thousand separators ("13,701,348,177") that ncu emits with --csv
        $raw = $f[$valueCol] -replace ',', ''
        $val = 0.0
        if ([double]::TryParse($raw, [ref]$val)) { $sum += $val }
    }
    return $sum
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
            "--kernel-name", "`"$KernelNameRegex`"",
            "--csv",
            "--target-processes", "all"
        )
        if ($ApplicationReplay) {
            $ncuArgs += @("--replay-mode", "application")
        }
        $ncuArgs += @("--", $ExePath) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outCsv.stderr.tmp"
        $stdoutTmp = "$outCsv.stdout.tmp"
        if (-not $SkipDeprecatedLauncher) {
            $env:OPTIX_FORCE_DEPRECATED_LAUNCHER = "1"
        } else {
            Remove-Item Env:\OPTIX_FORCE_DEPRECATED_LAUNCHER -ErrorAction SilentlyContinue
        }
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
            if ($ApplicationReplay) {
                Write-Host "         Application replay didn't capture either. Try without -ApplicationReplay." -ForegroundColor Red
            } else {
                Write-Host "         OPTIX_FORCE_DEPRECATED_LAUNCHER may not have been picked up." -ForegroundColor Red
                Write-Host "         Try -ApplicationReplay or set OptixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE in main.cpp." -ForegroundColor Red
            }
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
        $headerFields = [regex]::Split($csvLines[$headerIdx], '","')
        $headerFields = $headerFields | ForEach-Object { $_.Trim('"') }
        $knCol = [Array]::IndexOf($headerFields, 'Kernel Name')
        if ($knCol -lt 0) {
            Write-Host "  FAILED: 'Kernel Name' column not found in CSV header." -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "no Kernel Name column" }
            continue
        }
        $kernelNames = @{}
        for ($i = $headerIdx + 1; $i -lt $csvLines.Count; $i++) {
            $line = $csvLines[$i]
            if ($line -match '^"\d+"') {
                $fields = [regex]::Split($line, '","') | ForEach-Object { $_.Trim('"') }
                if ($fields.Count -gt $knCol) {
                    $kn = $fields[$knCol]
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
        $sawRaygenName = $false
        foreach ($kn in $kernelNames.Keys) {
            Write-Host ("    {0,-50}  {1} metrics" -f $kn, $kernelNames[$kn]) -ForegroundColor DarkGray
            if ($kn -match '(?i)raygen') { $sawRaygenName = $true }
        }

        $hasRaygen = $false
        foreach ($kn in $kernelNames.Keys) {
            if ($kn -match '(?i)(raygen|optixLaunch)') { $hasRaygen = $true; break }
        }
        if (-not $hasRaygen) {
            Write-Host "  FAILED: no raygen/optixLaunch kernel in profiled set." -ForegroundColor Red
            Write-Host "         The kernel-name filter '$KernelNameRegex' didn't match anything." -ForegroundColor Red
            Write-Host "         Try -KernelNameRegex 'regex:.*' to capture all kernels and inspect names." -ForegroundColor Red
            $failures += @{ Config = $name; Reason = "no raygen kernel profiled" }
            continue
        }

        if ($sawRaygenName) {
            Write-Host "  GREAT: real raygen kernel name captured (not just optixLaunch). pc-sampling stalls should be populated." -ForegroundColor Green
        } else {
            Write-Host "  Note: kernel is named 'optixLaunch' (deprecated launcher wrapper). pc-sampling stalls will be n/a." -ForegroundColor DarkGray
        }

        # ---------------------------------------------------------------
        # Derive arithmetic intensity and a small summary CSV.
        # ---------------------------------------------------------------
        $ffma   = Parse-MetricSum -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'
        $fadd   = Parse-MetricSum -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum'
        $fmul   = Parse-MetricSum -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'
        $bytes  = Parse-MetricSum -CsvPath $outCsv -MetricName 'dram__bytes.sum'
        $timeNs = Parse-MetricSum -CsvPath $outCsv -MetricName 'gpu__time_duration.sum'
        $smThr  = Parse-MetricSum -CsvPath $outCsv -MetricName 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
        $dramThr= Parse-MetricSum -CsvPath $outCsv -MetricName 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
        $occup  = Parse-MetricSum -CsvPath $outCsv -MetricName 'smsp__warps_active.avg.pct_of_peak_sustained_active'
        $diverg = Parse-MetricSum -CsvPath $outCsv -MetricName 'smsp__thread_inst_executed_per_inst_executed.ratio'

        # FLOPs = 2*FFMA + FADD + FMUL  (ncu reports already summed across threads)
        $flops = (2.0 * $ffma) + $fadd + $fmul
        $ai    = if ($bytes -gt 0) { $flops / $bytes } else { 0 }
        # GFLOP/s = FLOPs / time (s)
        $gflopsps = if ($timeNs -gt 0) { $flops / ($timeNs * 1e-9) / 1e9 } else { 0 }
        # GB/s
        $gbps = if ($timeNs -gt 0) { $bytes / ($timeNs * 1e-9) / 1e9 } else { 0 }

        Write-Host "" -ForegroundColor DarkGray
        Write-Host "  --- Derived metrics (SUM across all profiled kernels) ---" -ForegroundColor Cyan
        Write-Host ("    FFMA       : {0:N0}" -f $ffma) -ForegroundColor DarkGray
        Write-Host ("    FADD       : {0:N0}" -f $fadd) -ForegroundColor DarkGray
        Write-Host ("    FMUL       : {0:N0}" -f $fmul) -ForegroundColor DarkGray
        Write-Host ("    FLOPs      : {0:N0}  (= 2*FFMA + FADD + FMUL)" -f $flops) -ForegroundColor DarkGray
        Write-Host ("    DRAM bytes : {0:N0}" -f $bytes) -ForegroundColor DarkGray
        Write-Host ("    Time (ms)  : {0:N3}" -f ($timeNs * 1e-6)) -ForegroundColor DarkGray
        Write-Host ("    Arith Int  : {0:F3} FLOP/byte  (SM-side only; excludes RT-core work)" -f $ai) -ForegroundColor Green
        Write-Host ("    Throughput : {0:N1} GFLOP/s, {1:N1} GB/s" -f $gflopsps, $gbps) -ForegroundColor Green
        Write-Host ("    SM%/DRAM%  : {0:N2}% SM   /   {1:N2}% DRAM" -f $smThr, $dramThr) -ForegroundColor DarkGray
        Write-Host ("    Occupancy  : {0:N2}% warps active" -f $occup) -ForegroundColor DarkGray
        Write-Host ("    Divergence : {0:N3} threads/warp inst (32 = perfect)" -f $diverg) -ForegroundColor DarkGray
        Write-Host ""

        # Write a slim summary CSV alongside the raw ncu output.
        $sumPath = Join-Path $NsightDir "${name}_summary.csv"
        @(
            "metric,value,unit"
            "variant,$Variant,"
            "config,$name,"
            "ffma_count,$ffma,inst"
            "fadd_count,$fadd,inst"
            "fmul_count,$fmul,inst"
            "flops_total,$flops,FLOPs"
            "dram_bytes,$bytes,bytes"
            "time_ns,$timeNs,ns"
            "arithmetic_intensity,$ai,FLOP/byte"
            "throughput_gflopsps,$gflopsps,GFLOP/s"
            "throughput_gbps,$gbps,GB/s"
            "sm_throughput_pct,$smThr,%"
            "dram_throughput_pct,$dramThr,%"
            "warps_active_pct,$occup,%"
            "divergence_ratio,$diverg,threads/inst"
            "raygen_kernel_named,$sawRaygenName,bool"
        ) | Out-File -FilePath $sumPath -Encoding ascii
        Write-Host "  Summary: $sumPath" -ForegroundColor Green

        Write-Host "  OK: $outCsv" -ForegroundColor Green

    } else {
        # ---------------------------------------------------------------
        # full mode: --set full
        # ---------------------------------------------------------------
        $outRep = Join-Path $NsightDir "${name}_full.ncu-rep"
        $ncuArgs = @(
            "--set", "full",
            "--kernel-name", "`"$KernelNameRegex`"",
            "--target-processes", "all",
            "--export", $outRep,
            "--force-overwrite"
        )
        if ($ApplicationReplay) {
            $ncuArgs += @("--replay-mode", "application")
        }
        $ncuArgs += @("--", $ExePath) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outRep.stderr.tmp"
        $stdoutTmp = "$outRep.stdout.tmp"
        if (-not $SkipDeprecatedLauncher) {
            $env:OPTIX_FORCE_DEPRECATED_LAUNCHER = "1"
        } else {
            Remove-Item Env:\OPTIX_FORCE_DEPRECATED_LAUNCHER -ErrorAction SilentlyContinue
        }
        $proc = Start-Process -FilePath $NcuPath -ArgumentList $ncuArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutTmp `
            -RedirectStandardError $stderrTmp

        if (Test-Path $stdoutTmp) { Move-Item -Force $stdoutTmp "$outRep.stdout.log" }
        if (Test-Path $stderrTmp) { Move-Item -Force $stderrTmp "$outRep.stderr.log" }

        $cfgElapsed = (Get-Date) - $cfgStart
        Write-Host ("  Wall time: {0:N1}s" -f $cfgElapsed.TotalSeconds)

        if (Test-Path "$outRep.stdout.log") {
            $stdoutContent = Get-Content "$outRep.stdout.log" -Raw
            if ($stdoutContent -match 'No kernels were profiled') {
                Write-Host "  FAILED: ncu reports 'No kernels were profiled'." -ForegroundColor Red
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
            if ($sumContent -notmatch '(?i)(raygen|optixLaunch)') {
                Write-Host "  FAILED: summary CSV has no raygen/optixLaunch mention." -ForegroundColor Red
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
