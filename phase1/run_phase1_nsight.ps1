# =============================================================================
# Nsight Compute runner for phase 1 cu-smallpt baseline.
#
# Adapted from run_nsight.ps1 (phase 3, OptiX) for the phase 1 vanilla CUDA
# kernel. The OptiX-specific machinery is stripped:
#   - No -PtxPath (phase 1 has no PTX).
#   - No OPTIX_FORCE_DEPRECATED_LAUNCHER environment variable. That env var
#     forces OptiX to use a launcher path that ncu can intercept; it has no
#     effect on a plain CUDA kernel like phase 1's, so we don't set it.
#   - No -ApplicationReplay, no -SkipDeprecatedLauncher. Same reason.
#   - Default -KernelNameRegex matches the CUDA kernel name in kernel.cu
#     ("kernel" inside namespace smallpt -> mangled name contains "kernel").
#     ncu will see something like "smallpt::kernel(const Sphere*, ...)".
#   - Phase 1 prints a "CSV: cuda_baseline,..." line; we don't parse it here
#     (run_phase1_benchmark.ps1 already does), but the rendered PPM is still
#     written so we can confirm the profiled run produced a real image.
#
# What is kept (because it's generic, not OptiX-specific):
#   - The full metric set: FFMA/FADD/FMUL counters, DRAM bytes, SM/DRAM%,
#     L1TEX%, L2 hit rate, occupancy, divergence, XU pipe.
#   - Sum-vs-weighted-mean aggregation discipline. Counters get summed,
#     percentages and ratios get time-weighted-mean'd. This is the SAME
#     correctness fix from the v4 phase 3 script (the v3 script summed
#     percentages and produced absurd numbers like "DRAM 174%").
#   - quick (default) mode: targeted metrics CSV + derived summary CSV.
#   - full mode: --set full .ncu-rep export for the Nsight Compute GUI.
#   - PPM->PNG conversion via Pillow.
#   - Dry-run mode for inspecting the ncu command line.
#
# REQUIREMENTS:
#   - ncu.exe must be on PATH or passed via -NcuPath. With CUDA Toolkit 12.8
#     the default install puts it at:
#       "C:\Program Files\NVIDIA Corporation\Nsight Compute <ver>\ncu.exe"
#   - First-time use needs GPU performance counters unlocked:
#       NVIDIA Control Panel -> Manage 3D settings -> Developer ->
#       Manage GPU Performance Counters -> "Allow access to all users"
#     If you see ERR_NVGPUCTRPERM, this is what's missing. Reboot if the
#     setting doesn't take effect.
#   - The exe must already be built. This script does NOT rebuild.
#   - Python with Pillow on PATH for PPM->PNG conversion (or pass -SkipPng).
#
# MODES:
#   quick (default) - one targeted-metrics capture on 1024x768_64spp.
#                     Note the spp is lower than phase 3's 512spp because
#                     phase 1 takes ~25 ms/spp at this resolution; ncu adds
#                     replay passes per metric so the effective wall time
#                     would be brutal at 512spp. 64 is plenty for stable
#                     metric numbers.
#   full            - --set full capture across 3 configs.
#                     ~5-15 min wall time per config because of metric
#                     replays. Output: <config>_full.ncu-rep -- open in the
#                     Nsight Compute GUI for the full section view.
#
# USAGE:
#   .\run_phase1_nsight.ps1
#   .\run_phase1_nsight.ps1 -Mode full
#   .\run_phase1_nsight.ps1 -Variant phase1_baseline_clean
#
# =============================================================================

[CmdletBinding()]
param(
    [string]$Variant = "phase1_baseline",

    [string]$ExePath = "",  # auto-resolved to .\build\Release\smallpt.exe

    [ValidateSet("quick","full")]
    [string]$Mode = "quick",

    [string]$RunDir = "",

    [string]$NcuPath = "ncu",

    # Default matches the CUDA kernel name in kernel.cu. The kernel is just
    # called "kernel" inside namespace smallpt; the mangled name will look
    # like "_ZN7smallpt6kernel..." or render as "smallpt::kernel(...)".
    # The substring "kernel" matches both. Pass "regex:.*" to capture all.
    [string]$KernelNameRegex = "regex:(?i)kernel",

    [switch]$SkipPng,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# -----------------------------------------------------------------------------
# Resolve paths
# -----------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Default exe path: build\Release\smallpt.exe under the project root
if ($ExePath -eq "") {
    $ExePath = Join-Path $ScriptDir "build\Release\smallpt.exe"
}
if (-not [System.IO.Path]::IsPathRooted($ExePath)) {
    $ExePath = Join-Path $ScriptDir $ExePath
}
$ExePath = [System.IO.Path]::GetFullPath($ExePath)

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found: $ExePath" -ForegroundColor Red
    Write-Host "Build first:" -ForegroundColor Red
    Write-Host "  cmake -S . -B build -G `"Visual Studio 17 2022`" -A x64" -ForegroundColor Red
    Write-Host "  cmake --build build --config Release" -ForegroundColor Red
    exit 2
}

try {
    $ncuVer = & $NcuPath --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
} catch {
    Write-Host "ERROR: cannot invoke ncu at '$NcuPath'." -ForegroundColor Red
    Write-Host "  Add Nsight Compute to PATH, or pass:" -ForegroundColor Red
    Write-Host "    -NcuPath 'C:\Program Files\NVIDIA Corporation\Nsight Compute <ver>\ncu.exe'" -ForegroundColor Red
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

Write-Host "=== Nsight Compute capture (phase 1) ===" -ForegroundColor Cyan
Write-Host "Variant:           $Variant"
Write-Host "Mode:              $Mode"
Write-Host "Exe:               $ExePath"
Write-Host "Kernel:            $KernelNameRegex"
Write-Host "Output:            $NsightDir"
Write-Host "Renders:           $RendersDir"
Write-Host ""

# Snapshot git hash if available
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

# Capture GPU name for metadata
$gpuName = "(unknown)"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
} catch {}

@(
    "Nsight capture timestamp:          $(Get-Date -Format 'o')"
    "Variant:                           $Variant"
    "Mode:                              $Mode"
    "Exe:                               $ExePath"
    "Kernel filter:                     $KernelNameRegex"
    "GPU:                               $gpuName"
    "Host:                              $env:COMPUTERNAME"
    "User:                              $env:USERNAME"
) | Out-File -FilePath (Join-Path $NsightDir "info.txt") -Encoding ascii

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
# Phase 1 is much slower per spp than phase 3 (no BVH, O(N) sphere loop), so
# we use lower spp here than the phase 3 nsight script. ncu replays the
# kernel once per metric, which already adds 5-10x to wall time, and the
# numbers are stable at low spp anyway.
if ($Mode -eq "quick") {
    $Configs = @(
        @{ Name = "1024x768_64spp"; Width = 1024; Height = 768; Spp = 64 }
    )
} else {
    $Configs = @(
        @{ Name = "512x384_32spp";   Width = 512;  Height = 384;  Spp = 32 },
        @{ Name = "1024x768_32spp";  Width = 1024; Height = 768;  Spp = 32 },
        @{ Name = "2048x1536_16spp"; Width = 2048; Height = 1536; Spp = 16 }
    )
}

# Metrics for the deck. Why each one is here:
#
#   gpu__time_duration.sum                                    : kernel time
#   sm__throughput.avg.pct_of_peak_sustained_elapsed          : SM utilization
#   smsp__warps_active.avg.pct_of_peak_sustained_active       : occupancy
#   smsp__thread_inst_executed_per_inst_executed.ratio        : divergence
#   dram__throughput.avg.pct_of_peak_sustained_elapsed        : DRAM utilization (memory pressure)
#   dram__bytes.sum                                           : actual bytes hit DRAM (denominator for AI)
#   l1tex__throughput.avg.pct_of_peak_sustained_active        : L1/TEX pressure
#   lts__t_sectors.sum / lts__t_sectors_lookup_hit.sum        : L2 hit rate
#   sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active: transcendental pipe usage
#   smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg.*    : memory stall reason
#   smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg.*   : compute stall reason
#   smsp__sass_thread_inst_executed_op_ffma_pred_on.sum       : FFMA = 2 FLOPs/inst (numerator for AI)
#   smsp__sass_thread_inst_executed_op_fadd_pred_on.sum       : FADD = 1 FLOP/inst
#   smsp__sass_thread_inst_executed_op_fmul_pred_on.sum       : FMUL = 1 FLOP/inst
#
# NOTE: phase 1 uses double precision throughout; FFMA/FADD/FMUL counters
# are SP-only on most architectures. The DP equivalents are *_dadd_pred_on /
# *_dmul_pred_on / *_dfma_pred_on -- we add those too so AI is computable.
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
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
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

# Helper: parse all per-invocation values for a metric out of the ncu CSV.
# The CSV has columns: ID, Process ID, ..., Metric Name, Metric Unit, Metric Value.
# Phase 1 has only one kernel launch per render, so we expect one row per
# metric. But the parsing tolerates multiple rows (in case ncu does multiple
# replays for a metric set, which it can).
#
# IMPORTANT: ncu emits per-invocation values, not pre-aggregated ones. Naive
# summation is correct for COUNTERS (FFMA, FADD, FMUL, dram__bytes.sum,
# anything ending in .sum) but WRONG for percentages and ratios. Use
# Parse-MetricMean for those.
function Parse-MetricRows {
    param([string]$CsvPath, [string]$MetricName)
    if (-not (Test-Path $CsvPath)) { return @() }

    $lines = Get-Content $CsvPath
    $headerIdx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^"ID"') { $headerIdx = $i; break }
    }
    if ($headerIdx -lt 0) { return @() }

    $hdr = [regex]::Split($lines[$headerIdx], '","') | ForEach-Object { $_.Trim('"') }
    $idCol    = [Array]::IndexOf($hdr, 'ID')
    $nameCol  = [Array]::IndexOf($hdr, 'Metric Name')
    $valueCol = [Array]::IndexOf($hdr, 'Metric Value')
    if ($nameCol -lt 0 -or $valueCol -lt 0) { return @() }

    $rows = @()
    for ($i = $headerIdx + 1; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if (-not ($line -match '^"\d+"')) { continue }
        $f = [regex]::Split($line, '","') | ForEach-Object { $_.Trim('"') }
        if ($f.Count -le $valueCol) { continue }
        if ($f[$nameCol] -ne $MetricName) { continue }

        # Strip thousand separators ("13,701,348,177") that ncu emits with --csv
        $raw = $f[$valueCol] -replace ',', ''
        $val = 0.0
        if ([double]::TryParse($raw, [ref]$val)) {
            $rows += [PSCustomObject]@{ Id = $f[$idCol]; Value = $val }
        }
    }
    return $rows
}

# Sum across invocations. Use for counters (.sum metrics, byte counts,
# instruction counts).
#
# PowerShell strict-mode quirk: when a function returns a single object,
# PowerShell unwraps it to a scalar and `.Count` blows up. The @(...) array
# subexpression around the call forces a real array even for 0/1 elements,
# which is what the rest of the function assumes. Phase 3 didn't trip this
# because it had multiple tile launches and so always >=4 rows.
function Parse-MetricSum {
    param([string]$CsvPath, [string]$MetricName)
    $rows = @(Parse-MetricRows -CsvPath $CsvPath -MetricName $MetricName)
    if ($rows.Count -eq 0) { return 0.0 }
    return ($rows | Measure-Object -Property Value -Sum).Sum
}

# Time-weighted mean across invocations. Use for percentages and ratios.
# Pulls gpu__time_duration.sum from the same CSV for the weight; if absent,
# falls back to plain arithmetic mean.
function Parse-MetricMean {
    param([string]$CsvPath, [string]$MetricName)
    $rows = @(Parse-MetricRows -CsvPath $CsvPath -MetricName $MetricName)
    if ($rows.Count -eq 0) { return 0.0 }

    $timeRows = @(Parse-MetricRows -CsvPath $CsvPath -MetricName 'gpu__time_duration.sum')
    if ($timeRows.Count -eq 0) {
        return ($rows | Measure-Object -Property Value -Average).Average
    }

    $timeMap = @{}
    foreach ($r in $timeRows) { $timeMap[$r.Id] = $r.Value }

    $weightedSum = 0.0
    $totalWeight = 0.0
    foreach ($r in $rows) {
        $w = if ($timeMap.ContainsKey($r.Id)) { $timeMap[$r.Id] } else { 0.0 }
        $weightedSum += $w * $r.Value
        $totalWeight += $w
    }
    if ($totalWeight -le 0) {
        return ($rows | Measure-Object -Property Value -Average).Average
    }
    return $weightedSum / $totalWeight
}

# -----------------------------------------------------------------------------
# Capture loop
# -----------------------------------------------------------------------------
$grandStart = Get-Date
$failures = @()

foreach ($cfg in $Configs) {
    $name   = $cfg.Name
    $width  = $cfg.Width
    $height = $cfg.Height
    $spp    = $cfg.Spp

    Write-Host "--- $name ($width x $height, $spp spp) ---" -ForegroundColor Yellow

    $ppmPath = Join-Path $RendersDir "${name}.ppm"
    $pngPath = Join-Path $RendersDir "${name}.png"

    # Phase 1 CLI: only --width / --height / --spp / --output
    $appArgs = @(
        "--width",  $width,
        "--height", $height,
        "--spp",    $spp,
        "--output", $ppmPath
    )

    if ($Mode -eq "quick") {
        $outCsv = Join-Path $NsightDir "${name}_metrics.csv"
        $ncuArgs = @(
            "--metrics", $Metrics,
            "--kernel-name", "`"$KernelNameRegex`"",
            "--csv",
            "--target-processes", "all"
        )
        $ncuArgs += @("--", $ExePath) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outCsv.stderr.tmp"
        $stdoutTmp = "$outCsv.stdout.tmp"

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

        # Detect "no kernels were profiled" up front
        $csvRaw = Get-Content $outCsv -Raw
        if ($csvRaw -match 'No kernels were profiled') {
            Write-Host "  FAILED: ncu reports 'No kernels were profiled'." -ForegroundColor Red
            Write-Host "         The kernel-name filter '$KernelNameRegex' didn't match." -ForegroundColor Red
            Write-Host "         Try -KernelNameRegex 'regex:.*' to capture all kernels and inspect names." -ForegroundColor Red
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
        foreach ($kn in $kernelNames.Keys) {
            Write-Host ("    {0,-50}  {1} metrics" -f $kn, $kernelNames[$kn]) -ForegroundColor DarkGray
        }

        # ---------------------------------------------------------------
        # Derive arithmetic intensity and a small summary CSV.
        #
        # AGGREGATION DISCIPLINE:
        #   - Counters (.sum metrics, byte/inst counts) -> Parse-MetricSum
        #   - Percentages (*.pct_of_peak_*) and ratios (*.ratio)
        #     -> Parse-MetricMean (time-weighted across invocations)
        # ---------------------------------------------------------------
        # SP FLOP counters (probably zero in phase 1 -- it's all double)
        $ffma   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'
        $fadd   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum'
        $fmul   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'
        # DP FLOP counters (this is where the actual numbers live for phase 1)
        $dfma   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum'
        $dadd   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum'
        $dmul   = Parse-MetricSum  -CsvPath $outCsv -MetricName 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum'

        $bytes  = Parse-MetricSum  -CsvPath $outCsv -MetricName 'dram__bytes.sum'
        $timeNs = Parse-MetricSum  -CsvPath $outCsv -MetricName 'gpu__time_duration.sum'
        $ltsHit = Parse-MetricSum  -CsvPath $outCsv -MetricName 'lts__t_sectors_lookup_hit.sum'
        $ltsTot = Parse-MetricSum  -CsvPath $outCsv -MetricName 'lts__t_sectors.sum'
        $smThr  = Parse-MetricMean -CsvPath $outCsv -MetricName 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
        $dramThr= Parse-MetricMean -CsvPath $outCsv -MetricName 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
        $l1Thr  = Parse-MetricMean -CsvPath $outCsv -MetricName 'l1tex__throughput.avg.pct_of_peak_sustained_active'
        $xuPct  = Parse-MetricMean -CsvPath $outCsv -MetricName 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active'
        $occup  = Parse-MetricMean -CsvPath $outCsv -MetricName 'smsp__warps_active.avg.pct_of_peak_sustained_active'
        $diverg = Parse-MetricMean -CsvPath $outCsv -MetricName 'smsp__thread_inst_executed_per_inst_executed.ratio'

        # FLOPs = 2*FFMA + FADD + FMUL + 2*DFMA + DADD + DMUL
        # (FMA counts as 2 FLOPs because it's a multiply-add)
        $spFlops = (2.0 * $ffma) + $fadd + $fmul
        $dpFlops = (2.0 * $dfma) + $dadd + $dmul
        $flops   = $spFlops + $dpFlops
        $ai      = if ($bytes -gt 0) { $flops / $bytes } else { 0 }
        $gflopsps = if ($timeNs -gt 0) { $flops / ($timeNs * 1e-9) / 1e9 } else { 0 }
        $gbps     = if ($timeNs -gt 0) { $bytes / ($timeNs * 1e-9) / 1e9 } else { 0 }
        $l2hit    = if ($ltsTot -gt 0) { ($ltsHit / $ltsTot) * 100.0 } else { 0 }

        Write-Host "" -ForegroundColor DarkGray
        Write-Host "  --- Derived metrics (sum-for-counters, weighted-mean-for-rates) ---" -ForegroundColor Cyan
        Write-Host ("    FFMA       : {0:N0}  (SP)" -f $ffma) -ForegroundColor DarkGray
        Write-Host ("    FADD       : {0:N0}  (SP)" -f $fadd) -ForegroundColor DarkGray
        Write-Host ("    FMUL       : {0:N0}  (SP)" -f $fmul) -ForegroundColor DarkGray
        Write-Host ("    DFMA       : {0:N0}  (DP)" -f $dfma) -ForegroundColor DarkGray
        Write-Host ("    DADD       : {0:N0}  (DP)" -f $dadd) -ForegroundColor DarkGray
        Write-Host ("    DMUL       : {0:N0}  (DP)" -f $dmul) -ForegroundColor DarkGray
        Write-Host ("    FLOPs (SP) : {0:N0}" -f $spFlops) -ForegroundColor DarkGray
        Write-Host ("    FLOPs (DP) : {0:N0}" -f $dpFlops) -ForegroundColor DarkGray
        Write-Host ("    FLOPs total: {0:N0}" -f $flops) -ForegroundColor DarkGray
        Write-Host ("    DRAM bytes : {0:N0}" -f $bytes) -ForegroundColor DarkGray
        Write-Host ("    Time (ms)  : {0:N3}" -f ($timeNs * 1e-6)) -ForegroundColor DarkGray
        if ($flops -gt 0) {
            Write-Host ("    Arith Int  : {0:F3} FLOP/byte" -f $ai) -ForegroundColor Green
            Write-Host ("    Throughput : {0:N1} GFLOP/s, {1:N1} GB/s" -f $gflopsps, $gbps) -ForegroundColor Green
        } else {
            Write-Host "    Arith Int  : NOT COMPUTABLE -- FLOP counters all zero." -ForegroundColor Yellow
            Write-Host ("    Throughput : {0:N1} GB/s sustained DRAM" -f $gbps) -ForegroundColor Green
        }
        Write-Host ("    SM%/DRAM%  : {0:N2}% SM   /   {1:N2}% DRAM   (peak-sustained)" -f $smThr, $dramThr) -ForegroundColor DarkGray
        Write-Host ("    L1TEX%     : {0:N2}%" -f $l1Thr) -ForegroundColor DarkGray
        Write-Host ("    XU%        : {0:N2}%  (transcendental pipe)" -f $xuPct) -ForegroundColor DarkGray
        Write-Host ("    L2 hit     : {0:N2}%  ({1:N0} of {2:N0} sectors)" -f $l2hit, $ltsHit, $ltsTot) -ForegroundColor DarkGray
        Write-Host ("    Occupancy  : {0:N2}% warps active" -f $occup) -ForegroundColor DarkGray
        Write-Host ("    Divergence : {0:N3} threads/warp inst (32 = perfect; {1:N0}% utilization)" -f $diverg, ($diverg/32*100)) -ForegroundColor DarkGray
        Write-Host ""

        # Write a slim summary CSV alongside the raw ncu output.
        $sumPath = Join-Path $NsightDir "${name}_summary.csv"
        @(
            "metric,value,unit,aggregation"
            "variant,$Variant,,"
            "config,$name,,"
            "ffma_count,$ffma,inst,sum"
            "fadd_count,$fadd,inst,sum"
            "fmul_count,$fmul,inst,sum"
            "dfma_count,$dfma,inst,sum"
            "dadd_count,$dadd,inst,sum"
            "dmul_count,$dmul,inst,sum"
            "sp_flops,$spFlops,FLOPs,derived"
            "dp_flops,$dpFlops,FLOPs,derived"
            "flops_total,$flops,FLOPs,derived"
            "dram_bytes,$bytes,bytes,sum"
            "time_ns,$timeNs,ns,sum"
            "arithmetic_intensity,$ai,FLOP/byte,derived"
            "throughput_gflopsps,$gflopsps,GFLOP/s,derived"
            "throughput_gbps,$gbps,GB/s,derived"
            "sm_throughput_pct,$smThr,%,weighted_mean"
            "dram_throughput_pct,$dramThr,%,weighted_mean"
            "l1tex_throughput_pct,$l1Thr,%,weighted_mean"
            "xu_pipe_pct,$xuPct,%,weighted_mean"
            "l2_hit_pct,$l2hit,%,derived"
            "warps_active_pct,$occup,%,weighted_mean"
            "divergence_ratio,$diverg,threads/inst,weighted_mean"
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
        $ncuArgs += @("--", $ExePath) + $appArgs

        Write-Host "  ncu $($ncuArgs -join ' ')" -ForegroundColor DarkGray
        if ($DryRun) { Write-Host "  (dry run -- skipping)" -ForegroundColor DarkGray; continue }

        $cfgStart = Get-Date
        $stderrTmp = "$outRep.stderr.tmp"
        $stdoutTmp = "$outRep.stdout.tmp"

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
