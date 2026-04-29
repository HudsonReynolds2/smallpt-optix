# =============================================================================
# Phase 0 (CPU smallpt) thread-count benchmark.
#
# Adapted to follow the phase 1 / phase 2 runner layout so build_charts.py and
# any downstream tooling can consume all three phases identically.
#
# Each invocation creates a new timestamped folder under
#   results\phase0_cpu_smallpt\runs\<timestamp>\
# containing renders/ (PPM + PNG), logs/, timings.csv, and run_info.txt.
# Older runs are never overwritten.
#
# Phase 0's varied axis is OMP_NUM_THREADS at fixed (resolution, spp). This is
# different from phase 1/2 which sweep (resolution, spp), but the timings.csv
# schema is identical -- "scene" carries the thread count (e.g.
# phase0_cpu_1t, phase0_cpu_24t) so all phases concatenate cleanly.
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase0
#   powershell -ExecutionPolicy Bypass -File .\bench.ps1
#
# Common invocations:
#   (no flags)                quick: spp=40, runs=3, default thread set
#   -Spp 400 -Runs 3          recommended for stable numbers (~5-15s / config)
#   -Threads "1,4,8,24"       custom thread sweep
#   -SkipPng                  skip PPM->PNG conversion
#   -ConfigFilter <regex>     only thread counts matching the regex
#   -Tag <name>               append a tag to the run folder name
# =============================================================================

[CmdletBinding()]
param(
    [int]$Spp           = 40,
    [int]$Runs          = 3,
    [string]$Threads    = "1,2,4,6,8,12,16,24",
    [string]$ConfigFilter = "",
    [string]$Tag        = "",
    [switch]$SkipPng,
    [int]$Width         = 1024,
    [int]$Height        = 768
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\smallpt.exe"
$RunsRoot    = Join-Path $ProjectRoot "results\phase0_cpu_smallpt\runs"

$RunStamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$RunFolder   = if ($Tag -ne "") { "${RunStamp}_$Tag" } else { $RunStamp }
$RunDir      = Join-Path $RunsRoot $RunFolder
$RendersDir  = Join-Path $RunDir   "renders"
$LogsDir     = Join-Path $RunDir   "logs"
$CsvPath     = Join-Path $RunDir   "timings.csv"
$RunInfoPath = Join-Path $RunDir   "run_info.txt"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
Write-Host "=== Phase 0 (CPU smallpt) benchmark runner ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Run folder:   $RunDir"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "Build first: powershell -ExecutionPolicy Bypass -File build.ps1" -ForegroundColor Red
    exit 1
}

if (-not $SkipPng) {
    & cmd.exe /c "python -c `"from PIL import Image`" >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: python+PIL not found, using -SkipPng automatically" -ForegroundColor Yellow
        $SkipPng = $true
    }
}

# -----------------------------------------------------------------------------
# Thread sweep configuration
# -----------------------------------------------------------------------------
$ThreadCounts = $Threads -split '\s*,\s*' | ForEach-Object { [int]$_ } | Where-Object { $_ -gt 0 }

if ($ConfigFilter -ne "") {
    $ThreadCounts = $ThreadCounts | Where-Object { "$_" -match $ConfigFilter }
}

if ($ThreadCounts.Count -eq 0) {
    Write-Host "ERROR: no thread counts to run (after filter)" -ForegroundColor Red
    exit 1
}

# -----------------------------------------------------------------------------
# Output dirs + metadata
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null

$gitHash = ""
try { $gitHash = & git -C $ProjectRoot rev-parse --short HEAD 2>$null } catch {}

$cpuName = "(unknown)"
try {
    $cpuName = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()
} catch {}

$totalLogicalCores = [Environment]::ProcessorCount

@(
    "Run timestamp:    $RunStamp"
    "Tag:              $Tag"
    "Exe:              $ExePath"
    "Git hash:         $gitHash"
    "CPU:              $cpuName"
    "Logical cores:    $totalLogicalCores"
    "Resolution:       ${Width}x${Height}"
    "SPP (requested):  $Spp"
    "SPP (actual):     $([int]([math]::Floor($Spp / 4) * 4))"
    "Thread counts:    $($ThreadCounts -join ', ')"
    "Runs per config:  $Runs"
) | Out-File -FilePath $RunInfoPath -Encoding ascii

# Same schema as phase 1/2 timings.csv
"phase,scene,resolution,spp,time_ms,mrays_sec,notes,run_timestamp" | Out-File -FilePath $CsvPath -Encoding ascii

Write-Host "Resolution:    ${Width}x${Height}"
Write-Host "SPP:           $Spp"
Write-Host "Threads:       $($ThreadCounts -join ', ')"
Write-Host "Runs/count:    $Runs"
Write-Host ""

# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------
$grandStart = Get-Date
$failures   = @()
$Summary    = @()

# Primary rays per render = w * h * (spp/4) * 4 = w * h * spp_aligned
$sppAligned   = [int]([math]::Floor($Spp / 4) * 4)
$primaryRays  = [double]$Width * [double]$Height * [double]$sppAligned
$resolution   = "${Width}x${Height}"

foreach ($T in $ThreadCounts) {
    $configName = "${T}t"
    $RunTimes   = @()
    $MraysRuns  = @()
    $RaysRuns   = @()

    Write-Host "--- $configName  ($T thread$(if ($T -ne 1){'s'}), ${Width}x${Height}, $Spp spp) ---" -ForegroundColor Yellow

    $logPath = Join-Path $LogsDir "$configName.log"

    @(
        "# Phase 0 benchmark: $configName"
        "# Threads: $T"
        "# Resolution: ${Width}x${Height}"
        "# SPP: $Spp (aligned: $sppAligned)"
        "# Runs: $Runs"
        "# Run: $RunStamp"
        "# CPU: $cpuName"
        "# Git: $gitHash"
        "# Command: $ExePath $Spp   (with OMP_NUM_THREADS=$T)"
        "# ---"
    ) | Out-File -FilePath $logPath -Encoding ascii

    for ($r = 1; $r -le $Runs; $r++) {
        # Each run renders into the run dir; the FINAL run's PPM is moved to
        # renders/<configName>.ppm.
        $tmpPpm = Join-Path $RunDir "image.ppm"
        if (Test-Path $tmpPpm) { Remove-Item $tmpPpm }

        # ProcessStartInfo so we can override OMP_NUM_THREADS in child env
        # reliably on Windows (Start-Process / & don't propagate $env: edits).
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName               = $ExePath
        $psi.Arguments              = "$Spp"
        $psi.WorkingDirectory       = $RunDir
        $psi.UseShellExecute        = $false
        $psi.RedirectStandardError  = $true
        $psi.RedirectStandardOutput = $true
        $psi.CreateNoWindow         = $true

        foreach ($entry in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
            $psi.Environment[$entry.Key] = $entry.Value
        }
        $psi.Environment["OMP_NUM_THREADS"] = "$T"

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = [System.Diagnostics.Process]::Start($psi)
        # ReadToEnd BEFORE WaitForExit -- avoids pipe-buffer deadlock
        $stderrText = $proc.StandardError.ReadToEnd()
        $stdoutText = $proc.StandardOutput.ReadToEnd()
        $proc.WaitForExit()
        $sw.Stop()

        Add-Content -Path $logPath -Value "## run $r stderr"
        Add-Content -Path $logPath -Value $stderrText
        if ($stdoutText.Trim() -ne "") {
            Add-Content -Path $logPath -Value "## run $r stdout"
            Add-Content -Path $logPath -Value $stdoutText
        }

        if ($proc.ExitCode -ne 0) {
            Write-Warning "  Run ${r} FAILED (exit $($proc.ExitCode))"
            $failures += @{ Config = $configName; Reason = "run $r exit $($proc.ExitCode)" }
            continue
        }

        $sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)

        # Parse RAYS_TOTAL (now primary-ray count). Cross-check against the
        # analytic primary-ray count -- if they disagree by more than 0.1%
        # something is wrong with the binary (likely an unpatched smallpt).
        $rays  = $null
        $mrays = $null
        if ($stderrText -match 'RAYS_TOTAL:\s*(\d+)') {
            $rays  = [double]$Matches[1]
            $mrays = [math]::Round($rays / $sec / 1e6, 2)

            $expected = $primaryRays
            if ($expected -gt 0) {
                $deltaPct = [math]::Abs($rays - $expected) / $expected * 100.0
                if ($deltaPct -gt 0.1) {
                    Write-Host ("  WARN: RAYS_TOTAL=$rays differs from analytic primary count $expected by {0:F2}%% -- binary may not be patched" -f $deltaPct) -ForegroundColor Yellow
                }
            }
        }

        $RunTimes  += $sec
        $MraysRuns += $mrays
        $RaysRuns  += $rays

        $mraysStr = if ($null -ne $mrays) { "$mrays" } else { "N/A" }
        Write-Host ("  Run {0}: {1:F3} s  |  {2} Mrays/s" -f $r, $sec, $mraysStr)

        # Save the LAST successful run's PPM for this thread count
        if ($r -eq $Runs -and (Test-Path $tmpPpm)) {
            $finalPpm = Join-Path $RendersDir "$configName.ppm"
            Move-Item -Force $tmpPpm $finalPpm
        }
    }

    if ($RunTimes.Count -eq 0) {
        Write-Host "  no successful runs for $configName" -ForegroundColor Red
        Write-Host ""
        continue
    }

    $avgSec = [math]::Round(($RunTimes | Measure-Object -Average).Average, 3)
    $minSec = [math]::Round(($RunTimes | Measure-Object -Minimum).Minimum, 3)

    $validMr = $MraysRuns | Where-Object { $null -ne $_ }
    $avgMr   = $null
    if ($validMr.Count -gt 0) {
        $avgMr = [math]::Round(($validMr | Measure-Object -Average).Average, 2)
    }

    $validRays = $RaysRuns | Where-Object { $null -ne $_ } | Select-Object -First 1
    $raysReported = if ($null -ne $validRays) { [int64]$validRays } else { [int64]$primaryRays }

    # Emit one timings.csv row per thread count (using AVG seconds).
    $avgMs    = [math]::Round($avgSec * 1000, 2)
    $note     = "threads=$T;runs=$Runs;min_s=$minSec;rays=$raysReported"
    $sceneTag = "phase0_cpu_${T}t"
    "phase0_cpu_baseline,$sceneTag,$resolution,$Spp,$avgMs,$avgMr,$note,$RunStamp" |
        Out-File -FilePath $CsvPath -Append -Encoding ascii

    $Summary += [PSCustomObject]@{
        Threads  = $T
        AvgSec   = $avgSec
        MinSec   = $minSec
        AvgMrays = $avgMr
        Speedup  = $null
    }

    $mrStr = if ($null -ne $avgMr) { "  avg_mrays=$avgMr" } else { "" }
    Write-Host ("  avg={0:F3}s  min={1:F3}s{2}" -f $avgSec, $minSec, $mrStr) -ForegroundColor Green

    # PPM -> PNG for this thread count's render
    if (-not $SkipPng) {
        $ppmPath = Join-Path $RendersDir "$configName.ppm"
        $pngPath = Join-Path $RendersDir "$configName.png"
        if (Test-Path $ppmPath) {
            $pyCmd = "from PIL import Image; Image.open(r'$ppmPath').save(r'$pngPath')"
            $pyCmdEscaped = $pyCmd.Replace('"', '\"')
            & cmd.exe /c "python -c `"$pyCmdEscaped`" >NUL 2>NUL"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  PNG: $pngPath" -ForegroundColor DarkGreen
            } else {
                Write-Host "  PNG conversion failed (exit $LASTEXITCODE)" -ForegroundColor Yellow
                $failures += @{ Config = $configName; Reason = "PNG conversion exit $LASTEXITCODE" }
            }
        } else {
            Write-Host "  WARN: PPM not produced at $ppmPath" -ForegroundColor Yellow
            $failures += @{ Config = $configName; Reason = "PPM not produced" }
        }
    }

    Write-Host ""
}

# Speedup relative to 1-thread (using min for cleanest curve)
$baseRow = $Summary | Where-Object { $_.Threads -eq 1 } | Select-Object -First 1
if ($null -ne $baseRow -and $baseRow.MinSec -gt 0) {
    $base = $baseRow.MinSec
    foreach ($row in $Summary) {
        $row.Speedup = [math]::Round($base / $row.MinSec, 2)
    }
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
$grandElapsed = (Get-Date) - $grandStart

Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ("{0,8}  {1,9}  {2,9}  {3,10}  {4,9}" -f "Threads", "Avg (s)", "Min (s)", "Mrays/s", "Speedup")
Write-Host ("{0,8}  {1,9}  {2,9}  {3,10}  {4,9}" -f "-------", "-------", "-------", "---------", "-------")
foreach ($row in $Summary) {
    $spStr = if ($null -ne $row.Speedup)  { "{0:F2}x" -f $row.Speedup  } else { "N/A" }
    $mrStr = if ($null -ne $row.AvgMrays) { "{0:F2}"  -f $row.AvgMrays } else { "N/A" }
    Write-Host ("{0,8}  {1,9:F3}  {2,9:F3}  {3,10}  {4,9}" -f $row.Threads, $row.AvgSec, $row.MinSec, $mrStr, $spStr)
}

Write-Host ""
Write-Host ("Total wall time: {0:N1}s" -f $grandElapsed.TotalSeconds)
Write-Host "Run folder:      $RunDir"
Write-Host "Timings CSV:     $CsvPath"
Write-Host "Renders:         $RendersDir"

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures ($($failures.Count)):" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red
    }
    exit 1
}
