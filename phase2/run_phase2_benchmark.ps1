# =============================================================================
# Phase 2 benchmark grid runner
#
# Runs optix_smallpt.exe across the full benchmark grid, saves PPM + PNG renders,
# captures stdout/stderr per config to a log file, and appends timing rows to
# timings.csv in the format specified by plan.md.
#
# Run from VS 2022 Developer PowerShell (so cmake/cl are on PATH if you need to
# rebuild). Build is *not* triggered by this script - build first, then run.
#
# Usage:
#   cd C:\Users\hudsonre\527project\phase2
#   powershell -ExecutionPolicy Bypass -File .\run_phase2_benchmark.ps1
#
# Options:
#   -Force         Re-run configs even if their PNG already exists
#   -SkipPng       Don't convert PPM -> PNG (faster if you just want timings)
#   -ConfigFilter  Regex; only run configs whose name matches (e.g. "1080p")
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipPng,
    [string]$ConfigFilter = ""
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Paths - all relative to the script's own location so this works regardless
# of where it's invoked from.
# -----------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\optix_smallpt.exe"
$PtxPath     = Join-Path $ProjectRoot "build\shaders.ptx"
$ResultsDir  = Join-Path $ProjectRoot "results\phase2_optix_spheres"
$RendersDir  = Join-Path $ResultsDir  "renders"
$LogsDir     = Join-Path $ResultsDir  "logs"
$CsvPath     = Join-Path $ResultsDir  "timings.csv"

# -----------------------------------------------------------------------------
# Benchmark grid - tuples of (config_name, width, height, spp)
# config_name is used for filenames (PNG, PPM, log) so keep it filesystem-safe.
# -----------------------------------------------------------------------------
$Configs = @(
    @{ Name = "720p_256spp";   Width = 1280; Height = 720;  Spp = 256  },
    @{ Name = "720p_1024spp";  Width = 1280; Height = 720;  Spp = 1024 },
    @{ Name = "1080p_256spp";  Width = 1920; Height = 1080; Spp = 256  },
    @{ Name = "1080p_1024spp"; Width = 1920; Height = 1080; Spp = 1024 },
    @{ Name = "1080p_4096spp"; Width = 1920; Height = 1080; Spp = 4096 },
    @{ Name = "4k_256spp";     Width = 3840; Height = 2160; Spp = 256  },
    @{ Name = "4k_1024spp";    Width = 3840; Height = 2160; Spp = 1024 }
)

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
Write-Host "=== Phase 2 benchmark runner ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "Build first:  cd build; cmake --build . --config Release" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $PtxPath)) {
    Write-Host "ERROR: PTX not found at $PtxPath" -ForegroundColor Red
    Write-Host "PTX is generated as part of the build. Run cmake --build first." -ForegroundColor Red
    exit 1
}

# Verify the Python conversion path works once before we start, so we don't
# spend 30 minutes rendering and then fail to convert anything.
if (-not $SkipPng) {
    # Run the import; suppress all output via redirection to $null. We only
    # care about the exit code. Using cmd.exe avoids PowerShell's pipeline
    # semantics that conflate stderr writes with errors.
    & cmd.exe /c "python -c `"from PIL import Image`" >NUL 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: python with PIL not on PATH (needed for PPM->PNG conversion)" -ForegroundColor Red
        Write-Host "Install with: pip install pillow" -ForegroundColor Red
        Write-Host "Or re-run with -SkipPng to skip the conversion step." -ForegroundColor Red
        exit 1
    }
}

# -----------------------------------------------------------------------------
# Output dirs
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null

# -----------------------------------------------------------------------------
# CSV - create with header if missing. We don't truncate existing rows; the
# script appends, and a downstream merge step (or a quick pandas read) will
# pick up the latest run via the `run_timestamp` column.
# -----------------------------------------------------------------------------
if (-not (Test-Path $CsvPath)) {
    "phase,resolution,spp,time_ms,mrays_sec,notes,run_timestamp" | Out-File -FilePath $CsvPath -Encoding ascii
}

# Filter configs if requested
if ($ConfigFilter -ne "") {
    $Configs = $Configs | Where-Object { $_.Name -match $ConfigFilter }
    Write-Host "Filter '$ConfigFilter' matched $($Configs.Count) config(s)"
}

Write-Host "Configs to run: $($Configs.Count)"
Write-Host ""

# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------
$RunTimestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
$grandStart   = Get-Date
$failures     = @()

foreach ($cfg in $Configs) {
    $name   = $cfg.Name
    $width  = $cfg.Width
    $height = $cfg.Height
    $spp    = $cfg.Spp

    $ppmPath = Join-Path $RendersDir "$name.ppm"
    $pngPath = Join-Path $RendersDir "$name.png"
    $logPath = Join-Path $LogsDir    "$name.log"

    Write-Host "--- $name ($width x $height, $spp spp) ---" -ForegroundColor Yellow

    # Skip if PNG already exists and -Force not set
    if ((Test-Path $pngPath) -and (-not $Force)) {
        Write-Host "  PNG exists, skipping. Use -Force to re-run." -ForegroundColor DarkGray
        continue
    }

    $cfgStart = Get-Date

    # Invoke the renderer. Capture stdout+stderr together. The exe prints both
    # OptiX log lines and the timing summary to stdout/stderr; we want all of it.
    # Tee-Object writes to file *and* the console so you can watch progress.
    $exeArgs = @(
        "--width",  $width,
        "--height", $height,
        "--spp",    $spp,
        "--output", $ppmPath,
        "--ptx",    $PtxPath
    )

    # Header for the log so a single log file is self-describing
    $logHeader = @(
        "# Phase 2 benchmark: $name",
        "# Resolution: ${width}x${height}",
        "# SPP: $spp",
        "# Run timestamp: $RunTimestamp",
        "# Command: $ExePath $($exeArgs -join ' ')",
        "# ---"
    )
    $logHeader | Out-File -FilePath $logPath -Encoding ascii

    # ------------------------------------------------------------------
    # Run the exe. We use Start-Process with redirection rather than the
    # call operator + Tee-Object: PowerShell's pipeline treats stderr
    # output as error records when 2>&1 is used, and that trips try/catch
    # even on exit-code-0 runs (OptiX writes info-level log lines to
    # stderr). Process exit code is the only reliable success signal.
    # ------------------------------------------------------------------
    $stdoutTmp = "$logPath.stdout.tmp"
    $stderrTmp = "$logPath.stderr.tmp"

    $proc = Start-Process -FilePath $ExePath -ArgumentList $exeArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutTmp `
        -RedirectStandardError  $stderrTmp

    # Append both streams to the log file. We label each block so OptiX
    # info-level log lines (which go to stderr) don't masquerade as errors
    # when you read the log later.
    Add-Content -Path $logPath -Value "## stderr (OptiX log + diagnostics)"
    if (Test-Path $stderrTmp) { Get-Content $stderrTmp | Add-Content -Path $logPath }
    Add-Content -Path $logPath -Value "## stdout (program output)"
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | Add-Content -Path $logPath }

    # Echo program stdout to the console so progress is visible
    if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | ForEach-Object { Write-Host "  $_" } }

    Remove-Item $stdoutTmp,$stderrTmp -ErrorAction SilentlyContinue

    if ($proc.ExitCode -ne 0) {
        Write-Host "  FAILED: exit code $($proc.ExitCode) (see $logPath)" -ForegroundColor Red
        $failures += @{ Config = $name; Reason = "exit code $($proc.ExitCode)" }
        continue
    }

    # ------------------------------------------------------------------
    # Parse the timing line that the exe prints, of the form:
    #   CSV: optix_spheres,1280x720,256,499.99,471.86,sm_86
    # We re-emit it into our master CSV with a timestamp column.
    # ------------------------------------------------------------------
    $csvLine = Select-String -Path $logPath -Pattern "^CSV: " | Select-Object -Last 1
    if ($csvLine) {
        $payload = $csvLine.Line.Substring(5).Trim()  # strip "CSV: "
        "$payload,$RunTimestamp" | Out-File -FilePath $CsvPath -Append -Encoding ascii
        Write-Host "  Logged to timings.csv" -ForegroundColor Green
    } else {
        Write-Host "  WARN: no CSV line found in exe output - check $logPath" -ForegroundColor Yellow
        $failures += @{ Config = $name; Reason = "no CSV line in output" }
    }

    # ------------------------------------------------------------------
    # Convert PPM -> PNG. We don't delete the PPM - it's the canonical
    # float-correct output and disk is cheap.
    # ------------------------------------------------------------------
    if (-not $SkipPng) {
        if (Test-Path $ppmPath) {
            # Build a tiny inline script. Use cmd.exe redirection so stderr
            # noise from PIL doesn't confuse the parent shell. We rely solely
            # on the exit code from python to determine success.
            $pyCmd = "from PIL import Image; Image.open(r'$ppmPath').save(r'$pngPath')"
            # Escape embedded double-quotes for cmd's parser
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
Write-Host "Renders:  $RendersDir"
Write-Host "Logs:     $LogsDir"
Write-Host "Timings:  $CsvPath"

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures ($($failures.Count)):" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red
    }
    exit 1
}

# Print the timings CSV (most recent run only) so you can eyeball results
Write-Host ""
Write-Host "Timings (this run):" -ForegroundColor Cyan
Get-Content $CsvPath | Where-Object { $_ -match $RunTimestamp -or $_ -match "^phase," } | ForEach-Object {
    Write-Host "  $_"
}
