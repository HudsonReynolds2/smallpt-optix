# =============================================================================
# Phase 3 correctness suite (top-level orchestrator).
#
# Runs a canonical set of configurations, then for each output PNG:
#   1. Compares to the 4K reference render (downscaled) -> SSIM threshold
#   2. Runs invariant checks (no NaN, energy bounds, Cornell color sanity)
#
# Emits one PASS/FAIL line at the end with a per-check breakdown.
#
# Run from the phase3 directory:
#     cd C:\Users\hudsonre\527project\phase3
#     powershell -ExecutionPolicy Bypass -File .\correctness\run_correctness.ps1
#
# CI-style failure semantics: exit 0 only if EVERY check on EVERY render
# passes. Any failure -> exit 1 with details.
#
# Common invocations:
#   .\correctness\run_correctness.ps1                  # quick set + all checks
#   .\correctness\run_correctness.ps1 -SkipBuild       # don't rebuild first
#   .\correctness\run_correctness.ps1 -Quick           # minimal: 512x384_64spp only
#   .\correctness\run_correctness.ps1 -ReuseRunDir <path>  # don't re-render, just check
# =============================================================================

[CmdletBinding()]
param(
    # Which set of configs to render and check. Default: a 2-config quick set
    # that finishes in ~30 seconds total.
    [switch]$Quick,
    [switch]$Full,

    # Skip the build step. Use when the exe is already up to date.
    [switch]$SkipBuild,

    # Reuse renders from an existing run dir (path containing renders/ subfolder).
    # When set, no new renders are produced -- we just run checks against
    # whatever is already in <ReuseRunDir>/renders/.
    [string]$ReuseRunDir = "",

    # SSIM threshold for the reference comparison. Default 0.96 because we
    # compare downscaled-from-4K-reference to lower-spp renders, which is
    # noise-limited even on correct output. For matched-spp comparisons
    # (1024 spp test vs 1024 spp reference slice) raise to 0.98.
    [double]$SsimThreshold = 0.96,

    # Override the reference image. Default is phase3/reference/4096x3072_4096spp.png.
    [string]$ReferenceImage = "",

    # Override which python is used. Default: just "python" on PATH.
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
# Script lives in correctness/ ; project root is its parent
$ProjectRoot = Split-Path -Parent $ScriptDir
$ExePath     = Join-Path $ProjectRoot "build\Release\optix_smallpt.exe"
$PtxPath     = Join-Path $ProjectRoot "build\shaders.ptx"
$SceneHPath  = Join-Path $ProjectRoot "src\scene.h"
$RefDefault  = Join-Path $ProjectRoot "reference\4096x3072_4096spp.png"
$BenchScript = Join-Path $ProjectRoot "select_scene.ps1"

if ($ReferenceImage -eq "") { $ReferenceImage = $RefDefault }

$SsimScript       = Join-Path $ScriptDir "ssim_compare.py"
$RegressionScript = Join-Path $ScriptDir "render_regression.py"
$InvariantsScript = Join-Path $ScriptDir "check_render_invariants.py"

# Run dir
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunsRoot = Join-Path $ProjectRoot "results\correctness\runs"
if ($ReuseRunDir -ne "") {
    $RunDir = (Resolve-Path $ReuseRunDir -ErrorAction SilentlyContinue)
    if (-not $RunDir) {
        Write-Host "ERROR: -ReuseRunDir '$ReuseRunDir' does not exist" -ForegroundColor Red
        exit 2
    }
    $RunDir = $RunDir.Path
} else {
    $RunDir = Join-Path $RunsRoot $RunStamp
}
$RendersDir = Join-Path $RunDir "renders"
$ReportsDir = Join-Path $RunDir "reports"
$LogsDir    = Join-Path $RunDir "logs"

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
$QuickConfigs = @(
    @{ Name = "512x384_64spp"; Width = 512; Height = 384; Spp = 64 }
)
$DefaultConfigs = @(
    @{ Name = "512x384_64spp";   Width = 512;  Height = 384; Spp = 64  },
    @{ Name = "1024x768_256spp"; Width = 1024; Height = 768; Spp = 256 }
)
$FullConfigs = @(
    @{ Name = "512x384_64spp";    Width = 512;  Height = 384;  Spp = 64   },
    @{ Name = "1024x768_256spp";  Width = 1024; Height = 768;  Spp = 256  },
    @{ Name = "1024x768_1024spp"; Width = 1024; Height = 768;  Spp = 1024 },
    @{ Name = "2048x1536_256spp"; Width = 2048; Height = 1536; Spp = 256  }
)

if ($Quick)      { $Configs = $QuickConfigs }
elseif ($Full)   { $Configs = $FullConfigs }
else             { $Configs = $DefaultConfigs }

# -----------------------------------------------------------------------------
# Sanity
# -----------------------------------------------------------------------------
Write-Host "=== Phase 3 correctness suite ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Run dir:      $RunDir"
Write-Host "Reference:    $ReferenceImage"
Write-Host "SSIM thresh:  $SsimThreshold"
Write-Host "Configs:      $($Configs.Count)"
Write-Host ""

if (-not (Test-Path $ReferenceImage)) {
    Write-Host "ERROR: reference image not found: $ReferenceImage" -ForegroundColor Red
    Write-Host "       Pass -ReferenceImage <path> or place the 4K render at the default location." -ForegroundColor Red
    exit 2
}
foreach ($s in @($SsimScript, $RegressionScript, $InvariantsScript)) {
    if (-not (Test-Path $s)) { Write-Host "ERROR: missing helper script: $s" -ForegroundColor Red; exit 2 }
}

# Verify python deps once up front rather than discovering missing ones mid-run.
& $Python -c "import numpy, PIL, skimage" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: python deps missing. Install with:" -ForegroundColor Red
    Write-Host "       pip install numpy pillow scikit-image" -ForegroundColor Red
    exit 2
}

# -----------------------------------------------------------------------------
# Build (skipped if -SkipBuild or -ReuseRunDir set)
# -----------------------------------------------------------------------------
if ($ReuseRunDir -eq "" -and -not $SkipBuild) {
    if (-not (Test-Path $BenchScript)) {
        Write-Host "WARN: select_scene.ps1 not found at $BenchScript; skipping rebuild." -ForegroundColor Yellow
    } else {
        Write-Host "--- Rebuilding (default scene) ---" -ForegroundColor Yellow
        & powershell -ExecutionPolicy Bypass -File $BenchScript -Name default
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: rebuild failed (select_scene.ps1 exit $LASTEXITCODE)" -ForegroundColor Red
            exit 1
        }
        Write-Host ""
    }
}

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: exe not found at $ExePath" -ForegroundColor Red
    Write-Host "       Run without -SkipBuild, or build manually first." -ForegroundColor Red
    exit 2
}

# -----------------------------------------------------------------------------
# Output dirs
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RendersDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir    | Out-Null

# -----------------------------------------------------------------------------
# Render phase (skipped if -ReuseRunDir set)
# -----------------------------------------------------------------------------
$renderFailures = @()

if ($ReuseRunDir -eq "") {
    Write-Host "--- Rendering $($Configs.Count) config(s) ---" -ForegroundColor Yellow

    foreach ($cfg in $Configs) {
        $name   = $cfg.Name
        $width  = $cfg.Width
        $height = $cfg.Height
        $spp    = $cfg.Spp
        $ppm    = Join-Path $RendersDir "$name.ppm"
        $png    = Join-Path $RendersDir "$name.png"
        $log    = Join-Path $LogsDir    "$name.log"

        Write-Host ("  [{0}] rendering..." -f $name)

        $exeArgs = @(
            "--width", $width, "--height", $height, "--spp", $spp,
            "--output", $ppm, "--ptx", $PtxPath
        )
        $stdoutTmp = "$log.stdout"
        $stderrTmp = "$log.stderr"
        $proc = Start-Process -FilePath $ExePath -ArgumentList $exeArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutTmp -RedirectStandardError $stderrTmp

        if (Test-Path $stderrTmp) { Get-Content $stderrTmp | Add-Content $log }
        if (Test-Path $stdoutTmp) { Get-Content $stdoutTmp | Add-Content $log }
        Remove-Item $stdoutTmp,$stderrTmp -ErrorAction SilentlyContinue

        if ($proc.ExitCode -ne 0) {
            Write-Host ("    FAIL: exe exited with {0}" -f $proc.ExitCode) -ForegroundColor Red
            $renderFailures += @{ Config = $name; Reason = "exe exit $($proc.ExitCode)" }
            continue
        }
        if (-not (Test-Path $ppm)) {
            Write-Host "    FAIL: PPM not produced" -ForegroundColor Red
            $renderFailures += @{ Config = $name; Reason = "no PPM" }
            continue
        }

        # Convert PPM -> PNG (Pillow). The check tools accept both, but PNG
        # is faster to load repeatedly and what humans expect to see.
        $pyCmd = "from PIL import Image; Image.open(r'$ppm').save(r'$png')"
        & $Python -c $pyCmd 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    WARN: PNG conversion failed; using PPM directly" -ForegroundColor Yellow
            $png = $ppm  # fall back so checks still run
        }
    }
    Write-Host ""
}

# -----------------------------------------------------------------------------
# Check phase: invariants + reference regression for each render.
# -----------------------------------------------------------------------------
Write-Host "--- Running checks ---" -ForegroundColor Yellow
Write-Host ""

# Single regression sweep across all renders in $RendersDir
$regressionCsv = Join-Path $ReportsDir "regression.csv"
$regressionLog = Join-Path $ReportsDir "regression.log"

Write-Host "[regression] all renders vs reference (SSIM >= $SsimThreshold)"
& $Python $RegressionScript `
    --reference $ReferenceImage `
    --renders $RendersDir `
    --report-csv $regressionCsv `
    --threshold $SsimThreshold `
    --match "*.png" *> $regressionLog
$regressionExit = $LASTEXITCODE
Get-Content $regressionLog | ForEach-Object { Write-Host "  $_" }
Write-Host ""

# Per-render invariant checks
$invariantFailures = @()
$pngs = @(Get-ChildItem -Path $RendersDir -Filter "*.png" -File -ErrorAction SilentlyContinue)
if ($pngs.Count -eq 0) {
    # Fall back to PPMs if PNG conversion didn't happen
    $pngs = @(Get-ChildItem -Path $RendersDir -Filter "*.ppm" -File -ErrorAction SilentlyContinue)
}

foreach ($img in $pngs) {
    $name = $img.BaseName
    $jsonOut = Join-Path $ReportsDir "$name.invariants.json"
    Write-Host "[invariants] $($img.Name)"
    & $Python $InvariantsScript $img.FullName --json $jsonOut --quiet
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        $invariantFailures += @{ Config = $name; Reason = "invariant check failed (exit $rc)" }

        # Re-run verbose so the user can see WHICH check failed without
        # having to grep the JSON. This adds maybe 30ms per failure -- nbd.
        Write-Host "  --- detail ---" -ForegroundColor Yellow
        & $Python $InvariantsScript $img.FullName | ForEach-Object { Write-Host "    $_" }
    }
}

Write-Host ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
$totalFailures = $renderFailures.Count + $invariantFailures.Count + ([int]($regressionExit -ne 0))

Write-Host "=== Correctness summary ===" -ForegroundColor Cyan
Write-Host ("  renders:    {0} attempted, {1} failed" -f $Configs.Count, $renderFailures.Count)
Write-Host ("  regression: {0} (see $regressionCsv)" -f $(if ($regressionExit -eq 0) { 'PASS' } else { 'FAIL' }))
Write-Host ("  invariants: {0} renders checked, {1} failed" -f $pngs.Count, $invariantFailures.Count)
Write-Host ""
if ($renderFailures.Count -gt 0) {
    Write-Host "Render failures:" -ForegroundColor Red
    foreach ($f in $renderFailures) { Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red }
}
if ($invariantFailures.Count -gt 0) {
    Write-Host "Invariant failures:" -ForegroundColor Red
    foreach ($f in $invariantFailures) { Write-Host "  $($f.Config): $($f.Reason)" -ForegroundColor Red }
}

if ($totalFailures -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host ("FAILED: {0} total failure(s). See $ReportsDir for details." -f $totalFailures) -ForegroundColor Red
    exit 1
}
