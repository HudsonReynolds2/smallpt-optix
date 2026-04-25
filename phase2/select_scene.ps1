# =============================================================================
# select_scene.ps1 — choose which scene scene.h points at, then rebuild
#
# Usage:
#   .\select_scene.ps1 default
#   .\select_scene.ps1 default -NoBuild
#   .\select_scene.ps1                    # lists available scenes
#
# Scenes live in .\scenes\scene_<name>.h. This script copies the chosen one
# over .\scene.h (which is what main.cpp #includes), then runs cmake --build
# unless -NoBuild is passed.
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Position=0)] [string]$Name = "",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScenesDir   = Join-Path $ScriptDir "scenes"
$ActiveScene = Join-Path $ScriptDir "scene.h"
$BuildDir    = Join-Path $ScriptDir "build"

# List available scenes if no argument was provided
if ($Name -eq "") {
    Write-Host "Available scenes (in $ScenesDir):" -ForegroundColor Cyan
    Get-ChildItem $ScenesDir -Filter "scene_*.h" | ForEach-Object {
        $sn = $_.BaseName -replace '^scene_',''
        # Try to read SCENE_NAME marker from the file for a nicer listing
        $line = Select-String -Path $_.FullName -Pattern '^//\s*SCENE_NAME:\s*(.*)$' | Select-Object -First 1
        $label = if ($line) { $line.Matches[0].Groups[1].Value.Trim() } else { $sn }
        Write-Host ("  {0,-15} ({1})" -f $sn, $label)
    }
    Write-Host ""
    Write-Host "Active scene.h:" -ForegroundColor Cyan
    if (Test-Path $ActiveScene) {
        $line = Select-String -Path $ActiveScene -Pattern '^//\s*SCENE_NAME:\s*(.*)$' | Select-Object -First 1
        if ($line) {
            Write-Host ("  {0}" -f $line.Matches[0].Groups[1].Value.Trim())
        } else {
            Write-Host "  (no SCENE_NAME marker: unknown variant)"
        }
    } else {
        Write-Host "  (scene.h does not exist)"
    }
    exit 0
}

# Resolve the chosen scene file
$src = Join-Path $ScenesDir "scene_$Name.h"
if (-not (Test-Path $src)) {
    Write-Host "ERROR: scene '$Name' not found at $src" -ForegroundColor Red
    Write-Host "Available scenes:" -ForegroundColor Yellow
    Get-ChildItem $ScenesDir -Filter "scene_*.h" | ForEach-Object {
        Write-Host ("  {0}" -f ($_.BaseName -replace '^scene_',''))
    }
    exit 1
}

Write-Host "Selecting scene: $Name" -ForegroundColor Cyan
Write-Host "  source: $src"
Write-Host "  dest:   $ActiveScene"

Copy-Item -Path $src -Destination $ActiveScene -Force

if ($NoBuild) {
    Write-Host "Skipping rebuild (-NoBuild). You will need to rebuild manually." -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $BuildDir)) {
    Write-Host "ERROR: build directory not found at $BuildDir" -ForegroundColor Red
    Write-Host "Run cmake -B build first." -ForegroundColor Red
    exit 1
}

Write-Host "Building (Release)..." -ForegroundColor Cyan
& cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Done. Active scene is now: $Name" -ForegroundColor Green
