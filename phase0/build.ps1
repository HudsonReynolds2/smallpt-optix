# build.ps1 — Configure and build smallpt with Visual Studio 2022
# Run from the smallpt-bench directory:
#   powershell -ExecutionPolicy Bypass -File build.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$BuildDir = "build"

Write-Host "=== smallpt build ===" -ForegroundColor Cyan

# Remove stale build dir if requested
if ($args -contains "--clean" -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning $BuildDir ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Configure
Write-Host "`nConfiguring..." -ForegroundColor Cyan
cmake -S . -B $BuildDir `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

# Build
Write-Host "`nBuilding (Release)..." -ForegroundColor Cyan
cmake --build $BuildDir --config Release --parallel

if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

$Exe = Join-Path $BuildDir "Release\smallpt.exe"
if (Test-Path $Exe) {
    Write-Host "`nBuild succeeded: $Exe" -ForegroundColor Green
} else {
    Write-Error "Executable not found at expected path: $Exe"
}
