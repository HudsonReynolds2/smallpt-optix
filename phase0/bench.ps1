# bench.ps1 - Thread-count sweep benchmark for smallpt
# Usage:
#   powershell -ExecutionPolicy Bypass -File bench.ps1 -Spp 40 -Runs 1
#   powershell -ExecutionPolicy Bypass -File bench.ps1 -Spp 1000 -Runs 3 -OutCsv results.csv

param(
    [int]$Spp       = 40,
    [int]$Runs      = 3,
    [string]$OutCsv = "results.csv"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Exe = Join-Path $ScriptDir "build\Release\smallpt.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Executable not found at $Exe - run build.ps1 first"
    exit 1
}

$ThreadCounts = @(1, 2, 4, 6, 8, 12, 16, 24)

Write-Host "=== smallpt thread-count benchmark ===" -ForegroundColor Cyan
Write-Host "Executable : $Exe"
Write-Host "Spp        : $Spp"
Write-Host "Runs/count : $Runs"
Write-Host "Output CSV : $OutCsv"
Write-Host ""

"threads,run,wall_sec,spp,mrays_per_sec" | Set-Content $OutCsv

$Summary = @()

foreach ($T in $ThreadCounts) {
    $RunTimes  = @()
    $MraysRuns = @()

    Write-Host "Threads = $T" -ForegroundColor Yellow

    for ($r = 1; $r -le $Runs; $r++) {
        if (Test-Path "image.ppm") { Remove-Item "image.ppm" }

        # Build a ProcessStartInfo so we can control the environment directly.
        # This is necessary because Start-Process / & operator do not reliably
        # inherit $env: mutations into child process environments on Windows.
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName              = $Exe
        $psi.Arguments             = "$Spp"
        $psi.UseShellExecute       = $false
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow        = $false

        # Copy current environment then override OMP_NUM_THREADS
        foreach ($entry in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
            $psi.Environment[$entry.Key] = $entry.Value
        }
        $psi.Environment["OMP_NUM_THREADS"] = "$T"

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = [System.Diagnostics.Process]::Start($psi)
        # ReadToEnd BEFORE WaitForExit - avoids deadlock on full stderr pipe buffer
        $stderrText = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()
        $sw.Stop()

        if ($proc.ExitCode -ne 0) {
            Write-Warning "  Run $r FAILED (exit $($proc.ExitCode))"
            continue
        }

        $sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)

        $mrays = $null
        if ($stderrText -match 'RAYS_TOTAL:\s*(\d+)') {
            $totalRays = [double]$Matches[1]
            $mrays = [math]::Round($totalRays / $sec / 1e6, 2)
        }

        $RunTimes  += $sec
        $MraysRuns += $mrays

        $mraysStr = if ($null -ne $mrays) { "$mrays" } else { "N/A" }
        Write-Host ("  Run {0}: {1:F3} s  |  {2} Mrays/s" -f $r, $sec, $mraysStr)
    }

    if ($RunTimes.Count -gt 0) {
        $avg = [math]::Round(($RunTimes | Measure-Object -Average).Average, 3)
        $min = [math]::Round(($RunTimes | Measure-Object -Minimum).Minimum, 3)

        $validMr = $MraysRuns | Where-Object { $null -ne $_ }
        $avgMr   = $null
        if ($validMr.Count -gt 0) {
            $avgMr = [math]::Round(($validMr | Measure-Object -Average).Average, 2)
        }

        $Summary += [PSCustomObject]@{
            Threads  = $T
            AvgSec   = $avg
            MinSec   = $min
            AvgMrays = $avgMr
            Speedup  = $null
        }

        $mrStr = if ($null -ne $avgMr) { "  avg_mrays=$avgMr" } else { "" }
        Write-Host ("  avg={0:F3}s  min={1:F3}s{2}" -f $avg, $min, $mrStr) -ForegroundColor Green
    }
    Write-Host ""
}

# Speedup relative to 1-thread
$baseRow = $Summary | Where-Object { $_.Threads -eq 1 }
if ($null -ne $baseRow -and $baseRow.AvgSec -gt 0) {
    $base = $baseRow.AvgSec
    foreach ($row in $Summary) {
        $row.Speedup = [math]::Round($base / $row.AvgSec, 2)
    }
}

Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ("{0,8}  {1,9}  {2,9}  {3,10}  {4,9}" -f "Threads", "Avg (s)", "Min (s)", "Mrays/s", "Speedup")
Write-Host ("{0,8}  {1,9}  {2,9}  {3,10}  {4,9}" -f "-------", "-------", "-------", "---------", "-------")
foreach ($row in $Summary) {
    $spStr = if ($null -ne $row.Speedup)  { "{0:F2}x" -f $row.Speedup  } else { "N/A" }
    $mrStr = if ($null -ne $row.AvgMrays) { "{0:F2}"  -f $row.AvgMrays } else { "N/A" }
    Write-Host ("{0,8}  {1,9:F3}  {2,9:F3}  {3,10}  {4,9}" -f $row.Threads, $row.AvgSec, $row.MinSec, $mrStr, $spStr)
}

Write-Host ""
Write-Host "Raw data written to: $OutCsv" -ForegroundColor Cyan

"`n# Summary" | Add-Content $OutCsv
"threads,avg_sec,min_sec,avg_mrays_per_sec,speedup_vs_1t" | Add-Content $OutCsv
foreach ($row in $Summary) {
    "$($row.Threads),$($row.AvgSec),$($row.MinSec),$($row.AvgMrays),$($row.Speedup)" | Add-Content $OutCsv
}
