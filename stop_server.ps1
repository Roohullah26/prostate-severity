<# stop_server.ps1 - simple helper to stop a running uvicorn/python server bound to a port

Usage: run from any folder
  ./stop_server.ps1 -Port 8000

This script finds the process id(s) for the given port and attempts to stop them.
#>

param(
  [int]$Port = 8000
)

Write-Host "Looking for processes listening on port $Port..."
$lines = netstat -aon | Select-String ":$Port"
if (-not $lines) {
    Write-Host "No processes found listening on port $Port"
    exit 0
}

$pids = @()
foreach ($l in $lines) {
    $parts = $l.ToString().Trim() -split '\s+' | Where-Object { $_ -ne '' }
    $pid = $parts[-1]
    if ($pid -and ($pid -as [int])) { $pids += [int]$pid }
}

$pids = $pids | Sort-Object -Unique
if (-not $pids) { Write-Host "No PIDs parsed"; exit 0 }

foreach ($pid in $pids) {
    try {
        Write-Host "Stopping process $pid..."
        Stop-Process -Id $pid -Force -ErrorAction Stop
        Write-Host "Stopped $pid"
    } catch {
        Write-Warning "Failed to stop $pid: $_"
    }
}

Write-Host "Done."
