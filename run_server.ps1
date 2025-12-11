<#
Tiny helper script to start the inference server from anywhere.

Usage examples (from any folder):
  # start with default toy model if present:
  .\run_server.ps1

  # specify an absolute or relative state-dict path and other options:
  .\run_server.ps1 -StatePath 'D:\models\baseline_real_t2_adc_3s.pth' -Port 8000 -InCh 6

Notes:
- The script sets PYTHONPATH to the script's parent folder so Python can import the package
- By default it runs `python -m webapp.fastapi_server` so CLI args like --state-dict are supported
#>

[CmdletBinding()]
param(
    [string]$StatePath = ".\models\prototype_toy.pth",
    [int]$InCh = 3,
    [string]$Host = '127.0.0.1',
    [int]$Port = 8000,
    [string]$LogLevel = 'info'
)

try {
    # resolve script folder and set PYTHONPATH so module imports work regardless of cwd
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = (Resolve-Path $scriptDir).Path
    Write-Host "Setting PYTHONPATH=$repoRoot"
    $env:PYTHONPATH = $repoRoot

    # resolve state path if present
    if ($StatePath -and (Test-Path $StatePath)) {
        $stateResolved = (Resolve-Path $StatePath).Path
        Write-Host "Starting server (module entrypoint) with state-dict=$stateResolved"
        & py -3 -m webapp.fastapi_server --state-dict "$stateResolved" --in-ch $InCh --host $Host --port $Port
    } else {
        Write-Host "Starting server (uvicorn) without preloaded state-dict"
        & py -3 -m uvicorn webapp.fastapi_server:app --host $Host --port $Port --log-level $LogLevel
    }
}
catch {
    Write-Error "Failed to start server: $_"
    exit 1
}
