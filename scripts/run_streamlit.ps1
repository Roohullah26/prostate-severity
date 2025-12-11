<# Run the Streamlit demo UI from any folder.

Usage:
  .\run_streamlit.ps1  # runs streamlit app in the repo using default port 8501
  .\run_streamlit.ps1 -Port 8502 -ServerUrl 'http://127.0.0.1:8000/predict' -ApiKey 'mysecret'
#>

[CmdletBinding()]
param(
    [int]$Port = 8501,
    [string]$ServerUrl = 'http://127.0.0.1:8000/predict',
    [string]$ApiKey = $null
)

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = (Resolve-Path $scriptDir).Path
    Write-Host "Setting PYTHONPATH=$repoRoot"
    $env:PYTHONPATH = $repoRoot

    # pass values as environment variables so the Streamlit app can pick them up
    if ($ApiKey) { $env:DEMO_API_KEY = $ApiKey }
    $env:DEMO_SERVER_URL = $ServerUrl

    Write-Host "Starting Streamlit demo on port $Port (server: $ServerUrl)"
    & py -3 -m streamlit run $repoRoot\webapp\streamlit_demo.py --server.port $Port
}
catch {
    Write-Error "Failed to run Streamlit demo: $_"
    exit 1
}
