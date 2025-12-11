param(
    [string]$server = "http://127.0.0.1:8000/predict",
    [string]$image = "./data/example.jpg",
    [string]$state = "./models/prototype_toy.pth",
    [int]$in_ch = 3,
    [string]$apiKey = $null
)

if (-not (Test-Path $image)) {
    Write-Host "Image not found: $image" -ForegroundColor Yellow
    exit 1
}

$form = @{ file = Get-Item $image }
if ($apiKey) {
    $headers = @{ 'X-API-Key' = $apiKey }
} else {
    $headers = @{}
}

$uri = "$server?state_dict=$([System.Uri]::EscapeDataString((Resolve-Path $state).Path))&in_channels=$in_ch"
Write-Host "POSTing to: $uri"
$resp = Invoke-RestMethod -Uri $uri -Method Post -Form $form -Headers $headers
Write-Host "Server response:`n" ($resp | ConvertTo-Json -Depth 4)
