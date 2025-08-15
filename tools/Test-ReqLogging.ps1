param(
  [string]$BaseUrl = "https://personal-agent-ste.fly.dev",
  [int]$Repeat = 5,
  [string]$ApiKey = $env:API_KEY  # allow passing or env
)

function Invoke-And-Check($uri, $headers) {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Method GET -Uri $uri -Headers $headers
    $cid  = $resp.Headers["x-correlation-id"]
    [PSCustomObject]@{
      time = (Get-Date).ToString("s")
      url  = $uri
      code = [int]$resp.StatusCode
      cid  = $cid
      ok   = -not [string]::IsNullOrWhiteSpace($cid)
    }
  }
  catch {
    $response = $_.Exception.Response
    $status   = if ($response) { [int]$response.StatusCode } else { -1 }
    $cidHdr   = if ($response) { $response.Headers["x-correlation-id"] } else { "" }
    [PSCustomObject]@{
      time = (Get-Date).ToString("s")
      url  = $uri
      code = $status
      cid  = $cidHdr
      ok   = -not [string]::IsNullOrWhiteSpace($cidHdr)
      err  = $_.Exception.Message
    }
  }
}

$endpoints = @(
  "/health",
  "/app/api/repo/health"
) | ForEach-Object { $_.Trim() }

$seen = New-Object 'System.Collections.Generic.HashSet[string]'
$headers = @{}
if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
  $headers["X-API-Key"] = $ApiKey
}

Write-Host "Hitting $($endpoints.Count) endpoints on $BaseUrl, $Repeat times each..." -ForegroundColor Cyan

$results = @()
foreach ($ep in $endpoints) {
  for ($i = 1; $i -le $Repeat; $i++) {
    $uri = "$BaseUrl$ep"
    $r = Invoke-And-Check $uri $headers
    if (-not [string]::IsNullOrWhiteSpace($r.cid)) { $null = $seen.Add($r.cid) }
    $results += $r
    $cidOut = if ([string]::IsNullOrWhiteSpace($r.cid)) { "<none>" } else { $r.cid }
    "{0}  {1,-28}  {2}  cid={3}" -f $r.time, $ep, $r.code, $cidOut | Write-Host
    Start-Sleep -Milliseconds 150
  }
}

Write-Host ""
$missing = $results | Where-Object { -not $_.ok }
if ($missing.Count -gt 0) {
  Write-Host "Some responses lacked X-Correlation-ID:"
  $missing | Format-Table time, url, code, cid, err -Auto
} else {
  Write-Host "All responses included X-Correlation-ID."
}

$unique = $seen.Count
$total  = $results.Count
Write-Host ("Unique IDs observed: {0} (across {1} requests)" -f $unique, $total)
