<#  Get-RmsPatch.ps1
    Fetch, validate, normalize, and (optionally) apply an RMS patch safely on Windows.

    EXAMPLES
      # Validate only
      pwsh -File tools/Get-RmsPatch.ps1 -TaskJson rms_prompts/request_logging_with_strict_contract.json

      # Validate + apply + commit
      pwsh -File tools/Get-RmsPatch.ps1 -TaskJson rms_prompts/request_logging_with_strict_contract.json -Apply -CommitMessage "feat: request logging + correlation IDs"

      # Custom endpoint / filenames
      pwsh -File tools/Get-RmsPatch.ps1 -TaskJson rms_prompts/X.json -Endpoint "https://personal-agent-ste.fly.dev/app/api/repo/plan" -OutBase ".rms/request_logging"
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$TaskJson,

  [string]$Endpoint = "https://personal-agent-ste.fly.dev/app/api/repo/plan",
  [string]$OutBase = ".rms/patch",
  [switch]$Apply,
  [string]$CommitMessage = "chore: apply RMS patch"
)

# --- Prep
$ErrorActionPreference = "Stop"
$null = New-Item -Force -ItemType Directory -Path (Split-Path $OutBase) -ErrorAction SilentlyContinue

$headersPath = "$OutBase.headers"
$rawPath     = "$OutBase.patch"
$unixPath    = "$OutBase.unix.patch"

Write-Host ">> Requesting patch from $Endpoint with payload $TaskJson"

# --- Fetch (PATCH MODE) ---
# Ensure TLS 1.2 (older Windows PowerShell defaults can fail HTTPS)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# Build a clean absolute Uri (avoid hidden characters/newlines)
$uriString = ($Endpoint.TrimEnd('?','/')) + "?format=patch"
if (-not [Uri]::IsWellFormedUriString($uriString, [UriKind]::Absolute)) {
  throw "Bad URI: '$uriString'"
}
$uri = [Uri]$uriString
Write-Host ">> Using URI: $($uri.AbsoluteUri)"

try {
  $resp = Invoke-WebRequest -Method POST -Uri $uri -ContentType "application/json" -InFile $TaskJson -UseBasicParsing
} catch {
  Write-Warning "Invoke-WebRequest failed: $($_.Exception.Message)"
  Write-Warning "Falling back to curl.exe for the request..."
  $headersPath = "$OutBase.headers"
  $rawPath     = "$OutBase.patch"
  $curl = "${env:SystemRoot}\System32\curl.exe"
  & $curl -sS -X POST "$($uri.AbsoluteUri)" -H "Content-Type: application/json" --data-binary "@$TaskJson" -o "$rawPath" -D "$headersPath"
  if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE" }
  # Synthesize a $resp-like object for the rest of the script
  $contentType = (Get-Content $headersPath | Select-String -Pattern '^Content-Type:\s*(.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }) -join ', '
  $content     = Get-Content $rawPath -Raw
  $resp = [pscustomobject]@{ Headers = @{ 'Content-Type' = $contentType }; Content = $content }
}


# Save headers + body exactly as returned
$resp.Headers.GetEnumerator() | ForEach-Object { "$($_.Name): $($_.Value)" } | Set-Content -Encoding UTF8 $headersPath
[System.IO.File]::WriteAllBytes($rawPath, [System.Text.Encoding]::UTF8.GetBytes($resp.Content))

Write-Host ">> Saved: $rawPath and $headersPath"

# --- Header checks ---
$contentType = "$($resp.Headers.'Content-Type')"
if (-not ($contentType -match 'text/x-patch')) {
  Write-Error "Response Content-Type was not 'text/x-patch' (got '$contentType'). Aborting."
}

# --- Normalize to UTF-8 (no BOM) + LF ---
$raw = Get-Content $rawPath -Raw
$hadCRLF = $raw.Contains("`r`n")
$unix = $raw -replace "`r`n","`n" -replace "`r","`n"
[IO.File]::WriteAllText($unixPath, $unix, [Text.UTF8Encoding]::new($false))

Write-Host ">> Normalized LF file: $unixPath (CRLF present originally: $hadCRLF)"

# --- Fast content checks (prevent corrupt patches) ---
$firstLine = ($unix -split "`n", 2)[0]
if (($firstLine -notmatch '^diff --git ') -and ($firstLine -notmatch '^--- ')) {
  Write-Error "Patch does not start with a unified diff header (found: '$firstLine')."
}

$forbidden = @(
  '```',         # markdown fences
  '^\.\.\.$',    # ellipsis line as placeholder
  '/\*.*?\*/'    # C-style comment blocks
)
$foundBad = @()
foreach ($pat in $forbidden) {
  if ($unix -match $pat) { $foundBad += $pat }
}
if ($foundBad.Count -gt 0) {
  Write-Error "Forbidden patterns found in patch: $($foundBad -join ', ')"
}

# Optional: quick structural sanity (every hunk header well-formed)
$badHunks = Select-String -InputObject $unix -Pattern '^@@ (?!-\d+,\d+ \+\d+,\d+ @@)' -AllMatches
if ($badHunks) {
  Write-Warning "One or more malformed hunk headers detected. git apply may fail."
}

# --- Dry run with git ---
Write-Host ">> Running: git apply --check $unixPath"
$check = & git apply --check $unixPath 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host $check
  Write-Error "git apply --check failed. Inspect $unixPath and headers at $headersPath."
}

if (-not $Apply) {
  Write-Host ">> Validation succeeded. Re-run with -Apply to apply & commit."
  exit 0
}

# --- Apply & commit ---
Write-Host ">> Applying patch..."
& git apply --whitespace=fix $unixPath
if ($LASTEXITCODE -ne 0) { Write-Error "git apply failed." }

& git add -A
Write-Host ">> Staged changes:"
& git diff --cached

& git commit -m $CommitMessage
& git push
Write-Host ">> Patch applied and pushed."

# Done.
