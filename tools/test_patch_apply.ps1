# tools/test_patch_apply.ps1
param(
  [Parameter(Mandatory=$true)][string]$Path
)
if (-not (Test-Path $Path)) { Write-Error "File not found: $Path"; exit 2 }
# Ensure LF-only before piping to git
$txt = Get-Content -LiteralPath $Path -Raw
$txt = $txt -replace "`r`n","`n" -replace "`r","`n"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($txt)
[IO.File]::WriteAllBytes($Path, $bytes)
git apply --check --whitespace=nowarn --unsafe-paths $Path
if ($LASTEXITCODE -eq 0) { Write-Host "Patch OK" -ForegroundColor Green } else { exit $LASTEXITCODE }
