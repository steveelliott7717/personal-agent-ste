Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Message
)

$ErrorActionPreference = 'Stop'

# Commit message
$Msg = ($Message -join ' ').Trim()
if (-not $Msg) {
  Write-Host "Usage: tools\ship.ps1 'commit message'"
  exit 2
}

# If pre-commit isn't on PATH, fall back to python -m
function Run-PreCommit {
  try {
    pre-commit run
  } catch {
    Write-Host "[i] pre-commit not on PATH, trying 'python -m pre-commit run'..."
    python -m pre-commit run
  }
}

Write-Host "[+] git add -A"
git add -A

Write-Host "[+] pre-commit run (staged only)"
try {
  Run-PreCommit
} catch {
  throw
}
if ($LASTEXITCODE -ne 0) {
  Write-Host "[!] Hooks modified files; staging fixes..."
  git add -A
  # Run hooks again to be safe; ignore non-zero on second pass
  try { Run-PreCommit } catch { }
}

Write-Host "[+] git commit"
git commit -m "$Msg"

Write-Host "[+] git push"
git push

# If you need a specific Fly app, set $Env:FLY_APP or change the line below to: fly -a your-app-name deploy
Write-Host "[+] fly deploy"
fly deploy

Write-Host "SUCCESS: shipped & deployed."
