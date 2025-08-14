param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [Parameter(Mandatory = $true)][string]$OutPath
)

$ErrorActionPreference = "Stop"

# Regexes (PS5-safe)
$fence = New-Object System.Text.RegularExpressions.Regex '```(?:diff|patch)?\s*([\s\S]*?)```', ([System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
$cb = New-Object System.Text.RegularExpressions.Regex '/\*.*?\*/', ([System.Text.RegularExpressions.RegexOptions]::Singleline)
$ell = New-Object System.Text.RegularExpressions.Regex '(?m)^\s*\.\.\.\s*$'
$rxDiff = New-Object System.Text.RegularExpressions.Regex '^diff --git a/(.+?) b/(.+?)$'
$rxHunk = New-Object System.Text.RegularExpressions.Regex '^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@$'

$summary = [ordered]@{
    fixed_header_order = 0
    dedup_headers      = 0
    newfile_prefixed   = 0
    modified_prefixed  = 0
    recomputed_hunks   = 0
}

function SafeStr([object]$x) { if ($null -eq $x) { "" } else { [string]$x } }

function Unwrap([string]$s) {
    if ([string]::IsNullOrWhiteSpace($s)) { return "" }
    $s = $s -replace "`r`n", "`n" -replace "`r", "`n"
    while ($true) {
        $before = $s
        $s = $fence.Replace($s, { param($m) $m.Groups[1].Value })
        if ($s -eq $before) { break }
    }
    $s = [regex]::Replace($s, '(?m)^\s*```\s*$', '')
    $s = $cb.Replace($s, '')
    $s = $ell.Replace($s, '')
    $s.Trim()
}

function FixPatch([string]$text) {
    $lines = $text -split "`n", 0
    $out = New-Object System.Collections.Generic.List[string]

    $inSection = $false
    $pendingHeaders = New-Object System.Collections.Generic.List[string]
    $isNewFile = $false
    $pathB = $null

    $inHunk = $false
    $hunkHeaderIdx = $null
    $oldCnt = 0
    $newCnt = 0

    function ResetSection {
        $script:pendingHeaders = New-Object System.Collections.Generic.List[string]
        $script:isNewFile = $false
        $script:pathB = $null
        $script:inHunk = $false
        $script:hunkHeaderIdx = $null
        $script:oldCnt = 0
        $script:newCnt = 0
    }

    function FlushHunk {
        if ($null -ne $hunkHeaderIdx -and $hunkHeaderIdx -lt $out.Count) {
            $hdr = $out[$hunkHeaderIdx]
            $m = $rxHunk.Match([string]$hdr)
            if ($m.Success) {
                $aStart = [int]$m.Groups[1].Value
                $bStart = [int]$m.Groups[3].Value
                $out[$hunkHeaderIdx] = "@@ -$aStart,$oldCnt +$bStart,$newCnt @@"
                $summary.recomputed_hunks++
            }
        }
        $script:hunkHeaderIdx = $null
        $script:oldCnt = 0
        $script:newCnt = 0
        $script:inHunk = $false
    }

    function FlushHeaders {
        $haveDevNull = $false
        foreach ($h in $pendingHeaders) {
            $hs = SafeStr $h
            if ($hs -match '^\s*---\s*/dev/null\s*$') { $haveDevNull = $true }
            elseif (-not $pathB -and $hs -match '^\s*\+\+\+\s+b/(.+)\s*$') { $pathB = $Matches[1] }
        }
        $isNewFile = $haveDevNull
        if (-not $pathB) { $pathB = 'unknown' }

        $ordered = New-Object System.Collections.Generic.List[string]
        foreach ($h in $pendingHeaders) {
            $hs = SafeStr $h
            if ($hs.StartsWith('index ') -or $hs.StartsWith('new file mode ') -or
                $hs.StartsWith('deleted file mode ') -or $hs.StartsWith('rename ')) {
                $ordered.Add($hs)
            }
        }

        if ($isNewFile) {
            $ordered.Add('--- /dev/null')
            $ordered.Add("+++ b/$pathB")
            $summary.fixed_header_order++
        }
        else {
            $minus = $null; $plus = $null
            foreach ($h in $pendingHeaders) { $hs = SafeStr $h; if (-not $minus -and $hs -match '^\s*---\s+a/(.+)\s*$') { $minus = $hs } }
            foreach ($h in $pendingHeaders) { $hs = SafeStr $h; if (-not $plus -and $hs -match '^\s*\+\+\+\s+b/(.+)\s*$') { $plus = $hs; if (-not $pathB) { $pathB = $Matches[1] } } }
            if ($minus) { $ordered.Add($minus) } else { $ordered.Add("--- a/$pathB") }
            if ($plus) { $ordered.Add($plus) } else { $ordered.Add("+++ b/$pathB") }
        }

        $plusCount = 0
        foreach ($h in $pendingHeaders) { $hs = SafeStr $h; if ($hs -match '^\s*\+\+\+\s+b/') { $plusCount++ } }
        if ($plusCount -gt 1) { $summary.dedup_headers += ($plusCount - 1) }

        foreach ($h in $ordered) { $out.Add([string]$h) }
        $script:pendingHeaders = New-Object System.Collections.Generic.List[string]
    }

    ResetSection

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = SafeStr $lines[$i]

        if ($rxDiff.IsMatch($line)) {
            FlushHunk
            $inSection = $true
            ResetSection
            $m = $rxDiff.Match($line)
            if ($m.Success) { $pathB = $m.Groups[2].Value }
            $out.Add($line)
            continue
        }

        if (-not $inSection) { $out.Add($line); continue }

        if ($line.StartsWith('index ') -or $line.StartsWith('new file mode ') -or
            $line.StartsWith('deleted file mode ') -or $line.StartsWith('rename ') -or
            $line.StartsWith('--- ') -or $line.StartsWith('+++ ')) {
            $pendingHeaders.Add($line)
            continue
        }

        if ($line.StartsWith('@@ ')) {
            if ($pendingHeaders.Count -gt 0) { FlushHeaders }
            FlushHunk
            $inHunk = $true
            $hunkHeaderIdx = $out.Count
            $out.Add($line)
            $oldCnt = 0; $newCnt = 0
            continue
        }

        if ($inHunk) {
            $ll = SafeStr $line
            if ($ll.Length -gt 0) { $first = $ll[0] } else { $first = '' }
            $isBackslashLine = ($first -ceq '\')

            if ($ll -eq "") {
                if ($isNewFile) { $ll = "+"; $summary.newfile_prefixed++; $newCnt++ }
                else { $ll = " "; $summary.modified_prefixed++; $oldCnt++; $newCnt++ }
            }
            elseif (-not $isBackslashLine -and $first -notin @(' ', '+', '-')) {
                if ($isNewFile) { $ll = '+' + $ll; $summary.newfile_prefixed++; $newCnt++ }
                else { $ll = ' ' + $ll; $summary.modified_prefixed++; $oldCnt++; $newCnt++ }
            }
            else {
                if ($first -eq ' ') { $oldCnt++; $newCnt++ }
                elseif ($first -eq '+') { $newCnt++ }
                elseif ($first -eq '-') { $oldCnt++ }
            }

            $out.Add($ll)
            continue
        }

        if ($pendingHeaders.Count -gt 0) { FlushHeaders }
        $out.Add($line)
    }

    FlushHunk
    if ($pendingHeaders.Count -gt 0) { FlushHeaders }

    $fixed = [string]::Join("`n", $out)
    if (-not $fixed.EndsWith("`n")) { $fixed += "`n" }
    $fixed
}

# Pipeline
try {
    $raw = Get-Content -Raw -LiteralPath $InPath
}
catch {
    Write-Error ("Failed to read input file: {0} - {1}" -f $InPath, $_.Exception.Message)
    exit 1
}

try {
    $stage = Unwrap (SafeStr $raw)
    $stage = FixPatch $stage
    [IO.File]::WriteAllText($OutPath, $stage, [Text.UTF8Encoding]::new($false))
    Write-Host (
        "[fix-patch] fixed_header_order={0} dedup_headers={1} newfile_prefixed={2} modified_prefixed={3} recomputed_hunks={4}" -f `
            $summary.fixed_header_order, $summary.dedup_headers, $summary.newfile_prefixed, $summary.modified_prefixed, $summary.recomputed_hunks
    )
}
catch {
    Write-Error ("fix-patch.ps1 fatal: {0}" -f $_.Exception.Message)
    exit 1
}
