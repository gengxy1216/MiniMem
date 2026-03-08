[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:20195",
    [int]$TimeoutSec = 60,
    [switch]$NoAutoStart,
    [switch]$StrictCollective,
    [switch]$StrictOpenClaw,
    [switch]$WriteChecklist,
    [string]$ChecklistPath = "docs/reports/team-release-readiness-2026-03-08.md"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Resolve-HostExecutable {
    $path = (Get-Process -Id $PID).Path
    if ($path) {
        return $path
    }
    if (Get-Command "pwsh" -ErrorAction SilentlyContinue) {
        return "pwsh"
    }
    if (Get-Command "powershell" -ErrorAction SilentlyContinue) {
        return "powershell"
    }
    throw "Cannot resolve PowerShell host executable."
}

function Escape-MarkdownCell {
    param([string]$Value)
    if ($null -eq $Value) {
        return ""
    }
    return (($Value -replace "\|", "&#124;") -replace "\r?\n", "<br/>")
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$smokeScript = Join-Path $repoRoot "scripts/smoke_collective_release.ps1"
if (-not (Test-Path $smokeScript)) {
    throw "Missing smoke script: $smokeScript"
}

$runtimeDir = Join-Path $repoRoot "tmp_test_runtime"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$rawLogPath = Join-Path $runtimeDir "validate-collective-release.raw.log"
$stdoutLogPath = Join-Path $runtimeDir "validate-collective-release.stdout.log"
$stderrLogPath = Join-Path $runtimeDir "validate-collective-release.stderr.log"

$hostExecutable = Resolve-HostExecutable
$smokeArgs = @(
    "-NoProfile",
    "-File",
    $smokeScript,
    "-BaseUrl",
    $BaseUrl,
    "-TimeoutSec",
    $TimeoutSec.ToString()
)
if ($NoAutoStart) {
    $smokeArgs += "-NoAutoStart"
}
if (-not $StrictCollective) {
    $smokeArgs += "-AllowMissingCollective"
}
if ($StrictOpenClaw) {
    $smokeArgs += "-StrictOpenClaw"
}

Write-Host "== Collective Release Dry-Run Validation =="
Write-Host "base_url=$BaseUrl timeout_sec=$TimeoutSec strict_collective=$StrictCollective strict_openclaw=$StrictOpenClaw no_auto_start=$NoAutoStart"

$smokeOutput = @()
$smokeExitCode = 1
try {
    if (Test-Path $stdoutLogPath) {
        Remove-Item -Path $stdoutLogPath -Force
    }
    if (Test-Path $stderrLogPath) {
        Remove-Item -Path $stderrLogPath -Force
    }
    $proc = Start-Process -FilePath $hostExecutable `
        -ArgumentList $smokeArgs `
        -WorkingDirectory $repoRoot `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutLogPath `
        -RedirectStandardError $stderrLogPath
    $smokeExitCode = [int]$proc.ExitCode
    if (Test-Path $stdoutLogPath) {
        $smokeOutput += Get-Content -Path $stdoutLogPath
    }
    if (Test-Path $stderrLogPath) {
        $smokeOutput += Get-Content -Path $stderrLogPath
    }
}
catch {
    $smokeOutput = @("Smoke script invocation failed: $($_.Exception.Message)")
    $smokeExitCode = 1
}

$smokeLines = @($smokeOutput | ForEach-Object { $_.ToString() })
$smokeLines | Set-Content -Path $rawLogPath -Encoding UTF8
$smokeLines | ForEach-Object { Write-Host $_ }

$checks = @(
    @{ Name = "health"; Display = "Health"; Criteria = "/health returns 200"; Blocking = $true },
    @{ Name = "collective-ingest"; Display = "Collective ingest"; Criteria = "POST /api/v1/collective/ingest returns non-404/non-5xx"; Blocking = $true },
    @{ Name = "collective-context"; Display = "Collective context"; Criteria = "POST /api/v1/collective/context returns non-404/non-5xx"; Blocking = $true },
    @{ Name = "collective-feedback"; Display = "Collective feedback"; Criteria = "POST /api/v1/collective/feedback returns non-404/non-5xx"; Blocking = $true },
    @{ Name = "mcp-syntax"; Display = "MCP syntax"; Criteria = "integrations/flockmem-mcp/server.py compiles"; Blocking = $true },
    @{ Name = "mcp-import"; Display = "MCP import"; Criteria = "MCP server module import succeeds"; Blocking = $true },
    @{ Name = "plugin-install-scripts"; Display = "Plugin install scripts"; Criteria = "openclaw plugin install.sh/install.ps1 both exist"; Blocking = $true },
    @{ Name = "plugin-pack-dry-run"; Display = "Plugin pack dry-run"; Criteria = "npm --prefix integrations/openclaw-plugin pack --dry-run succeeds"; Blocking = $true },
    @{ Name = "openclaw-cli"; Display = "OpenClaw CLI"; Criteria = "openclaw command exists (strict mode)"; Blocking = [bool]$StrictOpenClaw },
    @{ Name = "stderr-anomalies"; Display = "Stderr anomalies"; Criteria = "no Python traceback/import/runtime errors in smoke stderr"; Blocking = $true }
)

$checkResults = @{}
foreach ($check in $checks) {
    $checkResults[$check.Name] = @{
        Level = "NOT_RUN"
        Detail = "not observed in smoke output"
    }
}

[int]$passCount = 0
[int]$warnCount = 0
[int]$failCount = 0
$summaryFound = $false
foreach ($line in $smokeLines) {
    if ($line -match "^\[(PASS|WARN|FAIL)\]\s+(.+?)\s+-\s+(.*)$") {
        $level = $matches[1]
        $name = $matches[2].Trim()
        $detail = $matches[3].Trim()

        switch ($level) {
            "PASS" { $passCount += 1 }
            "WARN" { $warnCount += 1 }
            "FAIL" { $failCount += 1 }
        }
        if ($checkResults.ContainsKey($name)) {
            $checkResults[$name] = @{
                Level = $level
                Detail = $detail
            }
        }
        continue
    }

    if ($line -match "^PASS=(\d+)\s+WARN=(\d+)\s+FAIL=(\d+)$") {
        $passCount = [int]$matches[1]
        $warnCount = [int]$matches[2]
        $failCount = [int]$matches[3]
        $summaryFound = $true
    }
}

if (-not $summaryFound) {
    Write-Host "WARN: smoke summary line not found; using parsed counters."
}

$tracebackLine = $smokeLines | Where-Object { $_ -match "^Traceback \(most recent call last\):" } | Select-Object -First 1
$pythonErrorLine = $smokeLines | Where-Object { $_ -match "^(ImportError|ModuleNotFoundError|RuntimeError|SyntaxError|NameError):" } | Select-Object -First 1
if ($tracebackLine -or $pythonErrorLine) {
    $traceDetail = @()
    if ($tracebackLine) {
        $traceDetail += $tracebackLine.Trim()
    }
    if ($pythonErrorLine) {
        $traceDetail += $pythonErrorLine.Trim()
    }
    $checkResults["stderr-anomalies"] = @{
        Level = "FAIL"
        Detail = ($traceDetail -join "; ")
    }
    $failCount += 1
}
else {
    $checkResults["stderr-anomalies"] = @{
        Level = "PASS"
        Detail = "no traceback/import/runtime errors detected"
    }
    $passCount += 1
}

$decision = "READY"
if ($smokeExitCode -ne 0 -or $failCount -gt 0) {
    $decision = "BLOCKED"
}
elseif ($warnCount -gt 0) {
    $decision = "READY_WITH_WARNINGS"
}

$mode = if ($StrictCollective -or $StrictOpenClaw) { "strict" } else { "compat" }
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

Write-Host "== Release Gate =="
Write-Host "decision=$decision mode=$mode smoke_exit=$smokeExitCode pass=$passCount warn=$warnCount fail=$failCount"
Write-Host "raw_log=$rawLogPath"

$report = [System.Collections.Generic.List[string]]::new()
$report.Add("# Team Release Readiness - Collective (2026-03-08)")
$report.Add("")
$report.Add("- Generated At: $timestamp")
$report.Add("- Owner: DevOps-lite")
$report.Add("- Validation Mode: $mode")
$report.Add("- Decision: $decision")
$report.Add("- Smoke Exit Code: $smokeExitCode")
$report.Add('- Smoke Raw Log: `tmp_test_runtime/validate-collective-release.raw.log`')
$report.Add("")
$report.Add("## Scope")
$report.Add("")
$report.Add("Minimum release dry-run coverage for collective path:")
$report.Add("1. health endpoint")
$report.Add("2. collective ingest/context/feedback endpoint callability")
$report.Add("3. MCP script syntax/import readiness")
$report.Add("4. OpenClaw plugin install/package readiness")
$report.Add("")
$report.Add("## CI Smoke")
$report.Add("")
$report.Add('- Workflow: `.github/workflows/collective-smoke.yml`')
$report.Add('- Trigger: `pull_request` / `workflow_dispatch`')
$report.Add('- Runtime: Linux + Windows smoke jobs, both call `scripts/validate_collective_release.ps1`')
$report.Add("- Dispatch knobs:")
$report.Add('  - `strict_collective=true`: fail when collective endpoints are missing')
$report.Add('  - `strict_openclaw=true`: fail when openclaw CLI is missing')
$report.Add("- Artifacts: readiness checklist markdown from each smoke job")
$report.Add("")
$report.Add("## Checklist")
$report.Add("")
$report.Add("| Check | Criteria | Blocking | Result | Detail |")
$report.Add("|---|---|---|---|---|")
foreach ($check in $checks) {
    $result = $checkResults[$check.Name]
    $blocking = if ($check.Blocking) { "Yes (P0/P1)" } else { "No" }
    $row = "| {0} | {1} | {2} | {3} | {4} |" -f `
        (Escape-MarkdownCell $check.Display), `
        (Escape-MarkdownCell $check.Criteria), `
        $blocking, `
        (Escape-MarkdownCell $result.Level), `
        (Escape-MarkdownCell $result.Detail)
    $report.Add($row)
}
$report.Add("")
$report.Add("## Blocking Rules (P0/P1)")
$report.Add("")
$report.Add('1. `health` or any strict collective endpoint check is `FAIL`.')
$report.Add('2. `mcp-syntax` / `mcp-import` / `plugin-install-scripts` / `plugin-pack-dry-run` is `FAIL`.')
$report.Add('3. `openclaw-cli` is `FAIL` when strict mode is enabled.')
$report.Add("")
$report.Add("## Local Commands")
$report.Add("")
$report.Add("Windows compat dry-run:")
$report.Add("")
$report.Add('```powershell')
$report.Add("powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_collective_release.ps1 -WriteChecklist -ChecklistPath docs/reports/team-release-readiness-2026-03-08.md")
$report.Add('```')
$report.Add("")
$report.Add("Windows strict release gate:")
$report.Add("")
$report.Add('```powershell')
$report.Add("powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_collective_release.ps1 -StrictCollective -StrictOpenClaw -WriteChecklist -ChecklistPath docs/reports/team-release-readiness-2026-03-08.md")
$report.Add('```')
$report.Add("")
$report.Add("Cross-platform strict gate (pwsh):")
$report.Add("")
$report.Add('```powershell')
$report.Add("pwsh -NoProfile -File scripts/validate_collective_release.ps1 -StrictCollective -StrictOpenClaw -WriteChecklist -ChecklistPath docs/reports/team-release-readiness-2026-03-08.md")
$report.Add('```')
$report.Add("")
$report.Add("## Current Run Summary")
$report.Add("")
$report.Add("- PASS: $passCount")
$report.Add("- WARN: $warnCount")
$report.Add("- FAIL: $failCount")
$report.Add("- Notes:")
$report.Add('  - `READY` means no fail and no warning.')
$report.Add('  - `READY_WITH_WARNINGS` means no fail but warning exists (non-blocking in compat mode).')
$report.Add('  - `BLOCKED` means release should be stopped until fail items are fixed.')

if ($WriteChecklist) {
    $targetPath = $ChecklistPath
    if (-not [System.IO.Path]::IsPathRooted($targetPath)) {
        $targetPath = Join-Path $repoRoot $targetPath
    }
    $targetDir = Split-Path -Parent $targetPath
    if ($targetDir -and -not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    $report | Set-Content -Path $targetPath -Encoding UTF8
    Write-Host "checklist=$targetPath"
}

if ($smokeExitCode -ne 0) {
    exit $smokeExitCode
}
if ($decision -eq "BLOCKED" -or $failCount -gt 0) {
    exit 1
}
exit 0
