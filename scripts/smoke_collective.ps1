[CmdletBinding()]
param(
    [ValidateSet("dry-run", "health-check", "full")]
    [string]$Mode = "dry-run",
    [string]$BaseUrl = "http://127.0.0.1:20195",
    [int]$TimeoutSec = 20,
    [switch]$AllowMissingCollective,
    [switch]$StrictOpenClaw,
    [switch]$FailOnWarn
)

$ErrorActionPreference = "Stop"

try {
    Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
}
catch {
    # Best effort; request checks will report concrete errors if runtime type is missing.
}

$script:PassCount = 0
$script:WarnCount = 0
$script:FailCount = 0
$script:Results = @()
$script:HealthOk = $false

function Add-Result {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("PASS", "WARN", "FAIL")] [string]$Level,
        [Parameter(Mandatory = $true)][ValidateSet("P0", "P1", "P2")] [string]$Severity,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Detail
    )

    switch ($Level) {
        "PASS" { $script:PassCount += 1 }
        "WARN" { $script:WarnCount += 1 }
        "FAIL" { $script:FailCount += 1 }
    }

    $item = [PSCustomObject]@{
        level    = $Level
        severity = $Severity
        name     = $Name
        detail   = $Detail
    }
    $script:Results += $item
    Write-Host "[$Level][$Severity] $Name - $Detail"
}

function Test-CommandExists {
    param([Parameter(Mandatory = $true)] [string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-PathExists {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)][ValidateSet("P0", "P1", "P2")] [string]$Severity
    )
    if (Test-Path $Path) {
        Add-Result -Level PASS -Severity $Severity -Name $Name -Detail "$Path exists"
    }
    else {
        Add-Result -Level FAIL -Severity $Severity -Name $Name -Detail "$Path missing"
    }
}

function Get-HttpStatus {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("GET", "POST")] [string]$Method,
        [Parameter(Mandatory = $true)] [string]$Path,
        [string]$JsonBody = "{}"
    )
    $url = "$($BaseUrl.TrimEnd('/'))$Path"
    $client = [System.Net.Http.HttpClient]::new()
    try {
        if ($Method -eq "GET") {
            $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $url)
        }
        else {
            $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, $url)
            $request.Content = [System.Net.Http.StringContent]::new(
                $JsonBody,
                [System.Text.Encoding]::UTF8,
                "application/json"
            )
        }
        $response = $client.Send($request)
        return [int]$response.StatusCode
    }
    catch {
        return 0
    }
    finally {
        $client.Dispose()
    }
}

function Check-Health {
    for ($i = 1; $i -le $TimeoutSec; $i += 1) {
        $status = Get-HttpStatus -Method GET -Path "/health" -JsonBody ""
        if ($status -eq 200) {
            $script:HealthOk = $true
            Add-Result -Level PASS -Severity P0 -Name "health-check" -Detail "/health -> 200 in ${i}s"
            return
        }
        Start-Sleep -Seconds 1
    }
    Add-Result -Level FAIL -Severity P0 -Name "health-check" -Detail "/health not ready after ${TimeoutSec}s"
}

function Check-CollectiveEndpoint {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [string]$Path
    )
    $status = Get-HttpStatus -Method POST -Path $Path -JsonBody "{}"
    if ($status -eq 404) {
        if ($AllowMissingCollective) {
            Add-Result -Level WARN -Severity P2 -Name $Name -Detail "$Path -> 404 (allowed)"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name $Name -Detail "$Path -> 404 (missing)"
        }
        return
    }
    if ($status -eq 0) {
        Add-Result -Level FAIL -Severity P1 -Name $Name -Detail "$Path -> request failed"
        return
    }
    if ($status -ge 500) {
        Add-Result -Level FAIL -Severity P1 -Name $Name -Detail "$Path -> $status (server error)"
        return
    }
    Add-Result -Level PASS -Severity P1 -Name $Name -Detail "$Path -> $status"
}

function Check-McpSyntax {
    param([Parameter(Mandatory = $true)] [string]$RepoRoot)
    if (-not (Test-CommandExists "python")) {
        Add-Result -Level FAIL -Severity P0 -Name "python" -Detail "python command not found"
        return
    }
    Push-Location $RepoRoot
    try {
        python -m py_compile integrations/flockmem-mcp/server.py
        if ($LASTEXITCODE -eq 0) {
            Add-Result -Level PASS -Severity P1 -Name "mcp-syntax" -Detail "integrations/flockmem-mcp/server.py compiled"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name "mcp-syntax" -Detail "python compile failed"
        }
    }
    catch {
        Add-Result -Level FAIL -Severity P1 -Name "mcp-syntax" -Detail "python compile failed"
    }
    finally {
        Pop-Location
    }
}

function Check-McpImport {
    param([Parameter(Mandatory = $true)] [string]$RepoRoot)
    if (-not (Test-CommandExists "python")) {
        Add-Result -Level FAIL -Severity P0 -Name "python" -Detail "python command not found"
        return
    }
    Push-Location $RepoRoot
    try {
@'
import importlib.util
import pathlib

path = pathlib.Path("integrations/flockmem-mcp/server.py")
spec = importlib.util.spec_from_file_location("flockmem_mcp_server", path)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to build module spec")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(getattr(module, "MCP_NAME", "unknown"))
'@ | python -
        if ($LASTEXITCODE -eq 0) {
            Add-Result -Level PASS -Severity P1 -Name "mcp-import" -Detail "MCP module import succeeded"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name "mcp-import" -Detail "MCP module import failed (check fastmcp dependency)"
        }
    }
    catch {
        Add-Result -Level FAIL -Severity P1 -Name "mcp-import" -Detail "MCP module import failed (check fastmcp dependency)"
    }
    finally {
        Pop-Location
    }
}

function Check-PluginPackDryRun {
    param([Parameter(Mandatory = $true)] [string]$RepoRoot)
    if (-not (Test-CommandExists "npm")) {
        Add-Result -Level FAIL -Severity P1 -Name "plugin-pack-dry-run" -Detail "npm command not found"
        return
    }
    $pluginRoot = Join-Path $RepoRoot "integrations/openclaw-plugin"
    Push-Location $pluginRoot
    try {
        npm pack --dry-run | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Add-Result -Level PASS -Severity P1 -Name "plugin-pack-dry-run" -Detail "npm pack --dry-run succeeded"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name "plugin-pack-dry-run" -Detail "npm pack --dry-run failed"
        }
    }
    catch {
        Add-Result -Level FAIL -Severity P1 -Name "plugin-pack-dry-run" -Detail "npm pack --dry-run failed"
    }
    finally {
        Pop-Location
    }
}

function Check-OpenClawCli {
    if (Test-CommandExists "openclaw") {
        Add-Result -Level PASS -Severity P2 -Name "openclaw-cli" -Detail "openclaw command found"
        return
    }
    if ($StrictOpenClaw) {
        Add-Result -Level FAIL -Severity P1 -Name "openclaw-cli" -Detail "openclaw command not found (strict mode)"
    }
    else {
        Add-Result -Level WARN -Severity P2 -Name "openclaw-cli" -Detail "openclaw command not found"
    }
}

function Check-SdkImport {
    param([Parameter(Mandatory = $true)] [string]$RepoRoot)
    if (-not (Test-CommandExists "python")) {
        Add-Result -Level FAIL -Severity P0 -Name "python" -Detail "python command not found"
        return
    }
    Push-Location $RepoRoot
    try {
@'
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path("src").resolve()))
import flockmem  # noqa: F401
print("flockmem-import-ok")
'@ | python -
        if ($LASTEXITCODE -eq 0) {
            Add-Result -Level PASS -Severity P1 -Name "sdk-import" -Detail "flockmem import succeeded from local source"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name "sdk-import" -Detail "flockmem import failed"
        }
    }
    catch {
        Add-Result -Level FAIL -Severity P1 -Name "sdk-import" -Detail "flockmem import failed"
    }
    finally {
        Pop-Location
    }
}

function Check-SdkHealthCall {
    if (-not (Test-CommandExists "python")) {
        Add-Result -Level FAIL -Severity P0 -Name "python" -Detail "python command not found"
        return
    }
    try {
        $env:FLOCKMEM_BASE_URL = $BaseUrl
@'
import json
import os
from urllib import request

base = os.environ.get("FLOCKMEM_BASE_URL", "http://127.0.0.1:20195").rstrip("/")
url = base + "/health"
with request.urlopen(url, timeout=10) as resp:
    status = resp.status
    body = resp.read().decode("utf-8", errors="ignore")
print(status)
print(body[:200])
'@ | python -
        if ($LASTEXITCODE -eq 0) {
            Add-Result -Level PASS -Severity P1 -Name "sdk-health-call" -Detail "python SDK health probe succeeded"
        }
        else {
            Add-Result -Level FAIL -Severity P1 -Name "sdk-health-call" -Detail "python SDK health probe failed"
        }
    }
    catch {
        Add-Result -Level FAIL -Severity P1 -Name "sdk-health-call" -Detail "python SDK health probe failed"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "tmp_test_runtime"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "== FlockMem Collective Smoke (DevOps-lite) =="
Write-Host "mode=$Mode base_url=$BaseUrl timeout_sec=$TimeoutSec allow_missing_collective=$AllowMissingCollective strict_openclaw=$StrictOpenClaw fail_on_warn=$FailOnWarn"

Test-PathExists -Path (Join-Path $repoRoot "integrations/flockmem-mcp/server.py") -Name "mcp-server-path" -Severity P1
Test-PathExists -Path (Join-Path $repoRoot "integrations/openclaw-plugin/install.ps1") -Name "plugin-install-ps1" -Severity P1
Test-PathExists -Path (Join-Path $repoRoot "integrations/openclaw-plugin/install.sh") -Name "plugin-install-sh" -Severity P1
Test-PathExists -Path (Join-Path $repoRoot "integrations/openclaw-plugin/package.json") -Name "plugin-package-json" -Severity P1
Test-PathExists -Path (Join-Path $repoRoot "pyproject.toml") -Name "sdk-pyproject" -Severity P1

if (Test-CommandExists "python") {
    Add-Result -Level PASS -Severity P0 -Name "python" -Detail "python command found"
}
else {
    Add-Result -Level FAIL -Severity P0 -Name "python" -Detail "python command not found"
}

if (Test-CommandExists "npm") {
    Add-Result -Level PASS -Severity P1 -Name "npm" -Detail "npm command found"
}
else {
    Add-Result -Level FAIL -Severity P1 -Name "npm" -Detail "npm command not found"
}

Check-OpenClawCli
Check-McpSyntax -RepoRoot $repoRoot
Check-PluginPackDryRun -RepoRoot $repoRoot
Check-SdkImport -RepoRoot $repoRoot

if ($Mode -in @("health-check", "full")) {
    Check-Health
    if ($script:HealthOk) {
        Check-SdkHealthCall
    }
}

if ($Mode -eq "full") {
    Check-McpImport -RepoRoot $repoRoot
    Check-CollectiveEndpoint -Name "collective-ingest" -Path "/api/v1/collective/ingest"
    Check-CollectiveEndpoint -Name "collective-context" -Path "/api/v1/collective/context"
    Check-CollectiveEndpoint -Name "collective-feedback" -Path "/api/v1/collective/feedback"
}

$blockingFails = @(
    $script:Results | Where-Object { $_.level -eq "FAIL" -and $_.severity -in @("P0", "P1") }
)
$warns = @($script:Results | Where-Object { $_.level -eq "WARN" })

Write-Host "== Summary =="
Write-Host "PASS=$($script:PassCount) WARN=$($script:WarnCount) FAIL=$($script:FailCount) BLOCKING_FAILS=$($blockingFails.Count)"

if ($blockingFails.Count -gt 0) {
    Write-Host "== Blocking Failures (P0/P1) =="
    foreach ($item in $blockingFails) {
        Write-Host "- [$($item.severity)] $($item.name): $($item.detail)"
    }
    exit 1
}

if ($FailOnWarn -and $warns.Count -gt 0) {
    Write-Host "== Warn As Fail Enabled =="
    foreach ($item in $warns) {
        Write-Host "- [$($item.severity)] $($item.name): $($item.detail)"
    }
    exit 2
}

exit 0
