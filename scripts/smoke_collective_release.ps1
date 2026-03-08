[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:20195",
    [int]$TimeoutSec = 60,
    [switch]$NoAutoStart,
    [switch]$AllowMissingCollective,
    [switch]$StrictOpenClaw
)

$ErrorActionPreference = "Stop"

$script:PassCount = 0
$script:WarnCount = 0
$script:FailCount = 0
$script:StartedServer = $null

function Add-Result {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("PASS", "WARN", "FAIL")] [string]$Level,
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [string]$Detail
    )
    switch ($Level) {
        "PASS" { $script:PassCount += 1 }
        "WARN" { $script:WarnCount += 1 }
        "FAIL" { $script:FailCount += 1 }
    }
    Write-Host "[$Level] $Name - $Detail"
}

function Test-CommandExists {
    param([Parameter(Mandatory = $true)] [string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-HttpStatus {
    param(
        [Parameter(Mandatory = $true)] [ValidateSet("GET", "POST")] [string]$Method,
        [Parameter(Mandatory = $true)] [string]$Path,
        [string]$JsonBody = "{}"
    )
    $url = "$($BaseUrl.TrimEnd('/'))$Path"
    try {
        if ($Method -eq "GET") {
            $response = Invoke-WebRequest -Uri $url -Method Get -UseBasicParsing -TimeoutSec 5
        }
        else {
            $response = Invoke-WebRequest -Uri $url -Method Post -Body $JsonBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 5
        }
        return [int]$response.StatusCode
    }
    catch {
        if ($null -ne $_.Exception.Response -and $null -ne $_.Exception.Response.StatusCode) {
            return [int]$_.Exception.Response.StatusCode.value__
        }
        return 0
    }
}

function Test-HealthOk {
    return (Get-HttpStatus -Method GET -Path "/health" -JsonBody "") -eq 200
}

function Start-ServerIfNeeded {
    param(
        [Parameter(Mandatory = $true)] [string]$RepoRoot,
        [Parameter(Mandatory = $true)] [string]$ServerLog,
        [Parameter(Mandatory = $true)] [string]$ServerErrLog
    )
    if (Test-HealthOk) {
        Add-Result -Level PASS -Name "health-precheck" -Detail "service already reachable at $BaseUrl/health"
        return
    }
    if ($NoAutoStart) {
        Add-Result -Level FAIL -Name "health-precheck" -Detail "service unreachable and -NoAutoStart enabled"
        return
    }
    if (-not (Test-CommandExists "python")) {
        Add-Result -Level FAIL -Name "server-start" -Detail "python command not found"
        return
    }

    $uri = [Uri]$BaseUrl
    $env:LITE_HOST = $uri.Host
    $env:LITE_PORT = [string]$uri.Port

    $proc = Start-Process -FilePath "python" -ArgumentList "main.py" `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $ServerLog `
        -RedirectStandardError $ServerErrLog `
        -PassThru
    $script:StartedServer = $proc
    Add-Result -Level PASS -Name "server-start" -Detail "spawned local server pid=$($proc.Id) (log: $ServerLog)"

    for ($i = 1; $i -le $TimeoutSec; $i += 1) {
        if (Test-HealthOk) {
            Add-Result -Level PASS -Name "health-check" -Detail "service became healthy in ${i}s"
            return
        }
        Start-Sleep -Seconds 1
    }
    Add-Result -Level FAIL -Name "health-check" -Detail "service not healthy after ${TimeoutSec}s; see $ServerLog"
}

function Check-CollectiveEndpoint {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$JsonBody,
        [int[]]$ExpectedStatuses = @(200)
    )
    $status = Get-HttpStatus -Method POST -Path $Path -JsonBody $JsonBody
    if ($status -eq 404) {
        if ($AllowMissingCollective) {
            Add-Result -Level WARN -Name $Name -Detail "$Path -> 404 (allowed)"
        }
        else {
            Add-Result -Level FAIL -Name $Name -Detail "$Path -> 404 (endpoint missing)"
        }
        return
    }
    if ($status -eq 0) {
        Add-Result -Level FAIL -Name $Name -Detail "$Path -> request failed"
        return
    }
    if ($status -ge 500) {
        Add-Result -Level FAIL -Name $Name -Detail "$Path -> $status (server error)"
        return
    }
    if ($ExpectedStatuses -contains $status) {
        Add-Result -Level PASS -Name $Name -Detail "$Path -> $status"
        return
    }
    Add-Result -Level FAIL -Name $Name -Detail "$Path -> $status (expected: $($ExpectedStatuses -join ','))"
}

function Check-McpAndPlugin {
    param([Parameter(Mandatory = $true)] [string]$RepoRoot)

    if (Test-CommandExists "python") {
        Push-Location $RepoRoot
        try {
            python -m py_compile integrations/flockmem-mcp/server.py
            Add-Result -Level PASS -Name "mcp-syntax" -Detail "integrations/flockmem-mcp/server.py compiled"
        }
        catch {
            Add-Result -Level FAIL -Name "mcp-syntax" -Detail "python compile failed"
        }
        try {
            python -c "import importlib.util,pathlib; p=pathlib.Path('integrations/flockmem-mcp/server.py'); s=importlib.util.spec_from_file_location('flockmem_mcp_server', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(getattr(m,'MCP_NAME','unknown'))"
            Add-Result -Level PASS -Name "mcp-import" -Detail "MCP module import succeeded"
        }
        catch {
            Add-Result -Level FAIL -Name "mcp-import" -Detail "MCP module import failed (check fastmcp dependency)"
        }
        Pop-Location
    }
    else {
        Add-Result -Level FAIL -Name "mcp-check" -Detail "python command not found"
    }

    $pluginInstallSh = Join-Path $RepoRoot "integrations/openclaw-plugin/install.sh"
    $pluginInstallPs1 = Join-Path $RepoRoot "integrations/openclaw-plugin/install.ps1"
    if ((Test-Path $pluginInstallSh) -and (Test-Path $pluginInstallPs1)) {
        Add-Result -Level PASS -Name "plugin-install-scripts" -Detail "install scripts found"
    }
    else {
        Add-Result -Level FAIL -Name "plugin-install-scripts" -Detail "install.sh/install.ps1 missing"
    }

    if (Test-CommandExists "npm") {
        $pluginDir = Join-Path $RepoRoot "integrations/openclaw-plugin"
        Push-Location $pluginDir
        try {
            npm pack --dry-run | Out-Null
            Add-Result -Level PASS -Name "plugin-pack-dry-run" -Detail "npm pack --dry-run succeeded"
        }
        catch {
            Add-Result -Level FAIL -Name "plugin-pack-dry-run" -Detail "npm pack --dry-run failed"
        }
        Pop-Location
    }
    else {
        Add-Result -Level FAIL -Name "plugin-pack-dry-run" -Detail "npm command not found"
    }

    if (Test-CommandExists "openclaw") {
        Add-Result -Level PASS -Name "openclaw-cli" -Detail "openclaw command found"
    }
    elseif ($StrictOpenClaw) {
        Add-Result -Level FAIL -Name "openclaw-cli" -Detail "openclaw command not found (strict mode)"
    }
    else {
        Add-Result -Level WARN -Name "openclaw-cli" -Detail "openclaw command not found"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "tmp_test_runtime"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$serverLog = Join-Path $logDir "collective-smoke-server.log"
$serverErrLog = Join-Path $logDir "collective-smoke-server.err.log"

Write-Host "== FlockMem Collective Release Smoke =="
Write-Host "base_url=$BaseUrl timeout_sec=$TimeoutSec auto_start=$(-not $NoAutoStart) allow_missing_collective=$AllowMissingCollective strict_openclaw=$StrictOpenClaw"

if (-not (Test-CommandExists "curl")) {
    Add-Result -Level FAIL -Name "prerequisite" -Detail "curl command not found"
}

try {
    Start-ServerIfNeeded -RepoRoot $repoRoot -ServerLog $serverLog -ServerErrLog $serverErrLog

    if (Test-HealthOk) {
        Add-Result -Level PASS -Name "health" -Detail "/health returned 200"
    }
    else {
        Add-Result -Level FAIL -Name "health" -Detail "/health not reachable"
    }

    $ingestPayload = @{
        knowledge_id = "smoke-k-1"
        scope_type = "personal"
        scope_id = "smoke-user"
        content = @{ text = "collective smoke ingest" }
        changed_by = "agent"
        actor_id = "smoke-agent"
        write_acl = @("smoke-agent")
        coordination_mode = "inruntime_a2a"
        coordination_id = "smoke-coord-1"
        runtime_id = "codex"
        agent_id = "smoke-agent"
        session_id = "smoke-session-1"
    } | ConvertTo-Json -Depth 8 -Compress
    $contextPayload = @{
        query = "collective smoke context"
        actor_id = "smoke-agent"
        personal_scope_id = "smoke-user"
        include_global = $false
        top_k = 5
        coordination_mode = "inruntime_a2a"
        coordination_id = "smoke-coord-2"
        runtime_id = "codex"
        agent_id = "smoke-agent"
        session_id = "smoke-session-2"
    } | ConvertTo-Json -Depth 8 -Compress
    $feedbackPayload = @{
        knowledge_id = "smoke-k-1"
        feedback_type = "execution_signal"
        feedback_payload = @{ outcome_status = "success" }
        actor = "smoke-agent"
        coordination_mode = "inruntime_a2a"
        coordination_id = "smoke-coord-3"
        runtime_id = "codex"
        agent_id = "smoke-agent"
        session_id = "smoke-session-3"
    } | ConvertTo-Json -Depth 8 -Compress

    Check-CollectiveEndpoint -Name "collective-ingest" -Path "/api/v1/collective/ingest" -JsonBody $ingestPayload -ExpectedStatuses @(200)
    Check-CollectiveEndpoint -Name "collective-context" -Path "/api/v1/collective/context" -JsonBody $contextPayload -ExpectedStatuses @(200)
    Check-CollectiveEndpoint -Name "collective-feedback" -Path "/api/v1/collective/feedback" -JsonBody $feedbackPayload -ExpectedStatuses @(200)

    Check-McpAndPlugin -RepoRoot $repoRoot
}
finally {
    if ($null -ne $script:StartedServer) {
        try {
            Stop-Process -Id $script:StartedServer.Id -Force -ErrorAction Stop
            Add-Result -Level PASS -Name "server-stop" -Detail "stopped local server pid=$($script:StartedServer.Id)"
        }
        catch {
            Add-Result -Level WARN -Name "server-stop" -Detail "failed to stop local server pid=$($script:StartedServer.Id): $($_.Exception.Message)"
        }
    }
}

Write-Host "== Summary =="
Write-Host "PASS=$($script:PassCount) WARN=$($script:WarnCount) FAIL=$($script:FailCount)"

if ($script:FailCount -gt 0) {
    exit 1
}
exit 0
