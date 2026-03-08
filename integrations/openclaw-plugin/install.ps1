param(
  [string]$BaseUrl = "http://127.0.0.1:20195",
  [ValidateSet("shared", "per_role", "per_user")]
  [string]$GroupStrategy = "per_role",
  [string]$SharedGroupId = "shared:openclaw",
  [switch]$EnableSharedMemory,
  [switch]$DisablePrimarySync,
  [switch]$ForcePrimarySync,
  [string]$FlockMemConfigPath = "",
  [switch]$TryRestartGateway
)

$ErrorActionPreference = "Stop"

function Ensure-ObjectProperty {
  param(
    [Parameter(Mandatory = $true)] [object]$Object,
    [Parameter(Mandatory = $true)] [string]$Name
  )
  if (-not ($Object.PSObject.Properties.Name -contains $Name)) {
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue ([pscustomobject]@{})
  }
  return $Object.$Name
}

function Set-ObjectProperty {
  param(
    [Parameter(Mandatory = $true)] [object]$Object,
    [Parameter(Mandatory = $true)] [string]$Name,
    [Parameter(Mandatory = $true)] $Value
  )
  if ($Object.PSObject.Properties.Name -contains $Name) {
    $Object.$Name = $Value
  } else {
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
  }
}

if ($EnableSharedMemory) {
  $GroupStrategy = "shared"
}
$inheritPrimaryModel = -not $DisablePrimarySync.IsPresent
$forcePrimarySyncEnabled = $ForcePrimarySync.IsPresent

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$homeDir = [Environment]::GetFolderPath("UserProfile")
$openclawDir = Join-Path $homeDir ".openclaw"
$configPath = Join-Path $openclawDir "openclaw.json"
$extensionsDir = Join-Path $openclawDir "extensions"
$targetDir = Join-Path $extensionsDir "flockmem-memory"

if (-not (Test-Path $configPath)) {
  throw "OpenClaw config not found: $configPath"
}

New-Item -ItemType Directory -Path $extensionsDir -Force | Out-Null
if (Test-Path $targetDir) {
  Remove-Item -Recurse -Force $targetDir
}
Copy-Item -Recurse -Force $scriptDir $targetDir

$jsonText = Get-Content -Raw -Path $configPath -Encoding UTF8
$config = $jsonText | ConvertFrom-Json

$plugins = Ensure-ObjectProperty -Object $config -Name "plugins"
$slots = Ensure-ObjectProperty -Object $plugins -Name "slots"
$entries = Ensure-ObjectProperty -Object $plugins -Name "entries"
$installs = Ensure-ObjectProperty -Object $plugins -Name "installs"

$syncResult = [pscustomobject]@{
  status = "sync_unavailable"
  warning = ""
  snapshot = [pscustomobject]@{
    provider = ""
    baseUrl = ""
    model = ""
  }
  senderMap = @{}
  channelGroupMap = @{}
  sharePolicy = @{}
}
try {
  $syncPy = @'
import json
import sys
from pathlib import Path

args = sys.argv[1:]
if len(args) < 4:
    raise SystemExit("expected at least 4 args: config_path script_dir inherit_raw force_raw [minimem_config_path]")
config_path, script_dir, inherit_raw, force_raw = args[:4]
minimem_config_path = args[4] if len(args) >= 5 else ""
if minimem_config_path == "__EMPTY__":
    minimem_config_path = ""

result = {
    "status": "sync_unavailable",
    "warning": "",
    "snapshot": {"provider": "", "baseUrl": "", "model": ""},
    "senderMap": {},
    "channelGroupMap": {},
    "sharePolicy": {},
}
try:
    repo_root = Path(script_dir).resolve().parents[1]
    repo_src = repo_root / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    from flockmem.config.openclaw_primary_sync import (
        detect_primary_model_snapshot,
        sync_openclaw_primary_to_minimem_config,
        to_public_primary_snapshot,
    )

    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    def _to_str(value):
        return str(value or "").strip()

    def _iter_items(raw):
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    item = dict(value)
                    if "id" not in item:
                        item["id"] = key
                    yield item
            return
        if isinstance(raw, list):
            for value in raw:
                if isinstance(value, dict):
                    yield value

    def _as_list(raw):
        if isinstance(raw, list):
            return [_to_str(x) for x in raw if _to_str(x)]
        text = _to_str(raw)
        if not text:
            return []
        return [seg.strip() for seg in text.replace(";", ",").split(",") if seg.strip()]

    sender_map = {}
    for item in _iter_items(cfg.get("agents")):
        agent_id = _to_str(item.get("id") or item.get("agent_id") or item.get("agentId") or item.get("name"))
        sender = _to_str(item.get("sender") or item.get("sender_id") or item.get("senderId") or item.get("memorySender"))
        if agent_id.lower() == "defaults":
            continue
        if agent_id:
            sender_map[agent_id] = sender or agent_id
    result["senderMap"] = sender_map

    channel_group_map = {}
    for item in _iter_items(cfg.get("channels")):
        channel_id = _to_str(
            item.get("id") or item.get("channel") or item.get("channel_id") or item.get("channelId") or item.get("name")
        )
        group_id = _to_str(
            item.get("group_id") or item.get("groupId") or item.get("memoryGroup") or item.get("memory_group")
        )
        if channel_id:
            channel_group_map[channel_id] = group_id or f"channel:{channel_id}"
    result["channelGroupMap"] = channel_group_map

    share_policy = {}
    raw_policy = cfg.get("sharePolicy")
    if isinstance(raw_policy, dict):
        for group_id, policy in raw_policy.items():
            gid = _to_str(group_id)
            if not gid or not isinstance(policy, dict):
                continue
            share_policy[gid] = {
                "readableAgents": _as_list(policy.get("readableAgents") or policy.get("readable_agents") or policy.get("readers")),
                "writableAgents": _as_list(policy.get("writableAgents") or policy.get("writable_agents") or policy.get("writers")),
            }
    result["sharePolicy"] = share_policy

    snap = detect_primary_model_snapshot(cfg)
    public_snap = to_public_primary_snapshot(snap)
    result["snapshot"] = {
        "provider": str(public_snap.get("provider", "")).strip(),
        "baseUrl": str(public_snap.get("base_url", "")).strip(),
        "model": str(public_snap.get("model", "")).strip(),
    }

    inherit_enabled = str(inherit_raw).strip().lower() not in {"0", "false", "no", "off"}
    force_enabled = str(force_raw).strip().lower() in {"1", "true", "yes", "on"}
    target = str(minimem_config_path).strip()
    sync_result = sync_openclaw_primary_to_minimem_config(
        openclaw_config_path=Path(config_path),
        minimem_config_path=Path(target) if target else None,
        inherit_primary_model=inherit_enabled,
        force=force_enabled,
    )
    result["status"] = str(sync_result.get("status", "")).strip() or "unknown"
except Exception as exc:  # noqa: BLE001
    result["status"] = "sync_failed"
    result["warning"] = str(exc)

print(json.dumps(result, ensure_ascii=False))
'@
  $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ("minimem_openclaw_sync_" + [Guid]::NewGuid().ToString("N") + ".py")
  $tmpErr = Join-Path ([System.IO.Path]::GetTempPath()) ("minimem_openclaw_sync_err_" + [Guid]::NewGuid().ToString("N") + ".log")
  $miniMemArg = if ([string]::IsNullOrWhiteSpace($FlockMemConfigPath)) { "__EMPTY__" } else { $FlockMemConfigPath }
  try {
    Set-Content -Path $tmpPy -Value $syncPy -Encoding UTF8
    $syncJson = python $tmpPy $configPath $scriptDir $inheritPrimaryModel $forcePrimarySyncEnabled $miniMemArg 2> $tmpErr
    if ($LASTEXITCODE -ne 0) {
      $errText = ""
      if (Test-Path $tmpErr) {
        $errText = (Get-Content -Raw -Path $tmpErr -Encoding UTF8)
      }
      throw "sync helper failed (exit=$LASTEXITCODE): $errText"
    }
  } finally {
    if (Test-Path $tmpPy) {
      Remove-Item -Force $tmpPy
    }
    if (Test-Path $tmpErr) {
      Remove-Item -Force $tmpErr
    }
  }
  $parsed = $syncJson | ConvertFrom-Json
  if ($null -ne $parsed) {
    $syncResult = $parsed
  }
} catch {
  Write-Warning "Primary model sync is unavailable: $($_.Exception.Message)"
}

$senderMap = if ($null -ne $syncResult.senderMap) { $syncResult.senderMap } else { @{} }
$channelGroupMap = if ($null -ne $syncResult.channelGroupMap) { $syncResult.channelGroupMap } else { @{} }
$sharePolicy = if ($null -ne $syncResult.sharePolicy) { $syncResult.sharePolicy } else { @{} }

$slots.memory = "flockmem-memory"

if (-not ($entries.PSObject.Properties.Name -contains "flockmem-memory")) {
  $entries | Add-Member -NotePropertyName "flockmem-memory" -NotePropertyValue ([pscustomobject]@{})
}
$entry = $entries."flockmem-memory"
Set-ObjectProperty -Object $entry -Name "enabled" -Value $true
Set-ObjectProperty -Object $entry -Name "config" -Value ([pscustomobject]@{
  baseUrl = $BaseUrl
  groupStrategy = $GroupStrategy
  sharedGroupId = $SharedGroupId
  defaultAgentId = "main"
  autoSenderFromAgent = $true
  defaultRetrieveMethod = "agentic"
  defaultDecisionMode = "rule"
  autoInjectOnStart = $true
  autoCaptureOnEnd = $true
  autoCaptureCompression = $true
  inheritPrimaryModel = [bool]$inheritPrimaryModel
  primaryModelSnapshot = [pscustomobject]@{
    provider = [string]$syncResult.snapshot.provider
    baseUrl = [string]$syncResult.snapshot.baseUrl
    model = [string]$syncResult.snapshot.model
  }
  primaryModelSyncStatus = [string]$syncResult.status
  senderMap = $senderMap
  channelGroupMap = $channelGroupMap
  sharePolicy = $sharePolicy
})

if (-not ($installs.PSObject.Properties.Name -contains "flockmem-memory")) {
  $installs | Add-Member -NotePropertyName "flockmem-memory" -NotePropertyValue ([pscustomobject]@{})
}
$install = $installs."flockmem-memory"
Set-ObjectProperty -Object $install -Name "source" -Value "path"
Set-ObjectProperty -Object $install -Name "sourcePath" -Value $scriptDir
Set-ObjectProperty -Object $install -Name "installPath" -Value $targetDir
Set-ObjectProperty -Object $install -Name "version" -Value "0.2.0"
Set-ObjectProperty -Object $install -Name "installedAt" -Value ((Get-Date).ToUniversalTime().ToString("o"))

$config | ConvertTo-Json -Depth 32 | Set-Content -Path $configPath -Encoding UTF8

Write-Host "FlockMem OpenClaw plugin installed."
Write-Host "Config: $configPath"
Write-Host "Strategy: $GroupStrategy"
if ($GroupStrategy -eq "shared") {
  Write-Host "Shared group: $SharedGroupId"
}
if ($inheritPrimaryModel) {
  Write-Host "Primary model inheritance: enabled"
} else {
  Write-Host "Primary model inheritance: disabled"
}
Write-Host "Primary model sync status: $($syncResult.status)"
if ($syncResult.warning) {
  Write-Host "Primary model sync warning: $($syncResult.warning)"
}

if ($TryRestartGateway) {
  try {
    openclaw gateway call health --json | Out-Null
    Write-Host "Gateway detected. Restart if needed: openclaw gateway restart"
  } catch {
    Write-Host "Gateway not reachable. Start it with: openclaw gateway"
  }
}


