param(
  [string]$BaseUrl = "http://127.0.0.1:20195",
  [ValidateSet("shared", "per_role", "per_user")]
  [string]$GroupStrategy = "per_role",
  [string]$SharedGroupId = "shared:openclaw",
  [switch]$EnableSharedMemory,
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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$homeDir = [Environment]::GetFolderPath("UserProfile")
$openclawDir = Join-Path $homeDir ".openclaw"
$configPath = Join-Path $openclawDir "openclaw.json"
$extensionsDir = Join-Path $openclawDir "extensions"
$targetDir = Join-Path $extensionsDir "minimem-memory"

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

$slots.memory = "minimem-memory"

if (-not ($entries.PSObject.Properties.Name -contains "minimem-memory")) {
  $entries | Add-Member -NotePropertyName "minimem-memory" -NotePropertyValue ([pscustomobject]@{})
}
$entry = $entries."minimem-memory"
Set-ObjectProperty -Object $entry -Name "enabled" -Value $true
Set-ObjectProperty -Object $entry -Name "config" -Value ([pscustomobject]@{
  baseUrl = $BaseUrl
  groupStrategy = $GroupStrategy
  sharedGroupId = $SharedGroupId
  autoSenderFromAgent = $true
  defaultRetrieveMethod = "agentic"
  defaultDecisionMode = "rule"
  autoInjectOnStart = $true
  autoCaptureOnEnd = $true
  autoCaptureCompression = $true
})

if (-not ($installs.PSObject.Properties.Name -contains "minimem-memory")) {
  $installs | Add-Member -NotePropertyName "minimem-memory" -NotePropertyValue ([pscustomobject]@{})
}
$install = $installs."minimem-memory"
Set-ObjectProperty -Object $install -Name "source" -Value "path"
Set-ObjectProperty -Object $install -Name "sourcePath" -Value $scriptDir
Set-ObjectProperty -Object $install -Name "installPath" -Value $targetDir
Set-ObjectProperty -Object $install -Name "version" -Value "0.1.0"
Set-ObjectProperty -Object $install -Name "installedAt" -Value ((Get-Date).ToUniversalTime().ToString("o"))

$config | ConvertTo-Json -Depth 32 | Set-Content -Path $configPath -Encoding UTF8

Write-Host "MiniMem OpenClaw plugin installed."
Write-Host "Config: $configPath"
Write-Host "Strategy: $GroupStrategy"
if ($GroupStrategy -eq "shared") {
  Write-Host "Shared group: $SharedGroupId"
}

if ($TryRestartGateway) {
  try {
    openclaw gateway call health --json | Out-Null
    Write-Host "Gateway detected. Restart if needed: openclaw gateway restart"
  } catch {
    Write-Host "Gateway not reachable. Start it with: openclaw gateway"
  }
}
