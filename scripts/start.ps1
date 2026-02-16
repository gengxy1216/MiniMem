$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Launcher = Join-Path $Root ".venv\Scripts\minimem.exe"
if (!(Test-Path $Launcher)) {
    Write-Host "[MiniMem] Not installed yet. Running installer first..."
    & (Join-Path $Root "scripts\install.ps1")
}

& (Join-Path $Root ".venv\Scripts\minimem.exe")
