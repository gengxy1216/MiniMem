$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Launcher = Join-Path $Root ".venv\Scripts\flockmem.exe"
if (!(Test-Path $Launcher)) {
    $Launcher = Join-Path $Root ".venv\Scripts\minimem.exe"
}
if (!(Test-Path $Launcher)) {
    Write-Host "[FlockMem] Not installed yet. Running installer first..."
    & (Join-Path $Root "scripts\install.ps1")
    $Launcher = Join-Path $Root ".venv\Scripts\flockmem.exe"
    if (!(Test-Path $Launcher)) {
        $Launcher = Join-Path $Root ".venv\Scripts\minimem.exe"
    }
}

& $Launcher

