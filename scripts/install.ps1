param(
    [switch]$RunAfterInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Cmd = "py"; Args = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python3"; Args = @() }
    }
    throw "Python not found. Please install Python 3.11+ first."
}

$PyCmd = Resolve-PythonCommand

if (!(Test-Path ".venv")) {
    Write-Host "[MiniMem] Creating virtual environment..."
    & $PyCmd.Cmd @($PyCmd.Args) -m venv .venv
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    throw "Virtual environment is broken: $VenvPython not found."
}

Write-Host "[MiniMem] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e .

Write-Host ""
Write-Host "[MiniMem] Install completed."
Write-Host "Start command: .venv\Scripts\minimem.exe"
Write-Host "UI URL: http://127.0.0.1:20195/ui"

if ($RunAfterInstall) {
    Write-Host "[MiniMem] Launching..."
    & (Join-Path $Root ".venv\Scripts\minimem.exe")
}
