@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start.ps1"
if errorlevel 1 (
  echo.
  echo [MiniMem] Failed to start.
  pause
  exit /b 1
)
