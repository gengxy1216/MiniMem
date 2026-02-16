#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_AFTER_INSTALL="${1:-}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[MiniMem] Python not found. Please install Python 3.11+ first."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[MiniMem] Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "[MiniMem] Virtual environment is broken: $VENV_PY not found."
  exit 1
fi

echo "[MiniMem] Installing dependencies..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -e .

echo
echo "[MiniMem] Install completed."
echo "Start command: .venv/bin/minimem"
echo "UI URL: http://127.0.0.1:20195/ui"

if [ "$RUN_AFTER_INSTALL" = "--run" ]; then
  echo "[MiniMem] Launching..."
  exec ".venv/bin/minimem"
fi
