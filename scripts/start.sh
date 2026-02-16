#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/minimem" ]; then
  echo "[MiniMem] Not installed yet. Running installer first..."
  bash scripts/install.sh
fi

exec ".venv/bin/minimem"
