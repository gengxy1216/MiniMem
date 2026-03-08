#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/flockmem" ] && [ ! -x ".venv/bin/minimem" ]; then
  echo "[FlockMem] Not installed yet. Running installer first..."
  bash scripts/install.sh
fi

if [ -x ".venv/bin/flockmem" ]; then
  exec ".venv/bin/flockmem"
fi
exec ".venv/bin/minimem"

