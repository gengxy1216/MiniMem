#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:20195"
GROUP_STRATEGY="per_role"
SHARED_GROUP_ID="shared:openclaw"
ENABLE_SHARED_MEMORY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --group-strategy)
      GROUP_STRATEGY="${2:-}"
      shift 2
      ;;
    --shared-group-id)
      SHARED_GROUP_ID="${2:-}"
      shift 2
      ;;
    --shared)
      ENABLE_SHARED_MEMORY="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$ENABLE_SHARED_MEMORY" == "true" ]]; then
  GROUP_STRATEGY="shared"
fi

case "$GROUP_STRATEGY" in
  shared|per_role|per_user) ;;
  *)
    echo "Invalid group strategy: $GROUP_STRATEGY" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_DIR="${HOME}/.openclaw"
CONFIG_PATH="${OPENCLAW_DIR}/openclaw.json"
EXT_DIR="${OPENCLAW_DIR}/extensions"
TARGET_DIR="${EXT_DIR}/minimem-memory"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "OpenClaw config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$EXT_DIR"
rm -rf "$TARGET_DIR"
cp -R "$SCRIPT_DIR" "$TARGET_DIR"

python3 - "$CONFIG_PATH" "$SCRIPT_DIR" "$TARGET_DIR" "$BASE_URL" "$GROUP_STRATEGY" "$SHARED_GROUP_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone

config_path, source_path, install_path, base_url, strategy, shared_group = sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

plugins = cfg.setdefault("plugins", {})
slots = plugins.setdefault("slots", {})
entries = plugins.setdefault("entries", {})
installs = plugins.setdefault("installs", {})

slots["memory"] = "minimem-memory"

entry = entries.setdefault("minimem-memory", {})
entry["enabled"] = True
entry["config"] = {
    "baseUrl": base_url,
    "groupStrategy": strategy,
    "sharedGroupId": shared_group,
    "autoSenderFromAgent": True,
    "defaultRetrieveMethod": "agentic",
    "defaultDecisionMode": "rule",
    "autoInjectOnStart": True,
    "autoCaptureOnEnd": True,
    "autoCaptureCompression": True,
}

install = installs.setdefault("minimem-memory", {})
install["source"] = "path"
install["sourcePath"] = source_path
install["installPath"] = install_path
install["version"] = "0.1.0"
install["installedAt"] = datetime.now(timezone.utc).isoformat()

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY

echo "MiniMem OpenClaw plugin installed."
echo "Config: $CONFIG_PATH"
echo "Strategy: $GROUP_STRATEGY"
if [[ "$GROUP_STRATEGY" == "shared" ]]; then
  echo "Shared group: $SHARED_GROUP_ID"
fi

echo "If gateway is running, restart it: openclaw gateway restart"
