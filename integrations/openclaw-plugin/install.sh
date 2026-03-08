#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:20195"
GROUP_STRATEGY="per_role"
SHARED_GROUP_ID="shared:openclaw"
ENABLE_SHARED_MEMORY="false"
INHERIT_PRIMARY_MODEL="true"
FORCE_PRIMARY_SYNC="false"
MINIMEM_CONFIG_PATH=""

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
    --disable-primary-sync)
      INHERIT_PRIMARY_MODEL="false"
      shift
      ;;
    --force-primary-sync)
      FORCE_PRIMARY_SYNC="true"
      shift
      ;;
    --minimem-config)
      MINIMEM_CONFIG_PATH="${2:-}"
      shift 2
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
TARGET_DIR="${EXT_DIR}/flockmem-memory"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "OpenClaw config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$EXT_DIR"
rm -rf "$TARGET_DIR"
cp -R "$SCRIPT_DIR" "$TARGET_DIR"

python3 - "$CONFIG_PATH" "$SCRIPT_DIR" "$TARGET_DIR" "$BASE_URL" "$GROUP_STRATEGY" "$SHARED_GROUP_ID" "$INHERIT_PRIMARY_MODEL" "$FORCE_PRIMARY_SYNC" "$MINIMEM_CONFIG_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

config_path, source_path, install_path, base_url, strategy, shared_group, inherit_primary_raw, force_sync_raw, minimem_config_raw = sys.argv[1:]
with open(config_path, "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)

inherit_primary_model = str(inherit_primary_raw).strip().lower() not in {"0", "false", "no", "off"}
force_primary_sync = str(force_sync_raw).strip().lower() in {"1", "true", "yes", "on"}
minimem_config_path = str(minimem_config_raw).strip() or None

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

def _collect_sender_map(cfg_obj):
    out = {}
    for item in _iter_items(cfg_obj.get("agents")):
        agent_id = _to_str(item.get("id") or item.get("agent_id") or item.get("agentId") or item.get("name"))
        sender = _to_str(item.get("sender") or item.get("sender_id") or item.get("senderId") or item.get("memorySender"))
        if agent_id.lower() == "defaults":
            continue
        if agent_id:
            out[agent_id] = sender or agent_id
    return out

def _collect_channel_group_map(cfg_obj):
    out = {}
    for item in _iter_items(cfg_obj.get("channels")):
        channel_id = _to_str(
            item.get("id") or item.get("channel") or item.get("channel_id") or item.get("channelId") or item.get("name")
        )
        group_id = _to_str(
            item.get("group_id") or item.get("groupId") or item.get("memoryGroup") or item.get("memory_group")
        )
        if channel_id:
            out[channel_id] = group_id or f"channel:{channel_id}"
    return out

def _collect_share_policy(cfg_obj):
    raw = cfg_obj.get("sharePolicy")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for group_id, policy in raw.items():
        gid = _to_str(group_id)
        if not gid or not isinstance(policy, dict):
            continue
        readable = _as_list(policy.get("readableAgents") or policy.get("readable_agents") or policy.get("readers"))
        writable = _as_list(policy.get("writableAgents") or policy.get("writable_agents") or policy.get("writers"))
        out[gid] = {
            "readableAgents": readable,
            "writableAgents": writable,
        }
    return out

plugins = cfg.setdefault("plugins", {})
slots = plugins.setdefault("slots", {})
entries = plugins.setdefault("entries", {})
installs = plugins.setdefault("installs", {})

slots["memory"] = "flockmem-memory"

snapshot_public = {
    "provider": "",
    "baseUrl": "",
    "model": "",
}
sync_status = "sync_unavailable"
sync_error = ""
try:
    repo_root = Path(source_path).resolve().parents[1]
    repo_src = repo_root / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    from flockmem.config.openclaw_primary_sync import (
        detect_primary_model_snapshot,
        sync_openclaw_primary_to_minimem_config,
        to_public_primary_snapshot,
    )

    primary_snapshot = detect_primary_model_snapshot(cfg)
    public_snapshot = to_public_primary_snapshot(primary_snapshot)
    snapshot_public = {
        "provider": str(public_snapshot.get("provider", "")).strip(),
        "baseUrl": str(public_snapshot.get("base_url", "")).strip(),
        "model": str(public_snapshot.get("model", "")).strip(),
    }
    sync_result = sync_openclaw_primary_to_minimem_config(
        openclaw_config_path=Path(config_path),
        minimem_config_path=Path(minimem_config_path) if minimem_config_path else None,
        inherit_primary_model=inherit_primary_model,
        force=force_primary_sync,
    )
    sync_status = str(sync_result.get("status", "")).strip() or "unknown"
except Exception as exc:  # noqa: BLE001
    sync_status = "sync_failed"
    sync_error = str(exc)

sender_map = _collect_sender_map(cfg)
channel_group_map = _collect_channel_group_map(cfg)
share_policy = _collect_share_policy(cfg)

entry = entries.setdefault("flockmem-memory", {})
entry["enabled"] = True
entry["config"] = {
    "baseUrl": base_url,
    "groupStrategy": strategy,
    "sharedGroupId": shared_group,
    "defaultAgentId": "main",
    "autoSenderFromAgent": True,
    "defaultRetrieveMethod": "agentic",
    "defaultDecisionMode": "rule",
    "autoInjectOnStart": True,
    "autoCaptureOnEnd": True,
    "autoCaptureCompression": True,
    "inheritPrimaryModel": inherit_primary_model,
    "primaryModelSnapshot": snapshot_public,
    "primaryModelSyncStatus": sync_status,
    "senderMap": sender_map,
    "channelGroupMap": channel_group_map,
    "sharePolicy": share_policy,
}

install = installs.setdefault("flockmem-memory", {})
install["source"] = "path"
install["sourcePath"] = source_path
install["installPath"] = install_path
install["version"] = "0.2.0"
install["installedAt"] = datetime.now(timezone.utc).isoformat()

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"Primary model sync status: {sync_status}")
if sync_error:
    print(f"Primary model sync warning: {sync_error[:240]}", file=sys.stderr)
PY

echo "FlockMem OpenClaw plugin installed."
echo "Config: $CONFIG_PATH"
echo "Strategy: $GROUP_STRATEGY"
if [[ "$GROUP_STRATEGY" == "shared" ]]; then
  echo "Shared group: $SHARED_GROUP_ID"
fi
if [[ "$INHERIT_PRIMARY_MODEL" == "true" ]]; then
  echo "Primary model inheritance: enabled"
else
  echo "Primary model inheritance: disabled"
fi

echo "If gateway is running, restart it: openclaw gateway restart"


