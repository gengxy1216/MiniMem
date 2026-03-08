#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:20195"
TIMEOUT_SEC=60
AUTO_START=1
ALLOW_MISSING_COLLECTIVE=0
STRICT_OPENCLAW=0

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
STARTED_SERVER_PID=""

usage() {
  cat <<'USAGE'
Usage: scripts/smoke_collective_release.sh [options]

Options:
  --base-url <url>               API base URL (default: http://127.0.0.1:20195)
  --timeout-sec <n>              Startup wait timeout seconds (default: 60)
  --no-auto-start                Do not auto-start local server when unreachable
  --allow-missing-collective     Treat missing collective endpoints (404) as warning
  --strict-openclaw              Fail when openclaw CLI is unavailable
  -h, --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --no-auto-start)
      AUTO_START=0
      shift
      ;;
    --allow-missing-collective)
      ALLOW_MISSING_COLLECTIVE=1
      shift
      ;;
    --strict-openclaw)
      STRICT_OPENCLAW=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/tmp_test_runtime"
SERVER_LOG="$LOG_DIR/collective-smoke-server.log"
mkdir -p "$LOG_DIR"

add_result() {
  local level="$1"
  local name="$2"
  local detail="$3"
  case "$level" in
    PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    WARN) WARN_COUNT=$((WARN_COUNT + 1)) ;;
  esac
  printf '[%s] %s - %s\n' "$level" "$name" "$detail"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

http_status() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local url="${BASE_URL%/}${path}"
  if [[ "$method" == "GET" ]]; then
    curl -sS -o /dev/null -w "%{http_code}" "$url" || echo "000"
    return
  fi
  curl -sS -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X "$method" \
    --data "$body" \
    "$url" || echo "000"
}

is_health_ok() {
  local code
  code="$(http_status GET "/health")"
  [[ "$code" == "200" ]]
}

cleanup() {
  if [[ -n "$STARTED_SERVER_PID" ]]; then
    if kill -0 "$STARTED_SERVER_PID" >/dev/null 2>&1; then
      kill "$STARTED_SERVER_PID" >/dev/null 2>&1 || true
      wait "$STARTED_SERVER_PID" 2>/dev/null || true
      add_result "PASS" "server-stop" "stopped local server pid=$STARTED_SERVER_PID"
    fi
  fi
}
trap cleanup EXIT

start_server_if_needed() {
  if is_health_ok; then
    add_result "PASS" "health-precheck" "service already reachable at ${BASE_URL}/health"
    return
  fi
  if [[ "$AUTO_START" -ne 1 ]]; then
    add_result "FAIL" "health-precheck" "service unreachable and --no-auto-start enabled"
    return
  fi
  if ! has_cmd python; then
    add_result "FAIL" "server-start" "python command not found"
    return
  fi

  local host port
  host="$(python - <<'PY' "$BASE_URL"
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
print(u.hostname or "127.0.0.1")
PY
)"
  port="$(python - <<'PY' "$BASE_URL"
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
print(u.port or 80)
PY
)"

  (
    cd "$REPO_ROOT"
    LITE_HOST="$host" LITE_PORT="$port" python main.py >"$SERVER_LOG" 2>&1
  ) &
  STARTED_SERVER_PID="$!"
  add_result "PASS" "server-start" "spawned local server pid=$STARTED_SERVER_PID (log: $SERVER_LOG)"

  local i
  for ((i=1; i<=TIMEOUT_SEC; i++)); do
    if is_health_ok; then
      add_result "PASS" "health-check" "service became healthy in ${i}s"
      return
    fi
    sleep 1
  done
  add_result "FAIL" "health-check" "service not healthy after ${TIMEOUT_SEC}s; see $SERVER_LOG"
}

check_collective_endpoint() {
  local name="$1"
  local path="$2"
  local body="$3"
  shift 3
  local expected=("$@")
  local code
  code="$(http_status POST "$path" "$body")"

  if [[ "$code" == "404" ]]; then
    if [[ "$ALLOW_MISSING_COLLECTIVE" -eq 1 ]]; then
      add_result "WARN" "$name" "$path -> 404 (allowed)"
    else
      add_result "FAIL" "$name" "$path -> 404 (endpoint missing)"
    fi
    return
  fi
  if [[ "$code" == "000" ]]; then
    add_result "FAIL" "$name" "$path -> request failed"
    return
  fi
  if [[ "$code" =~ ^5 ]]; then
    add_result "FAIL" "$name" "$path -> $code (server error)"
    return
  fi
  local ok=0
  for status in "${expected[@]}"; do
    if [[ "$code" == "$status" ]]; then
      ok=1
      break
    fi
  done
  if [[ "$ok" -eq 1 ]]; then
    add_result "PASS" "$name" "$path -> $code"
  else
    add_result "FAIL" "$name" "$path -> $code (expected: ${expected[*]})"
  fi
}

check_mcp_and_plugin() {
  if has_cmd python; then
    if (cd "$REPO_ROOT" && python -m py_compile integrations/flockmem-mcp/server.py); then
      add_result "PASS" "mcp-syntax" "integrations/flockmem-mcp/server.py compiled"
    else
      add_result "FAIL" "mcp-syntax" "python compile failed"
    fi

    if (cd "$REPO_ROOT" && python - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("integrations/flockmem-mcp/server.py")
spec = importlib.util.spec_from_file_location("flockmem_mcp_server", path)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to build module spec")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(getattr(module, "MCP_NAME", "unknown"))
PY
    ); then
      add_result "PASS" "mcp-import" "MCP module import succeeded"
    else
      add_result "FAIL" "mcp-import" "MCP module import failed (check fastmcp dependency)"
    fi
  else
    add_result "FAIL" "mcp-check" "python command not found"
  fi

  if [[ -f "$REPO_ROOT/integrations/openclaw-plugin/install.sh" && -f "$REPO_ROOT/integrations/openclaw-plugin/install.ps1" ]]; then
    add_result "PASS" "plugin-install-scripts" "install scripts found"
  else
    add_result "FAIL" "plugin-install-scripts" "install.sh/install.ps1 missing"
  fi

  if has_cmd npm; then
    if (cd "$REPO_ROOT/integrations/openclaw-plugin" && npm pack --dry-run >/dev/null); then
      add_result "PASS" "plugin-pack-dry-run" "npm pack --dry-run succeeded"
    else
      add_result "FAIL" "plugin-pack-dry-run" "npm pack --dry-run failed"
    fi
  else
    add_result "FAIL" "plugin-pack-dry-run" "npm command not found"
  fi

  if has_cmd openclaw; then
    add_result "PASS" "openclaw-cli" "openclaw command found"
  else
    if [[ "$STRICT_OPENCLAW" -eq 1 ]]; then
      add_result "FAIL" "openclaw-cli" "openclaw command not found (strict mode)"
    else
      add_result "WARN" "openclaw-cli" "openclaw command not found"
    fi
  fi
}

echo "== FlockMem Collective Release Smoke =="
echo "base_url=${BASE_URL} timeout_sec=${TIMEOUT_SEC} auto_start=${AUTO_START} allow_missing_collective=${ALLOW_MISSING_COLLECTIVE} strict_openclaw=${STRICT_OPENCLAW}"

if ! has_cmd curl; then
  add_result "FAIL" "prerequisite" "curl command not found"
fi

start_server_if_needed

if is_health_ok; then
  add_result "PASS" "health" "/health returned 200"
else
  add_result "FAIL" "health" "/health not reachable"
fi

INGEST_PAYLOAD='{"knowledge_id":"smoke-k-1","scope_type":"personal","scope_id":"smoke-user","content":{"text":"collective smoke ingest"},"changed_by":"agent","actor_id":"smoke-agent","write_acl":["smoke-agent"],"coordination_mode":"inruntime_a2a","coordination_id":"smoke-coord-1","runtime_id":"codex","agent_id":"smoke-agent","session_id":"smoke-session-1"}'
CONTEXT_PAYLOAD='{"query":"collective smoke context","actor_id":"smoke-agent","personal_scope_id":"smoke-user","include_global":false,"top_k":5,"coordination_mode":"inruntime_a2a","coordination_id":"smoke-coord-2","runtime_id":"codex","agent_id":"smoke-agent","session_id":"smoke-session-2"}'
FEEDBACK_PAYLOAD='{"knowledge_id":"smoke-k-1","feedback_type":"execution_signal","feedback_payload":{"outcome_status":"success"},"actor":"smoke-agent","coordination_mode":"inruntime_a2a","coordination_id":"smoke-coord-3","runtime_id":"codex","agent_id":"smoke-agent","session_id":"smoke-session-3"}'

check_collective_endpoint "collective-ingest" "/api/v1/collective/ingest" "$INGEST_PAYLOAD" "200"
check_collective_endpoint "collective-context" "/api/v1/collective/context" "$CONTEXT_PAYLOAD" "200"
check_collective_endpoint "collective-feedback" "/api/v1/collective/feedback" "$FEEDBACK_PAYLOAD" "200"

check_mcp_and_plugin

echo "== Summary =="
echo "PASS=${PASS_COUNT} WARN=${WARN_COUNT} FAIL=${FAIL_COUNT}"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
exit 0
