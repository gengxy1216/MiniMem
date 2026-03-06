# OpenClaw MiniMem Plugin

Lightweight memory bridge for OpenClaw.

It provides:

- auto memory injection before agent run
- auto memory capture after agent run
- manual tools for write/retrieve
- cross-session and cross-bot memory reuse

## 1-Minute Quick Start

Prerequisites:

- MiniMem running at `http://127.0.0.1:20195` (or your custom URL)
- OpenClaw already initialized (`~/.openclaw/openclaw.json` exists)

### Option A: Install from npm (recommended for distribution)

```bash
openclaw plugins install openclaw-plugin-minimem
openclaw plugins enable minimem-memory
```

Then set minimal config in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": { "memory": "minimem-memory" },
    "entries": {
      "minimem-memory": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:20195",
          "groupStrategy": "per_role"
        }
      }
    }
  }
}
```

### Option B: Install from local path

Install and enable:

```powershell
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/install.ps1
```

```bash
bash integrations/openclaw-plugin/install.sh
```

Restart gateway:

```bash
openclaw gateway restart
```

## Simple by Default, Advanced When Needed

Default installation keeps config minimal and practical:

- `groupStrategy = per_role`
- auto inject on start: on
- auto capture on end: on
- auto compression capture: on
- compression mode: `truncate` (fast/fail-open default)
- inherit OpenClaw primary model to MiniMem `config.json`: on

This means most users can start without editing config.

## OpenClaw Primary Model Sync

Install scripts now do an install-time sync:

1. detect OpenClaw primary model from `~/.openclaw/openclaw.json`
2. write snapshot to plugin config:
   - `inheritPrimaryModel`
   - `primaryModelSnapshot`
   - `primaryModelSyncStatus`
3. sync model trio into MiniMem `config.json` (chat/extractor follows the same model source)

Priority rule:

- explicit MiniMem `config.json` override > OpenClaw primary snapshot > default

Manual override protection:

- If MiniMem model fields were manually changed after previous sync, next sync is skipped (`skipped_manual_override`).
- Use force sync when you intentionally want OpenClaw primary to overwrite manual values.

Examples:

```powershell
# force sync OpenClaw primary model into MiniMem config.json
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/install.ps1 -ForcePrimarySync
```

```bash
# disable primary-model inheritance
bash integrations/openclaw-plugin/install.sh --disable-primary-sync
```

```bash
# sync to an explicit MiniMem config.json path
bash integrations/openclaw-plugin/install.sh --minimem-config /path/to/config.json
```

## Publish to GitHub Packages (npm registry)

```bash
cd integrations/openclaw-plugin
npm run pack:check
npm run publish:github
```

One-click publish:

```powershell
$env:NODE_AUTH_TOKEN="<github_pat_with_write_packages>"
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/publish-github-package.ps1
```

```bash
export NODE_AUTH_TOKEN="<github_pat_with_write_packages>"
bash integrations/openclaw-plugin/publish-github-package.sh
```

If login is required (manual):

```bash
npm login --scope=@gengxy1216 --registry=https://npm.pkg.github.com --auth-type=legacy
```

## Group Strategy (Keep Per-Role Support)

`groupStrategy` controls auto-assigned `group_id` when tool call does not pass one.

| Strategy | Behavior | Best for |
|---|---|---|
| `per_role` | `default:<sender>` | multi-bot isolation (recommended default) |
| `per_user` | `default:<user_id>` | user-centric memory |
| `shared` | `sharedGroupId` (default `shared:openclaw`) | cross-bot shared memory |

Note:

- Auto hooks (`before_agent_start`/`agent_end`) can infer role from agent runtime context.
- Manual tool calls should pass `sender`/`group_id` when you need exact routing.

Examples:

```powershell
# Shared team memory
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/install.ps1 -EnableSharedMemory -SharedGroupId "shared:team"
```

```bash
# Explicit per-role strategy
bash integrations/openclaw-plugin/install.sh --group-strategy per_role
```

## Agent Behavior: Auto + Policy File

The plugin itself does two automatic hooks:

- `before_agent_start`: retrieve and prepend memory context
- `agent_end`: write latest dialogue and optional compressed context

Tool usage can still be guided by your workspace policy file:

- Use template: `integrations/openclaw-plugin/examples/AGENTS.memory-policy.md`
- Put rules into your workspace `AGENTS.md`

This gives you:

- no-config startup for new users
- strict controllability for production bots

## Compression Governance (Task04)

`agent_end` context compression now supports explicit strategy and budget controls:

- `compressionMode`: `truncate | llm_summary | hybrid`
- `compressionTimeoutMs`: timeout budget for LLM compression path
- `compressionMinTurns`: skip compression capture when turn count is too small
- `compressionLlmBaseUrl` / `compressionLlmApiKey` / `compressionLlmModel`: optional OpenAI-compatible summary model

Behavior guarantees:

- default is `truncate` (lowest latency)
- `llm_summary` and `hybrid` both fail-open to `truncate` on timeout/error
- fallback reason and compression diagnostics are written to memory metadata and debug logs

## Agent/Channel Routing (Task03)

Plugin config now supports:

- `senderMap`: `agent_id -> sender`
- `channelGroupMap`: `channel -> group_id`
- `sharePolicy`: ACL per `group_id`
  - `readableAgents`
  - `writableAgents`

Resolution order:

1. explicit `group_id` from tool call
2. `channelGroupMap[channel]`
3. fallback from `groupStrategy`

ACL behavior:

- If `sharePolicy[group_id]` denies current `agent_id`, plugin falls back to default group (`groupStrategy`) instead of hard failing.
- This keeps runtime fail-open while reducing cross-group leakage risk.

Metadata automatically written into memory content:

- `agent_id`
- `channel`
- `task_id`
- `trace_id`
- `route_acl_fallback` (when ACL fallback happened)

Install script behavior:

- tries to auto-extract `senderMap` from OpenClaw `agents`
- tries to auto-extract `channelGroupMap` from OpenClaw `channels`
- reads optional root `sharePolicy` from OpenClaw config

You can always edit these mappings manually in `~/.openclaw/openclaw.json`.

## OpenClaw Backtest Checklist

Recommended after each major plugin change:

1. run one installation sync (`install.ps1` or `install.sh`)
2. run two agents in one shared group and verify cross-recall works
3. run two agents in isolated groups and verify no cross-recall
4. set `sharePolicy` deny rule and verify ACL fallback to default group
5. verify `before_agent_start` still injects context and `agent_end` still writes memory
6. record P50/P95 for startup injection and compare to previous baseline

## Tools

- `minimem_memory_write`
  - Write one memory item (`dialogue`/`bot_profile`/`context_compression`/`note`)
- `minimem_memory_retrieve`
  - Retrieve by strategy (`keyword`/`vector`/`hybrid`/`rrf`/`agentic`)
  - Returns `context_for_agent` for direct prompt injection

## Environment Variables (Optional)

You can configure without touching `openclaw.json`:

- `MINIMEM_BASE_URL`
- `MINIMEM_BEARER_TOKEN`
- `MINIMEM_BASIC_USER` + `MINIMEM_BASIC_PASSWORD`
- `MINIMEM_GROUP_STRATEGY`
- `MINIMEM_SHARED_GROUP_ID`
- `MINIMEM_AUTO_CAPTURE_ON_END`
- `MINIMEM_AUTO_INJECT_ON_START`
- `MINIMEM_COMPRESSION_MODE`
- `MINIMEM_COMPRESSION_TIMEOUT_MS`
- `MINIMEM_COMPRESSION_MIN_TURNS`
- `MINIMEM_COMPRESSION_LLM_BASE_URL`
- `MINIMEM_COMPRESSION_LLM_API_KEY`
- `MINIMEM_COMPRESSION_LLM_MODEL`

## UTF-8 / Windows

- Requests use `application/json; charset=utf-8`
- payload serialization is UTF-8 end-to-end
- install scripts support Windows and Linux/macOS
