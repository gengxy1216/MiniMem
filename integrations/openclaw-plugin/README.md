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

This means most users can start without editing config.

## Publish to npm

```bash
cd integrations/openclaw-plugin
npm run pack:check
npm publish
```

If login is required:

```bash
npm adduser
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

## UTF-8 / Windows

- Requests use `application/json; charset=utf-8`
- payload serialization is UTF-8 end-to-end
- install scripts support Windows and Linux/macOS
