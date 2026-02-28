# Memory Policy Template (OpenClaw + MiniMem)

Use these rules in your workspace `AGENTS.md` to keep memory behavior predictable.

## Memory Read Rule

Before answering user questions that depend on history, call `minimem_memory_retrieve` once:

- `query`: current user intent in one sentence
- `top_k`: 6
- `group_id`: use your bot's memory group

If retrieval returns `count > 0`, use `context_for_agent` as primary long-term context.

## Memory Write Rule

At the end of each completed task, call `minimem_memory_write` once:

- `memory_type`: `note`
- `content`: one concise summary of new facts
- `sender`: current bot id
- `group_id`: current bot group

## Group Strategy

- Per-role isolation: `group_id = default:<bot_id>`
- Shared team memory: `group_id = shared:team`

Prefer per-role by default. Write to shared memory only for facts that other bots should reuse.

