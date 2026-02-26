---
name: minimem-mcp-integration
description: Integrate MiniMem as an MCP server for cross-agent memory operations. Use this when an agent needs to call MiniMem memory APIs through MCP (search memories, write memory, memory-cited chat, graph search/neighbors), wire MCP client config, or run a reusable local MCP bridge process.
---

# Minimem Mcp Integration

## Overview

Use this skill to connect any MCP-compatible agent to MiniMem through a standalone bridge server.
Prefer this skill when users ask to "connect MiniMem with MCP", "expose MiniMem tools to another agent", or "share MiniMem memory retrieval across agents".

## Quick Start

1. Ensure MiniMem HTTP service is running at `MINIMEM_BASE_URL` (default `http://127.0.0.1:20195`).
2. Start MCP bridge with:
```bash
python skills/minimem-mcp-integration/scripts/run_minimem_mcp.py
```
3. Add MCP config to your agent client. Use:
```bash
python skills/minimem-mcp-integration/scripts/generate_mcp_config.py
```
4. Validate by calling MCP tool `minimem_health`.

## Available MCP Tools

- `minimem_health`: health check
- `search_memories`: memory retrieval (`keyword|vector|hybrid|rrf|agentic`)
- `write_memory`: write one memory item
- `chat_with_memory`: memory-cited chat answer
- `graph_search`: graph triple retrieval
- `graph_neighbors`: graph neighborhood retrieval

## Environment Variables

- `MINIMEM_BASE_URL`: MiniMem base URL
- `MINIMEM_TIMEOUT_SEC`: HTTP timeout
- `MINIMEM_USER_ID`: default `user_id` for tools
- `MINIMEM_GROUP_ID`: default `group_id` for tools
- `MINIMEM_BEARER_TOKEN`: optional bearer auth
- `MINIMEM_BASIC_USER` + `MINIMEM_BASIC_PASSWORD`: optional basic auth

## References

- MCP config examples and client wiring:
  [references/mcp-config.md](references/mcp-config.md)

## Scripts

- Start MCP bridge:
  [scripts/run_minimem_mcp.py](scripts/run_minimem_mcp.py)
- Print MCP client config snippet:
  [scripts/generate_mcp_config.py](scripts/generate_mcp_config.py)

## Integration Notes

- Keep MiniMem and MCP bridge in the same machine for lowest latency.
- If MiniMem requires auth in your deployment, set bearer or basic env vars before running the bridge.
- For production, run the bridge under a process supervisor (systemd / PM2 / task scheduler).
