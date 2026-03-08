# <img src="https://raw.githubusercontent.com/gengxy1216/FlockMem/main/docs/assets/flockmem-icon.svg" alt="FlockMem logo" width="32" /> FlockMem

> Lightweight local-first memory system for AI agents ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](pyproject.toml)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=plastic&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS42MDE1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/gengxy1216/FlockMem)

<p align="center">
  <img src="docs/assets/flockmem-banner.svg" alt="FlockMem banner" />
</p>

English | [简体中文](README.zh-CN.md)

FlockMem is a lightweight, local-first long-term memory system designed for AI agents. It is also built as a memory infrastructure layer for multi-agent collaboration. With edge deployment in mind, it runs with a minimal memory footprint under 50MB, making it suitable for resource-constrained environments 🖥️

## Why FlockMem? 💡

- 🧱 **Multi-Agent Memory Infrastructure** - One shared memory substrate across agents, runtimes, and workflows
- 🔄 **Local-first** - No cloud dependencies, runs entirely on your device
- 🔍 **Hybrid Retrieval** - Text + Vector + Graph search with citation traces
- 🧠 **Structured + Episodic Memory** - Entities, relationships, and conversation memories in one system
- ⚡ **Lightweight** - Under 50MB memory usage, suitable for edge devices
- 🔌 **Integration-ready** - REST API + MCP bridge + OpenClaw plugin
- 🚀 **One-click Install** - Get started in seconds

## Key Positioning: Memory Infrastructure for Multi-Agent Collaboration

FlockMem is not just a memory store for a single assistant. It provides:

- Shared memory infrastructure for multiple agents via REST, MCP, and plugin bridges
- Controlled collaboration boundaries with role-based, shared-group, and per-user memory strategies
- Retrieval traces and citations for auditability and easier debugging
- Local-first deployment with predictable cost, data ownership, and operational simplicity

## Quick Start ⚡

### One-click Install

`pip`:

```bash
pip install flockmem
flockmem
```

`npm`:

```bash
npm install openclaw-flockmem
```

`openclaw plugin`:

```bash
openclaw plugins install openclaw-flockmem
openclaw plugins enable flockmem-memory
```

Local install (kept):

| Platform | Command |
|----------|---------|
| 🪟 **Windows** | `powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -RunAfterInstall` |
| 🐧 **Linux** | `bash scripts/install.sh --run` |
| 🍎 **macOS** | `bash scripts/install.sh --run` |

Or use the launcher:

```bash
# Windows
start_flockmem.bat

# Linux/macOS
bash scripts/start.sh
```

### Manual Install

```bash
pip install -e .
flockmem
```

### Access the UI 🌐

Open your browser:

```
http://127.0.0.1:20195/ui
```

> 🔐 **Default Credentials**: `admin` / `admin123`

## Features 🎯

| Feature | Description |
|---------|-------------|
| 📝 **Memory Storage** | Store and manage conversation memories |
| 🔎 **Semantic Search** | Find relevant memories using vector similarity |
| 🕸️ **Graph Search** | Explore entity relationships in knowledge graph |
| 💬 **Chat with Memory** | Context-aware conversations with retrieval traces |
| ⚙️ **Runtime Config** | Change providers and settings on-the-fly |

## API Overview 📡

> For complete API documentation, see [API Reference](docs/api-reference.md)

```bash
# Health check
GET /health

# Store memory
POST /api/v1/memories

# Search memories
GET /api/v1/memories/search

# Chat with context
POST /api/v1/chat/simple

# Graph queries
GET /api/v1/graph/search
GET /api/v1/graph/neighbors
```

## MCP Integration 🔌

FlockMem now includes a standalone MCP bridge server for cross-agent integration:

- Server path: `integrations/flockmem-mcp/server.py`
- Skill path: `skills/flockmem-mcp-integration`

Run the MCP bridge:

```bash
python skills/flockmem-mcp-integration/scripts/run_flockmem_mcp.py
```

Generate a client config snippet:

```bash
python skills/flockmem-mcp-integration/scripts/generate_mcp_config.py
```

Run MCP integration test:

```bash
pytest -q tests/test_flockmem_mcp_server.py
```

Package the MCP skill as a distributable artifact:

```bash
python C:/Users/user/.agents/skills/skill-creator/scripts/package_skill.py skills/flockmem-mcp-integration dist
```

Packaged artifact:

- `dist/flockmem-mcp-integration.skill`

Environment variables (optional):

- `MINIMEM_BASE_URL` (default: `http://127.0.0.1:20195`)
- `MINIMEM_USER_ID`
- `MINIMEM_GROUP_ID`
- `MINIMEM_BEARER_TOKEN` or `MINIMEM_BASIC_USER` + `MINIMEM_BASIC_PASSWORD`

## OpenClaw Plugin Integration

FlockMem also provides a lightweight OpenClaw plugin bridge:

- Plugin path: `integrations/openclaw-plugin`
- Plugin manifest: `integrations/openclaw-plugin/openclaw.plugin.json`
- Plugin doc: `integrations/openclaw-plugin/README.md`
- Policy template: `integrations/openclaw-plugin/examples/AGENTS.memory-policy.md`

Core capabilities:

- write dialogue / bot profile / context compression into FlockMem
- retrieve memory by strategy (`keyword`/`vector`/`hybrid`/`rrf`/`agentic`)
- return `context_for_agent` for prompt injection
- optional auto capture (`agent_end`) and auto inject (`before_agent_start`)
- role-based / shared / per-user group strategies

One-command install:

```powershell
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/install.ps1
```

```bash
bash integrations/openclaw-plugin/install.sh
```

Or install distributed plugin package:

```bash
openclaw plugins install openclaw-flockmem
openclaw plugins enable flockmem-memory
```

Advanced entry example:

```json
{
  "path": "C:/path/to/flockmem/integrations/openclaw-plugin",
  "enabled": true,
  "config": {
    "baseUrl": "http://127.0.0.1:20195",
    "groupStrategy": "per_role",
    "sharedGroupId": "shared:team",
    "autoSenderFromAgent": true,
    "autoInjectOnStart": true,
    "autoCaptureOnEnd": true
  }
}
```

## Architecture 🏗️

```
┌─────────────────────────────────────┐
│            FlockMem UI               │
├─────────────────────────────────────┤
│            REST API                 │
├──────────────┬──────────────────────┤
│   Retrieval  │    Extraction       │
│   (Fusion)   │    (Atomic Facts)   │
├──────────────┼──────────────────────┤
│   Vector DB  │   Graph Store        │
│  (LanceDB)   │   (Local Persistent) │
├──────────────┴──────────────────────┤
│          SQLite (Metadata)          │
└─────────────────────────────────────┘
```

### Memory Tiering (L0 -> L2)

FlockMem uses progressive memory tiers to balance simplicity, recall quality, and retrieval performance:

| Tier | Name | Core Storage | Retrieval Role |
|------|------|--------------|----------------|
| `L0` | Plain Text | SQLite (`episodic_memory`) | Full-fidelity source of truth, keyword/metadata recall |
| `L0.5` | Vector-Text (Hot) | Local vector cache (`jsonl` + snapshot) | Fast local semantic recall and restart-safe warm cache |
| `L1` | Vector (Persistent) | LanceDB (`memory_vector_index`) | ANN-style semantic retrieval for higher-importance memories |
| `L2` | Graph | Local graph triples store | Structured relation recall and graph expansion |

### Retrieval Flow

1. Query enters policy/selector layer.
2. Keyword + vector recall run in hybrid mode (`L0` + `L0.5` + `L1`).
3. Graph evidence from `L2` is merged when enabled.
4. Final hits are reranked and rendered with query-aware citation snippets.

## Configuration ⚙️

Set environment variables to configure providers:

```bash
# Chat provider
LITE_CHAT_PROVIDER=openai
LITE_CHAT_MODEL=gpt-4o-mini

# Embedding provider
LITE_EMBEDDING_PROVIDER=openai
LITE_EMBEDDING_MODEL=text-embedding-3-small

# Graph module (optional)
LITE_GRAPH_ENABLED=true

# Retrieval profile
LITE_RETRIEVAL_PROFILE=agentic

  # Hybrid vector persistence (partial to LanceDB)
  LITE_VECTOR_LANCEDB_ENABLED=true
  LITE_VECTOR_LANCEDB_MIN_IMPORTANCE=0.72

  # Admin protection for sensitive config APIs
  LITE_ADMIN_TOKEN=change-me
  LITE_ADMIN_ALLOW_LOCALHOST=true
  
  # Retrieval latency/recall tuning (edge-side)
  LITE_SEARCH_BUDGET_FACTOR=4
LITE_SEARCH_MIN_PROBE_K=12
LITE_KEYWORD_CONFIDENT_BEST_SCORE=9.0
LITE_KEYWORD_CONFIDENT_KTH_SCORE=2.8
LITE_SEMANTIC_VECTOR_BUDGET_CAP=32
LITE_SEMANTIC_KEYWORD_BUDGET_CAP=16
  LITE_QUERY_EMBED_CACHE_SIZE=256
  LITE_QUERY_EMBED_CACHE_TTL_SEC=900
  ```

- Sensitive endpoints (`/api/v1/config/raw`, `/api/v1/model-config*`) accept
  `Authorization: Bearer <LITE_ADMIN_TOKEN>` or `X-API-Key`.
- When `LITE_ADMIN_TOKEN` is empty, localhost access is allowed by default
  (`LITE_ADMIN_ALLOW_LOCALHOST=true`).

Default local storage location (when `LITE_DATA_DIR` is not set):

- Windows: `%LOCALAPPDATA%\\MiniMem` (kept for backward compatibility)
- Linux/macOS: `$XDG_DATA_HOME/minimem` or `~/.local/share/minimem`

## Tech Stack 🛠️

- **FastAPI** - Modern async web framework
- **SQLite** - Local structured data
- **LanceDB** - High-performance vector database
- **Local Graph Store** - Persistent triple storage and graph retrieval

## Acknowledgments 🙏

FlockMem builds on the shoulders of giants ❤️

- **[EverMemOs](https://github.com/EverMind-AI/EverMemOS**)** - Original inspiration for agent memory systems
- **[LanceDB](https://lancedb.com/)** - Developer-friendly vector database
- **[SQLite](https://www.sqlite.org/)** - The most used database in the world

## License 📄

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ for AI agents everywhere 🤖



