# MiniMem 🧠

> Lightweight local-first memory system for AI agents ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](pyproject.toml)

English | [简体中文](README.zh-CN.md)

MiniMem is a lightweight, local-first long-term memory system designed for AI agents. Built with edge deployment in mind, it runs with a minimal memory footprint under 50MB, making it perfect for resource-constrained environments 🖥️

## Why MiniMem? 💡

- 🔄 **Local-first** - No cloud dependencies, runs entirely on your device
- 🧠 **Knowledge Graph** - Rich graph-based memory with entities and relationships
- ⚡ **Lightweight** - Under 50MB memory usage, perfect for edge devices
- 🚀 **One-click Install** - Get started in seconds
- 🔌 **Clean API** - Easy integration with any agent framework
- 🔍 **Hybrid Retrieval** - Text + Vector + Graph search with citations

## Quick Start ⚡

### One-click Install

| Platform | Command |
|----------|---------|
| 🪟 **Windows** | `powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -RunAfterInstall` |
| 🐧 **Linux** | `bash scripts/install.sh --run` |
| 🍎 **macOS** | `bash scripts/install.sh --run` |

Or use the launcher:

```bash
# Windows
start_minimem.bat

# Linux/macOS
bash scripts/start.sh
```

### Manual Install

```bash
pip install -e .
minimem
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

MiniMem now includes a standalone MCP bridge server for cross-agent integration:

- Server path: `integrations/minimem-mcp/server.py`
- Skill path: `skills/minimem-mcp-integration`

Run the MCP bridge:

```bash
python skills/minimem-mcp-integration/scripts/run_minimem_mcp.py
```

Generate a client config snippet:

```bash
python skills/minimem-mcp-integration/scripts/generate_mcp_config.py
```

Run MCP integration test:

```bash
pytest -q tests/test_minimem_mcp_server.py
```

Package the MCP skill as a distributable artifact:

```bash
python C:/Users/user/.agents/skills/skill-creator/scripts/package_skill.py skills/minimem-mcp-integration dist
```

Packaged artifact:

- `dist/minimem-mcp-integration.skill`

Environment variables (optional):

- `MINIMEM_BASE_URL` (default: `http://127.0.0.1:20195`)
- `MINIMEM_USER_ID`
- `MINIMEM_GROUP_ID`
- `MINIMEM_BEARER_TOKEN` or `MINIMEM_BASIC_USER` + `MINIMEM_BASIC_PASSWORD`

## OpenClaw Plugin Integration

MiniMem also provides a lightweight OpenClaw plugin bridge:

- Plugin path: `integrations/openclaw-plugin`
- Plugin manifest: `integrations/openclaw-plugin/openclaw.plugin.json`
- Plugin doc: `integrations/openclaw-plugin/README.md`
- Policy template: `integrations/openclaw-plugin/examples/AGENTS.memory-policy.md`

Core capabilities:

- write dialogue / bot profile / context compression into MiniMem
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

Advanced entry example:

```json
{
  "path": "C:/MiniMem-main/MiniMem/integrations/openclaw-plugin",
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
│            MiniMem UI               │
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

MiniMem uses progressive memory tiers to balance simplicity, recall quality, and retrieval performance:

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

Default local storage location (when `LITE_DATA_DIR` is not set):

- Windows: `%LOCALAPPDATA%\\MiniMem`
- Linux/macOS: `$XDG_DATA_HOME/minimem` or `~/.local/share/minimem`

## Tech Stack 🛠️

- **FastAPI** - Modern async web framework
- **SQLite** - Local structured data
- **LanceDB** - High-performance vector database
- **Local Graph Store** - Persistent triple storage and graph retrieval

## Acknowledgments 🙏

MiniMem builds on the shoulders of giants ❤️

- **[EverMemOs](https://github.com/EverMind-AI/EverMemOS**)** - Original inspiration for agent memory systems
- **[LanceDB](https://lancedb.com/)** - Developer-friendly vector database
- **[SQLite](https://www.sqlite.org/)** - The most used database in the world

## License 📄

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ for AI agents everywhere 🤖
