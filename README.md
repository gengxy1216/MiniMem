# MiniMem 🧠

> Lightweight local-first memory system for AI agents ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](pyproject.toml)

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
│   Vector DB  │   Knowledge Graph    │
│  (LanceDB)   │     (Kuzu)           │
├──────────────┴──────────────────────┤
│          SQLite (Metadata)          │
└─────────────────────────────────────┘
```

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
LITE_RETRIEVAL_PROFILE=balanced
```

## Tech Stack 🛠️

- **FastAPI** - Modern async web framework
- **SQLite** - Local structured data
- **LanceDB** - High-performance vector database
- **Kuzu** - Fast embedded graph database

## Acknowledgments 🙏

MiniMem builds on the shoulders of giants ❤️

- **[EverMemOs](https://github.com/Any机器人/**)** - Original inspiration for agent memory systems
- **[Kuzu](https://kuzudb.com/)** - High-performance embedded graph database
- **[LanceDB](https://lancedb.com/)** - Developer-friendly vector database
- **[SQLite](https://www.sqlite.org/)** - The most used database in the world

## License 📄

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ for AI agents everywhere 🤖