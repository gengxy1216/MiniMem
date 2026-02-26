from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_server(repo_root: Path) -> Path:
    server = repo_root / "integrations" / "minimem-mcp" / "server.py"
    if not server.exists():
        raise FileNotFoundError(f"MiniMem MCP server not found: {server}")
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMem MCP bridge server")
    parser.add_argument(
        "--repo-root",
        default=str(_default_repo_root()),
        help="MiniMem repository root path",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    server_path = _find_server(repo_root)
    os.execv(sys.executable, [sys.executable, str(server_path)])


if __name__ == "__main__":
    main()
