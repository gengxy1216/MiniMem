from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCP config snippet for MiniMem bridge")
    parser.add_argument(
        "--repo-root",
        default=str(_default_repo_root()),
        help="MiniMem repository root path",
    )
    parser.add_argument(
        "--server-name",
        default="minimem",
        help="MCP server name key",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable path for launching the bridge",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    launcher = repo_root / "skills" / "minimem-mcp-integration" / "scripts" / "run_minimem_mcp.py"
    payload = {
        "mcpServers": {
            args.server_name: {
                "command": str(args.python_bin),
                "args": [str(launcher)],
                "env": {
                    "MINIMEM_BASE_URL": "http://127.0.0.1:20195",
                    "MINIMEM_USER_ID": "admin",
                    "MINIMEM_GROUP_ID": "default:admin",
                },
            }
        }
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
