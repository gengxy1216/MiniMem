from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import importlib.util

import uvicorn

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings


def main() -> int:
    repo_root = Path.cwd()
    tmp_data = repo_root / ".tmp-qa-gate-data"
    tmp_config = repo_root / ".tmp-qa-gate-config"
    tmp_data.mkdir(parents=True, exist_ok=True)
    tmp_config.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "LITE_DATA_DIR": str(tmp_data),
            "LITE_CONFIG_DIR": str(tmp_config),
            "LITE_ADMIN_TOKEN": "qa-gate-admin-token",
            "LITE_ADMIN_ALLOW_LOCALHOST": "false",
            "LITE_RETRIEVAL_PROFILE": "keyword",
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "qa-chat-key",
            "LITE_CHAT_MODEL": "qa-chat-model",
            "LITE_EMBEDDING_PROVIDER": "openai",
            "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
            "LITE_EMBEDDING_API_KEY": "qa-embed-key",
            "LITE_EMBEDDING_MODEL": "qa-embed-model",
            "LITE_EXTRACTOR_PROVIDER": "rule",
            "LITE_RERANK_PROVIDER": "chat_model",
            "LITE_HOST": "127.0.0.1",
            "LITE_PORT": "20195",
        }
    )

    bench_path = repo_root / "tools" / "benchmark_collective_baseline.py"
    bench_spec = importlib.util.spec_from_file_location("benchmark_collective_baseline", bench_path)
    if bench_spec is None or bench_spec.loader is None:
        raise RuntimeError("Unable to import tools/benchmark_collective_baseline.py")
    bench_module = importlib.util.module_from_spec(bench_spec)
    bench_spec.loader.exec_module(bench_module)

    settings = LiteSettings.from_env()
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=20195,
            log_level="error",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    ready = False
    for _ in range(80):
        time.sleep(0.25)
        try:
            with urllib.request.urlopen("http://127.0.0.1:20195/health", timeout=1.0) as resp:
                if 200 <= int(resp.status) < 500:
                    ready = True
                    break
        except Exception:
            continue
    if not ready:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Timed out waiting for local QA gate server to become ready.")

    sys.argv = [
        "benchmark_collective_baseline.py",
        "--base-url",
        "http://127.0.0.1:20195",
        "--loops",
        "20",
        "--timeout-sec",
        "3",
        "--output",
        "docs/reports/collective-baseline-2026-03-08.json",
        "--fail-on-server-error",
    ]
    try:
        return int(bench_module.main())
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
