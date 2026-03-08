from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
src_path = str(SRC_DIR)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@dataclass(frozen=True)
class BenchCase:
    name: str
    path: str
    payload: dict[str, Any]


DEFAULT_CASES: tuple[BenchCase, ...] = (
    BenchCase(
        name="ingest",
        path="/api/v1/collective/ingest",
        payload={
            "knowledge_id": "bench-k-1",
            "scope_type": "personal",
            "scope_id": "u-bench",
            "content": {"text": "collective bench ingest"},
            "changed_by": "agent",
            "actor_id": "bench-agent",
            "write_acl": ["bench-agent"],
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "bench-coord-1",
            "runtime_id": "codex",
            "agent_id": "bench-agent",
            "session_id": "bench-session-1",
        },
    ),
    BenchCase(
        name="context",
        path="/api/v1/collective/context",
        payload={
            "query": "collective bench context",
            "personal_scope_id": "u-bench",
            "include_global": False,
            "actor_id": "bench-agent",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "bench-coord-2",
            "runtime_id": "codex",
            "agent_id": "bench-agent",
            "session_id": "bench-session-2",
        },
    ),
    BenchCase(
        name="feedback",
        path="/api/v1/collective/feedback",
        payload={
            "knowledge_id": "bench-k-1",
            "feedback_type": "execution_signal",
            "feedback_payload": {
                "outcome_status": "success",
                "tool_error_count": 0,
                "retry_count": 0,
                "rollback_flag": False,
                "reuse_hit": True,
            },
            "actor": "bench-agent",
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "bench-coord-3",
            "runtime_id": "codex",
            "agent_id": "bench-agent",
            "session_id": "bench-session-3",
        },
    ),
)


def _percentile_ms(values_ms: list[float], percentile: float) -> float:
    if not values_ms:
        return 0.0
    ordered = sorted(values_ms)
    pos = int(round((len(ordered) - 1) * percentile))
    return round(float(ordered[pos]), 3)


def _current_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)

    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def _post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        payload_text = err.read().decode("utf-8", errors="replace")
        return int(err.code), payload_text


def _run_case_http(
    *,
    base_url: str,
    loops: int,
    timeout_sec: float,
    case: BenchCase,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    status_hist: dict[str, int] = {}
    exceptions = 0
    for _ in range(loops):
        start = time.perf_counter()
        try:
            status, _ = _post_json(
                f"{base_url}{case.path}",
                payload=case.payload,
                timeout_sec=timeout_sec,
            )
            status_key = str(status)
            status_hist[status_key] = status_hist.get(status_key, 0) + 1
        except Exception:
            exceptions += 1
            status_hist["exception"] = status_hist.get("exception", 0) + 1
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed)

    avg_ms = round(float(statistics.mean(latencies_ms)), 3) if latencies_ms else 0.0
    return {
        "name": case.name,
        "path": case.path,
        "loops": loops,
        "avg_ms": avg_ms,
        "p50_ms": _percentile_ms(latencies_ms, 0.50),
        "p95_ms": _percentile_ms(latencies_ms, 0.95),
        "max_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        "status_histogram": status_hist,
        "exceptions": exceptions,
        "server_error_count": sum(
            count
            for code, count in status_hist.items()
            if code.isdigit() and int(code) >= 500
        ),
    }


def _run_case_in_process(
    *,
    client: Any,
    loops: int,
    case: BenchCase,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    status_hist: dict[str, int] = {}
    exceptions = 0
    for _ in range(loops):
        start = time.perf_counter()
        try:
            resp = client.post(case.path, json=case.payload)
            status_key = str(int(resp.status_code))
            status_hist[status_key] = status_hist.get(status_key, 0) + 1
        except Exception:
            exceptions += 1
            status_hist["exception"] = status_hist.get("exception", 0) + 1
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed)

    avg_ms = round(float(statistics.mean(latencies_ms)), 3) if latencies_ms else 0.0
    return {
        "name": case.name,
        "path": case.path,
        "loops": loops,
        "avg_ms": avg_ms,
        "p50_ms": _percentile_ms(latencies_ms, 0.50),
        "p95_ms": _percentile_ms(latencies_ms, 0.95),
        "max_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        "status_histogram": status_hist,
        "exceptions": exceptions,
        "server_error_count": sum(
            count
            for code, count in status_hist.items()
            if code.isdigit() and int(code) >= 500
        ),
    }


def _build_in_process_client() -> tuple[Any, tempfile.TemporaryDirectory[str], dict[str, str | None]]:
    from fastapi.testclient import TestClient
    from flockmem.bootstrap.app_factory import create_app
    from flockmem.config.settings import LiteSettings

    tmp = tempfile.TemporaryDirectory(
        prefix="collective-bench-",
        ignore_cleanup_errors=True,
    )
    data_dir = Path(tmp.name) / "data"
    config_dir = Path(tmp.name) / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "LITE_DATA_DIR": str(data_dir),
        "LITE_CONFIG_DIR": str(config_dir),
        "LITE_ADMIN_TOKEN": "bench-admin-token",
        "LITE_ADMIN_ALLOW_LOCALHOST": "false",
        "LITE_RETRIEVAL_PROFILE": "keyword",
        "LITE_CHAT_PROVIDER": "openai",
        "LITE_CHAT_BASE_URL": "https://chat.example/v1",
        "LITE_CHAT_API_KEY": "bench-chat-key",
        "LITE_CHAT_MODEL": "bench-chat-model",
        "LITE_EMBEDDING_PROVIDER": "openai",
        "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
        "LITE_EMBEDDING_API_KEY": "bench-embed-key",
        "LITE_EMBEDDING_MODEL": "bench-embed-model",
        "LITE_EXTRACTOR_PROVIDER": "rule",
        "LITE_RERANK_PROVIDER": "chat_model",
    }
    previous = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    settings = LiteSettings.from_env()
    app = create_app(settings)
    return TestClient(app), tmp, previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collective v1 endpoint baseline benchmark (latency + rss)."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Service base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--loops", type=int, default=20, help="Requests per endpoint.")
    parser.add_argument("--timeout-sec", type=float, default=3.0, help="Request timeout.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--fail-on-server-error",
        action="store_true",
        help="Exit non-zero when any endpoint returns HTTP >= 500.",
    )
    parser.add_argument(
        "--allow-exceptions",
        action="store_true",
        help="Do not fail when request exceptions are observed.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run benchmark in-process via FastAPI TestClient (no external server needed).",
    )
    args = parser.parse_args()

    base_url = str(args.base_url).rstrip("/")
    loops = max(1, int(args.loops))
    timeout_sec = max(0.1, float(args.timeout_sec))

    started_at = datetime.now(tz=timezone.utc).isoformat()
    mode = "in-process-testclient" if args.in_process else "http"
    if args.in_process:
        client, tmp, previous = _build_in_process_client()
        try:
            results = [
                _run_case_in_process(client=client, loops=loops, case=case)
                for case in DEFAULT_CASES
            ]
        finally:
            client.close()
            try:
                tmp.cleanup()
            except Exception:
                pass
            _restore_env(previous)
    else:
        results = [
            _run_case_http(base_url=base_url, loops=loops, timeout_sec=timeout_sec, case=case)
            for case in DEFAULT_CASES
        ]
    summary = {
        "started_at_utc": started_at,
        "mode": mode,
        "base_url": base_url,
        "loops_per_endpoint": loops,
        "rss_mb_current_process": _current_rss_mb(),
        "results": results,
        "notes": [
            "This script benchmarks client-observed endpoint latency.",
            "RSS metric is for benchmark process. Use external monitor for server RSS.",
        ],
    }

    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")

    total_server_errors = sum(item.get("server_error_count", 0) for item in results)
    total_exceptions = sum(item.get("exceptions", 0) for item in results)
    if args.fail_on_server_error and total_server_errors > 0:
        return 1
    if not args.allow_exceptions and total_exceptions > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
