from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


PERF_BUDGETS = {
    "ingest": {"p50_ms": 150.0, "p95_ms": 450.0},
    "context": {"p50_ms": 120.0, "p95_ms": 350.0},
}
RSS_BUDGET_MB = 350.0


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


def _build_client() -> tuple[TestClient, WritableTempDir]:
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    env = {
        "LITE_DATA_DIR": str(Path(tmp.name) / "qa-baseline-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / "qa-baseline-config"),
        "LITE_CHAT_PROVIDER": "openai",
        "LITE_CHAT_BASE_URL": "https://chat.example/v1",
        "LITE_CHAT_API_KEY": "qa-chat-key",
        "LITE_CHAT_MODEL": "qa-chat-model",
        "LITE_EMBEDDING_PROVIDER": "local",
        "LITE_EMBEDDING_MODEL": "local-hash-384",
        "LITE_EXTRACTOR_PROVIDER": "rule",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = LiteSettings.from_env()
    app = create_app(settings)
    return TestClient(app), tmp


def _ingest_payload(idx: int) -> dict[str, object]:
    return {
        "knowledge_id": f"k-baseline-{idx}",
        "scope_type": "personal",
        "scope_id": "u-baseline",
        "content": {"fact": f"baseline ingest #{idx}"},
        "change_type": "create",
        "changed_by": "agent",
        "actor_id": "qa-bench",
        "read_acl": ["qa-bench"],
        "write_acl": ["qa-bench"],
        "coordination_mode": "inruntime_a2a",
        "coordination_id": f"coord-baseline-{idx}",
        "runtime_id": "codex",
        "agent_id": "qa-gate",
    }


def _context_payload() -> dict[str, object]:
    return {
        "query": "collective baseline context lookup",
        "actor_id": "qa-bench",
        "personal_scope_id": "u-baseline",
        "include_global": False,
        "top_k": 20,
    }


def _feedback_payload(knowledge_id: str, revision_id: str) -> dict[str, object]:
    return {
        "knowledge_id": knowledge_id,
        "revision_id": revision_id,
        "feedback_type": "execution_signal",
        "feedback_payload": {"outcome_status": "success", "retry_count": 0},
        "actor": "qa-bench",
        "coordination_mode": "inruntime_a2a",
        "coordination_id": "coord-baseline-feedback",
    }


def _run_case(
    *,
    client: TestClient,
    loops: int,
    name: str,
    path: str,
    payload_builder,
) -> dict[str, object]:
    latencies_ms: list[float] = []
    status_hist: dict[str, int] = {}
    exceptions = 0
    for idx in range(loops):
        payload = payload_builder(idx)
        start = time.perf_counter()
        try:
            response = client.post(path, json=payload)
            key = str(int(response.status_code))
            status_hist[key] = status_hist.get(key, 0) + 1
        except Exception:
            exceptions += 1
            status_hist["exception"] = status_hist.get("exception", 0) + 1
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    avg_ms = round(float(statistics.mean(latencies_ms)), 3) if latencies_ms else 0.0
    p50_ms = _percentile_ms(latencies_ms, 0.50)
    p95_ms = _percentile_ms(latencies_ms, 0.95)
    server_error_count = sum(
        count
        for status_code, count in status_hist.items()
        if status_code.isdigit() and int(status_code) >= 500
    )
    case_result: dict[str, object] = {
        "name": name,
        "path": path,
        "loops": loops,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "max_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        "status_histogram": status_hist,
        "exceptions": exceptions,
        "server_error_count": server_error_count,
    }
    budget = PERF_BUDGETS.get(name)
    if budget:
        case_result["budget"] = {
            "p50_ms_target": budget["p50_ms"],
            "p95_ms_target": budget["p95_ms"],
            "pass": bool(
                p50_ms <= float(budget["p50_ms"]) and p95_ms <= float(budget["p95_ms"])
            ),
        }
    return case_result


def _run_ingest_concurrency_case(
    *,
    client: TestClient,
    workers: int,
    total_requests: int,
) -> dict[str, object]:
    workers = max(1, int(workers))
    total_requests = max(1, int(total_requests))
    latencies_ms: list[float] = []
    status_hist: dict[str, int] = {}
    exceptions = 0
    lock = threading.Lock()

    def _one(idx: int) -> None:
        nonlocal exceptions
        payload = _ingest_payload(100000 + idx)
        start = time.perf_counter()
        try:
            response = client.post("/api/v1/collective/ingest", json=payload)
            status_key = str(int(response.status_code))
            with lock:
                status_hist[status_key] = status_hist.get(status_key, 0) + 1
        except Exception:
            with lock:
                exceptions += 1
                status_hist["exception"] = status_hist.get("exception", 0) + 1
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with lock:
                latencies_ms.append(elapsed_ms)

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_one, range(total_requests)))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    avg_ms = round(float(statistics.mean(latencies_ms)), 3) if latencies_ms else 0.0
    p50_ms = _percentile_ms(latencies_ms, 0.50)
    p95_ms = _percentile_ms(latencies_ms, 0.95)
    server_error_count = sum(
        count
        for status_code, count in status_hist.items()
        if status_code.isdigit() and int(status_code) >= 500
    )
    throughput_rps = round((1000.0 * total_requests / wall_ms), 3) if wall_ms > 0 else 0.0
    return {
        "name": "ingest_concurrency",
        "path": "/api/v1/collective/ingest",
        "workers": workers,
        "total_requests": total_requests,
        "wall_ms": round(wall_ms, 3),
        "throughput_rps": throughput_rps,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "max_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        "status_histogram": status_hist,
        "exceptions": exceptions,
        "server_error_count": server_error_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collective v1 product-gate baseline (in-process)."
    )
    parser.add_argument("--loops", type=int, default=30, help="Requests per endpoint.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent workers for ingest concurrency benchmark.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=120,
        help="Total ingest requests for ingest concurrency benchmark.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--fail-on-server-error",
        action="store_true",
        help="Exit non-zero when any case returns HTTP >= 500.",
    )
    parser.add_argument(
        "--enforce-budget",
        action="store_true",
        help="Exit non-zero when latency/RSS budget is exceeded.",
    )
    args = parser.parse_args()

    loops = max(1, int(args.loops))
    client, tmp = _build_client()
    try:
        seed_ingest = client.post("/api/v1/collective/ingest", json=_ingest_payload(0))
        if seed_ingest.status_code != 200:
            raise RuntimeError(
                "baseline warmup ingest failed: "
                + str(seed_ingest.status_code)
                + " "
                + seed_ingest.text
            )
        seed_result = seed_ingest.json().get("result", {})
        feedback_knowledge_id = str(seed_result.get("knowledge_id") or "")
        feedback_revision_id = str(seed_result.get("revision_id") or "")
        if not feedback_knowledge_id or not feedback_revision_id:
            raise RuntimeError("baseline warmup ingest missing knowledge_id/revision_id")

        started_at = datetime.now(tz=timezone.utc).isoformat()
        results = [
            _run_case(
                client=client,
                loops=loops,
                name="ingest",
                path="/api/v1/collective/ingest",
                payload_builder=lambda idx: _ingest_payload(idx + 1),
            ),
            _run_case(
                client=client,
                loops=loops,
                name="context",
                path="/api/v1/collective/context",
                payload_builder=lambda _: _context_payload(),
            ),
            _run_case(
                client=client,
                loops=loops,
                name="feedback",
                path="/api/v1/collective/feedback",
                payload_builder=lambda _: _feedback_payload(
                    feedback_knowledge_id, feedback_revision_id
                ),
            )
        ]
        ingest_concurrency = _run_ingest_concurrency_case(
            client=client,
            workers=args.workers,
            total_requests=args.requests,
        )
        results.append(ingest_concurrency)
        rss_mb = _current_rss_mb()
        summary = {
            "started_at_utc": started_at,
            "mode": "inproc-testclient",
            "loops_per_endpoint": loops,
            "ingest_concurrency_workers": max(1, int(args.workers)),
            "ingest_concurrency_requests": max(1, int(args.requests)),
            "rss_mb_current_process": rss_mb,
            "rss_budget_mb": RSS_BUDGET_MB,
            "rss_budget_pass": bool(rss_mb is not None and rss_mb <= RSS_BUDGET_MB),
            "results": results,
            "notes": [
                "This benchmark measures in-process client-observed latency.",
                "For release gate, pair with server process RSS monitor.",
            ],
        }

        output_text = json.dumps(summary, ensure_ascii=False, indent=2)
        print(output_text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text + "\n")

        total_server_errors = sum(
            int(item.get("server_error_count", 0)) for item in results
        )
        if args.fail_on_server_error and total_server_errors > 0:
            return 1
        if args.enforce_budget:
            budget_failed = False
            for item in results:
                budget = item.get("budget")
                if isinstance(budget, dict) and not bool(budget.get("pass", False)):
                    budget_failed = True
                    break
            if not bool(summary.get("rss_budget_pass", False)):
                budget_failed = True
            total_exceptions = sum(int(item.get("exceptions", 0)) for item in results)
            if total_exceptions > 0:
                budget_failed = True
            if budget_failed:
                return 1
        return 0
    finally:
        client.close()
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
