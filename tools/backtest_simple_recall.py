from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from flockmem.domain.policy import EffectivePolicy
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.infra.vector.lancedb_store import LanceVectorStore
from flockmem.service.extractor import RuleMemoryExtractor
from flockmem.service.local_embedding import LocalHashEmbeddingProvider
from flockmem.service.memory_service import MemorizeInput, MemoryService
from flockmem.testing.writable_tempdir import WritableTempDir


class _NoopGraphStore:
    enabled = False


CASES: list[dict[str, str]] = [
    {"content": "alpha outage runbook updated monday", "query": "alpha runbook monday"},
    {"content": "beta payroll deadline moved friday", "query": "beta payroll friday"},
    {"content": "gamma migration owner is nora", "query": "gamma migration nora"},
    {"content": "delta onboarding batch starts april", "query": "delta onboarding april"},
    {"content": "epsilon support rota changed night", "query": "epsilon rota night"},
    {"content": "zeta invoice issue resolved today", "query": "zeta invoice today"},
]


def _build_service(
    *,
    vector_write_min_importance: float,
    lance_min_importance: float,
    search_budget_factor: int,
    search_min_probe_k: int,
    semantic_vector_budget_cap: int,
    semantic_keyword_budget_cap: int,
) -> tuple[MemoryService, WritableTempDir]:
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    base = Path(tmp.name)
    engine = SQLiteEngine(base / "lite.db")
    init_schema(engine)
    vector_store = LanceVectorStore(
        base / "vec",
        vector_dim=384,
        use_lancedb=False,
        lance_persist_min_importance=lance_min_importance,
    )
    embedder = LocalHashEmbeddingProvider(model="local-hash-384", vector_dim=384)
    service = MemoryService(
        engine=engine,
        vector_store=vector_store,
        embedding_provider=embedder,
        extractor=RuleMemoryExtractor(),
        graph_store=_NoopGraphStore(),
        vector_write_min_importance=vector_write_min_importance,
        search_budget_factor=search_budget_factor,
        search_min_probe_k=search_min_probe_k,
        semantic_vector_budget_cap=semantic_vector_budget_cap,
        semantic_keyword_budget_cap=semantic_keyword_budget_cap,
    )
    return service, tmp


def _vector_policy() -> EffectivePolicy:
    return EffectivePolicy(
        vector_enabled=True,
        keyword_enabled=False,
        agentic_enabled=False,
        importance_threshold=0.15,
        keyword_top_k=20,
        vector_top_k=20,
        rrf_k=60,
        profile="vector_only_backtest",
        reason="simple_backtest",
    )


def _ingest_cases(service: MemoryService, policy: EffectivePolicy) -> list[str]:
    event_ids: list[str] = []
    now = int(time.time())
    for idx, item in enumerate(CASES, start=1):
        request_id = f"req-{idx}"
        result = service.memorize(
            MemorizeInput(
                message_id=f"m-{idx}",
                create_time=now + idx,
                sender="u1",
                content=item["content"],
                group_id="g1",
                group_name="g1",
                sender_name="u1",
                role="user",
            ),
            request_id=request_id,
        )
        memory = result.get("memory", {})
        service.maybe_index_vector(policy, memory)
        event_ids.append(str(memory.get("event_id", request_id)))
    return event_ids


def _evaluate(service: MemoryService, policy: EffectivePolicy, expected_event_ids: list[str]) -> dict[str, Any]:
    hits = 0
    mrr = 0.0
    rows_out: list[dict[str, Any]] = []
    for idx, item in enumerate(CASES):
        expected = expected_event_ids[idx]
        rows = service.search(
            policy=policy,
            query=item["query"],
            user_id="u1",
            group_id="g1",
            top_k=5,
        )
        rank = 0
        for r_idx, row in enumerate(rows, start=1):
            if str(row.get("event_id", "")) == expected:
                rank = r_idx
                break
        if rank > 0:
            hits += 1
            mrr += 1.0 / float(rank)
        rows_out.append(
            {
                "query": item["query"],
                "expected_event_id": expected,
                "rank": rank,
                "top_event_ids": [str(x.get("event_id", "")) for x in rows[:5]],
            }
        )
    total = len(CASES)
    return {
        "total": total,
        "recall_at_5": round(hits / float(total), 4) if total else 0.0,
        "mrr_at_5": round(mrr / float(total), 4) if total else 0.0,
        "details": rows_out,
    }


def run() -> dict[str, Any]:
    policy = _vector_policy()
    baseline_service, baseline_tmp = _build_service(
        vector_write_min_importance=0.30,
        lance_min_importance=0.72,
        search_budget_factor=4,
        search_min_probe_k=12,
        semantic_vector_budget_cap=32,
        semantic_keyword_budget_cap=16,
    )
    improved_service, improved_tmp = _build_service(
        vector_write_min_importance=0.10,
        lance_min_importance=0.55,
        search_budget_factor=8,
        search_min_probe_k=24,
        semantic_vector_budget_cap=64,
        semantic_keyword_budget_cap=32,
    )
    try:
        baseline_events = _ingest_cases(baseline_service, policy)
        improved_events = _ingest_cases(improved_service, policy)
        baseline_eval = _evaluate(baseline_service, policy, baseline_events)
        improved_eval = _evaluate(improved_service, policy, improved_events)
        return {
            "suite": "simple-vector-recall-backtest",
            "baseline": {
                "config": {
                    "vector_write_min_importance": 0.30,
                    "lance_min_importance": 0.72,
                    "search_budget_factor": 4,
                    "search_min_probe_k": 12,
                    "semantic_vector_budget_cap": 32,
                    "semantic_keyword_budget_cap": 16,
                },
                "vector_index_stats": baseline_service.get_vector_index_stats(),
                "metrics": baseline_eval,
            },
            "improved": {
                "config": {
                    "vector_write_min_importance": 0.10,
                    "lance_min_importance": 0.55,
                    "search_budget_factor": 8,
                    "search_min_probe_k": 24,
                    "semantic_vector_budget_cap": 64,
                    "semantic_keyword_budget_cap": 32,
                },
                "vector_index_stats": improved_service.get_vector_index_stats(),
                "metrics": improved_eval,
            },
        }
    finally:
        baseline_tmp.cleanup()
        improved_tmp.cleanup()


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

