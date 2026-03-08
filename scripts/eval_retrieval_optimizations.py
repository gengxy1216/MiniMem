from __future__ import annotations

import json
import time
from pathlib import Path

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.memory_service import MemoryService
from evermemos_lite.testing.writable_tempdir import WritableTempDir


class _NoopVectorStore:
    enabled = False
    vector_dim = 4

    def search(self, **kwargs):
        return []

    def upsert(self, row_id: str, memory_id: str, vector: list[float], metadata: dict) -> None:
        return None


class _NoopEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _NoopExtractor:
    def extract(self, content: str, sender: str, group_id: str | None):
        raise NotImplementedError


class _StubGraphStore:
    def __init__(self, rows: list[dict]) -> None:
        self.enabled = True
        self._rows = list(rows)

    def search(self, **kwargs) -> list[dict]:
        return list(self._rows)


class _LegacyNoDedupeService(MemoryService):
    def _dedup_and_density_merge_rows(self, *, query: str, rows: list[dict], top_k: int) -> list[dict]:
        ranked = sorted(
            [dict(r) for r in rows if isinstance(r, dict)],
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )
        return ranked[: max(1, int(top_k))]


class _LegacyNoGraphRrfService(MemoryService):
    def _merge_graph_hits(
        self,
        query: str,
        user_id: str | None,
        group_id: str | None,
        base_hits: list[dict],
        top_k: int,
        rrf_k: int = 60,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        merged = [dict(x) for x in base_hits]
        if not self.graph_store.enabled:
            return merged[: max(1, int(top_k))]
        try:
            graph_rows = self.graph_store.search(
                query=query,
                user_id=user_id,
                group_id=group_id,
                limit=max(1, min(self.graph_top_k, top_k)),
            )
        except Exception:
            graph_rows = []
        for g in graph_rows:
            summary = f"{g['subject']} -[{g['relation']}]-> {g['obj']}"
            confidence = float(g.get("confidence", 0.5))
            match_score = float(g.get("match_score", 0.0))
            graph_score = 0.38 + 0.27 * confidence + 0.35 * match_score
            merged.append(
                {
                    "id": f"graph:{int(time.time() * 1000000)}",
                    "event_id": g.get("event_id"),
                    "timestamp": g.get("timestamp"),
                    "summary": summary,
                    "subject": g.get("subject", ""),
                    "episode": summary,
                    "score": graph_score,
                    "importance_score": 0.6,
                    "source": "graph_kuzu",
                }
            )
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[: max(1, int(top_k))]


_TMP_HOLD: list[WritableTempDir] = []


def _build_service(service_cls, *, graph_rows: list[dict]) -> MemoryService:
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    _TMP_HOLD.append(tmp)
    engine = SQLiteEngine(Path(tmp.name) / "lite.db")
    init_schema(engine)
    return service_cls(
        engine=engine,
        vector_store=_NoopVectorStore(),
        embedding_provider=_NoopEmbeddingProvider(),
        extractor=_NoopExtractor(),
        graph_store=_StubGraphStore(graph_rows),
        phase4_reasoning_enabled=True,
    )


def _mrr(rows: list[dict], target_id: str) -> float:
    for idx, row in enumerate(rows, start=1):
        if str(row.get("id", "")) == target_id:
            return 1.0 / float(idx)
    return 0.0


def _scenario_dedup_density() -> dict:
    rows = [
        {
            "id": "m-a-1",
            "event_id": "evt-a-1",
            "timestamp": 1730000000,
            "summary": "项目A 风险 延期",
            "episode": "项目A 风险 延期，关键路径滞后",
            "score": 0.95,
            "source": "keyword",
        },
        {
            "id": "m-a-2",
            "event_id": "evt-a-2",
            "timestamp": 1730000100,
            "summary": "项目A 风险 延期",
            "episode": "项目A 风险 延期，关键路径滞后",
            "score": 0.93,
            "source": "vector",
        },
        {
            "id": "m-b-1",
            "event_id": "evt-b-1",
            "timestamp": 1730000200,
            "summary": "项目B 预算超支",
            "episode": "项目B 预算超支，需要修复计划",
            "score": 0.90,
            "source": "keyword",
        },
    ]
    legacy = _build_service(_LegacyNoDedupeService, graph_rows=[])
    optimized = _build_service(MemoryService, graph_rows=[])
    before = legacy._dedup_and_density_merge_rows(query="项目B 预算超支", rows=rows, top_k=3)
    after = optimized._dedup_and_density_merge_rows(query="项目B 预算超支", rows=rows, top_k=3)
    return {
        "mrr_before": _mrr(before, "m-b-1"),
        "mrr_after": _mrr(after, "m-b-1"),
        "top_k_before": len(before),
        "top_k_after": len(after),
    }


def _scenario_graph_rrf() -> dict:
    graph_rows = [
        {
            "subject": "项目A",
            "relation": "caused_by",
            "obj": "预算超支",
            "confidence": 0.55,
            "match_score": 0.60,
            "event_id": "evt-x",
            "timestamp": 1730000300,
            "user_id": "u1",
            "group_id": "g1",
        }
    ]
    base_hits = [
        {"id": "m-top", "event_id": "evt-top", "summary": "一般状态", "subject": "项目A", "episode": "一般状态", "score": 0.90},
        {"id": "m-mid", "event_id": "evt-mid", "summary": "一般状态2", "subject": "项目A", "episode": "一般状态2", "score": 0.85},
        {"id": "m-x", "event_id": "evt-x", "summary": "预算项", "subject": "项目A", "episode": "预算项", "score": 0.60},
    ]
    legacy = _build_service(_LegacyNoGraphRrfService, graph_rows=graph_rows)
    optimized = _build_service(MemoryService, graph_rows=graph_rows)
    before = legacy._merge_graph_hits(
        query="谁导致了预算超支关系？",
        user_id="u1",
        group_id="g1",
        base_hits=base_hits,
        top_k=3,
        rrf_k=60,
    )
    after = optimized._merge_graph_hits(
        query="谁导致了预算超支关系？",
        user_id="u1",
        group_id="g1",
        base_hits=base_hits,
        top_k=3,
        rrf_k=60,
    )

    def _rank_by_event(rows: list[dict], target_event_id: str) -> int:
        for idx, row in enumerate(rows, start=1):
            if str(row.get("event_id", "")) == target_event_id:
                return idx
        return 999

    def _mrr_by_event(rows: list[dict], target_event_id: str) -> float:
        rank = _rank_by_event(rows, target_event_id)
        if rank <= 0 or rank >= 999:
            return 0.0
        return 1.0 / float(rank)

    return {
        "target_event_rank_before": _rank_by_event(before, "evt-x"),
        "target_event_rank_after": _rank_by_event(after, "evt-x"),
        "event_mrr_before": _mrr_by_event(before, "evt-x"),
        "event_mrr_after": _mrr_by_event(after, "evt-x"),
    }


def _scenario_overhead() -> dict:
    optimized = _build_service(MemoryService, graph_rows=[])
    legacy = _build_service(_LegacyNoGraphRrfService, graph_rows=[])
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"m-{i}",
                "event_id": f"evt-{i // 2}",
                "timestamp": 1730000000 + i * 30,
                "summary": f"项目{i // 2} 风险 延期",
                "episode": f"项目{i // 2} 风险 延期，细节{i}",
                "score": 0.99 - 0.01 * i,
                "source": "keyword" if i % 2 == 0 else "vector",
            }
        )
    rounds = 400
    started = time.perf_counter()
    for _ in range(rounds):
        optimized._dedup_and_density_merge_rows(query="项目风险", rows=rows, top_k=10)
    dedup_ms = (time.perf_counter() - started) * 1000.0 / float(rounds)
    base_hits = [
        {
            "id": f"m-{i}",
            "event_id": f"evt-{i}",
            "summary": f"项目{i} 概况",
            "subject": "项目",
            "episode": f"项目{i} 概况",
            "score": 1.0 - 0.01 * i,
        }
        for i in range(20)
    ]
    graph_rows = [
        {
            "subject": f"项目{i}",
            "relation": "related_to",
            "obj": "预算风险",
            "confidence": 0.56,
            "match_score": 0.62,
            "event_id": f"evt-{i}",
            "timestamp": 1730000000 + i * 60,
            "user_id": "u1",
            "group_id": "g1",
        }
        for i in range(5)
    ]
    optimized.graph_store = _StubGraphStore(graph_rows)
    legacy.graph_store = _StubGraphStore(graph_rows)
    started = time.perf_counter()
    for _ in range(rounds):
        legacy._merge_graph_hits(
            query="谁和预算风险相关？",
            user_id="u1",
            group_id="g1",
            base_hits=base_hits,
            top_k=10,
            rrf_k=60,
        )
    graph_merge_legacy_ms = (time.perf_counter() - started) * 1000.0 / float(rounds)
    started = time.perf_counter()
    for _ in range(rounds):
        optimized._merge_graph_hits(
            query="谁和预算风险相关？",
            user_id="u1",
            group_id="g1",
            base_hits=base_hits,
            top_k=10,
            rrf_k=60,
        )
    graph_merge_rrf_ms = (time.perf_counter() - started) * 1000.0 / float(rounds)
    return {
        "avg_dedup_merge_ms": round(dedup_ms, 3),
        "avg_graph_merge_legacy_ms": round(graph_merge_legacy_ms, 3),
        "avg_graph_merge_rrf_ms": round(graph_merge_rrf_ms, 3),
    }


def main() -> None:
    report = {
        "date": "2026-03-06",
        "scenarios": {
            "dedup_density_merge": _scenario_dedup_density(),
            "graph_rrf_fusion": _scenario_graph_rrf(),
            "overhead": _scenario_overhead(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
