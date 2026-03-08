from __future__ import annotations

import os
import unittest
from pathlib import Path

from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.service.memory_service import MemoryService
from flockmem.testing.writable_tempdir import WritableTempDir


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
        self._rows = rows

    def search(self, **kwargs) -> list[dict]:
        return list(self._rows)


class FtsTokenizerConfigTests(unittest.TestCase):
    def test_invalid_fts_tokenizer_falls_back_to_unicode61(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            old = os.environ.get("LITE_FTS_TOKENIZER")
            os.environ["LITE_FTS_TOKENIZER"] = "not_supported_tokenizer"
            try:
                engine = SQLiteEngine(Path(tmp) / "lite.db")
                init_schema(engine)
            finally:
                if old is None:
                    os.environ.pop("LITE_FTS_TOKENIZER", None)
                else:
                    os.environ["LITE_FTS_TOKENIZER"] = old
            row = engine.query_one(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type='table' AND name='memory_keyword_fts'
                """
            )
            ddl = str((row or {}).get("sql") or "").lower()
            self.assertIn("unicode61", ddl)


class RetrievalFusionOptimizationTests(unittest.TestCase):
    def _build_service(self, *, graph_store) -> MemoryService:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=graph_store,
            phase4_reasoning_enabled=True,
        )
        return service

    @staticmethod
    def _mrr_for_target(rows: list[dict], target_id: str) -> float:
        for idx, row in enumerate(rows, start=1):
            if str(row.get("id", "")) == target_id:
                return 1.0 / float(idx)
        return 0.0

    def test_rrf_graph_fusion_promotes_cross_channel_event(self) -> None:
        graph_store = _StubGraphStore(
            [
                {
                    "subject": "项目A",
                    "relation": "caused_by",
                    "obj": "预算超支",
                    "confidence": 0.92,
                    "match_score": 0.95,
                    "event_id": "evt-x",
                    "timestamp": 1730000300,
                    "user_id": "u1",
                    "group_id": "g1",
                }
            ]
        )
        service = self._build_service(graph_store=graph_store)
        base_hits = [
            {"id": "m-top", "event_id": "evt-top", "summary": "一般状态", "subject": "项目A", "episode": "一般状态", "score": 0.90},
            {"id": "m-mid", "event_id": "evt-mid", "summary": "一般状态2", "subject": "项目A", "episode": "一般状态2", "score": 0.85},
            {"id": "m-x", "event_id": "evt-x", "summary": "预算项", "subject": "项目A", "episode": "预算项", "score": 0.60},
        ]
        merged = service._merge_graph_hits(
            query="谁导致了预算超支关系？",
            user_id="u1",
            group_id="g1",
            base_hits=base_hits,
            top_k=3,
            rrf_k=60,
        )
        self.assertGreaterEqual(len(merged), 1)
        self.assertEqual("m-x", str(merged[0].get("id")))

    def test_dedup_density_merge_reduces_redundancy_and_improves_mrr(self) -> None:
        service = self._build_service(graph_store=_StubGraphStore([]))
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
        baseline = sorted(rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:3]
        optimized = service._dedup_and_density_merge_rows(
            query="项目B 预算超支",
            rows=rows,
            top_k=3,
        )
        baseline_mrr = self._mrr_for_target(baseline, "m-b-1")
        optimized_mrr = self._mrr_for_target(optimized, "m-b-1")
        self.assertGreater(optimized_mrr, baseline_mrr)
        self.assertLess(len(optimized), len(baseline))
        dense = optimized[0]
        self.assertIn("项目A", str(dense.get("episode", "")))


if __name__ == "__main__":
    unittest.main()

