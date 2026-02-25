from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.domain.policy import EffectivePolicy
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.memory_service import MemoryService
from evermemos_lite.service.query_rewriter import QueryExpansionDecision


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


class _NoopGraphStore:
    enabled = False


class _StubExpandingRewriter:
    def __init__(self, decision: QueryExpansionDecision | None) -> None:
        self.decision = decision
        self.calls = 0

    def expand_queries(
        self,
        *,
        query: str,
        hits: list[dict],
        insufficiency_reason: str,
        max_queries: int = 3,
    ) -> QueryExpansionDecision | None:
        self.calls += 1
        return self.decision


class PhaseFourReasoningTests(unittest.TestCase):
    def _build_service(self, *, query_rewriter=None) -> MemoryService:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        return MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=_NoopGraphStore(),
            query_rewriter=query_rewriter,
            phase4_reasoning_enabled=True,
            temporal_rerank_weight=0.4,
            multi_hop_max_queries=3,
        )

    @staticmethod
    def _policy() -> EffectivePolicy:
        return EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=True,
            importance_threshold=0.1,
            keyword_top_k=30,
            vector_top_k=30,
            rrf_k=60,
            profile="agentic",
            reason="test",
        )

    def test_temporal_rerank_prefers_target_year(self) -> None:
        service = self._build_service()
        ts_2024 = int(time.mktime(time.strptime("2024-05-01", "%Y-%m-%d")))
        ts_2022 = int(time.mktime(time.strptime("2022-05-01", "%Y-%m-%d")))
        rows = [
            {"id": "m-2022", "score": 0.5, "timestamp": ts_2022},
            {"id": "m-2024", "score": 0.5, "timestamp": ts_2024},
        ]
        reranked = service._rerank_temporal_rows(query="请回顾2024年的关键事件", rows=rows)
        self.assertEqual("m-2024", str(reranked[0].get("id")))

    def test_multi_hop_second_round_runs_multiple_queries(self) -> None:
        service = self._build_service()
        calls: list[str] = []

        def _fake_basic_search(
            policy,
            query,
            user_id,
            group_id,
            top_k,
            candidate_episode_ids,
            attach_foresight=True,
            **kwargs,
        ):
            calls.append(str(query))
            return [
                {
                    "id": f"id:{query}",
                    "score": 0.5,
                    "summary": str(query),
                    "timestamp": int(time.time()),
                }
            ]

        service._basic_search = _fake_basic_search
        rows = service._run_agentic_second_round(
            policy=self._policy(),
            original_query="先说项目进度，然后说风险以及对应措施",
            rewritten_query="先说项目进度，然后说风险以及对应措施",
            first_round_hits=[{"id": "m1", "summary": "项目进度"}],
            insufficiency_reason="need_more_context",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        self.assertGreaterEqual(len(calls), 2)
        self.assertGreaterEqual(len(rows), 1)

    def test_query_expansion_applies_for_non_multi_hop(self) -> None:
        rewriter = _StubExpandingRewriter(
            QueryExpansionDecision(
                queries=["项目风险 关键节点", "风险缓解措施 截止时间"],
                confidence=0.9,
                reason="missing_clues",
            )
        )
        service = self._build_service(query_rewriter=rewriter)
        calls: list[str] = []

        def _fake_basic_search(
            policy,
            query,
            user_id,
            group_id,
            top_k,
            candidate_episode_ids,
            attach_foresight=True,
            **kwargs,
        ):
            calls.append(str(query))
            return [
                {
                    "id": f"id:{query}",
                    "score": 0.45,
                    "summary": str(query),
                    "timestamp": int(time.time()),
                }
            ]

        service._basic_search = _fake_basic_search
        rows = service._run_agentic_second_round(
            policy=self._policy(),
            original_query="项目风险怎么样",
            rewritten_query="项目风险怎么样",
            first_round_hits=[{"id": "m0", "summary": "风险待跟进"}],
            insufficiency_reason="missing_details",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        self.assertGreaterEqual(rewriter.calls, 1)
        self.assertIn("项目风险 关键节点", calls)
        self.assertGreaterEqual(len(rows), 1)

    def test_second_round_overlap_rerank_promotes_query_aligned_row(self) -> None:
        rewriter = _StubExpandingRewriter(
            QueryExpansionDecision(
                queries=["关键节点 里程碑"],
                confidence=0.95,
                reason="focus_timeline",
            )
        )
        service = self._build_service(query_rewriter=rewriter)

        def _fake_basic_search(
            policy,
            query,
            user_id,
            group_id,
            top_k,
            candidate_episode_ids,
            attach_foresight=True,
            **kwargs,
        ):
            if "关键节点 里程碑" in str(query):
                return [
                    {
                        "id": "m-expanded",
                        "score": 0.7,
                        "summary": "关键节点 里程碑 延期两周",
                        "episode": "项目风险来自关键路径延期",
                        "timestamp": int(time.time()),
                    }
                ]
            return [
                {
                    "id": "m-base",
                    "score": 0.7,
                    "summary": "通用项目状态",
                    "episode": "项目整体正常推进",
                    "timestamp": int(time.time()),
                }
            ]

        service._basic_search = _fake_basic_search
        rows = service._run_agentic_second_round(
            policy=self._policy(),
            original_query="项目风险",
            rewritten_query="项目风险",
            first_round_hits=[{"id": "m0", "summary": "风险问题"}],
            insufficiency_reason="need_specifics",
            user_id="u1",
            group_id="g1",
            top_k=2,
            candidate_episode_ids=None,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        self.assertEqual("m-expanded", str(rows[0].get("id")))


if __name__ == "__main__":
    unittest.main()
