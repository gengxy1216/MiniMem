from __future__ import annotations

import unittest
from pathlib import Path

from flockmem.domain.policy import EffectivePolicy
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
        self._rows = list(rows)
        self.search_calls = 0

    def search(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        limit: int,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        self.search_calls += 1
        return list(self._rows)[: max(1, int(limit))]


class GraphRecallGateTests(unittest.TestCase):
    def _build_service(self, *, graph_rows: list[dict]) -> tuple[MemoryService, _StubGraphStore]:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        graph_store = _StubGraphStore(graph_rows)
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=graph_store,
            phase4_reasoning_enabled=True,
            graph_top_k=6,
        )
        return service, graph_store

    @staticmethod
    def _save_memory(
        service: MemoryService,
        *,
        message_id: str,
        event_id: str,
        user_id: str,
        group_id: str,
        ts: int,
        content: str,
    ) -> str:
        row = service.repo.save_message_as_memory(
            message_id=message_id,
            create_time=ts,
            sender=user_id,
            content=content,
            user_id=user_id,
            group_id=group_id,
            group_name=None,
            sender_name=user_id,
            role="user",
            importance_score=0.8,
            storage_tier="graph_text",
            summary=content,
            subject="graph-test",
            atomic_facts=[],
            foresights=[],
            profile_patch={},
            event_id=event_id,
        )
        return str(row["id"])

    def test_graph_query_gate_only_opens_for_complex_queries(self) -> None:
        service, graph_store = self._build_service(graph_rows=[])
        self.assertFalse(service._is_graph_query_eligible("咖啡"))
        self.assertTrue(service._is_graph_query_eligible("张三和李四是什么关系？"))
        self.assertTrue(service._is_graph_query_eligible("比较发布前后故障变化"))
        self.assertTrue(service._is_graph_query_eligible("先说项目风险然后说对应措施"))
        self.assertTrue(service._is_graph_query_eligible("请给我去年到今年的事件时间线"))

        base = [{"id": "m-base", "score": 0.5, "summary": "base"}]
        rows = service._merge_graph_hits(
            query="咖啡",
            user_id="u1",
            group_id="g1",
            base_hits=base,
            top_k=5,
        )
        self.assertEqual(["m-base"], [str(x.get("id")) for x in rows])
        self.assertEqual(0, graph_store.search_calls)

    def test_graph_guided_candidates_have_hit_and_miss_distribution(self) -> None:
        graph_rows = [
            {
                "subject": "张三",
                "relation": "has_son",
                "obj": "小明",
                "confidence": 0.92,
                "event_id": "evt-hit-rel",
                "timestamp": 1735689600,
                "match_score": 0.78,
            },
            {
                "subject": "项目A",
                "relation": "before",
                "obj": "项目B",
                "confidence": 0.82,
                "event_id": "evt-hit-time",
                "timestamp": 1711929600,
                "match_score": 0.62,
            },
            {
                "subject": "张三",
                "relation": "likes",
                "obj": "咖啡",
                "confidence": 0.25,
                "event_id": "evt-low-score",
                "timestamp": 1711929700,
                "match_score": 0.08,
            },
            {
                "subject": "张三",
                "relation": "knows",
                "obj": "王五",
                "confidence": 0.95,
                "event_id": "evt-group-miss",
                "timestamp": 1711929800,
                "match_score": 0.88,
            },
            {
                "subject": "张三",
                "relation": "knows",
                "obj": "赵六",
                "confidence": 0.95,
                "event_id": "evt-not-exists",
                "timestamp": 1711929900,
                "match_score": 0.9,
            },
        ]
        service, graph_store = self._build_service(graph_rows=graph_rows)
        rel_id = self._save_memory(
            service,
            message_id="m-hit-rel",
            event_id="evt-hit-rel",
            user_id="u1",
            group_id="g1",
            ts=1735689600,
            content="hit relation memory",
        )
        time_id = self._save_memory(
            service,
            message_id="m-hit-time",
            event_id="evt-hit-time",
            user_id="u1",
            group_id="g1",
            ts=1711929600,
            content="hit temporal memory",
        )
        low_score_id = self._save_memory(
            service,
            message_id="m-low-score",
            event_id="evt-low-score",
            user_id="u1",
            group_id="g1",
            ts=1711929700,
            content="low score memory",
        )
        group_miss_id = self._save_memory(
            service,
            message_id="m-group-miss",
            event_id="evt-group-miss",
            user_id="u1",
            group_id="g2",
            ts=1711929800,
            content="group miss memory",
        )

        candidate_ids = service._graph_guided_candidate_ids(
            query="张三和李四是什么关系？然后回顾之前发生了什么？",
            user_id="u1",
            group_id="g1",
            top_k=5,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        self.assertIn(rel_id, candidate_ids)
        self.assertIn(time_id, candidate_ids)
        self.assertNotIn(low_score_id, candidate_ids)
        self.assertNotIn(group_miss_id, candidate_ids)
        self.assertGreaterEqual(graph_store.search_calls, 1)

        calls_before = graph_store.search_calls
        no_graph_ids = service._graph_guided_candidate_ids(
            query="咖啡",
            user_id="u1",
            group_id="g1",
            top_k=5,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        self.assertEqual(set(), no_graph_ids)
        self.assertEqual(calls_before, graph_store.search_calls)

    def test_agentic_second_round_uses_graph_candidates(self) -> None:
        graph_rows = [
            {
                "subject": "张三",
                "relation": "has_son",
                "obj": "小明",
                "confidence": 0.94,
                "event_id": "evt-graph-candidate",
                "timestamp": 1735689600,
                "match_score": 0.82,
            }
        ]
        service, graph_store = self._build_service(graph_rows=graph_rows)
        target_id = self._save_memory(
            service,
            message_id="m-graph-target",
            event_id="evt-graph-candidate",
            user_id="u1",
            group_id="g1",
            ts=1735689600,
            content="graph candidate memory",
        )
        policy = EffectivePolicy(
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

        candidate_calls: list[set[str]] = []

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
            candidate_calls.append(set(candidate_episode_ids or set()))
            if len(candidate_calls) == 1:
                return [
                    {
                        "id": "m-round1",
                        "score": 0.12,
                        "summary": "round1",
                        "timestamp": 1735689600,
                        "source": "keyword",
                    }
                ]
            if not candidate_episode_ids:
                return []
            return [
                {
                    "id": str(mid),
                    "score": 0.6,
                    "summary": "graph candidate hit",
                    "timestamp": 1735689601,
                    "source": "keyword",
                }
                for mid in sorted(candidate_episode_ids)
            ][: max(1, int(top_k))]

        service._basic_search = _fake_basic_search
        service._scene_guided_candidate_ids = lambda *args, **kwargs: {
            "episode_ids": [],
            "scene_score_map": {},
            "episode_scene_map": {},
        }

        rows = service._agentic_search(
            policy=policy,
            query="张三和李四是什么关系？然后回顾之前发生了什么？",
            user_id="u1",
            group_id="g1",
            top_k=3,
        )
        self.assertTrue(any(target_id == str(row.get("id", "")) for row in rows))
        self.assertGreaterEqual(graph_store.search_calls, 1)
        self.assertGreaterEqual(len(candidate_calls), 2)
        self.assertTrue(any(target_id in s for s in candidate_calls[1:]))


if __name__ == "__main__":
    unittest.main()

