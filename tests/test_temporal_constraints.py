from __future__ import annotations

import time
import unittest
from pathlib import Path

from evermemos_lite.domain.policy import EffectivePolicy
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


class _NoopGraphStore:
    enabled = False


class TemporalConstraintTests(unittest.TestCase):
    def _build_service(self) -> MemoryService:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=_NoopGraphStore(),
            phase4_reasoning_enabled=True,
        )
        service.repo.get_valid_foresights_for_episodes = lambda **_: {}
        return service

    @staticmethod
    def _policy_keyword_only() -> EffectivePolicy:
        return EffectivePolicy(
            vector_enabled=False,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=30,
            vector_top_k=30,
            rrf_k=60,
            profile="default",
            reason="test",
        )

    @staticmethod
    def _ts(date_str: str) -> int:
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")))

    def test_query_single_year_derives_bounds(self) -> None:
        service = self._build_service()
        _, start_ts, end_ts, reason = service._resolve_query_time_constraints(
            query="请回顾2024年的关键事项",
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        )
        expected_start, expected_end = service._year_bounds(2024)
        self.assertEqual(expected_start, int(start_ts or 0))
        self.assertEqual(expected_end, int(end_ts or 0))
        self.assertEqual("query_single_year", reason)

    def test_explicit_time_overrides_query_derived_bounds(self) -> None:
        service = self._build_service()
        as_of_ts, start_ts, end_ts, reason = service._resolve_query_time_constraints(
            query="请回顾2024年的关键事项",
            as_of_ts=1_700_000_000,
            start_ts=None,
            end_ts=None,
        )
        self.assertEqual(1_700_000_000, int(as_of_ts or 0))
        self.assertIsNone(start_ts)
        self.assertIsNone(end_ts)
        self.assertEqual("explicit_as_of", reason)

    def test_basic_search_applies_temporal_filter(self) -> None:
        service = self._build_service()
        policy = self._policy_keyword_only()
        service.repo.search_keyword = lambda **_: [
            {"memory_id": "m-2023", "score": 1.2},
            {"memory_id": "m-2024", "score": 1.1},
        ]
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": "m-2023",
                "event_id": "e-2023",
                "timestamp": self._ts("2023-06-01"),
                "summary": "older memory",
                "subject": "u1",
                "episode": "event in 2023",
                "importance_score": 0.7,
                "storage_tier": "text_only",
            },
            {
                "id": "m-2024",
                "event_id": "e-2024",
                "timestamp": self._ts("2024-06-01"),
                "summary": "in-range memory",
                "subject": "u1",
                "episode": "event in 2024",
                "importance_score": 0.7,
                "storage_tier": "text_only",
            },
        ]
        service.repo.fetch_episodes = lambda **_: []
        start_2024, end_2024 = service._year_bounds(2024)
        rows = service._basic_search(
            policy=policy,
            query="回顾项目里程碑",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
            attach_foresight=False,
            start_ts=start_2024,
            end_ts=end_2024,
        )
        self.assertEqual(1, len(rows))
        self.assertEqual("m-2024", str(rows[0].get("id")))

    def test_fallback_recent_applies_temporal_filter_and_drops_invalid_timestamp(self) -> None:
        service = self._build_service()
        service.repo.fetch_episodes = lambda **_: [
            {
                "id": "m-invalid-ts",
                "event_id": "e-0",
                "timestamp": None,
                "summary": "missing timestamp",
                "subject": "u1",
                "episode": "invalid row",
                "importance_score": 0.4,
                "storage_tier": "text_only",
            },
            {
                "id": "m-2023",
                "event_id": "e-1",
                "timestamp": self._ts("2023-05-01"),
                "summary": "old row",
                "subject": "u1",
                "episode": "old row",
                "importance_score": 0.4,
                "storage_tier": "text_only",
            },
            {
                "id": "m-2024",
                "event_id": "e-2",
                "timestamp": self._ts("2024-05-01"),
                "summary": "in-range row",
                "subject": "u1",
                "episode": "in-range row",
                "importance_score": 0.4,
                "storage_tier": "text_only",
            },
        ]
        start_2024, end_2024 = service._year_bounds(2024)
        rows = service._fallback_recent(
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
            attach_foresight=False,
            start_ts=start_2024,
            end_ts=end_2024,
        )
        self.assertEqual(["m-2024"], [str(row.get("id")) for row in rows])


if __name__ == "__main__":
    unittest.main()
