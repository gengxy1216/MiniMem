from __future__ import annotations

import unittest
from pathlib import Path

from flockmem.domain.policy import EffectivePolicy
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.service.memory_service import MemoryService
from flockmem.testing.writable_tempdir import WritableTempDir


class _StubVectorStore:
    def __init__(self, hits: list[dict], *, enabled: bool = True) -> None:
        self.hits = list(hits)
        self.enabled = bool(enabled)
        self.vector_dim = 4

    def search(self, **kwargs) -> list[dict]:
        return list(self.hits)

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


class EventLogRetrievalChannelTests(unittest.TestCase):
    def _build_service(
        self,
        *,
        vector_hits: list[dict],
        event_vector_hits: list[dict],
        enable_vector: bool = True,
        enable_event_vector: bool = True,
    ) -> MemoryService:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        service = MemoryService(
            engine=engine,
            vector_store=_StubVectorStore(vector_hits, enabled=enable_vector),
            event_log_vector_store=_StubVectorStore(
                event_vector_hits,
                enabled=enable_event_vector,
            ),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=_NoopGraphStore(),
        )
        service.repo.get_valid_foresights_for_episodes = lambda **_: {}
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": str(mid),
                "event_id": f"evt-{mid}",
                "timestamp": 1700000000,
                "summary": f"summary-{mid}",
                "subject": "u1",
                "episode": f"episode-{mid}",
                "importance_score": 0.7,
                "storage_tier": "vector_only",
            }
            for mid in episode_ids
        ]
        return service

    @staticmethod
    def _policy(*, keyword_enabled: bool, vector_enabled: bool) -> EffectivePolicy:
        return EffectivePolicy(
            vector_enabled=vector_enabled,
            keyword_enabled=keyword_enabled,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=20,
            vector_top_k=20,
            rrf_k=60,
            profile="test",
            reason="event_log_channel_test",
        )

    def test_basic_search_can_return_event_log_keyword_hits(self) -> None:
        service = self._build_service(vector_hits=[], event_vector_hits=[], enable_vector=False)
        service.repo.search_keyword = lambda **_: []
        service.repo.search_event_log_keyword = lambda **_: [
            {
                "memory_id": "m-el-kw",
                "score": 0.93,
                "source": "event_log_keyword",
                "in_event_log_keyword_hits": True,
                "event_log_fact_hint": "owner is nora",
            }
        ]
        rows = service._basic_search(
            policy=self._policy(keyword_enabled=True, vector_enabled=False),
            query="who is nora",
            user_id="u1",
            group_id="g1",
            top_k=3,
            candidate_episode_ids=None,
            attach_foresight=False,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual("m-el-kw", str(rows[0].get("id")))
        self.assertTrue(bool(rows[0].get("in_event_log_keyword_hits", False)))

    def test_basic_search_can_return_event_log_vector_hits(self) -> None:
        service = self._build_service(
            vector_hits=[],
            event_vector_hits=[
                {
                    "id": "ev-1",
                    "memory_id": "m-el-vec",
                    "score": 0.89,
                    "source": "event_log_vector",
                    "in_event_log_vector_hits": True,
                }
            ],
        )
        service.repo.search_keyword = lambda **_: []
        service.repo.search_event_log_keyword = lambda **_: []
        rows = service._basic_search(
            policy=self._policy(keyword_enabled=False, vector_enabled=True),
            query="deadline approved by nora",
            user_id="u1",
            group_id="g1",
            top_k=3,
            candidate_episode_ids=None,
            attach_foresight=False,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual("m-el-vec", str(rows[0].get("id")))
        self.assertTrue(bool(rows[0].get("in_event_log_vector_hits", False)))

    def test_basic_search_can_return_foresight_keyword_hits(self) -> None:
        service = self._build_service(vector_hits=[], event_vector_hits=[], enable_vector=False)
        service.repo.search_keyword = lambda **_: []
        service.repo.search_event_log_keyword = lambda **_: []
        service.repo.search_foresight_keyword = lambda **_: [
            {
                "memory_id": "m-fs-kw",
                "score": 0.88,
                "source": "foresight_keyword",
                "in_foresight_keyword_hits": True,
                "foresight_text_hint": "next friday finalize payroll checklist",
            }
        ]
        rows = service._basic_search(
            policy=self._policy(keyword_enabled=True, vector_enabled=False),
            query="what is next friday checklist",
            user_id="u1",
            group_id="g1",
            top_k=3,
            candidate_episode_ids=None,
            attach_foresight=False,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual("m-fs-kw", str(rows[0].get("id")))
        self.assertTrue(bool(rows[0].get("in_foresight_keyword_hits", False)))


if __name__ == "__main__":
    unittest.main()

