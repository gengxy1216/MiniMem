from __future__ import annotations

import unittest
from pathlib import Path
from flockmem.testing.writable_tempdir import WritableTempDir

from flockmem.domain.policy import EffectivePolicy
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.service.memory_service import MemoryService, RetrievalSufficiency


class _StubVectorStore:
    def __init__(self) -> None:
        self.enabled = True
        self.vector_dim = 4
        self.last_top_k: int | None = None
        self.last_vector_len: int | None = None
        self.search_calls = 0
        self.upsert_calls = 0
        self.hits: list[dict] = [
            {
                "id": "v-row",
                "memory_id": "m-1",
                "score": 0.72,
                "source": "vector",
            }
        ]

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict]:
        self.search_calls += 1
        self.last_top_k = int(top_k)
        self.last_vector_len = len(vector)
        return list(self.hits)

    def upsert(
        self,
        row_id: str,
        memory_id: str,
        vector: list[float],
        metadata: dict[str, object],
    ) -> None:
        self.upsert_calls += 1


class _StubEmbeddingProvider:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.calls = 0
        self.vector = list(vector) if vector else [0.9, 0.1, 0.0, 0.0]

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return list(self.vector)


class _FailingEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("mock embedding failure")


class _StubExtractor:
    def extract(self, *, content: str, sender: str, group_id: str | None):
        raise NotImplementedError


class _StubGraphStore:
    enabled = False


class MemoryServiceSemanticLatencyTests(unittest.TestCase):
    def _build_service(
        self,
        *,
        semantic_vector_budget_cap: int = 18,
        semantic_keyword_budget_cap: int = 12,
        embedding_vector: list[float] | None = None,
    ) -> tuple[MemoryService, _StubVectorStore, _StubEmbeddingProvider]:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        vector_store = _StubVectorStore()
        embedding = _StubEmbeddingProvider(vector=embedding_vector)
        service = MemoryService(
            engine=engine,
            vector_store=vector_store,
            embedding_provider=embedding,
            extractor=_StubExtractor(),
            graph_store=_StubGraphStore(),
            search_budget_factor=4,
            search_min_probe_k=12,
            keyword_confident_best_score=9.0,
            keyword_confident_kth_score=2.8,
            semantic_vector_budget_cap=semantic_vector_budget_cap,
            semantic_keyword_budget_cap=semantic_keyword_budget_cap,
            query_embed_cache_size=64,
            query_embed_cache_ttl_sec=3600,
        )
        service.repo.search_keyword = lambda **_: []
        service.repo.fetch_episodes = lambda **_: [
            {
                "id": "m-1",
                "event_id": "e-1",
                "timestamp": 1700000000,
                "summary": "semantic memory",
                "subject": "user",
                "episode": "incident concept and workflow details",
                "importance_score": 0.7,
                "storage_tier": "vector_only",
            }
        ]
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": str(mid),
                "event_id": f"e-{mid}",
                "timestamp": 1700000000,
                "summary": "semantic memory",
                "subject": "user",
                "episode": "incident concept and workflow details",
                "importance_score": 0.7,
                "storage_tier": "vector_only",
            }
            for mid in episode_ids
        ]
        service.repo.get_valid_foresights_for_episodes = lambda **_: {}
        return service, vector_store, embedding

    def test_semantic_query_caps_vector_budget(self) -> None:
        service, vector_store, _ = self._build_service(semantic_vector_budget_cap=18)
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        service._basic_search(
            policy=policy,
            query="how to handle semantic workflow regression",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
        )
        self.assertEqual(18, vector_store.last_top_k)

    def test_basic_search_fits_query_vector_before_probe(self) -> None:
        service, vector_store, _ = self._build_service(
            embedding_vector=[0.9, 0.1, 0.0, 0.0, 0.2, 0.3]
        )
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        service._basic_search(
            policy=policy,
            query="how to handle semantic workflow regression",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
        )
        self.assertEqual(4, vector_store.last_vector_len)

    def test_id_like_confident_keyword_skips_vector_probe(self) -> None:
        service, vector_store, embedding = self._build_service()
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        service.repo.search_keyword = lambda **_: [
            {"memory_id": "m-1", "score": 15.0, "source": "keyword"},
            {"memory_id": "m-2", "score": 6.0, "source": "keyword"},
            {"memory_id": "m-3", "score": 5.8, "source": "keyword"},
            {"memory_id": "m-4", "score": 4.2, "source": "keyword"},
            {"memory_id": "m-5", "score": 3.0, "source": "keyword"},
        ]
        service.repo.fetch_episodes = lambda **_: [
            {
                "id": f"m-{idx}",
                "event_id": f"e-{idx}",
                "timestamp": 1700000000 + idx,
                "summary": f"memory-{idx}",
                "subject": "user",
                "episode": f"ticket details m-{idx}",
                "importance_score": 0.7,
                "storage_tier": "text_only",
            }
            for idx in range(1, 6)
        ]
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": str(mid),
                "event_id": f"e-{mid}",
                "timestamp": 1700000000,
                "summary": f"memory-{mid}",
                "subject": "user",
                "episode": f"ticket details {mid}",
                "importance_score": 0.7,
                "storage_tier": "text_only",
            }
            for mid in episode_ids
        ]
        service._basic_search(
            policy=policy,
            query="ticket tck-00123 detail",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
        )
        self.assertEqual(0, vector_store.search_calls)
        self.assertEqual(0, embedding.calls)

    def test_query_embedding_cache_reuses_same_query(self) -> None:
        service, _, embedding = self._build_service()
        first = service._embed_query("semantic query A")
        second = service._embed_query("semantic query A")
        self.assertEqual(first, second)
        self.assertEqual(1, embedding.calls)

    def test_agentic_skip_second_round_for_semantic_hits(self) -> None:
        service, _, _ = self._build_service()
        skip = service._should_skip_agentic_second_round(
            query="semantic relation question",
            hits=[
                {"score": 0.66, "source": "hybrid_rrf", "in_keyword_hits": True, "in_vector_hits": True},
                {"score": 0.52, "source": "hybrid_rrf", "in_keyword_hits": True, "in_vector_hits": False},
            ],
            top_k=5,
        )
        self.assertTrue(skip)

    def test_agentic_never_skips_when_vector_probe_errors(self) -> None:
        service, _, _ = self._build_service()
        skip = service._should_skip_agentic_second_round(
            query="semantic relation question",
            hits=[{"score": 0.8, "source": "keyword"}],
            top_k=5,
            vector_probe_error=True,
        )
        self.assertFalse(skip)

    def test_agentic_does_not_skip_complex_temporal_query_with_weak_hits(self) -> None:
        service, _, _ = self._build_service()
        skip = service._should_skip_agentic_second_round(
            query="请按时间线回顾去年项目进展然后说明风险",
            hits=[
                {"score": 0.14, "source": "keyword"},
                {"score": 0.13, "source": "keyword"},
            ],
            top_k=5,
        )
        self.assertFalse(skip)

    def test_agentic_heuristic_sufficient_still_runs_second_round_for_precision_query(self) -> None:
        service, _, _ = self._build_service()
        sufficiency = RetrievalSufficiency(
            sufficient=True,
            source="heuristic",
            confidence=1.0,
            reason="heuristic",
        )
        skip = service._should_skip_agentic_second_round(
            query="When did customer alpha submit the escalation ticket?",
            hits=[
                {"score": 0.50, "source": "keyword", "in_keyword_hits": True, "in_vector_hits": False},
                {"score": 0.44, "source": "keyword", "in_keyword_hits": True, "in_vector_hits": False},
            ],
            top_k=5,
            sufficiency=sufficiency,
        )
        self.assertFalse(skip)

    def test_agentic_llm_sufficient_complex_query_still_runs_second_round_when_scores_not_strong(self) -> None:
        service, _, _ = self._build_service()
        sufficiency = RetrievalSufficiency(
            sufficient=True,
            source="llm_verifier",
            confidence=0.93,
            reason="llm_sufficient",
        )
        skip = service._should_skip_agentic_second_round(
            query="When did project alpha slip, and what changed after that?",
            hits=[
                {"score": 0.66, "source": "keyword", "in_keyword_hits": True, "in_vector_hits": False},
                {"score": 0.58, "source": "vector", "in_keyword_hits": False, "in_vector_hits": True},
            ],
            top_k=5,
            sufficiency=sufficiency,
        )
        self.assertFalse(skip)

    def test_agentic_llm_sufficient_complex_query_can_skip_when_dual_source_and_scores_strong(self) -> None:
        service, _, _ = self._build_service()
        sufficiency = RetrievalSufficiency(
            sufficient=True,
            source="llm_verifier",
            confidence=0.97,
            reason="llm_sufficient",
        )
        skip = service._should_skip_agentic_second_round(
            query="When did project alpha slip, and what changed after that?",
            hits=[
                {"score": 0.80, "source": "hybrid_rrf", "in_keyword_hits": True, "in_vector_hits": True},
                {"score": 0.61, "source": "vector", "in_keyword_hits": False, "in_vector_hits": True},
                {"score": 0.54, "source": "keyword", "in_keyword_hits": True, "in_vector_hits": False},
            ],
            top_k=3,
            sufficiency=sufficiency,
        )
        self.assertTrue(skip)

    def test_expand_agentic_round_top_k_uses_deeper_candidate_budget(self) -> None:
        service, _, _ = self._build_service()
        self.assertEqual(24, service._expand_agentic_round_top_k(5))
        self.assertEqual(90, service._expand_agentic_round_top_k(30))

    def test_basic_search_can_defer_foresight_attachment(self) -> None:
        service, _, _ = self._build_service()
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        calls = {"count": 0}

        def _count_foresight(**kwargs):
            calls["count"] += 1
            return {}

        service.repo.get_valid_foresights_for_episodes = _count_foresight
        service._basic_search(
            policy=policy,
            query="semantic workflow question",
            user_id="u1",
            group_id="g1",
            top_k=5,
            candidate_episode_ids=None,
            attach_foresight=False,
        )
        self.assertEqual(0, calls["count"])

    def test_agentic_runs_second_round_when_no_scene_and_heuristic_not_confident(self) -> None:
        service, _, _ = self._build_service()
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=True,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        calls = {"count": 0}

        def _fake_basic_search(
            policy,
            query,
            user_id,
            group_id,
            top_k,
            candidate_episode_ids,
            attach_foresight=True,
            as_of_ts=None,
            start_ts=None,
            end_ts=None,
        ):
            calls["count"] += 1
            return []

        service._basic_search = _fake_basic_search
        service._scene_guided_candidate_ids = lambda *args, **kwargs: {
            "episode_ids": [],
            "episode_scene_map": {},
            "scene_score_map": {},
        }
        rows = service._agentic_search(
            policy=policy,
            query="semantic noop query",
            user_id="u1",
            group_id="g1",
            top_k=5,
        )
        self.assertGreaterEqual(calls["count"], 2)
        self.assertGreaterEqual(len(rows), 1)

    def test_semantic_query_prefers_vector_priority_merge(self) -> None:
        service, vector_store, _ = self._build_service()
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        vector_store.hits = [
            {"id": "v1", "memory_id": "m-1", "score": 0.75, "source": "vector"},
            {"id": "v2", "memory_id": "m-9", "score": 0.62, "source": "vector"},
        ]
        service.repo.search_keyword = lambda **_: [
            {"memory_id": "m-2", "score": 0.12, "source": "keyword"},
            {"memory_id": "m-3", "score": 0.09, "source": "keyword"},
            {"memory_id": "m-1", "score": 0.08, "source": "keyword"},
        ]
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": str(mid),
                "event_id": f"e-{mid}",
                "timestamp": 1700000000,
                "summary": f"memory-{mid}",
                "subject": "user",
                "episode": f"semantic details {mid}",
                "importance_score": 0.7,
                "storage_tier": "vector_only",
            }
            for mid in episode_ids
        ]
        rows = service._basic_search(
            policy=policy,
            query="debug semantic workflow under customer alpha operations",
            user_id="u1",
            group_id="g1",
            top_k=3,
            candidate_episode_ids=None,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual("m-1", str(rows[0].get("id")))
        self.assertEqual("semantic_vector_priority", str(rows[0].get("source")))

    def test_temporal_query_does_not_prefer_vector_priority_merge(self) -> None:
        service, vector_store, _ = self._build_service()
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=50,
            vector_top_k=50,
            rrf_k=80,
            profile="agentic",
            reason="test",
        )
        vector_store.hits = [
            {"id": "v1", "memory_id": "m-1", "score": 0.78, "source": "vector"},
            {"id": "v2", "memory_id": "m-9", "score": 0.64, "source": "vector"},
        ]
        service.repo.search_keyword = lambda **_: [
            {"memory_id": "m-2", "score": 0.12, "source": "keyword"},
            {"memory_id": "m-3", "score": 0.09, "source": "keyword"},
            {"memory_id": "m-1", "score": 0.08, "source": "keyword"},
        ]
        service.repo.fetch_episodes_by_ids = lambda episode_ids: [
            {
                "id": str(mid),
                "event_id": f"e-{mid}",
                "timestamp": 1700000000,
                "summary": f"memory-{mid}",
                "subject": "user",
                "episode": f"semantic details {mid}",
                "importance_score": 0.7,
                "storage_tier": "vector_only",
            }
            for mid in episode_ids
        ]
        rows = service._basic_search(
            policy=policy,
            query="When did customer alpha escalate the incident?",
            user_id="u1",
            group_id="g1",
            top_k=3,
            candidate_episode_ids=None,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertNotEqual("semantic_vector_priority", str(rows[0].get("source")))

    def test_maybe_index_vector_returns_error_when_embedding_fails(self) -> None:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        vector_store = _StubVectorStore()
        service = MemoryService(
            engine=engine,
            vector_store=vector_store,
            embedding_provider=_FailingEmbeddingProvider(),
            extractor=_StubExtractor(),
            graph_store=_StubGraphStore(),
            vector_write_min_importance=0.1,
        )
        policy = EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=20,
            vector_top_k=20,
            rrf_k=60,
            profile="hybrid",
            reason="test",
        )
        result = service.maybe_index_vector(
            policy,
            {
                "id": "m-1",
                "episode": "test memory content",
                "user_id": "u1",
                "group_id": "g1",
                "timestamp": 1700000000,
                "importance_score": 0.9,
                "storage_tier": "vector_only",
            },
        )
        self.assertEqual("error", str(result.get("status")))
        self.assertEqual("embed_failed", str(result.get("reason")))
        self.assertEqual(0, vector_store.upsert_calls)

    def test_split_embedding_chunks_splits_long_text(self) -> None:
        service, _, _ = self._build_service()
        service.vector_embed_chunk_chars = 50
        service.vector_embed_max_chunks = 8
        text = "这是一段很长的文本，用来验证向量分块机制是否生效。" * 20
        chunks = service._split_embedding_chunks(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

