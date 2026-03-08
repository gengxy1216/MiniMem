from __future__ import annotations

import re
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


class _NoopGraphStore:
    enabled = False


class _StubRerankProvider:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        return [0.2, 0.9, 0.4][: len(documents)]


class MemoryServiceAlignmentGuardTests(unittest.TestCase):
    def _build_service(
        self,
        *,
        recall_mode: bool = True,
        rerank_provider: _StubRerankProvider | None = None,
        rerank_trigger_k: int = 12,
    ) -> MemoryService:
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
            recall_mode=recall_mode,
            rerank_provider=rerank_provider,
            rerank_trigger_k=rerank_trigger_k,
            rerank_top_n=16,
            rerank_timeout_ms=4000,
        )
        return service

    def test_profile_hints_are_blocked_for_non_profile_third_person_query(self) -> None:
        service = self._build_service()
        service.repo.upsert_profile_snapshot(
            event_id="evt-1",
            user_id="u1",
            group_id="g1",
            profile_patch={
                "explicit_facts": {
                    "name": {"value": "Caroline", "timestamp": 1700000000},
                    "identity": {"value": "transgender woman", "timestamp": 1700000000},
                }
            },
            timestamp=1700000000,
        )
        rows = service._profile_hint_hits(
            query="What is Caroline's identity?",
            user_id=None,
            group_id="g1",
        )
        self.assertEqual([], rows)

    def test_profile_hints_keep_working_for_self_identity_query(self) -> None:
        service = self._build_service()
        service.repo.upsert_profile_snapshot(
            event_id="evt-1",
            user_id="u1",
            group_id="g1",
            profile_patch={
                "explicit_facts": {
                    "name": {"value": "Caroline", "timestamp": 1700000000},
                }
            },
            timestamp=1700000000,
        )
        rows = service._profile_hint_hits(query="who am i", user_id="u1", group_id="g1")
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(str(x.get("source")) == "profile_snapshot" for x in rows))

    def test_english_temporal_expansion_does_not_inject_cjk_tokens(self) -> None:
        service = self._build_service()
        extras = service._expand_temporal_boundary_queries(
            seed="When did Melanie paint a sunrise",
            original_query="When did Melanie paint a sunrise?",
            insufficiency_reason="need timeline and person details",
        )
        self.assertGreaterEqual(len(extras), 1)
        self.assertTrue(all(re.search(r"[\u4e00-\u9fff]", x) is None for x in extras))

    def test_recall_mode_applies_rerank_even_when_rows_below_default_trigger(self) -> None:
        reranker = _StubRerankProvider()
        service = self._build_service(
            recall_mode=True,
            rerank_provider=reranker,
            rerank_trigger_k=12,
        )
        rows = [
            {"id": "m1", "summary": "a", "episode": "a", "subject": "u", "score": 0.7},
            {"id": "m2", "summary": "b", "episode": "b", "subject": "u", "score": 0.6},
            {"id": "m3", "summary": "c", "episode": "c", "subject": "u", "score": 0.5},
        ]
        out = service._apply_model_rerank(query="test query", rows=rows)
        self.assertEqual(1, reranker.calls)
        self.assertEqual("m2", str(out[0].get("id")))
        self.assertTrue(bool(out[0].get("rerank_applied")))


if __name__ == "__main__":
    unittest.main()
