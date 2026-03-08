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


class _NoopGraphStore:
    enabled = False


class RetrievalExtractChainTests(unittest.TestCase):
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
            keyword_top_k=20,
            vector_top_k=20,
            rrf_k=60,
            profile="default",
            reason="test_extract_chain",
        )

    def test_compact_rows_runs_extract_and_keeps_query_window(self) -> None:
        service = self._build_service()
        long_episode = (
            "[metadata] {\"source\":\"raw\"}\n"
            + "背景信息 " * 180
            + "关键问题：预算超支，需要降本方案。"
            + " 收尾信息" * 60
        )
        rows = [
            {
                "id": "m1",
                "summary": "",
                "episode": long_episode,
                "subject": "u1",
                "score": 0.8,
            }
        ]
        compacted = service._compress_retrieval_rows(query="预算超支", rows=rows)
        self.assertEqual(1, len(compacted))
        row = compacted[0]
        self.assertTrue(str(row.get("summary", "")).strip())
        self.assertLessEqual(len(str(row.get("episode", ""))), 420)
        self.assertIn("预算超支", str(row.get("episode", "")))

    def test_search_returns_compacted_episode_not_full_raw_text(self) -> None:
        service = self._build_service()
        policy = self._policy_keyword_only()
        long_episode = (
            "常规背景 " * 220
            + "本次检索目标：预算超支修复计划。"
            + " 结束语" * 40
        )
        service.repo.save_message_as_memory(
            message_id="m-extract-1",
            create_time=1730000000,
            sender="u1",
            content=long_episode,
            user_id="u1",
            group_id="g1",
            group_name=None,
            sender_name="u1",
            role="user",
            importance_score=0.8,
            storage_tier="text_only",
            summary="",
            subject="project",
            atomic_facts=[],
            foresights=[],
            profile_patch={},
            memory_category="event",
        )
        rows = service.search(
            policy=policy,
            query="预算超支修复计划",
            user_id="u1",
            group_id="g1",
            top_k=1,
        )
        self.assertEqual(1, len(rows))
        episode = str(rows[0].get("episode", ""))
        self.assertLessEqual(len(episode), 420)
        self.assertIn("预算超支", episode)


if __name__ == "__main__":
    unittest.main()

