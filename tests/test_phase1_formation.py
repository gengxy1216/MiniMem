from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.domain.policy import EffectivePolicy
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.formation_enhancer import BoundaryDecision
from evermemos_lite.service.memory_service import ChatTurnInput, MemoryService


class _NoopVectorStore:
    enabled = False
    vector_dim = 4

    def search(self, **kwargs):
        return []

    def upsert(self, row_id: str, memory_id: str, vector: list[float], metadata: dict) -> None:
        return None


class _NoopEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.1, 0.1, 0.1]


class _NoopExtractor:
    def extract(self, content: str, sender: str, group_id: str | None):
        raise NotImplementedError


class _NoopGraphStore:
    enabled = False


class _StubFormationEnhancer:
    def __init__(
        self,
        *,
        boundary: BoundaryDecision | None = None,
        narrative: str | None = None,
    ) -> None:
        self.boundary = boundary
        self.narrative = narrative
        self.detect_calls = 0
        self.narrative_calls = 0

    def detect_boundary(
        self,
        *,
        query: str,
        recent_user_queries: list[str],
        turn_count: int,
        idle_seconds: int,
    ) -> BoundaryDecision | None:
        self.detect_calls += 1
        return self.boundary

    def synthesize_narrative(
        self, *, turns_markdown: str, user_id: str, group_id: str | None
    ) -> str | None:
        self.narrative_calls += 1
        return self.narrative


class PhaseOneFormationTests(unittest.TestCase):
    def _build_service(self, enhancer: _StubFormationEnhancer) -> MemoryService:
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
            formation_enhancer=enhancer,
            semantic_boundary_min_confidence=0.68,
        )

    @staticmethod
    def _policy(vector_enabled: bool = False) -> EffectivePolicy:
        return EffectivePolicy(
            vector_enabled=vector_enabled,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=20,
            vector_top_k=20,
            rrf_k=60,
            profile="keyword",
            reason="test",
        )

    def test_append_chat_turn_uses_high_confidence_semantic_boundary(self) -> None:
        enhancer = _StubFormationEnhancer(
            boundary=BoundaryDecision(
                should_cut=True,
                confidence=0.91,
                reason="topic_shift",
            )
        )
        service = self._build_service(enhancer)
        service.repo.upsert_conversation_segment(
            conversation_id="conv-1",
            user_id="u1",
            group_id="g1",
            segment_seq=1,
            turns_markdown="**User**: 先聊旅游\n\n**Assistant**: 好的",
            last_query="先聊旅游",
            turn_count=2,
            start_time=1700000000,
            last_time=1700000010,
        )

        commit_calls = {"count": 0}

        def _fake_commit(**kwargs) -> bool:
            commit_calls["count"] += 1
            return True

        service._commit_segment_as_memory = _fake_commit
        result = service.append_chat_turn(
            payload=ChatTurnInput(
                conversation_id="conv-1",
                user_id="u1",
                group_id="g1",
                user_text="换个话题，我想聊工作计划",
                assistant_text="好的，我们聊工作计划。",
                timestamp=1700000020,
            ),
            policy=self._policy(),
        )

        self.assertTrue(bool(result.get("boundary_detected")))
        self.assertEqual(1, commit_calls["count"])
        self.assertTrue(str(result.get("boundary_reason", "")).startswith("semantic:"))
        self.assertEqual(1, enhancer.detect_calls)

    def test_low_confidence_semantic_boundary_falls_back_to_rule(self) -> None:
        enhancer = _StubFormationEnhancer(
            boundary=BoundaryDecision(
                should_cut=True,
                confidence=0.22,
                reason="weak_shift",
            )
        )
        service = self._build_service(enhancer)
        service.repo.upsert_conversation_segment(
            conversation_id="conv-2",
            user_id="u1",
            group_id="g1",
            segment_seq=1,
            turns_markdown="**User**: 先聊电影\n\n**Assistant**: 好的",
            last_query="先聊电影",
            turn_count=2,
            start_time=1700000000,
            last_time=1700000010,
        )

        service._should_cut_segment = lambda **kwargs: False
        result = service.append_chat_turn(
            payload=ChatTurnInput(
                conversation_id="conv-2",
                user_id="u1",
                group_id="g1",
                user_text="我再补充一下刚才电影内容",
                assistant_text="继续说。",
                timestamp=1700000020,
            ),
            policy=self._policy(),
        )

        self.assertFalse(bool(result.get("boundary_detected")))
        self.assertTrue(str(result.get("boundary_reason", "")).startswith("semantic_no_cut:"))

    def test_commit_segment_includes_narrative_when_available(self) -> None:
        enhancer = _StubFormationEnhancer(
            narrative="用户以第三人称描述了近期旅行和工作安排。"
        )
        service = self._build_service(enhancer)

        captured: dict[str, str] = {}

        def _fake_memorize(payload, request_id: str):
            captured["content"] = str(payload.content)
            return {
                "memory": {
                    "id": "m1",
                    "importance_score": 0.9,
                    "storage_tier": "text_only",
                }
            }

        service.memorize = _fake_memorize
        ok = service._commit_segment_as_memory(
            conversation_id="conv-3",
            segment={
                "turns_markdown": "### Turn @ 2026-02-25 10:00:00\n**User**: 我明天去上海\n\n**Assistant**: 好的",
                "segment_seq": 3,
                "last_time": 1700001000,
            },
            user_id="u1",
            group_id="g1",
            policy=self._policy(vector_enabled=False),
        )

        self.assertTrue(ok)
        content = captured.get("content", "")
        self.assertIn("### Narrative", content)
        self.assertIn("第三人称", content)
        self.assertIn("### Conversation", content)
        self.assertEqual(1, enhancer.narrative_calls)


if __name__ == "__main__":
    unittest.main()
