from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.domain.policy import EffectivePolicy
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.memory_service import MemoryService
from evermemos_lite.service.query_rewriter import RewriteDecision
from evermemos_lite.service.retrieval_verifier import SufficiencyDecision


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


class _StubVerifier:
    def __init__(self, decision: SufficiencyDecision | None) -> None:
        self.decision = decision
        self.calls = 0

    def judge_sufficiency(
        self, *, query: str, hits: list[dict], top_k: int
    ) -> SufficiencyDecision | None:
        self.calls += 1
        return self.decision


class _StubRewriter:
    def __init__(self, decision: RewriteDecision | None) -> None:
        self.decision = decision
        self.calls = 0

    def rewrite(self, *, query: str, hits: list[dict], insufficiency_reason: str):
        self.calls += 1
        return self.decision


class AgenticVerifierTests(unittest.TestCase):
    def _build_service(
        self,
        verifier: _StubVerifier,
        min_confidence: float = 0.66,
        rewriter: _StubRewriter | None = None,
        rewrite_min_conf: float = 0.62,
    ) -> MemoryService:
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
            retrieval_verifier=verifier,
            retrieval_verifier_min_confidence=min_confidence,
            query_rewriter=rewriter,
            query_rewriter_min_confidence=rewrite_min_conf,
            phase4_reasoning_enabled=False,
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

    def test_llm_verifier_insufficient_forces_second_round(self) -> None:
        verifier = _StubVerifier(
            SufficiencyDecision(sufficient=False, confidence=0.92, reason="insufficient")
        )
        service = self._build_service(verifier)
        calls = {"count": 0}

        def _fake_basic_search(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return [{"id": "m1", "score": 0.45, "source": "keyword", "summary": "s1"}]
            return [{"id": "m2", "score": 0.44, "source": "keyword", "summary": "s2"}]

        service._basic_search = _fake_basic_search
        service._scene_guided_candidate_ids = lambda *args, **kwargs: {
            "episode_ids": ["m2"],
            "episode_scene_map": {"m2": "scene-2"},
            "scene_score_map": {"scene-2": 0.8},
        }
        service._attach_valid_foresight = (
            lambda rows, user_id, group_id, as_of_ts=None, **kwargs: rows
        )

        rows = service._agentic_search(
            policy=self._policy(),
            query="我上周和谁开会以及结论是什么？",
            user_id="u1",
            group_id="g1",
            top_k=3,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(2, calls["count"])
        self.assertEqual(1, verifier.calls)

    def test_llm_verifier_sufficient_skips_second_round(self) -> None:
        verifier = _StubVerifier(
            SufficiencyDecision(sufficient=True, confidence=0.9, reason="enough")
        )
        service = self._build_service(verifier)
        calls = {"count": 0}

        def _fake_basic_search(*args, **kwargs):
            calls["count"] += 1
            return [{"id": "m1", "score": 0.01, "source": "keyword", "summary": "s1"}]

        service._basic_search = _fake_basic_search
        service._attach_valid_foresight = (
            lambda rows, user_id, group_id, as_of_ts=None, **kwargs: rows
        )

        rows = service._agentic_search(
            policy=self._policy(),
            query="复杂推理问题",
            user_id="u1",
            group_id="g1",
            top_k=5,
        )
        self.assertEqual(1, calls["count"])
        self.assertEqual(1, len(rows))
        self.assertEqual(1, verifier.calls)

    def test_low_confidence_llm_decision_falls_back_to_heuristic(self) -> None:
        verifier = _StubVerifier(
            SufficiencyDecision(sufficient=False, confidence=0.2, reason="weak")
        )
        service = self._build_service(verifier, min_confidence=0.66)
        calls = {"count": 0}

        def _fake_basic_search(*args, **kwargs):
            calls["count"] += 1
            return [{"id": "m1", "score": 0.6, "source": "keyword", "summary": "s1"}]

        service._basic_search = _fake_basic_search
        service._attach_valid_foresight = (
            lambda rows, user_id, group_id, as_of_ts=None, **kwargs: rows
        )

        rows = service._agentic_search(
            policy=self._policy(),
            query="回顾本周任务进展",
            user_id="u1",
            group_id="g1",
            top_k=3,
        )
        self.assertEqual(1, calls["count"])
        self.assertEqual(1, len(rows))
        self.assertEqual(1, verifier.calls)

    def test_confident_llm_rewrite_is_used_in_second_round(self) -> None:
        verifier = _StubVerifier(
            SufficiencyDecision(sufficient=False, confidence=0.9, reason="insufficient")
        )
        rewriter = _StubRewriter(
            RewriteDecision(
                query="原问题 加上时间 人物 地点",
                confidence=0.9,
                reason="expand_entities",
            )
        )
        service = self._build_service(verifier, rewriter=rewriter)
        captured: list[str] = []

        def _fake_basic_search(*args, **kwargs):
            query = str(args[1] if len(args) > 1 else kwargs.get("query", ""))
            captured.append(query)
            if len(captured) == 1:
                return [{"id": "m1", "score": 0.21, "source": "keyword", "summary": "s1"}]
            return [{"id": "m2", "score": 0.41, "source": "keyword", "summary": "s2"}]

        service._basic_search = _fake_basic_search
        service._scene_guided_candidate_ids = lambda *args, **kwargs: {
            "episode_ids": ["m2"],
            "episode_scene_map": {"m2": "scene-2"},
            "scene_score_map": {"scene-2": 0.9},
        }
        service._attach_valid_foresight = (
            lambda rows, user_id, group_id, as_of_ts=None, **kwargs: rows
        )

        rows = service._agentic_search(
            policy=self._policy(),
            query="原问题",
            user_id="u1",
            group_id="g1",
            top_k=3,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertGreaterEqual(len(captured), 2)
        self.assertEqual("原问题 加上时间 人物 地点", captured[1])
        self.assertEqual(1, rewriter.calls)


if __name__ == "__main__":
    unittest.main()
