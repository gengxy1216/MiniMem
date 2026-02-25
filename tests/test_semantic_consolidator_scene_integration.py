from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.infra.sqlite.memory_repository import MemoryRepository
from evermemos_lite.service.semantic_consolidator import ConsolidateInput, SemanticConsolidator


class _StableEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.2, 0.4, 0.6]


class SemanticConsolidatorSceneIntegrationTests(unittest.TestCase):
    def _build_repo_and_consolidator(self) -> tuple[MemoryRepository, SemanticConsolidator]:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        repo = MemoryRepository(engine)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_StableEmbedding())
        return repo, consolidator

    @staticmethod
    def _save_episode(repo: MemoryRepository, *, ts: int, msg_id: str, summary: str) -> dict:
        return repo.save_message_as_memory(
            message_id=msg_id,
            create_time=ts,
            sender="u1",
            content=summary,
            user_id="u1",
            group_id="g1",
            group_name=None,
            sender_name=None,
            role="user",
            importance_score=0.8,
            storage_tier="text_only",
            summary=summary,
            subject="u1",
            atomic_facts=[],
            foresights=[],
            profile_patch={},
            event_id=f"evt-{msg_id}",
        )

    @staticmethod
    def _payload(*, memory_id: str, ts: int, summary: str, event_id: str) -> ConsolidateInput:
        return ConsolidateInput(
            user_id="u1",
            group_id="g1",
            event_id=event_id,
            memory_id=memory_id,
            timestamp=ts,
            episode=summary,
            summary=summary,
            importance_score=0.8,
            atomic_facts=[],
            profile_patch={},
            storage_tier="text_only",
        )

    def test_scene_memory_count_accumulates_and_is_reflected_in_profile_stats(self) -> None:
        repo, consolidator = self._build_repo_and_consolidator()
        ts0 = 1_730_000_000
        episode1 = self._save_episode(
            repo,
            ts=ts0,
            msg_id="m1",
            summary="我在上海工作，今天安排了项目计划。",
        )
        out1 = consolidator.consolidate(
            self._payload(
                memory_id=str(episode1["id"]),
                ts=ts0,
                summary="我在上海工作，今天安排了项目计划。",
                event_id="evt-c1",
            )
        )
        scene_id = str(out1.get("scene_id") or "")
        self.assertTrue(scene_id)
        scene1 = repo.get_memscene(scene_id)
        self.assertIsNotNone(scene1)
        self.assertEqual(1, int(scene1.get("memory_count") or 0))

        episode2 = self._save_episode(
            repo,
            ts=ts0 + 1200,
            msg_id="m2",
            summary="我在上海工作，继续推进这个项目计划。",
        )
        out2 = consolidator.consolidate(
            self._payload(
                memory_id=str(episode2["id"]),
                ts=ts0 + 1200,
                summary="我在上海工作，继续推进这个项目计划。",
                event_id="evt-c2",
            )
        )
        self.assertEqual(scene_id, str(out2.get("scene_id") or ""))
        scene2 = repo.get_memscene(scene_id)
        self.assertIsNotNone(scene2)
        self.assertEqual(2, int(scene2.get("memory_count") or 0))

        profile = repo.get_latest_profile_snapshot(user_id="u1", group_id="g1")
        self.assertIsNotNone(profile)
        stats = profile.get("profile", {}).get("scene_stats", {})
        self.assertEqual(2, int(stats.get("scene_memory_count") or 0))


if __name__ == "__main__":
    unittest.main()
