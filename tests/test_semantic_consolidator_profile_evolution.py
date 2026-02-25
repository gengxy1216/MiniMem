from __future__ import annotations

import unittest

from evermemos_lite.service.semantic_consolidator import ConsolidateInput, SemanticConsolidator


class _StubRepo:
    def __init__(self, initial_profile: dict | None = None, initial_ts: int = 0) -> None:
        self.profile = initial_profile or {}
        self.profile_ts = int(initial_ts)
        self.conflicts: list[dict] = []

    def get_latest_profile_snapshot(self, user_id: str | None, group_id: str | None):
        if not self.profile:
            return None
        return {
            "event_id": "prev",
            "user_id": user_id or "u1",
            "group_id": group_id,
            "timestamp": self.profile_ts,
            "profile": self.profile,
        }

    def upsert_profile_snapshot(
        self,
        *,
        event_id: str,
        user_id: str,
        group_id: str | None,
        profile_patch: dict,
        timestamp: int,
    ) -> None:
        self.profile = profile_patch
        self.profile_ts = int(timestamp)

    def insert_profile_conflict(self, **kwargs) -> None:
        self.conflicts.append(dict(kwargs))


class _NoopEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class SemanticConsolidatorProfileEvolutionTests(unittest.TestCase):
    def _payload(self, *, ts: int, summary: str, patch: dict | None = None) -> ConsolidateInput:
        return ConsolidateInput(
            user_id="u1",
            group_id="g1",
            event_id=f"e-{ts}",
            memory_id=f"m-{ts}",
            timestamp=ts,
            episode=summary,
            summary=summary,
            importance_score=0.8,
            atomic_facts=[],
            profile_patch=patch or {},
            storage_tier="text_only",
        )

    def test_recent_update_can_replace_stale_conflicting_fact(self) -> None:
        initial = {
            "explicit_facts": {
                "name": {
                    "value": "Alice",
                    "timestamp": 1_700_000_000,
                    "confidence": 0.9,
                    "decay_score": 0.92,
                    "support_count": 1,
                }
            }
        }
        repo = _StubRepo(initial_profile=initial, initial_ts=1_700_000_000)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_NoopEmbedding())
        payload = self._payload(
            ts=1_700_000_000 + 120 * 86400,
            summary="我叫Bob，最近换了工作城市。",
        )
        consolidator._evolve_profile(
            payload=payload, scene_id="scene-1", scene_summary=payload.summary, memory_count_delta=1
        )
        name_obj = repo.profile.get("explicit_facts", {}).get("name", {})
        self.assertEqual("Bob", str(name_obj.get("value")))
        self.assertEqual("Alice", str(name_obj.get("previous_value")))
        self.assertGreaterEqual(len(repo.conflicts), 1)

    def test_recent_supported_fact_can_resist_weaker_new_conflict(self) -> None:
        now = 1_710_000_000
        initial = {
            "explicit_facts": {
                "city": {
                    "value": "北京",
                    "timestamp": now - 86400,
                    "confidence": 0.95,
                    "decay_score": 0.96,
                    "support_count": 6,
                }
            }
        }
        repo = _StubRepo(initial_profile=initial, initial_ts=now - 86400)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_NoopEmbedding())
        payload = self._payload(ts=now, summary="我现在住在上海工作，最近也有出差安排。")
        consolidator._evolve_profile(
            payload=payload, scene_id="scene-2", scene_summary=payload.summary, memory_count_delta=1
        )
        city_obj = repo.profile.get("explicit_facts", {}).get("city", {})
        self.assertEqual("北京", str(city_obj.get("value")))
        self.assertGreaterEqual(len(repo.conflicts), 1)
        self.assertIn("scene_profile_links", repo.profile)
        links = repo.profile.get("scene_profile_links", [])
        self.assertTrue(any(str(x.get("scene_id")) == "scene-2" for x in links))

    def test_profile_patch_has_higher_priority_confidence(self) -> None:
        repo = _StubRepo(initial_profile={}, initial_ts=0)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_NoopEmbedding())
        payload = self._payload(
            ts=1_720_000_000,
            summary="今天聊了工作安排。",
            patch={"name": "Carol"},
        )
        consolidator._evolve_profile(
            payload=payload, scene_id="scene-3", scene_summary=payload.summary, memory_count_delta=1
        )
        name_obj = repo.profile.get("explicit_facts", {}).get("name", {})
        self.assertEqual("Carol", str(name_obj.get("value")))
        self.assertGreaterEqual(float(name_obj.get("confidence", 0.0)), 0.85)

    def test_scene_memory_count_boosts_support_and_decay(self) -> None:
        repo = _StubRepo(initial_profile={}, initial_ts=0)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_NoopEmbedding())
        payload = self._payload(
            ts=1_725_000_000,
            summary="我叫Dylan，今天也复盘了当前项目状态。",
        )
        consolidator._evolve_profile(
            payload=payload,
            scene_id="scene-strong",
            scene_summary=payload.summary,
            memory_count_delta=1,
            scene_memory_count=6,
        )
        name_obj = repo.profile.get("explicit_facts", {}).get("name", {})
        self.assertEqual("Dylan", str(name_obj.get("value")))
        self.assertGreaterEqual(int(name_obj.get("support_count", 0)), 4)
        self.assertGreaterEqual(float(name_obj.get("decay_score", 0.0)), 0.85)
        stats = repo.profile.get("scene_stats", {})
        self.assertEqual(6, int(stats.get("scene_memory_count", 0)))
        links = repo.profile.get("scene_profile_links", [])
        self.assertEqual(6, int(links[-1].get("scene_memory_count", 0)))

    def test_strong_scene_can_flip_close_conflict(self) -> None:
        now = 1_726_000_000
        initial = {
            "explicit_facts": {
                "city": {
                    "value": "北京",
                    "timestamp": now - 3600,
                    "confidence": 0.76,
                    "decay_score": 0.76,
                    "support_count": 1,
                }
            }
        }
        repo = _StubRepo(initial_profile=initial, initial_ts=now - 3600)
        consolidator = SemanticConsolidator(repo=repo, embedding_provider=_NoopEmbedding())
        payload = self._payload(ts=now, summary="我在上海工作，最近任务节奏比较稳定。")
        consolidator._evolve_profile(
            payload=payload,
            scene_id="scene-city",
            scene_summary=payload.summary,
            memory_count_delta=1,
            scene_memory_count=6,
        )
        city_obj = repo.profile.get("explicit_facts", {}).get("city", {})
        self.assertIn("上海", str(city_obj.get("value")))
        self.assertEqual("北京", str(city_obj.get("previous_value")))


if __name__ == "__main__":
    unittest.main()
