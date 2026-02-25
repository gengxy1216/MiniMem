from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.infra.sqlite.memory_repository import MemoryRepository


class ForesightIntervalTests(unittest.TestCase):
    def _build_repo(self) -> MemoryRepository:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        return MemoryRepository(engine)

    def test_get_valid_foresights_respects_as_of_interval(self) -> None:
        repo = self._build_repo()
        episode = repo.save_message_as_memory(
            message_id="m1",
            create_time=100,
            sender="u1",
            content="明天去上海出差",
            user_id="u1",
            group_id="g1",
            group_name=None,
            sender_name=None,
            role="user",
            importance_score=0.8,
            storage_tier="text_only",
            summary="出差计划",
            subject="u1",
            atomic_facts=["计划去上海"],
            foresights=[
                {"content": "长期提醒", "start_time": None, "end_time": None, "confidence": 0.5},
                {"content": "Q1 计划", "start_time": 200, "end_time": 300, "confidence": 0.8},
                {"content": "过期事项", "start_time": 10, "end_time": 20, "confidence": 0.9},
            ],
            profile_patch={},
            event_id="e1",
        )
        eid = str(episode["id"])

        as_of_50 = repo.get_valid_foresights_for_episodes(
            episode_ids=[eid], user_id="u1", group_id="g1", as_of_ts=50
        )
        as_of_250 = repo.get_valid_foresights_for_episodes(
            episode_ids=[eid], user_id="u1", group_id="g1", as_of_ts=250
        )

        contents_50 = {str(x.get("content")) for x in as_of_50.get(eid, [])}
        contents_250 = {str(x.get("content")) for x in as_of_250.get(eid, [])}
        self.assertEqual({"长期提醒"}, contents_50)
        self.assertEqual({"长期提醒", "Q1 计划"}, contents_250)

    def test_invalid_interval_is_ignored(self) -> None:
        repo = self._build_repo()
        episode = repo.save_message_as_memory(
            message_id="m2",
            create_time=100,
            sender="u1",
            content="错误区间测试",
            user_id="u1",
            group_id="g1",
            group_name=None,
            sender_name=None,
            role="user",
            importance_score=0.8,
            storage_tier="text_only",
            summary="错误区间",
            subject="u1",
            atomic_facts=[],
            foresights=[
                {"content": "非法区间", "start_time": 500, "end_time": 300, "confidence": 0.7}
            ],
            profile_patch={},
            event_id="e2",
        )
        eid = str(episode["id"])
        result = repo.get_valid_foresights_for_episodes(
            episode_ids=[eid], user_id="u1", group_id="g1", as_of_ts=400
        )
        self.assertEqual([], result.get(eid, []))

    def test_get_valid_foresights_respects_overlap_window(self) -> None:
        repo = self._build_repo()
        episode = repo.save_message_as_memory(
            message_id="m3",
            create_time=100,
            sender="u1",
            content="区间过滤测试",
            user_id="u1",
            group_id="g1",
            group_name=None,
            sender_name=None,
            role="user",
            importance_score=0.8,
            storage_tier="text_only",
            summary="区间测试",
            subject="u1",
            atomic_facts=[],
            foresights=[
                {"content": "阶段一", "start_time": 100, "end_time": 160, "confidence": 0.8},
                {"content": "阶段二", "start_time": 170, "end_time": 240, "confidence": 0.8},
                {"content": "长期事项", "start_time": None, "end_time": None, "confidence": 0.6},
                {"content": "太晚事项", "start_time": 300, "end_time": 360, "confidence": 0.7},
            ],
            profile_patch={},
            event_id="e3",
        )
        eid = str(episode["id"])
        result = repo.get_valid_foresights_for_episodes(
            episode_ids=[eid],
            user_id="u1",
            group_id="g1",
            start_ts=150,
            end_ts=220,
        )
        contents = {str(x.get("content")) for x in result.get(eid, [])}
        self.assertEqual({"阶段一", "阶段二", "长期事项"}, contents)


if __name__ == "__main__":
    unittest.main()
