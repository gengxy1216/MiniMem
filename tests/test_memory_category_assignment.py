from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.extractor import ExtractedMemory
from evermemos_lite.service.memory_service import MemorizeInput, MemoryService


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


class _NoopGraphStore:
    enabled = False


class _QueueExtractor:
    def __init__(self, items: list[ExtractedMemory]) -> None:
        self._items = list(items)

    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        if not self._items:
            raise RuntimeError("no extracted memory queued")
        return self._items.pop(0)


class MemoryCategoryAssignmentTests(unittest.TestCase):
    def _build_service(self, extracted_items: list[ExtractedMemory]) -> tuple[MemoryService, TemporaryDirectory]:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_QueueExtractor(extracted_items),
            graph_store=_NoopGraphStore(),
        )
        return service, tmp

    def test_memory_category_prefers_profile_when_profile_patch_present(self) -> None:
        service, tmp = self._build_service(
            [
                ExtractedMemory(
                    episode="我叫小明，住在上海",
                    summary="用户自我介绍",
                    subject="u1",
                    importance_score=0.8,
                    atomic_facts=["名字是小明", "住在上海"],
                    foresights=[],
                    profile_patch={"name": "小明"},
                )
            ]
        )
        self.addCleanup(tmp.cleanup)
        result = service.memorize(
            MemorizeInput(
                message_id="m-profile",
                create_time=1700000600,
                sender="u1",
                content="placeholder",
                group_id="g1",
                group_name=None,
                sender_name=None,
                role="user",
            ),
            request_id="req-profile",
        )
        self.assertEqual("profile", str(result.get("memory_category")))
        memory = result.get("memory", {})
        self.assertEqual("profile", str(memory.get("memory_category")))

    def test_memory_category_marks_plan_when_foresight_present(self) -> None:
        service, tmp = self._build_service(
            [
                ExtractedMemory(
                    episode="我明天要去杭州开会",
                    summary="出差安排",
                    subject="u1",
                    importance_score=0.7,
                    atomic_facts=["明天去杭州开会"],
                    foresights=[{"content": "明天杭州会议", "start_time": 1700000700, "end_time": 1700087100}],
                    profile_patch={},
                )
            ]
        )
        self.addCleanup(tmp.cleanup)
        result = service.memorize(
            MemorizeInput(
                message_id="m-plan",
                create_time=1700000601,
                sender="u1",
                content="placeholder",
                group_id="g1",
                group_name=None,
                sender_name=None,
                role="user",
            ),
            request_id="req-plan",
        )
        self.assertEqual("plan", str(result.get("memory_category")))

    def test_memory_category_marks_task_by_task_keywords(self) -> None:
        service, tmp = self._build_service(
            [
                ExtractedMemory(
                    episode="本周待办：完成接口联调并更新任务进度",
                    summary="项目任务",
                    subject="u1",
                    importance_score=0.6,
                    atomic_facts=["完成接口联调"],
                    foresights=[],
                    profile_patch={},
                )
            ]
        )
        self.addCleanup(tmp.cleanup)
        result = service.memorize(
            MemorizeInput(
                message_id="m-task",
                create_time=1700000602,
                sender="u1",
                content="placeholder",
                group_id="g1",
                group_name=None,
                sender_name=None,
                role="user",
            ),
            request_id="req-task",
        )
        self.assertEqual("task", str(result.get("memory_category")))


if __name__ == "__main__":
    unittest.main()
