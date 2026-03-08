from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.service.event_log_extractor import RuleEventLogExtractor
from evermemos_lite.service.extractor import ExtractedMemory
from evermemos_lite.service.memcell_extractor import RuleMemCellExtractor
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
        return [0.1, 0.2, 0.3, 0.4]


class _NoopGraphStore:
    enabled = False


class _QueueExtractor:
    def __init__(self, items: list[ExtractedMemory]) -> None:
        self._items = list(items)

    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        if not self._items:
            raise RuntimeError("no extracted memory queued")
        return self._items.pop(0)


class _FailOnceExtractor:
    def __init__(self, final_item: ExtractedMemory) -> None:
        self._final = final_item
        self.calls = 0

    def extract(self, content: str, sender: str, group_id: str | None) -> ExtractedMemory:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient extractor failure")
        return self._final


class MemCellEventLogPipelineTests(unittest.TestCase):
    def test_memcell_extractor_splits_long_episode(self) -> None:
        extractor = RuleMemCellExtractor(max_chars_per_cell=120, min_chars_per_cell=40, max_cells=10)
        text = (
            "alpha release timeline changed because dependency migration slipped. "
            "beta rollback plan prepared with mitigation and owner assignments. "
            "gamma support checklist updated for night shift operation."
        )
        cells = extractor.split(text)
        self.assertGreaterEqual(len(cells), 2)
        self.assertEqual([1, 2], [cells[0].order, cells[1].order])
        self.assertTrue(all(len(c.content) <= 120 for c in cells))

    def test_event_log_extractor_dedupes_normalized_facts(self) -> None:
        extractor = RuleEventLogExtractor(max_items=8)
        rows = extractor.extract(
            atomic_facts=[
                "Payroll deadline moved to Friday",
                "payroll deadline moved to friday!",
                "Owner is Nora",
            ],
            episode="",
        )
        self.assertEqual(2, len(rows))
        self.assertEqual("payrolldeadlinemovedtofriday", rows[0].fact_norm)
        self.assertEqual("ownerisnora", rows[1].fact_norm)

    def test_memorize_persists_memcells_and_event_logs(self) -> None:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        episode_text = (
            "alpha incident started monday afternoon and impacted payroll processing. "
            "the response team prepared rollback steps and reviewed support checklist. "
            "final mitigation was approved by nora before friday deadline."
        )
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=_QueueExtractor(
                [
                    ExtractedMemory(
                        episode=episode_text,
                        summary="incident summary",
                        subject="u1",
                        importance_score=0.8,
                        atomic_facts=[
                            "incident started monday afternoon",
                            "rollback steps prepared",
                            "approved by nora before friday deadline",
                            "rollback steps prepared",
                        ],
                        foresights=[],
                        profile_patch={},
                    )
                ]
            ),
            graph_store=_NoopGraphStore(),
        )
        result = service.memorize(
            MemorizeInput(
                message_id="m-p1",
                create_time=1700000800,
                sender="u1",
                content=episode_text,
                group_id="g1",
                group_name=None,
                sender_name=None,
                role="user",
            ),
            request_id="req-p1",
        )
        memory = result.get("memory", {})
        memory_id = str(memory.get("id", ""))
        self.assertTrue(memory_id)
        memcells = service.repo.get_memcells_by_memory_id(memory_id)
        event_logs = service.repo.get_event_logs_by_memory_id(memory_id)
        self.assertGreaterEqual(len(memcells), 1)
        self.assertEqual(int(result.get("memcell_count", 0)), len(memcells))
        self.assertEqual(int(result.get("event_log_count", 0)), len(event_logs))
        self.assertEqual(3, len(event_logs))
        self.assertEqual(1, int(event_logs[0].get("fact_order", 0)))

    def test_memorize_retries_episode_extract_and_applies_quality_gate(self) -> None:
        tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "lite.db")
        init_schema(engine)
        long_episode = (
            "caroline discussed adoption plans and timeline with mentors. " * 14
            + "she also prepared action checklist for next friday."
        )
        extractor = _FailOnceExtractor(
            ExtractedMemory(
                episode=long_episode,
                summary="",
                subject="u1",
                importance_score=0.6,
                atomic_facts=["adoption plans"],
                foresights=[],
                profile_patch={},
            )
        )
        service = MemoryService(
            engine=engine,
            vector_store=_NoopVectorStore(),
            embedding_provider=_NoopEmbeddingProvider(),
            extractor=extractor,
            graph_store=_NoopGraphStore(),
            extract_max_retries=3,
        )
        out = service.memorize(
            MemorizeInput(
                message_id="m-p2",
                create_time=1700000900,
                sender="u1",
                content=long_episode,
                group_id="g1",
                group_name=None,
                sender_name=None,
                role="user",
            ),
            request_id="req-p2",
        )
        self.assertGreaterEqual(extractor.calls, 2)
        memory = out.get("memory", {})
        memory_id = str(memory.get("id", ""))
        self.assertTrue(memory_id)
        facts = engine.query_all(
            "SELECT fact FROM memory_fact WHERE memory_id=? ORDER BY id ASC",
            (memory_id,),
        )
        self.assertGreaterEqual(len(facts), 2)
        event_logs = service.repo.get_event_logs_by_memory_id(memory_id)
        self.assertGreaterEqual(len(event_logs), 1)


if __name__ == "__main__":
    unittest.main()
