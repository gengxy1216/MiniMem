from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.init_schema import init_schema
from evermemos_lite.infra.sqlite.memory_repository import MemoryRepository


class KeywordSearchRepositoryTests(unittest.TestCase):
    def _build_repo(self, tmp: str) -> MemoryRepository:
        engine = SQLiteEngine(Path(tmp) / "lite.db")
        init_schema(engine)
        return MemoryRepository(engine)

    def test_keyword_search_scans_full_corpus_not_only_recent_window(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            target_memory_id = ""
            for idx in range(220):
                marker = f"mk-{idx:05d}"
                row = repo.save_message_as_memory(
                    message_id=f"m-{idx:05d}",
                    create_time=1700000000 + idx,
                    sender="u1",
                    content=f"Ticket {marker} belongs to project alpha",
                    user_id="u1",
                    group_id="g1",
                    group_name=None,
                    sender_name="u1",
                    role="user",
                    importance_score=0.8,
                    storage_tier="vector_only",
                    summary=f"Summary {marker}",
                    subject="ticket",
                    atomic_facts=[f"ticket {marker}"],
                    foresights=[],
                    profile_patch={},
                )
                if idx == 0:
                    target_memory_id = str(row["id"])

            hits = repo.search_keyword(
                query="mk-00000",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(target_memory_id, ids)

    def test_keyword_search_respects_candidate_episode_ids(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            a = repo.save_message_as_memory(
                message_id="m-a",
                create_time=1700000001,
                sender="u1",
                content="Alpha marker keep-me",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="vector_only",
                summary="alpha",
                subject="alpha",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            repo.save_message_as_memory(
                message_id="m-b",
                create_time=1700000002,
                sender="u1",
                content="Alpha marker drop-me",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="vector_only",
                summary="alpha",
                subject="alpha",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            keep_id = str(a["id"])
            hits = repo.search_keyword(
                query="alpha marker",
                user_id="u1",
                group_id="g1",
                top_k=5,
                candidate_episode_ids={keep_id},
            )
            self.assertEqual([keep_id], [str(x.get("memory_id", "")) for x in hits])

    def test_keyword_search_falls_back_when_fts_unavailable(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            row = repo.save_message_as_memory(
                message_id="m-fallback",
                create_time=1700000100,
                sender="u1",
                content="Fallback marker ZX-991",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="vector_only",
                summary="fallback",
                subject="ticket",
                atomic_facts=["marker ZX-991"],
                foresights=[],
                profile_patch={},
            )
            memory_id = str(row["id"])
            # Force FTS failure path.
            repo.engine.execute("DROP TABLE IF EXISTS memory_keyword_fts")
            hits = repo.search_keyword(
                query="ZX-991",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(memory_id, ids)

    def test_keyword_fts_search_does_not_require_memory_fact_join(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            row = repo.save_message_as_memory(
                message_id="m-fts-only",
                create_time=1700000200,
                sender="u1",
                content="Ticket marker ZX-fts-001 belongs to project beta",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="vector_only",
                summary="fts-only",
                subject="ticket",
                atomic_facts=["marker ZX-fts-001"],
                foresights=[],
                profile_patch={},
            )
            memory_id = str(row["id"])
            repo.engine.execute("DROP TABLE memory_fact")
            hits = repo.search_keyword(
                query="ZX-fts-001",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(memory_id, ids)

    def test_fetch_episodes_by_ids_preserves_input_order(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            ids: list[str] = []
            for idx in range(3):
                row = repo.save_message_as_memory(
                    message_id=f"m-order-{idx}",
                    create_time=1700000300 + idx,
                    sender="u1",
                    content=f"order marker {idx}",
                    user_id="u1",
                    group_id="g1",
                    group_name=None,
                    sender_name="u1",
                    role="user",
                    importance_score=0.7,
                    storage_tier="vector_only",
                    summary=f"summary-{idx}",
                    subject="order",
                    atomic_facts=[],
                    foresights=[],
                    profile_patch={},
                )
                ids.append(str(row["id"]))

            asked = [ids[2], ids[0]]
            rows = repo.fetch_episodes_by_ids(asked)
            self.assertEqual(asked, [str(r.get("id")) for r in rows])

    def test_keyword_index_contains_category_and_tier_tokens(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            row = repo.save_message_as_memory(
                message_id="m-cat-tier",
                create_time=1700000400,
                sender="u1",
                content="整理本周待办",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.8,
                storage_tier="vector_only",
                memory_category="task",
                summary="待办事项",
                subject="work",
                atomic_facts=["本周待办"],
                foresights=[],
                profile_patch={},
            )
            memory_id = str(row["id"])
            indexed = repo.engine.query_one(
                "SELECT search_text FROM memory_keyword_fts WHERE memory_id=?",
                (memory_id,),
            )
            self.assertIsNotNone(indexed)
            search_text = str((indexed or {}).get("search_text", ""))
            self.assertIn("cat_task", search_text)
            self.assertIn("tier_vector_only", search_text)

    def test_keyword_search_can_use_category_hint(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            task_row = repo.save_message_as_memory(
                message_id="m-task-1",
                create_time=1700000500,
                sender="u1",
                content="alpha record",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                memory_category="task",
                summary="alpha",
                subject="alpha",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            repo.save_message_as_memory(
                message_id="m-event-1",
                create_time=1700000501,
                sender="u1",
                content="alpha record",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                memory_category="event",
                summary="alpha",
                subject="alpha",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            hits = repo.search_keyword(
                query="待办任务",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(str(task_row["id"]), ids)


if __name__ == "__main__":
    unittest.main()
