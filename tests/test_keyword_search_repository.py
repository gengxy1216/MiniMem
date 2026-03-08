from __future__ import annotations

import unittest
from pathlib import Path
from flockmem.testing.writable_tempdir import WritableTempDir

from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.infra.sqlite.memory_repository import MemoryRepository


class KeywordSearchRepositoryTests(unittest.TestCase):
    def _build_repo(self, tmp: str) -> MemoryRepository:
        engine = SQLiteEngine(Path(tmp) / "lite.db")
        init_schema(engine)
        return MemoryRepository(engine)

    def test_keyword_search_scans_full_corpus_not_only_recent_window(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
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

    def test_keyword_search_respects_time_window(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            m2023 = repo.save_message_as_memory(
                message_id="m-kw-2023",
                create_time=1686758400,
                sender="u1",
                content="alpha milestone record",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                summary="alpha",
                subject="project",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            repo.save_message_as_memory(
                message_id="m-kw-2025",
                create_time=1736611200,
                sender="u1",
                content="alpha milestone record",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                summary="alpha",
                subject="project",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            hits = repo.search_keyword(
                query="alpha milestone",
                user_id="u1",
                group_id="g1",
                top_k=5,
                start_ts=1672531200,
                end_ts=1704067199,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertEqual([str(m2023["id"])], ids)

    def test_get_episode_ids_by_event_ids_filters_hits_and_misses(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            hit_recent = repo.save_message_as_memory(
                message_id="m-evt-hit-recent",
                create_time=1735689600,
                sender="u1",
                content="hit recent",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="graph_text",
                summary="hit recent",
                subject="evt",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
                event_id="evt-hit-recent",
            )
            hit_old = repo.save_message_as_memory(
                message_id="m-evt-hit-old",
                create_time=1672531200,
                sender="u1",
                content="hit old",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="graph_text",
                summary="hit old",
                subject="evt",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
                event_id="evt-hit-old",
            )
            repo.save_message_as_memory(
                message_id="m-evt-miss-group",
                create_time=1735689601,
                sender="u1",
                content="miss group",
                user_id="u1",
                group_id="g2",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="graph_text",
                summary="miss group",
                subject="evt",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
                event_id="evt-miss-group",
            )
            repo.save_message_as_memory(
                message_id="m-evt-miss-user",
                create_time=1735689602,
                sender="u2",
                content="miss user",
                user_id="u2",
                group_id="g1",
                group_name=None,
                sender_name="u2",
                role="user",
                importance_score=0.7,
                storage_tier="graph_text",
                summary="miss user",
                subject="evt",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
                event_id="evt-miss-user",
            )
            sample_events = [
                "evt-hit-recent",
                "evt-hit-old",
                "evt-miss-group",
                "evt-miss-user",
                "evt-not-exists",
                "",
                "evt-hit-recent",
            ]

            only_recent = repo.get_episode_ids_by_event_ids(
                event_ids=sample_events,
                user_id="u1",
                group_id="g1",
                start_ts=1704067200,
                end_ts=1767225599,
                limit=20,
            )
            self.assertEqual([str(hit_recent["id"])], only_recent)

            with_history = repo.get_episode_ids_by_event_ids(
                event_ids=sample_events,
                user_id="u1",
                group_id="g1",
                limit=20,
            )
            self.assertEqual(
                {str(hit_recent["id"]), str(hit_old["id"])},
                set(with_history),
            )

    def test_event_log_keyword_search_hits_memory_by_atomic_fact(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            row = repo.save_message_as_memory(
                message_id="m-el-1",
                create_time=1700000600,
                sender="u1",
                content="weekly update without detailed owner names",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.8,
                storage_tier="vector_only",
                summary="weekly update",
                subject="ops",
                atomic_facts=["approval owner is nora", "deadline moved to friday"],
                foresights=[],
                profile_patch={},
                event_id="evt-el-1",
            )
            memory_id = str(row["id"])
            inserted = repo.save_event_logs(
                memory_id=memory_id,
                event_id="evt-el-1",
                user_id="u1",
                group_id="g1",
                event_logs=[
                    {"fact_order": 1, "fact": "approval owner is nora", "fact_norm": "approvalownerisnora"},
                    {"fact_order": 2, "fact": "deadline moved to friday", "fact_norm": "deadlinemovedtofriday"},
                ],
                created_at=1700000600,
            )
            self.assertEqual(2, inserted)
            hits = repo.search_event_log_keyword(
                query="who is nora",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(memory_id, ids)

    def test_foresight_keyword_search_hits_memory_by_foresight_content(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            row = repo.save_message_as_memory(
                message_id="m-fs-1",
                create_time=1700000602,
                sender="u1",
                content="planning weekly schedule",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.8,
                storage_tier="vector_only",
                summary="planning",
                subject="ops",
                atomic_facts=["weekly planning"],
                foresights=[
                    {
                        "content": "next friday finalize payroll checklist",
                        "start_time": 1700600000,
                        "end_time": 1700686400,
                        "confidence": 0.71,
                    }
                ],
                profile_patch={},
                event_id="evt-fs-1",
            )
            memory_id = str(row["id"])
            hits = repo.search_foresight_keyword(
                query="finalize payroll checklist friday",
                user_id="u1",
                group_id="g1",
                top_k=5,
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn(memory_id, ids)

    def test_list_groups_returns_desc_by_latest_timestamp(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            repo = self._build_repo(tmp)
            repo.save_message_as_memory(
                message_id="m-g1",
                create_time=1700000600,
                sender="u1",
                content="group g1",
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                summary="g1",
                subject="s1",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            repo.save_message_as_memory(
                message_id="m-g2",
                create_time=1700000601,
                sender="u1",
                content="group g2",
                user_id="u1",
                group_id="g2",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.7,
                storage_tier="text_only",
                summary="g2",
                subject="s2",
                atomic_facts=[],
                foresights=[],
                profile_patch={},
            )
            rows = repo.list_groups(user_id="u1", limit=10)
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual("g2", str(rows[0].get("group_id")))
            self.assertEqual("g1", str(rows[1].get("group_id")))


if __name__ == "__main__":
    unittest.main()

