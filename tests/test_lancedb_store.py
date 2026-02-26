from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.vector.lancedb_store import LanceVectorStore


class LanceVectorStorePersistenceTests(unittest.TestCase):
    def test_invalid_index_type_falls_back_to_hnsw_pq(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = LanceVectorStore(
                base,
                vector_dim=4,
                use_lancedb=False,
                index_type="UNKNOWN_INDEX",
            )
            self.assertEqual("IVF_HNSW_PQ", store._index_type)

    def test_local_upsert_is_persisted_and_restored_after_restart(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = LanceVectorStore(base, vector_dim=4, use_lancedb=False)
            store.upsert(
                "row-1",
                "mem-1",
                [1.0, 0.0, 0.0, 0.0],
                {"user_id": "u1", "group_id": "g1", "importance_score": 0.3},
            )
            reloaded = LanceVectorStore(base, vector_dim=4, use_lancedb=False)
            hits = reloaded.search(
                vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                user_id="u1",
                group_id="g1",
            )
            self.assertEqual(1, len(hits))
            self.assertEqual("mem-1", hits[0]["memory_id"])
            self.assertGreater(float(hits[0]["score"]), 0.99)

    def test_search_filters_by_user_group_and_candidate_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = LanceVectorStore(base, vector_dim=4, use_lancedb=False)
            store.upsert(
                "row-1",
                "mem-1",
                [1.0, 0.0, 0.0, 0.0],
                {"user_id": "u1", "group_id": "g1", "importance_score": 0.3},
            )
            store.upsert(
                "row-2",
                "mem-2",
                [0.9, 0.1, 0.0, 0.0],
                {"user_id": "u2", "group_id": "g2", "importance_score": 0.3},
            )
            hits = store.search(
                vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                user_id="u1",
                group_id="g1",
                candidate_episode_ids={"mem-1"},
            )
            self.assertEqual(["mem-1"], [x["memory_id"] for x in hits])

    def test_restore_ignores_corrupted_log_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = LanceVectorStore(base, vector_dim=4, use_lancedb=False)
            store.upsert(
                "row-1",
                "mem-1",
                [1.0, 0.0, 0.0, 0.0],
                {"user_id": "u1", "group_id": "g1", "importance_score": 0.3},
            )
            log_path = base / LanceVectorStore.LOG_FILE
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("{not-json}\n")
                fh.write(
                    '{"op":"upsert","row":{"id":"bad","memory_id":"bad","vector":[1,2],"metadata":{}}}\n'
                )
            reloaded = LanceVectorStore(base, vector_dim=4, use_lancedb=False)
            hits = reloaded.search(
                vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                user_id="u1",
                group_id="g1",
            )
            self.assertEqual(1, len(hits))
            self.assertEqual("mem-1", hits[0]["memory_id"])

    @unittest.skipUnless(
        importlib.util.find_spec("lancedb") is not None,
        "lancedb package is required for this test",
    )
    def test_high_importance_memory_is_persisted_to_lancedb(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = LanceVectorStore(
                base,
                vector_dim=4,
                use_lancedb=True,
                lance_persist_min_importance=0.7,
            )
            store.upsert(
                "row-high",
                "mem-high",
                [1.0, 0.0, 0.0, 0.0],
                {"user_id": "u1", "group_id": "g1", "importance_score": 0.92},
            )
            store.upsert(
                "row-low",
                "mem-low",
                [0.8, 0.2, 0.0, 0.0],
                {"user_id": "u1", "group_id": "g1", "importance_score": 0.2},
            )
            reloaded = LanceVectorStore(
                base,
                vector_dim=4,
                use_lancedb=True,
                lance_persist_min_importance=0.7,
            )
            hits = reloaded.search(
                vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                user_id="u1",
                group_id="g1",
            )
            ids = [str(x.get("memory_id", "")) for x in hits]
            self.assertIn("mem-high", ids)
            self.assertIn("mem-low", ids)


if __name__ == "__main__":
    unittest.main()
