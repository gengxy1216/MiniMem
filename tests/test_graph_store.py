from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from evermemos_lite.infra.graph.kuzu_store import KuzuGraphStore


@dataclass
class Triple:
    subject: str
    relation: str
    obj: str
    confidence: float = 0.8


class GraphStorePersistenceTests(unittest.TestCase):
    def test_upsert_persists_and_restores(self) -> None:
        with TemporaryDirectory() as tmp:
            store = KuzuGraphStore(Path(tmp), enabled=True)
            inserted = store.upsert_triples(
                [Triple("张三", "likes", "咖啡")],
                event_id="evt-1",
                timestamp=100,
                user_id="u1",
                group_id="g1",
            )
            self.assertEqual(1, inserted)
            reloaded = KuzuGraphStore(Path(tmp), enabled=True)
            hits = reloaded.search("喜欢 咖啡", user_id="u1", group_id="g1", limit=5)
            self.assertEqual(1, len(hits))
            self.assertEqual("张三", hits[0]["subject"])

    def test_search_and_neighbors_respect_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            store = KuzuGraphStore(Path(tmp), enabled=True)
            store.upsert_triples(
                [
                    Triple("李四", "likes", "茶"),
                    Triple("王五", "likes", "咖啡"),
                ],
                event_id="evt-2",
                timestamp=200,
                user_id="u1",
                group_id="g1",
            )
            store.upsert_triples(
                [Triple("李四", "likes", "红酒")],
                event_id="evt-3",
                timestamp=300,
                user_id="u2",
                group_id="g2",
            )
            hits = store.search("李四 喜欢", user_id="u1", group_id="g1", limit=10)
            self.assertTrue(all(h["group_id"] == "g1" for h in hits))
            neighbors = store.neighbors("李四", user_id="u1", group_id="g1", limit=10)
            self.assertTrue(all(n["group_id"] == "g1" for n in neighbors))

    def test_load_ignores_corrupted_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = KuzuGraphStore(base, enabled=True)
            store.upsert_triples(
                [Triple("赵六", "likes", "篮球")],
                event_id="evt-4",
                timestamp=400,
                user_id="u1",
                group_id="g1",
            )
            rows_path = base / "graph_rows.jsonl"
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write("{bad-json}\n")
            reloaded = KuzuGraphStore(base, enabled=True)
            hits = reloaded.search("篮球", user_id="u1", group_id="g1", limit=5)
            self.assertEqual(1, len(hits))
            self.assertEqual("赵六", hits[0]["subject"])


if __name__ == "__main__":
    unittest.main()
