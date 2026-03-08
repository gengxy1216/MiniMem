from __future__ import annotations

import time
import unittest
from pathlib import Path

from flockmem.domain.policy import EffectivePolicy
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.infra.vector.lancedb_store import LanceVectorStore
from flockmem.service.memory_service import MemoryService
from flockmem.testing.writable_tempdir import WritableTempDir


class _NoopExtractor:
    def extract(self, content: str, sender: str, group_id: str | None):
        raise NotImplementedError


class _NoopGraphStore:
    enabled = False


class _FixedEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _NoTemporalConstraintMemoryService(MemoryService):
    def _resolve_query_time_constraints(
        self,
        *,
        query: str,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> tuple[int | None, int | None, int | None, str]:
        return as_of_ts, start_ts, end_ts, "disabled_for_comparison"


class TemporalEndToEndImpactTests(unittest.TestCase):
    @staticmethod
    def _policy_hybrid() -> EffectivePolicy:
        return EffectivePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
            importance_threshold=0.1,
            keyword_top_k=30,
            vector_top_k=30,
            rrf_k=80,
            profile="default",
            reason="e2e_temporal_impact",
        )

    @staticmethod
    def _ts(date_str: str) -> int:
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")))

    def _build_service(
        self, *, tmp_dir: str, temporal_enabled: bool
    ) -> tuple[MemoryService, dict[str, str]]:
        db_path = Path(tmp_dir) / ("temporal.db" if temporal_enabled else "baseline.db")
        vec_path = Path(tmp_dir) / ("vec_temporal" if temporal_enabled else "vec_baseline")
        engine = SQLiteEngine(db_path)
        init_schema(engine)
        service_cls = MemoryService if temporal_enabled else _NoTemporalConstraintMemoryService
        service = service_cls(
            engine=engine,
            vector_store=LanceVectorStore(vec_path, vector_dim=4, use_lancedb=False),
            embedding_provider=_FixedEmbeddingProvider(),
            extractor=_NoopExtractor(),
            graph_store=_NoopGraphStore(),
            phase4_reasoning_enabled=True,
        )
        dataset = [
            ("2023", "alpha milestone record", self._ts("2023-06-15"), [0.82, 0.0, 0.0, 0.0]),
            ("2024a", "alpha milestone record", self._ts("2024-03-10"), [0.91, 0.0, 0.0, 0.0]),
            ("2024b", "alpha milestone record", self._ts("2024-11-20"), [0.89, 0.0, 0.0, 0.0]),
            ("2025", "alpha milestone record", self._ts("2025-01-12"), [1.0, 0.0, 0.0, 0.0]),
        ]
        ids: dict[str, str] = {}
        for key, text, ts, vec in dataset:
            row = service.repo.save_message_as_memory(
                message_id=f"m-{key}",
                create_time=ts,
                sender="u1",
                content=text,
                user_id="u1",
                group_id="g1",
                group_name=None,
                sender_name="u1",
                role="user",
                importance_score=0.8,
                storage_tier="vector_only",
                summary="alpha milestone",
                subject="project",
                atomic_facts=[text],
                foresights=[],
                profile_patch={},
                memory_category="event",
            )
            mid = str(row["id"])
            ids[key] = mid
            service.vector_store.upsert(
                row_id=f"vec-{key}",
                memory_id=mid,
                vector=vec,
                metadata={
                    "id": mid,
                    "user_id": "u1",
                    "group_id": "g1",
                    "timestamp": ts,
                    "importance_score": 0.8,
                },
            )
        return service, ids

    @staticmethod
    def _recall_and_mrr(rows: list[dict], target_ids: set[str]) -> tuple[float, float]:
        ranked = [str(r.get("id", "")) for r in rows]
        hit_count = len([x for x in ranked if x in target_ids])
        recall = float(hit_count) / float(max(1, len(target_ids)))
        rr = 0.0
        for idx, item_id in enumerate(ranked, start=1):
            if item_id in target_ids:
                rr = 1.0 / float(idx)
                break
        return recall, rr

    def test_temporal_constraints_improve_full_chain_recall_and_mrr(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            temporal_service, temporal_ids = self._build_service(
                tmp_dir=tmp, temporal_enabled=True
            )
            baseline_service, baseline_ids = self._build_service(
                tmp_dir=tmp, temporal_enabled=False
            )
            policy = self._policy_hybrid()
            query = "alpha milestone 2024"

            temporal_rows = temporal_service.search(
                policy=policy,
                query=query,
                user_id="u1",
                group_id="g1",
                top_k=2,
            )
            baseline_rows = baseline_service.search(
                policy=policy,
                query=query,
                user_id="u1",
                group_id="g1",
                top_k=2,
            )
            target_temporal = {temporal_ids["2024a"], temporal_ids["2024b"]}
            target_baseline = {baseline_ids["2024a"], baseline_ids["2024b"]}
            recall_with, mrr_with = self._recall_and_mrr(temporal_rows, target_temporal)
            recall_without, mrr_without = self._recall_and_mrr(
                baseline_rows, target_baseline
            )

            self.assertGreater(recall_with, recall_without)
            self.assertGreater(mrr_with, mrr_without)


if __name__ == "__main__":
    unittest.main()

