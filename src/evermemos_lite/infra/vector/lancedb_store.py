from __future__ import annotations

import json
import heapq
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lancedb = None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LanceVectorStore:
    SNAPSHOT_FILE = "vector_rows.snapshot.json"
    LOG_FILE = "vector_rows.log.jsonl"
    COMPACT_MIN_OPS = 200
    TABLE_NAME = "memory_vector_index"
    SUPPORTED_INDEX_TYPES = {
        "IVF_FLAT",
        "IVF_SQ",
        "IVF_PQ",
        "IVF_RQ",
        "IVF_HNSW_SQ",
        "IVF_HNSW_PQ",
    }

    def __init__(
        self,
        db_dir: Path,
        vector_dim: int,
        *,
        use_lancedb: bool = True,
        lance_persist_min_importance: float = 0.72,
        index_type: str = "IVF_HNSW_PQ",
        index_metric: str = "cosine",
        hnsw_m: int = 20,
        hnsw_ef_construction: int = 300,
        search_nprobes: int = 20,
        search_ef: int = 80,
        search_refine_factor: int = 2,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = int(vector_dim)
        self.enabled = True
        self.lance_enabled = False
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._snapshot_path = self.db_dir / self.SNAPSHOT_FILE
        self._log_path = self.db_dir / self.LOG_FILE
        self._log_ops = 0
        self._use_lancedb = bool(use_lancedb)
        self._lance_persist_min_importance = max(
            0.0, min(1.0, float(lance_persist_min_importance))
        )
        index_key = str(index_type or "IVF_HNSW_PQ").strip().upper()
        self._index_type = (
            index_key if index_key in self.SUPPORTED_INDEX_TYPES else "IVF_HNSW_PQ"
        )
        self._index_metric = str(index_metric or "cosine").strip().lower() or "cosine"
        self._hnsw_m = max(4, int(hnsw_m))
        self._hnsw_ef_construction = max(32, int(hnsw_ef_construction))
        self._search_nprobes = max(1, int(search_nprobes))
        self._search_ef = max(8, int(search_ef))
        self._search_refine_factor = max(0, int(search_refine_factor))
        self._index_rebuild_pending = 0
        self._index_rebuild_every = 128
        self._lance_db = None
        self._lance_table = None
        self._init_lance()
        self._load_from_disk()
        self._promote_local_rows_to_lance()

    def upsert(
        self, row_id: str, memory_id: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        row = {
            "id": row_id,
            "memory_id": memory_id,
            "vector": list(vector),
            "metadata": dict(metadata),
        }
        with self._lock:
            if self._should_store_in_lancedb(row) and self._upsert_lancedb_locked(row):
                if row_id in self._rows:
                    self._rows.pop(row_id, None)
                    self._append_log_locked({"op": "delete", "id": row_id})
                return
            self._rows[row_id] = row
            self._append_log_locked({"op": "upsert", "row": row})
            if self._should_compact_locked():
                self._compact_locked()

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._rows.values())
            lance_table = self._lance_table if self.lance_enabled else None
        in_mem_hits = self._search_local(
            rows=rows,
            vector=vector,
            top_k=top_k,
            user_id=user_id,
            group_id=group_id,
            candidate_episode_ids=candidate_episode_ids,
        )
        lance_hits = self._search_lancedb(
            table=lance_table,
            vector=vector,
            top_k=top_k,
            user_id=user_id,
            group_id=group_id,
            candidate_episode_ids=candidate_episode_ids,
        )
        merged: dict[str, dict[str, Any]] = {}
        for row in [*in_mem_hits, *lance_hits]:
            key = str(row.get("memory_id") or row.get("id") or "")
            if not key:
                continue
            existing = merged.get(key)
            if existing is None or float(row.get("score", 0.0)) > float(
                existing.get("score", 0.0)
            ):
                merged[key] = row
        result = list(merged.values())
        result.sort(key=lambda x: float(x["score"]), reverse=True)
        return result[: max(1, int(top_k))]

    def _search_local(
        self,
        *,
        rows: list[dict[str, Any]],
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None,
    ) -> list[dict[str, Any]]:
        limit = max(24, int(top_k) * 6)
        heap: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            memory_id = str(row["memory_id"])
            meta = row["metadata"]
            if user_id and meta.get("user_id") != user_id:
                continue
            if group_id and meta.get("group_id") != group_id:
                continue
            if candidate_episode_ids is not None and memory_id not in candidate_episode_ids:
                continue
            sim = _cosine(vector, row["vector"])
            item = {
                "id": row["id"],
                "memory_id": memory_id,
                "distance": float(max(0.0, 1.0 - sim)),
                "score": float(sim),
                "source": "vector_local",
            }
            if len(heap) < limit:
                heapq.heappush(heap, (float(sim), item))
            else:
                heapq.heappushpop(heap, (float(sim), item))
        out = [entry[1] for entry in heap]
        out.sort(key=lambda x: float(x["score"]), reverse=True)
        return out

    def _search_lancedb(
        self,
        *,
        table: Any,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None,
    ) -> list[dict[str, Any]]:
        if table is None:
            return []
        expr = self._build_lancedb_filter(user_id=user_id, group_id=group_id)
        limit = self._lancedb_query_limit(
            top_k=top_k, candidate_episode_ids=candidate_episode_ids
        )
        try:
            query = table.search(vector)
            if hasattr(query, "metric"):
                query = query.metric(self._index_metric)
            if hasattr(query, "nprobes"):
                query = query.nprobes(self._search_nprobes)
            if hasattr(query, "ef"):
                query = query.ef(self._search_ef)
            if self._search_refine_factor > 0 and hasattr(query, "refine_factor"):
                query = query.refine_factor(self._search_refine_factor)
            if expr:
                query = query.where(expr, prefilter=True)
            rows = query.limit(limit).to_list()
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            if (
                candidate_episode_ids is not None
                and memory_id not in candidate_episode_ids
            ):
                continue
            raw_distance = row.get("_distance")
            try:
                distance = float(raw_distance) if raw_distance is not None else 1.0
            except Exception:
                distance = 1.0
            score = max(0.0, 1.0 - distance)
            out.append(
                {
                    "id": str(row.get("id", "")),
                    "memory_id": memory_id,
                    "distance": float(max(0.0, distance)),
                    "score": float(score),
                    "source": "vector_lancedb",
                }
            )
        out.sort(key=lambda x: float(x["score"]), reverse=True)
        return out

    def _init_lance(self) -> None:
        if not self._use_lancedb or lancedb is None:
            self.lance_enabled = False
            return
        try:
            self._lance_db = lancedb.connect(str(self.db_dir))
            if hasattr(self._lance_db, "list_tables"):
                listed = self._lance_db.list_tables()
                table_names = getattr(listed, "tables", None)
                if isinstance(table_names, list):
                    names = {str(x) for x in table_names}
                elif isinstance(listed, dict):
                    names = {str(x) for x in listed.get("tables", [])}
                else:
                    names = set()
            else:
                names = set(str(x) for x in self._lance_db.table_names())
            if self.TABLE_NAME in names:
                self._lance_table = self._lance_db.open_table(self.TABLE_NAME)
                self._ensure_lance_index_locked(force=True)
            else:
                self._lance_table = None
            self.lance_enabled = True
        except Exception:
            self.lance_enabled = False
            self._lance_db = None
            self._lance_table = None

    def _ensure_lancedb_table_locked(self, row: dict[str, Any]) -> tuple[Any, bool]:
        if not self.lance_enabled or self._lance_db is None:
            return None, False
        if self._lance_table is not None:
            return self._lance_table, False
        self._lance_table = self._lance_db.create_table(
            self.TABLE_NAME,
            data=[self._to_lancedb_row(row)],
        )
        self._ensure_lance_index_locked(force=True)
        return self._lance_table, True

    def _upsert_lancedb_locked(self, row: dict[str, Any]) -> bool:
        if not self.lance_enabled:
            return False
        try:
            table, created = self._ensure_lancedb_table_locked(row)
            if table is None:
                return False
            if not created:
                table.add([self._to_lancedb_row(row)])
                self._index_rebuild_pending += 1
                if self._index_rebuild_pending >= self._index_rebuild_every:
                    self._ensure_lance_index_locked(force=True)
                    self._index_rebuild_pending = 0
            return True
        except Exception:
            return False

    def _ensure_lance_index_locked(self, *, force: bool) -> None:
        if not self.lance_enabled or self._lance_table is None:
            return
        if not force:
            return
        kwargs: dict[str, Any] = {
            "metric": self._index_metric,
            "index_type": self._index_type,
            "vector_column_name": "vector",
            "replace": True,
            "m": self._hnsw_m,
            "ef_construction": self._hnsw_ef_construction,
        }
        try:
            self._lance_table.create_index(**kwargs)
        except Exception:
            # Keep retrieval available even when index build fails.
            return

    def _should_store_in_lancedb(self, row: dict[str, Any]) -> bool:
        if not self.lance_enabled:
            return False
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            return False
        try:
            importance = float(meta.get("importance_score", 0.0))
        except Exception:
            importance = 0.0
        return importance >= self._lance_persist_min_importance

    def _promote_local_rows_to_lance(self) -> None:
        if not self.lance_enabled:
            return
        with self._lock:
            moved_ids: list[str] = []
            for row_id, row in list(self._rows.items()):
                if not self._should_store_in_lancedb(row):
                    continue
                if not self._upsert_lancedb_locked(row):
                    continue
                moved_ids.append(row_id)
            for row_id in moved_ids:
                self._rows.pop(row_id, None)
                self._append_log_locked({"op": "delete", "id": row_id})
            if moved_ids:
                self._ensure_lance_index_locked(force=True)
                self._compact_locked()

    def _load_from_disk(self) -> None:
        try:
            if self._snapshot_path.exists():
                self._load_snapshot()
            if self._log_path.exists():
                self._replay_log()
        except Exception:
            self._rows = {}
            self._log_ops = 0

    def _load_snapshot(self) -> None:
        raw = self._snapshot_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return
        items = payload.get("rows")
        if not isinstance(items, list):
            return
        restored: dict[str, dict[str, Any]] = {}
        for item in items:
            row = self._normalize_row(item)
            if not row:
                continue
            restored[str(row["id"])] = row
        self._rows = restored

    def _replay_log(self) -> None:
        ops = 0
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    evt = json.loads(text)
                except Exception:
                    continue
                if not isinstance(evt, dict):
                    continue
                op = str(evt.get("op"))
                if op == "delete":
                    row_id = str(evt.get("id", "")).strip()
                    if row_id:
                        self._rows.pop(row_id, None)
                        ops += 1
                    continue
                if op != "upsert":
                    continue
                row = self._normalize_row(evt.get("row"))
                if not row:
                    continue
                self._rows[str(row["id"])] = row
                ops += 1
        self._log_ops = ops

    @staticmethod
    def _quote_sql(value: str) -> str:
        return str(value).replace("'", "''")

    def _build_lancedb_filter(
        self, *, user_id: str | None, group_id: str | None
    ) -> str | None:
        clauses: list[str] = []
        if user_id:
            clauses.append(f"user_id = '{self._quote_sql(user_id)}'")
        if group_id:
            clauses.append(f"group_id = '{self._quote_sql(group_id)}'")
        if not clauses:
            return None
        return " AND ".join(clauses)

    @staticmethod
    def _lancedb_query_limit(
        *, top_k: int, candidate_episode_ids: set[str] | None
    ) -> int:
        base = max(24, int(top_k) * 6)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return max(1, base)
            base = max(base, min(4000, len(candidate_episode_ids) * 2))
        return max(1, base)

    def _to_lancedb_row(self, row: dict[str, Any]) -> dict[str, Any]:
        meta = row.get("metadata")
        metadata = dict(meta) if isinstance(meta, dict) else {}
        try:
            ts = int(metadata.get("timestamp", 0))
        except Exception:
            ts = 0
        try:
            imp = float(metadata.get("importance_score", 0.0))
        except Exception:
            imp = 0.0
        return {
            "id": str(row.get("id", "")),
            "memory_id": str(row.get("memory_id", "")),
            "vector": [float(x) for x in row.get("vector", []) if isinstance(x, (int, float))],
            "user_id": str(metadata.get("user_id", "") or ""),
            "group_id": str(metadata.get("group_id", "") or ""),
            "timestamp": ts,
            "importance_score": imp,
        }

    def _normalize_row(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        row_id = str(value.get("id", "")).strip()
        memory_id = str(value.get("memory_id", "")).strip()
        if not row_id or not memory_id:
            return None
        raw_vector = value.get("vector")
        if not isinstance(raw_vector, list):
            return None
        vector = [float(x) for x in raw_vector if isinstance(x, (int, float))]
        if len(vector) != self.vector_dim:
            return None
        raw_meta = value.get("metadata")
        metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        return {
            "id": row_id,
            "memory_id": memory_id,
            "vector": vector,
            "metadata": metadata,
        }

    def _append_log_locked(self, event: dict[str, Any]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False))
            fh.write("\n")
        self._log_ops += 1

    def _should_compact_locked(self) -> bool:
        row_count = len(self._rows)
        if row_count <= 0:
            return False
        if self._log_ops < self.COMPACT_MIN_OPS:
            return False
        return self._log_ops >= row_count * 2

    def _compact_locked(self) -> None:
        tmp_snapshot = self._snapshot_path.with_suffix(".tmp")
        payload = {"rows": list(self._rows.values())}
        tmp_snapshot.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_snapshot, self._snapshot_path)
        tmp_log = self._log_path.with_suffix(".tmp")
        tmp_log.write_text("", encoding="utf-8")
        os.replace(tmp_log, self._log_path)
        self._log_ops = 0
