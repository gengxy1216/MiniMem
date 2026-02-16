from __future__ import annotations

import math
from pathlib import Path
from typing import Any


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
    def __init__(self, db_dir: Path, vector_dim: int) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = int(vector_dim)
        self.enabled = True
        self._rows: dict[str, dict[str, Any]] = {}

    def upsert(self, row_id: str, memory_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._rows[row_id] = {
            "id": row_id,
            "memory_id": memory_id,
            "vector": list(vector),
            "metadata": dict(metadata),
        }

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in self._rows.values():
            memory_id = str(row["memory_id"])
            meta = row["metadata"]
            if user_id and meta.get("user_id") != user_id:
                continue
            if group_id and meta.get("group_id") != group_id:
                continue
            if candidate_episode_ids is not None and memory_id not in candidate_episode_ids:
                continue
            sim = _cosine(vector, row["vector"])
            result.append(
                {
                    "id": row["id"],
                    "memory_id": memory_id,
                    "distance": float(max(0.0, 1.0 - sim)),
                    "score": float(sim),
                    "source": "vector",
                }
            )
        result.sort(key=lambda x: float(x["score"]), reverse=True)
        return result[: max(1, int(top_k))]
