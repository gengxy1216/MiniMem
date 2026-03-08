from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    key: str = "id",
    score_key: str = "score",
    rrf_k: int = 60,
) -> list[dict]:
    if not ranked_lists:
        return []

    index: dict[str, dict] = {}
    fused_scores: defaultdict[str, float] = defaultdict(float)
    source_map: defaultdict[str, set[str]] = defaultdict(set)

    for rows in ranked_lists:
        for rank, row in enumerate(rows, start=1):
            rid = str(row.get(key) or "")
            if not rid:
                continue
            index[rid] = dict(row)
            fused_scores[rid] += 1.0 / (rrf_k + rank)
            source = row.get("source")
            if source:
                source_map[rid].add(str(source))

    fused: list[dict] = []
    for rid, score in fused_scores.items():
        row = dict(index[rid])
        row[score_key] = float(score)
        if source_map[rid]:
            row["source"] = ",".join(sorted(source_map[rid]))
        fused.append(row)

    fused.sort(key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    return fused
