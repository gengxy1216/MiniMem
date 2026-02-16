from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from evermemos_lite.bootstrap.app_factory import create_app
from evermemos_lite.config.settings import LiteSettings


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
    return rows


def _rank_of_first_match(hit_event_ids: list[str], expected: set[str]) -> int | None:
    for idx, eid in enumerate(hit_event_ids, start=1):
        if eid in expected:
            return idx
    return None


def run_eval(dataset: Path, top_k: int, method: str, decision_mode: str) -> dict[str, Any]:
    app = create_app(LiteSettings.from_env())
    client = TestClient(app)
    cases = _load_jsonl(dataset)
    if not cases:
        return {"total": 0, "recall_at_k": 0.0, "mrr": 0.0}

    recall_hits = 0
    mrr_sum = 0.0
    valid = 0
    for item in cases:
        query = str(item.get("query", "")).strip()
        expected_ids = set(str(x) for x in item.get("expected_event_ids", []) if str(x).strip())
        if not query or not expected_ids:
            continue
        valid += 1
        params = {
            "query": query,
            "user_id": item.get("user_id"),
            "group_id": item.get("group_id"),
            "retrieve_method": method,
            "decision_mode": decision_mode,
            "top_k": top_k,
        }
        resp = client.get("/api/v1/memories/search", params=params)
        if resp.status_code != 200:
            continue
        result = resp.json().get("result", {})
        rows = result.get("memories", [])
        hit_event_ids = [str(r.get("event_id", "")) for r in rows[:top_k]]
        rank = _rank_of_first_match(hit_event_ids, expected_ids)
        if rank is not None:
            recall_hits += 1
            mrr_sum += 1.0 / float(rank)

    if valid == 0:
        return {"total": 0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "total": valid,
        "recall_at_k": round(recall_hits / valid, 4),
        "mrr": round(mrr_sum / valid, 4),
        "method": method,
        "decision_mode": decision_mode,
        "top_k": top_k,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Memory retrieval benchmark harness for datasets such as "
            "LoCoMo / LongMemEval / PersonaMem-v2 (after conversion to JSONL)."
        )
    )
    p.add_argument("--dataset", required=True, help="Path to JSONL benchmark file")
    p.add_argument("--name", default="custom", help="Benchmark name")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--method", default="hybrid", choices=["keyword", "vector", "hybrid", "rrf", "agentic"])
    p.add_argument("--decision-mode", default="static", choices=["static", "rule", "agent"])
    args = p.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")
    report = run_eval(
        dataset=dataset,
        top_k=max(1, min(100, args.top_k)),
        method=args.method,
        decision_mode=args.decision_mode,
    )
    report["benchmark"] = args.name
    report["dataset"] = str(dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
