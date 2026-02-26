from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evermemos_lite.bootstrap.app_factory import create_app
from evermemos_lite.config.settings import LiteSettings

_DIA_ID_RE = re.compile(r"^[Dd](\d+):0*(\d+)$")
_DIA_ID_IN_TEXT_RE = re.compile(r"[Dd]\s*:?\s*\d+\s*:\s*\d+")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _sample_cases(
    rows: list[dict[str, Any]], sample_size: int, sample_seed: int
) -> list[dict[str, Any]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return rows
    rng = random.Random(int(sample_seed))
    indices = sorted(rng.sample(range(len(rows)), k=sample_size))
    return [rows[i] for i in indices]


def _as_str(value: Any) -> str:
    return str(value or "").strip()


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    return []


def _normalize_message_id(value: Any) -> str:
    token = _as_str(value)
    if not token:
        return ""
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"^[Dd]:", "D", token)
    m = _DIA_ID_RE.match(token)
    if m:
        return f"D{int(m.group(1))}:{int(m.group(2))}"
    return token


def _as_message_id_list(value: Any) -> list[str]:
    raw_tokens: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = _as_str(item)
            if not text:
                continue
            matched = _DIA_ID_IN_TEXT_RE.findall(text)
            if matched:
                raw_tokens.extend(matched)
            else:
                parts = [x for x in re.split(r"[,;\s]+", text) if x]
                raw_tokens.extend(parts if parts else [text])
    elif isinstance(value, str):
        text = value.strip()
        if text:
            matched = _DIA_ID_IN_TEXT_RE.findall(text)
            if matched:
                raw_tokens.extend(matched)
            else:
                parts = [x for x in re.split(r"[,;\s]+", text) if x]
                raw_tokens.extend(parts if parts else [text])
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        norm = _normalize_message_id(token)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _rank_of_first_match(hit_event_ids: list[str], expected: set[str]) -> int | None:
    for idx, eid in enumerate(hit_event_ids, start=1):
        if eid in expected:
            return idx
    return None


@contextmanager
def _temporary_env(patch: dict[str, str]) -> Any:
    old_values: dict[str, str | None] = {k: os.environ.get(k) for k in patch}
    try:
        for key, value in patch.items():
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _build_memories(case: dict[str, Any], base_ts: int) -> list[dict[str, Any]]:
    raw = case.get("memories") or case.get("conversation") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            continue
        message_id = _as_str(
            row.get("message_id") or row.get("turn_id") or row.get("id")
        ) or f"turn-{i}"
        message_id = _normalize_message_id(message_id)
        content = _as_str(
            row.get("content") or row.get("text") or row.get("utterance") or row.get("episode")
        )
        if not content:
            continue
        sender = _as_str(row.get("sender") or row.get("speaker") or row.get("role"))
        create_time = row.get("create_time", row.get("timestamp"))
        if create_time is None or (isinstance(create_time, str) and not create_time.strip()):
            create_time = int(base_ts + i)
        out.append(
            {
                "message_id": message_id,
                "content": content,
                "sender": sender,
                "create_time": create_time,
                "role": _as_str(row.get("role")) or "user",
            }
        )
    return out


def _evaluate_case(
    *,
    client: TestClient,
    case: dict[str, Any],
    top_k: int,
    method: str,
    decision_mode: str,
    default_user_id: str,
    isolate_by_case: bool,
    search_use_user_id: bool,
    ingest_cache: dict[str, dict[str, str]],
) -> dict[str, Any]:
    case_id = _as_str(case.get("case_id")) or f"case-{int(time.time() * 1000)}"
    user_id = _as_str(case.get("user_id")) or default_user_id
    raw_group = _as_str(case.get("group_id")) or "locomo_group"
    group_id = f"{raw_group}:{case_id}" if isolate_by_case else raw_group
    query = _as_str(case.get("query") or case.get("question"))
    if not query:
        return {"case_id": case_id, "status": "skipped", "reason": "empty_query"}

    memories = _build_memories(case, int(time.time()) - 86400)
    if not memories:
        return {"case_id": case_id, "status": "skipped", "reason": "empty_memories"}

    msg_to_event: dict[str, str] = dict(ingest_cache.get(group_id, {}))
    missing_ids = [
        row["message_id"] for row in memories if row["message_id"] not in msg_to_event
    ]
    if missing_ids:
        for row in memories:
            if row["message_id"] in msg_to_event:
                continue
            payload = {
                "message_id": row["message_id"],
                "create_time": row["create_time"],
                "sender": row["sender"] or user_id,
                "content": row["content"],
                "group_id": group_id,
                "role": row["role"],
            }
            resp = client.post("/api/v1/memories", json=payload)
            if resp.status_code != 200:
                return {
                    "case_id": case_id,
                    "status": "failed",
                    "reason": f"memorize_http_{resp.status_code}",
                    "detail": _as_str(resp.text)[:200],
                }
            body = resp.json()
            result = body.get("result", {}) if isinstance(body, dict) else {}
            event_id = _as_str(result.get("event_id"))
            if event_id:
                msg_to_event[row["message_id"]] = event_id
        ingest_cache[group_id] = msg_to_event

    expected_event_ids = _as_str_list(case.get("expected_event_ids"))
    unresolved_ids: list[str] = []
    if not expected_event_ids:
        expected_msg_ids = _as_str_list(
            case.get("expected_message_ids")
            or case.get("supporting_message_ids")
            or case.get("supporting_turn_ids")
        )
        expected_msg_ids = _as_message_id_list(expected_msg_ids)
        for msg_id in expected_msg_ids:
            event_id = msg_to_event.get(msg_id)
            if event_id:
                expected_event_ids.append(event_id)
            else:
                unresolved_ids.append(msg_id)

    if not expected_event_ids:
        return {
            "case_id": case_id,
            "status": "skipped",
            "reason": "no_expected_targets",
            "unresolved_message_ids": unresolved_ids,
        }

    params = {
        "query": query,
        "user_id": user_id if search_use_user_id else None,
        "group_id": group_id,
        "retrieve_method": method,
        "decision_mode": decision_mode,
        "top_k": top_k,
    }
    resp = client.get("/api/v1/memories/search", params=params)
    if resp.status_code != 200:
        return {
            "case_id": case_id,
            "status": "failed",
            "reason": f"search_http_{resp.status_code}",
            "detail": _as_str(resp.text)[:200],
        }

    body = resp.json()
    result = body.get("result", {}) if isinstance(body, dict) else {}
    hits = result.get("memories", []) if isinstance(result, dict) else []
    hit_event_ids = [_as_str(x.get("event_id")) for x in hits[:top_k] if isinstance(x, dict)]
    expected = {x for x in expected_event_ids if x}
    rank = _rank_of_first_match(hit_event_ids, expected)
    hit = rank is not None
    return {
        "case_id": case_id,
        "status": "ok",
        "query": query,
        "group_id": group_id,
        "expected_event_ids": sorted(expected),
        "unresolved_message_ids": unresolved_ids,
        "rank": rank,
        "hit": hit,
        "mrr_contrib": round(1.0 / rank, 8) if rank else 0.0,
    }


def run_eval(
    *,
    dataset: Path,
    top_k: int,
    sample_size: int,
    sample_seed: int,
    method: str,
    decision_mode: str,
    default_user_id: str,
    isolate_by_case: bool,
    search_use_user_id: bool,
    ingest_profile: str,
    graph_enabled: bool,
    data_dir: Path | None,
) -> dict[str, Any]:
    raw_rows = _load_jsonl(dataset)
    rows = _sample_cases(raw_rows, sample_size=sample_size, sample_seed=sample_seed)
    if not rows:
        return {
            "status": "ok",
            "total_cases": 0,
            "sampled_cases": 0,
            "source_cases": len(raw_rows),
            "evaluated_cases": 0,
            "skipped_cases": 0,
            "failed_cases": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "cases": [],
        }

    case_details: list[dict[str, Any]] = []
    ingest_cache: dict[str, dict[str, str]] = {}
    base_data_dir: Path
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if data_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="minimem-locomo-")
        base_data_dir = Path(temp_dir_obj.name).resolve()
    else:
        base_data_dir = data_dir.resolve()
        base_data_dir.mkdir(parents=True, exist_ok=True)

    env_patch = {
        "LITE_DATA_DIR": str(base_data_dir),
        "LITE_RETRIEVAL_PROFILE": ingest_profile,
        "LITE_GRAPH_ENABLED": "true" if graph_enabled else "false",
        "LITE_AGENT_POLICY_ENABLED": "false",
    }
    try:
        with _temporary_env(env_patch):
            app = create_app(LiteSettings.from_env())
            with TestClient(app) as client:
                for row in rows:
                    detail = _evaluate_case(
                        client=client,
                        case=row,
                        top_k=top_k,
                        method=method,
                        decision_mode=decision_mode,
                        default_user_id=default_user_id,
                        isolate_by_case=isolate_by_case,
                        search_use_user_id=search_use_user_id,
                        ingest_cache=ingest_cache,
                    )
                    case_details.append(detail)
    finally:
        if temp_dir_obj is not None:
            try:
                temp_dir_obj.cleanup()
            except PermissionError:
                # Windows may keep sqlite file handles briefly after TestClient closes.
                pass

    evaluated = [x for x in case_details if x.get("status") == "ok"]
    skipped = [x for x in case_details if x.get("status") == "skipped"]
    failed = [x for x in case_details if x.get("status") == "failed"]

    recall_hits = sum(1 for x in evaluated if bool(x.get("hit")))
    mrr_sum = sum(float(x.get("mrr_contrib", 0.0)) for x in evaluated)
    n = len(evaluated)

    return {
        "status": "ok",
        "dataset": str(dataset),
        "method": method,
        "decision_mode": decision_mode,
        "top_k": top_k,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "source_cases": len(raw_rows),
        "sampled_cases": len(rows),
        "ingest_profile": ingest_profile,
        "graph_enabled": graph_enabled,
        "isolate_by_case": isolate_by_case,
        "search_use_user_id": search_use_user_id,
        "ingested_groups": len(ingest_cache),
        "total_cases": len(case_details),
        "evaluated_cases": n,
        "skipped_cases": len(skipped),
        "failed_cases": len(failed),
        "recall_at_k": round(recall_hits / n, 4) if n else 0.0,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "cases": case_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end LoCoMo retrieval evaluation in local MiniMem runtime."
    )
    parser.add_argument("--dataset", required=True, help="Prepared LoCoMo JSONL file")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Sample N cases for quick iteration. Set 0 to evaluate all cases.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size.",
    )
    parser.add_argument(
        "--method",
        default="keyword",
        choices=["keyword", "vector", "hybrid", "rrf", "agentic"],
    )
    parser.add_argument(
        "--decision-mode", default="static", choices=["static", "rule", "agent"]
    )
    parser.add_argument(
        "--default-user-id", default="locomo_user", help="Fallback user_id when missing"
    )
    parser.add_argument(
        "--isolate-by-case",
        action="store_true",
        help="Append case_id to group_id to avoid cross-case contamination.",
    )
    parser.add_argument(
        "--search-use-user-id",
        action="store_true",
        help="Pass user_id to /search. Default is disabled for multi-speaker LoCoMo cases.",
    )
    parser.add_argument(
        "--ingest-profile",
        default="keyword",
        choices=["keyword", "hybrid", "agentic"],
        help="Runtime profile used while ingesting memories.",
    )
    parser.add_argument(
        "--graph-enabled",
        action="store_true",
        help="Enable graph module during evaluation run.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional persistent LITE_DATA_DIR. Empty means temp dir.",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional path to save full JSON report.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    report = run_eval(
        dataset=dataset,
        top_k=max(1, min(100, int(args.top_k))),
        sample_size=max(0, int(args.sample_size)),
        sample_seed=int(args.sample_seed),
        method=args.method,
        decision_mode=args.decision_mode,
        default_user_id=args.default_user_id,
        isolate_by_case=bool(args.isolate_by_case),
        search_use_user_id=bool(args.search_use_user_id),
        ingest_profile=args.ingest_profile,
        graph_enabled=bool(args.graph_enabled),
        data_dir=Path(args.data_dir).resolve() if args.data_dir else None,
    )

    if args.report_out:
        out = Path(args.report_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    console_report = dict(report)
    console_report.pop("cases", None)
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
