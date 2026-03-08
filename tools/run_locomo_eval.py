from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
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

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings

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


def _default_eval_cache_dir(
    *, dataset: Path, ingest_profile: str, graph_enabled: bool
) -> Path:
    resolved = dataset.resolve()
    stat = resolved.stat()
    token = "|".join(
        [
            str(resolved),
            str(int(stat.st_size)),
            str(int(stat.st_mtime)),
            ingest_profile,
            "graph=1" if graph_enabled else "graph=0",
        ]
    )
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.stem) or "dataset"
    return (
        ROOT_DIR
        / "FlockMem_data"
        / "benchmarks"
        / "eval_cache"
        / f"{safe_stem}.{ingest_profile}.{digest}"
    )


def _load_dotenv_defaults(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except Exception:
        return
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.lower().startswith("export "):
            text = text[7:].strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        k = key.strip()
        if not k or k in os.environ:
            continue
        v = value.strip().strip("'").strip('"')
        os.environ[k] = v


def _component_ready(component: Any) -> tuple[bool, str]:
    if component is None:
        return False, "missing_component"
    enabled = bool(getattr(component, "enabled", False))
    base_url = str(getattr(component, "base_url", "") or "").strip()
    api_key = str(getattr(component, "api_key", "") or "").strip()
    model = str(getattr(component, "model", "") or "").strip()
    if not enabled:
        return False, "disabled"
    missing: list[str] = []
    if not base_url:
        missing.append("base_url")
    if not api_key:
        missing.append("api_key")
    if not model:
        missing.append("model")
    if missing:
        return False, "missing:" + ",".join(missing)
    return True, "ok"


def _run_model_preflight(
    *,
    app: Any,
    method: str,
) -> dict[str, Any]:
    memory_service = getattr(app.state, "memory_service", None)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    needs_vector = method in {"vector", "hybrid", "rrf", "agentic"}
    needs_agentic_llm = method == "agentic"

    if needs_vector:
        embed_ok = False
        embed_detail = "unknown"
        try:
            provider = getattr(memory_service, "embedding_provider", None)
            vec = provider.embed("model preflight probe")
            as_list = [float(x) for x in vec if isinstance(x, (int, float))]
            embed_ok = bool(as_list)
            embed_detail = f"dim={len(as_list)}" if embed_ok else "empty_embedding_vector"
        except Exception as exc:
            embed_ok = False
            embed_detail = str(exc)[:180]
        checks.append(
            {
                "name": "embedding_probe",
                "required": True,
                "ok": bool(embed_ok),
                "detail": embed_detail,
            }
        )
        if not embed_ok:
            errors.append(f"embedding_probe_failed:{embed_detail}")

    if needs_agentic_llm:
        verifier = getattr(memory_service, "retrieval_verifier", None)
        verifier_ok, verifier_detail = _component_ready(verifier)
        checks.append(
            {
                "name": "agentic_verifier_config",
                "required": True,
                "ok": bool(verifier_ok),
                "detail": verifier_detail,
            }
        )
        if not verifier_ok:
            errors.append(f"agentic_verifier_not_ready:{verifier_detail}")

        rewriter = getattr(memory_service, "query_rewriter", None)
        rewriter_ok, rewriter_detail = _component_ready(rewriter)
        checks.append(
            {
                "name": "agentic_rewriter_config",
                "required": True,
                "ok": bool(rewriter_ok),
                "detail": rewriter_detail,
            }
        )
        if not rewriter_ok:
            errors.append(f"agentic_rewriter_not_ready:{rewriter_detail}")

    return {
        "ok": len(errors) == 0,
        "checks": checks,
        "errors": errors,
    }


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


def _tokenize_eval_text(text: str) -> set[str]:
    raw = _as_str(text).lower()
    if not raw:
        return set()
    return {
        t
        for t in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", raw)
        if t and len(t) >= 2
    }


def _token_overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _extract_name_like_tokens(text: str) -> set[str]:
    raw = _as_str(text)
    if not raw:
        return set()
    stop = {
        "when",
        "what",
        "where",
        "which",
        "why",
        "who",
        "how",
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "you",
        "your",
        "i",
        "we",
        "he",
        "she",
        "they",
        "this",
        "that",
        "these",
        "those",
        "the",
        "and",
        "but",
        "by",
    }
    out: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", raw):
        token = match.group(0).lower()
        if token in stop:
            continue
        out.add(token)
    return out


def _evaluate_expected_target_consistency(
    *,
    query: str,
    expected_message_ids: list[str],
    memories: list[dict[str, Any]],
) -> tuple[bool, str, float]:
    expected_set = {x for x in expected_message_ids if _as_str(x)}
    if not expected_set:
        return True, "no_expected_message_ids", 1.0
    texts: list[str] = []
    expected_senders: set[str] = set()
    for item in memories:
        if not isinstance(item, dict):
            continue
        msg_id = _as_str(item.get("message_id") or item.get("id"))
        if msg_id not in expected_set:
            continue
        content = _as_str(item.get("content") or item.get("text"))
        if content:
            texts.append(content)
        sender = _as_str(item.get("sender"))
        if sender:
            expected_senders.add(sender.lower())
    if not texts:
        return True, "expected_text_missing", 1.0
    merged = " ".join(texts)
    query_names = _extract_name_like_tokens(query)
    if query_names and expected_senders:
        if not any(name in expected_senders for name in query_names):
            text_lower = merged.lower()
            if not any(name in text_lower for name in query_names):
                qnames = ",".join(sorted(query_names)[:3])
                senders = ",".join(sorted(expected_senders)[:3])
                return False, f"sender_name_mismatch:q={qnames};sender={senders}", 0.0
    q_tokens = _tokenize_eval_text(query)
    t_tokens = _tokenize_eval_text(merged)
    overlap = _token_overlap_ratio(q_tokens, t_tokens)
    return True, "ok", overlap


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


def _load_ingested_message_event_map(
    *, db_path: Path, group_id: str
) -> dict[str, str]:
    if not db_path.exists():
        return {}
    mapping: dict[str, str] = {}
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            """
            SELECT source_message_id, event_id
            FROM episodic_memory
            WHERE is_deleted=0
              AND COALESCE(group_id, '') = ?
            ORDER BY timestamp DESC, updated_at DESC
            """,
            (str(group_id or ""),),
        )
        for src_id, event_id in cur.fetchall():
            mid = _normalize_message_id(src_id)
            eid = _as_str(event_id)
            if not mid or not eid:
                continue
            # Keep first seen mapping to stay stable across repeated runs.
            mapping.setdefault(mid, eid)
    except Exception:
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return mapping


def _lookup_event_ids_by_source_message_ids(
    *,
    db_path: Path,
    group_id: str,
    message_ids: list[str],
) -> dict[str, list[str]]:
    if not db_path.exists() or not message_ids:
        return {}
    conn: sqlite3.Connection | None = None
    out: dict[str, list[str]] = {}
    unique_ids = [mid for mid in dict.fromkeys(message_ids) if _as_str(mid)]
    if not unique_ids:
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        placeholders = ",".join("?" for _ in unique_ids)
        params: list[str] = [str(group_id or "")]
        params.extend(unique_ids)
        cur = conn.execute(
            f"""
            SELECT source_message_id, event_id
            FROM episodic_memory
            WHERE is_deleted=0
              AND COALESCE(group_id, '') = ?
              AND source_message_id IN ({placeholders})
            ORDER BY timestamp DESC, updated_at DESC
            """,
            tuple(params),
        )
        for src_id, event_id in cur.fetchall():
            mid = _normalize_message_id(src_id)
            eid = _as_str(event_id)
            if not mid or not eid:
                continue
            bucket = out.setdefault(mid, [])
            if eid not in bucket:
                bucket.append(eid)
    except Exception:
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return out


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
    db_path: Path,
    reuse_ingested: bool,
    skip_ingest_missing: bool,
) -> dict[str, Any]:
    case_started = time.perf_counter()
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

    if group_id not in ingest_cache:
        ingest_cache[group_id] = (
            _load_ingested_message_event_map(db_path=db_path, group_id=group_id)
            if reuse_ingested
            else {}
        )
    msg_to_event: dict[str, str] = dict(ingest_cache.get(group_id, {}))
    ingest_started = time.perf_counter()
    missing_ids = [
        row["message_id"] for row in memories if row["message_id"] not in msg_to_event
    ]
    ingest_skipped_missing = 0
    if missing_ids and not skip_ingest_missing:
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
    elif missing_ids and skip_ingest_missing:
        ingest_skipped_missing = len(missing_ids)
    ingest_ms = (time.perf_counter() - ingest_started) * 1000.0

    expected_event_ids = _as_str_list(case.get("expected_event_ids"))
    expected_msg_ids = _as_message_id_list(
        _as_str_list(
            case.get("expected_message_ids")
            or case.get("supporting_message_ids")
            or case.get("supporting_turn_ids")
        )
    )
    unresolved_ids: list[str] = []
    if not expected_event_ids:
        expected_map = _lookup_event_ids_by_source_message_ids(
            db_path=db_path,
            group_id=group_id,
            message_ids=expected_msg_ids,
        )
        for msg_id in expected_msg_ids:
            resolved = list(expected_map.get(msg_id, []))
            if not resolved:
                fallback = msg_to_event.get(msg_id)
                if fallback:
                    resolved.append(fallback)
            if resolved:
                expected_event_ids.extend(resolved)
            else:
                unresolved_ids.append(msg_id)

    if not expected_event_ids:
        return {
            "case_id": case_id,
            "status": "skipped",
            "reason": "no_expected_targets",
            "unresolved_message_ids": unresolved_ids,
        }
    valid_case, invalid_reason, target_overlap = _evaluate_expected_target_consistency(
        query=query,
        expected_message_ids=expected_msg_ids,
        memories=memories,
    )
    invalid_case = not valid_case

    params = {
        "query": query,
        "user_id": user_id if search_use_user_id else None,
        "group_id": group_id,
        "retrieve_method": method,
        "decision_mode": decision_mode,
        "top_k": top_k,
    }
    search_started = time.perf_counter()
    resp = client.get("/api/v1/memories/search", params=params)
    search_ms = (time.perf_counter() - search_started) * 1000.0
    if resp.status_code != 200:
        return {
            "case_id": case_id,
            "status": "failed",
            "reason": f"search_http_{resp.status_code}",
            "detail": _as_str(resp.text)[:200],
            "ingest_skipped_missing": ingest_skipped_missing,
            "ingest_ms": round(float(ingest_ms), 3),
            "search_ms": round(float(search_ms), 3),
            "case_ms": round(float((time.perf_counter() - case_started) * 1000.0), 3),
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
        "invalid_case": invalid_case,
        "invalid_reason": invalid_reason if invalid_case else "",
        "target_overlap": round(float(target_overlap), 4),
        "rank": rank,
        "hit": hit,
        "mrr_contrib": round(1.0 / rank, 8) if rank else 0.0,
        "ingest_memory_total": len(memories),
        "ingest_memory_new": len(missing_ids),
        "ingest_skipped_missing": ingest_skipped_missing,
        "ingest_ms": round(float(ingest_ms), 3),
        "search_ms": round(float(search_ms), 3),
        "case_ms": round(float((time.perf_counter() - case_started) * 1000.0), 3),
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
    reuse_ingested: bool,
    skip_ingest_missing: bool,
    skip_model_preflight: bool,
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
        "LITE_CONFIG_PATH": str(base_data_dir / "eval.config.json"),
        "LITE_DB_PATH": str(base_data_dir / "lite.db"),
        "LITE_LANCEDB_DIR": str(base_data_dir / "lancedb"),
        "LITE_GRAPH_DIR": str(base_data_dir / "kuzu"),
        "LITE_RETRIEVAL_PROFILE": ingest_profile,
        "LITE_GRAPH_ENABLED": "true" if graph_enabled else "false",
        "LITE_AGENT_POLICY_ENABLED": "false",
    }
    preflight: dict[str, Any] = {"ok": False, "checks": [], "errors": []}
    try:
        with _temporary_env(env_patch):
            app = create_app(LiteSettings.from_env())
            db_path = Path(base_data_dir / "lite.db")
            preflight = _run_model_preflight(app=app, method=method)
            if not skip_model_preflight and not bool(preflight.get("ok")):
                reasons = "; ".join(str(x) for x in preflight.get("errors", []))
                raise RuntimeError(f"model preflight failed: {reasons}")
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
                        db_path=db_path,
                        reuse_ingested=reuse_ingested,
                        skip_ingest_missing=skip_ingest_missing,
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
    evaluated_valid = [x for x in evaluated if not bool(x.get("invalid_case"))]
    invalid = [x for x in evaluated if bool(x.get("invalid_case"))]
    skipped = [x for x in case_details if x.get("status") == "skipped"]
    failed = [x for x in case_details if x.get("status") == "failed"]

    recall_hits_raw = sum(1 for x in evaluated if bool(x.get("hit")))
    mrr_sum_raw = sum(float(x.get("mrr_contrib", 0.0)) for x in evaluated)
    n_raw = len(evaluated)
    recall_hits = sum(1 for x in evaluated_valid if bool(x.get("hit")))
    mrr_sum = sum(float(x.get("mrr_contrib", 0.0)) for x in evaluated_valid)
    n = len(evaluated_valid)
    ingest_ms_sum = sum(float(x.get("ingest_ms", 0.0)) for x in evaluated)
    search_ms_sum = sum(float(x.get("search_ms", 0.0)) for x in evaluated)
    case_ms_sum = sum(float(x.get("case_ms", 0.0)) for x in evaluated)
    ingested_total = sum(int(x.get("ingest_memory_total", 0)) for x in evaluated)
    ingested_new = sum(int(x.get("ingest_memory_new", 0)) for x in evaluated)
    ingested_skipped = sum(int(x.get("ingest_skipped_missing", 0)) for x in evaluated)

    return {
        "status": "ok",
        "dataset": str(dataset),
        "data_dir": str(base_data_dir),
        "model_preflight": preflight,
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
        "reuse_ingested": reuse_ingested,
        "skip_ingest_missing": skip_ingest_missing,
        "ingested_groups": len(ingest_cache),
        "ingested_memory_total": ingested_total,
        "ingested_memory_new": ingested_new,
        "ingested_missing_skipped": ingested_skipped,
        "ms_ingest_total": round(ingest_ms_sum, 3),
        "ms_search_total": round(search_ms_sum, 3),
        "ms_case_total": round(case_ms_sum, 3),
        "ms_case_avg": round(case_ms_sum / n_raw, 3) if n_raw else 0.0,
        "total_cases": len(case_details),
        "evaluated_cases": n_raw,
        "evaluated_cases_valid": n,
        "skipped_cases": len(skipped),
        "failed_cases": len(failed),
        "invalid_cases": len(invalid),
        "recall_at_k": round(recall_hits / n, 4) if n else 0.0,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "recall_at_k_raw": round(recall_hits_raw / n_raw, 4) if n_raw else 0.0,
        "mrr_raw": round(mrr_sum_raw / n_raw, 4) if n_raw else 0.0,
        "cases": case_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end LoCoMo retrieval evaluation in local FlockMem runtime."
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
        help=(
            "Optional persistent LITE_DATA_DIR. "
            "Default is auto cache dir by dataset fingerprint."
        ),
    )
    parser.add_argument(
        "--temp-data-dir",
        action="store_true",
        help="Force temp LITE_DATA_DIR for one-off run (disables persistent cache).",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional path to save full JSON report.",
    )
    parser.add_argument(
        "--no-reuse-ingested",
        action="store_true",
        help="Disable reuse of already ingested message_id->event_id mappings from existing lite.db.",
    )
    parser.add_argument(
        "--skip-ingest-missing",
        action="store_true",
        help="Do not ingest missing message_ids; search only over already persisted memories.",
    )
    parser.add_argument(
        "--skip-model-preflight",
        action="store_true",
        help="Skip model availability checks before evaluation.",
    )
    args = parser.parse_args()

    _load_dotenv_defaults(ROOT_DIR / ".env")

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    data_dir: Path | None
    if args.temp_data_dir:
        data_dir = None
    elif args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = _default_eval_cache_dir(
            dataset=dataset,
            ingest_profile=args.ingest_profile,
            graph_enabled=bool(args.graph_enabled),
        )

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
        data_dir=data_dir,
        reuse_ingested=not bool(args.no_reuse_ingested),
        skip_ingest_missing=bool(args.skip_ingest_missing),
        skip_model_preflight=bool(args.skip_model_preflight),
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


