from __future__ import annotations

import re
import time
from datetime import datetime
from functools import partial
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from evermemos_lite.service.policy_resolver import ResolveInput
from evermemos_lite.service.memory_service import ChatTurnInput

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    user_id: str | None = None
    group_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=30)
    provider: str | None = None
    conversation_id: str | None = None


_GREETING_PATTERNS = (
    r"^\s*(hi|hello|hey|yo|hola|你好|您好|嗨|哈喽|在吗|在不在|早上好|下午好|晚上好)[!！,.。~\s]*$",
)
_MEMORY_HINT_TERMS = (
    "记得",
    "还记得",
    "回忆",
    "之前",
    "上次",
    "以前",
    "历史",
    "profile",
    "memory",
    "remember",
    "before",
    "previous",
    "past",
    "history",
    "名字",
    "姓名",
    "我叫",
    "叫什么",
    "name",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}")
_IDENTITY_PATTERNS = (
    r"我叫.*什么",
    r"我的名字",
    r"我是谁",
    r"who am i",
    r"what(?:'s| is) my name",
)
_AGE_QUERY_TERMS = ("几岁", "多大", "年龄", "age", "old")
_AGE_SUBJECT_TERMS = (
    "我",
    "我的",
    "儿子",
    "女儿",
    "孩子",
    "宝宝",
    "son",
    "daughter",
    "child",
    "kid",
)
_MIN_IMPORTANCE = 0.45
_MIN_RELEVANCE = 0.12
_MIN_COMBINED_SCORE = 0.36
_MAX_MEMORY_CONTEXT = 4


def _is_smalltalk(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    if len(q) <= 6 and any(re.match(p, q, flags=re.IGNORECASE) for p in _GREETING_PATTERNS):
        return True
    return False


def _has_memory_hint(query: str) -> bool:
    q = query.lower()
    if any(term in q for term in _MEMORY_HINT_TERMS):
        return True
    return _is_identity_query(q) or _is_age_query(q)


def _is_identity_query(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in _IDENTITY_PATTERNS)


def _is_age_query(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    return any(term in q for term in _AGE_QUERY_TERMS) and any(
        term in q for term in _AGE_SUBJECT_TERMS
    )


def _query_tokens(query: str) -> set[str]:
    q = query.lower()
    tokens = {tok for tok in _TOKEN_RE.findall(q) if tok.strip()}
    chars = re.findall(r"[\u4e00-\u9fff]", q)
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])
    if chars:
        tokens.add("".join(chars))
    stop_words = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
    return {tok for tok in tokens if tok and tok not in stop_words}


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("summary") or ""),
            str(row.get("episode") or ""),
            str(row.get("subject") or ""),
            str(row.get("atomic_fact_text") or ""),
        ]
    ).lower()


def _lexical_relevance(query: str, row: dict[str, Any]) -> float:
    text = _row_text(row)
    if not text:
        return 0.0
    q = query.lower().strip()
    if q and q in text:
        return 1.0
    tokens = _query_tokens(query)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in text)
    return float(hits) / float(max(1, len(tokens)))


def _identity_relevance(row: dict[str, Any]) -> float:
    text = _row_text(row)
    if not text:
        return 0.0
    indicators = ("我叫", "名字", "姓名", "name_is", "name")
    if any(token in text for token in indicators):
        return 1.0
    return 0.0


def _age_relevance(row: dict[str, Any]) -> float:
    text = _row_text(row)
    if not text:
        return 0.0
    has_age = bool(re.search(r"(\d{1,3}\s*岁|age_is|年龄)", text))
    has_family = any(
        token in text
        for token in ("儿子", "女儿", "孩子", "has_son", "has_daughter", "has_child")
    )
    if has_age and has_family:
        return 1.0
    if has_age:
        return 0.75
    if has_family:
        return 0.55
    return 0.0


def _importance_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("importance_score", 0.0))
    except Exception:
        return 0.0


def _filter_memories_for_prompt(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    has_hint = _has_memory_hint(query)
    is_identity = _is_identity_query(query)
    is_age = _is_age_query(query)
    if _is_smalltalk(query) and not has_hint:
        return []
    if is_identity or is_age:
        min_importance = 0.05
        min_relevance = 0.0
        min_combined = 0.05
    else:
        min_importance = 0.30 if has_hint else _MIN_IMPORTANCE
        min_relevance = 0.05 if has_hint else _MIN_RELEVANCE
        min_combined = 0.24 if has_hint else _MIN_COMBINED_SCORE

    picked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        imp = _importance_score(row)
        rel = _lexical_relevance(query, row)
        if is_identity:
            rel = max(rel, _identity_relevance(row))
        if is_age:
            rel = max(rel, _age_relevance(row))
        combined = imp * 0.65 + rel * 0.35
        if imp < min_importance and rel < min_relevance:
            continue
        if combined < min_combined:
            continue
        picked.append((combined, row))
    picked.sort(
        key=lambda item: (
            item[0],
            _importance_score(item[1]),
            float(item[1].get("timestamp", 0.0)),
        ),
        reverse=True,
    )
    if picked:
        return [row for _, row in picked[:_MAX_MEMORY_CONTEXT]]
    if has_hint:
        # For explicit memory-intent queries, provide a small fallback context.
        by_importance = sorted(
            rows,
            key=lambda row: (
                _identity_relevance(row),
                _age_relevance(row),
                _importance_score(row),
            ),
            reverse=True,
        )
        return by_importance[: min(2, _MAX_MEMORY_CONTEXT)]
    return []


@router.post("/simple")
async def simple_chat(payload: ChatRequest, request: Request) -> dict:
    settings = request.app.state.settings
    resolver = request.app.state.policy_resolver
    memory_service = request.app.state.memory_service
    chat_responder = request.app.state.chat_responder
    model_config = request.app.state.runtime_model_config

    effective = resolver.resolve(
        ResolveInput(default_profile=settings.retrieval_profile, tenant_id="default")
    )
    raw_memories: list[dict[str, Any]] = []
    if not _is_smalltalk(payload.query) or _has_memory_hint(payload.query):
        raw_memories = memory_service.search(
            policy=effective,
            query=payload.query,
            user_id=payload.user_id,
            group_id=payload.group_id,
            top_k=payload.top_k,
        )
    live_segment_memories: list[dict[str, Any]] = []
    if payload.conversation_id:
        live_segment_memories = memory_service.retrieve_live_segment_context(
            conversation_id=str(payload.conversation_id),
            query=payload.query,
            limit=min(2, payload.top_k),
        )
    if live_segment_memories:
        dedup: set[str] = set()
        merged: list[dict[str, Any]] = []
        for row in [*live_segment_memories, *raw_memories]:
            key = str(row.get("id") or row.get("event_id") or row.get("summary") or row.get("episode"))
            if not key or key in dedup:
                continue
            dedup.add(key)
            merged.append(row)
        raw_memories = merged
    memories = _filter_memories_for_prompt(payload.query, raw_memories)
    provider_options = (
        model_config.get("chat_provider_options")
        if isinstance(model_config.get("chat_provider_options"), dict)
        else {}
    )
    default_provider = (
        str(model_config.get("chat_provider") or "openai").strip() or "openai"
    )
    active_provider = str(payload.provider or default_provider).strip() or default_provider
    if provider_options and active_provider not in provider_options:
        if default_provider in provider_options:
            active_provider = default_provider
        else:
            active_provider = next(iter(provider_options.keys()), "openai")
    try:
        response = chat_responder.respond(
            query=payload.query,
            memories=memories,
            system_time=datetime.now().astimezone(),
            provider=active_provider,
            provider_options=provider_options,
            model=str(model_config.get("chat_model", "")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    auto_memory_saved = False
    auto_memory_error: str | None = None
    boundary_detected = False
    user_id = (payload.user_id or "").strip() or "anonymous"
    group_id = (payload.group_id or "").strip() or f"default:{user_id}"
    conversation_id = (
        str(payload.conversation_id or "").strip() or f"session-{user_id}-{group_id}"
    )
    try:
        segment_result = await anyio.to_thread.run_sync(
            partial(
                memory_service.append_chat_turn,
                payload=ChatTurnInput(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    group_id=group_id,
                    user_text=payload.query,
                    assistant_text=str(response.get("answer", "")),
                    timestamp=int(time.time()),
                ),
                policy=effective,
            )
        )
        auto_memory_saved = bool(segment_result.get("memory_saved"))
        auto_memory_error = segment_result.get("memory_error")
        boundary_detected = bool(segment_result.get("boundary_detected"))
    except Exception as exc:  # best effort: chat should not fail because of memory persistence
        auto_memory_error = str(exc)

    title = payload.query.strip().replace("\n", " ")[:80] or "New Chat"
    try:
        request.app.state.conversation_meta_repo.upsert(
            user_id=user_id,
            group_id=group_id,
            title=title,
            conversation_id=conversation_id,
        )
    except Exception:
        pass
    return {
        "status": "ok",
        "result": {
            "answer": response["answer"],
            "citations": response["citations"],
            "provider": response.get("provider", active_provider),
            "model": response["model"],
            "effective_policy": effective.to_dict(),
            "memory_filter": {
                "retrieved_count": len(raw_memories),
                "used_count": len(memories),
                "smalltalk_bypass": _is_smalltalk(payload.query) and not _has_memory_hint(payload.query),
                "live_segment_count": len(live_segment_memories),
            },
            "memory_saved": auto_memory_saved,
            "memory_error": auto_memory_error,
            "boundary_detected": boundary_detected,
            "conversation_id": conversation_id,
        },
    }
