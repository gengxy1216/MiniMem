from __future__ import annotations

import time
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Literal

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from evermemos_lite.config.profiles import PROFILE_PRESETS
from evermemos_lite.domain.policy import RuntimePolicy
from evermemos_lite.service.memory_service import MemorizeInput
from evermemos_lite.service.policy_resolver import ResolveInput
from evermemos_lite.service.retrieval_mode_selector import SelectionInput

RetrieveMethod = Literal["keyword", "vector", "hybrid", "rrf", "agentic"]
DecisionMode = Literal["static", "rule", "agent"]

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


class MemorizeRequest(BaseModel):
    message_id: str
    create_time: str | int
    sender: str
    content: str = Field(min_length=1, max_length=10000)
    group_id: str | None = None
    group_name: str | None = None
    sender_name: str | None = None
    role: str | None = "user"

    @field_validator("message_id", "sender")
    @classmethod
    def _validate_required_ids(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("must not be empty")
        if len(v) > 128:
            raise ValueError("too long")
        return v

    @field_validator("group_id", "group_name", "sender_name", "role")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        return v or None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("content must not be blank")
        return v


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    user_id: str | None = None
    group_id: str | None = None
    retrieve_method: RetrieveMethod = "keyword"
    top_k: int = Field(default=20, ge=1, le=100)
    request_override: dict[str, Any] | None = None


def _build_retrieve_method_patch(retrieve_method: RetrieveMethod) -> RuntimePolicy:
    if retrieve_method == "keyword":
        return RuntimePolicy(
            vector_enabled=False,
            keyword_enabled=True,
            agentic_enabled=False,
        )
    if retrieve_method == "vector":
        return RuntimePolicy(
            vector_enabled=True,
            keyword_enabled=False,
            agentic_enabled=False,
        )
    if retrieve_method in ("hybrid", "rrf"):
        return RuntimePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=False,
        )
    if retrieve_method == "agentic":
        return RuntimePolicy(
            vector_enabled=True,
            keyword_enabled=True,
            agentic_enabled=True,
        )
    return RuntimePolicy()


def _to_unix(value: str | int) -> int:
    if isinstance(value, int):
        ts = value
    else:
        try:
            ts = int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid create_time: {exc}") from exc
    now = int(time.time())
    if ts <= 0 or ts > now + 315360000:
        raise HTTPException(status_code=400, detail="invalid create_time range")
    return ts


def _slice_content(content: str, max_len: int = 240, max_parts: int = 12) -> list[str]:
    parts = [seg.strip() for seg in content.split("\n\n") if seg.strip()]
    if not parts:
        stripped = content.strip()
        parts = [stripped] if stripped else []
    out: list[str] = []
    for part in parts:
        if len(part) <= max_len:
            out.append(part)
        else:
            for idx in range(0, len(part), max_len):
                out.append(part[idx : idx + max_len])
        if len(out) >= max_parts:
            return out[:max_parts]
    return out[:max_parts]


def _validate_time_filters(
    *,
    as_of_time: int | None,
    start_time: int | None,
    end_time: int | None,
) -> tuple[int | None, int | None, int | None]:
    as_of = int(as_of_time) if as_of_time is not None else None
    start = int(start_time) if start_time is not None else None
    end = int(end_time) if end_time is not None else None
    if as_of is not None and as_of <= 0:
        raise HTTPException(status_code=400, detail="as_of_time out of range")
    if start is not None and start <= 0:
        raise HTTPException(status_code=400, detail="start_time out of range")
    if end is not None and end <= 0:
        raise HTTPException(status_code=400, detail="end_time out of range")
    if as_of is not None and (start is not None or end is not None):
        raise HTTPException(
            status_code=400,
            detail="as_of_time cannot be combined with start_time/end_time",
        )
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start_time cannot be greater than end_time")
    return as_of, start, end


@router.post("")
async def memorize(payload: MemorizeRequest, request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    resolver = request.app.state.policy_resolver
    memory_service = request.app.state.memory_service
    status_repo = request.app.state.request_status_repo

    request_id = uuid.uuid4().hex
    now = int(time.time())
    await anyio.to_thread.run_sync(
        partial(
            status_repo.upsert,
            request_id=request_id,
            status="start",
            ttl_sec=settings.request_status_ttl_sec,
            url=str(request.url.path),
            method="POST",
            start_time=now,
        )
    )

    start_perf = time.perf_counter()
    group_id = payload.group_id or f"default:{payload.sender}"
    create_ts = _to_unix(payload.create_time)
    policy = await anyio.to_thread.run_sync(
        resolver.resolve,
        ResolveInput(default_profile=settings.retrieval_profile, tenant_id="default"),
    )
    try:
        result = await anyio.to_thread.run_sync(
            memory_service.memorize,
            MemorizeInput(
                message_id=payload.message_id,
                create_time=create_ts,
                sender=payload.sender,
                content=payload.content,
                group_id=group_id,
                group_name=payload.group_name,
                sender_name=payload.sender_name,
                role=payload.role or "user",
            ),
            request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    vector_index_result: dict[str, Any] = {"status": "skipped", "reason": "not_attempted"}
    if policy.vector_enabled and result.get("memory"):
        vector_index_result = await anyio.to_thread.run_sync(
            memory_service.maybe_index_vector, policy, result["memory"]
        )

    elapsed_ms = int((time.perf_counter() - start_perf) * 1000)
    await anyio.to_thread.run_sync(
        partial(
            status_repo.upsert,
            request_id=request_id,
            status="success",
            ttl_sec=settings.request_status_ttl_sec,
            url=str(request.url.path),
            method="POST",
            http_code=200,
            time_ms=elapsed_ms,
            start_time=now,
            end_time=int(time.time()),
        )
    )
    memory = result.get("memory") if isinstance(result, dict) else {}
    memory = memory if isinstance(memory, dict) else {}
    event_id = str(memory.get("event_id") or request_id)
    content_slices = _slice_content(payload.content)

    return {
        "status": "ok",
        "message": "memory written",
        "result": {
            "success": True,
            "message_id": payload.message_id,
            "sender": payload.sender,
            "group_id": group_id,
            "event_id": event_id,
            "write_time": int(memory.get("timestamp") or create_ts),
            "summary": str(result.get("summary", "")),
            "subject": str(result.get("subject", "")),
            "importance_score": float(result.get("importance_score", 0.0)),
            "storage_tier": str(result.get("storage_tier") or memory.get("storage_tier") or "text_only"),
            "memory_category": str(
                result.get("memory_category") or memory.get("memory_category") or "general"
            ),
            "scene_id": memory.get("scene_id") or result.get("scene_id"),
            "content_slices": content_slices,
            "vector_index": vector_index_result,
            "memory": memory,
        },
        "request_id": request_id,
    }


@router.get("")
async def fetch_memories(
    request: Request,
    user_id: str | None = None,
    group_id: str | None = None,
    limit: int = 40,
    as_of_time: int | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit out of range")
    as_of_ts, start_ts, end_ts = _validate_time_filters(
        as_of_time=as_of_time,
        start_time=start_time,
        end_time=end_time,
    )
    memory_service = request.app.state.memory_service
    episodes = await anyio.to_thread.run_sync(
        partial(
            memory_service.fetch,
            user_id=user_id,
            group_id=group_id,
            limit=limit,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    )
    conflicts = await anyio.to_thread.run_sync(
        partial(
            memory_service.repo.list_recent_conflicts,
            user_id=user_id,
            group_id=group_id,
            limit=20,
        )
    )
    profile = await anyio.to_thread.run_sync(
        memory_service.get_profile_snapshot, user_id, group_id
    )
    return {
        "status": "ok",
        "message": f"Memory retrieval successful, retrieved {len(episodes)} memories",
        "result": {
            "memories": episodes,
            "total_count": len(episodes),
            "has_more": False,
            "conflicts": conflicts,
            "profile": profile,
        },
    }


@router.get("/search")
async def search_memories(
    request: Request,
    query: str,
    user_id: str | None = None,
    group_id: str | None = None,
    retrieve_method: RetrieveMethod = "keyword",
    decision_mode: DecisionMode = "static",
    runtime_profile: str | None = None,
    top_k: int = 20,
    as_of_time: int | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be blank")
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k out of range")
    as_of_ts, start_ts, end_ts = _validate_time_filters(
        as_of_time=as_of_time,
        start_time=start_time,
        end_time=end_time,
    )

    settings = request.app.state.settings
    resolver = request.app.state.policy_resolver
    memory_service = request.app.state.memory_service
    rule_selector = request.app.state.rule_retrieval_mode_selector
    agent_selector = request.app.state.agent_retrieval_mode_selector

    patch = _build_retrieve_method_patch(retrieve_method)

    if runtime_profile:
        if runtime_profile not in PROFILE_PRESETS:
            raise HTTPException(status_code=400, detail="invalid runtime_profile")
        patch.profile = runtime_profile
        patch.reason = "manual.runtime_profile"

    if decision_mode in ("rule", "agent"):
        selector = rule_selector if decision_mode == "rule" else agent_selector
        selection = await anyio.to_thread.run_sync(
            selector.select,
            SelectionInput(query=query, top_k=top_k, user_id=user_id, group_id=group_id),
        )
        patch.profile = selection.policy.profile or patch.profile
        patch.reason = selection.policy.reason or patch.reason

    effective = await anyio.to_thread.run_sync(
        resolver.resolve,
        ResolveInput(
            default_profile=settings.retrieval_profile,
            tenant_id="default",
            request_override=patch,
        ),
    )
    hits = await anyio.to_thread.run_sync(
        partial(
            memory_service.search,
            policy=effective,
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    )
    conflicts = await anyio.to_thread.run_sync(
        partial(
            memory_service.repo.list_recent_conflicts,
            user_id=user_id,
            group_id=group_id,
            limit=20,
        )
    )
    profile = await anyio.to_thread.run_sync(
        memory_service.get_profile_snapshot, user_id, group_id
    )
    return {
        "status": "ok",
        "message": f"Memory search successful, retrieved {len(hits)} groups",
        "result": {
            "memories": hits,
            "effective_policy": effective.to_dict(),
            "decision_mode": decision_mode,
            "runtime_profile": runtime_profile,
            "conflicts": conflicts,
            "profile": profile,
        },
    }


@router.delete("")
async def delete_memory(request: Request, event_id: str) -> dict[str, Any]:
    memory_service = request.app.state.memory_service
    deleted = await anyio.to_thread.run_sync(memory_service.repo.delete_by_event_id, event_id)
    return {
        "status": "ok",
        "message": f"Deleted {deleted} memory records",
        "result": {"deleted_count": deleted},
    }
