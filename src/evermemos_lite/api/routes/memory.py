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

    if policy.vector_enabled and result.get("memory"):
        try:
            await anyio.to_thread.run_sync(
                memory_service.maybe_index_vector, policy, result["memory"]
            )
        except Exception:
            pass

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
            "scene_id": memory.get("scene_id") or result.get("scene_id"),
            "content_slices": content_slices,
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
) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit out of range")
    memory_service = request.app.state.memory_service
    episodes = await anyio.to_thread.run_sync(memory_service.fetch, user_id, group_id, limit)
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
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be blank")
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k out of range")

    settings = request.app.state.settings
    resolver = request.app.state.policy_resolver
    memory_service = request.app.state.memory_service
    rule_selector = request.app.state.rule_retrieval_mode_selector
    agent_selector = request.app.state.agent_retrieval_mode_selector

    patch = RuntimePolicy()
    if retrieve_method == "keyword":
        patch = RuntimePolicy(vector_enabled=False, keyword_enabled=True)
    elif retrieve_method == "vector":
        patch = RuntimePolicy(vector_enabled=True, keyword_enabled=False)
    elif retrieve_method in ("hybrid", "rrf"):
        patch = RuntimePolicy(vector_enabled=True, keyword_enabled=True)
    elif retrieve_method == "agentic":
        patch = RuntimePolicy(vector_enabled=True, keyword_enabled=True, agentic_enabled=True)

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
