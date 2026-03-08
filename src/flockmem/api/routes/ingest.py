from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from functools import partial
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from flockmem.service.memory_service import MemorizeInput
from flockmem.service.policy_resolver import ResolveInput

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


def _to_unix(value: str | int | None) -> int:
    if value is None:
        return int(time.time())
    if isinstance(value, int):
        ts = value
    else:
        try:
            ts = int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid create_time: {exc}") from exc
    now = int(time.time())
    if ts <= 0 or ts > now + 315360000:
        raise HTTPException(status_code=400, detail="invalid create_time range")
    return ts


def _parse_skill_whitelist(raw: str) -> set[str]:
    parts = [str(x or "").strip().lower() for x in str(raw or "").split(",")]
    return {x for x in parts if x}


def _join_skill_content(*, metadata: dict[str, Any], text: str) -> str:
    meta_line = "[metadata] " + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    body = str(text or "").strip()
    if not body:
        return meta_line
    return f"{meta_line}\n{body}"


class IngestChunk(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    metadata: dict[str, Any] | None = None

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("chunk text must not be blank")
        return text


class SkillIngestRequest(BaseModel):
    source_type: str = "skill_output"
    source_uri: str | None = None
    raw_text: str | None = None
    summary: str | None = None
    chunks: list[IngestChunk | str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None
    skill_version: str | None = None
    agent_id: str | None = None
    sender: str | None = None
    group_id: str | None = None
    task_id: str | None = None
    channel: str | None = None
    trace_id: str | None = None
    role: str = "user"
    create_time: str | int | None = None

    @field_validator(
        "source_type",
        "source_uri",
        "raw_text",
        "summary",
        "skill_name",
        "skill_version",
        "agent_id",
        "sender",
        "group_id",
        "task_id",
        "channel",
        "trace_id",
        "role",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}


def _collect_ingest_parts(payload: SkillIngestRequest) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if payload.summary:
        out.append((str(payload.summary).strip(), {"part_type": "summary"}))
    if payload.raw_text:
        out.append((str(payload.raw_text).strip(), {"part_type": "raw_text"}))
    for idx, chunk in enumerate(list(payload.chunks or []), start=1):
        if isinstance(chunk, str):
            text = str(chunk).strip()
            if text:
                out.append((text, {"part_type": "chunk", "chunk_index": idx}))
            continue
        if isinstance(chunk, IngestChunk):
            text = str(chunk.text or "").strip()
            if not text:
                continue
            meta = dict(chunk.metadata or {})
            meta["part_type"] = "chunk"
            meta["chunk_index"] = idx
            out.append((text, meta))
    return out


@router.post("/skill")
async def ingest_skill_output(payload: SkillIngestRequest, request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    skill_name = str(payload.skill_name or "").strip().lower()
    whitelist = _parse_skill_whitelist(settings.skill_adapter_whitelist)

    sender = str(payload.sender or payload.agent_id or "skill-agent").strip()
    group_id = str(payload.group_id or f"default:{sender}").strip()
    trace_id = str(payload.trace_id or f"trace-{uuid.uuid4().hex[:16]}").strip()
    create_ts = _to_unix(payload.create_time)

    if skill_name:
        if not bool(settings.skill_adapter_enabled):
            return {
                "status": "ok",
                "message": "skill adapter is disabled",
                "result": {
                    "accepted": False,
                    "hint": "set LITE_SKILL_ADAPTER_ENABLED=true to enable skill ingest",
                    "skill_name": skill_name,
                    "trace_id": trace_id,
                },
            }
        if whitelist and skill_name not in whitelist:
            allow = ", ".join(sorted(whitelist))
            return {
                "status": "ok",
                "message": "skill is not in whitelist",
                "result": {
                    "accepted": False,
                    "hint": f"allowed skills: {allow}",
                    "skill_name": skill_name,
                    "trace_id": trace_id,
                },
            }

    parts = _collect_ingest_parts(payload)
    if not parts:
        raise HTTPException(status_code=400, detail="at least one of summary/raw_text/chunks is required")

    base_meta = dict(payload.metadata or {})
    base_meta.update(
        {
            "source_type": str(payload.source_type or "skill_output"),
            "source_uri": payload.source_uri,
            "skill_name": payload.skill_name,
            "skill_version": payload.skill_version,
            "agent_id": payload.agent_id,
            "sender": sender,
            "group_id": group_id,
            "task_id": payload.task_id,
            "channel": payload.channel,
            "trace_id": trace_id,
        }
    )

    resolver = request.app.state.policy_resolver
    memory_service = request.app.state.memory_service
    event_ids: list[str] = []
    request_id_prefix = f"ingest-{trace_id}"
    policy = await anyio.to_thread.run_sync(
        resolver.resolve,
        ResolveInput(default_profile=settings.retrieval_profile, tenant_id="default"),
    )
    for idx, (text, part_meta) in enumerate(parts, start=1):
        merged_meta = dict(base_meta)
        merged_meta.update(part_meta)
        merged_meta["part_index"] = idx
        merged_meta["part_total"] = len(parts)
        content = _join_skill_content(metadata=merged_meta, text=text)
        message_id = f"{request_id_prefix}-m{idx}"
        result = await anyio.to_thread.run_sync(
            memory_service.memorize,
            MemorizeInput(
                message_id=message_id,
                create_time=create_ts + idx - 1,
                sender=sender,
                content=content,
                group_id=group_id,
                group_name=None,
                sender_name=sender,
                role=str(payload.role or "user"),
            ),
            f"{request_id_prefix}-r{idx}",
        )
        memory = result.get("memory") if isinstance(result, dict) else {}
        if isinstance(memory, dict) and memory:
            event_id = str(memory.get("event_id") or "").strip()
            if event_id:
                event_ids.append(event_id)
            if policy.vector_enabled:
                await anyio.to_thread.run_sync(memory_service.maybe_index_vector, policy, memory)

    return {
        "status": "ok",
        "message": "skill ingest completed",
        "result": {
            "accepted": True,
            "skill_name": payload.skill_name,
            "trace_id": trace_id,
            "sender": sender,
            "group_id": group_id,
            "ingested_count": len(parts),
            "event_ids": event_ids,
            "source_type": str(payload.source_type or "skill_output"),
        },
    }

