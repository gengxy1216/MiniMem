from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from fastmcp import FastMCP

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
src_path = str(SRC_DIR)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from flockmem.service.runtime_profile_service import (
    allowed_coordination_modes,
    allowed_runtime_profiles,
    build_collective_envelope,
    validate_adapter_payload,
)


MCP_NAME = "flockmem-mcp"
BASE_URL = os.getenv("MINIMEM_BASE_URL", "http://127.0.0.1:20195").strip().rstrip("/")
TIMEOUT_SEC = float(os.getenv("MINIMEM_TIMEOUT_SEC", "25"))
DEFAULT_USER_ID = os.getenv("MINIMEM_USER_ID", "").strip() or None
DEFAULT_GROUP_ID = os.getenv("MINIMEM_GROUP_ID", "").strip() or None
BEARER_TOKEN = os.getenv("MINIMEM_BEARER_TOKEN", "").strip()
BASIC_USER = os.getenv("MINIMEM_BASIC_USER", "").strip()
BASIC_PASSWORD = os.getenv("MINIMEM_BASIC_PASSWORD", "").strip()

ALLOWED_RETRIEVE_METHODS = {"keyword", "vector", "hybrid", "rrf", "agentic"}
ALLOWED_DECISION_MODES = {"static", "rule", "agent"}
ALLOWED_RUNTIME_PROFILES = allowed_runtime_profiles()
ALLOWED_SOURCE_TYPES = {
    "skill_output",
    "text",
    "markdown",
    "pdf",
    "pptx",
    "docx",
    "html",
    "json",
}
ALLOWED_SCOPE_TYPES = {"personal", "team", "global"}
ALLOWED_CHANGE_TYPES = {"create", "update", "deprecate", "rollback"}
ALLOWED_CHANGED_BY = {"agent", "user", "system"}
ALLOWED_COORDINATION_MODES = allowed_coordination_modes()

mcp = FastMCP(
    name=MCP_NAME,
    instructions=(
        "Bridge tools for FlockMem HTTP API. "
        "Use these tools for memory search, memory write, chat retrieval, and graph lookups."
    ),
)


def _auth_header() -> dict[str, str]:
    if BEARER_TOKEN:
        return {"Authorization": f"Bearer {BEARER_TOKEN}"}
    if BASIC_USER:
        raw = f"{BASIC_USER}:{BASIC_PASSWORD}".encode("utf-8")
        token = base64.b64encode(raw).decode("ascii")
        return {"Authorization": f"Basic {token}"}
    return {}


def _request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    if params:
        query = parse.urlencode(
            {k: v for k, v in params.items() if v is not None and v != ""},
            doseq=True,
        )
        if query:
            url = f"{url}?{query}"

    payload_bytes = (
        json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    )
    headers = {"Content-Type": "application/json", **_auth_header()}
    req = request.Request(url=url, data=payload_bytes, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"FlockMem API HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"FlockMem API request failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("response is not a JSON object")
        return parsed
    except Exception as exc:
        raise RuntimeError(f"FlockMem API returned invalid JSON: {raw[:280]}") from exc


def _resolved_scope(user_id: str | None, group_id: str | None) -> tuple[str | None, str | None]:
    return (user_id or DEFAULT_USER_ID, group_id or DEFAULT_GROUP_ID)


def _normalize_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _collective_envelope(
    *,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    normalized = build_collective_envelope(
        coordination_mode=coordination_mode,
        coordination_id=coordination_id,
        runtime_id=runtime_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        team_id=team_id,
        session_id=session_id,
        strict_coordination_mode=True,
    )
    if normalized.get("runtime_id") is None:
        normalized["runtime_id"] = "mcp"
    return normalized


def _merge_envelope_metadata(
    metadata: dict[str, Any] | None,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata or {})
    for key, value in envelope.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _attach_metadata_line(content: str, metadata: dict[str, Any]) -> str:
    if not metadata:
        return content
    meta_line = "[metadata] " + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    body = str(content or "").strip()
    if not body:
        return meta_line
    if meta_line in body:
        return body
    return f"{body}\n{meta_line}"


@mcp.tool(description="Health check for FlockMem service.")
def minimem_health() -> dict[str, Any]:
    return _request_json("GET", "/health")


@mcp.tool(description="Search memories with FlockMem hybrid/vector/keyword/agentic retrieval.")
def search_memories(
    query: str,
    top_k: int = 20,
    retrieve_method: str = "agentic",
    decision_mode: str = "rule",
    runtime_profile: str | None = None,
    user_id: str | None = None,
    group_id: str | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    method = retrieve_method.strip().lower()
    if method not in ALLOWED_RETRIEVE_METHODS:
        raise ValueError(f"invalid retrieve_method: {retrieve_method}")
    mode = decision_mode.strip().lower()
    if mode not in ALLOWED_DECISION_MODES:
        raise ValueError(f"invalid decision_mode: {decision_mode}")
    validated = validate_adapter_payload(
        "mcp",
        {"runtime_profile": runtime_profile},
        strict_runtime_profile=True,
    )
    profile = validated.get("runtime_profile")
    uid, gid = _resolved_scope(user_id, group_id)
    envelope = _collective_envelope(
        coordination_mode=coordination_mode,
        coordination_id=coordination_id,
        runtime_id=runtime_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        team_id=team_id,
        session_id=session_id,
    )
    return _request_json(
        "GET",
        "/api/v1/memories/search",
        params={
            "query": query,
            "user_id": uid,
            "group_id": gid,
            "retrieve_method": method,
            "decision_mode": mode,
            "runtime_profile": profile,
            "top_k": max(1, min(100, int(top_k))),
            **envelope,
        },
    )


@mcp.tool(description="Chat with FlockMem memory-cited context.")
def chat_with_memory(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    group_id: str | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    uid, gid = _resolved_scope(user_id, group_id)
    envelope = _collective_envelope(
        coordination_mode=coordination_mode,
        coordination_id=coordination_id,
        runtime_id=runtime_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        team_id=team_id,
        session_id=session_id,
    )
    body = {
        "query": query,
        "top_k": max(1, min(30, int(top_k))),
        "conversation_id": conversation_id,
        "provider": provider,
        "user_id": uid,
        "group_id": gid,
        **envelope,
    }
    return _request_json("POST", "/api/v1/chat/simple", body=body)


@mcp.tool(description="Write one memory item into FlockMem.")
def write_memory(
    content: str,
    sender: str,
    group_id: str | None = None,
    sender_name: str | None = None,
    role: str = "user",
    message_id: str | None = None,
    create_time: int | None = None,
    metadata: dict[str, Any] | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    gid = group_id or DEFAULT_GROUP_ID or f"default:{sender}"
    envelope = _collective_envelope(
        coordination_mode=coordination_mode,
        coordination_id=coordination_id,
        runtime_id=runtime_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        team_id=team_id,
        session_id=session_id,
    )
    merged_metadata = _merge_envelope_metadata(metadata, envelope)
    content_with_meta = _attach_metadata_line(content, merged_metadata)
    body = {
        "message_id": message_id or f"mcp-{uuid.uuid4().hex}",
        "create_time": int(create_time or time.time()),
        "sender": sender,
        "content": content_with_meta,
        "group_id": gid,
        "sender_name": sender_name or sender,
        "role": role,
        "metadata": merged_metadata,
        **envelope,
    }
    return _request_json("POST", "/api/v1/memories", body=body)


@mcp.tool(description="Ingest parsed skill output using FlockMem normalized ingest contract.")
def ingest_skill_output(
    source_type: str = "skill_output",
    source_uri: str | None = None,
    raw_text: str | None = None,
    summary: str | None = None,
    chunks: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    skill_name: str | None = None,
    skill_version: str | None = None,
    agent_id: str | None = None,
    sender: str | None = None,
    group_id: str | None = None,
    task_id: str | None = None,
    channel: str | None = None,
    trace_id: str | None = None,
    role: str = "user",
    create_time: int | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    st = str(source_type or "skill_output").strip().lower()
    if st not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {source_type}")
    gid = group_id or DEFAULT_GROUP_ID
    if not gid:
        sid = sender or agent_id or "skill-agent"
        gid = f"default:{sid}"
    envelope = _collective_envelope(
        coordination_mode=coordination_mode,
        coordination_id=coordination_id,
        runtime_id=runtime_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        team_id=team_id,
        session_id=session_id,
    )
    merged_metadata = _merge_envelope_metadata(metadata, envelope)
    body = {
        "source_type": st,
        "source_uri": source_uri,
        "raw_text": raw_text,
        "summary": summary,
        "chunks": list(chunks or []),
        "metadata": merged_metadata,
        "skill_name": skill_name,
        "skill_version": skill_version,
        "agent_id": agent_id,
        "sender": sender,
        "group_id": gid,
        "task_id": task_id,
        "channel": channel,
        "trace_id": trace_id,
        "role": role,
        "create_time": create_time,
        **envelope,
    }
    return _request_json("POST", "/api/v1/ingest/skill", body=body)


@mcp.tool(description="Write one revision through FlockMem collective ingest contract.")
def collective_ingest(
    scope_type: str,
    content: dict[str, Any],
    knowledge_id: str | None = None,
    revision_id: str | None = None,
    parent_revision_id: str | None = None,
    scope_id: str | None = None,
    change_type: str = "update",
    changed_by: str = "agent",
    actor_id: str | None = None,
    confidence: float = 0.6,
    trust_score: float = 0.5,
    read_acl: list[str] | None = None,
    write_acl: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    st = _normalize_token(scope_type)
    if st is None or st.lower() not in ALLOWED_SCOPE_TYPES:
        raise ValueError(f"invalid scope_type: {scope_type}")
    ct = _normalize_token(change_type)
    if ct is None or ct.lower() not in ALLOWED_CHANGE_TYPES:
        raise ValueError(f"invalid change_type: {change_type}")
    cb = _normalize_token(changed_by)
    if cb is None or cb.lower() not in ALLOWED_CHANGED_BY:
        raise ValueError(f"invalid changed_by: {changed_by}")
    if not isinstance(content, dict):
        raise ValueError("content must be an object")

    body = {
        "knowledge_id": _normalize_token(knowledge_id),
        "revision_id": _normalize_token(revision_id),
        "parent_revision_id": _normalize_token(parent_revision_id),
        "scope_type": st.lower(),
        "scope_id": _normalize_token(scope_id),
        "content": content,
        "change_type": ct.lower(),
        "changed_by": cb.lower(),
        "actor_id": _normalize_token(actor_id),
        "confidence": float(confidence),
        "trust_score": float(trust_score),
        "read_acl": list(read_acl or []),
        "write_acl": list(write_acl or []),
        "evidence": list(evidence or []),
        **_collective_envelope(
            coordination_mode=coordination_mode,
            coordination_id=coordination_id,
            runtime_id=runtime_id,
            agent_id=agent_id,
            subagent_id=subagent_id,
            team_id=team_id,
            session_id=session_id,
        ),
    }
    return _request_json("POST", "/api/v1/collective/ingest", body=body)


@mcp.tool(description="Resolve context through FlockMem collective context contract.")
def collective_context(
    query: str | None = None,
    actor_id: str | None = None,
    personal_scope_id: str | None = None,
    team_scope_id: str | None = None,
    include_global: bool = True,
    top_k: int = 20,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    body = {
        "query": _normalize_token(query),
        "actor_id": _normalize_token(actor_id),
        "personal_scope_id": _normalize_token(personal_scope_id),
        "team_scope_id": _normalize_token(team_scope_id),
        "include_global": bool(include_global),
        "top_k": max(1, min(100, int(top_k))),
        **_collective_envelope(
            coordination_mode=coordination_mode,
            coordination_id=coordination_id,
            runtime_id=runtime_id,
            agent_id=agent_id,
            subagent_id=subagent_id,
            team_id=team_id,
            session_id=session_id,
        ),
    }
    return _request_json("POST", "/api/v1/collective/context", body=body)


@mcp.tool(description="Submit execution feedback through FlockMem collective feedback contract.")
def collective_feedback(
    knowledge_id: str,
    feedback_type: str,
    feedback_payload: dict[str, Any],
    actor: str,
    revision_id: str | None = None,
    coordination_mode: str | None = None,
    coordination_id: str | None = None,
    runtime_id: str | None = None,
    agent_id: str | None = None,
    subagent_id: str | None = None,
    team_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(feedback_payload, dict):
        raise ValueError("feedback_payload must be an object")
    body = {
        "knowledge_id": _normalize_token(knowledge_id),
        "revision_id": _normalize_token(revision_id),
        "feedback_type": _normalize_token(feedback_type),
        "feedback_payload": feedback_payload,
        "actor": _normalize_token(actor),
        **_collective_envelope(
            coordination_mode=coordination_mode,
            coordination_id=coordination_id,
            runtime_id=runtime_id,
            agent_id=agent_id,
            subagent_id=subagent_id,
            team_id=team_id,
            session_id=session_id,
        ),
    }
    return _request_json("POST", "/api/v1/collective/feedback", body=body)


@mcp.tool(description="Search graph triples from FlockMem graph store.")
def graph_search(
    query: str,
    limit: int = 20,
    user_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    uid, gid = _resolved_scope(user_id, group_id)
    return _request_json(
        "GET",
        "/api/v1/graph/search",
        params={
            "query": query,
            "limit": max(1, min(100, int(limit))),
            "user_id": uid,
            "group_id": gid,
        },
    )


@mcp.tool(description="Get graph neighbors by entity name.")
def graph_neighbors(
    entity: str,
    limit: int = 20,
    user_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    uid, gid = _resolved_scope(user_id, group_id)
    return _request_json(
        "GET",
        "/api/v1/graph/neighbors",
        params={
            "entity": entity,
            "limit": max(1, min(200, int(limit))),
            "user_id": uid,
            "group_id": gid,
        },
    )


def main() -> None:
    mcp.run("stdio", show_banner=False)


if __name__ == "__main__":
    main()


