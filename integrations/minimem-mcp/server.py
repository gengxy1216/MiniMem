from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any
from urllib import error, parse, request

from fastmcp import FastMCP


MCP_NAME = "minimem-mcp"
BASE_URL = os.getenv("MINIMEM_BASE_URL", "http://127.0.0.1:20195").strip().rstrip("/")
TIMEOUT_SEC = float(os.getenv("MINIMEM_TIMEOUT_SEC", "25"))
DEFAULT_USER_ID = os.getenv("MINIMEM_USER_ID", "").strip() or None
DEFAULT_GROUP_ID = os.getenv("MINIMEM_GROUP_ID", "").strip() or None
BEARER_TOKEN = os.getenv("MINIMEM_BEARER_TOKEN", "").strip()
BASIC_USER = os.getenv("MINIMEM_BASIC_USER", "").strip()
BASIC_PASSWORD = os.getenv("MINIMEM_BASIC_PASSWORD", "").strip()

ALLOWED_RETRIEVE_METHODS = {"keyword", "vector", "hybrid", "rrf", "agentic"}
ALLOWED_DECISION_MODES = {"static", "rule", "agent"}
ALLOWED_RUNTIME_PROFILES = {"keyword", "hybrid", "agentic"}

mcp = FastMCP(
    name=MCP_NAME,
    instructions=(
        "Bridge tools for MiniMem HTTP API. "
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
        raise RuntimeError(f"MiniMem API HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"MiniMem API request failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("response is not a JSON object")
        return parsed
    except Exception as exc:
        raise RuntimeError(f"MiniMem API returned invalid JSON: {raw[:280]}") from exc


def _resolved_scope(user_id: str | None, group_id: str | None) -> tuple[str | None, str | None]:
    return (user_id or DEFAULT_USER_ID, group_id or DEFAULT_GROUP_ID)


@mcp.tool(description="Health check for MiniMem service.")
def minimem_health() -> dict[str, Any]:
    return _request_json("GET", "/health")


@mcp.tool(description="Search memories with MiniMem hybrid/vector/keyword/agentic retrieval.")
def search_memories(
    query: str,
    top_k: int = 20,
    retrieve_method: str = "agentic",
    decision_mode: str = "rule",
    runtime_profile: str | None = None,
    user_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    method = retrieve_method.strip().lower()
    if method not in ALLOWED_RETRIEVE_METHODS:
        raise ValueError(f"invalid retrieve_method: {retrieve_method}")
    mode = decision_mode.strip().lower()
    if mode not in ALLOWED_DECISION_MODES:
        raise ValueError(f"invalid decision_mode: {decision_mode}")
    profile = runtime_profile.strip().lower() if runtime_profile else None
    if profile is not None and profile not in ALLOWED_RUNTIME_PROFILES:
        raise ValueError(f"invalid runtime_profile: {runtime_profile}")
    uid, gid = _resolved_scope(user_id, group_id)
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
        },
    )


@mcp.tool(description="Chat with MiniMem memory-cited context.")
def chat_with_memory(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    uid, gid = _resolved_scope(user_id, group_id)
    body = {
        "query": query,
        "top_k": max(1, min(30, int(top_k))),
        "conversation_id": conversation_id,
        "provider": provider,
        "user_id": uid,
        "group_id": gid,
    }
    return _request_json("POST", "/api/v1/chat/simple", body=body)


@mcp.tool(description="Write one memory item into MiniMem.")
def write_memory(
    content: str,
    sender: str,
    group_id: str | None = None,
    sender_name: str | None = None,
    role: str = "user",
    message_id: str | None = None,
    create_time: int | None = None,
) -> dict[str, Any]:
    gid = group_id or DEFAULT_GROUP_ID or f"default:{sender}"
    body = {
        "message_id": message_id or f"mcp-{uuid.uuid4().hex}",
        "create_time": int(create_time or time.time()),
        "sender": sender,
        "content": content,
        "group_id": gid,
        "sender_name": sender_name or sender,
        "role": role,
    }
    return _request_json("POST", "/api/v1/memories", body=body)


@mcp.tool(description="Search graph triples from MiniMem graph store.")
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
