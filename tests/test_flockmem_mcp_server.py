from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import parse

from fastmcp import Client


class _StubFlockMemHandler(BaseHTTPRequestHandler):
    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        return

    @classmethod
    def clear_events(cls) -> None:
        with cls.events_lock:
            cls.events.clear()

    @classmethod
    def snapshot_events(cls) -> list[dict[str, Any]]:
        with cls.events_lock:
            return list(cls.events)

    @classmethod
    def _record_event(
        cls,
        *,
        method: str,
        path: str,
        query: dict[str, Any],
        body: dict[str, Any] | None,
    ) -> None:
        with cls.events_lock:
            cls.events.append(
                {"method": method, "path": path, "query": query, "body": body}
            )

    def _send_json(self, payload: dict[str, Any], *, code: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = parse.urlparse(self.path)
        query = {k: v[0] if len(v) == 1 else v for k, v in parse.parse_qs(parsed.query).items()}
        self._record_event(method="GET", path=parsed.path, query=query, body=None)

        if parsed.path == "/health":
            self._send_json({"status": "ok", "service": "stub"})
            return
        if parsed.path == "/api/v1/memories/search":
            self._send_json(
                {
                    "ok": True,
                    "query": query,
                    "hits": [
                        {
                            "id": "slice-5",
                            "snippet": "AI reflects new productive forces via technology and efficiency",
                        }
                    ],
                }
            )
            return
        if parsed.path == "/api/v1/graph/search":
            self._send_json({"ok": True, "query": query, "hits": []})
            return
        if parsed.path == "/api/v1/graph/neighbors":
            self._send_json({"ok": True, "query": query, "neighbors": []})
            return

        self._send_json({"ok": False, "error": "not found", "path": parsed.path}, code=404)

    def do_POST(self) -> None:
        parsed = parse.urlparse(self.path)
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size).decode("utf-8") if size else "{}"
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid json"}, code=400)
            return

        self._record_event(method="POST", path=parsed.path, query={}, body=body)

        if parsed.path == "/api/v1/memories":
            self._send_json({"ok": True, "saved": body})
            return
        if parsed.path == "/api/v1/ingest/skill":
            self._send_json({"ok": True, "ingested": body, "accepted": True})
            return
        if parsed.path == "/api/v1/collective/ingest":
            self._send_json(
                {
                    "status": "ok",
                    "message": "ingest accepted",
                    "result": {
                        "knowledge_id": body.get("knowledge_id") or "k-stub",
                        "revision_id": "r-stub-1",
                    },
                }
            )
            return
        if parsed.path == "/api/v1/collective/context":
            self._send_json(
                {
                    "status": "ok",
                    "message": "context resolved",
                    "result": {"count": 1, "items": [{"knowledge_id": "k-stub"}]},
                }
            )
            return
        if parsed.path == "/api/v1/collective/feedback":
            self._send_json(
                {
                    "status": "ok",
                    "message": "feedback accepted",
                    "result": {"knowledge_id": body.get("knowledge_id"), "feedback_id": "f-stub-1"},
                }
            )
            return
        if parsed.path == "/api/v1/chat/simple":
            self._send_json(
                {"ok": True, "answer": "stub-answer", "citations": [{"id": "slice-5"}]}
            )
            return

        self._send_json({"ok": False, "error": "not found", "path": parsed.path}, code=404)


def _tool_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structuredContent", None)
    if isinstance(payload, dict):
        return payload
    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}


class FlockMemMCPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.mcp_server = cls.repo_root / "integrations" / "flockmem-mcp" / "server.py"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubFlockMemHandler)
        cls.http_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.http_thread.join(timeout=2)

    def setUp(self) -> None:
        _StubFlockMemHandler.clear_events()

    def _client_config(self) -> dict[str, Any]:
        env = dict(os.environ)
        env.update(
            {
                "MINIMEM_BASE_URL": self.base_url,
                "MINIMEM_TIMEOUT_SEC": "8",
                "MINIMEM_USER_ID": "user-mcp-test",
                "MINIMEM_GROUP_ID": "group-mcp-test",
            }
        )
        return {
            "mcpServers": {
                "minimem": {
                    "command": sys.executable,
                    "args": [str(self.mcp_server)],
                    "cwd": str(self.repo_root),
                    "env": env,
                }
            }
        }

    def test_mcp_tools_are_exposed(self) -> None:
        async def _run() -> set[str]:
            async with Client(self._client_config()) as client:
                tools = await client.list_tools()
                return {tool.name for tool in tools}

        names = asyncio.run(_run())
        self.assertTrue(
            {
                "minimem_health",
                "search_memories",
                "chat_with_memory",
                "write_memory",
                "ingest_skill_output",
                "collective_ingest",
                "collective_context",
                "collective_feedback",
                "graph_search",
                "graph_neighbors",
            }.issubset(names)
        )

    def test_search_memories_uses_agentic_defaults_and_reaches_slice5(self) -> None:
        async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
            async with Client(self._client_config()) as client:
                health = await client.call_tool("minimem_health")
                search = await client.call_tool(
                    "search_memories",
                    {"query": "What is the relationship between AI and productivity?", "top_k": 7},
                )
                return _tool_payload(health), _tool_payload(search)

        health, search = asyncio.run(_run())
        self.assertEqual("ok", health.get("status"))
        self.assertEqual("slice-5", search.get("hits", [{}])[0].get("id"))

        search_events = [
            e
            for e in _StubFlockMemHandler.snapshot_events()
            if e["method"] == "GET" and e["path"] == "/api/v1/memories/search"
        ]
        self.assertEqual(1, len(search_events))
        query = search_events[0]["query"]
        self.assertEqual("agentic", query.get("retrieve_method"))
        self.assertEqual("rule", query.get("decision_mode"))
        self.assertEqual("7", query.get("top_k"))
        self.assertEqual("user-mcp-test", query.get("user_id"))
        self.assertEqual("group-mcp-test", query.get("group_id"))

    def test_write_memory_generates_message_id(self) -> None:
        async def _run() -> dict[str, Any]:
            async with Client(self._client_config()) as client:
                write_result = await client.call_tool(
                    "write_memory",
                    {
                        "content": "historical risk item should be persisted",
                        "sender": "tester",
                    },
                )
                return _tool_payload(write_result)

        payload = asyncio.run(_run())
        saved = payload.get("saved", {})
        self.assertTrue(str(saved.get("message_id", "")).startswith("mcp-"))
        self.assertEqual("group-mcp-test", saved.get("group_id"))

    def test_ingest_skill_output_uses_normalized_contract(self) -> None:
        async def _run() -> dict[str, Any]:
            async with Client(self._client_config()) as client:
                result = await client.call_tool(
                    "ingest_skill_output",
                    {
                        "source_type": "pdf",
                        "source_uri": "file:///tmp/a.pdf",
                        "summary": "pdf summary",
                        "chunks": ["chunk-a", "chunk-b"],
                        "skill_name": "pdf",
                        "agent_id": "agent-a",
                        "task_id": "task-1",
                        "channel": "chan-a",
                    },
                )
                return _tool_payload(result)

        payload = asyncio.run(_run())
        ingested = payload.get("ingested", {})
        self.assertEqual("pdf", ingested.get("source_type"))
        self.assertEqual("pdf", ingested.get("skill_name"))
        self.assertEqual("agent-a", ingested.get("agent_id"))
        self.assertEqual("task-1", ingested.get("task_id"))
        self.assertEqual("chan-a", ingested.get("channel"))
        self.assertEqual("group-mcp-test", ingested.get("group_id"))

    def test_collective_contract_envelope_passthrough_ingest_context_feedback(self) -> None:
        envelope = {
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-xyz-1",
            "runtime_id": "openclaw",
            "agent_id": "agent-root-1",
            "subagent_id": "agent-sub-1",
            "team_id": "team-red",
            "session_id": "sess-88",
        }

        async def _run() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            async with Client(self._client_config()) as client:
                ingest = await client.call_tool(
                    "collective_ingest",
                    {
                        "knowledge_id": "k-env-1",
                        "scope_type": "team",
                        "scope_id": "team-red",
                        "content": {"fact": "envelope forward"},
                        "change_type": "create",
                        "changed_by": "agent",
                        "actor_id": "agent-root-1",
                        **envelope,
                    },
                )
                context = await client.call_tool(
                    "collective_context",
                    {
                        "query": "need team context",
                        "actor_id": "agent-root-1",
                        "team_scope_id": "team-red",
                        "top_k": 5,
                        **envelope,
                    },
                )
                feedback = await client.call_tool(
                    "collective_feedback",
                    {
                        "knowledge_id": "k-env-1",
                        "feedback_type": "execution_signal",
                        "feedback_payload": {"outcome_status": "success"},
                        "actor": "agent-root-1",
                        **envelope,
                    },
                )
                return _tool_payload(ingest), _tool_payload(context), _tool_payload(feedback)

        ingest, context, feedback = asyncio.run(_run())
        self.assertEqual("ok", ingest.get("status"))
        self.assertEqual("ok", context.get("status"))
        self.assertEqual("ok", feedback.get("status"))

        events = _StubFlockMemHandler.snapshot_events()
        post_paths = {event["path"] for event in events if event["method"] == "POST"}
        self.assertIn("/api/v1/collective/ingest", post_paths)
        self.assertIn("/api/v1/collective/context", post_paths)
        self.assertIn("/api/v1/collective/feedback", post_paths)

        for path in (
            "/api/v1/collective/ingest",
            "/api/v1/collective/context",
            "/api/v1/collective/feedback",
        ):
            event = next(
                item
                for item in events
                if item["method"] == "POST" and item["path"] == path
            )
            body = event["body"]
            self.assertEqual(envelope["coordination_mode"], body.get("coordination_mode"))
            self.assertEqual(envelope["coordination_id"], body.get("coordination_id"))
            self.assertEqual(envelope["runtime_id"], body.get("runtime_id"))
            self.assertEqual(envelope["agent_id"], body.get("agent_id"))
            self.assertEqual(envelope["subagent_id"], body.get("subagent_id"))
            self.assertEqual(envelope["team_id"], body.get("team_id"))
            self.assertEqual(envelope["session_id"], body.get("session_id"))


if __name__ == "__main__":
    unittest.main()


