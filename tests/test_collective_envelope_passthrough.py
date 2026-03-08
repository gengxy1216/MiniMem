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

from flockmem.service.runtime_profile_service import RuntimeProfileService


class _StubEnvelopeHandler(BaseHTTPRequestHandler):
    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
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
        query = {
            k: v[0] if len(v) == 1 else v
            for k, v in parse.parse_qs(parsed.query).items()
        }
        self._record_event(method="GET", path=parsed.path, query=query, body=None)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/api/v1/memories/search":
            self._send_json({"ok": True, "query": query, "result": {"memories": []}})
            return
        self._send_json({"ok": False, "error": "not found"}, code=404)

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
        if parsed.path in {
            "/api/v1/chat/simple",
            "/api/v1/memories",
            "/api/v1/ingest/skill",
        }:
            self._send_json({"ok": True, "received": body})
            return
        self._send_json({"ok": False, "error": "not found"}, code=404)


class CollectiveEnvelopePassthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.mcp_server = cls.repo_root / "integrations" / "flockmem-mcp" / "server.py"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubEnvelopeHandler)
        cls.http_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.http_thread.join(timeout=2)

    def setUp(self) -> None:
        _StubEnvelopeHandler.clear_events()

    def _client_config(self) -> dict[str, Any]:
        env = dict(os.environ)
        env.update(
            {
                "MINIMEM_BASE_URL": self.base_url,
                "MINIMEM_TIMEOUT_SEC": "8",
                "MINIMEM_USER_ID": "user-envelope",
                "MINIMEM_GROUP_ID": "group-envelope",
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

    def test_memory_and_ingest_tools_passthrough_collective_envelope(self) -> None:
        async def _run() -> None:
            async with Client(self._client_config()) as client:
                await client.call_tool(
                    "search_memories",
                    {
                        "query": "coordination passthrough query",
                        "coordination_mode": "inruntime_a2a",
                        "coordination_id": "coord-mcp-1",
                        "runtime_id": "openclaw",
                        "agent_id": "agent-main",
                        "subagent_id": "agent-sub",
                        "team_id": "team-alpha",
                        "session_id": "session-123",
                        "runtime_profile": "hybrid",
                    },
                )
                await client.call_tool(
                    "chat_with_memory",
                    {
                        "query": "chat with envelope",
                        "coordination_mode": "federated_acp",
                        "coordination_id": "coord-chat-1",
                        "runtime_id": "codex",
                        "agent_id": "chat-agent",
                        "subagent_id": "chat-sub",
                        "team_id": "team-chat",
                        "session_id": "session-chat",
                    },
                )
                await client.call_tool(
                    "write_memory",
                    {
                        "content": "dialogue payload",
                        "sender": "writer-agent",
                        "coordination_mode": "inruntime_a2a",
                        "coordination_id": "coord-write-1",
                        "runtime_id": "openclaw",
                        "agent_id": "writer-main",
                        "subagent_id": "writer-sub",
                        "team_id": "team-write",
                        "session_id": "session-write",
                    },
                )
                await client.call_tool(
                    "ingest_skill_output",
                    {
                        "summary": "ingest summary",
                        "chunks": ["chunk-1"],
                        "skill_name": "browser-use",
                        "agent_id": "ingest-main",
                        "coordination_mode": "inruntime_a2a",
                        "coordination_id": "coord-ingest-1",
                        "runtime_id": "openclaw",
                        "subagent_id": "ingest-sub",
                        "team_id": "team-ingest",
                        "session_id": "session-ingest",
                    },
                )

        asyncio.run(_run())
        events = _StubEnvelopeHandler.snapshot_events()

        search_event = next(
            e for e in events if e["method"] == "GET" and e["path"] == "/api/v1/memories/search"
        )
        self.assertEqual("inruntime_a2a", search_event["query"].get("coordination_mode"))
        self.assertEqual("coord-mcp-1", search_event["query"].get("coordination_id"))
        self.assertEqual("openclaw", search_event["query"].get("runtime_id"))
        self.assertEqual("agent-main", search_event["query"].get("agent_id"))
        self.assertEqual("agent-sub", search_event["query"].get("subagent_id"))
        self.assertEqual("team-alpha", search_event["query"].get("team_id"))
        self.assertEqual("session-123", search_event["query"].get("session_id"))

        chat_event = next(
            e for e in events if e["method"] == "POST" and e["path"] == "/api/v1/chat/simple"
        )
        self.assertEqual("federated_acp", chat_event["body"].get("coordination_mode"))
        self.assertEqual("coord-chat-1", chat_event["body"].get("coordination_id"))
        self.assertEqual("codex", chat_event["body"].get("runtime_id"))
        self.assertEqual("chat-sub", chat_event["body"].get("subagent_id"))
        self.assertEqual("team-chat", chat_event["body"].get("team_id"))
        self.assertEqual("session-chat", chat_event["body"].get("session_id"))

        write_event = next(
            e for e in events if e["method"] == "POST" and e["path"] == "/api/v1/memories"
        )
        self.assertEqual("inruntime_a2a", write_event["body"].get("coordination_mode"))
        self.assertEqual("coord-write-1", write_event["body"].get("coordination_id"))
        self.assertEqual("writer-sub", write_event["body"].get("subagent_id"))
        self.assertEqual("team-write", write_event["body"].get("team_id"))
        self.assertEqual("session-write", write_event["body"].get("session_id"))
        metadata = write_event["body"].get("metadata", {})
        self.assertEqual("coord-write-1", metadata.get("coordination_id"))
        self.assertIn("[metadata]", str(write_event["body"].get("content", "")))

        ingest_event = next(
            e for e in events if e["method"] == "POST" and e["path"] == "/api/v1/ingest/skill"
        )
        self.assertEqual("inruntime_a2a", ingest_event["body"].get("coordination_mode"))
        self.assertEqual("coord-ingest-1", ingest_event["body"].get("coordination_id"))
        self.assertEqual("ingest-sub", ingest_event["body"].get("subagent_id"))
        self.assertEqual("team-ingest", ingest_event["body"].get("team_id"))
        self.assertEqual("session-ingest", ingest_event["body"].get("session_id"))
        ingest_meta = ingest_event["body"].get("metadata", {})
        self.assertEqual("coord-ingest-1", ingest_meta.get("coordination_id"))
        self.assertEqual("session-ingest", ingest_meta.get("session_id"))

    def test_runtime_profile_service_min_schema_validation(self) -> None:
        service = RuntimeProfileService()
        valid_profile = {
            "runtime_id": "openclaw-local",
            "adapter_type": "cli_bridge",
            "enabled": True,
            "launch": {"command": "openclaw", "args": ["run"], "timeout_seconds": 120},
            "contracts": {
                "ingest": {"mode": "event_push"},
                "context": {"mode": "pull_before_task", "endpoint": "/api/v1/collective/context"},
                "feedback": {"mode": "push_after_task", "endpoint": "/api/v1/collective/feedback"},
            },
            "scope_defaults": {"read_order": ["personal", "team", "global"], "write_scope": "team"},
            "reliability": {"retry_max": 3, "retry_backoff_ms": 300, "circuit_breaker": True},
            "security": {"auth_mode": "token_ref", "token_ref": "secret://flockmem/openclaw"},
        }
        ok = service.validate(valid_profile)
        self.assertTrue(ok.valid)
        self.assertEqual("cli_bridge", ok.normalized.get("adapter_type"))

        bad_profile = {
            "adapter_type": "cli_bridge",
            "enabled": "yes",
            "launch": {"args": []},
            "contracts": {"context": {}},
            "security": {"auth_mode": "bearer"},
        }
        bad = service.validate(bad_profile)
        self.assertFalse(bad.valid)
        self.assertGreaterEqual(len(bad.errors), 3)
        with self.assertRaises(ValueError):
            service.validate_or_raise(bad_profile)

        recovered = service.validate(valid_profile)
        self.assertTrue(recovered.valid)


if __name__ == "__main__":
    unittest.main()
