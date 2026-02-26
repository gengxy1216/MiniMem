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


class _StubMiniMemHandler(BaseHTTPRequestHandler):
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


class MiniMemMCPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.mcp_server = cls.repo_root / "integrations" / "minimem-mcp" / "server.py"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubMiniMemHandler)
        cls.http_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.http_thread.join(timeout=2)

    def setUp(self) -> None:
        _StubMiniMemHandler.clear_events()

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
            for e in _StubMiniMemHandler.snapshot_events()
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


if __name__ == "__main__":
    unittest.main()
