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
from unittest.mock import patch
from urllib import parse

from fastmcp import Client
from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.service.runtime_profile_service import (
    load_runtime_profile_schema,
    validate_adapter_payload,
)
from flockmem.testing.writable_tempdir import WritableTempDir


class _StubCollectiveHandler(BaseHTTPRequestHandler):
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
            cls.events.append({"method": method, "path": path, "query": query, "body": body})

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
            self._send_json({"ok": True, "query": query, "hits": []})
            return
        self._send_json({"ok": False, "path": parsed.path}, code=404)

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
            "/api/v1/collective/ingest",
            "/api/v1/collective/context",
            "/api/v1/collective/feedback",
            "/api/v1/ingest/skill",
        }:
            self._send_json({"ok": True, "echo": body})
            return
        self._send_json({"ok": False, "path": parsed.path}, code=404)


def _find_event(path: str, method: str = "POST") -> dict[str, Any]:
    events = _StubCollectiveHandler.snapshot_events()
    for event in reversed(events):
        if event.get("path") == path and event.get("method") == method:
            return event
    return {}


class CollectiveIntegrationEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.mcp_server = cls.repo_root / "integrations" / "flockmem-mcp" / "server.py"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubCollectiveHandler)
        cls.http_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.http_thread.join(timeout=2)

    def setUp(self) -> None:
        _StubCollectiveHandler.clear_events()

    def _client_config(self) -> dict[str, Any]:
        env = dict(os.environ)
        env.update(
            {
                "MINIMEM_BASE_URL": self.base_url,
                "MINIMEM_TIMEOUT_SEC": "8",
                "MINIMEM_USER_ID": "user-envelope-test",
                "MINIMEM_GROUP_ID": "group-envelope-test",
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

    def test_collective_envelope_passthrough_consistency(self) -> None:
        envelope = {
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-bridge-1",
            "runtime_id": "runtime-bridge",
            "agent_id": "agent-bridge",
            "subagent_id": "subagent-bridge",
            "team_id": "team-bridge",
            "session_id": "session-bridge",
        }

        async def _run() -> None:
            async with Client(self._client_config()) as client:
                await client.call_tool(
                    "collective_ingest",
                    {
                        "knowledge_id": "k-bridge-1",
                        "scope_type": "team",
                        "scope_id": "team-bridge",
                        "content": {"text": "collective-bridge"},
                        "change_type": "update",
                        "changed_by": "agent",
                        "actor_id": "agent-bridge",
                        **envelope,
                    },
                )
                await client.call_tool(
                    "collective_context",
                    {
                        "query": "collective envelope context",
                        "actor_id": "agent-bridge",
                        "team_scope_id": "team-bridge",
                        "include_global": False,
                        "top_k": 6,
                        **envelope,
                    },
                )
                await client.call_tool(
                    "collective_feedback",
                    {
                        "knowledge_id": "k-bridge-1",
                        "feedback_type": "execution_signal",
                        "feedback_payload": {"status": "ok"},
                        "actor": "agent-bridge",
                        **envelope,
                    },
                )

        asyncio.run(_run())

        ingest = _find_event("/api/v1/collective/ingest")
        context = _find_event("/api/v1/collective/context")
        feedback = _find_event("/api/v1/collective/feedback")
        for event in (ingest, context, feedback):
            body = event.get("body") if isinstance(event, dict) else {}
            self.assertIsInstance(body, dict)
            for key, value in envelope.items():
                self.assertEqual(value, body.get(key), msg=f"{key} not passed for {event.get('path')}")

    def test_ingest_skill_output_merges_envelope_into_metadata(self) -> None:
        async def _run() -> None:
            async with Client(self._client_config()) as client:
                await client.call_tool(
                    "ingest_skill_output",
                    {
                        "source_type": "text",
                        "raw_text": "skill payload text",
                        "skill_name": "bridge-test",
                        "agent_id": "agent-ingest",
                        "coordination_mode": "federated_acp",
                        "coordination_id": "coord-ingest-1",
                        "runtime_id": "runtime-ingest",
                        "subagent_id": "subagent-ingest",
                        "team_id": "team-ingest",
                        "session_id": "session-ingest",
                        "metadata": {"origin": "unit-test"},
                    },
                )

        asyncio.run(_run())
        event = _find_event("/api/v1/ingest/skill")
        body = event.get("body", {})
        self.assertEqual("federated_acp", body.get("coordination_mode"))
        self.assertEqual("runtime-ingest", body.get("runtime_id"))
        metadata = body.get("metadata", {})
        self.assertIsInstance(metadata, dict)
        self.assertEqual("unit-test", metadata.get("origin"))
        self.assertEqual("coord-ingest-1", metadata.get("coordination_id"))
        self.assertEqual("session-ingest", metadata.get("session_id"))

    def test_search_memories_runtime_profile_schema_validation(self) -> None:
        async def _run_valid() -> None:
            async with Client(self._client_config()) as client:
                await client.call_tool(
                    "search_memories",
                    {
                        "query": "runtime profile recall test",
                        "runtime_profile": "recall",
                        "top_k": 5,
                    },
                )

        asyncio.run(_run_valid())
        search_event = _find_event("/api/v1/memories/search", method="GET")
        query = search_event.get("query", {})
        self.assertEqual("recall", query.get("runtime_profile"))

        async def _run_invalid() -> None:
            async with Client(self._client_config()) as client:
                await client.call_tool(
                    "search_memories",
                    {
                        "query": "runtime profile bad test",
                        "runtime_profile": "not-a-profile",
                        "top_k": 5,
                    },
                )

        with self.assertRaises(Exception):
            asyncio.run(_run_invalid())

    def test_runtime_profile_service_supports_all_adapter_keysets(self) -> None:
        schema = load_runtime_profile_schema()
        keysets = schema.get("adapter_keysets", {})
        self.assertIn("mcp", keysets)
        self.assertIn("hook", keysets)
        self.assertIn("plugin", keysets)
        self.assertIn("cli_bridge", keysets)
        self.assertIn("webhook_bridge", keysets)

        normalized = validate_adapter_payload(
            "cli_bridge",
            {
                "runtime_profile": "AGENTIC",
                "coordination_mode": "INRUNTIME_A2A",
                "coordination_id": "coord-x",
                "runtime_id": "bridge-runtime",
                "agent_id": "agent-x",
                "session_id": "session-x",
            },
        )
        self.assertEqual("agentic", normalized.get("runtime_profile"))
        self.assertEqual("inruntime_a2a", normalized.get("coordination_mode"))
        self.assertEqual("bridge-runtime", normalized.get("runtime_id"))

        webhook_payload = validate_adapter_payload(
            "webhook_bridge",
            {
                "runtime_profile": "hybrid",
                "coordination_mode": "federated_acp",
                "coordination_id": "coord-webhook",
                "runtime_id": "webhook-runtime",
            },
        )
        self.assertEqual("hybrid", webhook_payload.get("runtime_profile"))
        self.assertEqual("federated_acp", webhook_payload.get("coordination_mode"))

        with self.assertRaises(ValueError):
            validate_adapter_payload(
                "plugin",
                {"runtime_profile": "hybrid", "coordination_mode": "invalid-mode"},
            )


if __name__ == "__main__":
    unittest.main()


class CollectiveEnvelopePersistenceTests(unittest.TestCase):
    def _build_app_client(self) -> tuple[TestClient, Any, WritableTempDir]:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        env = {
            "LITE_DATA_DIR": str(Path(tmp.name) / "envelope-db-data"),
            "LITE_CONFIG_DIR": str(Path(tmp.name) / "envelope-db-config"),
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "chat-key",
            "LITE_CHAT_MODEL": "chat-model-a",
            "LITE_EMBEDDING_PROVIDER": "local",
            "LITE_EMBEDDING_MODEL": "local-hash-384",
            "LITE_EXTRACTOR_PROVIDER": "rule",
            "LITE_RERANK_PROVIDER": "none",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        app = create_app(settings)
        return TestClient(app), app, tmp

    def test_envelope_fields_are_persisted_for_revision_and_feedback(self) -> None:
        client, app, tmp = self._build_app_client()
        self.addCleanup(tmp.cleanup)
        envelope = {
            "coordination_mode": "inruntime_a2a",
            "coordination_id": "coord-persist-1",
            "runtime_id": "runtime-persist",
            "agent_id": "agent-persist",
            "subagent_id": "subagent-persist",
            "team_id": "team-persist",
            "session_id": "session-persist",
        }

        ingest_resp = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-persist-envelope-1",
                "scope_type": "team",
                "scope_id": "team-persist",
                "content": {"text": "persist envelope ingest"},
                "change_type": "update",
                "changed_by": "agent",
                "actor_id": "agent-persist",
                "read_acl": ["agent-persist"],
                "write_acl": ["agent-persist"],
                **envelope,
            },
        )
        self.assertEqual(200, ingest_resp.status_code, msg=ingest_resp.text)
        revision_id = ingest_resp.json().get("result", {}).get("revision_id")
        self.assertTrue(str(revision_id or "").strip())

        context_resp = client.post(
            "/api/v1/collective/context",
            json={
                "query": "persist envelope context",
                "team_scope_id": "team-persist",
                "actor_id": "agent-persist",
                "include_global": False,
                "top_k": 5,
                **envelope,
            },
        )
        self.assertEqual(200, context_resp.status_code, msg=context_resp.text)
        self.assertGreaterEqual(
            int(context_resp.json().get("result", {}).get("count", 0)),
            1,
        )

        feedback_resp = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-persist-envelope-1",
                "revision_id": revision_id,
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "success"},
                "actor": "agent-persist",
                **envelope,
            },
        )
        self.assertEqual(200, feedback_resp.status_code, msg=feedback_resp.text)
        feedback_id = feedback_resp.json().get("result", {}).get("feedback_id")
        self.assertTrue(str(feedback_id or "").strip())

        engine = app.state.sqlite_engine
        revision_row = engine.query_one(
            """
            SELECT coordination_mode,coordination_id,runtime_id,agent_id,
                   subagent_id,team_id,session_id
            FROM knowledge_revision
            WHERE revision_id=?
            """,
            (revision_id,),
        )
        self.assertIsInstance(revision_row, dict)
        for key, value in envelope.items():
            self.assertEqual(value, revision_row.get(key), msg=f"revision.{key} mismatch")

        feedback_row = engine.query_one(
            """
            SELECT coordination_mode,coordination_id,runtime_id,agent_id,
                   subagent_id,team_id,session_id
            FROM knowledge_feedback
            WHERE feedback_id=?
            """,
            (feedback_id,),
        )
        self.assertIsInstance(feedback_row, dict)
        for key, value in envelope.items():
            self.assertEqual(value, feedback_row.get(key), msg=f"feedback.{key} mismatch")

        client.close()

    def test_schema_migration_adds_missing_envelope_columns(self) -> None:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engine = SQLiteEngine(Path(tmp.name) / "legacy-envelope.db")
        engine.execute(
            """
            CREATE TABLE knowledge_item (
                knowledge_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'draft',
                canonical_revision_id TEXT,
                read_acl TEXT NOT NULL DEFAULT '[]',
                write_acl TEXT NOT NULL DEFAULT '[]',
                trust_score REAL NOT NULL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        engine.execute(
            """
            CREATE TABLE knowledge_revision (
                revision_id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                parent_revision_id TEXT,
                content_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                change_type TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                coordination_mode TEXT,
                coordination_id TEXT,
                runtime_id TEXT,
                agent_id TEXT,
                subagent_id TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(knowledge_id) REFERENCES knowledge_item(knowledge_id) ON DELETE CASCADE
            )
            """
        )
        engine.execute(
            """
            CREATE TABLE knowledge_feedback (
                feedback_id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                revision_id TEXT,
                feedback_type TEXT NOT NULL,
                feedback_payload TEXT NOT NULL,
                actor TEXT NOT NULL,
                coordination_mode TEXT,
                coordination_id TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(knowledge_id) REFERENCES knowledge_item(knowledge_id) ON DELETE CASCADE,
                FOREIGN KEY(revision_id) REFERENCES knowledge_revision(revision_id) ON DELETE SET NULL
            )
            """
        )

        init_schema(engine)

        revision_columns = {
            str(row.get("name"))
            for row in engine.query_all("PRAGMA table_info(knowledge_revision)")
        }
        self.assertIn("team_id", revision_columns)
        self.assertIn("session_id", revision_columns)

        feedback_columns = {
            str(row.get("name"))
            for row in engine.query_all("PRAGMA table_info(knowledge_feedback)")
        }
        for column in ("runtime_id", "agent_id", "subagent_id", "team_id", "session_id"):
            self.assertIn(column, feedback_columns)
