from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


class CollectiveCoreSmokeTests(unittest.TestCase):
    def _build_client(self) -> TestClient:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = {
            "LITE_DATA_DIR": str(Path(tmp.name) / "smoke-data"),
            "LITE_CONFIG_DIR": str(Path(tmp.name) / "smoke-config"),
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "chat-key",
            "LITE_CHAT_MODEL": "chat-model-a",
            "LITE_EMBEDDING_PROVIDER": "local",
            "LITE_EMBEDDING_MODEL": "local-hash-384",
            "LITE_EXTRACTOR_PROVIDER": "rule",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        client = TestClient(create_app(settings))
        self.addCleanup(client.close)
        return client

    def test_collective_core_v1_smoke(self) -> None:
        client = self._build_client()
        route_paths = {getattr(route, "path", "") for route in client.app.routes}
        self.assertIn("/api/v1/collective/ingest", route_paths)
        self.assertIn("/api/v1/collective/context", route_paths)
        self.assertIn("/api/v1/collective/feedback", route_paths)

        ingest_response = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-smoke-v1",
                "scope_type": "personal",
                "scope_id": "u-smoke",
                "content": {"fact": "smoke baseline fact"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "agent-smoke",
                "read_acl": ["agent-smoke"],
                "write_acl": ["agent-smoke"],
                "confidence": 0.9,
                "trust_score": 0.7,
            },
        )
        self.assertEqual(200, ingest_response.status_code)
        ingest_result = ingest_response.json().get("result", {})
        revision_id = str(ingest_result.get("revision_id") or "").strip()
        self.assertEqual("k-smoke-v1", ingest_result.get("knowledge_id"))
        self.assertTrue(revision_id)

        context_response = client.post(
            "/api/v1/collective/context",
            json={
                "personal_scope_id": "u-smoke",
                "actor_id": "agent-smoke",
                "include_global": False,
                "top_k": 10,
            },
        )
        self.assertEqual(200, context_response.status_code)
        context_result = context_response.json().get("result", {})
        context_items = context_result.get("items", [])
        self.assertEqual(1, int(context_result.get("count", 0)))
        self.assertEqual("k-smoke-v1", context_items[0].get("knowledge_id"))
        self.assertEqual(revision_id, context_items[0].get("revision_id"))

        feedback_response = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-smoke-v1",
                "revision_id": revision_id,
                "feedback_type": "execution_result",
                "feedback_payload": {"outcome_status": "success"},
                "actor": "agent-smoke",
            },
        )
        self.assertEqual(200, feedback_response.status_code)
        feedback_result = feedback_response.json().get("result", {})
        feedback_id = str(feedback_result.get("feedback_id") or "").strip()
        self.assertTrue(feedback_id)

        engine = client.app.state.sqlite_engine
        item = engine.query_one(
            "SELECT knowledge_id, canonical_revision_id FROM knowledge_item WHERE knowledge_id=?",
            ("k-smoke-v1",),
        )
        self.assertIsNotNone(item)
        self.assertEqual(revision_id, item.get("canonical_revision_id"))

        revision = engine.query_one(
            "SELECT revision_id FROM knowledge_revision WHERE revision_id=?",
            (revision_id,),
        )
        self.assertIsNotNone(revision)

        feedback = engine.query_one(
            "SELECT feedback_id FROM knowledge_feedback WHERE feedback_id=?",
            (feedback_id,),
        )
        self.assertIsNotNone(feedback)


if __name__ == "__main__":
    unittest.main()
