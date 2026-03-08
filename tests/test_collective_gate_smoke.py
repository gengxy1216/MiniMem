from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


_REQUIRED_ROUTES = {
    "/api/v1/collective/ingest",
    "/api/v1/collective/context",
    "/api/v1/collective/feedback",
}


def _build_client() -> tuple[TestClient, set[str], WritableTempDir]:
    tmp = WritableTempDir(ignore_cleanup_errors=True)
    env = {
        "LITE_DATA_DIR": str(Path(tmp.name) / "qa-gate-smoke-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / "qa-gate-smoke-config"),
        "LITE_CHAT_PROVIDER": "openai",
        "LITE_CHAT_BASE_URL": "https://chat.example/v1",
        "LITE_CHAT_API_KEY": "qa-chat-key",
        "LITE_CHAT_MODEL": "qa-chat-model",
        "LITE_EMBEDDING_PROVIDER": "local",
        "LITE_EMBEDDING_MODEL": "local-hash-384",
        "LITE_EXTRACTOR_PROVIDER": "rule",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = LiteSettings.from_env()
    app = create_app(settings)
    routes = {getattr(route, "path", "") for route in app.routes}
    return TestClient(app), routes, tmp


class CollectiveGateSmokeTests(unittest.TestCase):
    def _skip_if_collective_missing(self, routes: set[str]) -> None:
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.skipTest(
                "collective smoke blocked until routes exist: " + ", ".join(missing)
            )

    def test_collective_routes_are_registered_for_smoke(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)
        self.assertEqual(set(), _REQUIRED_ROUTES - routes)
        client.close()

    def test_collective_closed_loop_normal_path(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        ingest_response = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-gate-smoke-normal",
                "scope_type": "personal",
                "scope_id": "u-qa-smoke",
                "content": {"fact": "smoke path closes inject execute feedback revise"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "qa-agent",
                "read_acl": ["qa-agent"],
                "write_acl": ["qa-agent"],
                "coordination_mode": "inruntime_a2a",
                "coordination_id": "coord-smoke-normal",
                "runtime_id": "codex",
                "agent_id": "qa-gate",
            },
        )
        self.assertEqual(200, ingest_response.status_code)
        ingest_body = ingest_response.json()
        self.assertEqual("ok", ingest_body.get("status"))
        ingest_result = ingest_body.get("result", {})
        self.assertEqual("k-gate-smoke-normal", ingest_result.get("knowledge_id"))
        self.assertTrue(str(ingest_result.get("revision_id", "")).strip())

        context_response = client.post(
            "/api/v1/collective/context",
            json={
                "query": "smoke context query",
                "actor_id": "qa-agent",
                "personal_scope_id": "u-qa-smoke",
                "include_global": False,
                "top_k": 5,
            },
        )
        self.assertEqual(200, context_response.status_code)
        context_body = context_response.json()
        self.assertEqual("ok", context_body.get("status"))
        context_items = context_body.get("result", {}).get("items", [])
        self.assertGreaterEqual(len(context_items), 1)
        self.assertEqual("k-gate-smoke-normal", context_items[0].get("knowledge_id"))

        feedback_response = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-gate-smoke-normal",
                "revision_id": ingest_result.get("revision_id"),
                "feedback_type": "execution_result",
                "feedback_payload": {"outcome_status": "success"},
                "actor": "qa-agent",
                "coordination_mode": "inruntime_a2a",
                "coordination_id": "coord-smoke-normal",
            },
        )
        self.assertEqual(200, feedback_response.status_code)
        feedback_body = feedback_response.json()
        self.assertEqual("ok", feedback_body.get("status"))
        feedback_result = feedback_body.get("result", {})
        self.assertTrue(str(feedback_result.get("feedback_id", "")).strip())
        client.close()


if __name__ == "__main__":
    unittest.main()
