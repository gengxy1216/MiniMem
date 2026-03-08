from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


class CollectiveRouteTests(unittest.TestCase):
    def _build_client(self) -> TestClient:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = {
            "LITE_DATA_DIR": str(Path(tmp.name) / "collective-routes-data"),
            "LITE_CONFIG_DIR": str(Path(tmp.name) / "collective-routes-config"),
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "chat-key",
            "LITE_CHAT_MODEL": "chat-model",
            "LITE_EMBEDDING_PROVIDER": "local",
            "LITE_EMBEDDING_MODEL": "local-hash-384",
            "LITE_EXTRACTOR_PROVIDER": "rule",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        return TestClient(create_app(settings))

    def test_collective_routes_closed_loop_normal_path(self) -> None:
        client = self._build_client()
        ingest_resp = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-route-1",
                "scope_type": "personal",
                "scope_id": "u-route",
                "content": {"fact": "route level closed loop seed"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "route-agent",
                "read_acl": ["route-agent"],
                "write_acl": ["route-agent"],
                "coordination_mode": "inruntime_a2a",
                "coordination_id": "coord-route-1",
            },
        )
        self.assertEqual(200, ingest_resp.status_code)
        ingest_result = ingest_resp.json()["result"]

        context_resp = client.post(
            "/api/v1/collective/context",
            json={
                "actor_id": "route-agent",
                "personal_scope_id": "u-route",
                "include_global": False,
                "top_k": 5,
                "coordination_mode": "inruntime_a2a",
                "coordination_id": "coord-route-ctx-1",
            },
        )
        self.assertEqual(200, context_resp.status_code)
        context_items = context_resp.json()["result"]["items"]
        self.assertGreaterEqual(len(context_items), 1)
        self.assertEqual("k-route-1", context_items[0]["knowledge_id"])

        feedback_resp = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-route-1",
                "revision_id": ingest_result["revision_id"],
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "success"},
                "actor": "route-agent",
                "coordination_mode": "inruntime_a2a",
                "coordination_id": "coord-route-fb-1",
                "runtime_id": "codex",
                "agent_id": "route-agent",
            },
        )
        self.assertEqual(200, feedback_resp.status_code)
        feedback_result = feedback_resp.json()["result"]
        self.assertEqual("k-route-1", feedback_result["knowledge_id"])
        self.assertTrue(str(feedback_result["feedback_id"]).strip())
        client.close()

    def test_collective_route_boundary_errors(self) -> None:
        client = self._build_client()
        denied = client.post(
            "/api/v1/collective/ingest",
            json={
                "scope_type": "team",
                "scope_id": "team-route",
                "content": {"fact": "write acl denied"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "intruder-route",
                "write_acl": ["owner-route"],
            },
        )
        self.assertEqual(403, denied.status_code)

        bad_context = client.post(
            "/api/v1/collective/context",
            json={"include_global": False},
        )
        self.assertEqual(400, bad_context.status_code)
        client.close()

    def test_collective_route_recovery_after_validation_error(self) -> None:
        client = self._build_client()
        bad_feedback = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-route-recovery",
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "failed"},
            },
        )
        self.assertEqual(422, bad_feedback.status_code)

        ingest_resp = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-route-recovery",
                "scope_type": "personal",
                "scope_id": "u-recovery",
                "content": {"fact": "recovery seed"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "route-agent",
            },
        )
        self.assertEqual(200, ingest_resp.status_code)
        client.close()


if __name__ == "__main__":
    unittest.main()
