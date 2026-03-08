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
        "LITE_DATA_DIR": str(Path(tmp.name) / "qa-recovery-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / "qa-recovery-config"),
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


class CollectiveErrorRecoveryTests(unittest.TestCase):
    def _skip_if_collective_missing(self, routes: set[str]) -> None:
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.skipTest(
                "collective error-recovery tests blocked until routes exist: "
                + ", ".join(missing)
            )

    def test_invalid_payload_returns_client_error_not_server_error(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        response = client.post(
            "/api/v1/collective/feedback",
            json={"knowledge_id": " ", "feedback_type": " ", "feedback_payload": {}},
        )
        self.assertGreaterEqual(response.status_code, 400)
        self.assertLess(response.status_code, 500)
        client.close()

    def test_acl_denied_then_valid_ingest_recovers(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        denied = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-acl-denied",
                "scope_type": "team",
                "scope_id": "team-recovery",
                "content": {"fact": "first write should be denied"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "intruder",
                "write_acl": ["owner-1"],
            },
        )
        self.assertEqual(403, denied.status_code)

        recovered = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-acl-denied",
                "scope_type": "team",
                "scope_id": "team-recovery",
                "content": {"fact": "owner write after deny"},
                "change_type": "update",
                "changed_by": "agent",
                "actor_id": "owner-1",
                "write_acl": ["owner-1"],
                "read_acl": ["owner-1"],
            },
        )
        self.assertEqual(200, recovered.status_code)
        self.assertEqual("ok", recovered.json().get("status"))
        client.close()

    def test_bad_context_then_followup_context_request_does_not_crash(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        bad = client.post(
            "/api/v1/collective/context",
            json={"query": "bad scope", "include_global": False},
        )
        self.assertEqual(400, bad.status_code)

        ingest = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-followup",
                "scope_type": "personal",
                "scope_id": "u-recovery",
                "content": {"fact": "service still handles followup"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "qa-recovery",
                "read_acl": ["qa-recovery"],
                "write_acl": ["qa-recovery"],
            },
        )
        self.assertEqual(200, ingest.status_code)

        followup = client.post(
            "/api/v1/collective/context",
            json={
                "query": "qa followup after bad request",
                "actor_id": "qa-recovery",
                "personal_scope_id": "u-recovery",
                "include_global": False,
                "top_k": 5,
            },
        )
        self.assertEqual(200, followup.status_code)
        followup_result = followup.json().get("result", {})
        self.assertGreaterEqual(int(followup_result.get("count", 0)), 1)
        client.close()

    def test_unknown_knowledge_feedback_then_valid_feedback_recovers(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        missing = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-recovery-missing-knowledge",
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "failed"},
                "actor": "qa-recovery",
            },
        )
        self.assertEqual(404, missing.status_code)

        ingest = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-feedback",
                "scope_type": "personal",
                "scope_id": "u-recovery",
                "content": {"fact": "feedback recovery target"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "qa-recovery",
                "read_acl": ["qa-recovery"],
                "write_acl": ["qa-recovery"],
            },
        )
        self.assertEqual(200, ingest.status_code)
        revision_id = ingest.json().get("result", {}).get("revision_id")

        recovered = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-recovery-feedback",
                "revision_id": revision_id,
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "success"},
                "actor": "qa-recovery",
            },
        )
        self.assertEqual(200, recovered.status_code)
        self.assertEqual("ok", recovered.json().get("status"))
        client.close()

    def test_existing_knowledge_with_unknown_revision_returns_not_found(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        ingest = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-recovery-revision-check",
                "scope_type": "personal",
                "scope_id": "u-recovery",
                "content": {"fact": "revision guard seed"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "qa-recovery",
                "read_acl": ["qa-recovery"],
                "write_acl": ["qa-recovery"],
            },
        )
        self.assertEqual(200, ingest.status_code)

        missing_revision = client.post(
            "/api/v1/collective/feedback",
            json={
                "knowledge_id": "k-recovery-revision-check",
                "revision_id": "rev-does-not-exist",
                "feedback_type": "execution_signal",
                "feedback_payload": {"outcome_status": "failed"},
                "actor": "qa-recovery",
            },
        )
        self.assertEqual(404, missing_revision.status_code)
        self.assertLess(missing_revision.status_code, 500)
        client.close()


if __name__ == "__main__":
    unittest.main()
