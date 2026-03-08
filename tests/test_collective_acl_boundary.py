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
        "LITE_DATA_DIR": str(Path(tmp.name) / "qa-acl-boundary-data"),
        "LITE_CONFIG_DIR": str(Path(tmp.name) / "qa-acl-boundary-config"),
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


class CollectiveAclBoundaryTests(unittest.TestCase):
    def _skip_if_collective_missing(self, routes: set[str]) -> None:
        missing = sorted(_REQUIRED_ROUTES - routes)
        if missing:
            self.skipTest(
                "collective acl-boundary blocked until routes exist: "
                + ", ".join(missing)
            )

    def test_ingest_rejects_write_acl_violation(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        denied = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-acl-denied",
                "scope_type": "team",
                "scope_id": "team-a",
                "content": {"fact": "intruder should be denied"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "intruder",
                "write_acl": ["owner-1"],
            },
        )
        self.assertEqual(403, denied.status_code)
        client.close()

    def test_context_requires_scope_when_global_disabled(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        response = client.post(
            "/api/v1/collective/context",
            json={
                "query": "missing scope",
                "include_global": False,
                "top_k": 5,
            },
        )
        self.assertEqual(400, response.status_code)
        client.close()

    def test_context_rejects_top_k_out_of_range(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        response = client.post(
            "/api/v1/collective/context",
            json={
                "query": "invalid top_k",
                "personal_scope_id": "u-boundary",
                "top_k": 101,
            },
        )
        self.assertEqual(422, response.status_code)
        client.close()

    def test_context_enforces_read_acl_filtering(self) -> None:
        client, routes, tmp = _build_client()
        self.addCleanup(tmp.cleanup)
        self._skip_if_collective_missing(routes)

        ingest = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-acl-read-guard",
                "scope_type": "personal",
                "scope_id": "u-boundary",
                "content": {"fact": "private to owner"},
                "change_type": "create",
                "changed_by": "agent",
                "actor_id": "owner-1",
                "read_acl": ["owner-1"],
                "write_acl": ["owner-1"],
            },
        )
        self.assertEqual(200, ingest.status_code)

        denied_context = client.post(
            "/api/v1/collective/context",
            json={
                "query": "acl denied read",
                "actor_id": "intruder",
                "personal_scope_id": "u-boundary",
                "include_global": False,
                "top_k": 5,
            },
        )
        self.assertEqual(200, denied_context.status_code)
        denied_result = denied_context.json().get("result", {})
        self.assertEqual(0, int(denied_result.get("count", -1)))
        self.assertEqual([], denied_result.get("items", []))

        allowed_context = client.post(
            "/api/v1/collective/context",
            json={
                "query": "acl allowed read",
                "actor_id": "owner-1",
                "personal_scope_id": "u-boundary",
                "include_global": False,
                "top_k": 5,
            },
        )
        self.assertEqual(200, allowed_context.status_code)
        allowed_items = allowed_context.json().get("result", {}).get("items", [])
        self.assertGreaterEqual(len(allowed_items), 1)
        self.assertEqual("k-acl-read-guard", allowed_items[0].get("knowledge_id"))
        client.close()


if __name__ == "__main__":
    unittest.main()
