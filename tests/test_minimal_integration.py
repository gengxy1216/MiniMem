from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


class MinimalIntegrationTests(unittest.TestCase):
    def _build_client(self) -> tuple[TestClient, LiteSettings, dict[str, str]]:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        data_dir = Path(tmp.name) / "integration-data"
        admin_token = "integration-admin-token"
        env = {
            "LITE_DATA_DIR": str(data_dir),
            "LITE_CONFIG_DIR": str(Path(tmp.name) / "integration-config"),
            "LITE_ADMIN_TOKEN": admin_token,
            "LITE_ADMIN_ALLOW_LOCALHOST": "false",
            "LITE_RETRIEVAL_PROFILE": "keyword",
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "int-chat-key",
            "LITE_CHAT_MODEL": "int-chat-model",
            "LITE_EMBEDDING_PROVIDER": "openai",
            "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
            "LITE_EMBEDDING_API_KEY": "int-embed-key",
            "LITE_EMBEDDING_MODEL": "int-embed-model",
            "LITE_EXTRACTOR_PROVIDER": "rule",
            "LITE_RERANK_PROVIDER": "chat_model",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        app = create_app(settings)
        headers = {"Authorization": f"Bearer {admin_token}"}
        return TestClient(app), settings, headers

    def test_protected_config_endpoints_require_auth(self) -> None:
        client, _, _ = self._build_client()
        self.assertEqual(401, client.get("/api/v1/model-config").status_code)
        self.assertEqual(401, client.get("/api/v1/config/raw").status_code)

    def test_admin_auth_and_redaction_work_for_model_config(self) -> None:
        client, _, headers = self._build_client()
        resp = client.get("/api/v1/model-config", headers=headers)
        self.assertEqual(200, resp.status_code)
        payload = resp.json()["result"]
        self.assertEqual("[REDACTED]", payload.get("chat_api_key"))
        self.assertEqual("[REDACTED]", payload.get("embedding_api_key"))
        resp_x = client.get(
            "/api/v1/model-config",
            headers={"X-API-Key": "integration-admin-token"},
        )
        self.assertEqual(200, resp_x.status_code)

    def test_raw_config_redaction_roundtrip_keeps_secret_values(self) -> None:
        client, settings, headers = self._build_client()
        raw = client.get("/api/v1/config/raw", headers=headers).json()["result"]["config"]
        self.assertEqual(
            "[REDACTED]",
            raw.get("models", {}).get("chat", {}).get("api_key"),
        )
        raw["settings"]["search_trace_enabled"] = True
        resp = client.put("/api/v1/config/raw", headers=headers, json={"config": raw})
        self.assertEqual(200, resp.status_code)
        stored = json.loads(settings.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            "int-chat-key",
            stored.get("models", {}).get("chat", {}).get("api_key"),
        )

    def test_minimal_memory_write_and_keyword_search_flow(self) -> None:
        client, _, _ = self._build_client()
        payload = {
            "message_id": "integration-msg-1",
            "create_time": int(time.time()),
            "sender": "integration-user",
            "content": "集成测试：我今天学习了 FlockMem 的最小集成流程。",
            "group_id": "integration-group",
            "role": "user",
        }
        write_resp = client.post(
            "/api/v1/memories",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json=payload,
        )
        self.assertEqual(200, write_resp.status_code)
        write_body = write_resp.json()
        self.assertEqual("ok", write_body.get("status"))
        req_id = str(write_body.get("request_id") or "")
        self.assertTrue(req_id)
        status_resp = client.get(f"/api/v1/status/{req_id}")
        self.assertEqual(200, status_resp.status_code)

        search_resp = client.get(
            "/api/v1/memories/search",
            params={
                "query": "FlockMem 集成流程",
                "group_id": "integration-group",
                "retrieve_method": "keyword",
                "decision_mode": "static",
                "top_k": 10,
            },
        )
        self.assertEqual(200, search_resp.status_code)
        memories = search_resp.json().get("result", {}).get("memories", [])
        self.assertGreaterEqual(len(memories), 1)


if __name__ == "__main__":
    unittest.main()


