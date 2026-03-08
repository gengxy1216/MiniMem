from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.infra.sqlite.app_config_repository import AppConfigRepository
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.testing.writable_tempdir import WritableTempDir


class ModelConfigRouteTests(unittest.TestCase):
    def test_put_model_config_persists_to_config_json_only(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp) / "mem-data"
            admin_token = "model-admin-token"
            env = {
                "LITE_DATA_DIR": str(data_dir),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_ADMIN_TOKEN": admin_token,
                "LITE_ADMIN_ALLOW_LOCALHOST": "false",
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "model-base",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-a",
                "LITE_EXTRACTOR_PROVIDER": "rule",
                "LITE_RERANK_PROVIDER": "chat_model",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            app = create_app(settings)
            client = TestClient(app)
            headers = {"Authorization": f"Bearer {admin_token}"}

            unauth_resp = client.put("/api/v1/model-config", json={"chat_model": "x"})
            self.assertEqual(401, unauth_resp.status_code)

            resp = client.put(
                "/api/v1/model-config",
                json={
                    "chat_model": "model-updated",
                    "rerank": {
                        "provider": "custom-rerank",
                        "base_url": "https://rerank.example/v1",
                        "api_key": "rerank-key",
                        "model": "rerank-model-a",
                    },
                },
                headers=headers,
            )
            self.assertEqual(200, resp.status_code)
            body = resp.json()
            self.assertEqual("model-updated", body["result"]["updated"]["chat_model"])
            self.assertEqual(
                "custom-rerank", body["result"]["updated"]["rerank_provider"]
            )
            self.assertEqual(
                "https://rerank.example/v1",
                body["result"]["updated"]["rerank_base_url"],
            )
            self.assertNotEqual("rerank-key", body["result"]["updated"]["rerank_api_key"])
            self.assertEqual("[REDACTED]", str(body["result"]["updated"]["rerank_api_key"]))
            self.assertEqual(
                "rerank-model-a", body["result"]["updated"]["rerank_model"]
            )

            resp_get = client.get("/api/v1/model-config", headers=headers)
            self.assertEqual(200, resp_get.status_code)
            self.assertEqual(
                "model-updated",
                resp_get.json()["result"]["chat_model"],
            )
            self.assertEqual(
                "custom-rerank",
                resp_get.json()["result"]["rerank_provider"],
            )
            self.assertEqual(
                "https://rerank.example/v1",
                resp_get.json()["result"]["rerank_base_url"],
            )
            self.assertEqual(
                "rerank-model-a",
                resp_get.json()["result"]["rerank_model"],
            )
            self.assertNotEqual(
                "rerank-key",
                resp_get.json()["result"]["rerank_api_key"],
            )
            self.assertEqual("[REDACTED]", str(resp_get.json()["result"]["rerank_api_key"]))
            self.assertIsInstance(resp_get.json()["result"].get("rerank"), dict)

            cfg = json.loads(settings.config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "model-updated",
                cfg.get("models", {}).get("chat", {}).get("model"),
            )
            self.assertEqual(
                "custom-rerank",
                cfg.get("models", {}).get("rerank", {}).get("provider"),
            )
            self.assertEqual(
                "https://rerank.example/v1",
                cfg.get("models", {}).get("rerank", {}).get("base_url"),
            )
            self.assertEqual(
                "rerank-model-a",
                cfg.get("models", {}).get("rerank", {}).get("model"),
            )
            self.assertIsInstance(cfg.get("models", {}).get("rerank"), dict)
            self.assertNotIn("chat_model", cfg.get("models", {}))
            self.assertNotIn("rerank_provider", cfg.get("models", {}))
            repo = AppConfigRepository(SQLiteEngine(settings.db_path))
            self.assertIsNone(repo.get("chat_model"))


if __name__ == "__main__":
    unittest.main()

