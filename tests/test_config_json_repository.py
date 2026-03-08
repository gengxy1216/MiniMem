from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.config_json import JsonConfigRepository
from flockmem.config.settings import LiteSettings
from flockmem.infra.sqlite.app_config_repository import AppConfigRepository
from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.testing.writable_tempdir import WritableTempDir


class ConfigJsonRepositoryTests(unittest.TestCase):
    def test_repository_creates_config_json_from_bootstrap_settings(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "chat-model-a",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-a",
                "LITE_RERANK_PROVIDER": "chat_model",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            repo = JsonConfigRepository(settings.config_path)
            payload = repo.ensure(settings)
            self.assertTrue(settings.config_path.exists())
            models = payload.get("models", {})
            self.assertEqual("chat-model-a", models.get("chat", {}).get("model"))
            self.assertEqual("chat_model", models.get("rerank", {}).get("provider"))
            self.assertEqual("https://chat.example/v1", models.get("rerank", {}).get("base_url"))
            self.assertEqual("chat-key", models.get("rerank", {}).get("api_key"))
            self.assertEqual("chat-model-a", models.get("rerank", {}).get("model"))
            self.assertIsInstance(payload.get("models", {}).get("embedding"), dict)
            self.assertIsInstance(payload.get("models", {}).get("rerank"), dict)
            self.assertNotIn("chat_model", models)
            self.assertNotIn("rerank_provider", models)
            repo.patch_model_config(
                settings=settings,
                patch={"chat_model": "chat-model-b"},
            )
            backup_path = settings.config_path.with_suffix(".json.bak")
            self.assertTrue(backup_path.exists())
            effective = repo.get_effective_settings(settings)
            self.assertEqual(settings.db_path, effective.db_path)
            runtime = repo.get_runtime_model_config(effective)
            self.assertEqual("chat-model-b", runtime.get("chat_model"))

    def test_create_app_ignores_legacy_sqlite_app_config_values(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp) / "mem-data"
            env = {
                "LITE_DATA_DIR": str(data_dir),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "model-from-env",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-a",
                "LITE_EXTRACTOR_PROVIDER": "rule",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            engine = SQLiteEngine(settings.db_path)
            init_schema(engine)
            legacy_repo = AppConfigRepository(engine)
            legacy_repo.upsert("chat_model", "legacy-model-in-sqlite")
            app = create_app(settings)
            self.assertEqual(
                "model-from-env",
                str(app.state.runtime_model_config.get("chat_model")),
            )
            raw = settings.config_path.read_text(encoding="utf-8")
            cfg = json.loads(raw)
            self.assertEqual("model-from-env", cfg.get("models", {}).get("chat", {}).get("model"))

    def test_repository_migrates_legacy_data_dir_config_to_new_config_path(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            data_dir = root / "mem-data"
            config_dir = root / "mem-config"
            env = {
                "LITE_DATA_DIR": str(data_dir),
                "LITE_CONFIG_DIR": str(config_dir),
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "model-from-env",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-a",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()

            legacy_path = data_dir / "config.json"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_payload = {
                "version": 1,
                "updated_at": 1735603200,
                "settings": {},
                "models": {
                    "chat_provider": "openai",
                    "chat_base_url": "https://chat.example/v1",
                    "chat_api_key": "chat-key",
                    "chat_model": "legacy-chat-model",
                    "embedding_provider": "openai",
                    "embedding_base_url": "https://embed.example/v1",
                    "embedding_api_key": "embed-key",
                    "embedding_model": "legacy-embed-model",
                    "extractor_provider": "rule",
                    "extractor_base_url": "https://chat.example/v1",
                    "extractor_api_key": "chat-key",
                    "extractor_model": "legacy-chat-model",
                },
            }
            legacy_path.write_text(
                json.dumps(legacy_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self.assertFalse(settings.config_path.exists())
            repo = JsonConfigRepository(settings.config_path)
            payload = repo.ensure(settings)

            self.assertTrue(settings.config_path.exists())
            self.assertEqual("legacy-chat-model", payload.get("models", {}).get("chat", {}).get("model"))
            self.assertEqual(
                "legacy-embed-model",
                payload.get("models", {}).get("embedding", {}).get("model"),
            )

    def test_repository_normalizes_structured_model_blocks(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "model-from-env",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-env",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            repo = JsonConfigRepository(settings.config_path)
            payload = repo.ensure(settings)
            payload["models"] = {
                "chat": {
                    "provider": "custom-chat",
                    "base_url": "https://chat-2.example/v1",
                    "api_key": "chat-2-key",
                    "model": "chat-2-model",
                },
                "embedding": {
                    "provider": "custom-embed",
                    "base_url": "https://embed-2.example/v1",
                    "api_key": "embed-2-key",
                    "model": "embed-2-model",
                },
                "extractor": {
                    "provider": "chat_model",
                    "base_url": "",
                    "api_key": "",
                    "model": "",
                },
                "rerank": {
                    "provider": "custom-rerank",
                    "base_url": "https://rerank.example/v1",
                    "api_key": "rerank-key",
                    "model": "rerank-model",
                },
            }
            normalized = repo.replace_raw_config(
                bootstrap_settings=settings,
                payload=payload,
            )
            models = normalized.get("models", {})
            self.assertEqual("custom-chat", models.get("chat", {}).get("provider"))
            self.assertEqual("chat-2-model", models.get("chat", {}).get("model"))
            self.assertEqual("custom-embed", models.get("embedding", {}).get("provider"))
            self.assertEqual("embed-2-model", models.get("embedding", {}).get("model"))
            self.assertEqual("custom-rerank", models.get("rerank", {}).get("provider"))
            self.assertEqual("rerank-model", models.get("rerank", {}).get("model"))
            self.assertIsInstance(models.get("chat"), dict)
            self.assertIsInstance(models.get("embedding"), dict)
            self.assertIsInstance(models.get("extractor"), dict)
            self.assertIsInstance(models.get("rerank"), dict)
            self.assertNotIn("chat_provider", models)
            self.assertNotIn("embedding_provider", models)
            self.assertNotIn("rerank_provider", models)

    def test_repository_normalizes_provider_prefixed_rerank_model(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_CHAT_PROVIDER": "openai",
                "LITE_CHAT_BASE_URL": "https://chat.example/v1",
                "LITE_CHAT_API_KEY": "chat-key",
                "LITE_CHAT_MODEL": "model-from-env",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-env",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            repo = JsonConfigRepository(settings.config_path)
            payload = repo.ensure(settings)
            payload["models"] = {
                "chat": {
                    "provider": "custom-maas",
                    "base_url": "https://maas.example/v2",
                    "api_key": "maas-key",
                    "model": "xopkimik25",
                },
                "embedding": {
                    "provider": "openai",
                    "base_url": "https://embed.example/v1",
                    "api_key": "embed-key",
                    "model": "embed-model-env",
                },
                "extractor": {
                    "provider": "chat_model",
                    "base_url": "https://maas.example/v2",
                    "api_key": "maas-key",
                    "model": "xopkimik25",
                },
                "rerank": {
                    "provider": "custom-maas",
                    "base_url": "https://maas.example/v2",
                    "api_key": "maas-key",
                    "model": "custom-maas/xopkimik25",
                },
                "chat_provider_options": {
                    "custom-maas": {
                        "base_url": "https://maas.example/v2",
                        "api_key": "maas-key",
                        "model": "xopkimik25",
                    }
                },
            }
            normalized = repo.replace_raw_config(
                bootstrap_settings=settings,
                payload=payload,
            )
            rerank = normalized.get("models", {}).get("rerank", {})
            self.assertEqual("custom-maas", rerank.get("provider"))
            self.assertEqual("xopkimik25", rerank.get("model"))


if __name__ == "__main__":
    unittest.main()

