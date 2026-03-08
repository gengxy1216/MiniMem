from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.service.embedding_factory import build_embedding_provider
from flockmem.service.local_embedding import LocalHashEmbeddingProvider
from flockmem.service.openai_embedding import OpenAIEmbeddingProvider
from flockmem.testing.writable_tempdir import WritableTempDir


class EmbeddingFactoryTests(unittest.TestCase):
    def test_build_openai_compatible_provider(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": "https://embed.example/v1",
                "LITE_EMBEDDING_API_KEY": "embed-key",
                "LITE_EMBEDDING_MODEL": "embed-model-a",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_embedding_provider(
                settings=settings,
                runtime_model_config={
                    "embedding_provider": "openai",
                    "embedding_base_url": "https://embed.example/v1",
                    "embedding_api_key": "embed-key",
                    "embedding_model": "embed-model-a",
                },
            )
            self.assertIsInstance(provider, OpenAIEmbeddingProvider)

    def test_build_local_provider_and_embed_is_deterministic(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_EMBEDDING_PROVIDER": "local",
                "LITE_EMBEDDING_MODEL": "local-hash-384",
                "LITE_LOCAL_EMBEDDING_DEVICE": "cpu",
                "LITE_LOCAL_EMBEDDING_BATCH_SIZE": "8",
                "LITE_LOCAL_EMBEDDING_MAX_CONCURRENCY": "2",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_embedding_provider(
                settings=settings,
                runtime_model_config={
                    "embedding_provider": "local",
                    "embedding_model": "local-hash-384",
                },
            )
            self.assertIsInstance(provider, LocalHashEmbeddingProvider)
            vec_a = provider.embed("测试 embedding local provider")
            vec_b = provider.embed("测试 embedding local provider")
            self.assertEqual(len(vec_a), 384)
            self.assertEqual(vec_a, vec_b)
            norm = math.sqrt(sum(float(x) * float(x) for x in vec_a))
            self.assertGreater(norm, 0.99)
            self.assertLess(norm, 1.01)

    def test_create_app_with_local_embedding_provider(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_PATH": str(Path(tmp) / "mem-config" / "config.json"),
                "LITE_EMBEDDING_PROVIDER": "local",
                "LITE_EMBEDDING_MODEL": "local-hash-384",
                "LITE_EXTRACTOR_PROVIDER": "rule",
                "LITE_CHAT_API_KEY": "",
                "LITE_EXTRACTOR_API_KEY": "",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
                app = create_app(settings)
                with TestClient(app) as client:
                    payload = {
                        "message_id": "local-embed-test-1",
                        "create_time": 1772627001,
                        "sender": "tester",
                        "content": "今天在上海和团队讨论了任务拆分、发布门禁、性能回测、质量基线和下周计划。",
                        "group_id": "unit-test",
                        "role": "user",
                    }
                    resp = client.post("/api/v1/memories", json=payload)
                    self.assertEqual(200, resp.status_code)
                    result = resp.json().get("result", {})
                    self.assertIn("vector_index", result)
                    self.assertIsInstance(result.get("vector_index"), dict)


if __name__ == "__main__":
    unittest.main()

