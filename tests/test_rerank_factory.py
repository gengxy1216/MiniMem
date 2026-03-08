from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evermemos_lite.config.settings import LiteSettings
from evermemos_lite.service.chat_model_rerank import ChatModelRerankProvider
from evermemos_lite.service.local_rerank import LocalHeuristicRerankProvider
from evermemos_lite.service.openai_rerank import OpenAIRerankProvider
from evermemos_lite.service.rerank_factory import build_rerank_provider
from evermemos_lite.testing.writable_tempdir import WritableTempDir


class RerankFactoryTests(unittest.TestCase):
    def test_build_local_rerank_provider(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_RERANK_PROVIDER": "local",
                "LITE_RERANK_MODEL": "local-rerank-lexical-v1",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_rerank_provider(
                settings=settings,
                runtime_model_config={
                    "rerank_provider": "local",
                    "rerank_model": "local-rerank-lexical-v1",
                },
            )
            self.assertIsInstance(provider, LocalHeuristicRerankProvider)
            assert provider is not None
            scores = provider.rerank(
                query="上海 项目 进度",
                documents=["今天在上海讨论项目进度", "天气晴朗适合跑步"],
            )
            self.assertEqual(2, len(scores))
            self.assertGreater(float(scores[0]), float(scores[1]))

    def test_build_openai_compatible_rerank_provider(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_RERANK_PROVIDER": "openai",
                "LITE_RERANK_BASE_URL": "https://rerank.example/v1",
                "LITE_RERANK_API_KEY": "rk",
                "LITE_RERANK_MODEL": "rerank-model-a",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_rerank_provider(
                settings=settings,
                runtime_model_config={
                    "rerank_provider": "openai",
                    "rerank_base_url": "https://rerank.example/v1",
                    "rerank_api_key": "rk",
                    "rerank_model": "rerank-model-a",
                },
            )
            self.assertIsInstance(provider, OpenAIRerankProvider)

    def test_chat_model_rerank_provider_is_disabled(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_RERANK_PROVIDER": "chat_model",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_rerank_provider(
                settings=settings,
                runtime_model_config={"rerank_provider": "chat_model"},
            )
            self.assertIsNone(provider)

    def test_chat_model_rerank_provider_is_enabled_with_credentials(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_RERANK_PROVIDER": "chat_model",
                "LITE_RERANK_BASE_URL": "https://chat.example/v1",
                "LITE_RERANK_API_KEY": "rk",
                "LITE_RERANK_MODEL": "qwen-rerank",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            provider = build_rerank_provider(
                settings=settings,
                runtime_model_config={
                    "rerank_provider": "chat_model",
                    "rerank_base_url": "https://chat.example/v1",
                    "rerank_api_key": "rk",
                    "rerank_model": "qwen-rerank",
                },
            )
            self.assertIsInstance(provider, ChatModelRerankProvider)


if __name__ == "__main__":
    unittest.main()
