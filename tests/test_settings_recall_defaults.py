from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from flockmem.config.settings import LiteSettings


class SettingsRecallDefaultsTests(unittest.TestCase):
    def test_recall_focused_defaults_align_with_bge_m3_stack(self) -> None:
        env = {
            "LITE_CHAT_API_KEY": "k",
            "LITE_EMBEDDING_API_KEY": "k",
        }
        if os.name == "nt":
            env["USERPROFILE"] = r"C:\Users\tester"
            env["LOCALAPPDATA"] = r"C:\Users\tester\AppData\Local"
        else:
            env["HOME"] = "/home/tester"
            env["XDG_DATA_HOME"] = "/home/tester/.local/share"
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        self.assertEqual("BAAI/bge-m3", settings.embedding_model)
        self.assertEqual("openai", settings.rerank_provider)
        self.assertEqual("BAAI/bge-reranker-v2-m3", settings.rerank_model)
        self.assertAlmostEqual(0.55, settings.vector_lancedb_min_importance, places=4)
        self.assertAlmostEqual(0.10, settings.vector_write_min_importance, places=4)
        self.assertEqual(8, settings.search_budget_factor)
        self.assertEqual(24, settings.search_min_probe_k)
        self.assertEqual(96, settings.semantic_vector_budget_cap)
        self.assertEqual(64, settings.semantic_keyword_budget_cap)
        self.assertEqual(6, settings.phase4_multi_hop_max_queries)
        self.assertFalse(settings.recall_mode)
        self.assertEqual(3, settings.extract_max_retries)
        self.assertEqual(50, settings.agentic_round_min_k)
        self.assertEqual(180, settings.agentic_round_max_k)
        self.assertTrue(settings.agentic_force_second_round)

    def test_recall_defaults_can_be_overridden_via_env(self) -> None:
        env = {
            "LITE_CHAT_API_KEY": "k",
            "LITE_EMBEDDING_API_KEY": "k",
            "LITE_EMBEDDING_MODEL": "custom-embed",
            "LITE_RERANK_PROVIDER": "chat_model",
            "LITE_RERANK_MODEL": "custom-rerank",
            "LITE_SEMANTIC_VECTOR_BUDGET_CAP": "40",
            "LITE_SEMANTIC_KEYWORD_BUDGET_CAP": "20",
        }
        if os.name == "nt":
            env["USERPROFILE"] = r"C:\Users\tester"
            env["LOCALAPPDATA"] = r"C:\Users\tester\AppData\Local"
        else:
            env["HOME"] = "/home/tester"
            env["XDG_DATA_HOME"] = "/home/tester/.local/share"
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        self.assertEqual("custom-embed", settings.embedding_model)
        self.assertEqual("chat_model", settings.rerank_provider)
        self.assertEqual("custom-rerank", settings.rerank_model)
        self.assertEqual(40, settings.semantic_vector_budget_cap)
        self.assertEqual(20, settings.semantic_keyword_budget_cap)


if __name__ == "__main__":
    unittest.main()

