from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evermemos_lite.config.settings import LiteSettings


class SettingsPathTests(unittest.TestCase):
    def test_default_data_dir_uses_user_scope_location(self) -> None:
        env = {
            "LITE_CHAT_API_KEY": "k",
            "LITE_EMBEDDING_API_KEY": "k",
        }
        if os.name == "nt":
            env["LOCALAPPDATA"] = r"C:\Users\tester\AppData\Local"
            expected_data_dir = Path(r"C:\Users\tester\AppData\Local\MiniMem").resolve()
        else:
            env["XDG_DATA_HOME"] = "/home/tester/.local/share"
            expected_data_dir = Path("/home/tester/.local/share/minimem").resolve()

        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        self.assertEqual(expected_data_dir, settings.data_dir)
        self.assertEqual((expected_data_dir / "lancedb").resolve(), settings.lancedb_dir)
        self.assertEqual((expected_data_dir / "kuzu").resolve(), settings.graph_dir)
        self.assertEqual((expected_data_dir / "lite.db").resolve(), settings.db_path)

    def test_lite_data_dir_env_overrides_default(self) -> None:
        env = {
            "LITE_DATA_DIR": r"C:\mem-data" if os.name == "nt" else "/tmp/mem-data",
            "LITE_CHAT_API_KEY": "k",
            "LITE_EMBEDDING_API_KEY": "k",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        expected_data_dir = Path(env["LITE_DATA_DIR"]).resolve()
        self.assertEqual(expected_data_dir, settings.data_dir)
        self.assertEqual((expected_data_dir / "lancedb").resolve(), settings.lancedb_dir)
        self.assertEqual((expected_data_dir / "kuzu").resolve(), settings.graph_dir)

    def test_search_trace_env_is_parsed(self) -> None:
        env = {
            "LITE_CHAT_API_KEY": "k",
            "LITE_EMBEDDING_API_KEY": "k",
            "LITE_SEARCH_TRACE_ENABLED": "true",
            "LITE_SEARCH_TRACE_SLOW_MS": "150",
        }
        if os.name == "nt":
            env["LOCALAPPDATA"] = r"C:\Users\tester\AppData\Local"
        else:
            env["XDG_DATA_HOME"] = "/home/tester/.local/share"
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        self.assertTrue(settings.search_trace_enabled)
        self.assertEqual(150, settings.search_trace_slow_ms)


if __name__ == "__main__":
    unittest.main()
