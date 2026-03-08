from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evermemos_lite.bootstrap.app_factory import create_app
from evermemos_lite.config.settings import LiteSettings
from evermemos_lite.testing.writable_tempdir import WritableTempDir


class EventLogVectorWiringTests(unittest.TestCase):
    def test_app_factory_builds_independent_event_log_vector_store(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            env = {
                "LITE_DATA_DIR": str(Path(tmp) / "mem-data"),
                "LITE_CONFIG_DIR": str(Path(tmp) / "mem-config"),
                "LITE_EMBEDDING_PROVIDER": "local",
                "LITE_EMBEDDING_MODEL": "local-hash-384",
                "LITE_EXTRACTOR_PROVIDER": "rule",
                "LITE_CHAT_API_KEY": "",
                "LITE_EXTRACTOR_API_KEY": "",
                "LITE_RETRIEVAL_PROFILE": "agentic",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = LiteSettings.from_env()
            app = create_app(settings)
            service = app.state.memory_service
            self.assertIsNotNone(service.event_log_vector_store)
            self.assertTrue(bool(getattr(service.event_log_vector_store, "enabled", False)))
            self.assertNotEqual(
                str(service.vector_store.db_dir),
                str(service.event_log_vector_store.db_dir),
            )
            self.assertTrue(str(service.event_log_vector_store.db_dir).endswith("eventlog"))


if __name__ == "__main__":
    unittest.main()
