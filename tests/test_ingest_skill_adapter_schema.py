from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings
from flockmem.testing.writable_tempdir import WritableTempDir


class IngestSkillAdapterSchemaTests(unittest.TestCase):
    def _build_client(self, *, skill_enabled: bool = True) -> tuple[TestClient, LiteSettings]:
        tmp = WritableTempDir(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = {
            "LITE_DATA_DIR": str(Path(tmp.name) / "mem-data"),
            "LITE_CONFIG_DIR": str(Path(tmp.name) / "mem-config"),
            "LITE_CHAT_PROVIDER": "openai",
            "LITE_CHAT_BASE_URL": "https://chat.example/v1",
            "LITE_CHAT_API_KEY": "chat-key",
            "LITE_CHAT_MODEL": "chat-model-a",
            "LITE_EMBEDDING_PROVIDER": "local",
            "LITE_EMBEDDING_MODEL": "local-hash-384",
            "LITE_EXTRACTOR_PROVIDER": "rule",
            "LITE_SKILL_ADAPTER_ENABLED": "true" if skill_enabled else "false",
            "LITE_SKILL_ADAPTER_WHITELIST": "markitdown,pdf,pptx",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = LiteSettings.from_env()
        app = create_app(settings)
        return TestClient(app), settings

    def test_ingest_skill_output_accepts_whitelisted_skill_and_writes_memories(self) -> None:
        client, _ = self._build_client(skill_enabled=True)
        resp = client.post(
            "/api/v1/ingest/skill",
            json={
                "source_type": "pdf",
                "source_uri": "file:///tmp/demo.pdf",
                "summary": "这是摘要",
                "chunks": ["第一段内容", "第二段内容"],
                "skill_name": "pdf",
                "agent_id": "agent-a",
                "task_id": "task-1",
                "channel": "chan-a",
                "trace_id": "trace-a",
            },
        )
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        result = body.get("result", {})
        self.assertTrue(bool(result.get("accepted")))
        self.assertEqual(3, int(result.get("ingested_count", 0)))
        self.assertEqual("default:agent-a", result.get("group_id"))

        mem_resp = client.get("/api/v1/memories", params={"user_id": "agent-a", "group_id": "default:agent-a"})
        self.assertEqual(200, mem_resp.status_code)
        memories = mem_resp.json().get("result", {}).get("memories", [])
        self.assertGreaterEqual(len(memories), 1)
        first = str(memories[0].get("content", ""))
        self.assertIn("[metadata]", first)
        self.assertIn("task-1", first)
        self.assertIn("trace-a", first)

    def test_ingest_skill_output_returns_hint_when_skill_not_whitelisted(self) -> None:
        client, _ = self._build_client(skill_enabled=True)
        resp = client.post(
            "/api/v1/ingest/skill",
            json={
                "source_type": "text",
                "raw_text": "hello",
                "skill_name": "untrusted-skill",
                "sender": "tester",
            },
        )
        self.assertEqual(200, resp.status_code)
        result = resp.json().get("result", {})
        self.assertFalse(bool(result.get("accepted")))
        self.assertIn("allowed skills", str(result.get("hint", "")))

    def test_ingest_skill_output_returns_hint_when_adapter_disabled(self) -> None:
        client, _ = self._build_client(skill_enabled=False)
        resp = client.post(
            "/api/v1/ingest/skill",
            json={
                "source_type": "pdf",
                "summary": "a",
                "skill_name": "pdf",
                "sender": "tester",
            },
        )
        self.assertEqual(200, resp.status_code)
        result = resp.json().get("result", {})
        self.assertFalse(bool(result.get("accepted")))
        self.assertIn("LITE_SKILL_ADAPTER_ENABLED", str(result.get("hint", "")))

    def test_ingest_skill_output_requires_content_parts(self) -> None:
        client, _ = self._build_client(skill_enabled=True)
        resp = client.post(
            "/api/v1/ingest/skill",
            json={
                "source_type": "pdf",
                "skill_name": "pdf",
                "sender": "tester",
            },
        )
        self.assertEqual(400, resp.status_code)


if __name__ == "__main__":
    unittest.main()

