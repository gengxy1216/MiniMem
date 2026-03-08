from __future__ import annotations

import unittest

from flockmem.service.chat_model_rerank import ChatModelRerankProvider


class ChatModelRerankProviderTests(unittest.TestCase):
    def test_build_completion_url(self) -> None:
        provider = ChatModelRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
        )
        self.assertEqual(
            "https://api.example.com/v1/chat/completions",
            provider._build_completion_url("https://api.example.com/v1"),
        )
        self.assertEqual(
            "https://api.example.com/chat/completions",
            provider._build_completion_url("https://api.example.com"),
        )

    def test_safe_json_extracts_fenced_payload(self) -> None:
        payload = "```json\n{\"scores\":[0.9,0.1]}\n```"
        data = ChatModelRerankProvider._safe_json(payload)
        self.assertEqual([0.9, 0.1], data.get("scores"))

    def test_build_headers_wraps_key_with_bearer(self) -> None:
        provider = ChatModelRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="raw-key",
            model="m",
        )
        headers = provider._build_headers("raw-key")
        self.assertEqual("application/json", headers.get("Content-Type"))
        self.assertEqual("Bearer raw-key", headers.get("Authorization"))


if __name__ == "__main__":
    unittest.main()

