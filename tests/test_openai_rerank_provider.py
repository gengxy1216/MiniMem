from __future__ import annotations

import unittest

from flockmem.service.openai_rerank import OpenAIRerankProvider


class OpenAIRerankProviderTests(unittest.TestCase):
    def test_build_headers_wraps_raw_key_with_bearer(self) -> None:
        provider = OpenAIRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="raw-key",
            model="rerank-model",
        )
        headers = provider._build_headers("raw-key")
        self.assertEqual("application/json", headers.get("Content-Type"))
        self.assertEqual("Bearer raw-key", headers.get("Authorization"))

    def test_build_headers_keeps_prefixed_bearer_key(self) -> None:
        provider = OpenAIRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="Bearer token-a",
            model="rerank-model",
        )
        headers = provider._build_headers("Bearer token-a")
        self.assertEqual("Bearer token-a", headers.get("Authorization"))

    def test_build_headers_supports_authorization_prefix_value(self) -> None:
        provider = OpenAIRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="Authorization: Bearer token-a",
            model="rerank-model",
        )
        headers = provider._build_headers("Authorization: Bearer token-a")
        self.assertEqual("Bearer token-a", headers.get("Authorization"))

    def test_build_rerank_url(self) -> None:
        provider = OpenAIRerankProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
        )
        self.assertEqual(
            "https://api.example.com/v1/rerank",
            provider._build_rerank_url("https://api.example.com/v1"),
        )
        self.assertEqual(
            "https://api.example.com/rerank",
            provider._build_rerank_url("https://api.example.com"),
        )


if __name__ == "__main__":
    unittest.main()

