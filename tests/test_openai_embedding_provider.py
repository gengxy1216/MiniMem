from __future__ import annotations

import unittest

from flockmem.service.openai_embedding import OpenAIEmbeddingProvider


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    def test_build_headers_wraps_raw_key_with_bearer(self) -> None:
        provider = OpenAIEmbeddingProvider(
            base_url="https://api.example.com/v1",
            api_key="raw-key",
            model="embed-model",
        )
        headers = provider._build_headers("raw-key")
        self.assertEqual("application/json", headers.get("Content-Type"))
        self.assertEqual("Bearer raw-key", headers.get("Authorization"))

    def test_build_headers_keeps_prefixed_bearer_key(self) -> None:
        provider = OpenAIEmbeddingProvider(
            base_url="https://api.example.com/v1",
            api_key="Bearer real-token",
            model="embed-model",
        )
        headers = provider._build_headers("Bearer real-token")
        self.assertEqual("Bearer real-token", headers.get("Authorization"))

    def test_build_headers_supports_authorization_prefix_value(self) -> None:
        provider = OpenAIEmbeddingProvider(
            base_url="https://api.example.com/v1",
            api_key="Authorization: Bearer real-token",
            model="embed-model",
        )
        headers = provider._build_headers("Authorization: Bearer real-token")
        self.assertEqual("Bearer real-token", headers.get("Authorization"))


if __name__ == "__main__":
    unittest.main()

