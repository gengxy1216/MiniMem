from __future__ import annotations

import unittest

from flockmem.service.http_auth import build_auth_headers, normalize_api_key_token


class HttpAuthHeaderTests(unittest.TestCase):
    def test_wraps_plain_token_with_bearer(self) -> None:
        headers = build_auth_headers("plain-token")
        self.assertEqual("application/json", headers.get("Content-Type"))
        self.assertEqual("Bearer plain-token", headers.get("Authorization"))

    def test_keeps_bearer_prefixed_token(self) -> None:
        headers = build_auth_headers("Bearer abc-123")
        self.assertEqual("Bearer abc-123", headers.get("Authorization"))

    def test_supports_full_authorization_value(self) -> None:
        headers = build_auth_headers("Authorization: Basic dXNlcjpwYXNz")
        self.assertEqual("Basic dXNlcjpwYXNz", headers.get("Authorization"))

    def test_normalize_api_key_token_strips_bearer_prefix(self) -> None:
        token = normalize_api_key_token("Bearer sk-test")
        self.assertEqual("sk-test", token)

    def test_normalize_api_key_token_strips_authorization_wrapper(self) -> None:
        token = normalize_api_key_token("Authorization: Bearer sk-test")
        self.assertEqual("sk-test", token)


if __name__ == "__main__":
    unittest.main()

