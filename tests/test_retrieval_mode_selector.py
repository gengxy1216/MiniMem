from __future__ import annotations

import unittest

from evermemos_lite.service.retrieval_mode_selector import (
    OpenAIRetrievalModeSelector,
    RuleRetrievalModeSelector,
    SelectionInput,
)


class StubOpenAISelector(OpenAIRetrievalModeSelector):
    def __init__(self, response_text: str) -> None:
        super().__init__(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
        )
        self._response_text = response_text

    def _chat_completion(self, *, model: str, messages: list[dict[str, str]]) -> str:
        return self._response_text


class RetrievalModeSelectorTests(unittest.TestCase):
    def test_rule_selector_defaults_to_agentic(self) -> None:
        selector = RuleRetrievalModeSelector()
        out = selector.select(SelectionInput(query="帮我梳理这个季度研发路线图和风险", top_k=20))
        self.assertTrue(bool(out.policy.agentic_enabled))
        self.assertEqual("agentic", out.policy.profile)

    def test_rule_selector_uses_keyword_for_short_precise_query(self) -> None:
        selector = RuleRetrievalModeSelector()
        out = selector.select(SelectionInput(query="我儿子是谁", top_k=10))
        self.assertFalse(bool(out.policy.vector_enabled))
        self.assertEqual("keyword", out.policy.profile)

    def test_openai_selector_parses_profile_json(self) -> None:
        selector = StubOpenAISelector('{"profile":"hybrid","reason":"short question"}')
        out = selector.select(SelectionInput(query="测试", top_k=10))
        self.assertEqual("hybrid", out.policy.profile)
        self.assertFalse(bool(out.policy.agentic_enabled))
        self.assertIn("short question", str(out.reason))


if __name__ == "__main__":
    unittest.main()
