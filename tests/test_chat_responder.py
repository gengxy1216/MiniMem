from __future__ import annotations

import unittest
from datetime import datetime, timezone

from evermemos_lite.service.chat_responder import ChatResponder


class StubChatResponder(ChatResponder):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            provider="openai",
        )
        self.last_messages: list[dict[str, str]] = []

    def _chat_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        self.last_messages = messages
        return "stub-answer"


class ChatResponderCitationTests(unittest.TestCase):
    def test_respond_picks_relevant_late_slice(self) -> None:
        responder = StubChatResponder()
        target = (
            "具体来看，AI如何体现新质生产力呢？第一，是它的高科技特征。"
            "第二，是它的高效能特征。"
        )
        episode = ("甲" * 960) + target + ("乙" * 120)
        memory = {
            "id": "m1",
            "summary": "这是开场介绍，不包含核心问答片段。",
            "episode": episode,
            "source": "vector",
            "storage_tier": "vector_only",
            "importance_score": 0.9,
            "score": 0.88,
            "timestamp": 1771900069,
        }
        result = responder.respond(
            query="AI和新质生产力是什么关系？",
            memories=[memory],
            system_time=datetime.now(timezone.utc),
        )
        citations = result["citations"]
        self.assertEqual(1, len(citations))
        citation = citations[0]
        self.assertEqual(5, citation.get("citation_slice"))
        self.assertIn("AI如何体现新质生产力", str(citation.get("citation_snippet")))
        self.assertGreater(float(citation.get("citation_match_score", 0.0)), 0.0)
        self.assertIn("AI如何体现新质生产力", responder.last_messages[1]["content"])

    def test_respond_fallbacks_to_first_slice_for_unmatched_query(self) -> None:
        responder = StubChatResponder()
        episode = ("第一段主题是财务流程优化。第二段主题是运营管理。") * 12
        result = responder.respond(
            query="完全无关的问题",
            memories=[
                {
                    "id": "m2",
                    "summary": "文档摘要",
                    "episode": episode,
                    "source": "vector",
                }
            ],
            system_time=datetime.now(timezone.utc),
        )
        citation = result["citations"][0]
        self.assertEqual(1, citation.get("citation_slice"))
        self.assertTrue(str(citation.get("citation_snippet", "")).startswith("第一段主题"))
        self.assertGreaterEqual(float(citation.get("citation_match_score", 0.0)), 0.0)

    def test_respond_handles_empty_memory_text_without_crash(self) -> None:
        responder = StubChatResponder()
        result = responder.respond(
            query="测试",
            memories=[
                {
                    "id": "m3",
                    "summary": None,
                    "episode": None,
                    "source": "vector",
                }
            ],
            system_time=datetime.now(timezone.utc),
        )
        self.assertEqual("stub-answer", result["answer"])
        self.assertEqual(1, len(result["citations"]))
        citation = result["citations"][0]
        self.assertEqual("", citation.get("citation_snippet"))
        self.assertEqual(0.0, float(citation.get("citation_match_score", 0.0)))
        self.assertIn("当前没有检索到可用记忆", responder.last_messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
