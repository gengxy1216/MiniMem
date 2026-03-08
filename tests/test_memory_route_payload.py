from __future__ import annotations

import unittest

from flockmem.api.routes.memory import (
    _is_utf8_json_content_type,
    _normalize_memory_row,
    _validate_text_integrity,
)


class MemoryRoutePayloadTests(unittest.TestCase):
    def test_utf8_json_content_type_accepts_default_json(self) -> None:
        self.assertTrue(_is_utf8_json_content_type("application/json"))

    def test_utf8_json_content_type_accepts_utf8_charset(self) -> None:
        self.assertTrue(_is_utf8_json_content_type("application/json; charset=utf-8"))
        self.assertTrue(_is_utf8_json_content_type("application/json;charset=UTF8"))

    def test_utf8_json_content_type_rejects_non_utf8_charset(self) -> None:
        self.assertFalse(_is_utf8_json_content_type("application/json; charset=latin1"))

    def test_utf8_json_content_type_rejects_non_json_media_type(self) -> None:
        self.assertFalse(_is_utf8_json_content_type("text/plain; charset=utf-8"))

    def test_validate_text_integrity_rejects_replacement_char(self) -> None:
        with self.assertRaises(ValueError):
            _validate_text_integrity("飞书\ufffd", "content")

    def test_validate_text_integrity_rejects_garbled_question_marks(self) -> None:
        with self.assertRaises(ValueError):
            _validate_text_integrity("??????", "content")

    def test_validate_text_integrity_accepts_normal_utf8_text(self) -> None:
        value = _validate_text_integrity(" 飞书知识库 ", "content")
        self.assertEqual("飞书知识库", value)

    def test_normalize_row_adds_content_alias(self) -> None:
        row = {"id": "m1", "episode": "记忆正文", "summary": "摘要"}
        normalized = _normalize_memory_row(row)
        self.assertEqual("记忆正文", normalized.get("content"))


if __name__ == "__main__":
    unittest.main()

