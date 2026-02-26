from __future__ import annotations

import unittest

from evermemos_lite.api.routes.memory import _build_retrieve_method_patch


class MemoryRoutePolicyTests(unittest.TestCase):
    def test_hybrid_explicitly_disables_agentic(self) -> None:
        patch = _build_retrieve_method_patch("hybrid")
        self.assertTrue(bool(patch.vector_enabled))
        self.assertTrue(bool(patch.keyword_enabled))
        self.assertFalse(bool(patch.agentic_enabled))

    def test_vector_explicitly_disables_agentic(self) -> None:
        patch = _build_retrieve_method_patch("vector")
        self.assertTrue(bool(patch.vector_enabled))
        self.assertFalse(bool(patch.keyword_enabled))
        self.assertFalse(bool(patch.agentic_enabled))

    def test_agentic_keeps_agent_mode_enabled(self) -> None:
        patch = _build_retrieve_method_patch("agentic")
        self.assertTrue(bool(patch.vector_enabled))
        self.assertTrue(bool(patch.keyword_enabled))
        self.assertTrue(bool(patch.agentic_enabled))


if __name__ == "__main__":
    unittest.main()
