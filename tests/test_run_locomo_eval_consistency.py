from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_locomo_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_locomo_eval_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("failed to load tools/run_locomo_eval.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_evaluate_expected_target_consistency = _MOD._evaluate_expected_target_consistency


class RunLocomoEvalConsistencyTests(unittest.TestCase):
    def test_detects_name_mismatch_between_query_and_expected_message(self) -> None:
        ok, reason, overlap = _evaluate_expected_target_consistency(
            query="When did Melanie paint a sunrise?",
            expected_message_ids=["D1:12"],
            memories=[
                {
                    "message_id": "D1:12",
                    "content": "Caroline said she would be a great counselor.",
                    "sender": "Caroline",
                }
            ],
        )
        self.assertFalse(ok)
        self.assertIn("sender_name_mismatch", reason)
        self.assertEqual(0.0, overlap)

    def test_passes_when_expected_message_semantically_matches_query(self) -> None:
        ok, reason, overlap = _evaluate_expected_target_consistency(
            query="When did Melanie paint a sunrise?",
            expected_message_ids=["D1:10"],
            memories=[
                {
                    "message_id": "D1:10",
                    "content": "Melanie said she painted a lake sunrise last year.",
                    "sender": "Melanie",
                }
            ],
        )
        self.assertTrue(ok)
        self.assertEqual("ok", reason)
        self.assertGreater(overlap, 0.0)


if __name__ == "__main__":
    unittest.main()
