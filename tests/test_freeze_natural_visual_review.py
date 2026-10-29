import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "freeze_natural_visual_review.py"
SPEC = importlib.util.spec_from_file_location("freeze_natural_visual_review", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FreezeNaturalVisualReviewTests(unittest.TestCase):
    def test_frozen_blind_manifest_integrity(self):
        rows = MODULE.load_blind_manifest()
        self.assertEqual(len(rows), 345)
        self.assertEqual([row["pair_id"] for row in rows], MODULE.EXPECTED_PAIR_IDS)

    def test_decision_sets_are_valid_and_disjoint(self):
        expected = set(MODULE.EXPECTED_PAIR_IDS)
        self.assertFalse(MODULE.SAME_SOURCE & MODULE.UNCERTAIN)
        self.assertTrue(MODULE.SAME_SOURCE <= expected)
        self.assertTrue(MODULE.UNCERTAIN <= expected)
        self.assertTrue(MODULE.MEDIUM_INDEPENDENT <= expected)

    def test_all_345_pairs_receive_one_allowed_decision(self):
        allowed_verdicts = {
            "same_source_candidate",
            "independent_lookalike",
            "uncertain",
        }
        decisions = [MODULE.decision_for(pair_id) for pair_id in MODULE.EXPECTED_PAIR_IDS]
        self.assertEqual(len(decisions), 345)
        self.assertTrue(all(item["visual_verdict"] in allowed_verdicts for item in decisions))
        self.assertEqual(
            sum(item["visual_verdict"] == "same_source_candidate" for item in decisions),
            21,
        )
        self.assertEqual(
            sum(item["visual_verdict"] == "independent_lookalike" for item in decisions),
            322,
        )
        self.assertEqual(
            sum(item["visual_verdict"] == "uncertain" for item in decisions),
            2,
        )


if __name__ == "__main__":
    unittest.main()

