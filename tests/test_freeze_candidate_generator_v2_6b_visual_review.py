import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "freeze_candidate_generator_v2_6b_visual_review.py"
SPEC = importlib.util.spec_from_file_location("freeze_v2_6b_visual_review", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FreezeCandidateGeneratorV26bVisualReviewTests(unittest.TestCase):
    def test_frozen_public_manifest_integrity(self):
        rows = MODULE.load_blind_manifest()
        self.assertEqual(len(rows), 160)
        self.assertEqual([row["pair_id"] for row in rows], MODULE.EXPECTED_PAIR_IDS)

    def test_decision_sets_are_valid_and_disjoint(self):
        MODULE.validate_decision_sets()

    def test_all_pairs_receive_one_allowed_decision(self):
        decisions = [MODULE.decision_for(pair_id) for pair_id in MODULE.EXPECTED_PAIR_IDS]
        self.assertEqual(len(decisions), 160)
        self.assertEqual(
            sum(item["visual_verdict"] == "same_source_candidate" for item in decisions),
            24,
        )
        self.assertEqual(
            sum(item["visual_verdict"] == "independent_lookalike" for item in decisions),
            135,
        )
        self.assertEqual(
            sum(item["visual_verdict"] == "uncertain" for item in decisions),
            1,
        )

    def test_same_source_note_matches_same_2d_target(self):
        for pair_id in MODULE.SAME_SOURCE:
            note = MODULE.decision_for(pair_id)["review_notes"]
            self.assertIn("same-2D", note)
            self.assertNotIn("adjacent-slice", note)


if __name__ == "__main__":
    unittest.main()
