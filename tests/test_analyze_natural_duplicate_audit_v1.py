import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "analyze_natural_duplicate_audit_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_natural_duplicate_audit_v1", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzeNaturalDuplicateAuditV1Tests(unittest.TestCase):
    def test_frozen_inputs_have_expected_hashes(self):
        private = PROJECT_ROOT / "research" / "natural_duplicate_audit_v1" / "selection_private.csv"
        visual = PROJECT_ROOT / "research" / "natural_duplicate_visual_review_v1.csv"
        self.assertEqual(MODULE.file_sha256(private), MODULE.EXPECTED_PRIVATE_SHA256)
        self.assertEqual(MODULE.file_sha256(visual), MODULE.EXPECTED_VISUAL_SHA256)

    def test_actual_unblinded_summary_matches_frozen_review(self):
        private = MODULE.read_csv(
            PROJECT_ROOT / "research" / "natural_duplicate_audit_v1" / "selection_private.csv"
        )
        visual = MODULE.read_csv(
            PROJECT_ROOT / "research" / "natural_duplicate_visual_review_v1.csv"
        )
        merged = MODULE.merge_rows(private, visual)
        summary = MODULE.summarize(merged)
        self.assertEqual(len(merged), 345)
        self.assertEqual(
            summary["verdict_counts"],
            {
                "independent_lookalike": 322,
                "same_source_candidate": 21,
                "uncertain": 2,
            },
        )
        self.assertEqual(summary["audit_region_counts"], {"candidate": 153, "outside": 192})
        self.assertEqual(summary["outside_confirmed_positive_count"], 0)
        self.assertTrue(summary["challenge_set_gate_passed"])
        self.assertEqual(summary["uncertain_pair_ids"], ["N000252", "N000337"])

    def test_merge_rejects_mismatched_pair_ids(self):
        private = [{"pair_id": "N1", "selection_stratum": "outside_random"}]
        visual = [
            {
                "pair_id": "N2",
                "visual_verdict": "independent_lookalike",
            }
        ]
        with self.assertRaisesRegex(ValueError, "kapsamı eşleşmiyor"):
            MODULE.merge_rows(private, visual)


if __name__ == "__main__":
    unittest.main()
