from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from duplicate_decision_rule import load_rule  # noqa: E402
from verify_duplicate_holdout import verify_score_rows  # noqa: E402


RULE = load_rule(PROJECT_ROOT / "config" / "duplicate_rule_v1.json")


def expected() -> dict[str, str]:
    return {
        "pair_id": "C001204",
        "audit_partition": "holdout",
        "pair_kind": "random_same_leaf_review",
        "left_leaf": "brain_glioma",
        "right_leaf": "brain_glioma",
        "modality_key": "mri",
        "source_group": "brain_ce_mri",
        "reference_label": "independent",
    }


def valid_score() -> dict[str, str]:
    row = {
        **expected(),
        "sha256_equal": "no",
        "orb_affine_plausible": "no",
        "orb_inliers": "3",
        "orb_inlier_ratio": "0.1",
        "orb_coverage_min": "0.01",
        "best_ssim": "0.3",
        "best_correlation": "0.4",
        "best_nmi": "1.0",
        "decision": "independent",
        "decision_path": "conservative_independent",
        "decision_reason": "Geometrik ve hizalanmis yapisal kanitlar birlikte dusuk.",
        "flag_sha256_exact": "no",
        "flag_strong_geometry": "no",
        "flag_strong_intensity_structure": "no",
        "flag_conservative_independent": "yes",
    }
    return row


class HoldoutVerificationTests(unittest.TestCase):
    def test_saved_decision_must_match_frozen_rule(self) -> None:
        score = {**valid_score(), "decision": "same_source"}
        with self.assertRaises(ValueError):
            verify_score_rows([score], [expected()], RULE)

    def test_duplicate_pair_id_is_rejected(self) -> None:
        score = valid_score()
        with self.assertRaises(ValueError):
            verify_score_rows([score, score], [expected()], RULE)

    def test_valid_saved_row_is_accepted(self) -> None:
        verify_score_rows([valid_score()], [expected()], RULE)


if __name__ == "__main__":
    unittest.main()
