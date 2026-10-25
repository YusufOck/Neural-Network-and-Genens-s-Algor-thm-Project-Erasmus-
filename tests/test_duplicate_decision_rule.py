from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from duplicate_decision_rule import classify_pair, load_frozen_rule, load_rule  # noqa: E402
from evaluate_duplicate_rule_development import read_development_scores  # noqa: E402


RULE_PATH = PROJECT_ROOT / "config" / "duplicate_rule_v1.json"


def row(**overrides: str) -> dict[str, str]:
    base = {
        "sha256_equal": "no",
        "orb_affine_plausible": "no",
        "orb_inliers": "0",
        "orb_inlier_ratio": "0",
        "orb_coverage_min": "0",
        "best_ssim": "0",
        "best_correlation": "0",
        "best_nmi": "0",
    }
    base.update(overrides)
    return base


class DuplicateDecisionRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rule = load_rule(RULE_PATH)

    def test_exact_hash_is_same_source(self) -> None:
        result = classify_pair(row(sha256_equal="yes"), self.rule)
        self.assertEqual(result["decision"], "same_source")
        self.assertEqual(result["decision_path"], "sha256_exact")

    def test_strong_geometry_is_same_source(self) -> None:
        result = classify_pair(
            row(
                orb_affine_plausible="yes",
                orb_inliers="449",
                orb_inlier_ratio="0.66",
                orb_coverage_min="0.27",
                best_ssim="0.77",
                best_correlation="0.96",
                best_nmi="1.20",
            ),
            self.rule,
        )
        self.assertEqual(result["decision"], "same_source")
        self.assertEqual(result["decision_path"], "strong_geometry")

    def test_dark_brightness_relative_uses_intensity_path(self) -> None:
        result = classify_pair(
            row(best_ssim="0.87", best_correlation="0.99", best_nmi="1.51"),
            self.rule,
        )
        self.assertEqual(result["decision"], "same_source")
        self.assertEqual(result["decision_path"], "strong_intensity_structure")

    def test_low_evidence_is_independent(self) -> None:
        result = classify_pair(
            row(orb_inliers="8", best_ssim="0.33", best_correlation="0.52"),
            self.rule,
        )
        self.assertEqual(result["decision"], "independent")

    def test_c000620_like_evidence_is_uncertain(self) -> None:
        result = classify_pair(
            row(
                orb_affine_plausible="yes",
                orb_inliers="44",
                orb_inlier_ratio="0.291",
                orb_coverage_min="0.095",
                best_ssim="0.754",
                best_correlation="0.942",
                best_nmi="1.255",
            ),
            self.rule,
        )
        self.assertEqual(result["decision"], "uncertain")

    def test_missing_evidence_is_uncertain_not_independent(self) -> None:
        result = classify_pair({}, self.rule)
        self.assertEqual(result["decision"], "uncertain")

    def test_development_evaluator_rejects_holdout_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("pair_id", "audit_partition")
                )
                writer.writeheader()
                writer.writerow({"pair_id": "H1", "audit_partition": "holdout"})
            with self.assertRaises(ValueError):
                read_development_scores(path)

    def test_changed_rule_is_rejected_by_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied_rule = Path(temporary) / "duplicate_rule_v1.json"
            copied_hash = copied_rule.with_suffix(".sha256")
            copied_rule.write_text(
                RULE_PATH.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            copied_hash.write_text(
                (PROJECT_ROOT / "config" / "duplicate_rule_v1.sha256").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_frozen_rule(copied_rule)


if __name__ == "__main__":
    unittest.main()
