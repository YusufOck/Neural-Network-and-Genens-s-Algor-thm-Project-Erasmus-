from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apply_calibration_reviews import apply_decisions  # noqa: E402


def review_row(pair_id: str, partition: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "audit_partition": partition,
        "human_verdict": "",
        "confidence": "",
        "reviewer": "",
        "review_date": "",
        "review_notes": "",
        "review_status": "pending_visual_review",
    }


def decision(pair_id: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "visual_verdict": "independent_lookalike",
        "confidence": "high",
        "reviewer": "Codex AI-assisted visual review",
        "review_date": "2026-10-21",
        "review_notes": "Distinct local landmarks.",
    }


class ApplyCalibrationReviewTests(unittest.TestCase):
    def test_applies_development_decision_and_preserves_holdout(self) -> None:
        rows = [review_row("C000001", "development"), review_row("C000002", "holdout")]
        result = apply_decisions(rows, [decision("C000001")])
        self.assertEqual(result[0]["human_verdict"], "independent_lookalike")
        self.assertEqual(result[0]["review_status"], "reviewed_ai_assisted")
        self.assertEqual(result[1]["human_verdict"], "")
        self.assertEqual(result[1]["review_status"], "pending_visual_review")

    def test_rejects_holdout_decision(self) -> None:
        with self.assertRaises(ValueError):
            apply_decisions([review_row("C000002", "holdout")], [decision("C000002")])

    def test_rejects_unknown_pair_id(self) -> None:
        with self.assertRaises(ValueError):
            apply_decisions(
                [review_row("C000001", "development")], [decision("C999999")]
            )

    def test_rejects_conflicting_existing_decision(self) -> None:
        row = review_row("C000001", "development")
        row["human_verdict"] = "same_source_candidate"
        with self.assertRaises(ValueError):
            apply_decisions([row], [decision("C000001")])


if __name__ == "__main__":
    unittest.main()
