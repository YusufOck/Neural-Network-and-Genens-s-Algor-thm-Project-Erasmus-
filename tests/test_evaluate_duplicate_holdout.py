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

from evaluate_duplicate_holdout import (  # noqa: E402
    evaluate_predictions,
    load_holdout_rows,
    validate_visual_decisions,
)


class HoldoutEvaluationTests(unittest.TestCase):
    def test_visual_decisions_must_match_expected_ids_exactly(self) -> None:
        decision = {
            "pair_id": "C001204",
            "visual_verdict": "independent_lookalike",
            "confidence": "high",
            "reviewer": "reviewer",
            "review_date": "2026-10-25",
            "review_notes": "note",
        }
        with self.assertRaises(ValueError):
            validate_visual_decisions([decision], {"C001204", "C001205"})

    def test_loader_rejects_development_decisions_as_holdout_scope(self) -> None:
        fields = (
            "pair_id",
            "audit_partition",
            "truth_status",
            "truth_basis",
            "human_verdict",
        )
        decision_fields = (
            "pair_id",
            "visual_verdict",
            "confidence",
            "reviewer",
            "review_date",
            "review_notes",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            decisions = root / "decisions.csv"
            with pairs.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "C001204",
                        "audit_partition": "holdout",
                        "truth_status": "review_required",
                        "truth_basis": "",
                        "human_verdict": "",
                    }
                )
            with decisions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=decision_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "C000533",
                        "visual_verdict": "independent_lookalike",
                        "confidence": "high",
                        "reviewer": "reviewer",
                        "review_date": "2026-10-25",
                        "review_notes": "note",
                    }
                )
            with self.assertRaises(ValueError):
                load_holdout_rows(pairs, decisions)

    def test_metrics_keep_uncertain_as_abstention(self) -> None:
        rows = [
            {"reference_label": "same_source", "decision": "same_source"},
            {"reference_label": "same_source", "decision": "uncertain"},
            {"reference_label": "independent", "decision": "independent"},
            {"reference_label": "independent", "decision": "same_source"},
        ]
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["overall_accuracy_counting_uncertain_as_not_correct"], 0.5)
        self.assertAlmostEqual(metrics["accuracy_on_decided_pairs"], 2 / 3)
        self.assertEqual(metrics["same_source_recall"], 0.5)
        self.assertEqual(metrics["independent_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
