from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finalize_natural_duplicate_adjudication_v1 import (  # noqa: E402
    build_adjudicated_rows,
    build_final_references,
    rule_comparison,
    validate_decisions,
)


def queue_row(pair_id: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "adjudication_verdict": "",
        "adjudication_confidence": "",
        "adjudicator": "",
        "adjudication_date": "",
        "adjudication_notes": "",
    }


def decision_row(pair_id: str, verdict: str = "independent") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "adjudication_verdict": verdict,
        "adjudication_confidence": "high",
        "adjudicator": "reviewer",
        "adjudication_date": "2026-11-04",
        "adjudication_notes": "traceable reason",
    }


def score_row(
    pair_id: str, visual: str, rule: str, audit_region: str = "candidate"
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "human_verdict": visual,
        "decision": rule,
        "selection_stratum": "candidate_robust_only",
        "audit_region": audit_region,
        "source_group": "kidney_ct",
        "modality_key": "ct",
        "left_leaf": "kidney_normal",
        "right_leaf": "kidney_normal",
    }


class FinalizeNaturalAdjudicationTests(unittest.TestCase):
    def test_validation_requires_exact_pair_coverage(self) -> None:
        queue = [queue_row(f"N{i:06d}") for i in range(25)]
        decisions = [decision_row(f"N{i:06d}") for i in range(24)]
        with self.assertRaisesRegex(ValueError, "25 satır"):
            validate_decisions(queue, list(decisions[0]), decisions)

    def test_changed_blank_queue_is_rejected(self) -> None:
        queue = [queue_row(f"N{i:06d}") for i in range(25)]
        queue[0]["adjudication_verdict"] = "independent"
        decisions = [decision_row(f"N{i:06d}") for i in range(25)]
        with self.assertRaisesRegex(ValueError, "blank queue"):
            validate_decisions(queue, list(decisions[0]), decisions)

    def test_build_adjudicated_rows_fills_separate_decisions(self) -> None:
        queue = [queue_row(f"N{i:06d}") for i in range(25)]
        decision_rows = [decision_row(f"N{i:06d}") for i in range(25)]
        decisions = validate_decisions(queue, list(decision_rows[0]), decision_rows)
        merged = build_adjudicated_rows(queue, decisions)
        self.assertEqual(len(merged), 25)
        self.assertTrue(all(row["adjudication_verdict"] == "independent" for row in merged))
        self.assertTrue(all(not row["adjudication_verdict"] for row in queue))

    def test_final_references_use_consensus_and_adjudication(self) -> None:
        scores = [
            score_row("N000001", "same_source_candidate", "same_source"),
            score_row("N000002", "independent_lookalike", "independent", "outside"),
            score_row("N000003", "independent_lookalike", "uncertain"),
        ]
        while len(scores) < 345:
            index = len(scores) + 1
            scores.append(
                score_row(
                    f"N{index:06d}", "independent_lookalike", "independent", "outside"
                )
            )
        decisions = {"N000003": decision_row("N000003", "same_source")}
        final = build_final_references(scores, decisions)
        by_id = {row["pair_id"]: row for row in final}
        self.assertEqual(by_id["N000001"]["final_reference"], "same_source")
        self.assertEqual(by_id["N000002"]["final_reference"], "independent")
        self.assertEqual(by_id["N000003"]["final_reference"], "same_source")
        self.assertEqual(
            by_id["N000003"]["reference_basis"],
            "ai_assisted_score_aware_adjudication",
        )

    def test_rule_comparison_counts_abstention_separately(self) -> None:
        rows = [
            {"final_reference": "same_source", "frozen_rule_decision": "same_source"},
            {"final_reference": "same_source", "frozen_rule_decision": "uncertain"},
            {"final_reference": "independent", "frozen_rule_decision": "independent"},
        ]
        metrics = rule_comparison(rows)
        self.assertEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["decided_pair_accuracy"], 1.0)
        self.assertEqual(metrics["same_source_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
