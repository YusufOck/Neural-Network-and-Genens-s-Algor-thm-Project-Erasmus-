from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_candidate_generator_v2_4_counterfactual import (
    controlled_policy_hit,
    evaluate_controlled_references,
    evaluate_controlled_summary,
    is_inside,
    load_frozen_config,
    membership_count,
    normalized_text_sha256,
    proposed_group_union,
)


class V24CounterfactualTests(unittest.TestCase):
    def test_proposal_disables_phash_but_keeps_exact(self) -> None:
        robust = np.array([1, 2, 3], dtype=np.uint64)
        phash = np.array([3, 4, 5], dtype=np.uint64)
        exact = np.array([5, 6], dtype=np.uint64)
        baseline, proposal = proposed_group_union(robust, phash, exact, False)
        np.testing.assert_array_equal(baseline, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(proposal, np.array([1, 2, 3, 5, 6]))

    def test_proposal_keeps_phash_in_enabled_group(self) -> None:
        baseline, proposal = proposed_group_union(
            np.array([1, 2], dtype=np.uint64),
            np.array([2, 3], dtype=np.uint64),
            np.array([4], dtype=np.uint64),
            True,
        )
        np.testing.assert_array_equal(baseline, np.array([1, 2, 3]))
        np.testing.assert_array_equal(proposal, np.array([1, 2, 3, 4]))

    def test_controlled_policy_uses_robust_in_disabled_group(self) -> None:
        robust_row = {
            "source_group": "brain_ce_mri",
            "robust_top15_candidate": "yes",
            "flip_aware_phash_distance": "2",
        }
        phash_only_row = {
            "source_group": "brain_ce_mri",
            "robust_top15_candidate": "no",
            "flip_aware_phash_distance": "2",
        }
        self.assertTrue(controlled_policy_hit(robust_row, {"all_blood_smear"}))
        self.assertFalse(controlled_policy_hit(phash_only_row, {"all_blood_smear"}))

    def test_controlled_summary_keeps_evidence_by_group(self) -> None:
        rows = [
            {
                "pair_id": "P1",
                "source_group": "brain_ce_mri",
                "robust_top15_candidate": "yes",
                "flip_aware_phash_distance": "30",
            },
            {
                "pair_id": "P2",
                "source_group": "all_blood_smear",
                "robust_top15_candidate": "no",
                "flip_aware_phash_distance": "12",
            },
        ]
        result = evaluate_controlled_references(
            rows,
            {"brain_ce_mri", "all_blood_smear"},
            {"all_blood_smear"},
        )
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["misses"], 0)
        self.assertEqual(result["by_source_group"]["brain_ce_mri"]["hits"], 1)

    def test_membership_count_reports_missing_positions(self) -> None:
        hits, missing = membership_count(
            np.array([2, 5, 8], dtype=np.uint64),
            np.array([1, 2, 8], dtype=np.uint64),
        )
        self.assertEqual(hits, 2)
        self.assertEqual(missing, [1])

    def test_tracked_controlled_summary_matches_raw_ignored_metrics(self) -> None:
        result = evaluate_controlled_summary(
            ROOT / "research" / "candidate_generator_v2_4_controlled_reference_summary.json",
            {
                "all_blood_smear",
                "brain_ce_mri",
                "breast_histology",
                "cervix_pap_smear",
                "kidney_ct",
                "lung_colon_histology",
                "lymphoma_histology",
                "oral_histology",
            },
            {"brain_ce_mri", "kidney_ct"},
            ROOT
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_2_holdout"
            / "evaluation"
            / "positive_pair_metrics.csv",
        )
        self.assertTrue(result["raw_metrics_reverified_in_this_run"])
        self.assertEqual(result["references"], 1040)
        self.assertEqual(result["hits"], 1040)

    def test_frozen_config_hash_is_required(self) -> None:
        payload = {
            "status": "predeclared_counterfactual_before_full_artifact_evaluation"
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "policy.json"
            encoded = (json.dumps(payload) + "\n").encode("utf-8")
            config.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            config.with_suffix(".sha256").write_text(
                f"{digest}  {config.name}\n", encoding="utf-8"
            )
            loaded, actual = load_frozen_config(config)
            self.assertEqual(loaded, payload)
            self.assertEqual(actual, digest)

    def test_text_dependency_hash_is_line_ending_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lf_path = Path(temporary) / "lf.csv"
            crlf_path = Path(temporary) / "crlf.csv"
            lf_path.write_bytes(b"a,b\n1,2\n")
            crlf_path.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(
                normalized_text_sha256(lf_path),
                normalized_text_sha256(crlf_path),
            )

    def test_report_location_guard(self) -> None:
        root = Path("/tmp/frozen")
        self.assertTrue(is_inside(root / "report.json", root))
        self.assertFalse(is_inside(Path("/tmp/report.json"), root))


if __name__ == "__main__":
    unittest.main()
