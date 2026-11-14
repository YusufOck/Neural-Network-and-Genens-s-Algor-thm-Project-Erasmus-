from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_candidate_generator_v2_6b_prospective_audit as audit


class CandidateGeneratorV26BProspectiveAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selection_dir = ROOT / "research" / "candidate_generator_v2_6b_prospective_selection_v1"
        self.private = audit.read_csv(self.selection_dir / "selection_private.csv")
        self.public = audit.read_csv(self.selection_dir / "review_manifest.csv")
        self.visual = audit.read_csv(ROOT / "research" / "candidate_generator_v2_6b_visual_review_v1.csv")

    def test_frozen_hashes_match(self) -> None:
        paths = {
            "config": ROOT / "config" / "candidate_generator_v2_6b_prospective_audit.json",
            "private_selection": self.selection_dir / "selection_private.csv",
            "public_manifest": self.selection_dir / "review_manifest.csv",
            "selection_report": self.selection_dir / "selection_report.json",
            "visual_decisions": ROOT / "research" / "candidate_generator_v2_6b_visual_review_v1.csv",
            "v2_5_audit_report": ROOT / "research" / "candidate_generator_v2_5_prospective_audit_v1" / "unblinded_report.json",
            "v2_6a_report": ROOT / "research" / "candidate_generator_v2_6a_bounded_grid_v1.json",
        }
        for name, path in paths.items():
            self.assertEqual(audit.file_sha256(path), audit.EXPECTED_HASHES[name])

    def test_real_unblind_counts_and_gates(self) -> None:
        merged = audit.merge_rows(self.private, self.public, self.visual)
        self.assertEqual(len(merged), 160)
        summary = audit.summarize_visual(merged)
        self.assertEqual(summary["verdict_counts"], {
            "independent_lookalike": 135,
            "same_source_candidate": 24,
            "uncertain": 1,
        })
        self.assertEqual(summary["region_counts"], {"candidate": 32, "outside": 128})
        self.assertEqual(summary["candidate_same_source_count"], 9)
        self.assertEqual(summary["candidate_uncertain_count"], 0)
        self.assertEqual(summary["outside_same_source_count"], 15)
        self.assertEqual(summary["outside_same_source_by_group"], {
            "kidney_ct": 5,
            "lymphoma_histology": 10,
        })
        self.assertEqual(summary["outside_uncertain_count"], 1)
        self.assertEqual(summary["outside_uncertain_pair_ids"], ["V26P000007"])

        result = audit.evaluate(
            audit.read_json(ROOT / "config" / "candidate_generator_v2_6b_prospective_audit.json"),
            audit.read_json(self.selection_dir / "selection_report.json"),
            audit.read_json(ROOT / "research" / "candidate_generator_v2_5_prospective_audit_v1" / "unblinded_report.json"),
            audit.read_json(ROOT / "research" / "candidate_generator_v2_6a_bounded_grid_v1.json"),
            merged,
        )
        gates = result["acceptance_gates"]
        self.assertEqual(result["observed_gate_values"]["full_candidate_pairs"], 3_116_598)
        self.assertAlmostEqual(result["observed_gate_values"]["candidate_pairs_per_image"], 23.97383076923077)
        self.assertEqual(result["observed_gate_values"]["development_same2d_hits"], 4)
        self.assertFalse(gates["new_outside_same2d"])
        self.assertFalse(gates["unresolved_outside_uncertain"])
        self.assertFalse(gates["all_passed"])
        self.assertTrue(all(
            value
            for key, value in gates.items()
            if key not in {"new_outside_same2d", "unresolved_outside_uncertain", "all_passed"}
        ))

    def test_private_public_identity_mismatch_is_rejected(self) -> None:
        tampered = [dict(row) for row in self.public]
        tampered[0]["left_path"] = "changed.jpg"
        with self.assertRaisesRegex(ValueError, "private/public left_path"):
            audit.merge_rows(self.private, tampered, self.visual)

    def test_cross_leaf_pair_is_rejected(self) -> None:
        tampered = [dict(row) for row in self.private]
        tampered[0]["right_leaf"] = "different_leaf"
        matching_public = [dict(row) for row in self.public]
        matching_public[0]["right_leaf"] = "different_leaf"
        with self.assertRaisesRegex(ValueError, "cross-leaf"):
            audit.merge_rows(tampered, matching_public, self.visual)


if __name__ == "__main__":
    unittest.main()
