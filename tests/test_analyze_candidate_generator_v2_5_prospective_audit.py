from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_candidate_generator_v2_5_prospective_audit as audit


class CandidateGeneratorV25ProspectiveAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selection_dir = ROOT / "research" / "candidate_generator_v2_5_prospective_selection_v1"
        self.private = audit.read_csv(self.selection_dir / "selection_private.csv")
        self.public = audit.read_csv(self.selection_dir / "review_manifest.csv")
        self.visual = audit.read_csv(ROOT / "research" / "candidate_generator_v2_5_visual_review_v1.csv")

    def test_frozen_hashes_match(self) -> None:
        self.assertEqual(audit.file_sha256(ROOT / "config" / "candidate_generator_v2_5_prospective_audit.json"), audit.EXPECTED_CONFIG_SHA256)
        self.assertEqual(audit.file_sha256(self.selection_dir / "selection_private.csv"), audit.EXPECTED_PRIVATE_SHA256)
        self.assertEqual(audit.file_sha256(self.selection_dir / "review_manifest.csv"), audit.EXPECTED_PUBLIC_MANIFEST_SHA256)
        self.assertEqual(audit.file_sha256(ROOT / "research" / "candidate_generator_v2_5_visual_review_v1.csv"), audit.EXPECTED_VISUAL_SHA256)

    def test_real_unblind_counts_and_gate_failure(self) -> None:
        merged = audit.merge_rows(self.private, self.public, self.visual)
        self.assertEqual(len(merged), 256)
        summary = audit.summarize_visual(merged)
        self.assertEqual(summary["verdict_counts"], {
            "independent_lookalike": 219,
            "same_source_candidate": 36,
            "uncertain": 1,
        })
        self.assertEqual(summary["candidate_same_source_count"], 20)
        self.assertEqual(summary["outside_same_source_count"], 16)
        self.assertEqual(summary["outside_uncertain_count"], 0)
        self.assertEqual(summary["outside_same_source_by_group"], {
            "brain_ce_mri": 6,
            "kidney_ct": 6,
            "lymphoma_histology": 4,
        })

        result = audit.evaluate(
            audit.read_json(ROOT / "config" / "candidate_generator_v2_5_prospective_audit.json"),
            audit.read_json(self.selection_dir / "selection_report.json"),
            audit.read_json(ROOT / "research" / "candidate_generator_v2_4_counterfactual_v1.json"),
            audit.read_json(ROOT / "research" / "candidate_generator_v2_4_rank_crowding_v1.json"),
            merged,
        )
        gates = result["acceptance_gates"]
        self.assertFalse(gates["new_outside_same_source"])
        self.assertFalse(gates["all_passed"])
        self.assertTrue(all(value for key, value in gates.items() if key not in {"new_outside_same_source", "all_passed"}))

    def test_private_public_identity_mismatch_is_rejected(self) -> None:
        tampered = [dict(row) for row in self.public]
        tampered[0]["left_path"] = "changed.jpg"
        with self.assertRaisesRegex(ValueError, "private/public left_path"):
            audit.merge_rows(self.private, tampered, self.visual)


if __name__ == "__main__":
    unittest.main()
