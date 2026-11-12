from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import score_v2_6_same2d_development as audit


class V26Same2DDevelopmentTests(unittest.TestCase):
    def test_frozen_family_target_hash_and_state(self) -> None:
        target, target_hash = audit.load_frozen_target(
            ROOT / "config" / "leakage_family_target_v1.json"
        )
        self.assertEqual(
            target_hash,
            "25c46dec6a912896ad22a6832dbc1ffee40e37650766d970453b4a110840d150",
        )
        self.assertEqual(target["target_id"], "same_2d_augmentation_family_v1")
        self.assertFalse(target["metadata_limitation"]["patient_scan_series_ids_available"])

    def test_actual_selection_is_exactly_the_16_frozen_outside_positives(self) -> None:
        rows = audit.read_csv(
            ROOT
            / "research"
            / "candidate_generator_v2_5_prospective_audit_v1"
            / "unblinded_pair_results.csv"
        )
        selected = audit.select_development_rows(rows)
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            Counter(row["source_group"] for row in selected),
            Counter({"brain_ce_mri": 6, "kidney_ct": 6, "lymphoma_histology": 4}),
        )
        self.assertTrue(all(row["audit_region"] == "outside" for row in selected))
        self.assertTrue(all(row["visual_verdict"] == "same_source_candidate" for row in selected))

    def test_score_input_preserves_traceability_without_claiming_truth(self) -> None:
        source = {
            "pair_id": "P1",
            "visual_verdict": "same_source_candidate",
            "left_leaf": "brain_glioma",
            "right_leaf": "brain_glioma",
            "source_group": "brain_ce_mri",
            "phash_distance": "18",
        }
        result = audit.make_score_input(source, "mri", False)
        self.assertEqual(result["audit_partition"], "development")
        self.assertEqual(result["truth_status"], "development_not_new_holdout")
        self.assertEqual(result["reference_label"], "")
        self.assertEqual(result["human_verdict"], "same_source_candidate")
        self.assertEqual(result["sha256_equal"], "no")


if __name__ == "__main__":
    unittest.main()
