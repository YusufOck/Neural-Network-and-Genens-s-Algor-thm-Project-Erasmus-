from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verify_candidate_generator_v2_3_freeze import (  # noqa: E402
    EXPECTED_CONFIG_SHA256,
    capacity_summary,
    load_frozen_config,
    verify_dependency_hashes,
    verify_natural_evidence,
)


class CandidateGeneratorV23FreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "config" / "candidate_generator_v2_3.json"
        self.hash_path = PROJECT_ROOT / "config" / "candidate_generator_v2_3.sha256"
        self.config = load_frozen_config(self.config_path, self.hash_path)

    def test_actual_config_and_dependency_hashes_are_frozen(self) -> None:
        self.assertEqual(self.hash_path.read_text(encoding="utf-8").strip(), EXPECTED_CONFIG_SHA256)
        verified = verify_dependency_hashes(PROJECT_ROOT.resolve(), self.config)
        self.assertEqual(len(verified), 7)

    def test_changed_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changed = dict(self.config)
            changed_rule = dict(changed["candidate_rule"])
            changed_robust = dict(changed_rule["robust_retrieval"])
            changed_robust["top_k"] = 20
            changed_rule["robust_retrieval"] = changed_robust
            changed["candidate_rule"] = changed_rule
            config_path = Path(tmp) / "changed.json"
            hash_path = Path(tmp) / "changed.sha256"
            config_path.write_text(json.dumps(changed), encoding="utf-8")
            hash_path.write_text(EXPECTED_CONFIG_SHA256 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen_config(config_path, hash_path)

    def test_natural_evidence_matches_23_inside_zero_outside(self) -> None:
        result = verify_natural_evidence(
            PROJECT_ROOT
            / "research"
            / "natural_duplicate_adjudication_v1"
            / "final_pair_references_v1.csv",
            self.config,
        )
        self.assertEqual(result["same_source_inside_candidate"], 23)
        self.assertEqual(result["same_source_outside_candidate"], 0)

    def test_capacity_summary_is_consistent(self) -> None:
        capacity = capacity_summary(self.config)
        self.assertEqual(capacity["estimated_candidate_pairs"], 2925170)
        self.assertEqual(capacity["maximum_candidate_pairs_before_pause"], 3250000)
        self.assertGreater(capacity["serial_verification_days"], 27)
        self.assertLess(capacity["ideal_eight_worker_days_without_overhead"], 4)


if __name__ == "__main__":
    unittest.main()
