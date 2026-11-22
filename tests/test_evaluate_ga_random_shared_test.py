from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_ga_random_shared_test import (  # noqa: E402
    exact_mcnemar_pvalue,
    paired_comparison,
    taxonomy_contract,
    validate_config,
)


class EvaluateGaRandomSharedTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "config/ga_random_shared_test_v1.json").read_text(encoding="utf-8")
        )

    def test_config_freezes_evaluation_only_shared_test(self) -> None:
        validate_config(self.config)
        self.assertTrue(self.config["model_contract"]["evaluation_only"])
        self.assertFalse(self.config["model_contract"]["training_allowed"])
        self.assertEqual(self.config["frozen_inputs"]["counts"]["test"], 19_500)
        self.assertFalse(self.config["test_policy"]["test_based_reselection_allowed"])
        self.assertTrue(self.config["test_policy"]["shared_benchmark_already_used_by_prior_models"])

    def test_taxonomy_contract_is_26_leaf_and_8_parent(self) -> None:
        class_names, mapping = taxonomy_contract(
            ROOT / "config/label_taxonomy.json",
            self.config["frozen_inputs"]["label_taxonomy_sha256"],
        )
        self.assertEqual(len(class_names), 26)
        self.assertEqual(len(set(mapping.values())), 8)

    def test_exact_mcnemar_is_symmetric_and_bounded(self) -> None:
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_pvalue(2, 8), exact_mcnemar_pvalue(8, 2))
        self.assertGreaterEqual(exact_mcnemar_pvalue(2, 8), 0.0)
        self.assertLessEqual(exact_mcnemar_pvalue(2, 8), 1.0)

    def test_paired_comparison_uses_identical_paths(self) -> None:
        ga = [
            {"relative_path": "a.jpg", "true_leaf": "a", "leaf_correct": "True", "parent_correct": "True"},
            {"relative_path": "b.jpg", "true_leaf": "b", "leaf_correct": "True", "parent_correct": "True"},
            {"relative_path": "c.jpg", "true_leaf": "c", "leaf_correct": "False", "parent_correct": "True"},
        ]
        random = [
            {"relative_path": "a.jpg", "true_leaf": "a", "leaf_correct": "True", "parent_correct": "True"},
            {"relative_path": "b.jpg", "true_leaf": "b", "leaf_correct": "False", "parent_correct": "False"},
            {"relative_path": "c.jpg", "true_leaf": "c", "leaf_correct": "True", "parent_correct": "True"},
        ]
        result = paired_comparison(ga, random)
        self.assertEqual(result["leaf"]["ga_only_correct"], 1)
        self.assertEqual(result["leaf"]["random_only_correct"], 1)
        self.assertEqual(result["parent"]["ga_only_correct"], 1)
        random.pop()
        with self.assertRaisesRegex(ValueError, "path"):
            paired_comparison(ga, random)


if __name__ == "__main__":
    unittest.main()
