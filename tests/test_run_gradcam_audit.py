from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_gradcam_audit import (  # noqa: E402
    derive_frozen_selection,
    gradcam_from_tensors,
)


CONFIG_PATH = ROOT / "config" / "gradcam_resnet18_lastblock_v1.json"
MANIFEST_PATH = (
    ROOT
    / "research"
    / "gradcam"
    / "resnet18_lastblock_gradcam_v1"
    / "sample_manifest.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GradCamAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_frozen_manifest_hash_and_counts(self) -> None:
        sample_contract = self.config["sample_manifest"]
        self.assertEqual(sha256(MANIFEST_PATH), sample_contract["sha256"])
        self.assertEqual(len(self.rows), 37)
        self.assertEqual(len({row["relative_path"] for row in self.rows}), 37)
        correct = [
            row
            for row in self.rows
            if row["selection_group"] == "correct_leaf_median_confidence"
        ]
        errors = [row for row in self.rows if row["sample_id"].startswith("ERROR_")]
        self.assertEqual(len(correct), 26)
        self.assertEqual(len({row["true_leaf"] for row in correct}), 26)
        self.assertEqual(len(errors), 11)
        self.assertEqual(
            sum(row["selection_group"] == "cross_parent_error" for row in errors),
            2,
        )
        self.assertEqual(
            sum(row["selection_group"] == "high_confidence_error" for row in errors),
            7,
        )
        self.assertEqual(
            sum(
                row["selection_group"] == "dominant_oral_confusion_median"
                for row in errors
            ),
            2,
        )
        self.assertTrue(all(row["split"] == "test" for row in self.rows))

    def test_config_forbids_training_and_freezes_model_inputs(self) -> None:
        self.assertTrue(self.config["does_not_train_or_select_a_model"])
        self.assertFalse(self.config["reportable_as_model_performance"])
        self.assertEqual(self.config["target_layer"], "layer4.1.conv2")
        self.assertEqual(
            self.config["checkpoint"]["sha256"],
            "538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d",
        )
        self.assertEqual(
            self.config["split_manifest"]["sha256"],
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
        )
        self.assertTrue(
            self.config["selection_contract"]["selected_before_heatmap_generation"]
        )

    def test_gradcam_tensor_math_is_normalized_and_nonnegative(self) -> None:
        activations = torch.tensor(
            [[[[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [1.0, 0.0]]]]
        )
        gradients = torch.tensor(
            [[[[2.0, 2.0], [2.0, 2.0]], [[-1.0, -1.0], [-1.0, -1.0]]]]
        )
        heatmap = gradcam_from_tensors(activations, gradients)
        self.assertEqual(heatmap.shape, (2, 2))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertAlmostEqual(float(heatmap.max()), 1.0)

    def test_gradcam_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            gradcam_from_tensors(torch.ones(2, 3), torch.ones(2, 3))
        with self.assertRaises(ValueError):
            gradcam_from_tensors(torch.ones(1, 2, 3, 3), torch.ones(1, 2, 2, 2))
        with self.assertRaises(ValueError):
            gradcam_from_tensors(torch.ones(2, 2, 3, 3), torch.ones(2, 2, 3, 3))

    def test_selection_algorithm_is_deterministic_and_score_blind_to_heatmaps(self) -> None:
        rows = []
        for leaf in ("leaf_a", "leaf_b"):
            for index, confidence in enumerate((0.7, 0.8, 0.9), start=1):
                rows.append(
                    {
                        "relative_path": f"P/{leaf}/{index}.jpg",
                        "true_leaf": leaf,
                        "predicted_leaf": leaf,
                        "leaf_correct": "True",
                        "parent_correct": "True",
                        "confidence": str(confidence),
                    }
                )
        error_specs = (
            ("err_cross.jpg", "leaf_a", "leaf_b", "False", 0.70),
            ("err_high.jpg", "leaf_a", "leaf_b", "True", 0.995),
            ("oral_a.jpg", "oral_scc", "oral_normal", "True", 0.60),
            ("oral_b.jpg", "oral_scc", "oral_normal", "True", 0.80),
            ("oral_c.jpg", "oral_normal", "oral_scc", "True", 0.65),
            ("oral_d.jpg", "oral_normal", "oral_scc", "True", 0.85),
        )
        for path, true_leaf, predicted_leaf, parent_correct, confidence in error_specs:
            rows.append(
                {
                    "relative_path": f"P/errors/{path}",
                    "true_leaf": true_leaf,
                    "predicted_leaf": predicted_leaf,
                    "leaf_correct": "False",
                    "parent_correct": parent_correct,
                    "confidence": str(confidence),
                }
            )
        first = derive_frozen_selection(rows)
        second = derive_frozen_selection(list(reversed(rows)))
        self.assertEqual(first, second)
        correct_paths = {
            row["relative_path"]
            for row in first
            if row["selection_group"] == "correct_leaf_median_confidence"
        }
        self.assertEqual(correct_paths, {"P/leaf_a/2.jpg", "P/leaf_b/2.jpg"})
        self.assertNotIn("heatmap", json.dumps(first).lower())


if __name__ == "__main__":
    unittest.main()
