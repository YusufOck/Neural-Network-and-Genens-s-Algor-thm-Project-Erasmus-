from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_image import (  # noqa: E402
    load_runtime,
    predict_one,
    render_prediction_card,
    verify_manifest,
)
from train import build_model, build_transforms  # noqa: E402
from training_support import sha256_file  # noqa: E402


class PredictImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.class_names = ["class_a", "class_b", "class_c"]
        model = build_model(
            "small_cnn", len(self.class_names), pretrained=False, fine_tune=False
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier[-1].bias[1] = 10.0
        self.checkpoint_path = self.root / "best_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": self.class_names,
                "args": {
                    "model": "small_cnn",
                    "target_level": "subclass",
                    "image_size": 32,
                    "normalization": "imagenet",
                    "fine_tune": False,
                },
            },
            self.checkpoint_path,
        )
        labels = {
            name: {
                "display_name": name.replace("_", " ").title(),
                "parent_group": f"parent_{name[-1]}",
                "safe_output": f"Synthetic test label {name}",
            }
            for name in self.class_names
        }
        self.taxonomy_path = self.root / "taxonomy.json"
        self.taxonomy_path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
        self.config_path = self.root / "config.json"
        self.config = {
            "schema_version": 1,
            "model": "small_cnn",
            "target_level": "subclass",
            "class_count": 3,
            "class_names": self.class_names,
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "image_size": 32,
            "normalization": "imagenet",
            "taxonomy": str(self.taxonomy_path),
            "verification_rows": 2,
            "confidence_absolute_tolerance": 1e-6,
            "warning": "Synthetic test only",
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.dataset_root = self.root / "dataset"
        self.image_paths = []
        for index in range(2):
            path = self.dataset_root / "Parent" / "class_b" / f"image_{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 50), (30 + index, 80, 120)).save(path)
            self.image_paths.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load(self):
        return load_runtime(self.checkpoint_path, self.config_path, "cpu")

    def test_single_prediction_and_card(self) -> None:
        model, transform, classes, taxonomy, metadata = self.load()
        result = predict_one(
            self.image_paths[0],
            model,
            transform,
            classes,
            taxonomy,
            metadata,
            top_k=3,
            dataset_root=self.dataset_root,
        )
        self.assertEqual(result["predicted_leaf"], "class_b")
        self.assertEqual(result["predicted_parent_group"], "parent_b")
        self.assertEqual(result["true_leaf_from_dataset_path"], "class_b")
        self.assertTrue(result["leaf_correct_when_known"])
        self.assertEqual(len(result["top_k"]), 3)
        card_path = self.root / "prediction_card.png"
        render_prediction_card(self.image_paths[0], result, card_path)
        self.assertTrue(card_path.is_file())
        self.assertGreater(card_path.stat().st_size, 1000)

    def test_frozen_manifest_replay(self) -> None:
        expected_confidence = math.exp(10.0) / (math.exp(10.0) + 2.0)
        manifest_path = self.root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "relative_path",
                    "true_leaf",
                    "predicted_leaf",
                    "expected_confidence",
                ),
            )
            writer.writeheader()
            for index, path in enumerate(self.image_paths):
                writer.writerow(
                    {
                        "sample_id": f"sample_{index}",
                        "relative_path": path.relative_to(self.dataset_root).as_posix(),
                        "true_leaf": "class_b",
                        "predicted_leaf": "class_b",
                        "expected_confidence": expected_confidence,
                    }
                )
        model, transform, classes, taxonomy, metadata = self.load()
        report, rows = verify_manifest(
            manifest_path,
            self.dataset_root,
            model,
            transform,
            classes,
            taxonomy,
            metadata,
            self.config,
        )
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["passed_rows"], 2)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        changed = dict(self.config)
        changed["checkpoint_sha256"] = "0" * 64
        self.config_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256 mismatch"):
            self.load()

    def test_checkpoint_class_order_mismatch_is_rejected(self) -> None:
        changed = dict(self.config)
        changed["class_names"] = list(reversed(self.class_names))
        self.config_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class order"):
            self.load()


if __name__ == "__main__":
    unittest.main()
