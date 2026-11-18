from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_index import ImageRecord  # noqa: E402
from training_support import (  # noqa: E402
    build_subclass_parent_map,
    derive_parent_predictions,
    prediction_rows,
    set_training_mode,
    sha256_file,
    write_prediction_csv,
)
from train import (  # noqa: E402
    build_model,
    resolve_dropout,
    run_epoch,
    validate_initial_checkpoint,
)


class TrainingSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("C:/dataset")
        self.records = [
            ImageRecord(self.root / "Brain" / "glioma" / "a.jpg", "Brain", "glioma"),
            ImageRecord(self.root / "Brain" / "meningioma" / "b.jpg", "Brain", "meningioma"),
            ImageRecord(self.root / "Kidney" / "normal" / "c.jpg", "Kidney", "normal"),
        ]

    def test_leaf_parent_projection_counts_parent_correct_when_leaf_is_wrong(self) -> None:
        mapping = build_subclass_parent_map(self.records)
        result = derive_parent_predictions(
            targets=[0, 2],
            predictions=[1, 2],
            leaf_class_names=["glioma", "meningioma", "normal"],
            leaf_to_parent=mapping,
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["targets"], result["predictions"])

    def test_ambiguous_leaf_parent_mapping_is_rejected(self) -> None:
        records = self.records + [
            ImageRecord(self.root / "Other" / "glioma" / "d.jpg", "Other", "glioma")
        ]
        with self.assertRaisesRegex(ValueError, "birden fazla"):
            build_subclass_parent_map(records)

    def test_frozen_batch_norm_stays_eval_while_trainable_batch_norm_trains(self) -> None:
        model = nn.Sequential(nn.BatchNorm2d(3), nn.BatchNorm2d(3))
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        set_training_mode(model, True)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)

    def test_prediction_rows_include_both_label_levels_and_probabilities(self) -> None:
        rows = prediction_rows(
            records=self.records,
            record_indices=[0, 2],
            targets=[0, 2],
            predictions=[1, 2],
            probabilities=[[0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
            leaf_class_names=["glioma", "meningioma", "normal"],
            leaf_to_parent=build_subclass_parent_map(self.records),
            dataset_root=self.root,
        )
        self.assertFalse(rows[0]["leaf_correct"])
        self.assertTrue(rows[0]["parent_correct"])
        self.assertEqual(rows[0]["confidence"], 0.8)
        self.assertIn('"meningioma": 0.8', rows[0]["probabilities_json"])

    def test_prediction_csv_and_sha256_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            rows = [{"relative_path": "a.jpg", "confidence": 0.75}]
            write_prediction_csv(rows, path)
            first_hash = sha256_file(path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["relative_path"], "a.jpg")
            self.assertEqual(first_hash, sha256_file(path))

    def test_run_epoch_keeps_dataset_indices_and_probabilities(self) -> None:
        inputs = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        targets = torch.tensor([0, 1])
        indices = torch.tensor([7, 4])
        loader = DataLoader(TensorDataset(inputs, targets, indices), batch_size=2)
        model = nn.Identity()
        result = run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            collect_outputs=True,
        )
        self.assertEqual(result["predictions"], [0, 1])
        self.assertEqual(result["record_indices"], [7, 4])
        self.assertEqual(len(result["probabilities"]), 2)
        self.assertAlmostEqual(sum(result["probabilities"][0]), 1.0, places=6)

    def test_initial_checkpoint_is_hash_and_contract_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best_model.pth"
            source_model = nn.Linear(3, 2)
            torch.save(
                {
                    "model_state_dict": source_model.state_dict(),
                    "class_names": ["a", "b"],
                    "args": {"model": "linear", "target_level": "subclass"},
                },
                path,
            )
            expected_hash = sha256_file(path)
            checkpoint, actual_hash = validate_initial_checkpoint(
                path,
                expected_hash,
                "linear",
                "subclass",
                ["a", "b"],
            )
            self.assertEqual(actual_hash, expected_hash)
            self.assertIn("model_state_dict", checkpoint)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_initial_checkpoint(
                    path, "0" * 64, "linear", "subclass", ["a", "b"]
                )
            with self.assertRaisesRegex(ValueError, "sinif sirasi"):
                validate_initial_checkpoint(
                    path, expected_hash, "linear", "subclass", ["b", "a"]
                )

    def test_resnet18_last_block_fine_tuning_only_unfreezes_layer4_and_head(self) -> None:
        original_resnet18 = models.resnet18

        def local_resnet18(*, weights):
            return original_resnet18(weights=None)

        with patch("train.models.resnet18", side_effect=local_resnet18):
            model = build_model("resnet18", 26, pretrained=True, fine_tune=True)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("layer4.") for name in trainable_names))
        self.assertTrue(any(name.startswith("fc.") for name in trainable_names))
        self.assertFalse(any(name.startswith("layer3.") for name in trainable_names))
        self.assertFalse(any(name.startswith("conv1.") for name in trainable_names))
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            8_407_066,
        )

    def test_dropout_is_configurable_without_changing_default_contract(self) -> None:
        original_resnet18 = models.resnet18

        def local_resnet18(*, weights):
            return original_resnet18(weights=None)

        with patch("train.models.resnet18", side_effect=local_resnet18):
            default_model = build_model("resnet18", 26, pretrained=True, fine_tune=True)
            searched_model = build_model(
                "resnet18", 26, pretrained=True, fine_tune=True, dropout=0.17
            )
        self.assertAlmostEqual(default_model.fc[0].p, 0.30)
        self.assertAlmostEqual(searched_model.fc[0].p, 0.17)
        self.assertAlmostEqual(resolve_dropout("small_cnn", None), 0.35)
        with self.assertRaisesRegex(ValueError, "Dropout"):
            resolve_dropout("resnet18", 1.0)


if __name__ == "__main__":
    unittest.main()
