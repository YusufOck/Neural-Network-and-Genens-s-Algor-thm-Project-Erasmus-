from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_model_errors import analyze, compute_calibration  # noqa: E402


class AnalyzeModelErrorsTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "data"
        rows = [
            {
                "relative_path": "Oral Cancer/oral_normal/a.jpg",
                "true_leaf": "oral_normal",
                "predicted_leaf": "oral_normal",
                "true_parent": "Oral Cancer",
                "predicted_parent": "Oral Cancer",
                "leaf_correct": "True",
                "parent_correct": "True",
                "confidence": "0.8",
                "probabilities_json": json.dumps({"oral_normal": 0.8, "oral_scc": 0.2}),
            },
            {
                "relative_path": "Oral Cancer/oral_normal/b.jpg",
                "true_leaf": "oral_normal",
                "predicted_leaf": "oral_scc",
                "true_parent": "Oral Cancer",
                "predicted_parent": "Oral Cancer",
                "leaf_correct": "False",
                "parent_correct": "True",
                "confidence": "0.9",
                "probabilities_json": json.dumps({"oral_normal": 0.1, "oral_scc": 0.9}),
            },
            {
                "relative_path": "Oral Cancer/oral_scc/c.jpg",
                "true_leaf": "oral_scc",
                "predicted_leaf": "oral_scc",
                "true_parent": "Oral Cancer",
                "predicted_parent": "Oral Cancer",
                "leaf_correct": "True",
                "parent_correct": "True",
                "confidence": "0.7",
                "probabilities_json": json.dumps({"oral_normal": 0.3, "oral_scc": 0.7}),
            },
            {
                "relative_path": "Oral Cancer/oral_scc/d.jpg",
                "true_leaf": "oral_scc",
                "predicted_leaf": "oral_normal",
                "true_parent": "Oral Cancer",
                "predicted_parent": "Other Parent",
                "leaf_correct": "False",
                "parent_correct": "False",
                "confidence": "0.6",
                "probabilities_json": json.dumps({"oral_normal": 0.6, "oral_scc": 0.4}),
            },
        ]
        for index, row in enumerate(rows):
            image_path = data_root / row["relative_path"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (30 * index, 80, 140)).save(image_path)
        metrics = {
            "best_epoch": 2,
            "class_names": ["oral_normal", "oral_scc"],
            "split_sizes": {"train": 4, "val": 4, "test": 4},
            "leaf_metrics": {"accuracy": 0.5},
            "parent_metrics": {"accuracy": 0.75},
        }
        package = root / "review.zip"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("test_metrics.json", json.dumps(metrics))
            archive.writestr("test_predictions.csv", buffer.getvalue())
        return package, data_root

    def test_full_error_audit_creates_traceable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, data_root = self.build_fixture(root)
            expected_hash = hashlib.sha256(package.read_bytes()).hexdigest()
            output = root / "output"
            summary = analyze(
                package,
                data_root,
                output,
                expected_hash,
                bin_count=2,
                images_per_sheet=2,
            )
            self.assertEqual(summary["errors"]["leaf_error_count"], 2)
            self.assertEqual(summary["errors"]["parent_error_count"], 1)
            self.assertEqual(summary["errors"]["contact_sheet_count"], 1)
            self.assertTrue((output / "contact_sheets/error_sheet_001.png").is_file())
            self.assertTrue((output / "reliability_diagram.png").is_file())
            self.assertTrue((output / "hash_manifest.json").is_file())
            for relative in (
                "error_rows.csv",
                "contact_sheet_index.csv",
                "reliability_bins.csv",
                "summary.json",
                "hash_manifest.json",
            ):
                self.assertNotIn(b"\r\n", (output / relative).read_bytes())
            with (output / "error_rows.csv").open(encoding="utf-8", newline="") as handle:
                errors = list(csv.DictReader(handle))
            self.assertEqual(errors[0]["priority_reason"], "cross_parent_error")
            self.assertEqual(errors[1]["priority_reason"], "confidence_ge_0_90")

    def test_package_hash_mismatch_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, data_root = self.build_fixture(root)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                analyze(package, data_root, output, "0" * 64)
            self.assertFalse(output.exists())

    def test_calibration_metrics_are_reproducible(self) -> None:
        rows = [
            {
                "true_leaf": "a",
                "predicted_leaf": "a",
                "leaf_correct": "True",
                "confidence": "0.8",
                "probabilities_json": json.dumps({"a": 0.8, "b": 0.2}),
            },
            {
                "true_leaf": "a",
                "predicted_leaf": "b",
                "leaf_correct": "False",
                "confidence": "0.9",
                "probabilities_json": json.dumps({"a": 0.1, "b": 0.9}),
            },
        ]
        summary, bins = compute_calibration(rows, ["a", "b"], 2)
        self.assertAlmostEqual(summary["top2_accuracy"], 1.0)
        self.assertAlmostEqual(summary["expected_calibration_error"], 0.35)
        self.assertEqual(summary["error_confidence_at_least_0_90"], 1)
        self.assertEqual(sum(row["count"] for row in bins), 2)


if __name__ == "__main__":
    unittest.main()
