from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "resnet18_26leaf_baseline_colab.ipynb"


class ResNet18BaselineNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.payload["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_code_cells_compile_and_notebook_is_clean(self) -> None:
        for index, cell in enumerate(self.payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_full_training_starts_locked(self) -> None:
        self.assertIn("AUTHORIZE_FULL_BASELINE = False", self.code)
        self.assertIn("assert AUTHORIZE_FULL_BASELINE", self.code)

    def test_interrupted_run_resumes_only_from_checkpoint(self) -> None:
        self.assertIn("last_checkpoint.pth", self.code)
        self.assertIn("command.append('--resume')", self.code)
        self.assertIn("not RESUME_RUN", self.code)

    def test_frozen_baseline_contract_is_explicit(self) -> None:
        required_fragments = (
            "--model', 'resnet18'",
            "--target-level', 'subclass'",
            "--split-manifest",
            "--pretrained",
            "--epochs', '15'",
            "--early-stopping-patience', '4'",
            "--batch-size', '32'",
            "--image-size', '224'",
            "--seed', '42'",
            "--experiment-type', 'baseline'",
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            "1839b0c65a26f0ea7d8719bcd58238b8c73eb3e78632b26e377937e018474490",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.code)

    def test_completion_checks_full_test_scope_and_artifacts(self) -> None:
        self.assertIn("'train': 91000, 'val': 19500, 'test': 19500", self.code)
        self.assertIn("prediction_rows == 19500", self.code)
        for artifact in (
            "test_metrics.json",
            "test_predictions.csv",
            "classification_reports.json",
            "leaf_confusion_matrix.png",
            "parent_confusion_matrix.png",
            "learning_curves.png",
            "split_manifest.csv",
        ):
            self.assertIn(artifact, self.code)


if __name__ == "__main__":
    unittest.main()
