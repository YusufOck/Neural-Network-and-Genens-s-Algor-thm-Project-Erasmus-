from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "smallcnn_26leaf_baseline_colab.ipynb"


class SmallCNNBaselineNotebookTests(unittest.TestCase):
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
        self.assertIn("AUTHORIZE_FULL_SMALLCNN = False", self.code)
        self.assertIn("assert AUTHORIZE_FULL_SMALLCNN", self.code)

    def test_interrupted_run_resumes_only_from_checkpoint(self) -> None:
        self.assertIn("last_checkpoint.pth", self.code)
        self.assertIn("command.append('--resume')", self.code)
        self.assertIn("not RESUME_RUN", self.code)
        self.assertIn("log_mode = 'a' if RESUME_RUN else 'w'", self.code)
        self.assertIn("COMPLETED_RUN = all", self.code)
        self.assertNotIn(
            "assert not (OUTPUT_DIR / 'test_metrics.json').exists()", self.code
        )

    def test_frozen_smallcnn_contract_is_explicit(self) -> None:
        required_fragments = (
            "--model', 'small_cnn'",
            "--target-level', 'subclass'",
            "--split-manifest",
            "--epochs', '25'",
            "--early-stopping-patience', '5'",
            "--batch-size', '32'",
            "--image-size', '224'",
            "--normalization', 'imagenet'",
            "--augmentation', 'minimal'",
            "--learning-rate', '0.001'",
            "--weight-decay', '0.0001'",
            "--seed', '42'",
            "--experiment-type', 'baseline'",
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            "1839b0c65a26f0ea7d8719bcd58238b8c73eb3e78632b26e377937e018474490",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.code)
        self.assertNotIn("'--pretrained'", self.code)
        self.assertNotIn("'--fine-tune'", self.code)

    def test_completion_checks_scope_and_builds_review_package(self) -> None:
        self.assertIn("'train': 91000, 'val': 19500, 'test': 19500", self.code)
        self.assertIn("prediction_rows == 19500", self.code)
        self.assertIn("smallcnn_26leaf_baseline_v1_review_package.zip", self.code)
        for artifact in (
            "test_metrics.json",
            "test_predictions.csv",
            "classification_reports.json",
            "leaf_confusion_matrix.png",
            "parent_confusion_matrix.png",
            "learning_curves.png",
            "split_manifest.csv",
            "artifact_hashes.json",
            "smallcnn_26leaf_baseline_v1_console.log",
        ):
            self.assertIn(artifact, self.code)


if __name__ == "__main__":
    unittest.main()
