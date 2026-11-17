from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "resnet18_lastblock_finetune_colab.ipynb"


class ResNet18LastBlockFineTuneNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.payload["cells"]
            if cell["cell_type"] == "code"
        )

    def test_code_cells_compile_and_notebook_is_clean(self) -> None:
        for index, cell in enumerate(self.payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_full_training_starts_locked_and_resumes_safely(self) -> None:
        self.assertIn("AUTHORIZE_FULL_FINETUNE = False", self.code)
        self.assertIn("assert AUTHORIZE_FULL_FINETUNE", self.code)
        self.assertIn("last_checkpoint.pth", self.code)
        self.assertIn("command.append('--resume')", self.code)
        self.assertIn("log_mode = 'a' if RESUME_RUN else 'w'", self.code)

    def test_frozen_ablation_contract_is_explicit(self) -> None:
        required_fragments = (
            "--model', 'resnet18'",
            "--target-level', 'subclass'",
            "--split-manifest",
            "--pretrained', '--fine-tune",
            "--init-checkpoint",
            "--init-checkpoint-sha256",
            "--epochs', '12'",
            "--early-stopping-patience', '4'",
            "--learning-rate', '0.0001'",
            "--batch-size', '32'",
            "--seed', '42'",
            "--experiment-type', 'ablation'",
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            "20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177",
            "65de201f74d9296b81280fbd7b4653542a05e6446de6eb987171eb3fa3a72650",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.code)

    def test_trainable_scope_and_completion_are_verified(self) -> None:
        self.assertIn("name.startswith(('layer4.', 'fc.'))", self.code)
        self.assertIn("8407066", self.code)
        self.assertIn("'train': 91000, 'val': 19500, 'test': 19500", self.code)
        self.assertIn("prediction_rows == 19500", self.code)
        self.assertIn("resnet18_lastblock_finetune_v1_review_package.zip", self.code)


if __name__ == "__main__":
    unittest.main()
