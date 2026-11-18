from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "resnet18_lastblock_gradcam_colab.ipynb"


class ResNet18LastBlockGradCamNotebookTests(unittest.TestCase):
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

    def test_run_is_locked_and_never_calls_training(self) -> None:
        self.assertIn("AUTHORIZE_GRADCAM = False", self.code)
        self.assertIn("assert AUTHORIZE_GRADCAM", self.code)
        self.assertIn("run_gradcam_audit.py", self.code)
        self.assertNotIn("train.py", self.code)
        self.assertNotIn("--epochs", self.code)
        self.assertNotIn("--learning-rate", self.code)

    def test_all_frozen_evidence_hashes_are_present(self) -> None:
        required = (
            "4a72f5e5dc28b0ef25e4195060be69d0fccdecdd44bf6e8ddf680618de89f538",
            "07f7a88c21a6e65ed6d96084d86f886eb47f2f4bc1be5096c32259cb5ca50df2",
            "3f797979b4d75b092dfcb4e2348ea1705ed34d35da884ac999c68abef8c87add",
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            "538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d",
            "237f3283b88beeaf0a93f182adab23ad478810b0944245e3023e6cfe885c57bd",
        )
        for value in required:
            self.assertIn(value, self.code)

    def test_completion_contract_and_review_package_are_checked(self) -> None:
        self.assertIn("summary['sample_count'] == 37", self.code)
        self.assertIn("summary['correct_control_count'] == 26", self.code)
        self.assertIn("summary['error_count'] == 11", self.code)
        self.assertIn("summary['prediction_reproduction_count'] == 37", self.code)
        self.assertIn("summary['contact_sheet_count'] == 7", self.code)
        self.assertIn("resnet18_lastblock_gradcam_v1_review_package.zip", self.code)


if __name__ == "__main__":
    unittest.main()
