from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/ga_random_shared_test_colab.ipynb"


class GaRandomSharedTestNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.payload["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    def test_notebook_is_clean_and_cells_compile(self) -> None:
        for index, cell in enumerate(self.payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_shared_test_starts_locked_and_has_separate_freeze(self) -> None:
        self.assertIn("--mode', 'freeze", self.code)
        self.assertIn("AUTHORIZE_SHARED_TEST = False", self.code)
        self.assertIn("FROZEN_GA_RANDOM_SHARED_TEST_V1", self.code)
        self.assertIn("test_opened_during_freeze", self.code)
        self.assertIn("Output klasorunu silme", self.code)

    def test_frozen_hashes_match_current_files(self) -> None:
        locked = {
            ROOT / "src/evaluate_ga_random_shared_test.py": "d11e0f3cac7b0a056ce87a163f83e8aa043206408352623e83170dd854bb7419",
            ROOT / "config/ga_random_shared_test_v1.json": "9ac37b7c8d400eb363fecfc4801d4df835ab4f629e3a371a0a5e67998dec934e",
            ROOT / "research/splits/practical_leakage_reduced_split_v1.csv": "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            ROOT / "config/label_taxonomy.json": "63820de352eb442cacf7f9701ac6006ff3c16a26642eae5042015941d8d8dd1b",
        }
        for path, expected in locked.items():
            self.assertIn(expected, self.code)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_long_cell_is_evaluation_only(self) -> None:
        self.assertIn("--mode', 'evaluate", self.code)
        self.assertNotIn("--epochs", self.code)
        self.assertNotIn("--learning-rate", self.code)
        self.assertNotIn("--resume", self.code)
        self.assertIn("test_based_reselection_performed", self.code)


if __name__ == "__main__":
    unittest.main()
