from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ga_resnet18_lastblock_search_colab.ipynb"


class GaResnet18LastBlockSearchNotebookTests(unittest.TestCase):
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

    def test_search_starts_locked_and_is_resumable(self) -> None:
        self.assertIn("AUTHORIZE_GA_RANDOM_SEARCH = False", self.code)
        self.assertIn("assert AUTHORIZE_GA_RANDOM_SEARCH", self.code)
        self.assertIn("FROZEN_GA_RANDOM_SEARCH_V1", self.code)
        self.assertIn("Output klasörünü silme", self.code)
        self.assertIn("--method', 'both'", self.code)

    def test_frozen_equal_budget_and_no_test_contract(self) -> None:
        for fragment in (
            "'train': 9100, 'val': 3900, 'test': 0",
            "'ga_trials': 12, 'random_trials': 12",
            "aday başına 4 epoch",
            "test_evaluated",
            "test_metrics.json",
            "fd6a55def92aa5f3582679ef0be35bde6fcb722be00fb508db9a86a7030aff03",
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
            "20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177",
        ):
            self.assertIn(fragment, self.code)

    def test_code_and_config_hashes_are_locked(self) -> None:
        locked = {
            ROOT / "src/train.py": "807a23e40ea3aaa5c0c98737beddb637d71c5e5a9e5f230aa0b75d0be7212a77",
            ROOT / "src/run_ga_random_search.py": "657190d6b61ec1a1ce7a7aba6b5345e4fbcee376431a1ea9825d3b1dc4c1df7d",
            ROOT / "config/ga_resnet18_lastblock_v1.json": "fefe845a8fb84c0f252954f42cee3a8f4fa84210a4313d24af339fdeb90c7638",
            ROOT / "research/splits/ga_search_proxy_v1.csv": "fd6a55def92aa5f3582679ef0be35bde6fcb722be00fb508db9a86a7030aff03",
        }
        for path, expected in locked.items():
            self.assertIn(expected, self.code)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
