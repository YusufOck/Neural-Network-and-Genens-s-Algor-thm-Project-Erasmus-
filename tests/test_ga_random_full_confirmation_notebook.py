from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/ga_random_full_confirmation_colab.ipynb"


class GaRandomFullConfirmationNotebookTests(unittest.TestCase):
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

    def test_confirmation_starts_locked_and_is_resumable(self) -> None:
        self.assertIn("AUTHORIZE_FULL_CONFIRMATION = False", self.code)
        self.assertIn("assert AUTHORIZE_FULL_CONFIRMATION", self.code)
        self.assertIn("FROZEN_GA_RANDOM_FULL_CONFIRMATION_V1", self.code)
        self.assertIn("Output klasörünü silme", self.code)
        self.assertIn("last_checkpoint.pth", (ROOT / "src/run_ga_random_full_confirmation.py").read_text(encoding="utf-8"))

    def test_full_validation_only_contract_is_visible(self) -> None:
        for fragment in (
            "91.000 train / 19.500 validation / 0 test",
            "'train': 91000, 'val': 19500, 'test': 0",
            "test_loader_constructed",
            "test_metrics.json",
            "GA_G01_C02",
            "RANDOM_002",
            "afe9d336d53ae24490e065c37b5cc764d0cfeff8f01b3fb7ee8af756bcd375f9",
            "debb373153f0c3264e599e6c91a00c1689a86d2a21730cee2bba6d70f4a36f9c",
        ):
            self.assertIn(fragment, self.code)

    def test_code_and_config_hashes_are_locked(self) -> None:
        locked = {
            ROOT / "src/train.py": "807a23e40ea3aaa5c0c98737beddb637d71c5e5a9e5f230aa0b75d0be7212a77",
            ROOT / "src/run_ga_random_full_confirmation.py": "69d045c449245b1e408b33017737691c153c25586461d3cf5c8691c2a449fc49",
            ROOT / "config/ga_resnet18_full_confirmation_v1.json": "7f0886527f24365373842081561a0a4b6318f16fb59d0272fff1266a9bbd34a9",
            ROOT / "research/splits/ga_full_confirmation_v1.csv": "afe9d336d53ae24490e065c37b5cc764d0cfeff8f01b3fb7ee8af756bcd375f9",
            ROOT / "research/splits/ga_full_confirmation_v1_report.json": "eba3d8b626cc8b2ba2f513f4ec58061a28470f9d99b5a095d62461ca5e3d4e6d",
        }
        for path, expected in locked.items():
            self.assertIn(expected, self.code)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
