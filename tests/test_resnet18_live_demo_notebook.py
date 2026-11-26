from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "resnet18_live_demo_colab.ipynb"


class ResNet18LiveDemoNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    def test_notebook_is_clean_and_code_compiles(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")

    def test_inference_only_contract(self) -> None:
        self.assertIn("src/predict_image.py", self.code)
        self.assertNotIn("src/train.py", self.code)
        self.assertNotIn("--epochs", self.code)
        self.assertNotIn("--learning-rate", self.code)
        self.assertNotIn("--resume", self.code)
        self.assertIn("never trains", "".join(self.notebook["cells"][0]["source"]).lower())

    def test_frozen_replay_and_single_image_are_present(self) -> None:
        self.assertIn("AUTHORIZE_FROZEN_REPLAY = False", self.code)
        self.assertIn("--verify-manifest", self.code)
        self.assertIn("report['verification_rows'] == 37", self.code)
        self.assertIn("report['passed_rows'] == 37", self.code)
        self.assertIn("--image", self.code)
        self.assertIn("prediction_card.png", self.code)

    def test_frozen_hashes_match_current_files(self) -> None:
        locked = {
            ROOT / "src/predict_image.py": "79b18d906e0580f0931441a597cec6b1e9f9f5b51e56158d4a3a2d0ce945a0b2",
            ROOT / "config/live_demo_resnet18_lastblock_v1.json": "546d6d2720ececb4f9dad4be0a66800ab0f71bc7d19f176a3f160fe5862b2424",
            ROOT / "research/gradcam/resnet18_lastblock_gradcam_v1/sample_manifest.csv": "3f797979b4d75b092dfcb4e2348ea1705ed34d35da884ac999c68abef8c87add",
        }
        for path, expected in locked.items():
            self.assertIn(expected, self.code)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_evidence_package_is_persistent(self) -> None:
        self.assertIn("/content/drive/MyDrive/NNGA_OUTPUTS/resnet18_live_demo_v1", self.code)
        self.assertIn("verification_report.json", self.code)
        self.assertIn("verification_rows.csv", self.code)
        self.assertIn("evidence_hashes.json", self.code)
        self.assertIn("resnet18_live_demo_v1_review_package.zip", self.code)


if __name__ == "__main__":
    unittest.main()
