from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from render_natural_duplicate_audit_v1 import (  # noqa: E402
    read_review_manifest,
    render_page,
    validate_review_rows,
)


def review_row(pair_id: str = "N000001") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "source_group": "brain_ce_mri",
        "left_leaf": "brain_glioma",
        "right_leaf": "brain_glioma",
        "left_path": "left.png",
        "right_path": "right.png",
        "visual_verdict": "",
        "confidence": "",
        "reviewer": "",
        "review_date": "",
        "review_notes": "",
    }


class NaturalDuplicateAuditRenderTests(unittest.TestCase):
    def test_validate_rejects_prelabelled_row(self) -> None:
        row = review_row()
        row["visual_verdict"] = "independent_lookalike"
        with self.assertRaises(ValueError):
            validate_review_rows([row], 1)

    def test_validate_rejects_duplicate_pair_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_review_rows([review_row(), review_row()], 2)

    def test_manifest_rejects_hidden_score_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            fields = list(review_row()) + ["phash_distance"]
            path.write_text(
                ",".join(fields) + "\n" + ",".join(["N000001"] + [""] * (len(fields) - 1)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_review_manifest(path)

    def test_render_page_has_expected_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (40, 30), "red").save(root / "left.png")
            Image.new("RGB", (30, 40), "blue").save(root / "right.png")
            page = render_page([review_row()], root, 1, 1, 128)
            self.assertEqual(page.width, 20 * 2 + 128 * 2 + 12)
            self.assertGreater(page.height, 128)


if __name__ == "__main__":
    unittest.main()
