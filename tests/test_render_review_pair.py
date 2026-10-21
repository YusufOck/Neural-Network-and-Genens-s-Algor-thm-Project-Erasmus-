from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from render_review_pair import (  # noqa: E402
    metadata_lines,
    read_review_row,
    render_pair,
    validate_review_access,
)


def review_row(partition: str = "development") -> dict[str, str]:
    return {
        "pair_id": "C000613",
        "audit_partition": partition,
        "pair_kind": "phash_hard_same_leaf_review",
        "source_group": "lung_colon_histology",
        "left_leaf": "colon_aca",
        "right_leaf": "colon_aca",
        "left_path": "left.jpg",
        "right_path": "right.jpg",
        "phash_distance": "10",
        "human_verdict": "",
    }


class SinglePairReviewTests(unittest.TestCase):
    def test_holdout_is_hidden_by_default(self) -> None:
        with self.assertRaises(PermissionError):
            validate_review_access(review_row("holdout"), allow_holdout=False)

    def test_development_is_available(self) -> None:
        validate_review_access(review_row("development"), allow_holdout=False)

    def test_metadata_keeps_pair_id_and_pending_verdict(self) -> None:
        lines = metadata_lines(review_row())
        self.assertTrue(any("C000613" in line for line in lines))
        self.assertTrue(any("BEKLIYOR" in line for line in lines))

    def test_rendered_pair_has_expected_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.png"
            right = root / "right.png"
            output = root / "pair.png"
            Image.new("RGB", (80, 60), "purple").save(left)
            Image.new("RGB", (60, 80), "pink").save(right)
            render_pair(review_row(), left, right, output, image_size=256)
            with Image.open(output) as rendered:
                self.assertEqual(rendered.width, 24 * 2 + 256 * 2 + 20)
                self.assertGreater(rendered.height, 256)


if __name__ == "__main__":
    unittest.main()
