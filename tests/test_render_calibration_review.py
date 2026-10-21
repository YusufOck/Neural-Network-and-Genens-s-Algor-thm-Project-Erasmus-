from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from render_calibration_review import (  # noqa: E402
    prepare_review_rows,
    safe_name,
    thumbnail,
    validate_candidates,
)


def candidate_row() -> dict[str, str]:
    return {
        "pair_id": "C000001",
        "audit_partition": "development",
        "pair_kind": "random_same_leaf_review",
        "truth_status": "review_required",
        "left_leaf": "brain_glioma",
        "right_leaf": "brain_glioma",
        "modality_key": "mri",
        "source_group": "brain_ce_mri",
        "left_root": "dataset",
        "left_path": "left.jpg",
        "right_root": "dataset",
        "right_path": "right.jpg",
        "phash_distance": "18",
        "human_verdict": "",
        "review_notes": "",
    }


class ReviewPreparationTests(unittest.TestCase):
    def test_candidate_validation_preserves_blank_human_decision(self) -> None:
        result = validate_candidates([candidate_row()])
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["blank_human_verdict_count"], 1)

    def test_candidate_validation_rejects_preset_decision(self) -> None:
        row = candidate_row()
        row["human_verdict"] = "independent_lookalike"
        with self.assertRaises(ValueError):
            validate_candidates([row])

    def test_review_fields_are_added_without_filling_verdict(self) -> None:
        source = candidate_row()
        fields, rows = prepare_review_rows([source], list(source))
        self.assertIn("review_status", fields)
        self.assertEqual(rows[0]["review_status"], "pending_visual_review")
        self.assertEqual(rows[0]["human_verdict"], "")

    def test_thumbnail_has_requested_canvas_size(self) -> None:
        image_path = PROJECT_ROOT / "tests" / "_temporary_review_image.png"
        try:
            Image.new("RGB", (40, 20), "navy").save(image_path)
            rendered = thumbnail(image_path, 128)
            self.assertEqual(rendered.size, (128, 128))
        finally:
            image_path.unlink(missing_ok=True)

    def test_safe_name_removes_path_punctuation(self) -> None:
        self.assertEqual(safe_name("MRI / hold-out"), "MRI___hold_out")


if __name__ == "__main__":
    unittest.main()
