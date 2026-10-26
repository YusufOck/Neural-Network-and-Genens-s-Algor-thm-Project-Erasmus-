from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_candidate_generation import (  # noqa: E402
    ImageItem,
    canonical_pair,
    evaluate_candidate_set,
    similarity_matrices,
    topk_candidate_sets,
)


def item(index: int, group: str = "g") -> ImageItem:
    return ImageItem(
        key=f"dataset:image_{index}.jpg",
        root_kind="dataset",
        relative_path=f"image_{index}.jpg",
        source_group=group,
        absolute_path=Path(f"image_{index}.jpg"),
    )


class CandidateGenerationBenchmarkTests(unittest.TestCase):
    def test_topk_union_can_retrieve_one_direction(self) -> None:
        items = [item(0), item(1), item(2)]
        original = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32
        )
        original /= np.linalg.norm(original, axis=1, keepdims=True)
        matrices = similarity_matrices(items, original, original)
        candidates = topk_candidate_sets(matrices)
        self.assertIn(canonical_pair(0, 1), candidates[1])
        self.assertNotIn(canonical_pair(0, 2), candidates[1])

    def test_candidate_metrics_treat_uncertain_separately(self) -> None:
        rows = [
            {
                "pair_id": "P1",
                "left_root": "dataset",
                "left_path": "image_0.jpg",
                "right_root": "dataset",
                "right_path": "image_1.jpg",
                "reference_label": "same_source",
                "truth_status": "same_source",
                "transform_name": "brightness_0_80",
                "pair_kind": "controlled_synthetic_positive",
            },
            {
                "pair_id": "P2",
                "left_root": "dataset",
                "left_path": "image_0.jpg",
                "right_root": "dataset",
                "right_path": "image_2.jpg",
                "reference_label": "uncertain",
                "truth_status": "review_required",
                "transform_name": "",
                "pair_kind": "phash_hard_same_leaf_review",
            },
        ]
        metrics = evaluate_candidate_set(
            {canonical_pair(0, 1), canonical_pair(0, 2)},
            rows,
            {
                "dataset:image_0.jpg": 0,
                "dataset:image_1.jpg": 1,
                "dataset:image_2.jpg": 2,
            },
            3,
        )
        self.assertEqual(metrics["positive_count"], 1)
        self.assertEqual(metrics["positive_recall"], 1.0)
        self.assertEqual(metrics["reviewed_independent_count"], 0)


if __name__ == "__main__":
    unittest.main()
