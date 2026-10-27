from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_candidate_generation_v2 import (  # noqa: E402
    acceptance_summary,
    best_pair_rank_and_similarity,
    channel_similarity_matrices,
    descriptor_batch,
    hybrid_candidate_sets,
    spatial_views,
)
from benchmark_candidate_generation import ImageItem  # noqa: E402


class CandidateGenerationV2Tests(unittest.TestCase):
    def test_spatial_views_have_expected_geometry(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        views = spatial_views(image, crop_ratio=0.8)
        self.assertEqual(len(views), 6)
        self.assertEqual(views[0].size, (100, 80))
        self.assertTrue(all(view.size == (80, 64) for view in views[1:]))

    def test_descriptor_batch_is_normalized(self) -> None:
        generator = torch.Generator().manual_seed(42)
        features = torch.rand((3, 2, 6, 2, 8), generator=generator)
        output = descriptor_batch(features)
        self.assertEqual(len(output), 4)
        for values in output.values():
            self.assertEqual(tuple(values.shape), (3, 8))
            np.testing.assert_allclose(
                torch.linalg.vector_norm(values, dim=1).numpy(),
                np.ones(3),
                atol=1e-6,
            )

    def test_hybrid_is_set_union(self) -> None:
        topk = {5: {(0, 1)}, 10: {(0, 1)}, 20: {(0, 1)}, 50: {(0, 1)}}
        phash = {8: {(1, 2)}, 12: {(1, 2)}, 16: {(1, 2)}, 24: {(1, 2)}}
        output = hybrid_candidate_sets(topk, phash)
        self.assertEqual(
            output["top5_union_phash8"], {(0, 1), (1, 2)}
        )

    def test_acceptance_requires_recall_and_workload(self) -> None:
        base = {
            "positive_recall": 1.0,
            "natural_positive_recall": 1.0,
            "candidate_pair_count": 25,
        }
        accepted = acceptance_summary({"good": base}, possible_pairs=100)
        self.assertTrue(accepted["pilot_gate_passed"])
        rejected = acceptance_summary(
            {"too_large": {**base, "candidate_pair_count": 26}},
            possible_pairs=100,
        )
        self.assertFalse(rejected["pilot_gate_passed"])

    def test_reported_rank_matches_stable_topk_order_when_scores_tie(self) -> None:
        items = [
            ImageItem(
                key=f"dataset:image_{index}.jpg",
                root_kind="dataset",
                relative_path=f"image_{index}.jpg",
                source_group="g",
                absolute_path=Path(f"image_{index}.jpg"),
            )
            for index in range(4)
        ]
        tied = np.asarray([[1.0, 0.0]] * 4, dtype=np.float32)
        descriptors = {name: tied for name in (
            "raw_full_flip_mean",
            "autocontrast_full_flip_mean",
            "raw_multicrop_flip_mean",
            "autocontrast_multicrop_flip_mean",
        )}
        matrices = channel_similarity_matrices(items, descriptors)
        _, rank, _ = best_pair_rank_and_similarity(2, 3, "g", matrices)
        self.assertEqual(rank, 3)


if __name__ == "__main__":
    unittest.main()
