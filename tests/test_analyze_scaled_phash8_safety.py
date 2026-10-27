from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze_scaled_phash8_safety import phash_radius_candidates  # noqa: E402
from benchmark_candidate_generation import ImageItem  # noqa: E402


class ScaledPHashSafetyTests(unittest.TestCase):
    def test_vectorized_radius_candidates_are_flip_aware_and_group_bounded(self) -> None:
        items = [
            ImageItem(
                key=f"dataset:image_{index}.jpg",
                root_kind="dataset",
                relative_path=f"image_{index}.jpg",
                source_group="a" if index < 3 else "b",
                absolute_path=Path(f"image_{index}.jpg"),
            )
            for index in range(4)
        ]
        original = np.asarray([0b0000, 0b1111, 0b0011, 0b0000], dtype=np.uint64)
        flipped = np.asarray([0b0000, 0b0001, 0b0011, 0b0000], dtype=np.uint64)
        candidates, counts = phash_radius_candidates(
            items, original, flipped, radius=1, block_size=2
        )
        self.assertIn((0, 1), candidates)
        self.assertNotIn((0, 3), candidates)
        self.assertEqual(counts["a"], len(candidates))


if __name__ == "__main__":
    unittest.main()
