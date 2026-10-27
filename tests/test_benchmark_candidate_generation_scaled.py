from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_candidate_generation import ImageItem  # noqa: E402
from benchmark_candidate_generation_scaled import (  # noqa: E402
    blockwise_topk_candidates,
    parse_stage_sizes,
    select_nested_distractors,
    stage_items_and_descriptors,
)
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES  # noqa: E402
from data_index import ImageRecord  # noqa: E402


def item(index: int, group: str = "g") -> ImageItem:
    return ImageItem(
        key=f"dataset:leaf/image_{index}.jpg",
        root_kind="dataset",
        relative_path=f"leaf/image_{index}.jpg",
        source_group=group,
        absolute_path=Path(f"image_{index}.jpg"),
    )


class ScaledCandidateGenerationTests(unittest.TestCase):
    def test_stage_sizes_require_sorted_unique_positive_values(self) -> None:
        self.assertEqual(parse_stage_sizes("50,100,250"), (50, 100, 250))
        with self.assertRaises(Exception):
            parse_stage_sizes("100,50")
        with self.assertRaises(Exception):
            parse_stage_sizes("50,50")

    def test_nested_distractors_exclude_calibration_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf_dir = root / "Main" / "leaf"
            leaf_dir.mkdir(parents=True)
            records = []
            for index in range(5):
                path = leaf_dir / f"image_{index}.jpg"
                path.touch()
                records.append(ImageRecord(path=path, main_class="Main", subclass="leaf"))
            excluded = {"dataset:Main/leaf/image_0.jpg"}
            selected, ranks, leaves = select_nested_distractors(
                records,
                root,
                excluded,
                {"leaf": "group"},
                maximum_per_leaf=3,
                seed=42,
            )
            self.assertEqual(len(selected), 3)
            self.assertTrue(excluded.isdisjoint(item.key for item in selected))
            self.assertEqual(sorted(ranks.values()), [1, 2, 3])
            self.assertEqual(set(leaves.values()), {"leaf"})

    def test_blockwise_retrieval_finds_nearest_pair(self) -> None:
        items = [item(index) for index in range(4)]
        base = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        descriptors = {name: base for name in DESCRIPTOR_NAMES}
        candidates, metadata = blockwise_topk_candidates(
            items, descriptors, block_size=2, topk_values=(1, 2)
        )
        self.assertIn((0, 1), candidates[1])
        self.assertEqual(metadata["block_size"], 2)

    def test_stage_selection_preserves_calibration_and_nested_prefix(self) -> None:
        items = [item(index) for index in range(5)]
        descriptors = {
            name: np.arange(10, dtype=np.float32).reshape(5, 2)
            for name in DESCRIPTOR_NAMES
        }
        stage_items, stage_descriptors = stage_items_and_descriptors(
            items,
            descriptors,
            calibration_keys={items[0].key, items[1].key},
            distractor_rank_by_key={
                items[2].key: 1,
                items[3].key: 2,
                items[4].key: 3,
            },
            stage_size=2,
        )
        self.assertEqual([entry.key for entry in stage_items], [item.key for item in items[:4]])
        self.assertTrue(all(values.shape == (4, 2) for values in stage_descriptors.values()))


if __name__ == "__main__":
    unittest.main()
