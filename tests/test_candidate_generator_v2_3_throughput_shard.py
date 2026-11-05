from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_candidate_generation import ImageItem  # noqa: E402
from data_index import ImageRecord  # noqa: E402
from run_candidate_generator_v2_3_throughput_shard import (  # noqa: E402
    build_candidate_rows,
    select_deterministic_shard,
    verify_candidate_integrity,
)


class CandidateGeneratorV23ThroughputShardTests(unittest.TestCase):
    def test_deterministic_shard_is_balanced_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records: list[ImageRecord] = []
            for leaf in ("leaf_a", "leaf_b"):
                leaf_dir = root / "Main" / leaf
                leaf_dir.mkdir(parents=True)
                for index in range(6):
                    path = leaf_dir / f"image_{index}.jpg"
                    path.touch()
                    records.append(
                        ImageRecord(path=path, main_class="Main", subclass=leaf)
                    )
            groups = {"leaf_a": "group_a", "leaf_b": "group_b"}
            first, first_records, first_ranks = select_deterministic_shard(
                records, root, groups, images_per_leaf=3, seed=42
            )
            second, _, _ = select_deterministic_shard(
                records, root, groups, images_per_leaf=3, seed=42
            )
            self.assertEqual([item.relative_path for item in first], [item.relative_path for item in second])
            self.assertEqual(len(first), 6)
            self.assertEqual(len(first_records), 6)
            self.assertEqual(sorted(first_ranks.values()), [1, 1, 2, 2, 3, 3])

    def test_candidate_integrity_rejects_cross_group_pair(self) -> None:
        items = [
            ImageItem("a", "dataset", "a.jpg", "group_a", Path("a.jpg")),
            ImageItem("b", "dataset", "b.jpg", "group_b", Path("b.jpg")),
        ]
        with self.assertRaisesRegex(ValueError, "Source-group"):
            verify_candidate_integrity(items, {(0, 1)})

    def test_candidate_rows_record_union_channels(self) -> None:
        items = [
            ImageItem("a", "dataset", "a.jpg", "group", Path("a.jpg")),
            ImageItem("b", "dataset", "b.jpg", "group", Path("b.jpg")),
            ImageItem("c", "dataset", "c.jpg", "group", Path("c.jpg")),
        ]
        records = {
            name: ImageRecord(Path(name), main_class="Main", subclass=f"leaf_{name[0]}")
            for name in ("a.jpg", "b.jpg", "c.jpg")
        }
        rows = build_candidate_rows(items, records, {(0, 1)}, {(0, 1), (1, 2)})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["candidate_channels"], "robust_top15+phash_le12")
        self.assertEqual(rows[1]["candidate_channels"], "phash_le12")


if __name__ == "__main__":
    unittest.main()
