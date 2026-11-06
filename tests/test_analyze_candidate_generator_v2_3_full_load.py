from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_candidate_generator_v2_3_full_load import channel_summary, unique_pair_keys


class FullLoadDiagnosticTests(unittest.TestCase):
    def test_unique_pair_keys_merges_and_deduplicates(self) -> None:
        result = unique_pair_keys(
            [
                np.array([9, 2, 9], dtype=np.uint64),
                np.array([4, 2], dtype=np.uint64),
            ]
        )
        np.testing.assert_array_equal(result, np.array([2, 4, 9], dtype=np.uint64))

    def test_channel_summary_reports_union_arithmetic(self) -> None:
        summary, union = channel_summary(
            [np.array([1, 2, 3], dtype=np.uint64)],
            [np.array([3, 4], dtype=np.uint64)],
        )
        self.assertEqual(
            summary,
            {
                "robust_unique_pairs": 3,
                "phash_unique_pairs": 2,
                "intersection_pairs": 1,
                "union_pairs": 4,
            },
        )
        np.testing.assert_array_equal(union, np.array([1, 2, 3, 4], dtype=np.uint64))

    def test_empty_parts_are_supported(self) -> None:
        self.assertEqual(len(unique_pair_keys([])), 0)


if __name__ == "__main__":
    unittest.main()
