from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_generator_v2_3_runtime import encode_pair_keys
from analyze_candidate_generator_v2_6_same_leaf_counterfactual import (
    proposal_union,
    same_leaf_mask,
)


def key(left: int, right: int) -> np.uint64:
    return encode_pair_keys(np.asarray([left]), np.asarray([right]))[0]


class SameLeafCounterfactualTests(unittest.TestCase):
    def test_same_leaf_mask_uses_image_index_leaf_labels(self) -> None:
        keys = np.asarray([key(0, 1), key(0, 2), key(2, 3)], dtype=np.uint64)
        mask = same_leaf_mask(keys, ["a", "a", "b", "b"])
        np.testing.assert_array_equal(mask, np.asarray([True, False, True]))

    def test_proposal_keeps_baseline_and_adds_only_same_leaf_rank_16_to_30(self) -> None:
        baseline = np.asarray([key(0, 1)], dtype=np.uint64)
        phash = np.asarray([key(2, 3)], dtype=np.uint64)
        ranked = np.asarray(
            [key(0, 1), key(0, 2), key(2, 3), key(2, 4), key(4, 5)],
            dtype=np.uint64,
        )
        ranks = np.asarray([15, 20, 25, 30, 31], dtype=np.uint16)
        proposal, additions = proposal_union(
            baseline,
            phash,
            ranked,
            ranks,
            ["a", "a", "b", "b", "b", "b"],
            30,
        )
        np.testing.assert_array_equal(additions, np.asarray([key(2, 4)], dtype=np.uint64))
        np.testing.assert_array_equal(
            proposal,
            np.sort(np.asarray([key(0, 1), key(2, 3), key(2, 4)], dtype=np.uint64)),
        )

    def test_out_of_range_pair_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            same_leaf_mask(np.asarray([key(0, 3)], dtype=np.uint64), ["a", "a"])


if __name__ == "__main__":
    unittest.main()
