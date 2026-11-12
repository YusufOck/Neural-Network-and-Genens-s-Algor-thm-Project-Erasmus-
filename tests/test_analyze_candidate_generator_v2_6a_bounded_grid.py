from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_candidate_generator_v2_6a_bounded_grid import (
    load_config,
    merge_selected_channels,
    profile_group_union,
    select_profile,
    verify_dependencies,
)
from candidate_generator_v2_3_runtime import encode_pair_keys


def key(left: int, right: int) -> np.uint64:
    return encode_pair_keys(np.asarray([left]), np.asarray([right]))[0]


class BoundedGridTests(unittest.TestCase):
    def test_frozen_config_and_dependencies_match(self) -> None:
        config, actual = load_config(
            ROOT / "config" / "candidate_generator_v2_6a_bounded_grid.json"
        )
        self.assertEqual(
            actual,
            "580d8d7bd43891c988c3b405b894993a9580d0801f0ff0bf172887350ba88804",
        )
        verify_dependencies(config)

    def test_rejected_v2_6_report_arithmetic(self) -> None:
        report = json.loads(
            (ROOT / "research" / "candidate_generator_v2_6_same_leaf_counterfactual_v1.json")
            .read_text(encoding="utf-8")
        )
        load = report["candidate_load"]
        self.assertEqual(112337 + 249922, load["new_same_leaf_pairs"])
        self.assertEqual(
            load["v2_5_baseline_pairs"] + load["new_same_leaf_pairs"],
            load["proposed_full_pairs"],
        )
        self.assertAlmostEqual(
            load["proposed_full_pairs"] / 130000,
            load["pairs_per_image"],
        )
        self.assertFalse(report["acceptance_gates"]["candidate_load"])

    def test_merge_selected_channels_uses_only_requested_channels_and_min_rank(self) -> None:
        channels = {
            "a": (
                np.asarray([key(0, 1), key(0, 2)], dtype=np.uint64),
                np.asarray([20, 18], dtype=np.uint16),
            ),
            "b": (
                np.asarray([key(0, 1), key(2, 3)], dtype=np.uint64),
                np.asarray([17, 19], dtype=np.uint16),
            ),
        }
        keys, ranks = merge_selected_channels(channels, ["a", "b"])
        self.assertEqual(keys.tolist(), sorted([key(0, 1), key(0, 2), key(2, 3)]))
        rank_by_key = dict(zip(keys.tolist(), ranks.tolist()))
        self.assertEqual(rank_by_key[int(key(0, 1))], 17)

    def test_profile_group_union_filters_rank_leaf_and_channel(self) -> None:
        robust = np.asarray([key(0, 1)], dtype=np.uint64)
        phash = np.asarray([key(4, 5)], dtype=np.uint64)
        channel_pairs = {
            "raw": (
                np.asarray([key(0, 2), key(2, 3), key(4, 6)], dtype=np.uint64),
                np.asarray([20, 29, 18], dtype=np.uint16),
            ),
            "other": (
                np.asarray([key(6, 7)], dtype=np.uint64),
                np.asarray([19], dtype=np.uint16),
            ),
        }
        leaves = ["a", "a", "b", "b", "c", "c", "d", "d"]
        union, additions = profile_group_union(
            robust, phash, channel_pairs, leaves, 20, ["raw"]
        )
        np.testing.assert_array_equal(additions, np.empty(0, dtype=np.uint64))
        np.testing.assert_array_equal(union, np.union1d(robust, phash))

        union, additions = profile_group_union(
            robust, phash, channel_pairs, leaves, 29, ["raw"]
        )
        np.testing.assert_array_equal(additions, np.asarray([key(2, 3)], dtype=np.uint64))
        self.assertNotIn(int(key(6, 7)), union.tolist())

    def test_select_profile_chooses_broadest_passing_profile_then_order(self) -> None:
        rows = [
            {
                "profile_id": "narrow",
                "development_same2d_reference_recall": True,
                "candidate_load_passed": True,
                "full_candidate_pairs": 100,
            },
            {
                "profile_id": "broad",
                "development_same2d_reference_recall": True,
                "candidate_load_passed": True,
                "full_candidate_pairs": 120,
            },
            {
                "profile_id": "too_large",
                "development_same2d_reference_recall": True,
                "candidate_load_passed": False,
                "full_candidate_pairs": 140,
            },
        ]
        self.assertEqual(select_profile(rows), "broad")
        rows[0]["full_candidate_pairs"] = 120
        self.assertEqual(select_profile(rows), "narrow")

    def test_select_profile_returns_none_without_recall_and_capacity(self) -> None:
        rows = [
            {
                "profile_id": "miss",
                "development_same2d_reference_recall": False,
                "candidate_load_passed": True,
                "full_candidate_pairs": 90,
            },
            {
                "profile_id": "large",
                "development_same2d_reference_recall": True,
                "candidate_load_passed": False,
                "full_candidate_pairs": 130,
            },
        ]
        self.assertIsNone(select_profile(rows))


if __name__ == "__main__":
    unittest.main()
