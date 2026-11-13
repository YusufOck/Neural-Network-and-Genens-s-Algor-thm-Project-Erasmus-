from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_candidate_generator_v2_6b_prospective_audit import (
    REVIEW_FIELDS,
    load_config,
    load_excluded_paths,
    random_same_leaf_outside_keys,
    rank_window_same_leaf,
    selected_ranked_pairs,
)
from candidate_generator_v2_3_runtime import encode_pair_keys


def keys(pairs: list[tuple[int, int]]) -> np.ndarray:
    return encode_pair_keys(
        np.asarray([left for left, _ in pairs]),
        np.asarray([right for _, right in pairs]),
    )


class V26BProspectiveAuditTests(unittest.TestCase):
    def test_actual_config_builder_and_exclusion_contract_are_frozen(self) -> None:
        config, digest = load_config(
            ROOT / "config" / "candidate_generator_v2_6b_prospective_audit.json"
        )
        self.assertEqual(
            digest,
            "964bf13d3cd73ecc29e777775a8a9f64af94ada86bec5cc191f1f770032ff6c4",
        )
        self.assertEqual(
            config["candidate_rule"]["selected_profile"],
            "kidney_all_lymphoma_raw_multicrop",
        )
        self.assertEqual(len(load_excluded_paths(config)), 23434)

    def test_rank_window_requires_same_leaf_and_bounded_rank(self) -> None:
        pair_keys = keys([(0, 1), (0, 2), (2, 3), (4, 5)])
        ranks = np.asarray([16, 17, 29, 30], dtype=np.uint16)
        leaves = ["a", "a", "b", "b", "c", "c"]
        selected = rank_window_same_leaf(pair_keys, ranks, leaves, 16, 29)
        np.testing.assert_array_equal(selected, pair_keys[[0, 2]])

    def test_selected_ranked_pairs_ignores_unselected_channel(self) -> None:
        pair_keys = {
            "raw": (keys([(0, 1)]), np.asarray([20], dtype=np.uint16)),
            "other": (keys([(2, 3)]), np.asarray([18], dtype=np.uint16)),
        }
        selected_keys, selected_ranks = selected_ranked_pairs(pair_keys, ["raw"])
        np.testing.assert_array_equal(selected_keys, keys([(0, 1)]))
        np.testing.assert_array_equal(selected_ranks, np.asarray([20], dtype=np.uint16))

    def test_random_outside_is_same_leaf_disjoint_and_deterministic(self) -> None:
        by_leaf = {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}
        forbidden = keys([(0, 1)])
        first_used: set[int] = set()
        second_used: set[int] = set()
        first = random_same_leaf_outside_keys(
            by_leaf, forbidden, 3, 42, "x", first_used
        )
        second = random_same_leaf_outside_keys(
            by_leaf, forbidden, 3, 42, "x", second_used
        )
        self.assertEqual([int(value) for value in first], [int(value) for value in second])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first_used), 6)
        for key in first:
            left, right = (int(value[0]) for value in decode(key))
            self.assertEqual(left // 4, right // 4)
            self.assertNotEqual((left, right), (0, 1))

    def test_review_manifest_hides_private_fields(self) -> None:
        for field in (
            "selection_stratum",
            "v2_6b_candidate",
            "selected_extension_best_rank",
            "phash_distance",
        ):
            self.assertNotIn(field, REVIEW_FIELDS)


def decode(key: np.uint64) -> tuple[np.ndarray, np.ndarray]:
    value = int(key)
    return np.asarray([value >> 32]), np.asarray([value & 0xFFFFFFFF])


if __name__ == "__main__":
    unittest.main()
