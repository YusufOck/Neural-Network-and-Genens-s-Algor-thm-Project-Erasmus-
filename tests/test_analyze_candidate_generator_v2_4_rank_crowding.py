from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_candidate_generator_v2_4_rank_crowding import (  # noqa: E402
    load_config,
    merge_ranked_pairs,
    ranked_channel_pairs,
    reduce_min_rank,
    stable_directional_rank,
    target_channel_result,
)
from candidate_generator_v2_3_runtime import robust_channel_candidate_keys  # noqa: E402
from duplicate_decision_rule import file_sha256  # noqa: E402


class RankCrowdingDiagnosticTests(unittest.TestCase):
    def test_frozen_config_matches_analyzer_hash(self) -> None:
        config, config_hash = load_config(
            ROOT / "config" / "candidate_generator_v2_4_rank_crowding.json"
        )
        self.assertEqual(
            config_hash,
            "c5073b99fcafcbfc87e691a8250d818ffc39ac2f7de8c8635e3c9c7bba867ab7",
        )
        self.assertEqual(
            config["integrity_dependencies"]["analyzer_sha256"],
            file_sha256(ROOT / "src" / "analyze_candidate_generator_v2_4_rank_crowding.py"),
        )

    def test_stable_directional_rank_breaks_ties_by_index(self) -> None:
        values = np.asarray([-np.inf, 0.9, 0.9, 0.8], dtype=np.float32)
        self.assertEqual(stable_directional_rank(values, 1), 1)
        self.assertEqual(stable_directional_rank(values, 2), 2)
        self.assertEqual(stable_directional_rank(values, 3), 3)

    def test_reduce_min_rank_deduplicates_both_directions(self) -> None:
        keys = np.asarray([9, 4, 9, 7, 4], dtype=np.uint64)
        ranks = np.asarray([5, 3, 2, 4, 1], dtype=np.uint16)
        unique, minimum = reduce_min_rank(keys, ranks)
        np.testing.assert_array_equal(unique, np.asarray([4, 7, 9], dtype=np.uint64))
        np.testing.assert_array_equal(minimum, np.asarray([1, 4, 2], dtype=np.uint16))

    def test_merge_ranked_pairs_keeps_minimum_channel_rank(self) -> None:
        keys, ranks = merge_ranked_pairs(
            np.asarray([1, 2], dtype=np.uint64),
            np.asarray([3, 2], dtype=np.uint16),
            np.asarray([1, 3], dtype=np.uint64),
            np.asarray([1, 4], dtype=np.uint16),
        )
        np.testing.assert_array_equal(keys, np.asarray([1, 2, 3], dtype=np.uint64))
        np.testing.assert_array_equal(ranks, np.asarray([1, 2, 4], dtype=np.uint16))

    def test_ranked_pairs_reproduce_frozen_top_k_sets(self) -> None:
        rng = np.random.default_rng(42)
        descriptors = rng.normal(size=(7, 8)).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        global_indices = np.asarray([2, 5, 9, 11, 20, 31, 44], dtype=np.int64)
        keys, ranks = ranked_channel_pairs(
            descriptors,
            global_indices,
            max_k=3,
            block_size=2,
            device=torch.device("cpu"),
        )
        for top_k in (1, 2, 3):
            expected = robust_channel_candidate_keys(
                descriptors,
                global_indices,
                top_k=top_k,
                block_size=3,
                device=torch.device("cpu"),
            )
            np.testing.assert_array_equal(keys[ranks <= top_k], expected)

    def test_target_channel_result_reports_both_directions(self) -> None:
        descriptors = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.6, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        result = target_channel_result(descriptors, 0, 2)
        self.assertAlmostEqual(result["cosine_similarity"], 0.6, places=6)
        self.assertEqual(result["left_to_right_rank"], 2)
        self.assertEqual(result["right_to_left_rank"], 3)
        self.assertEqual(result["best_directed_rank"], 2)


if __name__ == "__main__":
    unittest.main()
