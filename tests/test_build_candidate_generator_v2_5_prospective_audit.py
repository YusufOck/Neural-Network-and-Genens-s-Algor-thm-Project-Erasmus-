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

from build_candidate_generator_v2_5_prospective_audit import (  # noqa: E402
    REVIEW_FIELDS,
    filter_eligible_keys,
    flip_aware_distance,
    load_config,
    random_outside_keys,
    rank_range_keys,
    select_disjoint_keys,
    subtract_keys,
)
from candidate_generator_v2_3_runtime import decode_pair_keys, encode_pair_keys  # noqa: E402
from duplicate_decision_rule import file_sha256  # noqa: E402


def keys(pairs: list[tuple[int, int]]) -> np.ndarray:
    return encode_pair_keys(
        np.asarray([left for left, _ in pairs]),
        np.asarray([right for _, right in pairs]),
    )


class V25ProspectiveAuditTests(unittest.TestCase):
    def test_actual_config_is_frozen_and_matches_builder(self) -> None:
        path = ROOT / "config" / "candidate_generator_v2_5_prospective_audit.json"
        config, digest = load_config(path)
        self.assertEqual(
            digest,
            "2c4213d34c70991550b85ec2d4699a0c7cf49b804904cc9bb3b4d7518d8fe745",
        )
        self.assertEqual(
            config["integrity_dependencies"]["builder_sha256"],
            file_sha256(SRC / "build_candidate_generator_v2_5_prospective_audit.py"),
        )
        self.assertEqual(
            config["candidate_rule"]["robust_top_k_by_group"]["cervix_pap_smear"],
            40,
        )
        self.assertEqual(config["expected_full_state"]["v2_5_full_pairs"], 2_994_870)

    def test_actual_exclusion_manifest_is_frozen_and_complete(self) -> None:
        path = ROOT / "research" / "candidate_generator_v2_5_excluded_paths_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_path_count"], 22_922)
        self.assertEqual(
            payload["excluded_paths_sha256"],
            "1569fa19392cb840b81bfc018e5849050774f3f36beb5a5478281b9d2e0be0f6",
        )
        self.assertEqual(
            file_sha256(path),
            "bc90f337d0058648c27312886cd407177a7e329a85373b13c58f4ab68bf38c3f",
        )
        self.assertIn(
            "Cervical Cancer/cervix_dyk/cervix_dyk_4875.jpg",
            payload["excluded_paths"],
        )

    def test_rank_range_and_subtraction_define_boundary_pool(self) -> None:
        pair_keys = keys([(0, 1), (0, 2), (1, 2), (2, 3)])
        ranks = np.asarray([15, 16, 30, 31], dtype=np.uint16)
        boundary = rank_range_keys(pair_keys, ranks, 16, 30)
        np.testing.assert_array_equal(boundary, pair_keys[1:3])
        np.testing.assert_array_equal(
            subtract_keys(boundary, pair_keys[1:2]), pair_keys[2:3]
        )

    def test_filter_eligible_requires_both_endpoints(self) -> None:
        pair_keys = keys([(0, 1), (1, 2), (2, 3)])
        eligible = np.asarray([True, True, False, True])
        np.testing.assert_array_equal(
            filter_eligible_keys(pair_keys, eligible), pair_keys[:1]
        )

    def test_disjoint_selection_is_deterministic_and_one_pair_per_image(self) -> None:
        pair_keys = keys([(0, 1), (2, 3), (4, 5), (6, 7)])
        first_used: set[int] = set()
        second_used: set[int] = set()
        first = select_disjoint_keys(pair_keys, 3, 42, "x", first_used)
        second = select_disjoint_keys(pair_keys, 3, 42, "x", second_used)
        self.assertEqual([int(value) for value in first], [int(value) for value in second])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first_used), 6)

    def test_random_outside_avoids_forbidden_pairs_and_reused_images(self) -> None:
        forbidden = keys([(0, 1), (2, 3)])
        used = {8, 9}
        selected = random_outside_keys(
            list(range(10)), forbidden, 3, 123, "outside", used
        )
        self.assertEqual(len(selected), 3)
        endpoints: list[int] = []
        for key in selected:
            left, right = decode_pair_keys(np.asarray([key]))
            pair = (int(left[0]), int(right[0]))
            self.assertNotIn(pair, {(0, 1), (2, 3)})
            endpoints.extend(pair)
        self.assertEqual(len(endpoints), len(set(endpoints)))
        self.assertNotIn(8, endpoints)
        self.assertNotIn(9, endpoints)

    def test_flip_aware_distance_uses_best_orientation(self) -> None:
        original = np.asarray([0b0000, 0b1111], dtype=np.uint64)
        flipped = np.asarray([0b0011, 0b0000], dtype=np.uint64)
        self.assertEqual(flip_aware_distance(0, 1, original, flipped), 0)

    def test_review_manifest_hides_every_frozen_sensitive_field(self) -> None:
        config = json.loads(
            (ROOT / "config" / "candidate_generator_v2_5_prospective_audit.json").read_text(
                encoding="utf-8"
            )
        )
        hidden = set(config["blinding"]["review_manifest_hides"])
        self.assertTrue(hidden.isdisjoint(REVIEW_FIELDS))
        self.assertNotIn("selection_stratum", REVIEW_FIELDS)
        self.assertNotIn("phash_distance", REVIEW_FIELDS)
        self.assertNotIn("robust_best_rank", REVIEW_FIELDS)


if __name__ == "__main__":
    unittest.main()
