from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_candidate_generator_v2_6_missed_pair_ranks as diagnostic


class V26MissedPairRankDiagnosticTests(unittest.TestCase):
    def test_real_drive_report_matches_frozen_contract_and_arithmetic(self) -> None:
        report_path = (
            ROOT
            / "research"
            / "candidate_generator_v2_6_missed_pair_rank_diagnostic_v1.json"
        )
        self.assertEqual(
            diagnostic.file_sha256(report_path),
            "88902b9c3810e6f3d945208e96a7d35a3d60a5352eaed98b8dfa60658cd6973f",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["status"],
            "candidate_generator_v2_6_missed_pair_rank_diagnostic_complete",
        )
        self.assertEqual(report["input"]["registered_artifact_count"], 1107)
        self.assertEqual(report["input"]["verified_artifacts_in_this_diagnostic"], 139)
        self.assertEqual(report["input"]["target_pair_count"], 16)

        expected = {
            "brain_ce_mri": (6, 2076, None),
            "kidney_ct": (6, 28, 30),
            "lymphoma_histology": (4, 29, 30),
        }
        base_full = 2_994_870
        image_count = 130_000
        current_group_total = 0
        for group, (pair_count, minimum_k, first_grid) in expected.items():
            result = report["by_source_group"][group]
            self.assertEqual(result["target_pair_count"], pair_count)
            self.assertEqual(result["minimum_k_covering_all_observed_targets"], minimum_k)
            self.assertEqual(result["first_predeclared_grid_k_covering_all"], first_grid)
            self.assertTrue(result["baseline_reproduction"]["all_matched"])
            current_union = result["baseline_reproduction"]["current_v2_5_group_union_pairs"]
            current_group_total += current_union
            previous_union = -1
            for projection in result["candidate_load_projections"]:
                union = projection["target_group_union_pairs"]
                self.assertGreater(union, previous_union)
                previous_union = union
                full = base_full - current_union + union
                self.assertEqual(projection["full_pairs_if_only_this_group_changed"], full)
                self.assertAlmostEqual(projection["full_pairs_per_image"], full / image_count)
                self.assertEqual(
                    projection["passes_25_pairs_per_image_capacity"],
                    full / image_count <= 25.0,
                )
            for target in result["target_pairs"]:
                channel_minimum = min(
                    channel["best_directed_rank"]
                    for channel in target["by_descriptor_channel"].values()
                )
                self.assertEqual(target["minimum_required_robust_top_k"], channel_minimum)

        for projection in report["common_k_candidate_load_projections"]:
            projected_total = sum(projection["projected_union_pairs_by_group"].values())
            full = base_full - current_group_total + projected_total
            self.assertEqual(projection["full_candidate_pairs"], full)
            self.assertAlmostEqual(projection["full_pairs_per_image"], full / image_count)
            self.assertEqual(
                projection["passes_25_pairs_per_image_capacity"],
                full / image_count <= 25.0,
            )
            self.assertFalse(projection["all_16_targets_recovered"])

        brain_pairs = {
            row["pair_id"]: row
            for row in report["by_source_group"]["brain_ce_mri"]["target_pairs"]
        }
        self.assertEqual(brain_pairs["V25N000107"]["minimum_required_robust_top_k"], 2076)

    def test_frozen_config_and_dependencies_match(self) -> None:
        config, config_hash = diagnostic.load_config(
            ROOT / "config" / "candidate_generator_v2_6_missed_pair_rank_diagnostic.json"
        )
        self.assertEqual(
            config_hash,
            "21c6e164c0d2b9f28bb654d9534622ec75d5b99f14b93912cde260d8477645f7",
        )
        diagnostic.verify_dependency_hashes(config)
        rows = diagnostic.read_csv(
            ROOT
            / "research"
            / "candidate_generator_v2_5_prospective_audit_v1"
            / "unblinded_pair_results.csv"
        )
        grouped = diagnostic.validate_targets(config["target_pairs"], rows)
        self.assertEqual(sum(len(items) for items in grouped.values()), 16)

    def test_validate_targets_requires_exact_frozen_outside_positive_set(self) -> None:
        rows = diagnostic.read_csv(
            ROOT
            / "research"
            / "candidate_generator_v2_5_prospective_audit_v1"
            / "unblinded_pair_results.csv"
        )
        positives = [
            row
            for row in rows
            if row["audit_region"] == "outside"
            and row["visual_verdict"] == "same_source_candidate"
        ]
        targets = [
            {
                "pair_id": row["pair_id"],
                "source_group": row["source_group"],
                "left_path": row["left_path"],
                "right_path": row["right_path"],
                "selection_stratum": row["selection_stratum"],
                "phash_distance": int(row["phash_distance"]),
                "audit_robust_best_rank": None if not row["robust_best_rank"] else int(row["robust_best_rank"]),
                "audit_robust_rank_lower_bound": None if not row["robust_rank_lower_bound"] else int(row["robust_rank_lower_bound"]),
            }
            for row in positives
        ]
        grouped = diagnostic.validate_targets(targets, rows)
        self.assertEqual({group: len(items) for group, items in grouped.items()}, {
            "brain_ce_mri": 6,
            "kidney_ct": 6,
            "lymphoma_histology": 4,
        })
        with self.assertRaisesRegex(ValueError, "do not equal"):
            diagnostic.validate_targets(targets[:-1], rows)

    def test_group_projection_is_monotonic_and_uses_full_baseline(self) -> None:
        keys = np.asarray([1, 2, 3, 4, 5], dtype=np.uint64)
        ranks = np.asarray([1, 15, 20, 30, 40], dtype=np.uint16)
        phash = np.asarray([2, 8], dtype=np.uint64)
        result = diagnostic.group_projection(
            keys,
            ranks,
            phash,
            [15, 20, 30, 40],
            current_group_union=3,
            full_v2_5_pairs=100,
            image_count=10,
            required_k=30,
        )
        self.assertEqual([row["target_group_union_pairs"] for row in result], [3, 4, 5, 6])
        self.assertEqual([row["full_pairs_if_only_this_group_changed"] for row in result], [100, 101, 102, 103])
        self.assertEqual([row["all_group_targets_recovered"] for row in result], [False, False, True, True])
        self.assertEqual(diagnostic.first_grid_covering(result), 30)

    def test_first_grid_covering_can_be_absent(self) -> None:
        self.assertIsNone(
            diagnostic.first_grid_covering(
                [
                    {"top_k": 15, "all_group_targets_recovered": False},
                    {"top_k": 20, "all_group_targets_recovered": False},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
