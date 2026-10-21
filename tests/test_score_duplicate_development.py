from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from score_duplicate_development import (  # noqa: E402
    REFERENCE_NEGATIVE,
    REFERENCE_POSITIVE,
    REFERENCE_UNCERTAIN,
    load_development_rows,
    reference_label,
    score_ecc,
    score_orb,
    score_phase,
)


PAIR_FIELDS = (
    "pair_id",
    "audit_partition",
    "truth_status",
    "truth_basis",
    "human_verdict",
)


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class DevelopmentScorerTests(unittest.TestCase):
    def test_known_positive_wins_without_review(self) -> None:
        label, basis = reference_label(
            {
                "pair_id": "C1",
                "truth_status": "same_source",
                "truth_basis": "sha256_equal",
                "human_verdict": "",
            }
        )
        self.assertEqual(label, REFERENCE_POSITIVE)
        self.assertEqual(basis, "sha256_equal")

    def test_review_verdict_mapping(self) -> None:
        base = {"pair_id": "C1", "truth_status": "review_required"}
        self.assertEqual(
            reference_label({**base, "human_verdict": "same_source_candidate"})[0],
            REFERENCE_POSITIVE,
        )
        self.assertEqual(
            reference_label({**base, "human_verdict": "independent_lookalike"})[0],
            REFERENCE_NEGATIVE,
        )
        self.assertEqual(
            reference_label({**base, "human_verdict": "uncertain"})[0],
            REFERENCE_UNCERTAIN,
        )

    def test_holdout_rows_are_not_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            reviews = root / "reviews.csv"
            write_csv(
                pairs,
                [
                    {
                        "pair_id": "D1",
                        "audit_partition": "development",
                        "truth_status": "same_source",
                        "truth_basis": "generated_from_left_image",
                        "human_verdict": "same_source",
                    },
                    {
                        "pair_id": "H1",
                        "audit_partition": "holdout",
                        "truth_status": "same_source",
                        "truth_basis": "generated_from_left_image",
                        "human_verdict": "same_source",
                    },
                ],
                PAIR_FIELDS,
            )
            write_csv(
                reviews,
                [],
                ("pair_id", "audit_partition", "human_verdict", "review_status"),
            )
            rows, audit = load_development_rows(pairs, reviews)
            self.assertEqual([row["pair_id"] for row in rows], ["D1"])
            self.assertEqual(audit["holdout_rows_not_loaded"], 1)

    def test_blank_development_review_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            reviews = root / "reviews.csv"
            write_csv(
                pairs,
                [
                    {
                        "pair_id": "D1",
                        "audit_partition": "development",
                        "truth_status": "review_required",
                        "truth_basis": "random_pair_not_proven_independent",
                        "human_verdict": "",
                    }
                ],
                PAIR_FIELDS,
            )
            write_csv(
                reviews,
                [
                    {
                        "pair_id": "D1",
                        "audit_partition": "development",
                        "human_verdict": "",
                        "review_status": "pending",
                    }
                ],
                ("pair_id", "audit_partition", "human_verdict", "review_status"),
            )
            with self.assertRaises(ValueError):
                load_development_rows(pairs, reviews)


class GeometricEvidenceTests(unittest.TestCase):
    @staticmethod
    def textured_image() -> np.ndarray:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.circle(image, (70, 80), 28, 255, 3)
        cv2.rectangle(image, (145, 135), (220, 205), 15, 4)
        cv2.line(image, (20, 225), (230, 25), 180, 3)
        return image

    def test_phase_correlation_recovers_translation(self) -> None:
        moving = self.textured_image()
        matrix = np.float32([[1.0, 0.0, 9.0], [0.0, 1.0, -6.0]])
        fixed = cv2.warpAffine(moving, matrix, (256, 256))
        result = score_phase(moving, fixed)
        estimated = result["matrix"]
        self.assertAlmostEqual(float(estimated[0, 2]), 9.0, delta=0.2)
        self.assertAlmostEqual(float(estimated[1, 2]), -6.0, delta=0.2)
        self.assertGreater(float(result["response"]), 0.9)
        self.assertGreater(float(result["alignment"]["correlation"]), 0.98)

    def test_orb_ransac_finds_controlled_affine_relative(self) -> None:
        moving = self.textured_image()
        matrix = cv2.getRotationMatrix2D((128, 128), 7.0, 1.05).astype(np.float32)
        matrix[:, 2] += np.array([8.0, -5.0], dtype=np.float32)
        fixed = cv2.warpAffine(moving, matrix, (256, 256))
        result = score_orb(
            moving,
            fixed,
            features=4000,
            ratio_test=0.78,
            ransac_threshold=3.0,
        )
        self.assertGreater(int(result["inliers"]), 250)
        self.assertGreater(float(result["inlier_ratio"]), 0.75)
        self.assertGreater(float(result["alignment"]["correlation"]), 0.95)

    def test_ecc_refines_brightness_changed_translation(self) -> None:
        moving = self.textured_image()
        direct = np.float32([[1.0, 0.0, 7.0], [0.0, 1.0, 4.0]])
        fixed = cv2.warpAffine(moving, direct, (256, 256))
        fixed = np.clip(fixed.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        phase = score_phase(moving, fixed)
        result = score_ecc(
            moving,
            fixed,
            phase["matrix"],
            iterations=100,
        )
        self.assertTrue(result["success"])
        self.assertGreater(float(result["score"]), 0.98)
        self.assertGreater(float(result["alignment"]["correlation"]), 0.98)


if __name__ == "__main__":
    unittest.main()
