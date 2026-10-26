from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prepare_holdout_blind_review import (  # noqa: E402
    blind_metadata_lines,
    load_pending_holdout_rows,
)


FIELDS = (
    "pair_id",
    "audit_partition",
    "truth_status",
    "human_verdict",
    "review_status",
    "pair_kind",
    "source_group",
    "left_leaf",
    "right_leaf",
)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def holdout_row() -> dict[str, str]:
    return {
        "pair_id": "C001204",
        "audit_partition": "holdout",
        "truth_status": "review_required",
        "human_verdict": "",
        "review_status": "pending_visual_review",
        "pair_kind": "random_same_leaf_review",
        "source_group": "brain_ce_mri",
        "left_leaf": "brain_glioma",
        "right_leaf": "brain_glioma",
    }


class HoldoutBlindReviewTests(unittest.TestCase):
    def test_metadata_omits_all_algorithm_scores(self) -> None:
        lines = " ".join(blind_metadata_lines(holdout_row())).lower()
        for forbidden in ("phash", "orb", "ransac", "ecc", "ssim", "nmi", "correlation", "score"):
            self.assertNotIn(forbidden, lines)

    def test_only_pending_blank_holdout_rows_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            development = {**holdout_row(), "pair_id": "C000533", "audit_partition": "development"}
            write_rows(path, [development, holdout_row()])
            rows = load_pending_holdout_rows(path)
            self.assertEqual([row["pair_id"] for row in rows], ["C001204"])

    def test_prelabelled_holdout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            labelled = {**holdout_row(), "human_verdict": "independent_lookalike"}
            write_rows(path, [labelled])
            with self.assertRaises(ValueError):
                load_pending_holdout_rows(path)


if __name__ == "__main__":
    unittest.main()
