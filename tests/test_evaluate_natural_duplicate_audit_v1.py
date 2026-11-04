from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze_natural_duplicate_audit_v1 import read_csv  # noqa: E402
from evaluate_natural_duplicate_audit_v1 import (  # noqa: E402
    SOURCE_GROUP_TO_MODALITY,
    prepare_rows,
)


class EvaluateNaturalDuplicateAuditV1Tests(unittest.TestCase):
    def test_prepare_actual_frozen_rows(self) -> None:
        private = read_csv(
            PROJECT_ROOT / "research" / "natural_duplicate_audit_v1" / "selection_private.csv"
        )
        visual = read_csv(
            PROJECT_ROOT / "research" / "natural_duplicate_visual_review_v1.csv"
        )
        rows = prepare_rows(private, visual)
        self.assertEqual(len(rows), 345)
        self.assertEqual(
            Counter(row["reference_label"] for row in rows),
            Counter({"independent": 322, "same_source": 21, "uncertain": 2}),
        )
        self.assertEqual(
            {row["source_group"] for row in rows}, set(SOURCE_GROUP_TO_MODALITY)
        )
        self.assertTrue(all(row["audit_partition"] == "natural_audit_v1" for row in rows))

    def test_unknown_source_group_is_rejected(self) -> None:
        private = [
            {
                "pair_id": "N1",
                "selection_stratum": "outside_random",
                "source_group": "unknown",
            }
        ]
        visual = [
            {
                "pair_id": "N1",
                "visual_verdict": "independent_lookalike",
            }
        ]
        with self.assertRaisesRegex(ValueError, "Bilinmeyen source group"):
            prepare_rows(private, visual)


if __name__ == "__main__":
    unittest.main()
