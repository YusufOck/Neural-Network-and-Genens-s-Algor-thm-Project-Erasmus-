from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import finalize_v2_6_same2d_development as finalizer


class V26Same2DFinalizationTests(unittest.TestCase):
    def test_actual_final_reference_counts_and_traceability(self) -> None:
        rows = finalizer.read_csv(finalizer.INPUT)
        final = finalizer.finalize(rows)
        self.assertEqual(len(final), 16)
        self.assertEqual(
            Counter(row["final_reference"] for row in final),
            Counter({"independent": 12, "same_source": 4}),
        )
        self.assertEqual(
            {row["pair_id"] for row in final if row["adjudication_verdict"]},
            set(finalizer.ADJUDICATIONS),
        )
        self.assertTrue(
            all(
                row["reference_basis"].startswith("frozen_duplicate_rule_v1:")
                for row in final
                if not row["adjudication_verdict"]
            )
        )
        self.assertFalse(any(row["final_reference"] == "uncertain" for row in final))

    def test_missing_uncertain_decision_is_rejected(self) -> None:
        rows = finalizer.read_csv(finalizer.INPUT)
        tampered = [dict(row) for row in rows]
        target = next(row for row in tampered if row["pair_id"] == "V25N000084")
        target["rule_decision"] = "independent"
        with self.assertRaisesRegex(ValueError, "uncertain set"):
            finalizer.finalize(tampered)


if __name__ == "__main__":
    unittest.main()
