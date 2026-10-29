from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_natural_duplicate_audit_v1 import (  # noqa: E402
    balanced_phash_sample,
    deterministic_sample,
    reservoir_outside_sample,
)


class NaturalDuplicateAuditSelectionTests(unittest.TestCase):
    def test_deterministic_sample_is_stable_and_unique(self) -> None:
        pool = {(index, index + 100) for index in range(30)}
        first = deterministic_sample(pool, 8, 42, "group:stratum")
        second = deterministic_sample(reversed(sorted(pool)), 8, 42, "group:stratum")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)

    def test_balanced_phash_sample_uses_both_channels_when_available(self) -> None:
        phash_only = {(index, index + 10) for index in range(6)}
        both = {(index + 20, index + 30) for index in range(6)}
        selected = balanced_phash_sample(phash_only, both, 8, 4, 7, "g")
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected) & phash_only), 4)
        self.assertEqual(len(set(selected) & both), 4)

    def test_balanced_phash_sample_does_not_invent_missing_rows(self) -> None:
        selected = balanced_phash_sample({(1, 2)}, set(), 8, 4, 7, "g")
        self.assertEqual(selected, [(1, 2)])

    def test_reservoir_sample_excludes_declared_pairs(self) -> None:
        indices = [0, 1, 2, 3, 4]
        excluded = {(0, 1), (0, 2), (0, 3)}
        selected = reservoir_outside_sample(indices, excluded, 4, 9, "g")
        self.assertEqual(len(selected), 4)
        self.assertFalse(set(selected) & excluded)


if __name__ == "__main__":
    unittest.main()
