from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_ga_full_confirmation_manifest import (  # noqa: E402
    read_source_rows,
    select_full_train_val_rows,
    sha256_file,
)


SOURCE = ROOT / "research/splits/practical_leakage_reduced_split_v1.csv"


class BuildGaFullConfirmationManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = select_full_train_val_rows(read_source_rows(SOURCE))

    def test_source_hash_and_full_counts_are_frozen(self) -> None:
        self.assertEqual(
            sha256_file(SOURCE),
            "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171",
        )
        counts = Counter(row["split"] for row in self.rows)
        self.assertEqual(counts, {"train": 91_000, "val": 19_500})
        self.assertNotIn("test", counts)

    def test_every_leaf_has_all_train_and_validation_rows(self) -> None:
        counts = Counter((row["split"], row["subclass"]) for row in self.rows)
        leaves = sorted({row["subclass"] for row in self.rows})
        self.assertEqual(len(leaves), 26)
        for leaf in leaves:
            self.assertEqual(counts[("train", leaf)], 3500)
            self.assertEqual(counts[("val", leaf)], 750)

    def test_paths_are_unique_and_selection_rank_is_complete(self) -> None:
        paths = [row["relative_path"] for row in self.rows]
        self.assertEqual(len(paths), len(set(paths)))
        ranks: dict[tuple[str, str], list[int]] = {}
        for row in self.rows:
            ranks.setdefault((row["split"], row["subclass"]), []).append(
                int(row["selection_rank"])
            )
        for values in ranks.values():
            self.assertEqual(values, list(range(1, len(values) + 1)))


if __name__ == "__main__":
    unittest.main()
