from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_ga_proxy_manifest import select_proxy_rows, write_csv  # noqa: E402


def source_row(split: str, leaf: str, index: int) -> dict:
    return {
        "relative_path": f"Group/{leaf}/{split}_{leaf}_{index:04d}.jpg",
        "main_class": "Group",
        "subclass": leaf,
        "split": split,
        "family_id": f"fam_{split}_{leaf}_{index}",
        "family_size": "1",
        "family_sources": "singleton",
    }


class BuildGaProxyManifestTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_excludes_test(self) -> None:
        rows = []
        for split in ("train", "val", "test"):
            for leaf in ("a", "b"):
                rows.extend(source_row(split, leaf, index) for index in range(10))
        first = select_proxy_rows(rows, seed=42, train_per_leaf=4, val_per_leaf=3)
        second = select_proxy_rows(rows, seed=42, train_per_leaf=4, val_per_leaf=3)
        self.assertEqual(first, second)
        self.assertFalse(any(row["split"] == "test" for row in first))
        counts = Counter((row["split"], row["subclass"]) for row in first)
        self.assertEqual(counts[("train", "a")], 4)
        self.assertEqual(counts[("train", "b")], 4)
        self.assertEqual(counts[("val", "a")], 3)
        self.assertEqual(counts[("val", "b")], 3)
        self.assertEqual(len({row["relative_path"] for row in first}), len(first))

    def test_csv_round_trip_preserves_selected_rows(self) -> None:
        rows = [
            {
                **source_row("train", "a", 1),
                "selection_rank": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proxy.csv"
            write_csv(rows, path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(loaded[0]["relative_path"], rows[0]["relative_path"])
        self.assertEqual(loaded[0]["selection_rank"], "1")


if __name__ == "__main__":
    unittest.main()
