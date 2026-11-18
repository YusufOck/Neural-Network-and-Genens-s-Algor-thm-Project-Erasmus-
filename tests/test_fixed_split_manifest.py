from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_index import ImageRecord
from fixed_split_manifest import load_fixed_split_manifest, load_search_proxy_manifest


class FixedSplitManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = []
        for split, suffix in (("train", "a"), ("val", "b"), ("test", "c")):
            path = self.root / "Main" / "leaf" / f"{suffix}.jpg"
            self.records.append(ImageRecord(path=path, main_class="Main", subclass="leaf"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self, rows) -> Path:
        path = self.root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "main_class", "subclass", "split", "family_id"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def valid_rows(self):
        return [
            {
                "relative_path": record.path.relative_to(self.root).as_posix(),
                "main_class": record.main_class,
                "subclass": record.subclass,
                "split": split,
                "family_id": f"fam_{index}",
            }
            for index, (record, split) in enumerate(zip(self.records, ("train", "val", "test")))
        ]

    def test_loads_complete_fixed_manifest(self) -> None:
        loaded = load_fixed_split_manifest(self.records, self.root, self.write_manifest(self.valid_rows()))
        self.assertEqual({name: len(rows) for name, rows in loaded.items()}, {"train": 1, "val": 1, "test": 1})

    def test_rejects_duplicate_or_missing_path(self) -> None:
        rows = self.valid_rows()
        rows[1]["relative_path"] = rows[0]["relative_path"]
        with self.assertRaisesRegex(ValueError, "tekrarlanan path"):
            load_fixed_split_manifest(self.records, self.root, self.write_manifest(rows))

    def test_rejects_metadata_mismatch(self) -> None:
        rows = self.valid_rows()
        rows[0]["subclass"] = "wrong"
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_fixed_split_manifest(self.records, self.root, self.write_manifest(rows))

    def test_search_proxy_cannot_include_or_move_test_rows(self) -> None:
        fixed = load_fixed_split_manifest(
            self.records, self.root, self.write_manifest(self.valid_rows())
        )
        proxy_path = self.root / "proxy.csv"
        proxy_rows = self.valid_rows()[:2]
        with proxy_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "main_class", "subclass", "split", "family_id"],
            )
            writer.writeheader()
            writer.writerows(proxy_rows)
        loaded = load_search_proxy_manifest(fixed, self.root, proxy_path)
        self.assertEqual({name: len(rows) for name, rows in loaded.items()}, {"train": 1, "val": 1})

        proxy_rows[1]["split"] = "test"
        with proxy_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "main_class", "subclass", "split", "family_id"],
            )
            writer.writeheader()
            writer.writerows(proxy_rows)
        with self.assertRaisesRegex(ValueError, "yalniz train/val"):
            load_search_proxy_manifest(fixed, self.root, proxy_path)


if __name__ == "__main__":
    unittest.main()
