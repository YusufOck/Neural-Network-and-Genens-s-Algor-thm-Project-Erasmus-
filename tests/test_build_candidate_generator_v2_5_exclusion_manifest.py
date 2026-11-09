from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_candidate_generator_v2_5_exclusion_manifest import (
    build_manifest,
    canonical_paths_sha256,
    normalized_path,
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class V25ExclusionManifestTests(unittest.TestCase):
    def test_normalized_path_rejects_absolute_paths(self) -> None:
        self.assertEqual(normalized_path(r"Brain Cancer\\x.jpg"), "Brain Cancer/x.jpg")
        with self.assertRaises(ValueError):
            normalized_path("C:/dataset/x.jpg")

    def test_build_manifest_unions_and_deduplicates_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexes = []
            for ordinal, paths in enumerate((("a.jpg", "b.jpg"), ("b.jpg", "c.jpg"), ("d.jpg",))):
                path = root / f"index_{ordinal}.csv"
                write_csv(
                    path,
                    ["root_kind", "relative_path"],
                    [
                        {"root_kind": "dataset", "relative_path": value}
                        for value in paths
                    ]
                    + [{"root_kind": "generated", "relative_path": "ignore.png"}],
                )
                indexes.append(path)
            calibration = root / "pairs.csv"
            write_csv(
                calibration,
                ["left_root", "left_path", "right_root", "right_path"],
                [
                    {
                        "left_root": "dataset",
                        "left_path": "e.jpg",
                        "right_root": "generated",
                        "right_path": "ignore.png",
                    }
                ],
            )
            exact = root / "exact.json"
            exact.write_text(
                json.dumps({"duplicate_groups": [{"paths": ["a.jpg", "f.jpg"]}]}),
                encoding="utf-8",
            )
            payload = build_manifest(
                indexes[0], indexes[1], indexes[2], calibration, exact
            )
            self.assertEqual(payload["excluded_paths"], ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"])
            self.assertEqual(payload["excluded_path_count"], 6)
            self.assertEqual(
                payload["excluded_paths_sha256"],
                canonical_paths_sha256(payload["excluded_paths"]),
            )
            self.assertTrue(payload["does_not_open_images"])


if __name__ == "__main__":
    unittest.main()
