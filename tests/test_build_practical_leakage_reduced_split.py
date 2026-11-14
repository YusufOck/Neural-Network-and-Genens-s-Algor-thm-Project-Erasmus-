from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_practical_leakage_reduced_split as split_builder


class PracticalLeakageReducedSplitTests(unittest.TestCase):
    def test_normalize_relative_is_portable_and_safe(self) -> None:
        self.assertEqual(
            split_builder.normalize_relative(r"Kidney Cancer\kidney_normal\a.jpg"),
            "Kidney Cancer/kidney_normal/a.jpg",
        )
        with self.assertRaises(ValueError):
            split_builder.normalize_relative("../a.jpg")
        with self.assertRaises(ValueError):
            split_builder.normalize_relative("C:/data/a.jpg")

    def test_component_assignment_is_deterministic_and_exact(self) -> None:
        components = [[f"image_{index:03d}.jpg"] for index in range(91)]
        components.extend(
            [
                ["family_a_1.jpg", "family_a_2.jpg", "family_a_3.jpg", "family_a_4.jpg"],
                ["family_b_1.jpg", "family_b_2.jpg", "family_b_3.jpg"],
                ["family_c_1.jpg", "family_c_2.jpg"],
            ]
        )
        targets = {"train": 70, "val": 15, "test": 15}
        first = split_builder.assign_leaf_components(components, targets, 42, "leaf")
        second = split_builder.assign_leaf_components(components, targets, 42, "leaf")
        self.assertEqual(first, second)
        counts = {split: list(first.values()).count(split) for split in split_builder.SPLITS}
        self.assertEqual(counts, targets)
        for component in components:
            self.assertEqual(len({first[path] for path in component}), 1)

    def test_real_review_edges_and_frozen_hashes(self) -> None:
        config_path = ROOT / "config" / "practical_leakage_reduced_split_v1.json"
        config = split_builder.read_json(config_path)
        for name, info in config["evidence"].items():
            path = ROOT / info["path"]
            if "sha256" in info:
                actual = split_builder.file_sha256(path)
                expected = info["sha256"]
            else:
                actual = split_builder.normalized_text_sha256(path)
                expected = info["normalized_text_sha256"]
            self.assertEqual(actual, expected, name)
        edges = split_builder.collect_review_edges(ROOT, config)
        self.assertEqual(len(edges), 85)
        source_counts = {}
        for sources in edges.values():
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
        self.assertEqual(source_counts["natural_confirmed_same_source"], 23)
        self.assertEqual(source_counts["v2_5_review_same_source"], 36)
        self.assertEqual(source_counts["v2_6_confirmed_same2d"], 4)
        self.assertEqual(source_counts["v2_6b_confirmed_same2d"], 24)

    def test_generated_report_passes_all_invariants(self) -> None:
        report = split_builder.read_json(
            ROOT / "research" / "splits" / "practical_leakage_reduced_split_v1_report.json"
        )
        self.assertEqual(report["input"]["images"], 130000)
        self.assertEqual(report["input"]["leaves"], 26)
        self.assertEqual(report["split_counts"], {"train": 91000, "val": 19500, "test": 19500})
        self.assertTrue(report["invariants"]["all_passed"])
        self.assertEqual(report["invariants"]["cross_split_exact_groups"], 0)
        self.assertEqual(report["invariants"]["cross_split_review_edges"], 0)


if __name__ == "__main__":
    unittest.main()
