from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_candidate_generator_v2_1_holdout import select_holdout_records
from data_index import ImageRecord
from evaluate_candidate_generator_v2_1_holdout import evaluate_positive_pairs


class CandidateGeneratorV21HoldoutTests(unittest.TestCase):
    def test_select_holdout_records_excludes_prior_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            records = []
            for leaf in ("leaf_a", "leaf_b"):
                for index in range(8):
                    path = root / "main" / leaf / f"image_{index}.png"
                    records.append(
                        ImageRecord(path=path, main_class="main", subclass=leaf)
                    )
            prior = {"main/leaf_a/image_0.png", "main/leaf_b/image_0.png"}
            exact = {"main/leaf_a/image_1.png", "main/leaf_b/image_1.png"}
            sources, distractors = select_holdout_records(
                records,
                root,
                ["leaf_a", "leaf_b"],
                prior,
                exact,
                sources_per_leaf=2,
                distractors_per_leaf=3,
                seed=7,
            )
            selected = {
                record.path.relative_to(root).as_posix()
                for values in [*sources.values(), *distractors.values()]
                for record in values
            }
            self.assertEqual(len(selected), 10)
            self.assertFalse(selected & prior)
            self.assertFalse(selected & exact)
            self.assertTrue(all(len(values) == 2 for values in sources.values()))
            self.assertTrue(
                all(len(values) == 3 for values in distractors.values())
            )

    def test_evaluate_positive_pairs_reports_union_recall(self) -> None:
        rows = [
            {
                "pair_id": "P1",
                "left_key": "a",
                "right_key": "b",
                "leaf": "leaf",
                "source_group": "group",
                "source_id": "S1",
                "transform_name": "brightness",
            },
            {
                "pair_id": "P2",
                "left_key": "a",
                "right_key": "c",
                "leaf": "leaf",
                "source_group": "group",
                "source_id": "S1",
                "transform_name": "rotation",
            },
        ]
        original = np.asarray([0, 0, 3], dtype=np.uint64)
        flipped = original.copy()
        metric_rows, metrics = evaluate_positive_pairs(
            rows,
            {"a": 0, "b": 1, "c": 2},
            robust={(0, 1)},
            phash={(0, 2)},
            union={(0, 1), (0, 2)},
            phash_original=original,
            phash_flipped=flipped,
        )
        self.assertEqual(len(metric_rows), 2)
        self.assertEqual(metrics["robust_top10_hits"], 1)
        self.assertEqual(metrics["phash_radius8_hits"], 1)
        self.assertEqual(metrics["v2_1_union_hits"], 2)
        self.assertEqual(metrics["v2_1_union_recall"], 1.0)
        self.assertEqual(metrics["v2_1_missed_pair_ids"], [])
