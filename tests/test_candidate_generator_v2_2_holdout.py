from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_candidate_generator_v2_2_holdout import (
    evaluate_positive_pairs_v22,
    load_frozen_config,
)


class CandidateGeneratorV22HoldoutTests(unittest.TestCase):
    def test_evaluate_positive_pairs_reports_top15_union_recall(self) -> None:
        rows = [
            {
                "pair_id": "V22H00001",
                "left_key": "a",
                "right_key": "b",
                "leaf": "leaf",
                "source_group": "group",
                "source_id": "S1",
                "transform_name": "brightness",
            },
            {
                "pair_id": "V22H00002",
                "left_key": "a",
                "right_key": "c",
                "leaf": "leaf",
                "source_group": "group",
                "source_id": "S1",
                "transform_name": "combined_stress",
            },
        ]
        original = np.asarray([0, 0, 3], dtype=np.uint64)
        flipped = original.copy()
        metric_rows, metrics = evaluate_positive_pairs_v22(
            rows,
            {"a": 0, "b": 1, "c": 2},
            robust={(0, 2)},
            phash={(0, 1)},
            union={(0, 1), (0, 2)},
            phash_original=original,
            phash_flipped=flipped,
        )
        self.assertEqual(len(metric_rows), 2)
        self.assertEqual(metrics["robust_top15_hits"], 1)
        self.assertEqual(metrics["phash_radius8_hits"], 1)
        self.assertEqual(metrics["v2_2_union_hits"], 2)
        self.assertEqual(metrics["v2_2_union_recall"], 1.0)
        self.assertEqual(metrics["v2_2_missed_pair_ids"], [])

    def test_frozen_config_rejects_non_top15_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            hash_path = root / "config.sha256"
            payload = {
                "status": "frozen_before_second_prospective_controlled_holdout_evaluation",
                "candidate_rule": {
                    "robust_retrieval": {"top_k": 10},
                    "phash_retrieval": {"hamming_radius_inclusive": 8},
                },
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            import hashlib

            hash_path.write_text(
                hashlib.sha256(config_path.read_bytes()).hexdigest() + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "top-k"):
                load_frozen_config(config_path, hash_path)
