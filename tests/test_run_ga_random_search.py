from __future__ import annotations

import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ga_random_search import (  # noqa: E402
    candidate_key,
    in_search_space,
    make_ga_offspring,
    make_initial_ga_candidate,
    make_random_candidate,
    trial_command,
    validate_config,
)


CONFIG_PATH = ROOT / "config" / "ga_resnet18_lastblock_v1.json"


class RunGaRandomSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_equal_budget_and_test_policy_are_frozen(self) -> None:
        validate_config(self.config)
        budget = self.config["equal_budget"]
        self.assertEqual(budget["ga_trials"], 12)
        self.assertEqual(budget["random_trials"], 12)
        self.assertEqual(budget["population_size"] * budget["ga_rounds"], 12)
        self.assertFalse(self.config["test_policy"]["test_opened_during_search"])

    def test_initial_and_random_candidates_are_deterministic_unique_and_bounded(self) -> None:
        ga_keys = set()
        ga_candidates = []
        for slot in range(4):
            candidate = make_initial_ga_candidate(self.config, slot, ga_keys)
            ga_candidates.append(candidate)
            ga_keys.add(candidate_key(candidate))
            self.assertTrue(in_search_space(candidate, self.config))
        self.assertEqual(len(ga_keys), 4)
        self.assertEqual(
            ga_candidates[0], make_initial_ga_candidate(self.config, 0, set())
        )

        random_keys = set()
        for index in range(12):
            candidate = make_random_candidate(self.config, index, random_keys)
            random_keys.add(candidate_key(candidate))
            self.assertTrue(in_search_space(candidate, self.config))
        self.assertEqual(len(random_keys), 12)

    def test_offspring_is_deterministic_bounded_and_not_duplicate(self) -> None:
        archive = []
        existing = set()
        for slot in range(4):
            candidate = make_initial_ga_candidate(self.config, slot, existing)
            existing.add(candidate_key(candidate))
            archive.append(
                {
                    **candidate,
                    "fitness": 0.70 + slot * 0.01,
                    "sequence": slot + 1,
                }
            )
        first = make_ga_offspring(self.config, 1, 0, archive, existing)
        second = make_ga_offspring(self.config, 1, 0, archive, existing)
        self.assertEqual(first, second)
        self.assertTrue(in_search_space(first, self.config))
        self.assertNotIn(candidate_key(first), existing)

    def test_trial_command_is_proxy_only_and_carries_all_genes(self) -> None:
        args = Namespace(
            data_dir="/data",
            split_manifest="/split.csv",
            proxy_manifest="/proxy.csv",
            init_checkpoint="/best_model.pth",
            workers=4,
        )
        candidate = make_initial_ga_candidate(self.config, 0, set())
        command = trial_command(args, self.config, candidate, Path("trial"))
        joined = " ".join(command)
        self.assertIn("--search-proxy-manifest", command)
        self.assertIn("--search-proxy-manifest-sha256", command)
        self.assertIn("--experiment-type search_trial", joined)
        self.assertIn("--dropout", command)
        self.assertIn("--learning-rate", command)
        self.assertIn("--weight-decay", command)
        self.assertNotIn("test_metrics.json", joined)


if __name__ == "__main__":
    unittest.main()
