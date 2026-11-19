from __future__ import annotations

import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ga_random_full_confirmation import (  # noqa: E402
    candidate_command,
    validate_candidate_summary,
    validate_config,
    validate_source_search,
)


CONFIG_PATH = ROOT / "config/ga_resnet18_full_confirmation_v1.json"


class RunGaRandomFullConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_freezes_two_candidates_and_zero_test_policy(self) -> None:
        validate_config(self.config)
        self.assertEqual(
            {row["method"] for row in self.config["candidates"]}, {"ga", "random"}
        )
        self.assertEqual(self.config["frozen_inputs"]["counts"]["test"], 0)
        self.assertFalse(
            self.config["test_policy"]["test_opened_during_confirmation"]
        )
        self.assertFalse(self.config["test_policy"]["test_loader_constructed"])
        self.assertFalse(self.config["test_policy"]["test_metrics_allowed"])

    def test_source_search_winners_match_frozen_candidates(self) -> None:
        winners = {}
        for candidate in self.config["candidates"]:
            winners[candidate["method"]] = {
                "candidate_id": candidate["candidate_id"],
                "learning_rate": candidate["learning_rate"],
                "weight_decay": candidate["weight_decay"],
                "dropout": candidate["dropout"],
                "fitness": candidate["proxy_validation_macro_f1"],
                "summary_sha256": candidate["source_summary_sha256"],
            }
        summary = {
            "status": "ga_random_search_v1_complete",
            "test_evaluated": False,
            "completed_trials": {"ga": 12, "random": 12},
            "winners": winners,
        }
        validate_source_search(summary, self.config)
        summary["winners"]["ga"]["fitness"] += 0.01
        with self.assertRaisesRegex(ValueError, "fitness"):
            validate_source_search(summary, self.config)

    def test_candidate_command_is_full_validation_only_and_resumable(self) -> None:
        args = Namespace(
            data_dir="/data",
            split_manifest="/split.csv",
            confirmation_manifest="/full_train_val.csv",
            init_checkpoint="/best_model.pth",
            workers=2,
        )
        candidate = self.config["candidates"][0]
        command = candidate_command(args, self.config, candidate, Path("candidate"))
        joined = " ".join(command)
        self.assertIn("--search-proxy-manifest", command)
        manifest_argument = command[command.index("--search-proxy-manifest") + 1]
        self.assertTrue(manifest_argument.replace("\\", "/").endswith("/full_train_val.csv"))
        self.assertIn("--experiment-type search_trial", joined)
        self.assertIn("--epochs 12", joined)
        self.assertIn("--early-stopping-patience 4", joined)
        self.assertNotIn("test_metrics.json", joined)

    def test_candidate_summary_rejects_test_or_partial_counts(self) -> None:
        candidate = self.config["candidates"][0]
        summary = {
            "status": "hyperparameter_search_trial_complete",
            "test_evaluated": False,
            "best_validation_macro_f1": 0.95,
            "best_epoch": 5,
            "epochs_completed": 8,
            "split_sizes": {"train": 91_000, "val": 19_500, "test": 0},
            "search_proxy_manifest_sha256": self.config["frozen_inputs"][
                "confirmation_manifest_sha256"
            ],
            "initialization": {
                "checkpoint_sha256": self.config["frozen_inputs"][
                    "init_checkpoint_sha256"
                ]
            },
            "hyperparameters": {
                key: candidate[key]
                for key in ("learning_rate", "weight_decay", "dropout")
            },
        }
        self.assertEqual(
            validate_candidate_summary(summary, candidate, self.config), 0.95
        )
        summary["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "test"):
            validate_candidate_summary(summary, candidate, self.config)


if __name__ == "__main__":
    unittest.main()
