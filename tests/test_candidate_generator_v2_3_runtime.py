from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_candidate_generation import (  # noqa: E402
    ImageItem,
    phash_candidate_sets,
)
from benchmark_candidate_generation_scaled import blockwise_topk_candidates  # noqa: E402
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES  # noqa: E402
from candidate_generator_v2_3_runtime import (  # noqa: E402
    ResumeManifest,
    atomic_json,
    decode_pair_keys,
    encode_pair_keys,
    exact_duplicate_pair_keys,
    merge_candidate_channels,
    phash_block_candidate_keys,
    robust_channel_candidate_keys,
    stable_topk_index_set,
    verify_exact_duplicate_coverage,
    verify_exact_duplicate_manifest,
)
from duplicate_decision_rule import file_sha256  # noqa: E402
from run_candidate_generator_v2_3_full_retrieval import (  # noqa: E402
    FULL_CONFIRMATION,
    create_or_validate_contract,
    validate_args,
)


def item(index: int, group: str = "g") -> ImageItem:
    return ImageItem(
        key=f"dataset:leaf/image_{index}.jpg",
        root_kind="dataset",
        relative_path=f"leaf/image_{index}.jpg",
        source_group=group,
        absolute_path=Path(f"image_{index}.jpg"),
    )


class CandidateGeneratorV23RuntimeTests(unittest.TestCase):
    def test_pair_key_round_trip(self) -> None:
        left = np.asarray([0, 3, 100_000], dtype=np.uint32)
        right = np.asarray([1, 9, 129_999], dtype=np.uint32)
        keys = encode_pair_keys(left, right)
        decoded_left, decoded_right = decode_pair_keys(keys)
        np.testing.assert_array_equal(decoded_left, left)
        np.testing.assert_array_equal(decoded_right, right)

    def test_stable_topk_matches_stable_argsort_with_ties(self) -> None:
        row = np.asarray([0.7, 0.9, 0.9, 0.1, 0.7], dtype=np.float32)
        expected = set(np.argsort(-row, kind="stable")[:3].tolist())
        actual = set(stable_topk_index_set(row, 3).tolist())
        self.assertEqual(actual, expected)
        self.assertIn(0, actual)
        self.assertNotIn(4, actual)

    def test_robust_channel_matches_existing_stable_retrieval(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(size=(7, 8)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        items = [item(index) for index in range(len(values))]
        existing, _ = blockwise_topk_candidates(
            items,
            {name: values for name in DESCRIPTOR_NAMES},
            block_size=3,
            topk_values=(2,),
        )
        keys = robust_channel_candidate_keys(
            values,
            np.arange(len(values)),
            top_k=2,
            block_size=3,
            device=torch.device("cpu"),
        )
        left, right = decode_pair_keys(keys)
        actual = set(zip(left.tolist(), right.tolist()))
        self.assertEqual(actual, existing[2])

    def test_blockwise_phash_matches_bruteforce_flip_aware_radius12(self) -> None:
        original = np.asarray(
            [0, 1, (1 << 12) - 1, (1 << 20) - 1, 0xFFFFFFFFFFFFFFFF],
            dtype=np.uint64,
        )
        flipped = np.asarray(
            [0, 3, (1 << 13) - 1, (1 << 19) - 1, 0xFFFFFFFFFFFFFFFE],
            dtype=np.uint64,
        )
        items = [item(index) for index in range(len(original))]
        expected = phash_candidate_sets(items, original, flipped)[12]
        parts = [
            phash_block_candidate_keys(
                original,
                flipped,
                np.arange(len(original)),
                start,
                min(start + 2, len(original)),
                radius=12,
            )
            for start in range(0, len(original), 2)
        ]
        keys = np.unique(np.concatenate(parts))
        left, right = decode_pair_keys(keys)
        actual = set(zip(left.tolist(), right.tolist()))
        self.assertEqual(actual, expected)

    def test_channel_merge_preserves_flags(self) -> None:
        robust = encode_pair_keys(np.asarray([0, 1]), np.asarray([1, 2]))
        phash = encode_pair_keys(np.asarray([1, 2]), np.asarray([2, 3]))
        union, robust_flag, phash_flag = merge_candidate_channels([robust], [phash])
        self.assertEqual(len(union), 3)
        self.assertEqual(robust_flag.tolist(), [True, True, False])
        self.assertEqual(phash_flag.tolist(), [False, True, True])

    def test_exact_duplicate_manifest_and_coverage(self) -> None:
        normalized, _ = verify_exact_duplicate_manifest(
            PROJECT_ROOT / "artifacts" / "audit" / "exact_duplicates.json"
        )
        first_group = dict(normalized["duplicate_groups"][0])
        paths = list(first_group["paths"])
        index = {path: offset for offset, path in enumerate(paths)}
        expected = exact_duplicate_pair_keys(normalized, index)
        self.assertEqual(len(expected), len(paths) * (len(paths) - 1) // 2)
        result = verify_exact_duplicate_coverage(expected, expected)
        self.assertEqual(result["missing_exact_duplicate_pairs"], 0)

    def test_resume_manifest_verifies_and_rejects_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            manifest = ResumeManifest(root, "contract")
            artifact = root / "stage.txt"
            artifact.write_text("first", encoding="utf-8")
            manifest.register("stage.txt", "test", {"seconds": 1.25})
            resumed = ResumeManifest(root, "contract")
            resumed.verify_registered()
            self.assertTrue(resumed.is_complete("stage.txt"))
            self.assertEqual(resumed.stage_seconds("test"), 1.25)
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resumed.verify_registered()

    def test_contract_resume_requires_identical_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            path, first = create_or_validate_contract(output, {"scope": "smoke"})
            self.assertEqual(file_sha256(path), file_sha256(path))
            _, second = create_or_validate_contract(output, {"scope": "smoke"})
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "differs"):
                create_or_validate_contract(output, {"scope": "full"})

    def test_full_scope_requires_explicit_confirmation(self) -> None:
        base = dict(
            scope="full",
            confirm_full_run="",
            smoke_images_per_leaf=2,
            batch_size=8,
            workers=4,
            robust_block_size=256,
            phash_block_size=128,
        )
        with self.assertRaisesRegex(ValueError, "locked"):
            validate_args(argparse.Namespace(**base))
        base["confirm_full_run"] = FULL_CONFIRMATION
        validate_args(argparse.Namespace(**base))


if __name__ == "__main__":
    unittest.main()
