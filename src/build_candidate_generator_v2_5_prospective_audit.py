from __future__ import annotations

"""Build the frozen score-blind prospective audit selection for v2.5.

The builder reads only hash-registered indexes, descriptors, pHashes and candidate
artifacts. It never decodes image pixels, assigns visual labels, scores duplicates,
constructs families/splits, or trains a classifier.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from analyze_candidate_generator_v2_4_rank_crowding import (
    EXPECTED_STAGE_COUNTS,
    choose_device,
    load_group_channel,
    load_pair_union,
    merge_ranked_pairs,
    ranked_channel_pairs,
    read_index,
    verify_registered_artifact,
)
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES
from candidate_generator_v2_3_runtime import (
    atomic_csv,
    atomic_json,
    decode_pair_keys,
    encode_pair_keys,
    phash_block_candidate_keys,
)
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_5_prospective_audit.json"

PRIVATE_FIELDS = (
    "pair_id",
    "selection_stratum",
    "source_group",
    "left_image_index",
    "right_image_index",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "v2_5_candidate",
    "robust_acceptance_k",
    "robust_best_rank",
    "robust_rank_lower_bound",
    "phash_enabled",
    "phash_distance",
    "phash_radius12",
)
REVIEW_FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the predeclared v2.5 score-blind prospective audit selection "
            "from immutable v2.3 full-retrieval artifacts."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Immutable v2.3 artifact directory")
    parser.add_argument("--selection-dir", required=True, help="New output directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--phash-block-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def stable_subseed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def canonical_paths_sha256(paths: Iterable[str]) -> str:
    encoded = "\n".join(sorted(set(paths))).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path) -> tuple[dict[str, object], str]:
    hash_path = path.with_suffix(".sha256")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("V2.5 config hash file has an invalid format.")
    actual = file_sha256(path)
    if fields[0].lower() != actual:
        raise ValueError(f"V2.5 config hash mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_v2_5_prospective_pair_selection":
        raise ValueError("V2.5 config is not in the frozen pre-selection state.")
    expected_builder = str(payload["integrity_dependencies"]["builder_sha256"])  # type: ignore[index]
    if file_sha256(Path(__file__).resolve()) != expected_builder:
        raise ValueError("V2.5 builder hash differs from the frozen config.")
    return payload, actual


def load_excluded_paths(path: Path, expected_hash: str) -> set[str]:
    if file_sha256(path) != expected_hash:
        raise ValueError("V2.5 exclusion-manifest hash mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_v2_5_prospective_pair_selection":
        raise ValueError("V2.5 exclusion manifest is not frozen.")
    paths = {str(value).replace("\\", "/") for value in payload["excluded_paths"]}
    if len(paths) != int(payload["excluded_path_count"]):
        raise ValueError("V2.5 exclusion manifest contains duplicate paths.")
    if canonical_paths_sha256(paths) != payload["excluded_paths_sha256"]:
        raise ValueError("V2.5 exclusion path-list hash mismatch.")
    return paths


def sorted_contains(values: np.ndarray, key: np.uint64) -> bool:
    position = int(np.searchsorted(values, key))
    return position < len(values) and values[position] == key


def filter_eligible_keys(keys: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    values = np.asarray(keys, dtype=np.uint64)
    left, right = decode_pair_keys(values)
    return values[eligible[left] & eligible[right]]


def rank_range_keys(
    keys: np.ndarray,
    ranks: np.ndarray,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    pair_keys = np.asarray(keys, dtype=np.uint64)
    pair_ranks = np.asarray(ranks, dtype=np.uint16)
    return pair_keys[(pair_ranks >= minimum) & (pair_ranks <= maximum)]


def subtract_keys(keys: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    return np.setdiff1d(
        np.asarray(keys, dtype=np.uint64),
        np.asarray(excluded, dtype=np.uint64),
        assume_unique=True,
    )


def select_disjoint_keys(
    keys: np.ndarray,
    count: int,
    seed: int,
    label: str,
    used_indices: set[int],
) -> list[np.uint64]:
    ordered = np.asarray(keys, dtype=np.uint64)
    order = list(range(len(ordered)))
    random.Random(stable_subseed(seed, label)).shuffle(order)
    selected: list[np.uint64] = []
    for position in order:
        key = ordered[position]
        left, right = decode_pair_keys(np.asarray([key], dtype=np.uint64))
        first, second = int(left[0]), int(right[0])
        if first in used_indices or second in used_indices:
            continue
        selected.append(key)
        used_indices.update((first, second))
        if len(selected) == count:
            break
    return selected


def random_outside_keys(
    eligible_indices: Sequence[int],
    forbidden_keys: np.ndarray,
    count: int,
    seed: int,
    label: str,
    used_indices: set[int],
    maximum_attempts: int = 1_000_000,
) -> list[np.uint64]:
    available = [int(value) for value in eligible_indices if int(value) not in used_indices]
    rng = random.Random(stable_subseed(seed, label))
    selected: list[np.uint64] = []
    selected_set: set[int] = set()
    attempts = 0
    while len(selected) < count and len(available) >= 2 and attempts < maximum_attempts:
        attempts += 1
        first, second = rng.sample(available, 2)
        left, right = sorted((first, second))
        key = encode_pair_keys(np.asarray([left]), np.asarray([right]))[0]
        integer_key = int(key)
        if integer_key in selected_set or sorted_contains(forbidden_keys, key):
            continue
        selected.append(key)
        selected_set.add(integer_key)
        used_indices.update((left, right))
        available = [value for value in available if value not in (left, right)]
    return selected


def load_group_phashes(
    group_rows: Sequence[Mapping[str, object]],
    phash_paths: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
    position = {int(index): offset for offset, index in enumerate(global_indices)}
    original = np.empty(len(group_rows), dtype=np.uint64)
    flipped = np.empty(len(group_rows), dtype=np.uint64)
    for leaf in sorted({str(row["leaf"]) for row in group_rows}):
        leaf_rows = sorted(
            (row for row in group_rows if row["leaf"] == leaf),
            key=lambda row: int(row["selection_rank_within_leaf"]),
        )
        with np.load(phash_paths[leaf], allow_pickle=False) as loaded:
            first = np.asarray(loaded["original"], dtype=np.uint64)
            second = np.asarray(loaded["horizontal_flip"], dtype=np.uint64)
        if first.shape != (len(leaf_rows),) or second.shape != first.shape:
            raise ValueError(f"pHash shape mismatch: {leaf}")
        for offset, row in enumerate(leaf_rows):
            target = position[int(row["image_index"])]
            original[target] = first[offset]
            flipped[target] = second[offset]
    return original, flipped


def phash_radius_keys(
    original: np.ndarray,
    flipped: np.ndarray,
    global_indices: np.ndarray,
    radius: int,
    block_size: int,
) -> np.ndarray:
    parts = [
        phash_block_candidate_keys(
            original,
            flipped,
            global_indices,
            start,
            min(start + block_size, len(global_indices)),
            radius,
        )
        for start in range(0, len(global_indices), block_size)
    ]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)


def flip_aware_distance(
    left_local: int,
    right_local: int,
    original: np.ndarray,
    flipped: np.ndarray,
) -> int:
    values = (
        int(original[left_local] ^ original[right_local]).bit_count(),
        int(original[left_local] ^ flipped[right_local]).bit_count(),
        int(flipped[left_local] ^ original[right_local]).bit_count(),
        int(flipped[left_local] ^ flipped[right_local]).bit_count(),
    )
    return min(values)


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    if args.block_size < 1 or args.phash_block_size < 1:
        raise ValueError("Block sizes must be positive.")
    output_dir = Path(args.output_dir).resolve()
    selection_dir = Path(args.selection_dir).resolve()
    config_path = Path(args.config).resolve()
    if selection_dir.exists():
        raise FileExistsError(f"Selection output already exists: {selection_dir}")
    if is_inside(selection_dir, output_dir):
        raise ValueError("Selection output must be outside immutable v2.3 artifacts.")
    config, config_hash = load_config(config_path)
    rule = dict(config["candidate_rule"])
    sampling = dict(config["sampling"])
    expected = dict(config["expected_full_state"])
    dependencies = dict(config["integrity_dependencies"])
    exclusion_path = ROOT / str(dependencies["exclusion_manifest_path"])
    excluded_paths = load_excluded_paths(
        exclusion_path, str(dependencies["exclusion_manifest_sha256"])
    )
    for relative, expected_hash in dict(dependencies["tracked_evidence_sha256"]).items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise ValueError(f"Tracked evidence hash mismatch: {relative}")

    manifest_path = output_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = dict(sorted(Counter(str(info["stage"]) for info in artifacts.values()).items()))
    if stage_counts != dict(sorted(EXPECTED_STAGE_COUNTS.items())):
        raise ValueError(f"Unexpected stage inventory: {stage_counts}")
    if len(artifacts) != int(expected["registered_artifacts"]):
        raise ValueError("Registered artifact count mismatch.")
    if manifest["contract_sha256"] != expected["run_contract_sha256"]:
        raise ValueError("Run-contract hash mismatch.")

    index_relative = "stages/image_index.csv"
    rows = read_index(verify_registered_artifact(output_dir, index_relative, artifacts[index_relative]))
    if len(rows) != int(expected["image_count"]):
        raise ValueError("Full image count mismatch.")
    by_path = {str(row["relative_path"]): row for row in rows}
    missing_exclusions = sorted(excluded_paths - set(by_path))
    if missing_exclusions:
        raise ValueError(f"Excluded paths missing from full index: {missing_exclusions[:5]}")
    eligible = np.asarray(
        [str(row["relative_path"]) not in excluded_paths for row in rows], dtype=bool
    )
    if int(np.count_nonzero(~eligible)) != len(excluded_paths):
        raise ValueError("Excluded path count does not match full-index membership.")

    descriptor_paths: dict[str, Path] = {}
    phash_paths: dict[str, Path] = {}
    for leaf in sorted({str(row["leaf"]) for row in rows}):
        descriptor_relative = f"stages/descriptors/{leaf}.npz"
        phash_relative = f"stages/phash/{leaf}.npz"
        descriptor_paths[leaf] = verify_registered_artifact(
            output_dir, descriptor_relative, artifacts[descriptor_relative]
        )
        phash_paths[leaf] = verify_registered_artifact(
            output_dir, phash_relative, artifacts[phash_relative]
        )

    robust_k_by_group = {str(k): int(v) for k, v in dict(rule["robust_top_k_by_group"]).items()}
    phash_enabled = set(str(value) for value in rule["phash_radius12_source_groups"])
    phash_disabled = set(str(value) for value in rule["phash_disabled_source_groups"])
    all_groups = sorted({str(row["source_group"]) for row in rows})
    if set(robust_k_by_group) != set(all_groups):
        raise ValueError("Robust K policy does not cover exactly the full source-group set.")
    if phash_enabled & phash_disabled or phash_enabled | phash_disabled != set(all_groups):
        raise ValueError("pHash enabled/disabled policy does not cover all source groups.")

    device = choose_device(args.device)
    group_payloads: dict[str, dict[str, object]] = {}
    verified_artifacts = 1 + len(descriptor_paths) + len(phash_paths)
    v2_4_group_sum = 0
    v2_5_group_sum = 0
    for group in all_groups:
        print(f"Preparing v2.5 selection universe: {group}", flush=True)
        group_rows = sorted(
            (row for row in rows if row["source_group"] == group),
            key=lambda row: int(row["image_index"]),
        )
        global_indices = np.asarray(
            [int(row["image_index"]) for row in group_rows], dtype=np.int64
        )
        local_by_global = {int(value): offset for offset, value in enumerate(global_indices)}
        acceptance_k = robust_k_by_group[group]
        maximum_k = acceptance_k * 2
        combined_keys = np.empty(0, dtype=np.uint64)
        combined_ranks = np.empty(0, dtype=np.uint16)
        for channel in DESCRIPTOR_NAMES:
            descriptors = load_group_channel(group_rows, channel, descriptor_paths)
            channel_keys, channel_ranks = ranked_channel_pairs(
                descriptors, global_indices, maximum_k, args.block_size, device
            )
            combined_keys, combined_ranks = merge_ranked_pairs(
                combined_keys, combined_ranks, channel_keys, channel_ranks
            )
            del descriptors, channel_keys, channel_ranks

        robust_relatives = sorted(
            relative
            for relative, info in artifacts.items()
            if info["stage"] == "robust_retrieval"
            and info["metadata"]["source_group"] == group  # type: ignore[index]
        )
        stored_robust15, verified_robust = load_pair_union(
            output_dir, artifacts, robust_relatives
        )
        verified_artifacts += verified_robust
        computed_robust15 = combined_keys[combined_ranks <= 15]
        if not np.array_equal(computed_robust15, stored_robust15):
            raise ValueError(f"Frozen robust top-15 mismatch: {group}")

        original, flipped = load_group_phashes(group_rows, phash_paths)
        if group in phash_enabled:
            phash_relatives = sorted(
                relative
                for relative, info in artifacts.items()
                if info["stage"] == "phash_retrieval"
                and info["metadata"]["source_group"] == group  # type: ignore[index]
            )
            stored_phash12, verified_phash = load_pair_union(
                output_dir, artifacts, phash_relatives
            )
            verified_artifacts += verified_phash
            phash16 = phash_radius_keys(
                original, flipped, global_indices, 16, args.phash_block_size
            )
            if not np.all(np.isin(stored_phash12, phash16, assume_unique=True)):
                raise ValueError(f"Stored pHash<=12 is not contained in pHash<=16: {group}")
        else:
            stored_phash12 = np.empty(0, dtype=np.uint64)
            phash16 = np.empty(0, dtype=np.uint64)

        robust15 = combined_keys[combined_ranks <= 15]
        robust_policy = combined_keys[combined_ranks <= acceptance_k]
        v2_4_union = np.union1d(robust15, stored_phash12)
        v2_5_union = np.union1d(robust_policy, stored_phash12)
        v2_4_group_sum += len(v2_4_union)
        v2_5_group_sum += len(v2_5_union)

        eligible_rank_keys = filter_eligible_keys(combined_keys, eligible)
        rank_positions = np.searchsorted(combined_keys, eligible_rank_keys)
        eligible_ranks = combined_ranks[rank_positions]
        eligible_phash12 = filter_eligible_keys(stored_phash12, eligible)
        eligible_phash16 = filter_eligible_keys(phash16, eligible)
        edge_minimum = acceptance_k - 9
        candidate_edge = rank_range_keys(
            eligible_rank_keys, eligible_ranks, edge_minimum, acceptance_k
        )
        outside_robust = rank_range_keys(
            eligible_rank_keys, eligible_ranks, acceptance_k + 1, maximum_k
        )
        if group in phash_enabled:
            candidate_edge = subtract_keys(candidate_edge, eligible_phash12)
            outside_robust = subtract_keys(outside_robust, eligible_phash12)
            outside_phash = subtract_keys(eligible_phash16, eligible_phash12)
            outside_phash = subtract_keys(outside_phash, robust_policy)
        else:
            outside_phash = np.empty(0, dtype=np.uint64)
        if group == "cervix_pap_smear":
            cervix_new = rank_range_keys(eligible_rank_keys, eligible_ranks, 16, 30)
            cervix_new = subtract_keys(cervix_new, eligible_phash12)
        else:
            cervix_new = np.empty(0, dtype=np.uint64)

        group_payloads[group] = {
            "rows": group_rows,
            "global_indices": global_indices,
            "local_by_global": local_by_global,
            "acceptance_k": acceptance_k,
            "maximum_k": maximum_k,
            "combined_keys": combined_keys,
            "combined_ranks": combined_ranks,
            "phash12": stored_phash12,
            "phash16": phash16,
            "original": original,
            "flipped": flipped,
            "candidate_edge": candidate_edge,
            "outside_robust": outside_robust,
            "outside_phash": outside_phash,
            "cervix_new": cervix_new,
            "robust15_pairs": len(robust15),
            "robust_policy_pairs": len(robust_policy),
            "phash12_pairs": len(stored_phash12),
            "v2_4_union_pairs": len(v2_4_union),
            "v2_5_union_pairs": len(v2_5_union),
        }

    cervix = group_payloads["cervix_pap_smear"]
    if int(cervix["v2_4_union_pairs"]) != int(expected["cervix_v2_4_union_pairs"]):
        raise ValueError("Cervix v2.4 union baseline mismatch.")
    if int(cervix["v2_5_union_pairs"]) != int(expected["cervix_v2_5_union_pairs"]):
        raise ValueError("Cervix v2.5 union projection mismatch.")
    exact_outside_channel_union = int(expected["v2_4_full_pairs"]) - v2_4_group_sum
    if exact_outside_channel_union < 0:
        raise ValueError("Group unions exceed the verified v2.4 full count.")
    v2_5_full_pairs = v2_5_group_sum + exact_outside_channel_union
    if v2_5_full_pairs != int(expected["v2_5_full_pairs"]):
        raise ValueError("V2.5 full candidate projection mismatch.")
    if v2_5_full_pairs / len(rows) > float(expected["maximum_pairs_per_image"]):
        raise ValueError("V2.5 candidate load exceeds the frozen capacity gate.")

    seed = int(sampling["seed"])
    used_indices: set[int] = set()
    selected: list[tuple[str, str, np.uint64]] = []

    def choose(group: str, stratum: str, pool_name: str, target: int) -> None:
        chosen = select_disjoint_keys(
            np.asarray(group_payloads[group][pool_name], dtype=np.uint64),
            target,
            seed,
            f"{group}:{stratum}",
            used_indices,
        )
        selected.extend((stratum, group, key) for key in chosen)

    # Outside challenge evidence has priority when endpoints compete.
    for group in all_groups:
        choose(group, "outside_robust_boundary", "outside_robust", int(sampling["outside_robust_per_group"]))
    for group in sorted(phash_enabled):
        choose(group, "outside_phash_13_16", "outside_phash", int(sampling["outside_phash_max_per_enabled_group"]))
    for group in all_groups:
        payload = group_payloads[group]
        forbidden = np.union1d(
            np.asarray(payload["combined_keys"], dtype=np.uint64),
            np.asarray(payload["phash16"], dtype=np.uint64),
        )
        group_eligible = [
            int(value)
            for value in np.asarray(payload["global_indices"], dtype=np.int64)
            if eligible[int(value)]
        ]
        chosen = random_outside_keys(
            group_eligible,
            forbidden,
            int(sampling["outside_random_per_group"]),
            seed,
            f"{group}:outside_random",
            used_indices,
        )
        selected.extend(("outside_random", group, key) for key in chosen)
    choose(
        "cervix_pap_smear",
        "candidate_new_cervix_rank16_30",
        "cervix_new",
        int(sampling["candidate_new_cervix_rank16_30"]),
    )
    for group in all_groups:
        choose(group, "candidate_robust_edge", "candidate_edge", int(sampling["candidate_robust_edge_per_group"]))

    pair_keys = [int(key) for _, _, key in selected]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("Selected v2.5 pairs are not unique.")
    if len(used_indices) != 2 * len(selected):
        raise ValueError("A selected image appears in more than one v2.5 pair.")
    if len(selected) > int(sampling["maximum_pair_count"]):
        raise ValueError("Selected pair count exceeds the frozen maximum.")

    random.Random(stable_subseed(seed, "pair_id_order")).shuffle(selected)
    private_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for ordinal, (stratum, group, key) in enumerate(selected, start=1):
        left_values, right_values = decode_pair_keys(np.asarray([key], dtype=np.uint64))
        left, right = int(left_values[0]), int(right_values[0])
        payload = group_payloads[group]
        local_by_global = payload["local_by_global"]
        left_local = local_by_global[left]  # type: ignore[index]
        right_local = local_by_global[right]  # type: ignore[index]
        distance = flip_aware_distance(
            left_local,
            right_local,
            np.asarray(payload["original"], dtype=np.uint64),
            np.asarray(payload["flipped"], dtype=np.uint64),
        )
        combined_keys = np.asarray(payload["combined_keys"], dtype=np.uint64)
        combined_ranks = np.asarray(payload["combined_ranks"], dtype=np.uint16)
        rank_position = int(np.searchsorted(combined_keys, key))
        if rank_position < len(combined_keys) and combined_keys[rank_position] == key:
            best_rank: object = int(combined_ranks[rank_position])
            lower_bound: object = ""
        else:
            best_rank = ""
            lower_bound = int(payload["maximum_k"]) + 1
        acceptance_k = int(payload["acceptance_k"])
        phash_is_enabled = group in phash_enabled
        candidate = (
            (best_rank != "" and int(best_rank) <= acceptance_k)
            or (phash_is_enabled and distance <= 12)
        )
        pair_id = f"{sampling['pair_id_prefix']}{ordinal:06d}"
        left_row, right_row = rows[left], rows[right]
        private_rows.append(
            {
                "pair_id": pair_id,
                "selection_stratum": stratum,
                "source_group": group,
                "left_image_index": left,
                "right_image_index": right,
                "left_leaf": left_row["leaf"],
                "right_leaf": right_row["leaf"],
                "left_path": left_row["relative_path"],
                "right_path": right_row["relative_path"],
                "v2_5_candidate": "yes" if candidate else "no",
                "robust_acceptance_k": acceptance_k,
                "robust_best_rank": best_rank,
                "robust_rank_lower_bound": lower_bound,
                "phash_enabled": "yes" if phash_is_enabled else "no",
                "phash_distance": distance,
                "phash_radius12": "yes" if distance <= 12 else "no",
            }
        )
        review_rows.append(
            {
                "pair_id": pair_id,
                "source_group": group,
                "left_leaf": left_row["leaf"],
                "right_leaf": right_row["leaf"],
                "left_path": left_row["relative_path"],
                "right_path": right_row["relative_path"],
                "visual_verdict": "",
                "confidence": "",
                "reviewer": "",
                "review_date": "",
                "review_notes": "",
            }
        )

    counts_by_stratum = Counter(str(row["selection_stratum"]) for row in private_rows)
    counts_by_group = Counter(str(row["source_group"]) for row in private_rows)
    target_counts = {
        "outside_robust_boundary": int(sampling["outside_robust_per_group"]) * len(all_groups),
        "outside_phash_13_16": int(sampling["outside_phash_max_per_enabled_group"]) * len(phash_enabled),
        "outside_random": int(sampling["outside_random_per_group"]) * len(all_groups),
        "candidate_new_cervix_rank16_30": int(sampling["candidate_new_cervix_rank16_30"]),
        "candidate_robust_edge": int(sampling["candidate_robust_edge_per_group"]) * len(all_groups),
    }
    selection_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-5-prospective-", dir=selection_dir.parent) as temporary:
        staging = Path(temporary) / selection_dir.name
        staging.mkdir()
        private_path = staging / "selection_private.csv"
        review_path = staging / "review_manifest.csv"
        atomic_csv(private_path, PRIVATE_FIELDS, private_rows)
        atomic_csv(review_path, REVIEW_FIELDS, review_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "v2_5_prospective_selection_frozen_images_not_opened",
            "experiment_type": "score_blind_full_index_prospective_natural_audit_selection",
            "does_not_open_images": True,
            "does_not_score_duplicates": True,
            "does_not_authorize_family_split_or_training": True,
            "config_sha256": config_hash,
            "input": {
                "run_contract_sha256": manifest["contract_sha256"],
                "registered_artifacts": len(artifacts),
                "verified_artifacts": verified_artifacts,
                "image_count": len(rows),
                "excluded_paths": len(excluded_paths),
                "eligible_images": int(np.count_nonzero(eligible)),
                "device": str(device),
            },
            "candidate_policy": {
                "v2_4_full_pairs": int(expected["v2_4_full_pairs"]),
                "v2_5_full_pairs": v2_5_full_pairs,
                "v2_5_pairs_per_image": v2_5_full_pairs / len(rows),
                "exact_pairs_outside_retrieval_channel_unions": exact_outside_channel_union,
                "by_source_group": {
                    group: {
                        key: value
                        for key, value in group_payloads[group].items()
                        if key
                        in {
                            "acceptance_k",
                            "robust15_pairs",
                            "robust_policy_pairs",
                            "phash12_pairs",
                            "v2_4_union_pairs",
                            "v2_5_union_pairs",
                        }
                    }
                    for group in all_groups
                },
            },
            "selection": {
                "selected_pairs": len(private_rows),
                "maximum_pairs": int(sampling["maximum_pair_count"]),
                "unique_pair_ids": len({str(row["pair_id"]) for row in private_rows}),
                "unique_images": len(used_indices),
                "blank_review_verdicts": sum(
                    not str(row["visual_verdict"]) for row in review_rows
                ),
                "target_by_stratum": target_counts,
                "actual_by_stratum": dict(sorted(counts_by_stratum.items())),
                "shortfall_by_stratum": {
                    name: target - counts_by_stratum.get(name, 0)
                    for name, target in target_counts.items()
                },
                "by_source_group": dict(sorted(counts_by_group.items())),
            },
            "integrity": {
                "builder_sha256": file_sha256(Path(__file__).resolve()),
                "exclusion_manifest_sha256": file_sha256(exclusion_path),
                "stage_manifest_sha256": file_sha256(manifest_path),
                "selection_private_sha256": file_sha256(private_path),
                "review_manifest_sha256": file_sha256(review_path),
            },
            "decision": (
                "SELECTION FROZEN: inspect only the report and hashes; commit/push the "
                "private and blank review manifests before opening any selected image."
            ),
        }
        atomic_json(staging / "selection_report.json", report)
        shutil.move(str(staging), str(selection_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Selection: {selection_dir}")
    return selection_dir, report


def main() -> None:
    try:
        run(parse_args())
    except Exception as error:
        print(f"ERROR: {error}")
        raise


if __name__ == "__main__":
    main()
