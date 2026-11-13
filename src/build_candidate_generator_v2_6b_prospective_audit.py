from __future__ import annotations

"""Build a score-blind, path-disjoint prospective audit for selected v2.6a.

The builder reads only frozen metadata, descriptors, pHashes and candidate
artifacts. It does not decode images, assign visual labels, score duplicates,
construct families/splits, or train a classifier.
"""

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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
from analyze_candidate_generator_v2_6a_bounded_grid import same_leaf_mask
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES
from build_candidate_generator_v2_5_prospective_audit import (
    canonical_paths_sha256,
    filter_eligible_keys,
    flip_aware_distance,
    load_group_phashes,
    phash_radius_keys,
    select_disjoint_keys,
    stable_subseed,
    subtract_keys,
)
from candidate_generator_v2_3_runtime import (
    atomic_csv,
    atomic_json,
    decode_pair_keys,
)
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_6b_prospective_audit.json"
RAW_MULTICROP = "raw_multicrop_flip_mean"

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
    "v2_6b_candidate",
    "candidate_reason",
    "selected_extension_channels",
    "selected_extension_top_k",
    "selected_extension_best_rank",
    "all_channel_best_rank",
    "raw_multicrop_best_rank",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Immutable v2.3 artifact directory")
    parser.add_argument("--selection-dir", required=True, help="New selection output directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--phash-block-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {path}")
        return [dict(row) for row in reader]


def load_config(path: Path) -> tuple[dict[str, object], str]:
    hash_path = path.with_suffix(".sha256")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("V2.6b config hash file has an invalid format.")
    actual = file_sha256(path)
    if fields[0].lower() != actual:
        raise ValueError(f"V2.6b config hash mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_v2_6b_prospective_pair_selection":
        raise ValueError("V2.6b config is not in the frozen pre-selection state.")
    expected_builder = str(payload["integrity_dependencies"]["builder_sha256"])
    if file_sha256(Path(__file__).resolve()) != expected_builder:
        raise ValueError("V2.6b builder hash differs from the frozen config.")
    return payload, actual


def load_excluded_paths(config: Mapping[str, object]) -> set[str]:
    exclusions = config["path_exclusions"]
    base_path = ROOT / str(exclusions["base_manifest_path"])
    prior_path = ROOT / str(exclusions["prior_review_manifest_path"])
    if file_sha256(base_path) != str(exclusions["base_manifest_sha256"]):
        raise ValueError("Base exclusion manifest hash mismatch.")
    if file_sha256(prior_path) != str(exclusions["prior_review_manifest_sha256"]):
        raise ValueError("Prior review manifest hash mismatch.")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    paths = {str(value).replace("\\", "/") for value in base["excluded_paths"]}
    prior_rows = read_csv(prior_path)
    prior_paths = {
        row[field].replace("\\", "/")
        for row in prior_rows
        for field in ("left_path", "right_path")
    }
    if len(prior_rows) != int(exclusions["prior_pair_count"]):
        raise ValueError("Prior review pair count mismatch.")
    if len(prior_paths) != int(exclusions["prior_unique_path_count"]):
        raise ValueError("Prior review paths are not unique as frozen.")
    if paths & prior_paths:
        raise ValueError("Prior review paths unexpectedly overlap the base exclusions.")
    paths.update(prior_paths)
    if len(paths) != int(exclusions["combined_excluded_path_count"]):
        raise ValueError("Combined excluded path count mismatch.")
    if canonical_paths_sha256(paths) != str(exclusions["combined_paths_sha256"]):
        raise ValueError("Combined excluded path hash mismatch.")
    return paths


def selected_ranked_pairs(
    channel_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    channels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.empty(0, dtype=np.uint64)
    ranks = np.empty(0, dtype=np.uint16)
    for channel in channels:
        if channel not in channel_pairs:
            raise ValueError(f"Unknown selected channel: {channel}")
        channel_keys, channel_ranks = channel_pairs[channel]
        keys, ranks = merge_ranked_pairs(keys, ranks, channel_keys, channel_ranks)
    return keys, ranks


def rank_window_same_leaf(
    keys: np.ndarray,
    ranks: np.ndarray,
    leaf_by_index: Sequence[str],
    minimum: int,
    maximum: int,
) -> np.ndarray:
    pair_keys = np.asarray(keys, dtype=np.uint64)
    pair_ranks = np.asarray(ranks, dtype=np.uint16)
    mask = (
        (pair_ranks >= minimum)
        & (pair_ranks <= maximum)
        & same_leaf_mask(pair_keys, leaf_by_index)
    )
    return pair_keys[mask]


def pair_rank(keys: np.ndarray, ranks: np.ndarray, key: np.uint64, lower_bound: int) -> int:
    position = int(np.searchsorted(keys, key))
    if position < len(keys) and keys[position] == key:
        return int(ranks[position])
    return lower_bound


def random_same_leaf_outside_keys(
    eligible_indices_by_leaf: Mapping[str, Sequence[int]],
    forbidden_keys: np.ndarray,
    count: int,
    seed: int,
    label: str,
    used_indices: set[int],
    maximum_attempts: int = 1_000_000,
) -> list[np.uint64]:
    forbidden = {int(value) for value in np.asarray(forbidden_keys, dtype=np.uint64)}
    available = {
        leaf: [int(value) for value in indices if int(value) not in used_indices]
        for leaf, indices in eligible_indices_by_leaf.items()
    }
    leaves = sorted(leaf for leaf, indices in available.items() if len(indices) >= 2)
    rng = random.Random(stable_subseed(seed, label))
    selected: list[np.uint64] = []
    selected_set: set[int] = set()
    attempts = 0
    while len(selected) < count and leaves and attempts < maximum_attempts:
        attempts += 1
        leaf = rng.choice(leaves)
        first, second = rng.sample(available[leaf], 2)
        low, high = sorted((first, second))
        key = np.uint64((low << 32) | high)
        integer = int(key)
        if integer in forbidden or integer in selected_set:
            continue
        selected.append(key)
        selected_set.add(integer)
        used_indices.update((low, high))
        for values in available.values():
            if low in values:
                values.remove(low)
            if high in values:
                values.remove(high)
        leaves = sorted(name for name, indices in available.items() if len(indices) >= 2)
    return selected


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    if args.block_size < 1 or args.phash_block_size < 1:
        raise ValueError("Block sizes must be positive.")
    output_dir = Path(args.output_dir).resolve()
    selection_dir = Path(args.selection_dir).resolve()
    if selection_dir.exists():
        raise FileExistsError(f"Selection output already exists: {selection_dir}")
    if output_dir == selection_dir or output_dir in selection_dir.parents:
        raise ValueError("Selection output must be outside immutable v2.3 artifacts.")
    config, config_hash = load_config(Path(args.config).resolve())
    for relative, expected_hash in config["integrity_dependencies"]["tracked_file_sha256"].items():
        if file_sha256(ROOT / str(relative)) != str(expected_hash):
            raise ValueError(f"Tracked dependency hash mismatch: {relative}")
    excluded_paths = load_excluded_paths(config)

    expected = config["expected_full_state"]
    candidate_rule = config["candidate_rule"]
    sampling = config["sampling"]
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
    all_paths = {str(row["relative_path"]) for row in rows}
    missing = sorted(excluded_paths - all_paths)
    if missing:
        raise ValueError(f"Excluded paths missing from full index: {missing[:5]}")
    eligible = np.asarray(
        [str(row["relative_path"]) not in excluded_paths for row in rows], dtype=bool
    )
    if int(np.count_nonzero(~eligible)) != len(excluded_paths):
        raise ValueError("Excluded path count does not match full-index membership.")
    if int(np.count_nonzero(eligible)) != int(config["path_exclusions"]["expected_eligible_images"]):
        raise ValueError("Eligible image count differs from the frozen exclusion contract.")
    if (
        int(expected["selected_profile_full_pairs"]) / len(rows)
        > float(expected["maximum_pairs_per_image"])
    ):
        raise ValueError("Selected profile exceeds the frozen capacity gate.")
    leaf_by_index = [""] * len(rows)
    for row in rows:
        leaf_by_index[int(row["image_index"])] = str(row["leaf"])

    target_groups = [str(value) for value in candidate_rule["extension_groups"]]
    if target_groups != ["kidney_ct", "lymphoma_histology"]:
        raise ValueError("Unexpected v2.6b target groups.")
    selected_rules = candidate_rule["selected_extension_by_group"]
    device = choose_device(args.device)
    group_payloads: dict[str, dict[str, object]] = {}
    verified_artifacts = 1
    for group in target_groups:
        print(f"Preparing v2.6b prospective universe: {group}", flush=True)
        group_rows = sorted(
            (row for row in rows if row["source_group"] == group),
            key=lambda row: int(row["image_index"]),
        )
        global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
        local_by_global = {int(value): offset for offset, value in enumerate(global_indices)}
        rule = selected_rules[group]
        channels = [str(value) for value in rule["channels"]]
        top_k = int(rule["top_k"])

        descriptor_paths: dict[str, Path] = {}
        phash_paths: dict[str, Path] = {}
        for leaf in sorted({str(row["leaf"]) for row in group_rows}):
            descriptor_relative = f"stages/descriptors/{leaf}.npz"
            phash_relative = f"stages/phash/{leaf}.npz"
            descriptor_paths[leaf] = verify_registered_artifact(
                output_dir, descriptor_relative, artifacts[descriptor_relative]
            )
            phash_paths[leaf] = verify_registered_artifact(
                output_dir, phash_relative, artifacts[phash_relative]
            )
        verified_artifacts += len(descriptor_paths) + len(phash_paths)

        channel_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        combined_keys = np.empty(0, dtype=np.uint64)
        combined_ranks = np.empty(0, dtype=np.uint16)
        for channel in DESCRIPTOR_NAMES:
            descriptors = load_group_channel(group_rows, channel, descriptor_paths)
            channel_keys, channel_ranks = ranked_channel_pairs(
                descriptors, global_indices, 40, args.block_size, device
            )
            channel_pairs[channel] = (channel_keys, channel_ranks)
            combined_keys, combined_ranks = merge_ranked_pairs(
                combined_keys, combined_ranks, channel_keys, channel_ranks
            )
            del descriptors

        robust_relatives = sorted(
            relative for relative, info in artifacts.items()
            if info["stage"] == "robust_retrieval"
            and info["metadata"]["source_group"] == group
        )
        stored_robust15, robust_verified = load_pair_union(output_dir, artifacts, robust_relatives)
        verified_artifacts += robust_verified
        if not np.array_equal(combined_keys[combined_ranks <= 15], stored_robust15):
            raise ValueError(f"Frozen robust top-15 mismatch: {group}")

        original, flipped = load_group_phashes(group_rows, phash_paths)
        if bool(rule["phash_enabled"]):
            phash_relatives = sorted(
                relative for relative, info in artifacts.items()
                if info["stage"] == "phash_retrieval"
                and info["metadata"]["source_group"] == group
            )
            stored_phash12, phash_verified = load_pair_union(
                output_dir, artifacts, phash_relatives
            )
            verified_artifacts += phash_verified
            phash16 = phash_radius_keys(
                original, flipped, global_indices, 16, args.phash_block_size
            )
            if not np.all(np.isin(stored_phash12, phash16, assume_unique=True)):
                raise ValueError(f"Stored pHash<=12 is not contained in pHash<=16: {group}")
        else:
            stored_phash12 = np.empty(0, dtype=np.uint64)
            phash16 = np.empty(0, dtype=np.uint64)

        selected_keys, selected_ranks = selected_ranked_pairs(channel_pairs, channels)
        selected_extension = rank_window_same_leaf(
            selected_keys, selected_ranks, leaf_by_index, 16, top_k
        )
        baseline = np.union1d(stored_robust15, stored_phash12)
        additions = subtract_keys(selected_extension, baseline)
        final_union = np.union1d(baseline, additions)
        expected_group = expected["selected_profile_by_group"][group]
        if len(additions) != int(expected_group["new_same_leaf_pairs"]):
            raise ValueError(f"Selected-profile addition count mismatch: {group}")
        if len(final_union) != int(expected_group["proposed_group_union_pairs"]):
            raise ValueError(f"Selected-profile union count mismatch: {group}")

        eligible_additions = filter_eligible_keys(additions, eligible)
        boundary_min = top_k + 1
        boundary_max = int(rule["outside_boundary_max_rank"])
        selected_boundary = rank_window_same_leaf(
            selected_keys, selected_ranks, leaf_by_index, boundary_min, boundary_max
        )
        selected_boundary = filter_eligible_keys(
            subtract_keys(selected_boundary, final_union), eligible
        )
        if group == "lymphoma_histology":
            alternate_channels = [name for name in DESCRIPTOR_NAMES if name != RAW_MULTICROP]
            alternate_keys, alternate_ranks = selected_ranked_pairs(
                channel_pairs, alternate_channels
            )
            alternate = rank_window_same_leaf(
                alternate_keys, alternate_ranks, leaf_by_index, 16, top_k
            )
            alternate = filter_eligible_keys(subtract_keys(alternate, final_union), eligible)
        else:
            alternate = np.empty(0, dtype=np.uint64)

        eligible_by_leaf: dict[str, list[int]] = {}
        for row in group_rows:
            index = int(row["image_index"])
            if eligible[index]:
                eligible_by_leaf.setdefault(str(row["leaf"]), []).append(index)
        random_forbidden = np.union1d(np.union1d(combined_keys, phash16), final_union)
        group_payloads[group] = {
            "rows": group_rows,
            "global_indices": global_indices,
            "local_by_global": local_by_global,
            "channels": channels,
            "top_k": top_k,
            "channel_pairs": channel_pairs,
            "combined_keys": combined_keys,
            "combined_ranks": combined_ranks,
            "selected_keys": selected_keys,
            "selected_ranks": selected_ranks,
            "stored_phash12": stored_phash12,
            "original": original,
            "flipped": flipped,
            "final_union": final_union,
            "candidate_new_extension": eligible_additions,
            "outside_selected_boundary": selected_boundary,
            "outside_alternate_channel": alternate,
            "random_forbidden": random_forbidden,
            "eligible_by_leaf": eligible_by_leaf,
        }

    seed = int(sampling["seed"])
    used_indices: set[int] = set()
    selected: list[tuple[str, str, np.uint64]] = []

    def choose(group: str, stratum: str, pool_name: str, count: int) -> None:
        chosen = select_disjoint_keys(
            np.asarray(group_payloads[group][pool_name], dtype=np.uint64),
            count,
            seed,
            f"{group}:{stratum}",
            used_indices,
        )
        selected.extend((stratum, group, key) for key in chosen)

    # Outside evidence receives endpoint priority.
    for group in target_groups:
        choose(
            group,
            "outside_selected_boundary",
            "outside_selected_boundary",
            int(sampling["outside_selected_boundary_per_group"]),
        )
    choose(
        "lymphoma_histology",
        "outside_alternate_channel",
        "outside_alternate_channel",
        int(sampling["outside_alternate_channel_lymphoma"]),
    )
    for group in target_groups:
        payload = group_payloads[group]
        chosen = random_same_leaf_outside_keys(
            payload["eligible_by_leaf"],
            payload["random_forbidden"],
            int(sampling["outside_random_same_leaf_per_group"]),
            seed,
            f"{group}:outside_random_same_leaf",
            used_indices,
        )
        selected.extend(("outside_random_same_leaf", group, key) for key in chosen)
    for group in target_groups:
        choose(
            group,
            "candidate_new_extension",
            "candidate_new_extension",
            int(sampling["candidate_new_extension_per_group"]),
        )

    pair_keys = [int(key) for _, _, key in selected]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("Selected v2.6b pairs are not unique.")
    if len(used_indices) != 2 * len(selected):
        raise ValueError("A selected image appears in more than one v2.6b pair.")
    random.Random(stable_subseed(seed, "pair_id_order")).shuffle(selected)

    private_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for ordinal, (stratum, group, key) in enumerate(selected, start=1):
        payload = group_payloads[group]
        left_values, right_values = decode_pair_keys(np.asarray([key], dtype=np.uint64))
        left, right = int(left_values[0]), int(right_values[0])
        left_local = payload["local_by_global"][left]
        right_local = payload["local_by_global"][right]
        distance = flip_aware_distance(
            left_local,
            right_local,
            np.asarray(payload["original"], dtype=np.uint64),
            np.asarray(payload["flipped"], dtype=np.uint64),
        )
        final_union = np.asarray(payload["final_union"], dtype=np.uint64)
        union_position = int(np.searchsorted(final_union, key))
        candidate = union_position < len(final_union) and final_union[union_position] == key
        if stratum == "candidate_new_extension":
            reason = "selected_same_leaf_extension"
        elif candidate:
            reason = "preserved_v2_5_baseline"
        else:
            reason = "outside_selected_rule"
        if stratum.startswith("outside_") and candidate:
            raise ValueError(f"Outside stratum unexpectedly entered the candidate rule: {stratum}")
        if stratum == "candidate_new_extension" and not candidate:
            raise ValueError("Candidate-new stratum is outside the selected rule.")
        selected_rank = pair_rank(
            payload["selected_keys"], payload["selected_ranks"], key, 41
        )
        all_rank = pair_rank(
            payload["combined_keys"], payload["combined_ranks"], key, 41
        )
        raw_keys, raw_ranks = payload["channel_pairs"][RAW_MULTICROP]
        raw_rank = pair_rank(raw_keys, raw_ranks, key, 41)
        pair_id = f"{sampling['pair_id_prefix']}{ordinal:06d}"
        left_row, right_row = rows[left], rows[right]
        private_rows.append({
            "pair_id": pair_id,
            "selection_stratum": stratum,
            "source_group": group,
            "left_image_index": left,
            "right_image_index": right,
            "left_leaf": left_row["leaf"],
            "right_leaf": right_row["leaf"],
            "left_path": left_row["relative_path"],
            "right_path": right_row["relative_path"],
            "v2_6b_candidate": "yes" if candidate else "no",
            "candidate_reason": reason,
            "selected_extension_channels": "+".join(payload["channels"]),
            "selected_extension_top_k": payload["top_k"],
            "selected_extension_best_rank": selected_rank,
            "all_channel_best_rank": all_rank,
            "raw_multicrop_best_rank": raw_rank,
            "phash_enabled": "yes" if bool(selected_rules[group]["phash_enabled"]) else "no",
            "phash_distance": distance,
            "phash_radius12": "yes" if distance <= 12 else "no",
        })
        review_rows.append({
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
        })

    counts_by_stratum = Counter(str(row["selection_stratum"]) for row in private_rows)
    target_counts = {
        "outside_selected_boundary": int(sampling["outside_selected_boundary_per_group"]) * 2,
        "outside_alternate_channel": int(sampling["outside_alternate_channel_lymphoma"]),
        "outside_random_same_leaf": int(sampling["outside_random_same_leaf_per_group"]) * 2,
        "candidate_new_extension": int(sampling["candidate_new_extension_per_group"]) * 2,
    }
    if len(private_rows) != int(sampling["total_pair_count"]):
        raise ValueError("Selection did not reach the frozen total pair count.")
    if any(counts_by_stratum.get(name, 0) != target for name, target in target_counts.items()):
        raise ValueError("At least one frozen stratum target was not reached.")

    selection_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-6b-prospective-", dir=selection_dir.parent) as temporary:
        staging = Path(temporary) / selection_dir.name
        staging.mkdir()
        private_path = staging / "selection_private.csv"
        review_path = staging / "review_manifest.csv"
        atomic_csv(private_path, PRIVATE_FIELDS, private_rows)
        atomic_csv(review_path, REVIEW_FIELDS, review_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "v2_6b_prospective_selection_frozen_images_not_opened",
            "experiment_type": "score_blind_path_disjoint_selected_profile_audit",
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
                "selected_profile": candidate_rule["selected_profile"],
                "full_candidate_pairs": int(expected["selected_profile_full_pairs"]),
                "pairs_per_image": int(expected["selected_profile_full_pairs"]) / len(rows),
                "by_source_group": expected["selected_profile_by_group"],
            },
            "selection": {
                "selected_pairs": len(private_rows),
                "unique_images": len(used_indices),
                "blank_review_verdicts": sum(not row["visual_verdict"] for row in review_rows),
                "target_by_stratum": target_counts,
                "actual_by_stratum": dict(sorted(counts_by_stratum.items())),
                "shortfall_by_stratum": {
                    name: target - counts_by_stratum.get(name, 0)
                    for name, target in target_counts.items()
                },
                "by_source_group": dict(sorted(Counter(row["source_group"] for row in private_rows).items())),
            },
            "integrity": {
                "builder_sha256": file_sha256(Path(__file__).resolve()),
                "stage_manifest_sha256": file_sha256(manifest_path),
                "combined_excluded_paths_sha256": canonical_paths_sha256(excluded_paths),
                "selection_private_sha256": file_sha256(private_path),
                "review_manifest_sha256": file_sha256(review_path),
            },
            "decision": (
                "SELECTION FROZEN: inspect report/hashes only; commit the private and blank "
                "review manifests before opening any selected image."
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
        print(f"ERROR: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
