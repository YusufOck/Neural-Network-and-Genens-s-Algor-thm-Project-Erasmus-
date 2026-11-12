from __future__ import annotations

"""Evaluate bounded same-leaf/channel profiles from frozen v2.3 artifacts.

This is a read-only development diagnostic after the rejected v2.6 top-30
four-channel proposal.  It preserves the complete v2.5 candidate union and
compares a small, predeclared set of Kidney/Lymphoma extensions.  It does not
score duplicates, construct families, split data, or train a classifier.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analyze_candidate_generator_v2_4_rank_crowding import (
    choose_device,
    is_inside,
    load_group_channel,
    load_pair_union,
    merge_ranked_pairs,
    ranked_channel_pairs,
    read_index,
    verify_registered_artifact,
)
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES
from candidate_generator_v2_3_runtime import atomic_json, decode_pair_keys, encode_pair_keys
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_6a_bounded_grid.json"
EXPECTED_STAGE_COUNTS = {
    "descriptor_generation": 26,
    "index": 1,
    "phash_generation": 26,
    "phash_retrieval": 1022,
    "robust_retrieval": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--block-size", type=int, default=256)
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
        raise ValueError("Bounded-grid config hash file has an invalid format.")
    actual = file_sha256(path)
    if actual != fields[0].lower():
        raise ValueError(f"Bounded-grid config hash mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "predeclared_before_v2_6a_bounded_grid_diagnostic":
        raise ValueError("Bounded-grid config is not predeclared.")
    return payload, actual


def verify_dependencies(config: Mapping[str, object]) -> None:
    dependencies = config["integrity_dependencies"]
    if file_sha256(Path(__file__).resolve()) != str(dependencies["analyzer_sha256"]):
        raise ValueError("Analyzer hash differs from the predeclared config.")
    for relative, expected in dependencies["tracked_file_sha256"].items():
        if file_sha256(ROOT / str(relative)) != str(expected):
            raise ValueError(f"Tracked dependency hash mismatch: {relative}")


def same_leaf_mask(pair_keys: np.ndarray, leaf_by_index: Sequence[str]) -> np.ndarray:
    keys = np.asarray(pair_keys, dtype=np.uint64)
    left, right = decode_pair_keys(keys)
    leaves = np.asarray(leaf_by_index, dtype=object)
    if len(keys) and int(max(left.max(), right.max())) >= len(leaves):
        raise ValueError("Pair key references an image outside the leaf index.")
    return leaves[left.astype(np.int64)] == leaves[right.astype(np.int64)]


def merge_selected_channels(
    channel_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    channels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.empty(0, dtype=np.uint64)
    ranks = np.empty(0, dtype=np.uint16)
    for channel in channels:
        if channel not in channel_pairs:
            raise ValueError(f"Unknown descriptor channel in profile: {channel}")
        channel_keys, channel_ranks = channel_pairs[channel]
        keys, ranks = merge_ranked_pairs(keys, ranks, channel_keys, channel_ranks)
    return keys, ranks


def profile_group_union(
    stored_robust_top15: np.ndarray,
    stored_phash: np.ndarray,
    channel_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    leaf_by_index: Sequence[str],
    top_k: int,
    channels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not 15 < top_k <= 30:
        raise ValueError("Every bounded profile top_k must be in 16..30.")
    keys, ranks = merge_selected_channels(channel_pairs, channels)
    extension_mask = (
        (ranks > 15)
        & (ranks <= top_k)
        & same_leaf_mask(keys, leaf_by_index)
    )
    baseline = np.union1d(stored_robust_top15, stored_phash)
    additions = np.setdiff1d(keys[extension_mask], baseline, assume_unique=False)
    return np.union1d(baseline, additions), additions


def select_profile(profile_reports: Sequence[Mapping[str, object]]) -> str | None:
    """Choose the broadest passing bounded profile, then config order."""

    passing = [
        row for row in profile_reports
        if bool(row["development_same2d_reference_recall"])
        and bool(row["candidate_load_passed"])
    ]
    if not passing:
        return None
    best = max(int(row["full_candidate_pairs"]) for row in passing)
    return next(str(row["profile_id"]) for row in passing if int(row["full_candidate_pairs"]) == best)


def reference_keys(
    rows: Sequence[Mapping[str, str]],
    path_rows: Sequence[Mapping[str, str]],
    index_by_path: Mapping[str, int],
) -> tuple[list[str], np.ndarray]:
    selected = [row for row in rows if row["final_reference"] == "same_source"]
    pair_ids = [row["pair_id"] for row in selected]
    path_by_pair = {row["pair_id"]: row for row in path_rows}
    keys: list[np.uint64] = []
    for row in selected:
        source = path_by_pair[row["pair_id"]]
        left = index_by_path[source["left_path"]]
        right = index_by_path[source["right_path"]]
        low, high = sorted((left, right))
        keys.append(encode_pair_keys(np.asarray([low]), np.asarray([high]))[0])
    return pair_ids, np.asarray(keys, dtype=np.uint64)


def analyze(
    output_dir: Path,
    config_path: Path,
    block_size: int,
    requested_device: str,
) -> dict[str, object]:
    config, config_hash = load_config(config_path)
    verify_dependencies(config)
    expected = config["expected_full_artifact_state"]
    profiles = list(config["bounded_profiles"])
    target_groups = tuple(str(group) for group in config["target_groups"])
    if target_groups != ("kidney_ct", "lymphoma_histology"):
        raise ValueError("Target groups differ from the frozen bounded diagnostic.")
    if not profiles:
        raise ValueError("At least one bounded profile is required.")
    profile_ids = [str(profile["profile_id"]) for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Profile IDs must be unique.")

    manifest = json.loads((output_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = dict(sorted(Counter(str(info["stage"]) for info in artifacts.values()).items()))
    if stage_counts != EXPECTED_STAGE_COUNTS:
        raise ValueError(f"Unexpected stage inventory: {stage_counts}")
    if len(artifacts) != int(expected["registered_artifacts"]):
        raise ValueError("Registered artifact count mismatch.")
    if manifest["contract_sha256"] != expected["run_contract_sha256"]:
        raise ValueError("Run contract hash mismatch.")

    index_relative = "stages/image_index.csv"
    index_path = verify_registered_artifact(output_dir, index_relative, artifacts[index_relative])
    all_rows = read_index(index_path)
    if len(all_rows) != int(expected["image_count"]):
        raise ValueError("Image count mismatch.")
    leaf_by_index = [""] * len(all_rows)
    index_by_path: dict[str, int] = {}
    for row in all_rows:
        index = int(row["image_index"])
        leaf_by_index[index] = str(row["leaf"])
        index_by_path[str(row["relative_path"])] = index
    if not all(leaf_by_index):
        raise ValueError("Image index is not contiguous.")

    references_path = ROOT / str(config["same2d_reference"]["path"])
    reference_rows = read_csv(references_path)
    path_rows = read_csv(ROOT / str(config["same2d_reference"]["path_source"]))
    pair_ids, same2d_keys = reference_keys(reference_rows, path_rows, index_by_path)
    if pair_ids != list(config["same2d_reference"]["same_source_pair_ids"]):
        raise ValueError("Same-2D reference Pair IDs differ from the frozen config.")

    device = choose_device(requested_device)
    verified_artifacts = 1
    group_data: dict[str, dict[str, object]] = {}
    top15_reproduced = True

    for group in target_groups:
        state = config["target_group_state"][group]
        group_rows = sorted(
            (row for row in all_rows if row["source_group"] == group),
            key=lambda row: int(row["image_index"]),
        )
        global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
        if len(group_rows) != int(state["image_count"]):
            raise ValueError(f"{group}: image count mismatch")

        robust_relatives = sorted(
            relative for relative, info in artifacts.items()
            if info["stage"] == "robust_retrieval"
            and info["metadata"]["source_group"] == group
        )
        stored_robust, robust_verified = load_pair_union(output_dir, artifacts, robust_relatives)
        verified_artifacts += robust_verified
        if len(stored_robust) != int(state["robust_top15_pairs"]):
            raise ValueError(f"{group}: stored robust top-15 count mismatch")

        phash = np.empty(0, dtype=np.uint64)
        if bool(state["phash_enabled"]):
            phash_relatives = sorted(
                relative for relative, info in artifacts.items()
                if info["stage"] == "phash_retrieval"
                and info["metadata"]["source_group"] == group
            )
            phash, phash_verified = load_pair_union(output_dir, artifacts, phash_relatives)
            verified_artifacts += phash_verified
            if len(phash) != int(state["phash_pairs"]):
                raise ValueError(f"{group}: stored pHash count mismatch")

        descriptor_paths: dict[str, Path] = {}
        for leaf in sorted({str(row["leaf"]) for row in group_rows}):
            relative = f"stages/descriptors/{leaf}.npz"
            descriptor_paths[leaf] = verify_registered_artifact(
                output_dir, relative, artifacts[relative]
            )
        verified_artifacts += len(descriptor_paths)

        channel_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        combined_keys = np.empty(0, dtype=np.uint64)
        combined_ranks = np.empty(0, dtype=np.uint16)
        for channel in DESCRIPTOR_NAMES:
            print(f"{group}: loading {channel}", flush=True)
            descriptors = load_group_channel(group_rows, channel, descriptor_paths)
            channel_keys, channel_ranks = ranked_channel_pairs(
                descriptors, global_indices, 30, block_size, device
            )
            channel_pairs[channel] = (channel_keys, channel_ranks)
            combined_keys, combined_ranks = merge_ranked_pairs(
                combined_keys, combined_ranks, channel_keys, channel_ranks
            )
            del descriptors

        recomputed_top15 = combined_keys[combined_ranks <= 15]
        reproduced = np.array_equal(recomputed_top15, stored_robust)
        top15_reproduced = top15_reproduced and reproduced
        if not reproduced:
            raise ValueError(f"{group}: recomputed top-15 differs from frozen artifact union")
        current_union = np.union1d(stored_robust, phash)
        if len(current_union) != int(state["current_v2_5_union_pairs"]):
            raise ValueError(f"{group}: current v2.5 union count mismatch")
        group_data[group] = {
            "stored_robust": stored_robust,
            "phash": phash,
            "channel_pairs": channel_pairs,
            "current_union_pairs": len(current_union),
        }

    profile_reports: list[dict[str, object]] = []
    capacity = int(expected["image_count"]) * float(expected["maximum_pairs_per_image"])
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        additions_total = 0
        proposal_parts: list[np.ndarray] = []
        group_reports: dict[str, object] = {}
        for group in target_groups:
            rule = profile["groups"][group]
            channels = [str(channel) for channel in rule["channels"]]
            if not channels or any(channel not in DESCRIPTOR_NAMES for channel in channels):
                raise ValueError(f"{profile_id}/{group}: invalid channels")
            data = group_data[group]
            proposal, additions = profile_group_union(
                data["stored_robust"], data["phash"], data["channel_pairs"],
                leaf_by_index, int(rule["top_k"]), channels,
            )
            additions_total += len(additions)
            proposal_parts.append(proposal)
            group_reports[group] = {
                "top_k": int(rule["top_k"]),
                "channels": channels,
                "current_v2_5_union_pairs": int(data["current_union_pairs"]),
                "new_same_leaf_pairs": int(len(additions)),
                "proposed_group_union_pairs": int(len(proposal)),
            }

        full_pairs = int(expected["v2_5_full_pairs"]) + additions_total
        proposal_affected = np.unique(np.concatenate(proposal_parts))
        hits = np.isin(same2d_keys, proposal_affected)
        missed_ids = [pair_id for pair_id, hit in zip(pair_ids, hits) if not hit]
        profile_reports.append({
            "profile_id": profile_id,
            "description": str(profile["description"]),
            "by_source_group": group_reports,
            "new_same_leaf_pairs": int(additions_total),
            "full_candidate_pairs": int(full_pairs),
            "pairs_per_image": full_pairs / int(expected["image_count"]),
            "development_same2d_hits": int(np.count_nonzero(hits)),
            "development_same2d_missed_pair_ids": missed_ids,
            "development_same2d_reference_recall": not missed_ids,
            "candidate_load_passed": full_pairs <= capacity,
        })

    selected_profile = select_profile(profile_reports)
    diagnostic_passed = selected_profile is not None and top15_reproduced
    return {
        "schema_version": 1,
        "status": "candidate_generator_v2_6a_bounded_grid_diagnostic_complete",
        "experiment_type": "read_only_predeclared_bounded_same_leaf_channel_grid",
        "does_not_modify_v2_3_artifacts": True,
        "development_evidence_only": True,
        "does_not_authorize_scoring_family_split_or_training": True,
        "config": {
            "path_name": config_path.name,
            "sha256": config_hash,
            "analyzer_sha256": file_sha256(Path(__file__).resolve()),
        },
        "input": {
            "registered_artifact_count": len(artifacts),
            "verified_artifacts_in_this_run": verified_artifacts,
            "image_count": len(all_rows),
            "device": str(device),
            "block_size": block_size,
            "v2_6_rejected_report_sha256": config["rejected_v2_6_report"]["sha256"],
        },
        "selection_policy": config["selection_policy"],
        "development_same2d_pair_ids": pair_ids,
        "profiles": profile_reports,
        "selected_development_profile": selected_profile,
        "acceptance_gates": {
            "top15_baseline_reproduced": top15_reproduced,
            "at_least_one_profile_recovers_4_of_4": any(
                bool(row["development_same2d_reference_recall"]) for row in profile_reports
            ),
            "at_least_one_recall_profile_within_capacity": selected_profile is not None,
            "all_diagnostic_gates_passed": diagnostic_passed,
        },
        "limitation": (
            "The four same-2D pairs are development evidence that defined this bounded grid. "
            "A selected profile is only a prospective-audit hypothesis, not an accepted rule."
        ),
        "decision": (
            "DIAGNOSTIC PASS: freeze the selected profile and prepare a new path-disjoint "
            "score-blind prospective audit before candidate acceptance."
            if diagnostic_passed
            else "DIAGNOSTIC FAIL: no bounded profile supports prospective audit."
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    if is_inside(report_path, output_dir):
        raise ValueError("Report must be outside the immutable v2.3 output directory.")
    report = analyze(
        output_dir=output_dir,
        config_path=Path(args.config).resolve(),
        block_size=args.block_size,
        requested_device=args.device,
    )
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
