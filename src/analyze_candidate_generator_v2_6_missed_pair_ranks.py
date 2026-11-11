from __future__ import annotations

"""Read-only full-index rank diagnostic for the 16 v2.5 outside positives.

This diagnostic measures exact directed ranks in the four frozen descriptor
channels and bounded candidate-load projections for the three affected source
groups. It does not select v2.6, score duplicates, build families, create a
split, or train a classifier.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from analyze_candidate_generator_v2_4_rank_crowding import (
    choose_device,
    is_inside,
    load_group_channel,
    load_pair_union,
    merge_ranked_pairs,
    ranked_channel_pairs,
    read_index,
    target_channel_result,
    verify_registered_artifact,
)
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES
from candidate_generator_v2_3_runtime import atomic_json
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_6_missed_pair_rank_diagnostic.json"
EXPECTED_STAGE_COUNTS = {
    "descriptor_generation": 26,
    "index": 1,
    "phash_generation": 26,
    "phash_retrieval": 1022,
    "robust_retrieval": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact full-index robust ranks for the 16 frozen v2.5 "
            "outside positives and bounded candidate-load projections."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic config not found: {path}")
    hash_path = path.with_suffix(".sha256")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("Diagnostic config hash file has an invalid format.")
    actual = file_sha256(path)
    if actual != fields[0].lower():
        raise ValueError(f"Diagnostic config hash mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "predeclared_before_v2_6_missed_pair_rank_diagnostic":
        raise ValueError("Diagnostic config is not in the predeclared state.")
    return payload, actual


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header not found: {path}")
        return [dict(row) for row in reader]


def verify_dependency_hashes(config: Mapping[str, object]) -> None:
    dependencies = config["integrity_dependencies"]
    expected_analyzer = str(dependencies["analyzer_sha256"])
    if file_sha256(Path(__file__).resolve()) != expected_analyzer:
        raise ValueError("Diagnostic analyzer hash differs from the predeclared config.")
    for relative, expected in dependencies["tracked_file_sha256"].items():
        path = ROOT / str(relative)
        if file_sha256(path) != str(expected):
            raise ValueError(f"Tracked dependency hash mismatch: {relative}")


def validate_targets(
    config_targets: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, str]],
) -> dict[str, list[dict[str, object]]]:
    frozen = {
        row["pair_id"]: row
        for row in audit_rows
        if row.get("audit_region") == "outside"
        and row.get("visual_verdict") == "same_source_candidate"
    }
    configured_ids = [str(target["pair_id"]) for target in config_targets]
    if len(configured_ids) != len(set(configured_ids)):
        raise ValueError("Configured target Pair IDs are not unique.")
    if set(configured_ids) != set(frozen):
        raise ValueError("Configured targets do not equal the frozen outside positives.")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for target in config_targets:
        pair_id = str(target["pair_id"])
        row = frozen[pair_id]
        for field in ("source_group", "left_path", "right_path", "selection_stratum"):
            if str(target[field]) != row[field]:
                raise ValueError(f"{pair_id}: config/audit {field} mismatch")
        if int(target["phash_distance"]) != int(row["phash_distance"]):
            raise ValueError(f"{pair_id}: config/audit pHash distance mismatch")
        audit_rank = row.get("robust_best_rank", "").strip()
        audit_lower = row.get("robust_rank_lower_bound", "").strip()
        expected_rank = target.get("audit_robust_best_rank")
        expected_lower = target.get("audit_robust_rank_lower_bound")
        if (None if not audit_rank else int(audit_rank)) != expected_rank:
            raise ValueError(f"{pair_id}: config/audit robust rank mismatch")
        if (None if not audit_lower else int(audit_lower)) != expected_lower:
            raise ValueError(f"{pair_id}: config/audit robust lower bound mismatch")
        grouped[str(target["source_group"])].append(dict(target))
    return dict(sorted(grouped.items()))


def group_projection(
    robust_keys: np.ndarray,
    robust_ranks: np.ndarray,
    phash_keys: np.ndarray,
    top_k_grid: Sequence[int],
    current_group_union: int,
    full_v2_5_pairs: int,
    image_count: int,
    required_k: int,
) -> list[dict[str, object]]:
    projections: list[dict[str, object]] = []
    for top_k in top_k_grid:
        robust = robust_keys[robust_ranks <= top_k]
        union = np.union1d(robust, phash_keys)
        full_pairs = full_v2_5_pairs - current_group_union + len(union)
        projections.append(
            {
                "top_k": int(top_k),
                "all_group_targets_recovered": bool(top_k >= required_k),
                "target_group_robust_pairs": int(len(robust)),
                "target_group_union_pairs": int(len(union)),
                "full_pairs_if_only_this_group_changed": int(full_pairs),
                "full_pairs_per_image": full_pairs / image_count,
                "passes_25_pairs_per_image_capacity": full_pairs / image_count <= 25.0,
            }
        )
    return projections


def first_grid_covering(projections: Sequence[Mapping[str, object]]) -> int | None:
    for projection in projections:
        if bool(projection["all_group_targets_recovered"]):
            return int(projection["top_k"])
    return None


def analyze(
    output_dir: Path,
    config_path: Path,
    block_size: int,
    requested_device: str,
) -> dict[str, object]:
    config, config_hash = load_config(config_path)
    verify_dependency_hashes(config)
    top_k_grid = [int(value) for value in config["top_k_grid"]]
    if top_k_grid != sorted(set(top_k_grid)) or top_k_grid[0] != 15:
        raise ValueError("top_k_grid must be unique, sorted and start at 15.")
    max_k = top_k_grid[-1]

    audit_path = ROOT / str(config["audit_source"]["unblinded_pair_results"])
    targets_by_group = validate_targets(config["target_pairs"], read_csv(audit_path))
    expected_groups = set(config["target_group_state"])
    if set(targets_by_group) != expected_groups:
        raise ValueError("Target groups differ from the predeclared group state.")

    manifest_path = output_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = dict(sorted(Counter(str(info["stage"]) for info in artifacts.values()).items()))
    expected_full = config["expected_full_artifact_state"]
    if stage_counts != dict(sorted(EXPECTED_STAGE_COUNTS.items())):
        raise ValueError(f"Unexpected stage inventory: {stage_counts}")
    if len(artifacts) != int(expected_full["registered_artifacts"]):
        raise ValueError("Registered artifact count differs from the predeclared state.")
    if manifest["contract_sha256"] != expected_full["run_contract_sha256"]:
        raise ValueError("Run-contract hash differs from the predeclared state.")

    index_relative = "stages/image_index.csv"
    index_path = verify_registered_artifact(output_dir, index_relative, artifacts[index_relative])
    all_rows = read_index(index_path)
    if len(all_rows) != int(expected_full["image_count"]):
        raise ValueError("Full image count differs from the predeclared state.")
    device = choose_device(requested_device)
    verified_artifacts = 1
    group_reports: dict[str, object] = {}
    group_ranked: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for group, targets in targets_by_group.items():
        state = config["target_group_state"][group]
        group_rows = sorted(
            (row for row in all_rows if row["source_group"] == group),
            key=lambda row: int(row["image_index"]),
        )
        if len(group_rows) != int(state["image_count"]):
            raise ValueError(f"{group}: image count mismatch")
        global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
        by_path = {str(row["relative_path"]): offset for offset, row in enumerate(group_rows)}
        if any(str(target["left_path"]) not in by_path or str(target["right_path"]) not in by_path for target in targets):
            raise ValueError(f"{group}: target path absent from full index")

        robust_relatives = sorted(
            relative
            for relative, info in artifacts.items()
            if info["stage"] == "robust_retrieval"
            and info["metadata"]["source_group"] == group
        )
        if len(robust_relatives) != int(state["robust_artifacts"]):
            raise ValueError(f"{group}: robust artifact count mismatch")
        stored_robust, robust_verified = load_pair_union(output_dir, artifacts, robust_relatives)
        verified_artifacts += robust_verified
        if len(stored_robust) != int(state["robust_top15_pairs"]):
            raise ValueError(f"{group}: stored robust top-15 count mismatch")

        phash_keys = np.empty(0, dtype=np.uint64)
        phash_verified = 0
        if bool(state["phash_enabled"]):
            phash_relatives = sorted(
                relative
                for relative, info in artifacts.items()
                if info["stage"] == "phash_retrieval"
                and info["metadata"]["source_group"] == group
            )
            if len(phash_relatives) != int(state["phash_artifacts"]):
                raise ValueError(f"{group}: pHash artifact count mismatch")
            phash_keys, phash_verified = load_pair_union(output_dir, artifacts, phash_relatives)
            verified_artifacts += phash_verified
            if len(phash_keys) != int(state["phash_pairs"]):
                raise ValueError(f"{group}: stored pHash count mismatch")

        descriptor_paths: dict[str, Path] = {}
        for leaf in sorted({str(row["leaf"]) for row in group_rows}):
            relative = f"stages/descriptors/{leaf}.npz"
            descriptor_paths[leaf] = verify_registered_artifact(output_dir, relative, artifacts[relative])
        if len(descriptor_paths) != int(state["descriptor_artifacts"]):
            raise ValueError(f"{group}: descriptor artifact count mismatch")
        verified_artifacts += len(descriptor_paths)

        combined_keys = np.empty(0, dtype=np.uint64)
        combined_ranks = np.empty(0, dtype=np.uint16)
        pair_reports: dict[str, dict[str, object]] = {
            str(target["pair_id"]): {
                "pair_id": target["pair_id"],
                "left_path": target["left_path"],
                "right_path": target["right_path"],
                "selection_stratum": target["selection_stratum"],
                "phash_distance": target["phash_distance"],
                "audit_robust_best_rank": target.get("audit_robust_best_rank"),
                "audit_robust_rank_lower_bound": target.get("audit_robust_rank_lower_bound"),
                "by_descriptor_channel": {},
            }
            for target in targets
        }
        for channel in DESCRIPTOR_NAMES:
            print(f"{group}: loading {channel}", flush=True)
            descriptors = load_group_channel(group_rows, channel, descriptor_paths)
            norms = np.linalg.norm(descriptors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
                raise ValueError(f"{group}/{channel}: descriptor normalization mismatch")
            for target in targets:
                pair_id = str(target["pair_id"])
                left_local = by_path[str(target["left_path"])]
                right_local = by_path[str(target["right_path"])]
                pair_reports[pair_id]["by_descriptor_channel"][channel] = target_channel_result(
                    descriptors, left_local, right_local
                )
            channel_keys, channel_ranks = ranked_channel_pairs(
                descriptors, global_indices, max_k, block_size, device
            )
            combined_keys, combined_ranks = merge_ranked_pairs(
                combined_keys, combined_ranks, channel_keys, channel_ranks
            )
            del descriptors, channel_keys, channel_ranks

        recomputed_top15 = combined_keys[combined_ranks <= 15]
        if not np.array_equal(recomputed_top15, stored_robust):
            raise ValueError(f"{group}: recomputed top-15 differs from frozen robust union")
        current_union = np.union1d(stored_robust, phash_keys)
        if len(current_union) != int(state["current_v2_5_union_pairs"]):
            raise ValueError(f"{group}: current v2.5 group union mismatch")

        required_ranks: list[int] = []
        for target in targets:
            pair_id = str(target["pair_id"])
            channels = pair_reports[pair_id]["by_descriptor_channel"]
            minimum = min(int(result["best_directed_rank"]) for result in channels.values())
            pair_reports[pair_id]["minimum_required_robust_top_k"] = minimum
            required_ranks.append(minimum)
            expected_rank = target.get("audit_robust_best_rank")
            lower_bound = target.get("audit_robust_rank_lower_bound")
            if expected_rank is not None and minimum != int(expected_rank):
                raise ValueError(f"{pair_id}: recomputed rank differs from frozen audit rank")
            if lower_bound is not None and minimum < int(lower_bound):
                raise ValueError(f"{pair_id}: recomputed rank violates frozen lower bound")

        required_k = max(required_ranks)
        projections = group_projection(
            combined_keys,
            combined_ranks,
            phash_keys,
            top_k_grid,
            int(state["current_v2_5_union_pairs"]),
            int(expected_full["v2_5_full_pairs"]),
            int(expected_full["image_count"]),
            required_k,
        )
        group_reports[group] = {
            "image_count": len(group_rows),
            "target_pair_count": len(targets),
            "phash_enabled": bool(state["phash_enabled"]),
            "target_pairs": [pair_reports[str(target["pair_id"])] for target in targets],
            "minimum_k_covering_all_observed_targets": required_k,
            "first_predeclared_grid_k_covering_all": first_grid_covering(projections),
            "candidate_load_projections": projections,
            "baseline_reproduction": {
                "stored_robust_top15_pairs": int(len(stored_robust)),
                "recomputed_robust_top15_pairs": int(len(recomputed_top15)),
                "stored_phash_pairs_used_by_policy": int(len(phash_keys)),
                "current_v2_5_group_union_pairs": int(len(current_union)),
                "all_matched": True,
            },
        }
        group_ranked[group] = (combined_keys, combined_ranks, phash_keys)

    combined_grid: list[dict[str, object]] = []
    current_group_total = sum(
        int(config["target_group_state"][group]["current_v2_5_union_pairs"])
        for group in targets_by_group
    )
    for top_k in top_k_grid:
        projected_group_total = 0
        recovered = True
        by_group: dict[str, int] = {}
        for group in targets_by_group:
            keys, ranks, phash = group_ranked[group]
            union_count = len(np.union1d(keys[ranks <= top_k], phash))
            projected_group_total += union_count
            by_group[group] = int(union_count)
            required = int(group_reports[group]["minimum_k_covering_all_observed_targets"])
            recovered = recovered and top_k >= required
        full_pairs = int(expected_full["v2_5_full_pairs"]) - current_group_total + projected_group_total
        combined_grid.append(
            {
                "common_top_k": top_k,
                "all_16_targets_recovered": recovered,
                "projected_union_pairs_by_group": by_group,
                "full_candidate_pairs": full_pairs,
                "full_pairs_per_image": full_pairs / int(expected_full["image_count"]),
                "passes_25_pairs_per_image_capacity": full_pairs / int(expected_full["image_count"]) <= 25.0,
            }
        )

    per_group_grid_profile: dict[str, int | None] = {
        group: group_reports[group]["first_predeclared_grid_k_covering_all"]
        for group in targets_by_group
    }
    profile_available = all(value is not None for value in per_group_grid_profile.values())
    profile_pairs: int | None = None
    if profile_available:
        projected_group_total = 0
        for group, top_k in per_group_grid_profile.items():
            keys, ranks, phash = group_ranked[group]
            projected_group_total += len(np.union1d(keys[ranks <= int(top_k)], phash))
        profile_pairs = int(expected_full["v2_5_full_pairs"]) - current_group_total + projected_group_total

    return {
        "schema_version": 1,
        "status": "candidate_generator_v2_6_missed_pair_rank_diagnostic_complete",
        "experiment_type": "read_only_predeclared_full_index_multi_pair_rank_diagnostic",
        "does_not_modify_v2_3_artifacts": True,
        "not_a_v2_6_rule": True,
        "not_a_duplicate_decision": True,
        "does_not_authorize_family_split_or_training": True,
        "config": {"path_name": config_path.name, "sha256": config_hash, "analyzer_sha256": file_sha256(Path(__file__).resolve())},
        "input": {
            "run_contract_sha256": manifest["contract_sha256"],
            "registered_artifact_count": len(artifacts),
            "stage_counts": stage_counts,
            "verified_artifacts_in_this_diagnostic": verified_artifacts,
            "image_count": len(all_rows),
            "target_pair_count": sum(len(items) for items in targets_by_group.values()),
            "target_groups": list(targets_by_group),
            "device": str(device),
            "block_size": block_size,
        },
        "by_source_group": group_reports,
        "common_k_candidate_load_projections": combined_grid,
        "diagnostic_only_first_grid_profile": {
            "top_k_by_group": per_group_grid_profile,
            "all_targets_recovered_within_grid": profile_available,
            "full_candidate_pairs": profile_pairs,
            "full_pairs_per_image": None if profile_pairs is None else profile_pairs / int(expected_full["image_count"]),
            "passes_25_pairs_per_image_capacity": None if profile_pairs is None else profile_pairs / int(expected_full["image_count"]) <= 25.0,
            "warning": "Observed-target recovery profile is diagnostic evidence, not a selected v2.6 rule.",
        },
        "decision": "DIAGNOSTIC COMPLETE: review ranks and load projections before predeclaring any v2.6 rule.",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    if is_inside(report_path, output_dir):
        raise ValueError("Diagnostic report must be outside the immutable v2.3 output directory.")
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
