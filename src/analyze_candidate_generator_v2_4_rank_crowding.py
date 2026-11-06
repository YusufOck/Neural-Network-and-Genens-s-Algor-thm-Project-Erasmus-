from __future__ import annotations

"""Read-only full-index rank-crowding diagnostic for natural pair N000123."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES
from candidate_generator_v2_3_runtime import (
    atomic_json,
    decode_pair_keys,
    encode_pair_keys,
    stable_topk_index_set,
)
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_4_rank_crowding.json"
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
            "Measure N000123 full-cervix robust ranks and exact bounded top-K "
            "candidate-load projections from existing v2.3 artifacts."
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
    if payload.get("status") != "predeclared_before_full_rank_diagnostic":
        raise ValueError("Diagnostic config is not in the predeclared state.")
    return payload, actual


def verify_registered_artifact(
    output_dir: Path,
    relative: str,
    info: Mapping[str, object],
) -> Path:
    path = output_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"Registered artifact is missing: {path}")
    if path.stat().st_size != int(info["bytes"]):
        raise ValueError(f"Registered artifact size mismatch: {relative}")
    if file_sha256(path) != str(info["sha256"]):
        raise ValueError(f"Registered artifact hash mismatch: {relative}")
    return path


def read_index(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "image_index": int(raw["image_index"]),
                    "relative_path": raw["relative_path"].replace("\\", "/"),
                    "main_class": raw["main_class"],
                    "leaf": raw["leaf"],
                    "source_group": raw["source_group"],
                    "selection_rank_within_leaf": int(raw["selection_rank_within_leaf"]),
                }
            )
    if [row["image_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Image index is not contiguous and ordered.")
    return rows


def reduce_min_rank(
    keys: np.ndarray,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_keys = np.asarray(keys, dtype=np.uint64)
    pair_ranks = np.asarray(ranks, dtype=np.uint16)
    if pair_keys.ndim != 1 or pair_ranks.shape != pair_keys.shape:
        raise ValueError("Pair keys and ranks must be aligned one-dimensional arrays.")
    if len(pair_keys) == 0:
        return pair_keys, pair_ranks
    order = np.argsort(pair_keys, kind="stable")
    ordered_keys = pair_keys[order]
    ordered_ranks = pair_ranks[order]
    starts = np.flatnonzero(
        np.concatenate((np.asarray([True]), ordered_keys[1:] != ordered_keys[:-1]))
    )
    return ordered_keys[starts], np.minimum.reduceat(ordered_ranks, starts)


def merge_ranked_pairs(
    first_keys: np.ndarray,
    first_ranks: np.ndarray,
    second_keys: np.ndarray,
    second_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return reduce_min_rank(
        np.concatenate((np.asarray(first_keys), np.asarray(second_keys))),
        np.concatenate((np.asarray(first_ranks), np.asarray(second_ranks))),
    )


def stable_directional_rank(similarities: np.ndarray, target_index: int) -> int:
    row = np.asarray(similarities)
    target = row[target_index]
    positions = np.arange(len(row))
    greater = int(np.count_nonzero(row > target))
    equal_before = int(np.count_nonzero((row == target) & (positions < target_index)))
    return greater + equal_before + 1


def ranked_channel_pairs(
    descriptors: np.ndarray,
    global_indices: np.ndarray,
    max_k: int,
    block_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(descriptors, dtype=np.float32)
    indices = np.asarray(global_indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(indices):
        raise ValueError("Descriptor/global-index shapes do not match.")
    if not 1 <= max_k < len(indices):
        raise ValueError("max_k must be between 1 and group_size-1.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    gallery = torch.from_numpy(values).to(device)
    key_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    try:
        for start in range(0, len(indices), block_size):
            stop = min(start + block_size, len(indices))
            similarities = (gallery[start:stop] @ gallery.T).float().cpu().numpy()
            offsets = np.arange(stop - start)
            similarities[offsets, start + offsets] = -np.inf
            block_keys: list[np.ndarray] = []
            for offset, local_left in enumerate(range(start, stop)):
                row = similarities[offset]
                selected = stable_topk_index_set(row, max_k)
                selected = selected[np.lexsort((selected, -row[selected]))]
                left = np.full(max_k, indices[local_left], dtype=np.int64)
                right = indices[selected]
                block_keys.append(encode_pair_keys(np.minimum(left, right), np.maximum(left, right)))
            key_parts.append(np.concatenate(block_keys))
            rank_parts.append(
                np.tile(np.arange(1, max_k + 1, dtype=np.uint16), stop - start)
            )
            print(f"Ranked robust retrieval: {stop}/{len(indices)}", flush=True)
    finally:
        del gallery
        if device.type == "cuda":
            torch.cuda.synchronize()

    return reduce_min_rank(np.concatenate(key_parts), np.concatenate(rank_parts))


def load_pair_union(
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, object]],
    relatives: Sequence[str],
) -> tuple[np.ndarray, int]:
    parts: list[np.ndarray] = []
    for relative in relatives:
        path = verify_registered_artifact(output_dir, relative, artifacts[relative])
        with np.load(path, allow_pickle=False) as loaded:
            parts.append(np.asarray(loaded["pair_keys"], dtype=np.uint64))
    if not parts:
        return np.empty(0, dtype=np.uint64), 0
    return np.unique(np.concatenate(parts)), len(parts)


def load_group_channel(
    group_rows: Sequence[Mapping[str, object]],
    channel: str,
    descriptor_paths: Mapping[str, Path],
) -> np.ndarray:
    global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
    position = {int(index): offset for offset, index in enumerate(global_indices)}
    matrix = np.empty((len(group_rows), 512), dtype=np.float32)
    leaves = sorted({str(row["leaf"]) for row in group_rows})
    for leaf in leaves:
        leaf_rows = sorted(
            (row for row in group_rows if row["leaf"] == leaf),
            key=lambda row: int(row["selection_rank_within_leaf"]),
        )
        expected_ranks = list(range(1, len(leaf_rows) + 1))
        if [int(row["selection_rank_within_leaf"]) for row in leaf_rows] != expected_ranks:
            raise ValueError(f"Leaf selection ranks are not contiguous: {leaf}")
        path = descriptor_paths[leaf]
        with np.load(path, allow_pickle=False) as loaded:
            values = np.asarray(loaded[channel], dtype=np.float32)
        if values.shape != (len(leaf_rows), 512) or not np.isfinite(values).all():
            raise ValueError(f"Descriptor shape/finiteness mismatch: {leaf}/{channel}")
        for leaf_offset, row in enumerate(leaf_rows):
            matrix[position[int(row["image_index"])]] = values[leaf_offset]
    return matrix


def target_channel_result(
    descriptors: np.ndarray,
    left_local: int,
    right_local: int,
) -> dict[str, object]:
    left_similarities = descriptors @ descriptors[left_local]
    right_similarities = descriptors @ descriptors[right_local]
    left_similarities[left_local] = -np.inf
    right_similarities[right_local] = -np.inf
    left_rank = stable_directional_rank(left_similarities, right_local)
    right_rank = stable_directional_rank(right_similarities, left_local)
    return {
        "cosine_similarity": float(descriptors[left_local] @ descriptors[right_local]),
        "left_to_right_rank": left_rank,
        "right_to_left_rank": right_rank,
        "best_directed_rank": min(left_rank, right_rank),
    }


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def analyze(
    output_dir: Path,
    report_path: Path,
    config_path: Path,
    block_size: int,
    requested_device: str,
) -> dict[str, object]:
    config, config_hash = load_config(config_path)
    analyzer_hash = file_sha256(Path(__file__).resolve())
    if config["integrity_dependencies"]["analyzer_sha256"] != analyzer_hash:  # type: ignore[index]
        raise ValueError("Diagnostic analyzer hash differs from the predeclared config.")
    expected = config["expected_full_artifact_state"]
    target = config["target_pair"]
    top_k_grid = [int(value) for value in config["top_k_grid"]]
    if top_k_grid != sorted(set(top_k_grid)) or top_k_grid[0] != 15:
        raise ValueError("top_k_grid must be unique, sorted and start at 15.")
    max_k = top_k_grid[-1]

    manifest_path = output_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = dict(sorted(Counter(str(info["stage"]) for info in artifacts.values()).items()))
    if stage_counts != dict(sorted(EXPECTED_STAGE_COUNTS.items())):
        raise ValueError(f"Unexpected stage inventory: {stage_counts}")
    if len(artifacts) != int(expected["registered_artifacts"]):
        raise ValueError("Registered artifact count differs from the predeclared state.")
    if manifest["contract_sha256"] != expected["run_contract_sha256"]:
        raise ValueError("Run-contract hash differs from the predeclared state.")

    index_relative = "stages/image_index.csv"
    index_path = verify_registered_artifact(output_dir, index_relative, artifacts[index_relative])
    rows = read_index(index_path)
    if len(rows) != int(expected["image_count"]):
        raise ValueError("Full image count differs from the predeclared state.")
    group = str(target["source_group"])
    group_rows = [row for row in rows if row["source_group"] == group]
    if len(group_rows) != int(expected["target_group_image_count"]):
        raise ValueError("Target source-group image count differs from the predeclared state.")
    group_rows.sort(key=lambda row: int(row["image_index"]))
    global_indices = np.asarray([int(row["image_index"]) for row in group_rows], dtype=np.int64)
    by_path = {str(row["relative_path"]): offset for offset, row in enumerate(group_rows)}
    left_local = by_path[str(target["left_path"])]
    right_local = by_path[str(target["right_path"])]
    left_global = int(global_indices[left_local])
    right_global = int(global_indices[right_local])
    target_key = encode_pair_keys(
        np.asarray([min(left_global, right_global)]),
        np.asarray([max(left_global, right_global)]),
    )[0]

    robust_relatives = sorted(
        relative
        for relative, info in artifacts.items()
        if info["stage"] == "robust_retrieval"
        and info["metadata"]["source_group"] == group  # type: ignore[index]
    )
    phash_relatives = sorted(
        relative
        for relative, info in artifacts.items()
        if info["stage"] == "phash_retrieval"
        and info["metadata"]["source_group"] == group  # type: ignore[index]
    )
    if len(robust_relatives) != 4 or not phash_relatives:
        raise ValueError("Target-group retrieval inventory is incomplete.")
    if len(robust_relatives) != int(expected["target_group_robust_artifacts"]):
        raise ValueError("Target-group robust artifact count mismatch.")
    if len(phash_relatives) != int(expected["target_group_phash_artifacts"]):
        raise ValueError("Target-group pHash artifact count mismatch.")
    stored_robust, verified_robust = load_pair_union(output_dir, artifacts, robust_relatives)
    stored_phash, verified_phash = load_pair_union(output_dir, artifacts, phash_relatives)
    if len(stored_robust) != int(expected["target_group_robust_top15_pairs"]):
        raise ValueError("Stored target-group robust-top15 count mismatch.")
    if len(stored_phash) != int(expected["target_group_phash_pairs"]):
        raise ValueError("Stored target-group pHash count mismatch.")

    device = choose_device(requested_device)
    combined_keys = np.empty(0, dtype=np.uint64)
    combined_ranks = np.empty(0, dtype=np.uint16)
    channel_reports: dict[str, object] = {}
    descriptor_paths: dict[str, Path] = {}
    for leaf in sorted({str(row["leaf"]) for row in group_rows}):
        relative = f"stages/descriptors/{leaf}.npz"
        descriptor_paths[leaf] = verify_registered_artifact(
            output_dir, relative, artifacts[relative]
        )
    verified_descriptors = len(descriptor_paths)
    if verified_descriptors != int(expected["target_group_descriptor_artifacts"]):
        raise ValueError("Target-group descriptor artifact count mismatch.")
    for channel in DESCRIPTOR_NAMES:
        print(f"Loading descriptor channel: {channel}", flush=True)
        descriptors = load_group_channel(
            group_rows, channel, descriptor_paths
        )
        norms = np.linalg.norm(descriptors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Descriptor normalization mismatch: {channel}")
        target_result = target_channel_result(descriptors, left_local, right_local)
        print(
            f"Target {channel}: best directed rank="
            f"{target_result['best_directed_rank']}",
            flush=True,
        )
        channel_keys, channel_ranks = ranked_channel_pairs(
            descriptors, global_indices, max_k, block_size, device
        )
        channel_reports[channel] = {
            **target_result,
            "unique_pairs_at_max_k": int(len(channel_keys)),
        }
        combined_keys, combined_ranks = merge_ranked_pairs(
            combined_keys,
            combined_ranks,
            channel_keys,
            channel_ranks,
        )
        del descriptors, channel_keys, channel_ranks

    computed_top15 = combined_keys[combined_ranks <= 15]
    if not np.array_equal(computed_top15, stored_robust):
        raise ValueError("Recomputed robust top-15 differs from the frozen artifact union.")

    exact_target_ranks = [
        int(summary["best_directed_rank"]) for summary in channel_reports.values()  # type: ignore[union-attr]
    ]
    minimum_required_k = min(exact_target_ranks)
    projections: list[dict[str, object]] = []
    for top_k in top_k_grid:
        robust = combined_keys[combined_ranks <= top_k]
        target_union = np.union1d(robust, stored_phash)
        full_pairs = (
            int(expected["v2_4_counterfactual_pairs"])
            - int(expected["target_group_union_top15_pairs"])
            + len(target_union)
        )
        projections.append(
            {
                "top_k": top_k,
                "target_pair_recovered": bool(top_k >= minimum_required_k),
                "target_group_robust_pairs": int(len(robust)),
                "target_group_union_with_phash_pairs": int(len(target_union)),
                "full_counterfactual_pairs": int(full_pairs),
                "full_pairs_per_image": full_pairs / int(expected["image_count"]),
                "passes_25_pairs_per_image_capacity": (
                    full_pairs / int(expected["image_count"]) <= 25.0
                ),
            }
        )
    first = projections[0]
    if (
        first["target_group_union_with_phash_pairs"]
        != int(expected["target_group_union_top15_pairs"])
        or first["full_counterfactual_pairs"] != int(expected["v2_4_counterfactual_pairs"])
    ):
        raise ValueError("Top-15 projection does not reproduce the verified v2.4 baseline.")

    target_in_phash = bool(
        np.searchsorted(stored_phash, target_key) < len(stored_phash)
        and stored_phash[np.searchsorted(stored_phash, target_key)] == target_key
    )
    if target_in_phash != bool(expected["target_pair_in_phash_radius12"]):
        raise ValueError("Target pair pHash membership differs from the predeclared state.")
    return {
        "schema_version": 1,
        "status": "candidate_generator_v2_4_rank_crowding_diagnostic_complete",
        "experiment_type": "read_only_predeclared_full_index_rank_diagnostic",
        "does_not_modify_v2_3_artifacts": True,
        "not_a_v2_5_rule": True,
        "not_a_duplicate_decision": True,
        "does_not_authorize_family_split_or_training": True,
        "config": {
            "path_name": config_path.name,
            "sha256": config_hash,
            "analyzer_sha256": analyzer_hash,
        },
        "input": {
            "run_contract_sha256": manifest["contract_sha256"],
            "registered_artifact_count": len(artifacts),
            "stage_counts": stage_counts,
            "verified_artifacts_in_this_diagnostic": (
                1 + verified_robust + verified_phash + verified_descriptors
            ),
            "image_count": len(rows),
            "target_group": group,
            "target_group_image_count": len(group_rows),
            "device": str(device),
            "block_size": block_size,
        },
        "target_pair": {
            "pair_id": target["pair_id"],
            "left_path": target["left_path"],
            "right_path": target["right_path"],
            "left_global_index": left_global,
            "right_global_index": right_global,
            "phash_distance_from_audit": target["phash_distance_from_audit"],
            "present_in_full_phash_radius12": target_in_phash,
            "minimum_required_robust_top_k": minimum_required_k,
            "recovered_within_predeclared_grid": minimum_required_k <= max_k,
            "by_descriptor_channel": channel_reports,
        },
        "candidate_load_projections": projections,
        "baseline_reproduction": {
            "stored_robust_top15_pairs": int(len(stored_robust)),
            "recomputed_robust_top15_pairs": int(len(computed_top15)),
            "stored_phash_pairs": int(len(stored_phash)),
            "v2_4_full_pairs": int(first["full_counterfactual_pairs"]),
            "all_matched": True,
        },
        "decision": (
            "DIAGNOSTIC COMPLETE: review measured ranks and bounded load projections "
            "before predeclaring any v2.5 rule."
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    if is_inside(report_path, output_dir):
        raise ValueError("Diagnostic report must be outside the immutable v2.3 output directory.")
    report = analyze(
        output_dir=output_dir,
        report_path=report_path,
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
