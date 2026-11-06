from __future__ import annotations

"""Read-only workload decomposition for a paused Candidate Generator v2.3 full run."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from candidate_generator_v2_3_runtime import (
    atomic_json,
    decode_pair_keys,
    exact_duplicate_pair_keys,
    verify_exact_duplicate_manifest,
)


EXPECTED_STAGE_COUNTS = {
    "index": 1,
    "phash_generation": 26,
    "descriptor_generation": 26,
    "robust_retrieval": 32,
    "phash_retrieval": 1022,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the frozen v2.3 full candidate load without changing the "
            "retrieval artifacts or producing final candidate files."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exact-duplicates", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pause-gate", type=float, default=25.0)
    parser.add_argument("--scorer-seconds-per-pair", type=float, default=0.815)
    return parser.parse_args()


def unique_pair_keys(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.uint64) for part in parts]
    if not arrays:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(arrays))


def load_pair_parts(output_dir: Path, relatives: Sequence[str]) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for relative in relatives:
        path = output_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Registered pair artifact is missing: {path}")
        with np.load(path) as loaded:
            parts.append(np.asarray(loaded["pair_keys"], dtype=np.uint64))
    return parts


def channel_summary(
    robust_parts: Sequence[np.ndarray], phash_parts: Sequence[np.ndarray]
) -> tuple[dict[str, int], np.ndarray]:
    robust = unique_pair_keys(robust_parts)
    phash = unique_pair_keys(phash_parts)
    intersection = np.intersect1d(robust, phash, assume_unique=True)
    union = np.union1d(robust, phash)
    summary = {
        "robust_unique_pairs": int(len(robust)),
        "phash_unique_pairs": int(len(phash)),
        "intersection_pairs": int(len(intersection)),
        "union_pairs": int(len(union)),
    }
    if summary["robust_unique_pairs"] + summary["phash_unique_pairs"] - summary[
        "intersection_pairs"
    ] != summary["union_pairs"]:
        raise ValueError("Candidate channel union arithmetic failed.")
    return summary, union


def read_index(path: Path) -> tuple[list[str], dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["image_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Image index is not contiguous and deterministic.")
    groups = [row["source_group"] for row in rows]
    by_relative = {row["relative_path"]: int(row["image_index"]) for row in rows}
    if len(by_relative) != len(rows):
        raise ValueError("Image index contains duplicate relative paths.")
    return groups, by_relative


def validate_group_pairs(keys: np.ndarray, group: str, groups: Sequence[str]) -> None:
    left, right = decode_pair_keys(keys)
    if np.any(left >= right):
        raise ValueError(f"Non-canonical pair found in group {group}.")
    if len(right) and int(right.max()) >= len(groups):
        raise ValueError(f"Pair index exceeds image inventory in group {group}.")
    for left_index, right_index in zip(left, right, strict=True):
        if groups[int(left_index)] != group or groups[int(right_index)] != group:
            raise ValueError(f"Cross-group pair found in artifact group {group}.")


def analyze(
    output_dir: Path,
    exact_path: Path,
    pause_gate: float,
    scorer_seconds_per_pair: float,
) -> dict[str, object]:
    manifest_path = output_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = Counter(str(info["stage"]) for info in artifacts.values())
    actual_stage_counts = dict(sorted(stage_counts.items()))
    if actual_stage_counts != dict(sorted(EXPECTED_STAGE_COUNTS.items())):
        raise ValueError(
            "Full retrieval stage inventory is incomplete or unexpected: "
            f"{actual_stage_counts}"
        )

    groups, index_by_relative = read_index(output_dir / "stages" / "image_index.csv")
    source_groups = sorted(set(groups))
    if len(groups) != 130_000 or len(source_groups) != 8:
        raise ValueError(
            f"Expected 130000 images/8 groups, got {len(groups)}/{len(source_groups)}."
        )

    group_reports: dict[str, object] = {}
    union_parts: list[np.ndarray] = []
    for group in source_groups:
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
            raise ValueError(f"Unexpected retrieval artifact inventory for {group}.")

        robust_parts = load_pair_parts(output_dir, robust_relatives)
        phash_parts = load_pair_parts(output_dir, phash_relatives)
        summary, union = channel_summary(robust_parts, phash_parts)
        validate_group_pairs(union, group, groups)
        image_count = groups.count(group)
        summary.update(
            {
                "image_count": image_count,
                "union_pairs_per_image": len(union) / image_count,
                "robust_channel_raw_pair_counts": {
                    str(artifacts[relative]["metadata"]["channel"]): int(
                        artifacts[relative]["metadata"]["pairs"]
                    )
                    for relative in robust_relatives
                },
                "phash_block_count": len(phash_relatives),
            }
        )
        group_reports[group] = summary
        union_parts.append(union)

    full_union = unique_pair_keys(union_parts)
    robust_total = sum(
        int(report["robust_unique_pairs"])  # type: ignore[index]
        for report in group_reports.values()
    )
    phash_total = sum(
        int(report["phash_unique_pairs"])  # type: ignore[index]
        for report in group_reports.values()
    )
    intersection_total = sum(
        int(report["intersection_pairs"])  # type: ignore[index]
        for report in group_reports.values()
    )
    union_total = len(full_union)
    if robust_total + phash_total - intersection_total != union_total:
        raise ValueError("Full candidate union arithmetic failed.")

    normalized_exact, normalized_exact_hash = verify_exact_duplicate_manifest(exact_path)
    expected_exact = exact_duplicate_pair_keys(normalized_exact, index_by_relative)
    missing_exact = np.setdiff1d(expected_exact, full_union, assume_unique=True)

    pairs_per_image = union_total / len(groups)
    serial_seconds = union_total * scorer_seconds_per_pair
    return {
        "schema_version": 1,
        "status": "candidate_generator_v2_3_full_load_diagnostic_complete",
        "experiment_type": "read_only_candidate_load_diagnostic",
        "does_not_modify_frozen_artifacts": True,
        "not_a_duplicate_decision": True,
        "run_contract_sha256": manifest["contract_sha256"],
        "registered_artifact_count": len(artifacts),
        "stage_counts": actual_stage_counts,
        "input": {
            "image_count": len(groups),
            "source_group_count": len(source_groups),
        },
        "candidate_counts": {
            "robust_unique_pairs": robust_total,
            "phash_unique_pairs": phash_total,
            "intersection_pairs": intersection_total,
            "union_pairs": union_total,
            "union_pairs_per_image": pairs_per_image,
            "pause_gate_pairs_per_image": pause_gate,
            "pause_gate_passed": pairs_per_image <= pause_gate,
        },
        "exact_duplicate_coverage": {
            "normalized_manifest_sha256": normalized_exact_hash,
            "expected_pairs": int(len(expected_exact)),
            "missing_pairs": int(len(missing_exact)),
            "all_covered": len(missing_exact) == 0,
        },
        "capacity_projection": {
            "frozen_scorer_seconds_per_pair": scorer_seconds_per_pair,
            "serial_days": serial_seconds / 86_400,
            "ideal_eight_worker_days": serial_seconds / 86_400 / 8,
            "warning": "Eight-worker value is an ideal lower bound, not a measured runtime.",
        },
        "by_source_group": group_reports,
        "decision": (
            "PAUSE: candidate load exceeds the frozen operational gate; do not score or "
            "change v2.3 before reviewing this decomposition."
            if pairs_per_image > pause_gate
            else "Candidate load is within the frozen operational gate."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.pause_gate <= 0 or args.scorer_seconds_per_pair <= 0:
        raise ValueError("Pause gate and scorer seconds must be positive.")
    report = analyze(
        Path(args.output_dir).resolve(),
        Path(args.exact_duplicates).resolve(),
        args.pause_gate,
        args.scorer_seconds_per_pair,
    )
    report_path = Path(args.report).resolve()
    atomic_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "by_source_group"}, indent=2))
    print("By source group:")
    for group, summary in report["by_source_group"].items():  # type: ignore[union-attr]
        print(
            f"  {group}: union={summary['union_pairs']}, "  # type: ignore[index]
            f"pairs/image={summary['union_pairs_per_image']:.3f}, "  # type: ignore[index]
            f"robust={summary['robust_unique_pairs']}, "  # type: ignore[index]
            f"phash={summary['phash_unique_pairs']}"  # type: ignore[index]
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
