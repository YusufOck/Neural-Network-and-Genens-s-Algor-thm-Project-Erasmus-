from __future__ import annotations

"""Score the 16 v2.5 outside positives under the frozen same-2D target.

This is a small development audit. It preserves the historical v2.5 decisions,
extracts the already-frozen ORB/phase/ECC/SSIM evidence, and applies duplicate
rule v1. It does not create families, split the dataset, or train a classifier.
"""

import argparse
import csv
import json
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import cv2

from data_index import project_root, resolve_dataset_root
from duplicate_decision_rule import classify_pair, file_sha256, load_frozen_rule
from score_duplicate_development import OUTPUT_FIELDS as SCORE_OUTPUT_FIELDS
from score_duplicate_development import score_pair


ROOT = project_root()
DEFAULT_TARGET = ROOT / "config" / "leakage_family_target_v1.json"
DEFAULT_INPUT = (
    ROOT
    / "research"
    / "candidate_generator_v2_5_prospective_audit_v1"
    / "unblinded_pair_results.csv"
)
DEFAULT_GROUPS = ROOT / "config" / "duplicate_audit_groups.json"
DEFAULT_RULE = ROOT / "config" / "duplicate_rule_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "v2_6_same2d_development_v1"

DECISION_FIELDS = (
    "same2d_target_id",
    "frozen_v2_5_visual_verdict",
    "rule_decision",
    "rule_decision_path",
    "rule_decision_reason",
    "flag_sha256_exact",
    "flag_strong_geometry",
    "flag_strong_intensity_structure",
    "flag_conservative_independent",
)
OUTPUT_FIELDS = SCORE_OUTPUT_FIELDS + DECISION_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the 16 frozen v2.5 outside positives against the approved "
            "same-2D augmentation-family target."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--target-config", default=str(DEFAULT_TARGET))
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--groups-config", default=str(DEFAULT_GROUPS))
    parser.add_argument("--rule", default=str(DEFAULT_RULE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {path}")
        return [dict(row) for row in reader]


def load_frozen_target(path: Path) -> tuple[dict[str, object], str]:
    sidecar = path.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("Family target SHA-256 sidecar has invalid format.")
    actual = file_sha256(path)
    if actual != fields[0].lower():
        raise ValueError("Family target SHA-256 mismatch.")
    target = read_json(path)
    if target.get("status") != "frozen_by_mehmet_before_accelerated_split_pipeline":
        raise ValueError("Family target is not in the approved frozen state.")
    return target, actual


def select_development_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if row.get("audit_region") == "outside"
        and row.get("visual_verdict") == "same_source_candidate"
    ]
    selected.sort(key=lambda row: row["pair_id"])
    if len(selected) != 16 or len({row["pair_id"] for row in selected}) != 16:
        raise ValueError("Expected exactly 16 unique frozen v2.5 outside positives.")
    expected_groups = Counter({"brain_ce_mri": 6, "kidney_ct": 6, "lymphoma_histology": 4})
    actual_groups = Counter(row["source_group"] for row in selected)
    if actual_groups != expected_groups:
        raise ValueError(f"Unexpected target-group counts: {actual_groups}")
    return selected


def group_metadata(path: Path) -> dict[str, dict[str, object]]:
    payload = read_json(path)
    groups = payload.get("source_groups")
    if not isinstance(groups, dict):
        raise ValueError("source_groups missing from audit-group config.")
    return {str(name): dict(value) for name, value in groups.items()}


def make_score_input(
    source: Mapping[str, str],
    modality_key: str,
    sha256_equal: bool,
) -> dict[str, str]:
    return {
        "pair_id": source["pair_id"],
        "audit_partition": "development",
        "pair_kind": "v2_5_outside_positive_same2d_reassessment",
        "truth_status": "development_not_new_holdout",
        "human_verdict": source["visual_verdict"],
        "reference_label": "",
        "reference_basis": "frozen_v2_5_broad_acquisition_family_visual_label",
        "left_leaf": source["left_leaf"],
        "right_leaf": source["right_leaf"],
        "modality_key": modality_key,
        "source_group": source["source_group"],
        "transform_name": "",
        "sha256_equal": "yes" if sha256_equal else "no",
        "phash_distance": source.get("phash_distance", ""),
    }


def scorer_args() -> SimpleNamespace:
    return SimpleNamespace(
        image_size=256,
        orb_features=4000,
        ratio_test=0.78,
        ransac_threshold=3.0,
        ecc_iterations=100,
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    target_path = Path(args.target_config).resolve()
    input_path = Path(args.input_csv).resolve()
    groups_path = Path(args.groups_config).resolve()
    rule_path = Path(args.rule).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_csv = output_dir / "pair_scores_and_rule.csv"
    output_json = output_dir / "report.json"
    if not args.overwrite and (output_csv.exists() or output_json.exists()):
        raise FileExistsError("Output exists; use --overwrite to regenerate atomically.")

    target, target_hash = load_frozen_target(target_path)
    dependencies = target["frozen_dependencies"]
    if file_sha256(input_path) != dependencies["v2_5_unblinded_pairs_sha256"]:
        raise ValueError("Frozen v2.5 unblinded input hash mismatch.")
    if file_sha256(ROOT / "research" / "candidate_generator_v2_6_missed_pair_rank_diagnostic_v1.json") != dependencies["v2_6_rank_report_sha256"]:
        raise ValueError("Frozen v2.6 rank report hash mismatch.")
    if file_sha256(ROOT / "src" / "score_duplicate_development.py") != dependencies["scorer_sha256"]:
        raise ValueError("Frozen scorer hash mismatch.")
    rule, rule_hash = load_frozen_rule(rule_path)
    if rule_hash != dependencies["duplicate_rule_v1_sha256"]:
        raise ValueError("Frozen duplicate rule hash differs from the target config.")

    dataset_root = resolve_dataset_root(args.data_dir)
    rows = select_development_rows(read_csv(input_path))
    metadata = group_metadata(groups_path)
    settings = scorer_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)

    scored: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, source in enumerate(rows, start=1):
        left_path = dataset_root / source["left_path"]
        right_path = dataset_root / source["right_path"]
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"Pair image missing: {source['pair_id']}")
        same_hash = file_sha256(left_path) == file_sha256(right_path)
        group = source["source_group"]
        score_input = make_score_input(source, str(metadata[group]["modality_key"]), same_hash)
        evidence = score_pair(score_input, left_path, right_path, settings)
        decision = classify_pair(evidence, rule)
        combined = {
            **evidence,
            "same2d_target_id": str(target["target_id"]),
            "frozen_v2_5_visual_verdict": source["visual_verdict"],
            "rule_decision": decision["decision"],
            "rule_decision_path": decision["decision_path"],
            "rule_decision_reason": decision["decision_reason"],
            "flag_sha256_exact": decision["flag_sha256_exact"],
            "flag_strong_geometry": decision["flag_strong_geometry"],
            "flag_strong_intensity_structure": decision["flag_strong_intensity_structure"],
            "flag_conservative_independent": decision["flag_conservative_independent"],
        }
        scored.append(combined)
        print(
            f"{index:02d}/16 {source['pair_id']} {group}: "
            f"{decision['decision']} ({decision['decision_path']})",
            flush=True,
        )
    total_elapsed = time.perf_counter() - started
    elapsed_values = [float(row["elapsed_seconds"]) for row in scored]
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        by_group[row["source_group"]][row["rule_decision"]] += 1
    decision_counts = Counter(row["rule_decision"] for row in scored)
    path_counts = Counter(row["rule_decision_path"] for row in scored)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "v2_6_same2d_development_scoring_complete",
        "experiment_type": "small_development_reassessment_under_frozen_same_2d_target",
        "not_a_new_holdout": True,
        "does_not_rewrite_v2_5": True,
        "does_not_create_family_split_or_model": True,
        "inputs": {
            "family_target_sha256": target_hash,
            "v2_5_unblinded_pairs_sha256": file_sha256(input_path),
            "v2_6_rank_report_sha256": dependencies["v2_6_rank_report_sha256"],
            "scorer_sha256": dependencies["scorer_sha256"],
            "duplicate_rule_v1_sha256": rule_hash,
            "dataset_root": str(dataset_root),
        },
        "scorer_parameters": vars(settings),
        "pair_count": len(scored),
        "decision_counts": dict(sorted(decision_counts.items())),
        "decision_path_counts": dict(sorted(path_counts.items())),
        "decision_counts_by_source_group": {
            group: dict(sorted(counts.items())) for group, counts in sorted(by_group.items())
        },
        "same_source_pair_ids": [row["pair_id"] for row in scored if row["rule_decision"] == "same_source"],
        "independent_pair_ids": [row["pair_id"] for row in scored if row["rule_decision"] == "independent"],
        "uncertain_pair_ids": [row["pair_id"] for row in scored if row["rule_decision"] == "uncertain"],
        "timing": {
            "total_elapsed_seconds": total_elapsed,
            "mean_pair_seconds": statistics.mean(elapsed_values),
            "median_pair_seconds": statistics.median(elapsed_values),
            "maximum_pair_seconds": max(elapsed_values),
        },
        "interpretation": (
            "Frozen-rule decisions are development evidence under the approved same-2D target. "
            "Uncertain rows require separate adjudication; historical v2.5 labels remain unchanged."
        ),
    }
    atomic_csv(output_csv, scored)
    atomic_json(output_json, report)
    return output_csv, output_json, report


def main() -> None:
    args = parse_args()
    output_csv, output_json, report = run(args)
    print(f"Decisions: {report['decision_counts']}")
    print(f"Timing: {report['timing']}")
    print(f"Scores: {output_csv}")
    print(f"Report: {output_json}")


if __name__ == "__main__":
    main()
