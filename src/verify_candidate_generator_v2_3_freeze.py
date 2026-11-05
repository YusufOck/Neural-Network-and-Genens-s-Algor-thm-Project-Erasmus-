from __future__ import annotations

"""Verify the Candidate Generator v2.3 freeze before any full-data execution."""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torchvision.models import ResNet18_Weights

from data_index import discover_records, project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


EXPECTED_CONFIG_SHA256 = (
    "daef2f4d1c4199846a64b549a5070f5c8cc3f322cc949fb55dde2700ef2e3776"
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Candidate Generator v2.3 config, source hash, dataset inventory and "
            "natural-audit evidence chain; does not run retrieval/family/split/model."
        )
    )
    parser.add_argument(
        "--config", default=str(root / "config" / "candidate_generator_v2_3.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "candidate_generator_v2_3.sha256"),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--natural-references",
        default=str(
            root
            / "research"
            / "natural_duplicate_adjudication_v1"
            / "final_pair_references_v1.csv"
        ),
    )
    parser.add_argument(
        "--v2-2-report",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_2_holdout"
            / "evaluation"
            / "report.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(root / "research" / "candidate_generator_v2_3_freeze_report.json"),
    )
    return parser.parse_args()


def load_frozen_config(config_path: Path, hash_path: Path) -> Mapping[str, object]:
    expected_sidecar = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = file_sha256(config_path)
    if expected_sidecar != EXPECTED_CONFIG_SHA256 or actual != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            "Candidate Generator v2.3 frozen config hash mismatch: "
            f"constant={EXPECTED_CONFIG_SHA256}, sidecar={expected_sidecar}, actual={actual}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_full_dataset_candidate_retrieval":
        raise ValueError("V2.3 config full retrieval before frozen state is not set.")
    rule = dict(config["candidate_rule"])
    robust = dict(rule["robust_retrieval"])
    phash = dict(rule["phash_retrieval"])
    if int(robust["top_k"]) != 15:
        raise ValueError("Frozen v2.3 robust top-k must equal 15.")
    if int(phash["hamming_radius_inclusive"]) != 12:
        raise ValueError("Frozen v2.3 pHash radius must equal 12.")
    if rule["search_scope"] != "within_compatible_source_group_only":
        raise ValueError("Frozen v2.3 scope must stay within compatible source groups.")
    if len(robust["descriptor_channels"]) != 4:
        raise ValueError("Frozen v2.3 must contain exactly four descriptor channels.")
    return config


def verify_dependency_hashes(
    root: Path, config: Mapping[str, object]
) -> dict[str, str]:
    expected = dict(config["integrity_dependencies"])
    actual: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = (root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise FileNotFoundError(f"Frozen dependency is missing or outside project: {relative}")
        digest = file_sha256(path)
        if digest != str(expected_hash):
            raise ValueError(
                f"Frozen dependency changed: {relative}; expected={expected_hash}, actual={digest}"
            )
        actual[str(relative)] = digest
    return actual


def verify_checkpoint(config: Mapping[str, object]) -> dict[str, str]:
    robust = dict(dict(config["candidate_rule"])["robust_retrieval"])
    weights = ResNet18_Weights.DEFAULT
    checkpoint_name = Path(weights.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ResNet18 checkpoint not found: {checkpoint_path}")
    actual = file_sha256(checkpoint_path)
    expected = str(robust["checkpoint_sha256"])
    if actual != expected:
        raise ValueError(
            f"ResNet18 checkpoint hash mismatch: expected={expected}, actual={actual}"
        )
    if str(weights) != str(robust["weights"]):
        raise ValueError("Torchvision default ResNet18 weights no longer match frozen config.")
    return {"path": str(checkpoint_path), "sha256": actual, "weights": str(weights)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is missing: {path}")
        return [dict(row) for row in reader]


def verify_natural_evidence(
    path: Path, config: Mapping[str, object]
) -> dict[str, object]:
    evidence = dict(config["evidence"])
    expected_hash = str(evidence["natural_final_references_sha256"])
    if file_sha256(path) != expected_hash:
        raise ValueError("Natural final references hash does not match frozen evidence.")
    rows = read_csv(path)
    if len(rows) != int(evidence["natural_reference_count"]):
        raise ValueError("Natural final reference count does not match frozen evidence.")
    same = [row for row in rows if row["final_reference"] == "same_source"]
    inside = sum(row["audit_region"] == "candidate" for row in same)
    outside = sum(row["audit_region"] == "outside" for row in same)
    if inside != int(evidence["natural_same_source_inside_candidate"]):
        raise ValueError("Natural same-source candidate count changed.")
    if outside != int(evidence["natural_same_source_outside_candidate"]):
        raise ValueError("Natural outside same-source count changed.")
    return {
        "reference_count": len(rows),
        "final_reference_counts": dict(
            sorted(Counter(row["final_reference"] for row in rows).items())
        ),
        "same_source_inside_candidate": inside,
        "same_source_outside_candidate": outside,
        "sha256": expected_hash,
    }


def verify_v22_evidence(
    path: Path, config: Mapping[str, object]
) -> dict[str, object]:
    evidence = dict(config["evidence"])
    expected_hash = str(evidence["candidate_generator_v2_2_evaluation_report_sha256"])
    if file_sha256(path) != expected_hash:
        raise ValueError("V2.2 evaluation report hash does not match frozen evidence.")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": report["status"],
        "image_count": report["input"]["image_count"],
        "positive_pair_count": report["input"]["positive_pair_count"],
        "frozen_v2_2_recall": report["positive_metrics"]["v2_2_union_recall"],
        "frozen_v2_2_gate_passed": report["acceptance"]["gate_passed"],
        "sha256": expected_hash,
    }


def load_source_group_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for group, details in dict(payload["source_groups"]).items():
        for label in details["labels"]:
            if label in mapping:
                raise ValueError(f"Leaf is assigned to multiple source groups: {label}")
            mapping[str(label)] = str(group)
    return mapping


def verify_dataset_inventory(
    dataset_root: Path, config: Mapping[str, object], groups_path: Path
) -> dict[str, object]:
    scope = dict(config["full_run_scope"])
    source_by_leaf = load_source_group_map(groups_path)
    records = discover_records(dataset_root)
    leaf_counts = Counter(record.subclass for record in records)
    if len(records) != int(scope["expected_dataset_images"]):
        raise ValueError(f"Dataset image count changed: {len(records)}")
    if set(leaf_counts) != set(source_by_leaf):
        raise ValueError("Dataset leaves do not match the frozen source-group map.")
    if len(leaf_counts) != int(scope["expected_leaf_count"]):
        raise ValueError("Dataset leaf count changed.")
    if any(count != int(scope["expected_images_per_leaf"]) for count in leaf_counts.values()):
        raise ValueError("At least one leaf no longer has the frozen image count.")
    if len(set(source_by_leaf.values())) != int(scope["expected_source_group_count"]):
        raise ValueError("Frozen source-group count changed.")
    return {
        "dataset_root": str(dataset_root),
        "image_count": len(records),
        "leaf_count": len(leaf_counts),
        "source_group_count": len(set(source_by_leaf.values())),
        "images_per_leaf": dict(sorted(leaf_counts.items())),
    }


def capacity_summary(config: Mapping[str, object]) -> dict[str, object]:
    basis = dict(config["capacity_plan_basis"])
    estimated_pairs = int(basis["estimated_full_unique_candidate_pairs"])
    seconds_per_pair = float(basis["measured_frozen_scorer_seconds_per_pair"])
    serial_seconds = estimated_pairs * seconds_per_pair
    return {
        "estimated_candidate_pairs": estimated_pairs,
        "maximum_candidate_pairs_before_pause": int(
            basis["maximum_pairs_at_integrity_gate"]
        ),
        "descriptor_bytes": int(basis["four_descriptor_float32_bytes"]),
        "phash_bytes": int(basis["two_uint64_phash_arrays_bytes"]),
        "serial_verification_seconds": serial_seconds,
        "serial_verification_days": serial_seconds / 86400,
        "ideal_eight_worker_days_without_overhead": serial_seconds / 86400 / 8,
    }


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    root = project_root().resolve()
    config_path = Path(args.config).resolve()
    hash_path = Path(args.config_sha256).resolve()
    natural_path = Path(args.natural_references).resolve()
    v22_report_path = Path(args.v2_2_report).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Freeze verification report already exists: {output_path}")
    config = load_frozen_config(config_path, hash_path)
    dependencies = verify_dependency_hashes(root, config)
    checkpoint = verify_checkpoint(config)
    natural = verify_natural_evidence(natural_path, config)
    v22 = verify_v22_evidence(v22_report_path, config)
    groups_path = root / "config" / "duplicate_audit_groups.json"
    dataset = verify_dataset_inventory(
        resolve_dataset_root(args.data_dir), config, groups_path
    )
    capacity = capacity_summary(config)

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "candidate_generator_v2_3_frozen_full_run_not_started",
        "verification_type": "pre_full_run_configuration_and_capacity_freeze",
        "candidate_rule": config["candidate_rule"],
        "dataset_inventory": dataset,
        "natural_evidence": natural,
        "v2_2_evidence": v22,
        "capacity_estimate": capacity,
        "integrity": {
            "frozen_config_sha256": file_sha256(config_path),
            "dependency_hashes": dependencies,
            "resnet18_checkpoint": checkpoint,
            "verifier_sha256": file_sha256(Path(__file__).resolve()),
        },
        "next_step": (
            "Create a Colab/local orchestration notebook that refuses changed hashes, "
            "then benchmark one bounded shard before the 130000-image retrieval."
        ),
        "limitations": list(config["interpretation_limits"]),
    }
    atomic_json(output_path, report)
    print("Operation: CANDIDATE GENERATOR v2.3 FREEZE VERIFICATION")
    print(f"Frozen config SHA-256: {file_sha256(config_path)}")
    print(
        "Dataset inventory: "
        f"{dataset['image_count']} images, {dataset['leaf_count']} leaves, "
        f"{dataset['source_group_count']} source groups"
    )
    print(
        "Natural evidence: "
        f"{natural['same_source_inside_candidate']} same-source inside, "
        f"{natural['same_source_outside_candidate']} outside"
    )
    print(
        "Capacity estimate: "
        f"{capacity['estimated_candidate_pairs']} candidates, "
        f"{capacity['serial_verification_days']:.2f} serial CPU days"
    )
    print(f"Report: {output_path}")
    print("NOTE: Full retrieval, family, split, and model training were not started.")
    return output_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
