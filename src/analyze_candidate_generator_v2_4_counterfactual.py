from __future__ import annotations

"""Read-only evaluation of the predeclared modality-aware v2.4 hypothesis."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from candidate_generator_v2_3_runtime import (
    atomic_json,
    decode_pair_keys,
    encode_pair_keys,
    exact_duplicate_pair_keys,
    verify_exact_duplicate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "candidate_generator_v2_4_counterfactual.json"
DEFAULT_NATURAL_REFERENCES = (
    ROOT / "research" / "natural_duplicate_adjudication_v1" / "final_pair_references_v1.csv"
)
DEFAULT_NATURAL_RESULTS = (
    ROOT / "research" / "natural_duplicate_audit_v1" / "unblinded_pair_results.csv"
)
DEFAULT_CONTROLLED_METRICS = (
    ROOT
    / "artifacts"
    / "audit"
    / "candidate_generator_v2_2_holdout"
    / "evaluation"
    / "positive_pair_metrics.csv"
)
DEFAULT_CONTROLLED_SUMMARY = (
    ROOT / "research" / "candidate_generator_v2_4_controlled_reference_summary.json"
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
            "Evaluate the predeclared v2.4 modality-aware candidate hypothesis from "
            "existing v2.3 artifacts without modifying them or scoring duplicates."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Completed v2.3 retrieval directory")
    parser.add_argument("--exact-duplicates", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--natural-references", default=str(DEFAULT_NATURAL_REFERENCES))
    parser.add_argument("--natural-results", default=str(DEFAULT_NATURAL_RESULTS))
    parser.add_argument("--controlled-summary", default=str(DEFAULT_CONTROLLED_SUMMARY))
    parser.add_argument(
        "--controlled-metrics",
        default="",
        help=(
            "Optional raw ignored artifact for independently checking the tracked summary. "
            "The Colab run does not require this file."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_config(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Counterfactual config not found: {path}")
    sha_path = path.with_suffix(".sha256")
    if not sha_path.is_file():
        raise FileNotFoundError(f"Counterfactual config hash not found: {sha_path}")
    fields = sha_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("Counterfactual config .sha256 file has an invalid format.")
    actual = file_sha256(path)
    if actual != fields[0].lower():
        raise ValueError(
            f"Counterfactual config hash mismatch: expected={fields[0]}, actual={actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "predeclared_counterfactual_before_full_artifact_evaluation":
        raise ValueError("Counterfactual config is not in the predeclared state.")
    return payload, actual


def unique_pair_keys(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.uint64) for part in parts]
    if not arrays:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(arrays))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_index(path: Path) -> tuple[list[str], dict[str, int]]:
    rows = read_csv(path)
    indices = [int(row["image_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Image index is not contiguous and deterministic.")
    groups = [row["source_group"] for row in rows]
    by_relative = {row["relative_path"]: int(row["image_index"]) for row in rows}
    if len(by_relative) != len(rows):
        raise ValueError("Image index contains duplicate relative paths.")
    return groups, by_relative


def verify_registered_artifact(
    output_dir: Path, relative: str, info: Mapping[str, object]
) -> Path:
    path = output_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"Registered artifact is missing: {path}")
    actual = file_sha256(path)
    expected = str(info["sha256"])
    if actual != expected:
        raise ValueError(
            f"Registered artifact hash mismatch for {relative}: "
            f"expected={expected}, actual={actual}"
        )
    return path


def load_registered_pair_parts(
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, object]],
    relatives: Sequence[str],
) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for relative in relatives:
        path = verify_registered_artifact(output_dir, relative, artifacts[relative])
        with np.load(path) as loaded:
            keys = np.asarray(loaded["pair_keys"], dtype=np.uint64)
        if len(keys) and np.any(keys[:-1] >= keys[1:]):
            raise ValueError(f"Pair keys are not strictly sorted and unique: {relative}")
        parts.append(keys)
    return parts


def validate_group_pairs(keys: np.ndarray, group: str, groups: Sequence[str]) -> None:
    left, right = decode_pair_keys(keys)
    if np.any(left >= right):
        raise ValueError(f"Non-canonical pair found in group {group}.")
    if len(right) and int(right.max()) >= len(groups):
        raise ValueError(f"Pair index exceeds image inventory in group {group}.")
    if any(
        groups[int(left_index)] != group or groups[int(right_index)] != group
        for left_index, right_index in zip(left, right, strict=True)
    ):
        raise ValueError(f"Cross-source-group pair found in artifact group {group}.")


def proposed_group_union(
    robust: np.ndarray,
    phash: np.ndarray,
    exact: np.ndarray,
    phash_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.union1d(robust, phash)
    proposal_parts = [robust, exact]
    if phash_enabled:
        proposal_parts.append(phash)
    proposal = unique_pair_keys(proposal_parts)
    return baseline, proposal


def membership_count(reference_keys: np.ndarray, candidates: np.ndarray) -> tuple[int, list[int]]:
    reference = np.asarray(reference_keys, dtype=np.uint64)
    candidate = np.asarray(candidates, dtype=np.uint64)
    present = np.isin(reference, candidate, assume_unique=False)
    missing_indices = np.flatnonzero(~present).astype(int).tolist()
    return int(present.sum()), missing_indices


def controlled_policy_hit(row: Mapping[str, str], phash_groups: set[str]) -> bool:
    robust_hit = row["robust_top15_candidate"].strip().lower() == "yes"
    phash_hit = (
        row["source_group"] in phash_groups
        and int(row["flip_aware_phash_distance"]) <= 12
    )
    return robust_hit or phash_hit


def evaluate_controlled_references(
    rows: Sequence[Mapping[str, str]],
    all_groups: set[str],
    phash_groups: set[str],
) -> dict[str, object]:
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Controlled reference file contains duplicate Pair IDs.")
    unknown = sorted({row["source_group"] for row in rows} - all_groups)
    if unknown:
        raise ValueError(f"Controlled references contain unknown source groups: {unknown}")

    misses: list[str] = []
    by_group: dict[str, dict[str, int]] = {}
    for group in sorted(all_groups):
        group_rows = [row for row in rows if row["source_group"] == group]
        hits = sum(controlled_policy_hit(row, phash_groups) for row in group_rows)
        by_group[group] = {"references": len(group_rows), "hits": hits}
        misses.extend(
            row["pair_id"]
            for row in group_rows
            if not controlled_policy_hit(row, phash_groups)
        )
    return {
        "reference_count": len(rows),
        "hits": len(rows) - len(misses),
        "misses": len(misses),
        "missing_pair_ids": misses,
        "by_source_group": by_group,
        "evaluation_note": (
            "Generated holdout images are outside the 130000-image full index; this gate "
            "is recomputed from frozen robust-top15 and pHash-distance metrics."
        ),
    }


def evaluate_controlled_summary(
    summary_path: Path,
    all_groups: set[str],
    disabled_groups: set[str],
    raw_metrics_path: Path | None = None,
) -> dict[str, object]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Controlled reference summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != (
        "derived_from_frozen_v2_2_positive_pair_metrics_before_v2_4_full_artifact_evaluation"
    ):
        raise ValueError("Controlled reference summary has an unexpected status.")
    policy = payload["counterfactual_policy"]
    if not policy["robust_top15_all_source_groups"] or int(
        policy["phash_radius_inclusive"]
    ) != 12:
        raise ValueError("Controlled summary policy does not match robust-top15/pHash<=12.")
    if set(policy["phash_disabled_source_groups"]) != disabled_groups:
        raise ValueError("Controlled summary pHash-disabled groups do not match the config.")
    by_group = payload["by_source_group"]
    if set(by_group) != all_groups:
        raise ValueError("Controlled summary source groups do not match the config.")

    summed = {
        key: sum(int(values[key]) for values in by_group.values())
        for key in (
            "references",
            "robust_top15_hits",
            "v2_3_union_hits",
            "v2_4_counterfactual_hits",
        )
    }
    if summed != {key: int(value) for key, value in payload["totals"].items()}:
        raise ValueError("Controlled summary totals do not equal its group totals.")
    if summed["references"] != int(payload["source_metrics"]["row_count"]):
        raise ValueError("Controlled summary row count is inconsistent.")

    raw_verified = False
    if raw_metrics_path is not None:
        if not raw_metrics_path.is_file():
            raise FileNotFoundError(f"Optional raw controlled metrics not found: {raw_metrics_path}")
        actual_raw_hash = file_sha256(raw_metrics_path)
        if actual_raw_hash != payload["source_metrics"]["sha256"]:
            raise ValueError("Raw controlled metrics hash does not match the tracked summary.")
        rows = read_csv(raw_metrics_path)
        recomputed = evaluate_controlled_references(
            rows, all_groups, all_groups - disabled_groups
        )
        for group, values in by_group.items():
            if int(values["references"]) != recomputed["by_source_group"][group]["references"]:
                raise ValueError(f"Controlled summary reference count mismatch for {group}.")
            if int(values["v2_4_counterfactual_hits"]) != recomputed["by_source_group"][group]["hits"]:
                raise ValueError(f"Controlled summary proposal-hit mismatch for {group}.")
        if int(recomputed["hits"]) != summed["v2_4_counterfactual_hits"]:
            raise ValueError("Controlled summary total proposal hits do not match raw metrics.")
        raw_verified = True

    return {
        "summary_file_sha256": file_sha256(summary_path),
        "source_metrics_sha256": payload["source_metrics"]["sha256"],
        "raw_metrics_reverified_in_this_run": raw_verified,
        "references": summed["references"],
        "robust_top15_hits": summed["robust_top15_hits"],
        "v2_3_hits": summed["v2_3_union_hits"],
        "hits": summed["v2_4_counterfactual_hits"],
        "misses": summed["references"] - summed["v2_4_counterfactual_hits"],
        "by_source_group": by_group,
        "evaluation_note": (
            "This tracked summary was derived from the frozen 1040-row metrics before the "
            "full-artifact counterfactual. Generated images are outside the full index."
        ),
    }


def natural_positive_keys(
    reference_rows: Sequence[Mapping[str, str]],
    result_rows: Sequence[Mapping[str, str]],
    index_by_relative: Mapping[str, int],
) -> tuple[list[str], list[str], np.ndarray]:
    result_by_id = {row["pair_id"]: row for row in result_rows}
    if len(result_by_id) != len(result_rows):
        raise ValueError("Natural result file contains duplicate Pair IDs.")
    positive_rows = [row for row in reference_rows if row["final_reference"] == "same_source"]
    pair_ids: list[str] = []
    source_groups: list[str] = []
    left_indices: list[int] = []
    right_indices: list[int] = []
    for reference in positive_rows:
        pair_id = reference["pair_id"]
        if pair_id not in result_by_id:
            raise ValueError(f"Natural positive is absent from unblinded results: {pair_id}")
        result = result_by_id[pair_id]
        if reference["source_group"] != result["source_group"]:
            raise ValueError(f"Natural source-group mismatch for {pair_id}.")
        try:
            left = index_by_relative[result["left_path"]]
            right = index_by_relative[result["right_path"]]
        except KeyError as error:
            raise ValueError(f"Natural reference path is absent from the full index: {error}") from error
        pair_ids.append(pair_id)
        source_groups.append(reference["source_group"])
        left_indices.append(min(left, right))
        right_indices.append(max(left, right))
    keys = encode_pair_keys(np.asarray(left_indices), np.asarray(right_indices))
    return pair_ids, source_groups, keys


def source_group_report(
    group: str,
    image_count: int,
    robust: np.ndarray,
    phash: np.ndarray,
    exact: np.ndarray,
    baseline: np.ndarray,
    proposal: np.ndarray,
    phash_enabled: bool,
) -> dict[str, object]:
    dropped = np.setdiff1d(baseline, proposal, assume_unique=True)
    restored_exact = np.intersect1d(exact, proposal, assume_unique=True)
    return {
        "image_count": image_count,
        "phash_enabled": phash_enabled,
        "robust_unique_pairs": int(len(robust)),
        "phash_unique_pairs": int(len(phash)),
        "exact_pairs_in_group": int(len(exact)),
        "v2_3_union_pairs": int(len(baseline)),
        "v2_4_counterfactual_pairs": int(len(proposal)),
        "dropped_v2_3_pairs": int(len(dropped)),
        "exact_pairs_present_in_proposal": int(len(restored_exact)),
        "v2_4_pairs_per_image": len(proposal) / image_count,
    }


def analyze(
    output_dir: Path,
    exact_path: Path,
    config_path: Path,
    natural_reference_path: Path,
    natural_result_path: Path,
    controlled_summary_path: Path,
    controlled_metrics_path: Path | None = None,
) -> dict[str, object]:
    config, config_hash = load_frozen_config(config_path)
    rule = config["counterfactual_rule"]
    gates = config["predeclared_acceptance_gates"]
    all_groups = set(rule["robust_top15_source_groups"])  # type: ignore[index]
    phash_groups = set(rule["phash_radius12_source_groups"])  # type: ignore[index]
    disabled_groups = set(rule["phash_disabled_source_groups"])  # type: ignore[index]
    if phash_groups & disabled_groups or phash_groups | disabled_groups != all_groups:
        raise ValueError("Configured pHash enabled/disabled groups do not partition all groups.")

    dependency_paths = {
        "research/natural_duplicate_adjudication_v1/final_pair_references_v1.csv": natural_reference_path,
        "research/natural_duplicate_audit_v1/unblinded_pair_results.csv": natural_result_path,
        "research/candidate_generator_v2_4_controlled_reference_summary.json": controlled_summary_path,
    }
    for relative, path in dependency_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Integrity dependency not found: {path}")
        expected = config["integrity_dependencies"][relative]  # type: ignore[index]
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Integrity dependency hash mismatch for {relative}: "
                f"expected={expected}, actual={actual}"
            )

    manifest_path = output_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: Mapping[str, Mapping[str, object]] = manifest["artifacts"]
    stage_counts = dict(sorted(Counter(str(info["stage"]) for info in artifacts.values()).items()))
    if stage_counts != dict(sorted(EXPECTED_STAGE_COUNTS.items())):
        raise ValueError(f"Unexpected v2.3 stage inventory: {stage_counts}")
    if manifest["contract_sha256"] != config["base_rule"]["full_run_contract_sha256"]:  # type: ignore[index]
        raise ValueError("Full retrieval run-contract hash does not match the predeclared config.")
    if len(artifacts) != int(gates["expected_registered_v2_3_artifacts"]):  # type: ignore[index]
        raise ValueError("Registered v2.3 artifact count does not match the predeclared gate.")

    index_relative = "stages/image_index.csv"
    index_path = verify_registered_artifact(output_dir, index_relative, artifacts[index_relative])
    groups, index_by_relative = read_index(index_path)
    source_groups = set(groups)
    if len(groups) != int(gates["expected_image_count"]) or source_groups != all_groups:  # type: ignore[index]
        raise ValueError(
            f"Full index scope mismatch: images={len(groups)}, groups={sorted(source_groups)}"
        )

    normalized_exact, normalized_exact_hash = verify_exact_duplicate_manifest(exact_path)
    exact_keys = exact_duplicate_pair_keys(normalized_exact, index_by_relative)
    if len(exact_keys) != int(gates["expected_exact_duplicate_pairs"]):  # type: ignore[index]
        raise ValueError(f"Unexpected exact-pair count: {len(exact_keys)}")
    exact_left, exact_right = decode_pair_keys(exact_keys)
    if any(
        groups[int(left)] != groups[int(right)]
        for left, right in zip(exact_left, exact_right, strict=True)
    ):
        raise ValueError("Exact manifest contains a cross-source-group pair.")

    group_reports: dict[str, object] = {}
    baseline_parts: list[np.ndarray] = []
    proposal_parts: list[np.ndarray] = []
    verified_candidate_artifacts = 1
    for group in sorted(all_groups):
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
        robust = unique_pair_keys(
            load_registered_pair_parts(output_dir, artifacts, robust_relatives)
        )
        phash = unique_pair_keys(
            load_registered_pair_parts(output_dir, artifacts, phash_relatives)
        )
        verified_candidate_artifacts += len(robust_relatives) + len(phash_relatives)
        group_indices = np.flatnonzero(np.asarray(groups) == group)
        group_exact_mask = np.isin(exact_left, group_indices, assume_unique=False)
        group_exact = exact_keys[group_exact_mask]
        baseline, proposal = proposed_group_union(
            robust, phash, group_exact, group in phash_groups
        )
        validate_group_pairs(baseline, group, groups)
        validate_group_pairs(proposal, group, groups)
        image_count = groups.count(group)
        group_reports[group] = source_group_report(
            group,
            image_count,
            robust,
            phash,
            group_exact,
            baseline,
            proposal,
            group in phash_groups,
        )
        baseline_parts.append(baseline)
        proposal_parts.append(proposal)

    baseline_union = unique_pair_keys(baseline_parts)
    proposal_union = unique_pair_keys(proposal_parts)
    exact_hits, exact_missing_indices = membership_count(exact_keys, proposal_union)

    natural_references = read_csv(natural_reference_path)
    natural_results = read_csv(natural_result_path)
    natural_ids, natural_groups, natural_keys = natural_positive_keys(
        natural_references, natural_results, index_by_relative
    )
    natural_hits, natural_missing_indices = membership_count(natural_keys, proposal_union)
    baseline_natural_hits, _ = membership_count(natural_keys, baseline_union)
    natural_missing_ids = [natural_ids[index] for index in natural_missing_indices]
    natural_by_group: dict[str, dict[str, int]] = {}
    for group in sorted(all_groups):
        positions = [index for index, value in enumerate(natural_groups) if value == group]
        group_keys = natural_keys[positions]
        group_hits, _ = membership_count(group_keys, proposal_union)
        natural_by_group[group] = {"references": len(positions), "hits": group_hits}

    controlled = evaluate_controlled_summary(
        controlled_summary_path,
        all_groups,
        disabled_groups,
        controlled_metrics_path,
    )

    if len(baseline_union) != int(gates["expected_v2_3_union_pairs"]):  # type: ignore[index]
        raise ValueError(
            f"v2.3 union does not match the verified baseline: {len(baseline_union)}"
        )
    if len(natural_ids) != int(gates["expected_natural_same_source_references"]):  # type: ignore[index]
        raise ValueError(f"Unexpected natural same-source reference count: {len(natural_ids)}")
    if int(controlled["references"]) != int(gates["expected_controlled_positive_references"]):  # type: ignore[index]
        raise ValueError(
            f"Unexpected controlled-positive reference count: {controlled['references']}"
        )

    pairs_per_image = len(proposal_union) / len(groups)
    cross_group_pairs = 0
    gate_results = {
        "exact_duplicate_recall": exact_hits == int(gates["required_exact_duplicate_hits"]),  # type: ignore[index]
        "natural_same_source_recall": natural_hits
        == int(gates["required_natural_same_source_hits"]),  # type: ignore[index]
        "controlled_positive_recall": controlled["hits"]
        == int(gates["required_controlled_positive_hits"]),  # type: ignore[index]
        "candidate_load": pairs_per_image
        <= float(gates["maximum_candidate_pairs_per_image"]),  # type: ignore[index]
        "cross_source_group_pairs": cross_group_pairs
        == int(gates["required_cross_source_group_pairs"]),  # type: ignore[index]
    }
    all_passed = all(gate_results.values())

    return {
        "schema_version": 1,
        "status": "candidate_generator_v2_4_counterfactual_evaluation_complete",
        "experiment_type": "read_only_predeclared_modality_aware_ablation",
        "does_not_modify_v2_3_artifacts": True,
        "not_a_duplicate_decision": True,
        "does_not_authorize_family_split_or_training": True,
        "config": {
            "path_name": config_path.name,
            "sha256": config_hash,
        },
        "v2_3_input": {
            "run_contract_sha256": manifest["contract_sha256"],
            "registered_artifact_count": len(artifacts),
            "stage_counts": stage_counts,
            "hash_verified_candidate_artifact_count": verified_candidate_artifacts,
            "image_count": len(groups),
            "source_group_count": len(source_groups),
        },
        "candidate_load": {
            "v2_3_union_pairs": int(len(baseline_union)),
            "v2_3_pairs_per_image": len(baseline_union) / len(groups),
            "v2_4_counterfactual_pairs": int(len(proposal_union)),
            "v2_4_pairs_per_image": pairs_per_image,
            "removed_pairs": int(len(np.setdiff1d(baseline_union, proposal_union, assume_unique=True))),
            "pause_gate_pairs_per_image": float(gates["maximum_candidate_pairs_per_image"]),  # type: ignore[index]
        },
        "exact_duplicate_reference": {
            "normalized_manifest_sha256": normalized_exact_hash,
            "references": int(len(exact_keys)),
            "hits": exact_hits,
            "misses": len(exact_missing_indices),
        },
        "natural_same_source_reference": {
            "reference_file_sha256": file_sha256(natural_reference_path),
            "unblinded_results_sha256": file_sha256(natural_result_path),
            "references": len(natural_ids),
            "v2_3_hits": baseline_natural_hits,
            "v2_4_hits": natural_hits,
            "misses": len(natural_missing_ids),
            "missing_pair_ids": natural_missing_ids,
            "by_source_group": natural_by_group,
            "limitation": "AI-assisted non-expert sampled reference; not proof of full natural recall.",
        },
        "controlled_positive_reference": {
            **controlled,
        },
        "by_source_group": group_reports,
        "acceptance_gates": {
            **gate_results,
            "all_passed": all_passed,
        },
        "decision": (
            "PASS: the predeclared v2.4 counterfactual passed all reference and capacity "
            "gates; review is required before freezing or producing final candidates."
            if all_passed
            else "FAIL: keep v2.3 paused; the predeclared v2.4 counterfactual missed at "
            "least one acceptance gate."
        ),
    }


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    if is_inside(report_path, output_dir):
        raise ValueError("Report must be outside the immutable v2.3 output directory.")
    report = analyze(
        output_dir=output_dir,
        exact_path=Path(args.exact_duplicates).resolve(),
        config_path=Path(args.config).resolve(),
        natural_reference_path=Path(args.natural_references).resolve(),
        natural_result_path=Path(args.natural_results).resolve(),
        controlled_summary_path=Path(args.controlled_summary).resolve(),
        controlled_metrics_path=(
            Path(args.controlled_metrics).resolve() if args.controlled_metrics else None
        ),
    )
    atomic_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "by_source_group"}, indent=2))
    print("By source group:")
    for group, summary in report["by_source_group"].items():  # type: ignore[union-attr]
        print(
            f"  {group}: v2.3={summary['v2_3_union_pairs']}, "  # type: ignore[index]
            f"v2.4={summary['v2_4_counterfactual_pairs']}, "  # type: ignore[index]
            f"pairs/image={summary['v2_4_pairs_per_image']:.3f}, "  # type: ignore[index]
            f"pHash={'on' if summary['phash_enabled'] else 'off'}"  # type: ignore[index]
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
