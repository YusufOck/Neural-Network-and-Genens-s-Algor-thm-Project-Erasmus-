from __future__ import annotations

"""Build the frozen path-exclusion manifest for the prospective v2.5 audit.

This reads metadata only. It never decodes an image, assigns a visual verdict,
scores a duplicate, constructs a family/split, or trains a classifier.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from candidate_generator_v2_3_runtime import atomic_json
from duplicate_decision_rule import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v2.5 prospective-audit path exclusions from frozen local metadata."
    )
    parser.add_argument(
        "--v2-1-index",
        default=str(
            ROOT
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_1_holdout"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--v2-2-index",
        default=str(
            ROOT
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_2_holdout"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--scaled-index",
        default=str(
            ROOT
            / "artifacts"
            / "audit"
            / "candidate_generation_scaled_v1"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--calibration-pairs",
        default=str(
            ROOT / "artifacts" / "audit" / "duplicate_calibration_v1" / "pairs.csv"
        ),
    )
    parser.add_argument(
        "--exact-duplicates",
        default=str(ROOT / "artifacts" / "audit" / "exact_duplicates.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "research" / "candidate_generator_v2_5_excluded_paths_v1.json"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def normalized_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    if not path or path.startswith("/") or ":/" in path:
        raise ValueError(f"Expected a dataset-relative path, got: {value!r}")
    return path


def index_dataset_paths(path: Path) -> set[str]:
    rows = read_csv(path)
    required = {"root_kind", "relative_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected image-index schema: {path}")
    return {
        normalized_path(row["relative_path"])
        for row in rows
        if row["root_kind"] == "dataset"
    }


def calibration_dataset_paths(path: Path) -> set[str]:
    rows = read_csv(path)
    required = {"left_root", "left_path", "right_root", "right_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected calibration-pair schema: {path}")
    output: set[str] = set()
    for row in rows:
        if row["left_root"] == "dataset":
            output.add(normalized_path(row["left_path"]))
        if row["right_root"] == "dataset":
            output.add(normalized_path(row["right_path"]))
    return output


def exact_group_paths(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalized_path(str(relative))
        for group in payload["duplicate_groups"]
        for relative in group["paths"]
    }


def canonical_paths_sha256(paths: Iterable[str]) -> str:
    encoded = "\n".join(sorted(set(paths))).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    v2_1_index: Path,
    v2_2_index: Path,
    scaled_index: Path,
    calibration_pairs: Path,
    exact_duplicates: Path,
) -> dict[str, object]:
    source_paths: dict[str, set[str]] = {
        "v2_1_holdout_dataset": index_dataset_paths(v2_1_index),
        "v2_2_holdout_and_prior_natural_audit_pool": index_dataset_paths(v2_2_index),
        "scaled_candidate_pilot_dataset": index_dataset_paths(scaled_index),
        "calibration_dataset": calibration_dataset_paths(calibration_pairs),
        "exact_duplicate_members": exact_group_paths(exact_duplicates),
    }
    source_files = {
        "v2_1_holdout_dataset": v2_1_index,
        "v2_2_holdout_and_prior_natural_audit_pool": v2_2_index,
        "scaled_candidate_pilot_dataset": scaled_index,
        "calibration_dataset": calibration_pairs,
        "exact_duplicate_members": exact_duplicates,
    }
    union = sorted(set().union(*source_paths.values()))
    return {
        "schema_version": 1,
        "status": "frozen_before_v2_5_prospective_pair_selection",
        "experiment_type": "metadata_only_path_exclusion_manifest",
        "does_not_open_images": True,
        "excluded_path_count": len(union),
        "excluded_paths_sha256": canonical_paths_sha256(union),
        "sources": {
            name: {
                "path_count": len(source_paths[name]),
                "paths_sha256": canonical_paths_sha256(source_paths[name]),
                "source_file_sha256": file_sha256(source_files[name]),
            }
            for name in sorted(source_paths)
        },
        "excluded_paths": union,
    }


def main() -> None:
    args = parse_args()
    inputs = {
        "v2_1_index": Path(args.v2_1_index).resolve(),
        "v2_2_index": Path(args.v2_2_index).resolve(),
        "scaled_index": Path(args.scaled_index).resolve(),
        "calibration_pairs": Path(args.calibration_pairs).resolve(),
        "exact_duplicates": Path(args.exact_duplicates).resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    payload = build_manifest(**inputs)
    output = Path(args.output).resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
