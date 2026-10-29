from __future__ import annotations

"""Build the second prospective controlled holdout for Candidate Generator v2.2.

The builder does not run retrieval, create duplicate families, create classifier
splits, or train a model. It excludes every dataset path used in the scaled pilot
and v2.1 holdout, plus every exact-duplicate-group member.
"""

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping

from build_candidate_generator_v2_1_holdout import (
    IMAGE_FIELDS,
    PAIR_FIELDS,
    build_rows_and_images,
    exact_group_member_paths,
    prior_dataset_paths,
    read_json,
    select_holdout_records,
    validate_output_rows,
    write_csv,
)
from build_duplicate_calibration_set import SYNTHETIC_TRANSFORMS, load_label_audit_info
from data_index import discover_records, project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Candidate Generator v2.2 için scaled pilot, v2.1 holdout ve exact "
            "kopyalardan ayrılmış ikinci kontrollü holdout'u kurar."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--config", default=str(root / "config" / "candidate_generator_v2_2.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "candidate_generator_v2_2.sha256"),
    )
    parser.add_argument(
        "--taxonomy", default=str(root / "config" / "label_taxonomy.json")
    )
    parser.add_argument(
        "--audit-groups",
        default=str(root / "config" / "duplicate_audit_groups.json"),
    )
    parser.add_argument(
        "--exact-duplicates",
        default=str(root / "artifacts" / "audit" / "exact_duplicates.json"),
    )
    parser.add_argument(
        "--scaled-index",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generation_scaled_v1"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--v2-1-index",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_1_holdout"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_2_holdout"
        ),
    )
    return parser.parse_args()


def verify_frozen_config(config_path: Path, hash_path: Path) -> Mapping[str, object]:
    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = file_sha256(config_path)
    if actual != expected:
        raise ValueError(
            "Dondurulmuş Candidate Generator v2.2 config hash'i eşleşmiyor: "
            f"expected={expected}, actual={actual}"
        )
    config = read_json(config_path)
    if config.get("status") != "frozen_before_second_prospective_controlled_holdout_evaluation":
        raise ValueError("Config ikinci prospective holdout için dondurulmuş değil.")
    rule = dict(config["candidate_rule"])
    robust = dict(rule["robust_retrieval"])
    phash = dict(rule["phash_retrieval"])
    if int(robust["top_k"]) != 15 or int(phash["hamming_radius_inclusive"]) != 8:
        raise ValueError("V2.2 frozen kuralı top-15 ve pHash<=8 olmalıdır.")
    return config


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    config_path = Path(args.config).resolve()
    config_hash_path = Path(args.config_sha256).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    groups_path = Path(args.audit_groups).resolve()
    exact_path = Path(args.exact_duplicates).resolve()
    scaled_index_path = Path(args.scaled_index).resolve()
    v21_index_path = Path(args.v2_1_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"V2.2 holdout zaten var; üzerine yazılmadı: {output_dir}")

    config = verify_frozen_config(config_path, config_hash_path)
    design = dict(config["holdout_design"])
    transforms = [str(value) for value in design["controlled_transforms"]]
    if tuple(transforms) != SYNTHETIC_TRANSFORMS:
        raise ValueError("Dondurulan transform listesi uygulama listesiyle eşleşmiyor.")

    dataset_root = resolve_dataset_root(args.data_dir)
    label_info = load_label_audit_info(taxonomy_path, groups_path)
    source_group_by_leaf = {
        leaf: info.source_group for leaf, info in label_info.items()
    }
    scaled_paths = prior_dataset_paths(scaled_index_path)
    v21_paths = prior_dataset_paths(v21_index_path)
    excluded_prior_paths = scaled_paths | v21_paths
    exact_paths = exact_group_member_paths(exact_path)
    records = discover_records(dataset_root)
    sources, distractors = select_holdout_records(
        records,
        dataset_root,
        sorted(label_info),
        excluded_prior_paths,
        exact_paths,
        int(design["source_images_per_leaf"]),
        int(design["distractors_per_leaf"]),
        int(design["seed"]),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        image_rows, pair_rows = build_rows_and_images(
            staging,
            dataset_root,
            sources,
            distractors,
            source_group_by_leaf,
            transforms,
        )
        for index, row in enumerate(pair_rows, start=1):
            row["pair_id"] = f"V22H{index:05d}"
        validate_output_rows(
            image_rows,
            pair_rows,
            config,
            excluded_prior_paths,
            exact_paths,
        )
        selected_dataset_paths = {
            str(row["relative_path"])
            for row in image_rows
            if row["root_kind"] == "dataset"
        }
        if selected_dataset_paths & scaled_paths or selected_dataset_paths & v21_paths:
            raise ValueError("V2.2 holdout önceki deney yollarıyla kesişiyor.")

        image_path = staging / "image_index.csv"
        pair_path = staging / "positive_pairs.csv"
        write_csv(image_path, IMAGE_FIELDS, image_rows)
        write_csv(pair_path, PAIR_FIELDS, pair_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "second_prospective_controlled_holdout_built_not_evaluated",
            "purpose": "Candidate Generator v2.2 second source-disjoint controlled retrieval holdout",
            "counts": {
                "images": len(image_rows),
                "source_originals": sum(
                    row["pool_role"] == "source_original" for row in image_rows
                ),
                "controlled_transforms": sum(
                    row["pool_role"] == "controlled_transform" for row in image_rows
                ),
                "distractors": sum(
                    row["pool_role"] == "distractor_unlabelled" for row in image_rows
                ),
                "positive_pairs": len(pair_rows),
                "leaves": len({str(row["leaf"]) for row in image_rows}),
                "source_groups": len(
                    {str(row["source_group"]) for row in image_rows}
                ),
            },
            "leaf_role_counts": {
                leaf: dict(
                    sorted(
                        Counter(
                            str(row["pool_role"])
                            for row in image_rows
                            if row["leaf"] == leaf
                        ).items()
                    )
                )
                for leaf in sorted(label_info)
            },
            "disjointness": {
                "scaled_pilot_dataset_path_count": len(scaled_paths),
                "v2_1_holdout_dataset_path_count": len(v21_paths),
                "combined_prior_dataset_path_count": len(excluded_prior_paths),
                "exact_duplicate_group_member_path_count": len(exact_paths),
                "selected_dataset_path_overlap_with_scaled_pilot": 0,
                "selected_dataset_path_overlap_with_v2_1_holdout": 0,
                "selected_dataset_path_overlap_with_exact_duplicate_groups": 0,
                "warning": "Dataset patient IDs/provenance are unavailable; patient independence cannot be proved.",
            },
            "integrity": {
                "frozen_config_sha256": file_sha256(config_path),
                "scaled_index_sha256": file_sha256(scaled_index_path),
                "v2_1_index_sha256": file_sha256(v21_index_path),
                "exact_duplicate_audit_sha256": file_sha256(exact_path),
                "image_index_sha256": file_sha256(image_path),
                "positive_pairs_sha256": file_sha256(pair_path),
                "builder_script_sha256": file_sha256(Path(__file__).resolve()),
                "v2_1_builder_dependency_sha256": file_sha256(
                    Path(__file__).with_name("build_candidate_generator_v2_1_holdout.py")
                ),
            },
            "limitations": list(config["interpretation_limits"]),
        }
        (staging / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("İşlem türü: SECOND PROSPECTIVE CONTROLLED RETRIEVAL HOLDOUT BUILD")
    print(f"Holdout görüntüsü: {len(image_rows)}")
    print(f"Bilinen aynı-kaynak çifti: {len(pair_rows)}")
    print("Scaled pilot yol kesişimi: 0")
    print("V2.1 holdout yol kesişimi: 0")
    print("Exact-duplicate grup üyesi kesişimi: 0")
    print(f"Holdout: {output_dir}")
    print("NOT: V2.2 retrieval henüz çalıştırılmadı; family/split/model yok.")
    return output_dir / "build_report.json", report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
