from __future__ import annotations

"""Build a prospective controlled holdout for Candidate Generator v2.1.

This script only selects new source images, generates known transformations and
adds unlabelled distractors. It never runs the candidate generator, creates
duplicate families, creates classifier splits, or trains a model.
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from build_duplicate_calibration_set import (
    SYNTHETIC_TRANSFORMS,
    apply_controlled_transform,
    load_label_audit_info,
    transform_spec,
)
from data_index import ImageRecord, discover_records, project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


IMAGE_FIELDS = (
    "image_index",
    "image_key",
    "root_kind",
    "relative_path",
    "leaf",
    "source_group",
    "pool_role",
    "source_id",
    "transform_name",
)
PAIR_FIELDS = (
    "pair_id",
    "left_key",
    "right_key",
    "leaf",
    "source_group",
    "source_id",
    "transform_name",
    "transform_parameters",
    "truth_label",
    "truth_basis",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Candidate Generator v2.1 için önceki indeks ve exact kopyalardan "
            "ayrılmış kontrollü holdout kurar; retrieval/model/split çalıştırmaz."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--config", default=str(root / "config" / "candidate_generator_v2_1.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "candidate_generator_v2_1.sha256"),
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
        "--prior-index",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generation_scaled_v1"
            / "image_index.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_1_holdout"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_posix(record: ImageRecord, dataset_root: Path) -> str:
    return record.path.relative_to(dataset_root).as_posix()


def verify_frozen_config(config_path: Path, hash_path: Path) -> Mapping[str, object]:
    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = file_sha256(config_path)
    if actual != expected:
        raise ValueError(
            "Dondurulmuş Candidate Generator v2.1 config hash'i eşleşmiyor: "
            f"expected={expected}, actual={actual}"
        )
    config = read_json(config_path)
    if config.get("status") != "frozen_before_prospective_controlled_holdout_evaluation":
        raise ValueError("Config dondurulmuş prospective holdout durumunda değil.")
    return config


def prior_dataset_paths(prior_index_path: Path) -> set[str]:
    paths = {
        row["relative_path"].replace("\\", "/")
        for row in read_csv(prior_index_path)
        if row["root_kind"].strip() == "dataset"
    }
    if not paths:
        raise ValueError("Önceki scaled indeksinden dataset yolu okunamadı.")
    return paths


def exact_group_member_paths(exact_path: Path) -> set[str]:
    report = read_json(exact_path)
    paths: set[str] = set()
    for group_value in report.get("duplicate_groups", []):
        group = dict(group_value)
        paths.update(str(path).replace("\\", "/") for path in group["paths"])
    return paths


def select_holdout_records(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    leaves: Sequence[str],
    excluded_prior_paths: set[str],
    exact_group_paths: set[str],
    sources_per_leaf: int,
    distractors_per_leaf: int,
    seed: int,
) -> tuple[dict[str, list[ImageRecord]], dict[str, list[ImageRecord]]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        path = relative_posix(record, dataset_root)
        if path in excluded_prior_paths or path in exact_group_paths:
            continue
        grouped[record.subclass].append(record)

    selected_sources: dict[str, list[ImageRecord]] = {}
    selected_distractors: dict[str, list[ImageRecord]] = {}
    needed = sources_per_leaf + distractors_per_leaf
    for leaf in sorted(leaves):
        candidates = sorted(grouped.get(leaf, []), key=lambda item: str(item.path))
        random.Random(f"{seed}:{leaf}").shuffle(candidates)
        if len(candidates) < needed:
            raise ValueError(
                f"{leaf} için {needed} yeni görüntü gerekli, bulunan={len(candidates)}"
            )
        selected_sources[leaf] = candidates[:sources_per_leaf]
        selected_distractors[leaf] = candidates[sources_per_leaf:needed]
    return selected_sources, selected_distractors


def build_rows_and_images(
    staging_root: Path,
    dataset_root: Path,
    selected_sources: Mapping[str, Sequence[ImageRecord]],
    selected_distractors: Mapping[str, Sequence[ImageRecord]],
    source_group_by_leaf: Mapping[str, str],
    transforms: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    def add_image(
        *,
        key: str,
        root_kind: str,
        relative_path: str,
        leaf: str,
        role: str,
        source_id: str = "",
        transform_name: str = "",
    ) -> None:
        image_rows.append(
            {
                "image_index": len(image_rows),
                "image_key": key,
                "root_kind": root_kind,
                "relative_path": relative_path,
                "leaf": leaf,
                "source_group": source_group_by_leaf[leaf],
                "pool_role": role,
                "source_id": source_id,
                "transform_name": transform_name,
            }
        )

    for leaf in sorted(selected_sources):
        for source_number, record in enumerate(selected_sources[leaf], start=1):
            source_relative = relative_posix(record, dataset_root)
            source_id = f"{leaf}_S{source_number:02d}"
            source_key = f"dataset:{source_relative}"
            add_image(
                key=source_key,
                root_kind="dataset",
                relative_path=source_relative,
                leaf=leaf,
                role="source_original",
                source_id=source_id,
            )
            with Image.open(record.path) as handle:
                original = handle.convert("RGB")
                for transform_name in transforms:
                    generated = apply_controlled_transform(original, transform_name)
                    generated_relative = (
                        Path("generated")
                        / leaf
                        / f"{source_id}__{transform_name}.png"
                    ).as_posix()
                    generated_path = staging_root / generated_relative
                    generated_path.parent.mkdir(parents=True, exist_ok=True)
                    generated.save(generated_path, format="PNG", optimize=False)
                    generated_key = f"holdout:{generated_relative}"
                    add_image(
                        key=generated_key,
                        root_kind="holdout",
                        relative_path=generated_relative,
                        leaf=leaf,
                        role="controlled_transform",
                        source_id=source_id,
                        transform_name=transform_name,
                    )
                    pair_rows.append(
                        {
                            "pair_id": f"V21H{len(pair_rows) + 1:05d}",
                            "left_key": source_key,
                            "right_key": generated_key,
                            "leaf": leaf,
                            "source_group": source_group_by_leaf[leaf],
                            "source_id": source_id,
                            "transform_name": transform_name,
                            "transform_parameters": json.dumps(
                                transform_spec(transform_name),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "truth_label": "same_source",
                            "truth_basis": "generated_from_left_image",
                        }
                    )

    for leaf in sorted(selected_distractors):
        for record in selected_distractors[leaf]:
            relative = relative_posix(record, dataset_root)
            add_image(
                key=f"dataset:{relative}",
                root_kind="dataset",
                relative_path=relative,
                leaf=leaf,
                role="distractor_unlabelled",
            )
    return image_rows, pair_rows


def validate_output_rows(
    image_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    prior_paths: set[str],
    exact_paths: set[str],
) -> None:
    design = dict(config["holdout_design"])
    expected_images = int(design["expected_index_image_count"])
    expected_pairs = int(design["expected_controlled_positive_pair_count"])
    if len(image_rows) != expected_images or len(pair_rows) != expected_pairs:
        raise ValueError(
            f"Holdout kapsamı bozuk: images={len(image_rows)}/{expected_images}, "
            f"pairs={len(pair_rows)}/{expected_pairs}"
        )
    keys = [str(row["image_key"]) for row in image_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Holdout image_key değerleri benzersiz değil.")
    selected_dataset_paths = {
        str(row["relative_path"])
        for row in image_rows
        if row["root_kind"] == "dataset"
    }
    if selected_dataset_paths & prior_paths:
        raise ValueError("Yeni holdout önceki scaled indeks yollarıyla kesişiyor.")
    if selected_dataset_paths & exact_paths:
        raise ValueError("Yeni holdout exact-duplicate grup üyeleriyle kesişiyor.")


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    config_path = Path(args.config).resolve()
    config_hash_path = Path(args.config_sha256).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    groups_path = Path(args.audit_groups).resolve()
    exact_path = Path(args.exact_duplicates).resolve()
    prior_index_path = Path(args.prior_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Holdout klasörü zaten var; üzerine yazılmadı: {output_dir}")

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
    prior_paths = prior_dataset_paths(prior_index_path)
    exact_paths = exact_group_member_paths(exact_path)
    records = discover_records(dataset_root)
    sources, distractors = select_holdout_records(
        records,
        dataset_root,
        sorted(label_info),
        prior_paths,
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
        validate_output_rows(image_rows, pair_rows, config, prior_paths, exact_paths)
        image_path = staging / "image_index.csv"
        pair_path = staging / "positive_pairs.csv"
        write_csv(image_path, IMAGE_FIELDS, image_rows)
        write_csv(pair_path, PAIR_FIELDS, pair_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "prospective_controlled_holdout_built_not_evaluated",
            "purpose": "Candidate Generator v2.1 source-disjoint controlled retrieval holdout",
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
                "prior_scaled_dataset_path_count": len(prior_paths),
                "exact_duplicate_group_member_path_count": len(exact_paths),
                "selected_dataset_path_overlap_with_prior_scaled": 0,
                "selected_dataset_path_overlap_with_exact_duplicate_groups": 0,
                "warning": "Dataset patient IDs/provenance are unavailable; patient independence cannot be proved.",
            },
            "integrity": {
                "frozen_config_sha256": file_sha256(config_path),
                "prior_scaled_index_sha256": file_sha256(prior_index_path),
                "exact_duplicate_audit_sha256": file_sha256(exact_path),
                "image_index_sha256": file_sha256(image_path),
                "positive_pairs_sha256": file_sha256(pair_path),
                "builder_script_sha256": file_sha256(Path(__file__).resolve()),
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

    print("İşlem türü: PROSPECTIVE CONTROLLED RETRIEVAL HOLDOUT BUILD")
    print(f"Holdout görüntüsü: {len(image_rows)}")
    print(f"Bilinen aynı-kaynak çifti: {len(pair_rows)}")
    print("Önceki yol kesişimi: 0")
    print("Exact-duplicate grup üyesi kesişimi: 0")
    print(f"Holdout: {output_dir}")
    print("NOT: Candidate retrieval henüz çalıştırılmadı; family/split/model yok.")
    return output_dir / "build_report.json", report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
