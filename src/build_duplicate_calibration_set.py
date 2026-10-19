from __future__ import annotations

"""Build a reviewable, all-26-leaf calibration set for duplicate detection.

This is an audit utility. It does not create the classifier's train/validation/test
split and it does not declare unreviewed image pairs to be independent. Known
positives come only from SHA-256 equality or controlled transformations created by
this script. Random, pHash-hard and cross-leaf pairs remain review candidates.
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from audit_perceptual_duplicates import dct_matrix, perceptual_hash
from data_index import ImageRecord, discover_records, project_root, resolve_dataset_root


AUDIT_PARTITIONS = ("development", "holdout")
SYNTHETIC_TRANSFORMS = (
    "rotate_minus_10",
    "rotate_plus_10",
    "shift_plus10_minus10",
    "shear_factor_plus_0_10",
    "zoom_out_0_90",
    "zoom_in_1_10",
    "horizontal_flip",
    "brightness_0_20",
    "brightness_1_20",
    "combined_stress",
)
PAIR_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "truth_status",
    "truth_basis",
    "left_leaf",
    "right_leaf",
    "modality_key",
    "source_group",
    "left_root",
    "left_path",
    "right_root",
    "right_path",
    "source_key",
    "transform_name",
    "transform_parameters",
    "sha256_equal",
    "phash_distance",
    "human_verdict",
    "review_notes",
)


@dataclass(frozen=True)
class LabelAuditInfo:
    leaf: str
    modality_key: str
    modality_name: str
    source_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "26 alt sınıfın tamamı için kontrollü pozitifler ve insan incelemeli "
            "negatif adaylar üretir. Model split'i veya eğitim yapmaz."
        )
    )
    parser.add_argument("--data-dir", default=None, help="Multi Cancer veri kökü.")
    parser.add_argument(
        "--taxonomy",
        default=str(project_root() / "config" / "label_taxonomy.json"),
        help="26 yaprak etiketini içeren taxonomy JSON dosyası.",
    )
    parser.add_argument(
        "--audit-groups",
        default=str(project_root() / "config" / "duplicate_audit_groups.json"),
        help="Modalite ve uyumlu kaynak grubu eşlemesi.",
    )
    parser.add_argument(
        "--exact-duplicates",
        default=str(project_root() / "artifacts" / "audit" / "exact_duplicates.json"),
        help="Tam kapsamlı SHA-256 audit raporu.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            project_root() / "artifacts" / "audit" / "duplicate_calibration_v1"
        ),
        help="Yeni calibration-set v1 klasörü. Var olan klasör ezilmez.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Tekrar üretilebilir seçim tohumu.")
    parser.add_argument(
        "--sources-per-leaf",
        type=int,
        default=2,
        help="Her audit partition'ında, her yaprak için sentetik kaynak sayısı.",
    )
    parser.add_argument(
        "--random-pairs-per-leaf",
        type=int,
        default=2,
        help="Her partition/yaprak için rastgele insan-inceleme çifti.",
    )
    parser.add_argument(
        "--hard-pairs-per-leaf",
        type=int,
        default=2,
        help="Her partition/yaprak için pHash-benzer insan-inceleme çifti.",
    )
    parser.add_argument(
        "--hard-candidate-pool",
        type=int,
        default=64,
        help="Her partition/yaprakta pHash-hard çift aramak için görüntü havuzu.",
    )
    parser.add_argument(
        "--exact-pairs-per-leaf",
        type=int,
        default=4,
        help="SHA grubu bulunan her yapraktan en çok kaç exact pozitif çift alınacağı.",
    )
    parser.add_argument(
        "--cross-pairs-per-label-pair",
        type=int,
        default=1,
        help="Aynı kaynak grubundaki her farklı yaprak çifti için, partition başına aday sayısı.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="pHash hesaplamasında kullanılacak iş parçacığı sayısı.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "sources_per_leaf",
        "hard_candidate_pool",
        "workers",
    )
    nonnegative_names = (
        "random_pairs_per_leaf",
        "hard_pairs_per_leaf",
        "exact_pairs_per_leaf",
        "cross_pairs_per_label_pair",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} en az 1 olmalıdır.")
    for name in nonnegative_names:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} negatif olamaz.")
    if args.hard_pairs_per_leaf * 2 > args.hard_candidate_pool:
        raise ValueError(
            "--hard-candidate-pool, hard çiftlerde görüntü tekrarını önlemek için "
            "--hard-pairs-per-leaf değerinin en az iki katı olmalıdır."
        )


def read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Gerekli JSON dosyası bulunamadı: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_label_audit_info(
    taxonomy_path: Path, groups_path: Path
) -> Dict[str, LabelAuditInfo]:
    taxonomy = read_json(taxonomy_path)
    groups = read_json(groups_path)
    taxonomy_labels = set(dict(taxonomy.get("labels", {})))

    result: Dict[str, LabelAuditInfo] = {}
    for source_group, payload_value in dict(groups.get("source_groups", {})).items():
        payload = dict(payload_value)
        modality_key = str(payload["modality_key"])
        modality_name = str(payload["modality_name"])
        for leaf_value in payload.get("labels", []):
            leaf = str(leaf_value)
            if leaf in result:
                raise ValueError(f"Audit grup eşlemesinde tekrar eden yaprak: {leaf}")
            result[leaf] = LabelAuditInfo(
                leaf=leaf,
                modality_key=modality_key,
                modality_name=modality_name,
                source_group=str(source_group),
            )

    mapped_labels = set(result)
    if mapped_labels != taxonomy_labels:
        raise ValueError(
            "Taxonomy ve duplicate audit grupları aynı yaprakları içermiyor. "
            f"Eksik={sorted(taxonomy_labels - mapped_labels)}, "
            f"fazla={sorted(mapped_labels - taxonomy_labels)}"
        )
    if len(result) != 26:
        raise ValueError(f"Tam 26 yaprak bekleniyordu; bulunan={len(result)}")
    return result


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_exact_groups(
    exact_path: Path,
    records_by_relative: Mapping[str, ImageRecord],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, str]]:
    report = read_json(exact_path)
    groups_by_leaf: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    exact_group_by_path: Dict[str, str] = {}

    for group in report.get("duplicate_groups", []):
        group_payload = dict(group)
        group_id = str(group_payload["sha256"])
        normalized_paths = [str(path).replace("\\", "/") for path in group_payload["paths"]]
        records = [records_by_relative[path] for path in normalized_paths]
        leaves = {record.subclass for record in records}
        if len(leaves) != 1:
            raise ValueError(
                f"Exact grup tek yaprakta değil: {group_id}, leaves={sorted(leaves)}"
            )
        leaf = next(iter(leaves))
        if len(normalized_paths) >= 2:
            groups_by_leaf[leaf].append((group_id, "|".join(normalized_paths)))
        for path in normalized_paths:
            if path in exact_group_by_path:
                raise ValueError(f"Dosya iki exact grupta görünüyor: {path}")
            exact_group_by_path[path] = group_id
    return dict(groups_by_leaf), exact_group_by_path


def split_leaf_records(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    exact_group_by_path: Mapping[str, str],
    rng: random.Random,
) -> Dict[str, List[ImageRecord]]:
    available = [
        record
        for record in records
        if relative_posix(record.path, dataset_root) not in exact_group_by_path
    ]
    rng.shuffle(available)
    midpoint = len(available) // 2
    if midpoint == 0:
        raise ValueError("Development/holdout için yeterli exact-dışı görüntü yok.")
    return {
        "development": available[:midpoint],
        "holdout": available[midpoint:],
    }


def edge_fill_color(image: Image.Image) -> Tuple[int, int, int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    medians = np.median(border, axis=0).round().astype(np.uint8)
    return tuple(int(value) for value in medians)


def affine_shift(image: Image.Image, dx_fraction: float, dy_fraction: float) -> Image.Image:
    width, height = image.size
    dx = dx_fraction * width
    dy = dy_fraction * height
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -dx, 0.0, 1.0, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=edge_fill_color(image),
    )


def affine_shear(image: Image.Image, factor: float) -> Image.Image:
    _, height = image.size
    center_y = (height - 1) / 2.0
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, -factor, factor * center_y, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=edge_fill_color(image),
    )


def zoom_image(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    if factor <= 0:
        raise ValueError("Zoom factor pozitif olmalıdır.")

    if factor >= 1.0:
        new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
        resized = image.resize(new_size, Image.Resampling.BILINEAR)
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", image.size, edge_fill_color(image))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def transform_spec(name: str) -> Mapping[str, object]:
    specs: Dict[str, Mapping[str, object]] = {
        "rotate_minus_10": {"rotation_degrees": -10.0},
        "rotate_plus_10": {"rotation_degrees": 10.0},
        "shift_plus10_minus10": {"width_shift_fraction": 0.10, "height_shift_fraction": -0.10},
        "shear_factor_plus_0_10": {"x_shear_factor": 0.10},
        "zoom_out_0_90": {"zoom_factor": 0.90},
        "zoom_in_1_10": {"zoom_factor": 1.10},
        "horizontal_flip": {"horizontal_flip": True},
        "brightness_0_20": {"brightness_factor": 0.20},
        "brightness_1_20": {"brightness_factor": 1.20},
        "combined_stress": {
            "rotation_degrees": 7.0,
            "width_shift_fraction": 0.08,
            "height_shift_fraction": -0.08,
            "x_shear_factor": 0.08,
            "zoom_factor": 1.08,
            "horizontal_flip": True,
            "brightness_factor": 0.70,
        },
    }
    return specs[name]


def apply_controlled_transform(image: Image.Image, name: str) -> Image.Image:
    result = image.convert("RGB")
    spec = transform_spec(name)

    if "rotation_degrees" in spec:
        result = result.rotate(
            float(spec["rotation_degrees"]),
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=edge_fill_color(result),
        )
    if "width_shift_fraction" in spec or "height_shift_fraction" in spec:
        result = affine_shift(
            result,
            float(spec.get("width_shift_fraction", 0.0)),
            float(spec.get("height_shift_fraction", 0.0)),
        )
    if "x_shear_factor" in spec:
        result = affine_shear(result, float(spec["x_shear_factor"]))
    if "zoom_factor" in spec:
        result = zoom_image(result, float(spec["zoom_factor"]))
    if spec.get("horizontal_flip"):
        result = ImageOps.mirror(result)
    if "brightness_factor" in spec:
        result = ImageEnhance.Brightness(result).enhance(float(spec["brightness_factor"]))
    return result


def blank_pair(
    *,
    partition: str,
    pair_kind: str,
    truth_status: str,
    truth_basis: str,
    left_leaf: str,
    right_leaf: str,
    info: LabelAuditInfo,
    left_root: str,
    left_path: str,
    right_root: str,
    right_path: str,
    source_key: str = "",
    transform_name: str = "",
    transform_parameters: str = "",
    sha256_equal: str = "no",
) -> Dict[str, str]:
    return {
        "pair_id": "",
        "audit_partition": partition,
        "pair_kind": pair_kind,
        "truth_status": truth_status,
        "truth_basis": truth_basis,
        "left_leaf": left_leaf,
        "right_leaf": right_leaf,
        "modality_key": info.modality_key,
        "source_group": info.source_group,
        "left_root": left_root,
        "left_path": left_path,
        "right_root": right_root,
        "right_path": right_path,
        "source_key": source_key,
        "transform_name": transform_name,
        "transform_parameters": transform_parameters,
        "sha256_equal": sha256_equal,
        "phash_distance": "",
        "human_verdict": "" if truth_status == "review_required" else truth_status,
        "review_notes": "",
    }


def prepare_exact_rows(
    groups_by_leaf: Mapping[str, Sequence[Tuple[str, str]]],
    label_info: Mapping[str, LabelAuditInfo],
    limit_per_leaf: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if limit_per_leaf == 0:
        return rows

    for leaf in sorted(label_info):
        groups = list(groups_by_leaf.get(leaf, []))
        rng.shuffle(groups)
        for index, (group_id, paths_text) in enumerate(groups[:limit_per_leaf]):
            paths = paths_text.split("|")
            partition = AUDIT_PARTITIONS[index % len(AUDIT_PARTITIONS)]
            rows.append(
                blank_pair(
                    partition=partition,
                    pair_kind="sha256_exact_positive",
                    truth_status="same_source",
                    truth_basis="sha256_equal",
                    left_leaf=leaf,
                    right_leaf=leaf,
                    info=label_info[leaf],
                    left_root="dataset",
                    left_path=paths[0],
                    right_root="dataset",
                    right_path=paths[1],
                    source_key=f"sha256:{group_id}",
                    sha256_equal="yes",
                )
            )
    return rows


def generate_synthetic_rows(
    selections: Mapping[Tuple[str, str], Sequence[ImageRecord]],
    label_info: Mapping[str, LabelAuditInfo],
    dataset_root: Path,
    staging_root: Path,
    sources_per_leaf: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for partition in AUDIT_PARTITIONS:
        for leaf in sorted(label_info):
            sources = list(selections[(partition, leaf)])[:sources_per_leaf]
            for source in sources:
                source_relative = relative_posix(source.path, dataset_root)
                with Image.open(source.path) as image_handle:
                    original = image_handle.convert("RGB")
                    for transform_name in SYNTHETIC_TRANSFORMS:
                        generated = apply_controlled_transform(original, transform_name)
                        output_relative = Path("generated") / partition / leaf / (
                            f"{source.path.stem}__{transform_name}.png"
                        )
                        output_path = staging_root / output_relative
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        generated.save(output_path, format="PNG", optimize=False)
                        rows.append(
                            blank_pair(
                                partition=partition,
                                pair_kind="controlled_synthetic_positive",
                                truth_status="same_source",
                                truth_basis="generated_from_left_image",
                                left_leaf=leaf,
                                right_leaf=leaf,
                                info=label_info[leaf],
                                left_root="dataset",
                                left_path=source_relative,
                                right_root="artifact",
                                right_path=output_relative.as_posix(),
                                source_key=f"dataset:{source_relative}",
                                transform_name=transform_name,
                                transform_parameters=json.dumps(
                                    transform_spec(transform_name),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            )
                        )
    return rows


def make_random_review_rows(
    selections: Mapping[Tuple[str, str], Sequence[ImageRecord]],
    label_info: Mapping[str, LabelAuditInfo],
    dataset_root: Path,
    sources_per_leaf: int,
    pairs_per_leaf: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if pairs_per_leaf == 0:
        return rows

    start = sources_per_leaf
    for partition in AUDIT_PARTITIONS:
        for leaf in sorted(label_info):
            pool = list(selections[(partition, leaf)])
            chosen = pool[start : start + pairs_per_leaf * 2]
            if len(chosen) != pairs_per_leaf * 2:
                raise ValueError(f"Rastgele çift için yetersiz görüntü: {partition}/{leaf}")
            for pair_index in range(pairs_per_leaf):
                left = chosen[pair_index * 2]
                right = chosen[pair_index * 2 + 1]
                rows.append(
                    blank_pair(
                        partition=partition,
                        pair_kind="random_same_leaf_review",
                        truth_status="review_required",
                        truth_basis="random_pair_not_proven_independent",
                        left_leaf=leaf,
                        right_leaf=leaf,
                        info=label_info[leaf],
                        left_root="dataset",
                        left_path=relative_posix(left.path, dataset_root),
                        right_root="dataset",
                        right_path=relative_posix(right.path, dataset_root),
                    )
                )
    return rows


def compute_hash_map(paths: Iterable[Path], workers: int) -> Dict[Path, int]:
    unique_paths = sorted(set(paths), key=lambda path: str(path).lower())
    transform = dct_matrix(32)

    def compute(path: Path) -> Tuple[Path, int]:
        return path, perceptual_hash(path, transform)

    result: Dict[Path, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (path, value) in enumerate(executor.map(compute, unique_paths), start=1):
            result[path] = value
            if index % 500 == 0 or index == len(unique_paths):
                print(f"Kalibrasyon pHash: {index}/{len(unique_paths)}")
    return result


def nearest_nonoverlapping_pairs(
    records: Sequence[ImageRecord],
    hash_by_path: Mapping[Path, int],
    limit: int,
) -> List[Tuple[ImageRecord, ImageRecord, int]]:
    ranked: List[Tuple[int, str, str, ImageRecord, ImageRecord]] = []
    for left, right in combinations(records, 2):
        distance = (hash_by_path[left.path] ^ hash_by_path[right.path]).bit_count()
        ranked.append((distance, str(left.path), str(right.path), left, right))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))

    selected: List[Tuple[ImageRecord, ImageRecord, int]] = []
    used: set[Path] = set()
    for distance, _, _, left, right in ranked:
        if left.path in used or right.path in used:
            continue
        selected.append((left, right, distance))
        used.update((left.path, right.path))
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"İstenen {limit} non-overlap hard çift üretilemedi.")
    return selected


def make_hard_review_rows(
    selections: Mapping[Tuple[str, str], Sequence[ImageRecord]],
    label_info: Mapping[str, LabelAuditInfo],
    dataset_root: Path,
    start: int,
    candidate_pool_size: int,
    pairs_per_leaf: int,
    workers: int,
) -> Tuple[List[Dict[str, str]], Dict[Path, int]]:
    if pairs_per_leaf == 0:
        return [], {}

    pools: Dict[Tuple[str, str], List[ImageRecord]] = {}
    all_paths: List[Path] = []
    for partition in AUDIT_PARTITIONS:
        for leaf in sorted(label_info):
            pool = list(selections[(partition, leaf)])[start : start + candidate_pool_size]
            if len(pool) != candidate_pool_size:
                raise ValueError(f"Hard-negative havuzu yetersiz: {partition}/{leaf}")
            pools[(partition, leaf)] = pool
            all_paths.extend(record.path for record in pool)

    hash_by_path = compute_hash_map(all_paths, workers)
    rows: List[Dict[str, str]] = []
    for partition in AUDIT_PARTITIONS:
        for leaf in sorted(label_info):
            for left, right, distance in nearest_nonoverlapping_pairs(
                pools[(partition, leaf)], hash_by_path, pairs_per_leaf
            ):
                row = blank_pair(
                    partition=partition,
                    pair_kind="phash_hard_same_leaf_review",
                    truth_status="review_required",
                    truth_basis="low_phash_distance_not_ground_truth",
                    left_leaf=leaf,
                    right_leaf=leaf,
                    info=label_info[leaf],
                    left_root="dataset",
                    left_path=relative_posix(left.path, dataset_root),
                    right_root="dataset",
                    right_path=relative_posix(right.path, dataset_root),
                )
                row["phash_distance"] = str(distance)
                rows.append(row)
    return rows, hash_by_path


def make_cross_leaf_rows(
    selections: Mapping[Tuple[str, str], Sequence[ImageRecord]],
    label_info: Mapping[str, LabelAuditInfo],
    dataset_root: Path,
    start: int,
    pairs_per_label_pair: int,
) -> List[Dict[str, str]]:
    if pairs_per_label_pair == 0:
        return []

    leaves_by_group: Dict[str, List[str]] = defaultdict(list)
    for leaf, info in label_info.items():
        leaves_by_group[info.source_group].append(leaf)

    rows: List[Dict[str, str]] = []
    offsets: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    for partition in AUDIT_PARTITIONS:
        for source_group in sorted(leaves_by_group):
            leaves = sorted(leaves_by_group[source_group])
            for left_leaf, right_leaf in combinations(leaves, 2):
                for _ in range(pairs_per_label_pair):
                    left_pool = list(selections[(partition, left_leaf)])
                    right_pool = list(selections[(partition, right_leaf)])
                    left_index = start + offsets[(partition, left_leaf)]
                    right_index = start + offsets[(partition, right_leaf)]
                    if left_index >= len(left_pool) or right_index >= len(right_pool):
                        raise ValueError(
                            f"Çapraz-yaprak çift için yetersiz görüntü: {partition}/"
                            f"{left_leaf}/{right_leaf}"
                        )
                    left = left_pool[left_index]
                    right = right_pool[right_index]
                    offsets[(partition, left_leaf)] += 1
                    offsets[(partition, right_leaf)] += 1
                    rows.append(
                        blank_pair(
                            partition=partition,
                            pair_kind="cross_leaf_same_source_group_review",
                            truth_status="review_required",
                            truth_basis="different_leaf_same_source_group_not_ground_truth",
                            left_leaf=left_leaf,
                            right_leaf=right_leaf,
                            info=label_info[left_leaf],
                            left_root="dataset",
                            left_path=relative_posix(left.path, dataset_root),
                            right_root="dataset",
                            right_path=relative_posix(right.path, dataset_root),
                        )
                    )
    return rows


def resolve_pair_path(
    row: Mapping[str, str], side: str, dataset_root: Path, staging_root: Path
) -> Path:
    root_name = row[f"{side}_root"]
    relative = Path(row[f"{side}_path"])
    if root_name == "dataset":
        return dataset_root / relative
    if root_name == "artifact":
        return staging_root / relative
    raise ValueError(f"Bilinmeyen pair root: {root_name}")


def add_missing_phash_distances(
    rows: Sequence[MutableMapping[str, str]],
    dataset_root: Path,
    staging_root: Path,
    existing_hashes: Mapping[Path, int],
    workers: int,
) -> None:
    needed_paths: List[Path] = []
    for row in rows:
        needed_paths.append(resolve_pair_path(row, "left", dataset_root, staging_root))
        needed_paths.append(resolve_pair_path(row, "right", dataset_root, staging_root))

    hash_by_path = dict(existing_hashes)
    missing = [path for path in needed_paths if path not in hash_by_path]
    hash_by_path.update(compute_hash_map(missing, workers))

    for row in rows:
        left = resolve_pair_path(row, "left", dataset_root, staging_root)
        right = resolve_pair_path(row, "right", dataset_root, staging_root)
        row["phash_distance"] = str(
            (hash_by_path[left] ^ hash_by_path[right]).bit_count()
        )


def assign_pair_ids(rows: Sequence[MutableMapping[str, str]]) -> None:
    kind_order = {
        "sha256_exact_positive": 0,
        "controlled_synthetic_positive": 1,
        "random_same_leaf_review": 2,
        "phash_hard_same_leaf_review": 3,
        "cross_leaf_same_source_group_review": 4,
    }
    rows.sort(
        key=lambda row: (
            row["audit_partition"],
            kind_order[row["pair_kind"]],
            row["left_leaf"],
            row["right_leaf"],
            row["left_path"],
            row["right_path"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["pair_id"] = f"C{index:06d}"


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    label_info: Mapping[str, LabelAuditInfo],
) -> Mapping[str, object]:
    if not rows:
        raise ValueError("Kalibrasyon manifesti boş olamaz.")

    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Tekrarlanan pair_id bulundu.")

    pair_keys: set[Tuple[str, str, str]] = set()
    original_paths_by_partition: Dict[str, set[str]] = {
        partition: set() for partition in AUDIT_PARTITIONS
    }
    leaves_by_partition: Dict[str, set[str]] = {
        partition: set() for partition in AUDIT_PARTITIONS
    }
    exact_groups_by_partition: Dict[str, set[str]] = {
        partition: set() for partition in AUDIT_PARTITIONS
    }

    for row in rows:
        partition = row["audit_partition"]
        if partition not in AUDIT_PARTITIONS:
            raise ValueError(f"Bilinmeyen audit partition: {partition}")
        if row["left_leaf"] not in label_info or row["right_leaf"] not in label_info:
            raise ValueError(f"Bilinmeyen yaprak içeren çift: {row['pair_id']}")
        pair_key = (partition, row["left_path"], row["right_path"])
        if pair_key in pair_keys:
            raise ValueError(f"Tekrarlanan çift bulundu: {pair_key}")
        pair_keys.add(pair_key)

        leaves_by_partition[partition].update((row["left_leaf"], row["right_leaf"]))
        for side in ("left", "right"):
            if row[f"{side}_root"] == "dataset":
                original_paths_by_partition[partition].add(row[f"{side}_path"])
        if row["source_key"].startswith("sha256:"):
            exact_groups_by_partition[partition].add(row["source_key"])

        if row["truth_status"] == "same_source" and row["truth_basis"] not in {
            "sha256_equal",
            "generated_from_left_image",
        }:
            raise ValueError(
                f"Same-source ground truth kanıtsız görünüyor: {row['pair_id']}"
            )
        if row["truth_status"] == "review_required" and row["human_verdict"]:
            raise ValueError(
                f"İncelenmemiş aday önceden etiketlenmiş: {row['pair_id']}"
            )

    overlap = original_paths_by_partition["development"] & original_paths_by_partition["holdout"]
    if overlap:
        raise ValueError(
            "Aynı seçilmiş orijinal dosya development ve holdout'ta kullanılmış: "
            f"{sorted(overlap)[:5]}"
        )
    exact_overlap = exact_groups_by_partition["development"] & exact_groups_by_partition["holdout"]
    if exact_overlap:
        raise ValueError(
            "Aynı SHA-256 ailesi iki audit partition'ına bölünmüş: "
            f"{sorted(exact_overlap)[:5]}"
        )

    expected_leaves = set(label_info)
    for partition in AUDIT_PARTITIONS:
        missing = expected_leaves - leaves_by_partition[partition]
        if missing:
            raise ValueError(
                f"{partition} partition'ında eksik yapraklar var: {sorted(missing)}"
            )

    return {
        "selected_original_path_overlap_count": 0,
        "selected_exact_group_overlap_count": 0,
        "all_26_leaves_present_in_each_audit_partition": True,
    }


def counter_dict(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    args: argparse.Namespace,
    dataset_root: Path,
    rows: Sequence[Mapping[str, str]],
    label_info: Mapping[str, LabelAuditInfo],
    invariants: Mapping[str, object],
) -> Mapping[str, object]:
    review_rows = [row for row in rows if row["truth_status"] == "review_required"]
    positive_rows = [row for row in rows if row["truth_status"] == "same_source"]
    synthetic_rows = [
        row for row in rows if row["pair_kind"] == "controlled_synthetic_positive"
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Near-duplicate method calibration only; not a classifier data split.",
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "parameters": {
            "sources_per_leaf_per_partition": args.sources_per_leaf,
            "synthetic_transforms": list(SYNTHETIC_TRANSFORMS),
            "random_pairs_per_leaf_per_partition": args.random_pairs_per_leaf,
            "hard_pairs_per_leaf_per_partition": args.hard_pairs_per_leaf,
            "hard_candidate_pool_per_leaf_per_partition": args.hard_candidate_pool,
            "exact_pairs_per_leaf_maximum": args.exact_pairs_per_leaf,
            "cross_pairs_per_label_pair_per_partition": args.cross_pairs_per_label_pair,
        },
        "counts": {
            "total_pairs": len(rows),
            "known_positive_pairs": len(positive_rows),
            "review_required_pairs": len(review_rows),
            "generated_synthetic_images": len(synthetic_rows),
            "by_partition": counter_dict(row["audit_partition"] for row in rows),
            "by_pair_kind": counter_dict(row["pair_kind"] for row in rows),
            "by_modality": counter_dict(row["modality_key"] for row in rows),
            "by_source_group": counter_dict(row["source_group"] for row in rows),
            "by_left_leaf": counter_dict(row["left_leaf"] for row in rows),
            "synthetic_positive_by_transform": counter_dict(
                row["transform_name"] for row in synthetic_rows
            ),
        },
        "phash_distance_summary": {
            "controlled_synthetic_min": min(
                int(row["phash_distance"]) for row in synthetic_rows
            ),
            "controlled_synthetic_max": max(
                int(row["phash_distance"]) for row in synthetic_rows
            ),
            "controlled_synthetic_over_12": sum(
                int(row["phash_distance"]) > 12 for row in synthetic_rows
            ),
        },
        "invariants": dict(invariants),
        "label_count": len(label_info),
        "status": "generated_unreviewed_calibration_draft",
        "limitations": [
            "Random, pHash-hard and cross-leaf pairs are review candidates, not proven negatives.",
            "The audit development/holdout assignment is not the classifier 70/15/15 split.",
            "Selected original files are disjoint across audit partitions, but unknown natural augmentation families are not yet available and therefore cannot be guaranteed disjoint.",
            "PIL transformations are deterministic controlled surrogates within the README ranges; they are not claimed to reproduce Keras ImageDataGenerator pixel-for-pixel.",
            "No ORB, ECC, SSIM or family threshold is selected by this generator.",
        ],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_root = resolve_dataset_root(args.data_dir)
    taxonomy_path = Path(args.taxonomy).resolve()
    groups_path = Path(args.audit_groups).resolve()
    exact_path = Path(args.exact_duplicates).resolve()
    output_root = Path(args.output_dir).resolve()

    if output_root.exists():
        raise FileExistsError(
            f"Çıktı klasörü zaten var ve güvenlik için ezilmeyecek: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Geçici çıktı klasörü zaten var: {staging_root}")
    staging_root.mkdir(parents=False)

    try:
        label_info = load_label_audit_info(taxonomy_path, groups_path)
        records = discover_records(dataset_root)
        records_by_leaf: Dict[str, List[ImageRecord]] = defaultdict(list)
        records_by_relative: Dict[str, ImageRecord] = {}
        for record in records:
            relative = relative_posix(record.path, dataset_root)
            records_by_leaf[record.subclass].append(record)
            records_by_relative[relative] = record

        if set(records_by_leaf) != set(label_info):
            raise ValueError(
                "Yerel veri yaprakları ve audit config eşleşmiyor. "
                f"Veri-fazla={sorted(set(records_by_leaf) - set(label_info))}, "
                f"config-fazla={sorted(set(label_info) - set(records_by_leaf))}"
            )

        groups_by_leaf, exact_group_by_path = load_exact_groups(
            exact_path, records_by_relative
        )
        rng = random.Random(args.seed)
        selections: Dict[Tuple[str, str], List[ImageRecord]] = {}
        for leaf in sorted(label_info):
            partitioned = split_leaf_records(
                records_by_leaf[leaf], dataset_root, exact_group_by_path, rng
            )
            for partition, selected in partitioned.items():
                selections[(partition, leaf)] = selected

        required_per_partition_leaf = (
            args.sources_per_leaf
            + args.random_pairs_per_leaf * 2
            + args.hard_candidate_pool
            + max(len(info_group) - 1 for info_group in (
                [leaf for leaf, info in label_info.items() if info.source_group == group]
                for group in {info.source_group for info in label_info.values()}
            ))
            * args.cross_pairs_per_label_pair
        )
        for (partition, leaf), selected in selections.items():
            if len(selected) < required_per_partition_leaf:
                raise ValueError(
                    f"{partition}/{leaf} için en az {required_per_partition_leaf} exact-dışı "
                    f"görüntü gerekiyor; bulunan={len(selected)}"
                )

        rows: List[Dict[str, str]] = []
        rows.extend(
            prepare_exact_rows(
                groups_by_leaf,
                label_info,
                args.exact_pairs_per_leaf,
                rng,
            )
        )
        rows.extend(
            generate_synthetic_rows(
                selections,
                label_info,
                dataset_root,
                staging_root,
                args.sources_per_leaf,
            )
        )
        rows.extend(
            make_random_review_rows(
                selections,
                label_info,
                dataset_root,
                args.sources_per_leaf,
                args.random_pairs_per_leaf,
            )
        )

        hard_start = args.sources_per_leaf + args.random_pairs_per_leaf * 2
        hard_rows, hard_hashes = make_hard_review_rows(
            selections,
            label_info,
            dataset_root,
            hard_start,
            args.hard_candidate_pool,
            args.hard_pairs_per_leaf,
            args.workers,
        )
        rows.extend(hard_rows)

        cross_start = hard_start + args.hard_candidate_pool
        rows.extend(
            make_cross_leaf_rows(
                selections,
                label_info,
                dataset_root,
                cross_start,
                args.cross_pairs_per_label_pair,
            )
        )

        add_missing_phash_distances(
            rows,
            dataset_root,
            staging_root,
            hard_hashes,
            args.workers,
        )
        assign_pair_ids(rows)
        invariants = validate_rows(rows, label_info)

        write_csv(staging_root / "pairs.csv", rows)
        review_rows = [row for row in rows if row["truth_status"] == "review_required"]
        write_csv(staging_root / "review_candidates.csv", review_rows)
        report = build_report(args, dataset_root, rows, label_info, invariants)
        with (staging_root / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print("Calibration-set v1 üretildi (model split'i/eğitim yok).")
    print(f"Veri kökü: {dataset_root}")
    print(f"Çift manifesti: {output_root / 'pairs.csv'}")
    print(f"İnsan inceleme adayları: {output_root / 'review_candidates.csv'}")
    print(f"Rapor: {output_root / 'report.json'}")
    print(f"Toplam çift: {report['counts']['total_pairs']}")
    print(f"Bilinen pozitif: {report['counts']['known_positive_pairs']}")
    print(f"İnceleme gerekli: {report['counts']['review_required_pairs']}")
    print(
        "NOT: development/holdout yalnız duplicate yöntem kalibrasyonu içindir; "
        "70/15/15 sınıflandırma split'i değildir."
    )


if __name__ == "__main__":
    main()
