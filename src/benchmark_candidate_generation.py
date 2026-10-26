from __future__ import annotations

"""Benchmark scalable near-duplicate candidate generation on calibration images.

This is a retrieval/capacity pilot, not a full 130,000-image scan and not a duplicate
decision rule. It compares flip-aware pHash radii with top-k ImageNet ResNet18 embedding
retrieval, then measures how often known same-source calibration pairs enter each shortlist.
"""

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from audit_perceptual_duplicates import HASH_SIZE, dct_matrix
from data_index import project_root
from duplicate_decision_rule import file_sha256
from render_calibration_review import read_dataset_root, resolve_image_path


VISUAL_TO_REFERENCE = {
    "same_source_candidate": "same_source",
    "independent_lookalike": "independent",
    "uncertain": "uncertain",
}
PHASH_RADII = (8, 12, 16, 24, 32)
TOPK_VALUES = (1, 5, 10, 20, 50)
PAIR_METRIC_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "reference_label",
    "reference_basis",
    "modality_key",
    "source_group",
    "left_leaf",
    "right_leaf",
    "transform_name",
    "flip_aware_phash_distance",
    "resnet18_cosine_similarity",
    "resnet18_union_rank",
    "resnet18_worst_direction_rank",
)


@dataclass(frozen=True)
class ImageItem:
    key: str
    root_kind: str
    relative_path: str
    source_group: str
    absolute_path: Path


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Kalibrasyon görüntülerinde flip-aware pHash ve ResNet18 top-k aday üretimini "
            "karşılaştırır; tam veri taraması/family/split/model eğitimi yapmaz."
        )
    )
    parser.add_argument("--calibration-root", default=str(calibration_root))
    parser.add_argument("--pairs-csv", default=str(calibration_root / "pairs.csv"))
    parser.add_argument(
        "--development-decisions",
        default=str(root / "research" / "development_visual_review_v1.csv"),
    )
    parser.add_argument(
        "--holdout-decisions",
        default=str(root / "research" / "holdout_visual_review_v1.csv"),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(root / "artifacts" / "audit" / "candidate_generation_benchmark_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmalıdır.")
    if args.workers < 1:
        raise ValueError("--workers en az 1 olmalıdır.")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def load_reference_pairs(
    pairs_path: Path,
    development_decisions_path: Path,
    holdout_decisions_path: Path,
) -> list[dict[str, str]]:
    pairs = read_csv(pairs_path)
    decisions = read_csv(development_decisions_path) + read_csv(holdout_decisions_path)
    decision_by_id: dict[str, str] = {}
    for row in decisions:
        pair_id = row.get("pair_id", "").strip()
        verdict = row.get("visual_verdict", "").strip()
        if not pair_id or pair_id in decision_by_id:
            raise ValueError(f"Boş veya tekrarlanan review Pair ID: {pair_id!r}")
        if verdict not in VISUAL_TO_REFERENCE:
            raise ValueError(f"Geçersiz görsel karar: {pair_id}={verdict!r}")
        decision_by_id[pair_id] = VISUAL_TO_REFERENCE[verdict]

    required_review_ids = {
        row["pair_id"]
        for row in pairs
        if row.get("truth_status", "").strip() == "review_required"
    }
    if set(decision_by_id) != required_review_ids:
        missing = sorted(required_review_ids - set(decision_by_id))
        extra = sorted(set(decision_by_id) - required_review_ids)
        raise ValueError(f"Review kapsamı bozuk; eksik={missing}, fazla={extra}")

    output: list[dict[str, str]] = []
    for source in pairs:
        row = dict(source)
        if row.get("truth_status", "").strip() == "same_source":
            row["reference_label"] = "same_source"
            row["reference_basis"] = row.get("truth_basis", "known_positive")
        elif row.get("truth_status", "").strip() == "review_required":
            row["reference_label"] = decision_by_id[row["pair_id"]]
            row["reference_basis"] = "ai_assisted_visual_review"
        else:
            raise ValueError(f"Desteklenmeyen truth_status: {row.get('pair_id')}")
        output.append(row)
    return output


def image_key(row: Mapping[str, str], side: str) -> str:
    return f"{row[f'{side}_root'].strip()}:{row[f'{side}_path'].strip()}"


def collect_image_items(
    rows: Sequence[Mapping[str, str]], dataset_root: Path, calibration_root: Path
) -> list[ImageItem]:
    by_key: dict[str, ImageItem] = {}
    for row in rows:
        for side in ("left", "right"):
            key = image_key(row, side)
            item = ImageItem(
                key=key,
                root_kind=row[f"{side}_root"].strip(),
                relative_path=row[f"{side}_path"].strip(),
                source_group=row["source_group"].strip(),
                absolute_path=resolve_image_path(row, side, dataset_root, calibration_root),
            )
            existing = by_key.get(key)
            if existing is not None and existing != item:
                raise ValueError(f"Aynı görüntü anahtarı tutarsız metadata taşıyor: {key}")
            by_key[key] = item
    return sorted(by_key.values(), key=lambda item: item.key)


def phash_from_image(image: Image.Image, transform: np.ndarray) -> int:
    side = HASH_SIZE * 4
    pixels = np.asarray(
        image.convert("L").resize((side, side), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    coefficients = transform @ pixels @ transform.T
    low_frequency = coefficients[:HASH_SIZE, :HASH_SIZE].reshape(-1)
    threshold = float(np.median(low_frequency[1:]))
    value = 0
    for bit in low_frequency > threshold:
        value = (value << 1) | int(bit)
    return value


def compute_phash_variants(
    items: Sequence[ImageItem], workers: int
) -> tuple[np.ndarray, np.ndarray]:
    transform = dct_matrix(HASH_SIZE * 4)

    def one(item: ImageItem) -> tuple[int, int]:
        with Image.open(item.absolute_path) as image:
            rgb = image.convert("RGB")
            original = phash_from_image(rgb, transform)
            flipped = phash_from_image(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), transform)
        return original, flipped

    values: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(one, items), start=1):
            values.append(result)
            if index % 250 == 0 or index == len(items):
                print(f"pHash varyantları: {index}/{len(items)}", flush=True)
    return (
        np.asarray([value[0] for value in values], dtype=np.uint64),
        np.asarray([value[1] for value in values], dtype=np.uint64),
    )


class CalibrationImageDataset(Dataset[torch.Tensor]):
    def __init__(self, items: Sequence[ImageItem], preprocess: object) -> None:
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.items[index].absolute_path) as image:
            tensor = self.preprocess(image.convert("RGB"))  # type: ignore[operator]
        return tensor


def compute_resnet18_embeddings(
    items: Sequence[ImageItem], batch_size: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        CalibrationImageDataset(items, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    originals: list[np.ndarray] = []
    flips: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            joined = torch.cat((batch, torch.flip(batch, dims=(-1,))), dim=0)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = model(joined)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            split = batch.shape[0]
            originals.append(features[:split].cpu().numpy())
            flips.append(features[split:].cpu().numpy())
            processed = min(batch_index * batch_size, len(items))
            print(f"ResNet18 embedding: {processed}/{len(items)}", flush=True)

    checkpoint_name = Path(weights.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    metadata: dict[str, object] = {
        "architecture": "torchvision_resnet18_fc_identity",
        "weights": str(weights),
        "feature_dimension": 512,
        "orientation_variants": ["original", "horizontal_flip"],
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return np.concatenate(originals), np.concatenate(flips), metadata


def canonical_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def group_indices(items: Sequence[ImageItem]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item.source_group].append(index)
    return dict(sorted(groups.items()))


def flip_aware_hamming(
    left: int, right: int, original: np.ndarray, flipped: np.ndarray
) -> int:
    return min(
        int(int(original[left] ^ original[right]).bit_count()),
        int(int(original[left] ^ flipped[right]).bit_count()),
        int(int(flipped[left] ^ original[right]).bit_count()),
        int(int(flipped[left] ^ flipped[right]).bit_count()),
    )


def phash_candidate_sets(
    items: Sequence[ImageItem], original: np.ndarray, flipped: np.ndarray
) -> dict[int, set[tuple[int, int]]]:
    candidates = {radius: set() for radius in PHASH_RADII}
    for source_group, indices in group_indices(items).items():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                distance = flip_aware_hamming(left, right, original, flipped)
                for radius in PHASH_RADII:
                    if distance <= radius:
                        candidates[radius].add((left, right))
        print(f"pHash aday grubu tamamlandı: {source_group} ({len(indices)} görüntü)")
    return candidates


def similarity_matrices(
    items: Sequence[ImageItem], original: np.ndarray, flipped: np.ndarray
) -> dict[str, tuple[list[int], np.ndarray]]:
    output: dict[str, tuple[list[int], np.ndarray]] = {}
    for source_group, indices in group_indices(items).items():
        first = original[indices]
        second = flipped[indices]
        similarity = np.maximum.reduce(
            (
                first @ first.T,
                first @ second.T,
                second @ first.T,
                second @ second.T,
            )
        )
        np.fill_diagonal(similarity, -np.inf)
        output[source_group] = (indices, similarity)
    return output


def topk_candidate_sets(
    matrices: Mapping[str, tuple[list[int], np.ndarray]]
) -> dict[int, set[tuple[int, int]]]:
    candidates = {value: set() for value in TOPK_VALUES}
    for indices, similarity in matrices.values():
        group_size = len(indices)
        for row_index, global_left in enumerate(indices):
            order = np.argsort(-similarity[row_index], kind="stable")
            for topk in TOPK_VALUES:
                for local_right in order[: min(topk, group_size - 1)]:
                    candidates[topk].add(
                        canonical_pair(global_left, indices[int(local_right)])
                    )
    return candidates


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_candidate_set(
    candidate_pairs: set[tuple[int, int]],
    rows: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    image_count: int,
) -> dict[str, object]:
    labelled_pairs = []
    for row in rows:
        left = index_by_key[image_key(row, "left")]
        right = index_by_key[image_key(row, "right")]
        labelled_pairs.append((row, canonical_pair(left, right)))
    positives = [item for item in labelled_pairs if item[0]["reference_label"] == "same_source"]
    controlled = [item for item in positives if item[0]["truth_status"] == "same_source"]
    natural = [item for item in positives if item[0]["truth_status"] == "review_required"]
    negatives = [item for item in labelled_pairs if item[0]["reference_label"] == "independent"]

    def hits(examples: Sequence[tuple[Mapping[str, str], tuple[int, int]]]) -> int:
        return sum(pair in candidate_pairs for _, pair in examples)

    by_transform: dict[str, dict[str, object]] = {}
    transform_groups: dict[str, list[tuple[Mapping[str, str], tuple[int, int]]]] = defaultdict(list)
    for item in controlled:
        key = item[0].get("transform_name", "").strip() or item[0]["pair_kind"]
        transform_groups[key].append(item)
    for name, examples in sorted(transform_groups.items()):
        count = len(examples)
        hit_count = hits(examples)
        by_transform[name] = {
            "positive_count": count,
            "candidate_hits": hit_count,
            "candidate_recall": safe_rate(hit_count, count),
        }

    return {
        "candidate_pair_count": len(candidate_pairs),
        "candidate_pairs_per_indexed_image": safe_rate(len(candidate_pairs), image_count),
        "positive_count": len(positives),
        "positive_hits": hits(positives),
        "positive_recall": safe_rate(hits(positives), len(positives)),
        "controlled_positive_count": len(controlled),
        "controlled_positive_hits": hits(controlled),
        "controlled_positive_recall": safe_rate(hits(controlled), len(controlled)),
        "natural_positive_count": len(natural),
        "natural_positive_hits": hits(natural),
        "natural_positive_recall": safe_rate(hits(natural), len(natural)),
        "reviewed_independent_count": len(negatives),
        "reviewed_independent_candidate_hits": hits(negatives),
        "reviewed_independent_candidate_rate": safe_rate(hits(negatives), len(negatives)),
        "controlled_positive_recall_by_transform": by_transform,
    }


def pair_metric_rows(
    rows: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    items: Sequence[ImageItem],
    phash_original: np.ndarray,
    phash_flipped: np.ndarray,
    matrices: Mapping[str, tuple[list[int], np.ndarray]],
) -> list[dict[str, str]]:
    matrix_lookup: dict[str, tuple[dict[int, int], np.ndarray]] = {}
    for group, (indices, matrix) in matrices.items():
        matrix_lookup[group] = ({value: offset for offset, value in enumerate(indices)}, matrix)

    output: list[dict[str, str]] = []
    for source in rows:
        left = index_by_key[image_key(source, "left")]
        right = index_by_key[image_key(source, "right")]
        group = items[left].source_group
        positions, matrix = matrix_lookup[group]
        left_local = positions[left]
        right_local = positions[right]
        similarity = float(matrix[left_local, right_local])
        left_rank = 1 + int(np.count_nonzero(matrix[left_local] > similarity))
        right_rank = 1 + int(np.count_nonzero(matrix[right_local] > similarity))
        row = {
            field: str(source.get(field, ""))
            for field in PAIR_METRIC_FIELDS
            if field not in {
                "flip_aware_phash_distance",
                "resnet18_cosine_similarity",
                "resnet18_union_rank",
                "resnet18_worst_direction_rank",
            }
        }
        row.update(
            {
                "flip_aware_phash_distance": str(
                    flip_aware_hamming(left, right, phash_original, phash_flipped)
                ),
                "resnet18_cosine_similarity": f"{similarity:.8f}",
                "resnet18_union_rank": str(min(left_rank, right_rank)),
                "resnet18_worst_direction_rank": str(max(left_rank, right_rank)),
            }
        )
        output.append(row)
    return output


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    calibration_root = Path(args.calibration_root).resolve()
    pairs_path = Path(args.pairs_csv).resolve()
    development_decisions = Path(args.development_decisions).resolve()
    holdout_decisions = Path(args.holdout_decisions).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "report.json"
    pair_metrics_path = output_dir / "pair_candidate_metrics.csv"
    index_path = output_dir / "image_index.csv"
    embeddings_path = output_dir / "resnet18_embeddings.npz"
    if any(path.exists() for path in (report_path, pair_metrics_path, index_path, embeddings_path)):
        raise FileExistsError(
            "Benchmark çıktısı zaten var; bilinçsiz yeniden çalışma engellendi: "
            f"{output_dir}"
        )

    dataset_root = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else read_dataset_root(calibration_root)
    )
    rows = load_reference_pairs(pairs_path, development_decisions, holdout_decisions)
    items = collect_image_items(rows, dataset_root, calibration_root)
    index_by_key = {item.key: index for index, item in enumerate(items)}

    print("İşlem türü: candidate-generation ölçeklenebilirlik PİLOTU")
    print(f"Kalibrasyon çifti: {len(rows)}")
    print(f"Benzersiz görüntü: {len(items)}")
    print("Tam 130.000 görüntü taranmayacak; family/split/model üretilmeyecek.")
    started = time.perf_counter()
    phash_original, phash_flipped = compute_phash_variants(items, args.workers)
    embedding_original, embedding_flipped, embedding_metadata = compute_resnet18_embeddings(
        items, args.batch_size
    )
    matrices = similarity_matrices(items, embedding_original, embedding_flipped)
    phash_sets = phash_candidate_sets(items, phash_original, phash_flipped)
    topk_sets = topk_candidate_sets(matrices)

    phash_results = {
        str(radius): evaluate_candidate_set(
            phash_sets[radius], rows, index_by_key, len(items)
        )
        for radius in PHASH_RADII
    }
    topk_results = {
        str(topk): {
            **evaluate_candidate_set(topk_sets[topk], rows, index_by_key, len(items)),
            "full_dataset_directed_pair_upper_bound": 130_000 * topk,
        }
        for topk in TOPK_VALUES
    }
    metrics = pair_metric_rows(
        rows,
        index_by_key,
        items,
        phash_original,
        phash_flipped,
        matrices,
    )

    index_rows = [
        {
            "image_index": str(index),
            "image_key": item.key,
            "root_kind": item.root_kind,
            "relative_path": item.relative_path,
            "source_group": item.source_group,
        }
        for index, item in enumerate(items)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_path,
        original=embedding_original.astype(np.float32),
        horizontal_flip=embedding_flipped.astype(np.float32),
    )
    atomic_write_csv(
        index_path,
        ("image_index", "image_key", "root_kind", "relative_path", "source_group"),
        index_rows,
    )
    atomic_write_csv(pair_metrics_path, PAIR_METRIC_FIELDS, metrics)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "candidate_generation_calibration_pilot_not_full_dataset",
        "experiment_type": "retrieval_candidate_generation_capacity_pilot",
        "input": {
            "pair_count": len(rows),
            "unique_image_count": len(items),
            "source_group_image_counts": dict(
                sorted(Counter(item.source_group for item in items).items())
            ),
            "reference_counts": dict(
                sorted(Counter(row["reference_label"] for row in rows).items())
            ),
        },
        "phash": {
            "description": "64-bit DCT pHash; original/horizontal-flip minimum Hamming distance",
            "radii": list(PHASH_RADII),
            "results": phash_results,
        },
        "resnet18_embedding_retrieval": {
            "metadata": embedding_metadata,
            "topk_values": list(TOPK_VALUES),
            "results": topk_results,
        },
        "parameters": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "retrieval_scope": "within compatible source_group only",
        },
        "integrity": {
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "pairs_csv_sha256": file_sha256(pairs_path),
            "development_decisions_sha256": file_sha256(development_decisions),
            "holdout_decisions_sha256": file_sha256(holdout_decisions),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This benchmark indexes only 1,748 calibration images, not all 130,000 images.",
            "Candidate counts measured on the calibration pool do not directly predict full-dataset density.",
            "The prior holdout has already been opened for rule evaluation; these retrieval comparisons are pilot evidence, not a new untouched test.",
            "AI-assisted natural-pair labels are not an independent clinical-expert gold standard.",
            "Candidate generation only makes a shortlist; it does not replace the frozen duplicate decision rule.",
        ],
    }
    atomic_write_json(report_path, report)
    print(f"Aday benchmark raporu: {report_path}")
    print(f"Çift ölçümleri: {pair_metrics_path}")
    print(f"Görüntü indeksi: {index_path}")
    print(f"Embeddingler: {embeddings_path}")
    print("NOT: Bu pilot sonucu family veya split değildir.")
    return report_path, pair_metrics_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
