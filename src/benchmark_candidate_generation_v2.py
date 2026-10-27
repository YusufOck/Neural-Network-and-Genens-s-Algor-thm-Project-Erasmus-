from __future__ import annotations

"""Benchmark a brightness- and shift-robust duplicate candidate generator.

This is candidate-generator v2 on the existing calibration pool. It does not scan
the full dataset, decide duplicate families, create a classifier split, or train a
cancer classifier.
"""

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from benchmark_candidate_generation import (
    PHASH_RADII,
    TOPK_VALUES,
    ImageItem,
    atomic_write_csv,
    collect_image_items,
    compute_phash_variants,
    evaluate_candidate_set,
    group_indices,
    image_key,
    load_reference_pairs,
    phash_candidate_sets,
)
from data_index import project_root
from duplicate_decision_rule import file_sha256
from render_calibration_review import read_dataset_root


CROP_RATIO = 0.80
ACCEPTANCE_MAX_CANDIDATE_FRACTION = 0.25
DESCRIPTOR_NAMES = (
    "raw_full_flip_mean",
    "autocontrast_full_flip_mean",
    "raw_multicrop_flip_mean",
    "autocontrast_multicrop_flip_mean",
)
V2_PAIR_METRIC_FIELDS = (
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
    "v2_best_cosine_similarity",
    "v2_union_rank",
    "v2_best_descriptor",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Brightness/autocontrast ve multi-crop ResNet18 ile candidate-generator v2 "
            "kalibrasyon pilotu; tam tarama/family/split/model eğitimi yapmaz."
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
        default=str(root / "artifacts" / "audit" / "candidate_generation_benchmark_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmalıdır.")
    if args.workers < 1:
        raise ValueError("--workers en az 1 olmalıdır.")


def spatial_views(image: Image.Image, crop_ratio: float = CROP_RATIO) -> list[Image.Image]:
    """Return full plus five overlapping crops without altering the source image."""

    if not 0.5 <= crop_ratio < 1.0:
        raise ValueError("crop_ratio 0.5 ile 1.0 arasında olmalıdır.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    crop_width = max(1, round(width * crop_ratio))
    crop_height = max(1, round(height * crop_ratio))
    max_left = width - crop_width
    max_top = height - crop_height
    positions = (
        (max_left // 2, max_top // 2),
        (0, max_top // 2),
        (max_left, max_top // 2),
        (max_left // 2, 0),
        (max_left // 2, max_top),
    )
    return [rgb] + [
        rgb.crop((left, top, left + crop_width, top + crop_height))
        for left, top in positions
    ]


class RobustCalibrationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load raw and autocontrast full/multi-crop views for one image."""

    def __init__(self, items: Sequence[ImageItem], preprocess: object) -> None:
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def _tensors(self, image: Image.Image) -> torch.Tensor:
        return torch.stack(
            [self.preprocess(view) for view in spatial_views(image)], dim=0  # type: ignore[operator]
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.items[index].absolute_path) as image:
            raw = image.convert("RGB")
            normalized = ImageOps.autocontrast(raw, cutoff=1)
            return self._tensors(raw), self._tensors(normalized)


def normalize_mean(features: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    pooled = features.mean(dim=dimensions)
    return torch.nn.functional.normalize(pooled.float(), dim=1)


def descriptor_batch(features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Collapse [batch, photometric, spatial, orientation, feature] into four descriptors."""

    if features.ndim != 5 or features.shape[1:4] != (2, 6, 2):
        raise ValueError(f"Beklenmeyen feature şekli: {tuple(features.shape)}")
    return {
        "raw_full_flip_mean": normalize_mean(features[:, 0, 0:1], (1, 2)),
        "autocontrast_full_flip_mean": normalize_mean(features[:, 1, 0:1], (1, 2)),
        "raw_multicrop_flip_mean": normalize_mean(features[:, 0, 1:], (1, 2)),
        "autocontrast_multicrop_flip_mean": normalize_mean(
            features[:, 1, 1:], (1, 2)
        ),
    }


def compute_robust_descriptors(
    items: Sequence[ImageItem], batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        RobustCalibrationDataset(items, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in DESCRIPTOR_NAMES}
    with torch.inference_mode():
        for batch_index, (raw, normalized) in enumerate(loader, start=1):
            # [B, 2 photometric, 6 spatial, 3, H, W]
            views = torch.stack((raw, normalized), dim=1)
            batch_size_actual, photometric, spatial, channels, height, width = views.shape
            views = views.to(device, non_blocking=True)
            flipped = torch.flip(views, dims=(-1,))
            joined = torch.stack((views, flipped), dim=3)
            flat = joined.reshape(-1, channels, height, width)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                extracted = model(flat)
            extracted = torch.nn.functional.normalize(extracted.float(), dim=1)
            extracted = extracted.reshape(
                batch_size_actual, photometric, spatial, 2, -1
            )
            for name, values in descriptor_batch(extracted).items():
                collected[name].append(values.cpu().numpy())
            processed = min(batch_index * batch_size, len(items))
            print(f"V2 robust embedding: {processed}/{len(items)}", flush=True)

    descriptors = {
        name: np.concatenate(chunks).astype(np.float32)
        for name, chunks in collected.items()
    }
    checkpoint_name = Path(weights.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    metadata: dict[str, object] = {
        "architecture": "torchvision_resnet18_fc_identity",
        "weights": str(weights),
        "feature_dimension": 512,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "autocontrast_cutoff_percent": 1,
        "crop_ratio": CROP_RATIO,
        "spatial_views": ["full", "center", "left", "right", "top", "bottom"],
        "orientation_pool": ["original", "horizontal_flip"],
        "descriptors": list(DESCRIPTOR_NAMES),
    }
    return descriptors, metadata


def channel_similarity_matrices(
    items: Sequence[ImageItem], descriptors: Mapping[str, np.ndarray]
) -> dict[str, dict[str, tuple[list[int], np.ndarray]]]:
    output: dict[str, dict[str, tuple[list[int], np.ndarray]]] = {}
    for source_group, indices in group_indices(items).items():
        output[source_group] = {}
        for name in DESCRIPTOR_NAMES:
            selected = descriptors[name][indices]
            similarity = selected @ selected.T
            np.fill_diagonal(similarity, -np.inf)
            output[source_group][name] = (indices, similarity)
    return output


def multichannel_topk_candidate_sets(
    matrices: Mapping[str, Mapping[str, tuple[list[int], np.ndarray]]]
) -> dict[int, set[tuple[int, int]]]:
    candidates = {value: set() for value in TOPK_VALUES}
    for channels in matrices.values():
        for indices, similarity in channels.values():
            group_size = len(indices)
            for local_left, global_left in enumerate(indices):
                order = np.argsort(-similarity[local_left], kind="stable")
                for topk in TOPK_VALUES:
                    for local_right in order[: min(topk, group_size - 1)]:
                        left, right = global_left, indices[int(local_right)]
                        candidates[topk].add((left, right) if left < right else (right, left))
    return candidates


def total_possible_pairs(items: Sequence[ImageItem]) -> int:
    return sum(
        len(indices) * (len(indices) - 1) // 2
        for indices in group_indices(items).values()
    )


def hybrid_candidate_sets(
    topk_sets: Mapping[int, set[tuple[int, int]]],
    phash_sets: Mapping[int, set[tuple[int, int]]],
) -> dict[str, set[tuple[int, int]]]:
    output: dict[str, set[tuple[int, int]]] = {}
    for topk in (5, 10, 20, 50):
        for radius in (8, 12, 16, 24):
            output[f"top{topk}_union_phash{radius}"] = (
                topk_sets[topk] | phash_sets[radius]
            )
    return output


def best_pair_rank_and_similarity(
    left: int,
    right: int,
    source_group: str,
    matrices: Mapping[str, Mapping[str, tuple[list[int], np.ndarray]]],
) -> tuple[float, int, str]:
    best_similarity = -float("inf")
    best_rank = 10**9
    best_descriptor = ""
    for name, (indices, matrix) in matrices[source_group].items():
        positions = {value: offset for offset, value in enumerate(indices)}
        left_local = positions[left]
        right_local = positions[right]
        similarity = float(matrix[left_local, right_local])
        # Use the exact stable ordering used by candidate generation. Counting only
        # strictly greater values gives a misleading optimistic rank when several
        # images have identical cosine scores.
        left_order = np.argsort(-matrix[left_local], kind="stable")
        right_order = np.argsort(-matrix[right_local], kind="stable")
        left_rank = 1 + int(np.flatnonzero(left_order == right_local)[0])
        right_rank = 1 + int(np.flatnonzero(right_order == left_local)[0])
        union_rank = min(left_rank, right_rank)
        if union_rank < best_rank or (
            union_rank == best_rank and similarity > best_similarity
        ):
            best_similarity = similarity
            best_rank = union_rank
            best_descriptor = name
    return best_similarity, best_rank, best_descriptor


def pair_metric_rows_v2(
    rows: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    items: Sequence[ImageItem],
    phash_original: np.ndarray,
    phash_flipped: np.ndarray,
    matrices: Mapping[str, Mapping[str, tuple[list[int], np.ndarray]]],
) -> list[dict[str, str]]:
    from benchmark_candidate_generation import flip_aware_hamming

    output: list[dict[str, str]] = []
    for source in rows:
        left = index_by_key[image_key(source, "left")]
        right = index_by_key[image_key(source, "right")]
        similarity, rank, descriptor = best_pair_rank_and_similarity(
            left, right, items[left].source_group, matrices
        )
        row = {
            field: str(source.get(field, ""))
            for field in V2_PAIR_METRIC_FIELDS
            if field
            not in {
                "flip_aware_phash_distance",
                "v2_best_cosine_similarity",
                "v2_union_rank",
                "v2_best_descriptor",
            }
        }
        row.update(
            {
                "flip_aware_phash_distance": str(
                    flip_aware_hamming(
                        left, right, phash_original, phash_flipped
                    )
                ),
                "v2_best_cosine_similarity": f"{similarity:.8f}",
                "v2_union_rank": str(rank),
                "v2_best_descriptor": descriptor,
            }
        )
        output.append(row)
    return output


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def acceptance_summary(
    results: Mapping[str, Mapping[str, object]], possible_pairs: int
) -> dict[str, object]:
    accepted: list[str] = []
    for name, metrics in results.items():
        fraction = int(metrics["candidate_pair_count"]) / possible_pairs
        if (
            metrics["positive_recall"] == 1.0
            and metrics["natural_positive_recall"] == 1.0
            and fraction <= ACCEPTANCE_MAX_CANDIDATE_FRACTION
        ):
            accepted.append(name)
    return {
        "criteria": {
            "positive_recall": 1.0,
            "natural_positive_recall": 1.0,
            "maximum_candidate_fraction_of_possible_pairs": ACCEPTANCE_MAX_CANDIDATE_FRACTION,
        },
        "accepted_configurations": accepted,
        "pilot_gate_passed": bool(accepted),
        "note": "Passing this pilot gate does not authorize the 130,000-image scan.",
    }


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
    descriptors_path = output_dir / "resnet18_robust_descriptors.npz"
    if any(
        path.exists()
        for path in (report_path, pair_metrics_path, index_path, descriptors_path)
    ):
        raise FileExistsError(
            "V2 benchmark çıktısı zaten var; bilinçsiz yeniden çalışma engellendi: "
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
    possible_pairs = total_possible_pairs(items)

    print("İşlem türü: candidate-generator v2 KALİBRASYON PİLOTU")
    print(f"Kalibrasyon çifti: {len(rows)}")
    print(f"Benzersiz görüntü: {len(items)}")
    print("Tam 130.000 görüntü taranmayacak; family/split/model üretilmeyecek.")
    started = time.perf_counter()
    phash_original, phash_flipped = compute_phash_variants(items, args.workers)
    descriptors, descriptor_metadata = compute_robust_descriptors(
        items, args.batch_size
    )
    matrices = channel_similarity_matrices(items, descriptors)
    topk_sets = multichannel_topk_candidate_sets(matrices)
    phash_sets = phash_candidate_sets(items, phash_original, phash_flipped)
    hybrid_sets = hybrid_candidate_sets(topk_sets, phash_sets)

    topk_results = {
        str(topk): evaluate_candidate_set(
            topk_sets[topk], rows, index_by_key, len(items)
        )
        for topk in TOPK_VALUES
    }
    hybrid_results = {
        name: evaluate_candidate_set(candidates, rows, index_by_key, len(items))
        for name, candidates in hybrid_sets.items()
    }
    for metrics in list(topk_results.values()) + list(hybrid_results.values()):
        metrics["candidate_fraction_of_possible_pairs"] = (
            int(metrics["candidate_pair_count"]) / possible_pairs
        )

    metrics = pair_metric_rows_v2(
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
    np.savez_compressed(descriptors_path, **descriptors)
    atomic_write_csv(
        index_path,
        ("image_index", "image_key", "root_kind", "relative_path", "source_group"),
        index_rows,
    )
    atomic_write_csv(pair_metrics_path, V2_PAIR_METRIC_FIELDS, metrics)
    combined_results: dict[str, Mapping[str, object]] = {
        **{f"robust_top{k}": value for k, value in topk_results.items()},
        **hybrid_results,
    }
    report: dict[str, object] = {
        "schema_version": 2,
        "status": "candidate_generation_v2_calibration_pilot_not_full_dataset",
        "experiment_type": "robust_retrieval_candidate_generation_pilot",
        "input": {
            "pair_count": len(rows),
            "unique_image_count": len(items),
            "possible_within_source_group_pairs": possible_pairs,
            "source_group_image_counts": dict(
                sorted(Counter(item.source_group for item in items).items())
            ),
            "reference_counts": dict(
                sorted(Counter(row["reference_label"] for row in rows).items())
            ),
        },
        "robust_resnet18": {
            "metadata": descriptor_metadata,
            "topk_values": list(TOPK_VALUES),
            "results": topk_results,
        },
        "hybrid_union": {
            "description": "robust ResNet18 multichannel top-k union flip-aware pHash radius",
            "results": hybrid_results,
        },
        "acceptance": acceptance_summary(combined_results, possible_pairs),
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
            "The calibration pool includes the previously revealed duplicate-rule holdout.",
            "Candidate counts measured here do not directly predict full-dataset density or rank.",
            "AI-assisted natural-pair labels are not an independent expert gold standard.",
            "Candidate generation makes a shortlist and does not replace frozen duplicate rule v1.",
        ],
    }
    atomic_write_json(report_path, report)
    print(f"V2 benchmark raporu: {report_path}")
    print(f"Çift ölçümleri: {pair_metrics_path}")
    print(f"Görüntü indeksi: {index_path}")
    print(f"Robust descriptorler: {descriptors_path}")
    print(f"Pilot kabul kapısı: {report['acceptance']['pilot_gate_passed']}")  # type: ignore[index]
    print("NOT: Bu pilot sonucu family veya split değildir.")
    return report_path, pair_metrics_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
