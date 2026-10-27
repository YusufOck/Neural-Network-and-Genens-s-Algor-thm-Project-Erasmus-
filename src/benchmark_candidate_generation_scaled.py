from __future__ import annotations

"""Run a bounded index-density/capacity pilot for candidate generator v2.

The script keeps every labelled calibration image, adds deterministic distractors from
all 26 leaves, and measures exact blockwise top-k retrieval at increasing pool sizes.
It does not scan all 130,000 images, score candidates with the frozen duplicate rule,
build families, create a train/validation/test split, or train a classifier.
"""

import argparse
import csv
import json
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from benchmark_candidate_generation import (
    ImageItem,
    atomic_write_csv,
    collect_image_items,
    evaluate_candidate_set,
    group_indices,
    image_key,
    load_reference_pairs,
)
from benchmark_candidate_generation_v2 import (
    DESCRIPTOR_NAMES,
    RobustCalibrationDataset,
    descriptor_batch,
)
from data_index import ImageRecord, discover_records, project_root
from duplicate_decision_rule import file_sha256
from render_calibration_review import read_dataset_root


TOPK_VALUES = (5, 10, 20)
DEFAULT_STAGES = (50, 100, 250)
FROZEN_SCORER_SECONDS_PER_PAIR = 0.815
ACCEPTANCE_MAX_PAIRS_PER_IMAGE = 25.0
PAIR_STAGE_FIELDS = (
    "stage_distractors_per_leaf",
    "stage_image_count",
    "pair_id",
    "reference_label",
    "pair_kind",
    "source_group",
    "transform_name",
    "top5_candidate",
    "top10_candidate",
    "top20_candidate",
)
CANDIDATE_FIELDS = (
    "left_image_index",
    "right_image_index",
    "left_key",
    "right_key",
    "source_group",
    "known_pair_ids",
    "known_reference_labels",
)


def parse_stage_sizes(value: str) -> tuple[int, ...]:
    try:
        stages = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Stage değerleri tam sayı olmalıdır.") from error
    if not stages or any(stage < 1 for stage in stages):
        raise argparse.ArgumentTypeError("Stage değerleri pozitif olmalıdır.")
    if tuple(sorted(set(stages))) != stages:
        raise argparse.ArgumentTypeError(
            "Stage değerleri küçükten büyüğe ve tekrarsız verilmelidir."
        )
    return stages


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    v2_root = root / "artifacts" / "audit" / "candidate_generation_benchmark_v2"
    parser = argparse.ArgumentParser(
        description=(
            "Candidate generator v2 için artan indeks yoğunluğu pilotu; tam veri "
            "taraması/family/split/model eğitimi yapmaz."
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
    parser.add_argument("--v2-cache-dir", default=str(v2_root))
    parser.add_argument(
        "--output-dir",
        default=str(root / "artifacts" / "audit" / "candidate_generation_scaled_v1"),
    )
    parser.add_argument(
        "--stage-sizes",
        type=parse_stage_sizes,
        default=DEFAULT_STAGES,
        help="Her leaf için eklenecek iç içe distractor sayıları (varsayılan: 50,100,250).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retrieval-block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmalıdır.")
    if args.retrieval_block_size < 1:
        raise ValueError("--retrieval-block-size en az 1 olmalıdır.")
    if max(args.stage_sizes) > 1000:
        raise ValueError("Bu bounded pilotta stage başına en fazla 1000 distractor/leaf kullanılabilir.")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def leaf_and_source_group_maps(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    leaf_by_key: dict[str, str] = {}
    source_group_by_leaf: dict[str, str] = {}
    for row in rows:
        source_group = row["source_group"].strip()
        for side in ("left", "right"):
            key = image_key(row, side)
            leaf = row[f"{side}_leaf"].strip()
            previous_leaf = leaf_by_key.setdefault(key, leaf)
            if previous_leaf != leaf:
                raise ValueError(f"Görüntü birden çok leaf ile eşleşiyor: {key}")
            previous_group = source_group_by_leaf.setdefault(leaf, source_group)
            if previous_group != source_group:
                raise ValueError(f"Leaf birden çok source_group ile eşleşiyor: {leaf}")
    return leaf_by_key, source_group_by_leaf


def record_key(record: ImageRecord, dataset_root: Path) -> str:
    return f"dataset:{record.path.relative_to(dataset_root).as_posix()}"


def select_nested_distractors(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    excluded_keys: set[str],
    source_group_by_leaf: Mapping[str, str],
    maximum_per_leaf: int,
    seed: int,
) -> tuple[list[ImageItem], dict[str, int], dict[str, str]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        key = record_key(record, dataset_root)
        if key not in excluded_keys:
            grouped[record.subclass].append(record)
    if set(grouped) != set(source_group_by_leaf):
        raise ValueError(
            "Distractor leaf kapsamı kalibrasyonla eşleşmiyor; "
            f"eksik={sorted(set(source_group_by_leaf) - set(grouped))}, "
            f"fazla={sorted(set(grouped) - set(source_group_by_leaf))}"
        )

    items: list[ImageItem] = []
    rank_by_key: dict[str, int] = {}
    leaf_by_key: dict[str, str] = {}
    for leaf_index, leaf in enumerate(sorted(grouped)):
        candidates = sorted(grouped[leaf], key=lambda record: record_key(record, dataset_root))
        random.Random(seed + leaf_index * 1009).shuffle(candidates)
        if len(candidates) < maximum_per_leaf:
            raise ValueError(
                f"{leaf} için {maximum_per_leaf} distractor yok; bulunan={len(candidates)}"
            )
        for rank, record in enumerate(candidates[:maximum_per_leaf], start=1):
            relative = record.path.relative_to(dataset_root).as_posix()
            key = f"dataset:{relative}"
            items.append(
                ImageItem(
                    key=key,
                    root_kind="dataset",
                    relative_path=relative,
                    source_group=source_group_by_leaf[leaf],
                    absolute_path=record.path.resolve(),
                )
            )
            rank_by_key[key] = rank
            leaf_by_key[key] = leaf
    return sorted(items, key=lambda item: item.key), rank_by_key, leaf_by_key


def compute_distractor_descriptors(
    items: Sequence[ImageItem], batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, object], float]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        RobustCalibrationDataset(items, weights.transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in DESCRIPTOR_NAMES}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (raw, normalized) in enumerate(loader, start=1):
            views = torch.stack((raw, normalized), dim=1)
            batch_count, photometric, spatial, channels, height, width = views.shape
            views = views.to(device, non_blocking=True)
            joined = torch.stack((views, torch.flip(views, dims=(-1,))), dim=3)
            flat = joined.reshape(-1, channels, height, width)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = model(flat)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            features = features.reshape(batch_count, photometric, spatial, 2, -1)
            for name, values in descriptor_batch(features).items():
                collected[name].append(values.cpu().numpy())
            processed = min(batch_index * batch_size, len(items))
            if processed % 200 == 0 or processed == len(items):
                print(f"Scaled robust embedding: {processed}/{len(items)}", flush=True)
    elapsed = time.perf_counter() - started
    checkpoint_name = Path(weights.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    metadata: dict[str, object] = {
        "architecture": "torchvision_resnet18_fc_identity",
        "weights": str(weights),
        "device": str(device),
        "feature_dimension": 512,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return (
        {
            name: np.concatenate(chunks).astype(np.float32)
            for name, chunks in collected.items()
        },
        metadata,
        elapsed,
    )


def load_v2_cache(
    cache_dir: Path,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, object]]:
    index_path = cache_dir / "image_index.csv"
    descriptor_path = cache_dir / "resnet18_robust_descriptors.npz"
    report_path = cache_dir / "report.json"
    rows = read_csv(index_path)
    keys = [row["image_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("V2 cache image_index içinde tekrarlanan key var.")
    with np.load(descriptor_path) as loaded:
        if set(loaded.files) != set(DESCRIPTOR_NAMES):
            raise ValueError("V2 cache descriptor kanalları beklenenlerle eşleşmiyor.")
        descriptors = {
            name: np.asarray(loaded[name], dtype=np.float32) for name in DESCRIPTOR_NAMES
        }
    if any(values.shape != (len(keys), 512) for values in descriptors.values()):
        raise ValueError("V2 cache descriptor şekilleri bozuk.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = {
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": file_sha256(descriptor_path),
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "checkpoint_sha256": report["robust_resnet18"]["metadata"]["checkpoint_sha256"],
    }
    return keys, descriptors, metadata


def combine_descriptors(
    all_items: Sequence[ImageItem],
    calibration_keys: Sequence[str],
    calibration_descriptors: Mapping[str, np.ndarray],
    distractor_items: Sequence[ImageItem],
    distractor_descriptors: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    calibration_position = {key: index for index, key in enumerate(calibration_keys)}
    distractor_position = {item.key: index for index, item in enumerate(distractor_items)}
    output: dict[str, np.ndarray] = {}
    for name in DESCRIPTOR_NAMES:
        rows: list[np.ndarray] = []
        for item in all_items:
            if item.key in calibration_position:
                rows.append(calibration_descriptors[name][calibration_position[item.key]])
            elif item.key in distractor_position:
                rows.append(distractor_descriptors[name][distractor_position[item.key]])
            else:
                raise ValueError(f"Descriptor bulunamayan görüntü: {item.key}")
        output[name] = np.stack(rows).astype(np.float32)
    return output


def canonical_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def blockwise_topk_candidates(
    items: Sequence[ImageItem],
    descriptors: Mapping[str, np.ndarray],
    block_size: int,
    topk_values: Sequence[int] = TOPK_VALUES,
) -> tuple[dict[int, set[tuple[int, int]]], dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = {topk: set() for topk in topk_values}
    maximum_topk = max(topk_values)
    timings: dict[str, float] = defaultdict(float)
    started = time.perf_counter()
    for source_group, global_indices in group_indices(items).items():
        group_started = time.perf_counter()
        for descriptor_name in DESCRIPTOR_NAMES:
            gallery = torch.from_numpy(descriptors[descriptor_name][global_indices]).to(device)
            group_size = len(global_indices)
            kept = min(maximum_topk, group_size - 1)
            for start in range(0, group_size, block_size):
                stop = min(start + block_size, group_size)
                similarities = (gallery[start:stop] @ gallery.T).float().cpu().numpy()
                row_offsets = np.arange(stop - start)
                similarities[row_offsets, start + row_offsets] = -np.inf
                # Stable sorting makes tie behaviour reproducible and exactly matches v2.
                order = np.argsort(-similarities, axis=1, kind="stable")[:, :kept]
                for row_offset, local_left in enumerate(range(start, stop)):
                    global_left = global_indices[local_left]
                    neighbors = order[row_offset]
                    for topk in topk_values:
                        for local_right in neighbors[: min(topk, kept)]:
                            candidates[topk].add(
                                canonical_pair(global_left, global_indices[int(local_right)])
                            )
            del gallery
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings[source_group] = time.perf_counter() - group_started
        print(
            f"Scaled retrieval: {source_group} ({len(global_indices)} görüntü)", flush=True
        )
    return candidates, {
        "device": str(device),
        "block_size": block_size,
        "seconds_total": time.perf_counter() - started,
        "seconds_by_source_group": dict(sorted(timings.items())),
    }


def possible_pairs(items: Sequence[ImageItem]) -> int:
    return sum(
        len(indices) * (len(indices) - 1) // 2
        for indices in group_indices(items).values()
    )


def stage_items_and_descriptors(
    all_items: Sequence[ImageItem],
    all_descriptors: Mapping[str, np.ndarray],
    calibration_keys: set[str],
    distractor_rank_by_key: Mapping[str, int],
    stage_size: int,
) -> tuple[list[ImageItem], dict[str, np.ndarray]]:
    selected_indices = [
        index
        for index, item in enumerate(all_items)
        if item.key in calibration_keys or distractor_rank_by_key[item.key] <= stage_size
    ]
    return (
        [all_items[index] for index in selected_indices],
        {name: values[selected_indices] for name, values in all_descriptors.items()},
    )


def candidate_counts_by_group(
    candidates: set[tuple[int, int]], items: Sequence[ImageItem]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for left, right in candidates:
        if items[left].source_group != items[right].source_group:
            raise ValueError("Source-group dışı aday çifti üretildi.")
        counts[items[left].source_group] += 1
    return dict(sorted(counts.items()))


def pair_stage_rows(
    stage_size: int,
    items: Sequence[ImageItem],
    rows: Sequence[Mapping[str, str]],
    candidates: Mapping[int, set[tuple[int, int]]],
) -> list[dict[str, str]]:
    index_by_key = {item.key: index for index, item in enumerate(items)}
    output: list[dict[str, str]] = []
    for row in rows:
        pair = canonical_pair(
            index_by_key[image_key(row, "left")],
            index_by_key[image_key(row, "right")],
        )
        output.append(
            {
                "stage_distractors_per_leaf": str(stage_size),
                "stage_image_count": str(len(items)),
                "pair_id": row["pair_id"],
                "reference_label": row["reference_label"],
                "pair_kind": row["pair_kind"],
                "source_group": row["source_group"],
                "transform_name": row.get("transform_name", ""),
                "top5_candidate": "yes" if pair in candidates[5] else "no",
                "top10_candidate": "yes" if pair in candidates[10] else "no",
                "top20_candidate": "yes" if pair in candidates[20] else "no",
            }
        )
    return output


def maximum_stage_candidate_rows(
    items: Sequence[ImageItem],
    candidates: set[tuple[int, int]],
    references: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    index_by_key = {item.key: index for index, item in enumerate(items)}
    reference_by_pair: dict[tuple[int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in references:
        pair = canonical_pair(
            index_by_key[image_key(row, "left")],
            index_by_key[image_key(row, "right")],
        )
        reference_by_pair[pair].append(row)
    output: list[dict[str, str]] = []
    for left, right in sorted(candidates):
        known = reference_by_pair.get((left, right), [])
        output.append(
            {
                "left_image_index": str(left),
                "right_image_index": str(right),
                "left_key": items[left].key,
                "right_key": items[right].key,
                "source_group": items[left].source_group,
                "known_pair_ids": ";".join(row["pair_id"] for row in known),
                "known_reference_labels": ";".join(
                    row["reference_label"] for row in known
                ),
            }
        )
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


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    calibration_root = Path(args.calibration_root).resolve()
    pairs_path = Path(args.pairs_csv).resolve()
    development_path = Path(args.development_decisions).resolve()
    holdout_path = Path(args.holdout_decisions).resolve()
    v2_cache_dir = Path(args.v2_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "report.json"
    index_path = output_dir / "image_index.csv"
    pair_metrics_path = output_dir / "pair_stage_metrics.csv"
    candidates_path = output_dir / "max_stage_top10_candidates.csv"
    descriptors_path = output_dir / "robust_descriptors_max_stage.npz"
    protected_outputs = (
        report_path,
        index_path,
        pair_metrics_path,
        candidates_path,
        descriptors_path,
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            "Scaled pilot çıktısı zaten var; bilinçsiz yeniden çalışma engellendi: "
            f"{output_dir}"
        )

    dataset_root = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else read_dataset_root(calibration_root)
    )
    rows = load_reference_pairs(pairs_path, development_path, holdout_path)
    calibration_items = collect_image_items(rows, dataset_root, calibration_root)
    calibration_keys = {item.key for item in calibration_items}
    leaf_by_key, source_group_by_leaf = leaf_and_source_group_maps(rows)
    records = discover_records(dataset_root)
    distractor_items, distractor_rank_by_key, distractor_leaf_by_key = (
        select_nested_distractors(
            records,
            dataset_root,
            calibration_keys,
            source_group_by_leaf,
            max(args.stage_sizes),
            args.seed,
        )
    )
    all_items = sorted([*calibration_items, *distractor_items], key=lambda item: item.key)
    if len({item.key for item in all_items}) != len(all_items):
        raise ValueError("Scaled pilot görüntü listesinde tekrarlanan key var.")

    print("İşlem türü: SCALED RETRIEVAL/CAPACITY PİLOTU")
    print(f"Kalibrasyon görüntüsü: {len(calibration_items)}")
    print(f"Yeni distractor: {len(distractor_items)}")
    print(f"Maksimum indeks: {len(all_items)}")
    print(f"Stage değerleri: {args.stage_sizes}")
    print("Tam 130.000 görüntü taranmayacak; family/split/model üretilmeyecek.")

    total_started = time.perf_counter()
    cache_keys, cache_descriptors, cache_metadata = load_v2_cache(v2_cache_dir)
    if set(cache_keys) != calibration_keys:
        raise ValueError("V2 cache görüntü kapsamı kalibrasyon görüntüleriyle eşleşmiyor.")
    distractor_descriptors, embedding_metadata, embedding_seconds = (
        compute_distractor_descriptors(distractor_items, args.batch_size)
    )
    all_descriptors = combine_descriptors(
        all_items,
        cache_keys,
        cache_descriptors,
        distractor_items,
        distractor_descriptors,
    )

    stage_results: dict[str, object] = {}
    all_pair_stage_rows: list[dict[str, str]] = []
    max_stage_items: list[ImageItem] | None = None
    max_stage_candidates: dict[int, set[tuple[int, int]]] | None = None
    for stage_size in args.stage_sizes:
        stage_items, stage_descriptors = stage_items_and_descriptors(
            all_items,
            all_descriptors,
            calibration_keys,
            distractor_rank_by_key,
            stage_size,
        )
        candidate_sets, retrieval_metadata = blockwise_topk_candidates(
            stage_items,
            stage_descriptors,
            args.retrieval_block_size,
        )
        index_by_key = {item.key: index for index, item in enumerate(stage_items)}
        topk_results: dict[str, object] = {}
        for topk in TOPK_VALUES:
            metrics = evaluate_candidate_set(
                candidate_sets[topk], rows, index_by_key, len(stage_items)
            )
            metrics["candidate_fraction_of_possible_pairs"] = (
                len(candidate_sets[topk]) / possible_pairs(stage_items)
            )
            metrics["candidate_counts_by_source_group"] = candidate_counts_by_group(
                candidate_sets[topk], stage_items
            )
            topk_results[str(topk)] = metrics
        stage_results[str(stage_size)] = {
            "distractors_per_leaf": stage_size,
            "image_count": len(stage_items),
            "calibration_image_count": len(calibration_items),
            "distractor_image_count": len(stage_items) - len(calibration_items),
            "possible_within_source_group_pairs": possible_pairs(stage_items),
            "source_group_image_counts": dict(
                sorted(Counter(item.source_group for item in stage_items).items())
            ),
            "retrieval": retrieval_metadata,
            "topk_results": topk_results,
        }
        all_pair_stage_rows.extend(
            pair_stage_rows(stage_size, stage_items, rows, candidate_sets)
        )
        if stage_size == max(args.stage_sizes):
            max_stage_items = stage_items
            max_stage_candidates = candidate_sets

    if max_stage_items is None or max_stage_candidates is None:
        raise RuntimeError("Maksimum stage sonucu oluşmadı.")
    max_metrics = stage_results[str(max(args.stage_sizes))]["topk_results"]["10"]  # type: ignore[index]
    gate_passed = bool(
        max_metrics["positive_recall"] == 1.0  # type: ignore[index]
        and max_metrics["natural_positive_recall"] == 1.0  # type: ignore[index]
        and max_metrics["candidate_pairs_per_indexed_image"]
        <= ACCEPTANCE_MAX_PAIRS_PER_IMAGE  # type: ignore[index]
    )
    pairs_per_image = float(max_metrics["candidate_pairs_per_indexed_image"])  # type: ignore[index]
    estimated_full_pairs = round(pairs_per_image * 130_000)
    estimated_frozen_seconds = estimated_full_pairs * FROZEN_SCORER_SECONDS_PER_PAIR

    index_rows: list[dict[str, str]] = []
    for index, item in enumerate(all_items):
        is_calibration = item.key in calibration_keys
        leaf = leaf_by_key[item.key] if is_calibration else distractor_leaf_by_key[item.key]
        index_rows.append(
            {
                "image_index": str(index),
                "image_key": item.key,
                "root_kind": item.root_kind,
                "relative_path": item.relative_path,
                "leaf": leaf,
                "source_group": item.source_group,
                "pool_role": "calibration" if is_calibration else "distractor",
                "distractor_rank_within_leaf": (
                    "" if is_calibration else str(distractor_rank_by_key[item.key])
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(descriptors_path, **all_descriptors)
    atomic_write_csv(
        index_path,
        (
            "image_index",
            "image_key",
            "root_kind",
            "relative_path",
            "leaf",
            "source_group",
            "pool_role",
            "distractor_rank_within_leaf",
        ),
        index_rows,
    )
    atomic_write_csv(pair_metrics_path, PAIR_STAGE_FIELDS, all_pair_stage_rows)
    atomic_write_csv(
        candidates_path,
        CANDIDATE_FIELDS,
        maximum_stage_candidate_rows(
            max_stage_items, max_stage_candidates[10], rows
        ),
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "bounded_scaled_retrieval_capacity_pilot_not_full_dataset",
        "experiment_type": "candidate_generator_v2_index_density_and_capacity_pilot",
        "input": {
            "calibration_image_count": len(calibration_items),
            "reference_pair_count": len(rows),
            "distractors_per_leaf_maximum": max(args.stage_sizes),
            "leaf_count": len(source_group_by_leaf),
            "maximum_index_image_count": len(all_items),
            "stage_sizes": list(args.stage_sizes),
            "seed": args.seed,
        },
        "method": {
            "candidate_generator": "v2 robust ResNet18 four-channel descriptor union",
            "topk_values": list(TOPK_VALUES),
            "retrieval": "exact blockwise cosine similarity within compatible source_group",
            "calibration_descriptor_cache": cache_metadata,
            "new_embedding_metadata": embedding_metadata,
        },
        "stages": stage_results,
        "acceptance": {
            "evaluated_stage": max(args.stage_sizes),
            "criteria": {
                "positive_recall": 1.0,
                "natural_positive_recall": 1.0,
                "maximum_candidate_pairs_per_indexed_image": ACCEPTANCE_MAX_PAIRS_PER_IMAGE,
            },
            "pilot_gate_passed": gate_passed,
        },
        "capacity_estimate": {
            "basis": "linear extrapolation from maximum-stage unique top-10 candidate pairs per indexed image",
            "measured_candidate_pairs_per_image": pairs_per_image,
            "estimated_130k_unique_candidate_pairs_if_density_stays_constant": estimated_full_pairs,
            "four_channel_top10_directed_relation_upper_bound_130k": 5_200_000,
            "frozen_scorer_measured_seconds_per_pair": FROZEN_SCORER_SECONDS_PER_PAIR,
            "estimated_serial_frozen_scorer_seconds": estimated_frozen_seconds,
            "estimated_serial_frozen_scorer_days": estimated_frozen_seconds / 86_400,
            "warning": "This extrapolation is a capacity estimate, not a full-dataset measurement.",
        },
        "timing": {
            "new_distractor_embedding_seconds": embedding_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
        "integrity": {
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "pairs_csv_sha256": file_sha256(pairs_path),
            "development_decisions_sha256": file_sha256(development_path),
            "holdout_decisions_sha256": file_sha256(holdout_path),
        },
        "limitations": [
            "The maximum index contains only 8,248 images, not all 130,000 images.",
            "The same revealed calibration references are used to measure retrieval recall.",
            "Random distractors measure index density but are not labelled duplicate negatives.",
            "No frozen-rule scoring, family construction, classifier split or model training occurs.",
        ],
    }
    atomic_write_json(report_path, report)
    print(f"Scaled pilot raporu: {report_path}")
    print(f"Maksimum-stage top-10 adayları: {candidates_path}")
    print(f"Pilot kabul kapısı: {gate_passed}")
    print("NOT: Bu çıktı tam veri taraması, family veya split değildir.")
    return report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
