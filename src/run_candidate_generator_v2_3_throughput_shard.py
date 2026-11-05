from __future__ import annotations

"""Run one small, hash-guarded Candidate Generator v2.3 throughput shard.

This is an infrastructure smoke test and environment measurement. It never scans
all 130,000 images, scores duplicate decisions, builds families/splits, or trains a
classifier.
"""

import argparse
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import PIL
import torch
import torchvision
from torchvision.models import ResNet18_Weights

from benchmark_candidate_generation import (
    ImageItem,
    atomic_write_csv,
    compute_phash_variants,
    phash_candidate_sets,
)
from benchmark_candidate_generation_scaled import (
    blockwise_topk_candidates,
    candidate_counts_by_group,
    possible_pairs,
)
from benchmark_candidate_generation_v2 import (
    DESCRIPTOR_NAMES,
    compute_robust_descriptors,
)
from data_index import ImageRecord, discover_records, project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256
from verify_candidate_generator_v2_3_freeze import (
    load_frozen_config,
    load_source_group_map,
    verify_checkpoint,
    verify_dataset_inventory,
    verify_dependency_hashes,
)


INDEX_FIELDS = (
    "image_index",
    "relative_path",
    "main_class",
    "leaf",
    "source_group",
    "selection_rank_within_leaf",
)
CANDIDATE_FIELDS = (
    "left_image_index",
    "right_image_index",
    "left_relative_path",
    "right_relative_path",
    "left_leaf",
    "right_leaf",
    "source_group",
    "robust_top15",
    "phash_le12",
    "candidate_channels",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Candidate Generator v2.3 için küçük hash-korumalı throughput "
            "shard'ı; full retrieval/family/split/model çalıştırmaz."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--config", default=str(root / "config" / "candidate_generator_v2_3.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "candidate_generator_v2_3.sha256"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_3_throughput_shard"
        ),
    )
    parser.add_argument("--images-per-leaf", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.images_per_leaf <= 100:
        raise ValueError("--images-per-leaf bu bounded shard için 1..100 olmalıdır.")
    if args.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmalıdır.")
    if args.workers < 1:
        raise ValueError("--workers en az 1 olmalıdır.")
    if args.retrieval_block_size < 1:
        raise ValueError("--retrieval-block-size en az 1 olmalıdır.")


def ensure_checkpoint_available() -> None:
    """Download the official checkpoint when absent; its full hash is checked next."""

    weights = ResNet18_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if checkpoint.is_file():
        return
    torch.hub.load_state_dict_from_url(
        weights.url,
        model_dir=str(checkpoint.parent),
        progress=True,
        check_hash=True,
        map_location="cpu",
    )


def select_deterministic_shard(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    source_by_leaf: Mapping[str, str],
    images_per_leaf: int,
    seed: int,
) -> tuple[list[ImageItem], dict[str, ImageRecord], dict[str, int]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.subclass, []).append(record)
    if set(grouped) != set(source_by_leaf):
        raise ValueError("Shard leaf kapsamı frozen source-group haritasıyla eşleşmiyor.")

    selected: list[tuple[ImageItem, ImageRecord, int]] = []
    for leaf_index, leaf in enumerate(sorted(grouped)):
        candidates = sorted(
            grouped[leaf], key=lambda item: item.path.relative_to(dataset_root).as_posix()
        )
        random.Random(seed + leaf_index * 1009).shuffle(candidates)
        if len(candidates) < images_per_leaf:
            raise ValueError(f"{leaf} için yeterli görüntü yok: {len(candidates)}")
        for rank, record in enumerate(candidates[:images_per_leaf], start=1):
            relative = record.path.relative_to(dataset_root).as_posix()
            selected.append(
                (
                    ImageItem(
                        key=f"dataset:{relative}",
                        root_kind="dataset",
                        relative_path=relative,
                        source_group=source_by_leaf[leaf],
                        absolute_path=record.path.resolve(),
                    ),
                    record,
                    rank,
                )
            )

    selected.sort(key=lambda entry: entry[0].relative_path)
    items = [entry[0] for entry in selected]
    record_by_path = {entry[0].relative_path: entry[1] for entry in selected}
    rank_by_path = {entry[0].relative_path: entry[2] for entry in selected}
    if len(items) != len({item.relative_path for item in items}):
        raise ValueError("Shard içinde tekrarlanan görüntü yolu var.")
    return items, record_by_path, rank_by_path


def build_index_rows(
    items: Sequence[ImageItem],
    record_by_path: Mapping[str, ImageRecord],
    rank_by_path: Mapping[str, int],
) -> list[dict[str, str]]:
    return [
        {
            "image_index": str(index),
            "relative_path": item.relative_path,
            "main_class": record_by_path[item.relative_path].main_class,
            "leaf": record_by_path[item.relative_path].subclass,
            "source_group": item.source_group,
            "selection_rank_within_leaf": str(rank_by_path[item.relative_path]),
        }
        for index, item in enumerate(items)
    ]


def build_candidate_rows(
    items: Sequence[ImageItem],
    record_by_path: Mapping[str, ImageRecord],
    robust: set[tuple[int, int]],
    phash: set[tuple[int, int]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for left, right in sorted(robust | phash):
        channels = []
        if (left, right) in robust:
            channels.append("robust_top15")
        if (left, right) in phash:
            channels.append("phash_le12")
        left_item = items[left]
        right_item = items[right]
        output.append(
            {
                "left_image_index": str(left),
                "right_image_index": str(right),
                "left_relative_path": left_item.relative_path,
                "right_relative_path": right_item.relative_path,
                "left_leaf": record_by_path[left_item.relative_path].subclass,
                "right_leaf": record_by_path[right_item.relative_path].subclass,
                "source_group": left_item.source_group,
                "robust_top15": "yes" if (left, right) in robust else "no",
                "phash_le12": "yes" if (left, right) in phash else "no",
                "candidate_channels": "+".join(channels),
            }
        )
    return output


def verify_candidate_integrity(
    items: Sequence[ImageItem], pairs: set[tuple[int, int]]
) -> dict[str, object]:
    if any(left >= right for left, right in pairs):
        raise ValueError("Aday çiftleri canonical left < right biçiminde değil.")
    if any(left < 0 or right >= len(items) for left, right in pairs):
        raise ValueError("Aday çiftinde indeks sınır dışı.")
    cross_group = [
        pair
        for pair in pairs
        if items[pair[0]].source_group != items[pair[1]].source_group
    ]
    if cross_group:
        raise ValueError(f"Source-group dışı aday üretildi: {cross_group[:3]}")
    return {
        "unique_canonical_pairs": len(pairs),
        "cross_source_group_pairs": 0,
        "candidate_counts_by_source_group": candidate_counts_by_group(pairs, items),
    }


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def environment_summary() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    validate_args(args)
    root = project_root().resolve()
    config_path = Path(args.config).resolve()
    hash_path = Path(args.config_sha256).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            "Throughput shard çıktı klasörü zaten var; üzerine yazma reddedildi: "
            f"{output_dir}"
        )

    config = load_frozen_config(config_path, hash_path)
    dependencies = verify_dependency_hashes(root, config)
    ensure_checkpoint_available()
    checkpoint = verify_checkpoint(config)
    groups_path = root / "config" / "duplicate_audit_groups.json"
    dataset_root = resolve_dataset_root(args.data_dir)
    inventory = verify_dataset_inventory(dataset_root, config, groups_path)
    source_by_leaf = load_source_group_map(groups_path)
    records = discover_records(dataset_root)
    items, record_by_path, rank_by_path = select_deterministic_shard(
        records,
        dataset_root,
        source_by_leaf,
        args.images_per_leaf,
        args.seed,
    )

    print("İşlem türü: HASH-KORUMALI V2.3 THROUGHPUT SHARD")
    print(f"Görüntü: {len(items)} = 26 leaf x {args.images_per_leaf}")
    print("Bu çalışma full retrieval/family/split/model eğitimi değildir.")
    total_started = time.perf_counter()

    stage_started = time.perf_counter()
    phash_original, phash_flipped = compute_phash_variants(items, args.workers)
    phash_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    descriptors, descriptor_metadata = compute_robust_descriptors(items, args.batch_size)
    descriptor_seconds = time.perf_counter() - stage_started
    if any(values.shape != (len(items), 512) for values in descriptors.values()):
        raise ValueError("Robust descriptor şekilleri beklenen [N, 512] biçiminde değil.")
    if set(descriptors) != set(DESCRIPTOR_NAMES):
        raise ValueError("Dört frozen descriptor kanalı eksik veya fazla.")

    robust_sets, retrieval_metadata = blockwise_topk_candidates(
        items, descriptors, args.retrieval_block_size, topk_values=(15,)
    )
    robust = robust_sets[15]
    stage_started = time.perf_counter()
    phash = phash_candidate_sets(items, phash_original, phash_flipped)[12]
    phash_retrieval_seconds = time.perf_counter() - stage_started
    union = robust | phash
    integrity = verify_candidate_integrity(items, union)
    if len(union) > 25.0 * len(items):
        raise ValueError("Shard aday yükü frozen 25 çift/görüntü güvenlik kapısını aştı.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        index_path = staging_dir / "image_index.csv"
        phash_path = staging_dir / "phash_variants.npz"
        descriptor_path = staging_dir / "robust_descriptors.npz"
        candidate_path = staging_dir / "candidate_pairs.csv"
        report_path = staging_dir / "run_report.json"
        manifest_path = staging_dir / "hash_manifest.json"

        atomic_write_csv(
            index_path,
            INDEX_FIELDS,
            build_index_rows(items, record_by_path, rank_by_path),
        )
        atomic_npz(
            phash_path,
            original=phash_original,
            horizontal_flip=phash_flipped,
        )
        atomic_npz(descriptor_path, **descriptors)
        atomic_write_csv(
            candidate_path,
            CANDIDATE_FIELDS,
            build_candidate_rows(items, record_by_path, robust, phash),
        )

        total_seconds = time.perf_counter() - total_started
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "candidate_generator_v2_3_bounded_throughput_shard_complete",
            "experiment_type": "infrastructure_smoke_test_and_throughput_shard",
            "not_a_classifier_result": True,
            "input": {
                "dataset_root": str(dataset_root),
                "full_inventory": inventory,
                "shard_image_count": len(items),
                "leaf_count": len(source_by_leaf),
                "images_per_leaf": args.images_per_leaf,
                "seed": args.seed,
            },
            "frozen_method": {
                "config_sha256": file_sha256(config_path),
                "logic": dict(config["candidate_rule"])["logic"],
                "top_k": 15,
                "phash_radius": 12,
                "search_scope": "within_compatible_source_group_only",
            },
            "environment": environment_summary(),
            "candidate_counts": {
                "possible_within_source_group_pairs": possible_pairs(items),
                "robust_top15_pairs": len(robust),
                "phash_le12_pairs": len(phash),
                "intersection_pairs": len(robust & phash),
                "union_pairs": len(union),
                "union_pairs_per_image": len(union) / len(items),
            },
            "timing_seconds": {
                "phash_generation": phash_seconds,
                "robust_descriptor_generation": descriptor_seconds,
                "robust_retrieval": retrieval_metadata["seconds_total"],
                "phash_retrieval": phash_retrieval_seconds,
                "total_before_report_write": total_seconds,
            },
            "throughput": {
                "robust_descriptor_images_per_second": len(items)
                / descriptor_seconds,
                "warning": (
                    "This small-shard rate is an environment measurement, not a valid "
                    "full-run duration guarantee; full retrieval has different scaling."
                ),
            },
            "integrity": {
                **integrity,
                "dependency_hashes": dependencies,
                "checkpoint": checkpoint,
                "script_sha256": file_sha256(Path(__file__).resolve()),
                "descriptor_metadata": descriptor_metadata,
            },
            "forbidden_stages_confirmed_not_run": [
                "full_130000_image_retrieval",
                "frozen_duplicate_rule_scoring",
                "family_construction",
                "train_validation_test_split",
                "classifier_training",
            ],
            "limitations": [
                "This shard contains only a deterministic sample from each leaf.",
                "Candidate counts from this small shard are not full-dataset estimates.",
                "No candidate pair is labelled duplicate merely by retrieval inclusion.",
            ],
        }
        atomic_json(report_path, report)
        hash_manifest = {
            "schema_version": 1,
            "files": {
                path.name: file_sha256(path)
                for path in (
                    index_path,
                    phash_path,
                    descriptor_path,
                    candidate_path,
                    report_path,
                )
            },
        }
        atomic_json(manifest_path, hash_manifest)
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    final_report = output_dir / "run_report.json"
    print(f"Shard raporu: {final_report}")
    print(f"Aday birleşimi: {len(union)} ({len(union) / len(items):.3f} çift/görüntü)")
    print("NOT: Tam tarama, family, split ve model eğitimi başlatılmadı.")
    return final_report, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
