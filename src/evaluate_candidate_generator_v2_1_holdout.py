from __future__ import annotations

"""Evaluate frozen Candidate Generator v2.1 once on its prospective holdout."""

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analyze_scaled_phash8_safety import phash_radius_candidates
from benchmark_candidate_generation import (
    ImageItem,
    compute_phash_variants,
    flip_aware_hamming,
)
from benchmark_candidate_generation_scaled import (
    blockwise_topk_candidates,
    candidate_counts_by_group,
    possible_pairs,
)
from benchmark_candidate_generation_v2 import compute_robust_descriptors
from data_index import project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


PAIR_METRIC_FIELDS = (
    "pair_id",
    "leaf",
    "source_group",
    "source_id",
    "transform_name",
    "left_key",
    "right_key",
    "flip_aware_phash_distance",
    "robust_top10_candidate",
    "phash_radius8_candidate",
    "v2_1_union_candidate",
)
CANDIDATE_FIELDS = (
    "left_image_index",
    "right_image_index",
    "left_key",
    "right_key",
    "source_group",
    "robust_top10",
    "phash_radius8",
    "known_positive_pair_ids",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    holdout = root / "artifacts" / "audit" / "candidate_generator_v2_1_holdout"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmuş Candidate Generator v2.1'i yeni kontrollü holdout'ta "
            "tek sefer değerlendirir; family/split/model üretmez."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--holdout-root", default=str(holdout))
    parser.add_argument(
        "--config", default=str(root / "config" / "candidate_generator_v2_1.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "candidate_generator_v2_1.sha256"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-block-size", type=int, default=256)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_config(config_path: Path, hash_path: Path) -> Mapping[str, object]:
    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = file_sha256(config_path)
    if actual != expected:
        raise ValueError(
            f"Dondurulmuş v2.1 config değişmiş: expected={expected}, actual={actual}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_items(
    image_index_path: Path, dataset_root: Path, holdout_root: Path
) -> tuple[list[ImageItem], list[dict[str, str]]]:
    rows = read_csv(image_index_path)
    rows.sort(key=lambda row: int(row["image_index"]))
    items: list[ImageItem] = []
    for expected_index, row in enumerate(rows):
        if int(row["image_index"]) != expected_index:
            raise ValueError("Holdout image_index sırası kesintisiz değil.")
        if row["root_kind"] == "dataset":
            absolute = dataset_root / Path(row["relative_path"])
        elif row["root_kind"] == "holdout":
            absolute = holdout_root / Path(row["relative_path"])
        else:
            raise ValueError(f"Bilinmeyen root_kind: {row['root_kind']}")
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        items.append(
            ImageItem(
                key=row["image_key"],
                root_kind=row["root_kind"],
                relative_path=row["relative_path"],
                source_group=row["source_group"],
                absolute_path=absolute,
            )
        )
    if len({item.key for item in items}) != len(items):
        raise ValueError("Holdout image key değerleri benzersiz değil.")
    return items, rows


def canonical_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def evaluate_positive_pairs(
    positive_rows: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    robust: set[tuple[int, int]],
    phash: set[tuple[int, int]],
    union: set[tuple[int, int]],
    phash_original: np.ndarray,
    phash_flipped: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    by_transform: dict[str, list[bool]] = defaultdict(list)
    robust_hits = 0
    phash_hits = 0
    union_hits = 0
    for row in positive_rows:
        left = index_by_key[row["left_key"]]
        right = index_by_key[row["right_key"]]
        pair = canonical_pair(left, right)
        robust_hit = pair in robust
        phash_hit = pair in phash
        union_hit = pair in union
        robust_hits += robust_hit
        phash_hits += phash_hit
        union_hits += union_hit
        by_transform[row["transform_name"]].append(union_hit)
        metric_rows.append(
            {
                "pair_id": row["pair_id"],
                "leaf": row["leaf"],
                "source_group": row["source_group"],
                "source_id": row["source_id"],
                "transform_name": row["transform_name"],
                "left_key": row["left_key"],
                "right_key": row["right_key"],
                "flip_aware_phash_distance": flip_aware_hamming(
                    left, right, phash_original, phash_flipped
                ),
                "robust_top10_candidate": "yes" if robust_hit else "no",
                "phash_radius8_candidate": "yes" if phash_hit else "no",
                "v2_1_union_candidate": "yes" if union_hit else "no",
            }
        )
    count = len(positive_rows)
    metrics: dict[str, object] = {
        "positive_count": count,
        "robust_top10_hits": robust_hits,
        "robust_top10_recall": robust_hits / count,
        "phash_radius8_hits": phash_hits,
        "phash_radius8_recall": phash_hits / count,
        "v2_1_union_hits": union_hits,
        "v2_1_union_recall": union_hits / count,
        "v2_1_union_by_transform": {
            name: {
                "positive_count": len(values),
                "hits": sum(values),
                "recall": sum(values) / len(values),
            }
            for name, values in sorted(by_transform.items())
        },
        "v2_1_missed_pair_ids": [
            str(row["pair_id"])
            for row, metric in zip(positive_rows, metric_rows)
            if metric["v2_1_union_candidate"] == "no"
        ],
    }
    return metric_rows, metrics


def candidate_manifest_rows(
    items: Sequence[ImageItem],
    positive_rows: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    robust: set[tuple[int, int]],
    phash: set[tuple[int, int]],
    union: set[tuple[int, int]],
) -> list[dict[str, object]]:
    positive_ids: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in positive_rows:
        pair = canonical_pair(
            index_by_key[row["left_key"]], index_by_key[row["right_key"]]
        )
        positive_ids[pair].append(row["pair_id"])
    return [
        {
            "left_image_index": left,
            "right_image_index": right,
            "left_key": items[left].key,
            "right_key": items[right].key,
            "source_group": items[left].source_group,
            "robust_top10": "yes" if (left, right) in robust else "no",
            "phash_radius8": "yes" if (left, right) in phash else "no",
            "known_positive_pair_ids": ";".join(positive_ids.get((left, right), [])),
        }
        for left, right in sorted(union)
    ]


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if args.batch_size < 1 or args.workers < 1 or args.retrieval_block_size < 1:
        raise ValueError("batch-size, workers ve retrieval-block-size pozitif olmalıdır.")
    started = time.perf_counter()
    holdout_root = Path(args.holdout_root).resolve()
    config_path = Path(args.config).resolve()
    config_hash_path = Path(args.config_sha256).resolve()
    output_dir = holdout_root / "evaluation"
    if output_dir.exists():
        raise FileExistsError(
            f"Prospective evaluation zaten var; ikinci koşu engellendi: {output_dir}"
        )
    config = load_frozen_config(config_path, config_hash_path)
    build_report_path = holdout_root / "build_report.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report["status"] != "prospective_controlled_holdout_built_not_evaluated":
        raise ValueError("Holdout build raporu beklenen prospective durumda değil.")
    if build_report["integrity"]["frozen_config_sha256"] != file_sha256(config_path):
        raise ValueError("Holdout başka bir config hash'iyle oluşturulmuş.")

    dataset_root = resolve_dataset_root(args.data_dir)
    image_index_path = holdout_root / "image_index.csv"
    positive_pairs_path = holdout_root / "positive_pairs.csv"
    items, index_rows = load_items(image_index_path, dataset_root, holdout_root)
    positive_rows = read_csv(positive_pairs_path)
    expected_images = int(dict(config["holdout_design"])["expected_index_image_count"])
    expected_pairs = int(
        dict(config["holdout_design"])["expected_controlled_positive_pair_count"]
    )
    if len(items) != expected_images or len(positive_rows) != expected_pairs:
        raise ValueError("Holdout kapsamı frozen config ile eşleşmiyor.")

    print("İşlem türü: FROZEN CANDIDATE GENERATOR v2.1 HOLDOUT EVALUATION")
    print(f"Görüntü: {len(items)}, bilinen pozitif: {len(positive_rows)}")
    print("Kural: robust dört-kanal top-10 UNION flip-aware pHash<=8")
    print("Family/split/model eğitimi yapılmayacak.")

    phash_started = time.perf_counter()
    phash_original, phash_flipped = compute_phash_variants(items, args.workers)
    phash_candidates, phash_counts = phash_radius_candidates(
        items, phash_original, phash_flipped, radius=8
    )
    phash_seconds = time.perf_counter() - phash_started

    descriptors, embedding_metadata = compute_robust_descriptors(
        items, args.batch_size
    )
    robust_sets, retrieval_metadata = blockwise_topk_candidates(
        items,
        descriptors,
        args.retrieval_block_size,
        topk_values=(10,),
    )
    robust_candidates = robust_sets[10]
    union_candidates = robust_candidates | phash_candidates
    index_by_key = {item.key: index for index, item in enumerate(items)}
    pair_metrics, positive_metrics = evaluate_positive_pairs(
        positive_rows,
        index_by_key,
        robust_candidates,
        phash_candidates,
        union_candidates,
        phash_original,
        phash_flipped,
    )

    gate = dict(config["acceptance_gate"])
    per_transform = dict(positive_metrics["v2_1_union_by_transform"])
    pairs_per_image = len(union_candidates) / len(items)
    gate_passed = bool(
        positive_metrics["v2_1_union_recall"]
        >= float(gate["controlled_same_source_recall"])
        and min(float(value["recall"]) for value in per_transform.values())
        >= float(gate["minimum_recall_for_each_transform"])
        and pairs_per_image
        <= float(gate["maximum_unique_candidate_pairs_per_indexed_image"])
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".evaluation.staging-", dir=output_dir.parent)
    )
    try:
        descriptor_path = staging / "robust_descriptors.npz"
        np.savez_compressed(descriptor_path, **descriptors)
        np.savez_compressed(
            staging / "phash_variants.npz",
            original=phash_original,
            flipped=phash_flipped,
        )
        write_csv(staging / "positive_pair_metrics.csv", PAIR_METRIC_FIELDS, pair_metrics)
        manifest = candidate_manifest_rows(
            items,
            positive_rows,
            index_by_key,
            robust_candidates,
            phash_candidates,
            union_candidates,
        )
        write_csv(staging / "candidate_pairs.csv", CANDIDATE_FIELDS, manifest)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": (
                "prospective_controlled_holdout_passed"
                if gate_passed
                else "prospective_controlled_holdout_failed"
            ),
            "experiment_type": "candidate_generator_v2_1_frozen_prospective_controlled_holdout",
            "input": {
                "image_count": len(items),
                "positive_pair_count": len(positive_rows),
                "leaf_count": len({row["leaf"] for row in index_rows}),
                "source_group_count": len({item.source_group for item in items}),
            },
            "candidate_rule": config["candidate_rule"],
            "candidate_counts": {
                "robust_top10": len(robust_candidates),
                "phash_radius8": len(phash_candidates),
                "intersection": len(robust_candidates & phash_candidates),
                "v2_1_union": len(union_candidates),
                "union_increment_over_robust": len(union_candidates)
                - len(robust_candidates),
                "v2_1_union_pairs_per_indexed_image": pairs_per_image,
                "possible_within_source_group_pairs": possible_pairs(items),
                "union_fraction_of_possible": len(union_candidates)
                / possible_pairs(items),
                "robust_by_source_group": candidate_counts_by_group(
                    robust_candidates, items
                ),
                "phash_by_source_group": phash_counts,
                "union_by_source_group": candidate_counts_by_group(
                    union_candidates, items
                ),
            },
            "positive_metrics": positive_metrics,
            "acceptance": {
                "predeclared_gate": gate,
                "gate_passed": gate_passed,
            },
            "timing": {
                "phash_seconds": phash_seconds,
                "embedding_and_retrieval": retrieval_metadata,
                "total_seconds": time.perf_counter() - started,
            },
            "embedding_metadata": embedding_metadata,
            "integrity": {
                "frozen_config_sha256": file_sha256(config_path),
                "build_report_sha256": file_sha256(build_report_path),
                "image_index_sha256": file_sha256(image_index_path),
                "positive_pairs_sha256": file_sha256(positive_pairs_path),
                "evaluation_script_sha256": file_sha256(Path(__file__).resolve()),
            },
            "limitations": [
                *list(config["interpretation_limits"]),
                "This holdout contains controlled positives and unlabelled distractors; it does not estimate candidate precision or prospective natural-near-duplicate recall.",
            ],
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        "v2.1 kontrollü pozitif: "
        f"{positive_metrics['v2_1_union_hits']}/{positive_metrics['positive_count']}"
    )
    print(f"Aday yükü: {pairs_per_image:.3f} çift/görüntü")
    print(f"Önceden dondurulan kabul kapısı: {gate_passed}")
    print(f"Rapor: {output_dir / 'report.json'}")
    print("NOT: Bu duplicate retrieval sonucudur; sınıflandırma accuracy değildir.")
    return output_dir / "report.json", report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
