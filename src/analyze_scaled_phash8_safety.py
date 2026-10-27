from __future__ import annotations

"""Post-hoc pHash<=8 safety-channel analysis for the bounded scaled pilot.

This analysis was motivated by the two top-10 misses observed after the predeclared
scaled-pilot run. It is exploratory method-development evidence, not an untouched test
and not a retroactive pass of the original robust-top-10 acceptance gate.
"""

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from benchmark_candidate_generation import (
    ImageItem,
    compute_phash_variants,
    evaluate_candidate_set,
    group_indices,
    load_reference_pairs,
)
from data_index import project_root
from duplicate_decision_rule import file_sha256
from render_calibration_review import read_dataset_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    scaled_root = root / "artifacts" / "audit" / "candidate_generation_scaled_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Scaled pilot maksimum indeksinde robust top-10 + pHash<=8 post-hoc "
            "safety-channel analizi yapar."
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
    parser.add_argument("--scaled-root", default=str(scaled_root))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--radius", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def load_scaled_items(
    index_path: Path, dataset_root: Path, calibration_root: Path
) -> list[ImageItem]:
    rows = read_csv(index_path)
    items: list[ImageItem] = []
    for row in rows:
        root_kind = row["root_kind"]
        if root_kind == "dataset":
            absolute = dataset_root / Path(row["relative_path"])
        elif root_kind == "artifact":
            absolute = calibration_root / Path(row["relative_path"])
        else:
            raise ValueError(f"Bilinmeyen root_kind: {root_kind}")
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        items.append(
            ImageItem(
                key=row["image_key"],
                root_kind=root_kind,
                relative_path=row["relative_path"],
                source_group=row["source_group"],
                absolute_path=absolute,
            )
        )
    if len({item.key for item in items}) != len(items):
        raise ValueError("Scaled image index içinde tekrarlanan key var.")
    return items


def phash_radius_candidates(
    items: Sequence[ImageItem],
    original: np.ndarray,
    flipped: np.ndarray,
    radius: int,
    block_size: int = 256,
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    if radius < 0 or radius > 64:
        raise ValueError("pHash radius 0-64 arasında olmalıdır.")
    candidates: set[tuple[int, int]] = set()
    counts: Counter[str] = Counter()
    for source_group, indices in group_indices(items).items():
        global_indices = np.asarray(indices, dtype=np.int64)
        group_original = original[global_indices]
        group_flipped = flipped[global_indices]
        for start in range(0, len(indices), block_size):
            stop = min(start + block_size, len(indices))
            query_original = group_original[start:stop]
            query_flipped = group_flipped[start:stop]
            distances = np.minimum.reduce(
                (
                    np.bitwise_count(query_original[:, None] ^ group_original[None, :]),
                    np.bitwise_count(query_original[:, None] ^ group_flipped[None, :]),
                    np.bitwise_count(query_flipped[:, None] ^ group_original[None, :]),
                    np.bitwise_count(query_flipped[:, None] ^ group_flipped[None, :]),
                )
            )
            query_positions = np.arange(start, stop)[:, None]
            gallery_positions = np.arange(len(indices))[None, :]
            row_offsets, right_positions = np.nonzero(
                (distances <= radius) & (query_positions < gallery_positions)
            )
            for row_offset, right_position in zip(
                row_offsets.tolist(), right_positions.tolist()
            ):
                candidates.add(
                    (
                        int(global_indices[start + row_offset]),
                        int(global_indices[right_position]),
                    )
                )
        counts[source_group] = sum(
            1 for left, _ in candidates if items[left].source_group == source_group
        )
    return candidates, dict(sorted(counts.items()))


def load_top10_candidates(
    path: Path, items: Sequence[ImageItem]
) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for row in read_csv(path):
        left = int(row["left_image_index"])
        right = int(row["right_image_index"])
        if not (0 <= left < right < len(items)):
            raise ValueError(f"Geçersiz candidate index: {(left, right)}")
        if row["left_key"] != items[left].key or row["right_key"] != items[right].key:
            raise ValueError(f"Candidate key/index eşleşmiyor: {(left, right)}")
        candidates.add((left, right))
    return candidates


def robust_miss_rank_diagnostics(
    items: Sequence[ImageItem],
    descriptors_path: Path,
    references: Sequence[Mapping[str, str]],
    index_by_key: Mapping[str, int],
    top10: set[tuple[int, int]],
) -> list[dict[str, object]]:
    with np.load(descriptors_path) as loaded:
        descriptors = {name: loaded[name].copy() for name in loaded.files}
    if any(values.shape[0] != len(items) for values in descriptors.values()):
        raise ValueError("Scaled descriptor/image-index satır sayıları eşleşmiyor.")
    groups = group_indices(items)
    output: list[dict[str, object]] = []
    for row in references:
        if row["reference_label"] != "same_source":
            continue
        left = index_by_key[f"{row['left_root']}:{row['left_path']}"]
        right = index_by_key[f"{row['right_root']}:{row['right_path']}"]
        pair = (left, right) if left < right else (right, left)
        if pair in top10:
            continue
        source_group = items[left].source_group
        indices = groups[source_group]
        positions = {value: offset for offset, value in enumerate(indices)}
        left_local = positions[left]
        right_local = positions[right]
        best: tuple[int, int, int, float, str] | None = None
        for name, all_values in descriptors.items():
            values = all_values[indices]
            left_scores = values[left_local] @ values.T
            right_scores = values[right_local] @ values.T
            left_scores[left_local] = -np.inf
            right_scores[right_local] = -np.inf
            left_rank = 1 + int(
                np.flatnonzero(
                    np.argsort(-left_scores, kind="stable") == right_local
                )[0]
            )
            right_rank = 1 + int(
                np.flatnonzero(
                    np.argsort(-right_scores, kind="stable") == left_local
                )[0]
            )
            similarity = float(values[left_local] @ values[right_local])
            candidate = (
                min(left_rank, right_rank),
                left_rank,
                right_rank,
                similarity,
                name,
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("Rank diagnostic descriptor bulamadı.")
        output.append(
            {
                "pair_id": row["pair_id"],
                "transform_name": row.get("transform_name", ""),
                "best_union_rank": best[0],
                "left_to_right_rank": best[1],
                "right_to_left_rank": best[2],
                "cosine_similarity": best[3],
                "best_descriptor": best[4],
            }
        )
    return sorted(output, key=lambda entry: str(entry["pair_id"]))


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
    if args.workers < 1:
        raise ValueError("--workers en az 1 olmalıdır.")
    scaled_root = Path(args.scaled_root).resolve()
    calibration_root = Path(args.calibration_root).resolve()
    pairs_path = Path(args.pairs_csv).resolve()
    development_path = Path(args.development_decisions).resolve()
    holdout_path = Path(args.holdout_decisions).resolve()
    index_path = scaled_root / "image_index.csv"
    top10_path = scaled_root / "max_stage_top10_candidates.csv"
    descriptors_path = scaled_root / "robust_descriptors_max_stage.npz"
    output_path = scaled_root / f"posthoc_phash{args.radius}_safety.json"
    if output_path.exists():
        raise FileExistsError(f"Post-hoc çıktı zaten var: {output_path}")

    dataset_root = read_dataset_root(calibration_root)
    items = load_scaled_items(index_path, dataset_root, calibration_root)
    references = load_reference_pairs(
        pairs_path, development_path, holdout_path
    )
    index_by_key = {item.key: index for index, item in enumerate(items)}
    top10 = load_top10_candidates(top10_path, items)
    started = time.perf_counter()
    original, flipped = compute_phash_variants(items, args.workers)
    phash, phash_by_group = phash_radius_candidates(
        items, original, flipped, args.radius
    )
    union = top10 | phash

    top10_metrics = evaluate_candidate_set(
        top10, references, index_by_key, len(items)
    )
    phash_metrics = evaluate_candidate_set(
        phash, references, index_by_key, len(items)
    )
    union_metrics = evaluate_candidate_set(
        union, references, index_by_key, len(items)
    )
    recovered_pair_ids: list[str] = []
    for row in references:
        if row["reference_label"] != "same_source":
            continue
        left = index_by_key[f"{row['left_root']}:{row['left_path']}"]
        right = index_by_key[f"{row['right_root']}:{row['right_path']}"]
        pair = (left, right) if left < right else (right, left)
        if pair not in top10 and pair in phash:
            recovered_pair_ids.append(row["pair_id"])
    miss_diagnostics = robust_miss_rank_diagnostics(
        items, descriptors_path, references, index_by_key, top10
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "posthoc_exploratory_safety_channel_not_predeclared_acceptance",
        "analysis_type": "robust_top10_union_phash_radius_posthoc",
        "input": {
            "image_count": len(items),
            "reference_pair_count": len(references),
            "phash_radius": args.radius,
        },
        "candidate_counts": {
            "robust_top10": len(top10),
            "phash": len(phash),
            "intersection": len(top10 & phash),
            "union": len(union),
            "union_increment_over_top10": len(union) - len(top10),
            "union_pairs_per_image": len(union) / len(items),
            "phash_by_source_group": phash_by_group,
        },
        "reference_metrics": {
            "robust_top10": top10_metrics,
            "phash": phash_metrics,
            "union": union_metrics,
            "same_source_pairs_recovered_by_phash_only": sorted(recovered_pair_ids),
            "robust_top10_miss_rank_diagnostics": miss_diagnostics,
        },
        "timing_seconds": time.perf_counter() - started,
        "integrity": {
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "scaled_index_sha256": file_sha256(index_path),
            "scaled_top10_candidates_sha256": file_sha256(top10_path),
            "scaled_descriptors_sha256": file_sha256(descriptors_path),
            "pairs_csv_sha256": file_sha256(pairs_path),
        },
        "limitations": [
            "This safety channel was selected after observing the two scaled top-10 misses.",
            "It cannot retroactively pass the original predeclared robust-top-10 gate.",
            "The same revealed calibration references are used for method development.",
            "No full-dataset scan, frozen-rule scoring, family construction or split occurs.",
        ],
    }
    atomic_write_json(output_path, report)
    print(f"Post-hoc pHash safety raporu: {output_path}")
    print(f"Top-10 pozitif recall: {top10_metrics['positive_hits']}/{top10_metrics['positive_count']}")
    print(f"Union pozitif recall: {union_metrics['positive_hits']}/{union_metrics['positive_count']}")
    print(f"pHash-only kurtarılan Pair ID: {sorted(recovered_pair_ids)}")
    print("NOT: Bu post-hoc geliştirme kanıtıdır; orijinal pilot kapısını geriye dönük geçirmez.")
    return output_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
