from __future__ import annotations

"""Build a frozen, score-blind natural near-duplicate review manifest.

The builder reads only saved indexes, hashes and descriptors. It verifies that
dataset files exist but does not decode or render image pixels, label pairs,
construct families/splits, or train a classifier.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from analyze_scaled_phash8_safety import phash_radius_candidates
from benchmark_candidate_generation import ImageItem, flip_aware_hamming
from benchmark_candidate_generation_scaled import blockwise_topk_candidates
from data_index import project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


PRIVATE_FIELDS = (
    "pair_id",
    "selection_stratum",
    "source_group",
    "left_image_index",
    "right_image_index",
    "left_key",
    "right_key",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "robust_top15",
    "robust_top30",
    "phash_distance",
    "phash_radius12",
    "phash_radius16",
    "robust_best_union_rank",
    "robust_best_cosine_similarity",
)
REVIEW_FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
)
STRATA = (
    "candidate_robust_only",
    "candidate_phash_supported",
    "outside_robust_rank16_30",
    "outside_phash_distance14_16",
    "outside_random",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "V2.2 indeksindeki gerçek dataset görüntülerinden score-blind doğal "
            "near-duplicate audit manifesti seçer; görüntü açmaz, split/model üretmez."
        )
    )
    parser.add_argument(
        "--config", default=str(root / "config" / "natural_duplicate_audit_v1.json")
    )
    parser.add_argument(
        "--config-sha256",
        default=str(root / "config" / "natural_duplicate_audit_v1.sha256"),
    )
    parser.add_argument(
        "--holdout-root",
        default=str(root / "artifacts" / "audit" / "candidate_generator_v2_2_holdout"),
    )
    parser.add_argument(
        "--output-dir", default=str(root / "research" / "natural_duplicate_audit_v1")
    )
    parser.add_argument("--data-dir", default=None)
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


def stable_subseed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_sample(
    pool: Iterable[tuple[int, int]], count: int, seed: int, label: str
) -> list[tuple[int, int]]:
    ordered = sorted(set(pool))
    if count < 0:
        raise ValueError("Örnek sayısı negatif olamaz.")
    if len(ordered) <= count:
        return ordered
    rng = random.Random(stable_subseed(seed, label))
    return sorted(rng.sample(ordered, count))


def balanced_phash_sample(
    phash_only: Iterable[tuple[int, int]],
    both: Iterable[tuple[int, int]],
    maximum: int,
    per_channel_target: int,
    seed: int,
    label: str,
) -> list[tuple[int, int]]:
    selected = deterministic_sample(
        phash_only, min(per_channel_target, maximum), seed, f"{label}:phash_only"
    )
    remaining = maximum - len(selected)
    chosen_both = deterministic_sample(
        both, min(per_channel_target, remaining), seed, f"{label}:both"
    )
    selected.extend(chosen_both)
    remaining = maximum - len(selected)
    if remaining:
        leftovers = (set(phash_only) | set(both)) - set(selected)
        selected.extend(
            deterministic_sample(leftovers, remaining, seed, f"{label}:fill")
        )
    return sorted(selected)


def reservoir_outside_sample(
    indices: Sequence[int],
    excluded: set[tuple[int, int]],
    count: int,
    seed: int,
    label: str,
) -> list[tuple[int, int]]:
    rng = random.Random(stable_subseed(seed, label))
    reservoir: list[tuple[int, int]] = []
    eligible_seen = 0
    for pair in combinations(sorted(indices), 2):
        if pair in excluded:
            continue
        eligible_seen += 1
        if len(reservoir) < count:
            reservoir.append(pair)
        else:
            position = rng.randrange(eligible_seen)
            if position < count:
                reservoir[position] = pair
    if len(reservoir) != count:
        raise ValueError(
            f"{label}: {count} random dış çift için yalnız {eligible_seen} uygun çift var."
        )
    return sorted(reservoir)


def load_config(config_path: Path, hash_path: Path) -> Mapping[str, object]:
    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = file_sha256(config_path)
    if expected != actual:
        raise ValueError(
            f"Natural audit config hash uyuşmuyor: expected={expected}, actual={actual}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_natural_pair_selection":
        raise ValueError("Natural audit config frozen durumda değil.")
    return config


def load_items(
    index_path: Path, dataset_root: Path
) -> tuple[list[ImageItem], list[dict[str, str]]]:
    rows = read_csv(index_path)
    rows.sort(key=lambda row: int(row["image_index"]))
    items: list[ImageItem] = []
    for expected, row in enumerate(rows):
        if int(row["image_index"]) != expected:
            raise ValueError("image_index kesintisiz değil.")
        if row["root_kind"] == "dataset":
            absolute = (dataset_root / Path(row["relative_path"])).resolve()
        else:
            # Generated rows are kept only to preserve descriptor/index alignment.
            absolute = Path(row["relative_path"])
        items.append(
            ImageItem(
                key=row["image_key"],
                root_kind=row["root_kind"],
                relative_path=row["relative_path"],
                source_group=row["source_group"],
                absolute_path=absolute,
            )
        )
    return items, rows


def dataset_pairs(
    pairs: Iterable[tuple[int, int]], dataset_indices: set[int]
) -> set[tuple[int, int]]:
    return {
        pair for pair in pairs if pair[0] in dataset_indices and pair[1] in dataset_indices
    }


def pools_by_group(
    pairs: Iterable[tuple[int, int]], items: Sequence[ImageItem]
) -> dict[str, set[tuple[int, int]]]:
    output: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for left, right in pairs:
        if items[left].source_group != items[right].source_group:
            raise ValueError("Source-group dışı çift bulundu.")
        output[items[left].source_group].add((left, right))
    return output


def exact_directed_rank(
    query: int,
    target: int,
    group_indices: Sequence[int],
    descriptor: np.ndarray,
) -> int:
    gallery = descriptor[np.asarray(group_indices, dtype=np.int64)]
    similarities = gallery @ descriptor[query]
    target_position = group_indices.index(target)
    query_position = group_indices.index(query)
    similarities[query_position] = -np.inf
    target_similarity = similarities[target_position]
    preceding_ties = np.sum(
        (similarities == target_similarity)
        & (np.arange(len(group_indices)) < target_position)
    )
    return 1 + int(np.sum(similarities > target_similarity)) + int(preceding_ties)


def robust_pair_stats(
    left: int,
    right: int,
    group_indices: Sequence[int],
    descriptors: Mapping[str, np.ndarray],
) -> tuple[int, float]:
    best_rank = 10**9
    best_similarity = -1.0
    for values in descriptors.values():
        similarity = float(values[left] @ values[right])
        best_similarity = max(best_similarity, similarity)
        best_rank = min(
            best_rank,
            exact_directed_rank(left, right, group_indices, values),
            exact_directed_rank(right, left, group_indices, values),
        )
    return best_rank, best_similarity


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    if args.retrieval_block_size < 1:
        raise ValueError("--retrieval-block-size pozitif olmalıdır.")
    root = project_root()
    config_path = Path(args.config).resolve()
    config_hash_path = Path(args.config_sha256).resolve()
    holdout_root = Path(args.holdout_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Natural audit seçimi zaten var: {output_dir}")
    config = load_config(config_path, config_hash_path)
    source = dict(config["source"])
    sampling = dict(config["sampling"])
    seed = int(sampling["seed"])

    dataset_root = resolve_dataset_root(args.data_dir)
    index_path = holdout_root / "image_index.csv"
    descriptor_path = holdout_root / "evaluation" / "robust_descriptors.npz"
    phash_path = holdout_root / "evaluation" / "phash_variants.npz"
    source_report_path = holdout_root / "evaluation" / "report.json"
    items, index_rows = load_items(index_path, dataset_root)
    dataset_indices = {
        index for index, item in enumerate(items) if item.root_kind == "dataset"
    }
    if len(dataset_indices) != int(source["expected_dataset_image_count"]):
        raise ValueError("Dataset görüntü sayısı frozen config ile eşleşmiyor.")
    dataset_rows = [index_rows[index] for index in sorted(dataset_indices)]
    if len({row["leaf"] for row in dataset_rows}) != int(source["expected_leaf_count"]):
        raise ValueError("Leaf kapsamı frozen config ile eşleşmiyor.")
    if len({row["source_group"] for row in dataset_rows}) != int(
        source["expected_source_group_count"]
    ):
        raise ValueError("Source-group kapsamı frozen config ile eşleşmiyor.")
    for index in dataset_indices:
        if not items[index].absolute_path.is_file():
            raise FileNotFoundError(items[index].absolute_path)

    with np.load(descriptor_path) as loaded:
        descriptors = {name: loaded[name].copy() for name in loaded.files}
    with np.load(phash_path) as loaded:
        phash_original = loaded["original"].copy()
        phash_flipped = loaded["flipped"].copy()

    robust_sets, retrieval_metadata = blockwise_topk_candidates(
        items,
        descriptors,
        args.retrieval_block_size,
        topk_values=(
            int(dict(config["v2_3_hypothesis"])["robust_top_k"]),
            int(sampling["outside_robust_rank_max"]),
        ),
    )
    robust15 = dataset_pairs(robust_sets[15], dataset_indices)
    robust30 = dataset_pairs(robust_sets[30], dataset_indices)
    phash16_all, _ = phash_radius_candidates(
        items,
        phash_original,
        phash_flipped,
        int(sampling["outside_phash_distance_max"]),
        block_size=args.retrieval_block_size,
    )
    phash16 = dataset_pairs(phash16_all, dataset_indices)
    phash12 = {
        pair
        for pair in phash16
        if flip_aware_hamming(
            pair[0], pair[1], phash_original, phash_flipped
        )
        <= int(dict(config["v2_3_hypothesis"])["phash_radius_inclusive"])
    }

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index in sorted(dataset_indices):
        group_indices[items[index].source_group].append(index)
    robust_only_by = pools_by_group(robust15 - phash12, items)
    phash_only_by = pools_by_group(phash12 - robust15, items)
    both_by = pools_by_group(phash12 & robust15, items)
    robust_boundary_by = pools_by_group(robust30 - robust15 - phash12, items)
    phash_boundary_by = pools_by_group(phash16 - phash12 - robust15, items)

    selected_with_stratum: list[tuple[str, tuple[int, int]]] = []
    selected_pairs: set[tuple[int, int]] = set()
    for group in sorted(group_indices):
        chosen = deterministic_sample(
            robust_only_by[group],
            int(sampling["candidate_robust_only_per_source_group"]),
            seed,
            f"{group}:candidate_robust_only",
        )
        selected_with_stratum.extend(("candidate_robust_only", pair) for pair in chosen)
        selected_pairs.update(chosen)

        chosen = balanced_phash_sample(
            phash_only_by[group],
            both_by[group],
            int(sampling["candidate_phash_supported_max_per_source_group"]),
            int(sampling["candidate_phash_supported_target_per_channel"]),
            seed,
            f"{group}:candidate_phash_supported",
        )
        selected_with_stratum.extend(("candidate_phash_supported", pair) for pair in chosen)
        selected_pairs.update(chosen)

        chosen = deterministic_sample(
            robust_boundary_by[group] - selected_pairs,
            int(sampling["outside_robust_per_source_group"]),
            seed,
            f"{group}:outside_robust",
        )
        selected_with_stratum.extend(("outside_robust_rank16_30", pair) for pair in chosen)
        selected_pairs.update(chosen)

        chosen = deterministic_sample(
            phash_boundary_by[group] - selected_pairs,
            int(sampling["outside_phash_per_source_group"]),
            seed,
            f"{group}:outside_phash",
        )
        selected_with_stratum.extend(("outside_phash_distance14_16", pair) for pair in chosen)
        selected_pairs.update(chosen)

        chosen = reservoir_outside_sample(
            group_indices[group],
            robust30 | phash16 | selected_pairs,
            int(sampling["outside_random_per_source_group"]),
            seed,
            f"{group}:outside_random",
        )
        selected_with_stratum.extend(("outside_random", pair) for pair in chosen)
        selected_pairs.update(chosen)

    if len(selected_pairs) != len(selected_with_stratum):
        raise ValueError("Natural audit strata arasında tekrarlanan çift var.")
    if len(selected_pairs) != int(sampling["expected_pair_count"]):
        raise ValueError(
            f"Beklenen {sampling['expected_pair_count']} yerine {len(selected_pairs)} çift seçildi."
        )

    rng = random.Random(stable_subseed(seed, "pair_id_order"))
    rng.shuffle(selected_with_stratum)
    private_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for ordinal, (stratum, (left, right)) in enumerate(selected_with_stratum, start=1):
        pair_id = f"{sampling['pair_id_prefix']}{ordinal:06d}"
        left_row = index_rows[left]
        right_row = index_rows[right]
        group = items[left].source_group
        best_rank, best_similarity = robust_pair_stats(
            left, right, group_indices[group], descriptors
        )
        distance = flip_aware_hamming(
            left, right, phash_original, phash_flipped
        )
        private_rows.append(
            {
                "pair_id": pair_id,
                "selection_stratum": stratum,
                "source_group": group,
                "left_image_index": left,
                "right_image_index": right,
                "left_key": items[left].key,
                "right_key": items[right].key,
                "left_leaf": left_row["leaf"],
                "right_leaf": right_row["leaf"],
                "left_path": left_row["relative_path"],
                "right_path": right_row["relative_path"],
                "robust_top15": "yes" if (left, right) in robust15 else "no",
                "robust_top30": "yes" if (left, right) in robust30 else "no",
                "phash_distance": distance,
                "phash_radius12": "yes" if (left, right) in phash12 else "no",
                "phash_radius16": "yes" if (left, right) in phash16 else "no",
                "robust_best_union_rank": best_rank,
                "robust_best_cosine_similarity": f"{best_similarity:.9f}",
            }
        )
        review_rows.append(
            {
                "pair_id": pair_id,
                "source_group": group,
                "left_leaf": left_row["leaf"],
                "right_leaf": right_row["leaf"],
                "left_path": left_row["relative_path"],
                "right_path": right_row["relative_path"],
                "visual_verdict": "",
                "confidence": "",
                "reviewer": "",
                "review_date": "",
                "review_notes": "",
            }
        )

    counts_by_stratum = Counter(row["selection_stratum"] for row in private_rows)
    counts_by_group = Counter(row["source_group"] for row in private_rows)
    with tempfile.TemporaryDirectory(prefix="natural-audit-v1-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        staging.mkdir()
        private_path = staging / "selection_private.csv"
        review_path = staging / "review_manifest.csv"
        write_csv(private_path, PRIVATE_FIELDS, private_rows)
        write_csv(review_path, REVIEW_FIELDS, review_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "selection_frozen_images_not_opened",
            "experiment_type": "score_blind_natural_near_duplicate_audit_selection",
            "counts": {
                "dataset_images": len(dataset_indices),
                "selected_pairs": len(private_rows),
                "unique_pairs": len(selected_pairs),
                "unique_pair_ids": len({row["pair_id"] for row in private_rows}),
                "blank_review_verdicts": sum(
                    not str(row["visual_verdict"]) for row in review_rows
                ),
                "source_groups": len(group_indices),
                "leaves": len({row["leaf"] for row in dataset_rows}),
                "by_stratum": dict(sorted(counts_by_stratum.items())),
                "by_source_group": dict(sorted(counts_by_group.items())),
            },
            "candidate_universe": {
                "possible_dataset_pairs_within_source_group": sum(
                    len(indices) * (len(indices) - 1) // 2
                    for indices in group_indices.values()
                ),
                "robust_top15_pairs": len(robust15),
                "phash_radius12_pairs": len(phash12),
                "intersection_pairs": len(robust15 & phash12),
                "v2_3_union_pairs": len(robust15 | phash12),
                "robust_rank16_30_outside_pairs": len(robust30 - robust15 - phash12),
                "phash_distance14_16_outside_pairs": len(phash16 - phash12 - robust15),
            },
            "retrieval_metadata": retrieval_metadata,
            "integrity": {
                "config_sha256": file_sha256(config_path),
                "source_report_sha256": file_sha256(source_report_path),
                "image_index_sha256": file_sha256(index_path),
                "robust_descriptors_sha256": file_sha256(descriptor_path),
                "phash_variants_sha256": file_sha256(phash_path),
                "builder_sha256": file_sha256(Path(__file__).resolve()),
                "selection_private_sha256": file_sha256(private_path),
                "review_manifest_sha256": file_sha256(review_path),
            },
            "limitations": list(config["interpretation_limits"]),
        }
        report_path = staging / "selection_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(staging), str(output_dir))

    print("İşlem: NATURAL DUPLICATE AUDIT v1 SCORE-BLIND SELECTION")
    print(f"Dataset görüntüsü: {len(dataset_indices)}")
    print(f"Seçilen çift: {len(private_rows)}")
    print(f"Strata: {dict(sorted(counts_by_stratum.items()))}")
    print("Görüntü pikseli açılmadı; verdict alanları boştur.")
    print(f"Rapor: {output_dir / 'selection_report.json'}")
    return output_dir / "selection_report.json", report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
