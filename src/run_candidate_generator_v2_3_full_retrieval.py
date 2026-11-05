from __future__ import annotations

"""Resumable, hash-guarded Candidate Generator v2.3 retrieval orchestrator.

The default scope is deliberately a tiny smoke run. The 130,000-image scope needs
an explicit confirmation token and still performs candidate retrieval only: no
duplicate decision, family construction, data split, or classifier training.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import PIL
import torch
import torchvision
from torchvision.models import ResNet18_Weights

from benchmark_candidate_generation import ImageItem, compute_phash_variants
from benchmark_candidate_generation_v2 import DESCRIPTOR_NAMES, compute_robust_descriptors
from candidate_generator_v2_3_runtime import (
    ResumeManifest,
    atomic_csv,
    atomic_json,
    atomic_npz,
    decode_pair_keys,
    exact_duplicate_pair_keys,
    merge_candidate_channels,
    phash_block_candidate_keys,
    robust_channel_candidate_keys,
    verify_exact_duplicate_coverage,
    verify_exact_duplicate_manifest,
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


FULL_CONFIRMATION = "FROZEN_V2_3_130000"
INDEX_FIELDS = (
    "image_index",
    "relative_path",
    "main_class",
    "leaf",
    "source_group",
    "selection_rank_within_leaf",
)
CANDIDATE_FIELDS = (
    "pair_id",
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
STOP_STAGES = ("index", "phash", "descriptors", "retrieval", "final")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Candidate Generator v2.3 resumable retrieval orchestrator. Default is a "
            "small smoke scope; full scope requires an explicit confirmation token."
        )
    )
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--confirm-full-run", default="")
    parser.add_argument("--smoke-images-per-leaf", type=int, default=2)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--exact-duplicates",
        default=str(root / "artifacts" / "audit" / "exact_duplicates.json"),
    )
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
            / "candidate_generator_v2_3_resumable_smoke"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--robust-block-size", type=int, default=256)
    parser.add_argument("--phash-block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after", choices=STOP_STAGES, default="final")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.scope == "full" and args.confirm_full_run != FULL_CONFIRMATION:
        raise ValueError(
            "Full retrieval is locked. Pass --confirm-full-run " + FULL_CONFIRMATION
        )
    if args.scope == "smoke" and not 1 <= args.smoke_images_per_leaf <= 20:
        raise ValueError("--smoke-images-per-leaf must be in 1..20.")
    for name in ("batch_size", "workers", "robust_block_size", "phash_block_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")


def environment_summary() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def ensure_checkpoint_available() -> None:
    """Download official weights when absent; verify_checkpoint checks the full hash."""

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


def select_records(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    source_by_leaf: Mapping[str, str],
    scope: str,
    smoke_images_per_leaf: int,
    seed: int,
) -> list[ImageRecord]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.subclass].append(record)
    if set(grouped) != set(source_by_leaf):
        raise ValueError("Dataset leaves do not match frozen source groups.")
    selected: list[ImageRecord] = []
    for leaf_index, leaf in enumerate(sorted(grouped)):
        leaf_records = sorted(
            grouped[leaf], key=lambda row: row.path.relative_to(dataset_root).as_posix()
        )
        if scope == "smoke":
            random.Random(seed + leaf_index * 1009).shuffle(leaf_records)
            leaf_records = leaf_records[:smoke_images_per_leaf]
        selected.extend(leaf_records)
    return sorted(
        selected, key=lambda row: row.path.relative_to(dataset_root).as_posix()
    )


def build_items(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    source_by_leaf: Mapping[str, str],
) -> list[ImageItem]:
    return [
        ImageItem(
            key=f"dataset:{record.path.relative_to(dataset_root).as_posix()}",
            root_kind="dataset",
            relative_path=record.path.relative_to(dataset_root).as_posix(),
            source_group=source_by_leaf[record.subclass],
            absolute_path=record.path.resolve(),
        )
        for record in records
    ]


def index_rows(
    items: Sequence[ImageItem], records: Sequence[ImageRecord]
) -> Iterable[Mapping[str, object]]:
    ranks: Counter[str] = Counter()
    for index, (item, record) in enumerate(zip(items, records, strict=True)):
        ranks[record.subclass] += 1
        yield {
            "image_index": index,
            "relative_path": item.relative_path,
            "main_class": record.main_class,
            "leaf": record.subclass,
            "source_group": item.source_group,
            "selection_rank_within_leaf": ranks[record.subclass],
        }


def create_or_validate_contract(
    output_dir: Path, core: Mapping[str, object]
) -> tuple[Path, Mapping[str, object]]:
    contract_path = output_dir / "run_contract.json"
    if output_dir.exists() and not contract_path.is_file():
        raise FileExistsError(
            "Output directory exists without a run contract; refusing reuse: "
            f"{output_dir}"
        )
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("core") != core:
            raise ValueError(
                "Existing run contract differs from current code/config/environment/arguments."
            )
        return contract_path, contract
    output_dir.mkdir(parents=True, exist_ok=False)
    contract: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "core": dict(core),
    }
    atomic_json(contract_path, contract)
    return contract_path, contract


def indices_by_leaf(records: Sequence[ImageRecord]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        output[record.subclass].append(index)
    return dict(sorted(output.items()))


def indices_by_group(items: Sequence[ImageItem]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        output[item.source_group].append(index)
    return dict(sorted(output.items()))


def load_group_phashes(
    output_dir: Path,
    group_indices: Sequence[int],
    records: Sequence[ImageRecord],
    leaf_indices: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    position = {global_index: local for local, global_index in enumerate(group_indices)}
    original = np.empty(len(group_indices), dtype=np.uint64)
    flipped = np.empty(len(group_indices), dtype=np.uint64)
    leaves = sorted({records[index].subclass for index in group_indices})
    for leaf in leaves:
        with np.load(output_dir / "stages" / "phash" / f"{leaf}.npz") as loaded:
            first = np.asarray(loaded["original"], dtype=np.uint64)
            second = np.asarray(loaded["horizontal_flip"], dtype=np.uint64)
        expected = list(leaf_indices[leaf])
        if first.shape != (len(expected),) or second.shape != (len(expected),):
            raise ValueError(f"Saved pHash shape mismatch for leaf {leaf}.")
        for leaf_position, global_index in enumerate(expected):
            if global_index in position:
                output_position = position[global_index]
                original[output_position] = first[leaf_position]
                flipped[output_position] = second[leaf_position]
    return original, flipped


def load_group_descriptor_channel(
    output_dir: Path,
    group_indices: Sequence[int],
    records: Sequence[ImageRecord],
    leaf_indices: Mapping[str, Sequence[int]],
    channel: str,
) -> np.ndarray:
    position = {global_index: local for local, global_index in enumerate(group_indices)}
    descriptors = np.empty((len(group_indices), 512), dtype=np.float32)
    leaves = sorted({records[index].subclass for index in group_indices})
    for leaf in leaves:
        with np.load(output_dir / "stages" / "descriptors" / f"{leaf}.npz") as loaded:
            values = np.asarray(loaded[channel], dtype=np.float32)
        expected = list(leaf_indices[leaf])
        if values.shape != (len(expected), 512):
            raise ValueError(f"Saved descriptor shape mismatch for {leaf}/{channel}.")
        for leaf_position, global_index in enumerate(expected):
            if global_index in position:
                descriptors[position[global_index]] = values[leaf_position]
    return descriptors


def candidate_rows(
    keys: np.ndarray,
    robust_flags: np.ndarray,
    phash_flags: np.ndarray,
    items: Sequence[ImageItem],
    records: Sequence[ImageRecord],
    scope: str,
) -> Iterable[Mapping[str, object]]:
    left, right = decode_pair_keys(keys)
    prefix = "V23F" if scope == "full" else "V23S"
    for offset, (left_index, right_index, robust, phash) in enumerate(
        zip(left, right, robust_flags, phash_flags, strict=True), start=1
    ):
        left_int = int(left_index)
        right_int = int(right_index)
        channels = []
        if bool(robust):
            channels.append("robust_top15")
        if bool(phash):
            channels.append("phash_le12")
        yield {
            "pair_id": f"{prefix}{offset:09d}",
            "left_image_index": left_int,
            "right_image_index": right_int,
            "left_relative_path": items[left_int].relative_path,
            "right_relative_path": items[right_int].relative_path,
            "left_leaf": records[left_int].subclass,
            "right_leaf": records[right_int].subclass,
            "source_group": items[left_int].source_group,
            "robust_top15": "yes" if robust else "no",
            "phash_le12": "yes" if phash else "no",
            "candidate_channels": "+".join(channels),
        }


def stop_requested(args: argparse.Namespace, stage: str) -> bool:
    if args.stop_after == stage:
        print(f"Controlled stop after stage: {stage}")
        print("Resume with the same arguments and a later --stop-after value.")
        return True
    return False


def run(args: argparse.Namespace) -> Path | None:
    validate_args(args)
    root = project_root().resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    config_hash_path = Path(args.config_sha256).resolve()
    exact_path = Path(args.exact_duplicates).resolve()
    dataset_root = resolve_dataset_root(args.data_dir)

    config = load_frozen_config(config_path, config_hash_path)
    dependencies = verify_dependency_hashes(root, config)
    ensure_checkpoint_available()
    checkpoint = verify_checkpoint(config)
    groups_path = root / "config" / "duplicate_audit_groups.json"
    full_inventory = verify_dataset_inventory(dataset_root, config, groups_path)
    source_by_leaf = load_source_group_map(groups_path)
    normalized_exact, normalized_exact_hash = verify_exact_duplicate_manifest(exact_path)

    all_records = discover_records(dataset_root)
    records = select_records(
        all_records,
        dataset_root,
        source_by_leaf,
        args.scope,
        args.smoke_images_per_leaf,
        args.seed,
    )
    items = build_items(records, dataset_root, source_by_leaf)
    leaf_indices = indices_by_leaf(records)
    group_indices_map = indices_by_group(items)
    expected_images = 130_000 if args.scope == "full" else 26 * args.smoke_images_per_leaf
    if len(items) != expected_images:
        raise ValueError(f"Selected image count mismatch: {len(items)} != {expected_images}")

    script_path = Path(__file__).resolve()
    runtime_path = root / "src" / "candidate_generator_v2_3_runtime.py"
    core: dict[str, object] = {
        "scope": args.scope,
        "frozen_config_sha256": file_sha256(config_path),
        "orchestrator_sha256": file_sha256(script_path),
        "runtime_sha256": file_sha256(runtime_path),
        "frozen_dependency_hashes": dependencies,
        "checkpoint_sha256": checkpoint["sha256"],
        "normalized_exact_duplicate_sha256": normalized_exact_hash,
        "exact_duplicate_input_sha256": file_sha256(exact_path),
        "dataset_root": str(dataset_root),
        "full_inventory": full_inventory,
        "selected_image_count": len(items),
        "selected_leaf_counts": dict(sorted(Counter(r.subclass for r in records).items())),
        "source_group_counts": dict(sorted(Counter(i.source_group for i in items).items())),
        "method": {
            "top_k": 15,
            "phash_radius": 12,
            "descriptor_channels": list(DESCRIPTOR_NAMES),
            "search_scope": "within_compatible_source_group_only",
        },
        "arguments": {
            "smoke_images_per_leaf": args.smoke_images_per_leaf,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "robust_block_size": args.robust_block_size,
            "phash_block_size": args.phash_block_size,
            "seed": args.seed,
        },
        "environment": environment_summary(),
        "forbidden_stages": [
            "duplicate_decision_scoring",
            "family_construction",
            "train_validation_test_split",
            "classifier_training",
        ],
    }
    contract_path, _ = create_or_validate_contract(output_dir, core)
    contract_sha = file_sha256(contract_path)
    manifest = ResumeManifest(output_dir, contract_sha)
    manifest.verify_registered()

    print("Operation: RESUMABLE CANDIDATE GENERATOR v2.3 RETRIEVAL")
    print(f"Scope: {args.scope}; selected images: {len(items)}")
    print(f"Output: {output_dir}")
    print(f"Registered artifacts verified: {len(manifest.artifacts)}")
    print("No duplicate decision, family, split, or classifier training will run.")

    index_relative = "stages/image_index.csv"
    if manifest.is_complete(index_relative):
        print("RESUME index: verified and skipped")
    else:
        started = time.perf_counter()
        atomic_csv(output_dir / index_relative, INDEX_FIELDS, index_rows(items, records))
        manifest.register(
            index_relative,
            "index",
            {"rows": len(items), "seconds": time.perf_counter() - started},
        )
    if stop_requested(args, "index"):
        return None

    for leaf, selected_indices in leaf_indices.items():
        relative = f"stages/phash/{leaf}.npz"
        if manifest.is_complete(relative):
            print(f"RESUME pHash {leaf}: verified and skipped")
            continue
        started = time.perf_counter()
        leaf_items = [items[index] for index in selected_indices]
        original, flipped = compute_phash_variants(leaf_items, args.workers)
        atomic_npz(
            output_dir / relative,
            original=original,
            horizontal_flip=flipped,
        )
        manifest.register(
            relative,
            "phash_generation",
            {
                "leaf": leaf,
                "rows": len(leaf_items),
                "seconds": time.perf_counter() - started,
            },
        )
    if stop_requested(args, "phash"):
        return None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for leaf, selected_indices in leaf_indices.items():
        relative = f"stages/descriptors/{leaf}.npz"
        if manifest.is_complete(relative):
            print(f"RESUME descriptors {leaf}: verified and skipped")
            continue
        started = time.perf_counter()
        leaf_items = [items[index] for index in selected_indices]
        descriptors, metadata = compute_robust_descriptors(leaf_items, args.batch_size)
        if metadata["checkpoint_sha256"] != checkpoint["sha256"]:
            raise ValueError("Descriptor checkpoint differs from frozen checkpoint.")
        if any(values.shape != (len(leaf_items), 512) for values in descriptors.values()):
            raise ValueError(f"Descriptor shape mismatch for leaf {leaf}.")
        atomic_npz(output_dir / relative, **descriptors)
        manifest.register(
            relative,
            "descriptor_generation",
            {
                "leaf": leaf,
                "rows": len(leaf_items),
                "seconds": time.perf_counter() - started,
                "device": metadata["device"],
            },
        )
    if stop_requested(args, "descriptors"):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust_relatives: list[str] = []
    phash_relatives: list[str] = []
    for source_group, selected_indices_list in group_indices_map.items():
        selected_indices = np.asarray(selected_indices_list, dtype=np.int64)
        for channel in DESCRIPTOR_NAMES:
            relative = f"stages/robust_candidates/{source_group}/{channel}.npz"
            robust_relatives.append(relative)
            if manifest.is_complete(relative):
                print(f"RESUME robust {source_group}/{channel}: verified and skipped")
                continue
            descriptors = load_group_descriptor_channel(
                output_dir,
                selected_indices_list,
                records,
                leaf_indices,
                channel,
            )
            started = time.perf_counter()
            keys = robust_channel_candidate_keys(
                descriptors,
                selected_indices,
                top_k=15,
                block_size=args.robust_block_size,
                device=device,
            )
            atomic_npz(output_dir / relative, pair_keys=keys)
            manifest.register(
                relative,
                "robust_retrieval",
                {
                    "source_group": source_group,
                    "channel": channel,
                    "pairs": len(keys),
                    "seconds": time.perf_counter() - started,
                    "device": str(device),
                },
            )

        original, flipped = load_group_phashes(
            output_dir,
            selected_indices_list,
            records,
            leaf_indices,
        )
        for start in range(0, len(selected_indices), args.phash_block_size):
            stop = min(start + args.phash_block_size, len(selected_indices))
            relative = (
                f"stages/phash_candidates/{source_group}/"
                f"block_{start:06d}_{stop:06d}.npz"
            )
            phash_relatives.append(relative)
            if manifest.is_complete(relative):
                print(f"RESUME pHash retrieval {source_group} {start}:{stop}: skipped")
                continue
            started = time.perf_counter()
            keys = phash_block_candidate_keys(
                original,
                flipped,
                selected_indices,
                start,
                stop,
                radius=12,
            )
            atomic_npz(output_dir / relative, pair_keys=keys)
            manifest.register(
                relative,
                "phash_retrieval",
                {
                    "source_group": source_group,
                    "start": start,
                    "stop": stop,
                    "pairs": len(keys),
                    "seconds": time.perf_counter() - started,
                },
            )
    if stop_requested(args, "retrieval"):
        return None

    final_report_relative = "run_report.json"
    final_manifest_relative = "hash_manifest.json"
    if manifest.is_complete(final_report_relative) and manifest.is_complete(
        final_manifest_relative
    ):
        print("RESUME final outputs: verified and already complete")
        return output_dir / final_report_relative

    robust_parts = []
    for relative in robust_relatives:
        with np.load(output_dir / relative) as loaded:
            robust_parts.append(np.asarray(loaded["pair_keys"], dtype=np.uint64))
    phash_parts = []
    for relative in phash_relatives:
        with np.load(output_dir / relative) as loaded:
            phash_parts.append(np.asarray(loaded["pair_keys"], dtype=np.uint64))
    keys, robust_flags, phash_flags = merge_candidate_channels(
        robust_parts, phash_parts
    )
    left, right = decode_pair_keys(keys)
    if np.any(left >= right):
        raise ValueError("Final candidate pairs are not canonical.")
    cross_group = sum(
        items[int(left_index)].source_group != items[int(right_index)].source_group
        for left_index, right_index in zip(left, right, strict=True)
    )
    if cross_group:
        raise ValueError(f"Final candidate union contains {cross_group} cross-group pairs.")
    if len(keys) > 25.0 * len(items):
        raise ValueError(
            "Candidate load exceeds frozen pause gate: "
            f"{len(keys) / len(items):.3f} pairs/image"
        )

    index_by_relative = {item.relative_path: index for index, item in enumerate(items)}
    expected_exact = exact_duplicate_pair_keys(normalized_exact, index_by_relative)
    exact_coverage = verify_exact_duplicate_coverage(keys, expected_exact)
    if args.scope == "full" and len(expected_exact) != 1393:
        raise ValueError(
            f"Full scope expected 1393 exact-duplicate pairs, found {len(expected_exact)}."
        )

    candidate_npz_relative = "candidate_pairs.npz"
    if not manifest.is_complete(candidate_npz_relative):
        started = time.perf_counter()
        atomic_npz(
            output_dir / candidate_npz_relative,
            pair_keys=keys,
            robust_top15=robust_flags,
            phash_le12=phash_flags,
        )
        manifest.register(
            candidate_npz_relative,
            "finalization",
            {"pairs": len(keys), "seconds": time.perf_counter() - started},
        )

    candidate_csv_relative = "candidate_pairs.csv"
    if not manifest.is_complete(candidate_csv_relative):
        started = time.perf_counter()
        atomic_csv(
            output_dir / candidate_csv_relative,
            CANDIDATE_FIELDS,
            candidate_rows(keys, robust_flags, phash_flags, items, records, args.scope),
        )
        manifest.register(
            candidate_csv_relative,
            "finalization",
            {"rows": len(keys), "seconds": time.perf_counter() - started},
        )

    candidate_counts_by_group: Counter[str] = Counter()
    for left_index in left:
        candidate_counts_by_group[items[int(left_index)].source_group] += 1
    report: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "candidate_generator_v2_3_full_retrieval_complete"
            if args.scope == "full"
            else "candidate_generator_v2_3_resumable_smoke_complete"
        ),
        "experiment_type": "candidate_retrieval_only",
        "scope": args.scope,
        "not_a_classifier_result": True,
        "run_contract_sha256": contract_sha,
        "frozen_config_sha256": core["frozen_config_sha256"],
        "input": {
            "selected_image_count": len(items),
            "leaf_count": len(leaf_indices),
            "source_group_count": len(group_indices_map),
            "selected_leaf_counts": core["selected_leaf_counts"],
        },
        "candidate_counts": {
            "robust_unique_pairs": int(sum(robust_flags)),
            "phash_unique_pairs": int(sum(phash_flags)),
            "intersection_pairs": int(sum(robust_flags & phash_flags)),
            "union_pairs": len(keys),
            "union_pairs_per_image": len(keys) / len(items),
            "candidate_counts_by_source_group": dict(
                sorted(candidate_counts_by_group.items())
            ),
        },
        "integrity": {
            "unique_canonical_pairs": len(keys),
            "cross_source_group_pairs": 0,
            **exact_coverage,
            "normalized_exact_duplicate_sha256": normalized_exact_hash,
            "registered_non_report_artifact_count": sum(
                relative not in {final_report_relative, final_manifest_relative}
                for relative in manifest.artifacts
            ),
        },
        "timing_seconds_accumulated_across_resumes": {
            stage: manifest.stage_seconds(stage)
            for stage in (
                "index",
                "phash_generation",
                "descriptor_generation",
                "robust_retrieval",
                "phash_retrieval",
                "finalization",
            )
        },
        "environment": core["environment"],
        "limitations": [
            "Candidate inclusion is not a duplicate decision.",
            "No duplicate-rule scoring, family construction, split, or training ran.",
            "Patient-level leakage cannot be guaranteed because patient identifiers are absent.",
        ],
        "next_step": (
            "Score candidate pairs with frozen duplicate rule v1 in separately timed CPU "
            "shards; do not construct families before that verification."
        ),
    }
    if manifest.is_complete(final_report_relative):
        saved_report = json.loads(
            (output_dir / final_report_relative).read_text(encoding="utf-8")
        )
        if saved_report != report:
            raise ValueError("Registered final report differs from reconstructed result.")
    else:
        atomic_json(output_dir / final_report_relative, report)
        manifest.register(final_report_relative, "finalization", {"pairs": len(keys)})

    final_hash_payload = {
        "schema_version": 1,
        "run_contract.json": file_sha256(contract_path),
        "registered_artifacts": {
            relative: record["sha256"]
            for relative, record in sorted(manifest.artifacts.items())
        },
    }
    if not manifest.is_complete(final_manifest_relative):
        atomic_json(output_dir / final_manifest_relative, final_hash_payload)
        manifest.register(final_manifest_relative, "finalization")

    print(f"Final candidate pairs: {len(keys)}")
    print(f"Pairs per image: {len(keys) / len(items):.3f}")
    print(f"Report: {output_dir / final_report_relative}")
    print("NOTE: Duplicate scoring, family, split, and model training were not run.")
    return output_dir / final_report_relative


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
