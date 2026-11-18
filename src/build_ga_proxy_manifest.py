"""Build the frozen train/validation proxy used by the bounded GA search.

This builder reads only the accepted split manifest. It never opens an image and it never
selects a test row. Selection is deterministic by SHA-256 rank within each split/leaf.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


FROZEN_SPLIT_SHA256 = "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171"
OUTPUT_FIELDS = [
    "relative_path",
    "main_class",
    "subclass",
    "split",
    "family_id",
    "family_size",
    "family_sources",
    "selection_rank",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(row: dict, seed: int) -> str:
    payload = (
        f"ga-proxy-v1|{seed}|{row['split']}|{row['subclass']}|"
        f"{row['relative_path'].replace(chr(92), '/')}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_source_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(OUTPUT_FIELDS) - {"selection_rank"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Kaynak split manifest alanlari eksik: {path}")
        rows = list(reader)
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Kaynak split manifestte tekrarlanan relative path var")
    return rows


def select_proxy_rows(
    rows: Iterable[dict],
    seed: int,
    train_per_leaf: int,
    val_per_leaf: int,
) -> List[dict]:
    if train_per_leaf <= 0 or val_per_leaf <= 0:
        raise ValueError("Proxy leaf sayilari pozitif olmalidir")
    grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for source_row in rows:
        split = source_row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Gecersiz kaynak split: {split!r}")
        if split == "test":
            continue
        row = {field: source_row[field] for field in OUTPUT_FIELDS if field != "selection_rank"}
        row["relative_path"] = row["relative_path"].replace("\\", "/")
        grouped[(split, row["subclass"])].append(row)

    leaves = sorted({leaf for _, leaf in grouped})
    if not leaves:
        raise ValueError("Kaynak manifestte train/val leaf bulunamadi")
    selected: List[dict] = []
    expected = {"train": train_per_leaf, "val": val_per_leaf}
    for split in ("train", "val"):
        for leaf in leaves:
            candidates = grouped.get((split, leaf), [])
            count = expected[split]
            if len(candidates) < count:
                raise ValueError(
                    f"{split}/{leaf} icin yetersiz satir: {len(candidates)} < {count}"
                )
            ranked = sorted(
                candidates,
                key=lambda row: (selection_key(row, seed), row["relative_path"]),
            )
            for rank, row in enumerate(ranked[:count], start=1):
                selected.append({**row, "selection_rank": rank})
    return selected


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def build_report(
    source_path: Path,
    output_path: Path,
    rows: List[dict],
    seed: int,
    train_per_leaf: int,
    val_per_leaf: int,
) -> dict:
    counts = Counter((row["split"], row["subclass"]) for row in rows)
    leaves = sorted({row["subclass"] for row in rows})
    return {
        "schema_version": 1,
        "status": "ga_search_proxy_manifest_v1_complete",
        "experiment_type": "deterministic_train_validation_search_proxy",
        "opens_images": False,
        "contains_test_rows": False,
        "seed": seed,
        "source_split_manifest": str(source_path),
        "source_split_manifest_sha256": sha256_file(source_path),
        "proxy_manifest": str(output_path),
        "proxy_manifest_sha256": sha256_file(output_path),
        "leaf_count": len(leaves),
        "leaves": leaves,
        "counts": {
            "train": sum(1 for row in rows if row["split"] == "train"),
            "val": sum(1 for row in rows if row["split"] == "val"),
            "test": 0,
        },
        "per_leaf": {
            leaf: {
                "train": counts[("train", leaf)],
                "val": counts[("val", leaf)],
            }
            for leaf in leaves
        },
        "expected_per_leaf": {
            "train": train_per_leaf,
            "val": val_per_leaf,
        },
        "selection_rule": "SHA-256 rank of seed/split/subclass/relative_path",
        "claim": "Balanced deterministic proxy for validation-only hyperparameter search.",
        "forbidden_claim": "Final model performance or an independent test result.",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build frozen GA train/val proxy manifest")
    parser.add_argument(
        "--source-manifest",
        default=str(root / "research/splits/practical_leakage_reduced_split_v1.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        default=str(root / "research/splits/ga_search_proxy_v1.csv"),
    )
    parser.add_argument(
        "--report",
        default=str(root / "research/splits/ga_search_proxy_v1_report.json"),
    )
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--train-per-leaf", type=int, default=350)
    parser.add_argument("--val-per-leaf", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest).resolve()
    output_path = Path(args.output_manifest).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Proxy output zaten var; mevcut dondurulmus dosyayi ezme")
    source_hash = sha256_file(source_path)
    if source_hash != FROZEN_SPLIT_SHA256:
        raise ValueError(
            "Kaynak split SHA-256 dondurulmus manifestle eslesmiyor: "
            f"{source_hash}"
        )
    rows = select_proxy_rows(
        read_source_rows(source_path),
        args.seed,
        args.train_per_leaf,
        args.val_per_leaf,
    )
    write_csv(rows, output_path)
    report = build_report(
        source_path,
        output_path,
        rows,
        args.seed,
        args.train_per_leaf,
        args.val_per_leaf,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
