"""Build the full train/validation-only manifest for GA winner confirmation.

The source is the accepted practical split. Every train and validation row is copied,
while every test row is deliberately omitted. Images are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


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


def read_source_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(OUTPUT_FIELDS) - {"selection_rank"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Kaynak split manifest alanlari eksik: {path}")
        rows = list(reader)
    paths = [row["relative_path"].replace("\\", "/") for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Kaynak split manifestte tekrarlanan relative path var")
    return rows


def select_full_train_val_rows(rows: Iterable[dict]) -> List[dict]:
    selected = []
    for source_row in rows:
        split = source_row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Gecersiz kaynak split: {split!r}")
        if split == "test":
            continue
        row = {
            field: source_row[field]
            for field in OUTPUT_FIELDS
            if field != "selection_rank"
        }
        row["relative_path"] = row["relative_path"].replace("\\", "/")
        selected.append(row)

    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            row["subclass"],
            row["relative_path"],
        )
    )
    ranks: Counter[tuple[str, str]] = Counter()
    output = []
    for row in selected:
        key = (row["split"], row["subclass"])
        ranks[key] += 1
        output.append({**row, "selection_rank": ranks[key]})

    leaves = sorted({row["subclass"] for row in output})
    if len(leaves) != 26:
        raise ValueError(f"Beklenen 26 leaf yerine {len(leaves)} bulundu")
    counts = Counter((row["split"], row["subclass"]) for row in output)
    for leaf in leaves:
        if counts[("train", leaf)] != 3500 or counts[("val", leaf)] != 750:
            raise ValueError(
                f"Tam confirmation sayimi uyusmuyor: {leaf}, "
                f"train={counts[('train', leaf)]}, val={counts[('val', leaf)]}"
            )
    return output


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def build_report(source_path: Path, output_path: Path, rows: List[dict]) -> dict:
    counts = Counter((row["split"], row["subclass"]) for row in rows)
    leaves = sorted({row["subclass"] for row in rows})
    return {
        "schema_version": 1,
        "status": "ga_full_confirmation_manifest_v1_complete",
        "experiment_type": "full_train_validation_only_confirmation_manifest",
        "opens_images": False,
        "contains_test_rows": False,
        "source_split_manifest": str(source_path),
        "source_split_manifest_sha256": sha256_file(source_path),
        "confirmation_manifest": str(output_path),
        "confirmation_manifest_sha256": sha256_file(output_path),
        "leaf_count": len(leaves),
        "leaves": leaves,
        "counts": {"train": 91_000, "val": 19_500, "test": 0},
        "per_leaf": {
            leaf: {
                "train": counts[("train", leaf)],
                "val": counts[("val", leaf)],
            }
            for leaf in leaves
        },
        "selection_rule": (
            "All frozen train and validation rows, sorted by split/subclass/relative_path; "
            "all frozen test rows omitted."
        ),
        "claim": "Full-data validation-only confirmation input; not test performance.",
        "forbidden_claim": "Final test result, external validation or clinical evidence.",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build full train/validation-only GA confirmation manifest"
    )
    parser.add_argument(
        "--source-manifest",
        default=str(root / "research/splits/practical_leakage_reduced_split_v1.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        default=str(root / "research/splits/ga_full_confirmation_v1.csv"),
    )
    parser.add_argument(
        "--report",
        default=str(root / "research/splits/ga_full_confirmation_v1_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest).resolve()
    output_path = Path(args.output_manifest).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Confirmation output zaten var; dondurulmus dosyayi ezme")
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != FROZEN_SPLIT_SHA256:
        raise ValueError(
            "Kaynak split SHA-256 dondurulmus manifestle eslesmiyor: "
            f"{actual_source_hash}"
        )
    rows = select_full_train_val_rows(read_source_rows(source_path))
    write_csv(rows, output_path)
    report = build_report(source_path, output_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
