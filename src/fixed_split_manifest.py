from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

from data_index import ImageRecord


def load_fixed_split_manifest(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    manifest_path: Path,
) -> Dict[str, list[ImageRecord]]:
    """Load and strictly validate a fixed train/val/test manifest."""

    record_by_relative = {
        record.path.relative_to(dataset_root).as_posix(): record for record in records
    }
    if len(record_by_relative) != len(records):
        raise ValueError("Dataset kayitlarinda tekrarlanan relative path var")

    split_records: Dict[str, list[ImageRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "main_class", "subclass", "split", "family_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Split manifest alanlari eksik: {manifest_path}")
        for row in reader:
            relative = row["relative_path"].replace("\\", "/")
            if relative in seen:
                raise ValueError(f"Split manifestte tekrarlanan path: {relative}")
            seen.add(relative)
            if relative not in record_by_relative:
                raise ValueError(f"Split manifestte dataset disi path: {relative}")
            record = record_by_relative[relative]
            if row["main_class"] != record.main_class or row["subclass"] != record.subclass:
                raise ValueError(f"Split manifest metadata uyusmazligi: {relative}")
            split = row["split"]
            if split not in split_records:
                raise ValueError(f"Gecersiz split {split!r}: {relative}")
            if not row["family_id"].strip():
                raise ValueError(f"Bos family_id: {relative}")
            split_records[split].append(record)

    missing = set(record_by_relative) - seen
    extra = seen - set(record_by_relative)
    if missing or extra:
        raise ValueError(
            f"Split manifest dataset kapsamiyla eslesmiyor; eksik={len(missing)}, fazla={len(extra)}"
        )
    leaves = {record.subclass for record in records}
    for split, selected in split_records.items():
        split_leaves = {record.subclass for record in selected}
        if split_leaves != leaves:
            raise ValueError(
                f"{split} splitinde leaf kapsami eksik; eksik={sorted(leaves - split_leaves)}"
            )
    return split_records
