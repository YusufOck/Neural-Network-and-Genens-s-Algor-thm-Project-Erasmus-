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


def load_search_proxy_manifest(
    fixed_splits: Dict[str, list[ImageRecord]],
    dataset_root: Path,
    manifest_path: Path,
) -> Dict[str, list[ImageRecord]]:
    """Load a train/validation-only subset of an already validated fixed split.

    The proxy is intentionally unable to introduce test images or move a path between
    partitions. It is used only for bounded hyperparameter-search trials.
    """

    source_by_split: Dict[str, Dict[str, ImageRecord]] = {}
    for split in ("train", "val", "test"):
        source_by_split[split] = {
            record.path.relative_to(dataset_root).as_posix(): record
            for record in fixed_splits[split]
        }

    selected: Dict[str, list[ImageRecord]] = {"train": [], "val": []}
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "relative_path",
            "main_class",
            "subclass",
            "split",
            "family_id",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Search proxy manifest alanlari eksik: {manifest_path}")
        for row in reader:
            relative = row["relative_path"].replace("\\", "/")
            if relative in seen:
                raise ValueError(f"Search proxy manifestte tekrarlanan path: {relative}")
            seen.add(relative)
            split = row["split"]
            if split not in selected:
                raise ValueError(
                    "Search proxy yalniz train/val satirlari icerebilir; "
                    f"split={split!r}, path={relative}"
                )
            if relative not in source_by_split[split]:
                source_split = next(
                    (
                        name
                        for name, paths in source_by_split.items()
                        if relative in paths
                    ),
                    None,
                )
                raise ValueError(
                    "Search proxy path'i dondurulmus split bolumunu degistiriyor veya dataset "
                    f"disinda: path={relative}, proxy={split}, source={source_split}"
                )
            record = source_by_split[split][relative]
            if row["main_class"] != record.main_class or row["subclass"] != record.subclass:
                raise ValueError(f"Search proxy metadata uyusmazligi: {relative}")
            if not row["family_id"].strip():
                raise ValueError(f"Search proxy bos family_id: {relative}")
            selected[split].append(record)

    expected_leaves = {
        record.subclass
        for split in ("train", "val", "test")
        for record in fixed_splits[split]
    }
    for split, records in selected.items():
        leaves = {record.subclass for record in records}
        if leaves != expected_leaves:
            raise ValueError(
                f"Search proxy {split} leaf kapsami eksik; "
                f"eksik={sorted(expected_leaves - leaves)}"
            )
    return selected
