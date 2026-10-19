from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from data_index import discover_records, project_root, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi Cancer veri setinde byte-byte aynı görüntüleri SHA-256 ile arar."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output",
        default=str(project_root() / "artifacts" / "audit" / "exact_duplicates.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> Tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(path), digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.data_dir)
    records = discover_records(root)

    by_size: Dict[int, List[Path]] = defaultdict(list)
    for record in records:
        by_size[record.path.stat().st_size].append(record.path)
    candidates = [path for paths in by_size.values() if len(paths) > 1 for path in paths]

    print(f"Kabul edilen görüntü: {len(records)}")
    print(f"Aynı dosya boyutunu paylaşan hash adayı: {len(candidates)}")

    by_hash: Dict[str, List[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (path, digest) in enumerate(
            executor.map(sha256, candidates, chunksize=32), start=1
        ):
            by_hash[digest].append(path)
            if index % 10000 == 0 or index == len(candidates):
                print(f"Hash: {index}/{len(candidates)}")

    duplicate_groups = []
    cross_subclass_groups = 0
    duplicate_file_count = 0
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        relative_paths = sorted(str(Path(path).relative_to(root)) for path in paths)
        subclasses = sorted({Path(path).parts[1] for path in relative_paths})
        if len(subclasses) > 1:
            cross_subclass_groups += 1
        duplicate_file_count += len(paths)
        duplicate_groups.append(
            {
                "sha256": digest,
                "file_count": len(paths),
                "subclasses": subclasses,
                "paths": relative_paths,
            }
        )

    duplicate_groups.sort(key=lambda item: (-item["file_count"], item["sha256"]))
    report = {
        "dataset_root": str(root),
        "accepted_images": len(records),
        "same_size_hash_candidates": len(candidates),
        "exact_duplicate_group_count": len(duplicate_groups),
        "files_in_exact_duplicate_groups": duplicate_file_count,
        "redundant_exact_copies": sum(item["file_count"] - 1 for item in duplicate_groups),
        "cross_subclass_duplicate_group_count": cross_subclass_groups,
        "method": "SHA-256 for files sharing an identical byte size",
        "limitation": (
            "This detects byte-identical files only. Rotated, shifted, brightness-adjusted or "
            "re-encoded relatives require a separate perceptual-similarity audit."
        ),
        "duplicate_groups": duplicate_groups,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: value for key, value in report.items() if key != "duplicate_groups"}, indent=2))
    print(f"Çıktı: {output}")


if __name__ == "__main__":
    main()
