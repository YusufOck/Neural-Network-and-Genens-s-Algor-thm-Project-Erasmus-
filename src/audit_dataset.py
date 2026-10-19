from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image

from data_index import counts_by, discover_records, project_root, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi Cancer veri setini denetler.")
    parser.add_argument("--data-dir", default=None, help="Sekiz ana sınıfı içeren kök klasör")
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "artifacts" / "audit"),
        help="Denetim çıktılarının kaydedileceği klasör",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="130.000 görüntünün tamamını PIL ile açıp doğrular",
    )
    return parser.parse_args()


def verify_one_image(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.verify()
        return ""
    except Exception as exc:  # diagnostic tool: preserve filename and exception
        return f"{path}: {type(exc).__name__}: {exc}"


def verify_images(paths: List[Path]) -> List[str]:
    bad_files: List[str] = []
    total = len(paths)
    worker_count = min(16, max(4, (os.cpu_count() or 2) * 2))
    print(f"Görüntü doğrulama iş parçacığı: {worker_count}")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(verify_one_image, paths, chunksize=64)
        for index, error in enumerate(results, start=1):
            if error:
                bad_files.append(error)
            if index % 10000 == 0 or index == total:
                print(f"Doğrulandı: {index}/{total}")
    return bad_files


def save_distribution(main_counts: Dict[str, int], output_path: Path) -> None:
    labels = list(main_counts)
    values = [main_counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color="#24557a")
    ax.set_title("Multi Cancer Dataset - 8 Main Classes")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=28)
    ax.bar_label(bars, padding=3, fmt="%d")
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_sample_grid(records, output_path: Path) -> None:
    first_by_class = {}
    for record in records:
        first_by_class.setdefault(record.main_class, record)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, class_name in zip(axes.flat, sorted(first_by_class)):
        record = first_by_class[class_name]
        with Image.open(record.path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(class_name, fontsize=10)
        ax.axis("off")
    fig.suptitle("One example from each main class", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.90, wspace=0.05, hspace=0.18)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_subclass_grid(records, output_path: Path) -> None:
    first_by_subclass = {}
    for record in records:
        first_by_subclass.setdefault((record.main_class, record.subclass), record)

    entries = sorted(first_by_subclass.items())
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    for ax in axes.flat:
        ax.axis("off")
    for ax, ((main_class, subclass), record) in zip(axes.flat, entries):
        with Image.open(record.path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{subclass}\n({main_class})", fontsize=9)
        ax.axis("off")
    fig.suptitle("One local example from each of the 26 leaf classes", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = resolve_dataset_root(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Veri kökü: {data_root}")
    records = discover_records(data_root)
    main_counts = counts_by(records, "main_class")

    nested: Dict[str, Dict[str, int]] = defaultdict(dict)
    for record in records:
        nested[record.main_class][record.subclass] = (
            nested[record.main_class].get(record.subclass, 0) + 1
        )

    unexpected_backups = sorted(
        str(path)
        for path in data_root.rglob("*.bak.jpg")
        if path.is_file()
    )

    bad_files: List[str] = []
    if args.verify_all:
        bad_files = verify_images([record.path for record in records])

    report = {
        "dataset_root": str(data_root),
        "total_images": len(records),
        "main_class_count": len(main_counts),
        "main_class_counts": main_counts,
        "subclass_counts": {key: dict(sorted(value.items())) for key, value in nested.items()},
        "excluded_backup_files": unexpected_backups,
        "verified_all_images": bool(args.verify_all),
        "corrupt_files": bad_files,
        "notes": [
            "The dataset exposes 8 parent groups and 26 leaf labels.",
            "Current planned target is 26 leaf labels with derived 8-parent metrics.",
            "The dataset mixes microscopy, histopathology, MRI and CT modalities.",
            "The README states that the dataset already contains augmented images.",
            "Run audit_exact_duplicates.py and inspect dataset_metadata.json before splitting.",
        ],
    }

    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    save_distribution(main_counts, output_dir / "class_distribution.png")
    save_sample_grid(records, output_dir / "sample_grid.png")
    save_subclass_grid(records, output_dir / "subclass_sample_grid.png")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Denetim çıktıları: {output_dir}")


if __name__ == "__main__":
    main()

