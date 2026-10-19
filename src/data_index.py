from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_MAIN_CLASSES = {
    "ALL",
    "Brain Cancer",
    "Breast Cancer",
    "Cervical Cancer",
    "Kidney Cancer",
    "Lung and Colon Cancer",
    "Lymphoma",
    "Oral Cancer",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    main_class: str
    subclass: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root(requested: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.extend(
        [
            project_root() / "archive" / "Multi Cancer" / "Multi Cancer",
            project_root() / "archive" / "Multi Cancer",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_dir():
            continue
        child_names = {child.name for child in candidate.iterdir() if child.is_dir()}
        if EXPECTED_MAIN_CLASSES.issubset(child_names):
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Multi Cancer veri kökü bulunamadı. Kontrol edilen yollar:\n" + checked
    )


def is_supported_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.lower().endswith(".bak.jpg")
    )


def discover_records(dataset_root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    main_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    for main_dir in main_dirs:
        for path in sorted(main_dir.rglob("*")):
            if not is_supported_image(path):
                continue
            relative_parts = path.relative_to(main_dir).parts
            subclass = relative_parts[0] if len(relative_parts) > 1 else "(direct)"
            records.append(
                ImageRecord(path=path, main_class=main_dir.name, subclass=subclass)
            )

    found_classes = {record.main_class for record in records}
    if found_classes != EXPECTED_MAIN_CLASSES:
        missing = sorted(EXPECTED_MAIN_CLASSES - found_classes)
        extra = sorted(found_classes - EXPECTED_MAIN_CLASSES)
        raise ValueError(
            f"Beklenen 8 ana sınıf bulunamadı. Eksik={missing}, fazladan={extra}"
        )
    return records


def counts_by(records: Sequence[ImageRecord], attribute: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = getattr(record, attribute)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def target_label(record: ImageRecord, target_level: str) -> str:
    if target_level == "main":
        return record.main_class
    if target_level == "subclass":
        return record.subclass
    raise ValueError(f"Bilinmeyen hedef seviyesi: {target_level}")


def select_max_per_label(
    records: Sequence[ImageRecord],
    max_per_label: Optional[int],
    seed: int,
    target_level: str,
) -> List[ImageRecord]:
    if max_per_label is None:
        return list(records)

    import random

    rng = random.Random(seed)
    grouped: Dict[str, List[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(target_label(record, target_level), []).append(record)

    selected: List[ImageRecord] = []
    for label in sorted(grouped):
        label_records = list(grouped[label])
        rng.shuffle(label_records)
        selected.extend(label_records[:max_per_label])
    return selected


def select_max_per_class(
    records: Sequence[ImageRecord], max_per_class: Optional[int], seed: int
) -> List[ImageRecord]:
    return select_max_per_label(records, max_per_class, seed, "main")

