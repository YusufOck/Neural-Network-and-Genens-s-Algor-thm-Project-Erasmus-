from __future__ import annotations

"""Create a small, review-first pHash pilot for the Multi Cancer dataset.

This script does not create train/validation/test splits and does not declare
that visually similar pairs are duplicates. It produces candidate statistics
and contact sheets so a human can inspect different Hamming-distance bands
before any perceptual-family threshold is selected.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data_index import ImageRecord, discover_records, project_root, resolve_dataset_root


HASH_SIZE = 8
HIGH_FREQUENCY_FACTOR = 4
HASH_BITS = HASH_SIZE * HASH_SIZE
DEFAULT_SUBCLASSES = ("lung_scc", "brain_glioma")
DEFAULT_DISTANCE_BINS = ((0, 2), (3, 4), (5, 6), (7, 8), (9, 12))
BYTE_POPCOUNT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class HashedImage:
    record: ImageRecord
    relative_path: str
    phash: int


@dataclass(frozen=True)
class PairCandidate:
    subclass: str
    distance_band: str
    distance: int
    left: HashedImage
    right: HashedImage
    known_exact_duplicate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seçilen alt sınıflarda 64-bit pHash uzaklıklarını tarar ve manuel inceleme "
            "için aday çiftlerden temas sayfaları üretir. Split oluşturmaz."
        )
    )
    parser.add_argument("--data-dir", default=None, help="Multi Cancer veri kökü.")
    parser.add_argument(
        "--subclasses",
        nargs="+",
        default=list(DEFAULT_SUBCLASSES),
        help="İncelenecek alt sınıflar (varsayılan: lung_scc brain_glioma).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "artifacts" / "audit" / "phash_pilot"),
        help="JSON, CSV ve temas sayfalarının yazılacağı klasör.",
    )
    parser.add_argument(
        "--exact-duplicates",
        default=str(project_root() / "artifacts" / "audit" / "exact_duplicates.json"),
        help="Bilinen SHA-256 gruplarını pozitif kontrol olarak işaretleyen audit dosyası.",
    )
    parser.add_argument(
        "--samples-per-band",
        type=int,
        default=12,
        help="Her alt sınıf ve uzaklık aralığı için saklanacak aday çift sayısı.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Örnek seçimi için sabit tohum.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Görüntü hash'lerini hesaplarken kullanılacak iş parçacığı sayısı.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.subclasses:
        raise ValueError("En az bir alt sınıf verilmelidir.")
    if len(set(args.subclasses)) != len(args.subclasses):
        raise ValueError("--subclasses içinde aynı ad birden fazla kez kullanılamaz.")
    if args.samples_per_band < 1:
        raise ValueError("--samples-per-band en az 1 olmalıdır.")
    if args.workers < 1:
        raise ValueError("--workers en az 1 olmalıdır.")


def dct_matrix(size: int) -> np.ndarray:
    """Return an orthonormal DCT-II transform matrix."""

    positions = np.arange(size, dtype=np.float32)
    frequencies = positions.reshape(-1, 1)
    matrix = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size))
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix.astype(np.float32)


def perceptual_hash(path: Path, transform: np.ndarray) -> int:
    """Compute a deterministic 64-bit pHash from decoded image pixels."""

    side = HASH_SIZE * HIGH_FREQUENCY_FACTOR
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((side, side), resample=resampling)
        pixels = np.asarray(grayscale, dtype=np.float32)

    coefficients = transform @ pixels @ transform.T
    low_frequency = coefficients[:HASH_SIZE, :HASH_SIZE].reshape(-1)

    # The DC coefficient measures overall brightness. Excluding it from the median
    # prevents global brightness from dominating the other low-frequency features.
    threshold = float(np.median(low_frequency[1:]))
    bits = low_frequency > threshold

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def compute_hashes(
    records: Sequence[ImageRecord], root: Path, workers: int
) -> List[HashedImage]:
    side = HASH_SIZE * HIGH_FREQUENCY_FACTOR
    transform = dct_matrix(side)

    def hash_record(record: ImageRecord) -> HashedImage:
        return HashedImage(
            record=record,
            relative_path=relative_posix(record.path, root),
            phash=perceptual_hash(record.path, transform),
        )

    hashed: List[HashedImage] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(executor.map(hash_record, records), start=1):
            hashed.append(item)
            if index % 500 == 0 or index == len(records):
                print(f"pHash: {index}/{len(records)}")
    return hashed


def load_exact_group_map(path: Path) -> Tuple[Dict[str, str], Optional[str]]:
    """Map normalized relative paths to their known SHA-256 duplicate group."""

    if not path.is_file():
        return {}, f"SHA-256 audit dosyası bulunamadı: {path}"

    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    group_by_path: Dict[str, str] = {}
    for group in report.get("duplicate_groups", []):
        group_id = str(group["sha256"])
        for relative_path in group.get("paths", []):
            normalized = str(relative_path).replace("\\", "/")
            group_by_path[normalized] = group_id
    return group_by_path, None


def distance_band_name(low: int, high: int) -> str:
    return f"{low:02d}-{high:02d}"


def hamming_distances(value: int, others: np.ndarray) -> np.ndarray:
    """Vectorized Hamming distance from one uint64 hash to many hashes."""

    xor_values = np.bitwise_xor(np.uint64(value), others)
    bytes_view = xor_values.view(np.uint8).reshape(-1, 8)
    return BYTE_POPCOUNT[bytes_view].sum(axis=1, dtype=np.uint16)


def reservoir_add(
    reservoir: List[Tuple[int, int, int]],
    item: Tuple[int, int, int],
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    """Keep an unbiased fixed-size sample from a stream of matching pairs."""

    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < limit:
        reservoir[replacement_index] = item


def scan_subclass_pairs(
    items: Sequence[HashedImage],
    samples_per_band: int,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, List[Tuple[int, int, int]]]]:
    """Count all within-subclass distances and sample pairs from review bands."""

    values = np.asarray([item.phash for item in items], dtype=np.uint64)
    histogram = np.zeros(HASH_BITS + 1, dtype=np.int64)
    seen_by_band = {
        distance_band_name(low, high): 0 for low, high in DEFAULT_DISTANCE_BINS
    }
    samples_by_band: Dict[str, List[Tuple[int, int, int]]] = {
        distance_band_name(low, high): [] for low, high in DEFAULT_DISTANCE_BINS
    }

    for left_index in range(len(items) - 1):
        distances = hamming_distances(values[left_index], values[left_index + 1 :])
        histogram += np.bincount(distances, minlength=HASH_BITS + 1)

        for low, high in DEFAULT_DISTANCE_BINS:
            band = distance_band_name(low, high)
            matching_offsets = np.flatnonzero((distances >= low) & (distances <= high))
            for offset in matching_offsets.tolist():
                right_index = left_index + 1 + int(offset)
                distance = int(distances[offset])
                seen_by_band[band] += 1
                reservoir_add(
                    reservoir=samples_by_band[band],
                    item=(left_index, right_index, distance),
                    seen=seen_by_band[band],
                    limit=samples_per_band,
                    rng=rng,
                )

        if (left_index + 1) % 500 == 0 or left_index + 2 == len(items):
            print(f"Çift tarama ({items[0].record.subclass}): {left_index + 1}/{len(items) - 1}")

    return histogram, seen_by_band, samples_by_band


def is_known_exact_pair(
    left_path: str, right_path: str, exact_group_by_path: Mapping[str, str]
) -> bool:
    left_group = exact_group_by_path.get(left_path)
    return left_group is not None and left_group == exact_group_by_path.get(right_path)


def build_pair_candidates(
    subclass: str,
    items: Sequence[HashedImage],
    samples_by_band: Mapping[str, Sequence[Tuple[int, int, int]]],
    exact_group_by_path: Mapping[str, str],
) -> List[PairCandidate]:
    candidates: List[PairCandidate] = []
    for band, sampled_pairs in samples_by_band.items():
        for left_index, right_index, distance in sampled_pairs:
            left = items[left_index]
            right = items[right_index]
            candidates.append(
                PairCandidate(
                    subclass=subclass,
                    distance_band=band,
                    distance=distance,
                    left=left,
                    right=right,
                    known_exact_duplicate=is_known_exact_pair(
                        left.relative_path, right.relative_path, exact_group_by_path
                    ),
                )
            )
    return candidates


def known_exact_control(
    items: Sequence[HashedImage], exact_group_by_path: Mapping[str, str]
) -> Dict[str, object]:
    """Measure pHash distances for known byte-identical pairs in the pilot subset."""

    by_group: Dict[str, List[HashedImage]] = defaultdict(list)
    for item in items:
        group_id = exact_group_by_path.get(item.relative_path)
        if group_id is not None:
            by_group[group_id].append(item)

    histogram: Dict[int, int] = defaultdict(int)
    pair_count = 0
    for group_items in by_group.values():
        for left_index in range(len(group_items) - 1):
            for right_index in range(left_index + 1, len(group_items)):
                distance = (group_items[left_index].phash ^ group_items[right_index].phash)
                # Python 3.8 does not guarantee int.bit_count(), so use a portable count.
                distance = bin(distance).count("1")
                histogram[distance] += 1
                pair_count += 1

    return {
        "known_exact_groups_present": sum(len(group) >= 2 for group in by_group.values()),
        "known_exact_pair_count": pair_count,
        "distance_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "nonzero_distance_pair_count": sum(
            count for distance, count in histogram.items() if distance != 0
        ),
    }


def write_hash_csv(path: Path, items: Sequence[HashedImage]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["main_class", "subclass", "relative_path", "phash_hex"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "main_class": item.record.main_class,
                    "subclass": item.record.subclass,
                    "relative_path": item.relative_path,
                    "phash_hex": f"{item.phash:016x}",
                }
            )


def write_candidate_csv(path: Path, candidates: Sequence[PairCandidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subclass",
                "distance_band",
                "distance",
                "known_exact_duplicate",
                "left_path",
                "right_path",
                "left_phash",
                "right_phash",
            ],
        )
        writer.writeheader()
        for pair in candidates:
            writer.writerow(
                {
                    "subclass": pair.subclass,
                    "distance_band": pair.distance_band,
                    "distance": pair.distance,
                    "known_exact_duplicate": pair.known_exact_duplicate,
                    "left_path": pair.left.relative_path,
                    "right_path": pair.right.relative_path,
                    "left_phash": f"{pair.left.phash:016x}",
                    "right_phash": f"{pair.right.phash:016x}",
                }
            )


def thumbnail(path: Path, size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), resample=resampling)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def write_contact_sheets(
    output_dir: Path,
    candidates: Sequence[PairCandidate],
    pairs_per_page: int = 6,
    thumb_size: int = 224,
) -> List[str]:
    """Write vertically stacked, lossless PNG pages for manual pair review."""

    grouped: Dict[Tuple[str, str], List[PairCandidate]] = defaultdict(list)
    for pair in candidates:
        grouped[(pair.subclass, pair.distance_band)].append(pair)

    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    margin = 16
    gap = 12
    header_height = 54
    row_height = header_height + thumb_size + margin
    page_width = margin * 2 + thumb_size * 2 + gap

    for (subclass, band), pairs in sorted(grouped.items()):
        for page_number, start in enumerate(range(0, len(pairs), pairs_per_page), start=1):
            page_pairs = pairs[start : start + pairs_per_page]
            canvas = Image.new(
                "RGB", (page_width, margin + row_height * len(page_pairs)), "white"
            )
            draw = ImageDraw.Draw(canvas)

            for row, pair in enumerate(page_pairs):
                top = margin + row * row_height
                exact_text = "yes" if pair.known_exact_duplicate else "no"
                draw.text(
                    (margin, top),
                    f"{subclass} | distance={pair.distance} | SHA-256 exact={exact_text}",
                    fill="black",
                )
                draw.text(
                    (margin, top + 18),
                    f"L: {Path(pair.left.relative_path).name}",
                    fill="black",
                )
                draw.text(
                    (margin + thumb_size + gap, top + 18),
                    f"R: {Path(pair.right.relative_path).name}",
                    fill="black",
                )
                canvas.paste(
                    thumbnail(pair.left.record.path, thumb_size),
                    (margin, top + header_height),
                )
                canvas.paste(
                    thumbnail(pair.right.record.path, thumb_size),
                    (margin + thumb_size + gap, top + header_height),
                )

            filename = f"{safe_name(subclass)}__distance_{band}__page_{page_number:02d}.png"
            destination = sheets_dir / filename
            canvas.save(destination, format="PNG")
            written.append(destination.relative_to(output_dir).as_posix())
    return written


def nonzero_histogram(histogram: np.ndarray) -> Dict[str, int]:
    return {
        str(distance): int(count)
        for distance, count in enumerate(histogram.tolist())
        if count
    }


def candidates_for_json(candidates: Iterable[PairCandidate]) -> List[Dict[str, object]]:
    return [
        {
            "subclass": pair.subclass,
            "distance_band": pair.distance_band,
            "distance": pair.distance,
            "known_exact_duplicate": pair.known_exact_duplicate,
            "left_path": pair.left.relative_path,
            "right_path": pair.right.relative_path,
        }
        for pair in candidates
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = resolve_dataset_root(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = discover_records(root)
    requested = set(args.subclasses)
    records = sorted(
        (record for record in all_records if record.subclass in requested),
        key=lambda record: (record.subclass, str(record.path)),
    )
    found = {record.subclass for record in records}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Bulunamayan alt sınıflar: {missing}")

    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.subclass] += 1
    print(f"Pilot türü: pHash benzerlik audit'i (split/model eğitimi yok)")
    print(f"Veri kökü: {root}")
    print(f"Alt sınıf sayıları: {dict(sorted(counts.items()))}")

    exact_group_by_path, exact_warning = load_exact_group_map(
        Path(args.exact_duplicates).resolve()
    )
    if exact_warning:
        print(f"UYARI: {exact_warning}")

    hashed = compute_hashes(records, root=root, workers=args.workers)
    write_hash_csv(output_dir / "phashes.csv", hashed)

    hashed_by_subclass: Dict[str, List[HashedImage]] = defaultdict(list)
    for item in hashed:
        hashed_by_subclass[item.record.subclass].append(item)

    rng = random.Random(args.seed)
    all_candidates: List[PairCandidate] = []
    subclass_reports: Dict[str, Dict[str, object]] = {}
    for subclass in sorted(hashed_by_subclass):
        items = hashed_by_subclass[subclass]
        histogram, band_counts, samples_by_band = scan_subclass_pairs(
            items=items,
            samples_per_band=args.samples_per_band,
            rng=rng,
        )
        candidates = build_pair_candidates(
            subclass=subclass,
            items=items,
            samples_by_band=samples_by_band,
            exact_group_by_path=exact_group_by_path,
        )
        all_candidates.extend(candidates)
        subclass_reports[subclass] = {
            "image_count": len(items),
            "all_pair_count": len(items) * (len(items) - 1) // 2,
            "full_distance_histogram": nonzero_histogram(histogram),
            "candidate_pair_counts_by_band": band_counts,
            "known_exact_positive_control": known_exact_control(
                items, exact_group_by_path
            ),
        }

    write_candidate_csv(output_dir / "candidate_samples.csv", all_candidates)
    contact_sheets = write_contact_sheets(output_dir, all_candidates)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "pilot audit",
        "dataset_root": str(root),
        "selected_subclasses": sorted(requested),
        "selected_image_count": len(hashed),
        "seed": args.seed,
        "samples_per_subclass_per_band": args.samples_per_band,
        "method": {
            "name": "custom 64-bit perceptual hash (pHash)",
            "decoded_input": "Pillow grayscale pixels; file extension is not trusted",
            "resize": "32x32 Lanczos",
            "transform": "orthonormal 2D DCT-II",
            "signature": "8x8 low-frequency block; median excludes the DC coefficient",
            "distance": "Hamming distance between two 64-bit hashes",
            "distance_bands_for_manual_review": [
                {"low": low, "high": high, "name": distance_band_name(low, high)}
                for low, high in DEFAULT_DISTANCE_BINS
            ],
            "pair_scope": "pairs are compared only within the same selected subclass",
            "sampling": "seeded reservoir sample from every distance band",
        },
        "subclasses": subclass_reports,
        "sampled_candidate_pairs": candidates_for_json(all_candidates),
        "contact_sheets": contact_sheets,
        "exact_duplicate_audit_warning": exact_warning,
        "limitations": [
            "A small pHash distance is a candidate signal, not proof of a shared source image.",
            "The script deliberately does not choose a final family threshold.",
            "pHash is not invariant to every rotation, crop, shear, border or medical-image similarity pattern.",
            "This pilot compares only the requested subclasses and cannot validate one global threshold for all modalities.",
            "No patient/slide identifiers are available, so patient-level independence is not established.",
        ],
        "next_decision": (
            "Inspect the contact sheets manually, label reviewed pairs as same-source or independent, "
            "then decide whether pHash is reliable enough to expand beyond the pilot."
        ),
    }

    report_path = output_dir / "phash_pilot_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Hash tablosu: {output_dir / 'phashes.csv'}")
    print(f"Aday örnekler: {output_dir / 'candidate_samples.csv'}")
    print(f"Temas sayfası sayısı: {len(contact_sheets)}")
    print(f"Rapor: {report_path}")
    print("NOT: Bu çıktı otomatik olarak bir pHash eşiği seçmez.")


if __name__ == "__main__":
    main()
