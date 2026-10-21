from __future__ import annotations

"""Render one calibration-review pair at a time by Pair ID.

The command is deliberately read-only with respect to review_labels.csv.  It
does not assign a verdict, choose a pHash threshold, create a classifier split,
or train a model.  Holdout pairs are hidden by default so that the duplicate
method can be developed without looking at its later evaluation set.
"""

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw

from data_index import project_root
from render_calibration_review import read_dataset_root, resolve_image_path


def parse_args() -> argparse.Namespace:
    calibration_root = (
        project_root() / "artifacts" / "audit" / "duplicate_calibration_v1"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Bir development inceleme çiftini Pair ID ile büyük ve izlenebilir "
            "bir PNG olarak hazırlar. Karar yazmaz."
        )
    )
    parser.add_argument("--pair-id", required=True, help="Örnek: C000613")
    parser.add_argument(
        "--calibration-dir",
        default=str(calibration_root),
        help="Calibration-set v1 artefakt klasörü.",
    )
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Varsayılan: <calibration-dir>/review/review_labels.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Varsayılan: <calibration-dir>/review/pair_views/<Pair ID>.png",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Her görüntü için kare kutu boyutu (varsayılan: 640).",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Yöntem dondurulmadan kullanılmamalıdır; holdout çiftini açmaya izin verir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan tek-çift PNG çıktısını değiştirir.",
    )
    return parser.parse_args()


def read_review_row(path: Path, pair_id: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"İnceleme tablosu bulunamadı: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["pair_id"] == pair_id]
    if not matches:
        raise ValueError(f"Pair ID inceleme tablosunda bulunamadı: {pair_id}")
    if len(matches) != 1:
        raise ValueError(f"Pair ID benzersiz değil: {pair_id}")
    return dict(matches[0])


def validate_review_access(row: Mapping[str, str], allow_holdout: bool) -> None:
    partition = row["audit_partition"]
    if partition == "holdout" and not allow_holdout:
        raise PermissionError(
            "Bu çift holdout bölümünde. Yöntem dondurulana kadar holdout görüntüleri "
            "açılmamalıdır."
        )
    if partition not in {"development", "holdout"}:
        raise ValueError(f"Bilinmeyen audit partition: {partition}")


def fit_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def metadata_lines(row: Mapping[str, str]) -> Sequence[str]:
    verdict = row.get("human_verdict", "").strip() or "BEKLIYOR"
    return (
        f"Pair ID: {row['pair_id']} | bölüm: {row['audit_partition']}",
        f"aday türü: {row['pair_kind']} | source group: {row['source_group']}",
        f"pHash Hamming mesafesi: {row['phash_distance']} | insan kararı: {verdict}",
        f"L: {row['left_leaf']} / {Path(row['left_path']).name}",
        f"R: {row['right_leaf']} / {Path(row['right_path']).name}",
    )


def render_pair(
    row: Mapping[str, str],
    left_path: Path,
    right_path: Path,
    output_path: Path,
    image_size: int,
) -> None:
    if image_size < 256:
        raise ValueError("--image-size görsel inceleme için en az 256 olmalıdır.")

    margin = 24
    gap = 20
    line_height = 22
    lines = metadata_lines(row)
    header_height = margin + line_height * len(lines) + 16
    footer_height = 34
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + image_size * 2 + gap,
            header_height + image_size + footer_height + margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black")
        y += line_height

    left = fit_image(left_path, image_size)
    right = fit_image(right_path, image_size)
    canvas.paste(left, (margin, header_height))
    canvas.paste(right, (margin + image_size + gap, header_height))
    draw.text((margin, header_height + image_size + 10), "SOL (L)", fill="black")
    draw.text(
        (margin + image_size + gap, header_height + image_size + 10),
        "SAĞ (R)",
        fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def main() -> None:
    args = parse_args()
    calibration_root = Path(args.calibration_dir).resolve()
    review_csv = (
        Path(args.review_csv).resolve()
        if args.review_csv
        else calibration_root / "review" / "review_labels.csv"
    )
    pair_id = args.pair_id.strip().upper()
    row = read_review_row(review_csv, pair_id)
    validate_review_access(row, args.allow_holdout)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else calibration_root / "review" / "pair_views" / f"{pair_id}.png"
    )
    dataset_root = read_dataset_root(calibration_root)
    left_path = resolve_image_path(row, "left", dataset_root, calibration_root)
    right_path = resolve_image_path(row, "right", dataset_root, calibration_root)
    output_existed = output_path.exists()
    if not output_existed or args.overwrite:
        render_pair(row, left_path, right_path, output_path, args.image_size)

    print("İşlem türü: tek review çifti görselleştirme (karar/eşik/split/model yok)")
    print(f"Pair ID: {pair_id}")
    print(f"Bölüm: {row['audit_partition']}")
    print(f"Sol: {left_path}")
    print(f"Sağ: {right_path}")
    print(f"pHash Hamming mesafesi: {row['phash_distance']}")
    print(f"Çıktı: {output_path}")
    if output_existed and not args.overwrite:
        print("NOT: Var olan PNG korundu. Yeniden üretmek için --overwrite kullanın.")


if __name__ == "__main__":
    main()
