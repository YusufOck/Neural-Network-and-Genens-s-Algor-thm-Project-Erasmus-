from __future__ import annotations

"""Render score-blind natural near-duplicate audit contact sheets."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data_index import project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


REQUIRED_FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
)
FORBIDDEN_BLIND_FIELDS = {
    "selection_stratum",
    "phash_distance",
    "robust_rank",
    "robust_similarity",
    "candidate_channel",
    "robust_best_union_rank",
    "robust_best_cosine_similarity",
}
INDEX_FIELDS = (
    "pair_id",
    "sheet_file",
    "row_on_page",
    "source_group",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    selection = root / "research" / "natural_duplicate_audit_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmuş doğal duplicate audit manifestini score-blind Pair-ID temas "
            "sayfalarına dönüştürür; karar/split/model üretmez."
        )
    )
    parser.add_argument("--selection-dir", default=str(selection))
    parser.add_argument(
        "--output-dir",
        default=str(root / "artifacts" / "audit" / "natural_duplicate_audit_v1" / "review"),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pairs-per-page", type=int, default=6)
    parser.add_argument("--thumb-size", type=int, default=320)
    return parser.parse_args()


def read_review_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_FIELDS) - set(fields))
        if missing:
            raise ValueError(f"Review manifest eksik sütunlar içeriyor: {missing}")
        leaked = sorted(FORBIDDEN_BLIND_FIELDS & set(fields))
        if leaked:
            raise ValueError(f"Score-blind manifest gizli alan sızdırıyor: {leaked}")
        rows = [dict(row) for row in reader]
    return fields, rows


def validate_review_rows(rows: Sequence[Mapping[str, str]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Beklenen {expected_count} yerine {len(rows)} review satırı var.")
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Tekrarlanan Pair ID var.")
    for row in rows:
        if any(
            row[field].strip()
            for field in (
                "visual_verdict",
                "confidence",
                "reviewer",
                "review_date",
                "review_notes",
            )
        ):
            raise ValueError(f"Önceden doldurulmuş review kararı var: {row['pair_id']}")


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def render_page(
    rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    page_number: int,
    total_pages: int,
    thumb_size: int,
) -> Image.Image:
    margin = 20
    gap = 12
    header_height = 54
    label_height = 30
    row_gap = 18
    width = margin * 2 + thumb_size * 2 + gap
    row_height = label_height + thumb_size + row_gap
    height = header_height + len(rows) * row_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"Natural audit v1 | SCORE-BLIND | page {page_number}/{total_pages}",
        fill="black",
        font=font(18),
    )
    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        draw.text(
            (margin, y),
            f"{row['pair_id']} | {row['source_group']}",
            fill="black",
            font=font(16),
        )
        left_path = (dataset_root / Path(row["left_path"])).resolve()
        right_path = (dataset_root / Path(row["right_path"])).resolve()
        if not left_path.is_relative_to(dataset_root) or not right_path.is_relative_to(
            dataset_root
        ):
            raise ValueError("Dataset kökü dışına çıkan görüntü yolu bulundu.")
        left = thumbnail(left_path, thumb_size)
        right = thumbnail(right_path, thumb_size)
        image_y = y + label_height
        canvas.paste(left, (margin, image_y))
        canvas.paste(right, (margin + thumb_size + gap, image_y))
        draw.text((margin + 5, image_y + 5), "L", fill="yellow", font=font(16))
        draw.text(
            (margin + thumb_size + gap + 5, image_y + 5),
            "R",
            fill="yellow",
            font=font(16),
        )
    return canvas


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    if args.pairs_per_page < 1:
        raise ValueError("--pairs-per-page pozitif olmalıdır.")
    if args.thumb_size < 128:
        raise ValueError("--thumb-size en az 128 olmalıdır.")
    selection_dir = Path(args.selection_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Natural audit review paketi zaten var: {output_dir}")
    report_path = selection_dir / "selection_report.json"
    manifest_path = selection_dir / "review_manifest.csv"
    selection_report = json.loads(report_path.read_text(encoding="utf-8"))
    if selection_report["status"] != "selection_frozen_images_not_opened":
        raise ValueError("Selection report beklenen frozen durumda değil.")
    if file_sha256(manifest_path) != selection_report["integrity"][
        "review_manifest_sha256"
    ]:
        raise ValueError("Review manifest frozen hash ile eşleşmiyor.")
    _, rows = read_review_manifest(manifest_path)
    expected_count = int(selection_report["counts"]["selected_pairs"])
    validate_review_rows(rows, expected_count)
    dataset_root = resolve_dataset_root(args.data_dir)
    total_pages = (len(rows) + args.pairs_per_page - 1) // args.pairs_per_page

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="natural-review-v1-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        sheets = staging / "contact_sheets"
        sheets.mkdir(parents=True)
        for page_index, start in enumerate(
            range(0, len(rows), args.pairs_per_page), start=1
        ):
            page_rows = rows[start : start + args.pairs_per_page]
            sheet_name = f"natural_audit_page_{page_index:03d}.png"
            page = render_page(
                page_rows,
                dataset_root,
                page_index,
                total_pages,
                args.thumb_size,
            )
            page.save(sheets / sheet_name, format="PNG", optimize=True)
            for row_on_page, row in enumerate(page_rows, start=1):
                index_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "sheet_file": f"contact_sheets/{sheet_name}",
                        "row_on_page": row_on_page,
                        "source_group": row["source_group"],
                    }
                )
        index_path = staging / "sheet_index.csv"
        write_csv(index_path, INDEX_FIELDS, index_rows)
        render_report: dict[str, object] = {
            "schema_version": 1,
            "status": "score_blind_contact_sheets_rendered_not_reviewed",
            "counts": {
                "pair_count": len(rows),
                "sheet_count": total_pages,
                "index_count": len(index_rows),
                "blank_review_verdict_count": sum(
                    not row["visual_verdict"] for row in rows
                ),
            },
            "layout": {
                "pairs_per_page": args.pairs_per_page,
                "thumb_size": args.thumb_size,
                "visible_metadata": ["pair_id", "source_group", "left_right_marker"],
                "hidden_metadata": sorted(FORBIDDEN_BLIND_FIELDS),
            },
            "integrity": {
                "selection_report_sha256": file_sha256(report_path),
                "review_manifest_sha256": file_sha256(manifest_path),
                "renderer_sha256": file_sha256(Path(__file__).resolve()),
                "sheet_index_sha256": file_sha256(index_path),
            },
            "limitations": [
                "Rendering contact sheets does not label any pair.",
                "The review remains AI-assisted and is not a clinical-expert gold standard.",
                "No family, classifier split or model is created by this package.",
            ],
        }
        output_report = staging / "render_report.json"
        output_report.write_text(
            json.dumps(render_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staging), str(output_dir))

    print("İşlem: NATURAL DUPLICATE AUDIT v1 SCORE-BLIND RENDER")
    print(f"Çift: {len(rows)}, temas sayfası: {total_pages}")
    print("Verdict alanları değiştirilmedi; skor/stratum sayfalarda gösterilmedi.")
    print(f"Rapor: {output_dir / 'render_report.json'}")
    return output_dir / "render_report.json", render_report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
