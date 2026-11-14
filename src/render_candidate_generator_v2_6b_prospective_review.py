from __future__ import annotations

"""Render the public v2.6b prospective review manifest without private metadata."""

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
FORBIDDEN_PRIVATE_FIELDS = {
    "selection_stratum",
    "v2_6b_candidate",
    "candidate_membership",
    "descriptor_channel",
    "robust_acceptance_k",
    "robust_best_rank",
    "robust_rank_lower_bound",
    "phash_enabled",
    "phash_distance",
}
INDEX_FIELDS = ("pair_id", "sheet_file", "row_on_page", "source_group")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Render only the public frozen v2.6b review manifest into score-blind "
            "Pair-ID contact sheets; do not label, unblind, split or train."
        )
    )
    parser.add_argument(
        "--selection-dir",
        default=str(root / "research" / "candidate_generator_v2_6b_prospective_selection_v1"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "candidate_generator_v2_6b_prospective_review_v1"
        ),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pairs-per-page", type=int, default=6)
    parser.add_argument("--thumb-size", type=int, default=320)
    return parser.parse_args()


def read_public_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if fields != list(REQUIRED_FIELDS):
            raise ValueError(f"Public review schema differs from frozen fields: {fields}")
        leaked = sorted(FORBIDDEN_PRIVATE_FIELDS & set(fields))
        if leaked:
            raise ValueError(f"Private fields leaked into public review: {leaked}")
        rows = [dict(row) for row in reader]
    return rows


def validate_public_rows(rows: Sequence[Mapping[str, str]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} review rows, found {len(rows)}.")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate Pair ID in public review manifest.")
    paths = [
        str(row[field]).replace("\\", "/")
        for row in rows
        for field in ("left_path", "right_path")
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("A selected image occurs in more than one review pair.")
    for row in rows:
        if not str(row["pair_id"]).startswith("V26P"):
            raise ValueError(f"Unexpected Pair ID prefix: {row['pair_id']}")
        if any(
            str(row[field]).strip()
            for field in ("visual_verdict", "confidence", "reviewer", "review_date", "review_notes")
        ):
            raise ValueError(f"Public review contains a prefilled decision: {row['pair_id']}")


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def render_page(
    rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    page_number: int,
    total_pages: int,
    thumb_size: int,
) -> Image.Image:
    margin, gap = 20, 12
    header_height, label_height, row_gap = 58, 30, 18
    width = margin * 2 + thumb_size * 2 + gap
    row_height = label_height + thumb_size + row_gap
    height = header_height + len(rows) * row_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"V2.6b prospective audit | SCORE-BLIND | page {page_number}/{total_pages}",
        fill="black",
        font=font(18),
    )
    root = dataset_root.resolve()
    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        draw.text(
            (margin, y),
            f"{row['pair_id']} | {row['source_group']}",
            fill="black",
            font=font(16),
        )
        left_path = (root / Path(str(row["left_path"]))).resolve()
        right_path = (root / Path(str(row["right_path"]))).resolve()
        if not left_path.is_relative_to(root) or not right_path.is_relative_to(root):
            raise ValueError("Review image path escapes the dataset root.")
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(left_path if not left_path.is_file() else right_path)
        image_y = y + label_height
        canvas.paste(thumbnail(left_path, thumb_size), (margin, image_y))
        canvas.paste(thumbnail(right_path, thumb_size), (margin + thumb_size + gap, image_y))
        draw.text((margin + 5, image_y + 5), "L", fill="yellow", font=font(16))
        draw.text(
            (margin + thumb_size + gap + 5, image_y + 5),
            "R",
            fill="yellow",
            font=font(16),
        )
    return canvas


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, object]]:
    if args.pairs_per_page < 1:
        raise ValueError("--pairs-per-page must be positive.")
    if args.thumb_size < 128:
        raise ValueError("--thumb-size must be at least 128.")
    selection_dir = Path(args.selection_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"V2.6b review render already exists: {output_dir}")
    source_report_path = selection_dir / "selection_report.json"
    review_path = selection_dir / "review_manifest.csv"
    selection_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if selection_report.get("status") != "v2_6b_prospective_selection_frozen_images_not_opened":
        raise ValueError("V2.6b selection report is not in the frozen pre-review state.")
    if file_sha256(review_path) != selection_report["integrity"]["review_manifest_sha256"]:
        raise ValueError("Public review manifest hash differs from the frozen selection report.")
    rows = read_public_review(review_path)
    validate_public_rows(rows, int(selection_report["selection"]["selected_pairs"]))
    dataset_root = resolve_dataset_root(args.data_dir)
    total_pages = (len(rows) + args.pairs_per_page - 1) // args.pairs_per_page

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="v2-6b-review-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        sheets = staging / "contact_sheets"
        sheets.mkdir(parents=True)
        sheet_hashes: dict[str, str] = {}
        for page_number, start in enumerate(range(0, len(rows), args.pairs_per_page), start=1):
            page_rows = rows[start : start + args.pairs_per_page]
            sheet_name = f"v2_6b_prospective_page_{page_number:03d}.png"
            sheet_path = sheets / sheet_name
            render_page(page_rows, dataset_root, page_number, total_pages, args.thumb_size).save(
                sheet_path, format="PNG", optimize=True
            )
            relative = f"contact_sheets/{sheet_name}"
            sheet_hashes[relative] = file_sha256(sheet_path)
            for row_on_page, row in enumerate(page_rows, start=1):
                index_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "sheet_file": relative,
                        "row_on_page": row_on_page,
                        "source_group": row["source_group"],
                    }
                )
        index_path = staging / "sheet_index.csv"
        write_csv(index_path, index_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "v2_6b_score_blind_contact_sheets_rendered_not_reviewed",
            "does_not_read_private_manifest": True,
            "does_not_assign_visual_decisions": True,
            "counts": {
                "pair_count": len(rows),
                "unique_images": len(
                    {str(row[field]) for row in rows for field in ("left_path", "right_path")}
                ),
                "sheet_count": total_pages,
                "index_count": len(index_rows),
                "blank_review_verdict_count": sum(not str(row["visual_verdict"]) for row in rows),
            },
            "layout": {
                "pairs_per_page": args.pairs_per_page,
                "thumb_size": args.thumb_size,
                "visible_metadata": ["pair_id", "source_group", "left_right_marker"],
                "hidden_private_fields": sorted(FORBIDDEN_PRIVATE_FIELDS),
            },
            "integrity": {
                "selection_report_sha256": file_sha256(source_report_path),
                "review_manifest_sha256": file_sha256(review_path),
                "renderer_sha256": file_sha256(Path(__file__).resolve()),
                "sheet_index_sha256": file_sha256(index_path),
                "sheet_sha256": sheet_hashes,
            },
            "decision": "RENDER COMPLETE, NOT REVIEWED: freeze this package before recording verdicts.",
        }
        report_path_out = staging / "render_report.json"
        report_path_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(staging), str(output_dir))

    print("İşlem: V2.6b PROSPECTIVE SCORE-BLIND RENDER")
    print(f"Çift: {len(rows)}, benzersiz görüntü: {2 * len(rows)}, sayfa: {total_pages}")
    print("Private manifest okunmadı; verdict alanları değiştirilmedi.")
    print(f"Rapor: {output_dir / 'render_report.json'}")
    return output_dir / "render_report.json", report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
