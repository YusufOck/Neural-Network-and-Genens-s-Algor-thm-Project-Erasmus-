from __future__ import annotations

"""Create score-blind contact sheets for holdout visual reference labeling.

The frozen duplicate rule must pass its SHA-256 check before any holdout image
is opened. The generated sheets intentionally omit pHash and every hybrid
score so visual labels cannot be influenced by the algorithm under test.
"""

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw

from data_index import project_root
from duplicate_decision_rule import load_frozen_rule
from render_calibration_review import read_dataset_root, resolve_image_path, safe_name


ROOT = project_root()
DEFAULT_CALIBRATION = ROOT / "artifacts" / "audit" / "duplicate_calibration_v1"
DEFAULT_RULE = ROOT / "config" / "duplicate_rule_v1.json"
DEFAULT_OUTPUT = DEFAULT_CALIBRATION / "holdout_blind_review_v1"

LABEL_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "left_leaf",
    "right_leaf",
    "modality_key",
    "source_group",
    "left_root",
    "left_path",
    "right_root",
    "right_path",
    "visual_verdict",
    "review_notes",
    "confidence",
    "reviewer",
    "review_date",
)

INDEX_FIELDS = (
    "pair_id",
    "pair_kind",
    "source_group",
    "sheet_file",
    "row_on_page",
    "left_leaf",
    "right_leaf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--rule-json", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-page", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=420)
    return parser.parse_args()


def load_pending_holdout_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = [dict(row) for row in csv.DictReader(handle)]
    rows = [row for row in all_rows if row.get("audit_partition") == "holdout"]
    if not rows:
        raise ValueError("Holdout review adayi bulunamadi")
    pair_ids = [row.get("pair_id", "") for row in rows]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Holdout Pair ID bos veya tekrarlaniyor")
    for row in rows:
        if row.get("truth_status") != "review_required":
            raise ValueError(f"Kesin etiketli satir review paketine karisti: {row['pair_id']}")
        if row.get("human_verdict", "").strip():
            raise ValueError(f"Holdout karari onceden doldurulmus: {row['pair_id']}")
        if row.get("review_status") != "pending_visual_review":
            raise ValueError(f"Beklenmeyen review_status: {row['pair_id']}")
    return rows


def blind_metadata_lines(row: Mapping[str, str]) -> tuple[str, str]:
    return (
        f"{row['pair_id']} | {row['pair_kind']} | {row['source_group']}",
        f"L: {row['left_leaf']}    R: {row['right_leaf']}",
    )


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def render_sheets(
    rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    calibration_root: Path,
    output_root: Path,
    pairs_per_page: int,
    thumb_size: int,
) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair_kind"], row["source_group"])].append(row)

    sheets_root = output_root / "contact_sheets"
    sheets_root.mkdir(parents=True)
    margin = 24
    gap = 20
    group_header = 44
    pair_header = 52
    row_height = pair_header + thumb_size + margin
    page_width = margin * 2 + thumb_size * 2 + gap
    index_rows: list[dict[str, str]] = []

    for (pair_kind, source_group), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: row["pair_id"])
        for page_number, start in enumerate(range(0, len(ordered), pairs_per_page), start=1):
            page_rows = ordered[start : start + pairs_per_page]
            canvas = Image.new(
                "RGB",
                (page_width, margin + group_header + row_height * len(page_rows)),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (margin, margin),
                f"BLIND HOLDOUT REVIEW | {pair_kind} | {source_group} | page {page_number}",
                fill="black",
            )
            filename = (
                f"holdout_blind__{safe_name(pair_kind)}__{safe_name(source_group)}__"
                f"page_{page_number:02d}.png"
            )
            relative = (Path("contact_sheets") / filename).as_posix()

            for row_number, row in enumerate(page_rows, start=1):
                top = margin + group_header + (row_number - 1) * row_height
                line1, line2 = blind_metadata_lines(row)
                draw.text((margin, top), line1, fill="black")
                draw.text((margin, top + 22), line2, fill="black")
                left = resolve_image_path(row, "left", dataset_root, calibration_root)
                right = resolve_image_path(row, "right", dataset_root, calibration_root)
                canvas.paste(thumbnail(left, thumb_size), (margin, top + pair_header))
                canvas.paste(
                    thumbnail(right, thumb_size),
                    (margin + thumb_size + gap, top + pair_header),
                )
                index_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "pair_kind": pair_kind,
                        "source_group": source_group,
                        "sheet_file": relative,
                        "row_on_page": str(row_number),
                        "left_leaf": row["left_leaf"],
                        "right_leaf": row["right_leaf"],
                    }
                )
            canvas.save(sheets_root / filename, format="PNG")
    return index_rows


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.pairs_per_page < 1 or args.thumb_size < 256:
        raise ValueError("Gorsel inceleme parametreleri gecersiz")
    calibration_root = args.calibration_dir.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"Blind review ciktisi zaten var: {output_root}")

    rule, rule_hash = load_frozen_rule(args.rule_json.resolve())
    rows = load_pending_holdout_rows(calibration_root / "review" / "review_labels.csv")
    dataset_root = read_dataset_root(calibration_root)
    staging = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    staging.mkdir(parents=True)
    try:
        index_rows = render_sheets(
            rows,
            dataset_root,
            calibration_root,
            staging,
            args.pairs_per_page,
            args.thumb_size,
        )
        if {row["pair_id"] for row in index_rows} != {row["pair_id"] for row in rows}:
            raise ValueError("Blind sheet Pair ID kapsami hatali")
        label_rows = []
        for source in sorted(rows, key=lambda row: row["pair_id"]):
            prepared = {field: source.get(field, "") for field in LABEL_FIELDS}
            prepared["visual_verdict"] = ""
            prepared["review_notes"] = ""
            prepared["confidence"] = ""
            prepared["reviewer"] = ""
            prepared["review_date"] = ""
            label_rows.append(prepared)
        write_csv(staging / "blind_labels.csv", LABEL_FIELDS, label_rows)
        write_csv(staging / "contact_sheet_index.csv", INDEX_FIELDS, index_rows)
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "holdout_opened_for_blind_visual_reference_review",
            "purpose": "Visual reference labeling without algorithm scores",
            "frozen_rule_id": rule["rule_id"],
            "frozen_rule_sha256": rule_hash,
            "holdout_candidate_count": len(rows),
            "contact_sheet_count": len({row["sheet_file"] for row in index_rows}),
            "score_fields_exposed": [],
            "by_pair_kind": dict(sorted(Counter(row["pair_kind"] for row in rows).items())),
            "by_source_group": dict(sorted(Counter(row["source_group"] for row in rows).items())),
            "warning": "AI-assisted visual reference labels are not medical-expert gold standard.",
        }
        with (staging / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("Islem: HOLDOUT blind visual review paketi (algoritma skoru yok)")
    print(f"Dondurulmus kural SHA-256: {rule_hash}")
    print(f"Holdout aday cifti: {len(rows)}")
    print(f"Temas sayfasi: {len({row['sheet_file'] for row in index_rows})}")
    print(f"Etiket tablosu: {output_root / 'blind_labels.csv'}")
    print("pHash/ORB/ECC/SSIM/NMI/korelasyon skorlari sayfalara eklenmedi.")


if __name__ == "__main__":
    main()
