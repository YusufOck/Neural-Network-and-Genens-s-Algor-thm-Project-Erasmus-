from __future__ import annotations

"""Render traceable contact sheets for calibration review candidates.

This utility does not label pairs, choose similarity thresholds, create a
classifier split, or train a model. It preserves the candidate Pair IDs and
creates a separate human-editable worksheet plus an index from every Pair ID
to its contact-sheet page and row.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from data_index import project_root


AUDIT_PARTITIONS = ("development", "holdout")
ALLOWED_PAIR_KINDS = (
    "random_same_leaf_review",
    "phash_hard_same_leaf_review",
    "cross_leaf_same_source_group_review",
)
ALLOWED_VISUAL_VERDICTS = (
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
)
ALLOWED_CONFIDENCE = ("high", "medium", "low")
REQUIRED_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "truth_status",
    "left_leaf",
    "right_leaf",
    "modality_key",
    "source_group",
    "left_root",
    "left_path",
    "right_root",
    "right_path",
    "phash_distance",
    "human_verdict",
    "review_notes",
)
REVIEW_EXTRA_FIELDS = (
    "review_status",
    "confidence",
    "reviewer",
    "review_date",
)
INDEX_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "source_group",
    "sheet_file",
    "row_on_page",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
)


def parse_args() -> argparse.Namespace:
    calibration_root = (
        project_root() / "artifacts" / "audit" / "duplicate_calibration_v1"
    )
    parser = argparse.ArgumentParser(
        description=(
            "278 calibration adayını Pair ID'li temas sayfalarına ve güvenli insan "
            "inceleme tablosuna dönüştürür. Eşik/split/model üretmez."
        )
    )
    parser.add_argument(
        "--calibration-dir",
        default=str(calibration_root),
        help="Calibration-set v1 artefakt klasörü.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Varsayılan: <calibration-dir>/review_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Varsayılan: <calibration-dir>/review",
    )
    parser.add_argument(
        "--pairs-per-page",
        type=int,
        default=6,
        help="Her PNG sayfasındaki çift sayısı.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=320,
        help="Her görüntü kutusunun piksel boyutu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pairs_per_page < 1:
        raise ValueError("--pairs-per-page en az 1 olmalıdır.")
    if args.thumb_size < 128:
        raise ValueError("--thumb-size görsel inceleme için en az 128 olmalıdır.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"İnceleme adayları bulunamadı: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"Aday CSV içinde eksik sütunlar var: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("İnceleme adayı CSV dosyası boş.")
    return fieldnames, rows


def validate_candidates(rows: Sequence[Mapping[str, str]]) -> Mapping[str, object]:
    ids: set[str] = set()
    counts_by_partition: Counter[str] = Counter()
    counts_by_kind: Counter[str] = Counter()
    for row in rows:
        pair_id = row["pair_id"].strip()
        if not pair_id:
            raise ValueError("Boş Pair ID bulundu.")
        if pair_id in ids:
            raise ValueError(f"Tekrarlanan Pair ID: {pair_id}")
        ids.add(pair_id)

        partition = row["audit_partition"]
        if partition not in AUDIT_PARTITIONS:
            raise ValueError(f"Bilinmeyen audit partition: {partition}")
        pair_kind = row["pair_kind"]
        if pair_kind not in ALLOWED_PAIR_KINDS:
            raise ValueError(f"Bilinmeyen review pair kind: {pair_kind}")
        if row["truth_status"] != "review_required":
            raise ValueError(
                f"Review tablosuna kesin pozitif/etiketli satır karışmış: {pair_id}"
            )
        if row["human_verdict"].strip():
            raise ValueError(
                f"Kaynak aday tablosunda önceden doldurulmuş karar var: {pair_id}"
            )
        counts_by_partition[partition] += 1
        counts_by_kind[pair_kind] += 1

    return {
        "candidate_count": len(rows),
        "unique_pair_id_count": len(ids),
        "blank_human_verdict_count": sum(
            not row["human_verdict"].strip() for row in rows
        ),
        "by_partition": dict(sorted(counts_by_partition.items())),
        "by_pair_kind": dict(sorted(counts_by_kind.items())),
    }


def read_dataset_root(calibration_root: Path) -> Path:
    report_path = calibration_root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Calibration raporu bulunamadı: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    dataset_root = Path(str(report["dataset_root"])).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Rapordaki veri kökü bulunamadı: {dataset_root}")
    return dataset_root


def resolve_image_path(
    row: Mapping[str, str],
    side: str,
    dataset_root: Path,
    calibration_root: Path,
) -> Path:
    root_name = row[f"{side}_root"]
    relative = Path(row[f"{side}_path"])
    if root_name == "dataset":
        resolved = (dataset_root / relative).resolve()
        expected_parent = dataset_root
    elif root_name == "artifact":
        resolved = (calibration_root / relative).resolve()
        expected_parent = calibration_root
    else:
        raise ValueError(f"Bilinmeyen görüntü kökü: {root_name}")

    if not resolved.is_relative_to(expected_parent):
        raise ValueError(f"Görüntü yolu izin verilen kökün dışına çıkıyor: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Görüntü bulunamadı: {resolved}")
    return resolved


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")
    return result or "group"


def group_key(row: Mapping[str, str]) -> Tuple[str, str, str]:
    return row["audit_partition"], row["pair_kind"], row["source_group"]


def verdict_text(row: Mapping[str, str]) -> str:
    verdict = row.get("human_verdict", "").strip()
    if not verdict:
        return "HUMAN VERDICT: ________"
    confidence = row.get("confidence", "").strip() or "n/a"
    return f"VERDICT: {verdict} | confidence={confidence}"


def render_contact_sheets(
    rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    calibration_root: Path,
    staging_root: Path,
    pairs_per_page: int,
    thumb_size: int,
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    sheets_root = staging_root / "contact_sheets"
    sheets_root.mkdir(parents=True)
    index_rows: List[Dict[str, str]] = []
    margin = 18
    gap = 16
    group_header_height = 38
    pair_header_height = 78
    row_height = pair_header_height + thumb_size + margin
    page_width = margin * 2 + thumb_size * 2 + gap

    for key in sorted(grouped):
        partition, pair_kind, source_group = key
        group_rows = sorted(grouped[key], key=lambda row: row["pair_id"])
        for page_number, start in enumerate(
            range(0, len(group_rows), pairs_per_page), start=1
        ):
            page_rows = group_rows[start : start + pairs_per_page]
            canvas = Image.new(
                "RGB",
                (
                    page_width,
                    margin + group_header_height + row_height * len(page_rows),
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (margin, margin),
                f"{partition} | {pair_kind} | {source_group} | page {page_number}",
                fill="black",
            )

            for row_number, row in enumerate(page_rows, start=1):
                top = margin + group_header_height + (row_number - 1) * row_height
                distance = row["phash_distance"] or "n/a"
                draw.text(
                    (margin, top),
                    f"{row['pair_id']} | pHash distance={distance} | {verdict_text(row)}",
                    fill="black",
                )
                draw.text(
                    (margin, top + 20),
                    f"L leaf: {row['left_leaf']}    ->    R leaf: {row['right_leaf']}",
                    fill="black",
                )
                draw.text(
                    (margin, top + 40),
                    f"L: {Path(row['left_path']).name}",
                    fill="black",
                )
                draw.text(
                    (margin + thumb_size + gap, top + 40),
                    f"R: {Path(row['right_path']).name}",
                    fill="black",
                )

                left_path = resolve_image_path(
                    row, "left", dataset_root, calibration_root
                )
                right_path = resolve_image_path(
                    row, "right", dataset_root, calibration_root
                )
                canvas.paste(thumbnail(left_path, thumb_size), (margin, top + pair_header_height))
                canvas.paste(
                    thumbnail(right_path, thumb_size),
                    (margin + thumb_size + gap, top + pair_header_height),
                )

                filename = (
                    f"{safe_name(partition)}__{safe_name(pair_kind)}__"
                    f"{safe_name(source_group)}__page_{page_number:02d}.png"
                )
                relative_sheet = (Path("contact_sheets") / filename).as_posix()
                index_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "audit_partition": partition,
                        "pair_kind": pair_kind,
                        "source_group": source_group,
                        "sheet_file": relative_sheet,
                        "row_on_page": str(row_number),
                        "left_leaf": row["left_leaf"],
                        "right_leaf": row["right_leaf"],
                        "left_path": row["left_path"],
                        "right_path": row["right_path"],
                    }
                )

            filename = (
                f"{safe_name(partition)}__{safe_name(pair_kind)}__"
                f"{safe_name(source_group)}__page_{page_number:02d}.png"
            )
            canvas.save(sheets_root / filename, format="PNG")
    return index_rows


def prepare_review_rows(
    rows: Sequence[Mapping[str, str]], input_fields: Sequence[str]
) -> Tuple[List[str], List[Dict[str, str]]]:
    output_fields = list(input_fields)
    for field in REVIEW_EXTRA_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    prepared: List[Dict[str, str]] = []
    for source in rows:
        row = {field: source.get(field, "") for field in output_fields}
        row["review_status"] = "pending_visual_review"
        row["confidence"] = ""
        row["reviewer"] = ""
        row["review_date"] = ""
        prepared.append(row)
    return output_fields, prepared


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_rendered_output(
    rows: Sequence[Mapping[str, str]],
    index_rows: Sequence[Mapping[str, str]],
    staging_root: Path,
) -> Mapping[str, object]:
    expected_ids = {row["pair_id"] for row in rows}
    indexed_ids = [row["pair_id"] for row in index_rows]
    if len(indexed_ids) != len(set(indexed_ids)):
        raise ValueError("Contact-sheet indeksinde tekrarlanan Pair ID var.")
    if set(indexed_ids) != expected_ids:
        raise ValueError(
            "Contact-sheet indeksi ile aday Pair ID kümesi eşleşmiyor. "
            f"Eksik={sorted(expected_ids - set(indexed_ids))[:5]}, "
            f"fazla={sorted(set(indexed_ids) - expected_ids)[:5]}"
        )
    sheet_files = {row["sheet_file"] for row in index_rows}
    missing_sheets = [
        sheet for sheet in sheet_files if not (staging_root / sheet).is_file()
    ]
    if missing_sheets:
        raise FileNotFoundError(f"İndekste bulunamayan temas sayfaları: {missing_sheets[:5]}")
    return {
        "candidate_pair_count": len(rows),
        "indexed_pair_count": len(index_rows),
        "unique_indexed_pair_count": len(set(indexed_ids)),
        "contact_sheet_count": len(sheet_files),
        "pair_id_mapping_mismatch_count": 0,
    }


def counter_dict(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def main() -> None:
    args = parse_args()
    validate_args(args)
    calibration_root = Path(args.calibration_dir).resolve()
    candidates_path = (
        Path(args.candidates).resolve()
        if args.candidates
        else calibration_root / "review_candidates.csv"
    )
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else calibration_root / "review"
    )
    if output_root.exists():
        raise FileExistsError(
            f"İnceleme çıktısı zaten var ve güvenlik için ezilmeyecek: {output_root}"
        )

    dataset_root = read_dataset_root(calibration_root)
    input_fields, rows = read_candidates(candidates_path)
    source_validation = validate_candidates(rows)

    staging_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Geçici çıktı klasörü zaten var: {staging_root}")
    staging_root.mkdir(parents=True)

    try:
        review_fields, review_rows = prepare_review_rows(rows, input_fields)
        index_rows = render_contact_sheets(
            rows,
            dataset_root,
            calibration_root,
            staging_root,
            args.pairs_per_page,
            args.thumb_size,
        )
        output_validation = validate_rendered_output(rows, index_rows, staging_root)
        write_csv(staging_root / "review_labels.csv", review_fields, review_rows)
        write_csv(staging_root / "contact_sheet_index.csv", INDEX_FIELDS, index_rows)

        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Human review preparation only; no automatic labels, threshold, split, or model.",
            "calibration_root": str(calibration_root),
            "dataset_root": str(dataset_root),
            "candidate_source": str(candidates_path),
            "candidate_source_sha256": sha256_file(candidates_path),
            "parameters": {
                "pairs_per_page": args.pairs_per_page,
                "thumb_size": args.thumb_size,
            },
            "allowed_visual_verdicts": list(ALLOWED_VISUAL_VERDICTS),
            "allowed_confidence_values": list(ALLOWED_CONFIDENCE),
            "review_protocol": {
                "first": "Label development candidates and fit/freeze method thresholds.",
                "then": "Keep holdout labels unseen until the method is frozen; evaluate once.",
                "classifier_split": "This audit partition is not the future classifier train/validation/test split.",
            },
            "source_validation": source_validation,
            "output_validation": output_validation,
            "counts": {
                "by_partition": counter_dict(row["audit_partition"] for row in rows),
                "by_pair_kind": counter_dict(row["pair_kind"] for row in rows),
                "by_source_group": counter_dict(row["source_group"] for row in rows),
            },
            "status": "rendered_pending_human_review",
            "warning": (
                "pHash distance is evidence only. Blank verdicts must not be interpreted "
                "as independent pairs or negative ground truth."
            ),
        }
        with (staging_root / "review_render_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print("İşlem türü: calibration insan-inceleme paketi (eşik/split/model yok)")
    print(f"Kaynak aday: {len(rows)}")
    print(f"Temas sayfası: {output_validation['contact_sheet_count']}")
    print(f"Pair ID eşleme hatası: {output_validation['pair_id_mapping_mismatch_count']}")
    print(f"İnceleme tablosu: {output_root / 'review_labels.csv'}")
    print(f"Sayfa indeksi: {output_root / 'contact_sheet_index.csv'}")
    print(f"Temas sayfaları: {output_root / 'contact_sheets'}")
    print("Karar alanları boş bırakıldı; insan incelemesi yapılmadan etiket sayılmaz.")
    print("Kontrol: önce development incelenecek; holdout yöntem dondurulana kadar kapalı kalacak.")


if __name__ == "__main__":
    main()
