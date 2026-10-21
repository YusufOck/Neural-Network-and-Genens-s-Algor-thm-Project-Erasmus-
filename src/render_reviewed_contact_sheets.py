from __future__ import annotations

"""Render development contact sheets with recorded verdicts visible in PNG headers."""

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from data_index import project_root
from render_calibration_review import (
    INDEX_FIELDS,
    read_dataset_root,
    render_contact_sheets,
    validate_rendered_output,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    calibration = project_root() / "artifacts" / "audit" / "duplicate_calibration_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Kaydedilmiş development kararlarını başlıklarda gösteren temas sayfaları "
            "üretir. Holdout görüntülerini dahil etmez."
        )
    )
    parser.add_argument("--calibration-dir", default=str(calibration))
    parser.add_argument("--review-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pairs-per-page", type=int, default=6)
    parser.add_argument("--thumb-size", type=int, default=320)
    return parser.parse_args()


def read_completed_development(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row["audit_partition"] == "development"
        ]
    if not rows:
        raise ValueError("Development inceleme satırı bulunamadı.")
    pending = [row["pair_id"] for row in rows if not row["human_verdict"].strip()]
    if pending:
        raise ValueError(f"Kararı boş development satırları var: {pending[:10]}")
    return rows


def main() -> None:
    args = parse_args()
    if args.pairs_per_page < 1 or args.thumb_size < 128:
        raise ValueError("Geçersiz sayfa veya görsel boyutu.")
    calibration = Path(args.calibration_dir).resolve()
    review_csv = (
        Path(args.review_csv).resolve()
        if args.review_csv
        else calibration / "review" / "review_labels.csv"
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else calibration / "review" / "reviewed_development_v1"
    )
    expected_parent = (calibration / "review").resolve()
    if output.parent != expected_parent:
        raise ValueError(f"Çıktı review klasörünün doğrudan altında olmalıdır: {output}")
    if output.exists():
        raise FileExistsError(f"Çıktı zaten var; değiştirilmedi: {output}")

    rows = read_completed_development(review_csv)
    dataset_root = read_dataset_root(calibration)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".reviewed_development_staging_"
    ) as temporary:
        staging = Path(temporary)
        index_rows = render_contact_sheets(
            rows,
            dataset_root,
            calibration,
            staging,
            args.pairs_per_page,
            args.thumb_size,
        )
        validation = validate_rendered_output(rows, index_rows, staging)
        write_csv(staging / "contact_sheet_index.csv", INDEX_FIELDS, index_rows)
        report = {
            "purpose": "Development visual decisions rendered for inspection; holdout excluded.",
            "review_csv": str(review_csv),
            "development_pair_count": len(rows),
            "holdout_pair_count": 0,
            "verdict_counts": dict(
                sorted(Counter(row["human_verdict"] for row in rows).items())
            ),
            "validation": validation,
        }
        with (staging / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        staging.replace(output)

    print("İşlem türü: kararları gösteren development temas sayfaları")
    print(f"Development çiftleri: {len(rows)}")
    print(f"Temas sayfaları: {validation['contact_sheet_count']}")
    print("Holdout çifti: 0 (dahil edilmedi)")
    print(f"Çıktı: {output}")


if __name__ == "__main__":
    main()
