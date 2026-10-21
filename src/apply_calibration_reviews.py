from __future__ import annotations

"""Apply a traceable review-decision CSV to the local calibration worksheet.

The decision source is kept in Git while the image paths and rendered audit
artifacts remain local.  The legacy worksheet column is named
``human_verdict``; the reviewer field must therefore state clearly when a
decision was produced by an AI-assisted visual review.
"""

import argparse
import csv
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from data_index import project_root


ALLOWED_VERDICTS = {
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
DECISION_FIELDS = {
    "pair_id",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "İzlenebilir development kararlarını yerel calibration review tablosuna "
            "uygular; holdout satırlarını değiştirmez."
        )
    )
    parser.add_argument(
        "--decisions",
        default=str(root / "research" / "development_visual_review_v1.csv"),
    )
    parser.add_argument(
        "--review-csv",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "duplicate_calibration_v1"
            / "review"
            / "review_labels.csv"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def validate_decisions(decisions: Sequence[Mapping[str, str]]) -> None:
    if not decisions:
        raise ValueError("Karar CSV dosyası boş.")
    ids: set[str] = set()
    for decision in decisions:
        missing = DECISION_FIELDS - set(decision)
        if missing:
            raise ValueError(f"Karar satırında eksik alanlar: {sorted(missing)}")
        pair_id = decision["pair_id"].strip()
        if not pair_id or pair_id in ids:
            raise ValueError(f"Boş veya tekrarlanan Pair ID: {pair_id!r}")
        ids.add(pair_id)
        if decision["visual_verdict"] not in ALLOWED_VERDICTS:
            raise ValueError(f"Geçersiz karar ({pair_id}): {decision['visual_verdict']}")
        if decision["confidence"] not in ALLOWED_CONFIDENCE:
            raise ValueError(f"Geçersiz güven ({pair_id}): {decision['confidence']}")
        if not decision["reviewer"].strip() or not decision["review_date"].strip():
            raise ValueError(f"Reviewer/tarih boş bırakılamaz: {pair_id}")


def apply_decisions(
    review_rows: Sequence[Mapping[str, str]],
    decisions: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    validate_decisions(decisions)
    by_id = {decision["pair_id"]: decision for decision in decisions}
    known_ids = {row["pair_id"] for row in review_rows}
    missing = sorted(set(by_id) - known_ids)
    if missing:
        raise ValueError(f"Review tablosunda bulunmayan Pair ID: {missing}")

    output: list[dict[str, str]] = []
    applied: set[str] = set()
    for source_row in review_rows:
        row = dict(source_row)
        decision = by_id.get(row["pair_id"])
        if decision is None:
            output.append(row)
            continue
        if row["audit_partition"] != "development":
            raise ValueError(f"Holdout kararı uygulanamaz: {row['pair_id']}")
        existing = row.get("human_verdict", "").strip()
        if existing and existing != decision["visual_verdict"]:
            raise ValueError(
                f"Var olan karar farklı; otomatik değiştirilmedi: {row['pair_id']}"
            )
        row["human_verdict"] = decision["visual_verdict"]
        row["confidence"] = decision["confidence"]
        row["reviewer"] = decision["reviewer"]
        row["review_date"] = decision["review_date"]
        row["review_notes"] = decision["review_notes"]
        row["review_status"] = "reviewed_ai_assisted"
        applied.add(row["pair_id"])
        output.append(row)

    if applied != set(by_id):
        raise RuntimeError("Bütün karar satırları uygulanamadı.")
    return output


def write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    decisions_path = Path(args.decisions).resolve()
    review_path = Path(args.review_csv).resolve()
    _, decisions = read_csv(decisions_path)
    fields, review_rows = read_csv(review_path)
    missing_review_fields = {
        "human_verdict",
        "confidence",
        "reviewer",
        "review_date",
        "review_notes",
        "review_status",
    } - set(fields)
    if missing_review_fields:
        raise ValueError(
            f"Review tablosunda eksik alanlar: {sorted(missing_review_fields)}"
        )

    updated = apply_decisions(review_rows, decisions)
    backup = review_path.with_name("review_labels.before_development_review.csv")
    if not backup.exists():
        shutil.copy2(review_path, backup)
    write_csv_atomic(review_path, fields, updated)

    development = [row for row in updated if row["audit_partition"] == "development"]
    holdout = [row for row in updated if row["audit_partition"] == "holdout"]
    counts = Counter(row["human_verdict"] or "blank" for row in development)
    print("İşlem türü: development görsel kararlarını kaydetme (split/model yok)")
    print(f"Uygulanan karar: {len(decisions)}")
    print(f"Development dağılımı: {dict(sorted(counts.items()))}")
    print(
        "Holdout boş karar: "
        f"{sum(not row['human_verdict'].strip() for row in holdout)}/{len(holdout)}"
    )
    print(f"Güncellenen tablo: {review_path}")
    print(f"İlk durum yedeği: {backup}")


if __name__ == "__main__":
    main()
