from __future__ import annotations

"""Prepare a safe, human-editable review table from pHash candidate samples.

The script does not compare images, choose a pHash threshold, create a split, or
train a model. It copies candidate metadata into a review worksheet and refuses
to overwrite an existing worksheet unless ``--overwrite`` is explicitly used.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from data_index import project_root


INPUT_FIELDS = (
    "subclass",
    "distance_band",
    "distance",
    "known_exact_duplicate",
    "left_path",
    "right_path",
)

OUTPUT_FIELDS = (
    "pair_id",
    "subclass",
    "distance_band",
    "distance",
    "known_exact_duplicate",
    "left_path",
    "right_path",
    "review_status",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "notes",
)

ALLOWED_VISUAL_VERDICTS = (
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
)

ALLOWED_CONFIDENCE = ("high", "medium", "low")


def parse_args() -> argparse.Namespace:
    default_dir = project_root() / "artifacts" / "audit" / "phash_pilot"
    parser = argparse.ArgumentParser(
        description=(
            "pHash aday çiftlerini insan incelemesi için review_labels.csv tablosuna "
            "dönüştürür. Eşik seçmez, split oluşturmaz ve model eğitmez."
        )
    )
    parser.add_argument(
        "--candidates",
        default=str(default_dir / "candidate_samples.csv"),
        help="audit_perceptual_duplicates.py tarafından üretilen aday CSV dosyası.",
    )
    parser.add_argument(
        "--output",
        default=str(default_dir / "review_labels.csv"),
        help="Hazırlanacak insan inceleme tablosu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan inceleme tablosunu silip yeniden oluşturur. Normalde kullanılmamalıdır.",
    )
    return parser.parse_args()


def read_candidates(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Aday CSV bulunamadı: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(set(INPUT_FIELDS) - fieldnames)
        if missing:
            raise ValueError(f"Aday CSV içinde eksik sütunlar var: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("Aday CSV boş; inceleme tablosu hazırlanmadı.")
    return rows


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def prepare_review_rows(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        exact = is_true(candidate["known_exact_duplicate"])
        rows.append(
            {
                "pair_id": f"P{index:04d}",
                "subclass": candidate["subclass"],
                "distance_band": candidate["distance_band"],
                "distance": candidate["distance"],
                "known_exact_duplicate": "yes" if exact else "no",
                "left_path": candidate["left_path"],
                "right_path": candidate["right_path"],
                "review_status": "preconfirmed_sha256" if exact else "pending_visual_review",
                "visual_verdict": "exact_duplicate" if exact else "",
                "confidence": "certain" if exact else "",
                "reviewer": "SHA-256 audit" if exact else "",
                "review_date": "",
                "notes": (
                    "Byte-byte identical; visual review is optional." if exact else ""
                ),
            }
        )
    return rows


def write_review_table(path: Path, rows: List[Dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"İnceleme tablosu zaten var ve korunuyor: {path}\n"
            "Gerçekten sıfırlamak isterseniz --overwrite kullanın."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates).resolve()
    output_path = Path(args.output).resolve()

    try:
        candidates = read_candidates(candidates_path)
        review_rows = prepare_review_rows(candidates)
        write_review_table(output_path, review_rows, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"HATA: {error}") from None

    exact_count = sum(row["visual_verdict"] == "exact_duplicate" for row in review_rows)
    pending_count = len(review_rows) - exact_count
    print("İşlem türü: insan inceleme şablonu (eşik/split/model yok)")
    print(f"Kaynak aday sayısı: {len(candidates)}")
    print(f"SHA-256 ile önceden doğrulanan: {exact_count}")
    print(f"Görsel inceleme bekleyen: {pending_count}")
    print(f"İnceleme tablosu: {output_path}")
    print("Görsel karar seçenekleri:")
    for verdict in ALLOWED_VISUAL_VERDICTS:
        print(f"  - {verdict}")
    print(f"Güven seçenekleri: {', '.join(ALLOWED_CONFIDENCE)}")


if __name__ == "__main__":
    main()
