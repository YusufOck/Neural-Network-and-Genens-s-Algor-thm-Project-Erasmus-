from __future__ import annotations

"""Unblind and summarize natural duplicate audit v1.

This script joins the frozen, score-blind visual decisions to the private selection
strata. It does not open images, tune thresholds, create duplicate families, split the
dataset, or train a model.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from data_index import project_root


EXPECTED_PRIVATE_SHA256 = (
    "dc01803890352140c9e3465b86d144082850ebff735b73adc4f40a1ea97b86f0"
)
EXPECTED_VISUAL_SHA256 = (
    "91d857685ada7dbd6d3fb6e3add5a8fe914e3767389f1d37401a660ffd308bca"
)
EXPECTED_ROWS = 345

ALLOWED_STRATA = {
    "candidate_robust_only",
    "candidate_phash_supported",
    "outside_robust_rank16_30",
    "outside_phash_distance14_16",
    "outside_random",
}
ALLOWED_VERDICTS = {
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
}
OUTPUT_FIELDS = (
    "pair_id",
    "selection_stratum",
    "audit_region",
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
    "phash_distance",
    "phash_radius12",
    "phash_radius16",
    "robust_top15",
    "robust_top30",
    "robust_best_union_rank",
    "robust_best_cosine_similarity",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    audit_dir = root / "research" / "natural_duplicate_audit_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmuş doğal duplicate görsel kararlarını private seçim "
            "strata'sıyla birleştirir; eşik ayarlamaz veya split/model üretmez."
        )
    )
    parser.add_argument(
        "--private-selection", default=str(audit_dir / "selection_private.csv")
    )
    parser.add_argument(
        "--visual-decisions",
        default=str(root / "research" / "natural_duplicate_visual_review_v1.csv"),
    )
    parser.add_argument(
        "--output-csv", default=str(audit_dir / "unblinded_pair_results.csv")
    )
    parser.add_argument(
        "--output-json", default=str(audit_dir / "unblinded_report.json")
    )
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı bulunamadı: {path}")
        return [dict(row) for row in reader]


def index_unique(
    rows: Sequence[Mapping[str, str]], name: str
) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{name}: boş Pair ID")
        if pair_id in indexed:
            raise ValueError(f"{name}: tekrarlanan Pair ID {pair_id}")
        indexed[pair_id] = row
    return indexed


def merge_rows(
    private_rows: Sequence[Mapping[str, str]],
    visual_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    private_by_id = index_unique(private_rows, "private selection")
    visual_by_id = index_unique(visual_rows, "visual decisions")
    if set(private_by_id) != set(visual_by_id):
        missing = sorted(set(private_by_id) - set(visual_by_id))
        extra = sorted(set(visual_by_id) - set(private_by_id))
        raise ValueError(
            f"Pair ID kapsamı eşleşmiyor; visual eksik={missing}, visual fazla={extra}"
        )

    merged: list[dict[str, str]] = []
    for pair_id in sorted(private_by_id):
        private = private_by_id[pair_id]
        visual = visual_by_id[pair_id]
        stratum = private.get("selection_stratum", "").strip()
        verdict = visual.get("visual_verdict", "").strip()
        if stratum not in ALLOWED_STRATA:
            raise ValueError(f"{pair_id}: geçersiz stratum {stratum!r}")
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"{pair_id}: geçersiz görsel karar {verdict!r}")
        audit_region = "candidate" if stratum.startswith("candidate_") else "outside"
        row = {
            "pair_id": pair_id,
            "selection_stratum": stratum,
            "audit_region": audit_region,
            "source_group": private.get("source_group", ""),
            "left_leaf": private.get("left_leaf", ""),
            "right_leaf": private.get("right_leaf", ""),
            "left_path": private.get("left_path", ""),
            "right_path": private.get("right_path", ""),
            "visual_verdict": verdict,
            "confidence": visual.get("confidence", ""),
            "reviewer": visual.get("reviewer", ""),
            "review_date": visual.get("review_date", ""),
            "review_notes": visual.get("review_notes", ""),
            "phash_distance": private.get("phash_distance", ""),
            "phash_radius12": private.get("phash_radius12", ""),
            "phash_radius16": private.get("phash_radius16", ""),
            "robust_top15": private.get("robust_top15", ""),
            "robust_top30": private.get("robust_top30", ""),
            "robust_best_union_rank": private.get("robust_best_union_rank", ""),
            "robust_best_cosine_similarity": private.get(
                "robust_best_cosine_similarity", ""
            ),
        }
        merged.append(row)
    return merged


def nested_counts(
    rows: Sequence[Mapping[str, str]], group_field: str
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row[group_field]][row["visual_verdict"]] += 1
    return {
        group: {
            verdict: counts.get(verdict, 0)
            for verdict in sorted(ALLOWED_VERDICTS)
        }
        for group, counts in sorted(grouped.items())
    }


def summarize(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    verdict_counts = Counter(row["visual_verdict"] for row in rows)
    region_counts = Counter(row["audit_region"] for row in rows)
    candidate = [row for row in rows if row["audit_region"] == "candidate"]
    outside = [row for row in rows if row["audit_region"] == "outside"]
    outside_positive_ids = [
        row["pair_id"]
        for row in outside
        if row["visual_verdict"] == "same_source_candidate"
    ]
    uncertain_ids = [
        row["pair_id"] for row in rows if row["visual_verdict"] == "uncertain"
    ]
    candidate_positive = sum(
        row["visual_verdict"] == "same_source_candidate" for row in candidate
    )
    decided_candidate = sum(
        row["visual_verdict"] != "uncertain" for row in candidate
    )
    return {
        "total_rows": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "audit_region_counts": dict(sorted(region_counts.items())),
        "counts_by_stratum": nested_counts(rows, "selection_stratum"),
        "counts_by_source_group": nested_counts(rows, "source_group"),
        "candidate_positive_count": candidate_positive,
        "candidate_positive_yield_all": (
            candidate_positive / len(candidate) if candidate else None
        ),
        "candidate_positive_yield_decided": (
            candidate_positive / decided_candidate if decided_candidate else None
        ),
        "outside_confirmed_positive_count": len(outside_positive_ids),
        "outside_confirmed_positive_ids": outside_positive_ids,
        "uncertain_pair_ids": uncertain_ids,
        "challenge_set_gate_passed": not outside_positive_ids,
        "gate_interpretation": (
            "İncelenen outside challenge set içinde doğrulanmış aynı-kaynak çift yok. "
            "Bu sonuç 3.093.466 olası çiftin tamamında yüzde 100 recall kanıtı değildir."
        ),
    }


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def main() -> int:
    args = parse_args()
    root = project_root().resolve()
    private_path = Path(args.private_selection).resolve()
    visual_path = Path(args.visual_decisions).resolve()
    private_hash = file_sha256(private_path)
    visual_hash = file_sha256(visual_path)
    if not args.allow_hash_mismatch:
        if private_hash != EXPECTED_PRIVATE_SHA256:
            raise ValueError(
                f"Private selection hash değişmiş: {private_hash}"
            )
        if visual_hash != EXPECTED_VISUAL_SHA256:
            raise ValueError(f"Visual decision hash değişmiş: {visual_hash}")

    merged = merge_rows(read_csv(private_path), read_csv(visual_path))
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(f"Beklenen {EXPECTED_ROWS} satır, bulunan {len(merged)}")
    summary = summarize(merged)
    payload: dict[str, object] = {
        "audit_name": "natural_duplicate_audit_v1",
        "analysis_type": "frozen_score_blind_visual_decisions_unblinded_against_selection_strata",
        "inputs": {
            "private_selection": private_path.relative_to(root).as_posix(),
            "private_selection_sha256": private_hash,
            "visual_decisions": visual_path.relative_to(root).as_posix(),
            "visual_decisions_sha256": visual_hash,
        },
        "summary": summary,
    }
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    atomic_write_csv(output_csv, merged)
    atomic_write_json(output_json, payload)

    print("Doğal duplicate audit v1: score-blind kararlar private strata ile açıldı")
    print(f"Toplam çift: {summary['total_rows']}")
    print(f"Kararlar: {summary['verdict_counts']}")
    print(f"Bölgeler: {summary['audit_region_counts']}")
    print(
        "Outside doğrulanmış aynı-kaynak: "
        f"{summary['outside_confirmed_positive_count']}"
    )
    print(f"Challenge-set kapısı geçti: {summary['challenge_set_gate_passed']}")
    print(f"Birleştirilmiş CSV: {output_csv}")
    print(f"JSON rapor: {output_json}")
    print("NOT: Bu sonuç tüm olası çiftlerde yüzde 100 recall kanıtlamaz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
