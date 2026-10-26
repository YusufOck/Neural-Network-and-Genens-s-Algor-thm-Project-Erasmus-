from __future__ import annotations

"""Evaluate frozen duplicate rule v1 on the previously untouched holdout set.

The script verifies the frozen rule and the exact development scorer implementation,
loads the already-finished score-blind visual reference decisions, scores every holdout
pair once, and reports the frozen rule without tuning thresholds.

This is a duplicate-audit evaluation. It does not train a cancer classifier, create
duplicate families, or make a train/validation/test split.
"""

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import cv2

from data_index import project_root
from duplicate_decision_rule import classify_pair, file_sha256, load_frozen_rule
from render_calibration_review import read_dataset_root, resolve_image_path
from score_duplicate_development import OUTPUT_FIELDS, score_pair


REFERENCE_POSITIVE = "same_source"
REFERENCE_NEGATIVE = "independent"
REFERENCE_UNCERTAIN = "uncertain"
VISUAL_TO_REFERENCE = {
    "same_source_candidate": REFERENCE_POSITIVE,
    "independent_lookalike": REFERENCE_NEGATIVE,
    "uncertain": REFERENCE_UNCERTAIN,
}
RULE_OUTPUT_FIELDS = (
    "decision",
    "decision_path",
    "decision_reason",
    "flag_sha256_exact",
    "flag_strong_geometry",
    "flag_strong_intensity_structure",
    "flag_conservative_independent",
)
HOLDOUT_OUTPUT_FIELDS = OUTPUT_FIELDS + RULE_OUTPUT_FIELDS


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmuş duplicate rule v1'i holdout üzerinde yalnız bir kez ölçer; "
            "eşik ayarlamaz, family/split/model oluşturmaz."
        )
    )
    parser.add_argument("--calibration-root", default=str(calibration_root))
    parser.add_argument("--pairs-csv", default=str(calibration_root / "pairs.csv"))
    parser.add_argument(
        "--visual-decisions",
        default=str(root / "research" / "holdout_visual_review_v1.csv"),
    )
    parser.add_argument(
        "--rule", default=str(root / "config" / "duplicate_rule_v1.json")
    )
    parser.add_argument(
        "--scorer", default=str(root / "src" / "score_duplicate_development.py")
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "artifacts" / "audit" / "duplicate_holdout_evaluation_v1"),
    )
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def validate_visual_decisions(
    decisions: Sequence[Mapping[str, str]], expected_ids: set[str]
) -> dict[str, Mapping[str, str]]:
    required = {
        "pair_id",
        "visual_verdict",
        "confidence",
        "reviewer",
        "review_date",
        "review_notes",
    }
    by_id: dict[str, Mapping[str, str]] = {}
    for row in decisions:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Görsel karar satırında eksik alanlar: {sorted(missing)}")
        pair_id = row["pair_id"].strip()
        if not pair_id or pair_id in by_id:
            raise ValueError(f"Boş veya tekrarlanan Pair ID: {pair_id!r}")
        if row["visual_verdict"].strip() not in VISUAL_TO_REFERENCE:
            raise ValueError(f"Geçersiz görsel karar: {pair_id}")
        if row["confidence"].strip() not in {"high", "medium", "low"}:
            raise ValueError(f"Geçersiz güven değeri: {pair_id}")
        if not row["reviewer"].strip() or not row["review_date"].strip():
            raise ValueError(f"Reviewer/tarih boş: {pair_id}")
        by_id[pair_id] = row

    supplied = set(by_id)
    if supplied != expected_ids:
        missing = sorted(expected_ids - supplied)
        extra = sorted(supplied - expected_ids)
        raise ValueError(
            f"Holdout görsel karar kapsamı tam değil; eksik={missing}, fazla={extra}"
        )
    return by_id


def load_holdout_rows(
    pairs_csv: Path, decisions_csv: Path
) -> tuple[list[dict[str, str]], dict[str, object]]:
    all_pairs = read_csv(pairs_csv)
    holdout = [row for row in all_pairs if row.get("audit_partition", "").strip() == "holdout"]
    development_count = sum(
        row.get("audit_partition", "").strip() == "development" for row in all_pairs
    )
    unknown_partitions = sorted(
        {
            row.get("audit_partition", "").strip()
            for row in all_pairs
            if row.get("audit_partition", "").strip() not in {"development", "holdout"}
        }
    )
    if unknown_partitions:
        raise ValueError(f"Bilinmeyen audit partition: {unknown_partitions}")
    if not holdout:
        raise ValueError("Holdout satırı bulunamadı.")
    ids = [row["pair_id"] for row in holdout]
    if len(ids) != len(set(ids)):
        raise ValueError("Holdout Pair ID değerleri benzersiz değil.")

    review_ids = {
        row["pair_id"]
        for row in holdout
        if row.get("truth_status", "").strip() == "review_required"
    }
    decision_by_id = validate_visual_decisions(read_csv(decisions_csv), review_ids)

    output: list[dict[str, str]] = []
    for source in holdout:
        row = dict(source)
        truth_status = row.get("truth_status", "").strip()
        if truth_status == "same_source":
            row["reference_label"] = REFERENCE_POSITIVE
            row["reference_basis"] = row.get("truth_basis", "known_positive")
        elif truth_status == "review_required":
            decision = decision_by_id[row["pair_id"]]
            verdict = decision["visual_verdict"].strip()
            row["human_verdict"] = verdict
            row["reference_label"] = VISUAL_TO_REFERENCE[verdict]
            row["reference_basis"] = "ai_assisted_blind_visual_review"
        else:
            raise ValueError(
                f"Holdout satırında desteklenmeyen truth_status: {row['pair_id']}={truth_status!r}"
            )
        output.append(row)

    audit: dict[str, object] = {
        "all_pair_rows": len(all_pairs),
        "development_rows_not_loaded": development_count,
        "holdout_rows": len(output),
        "holdout_known_positive_rows": sum(
            row["truth_status"] == "same_source" for row in output
        ),
        "holdout_blind_visual_review_rows": len(review_ids),
        "reference_counts": dict(
            sorted(Counter(row["reference_label"] for row in output).items())
        ),
    }
    return output, audit


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_predictions(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    usable = [
        row
        for row in rows
        if row["reference_label"] in {REFERENCE_POSITIVE, REFERENCE_NEGATIVE}
    ]
    matrix = {
        reference: {
            decision: sum(
                row["reference_label"] == reference and row["decision"] == decision
                for row in usable
            )
            for decision in ("same_source", "independent", "uncertain")
        }
        for reference in (REFERENCE_POSITIVE, REFERENCE_NEGATIVE)
    }
    positives = sum(row["reference_label"] == REFERENCE_POSITIVE for row in usable)
    negatives = sum(row["reference_label"] == REFERENCE_NEGATIVE for row in usable)
    decided = sum(row["decision"] != "uncertain" for row in usable)
    correct = sum(row["decision"] == row["reference_label"] for row in usable)
    decided_correct = sum(
        row["decision"] != "uncertain" and row["decision"] == row["reference_label"]
        for row in usable
    )
    predicted_positive = sum(row["decision"] == "same_source" for row in usable)
    predicted_negative = sum(row["decision"] == "independent" for row in usable)
    tp = matrix[REFERENCE_POSITIVE]["same_source"]
    tn = matrix[REFERENCE_NEGATIVE]["independent"]
    return {
        "evaluated_reference_rows": len(usable),
        "reference_uncertain_rows_excluded": len(rows) - len(usable),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "confusion_with_uncertain": matrix,
        "coverage": safe_rate(decided, len(usable)),
        "overall_accuracy_counting_uncertain_as_not_correct": safe_rate(correct, len(usable)),
        "accuracy_on_decided_pairs": safe_rate(decided_correct, decided),
        "same_source_recall": safe_rate(tp, positives),
        "same_source_precision": safe_rate(tp, predicted_positive),
        "independent_recall": safe_rate(tn, negatives),
        "independent_precision": safe_rate(tn, predicted_negative),
    }


def grouped_metrics(
    rows: Sequence[Mapping[str, str]], field: str
) -> dict[str, dict[str, object]]:
    return {
        value: evaluate_predictions([row for row in rows if row.get(field, "") == value])
        for value in sorted({row.get(field, "") for row in rows})
    }


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=HOLDOUT_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def scoring_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        image_size=256,
        orb_features=4000,
        ratio_test=0.78,
        ransac_threshold=3.0,
        ecc_iterations=100,
        seed=42,
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    calibration_root = Path(args.calibration_root).resolve()
    pairs_csv = Path(args.pairs_csv).resolve()
    decisions_csv = Path(args.visual_decisions).resolve()
    rule_path = Path(args.rule).resolve()
    scorer_path = Path(args.scorer).resolve()
    output_dir = Path(args.output_dir).resolve()
    score_path = output_dir / "pair_scores_and_decisions.csv"
    report_path = output_dir / "report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(
            "Holdout çıktısı zaten var; bu değerlendirme yeniden çalıştırılmadı: "
            f"{output_dir}"
        )

    rule, rule_hash = load_frozen_rule(rule_path)
    expected_scorer_hash = str(
        rule.get("development_evidence", {}).get("scorer_sha256", "")  # type: ignore[union-attr]
    )
    actual_scorer_hash = file_sha256(scorer_path)
    if not expected_scorer_hash or actual_scorer_hash != expected_scorer_hash:
        raise ValueError(
            "Holdout scorer development ile aynı değil; görüntüler açılmadı. "
            f"expected={expected_scorer_hash}, actual={actual_scorer_hash}"
        )

    rows, input_audit = load_holdout_rows(pairs_csv, decisions_csv)
    dataset_root = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else read_dataset_root(calibration_root)
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Veri kökü bulunamadı: {dataset_root}")

    scorer_args = scoring_namespace()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(scorer_args.seed)
    print("İşlem türü: dondurulmuş duplicate rule v1 HOLDOUT değerlendirmesi")
    print(f"Holdout çiftleri: {len(rows)}")
    print(f"Görsel referans kararları: {input_audit['holdout_blind_visual_review_rows']}")
    print(f"Kural SHA-256: {rule_hash}")
    print(f"Scorer SHA-256 doğrulandı: {actual_scorer_hash}")
    print("Eşikler değiştirilmeyecek; model/family/split oluşturulmayacak.")

    started = time.perf_counter()
    scored: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        left_path = resolve_image_path(row, "left", dataset_root, calibration_root)
        right_path = resolve_image_path(row, "right", dataset_root, calibration_root)
        result = score_pair(row, left_path, right_path, scorer_args)
        result.update(classify_pair(result, rule))
        scored.append(result)
        if index % 25 == 0 or index == len(rows):
            print(f"Holdout skorlandı: {index}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    metrics = evaluate_predictions(scored)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "holdout_evaluated_once_with_frozen_rule_v1",
        "experiment_type": "near_duplicate_holdout_evaluation",
        "input_audit": input_audit,
        "integrity": {
            "frozen_rule_sha256": rule_hash,
            "development_scorer_expected_sha256": expected_scorer_hash,
            "holdout_scorer_actual_sha256": actual_scorer_hash,
            "visual_decisions_sha256": file_sha256(decisions_csv),
            "pairs_csv_sha256": file_sha256(pairs_csv),
        },
        "parameters": {
            "image_size": scorer_args.image_size,
            "orb_features": scorer_args.orb_features,
            "ratio_test": scorer_args.ratio_test,
            "ransac_threshold": scorer_args.ransac_threshold,
            "ecc_iterations": scorer_args.ecc_iterations,
            "seed": scorer_args.seed,
        },
        "overall_metrics": metrics,
        "by_modality": grouped_metrics(scored, "modality_key"),
        "by_pair_kind": grouped_metrics(scored, "pair_kind"),
        "elapsed_seconds": elapsed,
        "limitations": [
            "Holdout görsel referansları AI-assisted kör incelemedir; bağımsız uzman gold standard değildir.",
            "Kural aynı kaynak görüntü dönüşümünü arar; hasta kimliği veya klinik eşdeğerlik kanıtlamaz.",
            "Bu sonuç bir kanser sınıflandırma doğruluğu değildir.",
            "Holdout sonucuna bakılarak eşikler bu deney içinde değiştirilmemiştir.",
        ],
    }
    atomic_write_csv(score_path, scored)
    atomic_write_json(report_path, report)
    print(f"Skor + karar tablosu: {score_path}")
    print(f"Holdout raporu: {report_path}")
    print("NOT: Kural bu sonuçlara bakılarak yeniden ayarlanmadı.")
    return score_path, report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
