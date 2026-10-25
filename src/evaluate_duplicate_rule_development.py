from __future__ import annotations

"""Evaluate the frozen duplicate rule on development scores only.

This script reads numeric evidence only. It does not open dataset images, inspect
holdout rows, create duplicate families, or create classifier splits.
"""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from duplicate_decision_rule import classify_pair, file_sha256, load_frozen_rule


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = (
    ROOT / "artifacts" / "audit" / "duplicate_hybrid_development_v1" / "pair_scores.csv"
)
DEFAULT_RULE = ROOT / "config" / "duplicate_rule_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "audit" / "duplicate_rule_v1_development"

OUTPUT_FIELDS = (
    "pair_id",
    "audit_partition",
    "reference_label",
    "human_verdict",
    "modality_key",
    "source_group",
    "sha256_equal",
    "phash_distance",
    "orb_inliers",
    "orb_inlier_ratio",
    "orb_coverage_min",
    "best_ssim",
    "best_nmi",
    "best_correlation",
    "decision",
    "decision_path",
    "decision_reason",
    "flag_sha256_exact",
    "flag_strong_geometry",
    "flag_strong_intensity_structure",
    "flag_conservative_independent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--rule-json", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_development_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Score tablosu bos")
    partitions = {row.get("audit_partition", "") for row in rows}
    if partitions != {"development"}:
        raise ValueError(
            "Bu komut yalniz development skorlarini kabul eder; bulunan bolumler: "
            f"{sorted(partitions)}"
        )
    pair_ids = [row.get("pair_id", "") for row in rows]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Bos veya tekrar eden pair_id bulundu")
    return rows


def confusion(rows: Sequence[Mapping[str, str]]) -> dict[str, float | int]:
    tp = fp = tn = fn = uncertain_positive = uncertain_negative = 0
    for row in rows:
        reference = row["reference_label"]
        decision = row["decision"]
        if reference == "same_source":
            if decision == "same_source":
                tp += 1
            elif decision == "independent":
                fn += 1
            else:
                uncertain_positive += 1
        elif reference == "independent":
            if decision == "independent":
                tn += 1
            elif decision == "same_source":
                fp += 1
            else:
                uncertain_negative += 1
    decided = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn + uncertain_positive) if tp + fn + uncertain_positive else 0.0
    specificity = tn / (tn + fp + uncertain_negative) if tn + fp + uncertain_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    coverage = decided / (decided + uncertain_positive + uncertain_negative)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "uncertain_positive": uncertain_positive,
        "uncertain_negative": uncertain_negative,
        "precision_same_source": precision,
        "recall_same_source": recall,
        "specificity_independent": specificity,
        "f1_same_source": f1,
        "automatic_decision_coverage": coverage,
    }


def atomic_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    scores_path = args.scores_csv.resolve()
    rule_path = args.rule_json.resolve()
    output_dir = args.output_dir.resolve()
    decisions_path = output_dir / "decisions.csv"
    report_path = output_dir / "report.json"
    if not args.overwrite and (decisions_path.exists() or report_path.exists()):
        raise FileExistsError("Cikti zaten var; bilincli yenileme icin --overwrite kullanin")

    rule, rule_hash = load_frozen_rule(rule_path)
    score_hash = file_sha256(scores_path)
    evidence = rule.get("development_evidence")
    if not isinstance(evidence, dict) or score_hash != evidence.get("pair_scores_sha256"):
        raise ValueError("Development score dosyasi dondurulan kanitla ayni degil")

    input_rows = read_development_scores(scores_path)
    output_rows: list[dict[str, str]] = []
    for row in input_rows:
        result = classify_pair(row, rule)
        output_rows.append({field: row.get(field, result.get(field, "")) for field in OUTPUT_FIELDS})

    by_modality: dict[str, object] = {}
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output_rows:
        grouped[row["modality_key"]].append(row)
    for modality, rows in sorted(grouped.items()):
        by_modality[modality] = {
            "count": len(rows),
            "by_reference": dict(sorted(Counter(row["reference_label"] for row in rows).items())),
            "by_decision": dict(sorted(Counter(row["decision"] for row in rows).items())),
            "metrics": confusion(rows),
        }

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "frozen_rule_development_evaluation_only",
        "experiment_type": "near_duplicate_rule_calibration_pilot",
        "rule_id": rule["rule_id"],
        "rule_sha256": rule_hash,
        "score_input_sha256": score_hash,
        "holdout_policy": "no holdout row or image was read by this command",
        "pair_count": len(output_rows),
        "by_reference": dict(sorted(Counter(row["reference_label"] for row in output_rows).items())),
        "by_decision": dict(sorted(Counter(row["decision"] for row in output_rows).items())),
        "by_decision_path": dict(sorted(Counter(row["decision_path"] for row in output_rows).items())),
        "metrics_excluding_reference_uncertain": confusion(output_rows),
        "by_modality": by_modality,
        "limitations": [
            "Metrics are calculated on the same development evidence used to choose thresholds.",
            "This is not independent holdout performance and not classifier accuracy.",
            "The C000620 uncertain reference is not treated as positive or negative.",
            "No duplicate family or train/validation/test split was created.",
        ],
    }
    atomic_csv(decisions_path, output_rows)
    atomic_json(report_path, report)
    print("Islem: frozen duplicate rule DEVELOPMENT degerlendirmesi")
    print(f"Kural SHA-256: {report['rule_sha256']}")
    print(f"Development cifti: {len(output_rows)}")
    print("Holdout: okunmadi")
    print(f"Kararlar: {report['by_decision']}")
    print(f"Rapor: {report_path}")
    return decisions_path, report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
