from __future__ import annotations

"""Compare frozen duplicate rule v1 with natural audit visual references.

The visual decisions were frozen before their private candidate/outside strata and
scores were opened. This script keeps duplicate rule v1 unchanged, extracts the same
ORB/RANSAC, phase/ECC, SSIM, NMI and correlation evidence used in development, and
reports agreement. It does not create families, data splits, or cancer models.
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

from analyze_natural_duplicate_audit_v1 import (
    EXPECTED_PRIVATE_SHA256,
    EXPECTED_ROWS,
    EXPECTED_VISUAL_SHA256,
    file_sha256,
    merge_rows,
    read_csv,
)
from data_index import project_root
from duplicate_decision_rule import classify_pair, load_frozen_rule
from evaluate_duplicate_holdout import evaluate_predictions, grouped_metrics
from render_calibration_review import read_dataset_root
from score_duplicate_development import OUTPUT_FIELDS, score_pair


VISUAL_TO_REFERENCE = {
    "same_source_candidate": "same_source",
    "independent_lookalike": "independent",
    "uncertain": "uncertain",
}
SOURCE_GROUP_TO_MODALITY = {
    "all_blood_smear": "blood_smear",
    "brain_ce_mri": "mri",
    "breast_histology": "histopathology",
    "cervix_pap_smear": "pap_smear",
    "kidney_ct": "ct",
    "lung_colon_histology": "histopathology",
    "lymphoma_histology": "histopathology",
    "oral_histology": "histopathology",
}
AUDIT_FIELDS = (
    "selection_stratum",
    "audit_region",
    "robust_top15",
    "robust_best_union_rank",
    "robust_best_cosine_similarity",
)
RULE_FIELDS = (
    "decision",
    "decision_path",
    "decision_reason",
    "flag_sha256_exact",
    "flag_strong_geometry",
    "flag_strong_intensity_structure",
    "flag_conservative_independent",
)
OUTPUT_WITH_AUDIT_FIELDS = OUTPUT_FIELDS + AUDIT_FIELDS + RULE_FIELDS


def parse_args() -> argparse.Namespace:
    root = project_root()
    audit_dir = root / "research" / "natural_duplicate_audit_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Natural duplicate audit v1 çiftlerini frozen duplicate rule v1 ile "
            "skorlar; eşik/family/split/model üretmez."
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
        "--rule", default=str(root / "config" / "duplicate_rule_v1.json")
    )
    parser.add_argument(
        "--scorer", default=str(root / "src" / "score_duplicate_development.py")
    )
    parser.add_argument(
        "--calibration-root",
        default=str(root / "artifacts" / "audit" / "duplicate_calibration_v1"),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(
            root / "artifacts" / "audit" / "natural_duplicate_rule_evaluation_v1"
        ),
    )
    return parser.parse_args()


def prepare_rows(
    private_rows: Sequence[Mapping[str, str]],
    visual_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    merged = merge_rows(private_rows, visual_rows)
    output: list[dict[str, str]] = []
    for source in merged:
        source_group = source["source_group"]
        if source_group not in SOURCE_GROUP_TO_MODALITY:
            raise ValueError(f"Bilinmeyen source group: {source_group}")
        row = dict(source)
        row.update(
            {
                "audit_partition": "natural_audit_v1",
                "pair_kind": "natural_dataset_pair",
                "truth_status": "score_blind_visual_review",
                "human_verdict": source["visual_verdict"],
                "reference_label": VISUAL_TO_REFERENCE[source["visual_verdict"]],
                "reference_basis": "ai_assisted_score_blind_visual_review",
                "modality_key": SOURCE_GROUP_TO_MODALITY[source_group],
                "transform_name": "",
            }
        )
        output.append(row)
    return output


def scoring_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        image_size=256,
        orb_features=4000,
        ratio_test=0.78,
        ransac_threshold=3.0,
        ecc_iterations=100,
        seed=42,
    )


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_WITH_AUDIT_FIELDS)
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


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    private_path = Path(args.private_selection).resolve()
    visual_path = Path(args.visual_decisions).resolve()
    rule_path = Path(args.rule).resolve()
    scorer_path = Path(args.scorer).resolve()
    calibration_root = Path(args.calibration_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    score_path = output_dir / "pair_scores_and_decisions.csv"
    report_path = output_dir / "report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(
            "Natural audit frozen-rule çıktısı zaten var; üzerine yazılmadı: "
            f"{output_dir}"
        )

    if file_sha256(private_path) != EXPECTED_PRIVATE_SHA256:
        raise ValueError("Private selection hash değişmiş; görüntüler açılmadı.")
    if file_sha256(visual_path) != EXPECTED_VISUAL_SHA256:
        raise ValueError("Visual decision hash değişmiş; görüntüler açılmadı.")
    rule, rule_hash = load_frozen_rule(rule_path)
    expected_scorer_hash = str(
        rule.get("development_evidence", {}).get("scorer_sha256", "")  # type: ignore[union-attr]
    )
    actual_scorer_hash = file_sha256(scorer_path)
    if not expected_scorer_hash or actual_scorer_hash != expected_scorer_hash:
        raise ValueError(
            "Scorer frozen development sürümüyle aynı değil; görüntüler açılmadı. "
            f"expected={expected_scorer_hash}, actual={actual_scorer_hash}"
        )

    rows = prepare_rows(read_csv(private_path), read_csv(visual_path))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Beklenen {EXPECTED_ROWS} çift, bulunan {len(rows)}")
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
    print("İşlem türü: natural duplicate audit v1 / frozen duplicate rule karşılaştırması")
    print(f"Çift sayısı: {len(rows)}")
    print(f"Kural SHA-256: {rule_hash}")
    print(f"Scorer SHA-256 doğrulandı: {actual_scorer_hash}")
    print("CPU/OpenCV kullanılacak; eşik, family, split veya model değişmeyecek.")

    hash_cache: dict[Path, str] = {}
    scored: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        left_path = (dataset_root / row["left_path"]).resolve()
        right_path = (dataset_root / row["right_path"]).resolve()
        if not left_path.is_relative_to(dataset_root) or not right_path.is_relative_to(
            dataset_root
        ):
            raise ValueError(f"Dataset kökü dışına çıkan yol: {row['pair_id']}")
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {row['pair_id']}")
        for path in (left_path, right_path):
            if path not in hash_cache:
                hash_cache[path] = file_sha256(path)
        row["sha256_equal"] = (
            "yes" if hash_cache[left_path] == hash_cache[right_path] else "no"
        )
        result = score_pair(row, left_path, right_path, scorer_args)
        for field in AUDIT_FIELDS:
            result[field] = row.get(field, "")
        result.update(classify_pair(result, rule))
        scored.append(result)
        if index % 25 == 0 or index == len(rows):
            print(f"Natural audit skorlandı: {index}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "natural_audit_evaluated_with_frozen_rule_v1",
        "experiment_type": "near_duplicate_natural_challenge_set_evaluation",
        "integrity": {
            "private_selection_sha256": file_sha256(private_path),
            "visual_decisions_sha256": file_sha256(visual_path),
            "frozen_rule_sha256": rule_hash,
            "development_scorer_expected_sha256": expected_scorer_hash,
            "natural_audit_scorer_actual_sha256": actual_scorer_hash,
        },
        "input_counts": {
            "pairs": len(rows),
            "reference_counts": dict(
                sorted(Counter(row["reference_label"] for row in rows).items())
            ),
            "unique_images_hashed": len(hash_cache),
        },
        "parameters": vars(scorer_args),
        "overall_metrics": evaluate_predictions(scored),
        "by_selection_stratum": grouped_metrics(scored, "selection_stratum"),
        "by_audit_region": grouped_metrics(scored, "audit_region"),
        "by_source_group": grouped_metrics(scored, "source_group"),
        "by_modality": grouped_metrics(scored, "modality_key"),
        "elapsed_seconds": elapsed,
        "limitations": [
            "Görsel referanslar AI-destekli score-blind incelemedir; klinik uzman gold standard değildir.",
            "Challenge-set örneklemi bütün 3.093.466 olası doğal çifti kapsamaz.",
            "Kural aynı görüntü kaynağı benzerliğini arar; hasta kimliği veya klinik tanı vermez.",
            "Bu bir kanser sınıflandırma doğruluğu değildir.",
            "Frozen rule eşikleri bu sonuçlara bakılarak değiştirilmemiştir.",
        ],
    }
    atomic_write_csv(score_path, scored)
    atomic_write_json(report_path, report)
    print(f"Skor + karar tablosu: {score_path}")
    print(f"Rapor: {report_path}")
    print(f"Süre: {elapsed:.1f} saniye")
    print("NOT: Bu değerlendirme family/split/model oluşturmadı.")
    return score_path, report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
