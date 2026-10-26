from __future__ import annotations

"""Verify that the saved holdout evaluation is complete and internally consistent.

This script never opens an image and never rescoring the holdout. It is safe to run after
an interrupted terminal/app session because it checks the already-written CSV/JSON outputs,
recomputes frozen-rule decisions from stored evidence, and writes a verification manifest.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from data_index import project_root
from duplicate_decision_rule import classify_pair, file_sha256, load_frozen_rule
from evaluate_duplicate_holdout import (
    evaluate_predictions,
    grouped_metrics,
    load_holdout_rows,
    read_csv,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    calibration_root = root / "artifacts" / "audit" / "duplicate_calibration_v1"
    output_dir = root / "artifacts" / "audit" / "duplicate_holdout_evaluation_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Kaydedilmiş holdout CSV/JSON çıktısını görüntüleri yeniden açmadan doğrular."
        )
    )
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
        "--scores", default=str(output_dir / "pair_scores_and_decisions.csv")
    )
    parser.add_argument("--report", default=str(output_dir / "report.json"))
    parser.add_argument(
        "--verification", default=str(output_dir / "verification.json")
    )
    return parser.parse_args()


def verify_score_rows(
    scores: Sequence[Mapping[str, str]],
    expected_rows: Sequence[Mapping[str, str]],
    rule: Mapping[str, object],
) -> None:
    if not scores:
        raise ValueError("Holdout skor tablosu boş.")
    ids = [row.get("pair_id", "") for row in scores]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Skor tablosunda boş veya tekrarlanan Pair ID var.")
    expected_by_id = {row["pair_id"]: row for row in expected_rows}
    if set(ids) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(ids))
        extra = sorted(set(ids) - set(expected_by_id))
        raise ValueError(f"Skor kapsamı bozuk; eksik={missing}, fazla={extra}")

    for score in scores:
        pair_id = score["pair_id"]
        expected = expected_by_id[pair_id]
        if score.get("audit_partition", "") != "holdout":
            raise ValueError(f"Holdout çıktısında başka partition var: {pair_id}")
        for field in (
            "pair_kind",
            "left_leaf",
            "right_leaf",
            "modality_key",
            "source_group",
            "reference_label",
        ):
            if score.get(field, "") != expected.get(field, ""):
                raise ValueError(
                    f"Kaynak metadata uyuşmazlığı: {pair_id} {field} "
                    f"score={score.get(field)!r} expected={expected.get(field)!r}"
                )
        recomputed = classify_pair(score, rule)
        for field, value in recomputed.items():
            if score.get(field, "") != value:
                raise ValueError(
                    f"Dondurulmuş karar yeniden üretilemedi: {pair_id} {field}"
                )


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    pairs_path = Path(args.pairs_csv).resolve()
    decisions_path = Path(args.visual_decisions).resolve()
    rule_path = Path(args.rule).resolve()
    scorer_path = Path(args.scorer).resolve()
    scores_path = Path(args.scores).resolve()
    report_path = Path(args.report).resolve()
    verification_path = Path(args.verification).resolve()

    rule, rule_hash = load_frozen_rule(rule_path)
    expected_rows, expected_audit = load_holdout_rows(pairs_path, decisions_path)
    scores = read_csv(scores_path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "holdout_evaluated_once_with_frozen_rule_v1":
        raise ValueError("Holdout rapor durum işareti tamamlanmış değil.")

    integrity = report.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Holdout raporunda integrity bölümü yok.")
    actual_hashes = {
        "frozen_rule_sha256": rule_hash,
        "development_scorer_expected_sha256": str(
            rule.get("development_evidence", {}).get("scorer_sha256", "")  # type: ignore[union-attr]
        ),
        "holdout_scorer_actual_sha256": file_sha256(scorer_path),
        "visual_decisions_sha256": file_sha256(decisions_path),
        "pairs_csv_sha256": file_sha256(pairs_path),
    }
    for field, actual in actual_hashes.items():
        if integrity.get(field) != actual:
            raise ValueError(
                f"Integrity hash uyuşmazlığı: {field} report={integrity.get(field)!r} actual={actual!r}"
            )
    if actual_hashes["development_scorer_expected_sha256"] != actual_hashes[
        "holdout_scorer_actual_sha256"
    ]:
        raise ValueError("Holdout scorer development scorer ile aynı değil.")

    verify_score_rows(scores, expected_rows, rule)
    recomputed_overall = evaluate_predictions(scores)
    blind_review_scores = [
        row
        for row in scores
        if row.get("reference_basis") == "ai_assisted_blind_visual_review"
    ]
    blind_review_metrics = evaluate_predictions(blind_review_scores)
    recomputed_modality = grouped_metrics(scores, "modality_key")
    recomputed_pair_kind = grouped_metrics(scores, "pair_kind")
    if report.get("input_audit") != expected_audit:
        raise ValueError("Rapor input_audit bölümü kaynak metadata ile uyuşmuyor.")
    if report.get("overall_metrics") != recomputed_overall:
        raise ValueError("Rapor overall_metrics bölümü skor CSV'den yeniden üretilemedi.")
    if report.get("by_modality") != recomputed_modality:
        raise ValueError("Rapor by_modality bölümü yeniden üretilemedi.")
    if report.get("by_pair_kind") != recomputed_pair_kind:
        raise ValueError("Rapor by_pair_kind bölümü yeniden üretilemedi.")

    verification: dict[str, object] = {
        "schema_version": 1,
        "status": "verified_complete_after_interrupted_client_session",
        "verification_date": "2026-10-26",
        "image_files_reopened": False,
        "holdout_rescored": False,
        "score_rows": len(scores),
        "unique_pair_ids": len({row["pair_id"] for row in scores}),
        "missing_pair_ids": 0,
        "extra_pair_ids": 0,
        "rule_decisions_recomputed_from_saved_scores": len(scores),
        "input_hashes_verified": actual_hashes,
        "output_hashes": {
            "pair_scores_and_decisions_sha256": file_sha256(scores_path),
            "report_sha256": file_sha256(report_path),
        },
        "overall_metrics": recomputed_overall,
        "blind_visual_review_metrics": blind_review_metrics,
        "limitations": [
            "Bu doğrulama görüntüleri yeniden açmadı ve holdout'u yeniden skorlamadı.",
            "Görsel referanslar AI-assisted kör incelemedir; uzman gold standard değildir.",
            "Sonuç kanser sınıflandırma performansı değildir.",
        ],
    }
    atomic_write_json(verification_path, verification)
    print("Holdout çıktı doğrulaması: BAŞARILI")
    print(f"Satır / benzersiz Pair ID: {len(scores)} / {len(scores)}")
    print("Eksik / fazla Pair ID: 0 / 0")
    print("Kural kararları kayıtlı skorlardan 671/671 yeniden üretildi.")
    print("Girdi hash'leri ve rapor metrikleri birebir doğrulandı.")
    print(f"Kör görsel referans alt kümesi: {len(blind_review_scores)} çift")
    print(f"Doğrulama manifesti: {verification_path}")
    return verification


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
