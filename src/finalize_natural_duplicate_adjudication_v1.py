from __future__ import annotations

"""Validate and finalize the AI-assisted natural-audit adjudication source.

The original blank queue, score-blind visual decisions, and frozen duplicate rule
outputs remain immutable.  This script validates a separate 25-row decision source,
creates the filled adjudicated subset, and derives a traceable 345-pair final
reference table.  It does not create duplicate families, a classifier split, or a
model.
"""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from data_index import project_root
from duplicate_decision_rule import file_sha256


EXPECTED_QUEUE_SHA256 = (
    "113fe1ca33e5ab34863b9ae36b8ba52f2d75d7eb2a2e2633b21f6445e750ce2e"
)
EXPECTED_SCORES_SHA256 = (
    "76d8e739a9a59b26e7ef8de44bdb534e5f8be06b90d748dbfb7774e8c62df3da"
)
EXPECTED_QUEUE_ROWS = 25
EXPECTED_AUDIT_ROWS = 345
ALLOWED_VERDICTS = {"same_source", "independent", "uncertain"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
DECISION_FIELDS = (
    "pair_id",
    "adjudication_verdict",
    "adjudication_confidence",
    "adjudicator",
    "adjudication_date",
    "adjudication_notes",
)
FINAL_REFERENCE_FIELDS = (
    "pair_id",
    "final_reference",
    "reference_basis",
    "visual_verdict",
    "frozen_rule_decision",
    "selection_stratum",
    "audit_region",
    "source_group",
    "modality_key",
    "left_leaf",
    "right_leaf",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    base = root / "research" / "natural_duplicate_adjudication_v1"
    parser = argparse.ArgumentParser(
        description=(
            "25 çiftlik ayrı adjudication karar kaynağını doğrular ve 345 çiftlik "
            "nihai referans tablosuna dönüştürür; family/split/model üretmez."
        )
    )
    parser.add_argument("--queue", default=str(base / "adjudication_queue.csv"))
    parser.add_argument(
        "--decisions", default=str(base / "adjudication_decisions_v1.csv")
    )
    parser.add_argument(
        "--scores",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "natural_duplicate_rule_evaluation_v1"
            / "pair_scores_and_decisions.csv"
        ),
    )
    parser.add_argument("--output-dir", default=str(base))
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def index_unique(
    rows: Sequence[Mapping[str, str]], source_name: str
) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{source_name}: boş Pair ID bulundu.")
        if pair_id in output:
            raise ValueError(f"{source_name}: tekrarlanan Pair ID: {pair_id}")
        output[pair_id] = row
    return output


def validate_decisions(
    queue_rows: Sequence[Mapping[str, str]],
    decision_fields: Sequence[str],
    decision_rows: Sequence[Mapping[str, str]],
) -> dict[str, Mapping[str, str]]:
    missing_fields = sorted(set(DECISION_FIELDS) - set(decision_fields))
    if missing_fields:
        raise ValueError(f"Adjudication karar dosyasında eksik sütunlar: {missing_fields}")
    queue = index_unique(queue_rows, "queue")
    decisions = index_unique(decision_rows, "decisions")
    if len(queue) != EXPECTED_QUEUE_ROWS or len(decisions) != EXPECTED_QUEUE_ROWS:
        raise ValueError("Queue ve karar kaynağı tam olarak 25 satır içermelidir.")
    if set(queue) != set(decisions):
        raise ValueError("Karar Pair ID kapsamı adjudication kuyruğuyla eşleşmiyor.")

    queue_decision_fields = [
        "adjudication_verdict",
        "adjudication_confidence",
        "adjudicator",
        "adjudication_date",
        "adjudication_notes",
    ]
    for pair_id, row in queue.items():
        if any(row.get(field, "").strip() for field in queue_decision_fields):
            raise ValueError(f"Dondurulmuş blank queue değiştirilmiş: {pair_id}")
    for pair_id, row in decisions.items():
        verdict = row["adjudication_verdict"].strip()
        confidence = row["adjudication_confidence"].strip()
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Geçersiz adjudication verdict: {pair_id}={verdict}")
        if confidence not in ALLOWED_CONFIDENCE:
            raise ValueError(f"Geçersiz güven düzeyi: {pair_id}={confidence}")
        for field in ("adjudicator", "adjudication_date", "adjudication_notes"):
            if not row[field].strip():
                raise ValueError(f"Boş adjudication alanı: {pair_id}/{field}")
    return decisions


def build_adjudicated_rows(
    queue_rows: Sequence[Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for source in queue_rows:
        row = dict(source)
        decision = decisions[row["pair_id"]]
        for field in DECISION_FIELDS[1:]:
            row[field] = decision[field]
        output.append(row)
    return output


def _safe_consensus_reference(score: Mapping[str, str]) -> str | None:
    if (
        score["human_verdict"] == "same_source_candidate"
        and score["decision"] == "same_source"
    ):
        return "same_source"
    if (
        score["human_verdict"] == "independent_lookalike"
        and score["decision"] == "independent"
    ):
        return "independent"
    return None


def build_final_references(
    score_rows: Sequence[Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    scores = index_unique(score_rows, "scores")
    if len(scores) != EXPECTED_AUDIT_ROWS:
        raise ValueError("Frozen score kaynağı 345 satır içermelidir.")
    output: list[dict[str, str]] = []
    for pair_id in sorted(scores):
        score = scores[pair_id]
        if pair_id in decisions:
            final_reference = decisions[pair_id]["adjudication_verdict"]
            basis = "ai_assisted_score_aware_adjudication"
        else:
            final_reference = _safe_consensus_reference(score)
            if final_reference is None:
                raise ValueError(f"Kuyruk dışında konsensüs dışı çift bulundu: {pair_id}")
            basis = "frozen_visual_rule_safe_consensus"
        output.append(
            {
                "pair_id": pair_id,
                "final_reference": final_reference,
                "reference_basis": basis,
                "visual_verdict": score["human_verdict"],
                "frozen_rule_decision": score["decision"],
                "selection_stratum": score["selection_stratum"],
                "audit_region": score["audit_region"],
                "source_group": score["source_group"],
                "modality_key": score["modality_key"],
                "left_leaf": score["left_leaf"],
                "right_leaf": score["right_leaf"],
            }
        )
    return output


def rule_comparison(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row["final_reference"]][row["frozen_rule_decision"]] += 1
    same_total = sum(matrix["same_source"].values())
    same_found = matrix["same_source"]["same_source"]
    independent_total = sum(matrix["independent"].values())
    independent_correct = matrix["independent"]["independent"]
    decided = sum(
        count
        for truth, counts in matrix.items()
        for decision, count in counts.items()
        if decision != "uncertain" and truth != "uncertain"
    )
    correct = same_found + independent_correct
    return {
        "matrix": {
            truth: dict(sorted(counts.items()))
            for truth, counts in sorted(matrix.items())
        },
        "same_source_recall": same_found / same_total if same_total else None,
        "independent_recall": (
            independent_correct / independent_total if independent_total else None
        ),
        "coverage": decided / len(rows) if rows else None,
        "decided_pair_accuracy": correct / decided if decided else None,
        "accuracy_counting_uncertain_as_not_correct": (
            correct / len(rows) if rows else None
        ),
    }


def atomic_write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Mapping[str, object]]:
    queue_path = Path(args.queue).resolve()
    decisions_path = Path(args.decisions).resolve()
    scores_path = Path(args.scores).resolve()
    output_dir = Path(args.output_dir).resolve()
    adjudicated_path = output_dir / "adjudicated_pairs_v1.csv"
    references_path = output_dir / "final_pair_references_v1.csv"
    report_path = output_dir / "adjudication_report_v1.json"
    if any(path.exists() for path in (adjudicated_path, references_path, report_path)):
        raise FileExistsError("Adjudication final çıktısı zaten var; üzerine yazılmadı.")
    if file_sha256(queue_path) != EXPECTED_QUEUE_SHA256:
        raise ValueError("Blank adjudication queue frozen hash ile eşleşmiyor.")
    if file_sha256(scores_path) != EXPECTED_SCORES_SHA256:
        raise ValueError("Frozen natural-audit skor kaynağı hash ile eşleşmiyor.")

    queue_fields, queue_rows = read_csv(queue_path)
    decision_fields, decision_rows = read_csv(decisions_path)
    _, score_rows = read_csv(scores_path)
    decisions = validate_decisions(queue_rows, decision_fields, decision_rows)
    adjudicated_rows = build_adjudicated_rows(queue_rows, decisions)
    final_references = build_final_references(score_rows, decisions)

    same_source_rows = [
        row for row in final_references if row["final_reference"] == "same_source"
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "natural_audit_v1_ai_assisted_adjudication_complete",
        "experiment_type": "near_duplicate_second_stage_score_aware_adjudication",
        "counts": {
            "natural_audit_pairs": len(final_references),
            "safe_consensus_pairs": sum(
                row["reference_basis"] == "frozen_visual_rule_safe_consensus"
                for row in final_references
            ),
            "adjudicated_pairs": len(adjudicated_rows),
            "adjudication_verdicts": dict(
                sorted(
                    Counter(
                        row["adjudication_verdict"] for row in adjudicated_rows
                    ).items()
                )
            ),
            "final_references": dict(
                sorted(
                    Counter(row["final_reference"] for row in final_references).items()
                )
            ),
            "same_source_inside_candidate": sum(
                row["audit_region"] == "candidate" for row in same_source_rows
            ),
            "same_source_outside_candidate": sum(
                row["audit_region"] == "outside" for row in same_source_rows
            ),
        },
        "rule_comparison_after_score_aware_adjudication": rule_comparison(
            final_references
        ),
        "integrity": {
            "blank_queue_sha256": file_sha256(queue_path),
            "adjudication_decisions_sha256": file_sha256(decisions_path),
            "frozen_scores_sha256": file_sha256(scores_path),
        },
        "limitations": [
            "Adjudication Codex tarafından AI-destekli ve skorlar görünürken yapılmıştır; bağımsız uzman gold standard değildir.",
            "Kural karşılaştırması adjudication sırasında kural kanıtları görüldüğü için bağımsız holdout performansı değildir.",
            "Aynı hasta veya taramaya ait komşu kesitler yerel dosyalardan güvenilir biçimde gruplanamaz; karar hedefi aynı 2B kaynak görüntüdür.",
            "345 çiftlik örneklem 3.093.466 olası doğal çiftin tamamını kapsamaz.",
            "Bu çıktı duplicate family, classifier split veya kanser sınıflandırma modeli değildir.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(adjudicated_path, queue_fields, adjudicated_rows)
    atomic_write_csv(references_path, FINAL_REFERENCE_FIELDS, final_references)
    atomic_write_json(report_path, report)

    print("İşlem türü: NATURAL DUPLICATE ADJUDICATION v1 SONLANDIRMA")
    print(f"Adjudication kararları: {len(adjudicated_rows)}/{EXPECTED_QUEUE_ROWS}")
    print(f"Karar dağılımı: {report['counts']['adjudication_verdicts']}")
    print(f"345 çiftlik final referans: {report['counts']['final_references']}")
    print(
        "Final same-source candidate/outside: "
        f"{report['counts']['same_source_inside_candidate']}/"
        f"{report['counts']['same_source_outside_candidate']}"
    )
    print(f"Rapor: {report_path}")
    print("NOT: Family, split veya model oluşturulmadı.")
    return adjudicated_path, references_path, report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
