from __future__ import annotations

"""Finalize the 16-pair same-2D development reference without rewriting v2.5."""

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from data_index import project_root
from duplicate_decision_rule import file_sha256


ROOT = project_root()
INPUT = ROOT / "research" / "v2_6_same2d_development_v1" / "pair_scores_and_rule.csv"
OUTPUT = ROOT / "research" / "v2_6_same2d_development_v1" / "final_pair_references.csv"
REPORT = ROOT / "research" / "v2_6_same2d_development_v1" / "adjudication_report.json"
EXPECTED_INPUT_SHA256 = "9d84d5fd75c64cda1883c414f38d2d1c71ad6e86daa7ea98852d0c6292b2f6e4"

ADJUDICATIONS = {
    "V25N000084": (
        "independent",
        "high",
        "Coronal CT anatomy, organ cross-sections and internal landmarks differ; similar modality is not the same 2D source image.",
    ),
    "V25N000098": (
        "independent",
        "high",
        "After accounting for horizontal flip, tumor/organ contours and surrounding structures differ; compatible with a different slice/scan, not the same 2D image.",
    ),
    "V25N000113": (
        "independent",
        "high",
        "The axial level, bowel/tumor region and kidney cross-section differ; no repeatable one-to-one 2D content correspondence.",
    ),
    "V25N000185": (
        "independent",
        "high",
        "The axial level is similar, but renal contrast patterns, organ contours and bowel structures differ; this is an adjacent/different slice rather than an augmentation.",
    ),
    "V25N000212": (
        "independent",
        "high",
        "Kidney cross-sections and abdominal contents differ; the pair is not derived from the same 2D source image.",
    ),
}

FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "frozen_v2_5_visual_verdict",
    "frozen_rule_decision",
    "frozen_rule_path",
    "adjudication_verdict",
    "adjudication_confidence",
    "adjudicator",
    "adjudication_date",
    "adjudication_notes",
    "final_reference",
    "reference_basis",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {path}")
        return [dict(row) for row in reader]


def finalize(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if len(rows) != 16 or len({row["pair_id"] for row in rows}) != 16:
        raise ValueError("Expected 16 unique scored pairs.")
    uncertain_ids = {row["pair_id"] for row in rows if row["rule_decision"] == "uncertain"}
    if uncertain_ids != set(ADJUDICATIONS):
        raise ValueError("Frozen-rule uncertain set differs from the adjudication plan.")

    output: list[dict[str, str]] = []
    for source in sorted(rows, key=lambda row: row["pair_id"]):
        pair_id = source["pair_id"]
        rule_decision = source["rule_decision"]
        if rule_decision == "uncertain":
            verdict, confidence, notes = ADJUDICATIONS[pair_id]
            final_reference = verdict
            basis = "ai_assisted_score_aware_same2d_adjudication"
            adjudicator = "Codex AI-assisted non-expert review authorized by Mehmet"
            date = "2026-11-12"
        else:
            verdict = ""
            confidence = ""
            notes = ""
            final_reference = rule_decision
            basis = f"frozen_duplicate_rule_v1:{source['rule_decision_path']}"
            adjudicator = ""
            date = ""
        output.append(
            {
                "pair_id": pair_id,
                "source_group": source["source_group"],
                "left_leaf": source["left_leaf"],
                "right_leaf": source["right_leaf"],
                "frozen_v2_5_visual_verdict": source["frozen_v2_5_visual_verdict"],
                "frozen_rule_decision": rule_decision,
                "frozen_rule_path": source["rule_decision_path"],
                "adjudication_verdict": verdict,
                "adjudication_confidence": confidence,
                "adjudicator": adjudicator,
                "adjudication_date": date,
                "adjudication_notes": notes,
                "final_reference": final_reference,
                "reference_basis": basis,
            }
        )
    return output


def atomic_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def main() -> None:
    input_hash = file_sha256(INPUT)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Scored input hash mismatch: {input_hash}")
    final_rows = finalize(read_csv(INPUT))
    atomic_csv(OUTPUT, final_rows)
    counts = Counter(row["final_reference"] for row in final_rows)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "v2_6_same2d_development_adjudication_complete",
        "experiment_type": "ai_assisted_nonexpert_development_adjudication",
        "does_not_rewrite_v2_5": True,
        "not_a_new_holdout": True,
        "does_not_create_family_split_or_model": True,
        "input_scores_sha256": input_hash,
        "pair_count": len(final_rows),
        "rule_decision_counts": dict(sorted(Counter(row["frozen_rule_decision"] for row in final_rows).items())),
        "adjudicated_pair_count": sum(bool(row["adjudication_verdict"]) for row in final_rows),
        "final_reference_counts": dict(sorted(counts.items())),
        "same_source_pair_ids": [row["pair_id"] for row in final_rows if row["final_reference"] == "same_source"],
        "independent_pair_ids": [row["pair_id"] for row in final_rows if row["final_reference"] == "independent"],
        "uncertain_pair_ids": [row["pair_id"] for row in final_rows if row["final_reference"] == "uncertain"],
        "limitation": "Development-only, AI-assisted non-expert same-2D references; not a clinical or patient-level gold standard.",
    }
    atomic_json(REPORT, report)
    print(f"Final references: {dict(sorted(counts.items()))}")
    print(f"CSV: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
