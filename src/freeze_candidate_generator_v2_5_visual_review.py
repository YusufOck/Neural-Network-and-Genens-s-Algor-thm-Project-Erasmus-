"""Freeze the score-blind Candidate Generator v2.5 visual review.

This script reads only the frozen public review manifest.  It never opens the
private selection manifest or similarity/stratum fields.  Decisions come from
the completed AI-assisted visual inspection authorized by Mehmet.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "candidate_generator_v2_5_prospective_selection_v1"
    / "review_manifest.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "research" / "candidate_generator_v2_5_visual_review_v1.csv"
EXPECTED_INPUT_SHA256 = (
    "b551ef6bea86ee62c9fc831a48dd0b662a99344f25da7ec9ed550a2a27103bfb"
)
EXPECTED_PAIR_IDS = [f"V25N{number:06d}" for number in range(1, 257)]
REVIEWER = "Codex AI-assisted visual review authorized by Mehmet"
REVIEW_DATE = "2026-11-11"


# Repeatable cellular/tissue landmarks, or CT/MRI anatomy consistent with the
# same acquisition/source family (including adjacent acquisition slices).
SAME_SOURCE = {
    "V25N000011",
    "V25N000012",
    "V25N000024",
    "V25N000060",
    "V25N000070",
    "V25N000071",
    "V25N000075",
    "V25N000084",
    "V25N000097",
    "V25N000098",
    "V25N000103",
    "V25N000107",
    "V25N000113",
    "V25N000116",
    "V25N000120",
    "V25N000125",
    "V25N000135",
    "V25N000142",
    "V25N000145",
    "V25N000151",
    "V25N000152",
    "V25N000154",
    "V25N000159",
    "V25N000170",
    "V25N000174",
    "V25N000177",
    "V25N000183",
    "V25N000185",
    "V25N000187",
    "V25N000198",
    "V25N000212",
    "V25N000221",
    "V25N000224",
    "V25N000229",
    "V25N000245",
    "V25N000254",
}

# Similar two-cell configuration, but the surrounding red-cell constellation
# does not give reliable one-to-one evidence for a forced source decision.
UNCERTAIN = {"V25N000227"}

# Clearly non-identical after original-resolution review, but repetitive tissue
# texture or slice/cell similarity warrants medium rather than high confidence.
MEDIUM_INDEPENDENT = {
    "V25N000005",
    "V25N000008",
    "V25N000010",
    "V25N000019",
    "V25N000023",
    "V25N000046",
    "V25N000049",
    "V25N000063",
    "V25N000094",
    "V25N000102",
    "V25N000134",
    "V25N000143",
    "V25N000148",
    "V25N000176",
    "V25N000191",
    "V25N000195",
    "V25N000202",
    "V25N000211",
    "V25N000220",
    "V25N000225",
    "V25N000236",
    "V25N000238",
    "V25N000250",
    "V25N000252",
    "V25N000256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_blind_manifest() -> list[dict[str, str]]:
    actual_hash = sha256(INPUT_PATH)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"Public review manifest changed: {actual_hash} != {EXPECTED_INPUT_SHA256}"
        )
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pair_ids = [row["pair_id"] for row in rows]
    if pair_ids != EXPECTED_PAIR_IDS:
        raise RuntimeError("Public review Pair IDs are missing, extra, or out of order")
    decision_fields = (
        "visual_verdict",
        "confidence",
        "reviewer",
        "review_date",
        "review_notes",
    )
    nonblank = [
        row["pair_id"]
        for row in rows
        if any(row.get(field, "").strip() for field in decision_fields)
    ]
    if nonblank:
        raise RuntimeError(f"Public review manifest was mutated: {nonblank[:5]}")
    return rows


def decision_for(pair_id: str) -> dict[str, str]:
    if pair_id in SAME_SOURCE:
        return {
            "visual_verdict": "same_source_candidate",
            "confidence": "high",
            "review_notes": (
                "Repeatable cellular/tissue landmarks or acquisition anatomy agree "
                "after shift, crop, orientation, appearance, or adjacent-slice change."
            ),
        }
    if pair_id in UNCERTAIN:
        return {
            "visual_verdict": "uncertain",
            "confidence": "low",
            "review_notes": (
                "The two-cell configuration is similar, but surrounding red-cell "
                "landmarks do not support a reliable forced source decision."
            ),
        }
    return {
        "visual_verdict": "independent_lookalike",
        "confidence": "medium" if pair_id in MEDIUM_INDEPENDENT else "high",
        "review_notes": (
            "No repeatable one-to-one landmark correspondence; visually independent "
            "within the limits of AI-assisted non-expert review."
        ),
    }


def validate_decision_sets() -> None:
    expected = set(EXPECTED_PAIR_IDS)
    sets = (SAME_SOURCE, UNCERTAIN, MEDIUM_INDEPENDENT)
    for index, left in enumerate(sets):
        if not left <= expected:
            raise RuntimeError(f"Unknown Pair IDs: {sorted(left - expected)}")
        for right in sets[index + 1 :]:
            if left & right:
                raise RuntimeError(f"Decision sets overlap: {sorted(left & right)}")


def main() -> None:
    rows = load_blind_manifest()
    validate_decision_sets()
    fieldnames = [
        "pair_id",
        "visual_verdict",
        "confidence",
        "reviewer",
        "review_date",
        "review_notes",
    ]
    output_rows: list[dict[str, str]] = []
    for row in rows:
        output_rows.append(
            {
                "pair_id": row["pair_id"],
                **decision_for(row["pair_id"]),
                "reviewer": REVIEWER,
                "review_date": REVIEW_DATE,
            }
        )
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Frozen visual review already exists: {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    counts: dict[str, int] = {}
    for row in output_rows:
        verdict = row["visual_verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    print(f"Frozen v2.5 visual review: {OUTPUT_PATH}")
    print(f"Rows: {len(output_rows)}")
    print(f"Verdicts: {counts}")
    print(f"SHA-256: {sha256(OUTPUT_PATH)}")
    print("Private selection manifest was not read.")


if __name__ == "__main__":
    main()
