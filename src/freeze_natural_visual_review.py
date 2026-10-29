"""Freeze the score-blind natural near-duplicate visual review.

This script records the completed AI-assisted visual decisions without reading
the private candidate stratum or similarity-score file.  It deliberately reads
only the frozen blind review manifest and verifies its SHA-256 before writing
the separate decision source.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "research" / "natural_duplicate_audit_v1" / "review_manifest.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "research" / "natural_duplicate_visual_review_v1.csv"
EXPECTED_INPUT_SHA256 = (
    "84ae0e9fa6d09cf6162d5ecbeec47cab82786217a3895d03c67ed8e55729da7c"
)
EXPECTED_PAIR_IDS = [f"N{number:06d}" for number in range(1, 346)]
REVIEWER = "Codex AI-assisted visual review authorized by Mehmet"
REVIEW_DATE = "2026-10-29"


# Definite transformed/near-identical views with repeatable landmark agreement.
SAME_SOURCE = {
    "N000021",
    "N000055",
    "N000064",
    "N000075",
    "N000098",
    "N000120",
    "N000136",
    "N000145",
    "N000162",
    "N000170",
    "N000220",
    "N000234",
    "N000239",
    "N000248",
    "N000289",
    "N000294",
    "N000304",
    "N000308",
    "N000333",
    "N000342",
    "N000345",
}

# Image evidence was insufficient to distinguish a source-relative from a hard
# lookalike.  These rows must remain abstentions until scorer comparison/review.
UNCERTAIN = {"N000252", "N000337"}

# Visually distinct pairs that nevertheless received medium rather than high
# confidence because of repetitive tissue/cell texture or slice similarity.
MEDIUM_INDEPENDENT = {
    "N000009", "N000027", "N000031", "N000037", "N000039", "N000049",
    "N000063", "N000067", "N000077", "N000079", "N000089", "N000093",
    "N000101", "N000105", "N000110", "N000116", "N000123", "N000126",
    "N000131", "N000133", "N000137", "N000139", "N000148", "N000150",
    "N000159", "N000160", "N000169", "N000179", "N000191", "N000193",
    "N000194", "N000197", "N000201", "N000203", "N000206", "N000219",
    "N000221", "N000224", "N000229", "N000233", "N000253", "N000255",
    "N000270", "N000272", "N000280", "N000281", "N000282", "N000306",
    "N000311", "N000317", "N000318", "N000319", "N000329", "N000330",
    "N000344",
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
            f"Blind manifest hash changed: {actual_hash} != {EXPECTED_INPUT_SHA256}"
        )

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    pair_ids = [row["pair_id"] for row in rows]
    if pair_ids != EXPECTED_PAIR_IDS:
        raise RuntimeError("Blind manifest Pair IDs are missing, extra, or out of order")

    decision_fields = ("visual_verdict", "confidence", "reviewer", "review_date", "review_notes")
    nonblank = [
        row["pair_id"]
        for row in rows
        if any(row.get(field, "").strip() for field in decision_fields)
    ]
    if nonblank:
        raise RuntimeError(f"Blind manifest was mutated; nonblank decisions: {nonblank[:5]}")
    return rows


def decision_for(pair_id: str) -> dict[str, str]:
    if pair_id in SAME_SOURCE:
        confidence = "medium" if pair_id == "N000021" else "high"
        return {
            "visual_verdict": "same_source_candidate",
            "confidence": confidence,
            "review_notes": (
                "Repeatable anatomical/cellular landmarks agree after a simple "
                "flip, shift, crop, or appearance change."
            ),
        }
    if pair_id in UNCERTAIN:
        note = (
            "Similar CT anatomy may reflect an adjacent acquisition slice; the contact "
            "sheet and original images do not prove an augmented same-image relative."
            if pair_id == "N000252"
            else
            "Dense Pap-smear texture is highly similar, but repeatable one-to-one "
            "landmark correspondence is not reliable enough for a forced decision."
        )
        return {
            "visual_verdict": "uncertain",
            "confidence": "low",
            "review_notes": note,
        }

    confidence = "medium" if pair_id in MEDIUM_INDEPENDENT else "high"
    return {
        "visual_verdict": "independent_lookalike",
        "confidence": confidence,
        "review_notes": (
            "No repeatable one-to-one landmark correspondence; visually independent "
            "within the limits of AI-assisted non-expert review."
        ),
    }


def main() -> None:
    rows = load_blind_manifest()
    if SAME_SOURCE & UNCERTAIN:
        raise RuntimeError("A Pair ID appears in both SAME_SOURCE and UNCERTAIN")
    unknown = (SAME_SOURCE | UNCERTAIN | MEDIUM_INDEPENDENT) - set(EXPECTED_PAIR_IDS)
    if unknown:
        raise RuntimeError(f"Decision configuration has unknown Pair IDs: {sorted(unknown)}")

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
        decision = decision_for(row["pair_id"])
        output_rows.append(
            {
                "pair_id": row["pair_id"],
                **decision,
                "reviewer": REVIEWER,
                "review_date": REVIEW_DATE,
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    counts: dict[str, int] = {}
    for row in output_rows:
        verdict = row["visual_verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    print(f"Frozen visual review: {OUTPUT_PATH}")
    print(f"Rows: {len(output_rows)}")
    print(f"Verdicts: {counts}")
    print(f"SHA-256: {sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
