"""Freeze the score-blind Candidate Generator v2.6b visual review.

Only the public review manifest is read.  The private selection manifest and
its stratum/candidate labels remain unopened so they cannot influence visual
decisions.  The target is deliberately narrow: two files are a positive only
when they can be explained as transformations of the same two-dimensional
source image.  Merely adjacent MRI/CT slices are therefore independent here.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "candidate_generator_v2_6b_prospective_selection_v1"
    / "review_manifest.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "research" / "candidate_generator_v2_6b_visual_review_v1.csv"
EXPECTED_INPUT_SHA256 = (
    "f961d5e92eb4da2b581852e8151834ff16deaa245a3d7e6e35fb819acd4fb099"
)
EXPECTED_PAIR_IDS = [f"V26P{number:06d}" for number in range(1, 161)]
REVIEWER = "Codex AI-assisted visual review authorized by Mehmet"
REVIEW_DATE = "2026-11-14"


# Multiple repeatable fine-grained landmarks agree after a permitted same-2D
# transformation (flip, rotation, crop, shift, brightness/contrast change).
SAME_SOURCE = {
    "V26P000005",
    "V26P000006",
    "V26P000010",
    "V26P000018",
    "V26P000036",
    "V26P000039",
    "V26P000044",
    "V26P000057",
    "V26P000058",
    "V26P000076",
    "V26P000083",
    "V26P000089",
    "V26P000100",
    "V26P000105",
    "V26P000111",
    "V26P000115",
    "V26P000126",
    "V26P000128",
    "V26P000142",
    "V26P000143",
    "V26P000150",
    "V26P000151",
    "V26P000152",
    "V26P000153",
}

# Similar tissue appearance, but the public images do not contain enough
# repeatable one-to-one landmarks for an honest forced decision.
UNCERTAIN = {"V26P000007"}

# These are judged independent under the same-2D target, but receive medium
# confidence because repetitive histology or adjacent CT/MRI anatomy makes the
# visual distinction less decisive than in ordinary negative pairs.
MEDIUM_INDEPENDENT = {
    "V26P000014",
    "V26P000023",
    "V26P000024",
    "V26P000026",
    "V26P000027",
    "V26P000033",
    "V26P000037",
    "V26P000041",
    "V26P000042",
    "V26P000050",
    "V26P000051",
    "V26P000059",
    "V26P000062",
    "V26P000066",
    "V26P000073",
    "V26P000079",
    "V26P000088",
    "V26P000094",
    "V26P000095",
    "V26P000106",
    "V26P000109",
    "V26P000110",
    "V26P000113",
    "V26P000121",
    "V26P000123",
    "V26P000124",
    "V26P000125",
    "V26P000134",
    "V26P000140",
    "V26P000141",
    "V26P000146",
    "V26P000157",
    "V26P000158",
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
                "Repeatable fine-grained tissue or anatomical landmarks agree after "
                "an allowed same-2D transformation such as flip, shift, crop, "
                "rotation, or appearance change."
            ),
        }
    if pair_id in UNCERTAIN:
        return {
            "visual_verdict": "uncertain",
            "confidence": "low",
            "review_notes": (
                "Tissue appearance is similar, but the public images do not show "
                "enough repeatable one-to-one landmarks for a forced same-2D decision."
            ),
        }
    return {
        "visual_verdict": "independent_lookalike",
        "confidence": "medium" if pair_id in MEDIUM_INDEPENDENT else "high",
        "review_notes": (
            "No repeatable one-to-one same-2D landmark correspondence; visually "
            "independent under the frozen same-2D target."
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
    output_rows = [
        {
            "pair_id": row["pair_id"],
            **decision_for(row["pair_id"]),
            "reviewer": REVIEWER,
            "review_date": REVIEW_DATE,
        }
        for row in rows
    ]
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
    print(f"Frozen v2.6b visual review: {OUTPUT_PATH}")
    print(f"Rows: {len(output_rows)}")
    print(f"Verdicts: {counts}")
    print(f"SHA-256: {sha256(OUTPUT_PATH)}")
    print("Private selection manifest was not read.")


if __name__ == "__main__":
    main()
