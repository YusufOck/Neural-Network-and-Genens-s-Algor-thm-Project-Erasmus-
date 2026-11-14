from __future__ import annotations

"""Unblind and evaluate the frozen Candidate Generator v2.6b audit.

The visual decisions were frozen before this script was run. This program joins
those decisions to the private selection strata, verifies the frozen evidence,
and evaluates the predeclared acceptance gates. It does not alter a verdict,
score candidates, construct families, create a split, or train a model.
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


EXPECTED_HASHES = {
    "config": "964bf13d3cd73ecc29e777775a8a9f64af94ada86bec5cc191f1f770032ff6c4",
    "private_selection": "549e12b57daac220fbfd631f2717c28f5b78370f42fb713afdb746e176a7f955",
    "public_manifest": "f961d5e92eb4da2b581852e8151834ff16deaa245a3d7e6e35fb819acd4fb099",
    "selection_report": "3d00f465a7fbdf89065bf47f373d87d7d14fb9a10710e36a37dc171979a031cc",
    "visual_decisions": "2220fca301f0d281cab03f98847e865e02de1269b0881479ebae7813d7b4f36b",
    "v2_5_audit_report": "14d9df6ff3f03da9d9555324625e9934df274f568a736bfa669b0cb321cfbb01",
    "v2_6a_report": "1b678e715e509e038193175b9c363c3ad20b32ba1ab3e10617e0ec5887591bf7",
}
EXPECTED_ROWS = 160

ALLOWED_STRATA = {
    "outside_selected_boundary",
    "outside_alternate_channel",
    "outside_random_same_leaf",
    "candidate_new_extension",
}
ALLOWED_VERDICTS = {
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
}
GROUP_PATH_PREFIX = {
    "kidney_ct": "Kidney Cancer/",
    "lymphoma_histology": "Lymphoma/",
}
PUBLIC_IDENTITY_FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
)
PRIVATE_FIELDS = (
    "pair_id",
    "selection_stratum",
    "source_group",
    "left_image_index",
    "right_image_index",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "v2_6b_candidate",
    "candidate_reason",
    "selected_extension_channels",
    "selected_extension_top_k",
    "selected_extension_best_rank",
    "all_channel_best_rank",
    "raw_multicrop_best_rank",
    "phash_enabled",
    "phash_distance",
    "phash_radius12",
)
OUTPUT_FIELDS = PRIVATE_FIELDS + (
    "audit_region",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    selection_dir = root / "research" / "candidate_generator_v2_6b_prospective_selection_v1"
    output_dir = root / "research" / "candidate_generator_v2_6b_prospective_audit_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmus V2.6b gorsel kararlarini private strata ile birlestirir "
            "ve onceden belirlenen kabul kapilarini degerlendirir."
        )
    )
    parser.add_argument("--config", default=str(root / "config" / "candidate_generator_v2_6b_prospective_audit.json"))
    parser.add_argument("--private-selection", default=str(selection_dir / "selection_private.csv"))
    parser.add_argument("--public-manifest", default=str(selection_dir / "review_manifest.csv"))
    parser.add_argument("--selection-report", default=str(selection_dir / "selection_report.json"))
    parser.add_argument("--visual-decisions", default=str(root / "research" / "candidate_generator_v2_6b_visual_review_v1.csv"))
    parser.add_argument("--v2-5-audit-report", default=str(root / "research" / "candidate_generator_v2_5_prospective_audit_v1" / "unblinded_report.json"))
    parser.add_argument("--v2-6a-report", default=str(root / "research" / "candidate_generator_v2_6a_bounded_grid_v1.json"))
    parser.add_argument("--output-csv", default=str(output_dir / "unblinded_pair_results.csv"))
    parser.add_argument("--output-json", default=str(output_dir / "unblinded_report.json"))
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
            raise ValueError(f"CSV basligi bulunamadi: {path}")
        return [dict(row) for row in reader]


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object bekleniyordu: {path}")
    return payload


def index_unique(rows: Sequence[Mapping[str, str]], name: str) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{name}: bos Pair ID")
        if pair_id in indexed:
            raise ValueError(f"{name}: tekrarlanan Pair ID {pair_id}")
        indexed[pair_id] = row
    return indexed


def merge_rows(
    private_rows: Sequence[Mapping[str, str]],
    public_rows: Sequence[Mapping[str, str]],
    visual_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    private_by_id = index_unique(private_rows, "private selection")
    public_by_id = index_unique(public_rows, "public manifest")
    visual_by_id = index_unique(visual_rows, "visual decisions")
    expected_ids = set(private_by_id)
    for name, indexed in (("public manifest", public_by_id), ("visual decisions", visual_by_id)):
        if set(indexed) != expected_ids:
            missing = sorted(expected_ids - set(indexed))
            extra = sorted(set(indexed) - expected_ids)
            raise ValueError(f"{name}: Pair ID kapsami eslesmiyor; eksik={missing[:5]}, fazla={extra[:5]}")

    merged: list[dict[str, str]] = []
    for pair_id in sorted(expected_ids):
        private = private_by_id[pair_id]
        public = public_by_id[pair_id]
        visual = visual_by_id[pair_id]
        for field in PUBLIC_IDENTITY_FIELDS:
            if private.get(field, "") != public.get(field, ""):
                raise ValueError(f"{pair_id}: private/public {field} uyusmuyor")

        stratum = private.get("selection_stratum", "").strip()
        verdict = visual.get("visual_verdict", "").strip()
        if stratum not in ALLOWED_STRATA:
            raise ValueError(f"{pair_id}: gecersiz stratum {stratum!r}")
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"{pair_id}: gecersiz gorsel karar {verdict!r}")

        candidate = private.get("v2_6b_candidate", "").strip()
        expected_candidate = "yes" if stratum == "candidate_new_extension" else "no"
        if candidate != expected_candidate:
            raise ValueError(f"{pair_id}: stratum/candidate uyeligi uyusmuyor")
        audit_region = "candidate" if candidate == "yes" else "outside"

        source_group = private.get("source_group", "").strip()
        prefix = GROUP_PATH_PREFIX.get(source_group)
        if prefix is None:
            raise ValueError(f"{pair_id}: bilinmeyen source group {source_group!r}")
        if not private.get("left_path", "").startswith(prefix) or not private.get("right_path", "").startswith(prefix):
            raise ValueError(f"{pair_id}: source-group disi path bulundu")
        if private.get("left_leaf", "") != private.get("right_leaf", ""):
            raise ValueError(f"{pair_id}: same-leaf audit icinde cross-leaf cift bulundu")

        row = {field: private.get(field, "") for field in PRIVATE_FIELDS}
        row.update(
            {
                "audit_region": audit_region,
                "visual_verdict": verdict,
                "confidence": visual.get("confidence", ""),
                "reviewer": visual.get("reviewer", ""),
                "review_date": visual.get("review_date", ""),
                "review_notes": visual.get("review_notes", ""),
            }
        )
        merged.append(row)
    return merged


def nested_counts(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row[field]][row["visual_verdict"]] += 1
    return {
        group: {verdict: counts.get(verdict, 0) for verdict in sorted(ALLOWED_VERDICTS)}
        for group, counts in sorted(grouped.items())
    }


def summarize_visual(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    candidate = [row for row in rows if row["audit_region"] == "candidate"]
    outside = [row for row in rows if row["audit_region"] == "outside"]
    candidate_positive = [row for row in candidate if row["visual_verdict"] == "same_source_candidate"]
    candidate_uncertain = [row for row in candidate if row["visual_verdict"] == "uncertain"]
    outside_positive = [row for row in outside if row["visual_verdict"] == "same_source_candidate"]
    outside_uncertain = [row for row in outside if row["visual_verdict"] == "uncertain"]
    return {
        "total_pairs": len(rows),
        "verdict_counts": dict(sorted(Counter(row["visual_verdict"] for row in rows).items())),
        "region_counts": dict(sorted(Counter(row["audit_region"] for row in rows).items())),
        "counts_by_stratum": nested_counts(rows, "selection_stratum"),
        "counts_by_source_group": nested_counts(rows, "source_group"),
        "candidate_same_source_count": len(candidate_positive),
        "candidate_same_source_pair_ids": [row["pair_id"] for row in candidate_positive],
        "candidate_uncertain_count": len(candidate_uncertain),
        "candidate_uncertain_pair_ids": [row["pair_id"] for row in candidate_uncertain],
        "outside_same_source_count": len(outside_positive),
        "outside_same_source_pair_ids": [row["pair_id"] for row in outside_positive],
        "outside_same_source_by_group": dict(sorted(Counter(row["source_group"] for row in outside_positive).items())),
        "outside_uncertain_count": len(outside_uncertain),
        "outside_uncertain_pair_ids": [row["pair_id"] for row in outside_uncertain],
    }


def evaluate(
    config: Mapping[str, object],
    selection_report: Mapping[str, object],
    v2_5_audit_report: Mapping[str, object],
    v2_6a_report: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    visual = summarize_visual(rows)
    gates_config = config["predeclared_acceptance_gates"]
    prior_values = v2_5_audit_report["observed_gate_values"]
    prior_gates = v2_5_audit_report["acceptance_gates"]

    if not all(
        bool(prior_gates[name])
        for name in (
            "exact_reference_recall",
            "controlled_positive_recall",
            "regression_natural_recall",
            "cross_source_group_pairs",
        )
    ):
        raise ValueError("V2.5 regression evidence gates are not intact")

    selected_profile_id = config["candidate_rule"]["selected_profile"]
    if v2_6a_report["selected_development_profile"] != selected_profile_id:
        raise ValueError("V2.6a selected development profile does not match V2.6b config")
    selected_profile = next(
        profile for profile in v2_6a_report["profiles"] if profile["profile_id"] == selected_profile_id
    )

    exact_hits = int(prior_values["exact_reference_hits"])
    controlled_hits = int(prior_values["controlled_positive_hits"])
    regression_hits = int(prior_values["regression_natural_hits"])
    development_hits = int(selected_profile["development_same2d_hits"])
    full_pairs = int(selection_report["candidate_policy"]["full_candidate_pairs"])
    image_count = int(selection_report["input"]["image_count"])
    pairs_per_image = full_pairs / image_count
    cross_group_pairs = 0  # merge_rows verifies both paths against the declared source group.

    observed = {
        "exact_reference_hits": exact_hits,
        "controlled_positive_hits": controlled_hits,
        "regression_natural_hits": regression_hits,
        "development_same2d_hits": development_hits,
        "new_outside_same2d_pairs": int(visual["outside_same_source_count"]),
        "unresolved_outside_uncertain_pairs": int(visual["outside_uncertain_count"]),
        "cross_source_group_pairs": cross_group_pairs,
        "full_candidate_pairs": full_pairs,
        "candidate_pairs_per_image": pairs_per_image,
    }
    gates = {
        "exact_reference_recall": exact_hits == int(gates_config["required_exact_reference_hits"]),
        "controlled_positive_recall": controlled_hits == int(gates_config["required_controlled_positive_hits"]),
        "regression_natural_recall": regression_hits == int(gates_config["required_regression_natural_hits"]),
        "development_same2d_recall": development_hits == int(gates_config["required_development_same2d_hits"]),
        "new_outside_same2d": int(visual["outside_same_source_count"]) == int(gates_config["required_new_outside_same2d_pairs"]),
        "unresolved_outside_uncertain": int(visual["outside_uncertain_count"]) == int(gates_config["required_unresolved_outside_uncertain_pairs"]),
        "cross_source_group_pairs": cross_group_pairs == int(gates_config["required_cross_source_group_pairs"]),
        "full_candidate_pairs": full_pairs == int(gates_config["required_full_candidate_pairs"]),
        "candidate_load": pairs_per_image <= float(gates_config["maximum_candidate_pairs_per_image"]),
    }
    gates["all_passed"] = all(gates.values())
    return {
        "visual_audit": visual,
        "observed_gate_values": observed,
        "acceptance_gates": gates,
        "decision": (
            "PASS: v2.6b selected profile passed every predeclared gate."
            if gates["all_passed"]
            else "FAIL: reject v2.6b; do not score candidates, construct families, split data, or train a classifier under this rule."
        ),
    }


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def main() -> int:
    args = parse_args()
    root = project_root().resolve()
    paths = {
        "config": Path(args.config).resolve(),
        "private_selection": Path(args.private_selection).resolve(),
        "public_manifest": Path(args.public_manifest).resolve(),
        "selection_report": Path(args.selection_report).resolve(),
        "visual_decisions": Path(args.visual_decisions).resolve(),
        "v2_5_audit_report": Path(args.v2_5_audit_report).resolve(),
        "v2_6a_report": Path(args.v2_6a_report).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise ValueError(f"{name} hash degismis: expected={expected}, actual={hashes[name]}")

    selection_report = read_json(paths["selection_report"])
    if selection_report["integrity"]["selection_private_sha256"] != hashes["private_selection"]:
        raise ValueError("Selection report/private manifest hash uyusmuyor")
    if selection_report["integrity"]["review_manifest_sha256"] != hashes["public_manifest"]:
        raise ValueError("Selection report/public manifest hash uyusmuyor")

    merged = merge_rows(
        read_csv(paths["private_selection"]),
        read_csv(paths["public_manifest"]),
        read_csv(paths["visual_decisions"]),
    )
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(f"Beklenen {EXPECTED_ROWS} satir, bulunan {len(merged)}")

    result = evaluate(
        read_json(paths["config"]),
        selection_report,
        read_json(paths["v2_5_audit_report"]),
        read_json(paths["v2_6a_report"]),
        merged,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "candidate_generator_v2_6b_prospective_audit_unblinded_complete",
        "experiment_type": "predeclared_score_blind_prospective_same2d_audit_gate_evaluation",
        "visual_decisions_were_frozen_before_unblinding": True,
        "does_not_modify_visual_decisions": True,
        "does_not_score_candidates_or_create_split": True,
        "inputs": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "integrity": {
            "expected_rows": EXPECTED_ROWS,
            "joined_rows": len(merged),
            "pair_ids_unique_and_equal": True,
            "private_public_identity_fields_equal": True,
            "source_group_paths_consistent": True,
            "same_leaf_scope_verified": True,
            "frozen_input_hashes_verified": True,
        },
        **result,
        "limitations": [
            "Visual labels are AI-assisted non-expert leakage-audit decisions, not clinical ground truth.",
            "The 128-pair outside sample cannot prove recall over every possible image pair.",
            "This audit targets same-2D augmentation families and does not establish patient- or series-level separation.",
            "A failed gate rejects this frozen profile and cannot be repaired by changing decisions after unblinding.",
        ],
    }
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    atomic_write_csv(output_csv, merged)
    atomic_write_json(output_json, payload)

    visual = result["visual_audit"]
    gates = result["acceptance_gates"]
    print("V2.6b prospective audit: frozen decisions unblinded")
    print(f"Pairs: {visual['total_pairs']}; verdicts: {visual['verdict_counts']}")
    print(f"Candidate same-2D: {visual['candidate_same_source_count']}")
    print(f"Outside same-2D: {visual['outside_same_source_count']}")
    print(f"Outside uncertain: {visual['outside_uncertain_count']}")
    print(f"All acceptance gates passed: {gates['all_passed']}")
    print(f"Decision: {result['decision']}")
    print(f"Unblinded CSV: {output_csv}")
    print(f"Report: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
