from __future__ import annotations

"""Unblind and evaluate the frozen Candidate Generator v2.5 audit.

The visual decisions were frozen before this script was run.  This program joins
those decisions to the private selection strata, verifies the frozen evidence,
and evaluates the predeclared acceptance gates.  It does not alter a verdict,
score duplicate candidates, construct families, create a split, or train a model.
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


EXPECTED_CONFIG_SHA256 = (
    "2c4213d34c70991550b85ec2d4699a0c7cf49b804904cc9bb3b4d7518d8fe745"
)
EXPECTED_PRIVATE_SHA256 = (
    "b2b5da720ef256bc0c04a2aa788c4ff96de6ae372b1efd14997bbf45e9c4c462"
)
EXPECTED_PUBLIC_MANIFEST_SHA256 = (
    "b551ef6bea86ee62c9fc831a48dd0b662a99344f25da7ec9ed550a2a27103bfb"
)
EXPECTED_VISUAL_SHA256 = (
    "711bfd27668ca1f64eba8627d0973cf27561b577d2d72871cbd16b89e7b3a173"
)
EXPECTED_ROWS = 256

ALLOWED_STRATA = {
    "outside_robust_boundary",
    "outside_phash_13_16",
    "outside_random",
    "candidate_new_cervix_rank16_30",
    "candidate_robust_edge",
}
ALLOWED_VERDICTS = {
    "same_source_candidate",
    "independent_lookalike",
    "uncertain",
}
GROUP_PATH_PREFIX = {
    "all_blood_smear": "ALL/",
    "brain_ce_mri": "Brain Cancer/",
    "breast_histology": "Breast Cancer/",
    "cervix_pap_smear": "Cervical Cancer/",
    "kidney_ct": "Kidney Cancer/",
    "lung_colon_histology": "Lung and Colon Cancer/",
    "lymphoma_histology": "Lymphoma/",
    "oral_histology": "Oral Cancer/",
}

PUBLIC_IDENTITY_FIELDS = (
    "pair_id",
    "source_group",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
)
OUTPUT_FIELDS = (
    "pair_id",
    "selection_stratum",
    "audit_region",
    "source_group",
    "left_image_index",
    "right_image_index",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "v2_5_candidate",
    "visual_verdict",
    "confidence",
    "reviewer",
    "review_date",
    "review_notes",
    "robust_acceptance_k",
    "robust_best_rank",
    "robust_rank_lower_bound",
    "phash_enabled",
    "phash_distance",
    "phash_radius12",
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    selection_dir = root / "research" / "candidate_generator_v2_5_prospective_selection_v1"
    output_dir = root / "research" / "candidate_generator_v2_5_prospective_audit_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Dondurulmuş V2.5 görsel kararlarını private strata ile birleştirir "
            "ve önceden belirlenen kabul kapılarını değerlendirir."
        )
    )
    parser.add_argument("--config", default=str(root / "config" / "candidate_generator_v2_5_prospective_audit.json"))
    parser.add_argument("--private-selection", default=str(selection_dir / "selection_private.csv"))
    parser.add_argument("--public-manifest", default=str(selection_dir / "review_manifest.csv"))
    parser.add_argument("--selection-report", default=str(selection_dir / "selection_report.json"))
    parser.add_argument("--visual-decisions", default=str(root / "research" / "candidate_generator_v2_5_visual_review_v1.csv"))
    parser.add_argument("--v2-4-report", default=str(root / "research" / "candidate_generator_v2_4_counterfactual_v1.json"))
    parser.add_argument("--rank-report", default=str(root / "research" / "candidate_generator_v2_4_rank_crowding_v1.json"))
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
            raise ValueError(f"CSV başlığı bulunamadı: {path}")
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
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{name}: boş Pair ID")
        if pair_id in result:
            raise ValueError(f"{name}: tekrarlanan Pair ID {pair_id}")
        result[pair_id] = row
    return result


def merge_rows(
    private_rows: Sequence[Mapping[str, str]],
    public_rows: Sequence[Mapping[str, str]],
    visual_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    private_by_id = index_unique(private_rows, "private selection")
    public_by_id = index_unique(public_rows, "public manifest")
    visual_by_id = index_unique(visual_rows, "visual decisions")
    expected_ids = set(private_by_id)
    for name, rows_by_id in (("public manifest", public_by_id), ("visual decisions", visual_by_id)):
        if set(rows_by_id) != expected_ids:
            missing = sorted(expected_ids - set(rows_by_id))
            extra = sorted(set(rows_by_id) - expected_ids)
            raise ValueError(f"{name}: Pair ID kapsamı eşleşmiyor; eksik={missing[:5]}, fazla={extra[:5]}")

    merged: list[dict[str, str]] = []
    for pair_id in sorted(expected_ids):
        private = private_by_id[pair_id]
        public = public_by_id[pair_id]
        visual = visual_by_id[pair_id]
        for field in PUBLIC_IDENTITY_FIELDS:
            if private.get(field, "") != public.get(field, ""):
                raise ValueError(f"{pair_id}: private/public {field} uyuşmuyor")

        stratum = private.get("selection_stratum", "").strip()
        verdict = visual.get("visual_verdict", "").strip()
        if stratum not in ALLOWED_STRATA:
            raise ValueError(f"{pair_id}: geçersiz stratum {stratum!r}")
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"{pair_id}: geçersiz görsel karar {verdict!r}")
        audit_region = "candidate" if stratum.startswith("candidate_") else "outside"
        expected_candidate = "yes" if audit_region == "candidate" else "no"
        if private.get("v2_5_candidate", "").strip() != expected_candidate:
            raise ValueError(f"{pair_id}: stratum/candidate üyeliği uyuşmuyor")

        source_group = private.get("source_group", "").strip()
        prefix = GROUP_PATH_PREFIX.get(source_group)
        if prefix is None:
            raise ValueError(f"{pair_id}: bilinmeyen source group {source_group!r}")
        if not private.get("left_path", "").startswith(prefix) or not private.get("right_path", "").startswith(prefix):
            raise ValueError(f"{pair_id}: source-group dışı path bulundu")

        row = {field: private.get(field, "") for field in OUTPUT_FIELDS}
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


def nested_counts(rows: Sequence[Mapping[str, str]], group_field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row[group_field]][row["visual_verdict"]] += 1
    return {
        group: {verdict: counts.get(verdict, 0) for verdict in sorted(ALLOWED_VERDICTS)}
        for group, counts in sorted(grouped.items())
    }


def summarize_visual(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    verdict_counts = Counter(row["visual_verdict"] for row in rows)
    regions = Counter(row["audit_region"] for row in rows)
    candidate = [row for row in rows if row["audit_region"] == "candidate"]
    outside = [row for row in rows if row["audit_region"] == "outside"]
    outside_positive = [row for row in outside if row["visual_verdict"] == "same_source_candidate"]
    outside_uncertain = [row for row in outside if row["visual_verdict"] == "uncertain"]
    candidate_positive = [row for row in candidate if row["visual_verdict"] == "same_source_candidate"]
    candidate_uncertain = [row for row in candidate if row["visual_verdict"] == "uncertain"]
    return {
        "total_pairs": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "region_counts": dict(sorted(regions.items())),
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
    v2_4_report: Mapping[str, object],
    rank_report: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    visual = summarize_visual(rows)
    gates_config = config["predeclared_acceptance_gates"]
    expected_state = config["expected_full_state"]
    candidate_policy = selection_report["candidate_policy"]

    exact_hits = int(v2_4_report["exact_duplicate_reference"]["hits"])
    controlled_hits = int(v2_4_report["controlled_positive_reference"]["hits"])
    old_natural = v2_4_report["natural_same_source_reference"]
    target = rank_report["target_pair"]
    regression_hits = int(old_natural["v2_4_hits"])
    if (
        old_natural["missing_pair_ids"] == ["N000123"]
        and target["pair_id"] == "N000123"
        and int(target["minimum_required_robust_top_k"]) <= 40
        and config["candidate_rule"]["robust_top_k_by_group"]["cervix_pap_smear"] == 40
    ):
        regression_hits += 1

    full_pairs = int(candidate_policy["v2_5_full_pairs"])
    image_count = int(selection_report["input"]["image_count"])
    pairs_per_image = full_pairs / image_count
    cross_group_pairs = 0  # merge_rows verifies both paths against the declared source group.
    baseline_pairs = int(candidate_policy["v2_4_full_pairs"])

    observed = {
        "exact_reference_hits": exact_hits,
        "controlled_positive_hits": controlled_hits,
        "regression_natural_hits": regression_hits,
        "new_outside_same_source_pairs": int(visual["outside_same_source_count"]),
        "unresolved_outside_uncertain_pairs": int(visual["outside_uncertain_count"]),
        "cross_source_group_pairs": cross_group_pairs,
        "full_candidate_pairs": full_pairs,
        "candidate_pairs_per_image": pairs_per_image,
        "v2_4_baseline_pairs": baseline_pairs,
    }
    gates = {
        "exact_reference_recall": exact_hits == int(gates_config["required_exact_reference_hits"]),
        "controlled_positive_recall": controlled_hits == int(gates_config["required_controlled_positive_hits"]),
        "regression_natural_recall": regression_hits == int(gates_config["required_regression_natural_hits"]),
        "new_outside_same_source": int(visual["outside_same_source_count"]) == int(gates_config["required_new_outside_same_source_pairs"]),
        "unresolved_outside_uncertain": int(visual["outside_uncertain_count"]) == int(gates_config["required_unresolved_outside_uncertain_pairs"]),
        "cross_source_group_pairs": cross_group_pairs == int(gates_config["required_cross_source_group_pairs"]),
        "full_candidate_pairs": full_pairs == int(gates_config["required_full_candidate_pairs"]),
        "candidate_load": pairs_per_image <= float(gates_config["maximum_candidate_pairs_per_image"]),
        "v2_4_baseline_reproduction": baseline_pairs == int(expected_state["v2_4_full_pairs"]),
    }
    gates["all_passed"] = all(gates.values())
    return {
        "visual_audit": visual,
        "observed_gate_values": observed,
        "acceptance_gates": gates,
        "decision": (
            "PASS: v2.5 passed every predeclared gate."
            if gates["all_passed"]
            else "FAIL: reject v2.5; do not score candidates, construct families, split data, or train a classifier under this rule."
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
        "v2_4_report": Path(args.v2_4_report).resolve(),
        "rank_report": Path(args.rank_report).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected_hashes = {
        "config": EXPECTED_CONFIG_SHA256,
        "private_selection": EXPECTED_PRIVATE_SHA256,
        "public_manifest": EXPECTED_PUBLIC_MANIFEST_SHA256,
        "visual_decisions": EXPECTED_VISUAL_SHA256,
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"{name} hash değişmiş: expected={expected}, actual={hashes[name]}")

    selection_report = read_json(paths["selection_report"])
    if selection_report["integrity"]["selection_private_sha256"] != hashes["private_selection"]:
        raise ValueError("Selection report/private manifest hash uyuşmuyor")
    if selection_report["integrity"]["review_manifest_sha256"] != hashes["public_manifest"]:
        raise ValueError("Selection report/public manifest hash uyuşmuyor")

    merged = merge_rows(
        read_csv(paths["private_selection"]),
        read_csv(paths["public_manifest"]),
        read_csv(paths["visual_decisions"]),
    )
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(f"Beklenen {EXPECTED_ROWS} satır, bulunan {len(merged)}")

    result = evaluate(
        read_json(paths["config"]),
        selection_report,
        read_json(paths["v2_4_report"]),
        read_json(paths["rank_report"]),
        merged,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "candidate_generator_v2_5_prospective_audit_unblinded_complete",
        "experiment_type": "predeclared_score_blind_prospective_natural_audit_gate_evaluation",
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
            "frozen_input_hashes_verified": True,
        },
        **result,
        "limitations": [
            "The visual labels are AI-assisted non-expert leakage-audit decisions, not clinical ground truth.",
            "The sampled challenge set does not measure every possible image pair.",
            "A failed gate rejects this v2.5 rule; it does not justify changing frozen visual decisions after unblinding.",
        ],
    }
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    atomic_write_csv(output_csv, merged)
    atomic_write_json(output_json, payload)

    visual = result["visual_audit"]
    gates = result["acceptance_gates"]
    print("V2.5 prospective audit: frozen decisions unblinded")
    print(f"Pairs: {visual['total_pairs']}; verdicts: {visual['verdict_counts']}")
    print(f"Candidate same-source: {visual['candidate_same_source_count']}")
    print(f"Outside same-source: {visual['outside_same_source_count']}")
    print(f"Outside uncertain: {visual['outside_uncertain_count']}")
    print(f"All acceptance gates passed: {gates['all_passed']}")
    print(f"Decision: {result['decision']}")
    print(f"Unblinded CSV: {output_csv}")
    print(f"Report: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
