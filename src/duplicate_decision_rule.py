from __future__ import annotations

"""Apply the frozen, explainable near-duplicate decision rule to scored pairs.

This module does not open images and does not create train/validation/test splits.
It converts already-computed evidence into one of three conservative decisions:
same_source, independent, or uncertain.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


DECISIONS = {"same_source", "independent", "uncertain"}
REQUIRED_RULES = {
    "sha256_exact",
    "strong_geometry",
    "strong_intensity_structure",
    "conservative_independent",
    "fallback",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rule(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        rule = json.load(handle)
    if rule.get("schema_version") != 1:
        raise ValueError("Desteklenmeyen duplicate rule schema_version")
    if rule.get("status") != "frozen_before_holdout":
        raise ValueError("Kural holdout oncesinde dondurulmus degil")
    rules = rule.get("rules")
    if not isinstance(rules, dict) or set(rules) != REQUIRED_RULES:
        raise ValueError("Duplicate rule bolumleri eksik veya beklenmeyen bolum var")
    for name, section in rules.items():
        if not isinstance(section, dict) or section.get("decision") not in DECISIONS:
            raise ValueError(f"Gecersiz karar bolumu: {name}")
    return rule


def load_frozen_rule(path: Path) -> tuple[dict[str, object], str]:
    """Load a rule only when its adjacent SHA-256 sidecar still matches."""
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Kural SHA-256 dosyasi bulunamadi: {sidecar}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError("Kural SHA-256 dosyasi bos")
    expected = tokens[0].lower()
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(
            f"Dondurulmus kural hash uyusmazligi: expected={expected}, actual={actual}"
        )
    return load_rule(path), actual


def _float(row: Mapping[str, str], field: str) -> float | None:
    raw = str(row.get(field, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _all_available(values: Mapping[str, float | None]) -> bool:
    return all(value is not None for value in values.values())


def classify_pair(row: Mapping[str, str], rule: Mapping[str, object]) -> dict[str, str]:
    rules = rule["rules"]
    assert isinstance(rules, dict)

    exact_rule = rules["sha256_exact"]
    geometry_rule = rules["strong_geometry"]
    intensity_rule = rules["strong_intensity_structure"]
    independent_rule = rules["conservative_independent"]
    fallback_rule = rules["fallback"]
    assert isinstance(exact_rule, dict)
    assert isinstance(geometry_rule, dict)
    assert isinstance(intensity_rule, dict)
    assert isinstance(independent_rule, dict)
    assert isinstance(fallback_rule, dict)

    metrics = {
        "orb_inliers": _float(row, "orb_inliers"),
        "orb_inlier_ratio": _float(row, "orb_inlier_ratio"),
        "orb_coverage_min": _float(row, "orb_coverage_min"),
        "best_ssim": _float(row, "best_ssim"),
        "best_correlation": _float(row, "best_correlation"),
        "best_nmi": _float(row, "best_nmi"),
    }

    exact = str(row.get("sha256_equal", "")).strip().lower() == str(
        exact_rule["sha256_equal"]
    ).lower()

    geometry_values = {
        key: metrics[key]
        for key in ("orb_inliers", "orb_inlier_ratio", "orb_coverage_min")
    }
    strong_geometry = (
        _all_available(geometry_values)
        and str(row.get("orb_affine_plausible", "")).strip().lower()
        == str(geometry_rule["orb_affine_plausible"]).lower()
        and metrics["orb_inliers"] >= float(geometry_rule["orb_inliers_min"])
        and metrics["orb_inlier_ratio"]
        >= float(geometry_rule["orb_inlier_ratio_min"])
        and metrics["orb_coverage_min"] >= float(geometry_rule["orb_coverage_min"])
    )

    intensity_values = {
        key: metrics[key]
        for key in ("best_ssim", "best_correlation", "best_nmi")
    }
    strong_intensity = (
        _all_available(intensity_values)
        and metrics["best_ssim"] >= float(intensity_rule["best_ssim_min"])
        and metrics["best_correlation"]
        >= float(intensity_rule["best_correlation_min"])
        and metrics["best_nmi"] >= float(intensity_rule["best_nmi_min"])
    )

    independent_values = {
        "orb_inliers": metrics["orb_inliers"],
        "best_ssim": metrics["best_ssim"],
        "best_correlation": metrics["best_correlation"],
    }
    conservative_independent = (
        _all_available(independent_values)
        and metrics["orb_inliers"] <= float(independent_rule["orb_inliers_max"])
        and metrics["best_ssim"] <= float(independent_rule["best_ssim_max"])
        and metrics["best_correlation"]
        <= float(independent_rule["best_correlation_max"])
    )

    if exact:
        decision = str(exact_rule["decision"])
        path = "sha256_exact"
        reason = "Dosya icerigi SHA-256 ile birebir ayni."
    elif strong_geometry:
        decision = str(geometry_rule["decision"])
        path = "strong_geometry"
        reason = "ORB eslesmeleri tek ve yeterince kapsayici bir affine donusumu destekliyor."
    elif strong_intensity:
        decision = str(intensity_rule["decision"])
        path = "strong_intensity_structure"
        reason = "Hizalama sonrasi SSIM, korelasyon ve NMI birlikte guclu."
    elif conservative_independent:
        decision = str(independent_rule["decision"])
        path = "conservative_independent"
        reason = "Geometrik ve hizalanmis yapisal kanitlar birlikte dusuk."
    else:
        decision = str(fallback_rule["decision"])
        path = "uncertain"
        reason = "Kanitlar same_source veya independent icin birlikte yeterli degil."

    return {
        "decision": decision,
        "decision_path": path,
        "decision_reason": reason,
        "flag_sha256_exact": "yes" if exact else "no",
        "flag_strong_geometry": "yes" if strong_geometry else "no",
        "flag_strong_intensity_structure": "yes" if strong_intensity else "no",
        "flag_conservative_independent": "yes" if conservative_independent else "no",
    }
