from __future__ import annotations

"""Extract reproducible near-duplicate evidence from the development partition.

This is an audit/pilot utility, not a cancer classifier and not a train/validation/test
splitter.  It deliberately refuses to score holdout rows.  The output contains evidence
features only; it does not create augmentation families or freeze an automatic rule.
"""

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information, structural_similarity

from data_index import project_root
from render_calibration_review import read_dataset_root, resolve_image_path


REFERENCE_POSITIVE = "same_source"
REFERENCE_NEGATIVE = "independent"
REFERENCE_UNCERTAIN = "uncertain"

OUTPUT_FIELDS = (
    "pair_id",
    "audit_partition",
    "pair_kind",
    "truth_status",
    "human_verdict",
    "reference_label",
    "reference_basis",
    "left_leaf",
    "right_leaf",
    "modality_key",
    "source_group",
    "transform_name",
    "sha256_equal",
    "phash_distance",
    "analysis_size",
    "orb_orientation",
    "orb_keypoints_moving",
    "orb_keypoints_fixed",
    "orb_ratio_matches",
    "orb_inliers",
    "orb_inlier_ratio",
    "orb_coverage_min",
    "orb_affine_plausible",
    "orb_rotation_deg",
    "orb_scale_min",
    "orb_scale_max",
    "orb_translation_fraction",
    "orb_overlap",
    "orb_ssim",
    "orb_nmi",
    "orb_correlation",
    "phase_orientation",
    "phase_response",
    "phase_shift_fraction",
    "phase_overlap",
    "phase_ssim",
    "phase_nmi",
    "phase_correlation",
    "ecc_orientation",
    "ecc_attempted",
    "ecc_success",
    "ecc_score",
    "ecc_affine_plausible",
    "ecc_rotation_deg",
    "ecc_scale_min",
    "ecc_scale_max",
    "ecc_translation_fraction",
    "ecc_overlap",
    "ecc_ssim",
    "ecc_nmi",
    "ecc_correlation",
    "best_alignment_method",
    "best_alignment_orientation",
    "best_ssim",
    "best_nmi",
    "best_correlation",
    "best_overlap",
    "elapsed_seconds",
)

NUMERIC_DIAGNOSTIC_FIELDS = (
    "orb_ratio_matches",
    "orb_inliers",
    "orb_inlier_ratio",
    "orb_coverage_min",
    "orb_ssim",
    "orb_nmi",
    "orb_correlation",
    "phase_response",
    "phase_ssim",
    "phase_nmi",
    "phase_correlation",
    "ecc_score",
    "ecc_ssim",
    "ecc_nmi",
    "ecc_correlation",
    "best_ssim",
    "best_nmi",
    "best_correlation",
)

ECC_MODALITIES = frozenset({"mri", "ct", "pap_smear"})


def parse_args() -> argparse.Namespace:
    calibration_root = (
        project_root() / "artifacts" / "audit" / "duplicate_calibration_v1"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Yalnız development calibration çiftleri için pHash, ORB/RANSAC, "
            "phase correlation, ECC ve hizalanmış benzerlik kanıtlarını çıkarır. "
            "Holdout görüntülerini açmaz ve otomatik duplicate kararı vermez."
        )
    )
    parser.add_argument(
        "--calibration-root",
        default=str(calibration_root),
        help="Calibration-set v1 kökü.",
    )
    parser.add_argument(
        "--pairs-csv",
        default=str(calibration_root / "pairs.csv"),
        help="1.342 çiftlik ana tablo.",
    )
    parser.add_argument(
        "--review-csv",
        default=str(calibration_root / "review" / "review_labels.csv"),
        help="Development görsel kararlarını içeren tablo.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Veri kökü. Verilmezse calibration raporundaki kök kullanılır.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            project_root()
            / "artifacts"
            / "audit"
            / "duplicate_hybrid_development_v1"
        ),
        help="Development özellikleri ve raporu için çıktı klasörü.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Skor çıkarımı için kare analiz boyutu.",
    )
    parser.add_argument(
        "--orb-features",
        type=int,
        default=4000,
        help="ORB'nin arayacağı en yüksek anahtar nokta sayısı.",
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.78,
        help="ORB KNN eşleşmeleri için Lowe ratio sınırı.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="RANSAC affine inlier piksel toleransı (analiz boyutunda).",
    )
    parser.add_argument(
        "--ecc-iterations",
        type=int,
        default=100,
        help="MRI/CT/Pap için ECC en yüksek iterasyon sayısı.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="OpenCV RANSAC tekrar üretilebilirlik tohumu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan pair_scores.csv ve report.json dosyalarını atomik olarak yenile.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size < 64:
        raise ValueError("--image-size en az 64 olmalıdır.")
    if args.orb_features < 100:
        raise ValueError("--orb-features en az 100 olmalıdır.")
    if not 0.0 < args.ratio_test < 1.0:
        raise ValueError("--ratio-test 0 ile 1 arasında olmalıdır.")
    if args.ransac_threshold <= 0:
        raise ValueError("--ransac-threshold pozitif olmalıdır.")
    if args.ecc_iterations < 1:
        raise ValueError("--ecc-iterations en az 1 olmalıdır.")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV başlığı yok: {path}")
        return [dict(row) for row in reader]


def reference_label(row: Mapping[str, str]) -> tuple[str, str]:
    if row.get("truth_status", "").strip() == "same_source":
        return REFERENCE_POSITIVE, row.get("truth_basis", "known_positive")

    verdict = row.get("human_verdict", "").strip()
    if verdict == "same_source_candidate":
        return REFERENCE_POSITIVE, "ai_assisted_visual_review"
    if verdict == "independent_lookalike":
        return REFERENCE_NEGATIVE, "ai_assisted_visual_review"
    if verdict == "uncertain":
        return REFERENCE_UNCERTAIN, "ai_assisted_visual_review"
    raise ValueError(
        f"Development çifti için kullanılabilir karar yok: {row.get('pair_id', '?')}"
    )


def load_development_rows(
    pairs_csv: Path, review_csv: Path
) -> tuple[list[dict[str, str]], dict[str, int]]:
    all_pairs = read_csv(pairs_csv)
    reviews = read_csv(review_csv)
    review_by_id = {
        row["pair_id"]: row
        for row in reviews
        if row.get("audit_partition", "").strip() == "development"
    }

    rows: list[dict[str, str]] = []
    holdout_count = 0
    for source in all_pairs:
        partition = source.get("audit_partition", "").strip()
        if partition == "holdout":
            holdout_count += 1
            continue
        if partition != "development":
            raise ValueError(
                f"Bilinmeyen audit partition: {source.get('pair_id')}={partition!r}"
            )

        row = dict(source)
        if row.get("truth_status", "").strip() == "review_required":
            reviewed = review_by_id.get(row["pair_id"])
            if reviewed is None:
                raise ValueError(f"Development review satırı bulunamadı: {row['pair_id']}")
            row["human_verdict"] = reviewed.get("human_verdict", "").strip()
            row["review_status"] = reviewed.get("review_status", "").strip()

        label, basis = reference_label(row)
        row["reference_label"] = label
        row["reference_basis"] = basis
        rows.append(row)

    if not rows:
        raise ValueError("Development satırı bulunamadı.")
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Development Pair ID değerleri benzersiz değil.")

    audit = {
        "all_pair_rows": len(all_pairs),
        "development_rows": len(rows),
        "holdout_rows_not_loaded": holdout_count,
        "development_review_rows": len(review_by_id),
    }
    return rows, audit


def decode_gray(path: Path, size: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Görüntü decode edilemedi: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_LINEAR
    return cv2.resize(gray, (size, size), interpolation=interpolation)


def robust_normalize(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32) / 255.0
    low, high = np.percentile(values, (1.0, 99.0))
    if high - low < 1e-6:
        low = float(values.min())
        high = float(values.max())
    if high - low < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def safe_float(value: object, digits: int = 8) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def coverage_fraction(points: np.ndarray, size: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / float(size * size))


def affine_properties(matrix: np.ndarray | None, size: int) -> dict[str, float | bool]:
    blank = {
        "plausible": False,
        "rotation_deg": float("nan"),
        "scale_min": float("nan"),
        "scale_max": float("nan"),
        "translation_fraction": float("nan"),
    }
    if matrix is None or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return blank

    linear = matrix[:, :2].astype(np.float64)
    singular = np.linalg.svd(linear, compute_uv=False)
    scale_min = float(np.min(singular))
    scale_max = float(np.max(singular))
    rotation_deg = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    translation = matrix[:, 2].astype(np.float64)
    translation_fraction = float(np.linalg.norm(translation) / (math.sqrt(2.0) * size))
    determinant = float(np.linalg.det(linear))
    plausible = (
        0.65 <= scale_min <= 1.45
        and 0.65 <= scale_max <= 1.45
        and abs(determinant) >= 0.40
        and translation_fraction <= 0.30
    )
    return {
        "plausible": bool(plausible),
        "rotation_deg": rotation_deg,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "translation_fraction": translation_fraction,
    }


def aligned_similarity(
    fixed: np.ndarray, aligned: np.ndarray, valid_mask: np.ndarray
) -> dict[str, float]:
    mask = valid_mask.astype(np.uint8)
    mask = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    valid = mask > 0
    overlap = float(np.mean(valid))
    if np.count_nonzero(valid) < 256:
        return {
            "overlap": overlap,
            "ssim": float("nan"),
            "nmi": float("nan"),
            "correlation": float("nan"),
        }

    fixed_norm = robust_normalize(fixed)
    aligned_norm = robust_normalize(aligned)
    _, ssim_map = structural_similarity(
        fixed_norm, aligned_norm, data_range=1.0, full=True
    )
    ssim_value = float(np.mean(ssim_map[valid]))

    fixed_values = fixed_norm[valid]
    aligned_values = aligned_norm[valid]
    if np.std(fixed_values) < 1e-8 or np.std(aligned_values) < 1e-8:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(fixed_values, aligned_values)[0, 1])
    try:
        nmi = float(
            normalized_mutual_information(fixed_values, aligned_values, bins=64)
        )
    except (ValueError, ZeroDivisionError):
        nmi = float("nan")
    return {
        "overlap": overlap,
        "ssim": ssim_value,
        "nmi": nmi,
        "correlation": correlation,
    }


def warp_direct(moving: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = moving.shape[0]
    aligned = cv2.warpAffine(
        moving,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpAffine(
        np.ones_like(moving, dtype=np.uint8),
        matrix,
        (size, size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, valid


def score_orb(
    moving: np.ndarray,
    fixed: np.ndarray,
    *,
    features: int,
    ratio_test: float,
    ransac_threshold: float,
) -> dict[str, object]:
    size = moving.shape[0]
    orb = cv2.ORB_create(
        nfeatures=features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10,
    )
    moving_keypoints, moving_descriptors = orb.detectAndCompute(moving, None)
    fixed_keypoints, fixed_descriptors = orb.detectAndCompute(fixed, None)
    result: dict[str, object] = {
        "keypoints_moving": len(moving_keypoints),
        "keypoints_fixed": len(fixed_keypoints),
        "ratio_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "coverage_min": 0.0,
        "matrix": None,
        "alignment": {
            "overlap": 0.0,
            "ssim": float("nan"),
            "nmi": float("nan"),
            "correlation": float("nan"),
        },
    }
    if moving_descriptors is None or fixed_descriptors is None:
        return result
    if len(moving_descriptors) < 2 or len(fixed_descriptors) < 2:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    candidates = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
    good = [
        first
        for match in candidates
        if len(match) == 2
        for first, second in [match]
        if first.distance < ratio_test * second.distance
    ]
    result["ratio_matches"] = len(good)
    if len(good) < 3:
        return result

    moving_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in good]
    )
    fixed_points = np.float32(
        [fixed_keypoints[match.trainIdx].pt for match in good]
    )
    matrix, inlier_mask = cv2.estimateAffine2D(
        moving_points,
        fixed_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=5000,
        confidence=0.995,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return result
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    result["inliers"] = inlier_count
    result["inlier_ratio"] = inlier_count / len(good)
    result["matrix"] = matrix.astype(np.float32)
    if inlier_count >= 3:
        moving_coverage = coverage_fraction(moving_points[inliers], size)
        fixed_coverage = coverage_fraction(fixed_points[inliers], size)
        result["coverage_min"] = min(moving_coverage, fixed_coverage)
        aligned, valid = warp_direct(moving, result["matrix"])
        result["alignment"] = aligned_similarity(fixed, aligned, valid)
    return result


def score_phase(moving: np.ndarray, fixed: np.ndarray) -> dict[str, object]:
    size = moving.shape[0]
    moving_norm = robust_normalize(moving)
    fixed_norm = robust_normalize(fixed)
    window = cv2.createHanningWindow((size, size), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        moving_norm, fixed_norm, window
    )
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    aligned, valid = warp_direct(moving, matrix)
    return {
        "matrix": matrix,
        "response": float(response),
        "shift_fraction": float(math.hypot(dx, dy) / (math.sqrt(2.0) * size)),
        "alignment": aligned_similarity(fixed, aligned, valid),
    }


def score_ecc(
    moving: np.ndarray,
    fixed: np.ndarray,
    phase_matrix: np.ndarray,
    *,
    iterations: int,
) -> dict[str, object]:
    moving_norm = robust_normalize(moving)
    fixed_norm = robust_normalize(fixed)
    # phase_matrix maps moving -> fixed; ECC expects the inverse-map convention.
    warp = cv2.invertAffineTransform(phase_matrix).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        iterations,
        1e-6,
    )
    result: dict[str, object] = {
        "success": False,
        "score": float("nan"),
        "matrix": None,
        "alignment": {
            "overlap": 0.0,
            "ssim": float("nan"),
            "nmi": float("nan"),
            "correlation": float("nan"),
        },
    }
    try:
        score, inverse_matrix = cv2.findTransformECC(
            fixed_norm,
            moving_norm,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return result

    size = moving.shape[0]
    aligned = cv2.warpAffine(
        moving,
        inverse_matrix,
        (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpAffine(
        np.ones_like(moving, dtype=np.uint8),
        inverse_matrix,
        (size, size),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    direct_matrix = cv2.invertAffineTransform(inverse_matrix).astype(np.float32)
    result.update(
        {
            "success": True,
            "score": float(score),
            "matrix": direct_matrix,
            "alignment": aligned_similarity(fixed, aligned, valid),
        }
    )
    return result


def choose_best(
    candidates: Iterable[tuple[str, Mapping[str, object]]],
    primary: str,
    secondary: str | None = None,
) -> tuple[str, Mapping[str, object]]:
    def value(payload: Mapping[str, object], key: str) -> float:
        raw = payload.get(key)
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("-inf")
        return number if math.isfinite(number) else float("-inf")

    ranked = list(candidates)
    if not ranked:
        raise ValueError("Karşılaştırılacak aday yok.")
    return max(
        ranked,
        key=lambda item: (
            value(item[1], primary),
            value(item[1], secondary) if secondary else 0.0,
        ),
    )


def score_pair(
    row: Mapping[str, str],
    left_path: Path,
    right_path: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    started = time.perf_counter()
    left = decode_gray(left_path, args.image_size)
    fixed = decode_gray(right_path, args.image_size)
    variants = {"original": left, "horizontal_flip": cv2.flip(left, 1)}

    orb_results: list[tuple[str, Mapping[str, object]]] = []
    phase_results: list[tuple[str, Mapping[str, object]]] = []
    ecc_results: list[tuple[str, Mapping[str, object]]] = []
    alignment_candidates: list[tuple[str, str, Mapping[str, float]]] = []

    for orientation, moving in variants.items():
        orb = score_orb(
            moving,
            fixed,
            features=args.orb_features,
            ratio_test=args.ratio_test,
            ransac_threshold=args.ransac_threshold,
        )
        orb_results.append((orientation, orb))
        orb_alignment = orb["alignment"]
        alignment_candidates.append(
            ("orb", orientation, orb_alignment)  # type: ignore[arg-type]
        )

        phase = score_phase(moving, fixed)
        phase_results.append((orientation, phase))
        phase_alignment = phase["alignment"]
        alignment_candidates.append(
            ("phase", orientation, phase_alignment)  # type: ignore[arg-type]
        )

        if row["modality_key"] in ECC_MODALITIES:
            ecc = score_ecc(
                moving,
                fixed,
                phase["matrix"],  # type: ignore[arg-type]
                iterations=args.ecc_iterations,
            )
            ecc_results.append((orientation, ecc))
            ecc_alignment = ecc["alignment"]
            alignment_candidates.append(
                ("ecc", orientation, ecc_alignment)  # type: ignore[arg-type]
            )

    orb_orientation, orb = choose_best(
        orb_results, "inliers", "inlier_ratio"
    )
    phase_orientation, phase = choose_best(phase_results, "response")
    if ecc_results:
        ecc_orientation, ecc = choose_best(ecc_results, "score")
    else:
        ecc_orientation = ""
        ecc = {
            "success": False,
            "score": float("nan"),
            "matrix": None,
            "alignment": {
                "overlap": float("nan"),
                "ssim": float("nan"),
                "nmi": float("nan"),
                "correlation": float("nan"),
            },
        }

    usable_alignments = [
        (method, orientation, metrics)
        for method, orientation, metrics in alignment_candidates
        if math.isfinite(float(metrics.get("ssim", float("nan"))))
    ]
    if usable_alignments:
        best_method, best_orientation, best_alignment = max(
            usable_alignments,
            key=lambda item: (
                float(item[2]["ssim"]),
                float(item[2].get("correlation", float("-inf"))),
            ),
        )
    else:
        best_method, best_orientation = "", ""
        best_alignment = {
            "overlap": float("nan"),
            "ssim": float("nan"),
            "nmi": float("nan"),
            "correlation": float("nan"),
        }

    orb_properties = affine_properties(
        orb.get("matrix"), args.image_size  # type: ignore[arg-type]
    )
    ecc_properties = affine_properties(
        ecc.get("matrix"), args.image_size  # type: ignore[arg-type]
    )
    orb_alignment = orb["alignment"]  # type: ignore[assignment]
    phase_alignment = phase["alignment"]  # type: ignore[assignment]
    ecc_alignment = ecc["alignment"]  # type: ignore[assignment]

    return {
        "pair_id": row["pair_id"],
        "audit_partition": row["audit_partition"],
        "pair_kind": row["pair_kind"],
        "truth_status": row["truth_status"],
        "human_verdict": row.get("human_verdict", ""),
        "reference_label": row["reference_label"],
        "reference_basis": row["reference_basis"],
        "left_leaf": row["left_leaf"],
        "right_leaf": row["right_leaf"],
        "modality_key": row["modality_key"],
        "source_group": row["source_group"],
        "transform_name": row.get("transform_name", ""),
        "sha256_equal": row.get("sha256_equal", ""),
        "phash_distance": row.get("phash_distance", ""),
        "analysis_size": str(args.image_size),
        "orb_orientation": orb_orientation,
        "orb_keypoints_moving": str(orb["keypoints_moving"]),
        "orb_keypoints_fixed": str(orb["keypoints_fixed"]),
        "orb_ratio_matches": str(orb["ratio_matches"]),
        "orb_inliers": str(orb["inliers"]),
        "orb_inlier_ratio": safe_float(orb["inlier_ratio"]),
        "orb_coverage_min": safe_float(orb["coverage_min"]),
        "orb_affine_plausible": "yes" if orb_properties["plausible"] else "no",
        "orb_rotation_deg": safe_float(orb_properties["rotation_deg"]),
        "orb_scale_min": safe_float(orb_properties["scale_min"]),
        "orb_scale_max": safe_float(orb_properties["scale_max"]),
        "orb_translation_fraction": safe_float(
            orb_properties["translation_fraction"]
        ),
        "orb_overlap": safe_float(orb_alignment["overlap"]),
        "orb_ssim": safe_float(orb_alignment["ssim"]),
        "orb_nmi": safe_float(orb_alignment["nmi"]),
        "orb_correlation": safe_float(orb_alignment["correlation"]),
        "phase_orientation": phase_orientation,
        "phase_response": safe_float(phase["response"]),
        "phase_shift_fraction": safe_float(phase["shift_fraction"]),
        "phase_overlap": safe_float(phase_alignment["overlap"]),
        "phase_ssim": safe_float(phase_alignment["ssim"]),
        "phase_nmi": safe_float(phase_alignment["nmi"]),
        "phase_correlation": safe_float(phase_alignment["correlation"]),
        "ecc_orientation": ecc_orientation,
        "ecc_attempted": "yes" if ecc_results else "no",
        "ecc_success": "yes" if ecc["success"] else "no",
        "ecc_score": safe_float(ecc["score"]),
        "ecc_affine_plausible": "yes" if ecc_properties["plausible"] else "no",
        "ecc_rotation_deg": safe_float(ecc_properties["rotation_deg"]),
        "ecc_scale_min": safe_float(ecc_properties["scale_min"]),
        "ecc_scale_max": safe_float(ecc_properties["scale_max"]),
        "ecc_translation_fraction": safe_float(
            ecc_properties["translation_fraction"]
        ),
        "ecc_overlap": safe_float(ecc_alignment["overlap"]),
        "ecc_ssim": safe_float(ecc_alignment["ssim"]),
        "ecc_nmi": safe_float(ecc_alignment["nmi"]),
        "ecc_correlation": safe_float(ecc_alignment["correlation"]),
        "best_alignment_method": best_method,
        "best_alignment_orientation": best_orientation,
        "best_ssim": safe_float(best_alignment["ssim"]),
        "best_nmi": safe_float(best_alignment["nmi"]),
        "best_correlation": safe_float(best_alignment["correlation"]),
        "best_overlap": safe_float(best_alignment["overlap"]),
        "elapsed_seconds": safe_float(time.perf_counter() - started, digits=4),
    }


def numeric_values(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "q05": None,
            "median": None,
            "q95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def best_single_feature_threshold(
    rows: Sequence[Mapping[str, str]], field: str
) -> dict[str, object] | None:
    samples: list[tuple[float, int]] = []
    for row in rows:
        label = row["reference_label"]
        if label not in {REFERENCE_POSITIVE, REFERENCE_NEGATIVE}:
            continue
        raw = row.get(field, "").strip()
        if not raw:
            continue
        value = float(raw)
        if math.isfinite(value):
            samples.append((value, 1 if label == REFERENCE_POSITIVE else 0))
    if not samples or len({label for _, label in samples}) < 2:
        return None

    unique = sorted({value for value, _ in samples})
    thresholds = [unique[0] - 1e-12]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1e-12)
    best: dict[str, object] | None = None
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for value, label in samples:
            predicted = value >= threshold
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif not predicted and not label:
                tn += 1
            else:
                fn += 1
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        balanced_accuracy = (recall + specificity) / 2.0
        candidate = {
            "direction": "greater_or_equal_means_same_source",
            "threshold": float(threshold),
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sample_count": len(samples),
        }
        if best is None or (
            balanced_accuracy,
            precision,
            recall,
        ) > (
            float(best["balanced_accuracy"]),
            float(best["precision"]),
            float(best["recall"]),
        ):
            best = candidate
    return best


def count_nested(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "") or "(blank)" for row in rows).items()))


def build_report(
    rows: Sequence[Mapping[str, str]],
    input_audit: Mapping[str, int],
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> dict[str, object]:
    labelled = [
        row
        for row in rows
        if row["reference_label"] in {REFERENCE_POSITIVE, REFERENCE_NEGATIVE}
    ]
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["reference_label"]].append(row)

    feature_distributions: dict[str, object] = {}
    threshold_diagnostics: dict[str, object] = {}
    for field in NUMERIC_DIAGNOSTIC_FIELDS:
        feature_distributions[field] = {
            label: distribution(numeric_values(group, field))
            for label, group in sorted(by_label.items())
        }
        diagnostic = best_single_feature_threshold(labelled, field)
        if diagnostic is not None:
            threshold_diagnostics[field] = diagnostic

    by_modality: dict[str, object] = {}
    for modality in sorted({row["modality_key"] for row in rows}):
        group = [row for row in rows if row["modality_key"] == modality]
        by_modality[modality] = {
            "count": len(group),
            "by_reference_label": count_nested(group, "reference_label"),
            "best_single_feature_thresholds": {
                field: diagnostic
                for field in NUMERIC_DIAGNOSTIC_FIELDS
                if (diagnostic := best_single_feature_threshold(group, field)) is not None
            },
        }

    return {
        "schema_version": 1,
        "status": "development_feature_extraction_only_not_frozen",
        "experiment_type": "near_duplicate_development_pilot",
        "holdout_policy": "holdout_rows_counted_from_metadata_but_images_not_loaded_or_scored",
        "input_audit": dict(input_audit),
        "scored_pair_count": len(rows),
        "evaluation_pair_count_excluding_uncertain": len(labelled),
        "by_reference_label": count_nested(rows, "reference_label"),
        "by_pair_kind": count_nested(rows, "pair_kind"),
        "by_modality": by_modality,
        "by_source_group": count_nested(rows, "source_group"),
        "by_transform": count_nested(rows, "transform_name"),
        "ecc_attempted_count": sum(row["ecc_attempted"] == "yes" for row in rows),
        "ecc_success_count": sum(row["ecc_success"] == "yes" for row in rows),
        "feature_distributions": feature_distributions,
        "single_feature_threshold_diagnostics": threshold_diagnostics,
        "parameters": {
            "image_size": args.image_size,
            "orb_features": args.orb_features,
            "ratio_test": args.ratio_test,
            "ransac_threshold": args.ransac_threshold,
            "ecc_iterations": args.ecc_iterations,
            "seed": args.seed,
            "ecc_modalities": sorted(ECC_MODALITIES),
            "orientation_variants": ["original", "horizontal_flip"],
        },
        "elapsed_seconds": elapsed_seconds,
        "limitations": [
            "Single-feature thresholds are development diagnostics, not frozen rules.",
            "AI-assisted visual labels are not an independent expert-human gold standard.",
            "The uncertain C000620 pair is excluded from positive/negative threshold metrics.",
            "No holdout image was loaded; no family or classifier split was created.",
        ],
    }


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
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
    validate_args(args)
    calibration_root = Path(args.calibration_root).resolve()
    pairs_csv = Path(args.pairs_csv).resolve()
    review_csv = Path(args.review_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    score_path = output_dir / "pair_scores.csv"
    report_path = output_dir / "report.json"
    if not args.overwrite and (score_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Çıktı zaten var. Bilinçli yenileme için --overwrite kullanın: "
            f"{output_dir}"
        )

    if args.data_dir:
        dataset_root = Path(args.data_dir).resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Veri kökü bulunamadı: {dataset_root}")
    else:
        dataset_root = read_dataset_root(calibration_root)

    rows, input_audit = load_development_rows(pairs_csv, review_csv)
    print("İşlem türü: hybrid duplicate DEVELOPMENT pilotu (model/split/family yok)")
    print(f"Development çiftleri: {len(rows)}")
    print(f"Holdout: {input_audit['holdout_rows_not_loaded']} satır; görüntüler açılmayacak")
    print(f"Analiz boyutu: {args.image_size}x{args.image_size}")
    print(f"ECC modaliteleri: {', '.join(sorted(ECC_MODALITIES))}")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    started = time.perf_counter()
    scores: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        left_path = resolve_image_path(row, "left", dataset_root, calibration_root)
        right_path = resolve_image_path(row, "right", dataset_root, calibration_root)
        scores.append(score_pair(row, left_path, right_path, args))
        if index % 25 == 0 or index == len(rows):
            print(f"Skorlandı: {index}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    report = build_report(scores, input_audit, args, elapsed)
    atomic_write_csv(score_path, scores)
    atomic_write_json(report_path, report)
    print(f"Skor tablosu: {score_path}")
    print(f"Rapor: {report_path}")
    print("NOT: Eşik ve otomatik karar henüz dondurulmadı; holdout kapalı kaldı.")
    return score_path, report_path, report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
