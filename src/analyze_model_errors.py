from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_review_package(review_package: Path) -> tuple[dict, list[dict]]:
    with zipfile.ZipFile(review_package) as archive:
        names = set(archive.namelist())
        required = {"test_metrics.json", "test_predictions.csv"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Review paketinde zorunlu dosyalar eksik: {missing}")
        metrics = json.loads(archive.read("test_metrics.json").decode("utf-8"))
        predictions_text = archive.read("test_predictions.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(predictions_text, newline="")))
    return metrics, rows


def confidence_bin_index(confidence: float, bin_count: int) -> int:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence [0,1] disinda: {confidence}")
    return min(int(confidence * bin_count), bin_count - 1)


def compute_calibration(
    rows: Sequence[dict], class_names: Sequence[str], bin_count: int
) -> tuple[dict, list[dict]]:
    if not rows or bin_count <= 0:
        raise ValueError("Kalibrasyon icin bos olmayan satirlar ve pozitif bin sayisi gerekir")
    class_set = set(class_names)
    bins = [
        {"count": 0, "correct": 0, "confidence_sum": 0.0}
        for _ in range(bin_count)
    ]
    nll_sum = 0.0
    brier_sum = 0.0
    top2_correct = 0
    correct_confidences: list[float] = []
    error_confidences: list[float] = []

    for row in rows:
        probabilities = json.loads(row["probabilities_json"])
        if set(probabilities) != class_set:
            raise ValueError("Probability sinif anahtarlari metrics class_names ile eslesmiyor")
        probability_values = {name: float(probabilities[name]) for name in class_names}
        if any(not 0.0 <= value <= 1.0 for value in probability_values.values()):
            raise ValueError("Probability [0,1] disinda")
        if abs(sum(probability_values.values()) - 1.0) > 1e-5:
            raise ValueError("Probability toplami 1 degil")
        true_leaf = row["true_leaf"]
        predicted_leaf = row["predicted_leaf"]
        if true_leaf not in class_set or predicted_leaf not in class_set:
            raise ValueError("Tahmin satirinda bilinmeyen leaf etiketi")
        ranked = sorted(class_names, key=lambda name: probability_values[name], reverse=True)
        if ranked[0] != predicted_leaf:
            raise ValueError("Predicted leaf probability argmax ile eslesmiyor")
        confidence = float(row["confidence"])
        if abs(confidence - probability_values[predicted_leaf]) > 1e-6:
            raise ValueError("Confidence predicted probability ile eslesmiyor")
        correct = true_leaf == predicted_leaf
        if str(row["leaf_correct"]).lower() != str(correct).lower():
            raise ValueError("leaf_correct alani etiketlerle eslesmiyor")

        bin_index = confidence_bin_index(confidence, bin_count)
        bins[bin_index]["count"] += 1
        bins[bin_index]["correct"] += int(correct)
        bins[bin_index]["confidence_sum"] += confidence
        nll_sum += -math.log(max(probability_values[true_leaf], 1e-15))
        brier_sum += sum(
            (value - float(name == true_leaf)) ** 2
            for name, value in probability_values.items()
        )
        top2_correct += int(true_leaf in ranked[:2])
        (correct_confidences if correct else error_confidences).append(confidence)

    calibration_rows = []
    ece = 0.0
    for index, values in enumerate(bins):
        count = values["count"]
        accuracy = values["correct"] / count if count else None
        mean_confidence = values["confidence_sum"] / count if count else None
        gap = abs(accuracy - mean_confidence) if count else None
        contribution = count / len(rows) * gap if count else 0.0
        ece += contribution
        calibration_rows.append(
            {
                "bin_index": index,
                "lower_inclusive": index / bin_count,
                "upper_inclusive_if_last": (index + 1) / bin_count,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "absolute_gap": gap,
                "ece_contribution": contribution,
            }
        )

    summary = {
        "bin_count": bin_count,
        "expected_calibration_error": ece,
        "negative_log_likelihood": nll_sum / len(rows),
        "multiclass_brier_score": brier_sum / len(rows),
        "top2_accuracy": top2_correct / len(rows),
        "mean_confidence_correct": sum(correct_confidences) / len(correct_confidences),
        "mean_confidence_error": (
            sum(error_confidences) / len(error_confidences)
            if error_confidences
            else None
        ),
        "error_confidence_at_least_0_90": sum(value >= 0.90 for value in error_confidences),
        "error_confidence_at_least_0_99": sum(value >= 0.99 for value in error_confidences),
    }
    return summary, calibration_rows


def error_priority(row: dict) -> tuple:
    confidence = float(row["confidence"])
    parent_wrong = row["true_parent"] != row["predicted_parent"]
    if parent_wrong:
        priority = 0
        reason = "cross_parent_error"
    elif confidence >= 0.99:
        priority = 1
        reason = "confidence_ge_0_99"
    elif confidence >= 0.90:
        priority = 2
        reason = "confidence_ge_0_90"
    else:
        priority = 3
        reason = "remaining_leaf_error"
    return priority, -confidence, row["true_leaf"], row["predicted_leaf"], row["relative_path"], reason


def prepare_error_rows(rows: Sequence[dict]) -> list[dict]:
    errors = [dict(row) for row in rows if row["true_leaf"] != row["predicted_leaf"]]
    errors.sort(key=error_priority)
    prepared = []
    for index, row in enumerate(errors, start=1):
        priority, _, _, _, _, reason = error_priority(row)
        prepared.append(
            {
                "error_id": f"ERR{index:04d}",
                "review_priority": priority,
                "priority_reason": reason,
                **row,
            }
        )
    return prepared


def load_font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_error_sheets(
    error_rows: Sequence[dict],
    dataset_root: Path,
    output_dir: Path,
    images_per_sheet: int,
) -> list[dict]:
    if images_per_sheet <= 0:
        raise ValueError("images_per_sheet pozitif olmali")
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows_per_page = math.ceil(images_per_sheet / columns)
    card_width, card_height = 590, 390
    page_width, page_height = columns * card_width, rows_per_page * card_height
    title_font = load_font(21)
    text_font = load_font(17)
    small_font = load_font(14)
    index_rows = []

    for page_offset in range(0, len(error_rows), images_per_sheet):
        page_number = page_offset // images_per_sheet + 1
        page_rows = error_rows[page_offset : page_offset + images_per_sheet]
        canvas = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(page_rows):
            column = slot % columns
            line = slot // columns
            left, top = column * card_width, line * card_height
            image_path = (dataset_root / row["relative_path"]).resolve()
            try:
                image_path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError(f"Dataset disina tasan path: {row['relative_path']}") from error
            if image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES or not image_path.is_file():
                raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            thumbnail = ImageOps.contain(image, (540, 265))
            image_left = left + (card_width - thumbnail.width) // 2
            image_top = top + 8
            canvas.paste(thumbnail, (image_left, image_top))
            parent_wrong = row["true_parent"] != row["predicted_parent"]
            border = "#b91c1c" if parent_wrong else "#d97706"
            draw.rectangle(
                [left + 3, top + 3, left + card_width - 4, top + card_height - 4],
                outline=border,
                width=4,
            )
            text_top = top + 282
            draw.text((left + 18, text_top), row["error_id"], fill="black", font=title_font)
            draw.text(
                (left + 125, text_top + 2),
                f"confidence={float(row['confidence']):.4f}",
                fill="#7f1d1d" if float(row["confidence"]) >= 0.90 else "#374151",
                font=text_font,
            )
            draw.text(
                (left + 18, text_top + 31),
                f"TRUE: {row['true_leaf']}  ->  PRED: {row['predicted_leaf']}",
                fill="black",
                font=text_font,
            )
            draw.text(
                (left + 18, text_top + 57),
                f"parent_correct={row['true_parent'] == row['predicted_parent']} | {row['priority_reason']}",
                fill="#374151",
                font=small_font,
            )
            filename = row["relative_path"]
            if len(filename) > 72:
                filename = "..." + filename[-69:]
            draw.text((left + 18, text_top + 80), filename, fill="#4b5563", font=small_font)
            index_rows.append(
                {
                    "error_id": row["error_id"],
                    "page": page_number,
                    "slot": slot + 1,
                    "relative_path": row["relative_path"],
                }
            )
        canvas.save(sheets_dir / f"error_sheet_{page_number:03d}.png")
    return index_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Bos CSV yazilamaz: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_reliability_plot(calibration_rows: Sequence[dict], output_path: Path) -> None:
    populated = [row for row in calibration_rows if row["count"]]
    confidences = [row["mean_confidence"] for row in populated]
    accuracies = [row["accuracy"] for row in populated]
    sizes = [max(25, row["count"] / 15) for row in populated]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect calibration")
    ax.scatter(confidences, accuracies, s=sizes, alpha=0.75, color="#2563eb")
    ax.set(
        xlim=(0, 1.01),
        ylim=(0, 1.01),
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        title="Reliability diagram (test predictions)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_hash_manifest(output_dir: Path) -> dict:
    rows = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "hash_manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        rows[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return rows


def analyze(
    review_package: Path,
    dataset_root: Path,
    output_dir: Path,
    expected_package_sha256: str | None,
    bin_count: int = 15,
    images_per_sheet: int = 6,
) -> dict:
    review_package = review_package.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if not review_package.is_file():
        raise FileNotFoundError(f"Review package bulunamadi: {review_package}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root bulunamadi: {dataset_root}")
    package_sha256 = sha256_file(review_package)
    if expected_package_sha256 and package_sha256 != expected_package_sha256.lower():
        raise ValueError(
            "Review package SHA-256 uyusmuyor: "
            f"expected={expected_package_sha256.lower()}, actual={package_sha256}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output klasoru bos degil; uzerine yazilmayacak: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, rows = load_review_package(review_package)
    class_names = metrics.get("class_names")
    expected_test_rows = metrics.get("split_sizes", {}).get("test")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("test_metrics class_names icermiyor")
    if len(rows) != expected_test_rows:
        raise ValueError(f"Prediction satiri beklenen test sayisiyla eslesmiyor: {len(rows)}")
    paths = [row["relative_path"].replace("\\", "/") for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Prediction CSV benzersiz olmayan relative_path iceriyor")
    for row, relative in zip(rows, paths):
        row["relative_path"] = relative
        if (row["true_parent"] == row["predicted_parent"]) != (
            str(row["parent_correct"]).lower() == "true"
        ):
            raise ValueError("parent_correct alani parent etiketleriyle eslesmiyor")

    calibration_summary, calibration_rows = compute_calibration(rows, class_names, bin_count)
    error_rows = prepare_error_rows(rows)
    for row in error_rows:
        if not (dataset_root / row["relative_path"]).is_file():
            raise FileNotFoundError(f"Hata goruntusu bulunamadi: {row['relative_path']}")
    index_rows = render_error_sheets(
        error_rows, dataset_root, output_dir, images_per_sheet
    )
    write_csv(output_dir / "error_rows.csv", error_rows)
    write_csv(output_dir / "contact_sheet_index.csv", index_rows)
    write_csv(output_dir / "reliability_bins.csv", calibration_rows)
    save_reliability_plot(calibration_rows, output_dir / "reliability_diagram.png")

    confusion_counts = Counter(
        (row["true_leaf"], row["predicted_leaf"]) for row in error_rows
    )
    parent_confusion_counts = Counter(
        (row["true_parent"], row["predicted_parent"])
        for row in error_rows
        if row["true_parent"] != row["predicted_parent"]
    )
    summary = {
        "schema_version": 1,
        "status": "model_error_analysis_v1_complete",
        "analysis_type": "read_only_post_training_error_and_calibration_audit",
        "not_model_training": True,
        "input": {
            "review_package": str(review_package),
            "review_package_sha256": package_sha256,
            "dataset_root": str(dataset_root),
            "prediction_rows": len(rows),
            "class_count": len(class_names),
        },
        "model_result": {
            "best_epoch": metrics.get("best_epoch"),
            "leaf_metrics": metrics.get("leaf_metrics"),
            "parent_metrics": metrics.get("parent_metrics"),
        },
        "errors": {
            "leaf_error_count": len(error_rows),
            "parent_error_count": sum(
                row["true_parent"] != row["predicted_parent"] for row in error_rows
            ),
            "leaf_wrong_parent_correct_count": sum(
                row["true_parent"] == row["predicted_parent"] for row in error_rows
            ),
            "contact_sheet_count": math.ceil(len(error_rows) / images_per_sheet),
            "top_leaf_confusions": [
                {"true_leaf": true, "predicted_leaf": predicted, "count": count}
                for (true, predicted), count in confusion_counts.most_common()
            ],
            "cross_parent_confusions": [
                {"true_parent": true, "predicted_parent": predicted, "count": count}
                for (true, predicted), count in parent_confusion_counts.most_common()
            ],
        },
        "calibration": calibration_summary,
        "limitations": [
            "Calibration is measured on the already-used shared benchmark test set.",
            "Confidence is not clinical certainty.",
            "This audit does not remove residual unknown near-duplicate or modality shortcut risk.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    hash_manifest = build_hash_manifest(output_dir)
    write_json(output_dir / "hash_manifest.json", hash_manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review ZIP'indeki test tahminlerinden hata ve kalibrasyon audit'i uretir"
    )
    parser.add_argument("--review-package", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-package-sha256", default=None)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--images-per-sheet", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        Path(args.review_package),
        Path(args.data_dir),
        Path(args.output_dir),
        args.expected_package_sha256,
        args.bins,
        args.images_per_sheet,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Rapor: {Path(args.output_dir).resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
