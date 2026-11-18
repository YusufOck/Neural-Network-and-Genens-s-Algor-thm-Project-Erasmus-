"""Run the frozen post-hoc Grad-CAM audit for the fine-tuned ResNet18.

This command never trains, selects, or changes a model.  It verifies the frozen
checkpoint, split, predictions and sample manifest before opening an image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from train import build_model


EXPECTED_COLUMNS = (
    "sample_id",
    "selection_group",
    "relative_path",
    "true_leaf",
    "predicted_leaf",
    "expected_confidence",
    "target_mode",
    "split",
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen ResNet18 Grad-CAM explainability audit (no training)"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected True/False value, received {value!r}")
    return normalized == "true"


def resolve_relative_image(dataset_root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    if not parts or ".." in parts:
        raise ValueError(f"Unsafe relative image path: {relative_path}")
    candidate = dataset_root.joinpath(*parts).resolve()
    root = dataset_root.resolve()
    if root not in candidate.parents:
        raise ValueError(f"Image escapes dataset root: {relative_path}")
    return candidate


def choose_nearest_median(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    median_confidence = statistics.median(float(row["confidence"]) for row in rows)
    return min(
        rows,
        key=lambda row: (
            abs(float(row["confidence"]) - median_confidence),
            row["relative_path"],
        ),
    )


def derive_frozen_selection(
    prediction_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Reproduce the selection contract without using any Grad-CAM output."""

    correct_by_leaf: dict[str, list[dict[str, str]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    for row in prediction_rows:
        if bool_value(row["leaf_correct"]):
            correct_by_leaf[row["true_leaf"]].append(row)
        else:
            errors.append(row)

    selected: list[dict[str, str]] = []
    for leaf_name in sorted(correct_by_leaf):
        row = choose_nearest_median(correct_by_leaf[leaf_name])
        selected.append(
            {
                "sample_id": f"CORRECT_{leaf_name}",
                "selection_group": "correct_leaf_median_confidence",
                "relative_path": row["relative_path"],
                "true_leaf": row["true_leaf"],
                "predicted_leaf": row["predicted_leaf"],
                "expected_confidence": row["confidence"],
                "target_mode": "predicted_only",
                "split": "test",
            }
        )

    error_candidates = [
        row
        for row in errors
        if (not bool_value(row["parent_correct"])) or float(row["confidence"]) >= 0.99
    ]
    for true_leaf, predicted_leaf in (
        ("oral_scc", "oral_normal"),
        ("oral_normal", "oral_scc"),
    ):
        directional = [
            row
            for row in errors
            if row["true_leaf"] == true_leaf
            and row["predicted_leaf"] == predicted_leaf
        ]
        if not directional:
            raise ValueError(
                f"Missing dominant oral confusion direction: {true_leaf}->{predicted_leaf}"
            )
        error_candidates.append(choose_nearest_median(directional))

    unique_errors = {
        row["relative_path"]: row for row in error_candidates
    }
    for index, relative_path in enumerate(sorted(unique_errors), start=1):
        row = unique_errors[relative_path]
        if not bool_value(row["parent_correct"]):
            selection_group = "cross_parent_error"
        elif float(row["confidence"]) >= 0.99:
            selection_group = "high_confidence_error"
        else:
            selection_group = "dominant_oral_confusion_median"
        selected.append(
            {
                "sample_id": f"ERROR_{index:02d}",
                "selection_group": selection_group,
                "relative_path": relative_path,
                "true_leaf": row["true_leaf"],
                "predicted_leaf": row["predicted_leaf"],
                "expected_confidence": row["confidence"],
                "target_mode": "predicted_and_true",
                "split": "test",
            }
        )
    return selected


def validate_frozen_inputs(
    *,
    config: dict,
    checkpoint_path: Path,
    predictions_path: Path,
    split_path: Path,
    sample_manifest_path: Path,
    dataset_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    expected_hashes = {
        checkpoint_path: config["checkpoint"]["sha256"],
        predictions_path: config["test_predictions"]["sha256"],
        split_path: config["split_manifest"]["sha256"],
        sample_manifest_path: config["sample_manifest"]["sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {path}: expected={expected}, actual={actual}"
            )

    manifest_rows = read_csv(sample_manifest_path)
    prediction_rows = read_csv(predictions_path)
    split_rows = read_csv(split_path)
    if tuple(manifest_rows[0]) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected sample-manifest schema")
    if len(manifest_rows) != config["sample_manifest"]["sample_count"]:
        raise ValueError("Sample-manifest count differs from frozen config")
    if len({row["sample_id"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("Sample IDs are not unique")
    if len({row["relative_path"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("Selected image paths are not unique")

    rederived = derive_frozen_selection(prediction_rows)
    comparison_fields = (
        "sample_id",
        "selection_group",
        "relative_path",
        "true_leaf",
        "predicted_leaf",
        "target_mode",
        "split",
    )
    for frozen_row, derived_row in zip(manifest_rows, rederived, strict=True):
        if any(frozen_row[field] != derived_row[field] for field in comparison_fields):
            raise ValueError("Frozen sample manifest does not match its selection algorithm")
        if not math.isclose(
            float(frozen_row["expected_confidence"]),
            float(derived_row["expected_confidence"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Frozen sample confidence differs from prediction evidence")

    prediction_by_path = {row["relative_path"]: row for row in prediction_rows}
    test_paths = {
        row["relative_path"] for row in split_rows if row["split"] == "test"
    }
    for row in manifest_rows:
        relative_path = row["relative_path"]
        if relative_path not in test_paths:
            raise ValueError(f"Selected path is not in the frozen test split: {relative_path}")
        prediction = prediction_by_path.get(relative_path)
        if prediction is None:
            raise ValueError(f"Selected path is missing from test predictions: {relative_path}")
        if prediction["true_leaf"] != row["true_leaf"]:
            raise ValueError(f"True label mismatch for {relative_path}")
        if prediction["predicted_leaf"] != row["predicted_leaf"]:
            raise ValueError(f"Predicted label mismatch for {relative_path}")
        image_path = resolve_relative_image(dataset_root, relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    group_counts = Counter(row["selection_group"] for row in manifest_rows)
    expected_groups = {
        "correct_leaf_median_confidence": 26,
        "cross_parent_error": 2,
        "high_confidence_error": 7,
        "dominant_oral_confusion_median": 2,
    }
    if dict(group_counts) != expected_groups:
        raise ValueError(f"Frozen selection-group counts changed: {dict(group_counts)}")
    return manifest_rows, prediction_rows


def gradcam_from_tensors(
    activations: torch.Tensor, gradients: torch.Tensor
) -> np.ndarray:
    if activations.ndim != 4 or gradients.shape != activations.shape:
        raise ValueError("Grad-CAM expects matching NCHW activation and gradient tensors")
    if activations.shape[0] != 1:
        raise ValueError("Grad-CAM audit processes one image at a time")
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1))[0]
    cam = cam - cam.min()
    maximum = cam.max()
    if float(maximum) > 0.0:
        cam = cam / maximum
    return cam.detach().cpu().numpy().astype(np.float32)


class GradCamCapture:
    def __init__(self, target_layer: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def close(self) -> None:
        self.handle.remove()

    def heatmap(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_index: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[0, target_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        return gradcam_from_tensors(self.activations, self.gradients), logits.detach()


def resized_rgb(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_heatmap(heatmap: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def overlay_heatmap(
    rgb: np.ndarray, heatmap: np.ndarray, alpha: float, colormap: str
) -> np.ndarray:
    colored = plt.get_cmap(colormap)(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * rgb + alpha * colored, 0.0, 1.0)


def save_sample_panel(
    *,
    row: dict[str, str],
    rgb: np.ndarray,
    predicted_heatmap: np.ndarray,
    true_heatmap: np.ndarray | None,
    output_path: Path,
    alpha: float,
    colormap: str,
) -> None:
    size = rgb.shape[0]
    predicted_resized = resize_heatmap(predicted_heatmap, size)
    predicted_overlay = overlay_heatmap(rgb, predicted_resized, alpha, colormap)
    is_error = true_heatmap is not None
    if is_error:
        true_resized = resize_heatmap(true_heatmap, size)
        images = [
            rgb,
            predicted_overlay,
            overlay_heatmap(rgb, true_resized, alpha, colormap),
        ]
        titles = [
            "Model input view",
            f"Predicted target: {row['predicted_leaf']}",
            f"True target: {row['true_leaf']}",
        ]
    else:
        images = [rgb, predicted_resized, predicted_overlay]
        titles = [
            "Model input view",
            f"Grad-CAM: {row['predicted_leaf']}",
            "Overlay",
        ]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1))
    for axis, content, title in zip(axes, images, titles, strict=True):
        if content.ndim == 2:
            axis.imshow(content, cmap=colormap, vmin=0.0, vmax=1.0)
        else:
            axis.imshow(content)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        f"{row['sample_id']} | true={row['true_leaf']} | "
        f"pred={row['predicted_leaf']} | confidence={float(row['expected_confidence']):.4f}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "Coarse model-influence map; not medical localization or causal evidence.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def contact_sheets(
    panel_paths: Sequence[Path], output_dir: Path, columns: int = 3
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_page = columns * 2
    created: list[Path] = []
    font = ImageFont.load_default()
    for page_index in range(0, len(panel_paths), per_page):
        page_paths = panel_paths[page_index : page_index + per_page]
        thumbnails: list[Image.Image] = []
        for path in page_paths:
            with Image.open(path) as panel:
                thumbnail = panel.convert("RGB")
                thumbnail.thumbnail((720, 310), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (740, 345), "white")
                tile.paste(thumbnail, ((740 - thumbnail.width) // 2, 8))
                draw = ImageDraw.Draw(tile)
                draw.text((12, 322), path.stem, fill="black", font=font)
                thumbnails.append(tile)
        rows = math.ceil(len(thumbnails) / columns)
        sheet = Image.new("RGB", (740 * columns, 345 * rows), (240, 240, 240))
        for index, tile in enumerate(thumbnails):
            x = (index % columns) * 740
            y = (index // columns) * 345
            sheet.paste(tile, (x, y))
        output_path = output_dir / f"gradcam_sheet_{page_index // per_page + 1:02d}.jpg"
        sheet.save(output_path, quality=92, subsampling=0)
        created.append(output_path)
    return created


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_hash_manifest(output_dir: Path, excluded: Iterable[Path]) -> dict[str, dict]:
    excluded_resolved = {path.resolve() for path in excluded}
    result: dict[str, dict] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.resolve() in excluded_resolved:
            continue
        relative = path.relative_to(output_dir).as_posix()
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def run(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.data_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    predictions_path = Path(args.predictions).resolve()
    split_path = Path(args.split_manifest).resolve()
    sample_manifest_path = Path(args.sample_manifest).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["does_not_train_or_select_a_model"] is not True:
        raise ValueError("Grad-CAM contract must explicitly forbid training/model selection")
    manifest_rows, prediction_rows = validate_frozen_inputs(
        config=config,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        split_path=split_path,
        sample_manifest_path=sample_manifest_path,
        dataset_root=dataset_root,
    )

    if (output_dir / "summary.json").exists():
        raise FileExistsError("Completed Grad-CAM output already exists; refusing overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "sample_panels"
    sheets_dir = output_dir / "contact_sheets"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(class_names, list) or len(class_names) != config["class_count"]:
        raise ValueError("Checkpoint class-name contract is invalid")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint experiment arguments are missing")
    if checkpoint_args.get("model") != "resnet18":
        raise ValueError("Checkpoint is not ResNet18")
    if checkpoint_args.get("target_level") != "subclass":
        raise ValueError("Checkpoint is not a 26-leaf classifier")

    model = build_model("resnet18", len(class_names), pretrained=False, fine_tune=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    target_layer = model.layer4[-1].conv2
    capture = GradCamCapture(target_layer)
    preprocessing = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    prediction_by_path = {row["relative_path"]: row for row in prediction_rows}
    render = config["render_contract"]
    result_rows: list[dict[str, object]] = []
    panel_paths: list[Path] = []
    max_confidence_difference = 0.0
    try:
        for index, row in enumerate(manifest_rows, start=1):
            image_path = resolve_relative_image(dataset_root, row["relative_path"])
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            input_tensor = preprocessing(image).unsqueeze(0).to(device)
            predicted_index = class_to_index[row["predicted_leaf"]]
            predicted_heatmap, logits = capture.heatmap(
                model, input_tensor, predicted_index
            )
            probabilities = torch.softmax(logits, dim=1)[0]
            reproduced_index = int(probabilities.argmax().item())
            reproduced_leaf = class_names[reproduced_index]
            reproduced_confidence = float(probabilities[reproduced_index].item())
            expected_confidence = float(row["expected_confidence"])
            confidence_difference = abs(reproduced_confidence - expected_confidence)
            max_confidence_difference = max(
                max_confidence_difference, confidence_difference
            )
            if reproduced_leaf != row["predicted_leaf"]:
                raise ValueError(
                    f"Checkpoint prediction was not reproduced for {row['relative_path']}: "
                    f"expected={row['predicted_leaf']}, actual={reproduced_leaf}"
                )
            if confidence_difference > 5e-5:
                raise ValueError(
                    f"Checkpoint confidence differs for {row['relative_path']}: "
                    f"expected={expected_confidence}, actual={reproduced_confidence}"
                )

            true_heatmap = None
            if row["target_mode"] == "predicted_and_true":
                true_index = class_to_index[row["true_leaf"]]
                true_heatmap, _ = capture.heatmap(model, input_tensor, true_index)

            rgb = resized_rgb(image, config["image_size"])
            panel_path = panels_dir / f"{row['sample_id']}.png"
            save_sample_panel(
                row=row,
                rgb=rgb,
                predicted_heatmap=predicted_heatmap,
                true_heatmap=true_heatmap,
                output_path=panel_path,
                alpha=float(render["overlay_alpha"]),
                colormap=render["colormap"],
            )
            panel_paths.append(panel_path)
            prediction_evidence = prediction_by_path[row["relative_path"]]
            result_rows.append(
                {
                    **row,
                    "checkpoint_predicted_leaf": reproduced_leaf,
                    "checkpoint_confidence": f"{reproduced_confidence:.10f}",
                    "confidence_abs_difference": f"{confidence_difference:.10g}",
                    "prediction_reproduced": True,
                    "parent_correct": prediction_evidence["parent_correct"],
                    "panel_path": panel_path.relative_to(output_dir).as_posix(),
                }
            )
            print(f"Grad-CAM {index}/{len(manifest_rows)}: {row['sample_id']}")
    finally:
        capture.close()

    sheet_paths = contact_sheets(
        panel_paths,
        sheets_dir,
        columns=int(render["contact_sheet_columns"]),
    )
    results_path = output_dir / "gradcam_results.csv"
    write_csv(results_path, result_rows)
    copied_config = output_dir / "frozen_config.json"
    copied_manifest = output_dir / "sample_manifest.csv"
    shutil.copy2(config_path, copied_config)
    shutil.copy2(sample_manifest_path, copied_manifest)

    summary = {
        "schema_version": 1,
        "status": "resnet18_lastblock_gradcam_v1_complete",
        "experiment_type": config["experiment_type"],
        "does_not_train_or_select_a_model": True,
        "reportable_as_model_performance": False,
        "device": str(device),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": sha256_file(split_path),
        "test_predictions_sha256": sha256_file(predictions_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "target_layer": config["target_layer"],
        "sample_count": len(manifest_rows),
        "correct_control_count": sum(
            row["selection_group"] == "correct_leaf_median_confidence"
            for row in manifest_rows
        ),
        "error_count": sum(row["sample_id"].startswith("ERROR_") for row in manifest_rows),
        "prediction_reproduction_count": sum(
            bool(row["prediction_reproduced"]) for row in result_rows
        ),
        "max_confidence_abs_difference": max_confidence_difference,
        "panel_count": len(panel_paths),
        "contact_sheet_count": len(sheet_paths),
        "selection_was_frozen_before_heatmaps": True,
        "limitations": config["interpretation_limitations"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    hash_manifest_path = output_dir / "hash_manifest.json"
    hash_manifest = build_hash_manifest(output_dir, excluded=[hash_manifest_path])
    hash_manifest_path.write_text(
        json.dumps(hash_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {summary_path}")
    print(f"Hash manifest: {hash_manifest_path}")
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
