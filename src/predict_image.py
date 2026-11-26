from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

from train import build_model, build_transforms
from training_support import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "live_demo_resnet18_lastblock_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run hash-verified single-image inference or replay the frozen 37-image "
            "verification set. This command never trains or changes model weights."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image")
    parser.add_argument("--dataset-root")
    parser.add_argument("--verify-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if bool(args.image) == bool(args.verify_manifest):
        parser.error("Provide exactly one of --image or --verify-manifest")
    if args.verify_manifest and not args.dataset_root:
        parser.error("--verify-manifest requires --dataset-root")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_taxonomy(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    taxonomy_path = resolve_project_path(str(config["taxonomy"]))
    taxonomy_payload = read_json(taxonomy_path)
    labels = taxonomy_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Taxonomy does not contain a labels object")
    expected = list(config["class_names"])
    if set(labels) != set(expected):
        raise ValueError("Taxonomy labels do not match the frozen class names")
    return labels


def load_runtime(
    checkpoint_path: Path,
    config_path: Path,
    requested_device: str,
) -> tuple[torch.nn.Module, Any, list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected={config['checkpoint_sha256']}, actual={checkpoint_hash}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names")
    checkpoint_args = checkpoint.get("args")
    if class_names != config["class_names"]:
        raise ValueError("Checkpoint class order differs from the frozen demo contract")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint does not contain training arguments")
    for key in ("model", "target_level", "image_size", "normalization"):
        if checkpoint_args.get(key) != config[key]:
            raise ValueError(
                f"Checkpoint {key!r} differs from the frozen demo contract: "
                f"{checkpoint_args.get(key)!r} != {config[key]!r}"
            )

    model = build_model(
        checkpoint_args["model"],
        len(class_names),
        pretrained=False,
        fine_tune=bool(checkpoint_args.get("fine_tune", False)),
        dropout=checkpoint_args.get("dropout"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(requested_device)
    model.to(device).eval()
    _, transform = build_transforms(
        int(config["image_size"]), "none", str(config["normalization"])
    )
    taxonomy = load_taxonomy(config)
    metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "model": checkpoint_args["model"],
        "target_level": checkpoint_args["target_level"],
        "image_size": int(config["image_size"]),
        "normalization": config["normalization"],
        "warning": config["warning"],
    }
    return model, transform, list(class_names), taxonomy, metadata


def infer_true_leaf(image_path: Path, dataset_root: Path | None) -> str | None:
    if dataset_root is None:
        return None
    try:
        relative = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return None
    return relative.parts[1] if len(relative.parts) >= 3 else None


def predict_one(
    image_path: Path,
    model: torch.nn.Module,
    transform: Any,
    class_names: Sequence[str],
    taxonomy: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    top_k: int,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    started = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    count = min(top_k, len(class_names))
    values, indices = torch.topk(probabilities, k=count)
    ranked = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        leaf = class_names[index]
        details = taxonomy[leaf]
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "leaf": leaf,
                "display_name": details["display_name"],
                "parent_group": details["parent_group"],
                "probability": float(value),
            }
        )
    predicted = ranked[0]
    true_leaf = infer_true_leaf(image_path, dataset_root)
    result = {
        "schema_version": 1,
        "status": "live_demo_prediction_complete",
        "does_not_train_or_change_weights": True,
        "image_path": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        **metadata,
        "predicted_leaf": predicted["leaf"],
        "predicted_display_name": predicted["display_name"],
        "predicted_parent_group": predicted["parent_group"],
        "confidence": predicted["probability"],
        "supported_interpretation": taxonomy[predicted["leaf"]]["safe_output"],
        "top_k": ranked,
        "inference_milliseconds": elapsed_ms,
        "true_leaf_from_dataset_path": true_leaf,
        "leaf_correct_when_known": (
            predicted["leaf"] == true_leaf if true_leaf is not None else None
        ),
    }
    return result


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_prediction_card(image_path: Path, result: dict[str, Any], output_path: Path) -> None:
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    source.thumbnail((660, 660), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1500, 760), "white")
    image_x = (700 - source.width) // 2
    image_y = (760 - source.height) // 2
    canvas.paste(source, (image_x, image_y))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34)
    body_font = load_font(25)
    small_font = load_font(19)
    x = 735
    y = 45
    draw.text((x, y), "Hash-verified live prediction", fill="#1f2937", font=title_font)
    y += 70
    draw.text((x, y), f"Leaf: {result['predicted_leaf']}", fill="#166534", font=body_font)
    y += 42
    draw.text((x, y), f"Meaning: {result['predicted_display_name']}", fill="#1f2937", font=body_font)
    y += 42
    draw.text((x, y), f"Parent group: {result['predicted_parent_group']}", fill="#1f2937", font=body_font)
    y += 42
    draw.text((x, y), f"Confidence: {result['confidence']:.6f}", fill="#1f2937", font=body_font)
    y += 52
    if result["true_leaf_from_dataset_path"] is not None:
        correct = "YES" if result["leaf_correct_when_known"] else "NO"
        draw.text(
            (x, y),
            f"Known dataset label: {result['true_leaf_from_dataset_path']} | Correct: {correct}",
            fill="#166534" if result["leaf_correct_when_known"] else "#b91c1c",
            font=small_font,
        )
        y += 42
    draw.text((x, y), "Top probabilities", fill="#1f2937", font=body_font)
    y += 42
    for row in result["top_k"]:
        draw.text(
            (x, y),
            f"{row['rank']}. {row['leaf']}: {row['probability']:.6f}",
            fill="#374151",
            font=small_font,
        )
        y += 32
    y = 680
    draw.text(
        (x, y),
        "Educational dataset classification — not clinical diagnosis or cancer staging.",
        fill="#9a3412",
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def verify_manifest(
    manifest_path: Path,
    dataset_root: Path,
    model: torch.nn.Module,
    transform: Any,
    class_names: Sequence[str],
    taxonomy: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(config["verification_rows"]):
        raise ValueError("Verification manifest row count differs from the frozen contract")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Verification paths are not unique")
    tolerance = float(config["confidence_absolute_tolerance"])
    results = []
    for index, row in enumerate(rows, start=1):
        image_path = dataset_root / Path(row["relative_path"])
        prediction = predict_one(
            image_path,
            model,
            transform,
            class_names,
            taxonomy,
            metadata,
            top_k=5,
            dataset_root=dataset_root,
        )
        expected_leaf = row["predicted_leaf"]
        expected_confidence = float(row["expected_confidence"])
        confidence_difference = abs(prediction["confidence"] - expected_confidence)
        passed = (
            prediction["predicted_leaf"] == expected_leaf
            and confidence_difference <= tolerance
        )
        results.append(
            {
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_leaf": row["true_leaf"],
                "expected_leaf": expected_leaf,
                "reproduced_leaf": prediction["predicted_leaf"],
                "expected_confidence": expected_confidence,
                "reproduced_confidence": prediction["confidence"],
                "confidence_abs_difference": confidence_difference,
                "passed": passed,
            }
        )
        print(
            f"Verification {index}/{len(rows)}: {row['sample_id']} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )
    passed_count = sum(bool(row["passed"]) for row in results)
    report = {
        "schema_version": 1,
        "status": (
            "live_demo_frozen_replay_verified"
            if passed_count == len(results)
            else "live_demo_frozen_replay_failed"
        ),
        "does_not_train_or_change_weights": True,
        **metadata,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_root": str(dataset_root.resolve()),
        "verification_rows": len(results),
        "passed_rows": passed_count,
        "failed_rows": len(results) - passed_count,
        "maximum_confidence_abs_difference": max(
            row["confidence_abs_difference"] for row in results
        ),
        "confidence_absolute_tolerance": tolerance,
        "all_passed": passed_count == len(results),
    }
    return report, results


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    model, transform, class_names, taxonomy, metadata = load_runtime(
        checkpoint_path, config_path, args.device
    )

    if args.image:
        image_path = Path(args.image).resolve()
        dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
        result = predict_one(
            image_path,
            model,
            transform,
            class_names,
            taxonomy,
            metadata,
            args.top_k,
            dataset_root,
        )
        result_path = output_dir / "prediction.json"
        card_path = output_dir / "prediction_card.png"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        render_prediction_card(image_path, result, card_path)
        print(json.dumps(result, indent=2))
        print(f"Prediction JSON: {result_path}")
        print(f"Prediction card: {card_path}")
        return

    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.verify_manifest).resolve()
    report, rows = verify_manifest(
        manifest_path,
        dataset_root,
        model,
        transform,
        class_names,
        taxonomy,
        metadata,
        config,
    )
    report_path = output_dir / "verification_report.json"
    rows_path = output_dir / "verification_rows.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(rows_path, rows)
    print(json.dumps(report, indent=2))
    print(f"Verification report: {report_path}")
    print(f"Verification rows: {rows_path}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
