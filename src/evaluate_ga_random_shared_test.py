"""Evaluate the frozen GA and random winners on the shared test split without retraining.

The runner has two stages. ``freeze`` verifies the completed validation-only confirmation and
records both checkpoint hashes without opening dataset images. ``evaluate`` requires that frozen
contract, re-verifies every hash, and then evaluates both checkpoints on the same 19,500 test
records. Test results never change checkpoint selection or hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from data_index import discover_records, resolve_dataset_root
from fixed_split_manifest import load_fixed_split_manifest
from train import RecordDataset, build_model, build_transforms, run_epoch, save_confusion_matrix
from training_support import (
    build_subclass_parent_map,
    classification_metrics,
    derive_parent_predictions,
    prediction_rows,
    write_prediction_csv,
)


METHODS = ("ga", "random")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_config(config: dict) -> None:
    if config.get("experiment_type") != "shared_benchmark_test_only_evaluation":
        raise ValueError("Gecersiz shared-test experiment type")
    contract = config.get("model_contract", {})
    expected_contract = {
        "model": "resnet18",
        "target_level": "subclass",
        "image_size": 224,
        "normalization": "imagenet",
        "batch_size": 32,
        "seed": 42,
        "evaluation_only": True,
        "training_allowed": False,
        "checkpoint_selection": "already_frozen_by_full_validation_macro_f1",
    }
    if contract != expected_contract:
        raise ValueError("Shared-test model sozlesmesi degismis")
    counts = config.get("frozen_inputs", {}).get("counts")
    if counts != {
        "test": 19_500,
        "test_per_leaf": 750,
        "leaf_count": 26,
        "parent_count": 8,
    }:
        raise ValueError("Shared-test sayim sozlesmesi degismis")
    policy = config.get("test_policy", {})
    if not all(
        policy.get(key) is expected
        for key, expected in {
            "candidate_selection_completed_before_test": True,
            "evaluate_both_frozen_candidates_once": True,
            "retraining_allowed": False,
            "hyperparameter_change_allowed": False,
            "test_based_reselection_allowed": False,
            "shared_benchmark_already_used_by_prior_models": True,
            "untouched_external_test_claim_allowed": False,
        }.items()
    ):
        raise ValueError("Shared-test politikasi degismis")
    candidates = config.get("candidates", [])
    if len(candidates) != 2 or {row.get("method") for row in candidates} != set(METHODS):
        raise ValueError("Tam olarak bir GA ve bir random adayi gerekli")
    if {row.get("candidate_id") for row in candidates} != {"GA_G01_C02", "RANDOM_002"}:
        raise ValueError("Shared-test candidate ID'leri degismis")


def taxonomy_contract(path: Path, expected_sha256: str) -> tuple[list[str], dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Label taxonomy SHA-256 uyusmuyor")
    payload = load_json(path)
    labels = payload.get("labels", {})
    class_names = sorted(labels)
    parent_map = {name: labels[name]["parent_group"] for name in class_names}
    if len(class_names) != 26 or len(set(parent_map.values())) != 8:
        raise ValueError("Label taxonomy 26 leaf / 8 parent kapsamiyla uyusmuyor")
    return class_names, parent_map


def candidate_by_method(config: dict) -> dict[str, dict]:
    return {row["method"]: row for row in config["candidates"]}


def validate_confirmation_sources(confirmation_root: Path, config: dict) -> dict:
    frozen = config["frozen_inputs"]
    paths = {
        "confirmation_summary_sha256": confirmation_root / "confirmation_summary.json",
        "confirmation_run_contract_sha256": confirmation_root / "run_contract.json",
        "confirmation_config_sha256": confirmation_root / "frozen_config.json",
    }
    actual = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Eksik confirmation girdisi: {path}")
        actual[key] = sha256_file(path)
        if actual[key] != frozen[key]:
            raise ValueError(f"Confirmation girdi hash'i uyusmuyor: {key}")

    summary = load_json(paths["confirmation_summary_sha256"])
    if summary.get("status") != "ga_random_full_confirmation_v1_complete":
        raise ValueError("Full validation confirmation tamamlanmis degil")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Confirmation test acmis")
    if summary.get("validation_winner", {}).get("candidate_id") != "GA_G01_C02":
        raise ValueError("Dondurulmus validation winner degismis")

    configured = candidate_by_method(config)
    for method in METHODS:
        candidate = configured[method]
        result_path = confirmation_root / "candidates" / method / "confirmation_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Eksik confirmation result: {result_path}")
        if sha256_file(result_path) != candidate["confirmation_result_sha256"]:
            raise ValueError(f"Confirmation result hash'i uyusmuyor: {method}")
        result = load_json(result_path)
        source = summary["results"][method]
        for key in ("candidate_id", "method", "best_epoch"):
            if result.get(key) != candidate[key] or source.get(key) != candidate[key]:
                raise ValueError(f"Confirmation candidate alani uyusmuyor: {method}/{key}")
        for key in ("learning_rate", "weight_decay", "dropout", "full_validation_macro_f1"):
            if not math.isclose(
                float(result[key]), float(candidate[key]), rel_tol=0.0, abs_tol=1e-15
            ) or not math.isclose(
                float(source[key]), float(candidate[key]), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"Confirmation candidate sayisi uyusmuyor: {method}/{key}")
        if result.get("test_evaluated") is not False:
            raise ValueError(f"Confirmation result test acmis: {method}")
    return actual


def inspect_checkpoint(path: Path, candidate: dict, class_names: Sequence[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint bulunamadi: {path}")
    checkpoint_hash = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != list(class_names):
        raise ValueError(f"Checkpoint sinif sirasi uyusmuyor: {candidate['method']}")
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError("Checkpoint deney argumanlarini icermiyor")
    expected = {
        "model": "resnet18",
        "target_level": "subclass",
        "learning_rate": candidate["learning_rate"],
        "weight_decay": candidate["weight_decay"],
        "dropout": candidate["dropout"],
    }
    for key, value in expected.items():
        actual = args.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Checkpoint argumani uyusmuyor: {candidate['method']}/{key}")
        elif actual != value:
            raise ValueError(f"Checkpoint argumani uyusmuyor: {candidate['method']}/{key}")
    model = build_model(
        "resnet18", len(class_names), pretrained=False, fine_tune=False,
        dropout=float(candidate["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {"sha256": checkpoint_hash, "size_bytes": path.stat().st_size}


def build_checkpoint_freeze(args: argparse.Namespace, config: dict) -> dict:
    config_path = Path(args.config).resolve()
    split_path = Path(args.split_manifest).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    confirmation_root = Path(args.confirmation_root).resolve()
    frozen = config["frozen_inputs"]
    if sha256_file(split_path) != frozen["split_manifest_sha256"]:
        raise ValueError("Frozen split SHA-256 uyusmuyor")
    class_names, _ = taxonomy_contract(taxonomy_path, frozen["label_taxonomy_sha256"])
    source_hashes = validate_confirmation_sources(confirmation_root, config)
    checkpoints = {}
    for method, candidate in candidate_by_method(config).items():
        path = confirmation_root / "candidates" / method / "best_model.pth"
        checkpoints[method] = {
            "candidate_id": candidate["candidate_id"],
            **inspect_checkpoint(path, candidate, class_names),
        }
    return {
        "schema_version": 1,
        "status": "ga_random_shared_test_v1_checkpoints_frozen_test_not_opened",
        "experiment_type": config["experiment_type"],
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "split_manifest_sha256": frozen["split_manifest_sha256"],
        "label_taxonomy_sha256": frozen["label_taxonomy_sha256"],
        "confirmation_inputs": source_hashes,
        "class_names": class_names,
        "checkpoints": checkpoints,
        "test_opened_during_freeze": False,
    }


def freeze(args: argparse.Namespace, config: dict) -> dict:
    payload = build_checkpoint_freeze(args, config)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "checkpoint_freeze.json"
    if path.is_file() and load_json(path) != payload:
        raise ValueError("Mevcut checkpoint freeze yeni sozlesmeyle uyusmuyor")
    write_json_atomic(path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def exact_mcnemar_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_comparison(first_rows: Sequence[dict], second_rows: Sequence[dict]) -> dict:
    first = {row["relative_path"]: row for row in first_rows}
    second = {row["relative_path"]: row for row in second_rows}
    if len(first) != len(first_rows) or len(second) != len(second_rows) or set(first) != set(second):
        raise ValueError("Paired prediction path kapsami uyusmuyor")
    if any(first[path]["true_leaf"] != second[path]["true_leaf"] for path in first):
        raise ValueError("Paired prediction true label'lari uyusmuyor")

    def counts(field: str) -> dict:
        values = Counter(
            (
                first[path][field].lower() == "true",
                second[path][field].lower() == "true",
            )
            for path in first
        )
        first_only = values[(True, False)]
        second_only = values[(False, True)]
        return {
            "both_correct": values[(True, True)],
            "ga_only_correct": first_only,
            "random_only_correct": second_only,
            "both_wrong": values[(False, False)],
            "discordant": first_only + second_only,
            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(first_only, second_only),
        }

    return {"leaf": counts("leaf_correct"), "parent": counts("parent_correct")}


def save_reports(
    output_dir: Path,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    parent_metrics: dict,
) -> None:
    leaf_text = classification_report(
        targets, predictions, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )
    parent_text = classification_report(
        parent_metrics["targets"], parent_metrics["predictions"],
        labels=list(range(len(parent_metrics["class_names"]))),
        target_names=parent_metrics["class_names"], zero_division=0,
    )
    (output_dir / "leaf_classification_report.txt").write_text(leaf_text, encoding="utf-8")
    (output_dir / "parent_classification_report.txt").write_text(parent_text, encoding="utf-8")
    write_json_atomic(
        output_dir / "classification_reports.json",
        {
            "leaf": classification_report(
                targets, predictions, labels=list(range(len(class_names))),
                target_names=class_names, zero_division=0, output_dict=True,
            ),
            "parent": classification_report(
                parent_metrics["targets"], parent_metrics["predictions"],
                labels=list(range(len(parent_metrics["class_names"]))),
                target_names=parent_metrics["class_names"], zero_division=0,
                output_dict=True,
            ),
        },
    )
    save_confusion_matrix(targets, predictions, class_names, output_dir / "leaf_confusion_matrix.png")
    save_confusion_matrix(
        parent_metrics["targets"], parent_metrics["predictions"],
        parent_metrics["class_names"], output_dir / "parent_confusion_matrix.png",
    )


def completed_candidate(output_dir: Path, expected_checkpoint_hash: str) -> dict | None:
    marker = output_dir / "candidate_complete.json"
    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    if not marker.is_file():
        return None
    payload = load_json(marker)
    if payload.get("checkpoint_sha256") != expected_checkpoint_hash:
        raise ValueError("Tamamlanmis candidate checkpoint hash'i uyusmuyor")
    for name, expected in payload.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Tamamlanmis candidate artifact'i bozuk: {path}")
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise ValueError("Candidate complete marker temel ciktilari icermiyor")
    return load_json(metrics_path)


def evaluate_candidate(
    method: str,
    candidate: dict,
    checkpoint_path: Path,
    checkpoint_hash: str,
    test_records,
    class_names: Sequence[str],
    class_to_index: Mapping[str, int],
    parent_map: Mapping[str, str],
    dataset_root: Path,
    output_root: Path,
    config: dict,
    device: torch.device,
    workers: int,
) -> dict:
    candidate_dir = output_root / "candidates" / method
    candidate_dir.mkdir(parents=True, exist_ok=True)
    existing = completed_candidate(candidate_dir, checkpoint_hash)
    if existing is not None:
        print(f"RESUME: {method} shared-test sonucu doğrulandı ve atlandı")
        return existing

    _, eval_transform = build_transforms(
        config["model_contract"]["image_size"], "none",
        config["model_contract"]["normalization"],
    )
    dataset = RecordDataset(test_records, dict(class_to_index), "subclass", eval_transform)
    loader = DataLoader(
        dataset, batch_size=config["model_contract"]["batch_size"], shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    model = build_model(
        "resnet18", len(class_names), pretrained=False, fine_tune=False,
        dropout=float(candidate["dropout"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device, collect_outputs=True)
    leaf_metrics = classification_metrics(metrics["targets"], metrics["predictions"])
    parent_metrics = derive_parent_predictions(
        metrics["targets"], metrics["predictions"], class_names, parent_map
    )
    rows = prediction_rows(
        test_records, metrics["record_indices"], metrics["targets"], metrics["predictions"],
        metrics["probabilities"], class_names, parent_map, dataset_root,
    )
    write_prediction_csv(rows, candidate_dir / "test_predictions.csv")
    save_reports(
        candidate_dir, metrics["targets"], metrics["predictions"], class_names, parent_metrics
    )
    summary = {
        "schema_version": 1,
        "status": "ga_random_shared_test_candidate_complete",
        "experiment_type": config["experiment_type"],
        "candidate_id": candidate["candidate_id"],
        "method": method,
        "test_evaluated": True,
        "checkpoint_sha256": checkpoint_hash,
        "full_validation_macro_f1": candidate["full_validation_macro_f1"],
        "best_validation_epoch": candidate["best_epoch"],
        "test_loss": metrics["loss"],
        "leaf_metrics": leaf_metrics,
        "parent_metrics": {
            key: parent_metrics[key]
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "class_names")
        },
        "prediction_rows": len(rows),
        "warning": "Shared benchmark result; not an untouched external test or clinical evidence.",
    }
    write_json_atomic(candidate_dir / "test_metrics.json", summary)
    artifact_names = (
        "test_metrics.json", "test_predictions.csv", "classification_reports.json",
        "leaf_classification_report.txt", "parent_classification_report.txt",
        "leaf_confusion_matrix.png", "parent_confusion_matrix.png",
    )
    marker = {
        "schema_version": 1,
        "status": "candidate_test_artifacts_complete",
        "checkpoint_sha256": checkpoint_hash,
        "artifacts": {name: sha256_file(candidate_dir / name) for name in artifact_names},
    }
    write_json_atomic(candidate_dir / "candidate_complete.json", marker)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def read_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_final_hash_manifest(output_root: Path) -> dict:
    files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "hash_manifest.json"
    )
    payload = {
        path.relative_to(output_root).as_posix(): {
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size
        }
        for path in files
    }
    write_json_atomic(output_root / "hash_manifest.json", payload)
    return payload


def evaluate(args: argparse.Namespace, config: dict) -> dict:
    if args.authorize != config["authorization_token"]:
        raise ValueError("Shared test kilitli; frozen authorization token gerekli")
    output_root = Path(args.output_dir).resolve()
    freeze_path = output_root / "checkpoint_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Once checkpoint freeze asamasi calistirilmali")
    frozen_contract = load_json(freeze_path)
    current_contract = build_checkpoint_freeze(args, config)
    if current_contract != frozen_contract:
        raise ValueError("Checkpoint freeze ile mevcut girdiler uyusmuyor")

    dataset_root = resolve_dataset_root(args.data_dir)
    records = discover_records(dataset_root)
    fixed = load_fixed_split_manifest(
        records, dataset_root, Path(args.split_manifest).resolve()
    )
    test_records = fixed["test"]
    class_names, parent_map = taxonomy_contract(
        Path(args.taxonomy).resolve(), config["frozen_inputs"]["label_taxonomy_sha256"]
    )
    counts = Counter(record.subclass for record in test_records)
    expected = config["frozen_inputs"]["counts"]
    if len(test_records) != expected["test"] or set(counts) != set(class_names):
        raise ValueError("Test split kapsamı degismis")
    if any(count != expected["test_per_leaf"] for count in counts.values()):
        raise ValueError("Test split leaf dengesi degismis")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi fakat kullanilabilir degil")
    device = torch.device(device_name)
    print(f"Shared test aciliyor: {len(test_records)} goruntu | cihaz={device}")

    confirmation_root = Path(args.confirmation_root).resolve()
    results = {}
    for method in METHODS:
        candidate = candidate_by_method(config)[method]
        checkpoint_path = confirmation_root / "candidates" / method / "best_model.pth"
        results[method] = evaluate_candidate(
            method, candidate, checkpoint_path,
            frozen_contract["checkpoints"][method]["sha256"], test_records,
            class_names, class_to_index, parent_map, dataset_root, output_root,
            config, device, args.workers,
        )

    comparison = paired_comparison(
        read_prediction_rows(output_root / "candidates/ga/test_predictions.csv"),
        read_prediction_rows(output_root / "candidates/random/test_predictions.csv"),
    )
    summary = {
        "schema_version": 1,
        "status": "ga_random_shared_test_v1_complete",
        "experiment_type": config["experiment_type"],
        "test_evaluated": True,
        "test_rows": len(test_records),
        "frozen_validation_winner": "GA_G01_C02",
        "candidate_results": results,
        "ga_minus_random": {
            metric: results["ga"]["leaf_metrics"][metric]
            - results["random"]["leaf_metrics"][metric]
            for metric in ("accuracy", "macro_f1", "balanced_accuracy")
        },
        "paired_comparison": comparison,
        "test_based_reselection_performed": False,
        "warning": (
            "Shared benchmark already used by prior models; not untouched external test or "
            "clinical evidence. Validation selection remains frozen regardless of test ranking."
        ),
    }
    write_json_atomic(output_root / "shared_test_summary.json", summary)
    write_final_hash_manifest(output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Frozen GA/random shared-test evaluator")
    parser.add_argument("--mode", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--split-manifest",
        default=str(root / "research/splits/practical_leakage_reduced_split_v1.csv"),
    )
    parser.add_argument("--taxonomy", default=str(root / "config/label_taxonomy.json"))
    parser.add_argument("--config", default=str(root / "config/ga_random_shared_test_v1.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--authorize", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config).resolve())
    validate_config(config)
    if args.mode == "freeze":
        freeze(args, config)
    else:
        if not args.data_dir:
            raise ValueError("Evaluate modu icin --data-dir zorunludur")
        evaluate(args, config)


if __name__ == "__main__":
    main()
