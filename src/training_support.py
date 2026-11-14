from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import nn

from data_index import ImageRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_subclass_parent_map(records: Sequence[ImageRecord]) -> Dict[str, str]:
    """Return the unique dataset parent group for every leaf label."""

    mapping: Dict[str, str] = {}
    for record in records:
        previous = mapping.setdefault(record.subclass, record.main_class)
        if previous != record.main_class:
            raise ValueError(
                f"Alt sinif birden fazla ust gruba bagli: {record.subclass!r} "
                f"({previous!r}, {record.main_class!r})"
            )
    return dict(sorted(mapping.items()))


def classification_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Metrikler icin esit uzunlukta ve bos olmayan diziler gerekir")
    labels = sorted(set(targets) | set(predictions))
    recalls_by_label = {}
    f1_values = []
    for label in labels:
        true_positive = sum(
            target == label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_positive = sum(
            target != label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_negative = sum(
            target == label and prediction != label
            for target, prediction in zip(targets, predictions)
        )
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        f1_value = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls_by_label[label] = recall
        f1_values.append(f1_value)
    return {
        "accuracy": sum(
            target == prediction for target, prediction in zip(targets, predictions)
        )
        / len(targets),
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(
            recalls_by_label[label] for label in sorted(set(targets))
        )
        / len(set(targets)),
    }


def derive_parent_predictions(
    targets: Sequence[int],
    predictions: Sequence[int],
    leaf_class_names: Sequence[str],
    leaf_to_parent: Mapping[str, str],
) -> dict:
    """Project 26-leaf integer predictions to the eight dataset parent groups."""

    if len(targets) != len(predictions):
        raise ValueError("Target ve prediction uzunluklari farkli")
    missing = sorted(set(leaf_class_names) - set(leaf_to_parent))
    if missing:
        raise ValueError(f"Ust grup eslemesi olmayan alt siniflar: {missing}")

    parent_names = sorted({leaf_to_parent[name] for name in leaf_class_names})
    parent_to_index = {name: index for index, name in enumerate(parent_names)}
    leaf_index_to_parent_index = [
        parent_to_index[leaf_to_parent[name]] for name in leaf_class_names
    ]
    parent_targets = [leaf_index_to_parent_index[index] for index in targets]
    parent_predictions = [leaf_index_to_parent_index[index] for index in predictions]
    return {
        "class_names": parent_names,
        "targets": parent_targets,
        "predictions": parent_predictions,
        **classification_metrics(parent_targets, parent_predictions),
    }


def set_training_mode(model: nn.Module, training: bool) -> None:
    """Set model mode while keeping fully frozen BatchNorm modules in eval mode."""

    model.train(training)
    if not training:
        return
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def prediction_rows(
    records: Sequence[ImageRecord],
    record_indices: Sequence[int],
    targets: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    leaf_class_names: Sequence[str],
    leaf_to_parent: Mapping[str, str],
    dataset_root: Path,
) -> list[dict]:
    lengths = {
        len(record_indices), len(targets), len(predictions), len(probabilities)
    }
    if len(lengths) != 1:
        raise ValueError("Tahmin ciktilarinin satir sayilari eslesmiyor")

    rows = []
    for record_index, target, prediction, probability_vector in zip(
        record_indices, targets, predictions, probabilities
    ):
        record = records[record_index]
        true_leaf = leaf_class_names[target]
        predicted_leaf = leaf_class_names[prediction]
        vector = [float(value) for value in probability_vector]
        if len(vector) != len(leaf_class_names):
            raise ValueError("Probability vektoru sinif sayisiyla eslesmiyor")
        rows.append(
            {
                "relative_path": record.path.relative_to(dataset_root).as_posix(),
                "true_leaf": true_leaf,
                "predicted_leaf": predicted_leaf,
                "true_parent": leaf_to_parent[true_leaf],
                "predicted_parent": leaf_to_parent[predicted_leaf],
                "leaf_correct": true_leaf == predicted_leaf,
                "parent_correct": leaf_to_parent[true_leaf]
                == leaf_to_parent[predicted_leaf],
                "confidence": vector[prediction],
                "probabilities_json": json.dumps(
                    dict(zip(leaf_class_names, vector)), ensure_ascii=False
                ),
            }
        )
    return rows


def write_prediction_csv(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Bos prediction CSV yazilamaz")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curves(history: Sequence[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Ogrenme egrisi icin history bos olamaz")
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, title, axis in (
        ("loss", "Loss", axes[0]),
        ("accuracy", "Accuracy", axes[1]),
        ("macro_f1", "Macro-F1", axes[2]),
    ):
        axis.plot(epochs, [row[f"train_{key}"] for row in history], label="train")
        axis.plot(epochs, [row[f"val_{key}"] for row in history], label="validation")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
