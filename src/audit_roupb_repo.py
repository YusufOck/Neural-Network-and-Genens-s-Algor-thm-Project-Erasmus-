"""Statically audit the locally cloned RoupB/cancer-app repository.

This script parses notebook JSON and stored outputs. It never executes notebook
cells, imports third-party model code, downloads checkpoints, or trains a model.
It also simulates RoupB's file-level 70/15/15 split on the local dataset and
measures how many already-known exact-duplicate groups would cross splits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MODEL_PATTERN = re.compile(
    r"(densenet121|efficientnet_b0|mobilenet_v2|mobilenet_v3_small|"
    r"resnet18|vgg16|vit_b_16)\s*\(",
    re.IGNORECASE,
)


def first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def flatten_output_text(cell: Dict[str, Any]) -> str:
    parts: List[str] = []
    for output in cell.get("outputs", []):
        text = output.get("text", [])
        parts.extend(text if isinstance(text, list) else [text])
        data = output.get("data", {})
        for key in ("text/plain", "text/markdown"):
            value = data.get(key, [])
            parts.extend(value if isinstance(value, list) else ([value] if value else []))
    return "\n".join(str(part) for part in parts)


def float_matches(pattern: str, text: str) -> List[float]:
    return [float(value) for value in re.findall(pattern, text, re.IGNORECASE)]


def audit_notebook(path: Path, repo_root: Path) -> Dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    output = "\n".join(flatten_output_text(cell) for cell in notebook.get("cells", []))

    classes_match = re.search(r"Classes:\s*\[([^\]]+)\]", output)
    classes = (
        re.findall(r"['\"]([^'\"]+)['\"]", classes_match.group(1))
        if classes_match
        else []
    )
    model_match = MODEL_PATTERN.search(code)

    return {
        "path": str(path.relative_to(repo_root)).replace("\\", "/"),
        "cell_count": len(notebook.get("cells", [])),
        "model": model_match.group(1) if model_match else None,
        "max_images_per_class": int(value)
        if (value := first_match(r"MAX_IMAGES_PER_CLASS\s*=\s*([0-9_]+)", code))
        else None,
        "image_size": int(value)
        if (value := first_match(r"IMG_SIZE\s*=\s*([0-9_]+)", code))
        else None,
        "batch_size": int(value)
        if (value := first_match(r"BATCH_SIZE\s*=\s*([0-9_]+)", code))
        else None,
        "default_epochs": int(value)
        if (value := first_match(r"def\s+train_model\s*\(\s*epochs\s*=\s*([0-9_]+)", code))
        else None,
        "split_literal": first_match(r"SPLITS\s*=\s*(\{[^\n]+\})", code),
        "classes": classes,
        "validation_accuracy_outputs": float_matches(
            r"Validation Accuracy:\s*([0-9.]+)", output
        ),
        "test_accuracy_outputs": float_matches(r"Test Accuracy:\s*([0-9.]+)", output),
        "generic_accuracy_outputs": float_matches(r"(?<!Test )(?<!Validation )Accuracy:\s*([0-9.]+)", output),
        "uses_pretrained_weights": bool(
            re.search(r"weights\s*=\s*(?!None\b)[A-Za-z_]", code, re.IGNORECASE)
        ),
        "freezes_parameters": "requires_grad = False" in code,
        "uses_image_augmentation": any(
            name in code
            for name in (
                "RandomHorizontalFlip",
                "RandomVerticalFlip",
                "RandomRotation",
                "RandomResizedCrop",
                "ColorJitter",
            )
        ),
        "metrics_in_code": [
            metric
            for metric in (
                "classification_report",
                "f1_score",
                "precision_score",
                "recall_score",
                "confusion_matrix",
            )
            if metric in code
        ],
    }


def get_repo_head(repo_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def iter_leaf_directories(dataset_root: Path) -> Iterable[Path]:
    for parent_name in os.listdir(dataset_root):
        parent = dataset_root / parent_name
        if not parent.is_dir():
            continue
        for leaf_name in os.listdir(parent):
            leaf = parent / leaf_name
            if leaf.is_dir():
                yield leaf


def simulate_roupb_split(
    dataset_root: Path, duplicate_data: Dict[str, Any]
) -> Dict[str, Any]:
    random.seed(42)
    assignments: Dict[str, str] = {}
    selected_backups: List[str] = []
    omitted_after_cap: List[str] = []

    for leaf in iter_leaf_directories(dataset_root):
        names = os.listdir(leaf)
        random.shuffle(names)
        selected = names[:5000]
        omitted = names[5000:]

        selected_backups.extend(
            str((leaf / name).relative_to(dataset_root)).replace("/", "\\")
            for name in selected
            if ".bak." in name
        )
        omitted_after_cap.extend(
            str((leaf / name).relative_to(dataset_root)).replace("/", "\\")
            for name in omitted
            if ".bak." not in name
        )

        train_end = int(len(selected) * 0.70)
        validation_end = train_end + int(len(selected) * 0.15)
        split_names = {
            "train": selected[:train_end],
            "validation": selected[train_end:validation_end],
            "test": selected[validation_end:],
        }
        for split, filenames in split_names.items():
            for filename in filenames:
                relative = str((leaf / filename).relative_to(dataset_root)).replace("/", "\\")
                assignments[relative] = split

    crossing_groups = 0
    files_in_crossing_groups = 0
    split_patterns: Counter[str] = Counter()
    leaf_counts: Counter[str] = Counter()

    for group in duplicate_data["duplicate_groups"]:
        splits = [assignments.get(path) for path in group["paths"]]
        unique_splits = sorted({split for split in splits if split})
        if len(unique_splits) <= 1:
            continue
        crossing_groups += 1
        files_in_crossing_groups += len(splits)
        split_patterns["+".join(unique_splits)] += 1
        leaf_counts[group["subclasses"][0]] += 1

    return {
        "method": "Local simulation of RoupB seed-42 os.listdir/shuffle and 70/15/15 split",
        "warning": (
            "Filesystem listing order can differ from RoupB's machine; this proves the algorithm "
            "does not prevent duplicate leakage, not their exact historical assignment."
        ),
        "assigned_files": len(assignments),
        "selected_backup_files": selected_backups,
        "omitted_standard_files_after_5000_cap": omitted_after_cap,
        "exact_duplicate_groups_total": duplicate_data["exact_duplicate_group_count"],
        "cross_split_exact_duplicate_groups": crossing_groups,
        "files_in_cross_split_exact_duplicate_groups": files_in_crossing_groups,
        "split_pattern_counts": dict(sorted(split_patterns.items())),
        "crossing_group_counts_by_leaf": dict(sorted(leaf_counts.items())),
    }


def implied_mcnemar_accuracies(repo_root: Path) -> Dict[str, Any]:
    csv_path = repo_root / "mcnemar_results.csv"
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    # The first repository row is DenseNet vs EfficientNet.
    values = rows[1].split(",")
    both_wrong = int(values[2])
    model1_wrong_model2_correct = int(values[3])
    model1_correct_model2_wrong = int(values[4])
    both_correct = int(values[5])
    total = both_wrong + model1_wrong_model2_correct + model1_correct_model2_wrong + both_correct
    return {
        "first_pair": [values[0], values[1]],
        "sample_count": total,
        "model_1_implied_accuracy": (model1_correct_model2_wrong + both_correct) / total,
        "model_2_implied_accuracy": (model1_wrong_model2_correct + both_correct) / total,
        "repository_conclusion_conflicts_with_table": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notebooks = [
        audit_notebook(path, args.repo)
        for path in sorted((args.repo / "model_codes").glob("**/*.ipynb"))
    ]
    duplicate_data = json.loads(args.duplicates.read_text(encoding="utf-8"))
    result = {
        "audit_date": "2026-10-15",
        "execution_policy": "Static JSON/source inspection only; notebook cells were not executed",
        "repository": "https://github.com/RoupB/cancer-app",
        "local_repo": str(args.repo),
        "commit": get_repo_head(args.repo),
        "notebook_count": len(notebooks),
        "notebooks": notebooks,
        "split_simulation": simulate_roupb_split(args.dataset, duplicate_data),
        "mcnemar_csv_check": implied_mcnemar_accuracies(args.repo),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Audited {len(notebooks)} model notebooks without executing them")
    print(
        "Cross-split exact-duplicate groups:",
        result["split_simulation"]["cross_split_exact_duplicate_groups"],
    )


if __name__ == "__main__":
    main()
