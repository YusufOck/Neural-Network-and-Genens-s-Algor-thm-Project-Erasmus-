from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from data_index import (
    ImageRecord,
    discover_records,
    project_root,
    resolve_dataset_root,
    select_max_per_label,
    target_label,
)


class RecordDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        class_to_index: Dict[str, int],
        target_level: str,
        transform,
    ):
        self.records = list(records)
        self.class_to_index = class_to_index
        self.target_level = target_level
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
        label = target_label(record, self.target_level)
        return self.transform(image), self.class_to_index[label]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8/26 sınıflı Multi Cancer eğitim hattı")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--model",
        choices=["small_cnn", "resnet18", "densenet121", "efficientnet_b0"],
        default="resnet18",
    )
    parser.add_argument(
        "--target-level",
        choices=["main", "subclass"],
        default="main",
        help="main=8 üst klasör, subclass=26 alt klasör",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--normalization", choices=["imagenet", "half"], default="imagenet"
    )
    parser.add_argument(
        "--augmentation", choices=["none", "minimal"], default="minimal"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Seçilen omurganın son bloğunu da eğitir",
    )
    parser.add_argument(
        "--experiment-type",
        choices=["pilot", "reproduction", "ablation", "final"],
        default="pilot",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_records(records: Sequence[ImageRecord], seed: int, target_level: str):
    labels = [target_label(record, target_level) for record in records]
    train_records, temp_records = train_test_split(
        list(records), test_size=0.30, random_state=seed, stratify=labels
    )
    temp_labels = [target_label(record, target_level) for record in temp_records]
    val_records, test_records = train_test_split(
        temp_records, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    return train_records, val_records, test_records


def build_transforms(image_size: int, augmentation: str, normalization_name: str):
    if normalization_name == "imagenet":
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalization = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    train_steps = [transforms.Resize((image_size, image_size))]
    if augmentation == "minimal":
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    train_steps.extend([transforms.ToTensor(), normalization])
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def build_model(name: str, num_classes: int, pretrained: bool, fine_tune: bool):
    if name == "small_cnn":
        return SmallCNN(num_classes)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        classifier_in_features = model.fc.in_features
        if pretrained:
            for parameter in model.parameters():
                parameter.requires_grad = False
            if fine_tune:
                for parameter in model.layer4.parameters():
                    parameter.requires_grad = True
        model.fc = nn.Sequential(
            nn.Dropout(0.30), nn.Linear(classifier_in_features, num_classes)
        )
        return model

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        classifier_in_features = model.classifier.in_features
        if pretrained:
            for parameter in model.parameters():
                parameter.requires_grad = False
            if fine_tune:
                for parameter in model.features.denseblock4.parameters():
                    parameter.requires_grad = True
                for parameter in model.features.norm5.parameters():
                    parameter.requires_grad = True
        model.classifier = nn.Sequential(
            nn.Dropout(0.30), nn.Linear(classifier_in_features, num_classes)
        )
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        classifier_in_features = model.classifier[-1].in_features
        if pretrained:
            for parameter in model.parameters():
                parameter.requires_grad = False
            if fine_tune:
                for block in list(model.features.children())[-2:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
        model.classifier = nn.Sequential(
            nn.Dropout(0.30), nn.Linear(classifier_in_features, num_classes)
        )
        return model

    raise ValueError(f"Desteklenmeyen model: {name}")


def class_weight_tensor(records, class_to_index, target_level, device):
    counts = Counter(target_label(record, target_level) for record in records)
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[name]) for name in sorted(class_to_index)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())

        if training and (batch_index % 50 == 0 or batch_index == len(loader)):
            print(f"  batch {batch_index}/{len(loader)}")

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro", zero_division=0),
        "targets": all_targets,
        "predictions": all_predictions,
    }


def save_confusion_matrix(targets, predictions, class_names, output_path):
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.model = "small_cnn"
        args.target_level = "main"
        args.experiment_type = "smoke_test"
        args.epochs = 1
        args.max_per_class = 40
        args.batch_size = 16
        args.num_workers = 0
        args.pretrained = False

    seed_everything(args.seed)
    dataset_root = resolve_dataset_root(args.data_dir)
    records = discover_records(dataset_root)
    records = select_max_per_label(
        records, args.max_per_class, args.seed, args.target_level
    )

    class_names = sorted({target_label(record, args.target_level) for record in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_records, val_records, test_records = split_records(
        records, args.seed, args.target_level
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        project_root() / "artifacts" / "runs" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Deney türü: {args.experiment_type}")
    print(f"Hedef seviyesi: {args.target_level} ({len(class_names)} sınıf)")
    print(f"Kayıt sayısı: {len(records)} | train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    print(f"Sınıflar: {class_names}")

    train_transform, eval_transform = build_transforms(
        args.image_size, args.augmentation, args.normalization
    )
    datasets = {
        "train": RecordDataset(
            train_records, class_to_index, args.target_level, train_transform
        ),
        "val": RecordDataset(
            val_records, class_to_index, args.target_level, eval_transform
        ),
        "test": RecordDataset(
            test_records, class_to_index, args.target_level, eval_transform
        ),
    }
    pin_memory = device.type == "cuda"
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin_memory
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory
        ),
    }

    model = build_model(args.model, len(class_names), args.pretrained, args.fine_tune).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor(
            train_records, class_to_index, args.target_level, device
        )
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = []
    best_val_f1 = -1.0
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    test_summary = {
        "experiment_type": args.experiment_type,
        "model": args.model,
        "target_level": args.target_level,
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "balanced_accuracy": balanced_accuracy_score(
            test_metrics["targets"], test_metrics["predictions"]
        ),
        "class_names": class_names,
        "dataset_root": str(dataset_root),
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "warning": "Metrics are educational and are not evidence of clinical validity.",
    }

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    split_rows = []
    for split_name, split_records_list in (
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ):
        for record in split_records_list:
            split_rows.append(
                {
                    "split": split_name,
                    "label": target_label(record, args.target_level),
                    "main_class": record.main_class,
                    "subclass": record.subclass,
                    "path": str(record.path),
                }
            )
    with (output_dir / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "main_class", "subclass", "path"],
        )
        writer.writeheader()
        writer.writerows(split_rows)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_summary, handle, indent=2, ensure_ascii=False)
    report = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        test_metrics["targets"],
        test_metrics["predictions"],
        class_names,
        output_dir / "confusion_matrix.png",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Test sonuçları:")
    print(json.dumps(test_summary, indent=2, ensure_ascii=False))
    print(report)
    print(f"Çıktı klasörü: {output_dir}")


if __name__ == "__main__":
    main()

