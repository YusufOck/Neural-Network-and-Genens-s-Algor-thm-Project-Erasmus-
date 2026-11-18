from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
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
from fixed_split_manifest import load_fixed_split_manifest, load_search_proxy_manifest
from training_support import (
    build_subclass_parent_map,
    classification_metrics,
    derive_parent_predictions,
    prediction_rows,
    save_learning_curves,
    set_training_mode,
    sha256_file,
    write_prediction_csv,
)


FROZEN_SPLIT_SHA256 = "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171"


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
        return self.transform(image), self.class_to_index[label], index


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.35) -> None:
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
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

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
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Classifier dropout; varsayilan small_cnn=0.35, diger modeller=0.30",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Pilot icin label basina ust sinir; sabit manifest ile kullanilamaz",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Onceden uretilmis sabit train/val/test manifesti",
    )
    parser.add_argument(
        "--search-proxy-manifest",
        default=None,
        help=(
            "Yalniz hiperparametre aramasi icin sabit splitin train/val alt kumesi; "
            "test goruntuleri acilmaz"
        ),
    )
    parser.add_argument(
        "--search-proxy-manifest-sha256",
        default=None,
        help="Search proxy manifesti icin zorunlu beklenen SHA-256",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Validation macro-F1 iyilesmezse beklenecek epoch; 0 erken durdurmayi kapatir",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Seçilen omurganın son bloğunu da eğitir",
    )
    parser.add_argument(
        "--experiment-type",
        choices=[
            "pilot",
            "baseline",
            "reproduction",
            "ablation",
            "search_trial",
            "final",
        ],
        default="pilot",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ayni output klasorundeki last_checkpoint.pth dosyasindan devam et",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Yeni bir deneyin model agirliklarini bu best_model.pth dosyasindan baslatir; "
            "optimizer/history tasinmaz"
        ),
    )
    parser.add_argument(
        "--init-checkpoint-sha256",
        default=None,
        help="Init checkpoint icin beklenen SHA-256; verilirse yuklemeden once zorunlu dogrulanir",
    )
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


def resolve_dropout(name: str, requested: float | None) -> float:
    value = (0.35 if name == "small_cnn" else 0.30) if requested is None else requested
    if not 0.0 <= value < 1.0:
        raise ValueError(f"Dropout [0, 1) araliginda olmalidir: {value}")
    return value


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    fine_tune: bool,
    dropout: float | None = None,
):
    effective_dropout = resolve_dropout(name, dropout)
    if name == "small_cnn":
        return SmallCNN(num_classes, effective_dropout)

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
            nn.Dropout(effective_dropout), nn.Linear(classifier_in_features, num_classes)
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
            nn.Dropout(effective_dropout), nn.Linear(classifier_in_features, num_classes)
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
            nn.Dropout(effective_dropout), nn.Linear(classifier_in_features, num_classes)
        )
        return model

    raise ValueError(f"Desteklenmeyen model: {name}")


def class_weight_tensor(records, class_to_index, target_level, device):
    counts = Counter(target_label(record, target_level) for record in records)
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[name]) for name in sorted(class_to_index)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None, collect_outputs=False):
    training = optimizer is not None
    set_training_mode(model, training)
    total_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_record_indices: List[int] = []
    all_probabilities: List[List[float]] = []

    for batch_index, (inputs, targets, record_indices) in enumerate(loader, start=1):
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
        if collect_outputs:
            all_record_indices.extend(record_indices.tolist())
            all_probabilities.extend(
                torch.softmax(outputs, dim=1).detach().cpu().tolist()
            )

        if training and (batch_index % 50 == 0 or batch_index == len(loader)):
            print(f"  batch {batch_index}/{len(loader)}")

    result = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro", zero_division=0),
        "targets": all_targets,
        "predictions": all_predictions,
    }
    if collect_outputs:
        result["record_indices"] = all_record_indices
        result["probabilities"] = all_probabilities
    return result


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


def write_history(history: Sequence[dict], output_path: Path) -> None:
    if not history:
        return
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output_path)


def save_last_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_initial_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str | None,
    expected_model: str,
    expected_target_level: str,
    expected_class_names: Sequence[str],
) -> tuple[dict, str]:
    """Validate a prior best checkpoint before it initializes a new experiment."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Init checkpoint bulunamadi: {checkpoint_path}")
    actual_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "Init checkpoint SHA-256 beklenen degerle eslesmiyor: "
            f"expected={expected_sha256.lower()}, actual={actual_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Init checkpoint model_state_dict icermiyor")
    checkpoint_class_names = checkpoint.get("class_names")
    if checkpoint_class_names != list(expected_class_names):
        raise ValueError("Init checkpoint sinif sirasi mevcut deneyle eslesmiyor")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Init checkpoint deney argumanlarini icermiyor")
    if checkpoint_args.get("model") != expected_model:
        raise ValueError("Init checkpoint model mimarisi mevcut deneyle eslesmiyor")
    if checkpoint_args.get("target_level") != expected_target_level:
        raise ValueError("Init checkpoint hedef seviyesi mevcut deneyle eslesmiyor")
    return checkpoint, actual_sha256


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

    args.dropout = resolve_dropout(args.model, args.dropout)
    if args.search_proxy_manifest and not args.split_manifest:
        raise ValueError("Search proxy icin --split-manifest zorunludur")
    if args.search_proxy_manifest and args.experiment_type != "search_trial":
        raise ValueError("Search proxy yalniz --experiment-type search_trial ile kullanilir")
    if args.search_proxy_manifest and not args.search_proxy_manifest_sha256:
        raise ValueError("Search proxy icin beklenen SHA-256 zorunludur")
    if args.search_proxy_manifest_sha256 and not args.search_proxy_manifest:
        raise ValueError(
            "--search-proxy-manifest-sha256 icin --search-proxy-manifest verilmelidir"
        )

    seed_everything(args.seed)
    dataset_root = resolve_dataset_root(args.data_dir)
    records = discover_records(dataset_root)
    manifest_sha256 = None
    search_proxy_sha256 = None
    search_proxy_path = None
    if args.split_manifest:
        if args.max_per_class is not None:
            raise ValueError("Sabit split manifesti ile --max-per-class birlikte kullanilamaz")
        manifest_path = Path(args.split_manifest).resolve()
        manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 != FROZEN_SPLIT_SHA256:
            raise ValueError(
                "Split manifest SHA-256 dondurulmus proje manifestiyle eslesmiyor: "
                f"{manifest_sha256}"
            )
        fixed_splits = load_fixed_split_manifest(
            records, dataset_root, manifest_path
        )
        if args.search_proxy_manifest:
            search_proxy_path = Path(args.search_proxy_manifest).resolve()
            search_proxy_sha256 = sha256_file(search_proxy_path)
            if search_proxy_sha256 != args.search_proxy_manifest_sha256.lower():
                raise ValueError(
                    "Search proxy manifest SHA-256 beklenen degerle eslesmiyor: "
                    f"expected={args.search_proxy_manifest_sha256.lower()}, "
                    f"actual={search_proxy_sha256}"
                )
            proxy_splits = load_search_proxy_manifest(
                fixed_splits, dataset_root, search_proxy_path
            )
            train_records = proxy_splits["train"]
            val_records = proxy_splits["val"]
            test_records = []
        else:
            train_records = fixed_splits["train"]
            val_records = fixed_splits["val"]
            test_records = fixed_splits["test"]
    else:
        records = select_max_per_label(
            records, args.max_per_class, args.seed, args.target_level
        )
        train_records, val_records, test_records = split_records(
            records, args.seed, args.target_level
        )

    class_names = sorted({target_label(record, args.target_level) for record in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if args.target_level == "subclass":
        label_to_parent = build_subclass_parent_map(records)
    else:
        label_to_parent = {name: name for name in class_names}

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
    }
    if not args.search_proxy_manifest:
        datasets["test"] = RecordDataset(
            test_records, class_to_index, args.target_level, eval_transform
        )
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
    }
    if not args.search_proxy_manifest:
        loaders["test"] = DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory
        )

    model = build_model(
        args.model,
        len(class_names),
        args.pretrained,
        args.fine_tune,
        args.dropout,
    ).to(device)
    initialization = {
        "type": "default_model_initialization",
        "checkpoint_sha256": None,
    }
    initial_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint).resolve()
        initial_checkpoint, init_checkpoint_sha256 = validate_initial_checkpoint(
            init_checkpoint_path,
            args.init_checkpoint_sha256,
            args.model,
            args.target_level,
            class_names,
        )
        initialization = {
            "type": "prior_best_checkpoint",
            "checkpoint_path": str(init_checkpoint_path),
            "checkpoint_sha256": init_checkpoint_sha256,
        }
        if not args.resume:
            model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)
            print(
                "Yeni deney agirliklari onceki best checkpoint'ten baslatildi: "
                f"{init_checkpoint_sha256}"
            )
    elif args.init_checkpoint_sha256:
        raise ValueError("--init-checkpoint-sha256 icin --init-checkpoint de verilmelidir")
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
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_checkpoint.pth"
    training_contract = {
        "model": args.model,
        "target_level": args.target_level,
        "class_names": class_names,
        "pretrained": args.pretrained,
        "fine_tune": args.fine_tune,
        "image_size": args.image_size,
        "normalization": args.normalization,
        "augmentation": args.augmentation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_manifest_sha256": manifest_sha256,
        "search_proxy_manifest_sha256": search_proxy_sha256,
        "init_checkpoint_sha256": initialization["checkpoint_sha256"],
    }
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint bulunamadi: {last_path}")
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_state.get("training_contract") != training_contract:
            raise ValueError("Resume checkpoint eğitim sozlesmesi mevcut argumanlarla uyusmuyor")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        history = resume_state["history"]
        best_val_f1 = resume_state["best_val_f1"]
        best_epoch = resume_state["best_epoch"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        start_epoch = resume_state["epoch"] + 1
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
        if device.type == "cuda" and resume_state["cuda_random_states"] is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_random_states"])
        print(f"Resume: epoch {start_epoch} ile devam edilecek")
    elif last_path.exists():
        raise FileExistsError(
            f"Mevcut last checkpoint ezilmeyecek: {last_path}. Devam icin --resume kullan."
        )

    for epoch in range(start_epoch, args.epochs + 1):
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
        should_stop = False
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                should_stop = True
        write_history(history, output_dir / "history.csv")
        save_last_checkpoint(
            last_path,
            {
                "training_contract": training_contract,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
        )
        if should_stop:
            print(
                "Erken durdurma: validation macro-F1 "
                f"{args.early_stopping_patience} epoch boyunca iyilesmedi."
            )
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.search_proxy_manifest:
        trial_summary = {
            "schema_version": 1,
            "status": "hyperparameter_search_trial_complete",
            "experiment_type": "search_trial",
            "test_evaluated": False,
            "model": args.model,
            "target_level": args.target_level,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
            "epochs_completed": len(history),
            "split_sizes": {
                "train": len(datasets["train"]),
                "val": len(datasets["val"]),
                "test": 0,
            },
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
            },
            "split_manifest_sha256": manifest_sha256,
            "search_proxy_manifest_sha256": search_proxy_sha256,
            "initialization": initialization,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "warning": (
                "Validation-only proxy-search fitness; not test performance or clinical evidence."
            ),
        }
        shutil.copyfile(manifest_path, output_dir / "split_manifest.csv")
        shutil.copyfile(search_proxy_path, output_dir / "search_proxy_manifest.csv")
        if sha256_file(output_dir / "split_manifest.csv") != manifest_sha256:
            raise RuntimeError("Kopyalanan split manifestinin SHA-256 degeri degisti")
        if sha256_file(output_dir / "search_proxy_manifest.csv") != search_proxy_sha256:
            raise RuntimeError("Kopyalanan search proxy manifestinin SHA-256 degeri degisti")
        write_history(history, output_dir / "history.csv")
        save_learning_curves(history, output_dir / "learning_curves.png")
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "search_trial_summary.json").write_text(
            json.dumps(trial_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Search trial sonucu (test acilmadi):")
        print(json.dumps(trial_summary, indent=2, ensure_ascii=False))
        print(f"Çıktı klasörü: {output_dir}")
        return
    test_metrics = run_epoch(
        model, loaders["test"], criterion, device, collect_outputs=True
    )
    leaf_metrics = classification_metrics(
        test_metrics["targets"], test_metrics["predictions"]
    )
    parent_metrics = derive_parent_predictions(
        test_metrics["targets"],
        test_metrics["predictions"],
        class_names,
        label_to_parent,
    )
    test_summary = {
        "experiment_type": args.experiment_type,
        "model": args.model,
        "target_level": args.target_level,
        "loss": test_metrics["loss"],
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "epochs_completed": len(history),
        "leaf_metrics": leaf_metrics,
        "parent_metrics": {
            key: parent_metrics[key]
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "class_names")
        },
        "class_names": class_names,
        "dataset_root": str(dataset_root),
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "initialization": initialization,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warning": "Metrics are educational and are not evidence of clinical validity.",
    }

    write_history(history, output_dir / "history.csv")
    if args.split_manifest:
        source_manifest = manifest_path
        copied_manifest = output_dir / "split_manifest.csv"
        shutil.copyfile(source_manifest, copied_manifest)
        source_hash = sha256_file(source_manifest)
        copied_hash = sha256_file(copied_manifest)
        if copied_hash != source_hash:
            raise RuntimeError("Kopyalanan split manifestinin SHA-256 degeri degisti")
        test_summary["split_manifest"] = {
            "source": str(source_manifest),
            "sha256": source_hash,
        }
    else:
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
    leaf_report = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    (output_dir / "leaf_classification_report.txt").write_text(
        leaf_report, encoding="utf-8"
    )
    parent_report = classification_report(
        parent_metrics["targets"],
        parent_metrics["predictions"],
        labels=list(range(len(parent_metrics["class_names"]))),
        target_names=parent_metrics["class_names"],
        zero_division=0,
    )
    (output_dir / "parent_classification_report.txt").write_text(
        parent_report, encoding="utf-8"
    )
    report_json = {
        "leaf": classification_report(
            test_metrics["targets"],
            test_metrics["predictions"],
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "parent": classification_report(
            parent_metrics["targets"],
            parent_metrics["predictions"],
            labels=list(range(len(parent_metrics["class_names"]))),
            target_names=parent_metrics["class_names"],
            zero_division=0,
            output_dict=True,
        ),
    }
    (output_dir / "classification_reports.json").write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_confusion_matrix(
        test_metrics["targets"],
        test_metrics["predictions"],
        class_names,
        output_dir / "leaf_confusion_matrix.png",
    )
    save_confusion_matrix(
        parent_metrics["targets"],
        parent_metrics["predictions"],
        parent_metrics["class_names"],
        output_dir / "parent_confusion_matrix.png",
    )
    save_learning_curves(history, output_dir / "learning_curves.png")
    test_prediction_rows = prediction_rows(
        test_records,
        test_metrics["record_indices"],
        test_metrics["targets"],
        test_metrics["predictions"],
        test_metrics["probabilities"],
        class_names,
        label_to_parent,
        dataset_root,
    )
    write_prediction_csv(test_prediction_rows, output_dir / "test_predictions.csv")
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Test sonuçları:")
    print(json.dumps(test_summary, indent=2, ensure_ascii=False))
    print(leaf_report)
    print("Turetilmis 8 ust grup raporu:")
    print(parent_report)
    print(f"Çıktı klasörü: {output_dir}")


if __name__ == "__main__":
    main()

