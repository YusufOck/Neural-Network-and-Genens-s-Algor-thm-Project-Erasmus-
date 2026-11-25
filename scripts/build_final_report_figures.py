"""Build the verified visual assets used by the NNGA final report.

This script does not train or evaluate a model. It reads frozen local artifacts,
the supplied dataset and downloaded review packages, then creates publication-style
figures under research/final_report/figures.
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "archive" / "Multi Cancer" / "Multi Cancer"
OUT = ROOT / "research" / "final_report" / "figures"
DOWNLOADS = Path(r"C:\Users\mehmet\Downloads")

BEST_ZIP = DOWNLOADS / "resnet18_lastblock_finetune_v1_review_package.zip"
GA_SEARCH_ZIP = DOWNLOADS / "ga_resnet18_lastblock_search_v1_review_package.zip"

NAVY = "#17324D"
BLUE = "#2F6690"
TEAL = "#3A7D8C"
LIGHT_BLUE = "#DCEAF4"
GREEN = "#4E8C6A"
ORANGE = "#D9822B"
RED = "#B64949"
GRAY = "#6B7280"
LIGHT_GRAY = "#EEF2F5"
INK = "#1F2933"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#AAB4BE",
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def cover_image(path: Path, size: tuple[int, int]) -> Image.Image:
    """Resize and center-crop an image so dataset edge artefacts do not dominate a montage tile."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(size[0], round(image.width * scale)), max(size[1], round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def draw_box(ax, xy, width, height, text, facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.5):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=lw, color=color))


def project_workflow() -> None:
    labels = [
        "Dataset and\nlabel audit",
        "Exact and perceptual\nduplicate\nanalysis",
        "Known-family-grouped\n70/15/15\nsplit",
        "Controlled model\ntraining",
        "Error and Grad-CAM\naudit",
        "GA versus random\nsearch",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    x_positions = [0.15, 2.15, 4.15, 6.15, 8.15, 10.15]
    colors = ["#E7F0F7", "#E9F3F1", "#F4F1E7", "#EDEAF5", "#F7EDEA", "#E8F1E9"]
    edges = [BLUE, TEAL, "#8B7438", "#6C5B8C", RED, GREEN]
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        draw_box(ax, (x, 1.00), 1.55, 1.0, label, colors[index], edges[index], 8.1)
        ax.text(x + 0.78, 2.18, str(index + 1), ha="center", va="center", fontsize=10,
                color="white", bbox=dict(boxstyle="circle,pad=0.25", fc=edges[index], ec="none"))
        if index < len(labels) - 1:
            arrow(ax, (x + 1.58, 1.5), (x_positions[index + 1] - 0.05, 1.5))
    ax.text(6, 0.43, "Frozen evidence, shared split and hash-verified artifacts",
            ha="center", va="center", color=NAVY, fontsize=10.5, fontweight="bold")
    save_figure(fig, "01_project_workflow.png")


def modality_montage() -> None:
    parent_specs = [
        ("ALL", "all_benign", "Blood-smear microscopy"),
        ("Brain Cancer", "brain_glioma", "Contrast-enhanced MRI"),
        ("Breast Cancer", "breast_benign", "H&E histopathology"),
        ("Cervical Cancer", "cervix_dyk", "Pap-smear microscopy"),
        ("Kidney Cancer", "kidney_normal", "CT"),
        ("Lung and Colon Cancer", "lung_aca", "H&E histopathology"),
        ("Lymphoma", "lymph_cll", "H&E histopathology"),
        ("Oral Cancer", "oral_normal", "H&E histopathology"),
    ]
    tile_w, tile_h = 390, 330
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, True)
    subtitle_font = load_font(20, False)
    for index, (parent, leaf, modality) in enumerate(parent_specs):
        folder = DATA_ROOT / parent / leaf
        image_path = sorted(folder.glob("*.jpg"))[0]
        image = cover_image(image_path, (tile_w - 20, 245))
        col, row = index % 4, index // 4
        x, y = col * tile_w, row * tile_h
        canvas.paste(image, (x + 10, y + 8))
        draw.rectangle([x + 10, y + 8, x + tile_w - 10, y + 253], outline="#CBD5DF", width=2)
        draw.text((x + 14, y + 260), parent, fill=NAVY, font=title_font)
        draw.text((x + 14, y + 294), modality, fill=GRAY, font=subtitle_font)
    canvas.save(OUT / "02_dataset_modalities.png", quality=95)


def parent_distribution() -> None:
    groups = ["ALL", "Brain", "Breast", "Cervical", "Kidney", "Lung and\nColon", "Lymphoma", "Oral"]
    counts = np.array([20000, 15000, 10000, 25000, 10000, 25000, 15000, 10000])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(groups, counts / 1000, color=[BLUE, TEAL, "#7A6F9B", ORANGE, "#557A95", GREEN, "#9B6A6C", "#8B7438"])
    ax.set_ylabel("Images (thousands)")
    ax.set_title("Dataset distribution across eight parent groups")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.45,
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    ax.text(0.995, 0.96, "26 leaf classes; 5,000 images per leaf",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT_BLUE, ec="none"))
    save_figure(fig, "03_parent_distribution.png")


def phash_pipeline() -> None:
    labels = [
        "RGB image",
        "Grayscale and\nresize to 32 by 32",
        "Two-dimensional\nDCT",
        "Retain 8 by 8\nlow frequencies",
        "Median\nthreshold",
        "64-bit hash",
    ]
    fig, ax = plt.subplots(figsize=(11.3, 2.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.7)
    ax.axis("off")
    xs = [0.1, 2.05, 4.0, 5.95, 7.9, 9.85]
    for i, (x, label) in enumerate(zip(xs, labels)):
        color = LIGHT_BLUE if i not in (2, 4) else "#E9F3F1"
        edge = BLUE if i not in (2, 4) else TEAL
        draw_box(ax, (x, 1.05), 1.45, 0.78, label, color, edge, 9.2)
        if i < len(xs) - 1:
            arrow(ax, (x + 1.47, 1.44), (xs[i + 1] - 0.05, 1.44))
    ax.text(6, 0.45,
            "Hamming distance counts differing bits; pHash proposes candidates but does not make the final duplicate decision",
            ha="center", va="center", fontsize=9.5, color=NAVY)
    save_figure(fig, "04_phash_pipeline.png")


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions.reshape(-1, 1)
    matrix = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size))
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix.astype(np.float32)


def perceptual_hash(path: Path) -> int:
    transform = dct_matrix(32)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = transform @ pixels @ transform.T
    low = coefficients[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    bits = low > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def duplicate_examples() -> None:
    exact_json = json.loads((ROOT / "artifacts" / "audit" / "exact_duplicates.json").read_text(encoding="utf-8"))
    exact_paths = [DATA_ROOT / path for path in exact_json["duplicate_groups"][0]["paths"][:2]]
    perceptual_paths = [
        DATA_ROOT / "Lung and Colon Cancer" / "colon_aca" / "colon_aca_0700.jpg",
        DATA_ROOT / "Lung and Colon Cancer" / "colon_aca" / "colon_aca_4354.jpg",
    ]
    distance = (perceptual_hash(perceptual_paths[0]) ^ perceptual_hash(perceptual_paths[1])).bit_count()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.7))
    rows = [(exact_paths, "Byte-identical SHA-256 pair"),
            (perceptual_paths, f"Reviewed same-source candidate; pHash distance {distance}")]
    for row, (paths, row_title) in enumerate(rows):
        for col, path in enumerate(paths):
            with Image.open(path) as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].axis("off")
            axes[row, col].set_title(path.name, fontsize=9, color=GRAY)
        axes[row, 0].text(-0.02, 1.13, row_title, transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight="bold", color=NAVY)
    fig.suptitle("Examples motivating duplicate-aware splitting", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "05_duplicate_examples.png")


def candidate_load() -> None:
    groups = ["ALL", "Brain MRI", "Breast", "Cervix", "Kidney CT", "Lung/colon", "Lymphoma", "Oral"]
    v23 = np.array([21.136, 123.916, 27.788, 15.140, 65.979, 18.446, 22.634, 22.363])
    v24 = np.array([21.136, 18.187, 27.788, 15.140, 11.535, 18.446, 22.634, 22.363])
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    ax.barh(y + 0.18, v23, height=0.34, color="#A9C4D8", label="v2.3 global pHash rule")
    ax.barh(y - 0.18, v24, height=0.34, color=TEAL, label="v2.4 modality-aware counterfactual")
    ax.axvline(25, color=RED, linestyle="--", linewidth=1.4, label="Frozen capacity gate")
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("Unique candidate pairs per image")
    ax.set_title("Candidate retrieval load varied strongly by imaging modality")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "06_candidate_load.png")


def split_and_components() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.0, 1.4]})
    sizes = [91000, 19500, 19500]
    labels = ["Train\n91,000", "Validation\n19,500", "Test\n19,500"]
    colors = [BLUE, ORANGE, GREEN]
    axes[0].pie(sizes, labels=labels, colors=colors, startangle=90,
                wedgeprops=dict(width=0.42, edgecolor="white"),
                textprops=dict(fontsize=10, color=INK))
    axes[0].text(0, 0, "130,000\nimages", ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)
    axes[0].set_title("Fixed 70/15/15 assignment")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Known-family grouping before assignment")
    nodes = {"A": (1.1, 4.4), "B": (2.7, 4.4), "C": (4.3, 4.4), "D": (1.9, 2.1), "E": (5.9, 2.1)}
    for name, (x, y) in nodes.items():
        ax.text(x, y, name, ha="center", va="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.42", fc=BLUE if name in "ABC" else GRAY, ec="none"))
    ax.plot([1.35, 2.45], [4.4, 4.4], color=TEAL, lw=2)
    ax.plot([2.95, 4.05], [4.4, 4.4], color=TEAL, lw=2)
    ax.text(2.7, 5.15, "A-B and B-C links form one component", ha="center", fontsize=9.5, color=NAVY)
    draw_box(ax, (0.55, 0.35), 4.25, 0.95, "Component A-B-C\nassigned as one unit", "#E9F3F1", TEAL, 9.1)
    arrow(ax, (2.7, 3.95), (2.7, 1.3), TEAL)
    draw_box(ax, (5.15, 3.75), 4.3, 1.25,
             "128,560 components\n1,402 multi-image components\nmaximum component size 4",
             LIGHT_GRAY, GRAY, 9.3)
    draw_box(ax, (5.15, 0.55), 4.3, 1.25,
             "Zero cross-split exact groups\nZero reviewed-link crossings\nZero known-component crossings",
             "#E8F1E9", GREEN, 9.3)
    fig.tight_layout()
    save_figure(fig, "07_split_and_components.png")


def smallcnn_architecture() -> None:
    blocks = [
        ("Input\n3 x 224 x 224", LIGHT_GRAY, GRAY),
        ("Conv 3 to 32\nBatchNorm ReLU\nMaxPool", LIGHT_BLUE, BLUE),
        ("Conv 32 to 64\nBatchNorm ReLU\nMaxPool", LIGHT_BLUE, BLUE),
        ("Conv 64 to 128\nBatchNorm ReLU", "#E9F3F1", TEAL),
        ("Adaptive average\npooling 1 x 1", "#F4F1E7", "#8B7438"),
        ("Dropout 0.35\nLinear 128 to 26", "#EDEAF5", "#6C5B8C"),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 3.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3)
    ax.axis("off")
    xs = [0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
    for i, (x, (label, fc, ec)) in enumerate(zip(xs, blocks)):
        draw_box(ax, (x, 1.0), 1.55, 1.0, label, fc, ec, 9.2)
        if i < len(xs) - 1:
            arrow(ax, (x + 1.58, 1.5), (xs[i + 1] - 0.05, 1.5))
    ax.text(6.1, 0.38, "Approximately 97,050 trainable parameters; no pretrained weights",
            ha="center", va="center", fontsize=10.2, color=NAVY, fontweight="bold")
    save_figure(fig, "08_smallcnn_architecture.png")


def model_comparison() -> None:
    models = ["Frozen-head\nResNet18", "SmallCNN", "Last-block\nResNet18", "GA-selected\nResNet18", "Random-selected\nResNet18"]
    accuracy = np.array([0.9130256410, 0.9558461538, 0.9964615385, 0.9957948718, 0.9957435897]) * 100
    macro_f1 = np.array([0.9124547963, 0.9559161453, 0.9964624119, 0.9957937944, 0.9957445252]) * 100
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    width = 0.36
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color=BLUE)
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color=TEAL)
    ax.set_ylim(88, 100.8)
    ax.set_ylabel("Test score (%)")
    ax.set_title("All reportable models used the same 19,500-image test split")
    ax.set_xticks(x, models)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.2, rotation=90)
    save_figure(fig, "09_model_comparison.png")


def extract_best_model_visuals() -> None:
    if not BEST_ZIP.is_file():
        raise FileNotFoundError(BEST_ZIP)
    with zipfile.ZipFile(BEST_ZIP) as archive:
        for source, target in [
            ("learning_curves.png", "10_best_training_curves.png"),
            ("leaf_confusion_matrix.png", "11_best_leaf_confusion_matrix.png"),
            ("parent_confusion_matrix.png", "12_best_parent_confusion_matrix.png"),
        ]:
            (OUT / target).write_bytes(archive.read(source))


def error_pair_distribution() -> None:
    path = ROOT / "research" / "model_results" / "resnet18_lastblock_finetune_v1" / "error_analysis_v1" / "error_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(f"{row['true_leaf']} -> {row['predicted_leaf']}" for row in rows)
    top = counts.most_common(10)
    labels = [item[0] for item in top][::-1]
    values = [item[1] for item in top][::-1]
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    bars = ax.barh(labels, values, color=[TEAL if "oral" in label else BLUE for label in labels])
    ax.set_xlabel("Number of test errors")
    ax.set_title("Most frequent leaf-level error directions in the best model")
    ax.grid(axis="x", alpha=0.2)
    for bar in bars:
        ax.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())),
                va="center", fontsize=9)
    save_figure(fig, "13_error_pair_distribution.png")


def copy_audit_visuals() -> None:
    sources = [
        (ROOT / "research" / "model_results" / "resnet18_lastblock_finetune_v1" / "error_analysis_v1" / "reliability_diagram.png",
         OUT / "14_reliability_diagram.png"),
        (ROOT / "research" / "model_results" / "resnet18_lastblock_finetune_v1" / "gradcam_v1" / "contact_sheets" / "gradcam_sheet_01.jpg",
         OUT / "15_gradcam_examples.jpg"),
        (ROOT / "research" / "model_results" / "resnet18_lastblock_finetune_v1" / "error_analysis_v1" / "contact_sheets" / "error_sheet_001.png",
         OUT / "16_error_examples.png"),
    ]
    for source, target in sources:
        shutil.copy2(source, target)


def prediction_examples() -> None:
    with zipfile.ZipFile(BEST_ZIP) as archive:
        text = archive.read("test_predictions.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    correct_pool = [row for row in rows if row["leaf_correct"] == "True"]
    errors = [row for row in rows if row["leaf_correct"] == "False"]
    correct_targets = ["brain_glioma", "cervix_dyk", "kidney_tumor"]
    selected_correct = []
    for target in correct_targets:
        options = [row for row in correct_pool if row["true_leaf"] == target]
        options.sort(key=lambda row: abs(float(row["confidence"]) - 0.98))
        selected_correct.append(options[0])
    selected_errors = sorted(errors, key=lambda row: (row["parent_correct"] == "True", -float(row["confidence"])))[:3]
    selected = selected_correct + selected_errors
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.0))
    for index, (ax, row) in enumerate(zip(axes.ravel(), selected)):
        path = DATA_ROOT / Path(row["relative_path"])
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.axis("off")
        status = "Correct" if row["leaf_correct"] == "True" else "Error"
        color = GREEN if status == "Correct" else RED
        ax.set_title(
            f"{status}\nTrue: {row['true_leaf']}\nPredicted: {row['predicted_leaf']}\nConfidence: {float(row['confidence']):.3f}",
            fontsize=9, color=color, fontweight="bold"
        )
    fig.suptitle("Representative best-model predictions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "17_prediction_examples.png")


def ga_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    draw_box(ax, (0.25, 1.45), 1.65, 1.0, "Chromosome\nlearning rate\nweight decay\ndropout", LIGHT_BLUE, BLUE, 8.8)
    draw_box(ax, (2.4, 1.45), 1.55, 1.0, "Population\n4 candidates", "#E9F3F1", TEAL, 9.2)
    draw_box(ax, (4.45, 1.45), 1.55, 1.0, "Train each\nfor 4 epochs", "#F4F1E7", "#8B7438", 9.2)
    draw_box(ax, (6.5, 1.45), 1.55, 1.0, "Fitness\nbest validation\nmacro-F1", "#EDEAF5", "#6C5B8C", 9.2)
    draw_box(ax, (8.55, 1.45), 1.55, 1.0, "Tournament\nselection and\ncrossover", "#F7EDEA", RED, 9.2)
    draw_box(ax, (10.55, 1.45), 1.2, 1.0, "Mutation\nprobability\n0.35", "#E8F1E9", GREEN, 8.8)
    xs = [1.9, 3.95, 6.0, 8.05, 10.1]
    ends = [2.35, 4.4, 6.45, 8.5, 10.5]
    for start, end in zip(xs, ends):
        arrow(ax, (start, 1.95), (end, 1.95))
    ax.annotate("", xy=(3.15, 1.35), xytext=(11.15, 1.35),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.22", lw=1.4, color=GRAY))
    ax.text(7.15, 0.58, "Three rounds, 12 unique GA trials; equal budget of 12 random-search trials",
            ha="center", va="center", fontsize=10, color=NAVY, fontweight="bold")
    save_figure(fig, "18_ga_workflow.png")


def ga_search_comparison() -> None:
    with zipfile.ZipFile(GA_SEARCH_ZIP) as archive:
        text = archive.read("search_results.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    ga = sorted(float(row["fitness"]) for row in rows if row["method"] == "ga")
    random_values = sorted(float(row["fitness"]) for row in rows if row["method"] == "random")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(np.arange(1, 13), [float(row["fitness"]) * 100 for row in rows if row["method"] == "ga"],
                    color=BLUE, s=48, label="GA")
    axes[0].scatter(np.arange(1, 13), [float(row["fitness"]) * 100 for row in rows if row["method"] == "random"],
                    color=ORANGE, s=48, label="Random search")
    axes[0].set_xlabel("Trial sequence within method")
    axes[0].set_ylabel("Best proxy validation macro-F1 (%)")
    axes[0].set_title("Equal-budget proxy trials")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].boxplot([np.array(ga) * 100, np.array(random_values) * 100], tick_labels=["GA", "Random"],
                    patch_artist=True,
                    boxprops=dict(facecolor=LIGHT_BLUE, color=BLUE),
                    medianprops=dict(color=RED, linewidth=1.6),
                    whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY))
    axes[1].scatter([1, 2], [max(ga) * 100, max(random_values) * 100], color=[BLUE, ORANGE], s=70, zorder=3)
    axes[1].set_ylabel("Proxy validation macro-F1 (%)")
    axes[1].set_title("Fitness distribution and best candidates")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "19_ga_proxy_comparison.png")


def ga_shared_test() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    methods = ["GA", "Random"]
    accuracy = [99.579487, 99.574359]
    macro_f1 = [99.579379, 99.574453]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, accuracy, width, color=BLUE, label="Accuracy")
    axes[0].bar(x + width / 2, macro_f1, width, color=TEAL, label="Macro-F1")
    axes[0].set_xticks(x, methods)
    axes[0].set_ylim(99.45, 99.66)
    axes[0].set_ylabel("Shared-test score (%)")
    axes[0].set_title("Shared-test performance")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    paired_labels = ["Both correct", "GA only", "Random only", "Both wrong"]
    paired_counts = [19366, 52, 51, 31]
    colors = [GREEN, BLUE, ORANGE, RED]
    bars = axes[1].bar(paired_labels, paired_counts, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Images on logarithmic scale")
    axes[1].set_title("Paired correctness counts")
    axes[1].tick_params(axis="x", rotation=20)
    for bar, count in zip(bars, paired_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count * 1.15, f"{count:,}", ha="center", fontsize=9)
    axes[1].text(0.98, 0.95, "Exact McNemar p = 1.0", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9.5, color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT_BLUE, ec="none"))
    fig.tight_layout()
    save_figure(fig, "20_ga_shared_test.png")


def main() -> None:
    configure_plotting()
    OUT.mkdir(parents=True, exist_ok=True)
    project_workflow()
    modality_montage()
    parent_distribution()
    phash_pipeline()
    duplicate_examples()
    candidate_load()
    split_and_components()
    smallcnn_architecture()
    model_comparison()
    extract_best_model_visuals()
    error_pair_distribution()
    copy_audit_visuals()
    prediction_examples()
    ga_workflow()
    ga_search_comparison()
    ga_shared_test()
    generated = sorted(
        path.name for path in OUT.iterdir()
        if path.is_file() and path.name != "figure_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "status": "nnga_final_report_figures_complete",
        "figure_count": len(generated),
        "figures": generated,
        "does_not_train_or_evaluate_models": True,
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
