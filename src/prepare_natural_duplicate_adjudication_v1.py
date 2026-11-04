from __future__ import annotations

"""Prepare the conservative second-stage review queue for natural audit v1.

This script selects only pairs where the frozen score-blind visual review and
frozen duplicate rule v1 do not form a safe consensus.  It creates a traceable
CSV with blank adjudication fields and one score-aware PNG per Pair ID.  It does
not change the original visual decisions, create duplicate families, split the
dataset, or train a classifier.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data_index import project_root, resolve_dataset_root
from duplicate_decision_rule import file_sha256


EXPECTED_UNBLINDED_SHA256 = (
    "00070315628d2ccf300e79fba31e4773adf5e35baa9a8bc9964c2ef3648f1a08"
)
EXPECTED_SCORES_SHA256 = (
    "76d8e739a9a59b26e7ef8de44bdb534e5f8be06b90d748dbfb7774e8c62df3da"
)
EXPECTED_INPUT_ROWS = 345
EXPECTED_QUEUE_ROWS = 25

ADJUDICATION_FIELDS = (
    "adjudication_verdict",
    "adjudication_confidence",
    "adjudicator",
    "adjudication_date",
    "adjudication_notes",
)
QUEUE_FIELDS = (
    "pair_id",
    "review_priority",
    "queue_category",
    "queue_reason",
    "source_group",
    "modality_key",
    "left_leaf",
    "right_leaf",
    "left_path",
    "right_path",
    "visual_verdict",
    "visual_confidence",
    "visual_notes",
    "rule_decision",
    "rule_decision_path",
    "rule_decision_reason",
    "selection_stratum",
    "audit_region",
    "phash_distance",
    "robust_top15",
    "robust_best_union_rank",
    "robust_best_cosine_similarity",
    "orb_inliers",
    "orb_inlier_ratio",
    "orb_coverage_min",
    "orb_affine_plausible",
    "best_alignment_method",
    "best_alignment_orientation",
    "best_ssim",
    "best_nmi",
    "best_correlation",
    "best_overlap",
    "pair_view_file",
) + ADJUDICATION_FIELDS


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Natural audit v1 içindeki 25 konsensüs-dışı çifti boş son-karar "
            "alanları ve Pair-ID görselleriyle adjudication paketine dönüştürür."
        )
    )
    parser.add_argument(
        "--unblinded",
        default=str(
            root
            / "research"
            / "natural_duplicate_audit_v1"
            / "unblinded_pair_results.csv"
        ),
    )
    parser.add_argument(
        "--scores",
        default=str(
            root
            / "artifacts"
            / "audit"
            / "natural_duplicate_rule_evaluation_v1"
            / "pair_scores_and_decisions.csv"
        ),
    )
    parser.add_argument(
        "--queue-dir",
        default=str(root / "research" / "natural_duplicate_adjudication_v1"),
    )
    parser.add_argument(
        "--render-dir",
        default=str(
            root / "artifacts" / "audit" / "natural_duplicate_adjudication_v1"
        ),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--thumb-size", type=int, default=430)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, [dict(row) for row in reader]


def rows_by_pair_id(
    rows: Sequence[Mapping[str, str]], source_name: str
) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{source_name}: boş Pair ID bulundu.")
        if pair_id in indexed:
            raise ValueError(f"{source_name}: tekrarlanan Pair ID: {pair_id}")
        indexed[pair_id] = row
    return indexed


def is_safe_consensus(row: Mapping[str, str]) -> bool:
    visual = row["human_verdict"]
    rule = row["decision"]
    return (visual == "same_source_candidate" and rule == "same_source") or (
        visual == "independent_lookalike" and rule == "independent"
    )


def queue_category(row: Mapping[str, str]) -> tuple[int, str, str]:
    visual = row["human_verdict"]
    rule = row["decision"]
    if visual == "independent_lookalike" and rule == "same_source":
        return (
            1,
            "direct_conflict",
            "Score-blind görsel inceleme bağımsız, frozen kural aynı-kaynak dedi.",
        )
    if visual == "uncertain":
        return (
            2,
            "visual_uncertain",
            "Score-blind görsel inceleme kesin karar veremedi.",
        )
    if rule == "uncertain":
        return (
            3,
            "rule_uncertain",
            "Frozen kural güvenli bir kesin karar vermek yerine çekimser kaldı.",
        )
    return (
        4,
        "other_disagreement",
        "Görsel referans ile frozen kural güvenli konsensüs oluşturmadı.",
    )


def select_queue_rows(
    unblinded_rows: Sequence[Mapping[str, str]],
    score_rows: Sequence[Mapping[str, str]],
    render_relative_root: str = "artifacts/audit/natural_duplicate_adjudication_v1",
) -> list[dict[str, str]]:
    unblinded = rows_by_pair_id(unblinded_rows, "unblinded")
    scores = rows_by_pair_id(score_rows, "scores")
    if set(unblinded) != set(scores):
        missing_scores = sorted(set(unblinded) - set(scores))
        missing_unblinded = sorted(set(scores) - set(unblinded))
        raise ValueError(
            "Pair ID kapsamı eşleşmiyor: "
            f"scores_missing={missing_scores[:5]}, unblinded_missing={missing_unblinded[:5]}"
        )

    output: list[dict[str, str]] = []
    for pair_id, scored in scores.items():
        if is_safe_consensus(scored):
            continue
        source = unblinded[pair_id]
        if source["visual_verdict"] != scored["human_verdict"]:
            raise ValueError(f"Görsel karar kaynakları uyuşmuyor: {pair_id}")
        priority, category, reason = queue_category(scored)
        row = {
            "pair_id": pair_id,
            "review_priority": str(priority),
            "queue_category": category,
            "queue_reason": reason,
            "source_group": scored["source_group"],
            "modality_key": scored["modality_key"],
            "left_leaf": scored["left_leaf"],
            "right_leaf": scored["right_leaf"],
            "left_path": source["left_path"],
            "right_path": source["right_path"],
            "visual_verdict": scored["human_verdict"],
            "visual_confidence": source["confidence"],
            "visual_notes": source["review_notes"],
            "rule_decision": scored["decision"],
            "rule_decision_path": scored["decision_path"],
            "rule_decision_reason": scored["decision_reason"],
            "selection_stratum": scored["selection_stratum"],
            "audit_region": scored["audit_region"],
            "phash_distance": scored["phash_distance"],
            "robust_top15": scored["robust_top15"],
            "robust_best_union_rank": scored["robust_best_union_rank"],
            "robust_best_cosine_similarity": scored[
                "robust_best_cosine_similarity"
            ],
            "orb_inliers": scored["orb_inliers"],
            "orb_inlier_ratio": scored["orb_inlier_ratio"],
            "orb_coverage_min": scored["orb_coverage_min"],
            "orb_affine_plausible": scored["orb_affine_plausible"],
            "best_alignment_method": scored["best_alignment_method"],
            "best_alignment_orientation": scored["best_alignment_orientation"],
            "best_ssim": scored["best_ssim"],
            "best_nmi": scored["best_nmi"],
            "best_correlation": scored["best_correlation"],
            "best_overlap": scored["best_overlap"],
            "pair_view_file": (
                f"{render_relative_root}/pair_views/{pair_id}.png"
            ),
        }
        row.update({field: "" for field in ADJUDICATION_FIELDS})
        output.append(row)

    output.sort(key=lambda row: (int(row["review_priority"]), row["pair_id"]))
    return output


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def _fmt(value: str, digits: int = 3) -> str:
    if not value:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def render_pair_card(
    row: Mapping[str, str], dataset_root: Path, thumb_size: int
) -> Image.Image:
    left_path = (dataset_root / Path(row["left_path"])).resolve()
    right_path = (dataset_root / Path(row["right_path"])).resolve()
    if not left_path.is_relative_to(dataset_root) or not right_path.is_relative_to(
        dataset_root
    ):
        raise ValueError(f"Dataset kökü dışına çıkan yol: {row['pair_id']}")
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"Görüntü bulunamadı: {row['pair_id']}")

    margin = 24
    gap = 18
    header_height = 92
    text_height = 320
    width = margin * 2 + thumb_size * 2 + gap
    height = header_height + thumb_size + text_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(24)
    body_font = font(17)
    small_font = font(15)

    draw.text(
        (margin, 16),
        f"{row['pair_id']} | priority {row['review_priority']} | {row['queue_category']}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (margin, 52),
        f"{row['source_group']} | {row['left_leaf']} vs {row['right_leaf']}",
        fill="#333333",
        font=body_font,
    )

    image_y = header_height
    canvas.paste(thumbnail(left_path, thumb_size), (margin, image_y))
    canvas.paste(thumbnail(right_path, thumb_size), (margin + thumb_size + gap, image_y))
    draw.text((margin + 8, image_y + 8), "LEFT", fill="yellow", font=body_font)
    draw.text(
        (margin + thumb_size + gap + 8, image_y + 8),
        "RIGHT",
        fill="yellow",
        font=body_font,
    )

    text_y = image_y + thumb_size + 18
    lines = [
        f"Visual: {row['visual_verdict']} (confidence={row['visual_confidence']})",
        f"Frozen rule: {row['rule_decision']} via {row['rule_decision_path']}",
        (
            f"pHash={row['phash_distance']} | robust rank={row['robust_best_union_rank']} "
            f"| robust cosine={_fmt(row['robust_best_cosine_similarity'])}"
        ),
        (
            f"ORB inliers={row['orb_inliers']} | ratio={_fmt(row['orb_inlier_ratio'])} "
            f"| coverage={_fmt(row['orb_coverage_min'])} | plausible={row['orb_affine_plausible']}"
        ),
        (
            f"Best alignment={row['best_alignment_method']}/{row['best_alignment_orientation']} "
            f"| SSIM={_fmt(row['best_ssim'])} | NMI={_fmt(row['best_nmi'])} "
            f"| corr={_fmt(row['best_correlation'])} | overlap={_fmt(row['best_overlap'])}"
        ),
        "Review target: same transformed 2D source image, distinct image/slice, or uncertain?",
        "IMPORTANT: Adjacent CT slices remain independent 2D images unless a transformed copy is demonstrated.",
    ]
    for line in lines:
        for wrapped in textwrap.wrap(line, width=112) or [""]:
            draw.text((margin, text_y), wrapped, fill="black", font=small_font)
            text_y += 23
        text_y += 3
    return canvas


def write_csv(
    path: Path, rows: Sequence[Mapping[str, str]], fields: Sequence[str] = QUEUE_FIELDS
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, Mapping[str, object]]:
    unblinded_path = Path(args.unblinded).resolve()
    scores_path = Path(args.scores).resolve()
    queue_dir = Path(args.queue_dir).resolve()
    render_dir = Path(args.render_dir).resolve()
    if args.thumb_size < 256:
        raise ValueError("--thumb-size en az 256 olmalıdır.")
    if queue_dir.exists() or render_dir.exists():
        raise FileExistsError(
            "Adjudication çıktısı zaten var; üzerine yazılmadı: "
            f"queue={queue_dir}, render={render_dir}"
        )
    if file_sha256(unblinded_path) != EXPECTED_UNBLINDED_SHA256:
        raise ValueError("Unblinded audit tablosu beklenen frozen hash ile eşleşmiyor.")
    if file_sha256(scores_path) != EXPECTED_SCORES_SHA256:
        raise ValueError("Frozen-rule skor tablosu beklenen hash ile eşleşmiyor.")

    _, unblinded_rows = read_csv(unblinded_path)
    _, score_rows = read_csv(scores_path)
    if len(unblinded_rows) != EXPECTED_INPUT_ROWS or len(score_rows) != EXPECTED_INPUT_ROWS:
        raise ValueError("Natural audit girdileri 345 satır içermiyor.")
    queue_rows = select_queue_rows(unblinded_rows, score_rows)
    if len(queue_rows) != EXPECTED_QUEUE_ROWS:
        raise ValueError(
            f"Beklenen {EXPECTED_QUEUE_ROWS} yerine {len(queue_rows)} adjudication çifti var."
        )
    if any(row[field] for row in queue_rows for field in ADJUDICATION_FIELDS):
        raise ValueError("Adjudication karar alanları hazırlık aşamasında boş olmalıdır.")

    dataset_root = resolve_dataset_root(args.data_dir)
    queue_dir.parent.mkdir(parents=True, exist_ok=True)
    render_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="natural-adjudication-queue-", dir=queue_dir.parent
    ) as queue_tmp, tempfile.TemporaryDirectory(
        prefix="natural-adjudication-render-", dir=render_dir.parent
    ) as render_tmp:
        queue_staging = Path(queue_tmp) / queue_dir.name
        render_staging = Path(render_tmp) / render_dir.name
        pair_views = render_staging / "pair_views"
        queue_staging.mkdir()
        pair_views.mkdir(parents=True)

        queue_path = queue_staging / "adjudication_queue.csv"
        write_csv(queue_path, queue_rows)
        for index, row in enumerate(queue_rows, start=1):
            card = render_pair_card(row, dataset_root, args.thumb_size)
            card.save(pair_views / f"{row['pair_id']}.png", format="PNG", optimize=True)
            print(f"Adjudication kartı: {index}/{len(queue_rows)} {row['pair_id']}")

        rendered_ids = {path.stem for path in pair_views.glob("N*.png")}
        expected_ids = {row["pair_id"] for row in queue_rows}
        if rendered_ids != expected_ids:
            raise ValueError("Pair-ID görselleri ile kuyruk CSV eşleşmiyor.")

        category_counts = dict(
            sorted(Counter(row["queue_category"] for row in queue_rows).items())
        )
        modality_counts = dict(
            sorted(Counter(row["modality_key"] for row in queue_rows).items())
        )
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "adjudication_queue_prepared_decisions_blank",
            "experiment_type": "near_duplicate_second_stage_review_preparation",
            "counts": {
                "input_pairs": len(score_rows),
                "safe_consensus_pairs_excluded": len(score_rows) - len(queue_rows),
                "adjudication_pairs": len(queue_rows),
                "rendered_pair_views": len(rendered_ids),
                "blank_adjudication_rows": sum(
                    not row["adjudication_verdict"] for row in queue_rows
                ),
                "by_category": category_counts,
                "by_modality": modality_counts,
            },
            "integrity": {
                "unblinded_input_sha256": file_sha256(unblinded_path),
                "frozen_scores_input_sha256": file_sha256(scores_path),
                "adjudication_queue_sha256": file_sha256(queue_path),
                "preparer_sha256": file_sha256(Path(__file__).resolve()),
            },
            "decision_codes": {
                "same_source": "Aynı 2B kaynak görüntünün dönüştürülmüş/augment edilmiş hali.",
                "independent": "Farklı 2B görüntü veya farklı/komşu kesit.",
                "uncertain": "Kanıt güvenli karar için yetersiz.",
            },
            "guardrails": [
                "Dondurulmuş score-blind visual review değiştirilmedi.",
                "Frozen duplicate rule v1 değiştirilmedi.",
                "Hazırlık aşamasında 25 nihai karar alanının tamamı boş bırakıldı.",
                "Aynı hastaya ait olabilecek komşu CT kesitleri otomatik olarak aynı 2B görüntü ailesine bağlanmaz.",
                "Bu paket duplicate family, classifier split veya model üretmez.",
            ],
        }
        report_path = queue_staging / "package_report.json"
        atomic_json(report_path, report)

        shutil.move(str(render_staging), str(render_dir))
        shutil.move(str(queue_staging), str(queue_dir))

    print("İşlem türü: NATURAL DUPLICATE ADJUDICATION v1 PAKET HAZIRLIĞI")
    print(f"Girdi: {len(score_rows)} çift")
    print(f"Güvenli konsensüs dışarıda bırakıldı: {len(score_rows) - len(queue_rows)}")
    print(f"Adjudication kuyruğu: {len(queue_rows)} çift")
    print(f"Karar alanları boş: {len(queue_rows)}/{len(queue_rows)}")
    print(f"Kuyruk: {queue_dir / 'adjudication_queue.csv'}")
    print(f"Pair-ID görselleri: {render_dir / 'pair_views'}")
    print("NOT: Family, split veya model oluşturulmadı.")
    return (
        queue_dir / "adjudication_queue.csv",
        queue_dir / "package_report.json",
        report,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
