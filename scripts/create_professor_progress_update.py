from __future__ import annotations

"""Create the professor-facing NNGA progress PDF and representative PNG.

This script only formats already verified project evidence. It does not train a
model, inspect holdout images, or alter the dataset.
"""

from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "archive" / "Multi Cancer" / "Multi Cancer"
OUTPUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUTPUT_DIR / "NNGA_Progress_Update_2026-10-22.pdf"
PNG_PATH = OUTPUT_DIR / "NNGA_Duplicate_Examples_2026-10-22.png"

NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#1D7874")
ORANGE = colors.HexColor("#F4A261")
LIGHT = colors.HexColor("#F4F7FA")
MID = colors.HexColor("#D9E4EC")
TEXT = colors.HexColor("#22313F")
MUTED = colors.HexColor("#5B6B7A")
GREEN = colors.HexColor("#2A9D8F")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def fit_image(path: Path, size: tuple[int, int]) -> PILImage.Image:
    with PILImage.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.fit(image, size, method=PILImage.Resampling.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  selected_font: ImageFont.FreeTypeFont, fill: str) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=selected_font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=selected_font,
        fill=fill,
    )


def create_visual_example() -> None:
    canvas = PILImage.new("RGB", (1600, 900), "#F4F7FA")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((24, 20, 1576, 92), radius=18, fill="#16324F")
    draw.text(
        (52, 35),
        "Representative near-duplicate evidence",
        font=font(34, bold=True),
        fill="white",
    )
    draw.text(
        (1040, 43),
        "Development data only - holdout remains sealed",
        font=font(19),
        fill="#D9E4EC",
    )

    examples = [
        {
            "panel": (40, 120, 780, 730),
            "color": "#1D7874",
            "title": "C000613 - same-source candidate",
            "subtitle": "Colon histology: repeated local tissue landmarks align",
            "left": DATASET_ROOT
            / "Lung and Colon Cancer"
            / "colon_aca"
            / "colon_aca_0700.jpg",
            "right": DATASET_ROOT
            / "Lung and Colon Cancer"
            / "colon_aca"
            / "colon_aca_4354.jpg",
            "metrics": "pHash distance 10 | ORB/RANSAC inliers 2,270 | aligned SSIM 0.777",
        },
        {
            "panel": (820, 120, 1560, 730),
            "color": "#F4A261",
            "title": "C000620 - uncertain",
            "subtitle": "Kidney CT: similar level, but same slice vs adjacent slice is unclear",
            "left": DATASET_ROOT
            / "Kidney Cancer"
            / "kidney_tumor"
            / "kidney_tumor_3465.jpg",
            "right": DATASET_ROOT
            / "Kidney Cancer"
            / "kidney_tumor"
            / "kidney_tumor_4836.jpg",
            "metrics": "pHash distance 8 | ORB/RANSAC inliers 44 | aligned SSIM 0.754",
        },
    ]

    for example in examples:
        x1, y1, x2, y2 = example["panel"]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill="white", outline="#D9E4EC", width=3)
        draw.rounded_rectangle((x1 + 18, y1 + 18, x2 - 18, y1 + 73), radius=13, fill=example["color"])
        draw.text((x1 + 38, y1 + 31), example["title"], font=font(25, bold=True), fill="white")
        draw.text((x1 + 28, y1 + 91), example["subtitle"], font=font(17), fill="#354A5F")

        image_size = (320, 320)
        left_image = fit_image(example["left"], image_size)
        right_image = fit_image(example["right"], image_size)
        left_position = (x1 + 32, y1 + 140)
        right_position = (x1 + 388, y1 + 140)
        canvas.paste(left_image, left_position)
        canvas.paste(right_image, right_position)
        draw.rectangle(
            (*left_position, left_position[0] + image_size[0], left_position[1] + image_size[1]),
            outline="#9FB3C8",
            width=2,
        )
        draw.rectangle(
            (*right_position, right_position[0] + image_size[0], right_position[1] + image_size[1]),
            outline="#9FB3C8",
            width=2,
        )
        draw_centered(
            draw,
            (left_position[0], y1 + 467, left_position[0] + 320, y1 + 500),
            "Image A",
            font(17, bold=True),
            "#16324F",
        )
        draw_centered(
            draw,
            (right_position[0], y1 + 467, right_position[0] + 320, y1 + 500),
            "Image B",
            font(17, bold=True),
            "#16324F",
        )
        draw.rounded_rectangle((x1 + 28, y1 + 520, x2 - 28, y1 + 582), radius=12, fill="#EEF3F7")
        draw_centered(
            draw,
            (x1 + 36, y1 + 525, x2 - 36, y1 + 577),
            example["metrics"],
            font(17, bold=True),
            "#22313F",
        )

    draw.rounded_rectangle((40, 760, 1560, 865), radius=18, fill="#16324F")
    steps = [
        "pHash candidate",
        "ORB + RANSAC",
        "ECC / phase",
        "SSIM / NMI",
        "family-aware split",
    ]
    x_positions = [80, 370, 660, 950, 1240]
    for index, (label, x) in enumerate(zip(steps, x_positions)):
        draw.rounded_rectangle((x, 785, x + 235, 840), radius=14, fill="#1D7874")
        draw_centered(draw, (x, 785, x + 235, 840), label, font(18, bold=True), "white")
        if index < len(steps) - 1:
            draw.polygon(
                [(x + 250, 801), (x + 278, 812), (x + 250, 823)],
                fill="#F4A261",
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(PNG_PATH, format="PNG", optimize=True)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.6,
            textColor=TEXT,
            spaceAfter=1.6 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
        ),
        "card_number": ParagraphStyle(
            "CardNumber",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=21,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "timeline": ParagraphStyle(
            "Timeline",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
    }


def bullet(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"<font color='#1D7874'><b>-</b></font> {text}", st["body"])


def metric_card(number: str, label: str, st: dict[str, ParagraphStyle]) -> Table:
    card = Table(
        [[Paragraph(number, st["card_number"])], [Paragraph(label, st["card_label"])]],
        colWidths=[31 * mm],
        rowHeights=[9 * mm, 9 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return card


def draw_page(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 12 * mm, width, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(16 * mm, height - 7.5 * mm, "NNGA Multi-Cancer Classification Project")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 16 * mm, height - 7.5 * mm, "Progress update - 22 October 2026")
    canvas.setStrokeColor(MID)
    canvas.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(16 * mm, 8 * mm, "Prepared by Mehmet - POLITEHNICA Bucharest Erasmus")
    canvas.drawRightString(width - 16 * mm, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def create_pdf() -> None:
    st = styles()
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="NNGA Multi-Cancer Classification Project - Progress Update",
        author="Mehmet",
        subject="Dataset audit, leakage control and current project status",
    )
    story = []
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Multi-Cancer Image Classification", st["title"]))
    story.append(
        Paragraph(
            "Neural Networks and Genetic Algorithms - three-week progress update",
            st["subtitle"],
        )
    )

    stage = Table(
        [[Paragraph("CURRENT STAGE", st["timeline"]), Paragraph(
            "Data audit and leakage-aware split design before final model benchmarking",
            ParagraphStyle("StageText", parent=st["body"], fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=0),
        )]],
        colWidths=[35 * mm, 128 * mm],
    )
    stage.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), TEAL),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8F2F1")),
                ("BOX", (0, 0), (-1, -1), 0.8, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([stage, Spacer(1, 4 * mm)])

    cards = Table(
        [[
            metric_card("130,000", "verified images", st),
            metric_card("8", "top-level groups", st),
            metric_card("26", "leaf classes", st),
            metric_card("5", "imaging modalities", st),
        ]],
        colWidths=[41 * mm] * 4,
        hAlign="LEFT",
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story.extend([cards, Spacer(1, 4 * mm)])

    story.append(Paragraph("What has been completed", st["section"]))
    left = [
        Paragraph("<b>Dataset and label audit</b>", st["body"]),
        bullet("All 130,000 accepted images were decoded; no corrupt image was found.", st),
        bullet("The dataset contains 8 parent folders, 26 leaf labels and 5 acquisition modalities.", st),
        bullet("File format, size, channel and label-taxonomy checks were completed.", st),
        bullet("1,318 exact-duplicate groups were identified, containing 2,673 files.", st),
    ]
    right = [
        Paragraph("<b>Leakage-control pipeline</b>", st["body"]),
        bullet("A 1,342-pair calibration set covers all 26 leaf classes.", st),
        bullet("671 development pairs were scored; 671 holdout pairs remain sealed.", st),
        bullet("Implemented pHash, ORB/RANSAC, phase correlation, ECC, SSIM and NMI evidence.", st),
        bullet("23 automated tests pass in the Python 3.10 environment.", st),
    ]
    completion = Table([[left, right]], colWidths=[81 * mm, 81 * mm])
    completion.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([completion, Spacer(1, 4 * mm)])

    story.append(Paragraph("Why final training has not been reported yet", st["section"]))
    note = Table(
        [[Paragraph(
            "The downloaded dataset was already augmented. A naive random image-level split can place transformed versions of the same source image in both training and test sets, producing unrealistically high accuracy. Therefore, the final model comparison is intentionally postponed until the duplicate-family method and leakage-aware split are validated.",
            st["body"],
        )]],
        colWidths=[163 * mm],
    )
    note.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF3E6")),
                ("BOX", (0, 0), (-1, -1), 1, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    story.extend([note, Spacer(1, 4 * mm)])

    story.append(Paragraph("Project path", st["section"]))
    steps = [
        ("1. Audit", GREEN),
        ("2. Calibrate", GREEN),
        ("3. Freeze rule", ORANGE),
        ("4. Holdout test", NAVY),
        ("5. Model benchmark", NAVY),
    ]
    timeline = Table(
        [[Paragraph(label, st["timeline"]) for label, _ in steps]],
        colWidths=[32.6 * mm] * 5,
        rowHeights=[13 * mm],
    )
    timeline.setStyle(
        TableStyle(
            [
                *[("BACKGROUND", (index, 0), (index, 0), color) for index, (_, color) in enumerate(steps)],
                ("BOX", (0, 0), (-1, -1), 0.8, colors.white),
                ("INNERGRID", (0, 0), (-1, -1), 1.2, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([timeline, Spacer(1, 3 * mm)])
    story.append(
        Paragraph(
            "Scope note: the required 8 top-level group classification will be reported. The 26 leaf labels are being studied as a structured extension and can be mapped back to the 8 parent groups. No clinical diagnosis or cancer-stage claim is made.",
            st["small"],
        )
    )

    story.append(PageBreak())
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Representative evidence and current findings", st["title"]))
    story.append(
        Paragraph(
            "Examples below show why one similarity score is not sufficient and why an explicit uncertain region is required.",
            st["subtitle"],
        )
    )
    visual = Image(str(PNG_PATH), width=163 * mm, height=91.7 * mm)
    story.extend([visual, Spacer(1, 4 * mm)])

    findings_left = [
        Paragraph("<b>Development findings</b>", st["body"]),
        bullet("15 visually accepted natural candidates: 449-2,360 ORB/RANSAC inliers.", st),
        bullet("123 independent pairs: 3-17 ORB/RANSAC inliers.", st),
        bullet("C000620 falls between both groups and remains uncertain.", st),
        bullet("270/520 controlled positives had pHash distance above 12.", st),
    ]
    findings_right = [
        Paragraph("<b>Next controlled steps</b>", st["body"]),
        bullet("Freeze a conservative multi-evidence duplicate rule using development only.", st),
        bullet("Evaluate the frozen rule once on the sealed holdout set.", st),
        bullet("Build duplicate families and a leakage-aware 70/15/15 manifest.", st),
        bullet("Compare SmallCNN, ResNet18, DenseNet121 and EfficientNet-B0 on the same split.", st),
    ]
    findings = Table([[findings_left, findings_right]], colWidths=[81 * mm, 81 * mm])
    findings.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([findings, Spacer(1, 4 * mm)])

    warning = Table(
        [[Paragraph(
            "Interpretation limit: development separation is encouraging but is not holdout performance and is not classifier accuracy. The final claim will be made only after the frozen method is evaluated on unseen holdout pairs and the model is tested on a leakage-aware split.",
            st["body"],
        )]],
        colWidths=[163 * mm],
    )
    warning.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF3F8")),
                ("BOX", (0, 0), (-1, -1), 1, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    story.extend([warning, Spacer(1, 3 * mm)])
    story.append(
        Paragraph(
            "Code and documentation: <link href='https://github.com/YusufOck/Neural-Network-and-Genens-s-Algor-thm-Project-Erasmus-' color='#1D7874'>GitHub repository</link>. Dataset source: Obuli Sai Naren, Multi Cancer Dataset (Kaggle).",
            st["small"],
        )
    )

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


def main() -> None:
    for required in (
        DATASET_ROOT,
        DATASET_ROOT / "Lung and Colon Cancer" / "colon_aca" / "colon_aca_0700.jpg",
        DATASET_ROOT / "Kidney Cancer" / "kidney_tumor" / "kidney_tumor_3465.jpg",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    create_visual_example()
    create_pdf()
    print(f"PDF: {PDF_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
