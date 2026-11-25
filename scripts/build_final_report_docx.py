from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "research" / "final_report" / "figures"
OUTPUT = ROOT / "deliverables" / "NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx"
TAXONOMY = json.loads((ROOT / "config" / "label_taxonomy.json").read_text(encoding="utf-8"))

NAVY = "17324D"
BLUE = "2F6B95"
TEAL = "397E8C"
GREEN = "4F8C68"
RED = "B74A4A"
ORANGE = "DB8425"
LIGHT_BLUE = "E7F0F7"
LIGHT_TEAL = "E9F3F1"
LIGHT_GRAY = "F2F4F6"
MID_GRAY = "667085"
INK = "202833"
WHITE = "FFFFFF"


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text: str, *, bold=False, color=INK, size=8.5, align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def add_table(doc: Document, headers: list[str], rows: list[list[object]], widths=None, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, text in enumerate(headers):
        set_cell_text(header.cells[idx], text, bold=True, color=WHITE, size=font_size)
        shade_cell(header.cells[idx], NAVY)
        if widths:
            header.cells[idx].width = widths[idx]
    for row_index, row_data in enumerate(rows):
        row = table.add_row()
        for idx, value in enumerate(row_data):
            set_cell_text(row.cells[idx], str(value), size=font_size)
            shade_cell(row.cells[idx], WHITE if row_index % 2 == 0 else LIGHT_GRAY)
            if widths:
                row.cells[idx].width = widths[idx]
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def set_column_widths(section, widths: list[float]) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    if not cols:
        cols_node = OxmlElement("w:cols")
        sect_pr.append(cols_node)
    else:
        cols_node = cols[0]
    cols_node.set(qn("w:num"), str(len(widths)))
    cols_node.set(qn("w:space"), "360")


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, value, end])


def add_toc_field(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Right-click and select Update Field if page numbers are not refreshed."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def add_hyperlink(paragraph, text: str, url: str):
    part = paragraph.part
    relationship_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.extend([color, underline])
    run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_body(doc: Document, text: str, *, bold_prefix: str | None = None, italic=False, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        first = p.add_run(bold_prefix)
        first.bold = True
        p.add_run(text[len(bold_prefix):])
    else:
        run = p.add_run(text)
        run.italic = italic
    return p


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_numbered(doc: Document, items: list[str]) -> None:
    for index, item in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.55)
        p.paragraph_format.space_after = Pt(3)
        p.add_run(f"{index}. ").bold = True
        p.add_run(item)


def add_callout(doc: Document, title: str, body: str, color=LIGHT_BLUE, accent=BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, color)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(accent)
    p2 = cell.add_paragraph(body)
    p2.paragraph_format.space_after = Pt(0)
    set_cell_margins(cell, top=130, start=150, bottom=130, end=150)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_figure(doc: Document, filename: str, caption: str, width=6.35, page_break=False) -> None:
    if page_break:
        doc.add_page_break()
    path = FIG / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    picture = p.add_run().add_picture(str(path), width=Inches(width))
    # Store a meaningful Word alternative description for accessibility.
    # The figure caption is concise, unique, and already explains the visual.
    picture._inline.docPr.set("title", path.stem.replace("_", " "))
    picture._inline.docPr.set("descr", caption)
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    cap.add_run(caption)


def add_equation(doc: Document, expression: str, explanation: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(expression)
    run.font.name = "Cambria Math"
    run.font.size = Pt(10.5)
    run.italic = True
    p2 = doc.add_paragraph(explanation)
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(8)
    for run in p2.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor.from_string(MID_GRAY)


def add_reference(doc: Document, number: int, text: str, url: str | None = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.7)
    p.paragraph_format.space_after = Pt(4)
    p.add_run(f"[{number}] ").bold = True
    p.add_run(text)
    if url:
        p.add_run(" ")
        add_hyperlink(p, url, url)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.2)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(6)

    for name, size, color, before, after in (
        ("Title", 25, NAVY, 0, 12),
        ("Heading 1", 16, NAVY, 18, 8),
        ("Heading 2", 13, BLUE, 14, 6),
        ("Heading 3", 11, TEAL, 10, 4),
    ):
        style = styles[name]
        style.font.name = "Aptos Display" if name != "Heading 3" else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.left_indent = Cm(0)
        style.paragraph_format.first_line_indent = Cm(0)
    styles["Heading 1"].paragraph_format.page_break_before = True

    caption = styles["Caption"]
    caption.font.name = "Aptos"
    caption.font.size = Pt(8.5)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor.from_string(MID_GRAY)

    for list_name in ("List Bullet", "List Number"):
        style = styles[list_name]
        style.font.name = "Aptos"
        style.font.size = Pt(10)
        style.paragraph_format.space_after = Pt(3)

    if "Executive Summary" not in styles:
        style = styles.add_style("Executive Summary", WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "Aptos"
        style.font.size = Pt(10.5)
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = 1.12


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)
    header = section.header.paragraphs[0]
    header.text = "NNGA Multi Cancer Image Classification | Final Project Report"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(7.5)
        run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    add_page_number(section.footer.paragraphs[0])

    settings = doc.settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)


def add_title_page(doc: Document) -> None:
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("HIERARCHICAL MULTI-CANCER IMAGE CLASSIFICATION")
    r.font.name = "Aptos Display"
    r.font.size = Pt(25)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(NAVY)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("Duplicate-Aware Evaluation, Transfer Learning, Explainability,\nand Genetic-Algorithm Hyperparameter Search")
    r2.font.name = "Aptos Display"
    r2.font.size = Pt(16)
    r2.font.color.rgb = RGBColor.from_string(BLUE)
    p2.paragraph_format.space_after = Pt(28)

    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = line.add_run("—")
    rr.font.size = Pt(30)
    rr.font.color.rgb = RGBColor.from_string(TEAL)

    details = [
        ("Student", "MEHMET YUSUF OCAK"),
        ("Mobility", "ERASMUS"),
        ("Course", "Neural Networks and Genetic Algorithms"),
        ("Institution", "National University of Science and Technology POLITEHNICA Bucharest"),
        ("Report date", "November 2026"),
    ]
    table = doc.add_table(rows=len(details), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row, (key, value) in zip(table.rows, details):
        set_cell_text(row.cells[0], key, bold=True, color=NAVY, size=10.5, align=WD_ALIGN_PARAGRAPH.RIGHT)
        set_cell_text(row.cells[1], value, color=INK, size=10.5)
        row.cells[0].width = Cm(4.0)
        row.cells[1].width = Cm(10.0)
    doc.add_paragraph()
    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_before = Pt(16)
    run = p3.add_run("FINAL PROJECT REPORT")
    run.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor.from_string(GREEN)
    doc.add_page_break()


def build_report() -> Document:
    doc = Document()
    configure_styles(doc)
    configure_document(doc)
    doc.core_properties.title = "Hierarchical Multi-Cancer Image Classification"
    doc.core_properties.subject = "NNGA final project report"
    doc.core_properties.author = "MEHMET YUSUF OCAK"
    doc.core_properties.keywords = "ERASMUS, neural networks, genetic algorithms, multi-cancer, ResNet18"
    doc.core_properties.comments = "Generated from hash-verified project artifacts."

    add_title_page(doc)

    doc.add_heading("Abstract", level=1)
    add_body(doc, (
        "This project developed and audited a hierarchical image-classification pipeline for the Multi Cancer Dataset, "
        "a compilation of 130,000 images organized into eight parent groups and 26 leaf classes. The principal prediction "
        "task was the 26-class leaf problem; each leaf prediction was also mapped deterministically to its parent group so "
        "that organ or disease-family performance could be measured without replacing the clinically more informative leaf task. "
        "Because the released collection contains augmented data and repeated imagery, the project first performed a dedicated "
        "duplicate and leakage audit. SHA-256 exact matching identified 1,318 exact-duplicate groups containing 2,673 files. "
        "Perceptual and geometric analyses combined pHash, ORB, RANSAC, phase correlation, ECC alignment, SSIM, normalized mutual "
        "information and correlation. Known duplicate and reviewed same-source relations were converted into connected components "
        "before a fixed 70/15/15 split was assigned. The resulting split contained 91,000 training, 19,500 validation and 19,500 "
        "test images, with no known component crossing split boundaries. Three controlled baselines were then compared on the same "
        "test set: a frozen-head ImageNet-pretrained ResNet18, a 97,050-parameter SmallCNN trained from scratch, and a last-block "
        "ResNet18 fine-tuned in layer4 and the classifier. The last-block model achieved 99.6462% leaf accuracy and 99.6462% "
        "macro-F1, with 69 errors among 19,500 test images; 67 of those errors remained within the correct parent group. Post-hoc "
        "analysis included confusion matrices, calibration measures, representative predictions and Grad-CAM. Finally, a genetic "
        "algorithm optimized learning rate, weight decay and dropout and was compared with an equal-budget random search. The GA "
        "candidate obtained 99.5795% shared-test accuracy, while the random-search candidate obtained 99.5744%; exact McNemar "
        "testing found no statistically detectable paired difference (p = 1.0). The resulting work is a reproducible experimental "
        "pipeline rather than a clinical diagnostic system."
    ))
    p = doc.add_paragraph()
    p.add_run("Keywords: ").bold = True
    p.add_run("multi-cancer classification; transfer learning; duplicate leakage; ResNet18; Grad-CAM; genetic algorithm; macro-F1")

    doc.add_heading("Executive Summary", level=1)
    p = doc.add_paragraph(style="Executive Summary")
    p.add_run(
        "The final system classifies one of 26 dataset-defined leaf categories and derives one of eight parent groups from that leaf. "
        "Its strongest controlled baseline is the last-block fine-tuned ResNet18. The data split was frozen before reportable model "
        "comparison and was shared by every reported test result. The genetic-algorithm experiment was conducted after the baseline "
        "model study and did not use test images during search or validation-based candidate selection."
    )
    add_table(doc,
              ["Area", "Principal result", "Interpretation"],
              [
                  ["Dataset", "130,000 images; 8 parents; 26 leaves", "Balanced at 5,000 images per leaf; parent counts differ."],
                  ["Exact duplicates", "1,318 groups; 2,673 files", "Exact copies were grouped before splitting."],
                  ["Practical split", "91,000 / 19,500 / 19,500", "Fixed train/validation/test sets with known-family containment."],
                  ["Best baseline", "99.6462% leaf accuracy and macro-F1", "Last-block ResNet18, best epoch 11."],
                  ["Hierarchy", "99.9897% parent accuracy", "Most leaf errors remained inside the correct parent group."],
                  ["Explainability", "37/37 frozen predictions reproduced", "Grad-CAM supplied coarse influence maps for audit."],
                  ["GA vs random", "99.5795% vs 99.5744% test accuracy", "Negligible difference; McNemar p = 1.0."],
              ],
              widths=[Cm(3.0), Cm(4.3), Cm(9.0)], font_size=8.6)
    add_figure(doc, "01_project_workflow.png",
               "Figure 1. End-to-end experimental workflow. Each reportable result is linked to a frozen split and hash-verified artifact.",
               width=6.45)

    doc.add_heading("Table of Contents", level=1)
    add_toc_field(doc.add_paragraph())
    add_body(doc, "The table of contents is stored as an updateable Word field. Page numbers refresh automatically when the document is opened in Microsoft Word.", italic=True)

    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Motivation", level=2)
    add_body(doc, (
        "Medical-image repositories frequently combine data from different organs, acquisition devices and preparation protocols. "
        "A single classifier trained across such sources must therefore solve more than one visual problem: it must separate imaging "
        "modalities and anatomical groups while also distinguishing visually similar categories inside each group. The Multi Cancer "
        "Dataset is an unusually heterogeneous example, combining blood-smear microscopy, contrast-enhanced brain MRI, abdominal CT, "
        "Pap-smear microscopy and several histopathology collections [1]."
    ))
    add_body(doc, (
        "A second challenge is experimental validity. The dataset documentation states that augmentation was used to expand several "
        "source collections. If an original image and a transformed relative are assigned to different splits, a model may encounter "
        "nearly the same visual content during both training and testing. This produces data leakage: the reported score can reflect "
        "memorization of a source image family rather than generalization to an unseen family. Consequently, this project treated "
        "data auditing and split construction as first-class research tasks rather than as incidental preprocessing."
    ))

    doc.add_heading("1.2 Objectives", level=2)
    add_numbered(doc, [
        "Determine the exact semantic meaning of all 26 leaf labels and their relationship to the eight parent groups.",
        "Audit exact and perceptual duplicates and prevent known image families from crossing train, validation and test boundaries.",
        "Build a reproducible training pipeline that evaluates leaf and parent performance using the same frozen split.",
        "Compare a frozen pretrained model, a small model trained from scratch and a partially fine-tuned transfer-learning model.",
        "Inspect model errors, calibration and Grad-CAM influence maps using frozen post-hoc audit sets.",
        "Use a genetic algorithm for hyperparameter search and compare it fairly with an equal-budget random search.",
    ])

    doc.add_heading("1.3 Contributions", level=2)
    add_bullets(doc, [
        "A verified 26-to-8 label taxonomy that avoids unsupported medical interpretations.",
        "A duplicate-audit pipeline combining cryptographic, perceptual, local-feature, registration and aligned-similarity evidence.",
        "A practical leakage-reduced split with zero crossings among all known exact and reviewed same-source components.",
        "A controlled model comparison on 19,500 shared test images, including per-class and hierarchical metrics.",
        "A reproducible Grad-CAM audit and a paired GA-versus-random experiment with an exact McNemar test.",
    ])
    add_callout(doc, "Scope of the classification claim", (
        "The model predicts only labels represented in the dataset. It does not estimate cancer stage. It also cannot provide a universal "
        "healthy-versus-cancer decision for parent groups that contain no healthy control class. For example, the brain group contains "
        "glioma, meningioma and pituitary tumor images but no normal-brain category."
    ), LIGHT_TEAL, TEAL)

    doc.add_heading("2. Dataset Audit and Label Semantics", level=1)
    doc.add_heading("2.1 Dataset composition", level=2)
    add_body(doc, (
        "The project used the locally downloaded Multi Cancer Dataset exactly as provided for this coursework [1]. A file-system audit "
        "found 130,000 accepted JPEG images. Every leaf class contained 5,000 images, producing balanced leaf support. Parent groups "
        "were not equally sized because they contain different numbers of leaves. The parent label therefore identifies a broad source "
        "or organ family, whereas the leaf label carries the dataset's detailed target."
    ))
    add_figure(doc, "02_dataset_modalities.png",
               "Figure 2. Representative images from the eight parent groups. The collection spans five broad imaging modalities.",
               width=6.45)
    add_figure(doc, "03_parent_distribution.png",
               "Figure 3. Parent-group image counts. Leaf support is exactly balanced at 5,000 images per class.",
               width=6.35)

    doc.add_heading("2.2 Eight parent groups and 26 leaf classes", level=2)
    parent_rows = [
        ["ALL", "4", "20,000", "Benign hematogone; Early Pre-B, Pre-B and Pro-B ALL"],
        ["Brain Cancer", "3", "15,000", "Glioma; meningioma; pituitary tumor"],
        ["Breast Cancer", "2", "10,000", "Benign; malignant"],
        ["Cervical Cancer", "5", "25,000", "Dyskeratotic; koilocytotic; metaplastic; parabasal; superficial-intermediate"],
        ["Kidney Cancer", "2", "10,000", "Normal; tumor"],
        ["Lung and Colon Cancer", "5", "25,000", "Colon ACA/BNT; lung ACA/BNT/SCC"],
        ["Lymphoma", "3", "15,000", "CLL; follicular lymphoma; mantle-cell lymphoma"],
        ["Oral Cancer", "2", "10,000", "Normal; oral SCC"],
    ]
    add_table(doc, ["Parent group", "Leaves", "Images", "Leaf meaning"], parent_rows,
              widths=[Cm(3.3), Cm(1.4), Cm(2.2), Cm(9.3)], font_size=8.2)
    add_body(doc, (
        "The parent result is not produced by a second model. The network emits a probability distribution over 26 leaves. The "
        "highest-scoring leaf is selected, and a fixed taxonomy table maps that leaf to one parent. Thus a breast-benign image "
        "predicted as breast-malignant is a leaf error but remains a correct Breast Cancer parent prediction. Both levels are reported "
        "because the leaf score measures the detailed task and the parent score reveals whether an error also crosses organ/source groups."
    ))

    doc.add_heading("2.3 Semantic safeguards", level=2)
    add_bullets(doc, [
        "The ALL labels early, pre and pro are B-ALL immunophenotypic subtypes in the audited source context, not longitudinal cancer stages.",
        "The cervical labels describe cell morphologies; they are not cancer stages.",
        "The kidney tumor label does not supply a malignancy subtype or stage.",
        "The brain collection lacks a normal class, so it supports tumor-type classification among represented classes, not tumor detection against normal brain.",
        "The combined lung/colon parent contains two organs; the leaf prediction is required to resolve organ and tissue diagnosis.",
    ])

    doc.add_heading("2.4 Dataset provenance", level=2)
    add_body(doc, (
        "The Kaggle release is a compilation of eight upstream collections. Its README documents resizing to 512 × 512 pixels and "
        "augmentation with rotation, translation, shear, zoom, horizontal flipping and brightness variation. This provenance motivated "
        "the duplicate-family analysis. The complete source map is provided in Appendix B."
    ))

    doc.add_heading("3. Existing Work and Reproducibility Review", level=1)
    add_body(doc, (
        "Public notebooks associated with the dataset were reviewed before model development. Reported accuracies were treated as "
        "external claims unless code, data split and metrics could be reproduced locally. A GitHub implementation by RoupB was the "
        "closest structural reference because it addressed the same 26-class dataset. Its code informed the audit of common practices, "
        "but its reported scores were not imported as project results. Differences in splitting, preprocessing and evaluation prevent "
        "a direct scientific comparison unless the full protocol is shared."
    ))
    add_table(doc,
              ["Evidence label", "Meaning in this report"],
              [
                  ["Reported", "A value stated by another author; not independently confirmed here."],
                  ["Reproduced", "A result rerun locally from available code under a documented configuration."],
                  ["Project result", "A hash-linked artifact produced by this repository on the frozen split."],
                  ["Diagnostic", "An engineering or audit outcome that is not a classifier-performance claim."],
              ],
              widths=[Cm(3.2), Cm(12.8)], font_size=8.7)
    add_body(doc, (
        "The review also established an important reporting rule: accuracy alone is insufficient. A high accuracy may conceal weak "
        "minority-class behavior, class imbalance or easy source-group separation. This project therefore reports macro-F1, balanced "
        "accuracy, per-class precision/recall/F1 and confusion matrices in addition to accuracy."
    ))

    doc.add_heading("4. Duplicate and Leakage Audit", level=1)
    doc.add_heading("4.1 Why duplicate control precedes model training", level=2)
    add_body(doc, (
        "If two files derived from one source image enter different splits, the test set is no longer independent of the training set. "
        "A cryptographic hash can find byte-identical copies, but it changes completely after rotation, brightness adjustment, cropping "
        "or recompression. The audit therefore used a layered strategy: exact hashing for certain duplicates, perceptual retrieval for "
        "candidate generation, geometric registration and aligned similarity for stronger evidence, and human visual review for a "
        "frozen sampled reference set."
    ))

    doc.add_heading("4.2 SHA-256 exact duplicates", level=2)
    add_body(doc, (
        "SHA-256 converts the bytes of each file into a 256-bit digest. Equal digests imply byte-identical content with negligible "
        "collision risk for this application. The scan identified 1,318 exact-duplicate groups containing 2,673 files, equivalent to "
        "1,355 redundant copies beyond one representative per group. No exact group crossed leaf labels. Exact groups were retained "
        "but were treated as indivisible units during splitting."
    ))

    doc.add_heading("4.3 Perceptual hashing", level=2)
    add_body(doc, (
        "Perceptual hashing (pHash) represents coarse image structure rather than file bytes [2]. Each RGB image was converted to "
        "grayscale and resized to 32 × 32 pixels. A two-dimensional discrete cosine transform (DCT) decomposed this small image into "
        "spatial-frequency coefficients. The low-frequency 8 × 8 block describes broad intensity layout, whereas high frequencies "
        "mainly describe fine edges and noise. After excluding the dominant DC term from threshold estimation, coefficients were "
        "compared with their median to form a 64-bit binary signature. The Hamming distance between two signatures counts the positions "
        "at which their bits differ. Small distance proposes a visually related candidate; it is not, by itself, a duplicate decision."
    ))
    add_figure(doc, "04_phash_pipeline.png",
               "Figure 4. Perceptual-hash construction and its role as a candidate generator.", width=6.45)

    doc.add_heading("4.4 Geometric and aligned-similarity evidence", level=2)
    method_rows = [
        ["ORB", "Detects repeatable local keypoints with FAST and encodes their neighborhoods using rotated BRIEF binary descriptors [3].", "Candidate-local correspondence"],
        ["RANSAC", "Fits a geometric transformation while rejecting mismatched keypoints as outliers [4].", "Number and ratio of geometrically consistent inliers"],
        ["Phase correlation", "Uses the normalized cross-power spectrum to estimate translational displacement.", "Initial x/y shift"],
        ["ECC", "Optimizes an image warp by maximizing the enhanced correlation coefficient [5].", "Subpixel registration under intensity change"],
        ["SSIM", "Compares luminance, contrast and structural pattern after alignment [6].", "Aligned structural similarity"],
        ["NMI", "Measures dependence between intensity distributions without requiring equal raw intensities.", "Aligned information overlap"],
        ["Correlation", "Measures linear co-variation of aligned pixel intensities.", "Aligned intensity agreement"],
    ]
    add_table(doc, ["Method", "How it works", "Role in the audit"], method_rows,
              widths=[Cm(2.2), Cm(9.2), Cm(5.0)], font_size=8.0)
    add_body(doc, (
        "These measures answer complementary questions. ORB/RANSAC asks whether distinctive local structures support one coherent "
        "geometric mapping. ECC and phase correlation attempt to place images in a common coordinate frame. SSIM, NMI and correlation "
        "then quantify residual agreement. Strong scores cannot prove biological identity; they provide technical evidence that two "
        "files may originate from the same source image."
    ))
    add_figure(doc, "05_duplicate_examples.png",
               "Figure 5. A byte-identical pair and a non-identical histology pair accepted as a same-source candidate after visual and geometric review.",
               width=6.0)

    doc.add_heading("4.5 Modality-aware retrieval", level=2)
    add_body(doc, (
        "The eight parent groups reduce to five broad modality families: blood-smear microscopy, contrast-enhanced MRI, H&E "
        "histopathology, Pap-smear microscopy and CT. A single pHash rule behaved differently across them. Smooth MRI and CT slices "
        "produced many low-distance lookalikes, while histopathology could retain genuine same-source relations at larger pHash "
        "distance because localized tissue texture dominated the image. Consequently, equal treatment meant applying the same audit "
        "logic and validation gates to every modality, not imposing one numerical threshold on all modalities."
    ))
    add_body(doc, (
        "The full v2.3 retrieval produced 4,621,932 unique candidate pairs, or 35.553 pairs per image, exceeding the frozen operational "
        "gate of 25. A score-blind v2.4 counterfactual disabled the problematic pHash channel for brain MRI and kidney CT and reduced "
        "the candidate load to 2,491,553 pairs, or 19.166 per image. However, it recovered 95.65% of the sampled natural same-source "
        "reference set. This percentage describes a small, AI-assisted reference sample and is not an estimate of exhaustive full-dataset recall."
    ))
    add_figure(doc, "06_candidate_load.png",
               "Figure 6. Candidate-pair load by modality group before and after the modality-aware counterfactual.", width=6.35)

    doc.add_heading("4.6 Artifact registry and audit trace", level=2)
    add_body(doc, (
        "Candidate generation was executed in resumable blocks. The registry contained 1,107 verified intermediate artifacts: one "
        "index artifact, 26 pHash-generation artifacts, 26 descriptor-generation artifacts, 32 robust-retrieval blocks and 1,022 "
        "pHash-retrieval blocks. An artifact is a stored, hash-addressed intermediate result such as an image index, descriptor matrix "
        "or candidate block. This design allowed interrupted Colab sessions to skip verified work and made later counterfactual analysis "
        "possible without recomputing all 130,000 images."
    ))

    doc.add_heading("5. Practical Leakage-Reduced Split", level=1)
    doc.add_heading("5.1 Family graph construction", level=2)
    add_body(doc, (
        "The project represented every image as a graph vertex. An undirected edge joined files known to be related: membership in the "
        "same SHA-256 group or acceptance as a reviewed same-source relation. Connected components were then calculated. Transitivity "
        "is essential: if A is linked to B and B is linked to C, A, B and C form one component even when A-C was never compared directly. "
        "Each component, rather than each image, became the atomic split unit."
    ))
    add_body(doc, (
        "The final practical graph contained 128,560 components. Of these, 1,402 contained more than one image, accounting for 2,842 "
        "images; the largest component contained four images. The graph incorporated all 1,318 exact-duplicate groups and 85 reviewed "
        "same-source edges."
    ))

    doc.add_heading("5.2 Assignment algorithm", level=2)
    add_body(doc, (
        "Components were assigned separately inside each leaf class using a fixed random seed. A greedy deficit-minimization procedure "
        "targeted 70% training, 15% validation and 15% test support while keeping every component intact. Because each leaf contains "
        "5,000 images, the final counts are exactly 3,500 training, 750 validation and 750 test images per leaf. This yields 91,000, "
        "19,500 and 19,500 images overall."
    ))
    add_figure(doc, "07_split_and_components.png",
               "Figure 7. Fixed split proportions and connected-component containment.", width=6.45)
    add_table(doc,
              ["Integrity check", "Result"],
              [
                  ["Manifest rows", "130,000"],
                  ["Per-leaf train / validation / test", "3,500 / 750 / 750"],
                  ["Cross-split SHA-256 groups", "0"],
                  ["Cross-split reviewed same-source links", "0"],
                  ["Cross-split known components", "0"],
                  ["Frozen manifest SHA-256", "999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171"],
              ],
              widths=[Cm(6.0), Cm(10.2)], font_size=8.4)
    add_callout(doc, "Interpretation of the split", (
        "This is the strongest practical split supported by the verified relations in the project. It prevents every known exact or "
        "reviewed same-source component from crossing splits. It does not claim that all possible augmented relatives in the 130,000-image "
        "collection have been discovered."
    ), LIGHT_TEAL, TEAL)

    doc.add_heading("6. Model Development and Evaluation Protocol", level=1)
    doc.add_heading("6.1 Shared training contract", level=2)
    add_body(doc, (
        "All reportable baselines used the same frozen split, the same 26-leaf output space and a test set that remained untouched during "
        "checkpoint selection. Images were resized/cropped to 224 × 224 pixels and normalized with ImageNet channel statistics for "
        "ResNet18 experiments. Training augmentation was intentionally modest: horizontal flip with probability 0.5 and rotation up "
        "to ±5 degrees. Validation and test transforms were deterministic. AdamW was the optimizer [11]. The best checkpoint was chosen "
        "by validation macro-F1, not test accuracy."
    ))

    doc.add_heading("6.2 Evaluation metrics", level=2)
    add_equation(doc, "Accuracy = correct predictions / all predictions",
                 "Accuracy measures overall correctness. With equal leaf support, it is directly interpretable but still does not describe individual classes.")
    add_equation(doc, "Precision_c = TP_c / (TP_c + FP_c)     Recall_c = TP_c / (TP_c + FN_c)",
                 "Precision penalizes false alarms for class c; recall penalizes missed members of class c.")
    add_equation(doc, "F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)",
                 "F1 is the harmonic mean of precision and recall.")
    add_equation(doc, "Macro-F1 = (1 / C) × Σ F1_c",
                 "Each of the C = 26 leaf classes contributes equally, regardless of frequency [12].")
    add_equation(doc, "Balanced accuracy = (1 / C) × Σ Recall_c",
                 "Balanced accuracy is the mean per-class recall. Because the test split has 750 images per leaf, it equals ordinary accuracy here.")
    add_body(doc, (
        "The report also includes per-class precision, recall and F1, 26 × 26 leaf and 8 × 8 parent confusion matrices, top-2 accuracy "
        "and calibration metrics. Every test prediction was stored in CSV form to allow paired comparisons and visual sampling."
    ))

    doc.add_heading("6.3 Frozen-head ResNet18", level=2)
    add_body(doc, (
        "ResNet18 is a residual convolutional network in which shortcut connections learn residual functions and improve optimization "
        "of deep models [7]. ImageNet-pretrained weights [8] supplied general edge, texture and shape features. In the frozen-head "
        "baseline, every convolutional feature layer was frozen and only a new dropout layer (p = 0.30) and 26-output fully connected "
        "head were trained. AdamW used learning rate 1 × 10⁻³ and weight decay 1 × 10⁻⁴. The maximum was 15 epochs with early-stopping "
        "patience 4; the selected checkpoint was epoch 9. This experiment tests how far generic pretrained features can go without "
        "adapting the feature extractor to the cancer dataset."
    ))

    doc.add_heading("6.4 SmallCNN trained from scratch", level=2)
    add_body(doc, (
        "A compact network with approximately 97,050 trainable parameters provided a non-pretrained comparison. Three convolutional "
        "blocks expanded channels from 3 to 32, 64 and 128. Batch normalization and ReLU followed each convolution; the first two "
        "blocks used max pooling. Adaptive average pooling reduced the final feature maps to one value per channel, followed by dropout "
        "p = 0.35 and a 128-to-26 linear classifier. All weights were learned from the project training split. AdamW used learning rate "
        "1 × 10⁻³ and weight decay 1 × 10⁻⁴; the selected checkpoint was epoch 25."
    ))
    add_figure(doc, "08_smallcnn_architecture.png",
               "Figure 8. SmallCNN architecture used as a from-scratch baseline.", width=6.45)

    doc.add_heading("6.5 Last-block ResNet18 fine-tuning", level=2)
    add_body(doc, (
        "The strongest baseline started from the selected frozen-head checkpoint. Early ResNet18 stages remained frozen to preserve "
        "general features, while layer4 and the final classifier were unfrozen. This made 8,407,066 of 11,189,850 parameters trainable. "
        "The optimizer and training history were restarted so this was a new controlled stage, not a continuation that inherited "
        "optimizer momentum. A lower learning rate of 1 × 10⁻⁴ limited destructive changes to pretrained features; weight decay was "
        "1 × 10⁻⁴ and dropout remained 0.30. The maximum was 12 epochs with patience 4, and epoch 11 was selected by validation macro-F1."
    ))

    doc.add_heading("6.6 Reproducibility controls", level=2)
    add_bullets(doc, [
        "The split manifest hash was checked before every reportable run.",
        "Configuration, console log, history, test metrics, predictions, confusion matrices and checkpoint hashes were retained.",
        "Interrupted SmallCNN training could resume from the last checkpoint without repeating completed epochs.",
        "Local development targeted Python 3.10; recorded Colab runs reported their actual runtime and CUDA versions.",
        "No test image was used to choose an epoch, a baseline or a GA/random hyperparameter candidate.",
    ])

    doc.add_heading("7. Baseline Results", level=1)
    doc.add_heading("7.1 Shared-test comparison", level=2)
    model_rows = [
        ["Frozen-head ResNet18", "9", "91.3026%", "91.2455%", "91.3026%", "99.7744%"],
        ["SmallCNN", "25", "95.5846%", "95.5916%", "95.5846%", "99.9231%"],
        ["Last-block ResNet18", "11", "99.6462%", "99.6462%", "99.6462%", "99.9897%"],
        ["GA-selected ResNet18", "10", "99.5795%", "99.5794%", "99.5795%", "99.9641%"],
        ["Random-selected ResNet18", "11", "99.5744%", "99.5745%", "99.5744%", "99.9590%"],
    ]
    add_table(doc, ["Model", "Best epoch", "Leaf accuracy", "Macro-F1", "Balanced accuracy", "Parent accuracy"],
              model_rows, widths=[Cm(4.1), Cm(1.7), Cm(2.5), Cm(2.2), Cm(2.5), Cm(2.5)], font_size=7.8)
    add_figure(doc, "09_model_comparison.png",
               "Figure 9. Accuracy and macro-F1 on the same 19,500-image test split. The truncated y-axis is used only to make differences visible.",
               width=6.35)
    add_body(doc, (
        "Frozen ImageNet features were already useful, but adapting the model's last residual stage increased leaf accuracy by 8.3436 "
        "percentage points relative to the frozen-head baseline. SmallCNN outperformed the frozen head despite having no pretraining, "
        "which shows that the dataset contains strong learnable visual regularities. Last-block fine-tuning was nevertheless superior "
        "to both. The near equality of accuracy, macro-F1 and balanced accuracy is expected because every leaf has exactly 750 test images."
    ))

    doc.add_heading("7.2 Learning dynamics of the best baseline", level=2)
    add_figure(doc, "10_best_training_curves.png",
               "Figure 10. Training and validation loss, accuracy and macro-F1 for last-block ResNet18.", width=6.45)
    add_body(doc, (
        "Validation performance rose quickly and then plateaued near 99.6%. The training and validation curves remained close, with no "
        "late divergence characteristic of severe overfitting. Epoch 11 provided the highest validation macro-F1 and was frozen before "
        "the test set was evaluated."
    ))

    doc.add_heading("7.3 Leaf and parent confusion", level=2)
    add_figure(doc, "11_best_leaf_confusion_matrix.png",
               "Figure 11. Leaf-level confusion matrix for last-block ResNet18. Each row contains 750 test images.", width=6.35, page_break=True)
    add_figure(doc, "12_best_parent_confusion_matrix.png",
               "Figure 12. Parent-level confusion matrix derived from the same 26-class predictions.", width=5.65, page_break=True)
    add_body(doc, (
        "The best model made 69 leaf errors among 19,500 test images. Sixty-seven errors stayed within the correct parent group and two "
        "crossed parent groups. Therefore the parent accuracy is much higher than leaf accuracy: the hierarchy gives partial credit only "
        "in the separate parent evaluation, while the detailed 26-class metric still counts every incorrect leaf as an error."
    ))
    add_figure(doc, "13_error_pair_distribution.png",
               "Figure 13. Most frequent error directions. Oral normal versus oral SCC was the dominant confusion pair.", width=6.35)

    doc.add_heading("8. Error Analysis, Confidence and Explainability", level=1)
    doc.add_heading("8.1 Error structure", level=2)
    add_body(doc, (
        "The error concentration is informative. Oral normal and oral SCC accounted for 34 bidirectional errors (18 normal-to-SCC and "
        "16 SCC-to-normal). Breast benign-to-malignant contributed seven errors, and several brain and lymphoma confusions occurred "
        "inside their parent groups. The two cross-parent errors were lung SCC predicted as breast malignant and breast benign predicted "
        "as oral normal. This pattern explains the 99.9897% parent accuracy without weakening the 99.6462% leaf result: both are computed "
        "from the same predictions at different semantic resolutions."
    ))
    add_figure(doc, "16_error_examples.png",
               "Figure 14. Six frozen error-audit examples. The set intentionally includes high-confidence and cross-parent errors.", width=5.9)
    add_figure(doc, "17_prediction_examples.png",
               "Figure 15. Representative correct and incorrect predictions from the best baseline.", width=6.35)

    doc.add_heading("8.2 Confidence and calibration", level=2)
    add_body(doc, (
        "A classifier's maximum softmax probability is a confidence score, not a guarantee of correctness. Among the 69 errors, 23 had "
        "confidence at least 0.90 and seven had confidence at least 0.99. These cases were specifically retained for visual review. "
        "Dataset-wide calibration was summarized with expected calibration error (ECE), negative log-likelihood (NLL) and Brier score."
    ))
    add_table(doc,
              ["Metric", "Value", "Meaning"],
              [
                  ["ECE", "0.001032", "Weighted average gap between binned confidence and observed accuracy; lower is better."],
                  ["NLL", "0.010893", "Mean negative log probability assigned to the correct class; heavily penalizes confident errors."],
                  ["Brier", "0.005630", "Mean squared difference between predicted probabilities and one-hot targets; lower is better."],
                  ["Top-2 accuracy", "99.9949%", "The true leaf appeared among the two highest scores for all but one test image."],
              ],
              widths=[Cm(2.7), Cm(2.5), Cm(11.2)], font_size=8.3)
    add_figure(doc, "14_reliability_diagram.png",
               "Figure 16. Reliability diagram for test predictions. Bubble area indicates bin support.", width=4.7)

    doc.add_heading("8.3 Grad-CAM audit", level=2)
    add_body(doc, (
        "Gradient-weighted Class Activation Mapping (Grad-CAM) uses gradients of a target class with respect to convolutional feature "
        "maps to produce a coarse influence map [9]. The audit targeted layer4.1.conv2 of the selected ResNet18. Before any heatmap was "
        "examined, 37 test images were frozen: one correct control for each of the 26 leaf classes and 11 selected errors. The checkpoint "
        "reproduced all 37 stored predictions within a maximum confidence difference of 3.46 × 10⁻⁶, after which 37 panels and seven "
        "contact sheets were generated."
    ))
    add_body(doc, (
        "Grad-CAM achieved its project purpose by revealing which coarse regions influenced the model. Many maps emphasized tissue, "
        "cellular or lesion-containing regions; others also covered edges, background or acquisition artefacts. This is not a failure of "
        "the method: it is precisely the audit signal required to identify shortcut risk. A highlighted region is not object detection, "
        "causal proof or a medical tumor boundary."
    ))
    add_figure(doc, "15_gradcam_examples.jpg",
               "Figure 17. Selected Grad-CAM panels. Each triplet shows model input, heatmap and overlay.", width=6.45)

    doc.add_heading("9. Genetic Algorithm Hyperparameter Search", level=1)
    doc.add_heading("9.1 Search space and chromosome", level=2)
    add_body(doc, (
        "The course combines neural networks and genetic algorithms; the GA was therefore used for model hyperparameter search rather "
        "than for image classification itself. One chromosome encoded three continuous genes: learning rate, weight decay and dropout. "
        "The architecture, trainable scope (ResNet18 layer4 + classifier), initialization checkpoint, data, seed and computational budget "
        "were fixed. Fitness was the best validation macro-F1 observed within a short proxy training run."
    ))
    add_table(doc,
              ["Element", "Configuration"],
              [
                  ["Genes", "Learning rate; weight decay; dropout"],
                  ["Population", "4 candidates per generation"],
                  ["Generations", "3"],
                  ["GA evaluations", "12 unique candidates"],
                  ["Random-search evaluations", "12 candidates"],
                  ["Proxy training", "4 epochs per candidate"],
                  ["Proxy data", "9,100 train; 3,900 validation; 0 test"],
                  ["Fitness", "Best proxy validation macro-F1"],
                  ["Variation", "Tournament selection, crossover and mutation probability 0.35"],
              ], widths=[Cm(4.0), Cm(12.2)], font_size=8.6)
    add_figure(doc, "18_ga_workflow.png",
               "Figure 18. Genetic-algorithm search loop and equal-budget random-search control.", width=6.45)

    doc.add_heading("9.2 Equal-budget proxy comparison", level=2)
    add_body(doc, (
        "GA and random search each evaluated 12 candidates for four epochs on the same proxy split. The best GA candidate, GA_G01_C02, "
        "used learning rate 7.2466 × 10⁻⁵, weight decay 1.9454 × 10⁻⁵ and dropout 0.4619; its proxy validation macro-F1 was 95.2281%. "
        "The best random candidate, RANDOM_002, used learning rate 5.8648 × 10⁻⁵, weight decay 3.1478 × 10⁻⁴ and dropout 0.2827; its "
        "proxy validation macro-F1 was 94.9612%. The best-proxy difference was 0.2669 percentage points in favor of GA."
    ))
    add_figure(doc, "19_ga_proxy_comparison.png",
               "Figure 19. Proxy validation macro-F1 for all 12 GA and 12 random-search trials.", width=6.45)
    add_body(doc, (
        "The winning GA chromosome appeared in the first generation and later generations did not improve it. Therefore this run shows "
        "that GA found the best candidate under the defined budget; it does not show monotonic evolutionary improvement or universal "
        "superiority of GA over random search."
    ))

    doc.add_heading("9.3 Full-data confirmation and shared test", level=2)
    add_body(doc, (
        "The proxy winners were retrained on the full 91,000-image training split and selected by macro-F1 on all 19,500 validation "
        "images, still without accessing the test set. GA achieved 99.6104% full-validation macro-F1 at epoch 10; random search achieved "
        "99.5640% at epoch 11. Only after both candidates were frozen were they evaluated once on the shared 19,500-image test set."
    ))
    add_table(doc,
              ["Method", "Test accuracy", "Macro-F1", "Correct images", "Errors"],
              [
                  ["GA-selected", "99.5795%", "99.5794%", "19,418", "82"],
                  ["Random-selected", "99.5744%", "99.5745%", "19,417", "83"],
              ], widths=[Cm(3.7), Cm(2.8), Cm(2.8), Cm(3.0), Cm(2.0)], font_size=8.6)
    add_body(doc, (
        "Paired prediction analysis found 19,366 images correct for both models, 52 correct only for GA, 51 correct only for random "
        "search and 31 wrong for both. The exact two-sided McNemar p-value was 1.0. The accuracy difference, approximately 0.0051 "
        "percentage points or one net image, is not statistically distinguishable in this paired test. The scientifically defensible "
        "conclusion is that GA and random search performed equivalently under this budget, although the GA candidate ranked first by the "
        "predeclared validation-selection procedure."
    ))
    add_figure(doc, "20_ga_shared_test.png",
               "Figure 20. Shared-test scores and paired correctness counts for the GA- and random-selected candidates.", width=6.45)

    doc.add_heading("10. Discussion", level=1)
    doc.add_heading("10.1 What produced the strongest improvement", level=2)
    add_body(doc, (
        "The largest performance gain came from adapting the final residual stage, not from hyperparameter search. Frozen-head ResNet18 "
        "reached 91.30% leaf accuracy, SmallCNN reached 95.58%, and last-block fine-tuning reached 99.65%. This indicates that general "
        "ImageNet features were valuable but insufficiently specialized; allowing layer4 to adapt created representations suited to the "
        "dataset's microscopy, MRI, CT and histology patterns. By contrast, GA and random search began from the already strong fine-tuning "
        "recipe and differed by only one net test image."
    ))

    doc.add_heading("10.2 Interpretation of very high parent performance", level=2)
    add_body(doc, (
        "Parent accuracy should not be confused with detailed disease classification. A large part of the eight-parent problem can be "
        "solved by recognizing modality and organ/source appearance. The 26-leaf task remains the primary result. Parent accuracy is "
        "reported as a hierarchical diagnostic: it shows that 67 of 69 leaf errors remained in the correct broad group."
    ))

    doc.add_heading("10.3 Strengths", level=2)
    add_bullets(doc, [
        "The dataset hierarchy and medical meaning were audited before target definition.",
        "Known duplicate families were grouped before splitting rather than removed only after training.",
        "All model comparisons used one frozen, hash-identified test set.",
        "Accuracy was accompanied by macro-F1, balanced accuracy, per-class results, paired predictions and calibration measures.",
        "The GA experiment included an equal-budget random-search control and a paired significance test.",
        "The pipeline stores resumable checkpoints and hash manifests suitable for interrupted cloud runtimes.",
    ])

    doc.add_heading("10.4 Limitations", level=2)
    add_bullets(doc, [
        "The compilation does not include patient identifiers. Consequently, patient-level independence cannot be verified; the split is family-aware only for relations discoverable from the files.",
        "The sampled natural-reference audit is small. Its 95.65% retrieval result is evidence about that sample, not exhaustive duplicate recall.",
        "Dataset classes have heterogeneous medical meaning and do not support a universal cancer-stage or universal healthy-versus-cancer output.",
        "The source release contains augmented images and source-domain cues. High test performance may partly reflect dataset-specific regularities.",
        "Grad-CAM is a coarse post-hoc influence visualization and does not validate medical reasoning.",
        "The GA study uses one search space, one computational budget and one model family; it cannot establish broad algorithmic superiority.",
    ])

    doc.add_heading("10.5 Recommended future work", level=2)
    add_numbered(doc, [
        "Obtain patient or slide identifiers and construct a true patient-level or source-slide-level split.",
        "Evaluate the frozen model on an independently collected external dataset with compatible labels.",
        "Expand expert-reviewed duplicate references within each modality and measure retrieval precision/recall prospectively.",
        "Investigate modality-specific encoders or a hierarchical model with explicit parent and leaf losses.",
        "Use repeated GA and random-search runs with multiple seeds and budgets to estimate search-method variance.",
        "Add expert pathology/radiology review of representative errors and saliency maps before any medical interpretation.",
    ])

    doc.add_heading("11. Conclusion", level=1)
    add_body(doc, (
        "The project delivered a complete, auditable neural-network and genetic-algorithm workflow for the Multi Cancer Dataset. It "
        "converted the original eight-folder description into a semantically controlled 26-leaf classification task with derived "
        "parent evaluation; audited exact and near-duplicate risk; constructed a known-family-grouped 70/15/15 split; trained and "
        "compared three baseline strategies; analyzed errors, confidence and Grad-CAM maps; and compared GA with an equal-budget random "
        "search. Last-block ResNet18 was the strongest baseline at 99.6462% leaf accuracy and macro-F1. The GA-selected and random-selected "
        "models were statistically indistinguishable on the shared test set. The most important outcome is therefore not a single high "
        "score, but a reproducible chain connecting dataset meaning, leakage control, fixed evaluation, explainability and fair search comparison."
    ))

    doc.add_heading("References", level=1)
    references = [
        ("O. S. Naren, “Multi Cancer Dataset,” Kaggle, 2022, doi: 10.34740/KAGGLE/DSV/3415848.", "https://doi.org/10.34740/kaggle/dsv/3415848"),
        ("C. Zauner, “Implementation and Benchmarking of Perceptual Image Hash Functions,” M.Sc. thesis, Upper Austria University of Applied Sciences, 2010.", "https://www.phash.org/docs/pubs/thesis_zauner.pdf"),
        ("E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” in Proc. ICCV, 2011, pp. 2564–2571, doi: 10.1109/ICCV.2011.6126544.", "https://doi.org/10.1109/ICCV.2011.6126544"),
        ("M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981, doi: 10.1145/358669.358692.", "https://doi.org/10.1145/358669.358692"),
        ("G. D. Evangelidis and E. Z. Psarakis, “Parametric image alignment using enhanced correlation coefficient maximization,” IEEE TPAMI, vol. 30, no. 10, pp. 1858–1865, 2008, doi: 10.1109/TPAMI.2008.113.", "https://doi.org/10.1109/TPAMI.2008.113"),
        ("Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE TIP, vol. 13, no. 4, pp. 600–612, 2004, doi: 10.1109/TIP.2003.819861.", "https://doi.org/10.1109/TIP.2003.819861"),
        ("K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.", "https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf"),
        ("J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. CVPR, 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.", "https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf"),
        ("R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. ICCV, 2017, pp. 618–626.", "https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf"),
        ("J. H. Holland, Adaptation in Natural and Artificial Systems. Ann Arbor, MI, USA: University of Michigan Press, 1975.", None),
        ("I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019.", "https://openreview.net/pdf?id=Bkg6RiCqY7"),
        ("F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.", "https://jmlr.org/papers/v12/pedregosa11a.html"),
        ("C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,” J. Big Data, vol. 6, art. 60, 2019, doi: 10.1186/s40537-019-0197-0.", "https://doi.org/10.1186/s40537-019-0197-0"),
        ("C. L. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, pp. 153–157, 1947.", "https://doi.org/10.1007/BF02295996"),
        ("A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019.", "https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library"),
    ]
    for number, (text, url) in enumerate(references, 1):
        add_reference(doc, number, text, url)

    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)
    # Keep the landscape appendix header/footer independent.  A linked footer
    # inherits the wider landscape text frame and can push the preceding
    # portrait section's right-aligned page number beyond the printable area.
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)
    landscape_header = section.header.paragraphs[0]
    landscape_header.text = "NNGA Multi Cancer Image Classification | Final Project Report"
    landscape_header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in landscape_header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(7.5)
        run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    add_page_number(section.footer.paragraphs[0])
    doc.add_heading("Appendix A. Complete 26-Leaf Taxonomy", level=1)
    taxonomy_rows = []
    for leaf, meta in TAXONOMY["labels"].items():
        taxonomy_rows.append([
            leaf,
            meta["parent_group"],
            meta["display_name"],
            meta["modality"],
            meta["semantic_axis"],
            meta["safe_output"],
        ])
    taxonomy_table = add_table(doc,
              ["Leaf code", "Parent", "Display name", "Modality", "Semantic axis", "Supported output"],
              taxonomy_rows,
              widths=[Cm(3.0), Cm(3.4), Cm(4.0), Cm(4.6), Cm(4.5), Cm(6.8)], font_size=6.1)
    for row in taxonomy_table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=25, start=90, bottom=25, end=90)

    if doc.paragraphs and not doc.paragraphs[-1].text:
        doc._body._body.remove(doc.paragraphs[-1]._p)
    doc.add_page_break()

    appendix_b = doc.add_heading("Appendix B. Upstream Dataset Sources", level=1)
    appendix_b.paragraph_format.page_break_before = False
    appendix_b.alignment = WD_ALIGN_PARAGRAPH.LEFT
    source_rows = [
        ["ALL", "Mehrad Aria leukemia dataset", "https://www.kaggle.com/datasets/mehradaria/leukemia"],
        ["Brain Cancer", "Brain tumor dataset", "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"],
        ["Breast Cancer", "Breast cancer dataset", "https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset"],
        ["Cervical Cancer", "SIPaKMeD cervical-cell dataset", "https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"],
        ["Kidney Cancer", "CT kidney dataset", "https://www.kaggle.com/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"],
        ["Lung and Colon", "Lung and colon cancer histopathology", "https://www.kaggle.com/datasets/biplobdey/lung-and-colon-cancer"],
        ["Lymphoma", "Malignant lymphoma classification", "https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification"],
        ["Oral Cancer", "Oral histopathology dataset", "https://www.kaggle.com/datasets/ashenafifasilkebede/dataset"],
    ]
    add_table(doc, ["Parent group", "Upstream collection", "Source URL"], source_rows,
              widths=[Cm(4.0), Cm(8.0), Cm(14.0)], font_size=7.5)

    appendix_c = doc.add_heading("Appendix C. Reproducibility Evidence Map", level=1)
    appendix_c.paragraph_format.page_break_before = False
    appendix_c.paragraph_format.left_indent = Cm(0)
    appendix_c.paragraph_format.first_line_indent = Cm(0)
    appendix_c.alignment = WD_ALIGN_PARAGRAPH.LEFT
    evidence_rows = [
        ["Label taxonomy", "config/label_taxonomy.json", "Defines 26-to-8 mapping and supported semantics."],
        ["Practical split", "research/splits/practical_leakage_reduced_split_v1.csv", "Frozen 130,000-row split manifest."],
        ["Exact duplicates", "artifacts/audit/exact_duplicates.json", "SHA-256 groups and file membership."],
        ["Best baseline", "resnet18_lastblock_finetune_v1 review package", "Metrics, predictions, curves, matrices and hashes."],
        ["Grad-CAM", "resnet18_lastblock_gradcam_v1 review package", "Frozen sample manifest and 37 panels."],
        ["GA search", "ga_resnet18_lastblock_search_v1 review package", "All 24 proxy trials and search summary."],
        ["Full confirmation", "ga_random_full_confirmation_v1 review package", "Validation-only retraining of two selected candidates."],
        ["Shared test", "ga_random_shared_test_v1 review package", "Predictions, paired counts and McNemar result."],
    ]
    add_table(doc, ["Evidence", "Artifact", "Purpose"], evidence_rows,
              widths=[Cm(4.2), Cm(10.0), Cm(12.0)], font_size=7.5)

    return doc


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = build_report()
    document.save(OUTPUT)
    print(f"Report written: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
