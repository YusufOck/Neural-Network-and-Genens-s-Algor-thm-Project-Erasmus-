from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from render_candidate_generator_v2_5_prospective_review import (  # noqa: E402
    FORBIDDEN_PRIVATE_FIELDS,
    REQUIRED_FIELDS,
    read_public_review,
    render_page,
    run,
    validate_public_rows,
)
from data_index import EXPECTED_MAIN_CLASSES  # noqa: E402


def row(pair_id: str = "V25N000001") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "source_group": "brain_ce_mri",
        "left_leaf": "brain_glioma",
        "right_leaf": "brain_glioma",
        "left_path": "left.png",
        "right_path": "right.png",
        "visual_verdict": "",
        "confidence": "",
        "reviewer": "",
        "review_date": "",
        "review_notes": "",
    }


def write_review(path: Path, rows: list[dict[str, str]], fields=REQUIRED_FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V25ProspectiveRenderTests(unittest.TestCase):
    def test_rejects_private_field_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            fields = tuple(REQUIRED_FIELDS) + ("selection_stratum",)
            leaked = row()
            leaked["selection_stratum"] = "outside_random"
            write_review(path, [leaked], fields)
            with self.assertRaises(ValueError):
                read_public_review(path)

    def test_rejects_prefilled_or_reused_rows(self) -> None:
        prefilled = row()
        prefilled["visual_verdict"] = "independent_lookalike"
        with self.assertRaises(ValueError):
            validate_public_rows([prefilled], 1)
        reused = row()
        second = row("V25N000002")
        second["right_path"] = "other.png"
        with self.assertRaises(ValueError):
            validate_public_rows([reused, second], 2)

    def test_render_page_has_expected_blind_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (40, 30), "red").save(root / "left.png")
            Image.new("RGB", (30, 40), "blue").save(root / "right.png")
            page = render_page([row()], root, 1, 1, 128)
            self.assertEqual(page.width, 20 * 2 + 128 * 2 + 12)
            self.assertGreater(page.height, 128)

    def test_run_needs_no_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection"
            dataset = root / "dataset"
            output = root / "render"
            selection.mkdir()
            dataset.mkdir()
            for main_class in EXPECTED_MAIN_CLASSES:
                (dataset / main_class).mkdir()
            Image.new("RGB", (40, 30), "red").save(dataset / "left.png")
            Image.new("RGB", (30, 40), "blue").save(dataset / "right.png")
            review = selection / "review_manifest.csv"
            write_review(review, [row()])
            report = {
                "status": "v2_5_prospective_selection_frozen_images_not_opened",
                "selection": {"selected_pairs": 1},
                "integrity": {"review_manifest_sha256": sha256(review)},
            }
            (selection / "selection_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            _, rendered = run(
                Namespace(
                    selection_dir=str(selection),
                    output_dir=str(output),
                    data_dir=str(dataset),
                    pairs_per_page=1,
                    thumb_size=128,
                )
            )
            self.assertFalse((selection / "selection_private.csv").exists())
            self.assertTrue(rendered["does_not_read_private_manifest"])
            self.assertEqual(rendered["counts"]["pair_count"], 1)
            self.assertTrue(set(FORBIDDEN_PRIVATE_FIELDS).issubset(rendered["layout"]["hidden_private_fields"]))


if __name__ == "__main__":
    unittest.main()
