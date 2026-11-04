from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prepare_natural_duplicate_adjudication_v1 import (  # noqa: E402
    ADJUDICATION_FIELDS,
    is_safe_consensus,
    render_pair_card,
    select_queue_rows,
)


def unblinded(pair_id: str, verdict: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "left_path": "group/left.jpg",
        "right_path": "group/right.jpg",
        "visual_verdict": verdict,
        "confidence": "medium",
        "review_notes": "frozen note",
    }


def scored(pair_id: str, visual: str, decision: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "human_verdict": visual,
        "decision": decision,
        "decision_path": "uncertain" if decision == "uncertain" else "path",
        "decision_reason": "reason",
        "source_group": "kidney_ct",
        "modality_key": "ct",
        "left_leaf": "kidney_normal",
        "right_leaf": "kidney_normal",
        "selection_stratum": "candidate_phash_supported",
        "audit_region": "candidate",
        "phash_distance": "10",
        "robust_top15": "yes",
        "robust_best_union_rank": "4",
        "robust_best_cosine_similarity": "0.95",
        "orb_inliers": "42",
        "orb_inlier_ratio": "0.5",
        "orb_coverage_min": "0.4",
        "orb_affine_plausible": "yes",
        "best_alignment_method": "ecc",
        "best_alignment_orientation": "original",
        "best_ssim": "0.75",
        "best_nmi": "1.2",
        "best_correlation": "0.9",
        "best_overlap": "0.8",
    }


class AdjudicationSelectionTests(unittest.TestCase):
    def test_only_safe_consensus_is_excluded(self) -> None:
        self.assertTrue(
            is_safe_consensus(scored("N1", "same_source_candidate", "same_source"))
        )
        self.assertTrue(
            is_safe_consensus(scored("N2", "independent_lookalike", "independent"))
        )
        self.assertFalse(
            is_safe_consensus(scored("N3", "independent_lookalike", "uncertain"))
        )

    def test_queue_is_prioritized_and_decisions_are_blank(self) -> None:
        visual_rows = [
            unblinded("N000003", "independent_lookalike"),
            unblinded("N000001", "same_source_candidate"),
            unblinded("N000004", "uncertain"),
            unblinded("N000002", "independent_lookalike"),
        ]
        score_rows = [
            scored("N000001", "same_source_candidate", "same_source"),
            scored("N000002", "independent_lookalike", "same_source"),
            scored("N000003", "independent_lookalike", "uncertain"),
            scored("N000004", "uncertain", "independent"),
        ]
        queue = select_queue_rows(visual_rows, score_rows)
        self.assertEqual([row["pair_id"] for row in queue], ["N000002", "N000004", "N000003"])
        self.assertEqual(
            [row["queue_category"] for row in queue],
            ["direct_conflict", "visual_uncertain", "rule_uncertain"],
        )
        self.assertTrue(
            all(not row[field] for row in queue for field in ADJUDICATION_FIELDS)
        )

    def test_pair_id_coverage_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pair ID kapsamı"):
            select_queue_rows(
                [unblinded("N000001", "independent_lookalike")],
                [scored("N000002", "independent_lookalike", "uncertain")],
            )

    def test_renderer_creates_score_aware_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "group"
            image_dir.mkdir()
            Image.new("RGB", (64, 64), "red").save(image_dir / "left.jpg")
            Image.new("RGB", (64, 64), "blue").save(image_dir / "right.jpg")
            row = select_queue_rows(
                [unblinded("N000002", "independent_lookalike")],
                [scored("N000002", "independent_lookalike", "same_source")],
            )[0]
            card = render_pair_card(row, root.resolve(), 256)
            self.assertEqual(card.mode, "RGB")
            self.assertGreater(card.width, 512)
            self.assertGreater(card.height, 500)


if __name__ == "__main__":
    unittest.main()
