from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_duplicate_calibration_set import (  # noqa: E402
    SYNTHETIC_TRANSFORMS,
    apply_controlled_transform,
    load_label_audit_info,
)


class ControlledTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        yy, xx = np.mgrid[0:48, 0:64]
        pixels = np.stack(
            (
                (xx * 4) % 256,
                (yy * 5) % 256,
                ((xx + yy) * 3) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        self.image = Image.fromarray(pixels, mode="RGB")

    def test_every_controlled_transform_preserves_size_and_mode(self) -> None:
        original = np.asarray(self.image)
        for name in SYNTHETIC_TRANSFORMS:
            with self.subTest(name=name):
                transformed = apply_controlled_transform(self.image, name)
                self.assertEqual(transformed.size, self.image.size)
                self.assertEqual(transformed.mode, "RGB")
                self.assertFalse(np.array_equal(np.asarray(transformed), original))

    def test_audit_group_config_covers_exactly_26_taxonomy_leaves(self) -> None:
        mapping = load_label_audit_info(
            PROJECT_ROOT / "config" / "label_taxonomy.json",
            PROJECT_ROOT / "config" / "duplicate_audit_groups.json",
        )
        self.assertEqual(len(mapping), 26)
        self.assertEqual(len({info.source_group for info in mapping.values()}), 8)
        self.assertEqual(len({info.modality_key for info in mapping.values()}), 5)


if __name__ == "__main__":
    unittest.main()
