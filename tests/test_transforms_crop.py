from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("torch", types.SimpleNamespace())

from awseg.transforms.transform import AugmentedTransform


class ClassBalancedCropTest(unittest.TestCase):
    def test_selects_present_low_iou_class_and_connected_object(self) -> None:
        transform = AugmentedTransform(
            size=(4, 4),
            augmentation_config={
                "class_balanced_crop": {
                    "enabled": True,
                    "prob": 1.0,
                    "rare_classes": [3, 4],
                    "min_pixels": 1,
                }
            },
        )
        mask = np.full((10, 10), 255, dtype=np.int64)
        mask[1:3, 1:3] = 3
        mask[1:3, 7:9] = 4
        mask[7:9, 7:9] = 4

        with patch("awseg.transforms.transform.random.random", return_value=0.0):
            with patch("awseg.transforms.transform.random.choice", side_effect=lambda seq: seq[-1]):
                crop_box = transform._get_class_balanced_crop_box(mask)

        self.assertIsNotNone(crop_box)
        left, top, right, bottom = crop_box
        self.assertEqual((right - left, bottom - top), (4, 4))
        self.assertLessEqual(left, 7)
        self.assertLessEqual(top, 7)
        self.assertGreater(right, 8)
        self.assertGreater(bottom, 8)

    def test_returns_none_when_no_configured_class_is_present(self) -> None:
        transform = AugmentedTransform(
            size=(4, 4),
            augmentation_config={
                "class_balanced_crop": {
                    "enabled": True,
                    "prob": 1.0,
                    "rare_classes": [12, 17, 18],
                    "min_pixels": 1,
                }
            },
        )
        mask = np.zeros((10, 10), dtype=np.int64)

        with patch("awseg.transforms.transform.random.random", return_value=0.0):
            crop_box = transform._get_class_balanced_crop_box(mask)

        self.assertIsNone(crop_box)


if __name__ == "__main__":
    unittest.main()
