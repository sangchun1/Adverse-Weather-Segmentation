from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


class BaselineTransform:
    """Baseline transform for semantic segmentation.

    This transform is intentionally simple for the baseline experiment:
      1. Resize image and mask to the configured input size.
      2. Normalize image.
      3. Convert image and mask to torch.Tensor.

    Important:
        - Image uses bilinear interpolation.
        - Mask uses nearest-neighbor interpolation to preserve class IDs.
        - Enhancement preprocessing and augmentation are not applied here yet.
          They can be added later through build_transform().
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize baseline transform.

        Args:
            size: Target image size as (height, width).
            mean: RGB channel mean for normalization.
            std: RGB channel standard deviation for normalization.
        """
        self.height, self.width = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply transform to image and mask.

        Args:
            image: PIL RGB image.
            mask: Optional PIL mask image containing class IDs.

        Returns:
            image: FloatTensor with shape [3, H, W].
            mask: LongTensor with shape [H, W], or None.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((self.width, self.height), Image.BILINEAR)

        if mask is not None:
            mask = mask.resize((self.width, self.height), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if mask is None:
            return image_tensor, None

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


def build_transform(config: dict, split: str) -> BaselineTransform:
    """Build transform from config.

    Currently this returns only the baseline transform.

    The split argument is kept for future extension:
      - train: baseline + optional basic/weather augmentation
      - val/test: baseline preprocessing only

    Args:
        config: Experiment config dictionary.
        split: Dataset split name. Usually one of train, val, test.

    Returns:
        BaselineTransform.
    """
    data_config = config["data"]

    height = int(data_config["input_height"])
    width = int(data_config["input_width"])

    mean = tuple(data_config.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_config.get("std", [0.229, 0.224, 0.225]))

    return BaselineTransform(
        size=(height, width),
        mean=mean,
        std=std,
    )
