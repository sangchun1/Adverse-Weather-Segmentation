from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance


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


class AugmentedTransform(BaselineTransform):
    """Training transform with configurable image augmentation."""

    def __init__(
        self,
        size: Tuple[int, int],
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augmentation_config: Optional[dict] = None,
    ) -> None:
        super().__init__(size=size, mean=mean, std=std)
        self.augmentation_config = augmentation_config or {}

    def _apply_horizontal_flip(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        flip_config = self.augmentation_config.get("horizontal_flip", {})

        if not bool(flip_config.get("enabled", False)):
            return image, mask

        prob = float(flip_config.get("prob", 0.5))

        if random.random() >= prob:
            return image, mask

        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return image, mask

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        jitter_config = self.augmentation_config.get("color_jitter", {})

        if not bool(jitter_config.get("enabled", False)):
            return image

        prob = float(jitter_config.get("prob", 0.5))

        if random.random() >= prob:
            return image

        brightness = float(jitter_config.get("brightness", 0.0))
        contrast = float(jitter_config.get("contrast", 0.0))
        saturation = float(jitter_config.get("saturation", 0.0))

        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if saturation > 0:
            factor = 1.0 + random.uniform(-saturation, saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        return image

    def _find_connected_components(
        self,
        binary_mask: np.ndarray,
        min_pixels: int,
    ) -> list[tuple[int, int, int, int, int]]:
        height, width = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        components: list[tuple[int, int, int, int, int]] = []
        ys, xs = np.where(binary_mask)

        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if visited[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            pixel_count = 0

            while stack:
                y, x = stack.pop()
                pixel_count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                for next_y in range(max(0, y - 1), min(height, y + 2)):
                    for next_x in range(max(0, x - 1), min(width, x + 2)):
                        if visited[next_y, next_x] or not binary_mask[next_y, next_x]:
                            continue

                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

            if pixel_count >= min_pixels:
                components.append((min_x, min_y, max_x, max_y, pixel_count))

        return components

    def _get_class_balanced_crop_box(
        self,
        mask_np: np.ndarray,
    ) -> Optional[tuple[int, int, int, int]]:
        crop_config = self.augmentation_config.get("class_balanced_crop", {})

        if not bool(crop_config.get("enabled", False)):
            return None

        prob = float(crop_config.get("prob", 0.3))

        if random.random() >= prob:
            return None

        rare_classes = [int(class_id) for class_id in crop_config.get("rare_classes", [])]

        if len(rare_classes) == 0:
            return None

        min_pixels = int(crop_config.get("min_pixels", 30))
        present_classes = [
            class_id
            for class_id in rare_classes
            if int(np.count_nonzero(mask_np == class_id)) >= min_pixels
        ]

        if len(present_classes) == 0:
            return None

        target_class = random.choice(present_classes)
        components = self._find_connected_components(
            binary_mask=mask_np == target_class,
            min_pixels=min_pixels,
        )

        if len(components) == 0:
            return None

        min_x, min_y, max_x, max_y, _ = random.choice(components)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        mask_height, mask_width = mask_np.shape
        crop_width = min(self.width, mask_width)
        crop_height = min(self.height, mask_height)

        left = max(0, min(center_x - crop_width // 2, mask_width - crop_width))
        top = max(0, min(center_y - crop_height // 2, mask_height - crop_height))
        right = left + crop_width
        bottom = top + crop_height

        return left, top, right, bottom

    def _apply_class_balanced_crop(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        if mask is None:
            return image, mask

        mask_np = np.array(mask, dtype=np.int64)
        crop_box = self._get_class_balanced_crop_box(mask_np)

        if crop_box is None:
            return image, mask

        return image.crop(crop_box), mask.crop(crop_box)

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image, mask = self._apply_class_balanced_crop(image, mask)
        image, mask = self._apply_horizontal_flip(image, mask)
        image = self._apply_color_jitter(image)

        return super().__call__(image, mask)


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

    augmentation_config = config.get("augmentation", {})

    if split == "train" and bool(augmentation_config.get("enabled", False)):
        return AugmentedTransform(
            size=(height, width),
            mean=mean,
            std=std,
            augmentation_config=augmentation_config,
        )

    return BaselineTransform(
        size=(height, width),
        mean=mean,
        std=std,
    )
