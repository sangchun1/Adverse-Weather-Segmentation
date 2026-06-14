from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

import numpy as np
import torch
from PIL import Image

from awseg.transforms.augmentation import build_augmentation
from awseg.transforms.enhancement import build_enhancer
from awseg.transforms.weather_augmentation import build_weather_augmentation


class ImageMaskTransform(Protocol):
    """Protocol for image/mask transforms."""

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...


def _get_resample_mode(name: str) -> int:
    """Return PIL resampling mode with compatibility across Pillow versions."""
    if hasattr(Image, "Resampling"):
        if name == "bilinear":
            return Image.Resampling.BILINEAR
        if name == "nearest":
            return Image.Resampling.NEAREST

    if name == "bilinear":
        return Image.BILINEAR
    if name == "nearest":
        return Image.NEAREST

    raise ValueError(f"Unsupported resampling mode: {name}")


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _to_tensor(
    image: Image.Image,
    mask: Optional[Image.Image],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert PIL image/mask to normalized tensors."""
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = (image_np - mean) / std
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

    if mask is None:
        return image_tensor, None

    mask_np = np.array(mask, dtype=np.int64)
    mask_tensor = torch.from_numpy(mask_np).long()
    return image_tensor, mask_tensor


def _call_optional_transform(
    transform: Any,
    image: Image.Image,
    mask: Optional[Image.Image],
    condition: Optional[str],
) -> tuple[Image.Image, Optional[Image.Image]]:
    """Call optional augmentation transform with a tolerant signature.

    Supported augmentation call signatures:
        transform(image, mask, condition=condition)
        transform(image, mask)
        transform(image)
    """
    if transform is None:
        return image, mask

    try:
        output = transform(image, mask, condition=condition)
    except TypeError:
        try:
            output = transform(image, mask)
        except TypeError:
            output = transform(image)

    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(
                "Augmentation transform must return (image, mask) when returning a tuple."
            )
        out_image, out_mask = output
        return out_image, out_mask

    if isinstance(output, Image.Image):
        return output, mask

    raise TypeError(
        "Augmentation transform must return PIL Image or "
        "tuple[PIL Image, Optional[PIL Image]]."
    )


class SegmentationTransform:
    """Main transform for semantic segmentation.

    Pipeline:
        1. Convert image to RGB.
        2. Apply optional enhancement.
        3. Apply optional train-time basic augmentation.
        4. Apply optional weather augmentation depending on config.
        5. Resize image and mask.
        6. Normalize image.
        7. Convert image and mask to torch.Tensor.

    Important:
        - Image uses bilinear interpolation.
        - Mask uses nearest-neighbor interpolation to preserve class IDs.
        - Enhancement can be applied to train/val/test depending on config.
        - Basic augmentation is applied only for train split.
        - Weather augmentation can be applied to train/val/test through augmentation.weather.apply_splits.
        - Frequency augmentation / frequency map concat are intentionally excluded.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        split: str,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        enhancer: Any = None,
        augmentation: Any = None,
        weather_augmentation: Any = None,
    ) -> None:
        self.height, self.width = size
        self.split = str(split).lower()
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if self.mean.shape != (3,):
            raise ValueError(f"mean must have 3 values, got {self.mean}")
        if self.std.shape != (3,):
            raise ValueError(f"std must have 3 values, got {self.std}")
        if np.any(self.std == 0):
            raise ValueError(f"std must not contain zero, got {self.std}")

        self.enhancer = enhancer
        self.augmentation = augmentation
        self.weather_augmentation = weather_augmentation
        self.image_resample = _get_resample_mode("bilinear")
        self.mask_resample = _get_resample_mode("nearest")

    def _apply_enhancement(
        self,
        image: Image.Image,
        condition: Optional[str],
    ) -> Image.Image:
        if self.enhancer is None:
            return image

        image = self.enhancer(image, condition=condition)
        return _ensure_rgb(image)

    def _apply_augmentation(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
        condition: Optional[str],
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        # Basic augmentation, e.g. color jitter, is still train-only.
        if self.split == "train":
            image, mask = _call_optional_transform(
                transform=self.augmentation,
                image=image,
                mask=mask,
                condition=condition,
            )

        # Weather augmentation can be applied to train/val/test
        # depending on augmentation.weather.apply_splits.
        image, mask = _call_optional_transform(
            transform=self.weather_augmentation,
            image=image,
            mask=mask,
            condition=condition,
        )

        return _ensure_rgb(image), mask

    def _resize(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = image.resize((self.width, self.height), self.image_resample)

        if mask is not None:
            mask = mask.resize((self.width, self.height), self.mask_resample)

        return image, mask

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        image = _ensure_rgb(image)
        image = self._apply_enhancement(image=image, condition=condition)
        image, mask = self._apply_augmentation(
            image=image,
            mask=mask,
            condition=condition,
        )
        image, mask = self._resize(image=image, mask=mask)

        return _to_tensor(
            image=image,
            mask=mask,
            mean=self.mean,
            std=self.std,
        )


class BaselineTransform(SegmentationTransform):
    """Backward-compatible alias for the baseline transform."""

    def __init__(
        self,
        size: Tuple[int, int],
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        enhancer: Any = None,
    ) -> None:
        super().__init__(
            size=size,
            split="val",
            mean=mean,
            std=std,
            enhancer=enhancer,
            augmentation=None,
            weather_augmentation=None,
        )


def _get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise KeyError("config must contain a 'data' dictionary.")
    return data_config


def _get_input_size(data_config: dict[str, Any]) -> tuple[int, int]:
    """Read input size from config.

    Supported forms:
        data:
          input_height: 512
          input_width: 512

        data:
          input_size: [512, 512]
    """
    if "input_height" in data_config and "input_width" in data_config:
        return int(data_config["input_height"]), int(data_config["input_width"])

    input_size = data_config.get("input_size")
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])

    raise KeyError(
        "data config must contain either input_height/input_width or input_size."
    )


def _build_basic_augmentation(config: dict[str, Any], split: str) -> Any:
    if split != "train":
        return None

    augmentation_config = config.get("augmentation", {})
    if not isinstance(augmentation_config, dict):
        return None

    if not bool(augmentation_config.get("enabled", False)):
        return None

    return build_augmentation(config=config, split=split)


def _is_split_enabled_for_weather(weather_config: dict[str, Any], split: str) -> bool:
    split = str(split).lower()

    apply_splits = weather_config.get("apply_splits", ["train"])

    if isinstance(apply_splits, str):
        apply_splits = apply_splits.strip().lower()
        if apply_splits == "all":
            return True
        return split == apply_splits

    if isinstance(apply_splits, (list, tuple, set)):
        normalized = {str(item).strip().lower() for item in apply_splits}
        return "all" in normalized or split in normalized

    return split == "train"


def _build_weather_augmentation(config: dict[str, Any], split: str) -> Any:
    augmentation_config = config.get("augmentation", {})
    if not isinstance(augmentation_config, dict):
        return None

    weather_config = augmentation_config.get("weather", {})
    if not isinstance(weather_config, dict):
        return None

    if not bool(weather_config.get("enabled", False)):
        return None

    if not _is_split_enabled_for_weather(weather_config, split):
        return None

    return build_weather_augmentation(config=config, split=split)


def build_transform(config: dict[str, Any], split: str) -> SegmentationTransform:
    """Build transform from config.

    Args:
        config: Experiment config dictionary.
        split: Dataset split name. Usually one of train, val, test.

    Returns:
        SegmentationTransform.
    """
    split = str(split).lower()
    data_config = _get_data_config(config)
    height, width = _get_input_size(data_config)
    mean = tuple(data_config.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_config.get("std", [0.229, 0.224, 0.225]))

    enhancer = build_enhancer(config=config, split=split)
    augmentation = _build_basic_augmentation(config=config, split=split)
    weather_augmentation = _build_weather_augmentation(config=config, split=split)

    return SegmentationTransform(
        size=(height, width),
        split=split,
        mean=mean,
        std=std,
        enhancer=enhancer,
        augmentation=augmentation,
        weather_augmentation=weather_augmentation,
    )
