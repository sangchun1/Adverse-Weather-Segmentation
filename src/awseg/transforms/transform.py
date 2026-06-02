from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from awseg.transforms.enhancement import build_enhancer


class BaselineTransform:
    """Baseline transform for semantic segmentation.

    Pipeline:
      1. Convert image to RGB.
      2. Apply optional enhancement before resizing.
      3. Resize image and mask to the configured input size.
      4. Normalize image.
      5. Convert image and mask to torch.Tensor.

    Important:
      - Image uses bilinear interpolation.
      - Mask uses nearest-neighbor interpolation to preserve class IDs.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        enhancer=None,
    ) -> None:
        """Initialize baseline transform.

        Args:
            size: Target image size as (height, width).
            mean: RGB channel mean for normalization.
            std: RGB channel standard deviation for normalization.
            enhancer: Optional condition-aware image enhancer.
        """
        self.height, self.width = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.enhancer = enhancer

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply transform to image and mask.

        Args:
            image: PIL RGB image.
            mask: Optional PIL mask image containing class IDs.
            condition: Optional ACDC condition name, e.g. night/fog/rain/snow.

        Returns:
            image: FloatTensor with shape [3, H, W].
            mask: LongTensor with shape [H, W], or None.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.enhancer is not None:
            image = self.enhancer(image, condition=condition)
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


class FrequencyAugmentationTransform:
    """Image-level frequency augmentation transform.

    This transform keeps the input as RGB 3-channel.
    Frequency perturbation is applied only for the train split.

    Output:
      image: FloatTensor with shape [3, H, W]
      mask: LongTensor with shape [H, W], or None
    """

    def __init__(
        self,
        size: Tuple[int, int],
        split: str,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        p: float = 0.5,
        low_radius_ratio: float = 0.08,
        low_scale_range: tuple[float, float] = (0.8, 1.2),
        high_scale_range: tuple[float, float] = (0.9, 1.1),
        enhancer=None,
    ) -> None:
        self.height, self.width = size
        self.split = split
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.p = float(p)
        self.low_radius_ratio = float(low_radius_ratio)
        self.low_scale_range = tuple(low_scale_range)
        self.high_scale_range = tuple(high_scale_range)
        self.enhancer = enhancer

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.enhancer is not None:
            image = self.enhancer(image, condition=condition)
            if image.mode != "RGB":
                image = image.convert("RGB")

        image = image.resize((self.width, self.height), Image.BILINEAR)

        if mask is not None:
            mask = mask.resize((self.width, self.height), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32) / 255.0

        if self.split == "train" and np.random.rand() < self.p:
            image_np = _apply_frequency_augmentation(
                image_np=image_np,
                low_radius_ratio=self.low_radius_ratio,
                low_scale_range=self.low_scale_range,
                high_scale_range=self.high_scale_range,
            )

        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if mask is None:
            return image_tensor, None

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


class FrequencyMapConcatTransform:
    """RGB + frequency map concat transform.

    This transform changes the input channel count.

    Modes:
      - mode="low": output shape is [4, H, W] RGB + low-frequency map
      - mode="low_high": output shape is [5, H, W]
        RGB + low-frequency map + high-frequency map

    Important:
      This transform must be applied consistently to train/val/test, because the
      model input channel count changes.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mode: str,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        low_radius_ratio: float = 0.08,
        frequency_map_mean: float = 0.5,
        frequency_map_std: float = 0.5,
        enhancer=None,
    ) -> None:
        if mode not in {"low", "low_high"}:
            raise ValueError(
                f"Unsupported frequency map mode: {mode}. "
                "Expected one of {'low', 'low_high'}."
            )

        self.height, self.width = size
        self.mode = mode
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.low_radius_ratio = float(low_radius_ratio)
        self.frequency_map_mean = float(frequency_map_mean)
        self.frequency_map_std = float(frequency_map_std)
        self.enhancer = enhancer

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.enhancer is not None:
            image = self.enhancer(image, condition=condition)
            if image.mode != "RGB":
                image = image.convert("RGB")

        image = image.resize((self.width, self.height), Image.BILINEAR)

        if mask is not None:
            mask = mask.resize((self.width, self.height), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = _make_rgb_frequency_tensor(
            image_np=image_np,
            mode=self.mode,
            mean=self.mean,
            std=self.std,
            low_radius_ratio=self.low_radius_ratio,
            frequency_map_mean=self.frequency_map_mean,
            frequency_map_std=self.frequency_map_std,
        )

        if mask is None:
            return image_tensor, None

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


class FrequencyAugmentationMapConcatTransform:
    """Frequency augmentation + frequency map concat transform.

    This is for combined experiments.

    Example:
      RGB + low-frequency map input, while applying frequency augmentation only
      during training.

    Output channel count depends on mode:
      - mode="low": [4, H, W]
      - mode="low_high": [5, H, W]
    """

    def __init__(
        self,
        size: Tuple[int, int],
        split: str,
        mode: str,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        p: float = 0.5,
        low_radius_ratio: float = 0.08,
        low_scale_range: tuple[float, float] = (0.8, 1.2),
        high_scale_range: tuple[float, float] = (0.9, 1.1),
        frequency_map_mean: float = 0.5,
        frequency_map_std: float = 0.5,
        enhancer=None,
    ) -> None:
        if mode not in {"low", "low_high"}:
            raise ValueError(
                f"Unsupported frequency map mode: {mode}. "
                "Expected one of {'low', 'low_high'}."
            )

        self.height, self.width = size
        self.split = split
        self.mode = mode
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.p = float(p)
        self.low_radius_ratio = float(low_radius_ratio)
        self.low_scale_range = tuple(low_scale_range)
        self.high_scale_range = tuple(high_scale_range)
        self.frequency_map_mean = float(frequency_map_mean)
        self.frequency_map_std = float(frequency_map_std)
        self.enhancer = enhancer

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.enhancer is not None:
            image = self.enhancer(image, condition=condition)
            if image.mode != "RGB":
                image = image.convert("RGB")

        image = image.resize((self.width, self.height), Image.BILINEAR)

        if mask is not None:
            mask = mask.resize((self.width, self.height), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32) / 255.0

        if self.split == "train" and np.random.rand() < self.p:
            image_np = _apply_frequency_augmentation(
                image_np=image_np,
                low_radius_ratio=self.low_radius_ratio,
                low_scale_range=self.low_scale_range,
                high_scale_range=self.high_scale_range,
            )

        image_tensor = _make_rgb_frequency_tensor(
            image_np=image_np,
            mode=self.mode,
            mean=self.mean,
            std=self.std,
            low_radius_ratio=self.low_radius_ratio,
            frequency_map_mean=self.frequency_map_mean,
            frequency_map_std=self.frequency_map_std,
        )

        if mask is None:
            return image_tensor, None

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


def _apply_frequency_augmentation(
    image_np: np.ndarray,
    low_radius_ratio: float,
    low_scale_range: tuple[float, float],
    high_scale_range: tuple[float, float],
) -> np.ndarray:
    """Randomly perturb low/high frequency components of an RGB image.

    Args:
        image_np: RGB image array with shape [H, W, 3], range [0, 1].
        low_radius_ratio: Radius ratio for the low-frequency circular mask.
        low_scale_range: Random scale range for low-frequency components.
        high_scale_range: Random scale range for high-frequency components.

    Returns:
        Augmented RGB image array with shape [H, W, 3], range [0, 1].
    """
    height, width, channels = image_np.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {channels}.")

    low_mask = _make_low_frequency_mask(
        height=height,
        width=width,
        low_radius_ratio=low_radius_ratio,
    )

    low_scale = np.random.uniform(low_scale_range[0], low_scale_range[1])
    high_scale = np.random.uniform(high_scale_range[0], high_scale_range[1])

    augmented_channels = []
    for channel_idx in range(3):
        channel = image_np[:, :, channel_idx]
        freq = np.fft.fft2(channel)
        freq_shifted = np.fft.fftshift(freq)

        freq_shifted_aug = freq_shifted.copy()
        freq_shifted_aug[low_mask] *= low_scale
        freq_shifted_aug[~low_mask] *= high_scale

        freq_aug = np.fft.ifftshift(freq_shifted_aug)
        channel_aug = np.fft.ifft2(freq_aug).real
        augmented_channels.append(channel_aug)

    augmented = np.stack(augmented_channels, axis=-1)
    augmented = np.clip(augmented, 0.0, 1.0).astype(np.float32)
    return augmented


def _make_rgb_frequency_tensor(
    image_np: np.ndarray,
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    low_radius_ratio: float,
    frequency_map_mean: float,
    frequency_map_std: float,
) -> torch.Tensor:
    """Create tensor from RGB image and frequency maps.

    Args:
        image_np: RGB image array with shape [H, W, 3], range [0, 1].
        mode: "low" or "low_high".
        mean: RGB normalization mean.
        std: RGB normalization std.
        low_radius_ratio: Radius ratio for the low-frequency circular mask.
        frequency_map_mean: Mean for frequency map normalization.
        frequency_map_std: Std for frequency map normalization.

    Returns:
        FloatTensor with shape [4, H, W] or [5, H, W].
    """
    rgb_np = (image_np - mean) / std
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()

    frequency_maps = _compute_frequency_maps(
        image_np=image_np,
        mode=mode,
        low_radius_ratio=low_radius_ratio,
    )
    frequency_maps = (
        frequency_maps - frequency_map_mean
    ) / max(frequency_map_std, 1e-6)
    frequency_tensor = torch.from_numpy(frequency_maps).permute(2, 0, 1).float()

    image_tensor = torch.cat([rgb_tensor, frequency_tensor], dim=0)
    return image_tensor


def _compute_frequency_maps(
    image_np: np.ndarray,
    mode: str,
    low_radius_ratio: float,
) -> np.ndarray:
    """Compute low/high frequency maps from an RGB image.

    Args:
        image_np: RGB image array with shape [H, W, 3], range [0, 1].
        mode: "low" or "low_high".
        low_radius_ratio: Radius ratio for the low-frequency circular mask.

    Returns:
        Frequency map array:
          - mode="low": shape [H, W, 1]
          - mode="low_high": shape [H, W, 2]
    """
    if mode not in {"low", "low_high"}:
        raise ValueError(
            f"Unsupported frequency map mode: {mode}. "
            "Expected one of {'low', 'low_high'}."
        )

    gray = (
        0.299 * image_np[:, :, 0]
        + 0.587 * image_np[:, :, 1]
        + 0.114 * image_np[:, :, 2]
    ).astype(np.float32)

    height, width = gray.shape
    low_mask = _make_low_frequency_mask(
        height=height,
        width=width,
        low_radius_ratio=low_radius_ratio,
    )

    freq = np.fft.fft2(gray)
    freq_shifted = np.fft.fftshift(freq)

    low_freq_shifted = np.zeros_like(freq_shifted)
    low_freq_shifted[low_mask] = freq_shifted[low_mask]
    low_freq = np.fft.ifftshift(low_freq_shifted)
    low_map = np.fft.ifft2(low_freq).real
    low_map = np.clip(low_map, 0.0, 1.0).astype(np.float32)

    if mode == "low":
        return low_map[:, :, None]

    high_map = np.abs(gray - low_map).astype(np.float32)
    high_map = high_map / (high_map.max() + 1e-6)
    return np.stack([low_map, high_map], axis=-1).astype(np.float32)


def _make_low_frequency_mask(
    height: int,
    width: int,
    low_radius_ratio: float,
) -> np.ndarray:
    """Make circular low-frequency mask for centered FFT output."""
    center_y = height // 2
    center_x = width // 2
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    radius = max(1, int(min(height, width) * low_radius_ratio))
    return distance <= radius


def build_transform(config: dict, split: str):
    """Build transform from config.

    Supported augmentation names:
      - none
      - baseline
      - frequency_aug
      - frequency_map_low
      - frequency_map_low_high
      - frequency_aug_map_low
      - frequency_aug_map_low_high
      - frequency_map_low_freq_aug
      - frequency_map_low_high_freq_aug

    Enhancement is handled independently through config["enhancement"].
    It can be combined with baseline/frequency transforms, although the current
    all-data experiments mainly use augmentation.name="none".

    Args:
        config: Experiment config dictionary.
        split: Dataset split name. Usually one of train, val, test.

    Returns:
        Transform instance.
    """
    data_config = config["data"]
    height = int(data_config["input_height"])
    width = int(data_config["input_width"])
    mean = tuple(data_config.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_config.get("std", [0.229, 0.224, 0.225]))

    augmentation_config = config.get("augmentation", {})
    augmentation_enabled = bool(augmentation_config.get("enabled", False))
    augmentation_name = str(augmentation_config.get("name", "none")).lower()

    enhancer = build_enhancer(config, split=split)

    if not augmentation_enabled or augmentation_name in {"none", "baseline"}:
        return BaselineTransform(
            size=(height, width),
            mean=mean,
            std=std,
            enhancer=enhancer,
        )

    if augmentation_name == "frequency_aug":
        return FrequencyAugmentationTransform(
            size=(height, width),
            split=split,
            mean=mean,
            std=std,
            p=float(augmentation_config.get("p", 0.5)),
            low_radius_ratio=float(augmentation_config.get("low_radius_ratio", 0.08)),
            low_scale_range=tuple(augmentation_config.get("low_scale_range", [0.8, 1.2])),
            high_scale_range=tuple(augmentation_config.get("high_scale_range", [0.9, 1.1])),
            enhancer=enhancer,
        )

    if augmentation_name == "frequency_map_low":
        return FrequencyMapConcatTransform(
            size=(height, width),
            mode="low",
            mean=mean,
            std=std,
            low_radius_ratio=float(augmentation_config.get("low_radius_ratio", 0.08)),
            frequency_map_mean=float(augmentation_config.get("mean", 0.5)),
            frequency_map_std=float(augmentation_config.get("std", 0.5)),
            enhancer=enhancer,
        )

    if augmentation_name == "frequency_map_low_high":
        return FrequencyMapConcatTransform(
            size=(height, width),
            mode="low_high",
            mean=mean,
            std=std,
            low_radius_ratio=float(augmentation_config.get("low_radius_ratio", 0.08)),
            frequency_map_mean=float(augmentation_config.get("mean", 0.5)),
            frequency_map_std=float(augmentation_config.get("std", 0.5)),
            enhancer=enhancer,
        )

    if augmentation_name in {
        "frequency_aug_map_low",
        "frequency_map_low_freq_aug",
    }:
        return FrequencyAugmentationMapConcatTransform(
            size=(height, width),
            split=split,
            mode="low",
            mean=mean,
            std=std,
            p=float(augmentation_config.get("p", 0.5)),
            low_radius_ratio=float(augmentation_config.get("low_radius_ratio", 0.08)),
            low_scale_range=tuple(augmentation_config.get("low_scale_range", [0.8, 1.2])),
            high_scale_range=tuple(augmentation_config.get("high_scale_range", [0.9, 1.1])),
            frequency_map_mean=float(augmentation_config.get("mean", 0.5)),
            frequency_map_std=float(augmentation_config.get("std", 0.5)),
            enhancer=enhancer,
        )

    if augmentation_name in {
        "frequency_aug_map_low_high",
        "frequency_map_low_high_freq_aug",
    }:
        return FrequencyAugmentationMapConcatTransform(
            size=(height, width),
            split=split,
            mode="low_high",
            mean=mean,
            std=std,
            p=float(augmentation_config.get("p", 0.5)),
            low_radius_ratio=float(augmentation_config.get("low_radius_ratio", 0.08)),
            low_scale_range=tuple(augmentation_config.get("low_scale_range", [0.8, 1.2])),
            high_scale_range=tuple(augmentation_config.get("high_scale_range", [0.9, 1.1])),
            frequency_map_mean=float(augmentation_config.get("mean", 0.5)),
            frequency_map_std=float(augmentation_config.get("std", 0.5)),
            enhancer=enhancer,
        )

    raise ValueError(
        f"Unsupported augmentation.name: {augmentation_name}. "
        "Expected one of: "
        "'none', 'baseline', 'frequency_aug', "
        "'frequency_map_low', 'frequency_map_low_high', "
        "'frequency_aug_map_low', 'frequency_aug_map_low_high'."
    )
