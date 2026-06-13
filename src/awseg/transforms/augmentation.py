from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageEnhance


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _get_color_jitter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read color jitter config from supported schemas.

    Supported:
        augmentation:
          color_jitter: ...

        augmentation:
          basic:
            color_jitter: ...
    """
    aug_config = config.get("augmentation", {})
    if not isinstance(aug_config, dict):
        return {}

    if isinstance(aug_config.get("color_jitter"), dict):
        return dict(aug_config["color_jitter"])

    basic_config = aug_config.get("basic", {})
    if isinstance(basic_config, dict) and isinstance(
        basic_config.get("color_jitter"),
        dict,
    ):
        return dict(basic_config["color_jitter"])

    return {}


@dataclass
class IdentityAugmentation:
    """No-op augmentation."""

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        return _ensure_rgb(image), mask


@dataclass
class ColorJitterAugmentation:
    """Color jitter for semantic segmentation.

    Only the RGB image is changed. The segmentation mask is returned unchanged.

    Parameters follow torchvision-style ranges:
        brightness=0.2 means factor sampled from [0.8, 1.2]
        contrast=0.2 means factor sampled from [0.8, 1.2]
        saturation=0.1 means factor sampled from [0.9, 1.1]
        hue=0.05 means hue shift sampled from [-0.05, 0.05]
    """

    prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hue: float = 0.05

    def __post_init__(self) -> None:
        self.prob = float(self.prob)
        self.brightness = float(self.brightness)
        self.contrast = float(self.contrast)
        self.saturation = float(self.saturation)
        self.hue = float(self.hue)

        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")
        if self.brightness < 0:
            raise ValueError(f"brightness must be non-negative, got {self.brightness}")
        if self.contrast < 0:
            raise ValueError(f"contrast must be non-negative, got {self.contrast}")
        if self.saturation < 0:
            raise ValueError(f"saturation must be non-negative, got {self.saturation}")
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {self.hue}")

    @staticmethod
    def _sample_factor(amount: float) -> float:
        if amount <= 0:
            return 1.0
        low = max(0.0, 1.0 - amount)
        high = 1.0 + amount
        return random.uniform(low, high)

    def _apply_hue(self, image: Image.Image) -> Image.Image:
        if self.hue <= 0:
            return image

        hsv = np.array(image.convert("HSV"), dtype=np.uint8)
        shift = int(round(random.uniform(-self.hue, self.hue) * 255.0))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = _ensure_rgb(image)

        if random.random() > self.prob:
            return image, mask

        transforms = []

        if self.brightness > 0:
            transforms.append(
                lambda img: ImageEnhance.Brightness(img).enhance(
                    self._sample_factor(self.brightness)
                )
            )
        if self.contrast > 0:
            transforms.append(
                lambda img: ImageEnhance.Contrast(img).enhance(
                    self._sample_factor(self.contrast)
                )
            )
        if self.saturation > 0:
            transforms.append(
                lambda img: ImageEnhance.Color(img).enhance(
                    self._sample_factor(self.saturation)
                )
            )
        if self.hue > 0:
            transforms.append(self._apply_hue)

        random.shuffle(transforms)

        for transform in transforms:
            image = transform(image)

        return _ensure_rgb(image), mask


def build_augmentation(
    config: dict[str, Any],
    split: str = "train",
) -> IdentityAugmentation | ColorJitterAugmentation:
    """Build train-time basic augmentation.

    Current main-branch policy:
        - Basic augmentation supports only color jitter.
        - Weather-specific augmentation is handled separately by weather_augmentation.py.
        - Augmentation is disabled for val/test.
    """
    if str(split).lower() != "train":
        return IdentityAugmentation()

    aug_config = config.get("augmentation", {})
    if not isinstance(aug_config, dict):
        return IdentityAugmentation()

    enabled = bool(aug_config.get("enabled", False))
    name = _normalize_name(aug_config.get("name", "none"))

    if not enabled or name in {"none", "identity", "off", "null", ""}:
        return IdentityAugmentation()

    color_jitter_config = _get_color_jitter_config(config)
    color_jitter_enabled = bool(
        color_jitter_config.get("enabled", name in {"jitter", "color_jitter"})
    )

    # weather augmentation은 transform.py에서 별도 build_weather_augmentation으로 처리한다.
    # 따라서 final.yaml처럼 weather.enabled=true인 config에서 color_jitter가 꺼져 있거나
    # augmentation.name이 weather 계열 이름이어도 basic augmentation 단계에서는 no-op으로 넘긴다.
    weather_config = aug_config.get("weather", {})
    weather_enabled = isinstance(weather_config, dict) and bool(
        weather_config.get("enabled", False)
    )

    if weather_enabled and not color_jitter_enabled:
        return IdentityAugmentation()

    if name in {"jitter", "color_jitter"} or color_jitter_enabled:
        if not color_jitter_enabled:
            return IdentityAugmentation()

        return ColorJitterAugmentation(
            prob=float(color_jitter_config.get("prob", aug_config.get("prob", 0.5))),
            brightness=float(color_jitter_config.get("brightness", 0.2)),
            contrast=float(color_jitter_config.get("contrast", 0.2)),
            saturation=float(color_jitter_config.get("saturation", 0.1)),
            hue=float(color_jitter_config.get("hue", 0.05)),
        )

    raise ValueError(
        f"Unsupported augmentation name: {name!r}. "
        "Current main branch supports basic augmentation only for none, jitter/color_jitter. "
        "Weather-specific augmentation should be configured under augmentation.weather."
    )
