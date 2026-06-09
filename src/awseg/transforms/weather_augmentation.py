from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_condition(value: Any) -> Optional[str]:
    if value is None:
        return None
    condition = _normalize_name(value)
    if condition in {"", "none", "nan", "null", "unknown"}:
        return None
    return condition


def _sample_range(value: Any, default: tuple[float, float]) -> float:
    if value is None:
        low, high = default
    elif isinstance(value, (int, float)):
        low = high = float(value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        raise ValueError(f"Expected number or [low, high] range, got {value!r}")

    if high < low:
        raise ValueError(f"Range high must be >= low, got [{low}, {high}]")
    return random.uniform(low, high)


def _sample_int_range(value: Any, default: tuple[int, int]) -> int:
    sampled = _sample_range(value, (float(default[0]), float(default[1])))
    return int(round(sampled))


def _pil_to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(_ensure_rgb(image), dtype=np.float32)


def _array_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


@dataclass
class _ProbabilisticAugmentation:
    prob: float = 0.3

    def __post_init__(self) -> None:
        self.prob = float(self.prob)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")

    def _should_apply(self) -> bool:
        return random.random() <= self.prob


@dataclass
class IdentityWeatherAugmentation:
    """No-op weather augmentation.

    실제 weather-specific augmentation 구현 전까지 코드가 정상 실행되도록 하는 placeholder.
    """

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        return _ensure_rgb(image), mask


@dataclass
class FogAugmentation(_ProbabilisticAugmentation):
    """Single fog-specific augmentation.

    Supported effects:
        - haze: blend image with bright atmospheric light.
        - contrast: reduce global contrast.
    """

    effect: str = "haze"
    intensity_range: tuple[float, float] = (0.15, 0.4)
    atmospheric_light_range: tuple[float, float] = (180.0, 255.0)
    contrast_factor_range: tuple[float, float] = (0.65, 0.95)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = _normalize_name(self.effect)
        if self.effect not in {"haze", "contrast", "contrast_reduction"}:
            raise ValueError(f"Unsupported fog effect: {self.effect!r}")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = _ensure_rgb(image)
        if not self._should_apply():
            return image, mask

        if self.effect == "haze":
            intensity = _sample_range(self.intensity_range, (0.15, 0.4))
            atmospheric_light = _sample_range(
                self.atmospheric_light_range,
                (180.0, 255.0),
            )
            array = _pil_to_float_array(image)
            fogged = array * (1.0 - intensity) + atmospheric_light * intensity
            return _array_to_pil(fogged), mask

        factor = _sample_range(self.contrast_factor_range, (0.65, 0.95))
        return ImageEnhance.Contrast(image).enhance(factor), mask


@dataclass
class RainAugmentation(_ProbabilisticAugmentation):
    """Single rain-specific augmentation.

    Supported effects:
        - streak: draw diagonal rain streaks.
        - blur: apply Gaussian blur.
    """

    effect: str = "streak"
    streak_count_range: tuple[int, int] = (150, 600)
    streak_length_range: tuple[int, int] = (8, 24)
    streak_angle_range: tuple[float, float] = (-25.0, 25.0)
    alpha_range: tuple[float, float] = (0.1, 0.3)
    blur_radius_range: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = _normalize_name(self.effect)
        if self.effect not in {"streak", "rain_streak", "blur"}:
            raise ValueError(f"Unsupported rain effect: {self.effect!r}")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = _ensure_rgb(image)
        if not self._should_apply():
            return image, mask

        if self.effect == "blur":
            radius = _sample_range(self.blur_radius_range, (0.5, 1.5))
            return image.filter(ImageFilter.GaussianBlur(radius=radius)), mask

        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        count = _sample_int_range(self.streak_count_range, (150, 600))
        length = _sample_int_range(self.streak_length_range, (8, 24))
        angle = math.radians(_sample_range(self.streak_angle_range, (-25.0, 25.0)))
        alpha = int(round(_sample_range(self.alpha_range, (0.1, 0.3)) * 255))
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length

        for _ in range(max(0, count)):
            x = random.randint(0, max(0, width - 1))
            y = random.randint(0, max(0, height - 1))
            draw.line((x, y, x + dx, y + dy), fill=(220, 220, 220, alpha), width=1)

        rainy = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        return rainy, mask


@dataclass
class SnowAugmentation(_ProbabilisticAugmentation):
    """Single snow-specific augmentation.

    Supported effects:
        - particle: draw white snow particles.
        - whiteness: blend image with white.
    """

    effect: str = "particle"
    snowflake_count_range: tuple[int, int] = (300, 1200)
    snowflake_radius_range: tuple[int, int] = (1, 3)
    alpha_range: tuple[float, float] = (0.15, 0.35)
    whiteness_alpha_range: tuple[float, float] = (0.05, 0.18)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = _normalize_name(self.effect)
        if self.effect not in {"particle", "snow_particle", "whiteness"}:
            raise ValueError(f"Unsupported snow effect: {self.effect!r}")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = _ensure_rgb(image)
        if not self._should_apply():
            return image, mask

        if self.effect == "whiteness":
            alpha = _sample_range(self.whiteness_alpha_range, (0.05, 0.18))
            array = _pil_to_float_array(image)
            whitened = array * (1.0 - alpha) + 255.0 * alpha
            return _array_to_pil(whitened), mask

        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        count = _sample_int_range(self.snowflake_count_range, (300, 1200))
        radius = _sample_int_range(self.snowflake_radius_range, (1, 3))
        alpha = int(round(_sample_range(self.alpha_range, (0.15, 0.35)) * 255))

        for _ in range(max(0, count)):
            x = random.randint(0, max(0, width - 1))
            y = random.randint(0, max(0, height - 1))
            r = max(1, radius)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, alpha))

        snowy = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        return snowy, mask


@dataclass
class NightAugmentation(_ProbabilisticAugmentation):
    """Single night-specific augmentation.

    Supported effects:
        - darkening: reduce brightness.
        - noise: add sensor-like Gaussian noise.
    """

    effect: str = "darkening"
    brightness_factor_range: tuple[float, float] = (0.4, 0.85)
    noise_std_range: tuple[float, float] = (0.005, 0.03)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = _normalize_name(self.effect)
        if self.effect not in {"darkening", "noise"}:
            raise ValueError(f"Unsupported night effect: {self.effect!r}")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        image = _ensure_rgb(image)
        if not self._should_apply():
            return image, mask

        if self.effect == "darkening":
            factor = _sample_range(self.brightness_factor_range, (0.4, 0.85))
            return ImageEnhance.Brightness(image).enhance(factor), mask

        if self.effect == "noise":
            std = _sample_range(self.noise_std_range, (0.005, 0.03)) * 255.0
            array = _pil_to_float_array(image)
            noisy = array + np.random.normal(0.0, std, size=array.shape)
            return _array_to_pil(noisy), mask


@dataclass
class TargetConditionWeatherAugmentation:
    """Apply one weather augmentation only to its target condition."""

    target: str
    augmentation: Any

    def __post_init__(self) -> None:
        self.target = _normalize_name(self.target)

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        if _normalize_condition(condition) != self.target:
            return _ensure_rgb(image), mask
        return self.augmentation(image, mask, condition=condition)


@dataclass
class ConditionSpecificWeatherAugmentation:
    """Apply the augmentation matching each sample's weather condition."""

    augmentations: dict[str, Any]

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        normalized_condition = _normalize_condition(condition)
        augmentation = self.augmentations.get(normalized_condition or "")
        if augmentation is None:
            return _ensure_rgb(image), mask
        return augmentation(image, mask, condition=condition)


def _get_weather_config(config: dict[str, Any]) -> dict[str, Any]:
    augmentation_config = config.get("augmentation", {})
    if not isinstance(augmentation_config, dict):
        return {}
    weather_config = augmentation_config.get("weather", {})
    if not isinstance(weather_config, dict):
        return {}
    return weather_config


def _get_condition_config(weather_config: dict[str, Any], condition: str) -> dict[str, Any]:
    condition_config = weather_config.get(condition, {})
    if not isinstance(condition_config, dict):
        return {}
    return condition_config


def _is_condition_enabled(weather_config: dict[str, Any], condition: str) -> bool:
    condition_config = _get_condition_config(weather_config, condition)
    return bool(condition_config.get("enabled", True))


def _get_effect_config(
    condition_config: dict[str, Any],
    effect: str,
) -> dict[str, Any]:
    effect_config = condition_config.get(effect, {})
    if isinstance(effect_config, dict):
        return effect_config
    return {}


def _get_config_value(
    effect_config: dict[str, Any],
    condition_config: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in effect_config:
        return effect_config[key]
    return condition_config.get(key, default)


def _build_single_augmentation(
    condition: str,
    weather_config: dict[str, Any],
    effect: Optional[str] = None,
) -> Any:
    condition = _normalize_name(condition)
    condition_config = _get_condition_config(weather_config, condition)
    selected_effect = _normalize_name(
        effect or condition_config.get("effect") or weather_config.get("effect") or ""
    )
    effect_config = _get_effect_config(condition_config, selected_effect)
    prob = float(
        effect_config.get(
            "prob",
            condition_config.get("prob", weather_config.get("prob", 0.3)),
        )
    )

    if condition == "fog":
        return FogAugmentation(
            prob=prob,
            effect=selected_effect or "haze",
            intensity_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "intensity_range",
                    (0.15, 0.4),
                )
            ),
            atmospheric_light_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "atmospheric_light_range",
                    (180.0, 255.0),
                )
            ),
            contrast_factor_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "contrast_factor_range",
                    (0.65, 0.95),
                )
            ),
        )

    if condition == "rain":
        return RainAugmentation(
            prob=prob,
            effect=selected_effect or "streak",
            streak_count_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "streak_count_range",
                    (150, 600),
                )
            ),
            streak_length_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "streak_length_range",
                    (8, 24),
                )
            ),
            streak_angle_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "streak_angle_range",
                    (-25.0, 25.0),
                )
            ),
            alpha_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "alpha_range",
                    (0.1, 0.3),
                )
            ),
            blur_radius_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "blur_radius_range",
                    (0.5, 1.5),
                )
            ),
        )

    if condition == "snow":
        return SnowAugmentation(
            prob=prob,
            effect=selected_effect or "particle",
            snowflake_count_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "snowflake_count_range",
                    (300, 1200),
                )
            ),
            snowflake_radius_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "snowflake_radius_range",
                    (1, 3),
                )
            ),
            alpha_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "alpha_range",
                    (0.15, 0.35),
                )
            ),
            whiteness_alpha_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "whiteness_alpha_range",
                    (0.05, 0.18),
                )
            ),
        )

    if condition == "night":
        return NightAugmentation(
            prob=prob,
            effect=selected_effect or "darkening",
            brightness_factor_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "brightness_factor_range",
                    (0.4, 0.85),
                )
            ),
            noise_std_range=tuple(
                _get_config_value(
                    effect_config,
                    condition_config,
                    "noise_std_range",
                    (0.005, 0.03),
                )
            ),
        )

    raise ValueError(f"Unsupported weather target: {condition!r}")


def build_weather_augmentation(
    config: dict[str, Any],
    split: str = "train",
) -> Any:
    """Build weather-specific augmentation.

    Supported modes:
        target_condition:
            Apply one effect only when sample condition matches weather.target.
        condition_specific:
            Apply each enabled condition's configured single effect to matching samples.
    """
    if str(split).lower() != "train":
        return IdentityWeatherAugmentation()

    weather_config = _get_weather_config(config)
    if not bool(weather_config.get("enabled", False)):
        return IdentityWeatherAugmentation()

    mode = _normalize_name(weather_config.get("mode", "target_condition"))

    if mode in {"target", "target_condition", "single"}:
        target = _normalize_condition(weather_config.get("target"))
        if target is None:
            raise ValueError("weather.target is required for target_condition mode.")
        if not _is_condition_enabled(weather_config, target):
            return IdentityWeatherAugmentation()
        augmentation = _build_single_augmentation(
            condition=target,
            weather_config=weather_config,
            effect=weather_config.get("effect"),
        )
        return TargetConditionWeatherAugmentation(
            target=target,
            augmentation=augmentation,
        )

    if mode in {"condition_specific", "per_condition"}:
        augmentations = {}
        for condition in ("fog", "rain", "snow", "night"):
            if _is_condition_enabled(weather_config, condition):
                augmentations[condition] = _build_single_augmentation(
                    condition=condition,
                    weather_config=weather_config,
                )
        if not augmentations:
            return IdentityWeatherAugmentation()
        return ConditionSpecificWeatherAugmentation(augmentations=augmentations)

    raise ValueError(
        f"Unsupported weather augmentation mode: {mode!r}. "
        "Supported modes: target_condition, condition_specific."
    )
