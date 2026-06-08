from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import cv2
import numpy as np
from PIL import Image


class Enhancer(Protocol):
    """PIL RGB image enhancer interface."""

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        ...


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert input image to RGB PIL image."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to uint8 RGB numpy array."""
    image = _ensure_rgb(image)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {array.shape}")
    return array


def _rgb_array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert RGB numpy array to PIL RGB image."""
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_condition(condition: Optional[str]) -> Optional[str]:
    if condition is None:
        return None
    condition = str(condition).strip().lower()
    return condition or None


def _as_lower_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalize_name(value)}
    if isinstance(value, (list, tuple, set)):
        return {_normalize_name(item) for item in value if str(item).strip()}
    raise ValueError(f"Expected string or list-like value, got {value!r}")


def _should_apply_to_condition(condition: Optional[str], apply_conditions: Any) -> bool:
    """Return whether enhancement should be applied to this weather condition.

    Supported examples:
        apply_conditions: "all"
        apply_conditions: ["night"]
        apply_conditions: ["night", "fog"]
        apply_conditions: null
    """
    allowed_conditions = _as_lower_set(apply_conditions)
    if not allowed_conditions or allowed_conditions & {"all", "any", "*"}:
        return True

    normalized_condition = _normalize_condition(condition)
    return normalized_condition in allowed_conditions


def _should_apply_to_split(split: Optional[str], apply_to: Any) -> bool:
    """Return whether enhancement should be enabled for this dataset split."""
    allowed_splits = _as_lower_set(apply_to)
    if split is None or not allowed_splits or allowed_splits & {"all", "any", "*"}:
        return True

    return _normalize_name(split) in allowed_splits


def _parse_tile_grid_size(value: Any) -> tuple[int, int]:
    """Convert YAML int/list/tuple to OpenCV CLAHE tileGridSize tuple."""
    if value is None:
        return (8, 8)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid tile_grid_size: {value!r}")


def _get_enhancement_config(config: dict[str, Any]) -> dict[str, Any]:
    """Get enhancement config block.

    Preferred:
        enhancement: {...}

    Legacy-compatible:
        train:
          enhancement: {...}
    """
    if "enhancement" in config:
        return dict(config.get("enhancement") or {})

    train_config = config.get("train", {})
    if isinstance(train_config, dict) and "enhancement" in train_config:
        return dict(train_config.get("enhancement") or {})

    return {"enabled": False, "name": "none"}


@dataclass
class IdentityEnhancer:
    """No-op enhancer."""

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        return _ensure_rgb(image).copy()


@dataclass
class GammaCorrection:
    """Gamma correction.

    For normalized pixel x in [0, 1], output is x ** gamma.
    gamma < 1.0 brightens images, gamma > 1.0 darkens images.
    """

    gamma: float = 0.6

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        rgb = _pil_to_rgb_array(image).astype(np.float32) / 255.0
        enhanced = np.power(rgb, self.gamma) * 255.0
        return _rgb_array_to_pil(enhanced)


@dataclass
class CLAHEEnhancer:
    """Local contrast enhancement using CLAHE on a luminance-like channel."""

    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    color_space: str = "lab"

    def __post_init__(self) -> None:
        if self.clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {self.clip_limit}")

        if len(self.tile_grid_size) != 2:
            raise ValueError("tile_grid_size must have length 2")

        if self.tile_grid_size[0] <= 0 or self.tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size must be positive, got {self.tile_grid_size}")

        self.color_space = self.color_space.strip().lower()
        if self.color_space not in {"lab", "hsv", "ycrcb"}:
            raise ValueError(
                "color_space must be one of {'lab', 'hsv', 'ycrcb'}, "
                f"got {self.color_space!r}"
            )

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        rgb = _pil_to_rgb_array(image)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=tuple(self.tile_grid_size),
        )

        if self.color_space == "lab":
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.merge((l_channel, a_channel, b_channel))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            return _rgb_array_to_pil(enhanced)

        if self.color_space == "hsv":
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)
            v_channel = clahe.apply(v_channel)
            enhanced = cv2.merge((h_channel, s_channel, v_channel))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
            return _rgb_array_to_pil(enhanced)

        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_channel = clahe.apply(y_channel)
        enhanced = cv2.merge((y_channel, cr_channel, cb_channel))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2RGB)
        return _rgb_array_to_pil(enhanced)


@dataclass
class GammaCLAHEEnhancer:
    """Apply gamma correction first, then CLAHE."""

    gamma: float = 0.7
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    color_space: str = "lab"

    def __post_init__(self) -> None:
        self.gamma_enhancer = GammaCorrection(gamma=self.gamma)
        self.clahe_enhancer = CLAHEEnhancer(
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            color_space=self.color_space,
        )

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        image = self.gamma_enhancer(image, condition=condition)
        image = self.clahe_enhancer(image, condition=condition)
        return image


@dataclass
class ConditionalEnhancer:
    """Condition-aware wrapper for an enhancer."""

    enhancer: Enhancer
    apply_conditions: Any = "all"

    def __call__(
        self,
        image: Image.Image,
        condition: Optional[str] = None,
    ) -> Image.Image:
        if _should_apply_to_condition(
            condition=condition,
            apply_conditions=self.apply_conditions,
        ):
            return self.enhancer(image, condition=condition)

        return _ensure_rgb(image).copy()


def build_enhancer(config: dict[str, Any], split: Optional[str] = None) -> Enhancer:
    """Build image enhancer from config.

    Example:
        enhancement:
          enabled: true
          name: gamma_clahe
          apply_to: [train, val, test]
          apply_conditions: all
          gamma: 0.7
          clip_limit: 2.0
          tile_grid_size: [8, 8]
          color_space: lab
    """
    enhancement_config = _get_enhancement_config(config)
    enabled = bool(enhancement_config.get("enabled", False))
    name = _normalize_name(enhancement_config.get("name", "none"))

    if not enabled or name in {"none", "identity", "off", "null", ""}:
        return IdentityEnhancer()

    apply_to = enhancement_config.get("apply_to", None)
    if not _should_apply_to_split(split=split, apply_to=apply_to):
        return IdentityEnhancer()

    if name in {"gamma", "gamma_correction"}:
        enhancer: Enhancer = GammaCorrection(
            gamma=float(enhancement_config.get("gamma", 0.6)),
        )
    elif name in {"clahe", "lab_clahe", "luminance_clahe"}:
        enhancer = CLAHEEnhancer(
            clip_limit=float(enhancement_config.get("clip_limit", 2.0)),
            tile_grid_size=_parse_tile_grid_size(
                enhancement_config.get("tile_grid_size", [8, 8])
            ),
            color_space=str(enhancement_config.get("color_space", "lab")),
        )
    elif name in {"gamma_clahe", "gamma+clahe"}:
        enhancer = GammaCLAHEEnhancer(
            gamma=float(enhancement_config.get("gamma", 0.7)),
            clip_limit=float(enhancement_config.get("clip_limit", 2.0)),
            tile_grid_size=_parse_tile_grid_size(
                enhancement_config.get("tile_grid_size", [8, 8])
            ),
            color_space=str(enhancement_config.get("color_space", "lab")),
        )
    else:
        raise ValueError(
            f"Unknown enhancement method: {name!r}. "
            "Supported methods: none, gamma, clahe, gamma_clahe."
        )

    return ConditionalEnhancer(
        enhancer=enhancer,
        apply_conditions=enhancement_config.get("apply_conditions", "all"),
    )


def apply_enhancement(
    image: Image.Image,
    config: dict[str, Any],
    split: Optional[str] = None,
    condition: Optional[str] = None,
) -> Image.Image:
    """Functional API for applying configured enhancement to one image."""
    enhancer = build_enhancer(config=config, split=split)
    return enhancer(image, condition=condition)
