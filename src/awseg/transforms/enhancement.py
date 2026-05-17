from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import cv2
import numpy as np
from PIL import Image


class Enhancer(Protocol):
    """PIL RGB 이미지를 입력받아 enhancement된 PIL RGB 이미지를 반환하는 인터페이스."""

    def __call__(self, image: Image.Image) -> Image.Image:
        ...


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """입력 이미지를 RGB PIL 이미지로 통일한다."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """PIL RGB 이미지를 uint8 RGB numpy 배열로 변환한다."""
    image = _ensure_rgb(image)
    array = np.asarray(image, dtype=np.uint8)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {array.shape}")

    return array


def _rgb_array_to_pil(array: np.ndarray) -> Image.Image:
    """uint8 RGB numpy 배열을 PIL RGB 이미지로 변환한다."""
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@dataclass
class IdentityEnhancer:
    """아무 enhancement도 적용하지 않는 기본 enhancer."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return _ensure_rgb(image)


@dataclass
class GammaCorrection:
    """야간 이미지 밝기 보정을 위한 gamma correction.

    gamma < 1.0이면 이미지가 밝아지고,
    gamma > 1.0이면 이미지가 어두워진다.

    night 실험에서는 0.4~0.8 범위를 우선 추천한다.
    """

    gamma: float = 0.6

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = _pil_to_rgb_array(image).astype(np.float32) / 255.0
        enhanced = np.power(rgb, self.gamma) * 255.0
        return _rgb_array_to_pil(enhanced)


@dataclass
class CLAHEEnhancer:
    """밝기 채널에만 CLAHE를 적용하는 local contrast enhancement.

    RGB 전체에 histogram equalization을 적용하면 색이 많이 변할 수 있으므로,
    기본값은 LAB 색공간의 L 채널에만 CLAHE를 적용한다.
    """

    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    color_space: str = "lab"

    def __post_init__(self) -> None:
        if self.clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {self.clip_limit}")

        if len(self.tile_grid_size) != 2:
            raise ValueError("tile_grid_size must be a tuple of length 2")

        if self.tile_grid_size[0] <= 0 or self.tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size must be positive, got {self.tile_grid_size}")

        self.color_space = self.color_space.lower()

        if self.color_space not in {"lab", "hsv", "ycrcb"}:
            raise ValueError(
                "color_space must be one of {'lab', 'hsv', 'ycrcb'}, "
                f"got {self.color_space!r}"
            )

    def __call__(self, image: Image.Image) -> Image.Image:
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
    """Gamma correction 후 CLAHE를 적용하는 조합 실험용 enhancer.

    단일 gamma / CLAHE보다 강한 보정이 필요할 때 사용한다.
    단, noise나 halo가 커질 수 있어서 후순위 ablation으로 권장한다.
    """

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

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self.gamma_enhancer(image)
        image = self.clahe_enhancer(image)
        return image


def _get_enhancement_config(config: dict[str, Any]) -> dict[str, Any]:
    """config에서 enhancement 설정 block을 가져온다.

    train.enhancement 구조와 전역 enhancement 구조를 둘 다 지원한다.
    """

    if "enhancement" in config:
        return dict(config.get("enhancement") or {})

    train_config = config.get("train", {})

    if isinstance(train_config, dict) and "enhancement" in train_config:
        return dict(train_config.get("enhancement") or {})

    return {"enabled": False, "name": "none"}


def _parse_tile_grid_size(value: Any) -> tuple[int, int]:
    """YAML list/int/tuple을 CLAHE tileGridSize tuple로 변환한다."""

    if value is None:
        return (8, 8)

    if isinstance(value, int):
        return (value, value)

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))

    raise ValueError(f"Invalid tile_grid_size: {value!r}")


def build_enhancer(config: dict[str, Any], split: Optional[str] = None) -> Enhancer:
    """config에 맞는 image enhancement 객체를 생성한다.

    Args:
        config: 전체 experiment config dictionary.
        split: train / val / test 중 하나.
            apply_to가 설정된 경우 split별 적용 여부를 결정한다.

    Returns:
        PIL image를 입력받아 PIL image를 반환하는 enhancer.
    """

    enhancement_config = _get_enhancement_config(config)

    enabled = bool(enhancement_config.get("enabled", False))
    name = str(enhancement_config.get("name", "none")).lower().replace("-", "_")

    if not enabled or name in {"none", "identity", "off", "null", ""}:
        return IdentityEnhancer()

    apply_to = enhancement_config.get("apply_to", None)

    if split is not None and apply_to is not None:
        if isinstance(apply_to, str):
            apply_to = [apply_to]

        apply_to = {str(item).lower() for item in apply_to}

        if split.lower() not in apply_to:
            return IdentityEnhancer()

    if name in {"gamma", "gamma_correction"}:
        return GammaCorrection(
            gamma=float(enhancement_config.get("gamma", 0.6)),
        )

    if name in {"clahe", "lab_clahe", "luminance_clahe"}:
        return CLAHEEnhancer(
            clip_limit=float(enhancement_config.get("clip_limit", 2.0)),
            tile_grid_size=_parse_tile_grid_size(
                enhancement_config.get("tile_grid_size", [8, 8])
            ),
            color_space=str(enhancement_config.get("color_space", "lab")),
        )

    if name in {"gamma_clahe", "gamma+clahe"}:
        return GammaCLAHEEnhancer(
            gamma=float(enhancement_config.get("gamma", 0.7)),
            clip_limit=float(enhancement_config.get("clip_limit", 2.0)),
            tile_grid_size=_parse_tile_grid_size(
                enhancement_config.get("tile_grid_size", [8, 8])
            ),
            color_space=str(enhancement_config.get("color_space", "lab")),
        )

    if name in {"sci", "zero_dce", "zerodce", "retinexformer"}:
        raise ValueError(
            f"enhancement.name={name!r} is an offline enhancement method. "
            "Do not run SCI / Zero-DCE / Retinexformer inside enhancement.py. "
            "Use scripts/preprocess_enhancement.sh first, then train with "
            "enhancement.enabled=false and image paths pointing to data/enhanced/."
        )

    raise ValueError(
        f"Unknown enhancement method: {name!r}. "
        "Supported on-the-fly methods: none, gamma, clahe, gamma_clahe. "
        "Offline methods: sci, zero_dce, retinexformer."
    )