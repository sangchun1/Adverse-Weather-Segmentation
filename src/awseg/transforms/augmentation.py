from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image


@dataclass
class IdentityAugmentation:
    """No-op augmentation.

    실제 augmentation 구현 전까지 코드가 정상 실행되도록 하는 placeholder.
    """
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        condition: Optional[str] = None,
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        return image, mask


def build_augmentation(
    config: dict[str, Any],
    split: str = "train",
) -> IdentityAugmentation:
    """Build basic augmentation.

    현재는 placeholder만 반환한다.
    추후 HorizontalFlip, ColorJitter, ClassBalancedCrop 등을 여기서 연결하면 된다.
    """
    return IdentityAugmentation()