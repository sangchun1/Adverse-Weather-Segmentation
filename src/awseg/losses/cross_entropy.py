from __future__ import annotations

import torch.nn as nn


def build_cross_entropy_loss(ignore_index: int = 255) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss for semantic segmentation.

    Args:
        ignore_index: Label value ignored when computing loss.
            For ACDC/Cityscapes trainIds, 255 is commonly used as ignore index.

    Returns:
        torch.nn.CrossEntropyLoss with ignore_index.
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
