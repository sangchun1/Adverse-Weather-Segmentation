from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss wrapper for semantic segmentation."""

    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=pred.device, dtype=pred.dtype)

        return F.cross_entropy(
            pred,
            target.long(),
            weight=weight,
            ignore_index=self.ignore_index,
        )
