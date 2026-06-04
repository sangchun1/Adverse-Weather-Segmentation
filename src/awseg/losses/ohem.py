from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining CrossEntropyLoss.

    Keeps the top fraction of pixel-level CE losses.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        fraction: float = 0.2,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.fraction = float(fraction)

        if self.fraction <= 0 or self.fraction > 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=pred.device, dtype=pred.dtype)

        loss = F.cross_entropy(
            pred,
            target.long(),
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        loss = loss.view(-1)
        target_flat = target.view(-1)
        valid_mask = target_flat != self.ignore_index

        loss = loss[valid_mask]

        if loss.numel() == 0:
            return pred.sum() * 0.0

        num_hard = max(1, int(loss.numel() * self.fraction))
        hard_loss, _ = torch.topk(loss, k=num_hard)

        return hard_loss.mean()
