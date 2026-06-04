from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass Focal Loss.

    FL = (1 - pt)^gamma * CE
    """

    def __init__(
        self,
        ignore_index: int = 255,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.gamma = float(gamma)
        self.reduction = str(reduction)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4:
            raise ValueError(f"pred must have shape [B, C, H, W], got {tuple(pred.shape)}")
        if target.ndim != 3:
            raise ValueError(f"target must have shape [B, H, W], got {tuple(target.shape)}")

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = target.long()
        valid_mask = target != self.ignore_index

        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=pred.device, dtype=pred.dtype)

        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss[valid_mask]

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
