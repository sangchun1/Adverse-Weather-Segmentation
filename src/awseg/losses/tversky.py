from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Multiclass Tversky Loss.

    Tversky index = TP / (TP + alpha * FP + beta * FN)
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.eps = float(eps)

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative.")

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

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        prob = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target_safe, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()

        prob = prob * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)

        tp = (prob * target_one_hot).sum(dim=dims)
        fp = (prob * (1.0 - target_one_hot) * valid_mask).sum(dim=dims)
        fn = ((1.0 - prob) * target_one_hot).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps
        )

        loss = 1.0 - tversky

        valid_classes = target_one_hot.sum(dim=dims) > 0
        if valid_classes.sum() == 0:
            return pred.sum() * 0.0

        return loss[valid_classes].mean()
