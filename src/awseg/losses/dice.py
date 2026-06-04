from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multiclass Dice Loss.

    This implementation does not require segmentation_models_pytorch.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)
        self.eps = float(eps)

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
        intersection = (prob * target_one_hot).sum(dim=dims)
        cardinality = prob.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth + self.eps
        )
        loss = 1.0 - dice

        valid_classes = target_one_hot.sum(dim=dims) > 0
        if valid_classes.sum() == 0:
            return pred.sum() * 0.0

        return loss[valid_classes].mean()
