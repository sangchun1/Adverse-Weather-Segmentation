from __future__ import annotations

import torch
import torch.nn as nn

from awseg.losses.cross_entropy import CrossEntropyLoss
from awseg.losses.dice import DiceLoss
from awseg.losses.focal import FocalLoss
from awseg.losses.lovasz import LovaszSoftmaxLoss
from awseg.losses.ohem import OHEMCrossEntropyLoss
from awseg.losses.tversky import TverskyLoss


class UniversalHybridLoss(nn.Module):
    """Universal hybrid loss matching the fog branch combinations.

    Supported loss_name:
        - cross_entropy
        - dice
        - ce_dice
        - tversky
        - ce_tversky
        - focal
        - focal_tversky
        - ce_lovasz
        - ohem_tversky
        - tversky_lovasz
        - ce_focal_dice
        - ce_focal_tversky

    Important fog-branch compatibility:
        ce_tversky = ce_weight * CE + dice_weight * Tversky
        focal_tversky = ce_weight * Focal + dice_weight * Tversky
        ce_lovasz = ce_weight * CE + dice_weight * Lovasz
        ohem_tversky = ce_weight * OHEM + dice_weight * Tversky
        ce_focal_tversky = ce_weight * CE + Focal + dice_weight * Tversky
    """

    def __init__(
        self,
        loss_name: str,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        ohem_fraction: float = 0.2,
        gamma: float = 2.0,
        tversky_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.loss_name = str(loss_name).lower()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.lovasz_weight = float(lovasz_weight)

        self.ce_loss = CrossEntropyLoss(
            ignore_index=ignore_index,
            class_weights=class_weights,
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=smooth,
        )
        self.focal_loss = FocalLoss(
            ignore_index=ignore_index,
            gamma=gamma,
            class_weights=class_weights,
        )
        self.tversky_loss = TverskyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            smooth=smooth,
        )
        self.lovasz_loss = LovaszSoftmaxLoss(
            ignore_index=ignore_index,
        )
        self.ohem_loss = OHEMCrossEntropyLoss(
            ignore_index=ignore_index,
            fraction=ohem_fraction,
            class_weights=class_weights,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name in {"cross_entropy", "ce"}:
            return self.ce_loss(pred, target)

        if self.loss_name == "dice":
            return self.dice_loss(pred, target)

        if self.loss_name == "ce_dice":
            return (self.ce_weight * self.ce_loss(pred, target)) + (
                self.dice_weight * self.dice_loss(pred, target)
            )

        if self.loss_name == "tversky":
            return self.tversky_loss(pred, target)

        if self.loss_name == "ce_tversky":
            return (self.ce_weight * self.ce_loss(pred, target)) + (
                self.dice_weight * self.tversky_loss(pred, target)
            )

        if self.loss_name == "focal":
            return self.focal_loss(pred, target)

        if self.loss_name == "focal_tversky":
            return (self.ce_weight * self.focal_loss(pred, target)) + (
                self.dice_weight * self.tversky_loss(pred, target)
            )

        if self.loss_name == "ce_lovasz":
            return (self.ce_weight * self.ce_loss(pred, target)) + (
                self.dice_weight * self.lovasz_loss(pred, target)
            )

        if self.loss_name == "ohem_tversky":
            return (self.ce_weight * self.ohem_loss(pred, target)) + (
                self.dice_weight * self.tversky_loss(pred, target)
            )

        if self.loss_name == "tversky_lovasz":
            return (self.tversky_weight * self.tversky_loss(pred, target)) + (
                self.lovasz_weight * self.lovasz_loss(pred, target)
            )

        if self.loss_name == "ce_focal_dice":
            ce_val = self.ce_loss(pred, target)
            focal_val = self.focal_loss(pred, target)
            dice_val = self.dice_loss(pred, target)

            return (self.ce_weight * ce_val) + focal_val + (self.dice_weight * dice_val)

        if self.loss_name == "ce_focal_tversky":
            ce_val = self.ce_loss(pred, target)
            focal_val = self.focal_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)

            return (self.ce_weight * ce_val) + focal_val + (
                self.dice_weight * tversky_val
            )

        raise ValueError(f"Unsupported loss combination: {self.loss_name}")
