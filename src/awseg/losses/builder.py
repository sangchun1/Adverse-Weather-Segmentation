from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from awseg.losses.hybrid import UniversalHybridLoss


SUPPORTED_LOSSES = {
    "cross_entropy",
    "ce",
    "dice",
    "ce_dice",
    "tversky",
    "ce_tversky",
    "focal",
    "focal_tversky",
    "ce_lovasz",
    "ohem_tversky",
    "tversky_lovasz",
    "ce_focal_dice",
    "ce_focal_tversky",
}


def _build_class_weights(
    loss_config: dict[str, Any],
    num_classes: int,
) -> torch.Tensor | None:
    raw_weights = (
        loss_config.get("weights", None)
        or loss_config.get("class_weights", None)
        or loss_config.get("class_weight", None)
    )

    if raw_weights is None:
        return None

    if len(raw_weights) != num_classes:
        raise ValueError(
            f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})."
        )

    return torch.tensor(raw_weights, dtype=torch.float32)


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function from config.

    This builder reflects the fog branch loss menu while keeping the loss
    implementations split into separate files.
    """
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})

    loss_name = str(loss_config.get("name", "cross_entropy")).lower()

    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported loss name: {loss_name}. "
            f"Supported losses: {sorted(SUPPORTED_LOSSES)}"
        )

    if loss_name == "ce":
        loss_name = "cross_entropy"

    ignore_index = int(
        loss_config.get(
            "ignore_index",
            data_config.get("ignore_index", 255),
        )
    )
    num_classes = int(
        loss_config.get(
            "num_classes",
            data_config.get("num_classes", 19),
        )
    )

    class_weights = _build_class_weights(
        loss_config=loss_config,
        num_classes=num_classes,
    )

    return UniversalHybridLoss(
        loss_name=loss_name,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights,
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        alpha=float(loss_config.get("alpha", 0.5)),
        beta=float(loss_config.get("beta", 0.5)),
        ohem_fraction=float(loss_config.get("ohem_fraction", 0.2)),
        gamma=float(loss_config.get("gamma", 2.0)),
        tversky_weight=float(loss_config.get("tversky_weight", 1.0)),
        lovasz_weight=float(loss_config.get("lovasz_weight", 1.0)),
        smooth=float(loss_config.get("smooth", 1.0)),
    )
