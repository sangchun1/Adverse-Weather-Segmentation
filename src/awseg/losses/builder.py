from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .cross_entropy import build_cross_entropy_loss


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function from config.

    Currently supported:
        - cross_entropy

    Expected config example:
        loss:
          name: cross_entropy

        data:
          ignore_index: 255

    Args:
        config: Experiment config dictionary.

    Returns:
        PyTorch loss module.
    """
    loss_config = config.get("loss", {})
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()

    ignore_index = int(config["data"].get("ignore_index", 255))

    if loss_name in {"cross_entropy", "ce"}:
        return build_cross_entropy_loss(ignore_index=ignore_index)

    raise ValueError(
        f"Unknown loss name: {loss_name}. "
        "Currently supported losses: ['cross_entropy']"
    )
