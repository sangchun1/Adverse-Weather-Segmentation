from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .unet import UNet


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build segmentation model from config.

    Currently supported:
        - unet

    Expected config example:
        model:
          name: unet
          in_channels: 3
          num_classes: 19
          base_channels: 64
          bilinear: true

    Args:
        config: Experiment config dictionary.

    Returns:
        PyTorch segmentation model.
    """
    model_config = config["model"]
    model_name = str(model_config.get("name", "unet")).lower()

    if model_name == "unet":
        return UNet(
            in_channels=int(model_config.get("in_channels", 3)),
            num_classes=int(model_config.get("num_classes", config["data"]["num_classes"])),
            base_channels=int(model_config.get("base_channels", 64)),
            bilinear=bool(model_config.get("bilinear", True)),
        )

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Currently supported models: ['unet']"
    )
