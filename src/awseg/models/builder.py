from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .segformer import SegFormerModel
from .unet import UNet


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    data_config = config.get("data", {})

    model_name = str(model_config.get("name", "unet")).lower()
    num_classes = int(model_config.get("num_classes", data_config.get("num_classes", 19)))

    if model_name == "unet":
        return UNet(
            in_channels=int(model_config.get("in_channels", 3)),
            num_classes=num_classes,
            base_channels=int(model_config.get("base_channels", 64)),
            bilinear=bool(model_config.get("bilinear", True)),
        )

    if model_name in {"segformer", "segformer_b2", "segformer-b2"}:
        return SegFormerModel(
            pretrained_name=str(model_config.get("pretrained_name", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")),
            num_classes=num_classes,
            dropout=model_config.get("dropout", None),
            drop_path_rate=model_config.get("drop_path_rate", None),
            freeze_mode=str(model_config.get("freeze_mode", "full")),
            train_norm_when_frozen=bool(model_config.get("train_norm_when_frozen", False)),
            align_corners=bool(model_config.get("align_corners", False)),
            ignore_mismatched_sizes=model_config.get("ignore_mismatched_sizes", None),
        )

    raise ValueError(f"Unknown model name: {model_name}. Currently supported models: ['unet', 'segformer'].")
