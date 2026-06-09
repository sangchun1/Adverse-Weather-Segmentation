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
        night_adapter_config = model_config.get("night_adapter", {}) or {}
        return SegFormerModel(
            pretrained_name=str(
                model_config.get(
                    "pretrained_name",
                    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                )
            ),
            num_classes=num_classes,
            dropout=model_config.get("dropout", None),
            drop_path_rate=model_config.get("drop_path_rate", None),
            freeze_mode=str(model_config.get("freeze_mode", "full")),
            train_norm_when_frozen=bool(model_config.get("train_norm_when_frozen", False)),
            align_corners=bool(model_config.get("align_corners", False)),
            ignore_mismatched_sizes=model_config.get("ignore_mismatched_sizes", None),
            use_night_adapter=bool(night_adapter_config.get("enabled", False)),
            night_adapter_stages=night_adapter_config.get("stages", [3, 4]),
            night_adapter_num_bands=int(night_adapter_config.get("num_bands", 8)),
            night_adapter_num_tokens=int(night_adapter_config.get("num_tokens", 16)),
            night_adapter_randomize_t=float(night_adapter_config.get("randomize_t", 0.3)),
            night_adapter_randomize_probability=float(
                night_adapter_config.get("randomize_probability", 1.0)
            ),
            night_adapter_randomize_groups=night_adapter_config.get(
                "randomize_groups",
                ["H", "M1", "M2"],
            ),
            night_adapter_zero_init_fusion=bool(
                night_adapter_config.get("zero_init_fusion", True)
            ),
        )

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Currently supported models: ['unet', 'segformer']."
    )
