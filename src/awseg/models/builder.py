from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn

from .unet import UNet
# 고성능 세그멘테이션 모델 라이브러리 가져오기
import segmentation_models_pytorch as smp 


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build segmentation model from config.

    Currently supported:
        - unet
        - deeplabv3plus (새로 추가됨!)

    Expected config example:
        model:
          name: unet  # 또는 deeplabv3plus
          in_channels: 3
          num_classes: 19
          base_channels: 64  # unet 전용
          bilinear: true     # unet 전용
          backbone: resnet50 # deeplabv3plus 전용
          pretrained: true   # deeplabv3plus 전용

    Args:
        config: Experiment config dictionary.

    Returns:
        PyTorch segmentation model.
    """
    model_config = config["model"]
    model_name = str(model_config.get("name", "unet")).lower()

    # 1. 기존 UNet 조립 공정
    if model_name == "unet":
        return UNet(
            in_channels=int(model_config.get("in_channels", 3)),
            num_classes=int(model_config.get("num_classes", config["data"]["num_classes"])),
            base_channels=int(model_config.get("base_channels", 64)),
            bilinear=bool(model_config.get("bilinear", True)),
        )

    # 2. 새로 추가한 DeepLabV3+ 조립 공정
    elif model_name == "deeplabv3plus":
        # 사전학습 가중치를 쓸지 말지 결정 (기본값: 'imagenet')
        encoder_weights = "imagenet" if bool(model_config.get("pretrained", True)) else None
        
        return smp.DeepLabV3Plus(
            encoder_name=str(model_config.get("backbone", "resnet50")), # 백본 등뼈 모델 지정
            encoder_weights=encoder_weights,
            in_channels=int(model_config.get("in_channels", 3)),
            classes=int(model_config.get("num_classes", config["data"]["num_classes"])),
        )

    # 3. 아는 모델이 없을 때 예외 처리
    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Currently supported models: ['unet', 'deeplabv3plus']"
    )
