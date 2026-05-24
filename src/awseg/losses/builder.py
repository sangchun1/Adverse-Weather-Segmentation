from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_loss(config: Dict[str, Any]) -> Any:
    """Build loss function from config.

    Supported menu:
        - cross_entropy
        - dice
        - focal
    """
    loss_config = config["loss"]
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    ignore_index = int(config["data"].get("ignore_index", 255))
    num_classes = int(config["data"].get("num_classes", 19))

    # YAML 파일에서 클래스 가중치(weights) 설정 가져오기
    # 파이토치에 넣기 위해 리스트를 '토치 텐서(Tensor)' 형태로 변환해 줌
    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights 리스트의 길이는 클래스 개수({num_classes})와 같아야 합니다!"
        # GPU 연산을 위해 .cuda() 또는 지정된 장치로 보냄
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None # 가중치 설정 안 하면 기본값 None(모두 1배)

    # [1번 메뉴] 크로스 엔트로피 (가중치 반영)
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)

    # [2번 메뉴] 하이브리드 Dice Loss (가중치는 CE 부분에만 반영하는 것이 기본 정석)
    elif loss_name == "dice":
        ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        dice_loss = smp.losses.DiceLoss(
            mode="multiclass", 
            ignore_index=ignore_index
        )
        return lambda pred, target: ce_loss(pred, target) + dice_loss(pred, target)

    # [3번 메뉴] 하이브리드 Focal Loss
    elif loss_name == "focal":
        # Focal Loss 자체에 가중치(alpha)를 줄 수도 있지만, 
        # 안정성을 위해 우리 하이브리드 세팅의 CE 쪽에 뇌를 달아주는 방식을 씁니다.
        ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
            ignore_index=ignore_index
        )
        return lambda pred, target: ce_loss(pred, target) + focal_loss(pred, target)

    raise ValueError(
        f"Unknown loss name: {loss_name}. "
        "Currently supported losses: ['cross_entropy', 'dice', 'focal']"
    )
