from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UniversalHybridLoss(nn.Module):
    def __init__(
        self, 
        loss_name: str, 
        ignore_index: int, 
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0  # Focal Loss를 위한 gamma 파라미터 추가
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 1. 크로스 엔트로피 정의
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # 2. 다이스 로스 정의
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        
        # 3. 포컬 로스 정의 (gamma 값 동적 할당)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index, gamma=gamma)

        # 4. 트버스키 로스 정의
        self.tversky_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index, alpha=alpha, beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if self.loss_name == "cross_entropy":
            return self.ce_loss(pred, target)
            
        elif self.loss_name == "dice":
            return self.dice_loss(pred, target)
            
        elif self.loss_name == "ce_dice":
            ce_val = self.ce_loss(pred, target)
            dice_val = self.dice_loss(pred, target)
            return (self.ce_weight * ce_val) + (self.dice_weight * dice_val)
            
        elif self.loss_name == "combo_dice":
            return self.ce_loss(pred, target) + self.dice_loss(pred, target)
            
        elif self.loss_name == "focal":
            return self.focal_loss(pred, target)
            
        elif self.loss_name == "tversky":
            return self.tversky_loss(pred, target)
            
        elif self.loss_name == "ce_tversky":
            ce_val = self.ce_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            return (self.ce_weight * ce_val) + (self.dice_weight * tversky_val)
            
        # [추가된 최종 병기] Focal + Tversky 조합 (어려운 픽셀 집중 + 미탐 페널티)
        elif self.loss_name == "focal_tversky":
            focal_val = self.focal_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            # ce_weight를 focal의 가중치로 재활용
            return (self.ce_weight * focal_val) + (self.dice_weight * tversky_val)
            
        else:
            raise ValueError(f"Unsupported loss combination: {self.loss_name}")


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function from config."""
    loss_config = config["loss"]
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    ignore_index = int(config["data"].get("ignore_index", 255))
    num_classes = int(config["data"].get("num_classes", 19))

    ce_weight = float(loss_config.get("ce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))

    alpha = float(loss_config.get("alpha", 0.5))
    beta = float(loss_config.get("beta", 0.5))
    gamma = float(loss_config.get("gamma", 2.0)) # YAML에서 gamma 파라미터 가져오기

    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})"
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    else:
        class_weights = None

    return UniversalHybridLoss(
        loss_name=loss_name, 
        ignore_index=ignore_index, 
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
