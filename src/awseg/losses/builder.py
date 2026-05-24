from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



class UniversalHybridLoss(nn.Module):
    def __init__(self, loss_name: str, ignore_index: int, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.loss_name = loss_name
        
        # 1. 크로스 엔트로피 정의 (가중치가 들어오면 자동으로 반영됨)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # 2. 다이스 로스 정의
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        
        # 3. 포컬 로스 정의
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # YAML에 적힌 name 조건에 따라 실시간으로 다르게 계산하여 반환합니다.
        
        # [메뉴 A] 기본 크로스 엔트로피 (가중치 반영 가능)
        if self.loss_name == "cross_entropy":
            return self.ce_loss(pred, target)
            
        # [메뉴 B] 순수 Dice Loss 단독 사용 (지금 당장 하려는 실험!)
        elif self.loss_name == "dice":
            return self.dice_loss(pred, target)
            
        # [메뉴 C] 하이브리드 Dice Loss (가중치 CE 50% + Dice 50% 섞어 쓰기)
        elif self.loss_name == "combo_dice":
            return self.ce_loss(pred, target) + self.dice_loss(pred, target)
            
        # [메뉴 D] 하이브리드 Focal Loss (가중치 CE 50% + Focal 50% 섞어 쓰기)
        elif self.loss_name == "focal":
            return self.ce_loss(pred, target) + self.focal_loss(pred, target)
            
        else:
            raise ValueError(f"Unsupported loss combination: {self.loss_name}")


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function from config.
    
    Returns a true nn.Module object to prevent .to(device) device assignment errors.
    """
    loss_config = config["loss"]
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    ignore_index = int(config["data"].get("ignore_index", 255))
    num_classes = int(config["data"].get("num_classes", 19))

    # YAML 파일에서 클래스 가중치(weights) 설정 가져와 GPU 장치로 보내기
    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None

    
    return UniversalHybridLoss(loss_name, ignore_index, class_weights)
