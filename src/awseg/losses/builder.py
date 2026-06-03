from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# --- OHEM Loss 커스텀 구현 ---
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, fraction=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.fraction = fraction # 하위 몇 %의 어려운 픽셀만 학습할 것인가 (기본 20%)

    def forward(self, pred, target):
        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        loss = loss.view(-1)
        target_flat = target.view(-1)
        
        valid_mask = target_flat != self.ignore_index
        loss = loss[valid_mask]
        
        num_hard = int(loss.numel() * self.fraction)
        
        if num_hard == 0:
            return loss.sum() * 0.0
            
        hard_loss, _ = torch.topk(loss, num_hard)
        return hard_loss.mean()


# --- 메인 하이브리드 Loss 클래스 ---
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
        ohem_fraction: float = 0.2,
        gamma: float = 2.0  # ★ Focal Loss를 위한 gamma 파라미터 추가
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 1. 크로스 엔트로피 정의
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # 2. 다이스 / 포컬 / 트버스키 정의 (gamma 값 동적 반영)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index, gamma=gamma)
        self.tversky_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index, alpha=alpha, beta=beta)
        
        # 3. 로바스 로스 (Lovasz) 정의
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index)
        
        # 4. OHEM 로스 정의
        self.ohem_loss = OHEMCrossEntropyLoss(ignore_index=ignore_index, fraction=ohem_fraction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        # [기존 메뉴]
        if self.loss_name == "cross_entropy": 
            return self.ce_loss(pred, target)
        elif self.loss_name == "dice": 
            return self.dice_loss(pred, target)
        elif self.loss_name == "ce_dice": 
            return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.dice_loss(pred, target))
        elif self.loss_name == "tversky": 
            return self.tversky_loss(pred, target)
        elif self.loss_name == "ce_tversky": 
            return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
            
        # ★ [누락되었던 Focal 메뉴들 부활]
        elif self.loss_name == "focal": 
            return self.focal_loss(pred, target)
        elif self.loss_name == "focal_tversky":
            # ce_weight를 focal 가중치로, dice_weight를 tversky 가중치로 재활용합니다.
            return (self.ce_weight * self.focal_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
            
        # [신규 고급 메뉴]
        elif self.loss_name == "ce_lovasz":
            ce_val = self.ce_loss(pred, target)
            lovasz_val = self.lovasz_loss(pred, target)
            return (self.ce_weight * ce_val) + (self.dice_weight * lovasz_val)
            
        elif self.loss_name == "ohem_tversky":
            ohem_val = self.ohem_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            return (self.ce_weight * ohem_val) + (self.dice_weight * tversky_val)
            
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
    
    # ★ YAML에서 복합 파라미터 안전하게 가져오기
    ohem_fraction = float(loss_config.get("ohem_fraction", 0.2))
    gamma = float(loss_config.get("gamma", 2.0)) # 누락되었던 gamma 파라미터 빌더에 추가

    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})"
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    else:
        class_weights = None

    # 모든 변수를 파라미터 인스턴스에 온전하게 넘겨줍니다.
    return UniversalHybridLoss(
        loss_name=loss_name, 
        ignore_index=ignore_index, 
        class_weights=class_weights,
        ce_weight=ce_weight, 
        dice_weight=dice_weight, 
        alpha=alpha, 
        beta=beta,
        ohem_fraction=ohem_fraction,
        gamma=gamma # 추가됨
    )
