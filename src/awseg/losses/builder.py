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
        # 픽셀별로 개별 Loss 계산 (reduction='none')
        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # 1차원 텐서로 쭉 펴기
        loss = loss.view(-1)
        target_flat = target.view(-1)
        
        # ignore_index(255) 픽셀은 제외
        valid_mask = target_flat != self.ignore_index
        loss = loss[valid_mask]
        
        # 유효한 픽셀 중 상위 fraction(예: 20%)에 해당하는 어려운 픽셀 개수 산정
        num_hard = int(loss.numel() * self.fraction)
        
        if num_hard == 0:
            return loss.sum() * 0.0 # 에러 방지
            
        # 가장 Loss 값이 큰(제일 많이 틀린) 픽셀들만 추출하여 평균 냄
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
        ohem_fraction: float = 0.2 # OHEM 파라미터 추가
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 1. 크로스 엔트로피
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # 2. 다이스 / 포컬 / 트버스키
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index)
        self.tversky_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index, alpha=alpha, beta=beta)
        
        # 3. [신규 추가] 로바스 로스 (Lovasz)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index)
        
        # 4. [신규 추가] OHEM 로스
        self.ohem_loss = OHEMCrossEntropyLoss(ignore_index=ignore_index, fraction=ohem_fraction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        # 기존 로스들 생략 (이전과 동일하게 작동)
        if self.loss_name == "cross_entropy": return self.ce_loss(pred, target)
        elif self.loss_name == "dice": return self.dice_loss(pred, target)
        elif self.loss_name == "ce_dice": return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.dice_loss(pred, target))
        elif self.loss_name == "tversky": return self.tversky_loss(pred, target)
        elif self.loss_name == "ce_tversky": return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
            
        # [신규 메뉴 1] CE + Lovasz 콤보 (가장 안정적이고 강력한 mIoU 상승 기대)
        elif self.loss_name == "ce_lovasz":
            ce_val = self.ce_loss(pred, target)
            lovasz_val = self.lovasz_loss(pred, target)
            return (self.ce_weight * ce_val) + (self.dice_weight * lovasz_val) # dice_weight 변수 재활용
            
        # [신규 메뉴 2] OHEM + Tversky 콤보 (어려운 픽셀 상위 20%만 CE 학습 + Tversky 결합)
        elif self.loss_name == "ohem_tversky":
            ohem_val = self.ohem_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            return (self.ce_weight * ohem_val) + (self.dice_weight * tversky_val)
            
        else:
            raise ValueError(f"Unsupported loss combination: {self.loss_name}")


def build_loss(config: Dict[str, Any]) -> nn.Module:
    loss_config = config["loss"]
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    ignore_index = int(config["data"].get("ignore_index", 255))
    num_classes = int(config["data"].get("num_classes", 19))

    ce_weight = float(loss_config.get("ce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))
    alpha = float(loss_config.get("alpha", 0.5))
    beta = float(loss_config.get("beta", 0.5))
    
    # OHEM fraction 가져오기 (기본값 상위 20% 픽셀)
    ohem_fraction = float(loss_config.get("ohem_fraction", 0.2))

    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})"
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    else:
        class_weights = None

    return UniversalHybridLoss(
        loss_name=loss_name, ignore_index=ignore_index, class_weights=class_weights,
        ce_weight=ce_weight, dice_weight=dice_weight, alpha=alpha, beta=beta,
        ohem_fraction=ohem_fraction # 추가됨
    )
