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
        self.fraction = fraction

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
        gamma: float = 2.0,
        tversky_weight: float = 1.0,
        lovasz_weight: float = 1.0
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.lovasz_weight = lovasz_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index, gamma=gamma)
        self.tversky_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index, alpha=alpha, beta=beta)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index)
        self.ohem_loss = OHEMCrossEntropyLoss(ignore_index=ignore_index, fraction=ohem_fraction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if self.loss_name == "cross_entropy": return self.ce_loss(pred, target)
        elif self.loss_name == "dice": return self.dice_loss(pred, target)
        elif self.loss_name == "ce_dice": return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.dice_loss(pred, target))
        elif self.loss_name == "tversky": return self.tversky_loss(pred, target)
        elif self.loss_name == "ce_tversky": return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
        elif self.loss_name == "focal": return self.focal_loss(pred, target)
        elif self.loss_name == "focal_tversky": return (self.ce_weight * self.focal_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
        elif self.loss_name == "ce_lovasz": return (self.ce_weight * self.ce_loss(pred, target)) + (self.dice_weight * self.lovasz_loss(pred, target))
        elif self.loss_name == "ohem_tversky": return (self.ce_weight * self.ohem_loss(pred, target)) + (self.dice_weight * self.tversky_loss(pred, target))
        elif self.loss_name == "tversky_lovasz": return (self.tversky_weight * self.tversky_loss(pred, target)) + (self.lovasz_weight * self.lovasz_loss(pred, target))
        elif self.loss_name == "ce_focal_dice":
            ce_val = self.ce_loss(pred, target)
            focal_val = self.focal_loss(pred, target)
            dice_val = self.dice_loss(pred, target)
            return (self.ce_weight * ce_val) + focal_val + (self.dice_weight * dice_val)
            
        # ★ [추가된 신규 메뉴] CE(지탱) + Focal(동적 자극) + Tversky(소수 클래스 미탐 저격)
        elif self.loss_name == "ce_focal_tversky":
            ce_val = self.ce_loss(pred, target)
            focal_val = self.focal_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            # dice_weight 변수를 tversky의 배율 가중치로 재활용합니다.
            return (self.ce_weight * ce_val) + focal_val + (self.dice_weight * tversky_val)
            
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
    ohem_fraction = float(loss_config.get("ohem_fraction", 0.2))
    gamma = float(loss_config.get("gamma", 2.0)) 

    tversky_weight = float(loss_config.get("tversky_weight", 1.0))
    lovasz_weight = float(loss_config.get("lovasz_weight", 1.0))

    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})"
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    else:
        class_weights = None

    return UniversalHybridLoss(
        loss_name=loss_name, ignore_index=ignore_index, class_weights=class_weights,
        ce_weight=ce_weight, dice_weight=dice_weight, alpha=alpha, beta=beta,
        ohem_fraction=ohem_fraction, gamma=gamma,
        tversky_weight=tversky_weight, lovasz_weight=lovasz_weight
    )
