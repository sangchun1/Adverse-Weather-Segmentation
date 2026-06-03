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
        beta: float = 0.5
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 1. 크로스 엔트로피 정의 (가중치가 들어오면 자동으로 반영됨)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # 2. 다이스 로스 정의
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        
        # 3. 포컬 로스 정의
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index)

        # 4. 트버스키 로스 정의
        self.tversky_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index, alpha=alpha, beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # YAML에 적힌 name 조건에 따라 실시간으로 다르게 계산하여 반환합니다.
        
        # [메뉴 A] 기본 크로스 엔트로피 (가중치 반영 가능)
        if self.loss_name == "cross_entropy":
            return self.ce_loss(pred, target)
            
        # [메뉴 B] 순수 Dice Loss 단독 사용
        elif self.loss_name == "dice":
            return self.dice_loss(pred, target)
            
        # [메뉴 C] YAML 설정 파일 맞춤형 CE + Dice 조합 (가중치 비율 반영)
        elif self.loss_name == "ce_dice":
            ce_val = self.ce_loss(pred, target)
            dice_val = self.dice_loss(pred, target)
            return (self.ce_weight * ce_val) + (self.dice_weight * dice_val)
            
        # [메뉴 D] 기존 하이브리드 Dice Loss (50:50 고정 비율)
        elif self.loss_name == "combo_dice":
            return self.ce_loss(pred, target) + self.dice_loss(pred, target)
            
        # [메뉴 E] 하이브리드 Focal Loss (가중치 CE 50% + Focal 50% 섞어 쓰기)
        elif self.loss_name == "focal":
            return self.ce_loss(pred, target) + self.focal_loss(pred, target)
            
        # [메뉴 F] Tversky Loss 단독 사용 (작은 객체 탐지 특화)
        elif self.loss_name == "tversky":
            return self.tversky_loss(pred, target)
            
        # [추가된 최종 병기 G] CE + Tversky 콤보 로스 (배경 안정화 + 작은 객체 탐지)
        elif self.loss_name == "ce_tversky":
            ce_val = self.ce_loss(pred, target)
            tversky_val = self.tversky_loss(pred, target)
            # dice_weight 변수를 tversky의 가중치로 재활용하여 사용합니다.
            return (self.ce_weight * ce_val) + (self.dice_weight * tversky_val)
            
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

    # YAML에서 ce_weight와 dice_weight 가져오기 (없으면 기본값 1.0)
    ce_weight = float(loss_config.get("ce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))

    # Tversky 파라미터 가져오기 (없으면 일반 Dice와 동일하게 동작하도록 0.5 할당)
    alpha = float(loss_config.get("alpha", 0.5))
    beta = float(loss_config.get("beta", 0.5))

    # YAML 파일에서 클래스 가중치(weights) 설정 가져오기 (GPU 강제 할당 제거)
    raw_weights = loss_config.get("weights", None)
    if raw_weights is not None:
        assert len(raw_weights) == num_classes, f"Weights length ({len(raw_weights)}) must match num_classes ({num_classes})"
        # .to(device)를 제거하여 나중에 메인 루프에서 criterion.to(device)로 한 번에 이동되도록 수정
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    else:
        class_weights = None

    # 가중치 파라미터들을 모두 함께 인스턴스에 넘겨줍니다.
    return UniversalHybridLoss(
        loss_name=loss_name, 
        ignore_index=ignore_index, 
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        alpha=alpha,
        beta=beta
    )
