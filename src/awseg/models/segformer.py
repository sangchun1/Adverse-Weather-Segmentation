from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# encoder 4개 stage 중 freeze 변화
#   A. full       : freeze 없음         
#   B. freeze_s1  : stage 1 freeze
#   C. freeze_s12 : stage 1, 2 freeze
#   D. head_only  : encoder 전체 freeze, decode head 만 학습
FREEZE_MODES: Dict[str, List[int]] = {
    "full": [],
    "freeze_s1": [0],
    "freeze_s12": [0, 1],
    "head_only": [0, 1, 2, 3],
}


class SegFormerWrapper(nn.Module):

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_classes: int = 19,
        dropout: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
        freeze_mode: str = "full",  #
        train_norm_when_frozen: bool = False, #
        align_corners: bool = False, 
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.align_corners = bool(align_corners)

        # regularization
        hf_config = SegformerConfig.from_pretrained(pretrained_name)
        if dropout is not None:
            hf_config.hidden_dropout_prob = float(dropout)
            hf_config.classifier_dropout_prob = float(dropout)
        if drop_path_rate is not None:
            hf_config.drop_path_rate = float(drop_path_rate)

        # 모델 생성
        same_num_labels = int(getattr(hf_config, "num_labels", 19)) == self.num_classes

        self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                config=hf_config,
                ignore_mismatched_sizes=not same_num_labels,
            )


        if not same_num_labels:
            self.model.config.num_labels = self.num_classes

        # freeze
        if freeze_mode not in FREEZE_MODES:
            raise ValueError(
                f"Unknown freeze_mode: {freeze_mode}. "
                f"Choose from {list(FREEZE_MODES)}"
            )
        self.freeze_mode = freeze_mode
        self.train_norm_when_frozen = bool(train_norm_when_frozen)
        self._apply_freeze(FREEZE_MODES[freeze_mode], self.train_norm_when_frozen)


    def _get_stages(self) -> List[nn.Module]: 

        seg = self.model.segformer
        if hasattr(seg, "stages"): 
            return list(seg.stages)

        enc = seg.encoder
        n = len(enc.patch_embeddings)
        stages = []
        for i in range(n):
            stages.append(
                nn.ModuleList([enc.patch_embeddings[i], enc.block[i], enc.layer_norm[i]])
            )
        return stages

    def _apply_freeze(self, stages_to_freeze: List[int], train_norm: bool) -> None:
        stages = self._get_stages()

        for s in stages_to_freeze:
            stage = stages[s]
            for p in stage.parameters():
                p.requires_grad = False

            # stage 는 freeze 하고 LayerNorm만 학습
            if train_norm:
                for m in stage.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True

        # decoder head 학습
        for p in self.model.decode_head.parameters():
            p.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, C, H/4, W/4]

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return logits 


    def param_groups(self, encoder_lr: float, head_lr: float, weight_decay: float): #
        enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_head = name.startswith("decode_head")
            no_decay = (p.ndim == 1) or name.endswith(".bias")
            if is_head:
                (head_nodecay if no_decay else head_decay).append(p)
            else:
                (enc_nodecay if no_decay else enc_decay).append(p)

        groups = []
        if enc_decay:
            groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay})
        if enc_nodecay:
            groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
        if head_nodecay:
            groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})
        return groups
