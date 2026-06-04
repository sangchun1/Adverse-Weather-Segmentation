from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerWrapper(nn.Module):

    def __init__(
    self,
    pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    num_classes: int = 19,
    freeze_stages: int = 0,
    classifier_dropout_prob: Optional[float] = None,
    drop_path_rate: Optional[float] = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    align_corners: bool = False,
) -> None:
        
        super().__init__()

        self.num_classes = int(num_classes)
        self.align_corners = bool(align_corners)

        # regularization
        hf_config = SegformerConfig.from_pretrained(pretrained_name)
        if classifier_dropout_prob is not None:
            hf_config.classifier_dropout_prob = float(classifier_dropout_prob)
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

        # LoRA
        if self.use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        # freeze
        if not 0 <= int(freeze_stages) <= 4:
            raise ValueError(f"freeze_stages must be in [0, 4], got {freeze_stages}")
        self.freeze_stages = int(freeze_stages)
        self._apply_freeze(self.freeze_stages)


    def _get_stages(self) -> List[nn.Module]: 

        enc = self.model.segformer.encoder
        n = len(enc.patch_embeddings)
        return [
            nn.ModuleList([enc.patch_embeddings[i], enc.block[i], enc.layer_norm[i]])
            for i in range(n)
        ]
    
    def _apply_freeze(self, freeze_stages: int) -> None:
        for stage in self._get_stages()[:freeze_stages]:     # 앞에서 N개
            for p in stage.parameters():
                p.requires_grad = False
        for p in self.model.decode_head.parameters():
            p.requires_grad = True
    
    def _apply_lora(self, r: int, alpha: int, dropout: float) -> None:
        from peft import LoraConfig, get_peft_model
 
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)


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


    def param_groups(
        self,
        base_lr: float,
        encoder_lr_mult: float = 1.0,
        head_lr_mult: float = 1.0,
        weight_decay: float = 1e-2,
    ):
        """Build optimizer param groups with separate encoder/head LR and
        no weight-decay on 1-D params (norm weights / biases)."""
        encoder_lr = base_lr * encoder_lr_mult
        head_lr = base_lr * head_lr_mult
 
        enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_head = "decode_head" in name
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