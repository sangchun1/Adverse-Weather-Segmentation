from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FREEZE_MODES: dict[str, list[int]] = {
    "full": [],
    "freeze_s1": [0],
    "freeze_s12": [0, 1],
    "freeze_s123": [0, 1, 2],
    "head_only": [0, 1, 2, 3],
}


def _load_segformer_classes() -> tuple[type[Any], type[Any]]:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "SegFormerModel requires the 'transformers' package. "
            "Install with: pip install -e .[models]"
        ) from exc

    return SegformerConfig, SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    """HuggingFace SegFormer wrapper for semantic segmentation."""

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_classes: int = 19,
        dropout: float | None = None,
        drop_path_rate: float | None = None,
        freeze_mode: str = "full",
        train_norm_when_frozen: bool = False,
        align_corners: bool = False,
        ignore_mismatched_sizes: bool | None = None,
    ) -> None:
        super().__init__()

        self.pretrained_name = str(pretrained_name)
        self.num_classes = int(num_classes)
        self.dropout = None if dropout is None else float(dropout)
        self.drop_path_rate = None if drop_path_rate is None else float(drop_path_rate)
        self.freeze_mode = str(freeze_mode)
        self.train_norm_when_frozen = bool(train_norm_when_frozen)
        self.align_corners = bool(align_corners)

        if self.freeze_mode not in FREEZE_MODES:
            raise ValueError(
                f"Unknown freeze_mode={self.freeze_mode!r}. "
                f"Choose from {list(FREEZE_MODES.keys())}."
            )

        SegformerConfig, SegformerForSemanticSegmentation = _load_segformer_classes()
        hf_config = SegformerConfig.from_pretrained(self.pretrained_name)

        if self.dropout is not None:
            hf_config.hidden_dropout_prob = self.dropout
            hf_config.attention_probs_dropout_prob = self.dropout
            hf_config.classifier_dropout_prob = self.dropout

        if self.drop_path_rate is not None:
            hf_config.drop_path_rate = self.drop_path_rate

        pretrained_num_labels = int(getattr(hf_config, "num_labels", 19))
        same_num_labels = pretrained_num_labels == self.num_classes
        hf_config.num_labels = self.num_classes

        if ignore_mismatched_sizes is None:
            ignore_mismatched_sizes = not same_num_labels

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_name,
            config=hf_config,
            ignore_mismatched_sizes=bool(ignore_mismatched_sizes),
        )

        self._apply_freeze(
            stages_to_freeze=FREEZE_MODES[self.freeze_mode],
            train_norm=self.train_norm_when_frozen,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        return logits

    def _get_stages(self) -> list[nn.Module]:
        segformer = self.model.segformer

        if hasattr(segformer, "stages"):
            return list(segformer.stages)

        encoder = segformer.encoder
        stages: list[nn.Module] = []

        for idx in range(len(encoder.patch_embeddings)):
            stages.append(
                nn.ModuleList(
                    [
                        encoder.patch_embeddings[idx],
                        encoder.block[idx],
                        encoder.layer_norm[idx],
                    ]
                )
            )

        return stages

    def _apply_freeze(self, stages_to_freeze: list[int], train_norm: bool) -> None:
        stages = self._get_stages()

        for stage_idx in stages_to_freeze:
            if stage_idx < 0 or stage_idx >= len(stages):
                raise ValueError(
                    f"Invalid stage index {stage_idx}. "
                    f"Available stages: 0 to {len(stages) - 1}."
                )

            stage = stages[stage_idx]
            for param in stage.parameters():
                param.requires_grad = False

            if train_norm:
                for module in stage.modules():
                    if isinstance(module, nn.LayerNorm):
                        for param in module.parameters():
                            param.requires_grad = True

        for param in self.model.decode_head.parameters():
            param.requires_grad = True

    def get_encoder_parameters(self):
        return self.model.segformer.parameters()

    def get_head_parameters(self):
        return self.model.decode_head.parameters()

    def param_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        encoder_decay = []
        encoder_no_decay = []
        head_decay = []
        head_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_head = name.startswith("decode_head")
            no_decay = param.ndim == 1 or name.endswith(".bias")

            if is_head:
                if no_decay:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
            else:
                if no_decay:
                    encoder_no_decay.append(param)
                else:
                    encoder_decay.append(param)

        groups: list[dict[str, Any]] = []

        if encoder_decay:
            groups.append({"params": encoder_decay, "lr": float(encoder_lr), "weight_decay": float(weight_decay), "name": "encoder_decay"})
        if encoder_no_decay:
            groups.append({"params": encoder_no_decay, "lr": float(encoder_lr), "weight_decay": 0.0, "name": "encoder_no_decay"})
        if head_decay:
            groups.append({"params": head_decay, "lr": float(head_lr), "weight_decay": float(weight_decay), "name": "head_decay"})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": float(head_lr), "weight_decay": 0.0, "name": "head_no_decay"})

        if not groups:
            raise ValueError("No trainable SegFormer parameters found.")

        return groups
