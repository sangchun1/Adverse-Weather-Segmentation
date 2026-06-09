from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency import reject_spatial_feature_band
from .night_adapter import SegFormerNightAdapter


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
            "Install with: pip install -e ."
        ) from exc
    return SegformerConfig, SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    """HuggingFace SegFormer wrapper with optional stage-wise NightAdapter.

    This version matches the current proposed-model API:
      - pretrained_name
      - dropout
      - drop_path_rate
      - freeze_mode
      - train_norm_when_frozen
      - align_corners

    When night_adapter.enabled=true in config, stage features are extracted from
    the SegFormer encoder, adapted, and then passed to the original SegFormer
    decode head.
    """

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
        use_night_adapter: bool = False,
        night_adapter_stages: Sequence[int] = (3, 4),
        night_adapter_num_bands: int = 8,
        night_adapter_num_tokens: int = 16,
        night_adapter_randomize_t: float = 0.3,
        night_adapter_randomize_probability: float = 1.0,
        night_adapter_randomize_groups: Sequence[str] = ("H", "M1", "M2"),
        night_adapter_zero_init_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_name = str(pretrained_name)
        self.num_classes = int(num_classes)
        self.dropout = None if dropout is None else float(dropout)
        self.drop_path_rate = None if drop_path_rate is None else float(drop_path_rate)
        self.freeze_mode = str(freeze_mode)
        self.train_norm_when_frozen = bool(train_norm_when_frozen)
        self.align_corners = bool(align_corners)
        self.use_night_adapter = bool(use_night_adapter)
        self._band_rejection_stage: int | None = None
        self._band_rejection_band: int | None = None
        self._band_rejection_num_bands = int(night_adapter_num_bands)

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

        if self.use_night_adapter:
            self.night_adapter = SegFormerNightAdapter(
                stages=night_adapter_stages,
                num_bands=night_adapter_num_bands,
                num_tokens=night_adapter_num_tokens,
                randomize_t=night_adapter_randomize_t,
                randomize_probability=night_adapter_randomize_probability,
                randomize_groups=night_adapter_randomize_groups,
                zero_init_fusion=night_adapter_zero_init_fusion,
            )
        else:
            self.night_adapter = None

        self._apply_freeze(
            stages_to_freeze=FREEZE_MODES[self.freeze_mode],
            train_norm=self.train_norm_when_frozen,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Fast path for the plain proposed SegFormer model.
        if (
            self.night_adapter is None
            and self._band_rejection_stage is None
            and self._band_rejection_band is None
        ):
            outputs = self.model(pixel_values=x)
            return self._upsample_logits(outputs.logits, input_size)

        features = self.extract_encoder_features(x)

        if self._band_rejection_stage is not None:
            features = self.apply_band_rejection(features)

        if self.night_adapter is not None:
            features = self.night_adapter(features)

        logits = self.decode_features(features)
        return self._upsample_logits(logits, input_size)

    def _upsample_logits(
        self,
        logits: torch.Tensor,
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return logits

    def extract_encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return SegFormer encoder stage features.

        Expected output for SegFormer-B2:
          stage 1: (B, 64,  H/4,  W/4)
          stage 2: (B, 128, H/8,  W/8)
          stage 3: (B, 320, H/16, W/16)
          stage 4: (B, 512, H/32, W/32)
        """
        outputs = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer did not return hidden_states.")
        features = list(hidden_states)
        if len(features) != 4:
            raise RuntimeError(
                f"Expected 4 SegFormer hidden states, got {len(features)}. "
                "Check the installed transformers version."
            )
        for idx, feature in enumerate(features):
            if feature.ndim != 4:
                raise RuntimeError(
                    f"Expected stage {idx + 1} feature to have shape (B, C, H, W), "
                    f"got {tuple(feature.shape)}."
                )
        return features

    def decode_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pass possibly modified encoder features to the original decode head."""
        return self.model.decode_head(tuple(features))

    def set_band_rejection(
        self,
        stage: int | None,
        band: int | None,
        num_bands: int = 8,
    ) -> None:
        """Enable stage-wise frequency band rejection for analysis.

        Args:
            stage: 1-based SegFormer stage index. Use None to disable.
            band: 0-based band index. Use None to disable.
        """
        if stage is None or band is None:
            self.clear_band_rejection()
            return
        stage = int(stage)
        if stage < 1 or stage > 4:
            raise ValueError(f"stage must be in [1, 4], got {stage}.")
        self._band_rejection_stage = stage
        self._band_rejection_band = int(band)
        self._band_rejection_num_bands = int(num_bands)

    def clear_band_rejection(self) -> None:
        self._band_rejection_stage = None
        self._band_rejection_band = None

    def apply_band_rejection(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if self._band_rejection_stage is None or self._band_rejection_band is None:
            return list(features)
        output = list(features)
        feature_idx = self._band_rejection_stage - 1
        output[feature_idx] = reject_spatial_feature_band(
            output[feature_idx],
            rejected_band=self._band_rejection_band,
            num_bands=self._band_rejection_num_bands,
        )
        return output

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

        # Decode head should remain trainable for segmentation fine-tuning.
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

    def get_encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.segformer.parameters()

    def get_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.decode_head.parameters()

    def get_adapter_parameters(self) -> Iterable[nn.Parameter]:
        if self.night_adapter is None:
            return []
        return self.night_adapter.parameters()

    def param_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
        adapter_lr: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return optimizer parameter groups with decay/no-decay separation.

        The plain proposed model uses encoder/head groups. When NightAdapter is
        enabled, adapter groups are added without changing existing behavior for
        the plain SegFormer model.
        """
        if adapter_lr is None:
            adapter_lr = head_lr

        encoder_decay: list[nn.Parameter] = []
        encoder_no_decay: list[nn.Parameter] = []
        head_decay: list[nn.Parameter] = []
        head_no_decay: list[nn.Parameter] = []

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

        adapter_decay: list[nn.Parameter] = []
        adapter_no_decay: list[nn.Parameter] = []
        if self.night_adapter is not None:
            for name, param in self.night_adapter.named_parameters():
                if not param.requires_grad:
                    continue
                no_decay = param.ndim == 1 or name.endswith(".bias")
                if no_decay:
                    adapter_no_decay.append(param)
                else:
                    adapter_decay.append(param)

        groups: list[dict[str, Any]] = []
        if encoder_decay:
            groups.append(
                {
                    "params": encoder_decay,
                    "lr": float(encoder_lr),
                    "weight_decay": float(weight_decay),
                    "name": "encoder_decay",
                }
            )
        if encoder_no_decay:
            groups.append(
                {
                    "params": encoder_no_decay,
                    "lr": float(encoder_lr),
                    "weight_decay": 0.0,
                    "name": "encoder_no_decay",
                }
            )
        if adapter_decay:
            groups.append(
                {
                    "params": adapter_decay,
                    "lr": float(adapter_lr),
                    "weight_decay": float(weight_decay),
                    "name": "adapter_decay",
                }
            )
        if adapter_no_decay:
            groups.append(
                {
                    "params": adapter_no_decay,
                    "lr": float(adapter_lr),
                    "weight_decay": 0.0,
                    "name": "adapter_no_decay",
                }
            )
        if head_decay:
            groups.append(
                {
                    "params": head_decay,
                    "lr": float(head_lr),
                    "weight_decay": float(weight_decay),
                    "name": "head_decay",
                }
            )
        if head_no_decay:
            groups.append(
                {
                    "params": head_no_decay,
                    "lr": float(head_lr),
                    "weight_decay": 0.0,
                    "name": "head_no_decay",
                }
            )
        if not groups:
            raise ValueError("No trainable SegFormer parameters found.")
        return groups
