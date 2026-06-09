from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .frequency import dst_channel, flatten_spatial, idst_channel, split_frequency_bands, unflatten_spatial


class TokenBandAdapter(nn.Module):
    """Token-guided frequency-band adapter.

    This module follows the NightAdapter idea of using learnable tokens to build
    a similarity map for one frequency group, then fusing a linearly projected
    token response back into the original group feature.

    Input/Output shape:
        (B, D, N), where D is the group channel dimension and N=H*W.
    """

    def __init__(self, dim: int, num_tokens: int = 16, zero_init_fusion: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")

        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.tokens = nn.Parameter(torch.empty(self.num_tokens, self.dim))
        self.token_proj = nn.Linear(self.dim, self.dim)
        self.fusion = nn.Linear(self.dim, self.dim)
        self.reset_parameters(zero_init_fusion=zero_init_fusion)

    def reset_parameters(self, zero_init_fusion: bool = True) -> None:
        nn.init.trunc_normal_(self.tokens, std=0.02)
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        if zero_init_fusion:
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)
        else:
            nn.init.xavier_uniform_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, D, N), got {tuple(x.shape)}.")
        if int(x.shape[1]) != self.dim:
            raise ValueError(f"Expected D={self.dim}, got {x.shape[1]}.")

        # (B, D, N) -> (B, N, D)
        group = x.transpose(1, 2)

        # Similarity map: (B, N, M)
        scale = 1.0 / math.sqrt(float(self.dim))
        similarity = torch.matmul(group, self.tokens.t()) * scale
        similarity = torch.softmax(similarity, dim=-1)

        # Projected token response: (M, D)
        projected_tokens = self.token_proj(self.tokens)

        # Token-guided feature: (B, N, D)
        adapted = torch.matmul(similarity, projected_tokens)

        # Residual fusion. With zero-init fusion, this starts as identity.
        output = group + self.fusion(group + adapted)
        return output.transpose(1, 2)


class SensitiveBandRandomizer(nn.Module):
    """Randomize illumination-sensitive frequency responses during training."""

    def __init__(self, threshold: float = 0.3, probability: float = 1.0) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}.")
        if probability < 0 or probability > 1:
            raise ValueError(f"probability must be in [0, 1], got {probability}.")
        self.threshold = float(threshold)
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.threshold == 0 or self.probability == 0:
            return x
        if self.probability < 1.0:
            if torch.rand((), device=x.device) > self.probability:
                return x
        return torch.empty_like(x).uniform_(0.0, self.threshold)


class StageNightAdapter(nn.Module):
    """NightAdapter-style frequency adapter for one SegFormer stage.

    The channel-frequency axis is split into 8 equal bands. Bands are grouped as:
        H  = band 0 + band 1
        M1 = band 2 + band 3
        M2 = band 4 + band 5
        L  = band 6 + band 7

    L receives illumination-insensitive token adaptation.
    H/M1/M2 receive randomization during training and sensitive token adaptation.
    """

    GROUP_TO_BAND_INDICES: Mapping[str, tuple[int, int]] = {
        "H": (0, 1),
        "M1": (2, 3),
        "M2": (4, 5),
        "L": (6, 7),
    }

    def __init__(
        self,
        channels: int,
        num_bands: int = 8,
        num_tokens: int = 16,
        randomize_t: float = 0.3,
        randomize_probability: float = 1.0,
        randomize_groups: Sequence[str] = ("H", "M1", "M2"),
        zero_init_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.num_tokens = int(num_tokens)
        self.randomize_groups = tuple(str(group) for group in randomize_groups)

        if self.num_bands != 8:
            raise ValueError("StageNightAdapter currently expects num_bands=8.")
        if self.channels % self.num_bands != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_bands={num_bands}."
            )

        self.band_dim = self.channels // self.num_bands
        self.group_dim = self.band_dim * 2

        self.insensitive_adapter = TokenBandAdapter(
            dim=self.group_dim,
            num_tokens=self.num_tokens,
            zero_init_fusion=zero_init_fusion,
        )
        self.sensitive_adapters = nn.ModuleDict(
            {
                group: TokenBandAdapter(
                    dim=self.group_dim,
                    num_tokens=self.num_tokens,
                    zero_init_fusion=zero_init_fusion,
                )
                for group in ("H", "M1", "M2")
            }
        )
        self.randomizer = SensitiveBandRandomizer(
            threshold=randomize_t,
            probability=randomize_probability,
        )

    def _concat_group(self, bands: list[torch.Tensor], group: str) -> torch.Tensor:
        band_indices = self.GROUP_TO_BAND_INDICES[group]
        return torch.cat([bands[idx] for idx in band_indices], dim=1)

    def _write_group(self, bands: list[torch.Tensor], group: str, value: torch.Tensor) -> None:
        band_indices = self.GROUP_TO_BAND_INDICES[group]
        pieces = torch.chunk(value, len(band_indices), dim=1)
        for idx, piece in zip(band_indices, pieces):
            bands[idx] = piece

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}.")
        if int(x.shape[1]) != self.channels:
            raise ValueError(f"Expected C={self.channels}, got {x.shape[1]}.")

        flat, spatial_size = flatten_spatial(x)
        freq = dst_channel(flat)
        bands = split_frequency_bands(freq, num_bands=self.num_bands)

        # Illumination-sensitive groups: H, M1, M2.
        for group in ("H", "M1", "M2"):
            group_feature = self._concat_group(bands, group)
            if group in self.randomize_groups:
                group_feature = self.randomizer(group_feature)
            group_feature = self.sensitive_adapters[group](group_feature)
            self._write_group(bands, group, group_feature)

        # Illumination-insensitive low-frequency group: L.
        low_feature = self._concat_group(bands, "L")
        low_feature = self.insensitive_adapter(low_feature)
        self._write_group(bands, "L", low_feature)

        adapted_freq = torch.cat(bands, dim=1)
        restored = idst_channel(adapted_freq)
        return unflatten_spatial(restored, spatial_size)


class SegFormerNightAdapter(nn.Module):
    """Stage-wise NightAdapter module for SegFormer encoder features."""

    DEFAULT_STAGE_CHANNELS: Mapping[int, int] = {
        2: 128,
        3: 320,
        4: 512,
    }

    def __init__(
        self,
        stages: Sequence[int] = (3, 4),
        stage_channels: Mapping[int, int] | None = None,
        num_bands: int = 8,
        num_tokens: int = 16,
        randomize_t: float = 0.3,
        randomize_probability: float = 1.0,
        randomize_groups: Sequence[str] = ("H", "M1", "M2"),
        zero_init_fusion: bool = True,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = self.DEFAULT_STAGE_CHANNELS
        self.stages = tuple(int(stage) for stage in stages)
        self.stage_channels = {int(k): int(v) for k, v in stage_channels.items()}

        for stage in self.stages:
            if stage not in self.stage_channels:
                raise ValueError(
                    f"Unsupported stage {stage}. Available stage_channels: "
                    f"{sorted(self.stage_channels)}."
                )
            if stage < 1 or stage > 4:
                raise ValueError(f"SegFormer stage must be in [1, 4], got {stage}.")

        self.adapters = nn.ModuleDict(
            {
                str(stage): StageNightAdapter(
                    channels=self.stage_channels[stage],
                    num_bands=num_bands,
                    num_tokens=num_tokens,
                    randomize_t=randomize_t,
                    randomize_probability=randomize_probability,
                    randomize_groups=randomize_groups,
                    zero_init_fusion=zero_init_fusion,
                )
                for stage in self.stages
            }
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) < 4:
            raise ValueError(f"Expected 4 SegFormer stage features, got {len(features)}.")
        output = list(features)
        for stage in self.stages:
            feature_idx = stage - 1
            output[feature_idx] = self.adapters[str(stage)](output[feature_idx])
        return output
