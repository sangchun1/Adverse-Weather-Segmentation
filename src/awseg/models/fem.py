from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FEMCrossScale(nn.Module):
    """Cross-scale frequency enhancement module.

    This module keeps the key idea of the FEM block: use a shallow feature's
    high-frequency detail and a deep feature's low-frequency semantic response
    to enhance the shallow feature in the frequency domain.

    Args:
        shallow_channels: Number of channels in the shallow feature.
        deep_channels: Number of channels in the deep feature.
        low_radius_ratio: Radius ratio for the low-frequency circular mask.
            The radius is computed as min(H, W) * low_radius_ratio after fftshift.
        init_gamma: Initial residual scale. Use 0.0 to start as identity.
    """

    def __init__(
        self,
        shallow_channels: int,
        deep_channels: int,
        low_radius_ratio: float = 0.25,
        init_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.shallow_channels = int(shallow_channels)
        self.deep_channels = int(deep_channels)
        self.low_radius_ratio = float(low_radius_ratio)

        self.deep_proj = nn.Conv2d(self.deep_channels, self.shallow_channels, kernel_size=1)
        self.freq_fuse = nn.Sequential(
            nn.Conv2d(self.shallow_channels * 6, self.shallow_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.shallow_channels * 2, self.shallow_channels * 2, kernel_size=1),
        )

        # Channel-wise residual scale. With init_gamma=0, the whole FEM path is
        # an identity mapping at the start of fine-tuning.
        self.gamma = nn.Parameter(
            torch.full((1, self.shallow_channels, 1, 1), float(init_gamma))
        )

    @staticmethod
    def _low_frequency_mask(
        height: int,
        width: int,
        ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        yy = torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2.0
        xx = torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        distance = torch.sqrt(grid_y.pow(2) + grid_x.pow(2))
        radius = max(1.0, min(height, width) * float(ratio))
        return (distance <= radius).view(1, 1, height, width)

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        if shallow.ndim != 4 or deep.ndim != 4:
            raise ValueError(
                "FEMCrossScale expects 4D feature maps: "
                f"shallow={tuple(shallow.shape)}, deep={tuple(deep.shape)}"
            )

        residual = shallow
        original_dtype = shallow.dtype
        _, _, height, width = shallow.shape

        shallow_f = shallow.float()
        deep_f = deep.float()

        deep_f = self.deep_proj(deep_f)
        if deep_f.shape[-2:] != (height, width):
            deep_f = F.interpolate(
                deep_f,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        shallow_freq = torch.fft.fftshift(
            torch.fft.fft2(shallow_f, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )
        deep_freq = torch.fft.fftshift(
            torch.fft.fft2(deep_f, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )

        low_mask = self._low_frequency_mask(
            height=height,
            width=width,
            ratio=self.low_radius_ratio,
            device=shallow.device,
        )
        high_mask = ~low_mask

        shallow_low = shallow_freq * low_mask
        shallow_high = shallow_freq * high_mask
        deep_low = deep_freq * low_mask
        deep_high = deep_freq * high_mask

        # Paper-like cross-scale compensation terms.
        high_diff = shallow_high - deep_high
        low_diff = deep_low - shallow_low

        fused_real_imag = torch.cat(
            [
                shallow_freq.real,
                shallow_freq.imag,
                high_diff.real,
                high_diff.imag,
                low_diff.real,
                low_diff.imag,
            ],
            dim=1,
        )
        enhanced_real_imag = self.freq_fuse(fused_real_imag)
        enhanced_real, enhanced_imag = torch.chunk(enhanced_real_imag, chunks=2, dim=1)
        enhanced_freq = torch.complex(enhanced_real, enhanced_imag)

        enhanced_spatial = torch.fft.ifft2(
            torch.fft.ifftshift(enhanced_freq, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ).real

        enhanced_spatial = enhanced_spatial.to(dtype=original_dtype)
        gamma = self.gamma.to(dtype=original_dtype)
        return residual + gamma * enhanced_spatial


class SegFormerFEM(nn.Module):
    """Apply FEMCrossScale to selected SegFormer hidden-state pairs.

    Stage indices are 1-based to match the usual SegFormer stage notation:
    stage 1, 2, 3, 4.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        pairs: Sequence[Sequence[int]] | None = None,
        low_radius_ratio: float = 0.25,
        init_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 4:
            raise ValueError(f"SegFormerFEM expects 4 hidden sizes, got {hidden_sizes}.")

        self.hidden_sizes = [int(v) for v in hidden_sizes]
        self.pairs = self._normalize_pairs(pairs or [[3, 4]])

        modules: dict[str, FEMCrossScale] = {}
        for shallow_stage, deep_stage in self.pairs:
            shallow_channels = self.hidden_sizes[shallow_stage - 1]
            deep_channels = self.hidden_sizes[deep_stage - 1]
            modules[f"s{shallow_stage}_d{deep_stage}"] = FEMCrossScale(
                shallow_channels=shallow_channels,
                deep_channels=deep_channels,
                low_radius_ratio=low_radius_ratio,
                init_gamma=init_gamma,
            )
        self.blocks = nn.ModuleDict(modules)

    @staticmethod
    def _normalize_pairs(pairs: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
        normalized: list[tuple[int, int]] = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Each FEM pair must have two stages, got {pair}.")
            shallow_stage, deep_stage = int(pair[0]), int(pair[1])
            if shallow_stage < 1 or shallow_stage > 4 or deep_stage < 1 or deep_stage > 4:
                raise ValueError(f"FEM stages must be in [1, 4], got {pair}.")
            if shallow_stage == deep_stage:
                raise ValueError(f"FEM shallow/deep stages must be different, got {pair}.")
            if shallow_stage > deep_stage:
                raise ValueError(
                    "FEM expects shallow_stage < deep_stage, "
                    f"but got {pair}."
                )
            normalized.append((shallow_stage, deep_stage))
        return normalized

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(hidden_states) < 4:
            raise ValueError(
                f"SegFormerFEM expects at least 4 hidden states, got {len(hidden_states)}."
            )

        features = list(hidden_states)
        for shallow_stage, deep_stage in self.pairs:
            block = self.blocks[f"s{shallow_stage}_d{deep_stage}"]
            shallow_idx = shallow_stage - 1
            deep_idx = deep_stage - 1
            features[shallow_idx] = block(features[shallow_idx], features[deep_idx])
        return features
