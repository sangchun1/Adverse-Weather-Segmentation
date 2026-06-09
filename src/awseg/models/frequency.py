from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _cached_sine_basis(
    channels: int,
    device_type: str,
    device_index: int | None,
    dtype_name: str,
) -> torch.Tensor:
    """Build an orthonormal sine basis used as a differentiable DST surrogate.

    The basis is equivalent to the eigenbasis used by a DST-I style transform:
        basis[k, c] = sqrt(2 / (C + 1)) * sin(pi * (k + 1) * (c + 1) / (C + 1))

    It is orthonormal, so the inverse transform is the transpose matrix.
    Caching keeps repeated stage-wise transforms cheap.
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}.")

    if device_type == "cuda":
        device = torch.device(device_type, device_index)
    else:
        device = torch.device(device_type)
    dtype = getattr(torch, dtype_name)

    k = torch.arange(1, channels + 1, device=device, dtype=dtype).view(channels, 1)
    c = torch.arange(1, channels + 1, device=device, dtype=dtype).view(1, channels)
    scale = torch.sqrt(torch.tensor(2.0 / (channels + 1), device=device, dtype=dtype))
    return scale * torch.sin(torch.pi * k * c / float(channels + 1))


def _basis_for(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, C, N), got {tuple(x.shape)}.")
    dtype_name = str(x.dtype).replace("torch.", "")
    return _cached_sine_basis(
        int(x.shape[1]),
        x.device.type,
        x.device.index,
        dtype_name,
    )


def dst_channel(x: torch.Tensor) -> torch.Tensor:
    """Apply a differentiable channel-wise DST to a feature tensor.

    Args:
        x: Tensor with shape (B, C, N), where C is the channel-frequency axis
           and N is the flattened spatial/token axis.

    Returns:
        Tensor with shape (B, C, N).
    """
    basis = _basis_for(x)
    return torch.einsum("kc,bcn->bkn", basis, x)


def idst_channel(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse channel-wise DST.

    Because the sine basis is orthonormal, the inverse transform is basis.T.
    """
    basis = _basis_for(x)
    return torch.einsum("kc,bkn->bcn", basis, x)


def flatten_spatial(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert (B, C, H, W) to (B, C, H*W)."""
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}.")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    return x.flatten(2), (height, width)


def unflatten_spatial(x: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
    """Convert (B, C, H*W) back to (B, C, H, W)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, C, N), got {tuple(x.shape)}.")
    height, width = spatial_size
    expected_tokens = int(height) * int(width)
    if int(x.shape[-1]) != expected_tokens:
        raise ValueError(
            f"Token dimension mismatch: got N={x.shape[-1]}, "
            f"expected H*W={expected_tokens}."
        )
    return x.reshape(x.shape[0], x.shape[1], int(height), int(width))


def split_frequency_bands(x: torch.Tensor, num_bands: int = 8) -> list[torch.Tensor]:
    """Split a channel-frequency tensor into equal-width bands along C."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, C, N), got {tuple(x.shape)}.")
    channels = int(x.shape[1])
    if channels % int(num_bands) != 0:
        raise ValueError(
            f"channels={channels} must be divisible by num_bands={num_bands}."
        )
    return list(torch.chunk(x, int(num_bands), dim=1))


def reject_frequency_band(
    x: torch.Tensor,
    rejected_band: int | None,
    num_bands: int = 8,
) -> torch.Tensor:
    """Return a copy of x where one frequency band has been zeroed out.

    Args:
        x: Tensor with shape (B, C, N) in frequency domain.
        rejected_band: 0-based band index. If None, x is returned unchanged.
        num_bands: Number of equal-width channel-frequency bands.
    """
    if rejected_band is None:
        return x
    rejected_band = int(rejected_band)
    num_bands = int(num_bands)
    if rejected_band < 0 or rejected_band >= num_bands:
        raise ValueError(
            f"rejected_band must be in [0, {num_bands - 1}], got {rejected_band}."
        )

    bands = split_frequency_bands(x, num_bands=num_bands)
    bands[rejected_band] = torch.zeros_like(bands[rejected_band])
    return torch.cat(bands, dim=1)


def reject_spatial_feature_band(
    feature: torch.Tensor,
    rejected_band: int | None,
    num_bands: int = 8,
) -> torch.Tensor:
    """Apply DST -> band rejection -> IDST to a spatial feature map."""
    flat, spatial_size = flatten_spatial(feature)
    freq = dst_channel(flat)
    freq = reject_frequency_band(freq, rejected_band=rejected_band, num_bands=num_bands)
    restored = idst_channel(freq)
    return unflatten_spatial(restored, spatial_size)
