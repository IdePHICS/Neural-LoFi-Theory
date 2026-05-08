"""Fourier-space utilities for feature visualization.

Shared across architecture-specific visualizers.  All functions are pure
(no model dependency) and work with both PyTorch tensors and NumPy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Spatial shape inference
# ---------------------------------------------------------------------------


def infer_spatial_shape(dim: int) -> tuple[int, ...]:
    """Infer ``(H, W)`` or ``(C, H, W)`` from a flattened dimension.

    Checks for exact square (grayscale), then tries ``C ∈ {3, 1}``.
    """
    side = int(np.sqrt(dim))
    if side * side == dim:
        return (side, side)
    for c in (3, 1):
        spatial = dim // c
        s = int(np.sqrt(spatial))
        if s * s * c == dim:
            return (c, s, s)
    return (side, side)


def spatial_hw(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return ``(C, H, W)`` from a shape that is ``(H, W)`` or ``(C, H, W)``.

    For grayscale ``(H, W)`` inputs, ``C`` is set to 1.
    """
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    return shape[0], shape[1], shape[2]


# ---------------------------------------------------------------------------
# Decay mask construction
# ---------------------------------------------------------------------------


def build_decay_mask(
    H: int,
    W: int,
    f0: float,
    alpha: float,
    device: torch.device,
) -> Tensor:
    """Build a frequency-dependent decay mask for ``rfft2`` format.

    Parameters
    ----------
    H, W : int
        Spatial dimensions of the image.
    f0 : float
        Frequency scale.  Higher values allow more high-frequency content.
    alpha : float
        Decay exponent.  Higher values suppress high frequencies more
        aggressively.
    device : torch.device
        Target device.

    Returns
    -------
    Tensor
        Real-valued mask of shape ``(H, W // 2 + 1)``.
    """
    fy = torch.fft.fftfreq(H, device=device)       # (H,)
    fx = torch.fft.rfftfreq(W, device=device)       # (W//2+1,)
    freq = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return 1.0 / (1.0 + freq / f0) ** alpha


def build_numpy_decay_mask(
    H: int, W: int, f0: float, alpha: float,
) -> np.ndarray:
    """NumPy version of :func:`build_decay_mask` for post-processing."""
    fy = np.fft.fftfreq(H)
    fx = np.fft.rfftfreq(W)
    freq = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return 1.0 / (1.0 + freq / f0) ** alpha


# ---------------------------------------------------------------------------
# Fourier ↔ spatial conversions
# ---------------------------------------------------------------------------


def fourier_to_spatial(
    z_real: Tensor,
    z_imag: Tensor,
    decay_mask: Tensor,
    H: int,
    W: int,
) -> Tensor:
    """Convert Fourier coefficients to a spatial image.

    Parameters
    ----------
    z_real, z_imag : Tensor
        Real and imaginary parts, shape ``(C, H, W // 2 + 1)``.
    decay_mask : Tensor
        Frequency decay mask, shape ``(H, W // 2 + 1)``.
    H, W : int
        Target spatial dimensions.

    Returns
    -------
    Tensor
        Spatial image of shape ``(C, H, W)``.
    """
    z = torch.complex(z_real, z_imag) * decay_mask
    return torch.fft.irfft2(z, s=(H, W))


def smooth_spatial(
    v_flat: np.ndarray,
    shape: tuple[int, ...],
    f0: float,
    alpha: float,
) -> np.ndarray:
    """Low-pass filter a flat pixel-space vector via FFT.

    Parameters
    ----------
    v_flat : ndarray
        Flat vector of length ``prod(shape)``.
    shape : tuple
        ``(H, W)`` or ``(C, H, W)``.
    f0 : float
        Frequency scale for the decay mask.
    alpha : float
        Decay exponent.

    Returns
    -------
    ndarray
        Smoothed flat vector (same length as input).
    """
    C, H, W = spatial_hw(shape)
    img = v_flat.reshape(C, H, W) if len(shape) == 3 else v_flat.reshape(1, H, W)
    mask = build_numpy_decay_mask(H, W, f0, alpha)
    Z = np.fft.rfft2(img)            # (C, H, W//2+1)
    smoothed = np.fft.irfft2(Z * mask, s=(H, W))
    if len(shape) == 2:
        return smoothed.squeeze(0).reshape(-1)
    return smoothed.reshape(-1)
