"""Arc-cosine NNGP kernels for ReLU-family activations.

Two registered kernels:

* ``relu`` — Standard arc-cosine degree-1 kernel (Cho & Saul, 2009):

  ``K(x, x') = (1 / (2π)) (sin θ + (π − θ) cos θ) sqrt(q1 q2)``

  with ``cos θ = q12 / sqrt(q1 q2)`` and the NNGP scalars
  ``q_ii = sigma_w² / d ||x||² + sigma_b²``.

* ``relu_normalized`` — Spherical arc-cosine kernel (per-sample L2
  normalization of the ReLU output, killing the radial component):

  ``K(x, x') = (1 / π) (sin θ + (π − θ) cos θ)``

  with ``cos θ = q12 / sqrt(q1 q2)``.  The diagonal is identically 1
  (``θ = 0`` ⇒ ``K = (0 + π · 1) / π = 1``); ``sigma_w`` and the input
  scale drop out.

Both formulas are closed-form, so the ``n_quad`` argument from
:class:`KernelLayerSpec` is accepted but ignored.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import register_kernel
from .base import KernelBase


def _arc_cosine_j1(cos_theta: Tensor) -> Tensor:
    """Common angular factor ``(sin θ + (π − θ) cos θ) / (2π)``.

    Receives ``cos θ`` already clamped to ``[-1, 1]``; returns the
    Cho-Saul ``J_1(θ)`` factor.  Used by both relu kernels — the
    ``relu_normalized`` formula multiplies by 2 (the ``1/π`` convention),
    the standard ``relu`` formula multiplies by ``sqrt(q1 q2)``.
    """
    theta = torch.acos(cos_theta)
    return (torch.sin(theta) + (math.pi - theta) * cos_theta) / (2.0 * math.pi)


@register_kernel("relu")
class ReluKernel(KernelBase):
    """NNGP kernel for ``sigma(z) = max(0, z)`` (arc-cosine degree 1).

    Parameters
    ----------
    sigma_w : float
        Weight scale (``W ~ N(0, sigma_w² / d)``).
    sigma_b : float
        Bias standard deviation.
    n_quad : int
        Accepted for interface uniformity with quadrature kernels;
        ignored — the kernel has a closed form.
    """

    name = "relu"

    def __init__(
        self,
        *,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        n_quad: int = 30,
    ) -> None:
        super().__init__(sigma_w=sigma_w, sigma_b=sigma_b)
        self.n_quad = int(n_quad)  # accepted for API uniformity, unused

    def kernel_from_qs(self, q1: Tensor, q2: Tensor, q12: Tensor) -> Tensor:
        radial = (q1 * q2).clamp_min(1e-30).sqrt()
        cos_theta = (q12 / radial).clamp(-1.0, 1.0)
        return _arc_cosine_j1(cos_theta) * radial


@register_kernel("relu_normalized")
class ReluNormalizedKernel(KernelBase):
    """NNGP kernel for ReLU + per-sample L2 normalization.

    Activation: ``σ_norm(z) = ReLU(z) / ||ReLU(z)||``.  In the
    ``P → ∞`` limit the per-sample normalization removes the radial
    factor of the standard arc-cosine kernel, leaving the purely
    angular Cho-Saul factor scaled to a unit diagonal.

    The kernel is independent of ``sigma_w`` (the angle ``θ`` between
    inputs is scale-invariant), but the parameter is kept for API
    uniformity with other registered kernels.

    Parameters
    ----------
    sigma_w : float
        Accepted for API uniformity; does not change the kernel.
    sigma_b : float
        Bias standard deviation.  Non-zero values shift ``q1, q2, q12``
        and therefore ``cos θ``; ``sigma_b = 0`` is the standard choice.
    n_quad : int
        Accepted for API uniformity; ignored.
    """

    name = "relu_normalized"

    def __init__(
        self,
        *,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        n_quad: int = 30,
    ) -> None:
        super().__init__(sigma_w=sigma_w, sigma_b=sigma_b)
        self.n_quad = int(n_quad)

    def kernel_from_qs(self, q1: Tensor, q2: Tensor, q12: Tensor) -> Tensor:
        cos_theta = (q12 / (q1 * q2).clamp_min(1e-30).sqrt()).clamp(-1.0, 1.0)
        # K_norm = 2 * J_1(θ) = (sin θ + (π − θ) cos θ) / π.  Diagonal:
        # cos θ = 1 ⇒ θ = 0 ⇒ K = (0 + π · 1) / π = 1.
        return 2.0 * _arc_cosine_j1(cos_theta)
