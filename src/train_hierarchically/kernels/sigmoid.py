"""Sigmoid NNGP kernel via 2-D Gauss-Hermite quadrature.

Computes ``K(x, x') = E[sigma(h) sigma(h')]`` with pre-activations
``(h, h') ~ N(0, Sigma)``.  The 2-D Gaussian expectation is evaluated on a
tensor product of 1-D Hermite nodes using the **spectral** square root
of ``Sigma`` (super-algebraic convergence, unlike Cholesky for this
symmetric integrand — see Jaeckel, "A Note on Multivariate Gauss-Hermite
Quadrature").
"""

from __future__ import annotations

import torch
from numpy.polynomial.hermite import hermgauss
from torch import Tensor

from . import register_kernel
from .base import KernelBase

# Cache (n_quad, dtype, device) -> (nodes_scaled, weights_scaled).
# The scaling absorbs the ``u = sqrt(2) z`` substitution and the
# ``1/sqrt(pi)`` normalisation of the standard-normal Hermite rule.
_QUAD_CACHE: dict[tuple[int, torch.dtype, str], tuple[Tensor, Tensor]] = {}


def _get_quadrature(
    n_quad: int, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Scaled Gauss-Hermite nodes/weights for standard-normal expectations."""
    key = (int(n_quad), dtype, str(device))
    cached = _QUAD_CACHE.get(key)
    if cached is not None:
        return cached

    nodes_np, weights_np = hermgauss(int(n_quad))
    nodes = torch.from_numpy(nodes_np).to(dtype=dtype, device=device)
    weights = torch.from_numpy(weights_np).to(dtype=dtype, device=device)
    nodes_scaled = (nodes * (2.0**0.5)).contiguous()
    weights_scaled = (weights / (torch.pi**0.5)).contiguous()
    _QUAD_CACHE[key] = (nodes_scaled, weights_scaled)
    return nodes_scaled, weights_scaled


def _spectral_sqrt_2x2(
    q1: Tensor, q2: Tensor, q12: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form spectral square root of the 2x2 matrix ``[[q1,q12],[q12,q2]]``.

    Returns the three independent entries ``(a, b, c)`` of
    ``Sigma^{1/2} = [[a, b], [b, c]]``.  All inputs broadcast to a common
    shape ``S``; outputs share that shape.
    """
    trace = q1 + q2
    det = q1 * q2 - q12 * q12
    # Clamp tiny negatives from finite-precision (q12^2 just above q1 q2).
    disc = (trace * trace - 4.0 * det).clamp_min(0.0)
    sqrt_disc = disc.sqrt()
    lam_plus = 0.5 * (trace + sqrt_disc)
    lam_minus = (0.5 * (trace - sqrt_disc)).clamp_min(0.0)

    sqrt_lam_plus = lam_plus.sqrt()
    sqrt_lam_minus = lam_minus.sqrt()

    # Eigenvector for lam_plus: v_+ ∝ (q12, lam_plus - q1).  Degenerate
    # case (q12 ≈ 0) falls back to the standard basis, choosing the axis
    # matching the larger diagonal entry.
    eps = torch.finfo(q1.dtype).eps * (q1.abs() + q2.abs() + 1.0)
    near_diag = q12.abs() <= eps

    v1 = torch.where(near_diag, torch.where(q1 >= q2, 1.0, 0.0), q12)
    v2 = torch.where(near_diag, torch.where(q1 >= q2, 0.0, 1.0), lam_plus - q1)
    norm = (v1 * v1 + v2 * v2).sqrt().clamp_min(1e-30)
    v1 = v1 / norm
    v2 = v2 / norm

    # v_- is orthogonal to v_+: (-v2, v1).
    u1 = -v2
    u2 = v1

    a = sqrt_lam_plus * v1 * v1 + sqrt_lam_minus * u1 * u1
    b = sqrt_lam_plus * v1 * v2 + sqrt_lam_minus * u1 * u2
    c = sqrt_lam_plus * v2 * v2 + sqrt_lam_minus * u2 * u2
    return a, b, c


@register_kernel("sigmoid")
class SigmoidKernel(KernelBase):
    """NNGP kernel for ``sigma(z) = 1 / (1 + exp(-z))``.

    Parameters
    ----------
    sigma_w : float
        Weight scale (``W ~ N(0, sigma_w^2 / d)``).
    sigma_b : float
        Bias standard deviation (``b ~ N(0, sigma_b^2)``).
    n_quad : int
        Number of 1-D Gauss-Hermite nodes.  The 2-D rule has ``n_quad^2``
        evaluations per pair; ``30`` is overkill for typical ``sigma_w``
        scales but cheap.
    """

    name = "sigmoid"

    def __init__(
        self,
        *,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        n_quad: int = 30,
    ) -> None:
        super().__init__(sigma_w=sigma_w, sigma_b=sigma_b)
        if n_quad < 2:
            raise ValueError(f"n_quad must be >= 2, got {n_quad}")
        self.n_quad = int(n_quad)

    def _per_pair_working_bytes(self, dtype: torch.dtype) -> int:
        # The fast 4-D path materialises a ``(chunk, m, Q, Q)`` integrand
        # plus two same-shaped temporaries (h1, h2).  Charge ``3*Q*Q``
        # entries per pair so the auto ``row_chunk`` keeps peak memory
        # inside ``memory_budget_bytes``.
        bytes_per = torch.tensor([], dtype=dtype).element_size()
        return max(1, 3 * self.n_quad * self.n_quad) * bytes_per

    def kernel_from_qs(self, q1: Tensor, q2: Tensor, q12: Tensor) -> Tensor:
        """Vectorised ``E[sigma(h) sigma(h')]`` over a batch of pairs.

        The integrand is computed as the **centred product**
        ``(sigma(h1) - 1/2) * (sigma(h2) - 1/2)`` rather than the naive
        ``sigma(h1) * sigma(h2)``.  Since ``sigma(z) = 1/2 + δ(z)`` with
        ``E[δ(h)] = 0`` for zero-mean Gaussian pre-activations, the kernel
        splits as ``K = 1/4 + E[δ(h1) δ(h2)]``.  Integrating ``δ·δ``
        directly keeps the per-quadrature-point magnitude at ``O(δ²)``
        instead of ``O(1/4)``; the constant ``1/4`` term is added once
        after the sum.  This preserves about three extra digits of
        precision on the data-dependent part in float32 — critical when
        the downstream kernel ridge uses α ≈ 1e-6.  See the
        ``feature/kernel-fix-sqrtp`` branch notes for the full argument
        (naive summation ends up relying on the Gauss-Hermite rule
        cancelling the linear terms at float32 precision, which it does
        not do cleanly).

        Two code paths:

        * **Full 4-D** — when the full ``(S, Q, Q)`` integrand fits inside
          :attr:`KernelBase.memory_budget_bytes`, materialise it in one
          shot (minimum Python overhead, best BLAS / vector use).
        * **Streamed over j** — otherwise, loop over blocks of the outer
          quadrature axis ``j`` and accumulate.  Blocks are sized to the
          same budget, so peak memory stays bounded even when ``m`` is
          tens of thousands.

        The reduction uses the dense outer-product weight matrix
        ``w2 = outer(weights, weights)`` and a single 2-axis ``sum``:
        benchmarking on CPU showed this is ~30 % faster than the
        seemingly tighter ``integrand @ weights @ weights`` variant
        (PyTorch's CPU ``sum`` over trailing axes vectorises better than
        two chained batched matmuls).
        """
        q1, q2, q12 = torch.broadcast_tensors(q1, q2, q12)
        a, b, c = _spectral_sqrt_2x2(q1, q2, q12)

        nodes, weights = _get_quadrature(self.n_quad, q1.dtype, q1.device)
        Q = self.n_quad
        w2 = weights.unsqueeze(0) * weights.unsqueeze(1)  # (Q, Q)

        # Budget the outer (j) axis.  One j-block materialises three
        # (S, Jb, Q) working tensors: h1, h2, and their product.
        bytes_per_elem = torch.tensor([], dtype=a.dtype).element_size()
        S = a.numel()
        working_per_j = max(1, 3 * S * Q * bytes_per_elem)
        j_block = max(1, int(self.memory_budget_bytes // working_per_j))
        j_block = min(j_block, Q)

        a_e = a.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)
        b_e = b.unsqueeze(-1).unsqueeze(-1)
        c_e = c.unsqueeze(-1).unsqueeze(-1)
        z_i = nodes.view(*([1] * a.ndim), 1, Q)  # inner i-axis broadcast

        # Fast path: the full ``(S, Q, Q)`` integrand fits in one shot.
        # Out-of-place ops are intentionally used: PyTorch's CPU
        # sigmoid/mul kernels are 30–50 % faster when read and write
        # buffers are distinct (better SIMD + prefetching).
        if j_block >= Q:
            z_j = nodes.view(*([1] * a.ndim), Q, 1)
            h1 = a_e * z_i + b_e * z_j
            h2 = b_e * z_i + c_e * z_j
            integrand = (torch.sigmoid(h1) - 0.5) * (torch.sigmoid(h2) - 0.5)
            return (integrand * w2).sum(dim=(-2, -1)) + 0.25

        # Streamed path: loop over j-blocks, peak memory bounded by
        # ``memory_budget_bytes``.  Per-block we only need ``w2`` sliced
        # over the current rows of the outer axis.
        out = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
        for j0 in range(0, Q, j_block):
            j1 = min(j0 + j_block, Q)
            z_j = nodes[j0:j1].view(*([1] * a.ndim), -1, 1)
            w2_block = w2[j0:j1]  # (Jb, Q)

            h1 = a_e * z_i + b_e * z_j
            h2 = b_e * z_i + c_e * z_j
            integrand = (torch.sigmoid(h1) - 0.5) * (torch.sigmoid(h2) - 0.5)
            out = out + (integrand * w2_block).sum(dim=(-2, -1))

        return out + 0.25
