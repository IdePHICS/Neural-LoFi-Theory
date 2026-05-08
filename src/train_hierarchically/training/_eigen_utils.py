"""Shared signed-covariance eigendecomposition for ridge spectral trainers.

Provides three paths:

- **Dense + eigh**: materializes the full ``(P, P)`` covariance, then uses
  ``numpy.linalg.eigh`` for a full decomposition and selects the top-*k*.
  Preferred when *k* is large relative to *P* — ``eigh`` is O(P³) regardless
  of *k*, while ``eigsh`` slows dramatically as *k* approaches *P*.
- **Dense + eigsh**: materializes the ``(P, P)`` covariance, then calls
  ``scipy.sparse.linalg.eigsh``.  Used when *P* fits in memory and *k* is
  small enough that iterative Lanczos is faster than full ``eigh``.
- **LinearOperator + eigsh**: passes implicit matrix-vector products to
  ``eigsh`` so the ``(P, P)`` matrix is never formed.  O(N*P) per Lanczos
  iteration instead of O(N*P²) up front.  Activated only when *P* is very
  large (covariance matrix would be expensive to store/factor).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor

log = logging.getLogger(__name__)

# Default threshold: use LinearOperator when P exceeds this value.
# At P=25000 the (P, P) covariance is ~2.5 GB float32 — affordable on most
# machines.  The linop path only wins when P is so large that materializing
# the covariance is infeasible (e.g. P > 50000 → >10 GB).
_LINOP_THRESHOLD = 50000

# When k exceeds P // _EIGH_K_RATIO we use full eigh instead of eigsh.
# eigsh (Lanczos) time grows roughly as O(P² · k · restarts); eigh is a
# fixed O(P³).  In practice the crossover happens around k ≈ P/4.
_EIGH_K_RATIO = 4


def signed_covariance_eigen(
    z: Tensor,
    y: Tensor,
    k: int,
    *,
    linop_threshold: int = _LINOP_THRESHOLD,
) -> tuple[Tensor, Tensor]:
    """Eigendecompose the signed covariance ``C = (1/N) Z^T diag(y) Z``.

    Dispatches to the dense or LinearOperator path based on *P* and *k*.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets ``(N,)`` or ``(N, 1)``.
    k : int
        Number of top eigenvectors to keep.
    linop_threshold : int
        Use the LinearOperator path when ``P > linop_threshold`` and
        ``k <= P // 2``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(eigenvalues, eigenvectors)`` with shapes ``(k,)`` and ``(P, k)``,
        sorted by descending absolute eigenvalue, on the same device/dtype
        as *z*.
    """
    P = z.shape[1]
    if P <= linop_threshold or k > P // 2:
        return _signed_covariance_eigen_dense(z, y, k)
    return _signed_covariance_eigen_linop(z, y, k)


def _signed_covariance_eigen_dense(
    z: Tensor, y: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """Dense path: form the full (P, P) covariance, then eigh.

    On CUDA: forms ``C`` on the GPU and runs ``torch.linalg.eigh`` directly
    on it.  Full decomposition is O(P³) — at P=20K this is ~1-2 s on a
    modern GPU vs 30-60 s for CPU + scipy eigsh, since the BLAS3 GPU paths
    far outperform sequential Lanczos.

    On CPU (or when ``c`` is too large to fit in VRAM): falls back to
    numpy eigh, with eigsh used when ``k`` is small enough that Lanczos
    beats full O(P³).
    """
    n = z.shape[0]
    P = z.shape[1]

    # GPU fast path: form C and run eigh directly on the device.
    if z.is_cuda and P <= 25000:
        y_flat = y.view(n).to(z.dtype)
        weighted = y_flat.unsqueeze(1) * z  # (N, P)
        c = (weighted.T @ z) / n  # (P, P)
        c = 0.5 * (c + c.T)  # defensive symmetrise
        try:
            eigvals_t, eigvecs_t = torch.linalg.eigh(c)
        except RuntimeError as exc:
            log.warning(
                "GPU eigh at P=%d failed (%s); falling back to CPU.", P, exc,
            )
        else:
            order = torch.argsort(eigvals_t.abs(), descending=True)[:k]
            return eigvals_t[order], eigvecs_t[:, order]

    # CPU path (large-P or GPU-failure fallback).
    if P > 20000:
        z_cpu = z.cpu().float()
        y_cpu = y.view(n).cpu().float()
        weighted = y_cpu.unsqueeze(1) * z_cpu
        c_np = ((weighted.T @ z_cpu) / n).numpy()
    else:
        y_flat = y.view(n).to(z.dtype)
        weighted = y_flat.unsqueeze(1) * z  # (N, P)
        c = (weighted.T @ z) / n  # (P, P)
        c_np = c.cpu().numpy()

    if k >= P // _EIGH_K_RATIO:
        log.info("Using eigh (k=%d, P=%d)", k, P)
        eigvals_np, eigvecs_np = np.linalg.eigh(c_np)
        abs_order = np.argsort(np.abs(eigvals_np))[::-1][:k]
        eigvals_np = eigvals_np[abs_order]
        eigvecs_np = eigvecs_np[:, abs_order]
    else:
        v0 = np.ones(P, dtype=c_np.dtype) / np.sqrt(P)
        eigvals_np, eigvecs_np = eigsh(c_np, k=k, which="LM", v0=v0)

    return _to_sorted_tensors(eigvals_np, eigvecs_np, z.device, z.dtype)


def signed_covariance_full_eigenvalues(z: Tensor, y: Tensor) -> Tensor:
    """Compute all eigenvalues of the signed covariance for dense features.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    Tensor
        Eigenvalues of the signed covariance, sorted in ascending order
        (as returned by ``numpy.linalg.eigvalsh``).
    """
    n = z.shape[0]
    P = z.shape[1]
    if P > 20000:
        z_cpu = z.cpu().float()
        y_cpu = y.view(n).cpu().float()
        weighted = y_cpu.unsqueeze(1) * z_cpu
        c_np = ((weighted.T @ z_cpu) / n).numpy()
        eigvals_np = np.linalg.eigvalsh(c_np)
        return torch.from_numpy(eigvals_np.copy()).to(dtype=torch.float32)

    y_flat = y.view(n).to(z.dtype)
    weighted = y_flat.unsqueeze(1) * z
    c = (weighted.T @ z) / n
    eigvals_np = np.linalg.eigvalsh(c.cpu().numpy())
    return torch.from_numpy(eigvals_np.copy()).to(dtype=z.dtype)


def signed_covariance_full_eigenvalues_from_covariance(c: Tensor) -> Tensor:
    """Compute all eigenvalues from a dense signed-covariance matrix."""
    eigvals_np = np.linalg.eigvalsh(c.cpu().numpy())
    return torch.from_numpy(eigvals_np.copy()).to(dtype=c.dtype)


def _signed_covariance_eigen_linop(
    z: Tensor, y: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """LinearOperator path: implicit matvec, never forms the (P, P) matrix."""
    n = z.shape[0]
    P = z.shape[1]

    # Move to CPU — Lanczos matvecs are sequential, GPU kernel launch
    # overhead dominates for single-vector operations.
    z_cpu = z.cpu().float()
    y_cpu = y.view(-1).cpu().float()

    def matvec(v: np.ndarray) -> np.ndarray:
        v_t = torch.from_numpy(v).float()
        zv = z_cpu @ v_t  # (N,)
        yzv = y_cpu * zv  # (N,)
        return ((z_cpu.T @ yzv) / n).numpy()

    op = LinearOperator((P, P), matvec=matvec, dtype=np.float32)
    v0 = np.ones(P, dtype=np.float32) / np.sqrt(P)
    eigvals_np, eigvecs_np = eigsh(op, k=k, which="LM", v0=v0)

    return _to_sorted_tensors(eigvals_np, eigvecs_np, z.device, z.dtype)


def _to_sorted_tensors(
    eigvals_np: np.ndarray,
    eigvecs_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Convert numpy eigenpairs to sorted tensors on *device*/*dtype*."""
    eigvals = torch.from_numpy(eigvals_np.copy()).to(dtype=dtype, device=device)
    eigvecs = torch.from_numpy(eigvecs_np.copy()).to(dtype=dtype, device=device)

    order = torch.argsort(eigvals.abs(), descending=True)
    return eigvals[order], eigvecs[:, order]
