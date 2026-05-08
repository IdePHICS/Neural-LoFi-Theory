"""Kernel-trick variant of :class:`RidgeSpectralTrainer`.

Each :class:`KernelLayerSpec` carries a ``do_eigenreduction`` flag
(mirror of :attr:`LayerSpec.do_eigenreduction` on the finite-width
side) that selects between two per-layer behaviours.

- ``do_eigenreduction=True`` — signed-covariance eigenreduction in the
  kernel limit.  Eigendecompose ``G_l`` (clip at ``eps``), form
  ``G^{1/2}`` and ``G^{-1/2}``, diagonalise the symmetric indefinite
  ``M_tilde = G^{1/2} Y G^{1/2}``, keep the top ``m`` eigenpairs by
  ``|lambda|``, and emit features ``F_l = G^{1/2} B = G_l A`` of shape
  ``(n, m)`` where ``A = G^{-1/2} B`` satisfies the kernelised
  orthonormality ``A^T G_l A = I_m``.  This is the p → ∞ limit of the
  finite-width pipeline ``F_l = Phi_l H_l`` with ``H_l^T H_l = I_m``.

- ``do_eigenreduction=False`` — Standard deep-NNGP layer.  Compose the
  kernel recursion ``K_{l+1} = f(sigma_w^2 K_l + sigma_b^2, …)``
  directly on kernel matrices, with no ``1/m`` scaling.

A single :func:`fit_ridge_readout` call produces the final 1-D readout:
feature-space when the last layer eigenreduces, kernel-space
(kernel-ridge) when it does not.  The returned results dict matches
the schema of :class:`RidgeSpectralTrainer`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..kernels.base import KernelBase
from ..models.kernel_spectral import KernelLayerSpec, KernelSpectralModel
from .base import BaseTrainer, BaseTrainerConfig, fit_ridge_readout
from .deep_nngp import (
    _apply_kernel_rowchunked,
    _auto_rescale_layer0_sigma_w,
    _clone_kernel_with,
    _default_row_chunk,
    deep_nngp_kernel,
)
from .kernel_ridge import fit_kernel_ridge_readout

log = logging.getLogger(__name__)

# Dense-vs-Lanczos crossover: for small ``n`` (or large ``k/n``) it is
# faster to materialise ``M_tilde`` and call ``scipy.linalg.eigh`` than
# to feed a LinearOperator to ``eigsh``.  Thresholds from micro-
# benchmarks on CPU (n = 500…2000).
_DENSE_EIG_MAX_N = 4000
_DENSE_EIG_MIN_K_FRAC = 0.25


# ---------------------------------------------------------------------------
# Cache bundle shared between the trainer and the precompute script.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Layer1Bundle:
    """Precomputed layer-1 state consumed by :class:`KernelSpectralTrainer`.

    All arrays are NumPy float32.  The bundle may be partially populated:
    when ``alpha_full`` / ``lambda_full`` are absent the trainer runs its
    own eigendecomposition on ``g_train``.

    Attributes
    ----------
    g_train : ndarray
        ``(n_max, n_max)`` train Gram matrix.
    g_cross : ndarray | None
        ``(n_max, n_test)`` cross Gram matrix, or ``None``.
    alpha_full : ndarray | None
        ``(n_max, m_max)`` spec-A matrix (``A = G^{-1/2} B`` with ``B``
        the top eigenvectors of ``G^{1/2} Y G^{1/2}`` ranked by
        ``|lambda|``), or ``None``.  Any cache produced with the
        pre-rewrite "generalised-eigenproblem" convention must be
        regenerated — its ``alpha`` has different semantics.
    lambda_full : ndarray | None
        ``(m_max,)`` eigenvalues corresponding to ``alpha_full``.
    """

    g_train: np.ndarray
    g_cross: np.ndarray | None
    alpha_full: np.ndarray | None = None
    lambda_full: np.ndarray | None = None

    def slice_train(self, n: int) -> np.ndarray:
        """Top-left ``(n, n)`` sub-block of :attr:`g_train`."""
        if self.g_train.shape[0] < n or self.g_train.shape[1] < n:
            raise ValueError(
                f"Cached G has shape {self.g_train.shape}; need at least {n} x {n}"
            )
        return np.ascontiguousarray(self.g_train[:n, :n])

    def slice_cross(self, n: int, n_test: int) -> np.ndarray | None:
        """Top-left ``(n, n_test)`` sub-block of :attr:`g_cross`."""
        if self.g_cross is None:
            return None
        if self.g_cross.shape[0] < n or self.g_cross.shape[1] < n_test:
            raise ValueError(
                f"Cached G_cross has shape {self.g_cross.shape}; need at "
                f"least {n} x {n_test}"
            )
        return np.ascontiguousarray(self.g_cross[:n, :n_test])

    def slice_eigen(self, n: int, m: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(eigvals[:m], alpha[:n, :m])`` or ``None`` if absent."""
        if self.alpha_full is None or self.lambda_full is None:
            return None
        m_avail = self.alpha_full.shape[1]
        if m > m_avail:
            raise ValueError(
                f"Layer-1 cache has m_max={m_avail} but layer requests m={m}"
            )
        alpha = np.ascontiguousarray(self.alpha_full[:n, :m])
        eigvals = self.lambda_full[:m].astype(np.float32)
        return eigvals, alpha


@dataclass
class KernelSpectralConfig(BaseTrainerConfig):
    """Configuration for :class:`KernelSpectralTrainer`.

    Parameters
    ----------
    ridge_alpha_min, ridge_alpha_max, ridge_alpha_num : float / int
        Log10 bounds and count for the RidgeCV alpha grid (same defaults
        as :class:`RidgeSpectralConfig`).
    eps : float
        Numerical ridge added to ``G`` when forming the generalised
        eigenproblem ``G diag(y) G alpha = lambda (G + eps I) alpha``.
    m_max : int | None
        Cap on the number of eigenpairs requested per layer.  ``None``
        falls back to the largest layer ``m``.
    row_chunk : int
        Row-chunk size for Gram-matrix construction.
    layer1_cache : Layer1Bundle | None
        Optional precomputed layer-1 state; see :class:`Layer1Bundle`.
    """

    ridge_alpha_min: float = -6.0
    ridge_alpha_max: float = 6.0
    ridge_alpha_num: int = 500
    eps: float = 1e-6
    m_max: int | None = None
    row_chunk: int = 256
    layer1_cache: Layer1Bundle | None = field(default=None, repr=False)
    # Per-layer kernel streaming budget (bytes).  ``None`` keeps each
    # kernel's own default (1 GiB); override on cluster nodes with more
    # RAM to reduce Python overhead in the streamed quadrature path.
    memory_budget_bytes: int | None = None
    # Data-dependent per-layer sigma_w override (NLoFi-script convention).
    # See ``deep_nngp_kernel(auto_rescale=...)`` for the formula.  When
    # ``True``, any user-supplied ``sigma_w`` / ``sigma_b`` on the layer
    # specs is silently overridden in favour of the auto-computed scales,
    # and ``layer1_cache`` is ignored (the cache was built under the
    # fixed-sigma_w convention).
    auto_rescale: bool = False


# ---------------------------------------------------------------------------
# Per-layer state flowing through the trainer loop
# ---------------------------------------------------------------------------


@dataclass
class LayerState:
    """State passed between layers during ``_fit_mixed``.

    Exactly one branch is populated at a time:

    - **Features** (after an eigen layer): ``f_train`` of shape
      ``(n, m)`` and ``f_test`` of shape ``(n_test, m)``.  Rows index
      samples; columns index the kept eigendirections.
    - **Kernel** (after a no-eigen layer or as the initial raw-input
      state): ``k_train`` of shape ``(n, n)``, ``k_cross`` of shape
      ``(n, n_test)``, and ``k_test_diag`` of shape ``(n_test,)``.
    """

    # Features branch (after eigenreduction).
    f_train: Tensor | None = None
    f_test: Tensor | None = None

    # Kernel branch (after no-eigen or initial raw-input Gram).
    k_train: Tensor | None = None
    k_cross: Tensor | None = None
    k_test_diag: Tensor | None = None

    @property
    def is_features(self) -> bool:
        return self.f_train is not None


def _auto_rescale_kernel_for_state(
    kernel: KernelBase, state: LayerState
) -> KernelBase:
    """Clone *kernel* with ``sigma_w`` pinning the next layer's q-diag to 1.

    Mirrors ``deep_nngp_kernel(auto_rescale=True)`` but for the mixed
    eigen / no-eigen path of :class:`KernelSpectralTrainer`.  Two cases:

    - ``state`` carries features ``f_train`` of shape ``(n, m)`` (after
      an eigen layer): ``sigma_w^2 = m / mean(||f_train||^2)``, so that
      ``kernel.gram(f_train, ...)`` builds ``q_diag`` averaging 1.
    - ``state`` carries kernel-state ``k_train`` of shape ``(n, n)``
      (after a no-eigen layer): ``sigma_w^2 = 1 / mean(K_train_diag)``,
      so that ``q = sigma_w^2 K_train + sigma_b^2 = K_train / E[K_diag]``
      averages 1.

    In both cases ``sigma_b = 0``, matching the NLoFi-script convention.
    """
    if state.is_features:
        assert state.f_train is not None
        f = state.f_train
        m = f.shape[1]
        c2 = float((f * f).sum(dim=1).mean().item())
        sigma_w = math.sqrt(m / max(c2, 1e-30))
    else:
        assert state.k_train is not None
        c2 = float(state.k_train.diagonal().mean().item())
        sigma_w = 1.0 / math.sqrt(max(c2, 1e-30))
    return _clone_kernel_with(kernel, sigma_w, 0.0)


# ---------------------------------------------------------------------------
# Generalised signed-covariance eigendecomposition
# ---------------------------------------------------------------------------


def spectral_signed_eig(
    g: np.ndarray, y: np.ndarray, k: int, eps: float,
    *, use_torch_gpu_eigh: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-*k* signed-covariance eigenpairs in the whitened n×n form.

    Implements Section 3.2 of the multilayer signed-covariance kernel
    spec:

    1. ``G = U D U^T`` via ``eigh``; clip ``D`` at ``eps``.
    2. Build ``G^{1/2} = U D^{1/2} U^T`` and
       ``G^{-1/2} = U D^{-1/2} U^T`` from the same factorisation.
    3. Form ``M_tilde = G^{1/2} Y G^{1/2}`` (symmetric indefinite).
    4. Diagonalise ``M_tilde``; pick the top ``k`` eigenpairs ranked by
       ``|lambda|``.  ``B`` collects the eigenvectors
       (``B^T B = I_k``).
    5. Return ``A = G^{-1/2} B`` (satisfies ``A^T G A = I_k`` up to the
       clipping tolerance) and ``F = G^{1/2} B`` — the projected
       training features of shape ``(n, k)``, equivalently ``G A``.

    Two backends:

    - Default (``use_torch_gpu_eigh=False``): scipy on the CPU,
      dispatching between dense ``eigh`` on the full ``M_tilde`` and
      sparse ``eigsh`` on a ``LinearOperator`` based on ``n`` and
      ``k / n``.  Preserves bit-exact behaviour against any caller
      written before the flag was added.

    - Opt-in (``use_torch_gpu_eigh=True``): both eigendecompositions
      run via ``torch.linalg.eigh`` on the active CUDA device (or CPU
      torch when CUDA is unavailable).  Always uses the dense path —
      no Lanczos.  At ``n >= 5000`` this is typically 10x+ faster on a
      modern GPU than scipy's CPU ``eigsh``, even after the host↔device
      transfers required to keep the numpy in/out signature.

    Parameters
    ----------
    g : ndarray, shape (n, n)
        Symmetric kernel Gram matrix.  Upstream gram routines already
        symmetrise; a defensive ``0.5 * (g + g.T)`` is applied here.
    y : ndarray, shape (n,)
        Targets in ``{-1, +1}``.
    k : int
        Number of eigenpairs to return.  Must satisfy ``k < n``.
    eps : float
        Floor for the eigenvalues of ``G`` before forming ``G^{1/2}``
        and ``G^{-1/2}``.
    use_torch_gpu_eigh : bool, default ``False``
        Opt in to the torch-GPU dense path described above.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        ``(lambdas, A, F)`` with shapes ``(k,)``, ``(n, k)``, ``(n, k)``,
        sorted by descending ``|lambda|``.  ``F = G A = G^{1/2} B``.
    """
    n = g.shape[0]
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")

    if use_torch_gpu_eigh:
        return _spectral_signed_eig_torch(g, y, k, eps)

    # Symmetrise defensively — any residual asymmetry from float32
    # kernels biases eigh and propagates into G^{1/2}.
    g_sym = 0.5 * (g + g.T)

    # Eigendecompose G; clip eigenvalues from below by eps so
    # G^{-1/2} is well-defined in the presence of rank deficiency.
    mu, q = eigh(g_sym, check_finite=False)
    mu_clipped = np.clip(mu, eps, None)
    sqrt_d = np.sqrt(mu_clipped).astype(g.dtype, copy=False)
    inv_sqrt_d = (1.0 / sqrt_d).astype(g.dtype, copy=False)

    # G^{1/2} = Q diag(sqrt_d) Q^T.  Reused for the matvec path below
    # and for building F at the end, so materialise once.
    q_cast = q.astype(g.dtype, copy=False)
    g_half = (q_cast * sqrt_d) @ q_cast.T
    g_half = 0.5 * (g_half + g_half.T)

    # ``eigsh`` needs ``k < n``; full-spectrum requests go through eigh.
    use_dense = (
        k == n
        or (n <= _DENSE_EIG_MAX_N and k >= max(1, int(n * _DENSE_EIG_MIN_K_FRAC)))
    )

    if use_dense:
        # M_tilde = (G^{1/2} * y[None, :]) @ G^{1/2}  — broadcasting the
        # diagonal Y avoids an explicit n×n materialisation of Y.
        m_tilde = (g_half * y[None, :]) @ g_half
        m_tilde = 0.5 * (m_tilde + m_tilde.T)
        eigvals, b = eigh(m_tilde, check_finite=False)
        order = np.argsort(np.abs(eigvals))[::-1][:k]
        eigvals = eigvals[order]
        b = np.ascontiguousarray(b[:, order])
    else:

        def matvec(v: np.ndarray) -> np.ndarray:
            w = g_half @ v
            return g_half @ (y * w)

        op = LinearOperator((n, n), matvec=matvec, dtype=g.dtype)
        v0 = np.ones(n, dtype=g.dtype) / np.sqrt(n)
        eigvals, b = eigsh(op, k=k, which="LM", v0=v0)
        order = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[order]
        b = b[:, order]

    # A = Q diag(1/sqrt_d) Q^T @ B — apply G^{-1/2} via the stored
    # eigen-factors so we never form G^{-1/2} as a dense matrix.
    a = q_cast @ (inv_sqrt_d[:, None] * (q_cast.T @ b))
    f = g_half @ b  # equivalently G A (up to eps-level rounding)
    return eigvals, a, f


def _spectral_signed_eig_torch(
    g: np.ndarray, y: np.ndarray, k: int, eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torch-GPU dense implementation of :func:`spectral_signed_eig`.

    Both eigendecompositions run through ``torch.linalg.eigh`` on the
    *currently selected* CUDA device (so callers running multiple
    processes on different GPUs need only ``torch.cuda.set_device(N)``
    once at startup; CPU torch when CUDA is unavailable).  All
    intermediate algebra stays on the device until the final tensors
    are returned to numpy on the host.  Result dtype matches ``g``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dtype_in = g.dtype

    g_t = torch.from_numpy(np.ascontiguousarray(g)).to(
        device=device, dtype=torch.float64,
    )
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(
        device=device, dtype=torch.float64,
    )

    g_sym = 0.5 * (g_t + g_t.T)
    mu, q_t = torch.linalg.eigh(g_sym)
    mu_clipped = torch.clamp(mu, min=eps)
    sqrt_d = torch.sqrt(mu_clipped)
    inv_sqrt_d = 1.0 / sqrt_d

    g_half = (q_t * sqrt_d.unsqueeze(0)) @ q_t.T
    g_half = 0.5 * (g_half + g_half.T)

    m_tilde = (g_half * y_t.view(1, -1)) @ g_half
    m_tilde = 0.5 * (m_tilde + m_tilde.T)
    eigvals_t, b_t = torch.linalg.eigh(m_tilde)
    order = torch.argsort(eigvals_t.abs(), descending=True)[:k]
    eigvals_top = eigvals_t[order]
    b_top = b_t[:, order].contiguous()

    # A = Q diag(1/sqrt_d) Q^T @ B — apply G^{-1/2} without forming it.
    a_top = q_t @ (inv_sqrt_d.unsqueeze(1) * (q_t.T @ b_top))
    f_top = g_half @ b_top

    return (
        eigvals_top.detach().cpu().numpy().astype(dtype_in, copy=False),
        a_top.detach().cpu().numpy().astype(dtype_in, copy=False),
        f_top.detach().cpu().numpy().astype(dtype_in, copy=False),
    )


# Backwards-compatible alias.  Old callers (``precompute_kernel.py``,
# disk caches) expect ``(eigenvalues, alpha)`` where ``alpha`` has the
# spec's semantics.  ``F`` is cheap to recompute (``G @ alpha``) so we
# drop it in the wrapper.
def generalised_signed_eig(
    g: np.ndarray, y: np.ndarray, k: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated alias that returns ``(lambdas, A)`` from :func:`spectral_signed_eig`."""
    eigvals, a, _f = spectral_signed_eig(g, y, k, eps)
    return eigvals, a


class KernelSpectralTrainer(BaseTrainer):
    """Kernel-trick layer-wise trainer; see module docstring."""

    def __init__(
        self,
        model: KernelSpectralModel,
        config: KernelSpectralConfig,
    ) -> None:
        super().__init__(model, config)
        if not isinstance(model, KernelSpectralModel):
            raise TypeError(
                "KernelSpectralTrainer requires a KernelSpectralModel, "
                f"got {type(model).__name__}"
            )

    # Kept as a static method so ``precompute_kernel.py`` can reach the
    # same routine without importing from the module level.  Delegates to
    # :func:`generalised_signed_eig`.
    _generalised_signed_eig = staticmethod(generalised_signed_eig)

    @torch.no_grad()
    def fit(
        self,
        loader: DataLoader,
        *,
        test_loader: DataLoader | None = None,
        **_: object,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Fit all layers and the ridge readout.

        Dispatches on the ``do_eigenreduction`` flags of ``model.layer_specs``:

        - All ``False`` -> standard deep-NNGP recursion + kernel ridge.
        - Any ``True`` -> :class:`NotImplementedError` (the kernel-side
          eigenreduction code path is staged for a later PR).

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            Trained model and a results dict with keys ``"layers"`` and
            ``"final"``.
        """
        model = self._model
        config = self._config

        x_train, y_train = self._collect_dataset(loader)
        x_test: Tensor | None = None
        y_test_np: np.ndarray | None = None
        if test_loader is not None:
            x_test, y_test_t = self._collect_dataset(test_loader)
            y_test_np = y_test_t.cpu().numpy()

        # Propagate the streaming-memory budget to every layer's kernel.
        if config.memory_budget_bytes is not None:
            for k in model.kernels:
                k.memory_budget_bytes = int(config.memory_budget_bytes)

        if not any(spec.do_eigenreduction for spec in model.layer_specs):
            return self._fit_deep_nngp(
                x_train, y_train, x_test, y_test_np, model, config,
            )

        return self._fit_mixed(
            x_train, y_train, x_test, y_test_np, model, config,
        )

    # ------------------------------------------------------------------
    # Deep-NNGP branch (all layers: do_eigenreduction=False)
    # ------------------------------------------------------------------

    def _fit_deep_nngp(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor | None,
        y_test_np: np.ndarray | None,
        model: KernelSpectralModel,
        config: KernelSpectralConfig,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if config.verbose:
            log.info(
                "Deep NNGP fit: %d layers, specs=%s",
                model.n_layers,
                [(s.kernel, s.sigma_w, s.sigma_b) for s in model.layer_specs],
            )

        k_train, k_cross = deep_nngp_kernel(
            model.kernels,
            x_train,
            x_test,
            n_layers=model.n_layers,
            row_chunk=(
                config.row_chunk if config.row_chunk else None
            ),
            progress=lambda m: log.info("  %s", m) if config.verbose else None,
            auto_rescale=config.auto_rescale,
        )

        # Install the state so ``model.forward(x_test)`` recomputes
        # correctly.  No alphas / train_features in the no-eigen path.
        model.set_state(x_train)

        # Standard kernel ridge regression: the exact ``p -> infinity``
        # limit of the finite random-feature ridge readout.  Uses
        # eigendecomposition + leave-one-out CV for alpha selection, and
        # stores the dual coefficients on the model so ``forward`` can
        # evaluate ``K_cross.T @ dual_coef + bias``.
        final = fit_kernel_ridge_readout(
            model,
            k_train,
            y_train,
            K_cross=k_cross,
            y_test_np=y_test_np,
            alpha_min=config.ridge_alpha_min,
            alpha_max=config.ridge_alpha_max,
            alpha_num=config.ridge_alpha_num,
            verbose=config.verbose,
        )

        layer_history = [
            _deep_nngp_layer_entry(i, spec)
            for i, spec in enumerate(model.layer_specs)
        ]
        return model, {"layers": layer_history, "final": final}

    # ------------------------------------------------------------------
    # Mixed branch (at least one layer with do_eigenreduction=True)
    # ------------------------------------------------------------------

    def _fit_mixed(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor | None,
        y_test_np: np.ndarray | None,
        model: KernelSpectralModel,
        config: KernelSpectralConfig,
    ) -> tuple[nn.Module, dict[str, Any]]:
        n_train = x_train.shape[0]
        y_np = y_train.view(n_train).cpu().numpy().astype(np.float32)
        device = x_train.device
        dtype = x_train.dtype

        if config.verbose:
            log.info(
                "Mixed fit: %d layers, specs=%s",
                model.n_layers,
                [
                    (s.kernel, s.m, s.do_eigenreduction)
                    for s in model.layer_specs
                ],
            )

        # --- Initial state: layer-0 Gram from raw inputs ----------------
        # ``auto_rescale`` overrides the layer-0 sigma_w with the
        # data-dependent value ``sqrt(d / mean(||x_train||^2))`` (sigma_b
        # = 0); the user's spec sigma_w / sigma_b are silently dropped.
        # The layer1_cache is built under the fixed-sigma_w convention,
        # so it cannot be reused when auto_rescale is on — we skip it.
        layer1_cache = config.layer1_cache
        if config.auto_rescale and layer1_cache is not None:
            log.info(
                "auto_rescale=True: ignoring layer1_cache (built under "
                "fixed-sigma_w convention)"
            )
            layer1_cache = None

        k0_user = model.kernels[0]
        if config.auto_rescale:
            k0 = _clone_kernel_with(
                k0_user, _auto_rescale_layer0_sigma_w(x_train), 0.0
            )
        else:
            k0 = k0_user

        if layer1_cache is not None:
            g0_train = torch.from_numpy(
                layer1_cache.slice_train(n_train)
            ).to(device=device, dtype=dtype)
            g0_cross = (
                torch.from_numpy(
                    layer1_cache.slice_cross(
                        n_train, x_test.shape[0]
                    )
                ).to(device=device, dtype=dtype)
                if x_test is not None
                and layer1_cache.g_cross is not None
                else (
                    k0.gram(x_train, x_test)
                    if x_test is not None
                    else None
                )
            )
        else:
            g0_train = k0.gram(x_train)
            g0_cross = k0.gram(x_train, x_test) if x_test is not None else None

        # --- Process layer 0 -------------------------------------------
        spec0 = model.layer_specs[0]
        alphas: list[Tensor] = []
        train_features: list[Tensor] = []
        layer_history: list[dict[str, Any]] = []

        if spec0.do_eigenreduction:
            m0 = min(spec0.m, n_train - 1)
            cached = (
                layer1_cache.slice_eigen(n_train, m0)
                if layer1_cache is not None
                else None
            )
            g0_np = g0_train.cpu().float().numpy()
            if cached is not None:
                eigvals_np, alpha_np = cached
                # Features from cached A = G A.  Matches F = G^{1/2} B up
                # to the ``eps``-level rounding used when A was built.
                f_train_np = g0_np @ alpha_np
            else:
                eigvals_np, alpha_np, f_train_np = spectral_signed_eig(
                    g0_np, y_np, m0, config.eps
                )
            alpha_t = torch.from_numpy(alpha_np).to(device=device, dtype=dtype)
            f_train = torch.from_numpy(f_train_np).to(device=device, dtype=dtype)
            # Test-time features: f_test_row(x) = k(x, X_train) A.
            f_test = (
                g0_cross.T @ alpha_t  # (n_test, m)
                if g0_cross is not None
                else None
            )
            state = LayerState(f_train=f_train, f_test=f_test)
            alphas.append(alpha_t)
            train_features.append(f_train)
            layer_history.append(
                _eigen_layer_entry(0, spec0, eigvals_np)
            )
            if config.verbose:
                top = eigvals_np[: min(5, len(eigvals_np))]
                log.info("  Layer 0 (eigen m=%d): top |λ|=%s", m0, top)
        else:
            # No-eigen layer 0: G_0 becomes the kernel state.
            k_test_diag = self._test_diag_from_raw(
                x_test, k0
            ) if x_test is not None else None
            state = LayerState(
                k_train=g0_train, k_cross=g0_cross, k_test_diag=k_test_diag,
            )
            alphas.append(torch.empty(0))
            train_features.append(torch.empty(0))
            layer_history.append(_deep_nngp_layer_entry(0, spec0))
            if config.verbose:
                log.info("  Layer 0 (no-eigen)")

        del g0_train, g0_cross  # free layer-0 Gram early

        # --- Process layers 1 … L-1 ------------------------------------
        for layer_idx in range(1, model.n_layers):
            spec = model.layer_specs[layer_idx]
            kernel = model.kernels[layer_idx]

            # When the final layer is no-eigen (deep-NNGP readout on top
            # of eigenreduced features), promote the incoming features to
            # CPU/float64 before the Gram.  The sigmoid kernel on eigen-
            # features lives near ``σ(0)² = 1/4`` with signal in entries
            # that are themselves ``O(1e-4)``; float32 rounding at
            # magnitude 0.25 destabilises the kernel-ridge solve at
            # α ≈ 10⁻⁶.  MPS does not support float64, so this cast
            # also moves the layer to CPU.
            is_last_noeigen_from_features = (
                layer_idx == model.n_layers - 1
                and not spec.do_eigenreduction
                and state.is_features
            )
            if is_last_noeigen_from_features:
                assert state.f_train is not None
                state = LayerState(
                    f_train=state.f_train.detach().cpu().double(),
                    f_test=(
                        state.f_test.detach().cpu().double()
                        if state.f_test is not None
                        else None
                    ),
                )

            # ``auto_rescale`` clones the kernel with sigma_w pinning the
            # per-sample q-diag to 1.  Computed from the previous state
            # (``mean(||f_train||^2)`` after an eigen layer,
            # ``mean(K_diag)`` after a no-eigen layer); see
            # :func:`_auto_rescale_kernel_for_state`.
            if config.auto_rescale:
                kernel = _auto_rescale_kernel_for_state(kernel, state)

            # Step 1: build this layer's Gram from the previous state.
            g_train, g_cross = self._layer_gram(
                state, kernel, config.row_chunk
            )

            if spec.do_eigenreduction:
                m_l = min(spec.m, n_train - 1)
                g_np = g_train.cpu().float().numpy()
                eigvals_np, alpha_np, f_train_np = spectral_signed_eig(
                    g_np, y_np, m_l, config.eps
                )
                alpha_t = torch.from_numpy(alpha_np).to(
                    device=device, dtype=dtype
                )
                f_train = torch.from_numpy(f_train_np).to(
                    device=device, dtype=dtype
                )
                f_test = (
                    g_cross.T @ alpha_t  # (n_test, m)
                    if g_cross is not None
                    else None
                )
                state = LayerState(f_train=f_train, f_test=f_test)
                alphas.append(alpha_t)
                train_features.append(f_train)
                layer_history.append(
                    _eigen_layer_entry(layer_idx, spec, eigvals_np)
                )
                if config.verbose:
                    top = eigvals_np[: min(5, len(eigvals_np))]
                    log.info(
                        "  Layer %d (eigen m=%d): top |λ|=%s",
                        layer_idx, m_l, top,
                    )
            else:
                # No-eigen: this layer's Gram becomes the kernel state.
                k_test_diag = self._test_diag_from_state(
                    state, kernel
                ) if x_test is not None else None
                state = LayerState(
                    k_train=g_train,
                    k_cross=g_cross,
                    k_test_diag=k_test_diag,
                )
                alphas.append(torch.empty(0))
                train_features.append(torch.empty(0))
                layer_history.append(
                    _deep_nngp_layer_entry(layer_idx, spec)
                )
                if config.verbose:
                    log.info("  Layer %d (no-eigen)", layer_idx)

            del g_train, g_cross  # free Gram after each layer

        # --- Install model state ----------------------------------------
        model.set_state(x_train, alphas=alphas, train_features=train_features)

        # --- Readout depends on the last layer --------------------------
        last_spec = model.layer_specs[-1]
        if last_spec.do_eigenreduction:
            assert state.is_features
            final = fit_ridge_readout(
                model,
                state.f_train,  # (n, m_L)
                y_train,
                h_test=state.f_test if state.f_test is not None else None,
                y_test_np=y_test_np,
                alpha_min=config.ridge_alpha_min,
                alpha_max=config.ridge_alpha_max,
                alpha_num=config.ridge_alpha_num,
                verbose=config.verbose,
            )
        else:
            assert not state.is_features
            final = fit_kernel_ridge_readout(
                model,
                state.k_train,
                y_train,
                K_cross=state.k_cross,
                y_test_np=y_test_np,
                alpha_min=config.ridge_alpha_min,
                alpha_max=config.ridge_alpha_max,
                alpha_num=config.ridge_alpha_num,
                verbose=config.verbose,
            )

        return model, {"layers": layer_history, "final": final}

    # ------------------------------------------------------------------
    # Transition helpers for _fit_mixed
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_gram(
        state: LayerState,
        kernel: KernelBase,
        row_chunk: int | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute the current layer's Gram matrices from the previous state.

        Returns ``(G_train, G_cross)`` where ``G_train`` is ``(n, n)``
        and ``G_cross`` is ``(n, n_test)`` or ``None``.
        """
        if state.is_features:
            # Input is features from the previous eigen layer.
            # kernel.gram uses σ_w²/m automatically (m = f_train.shape[1]).
            assert state.f_train is not None
            g_train = kernel.gram(state.f_train, row_chunk=row_chunk)
            g_cross = (
                kernel.gram(state.f_train, state.f_test, row_chunk=row_chunk)
                if state.f_test is not None
                else None
            )
            return g_train, g_cross

        # Input is kernel state from a previous no-eigen layer.
        # Standard NNGP composition: q = σ_w² K + σ_b² (no 1/m factor).
        assert state.k_train is not None
        sigma_w2 = kernel.sigma_w**2
        sigma_b2 = kernel.sigma_b**2
        n = state.k_train.shape[0]

        chunk = row_chunk or _default_row_chunk(
            kernel, n, n, state.k_train.dtype,
        )

        k_diag = state.k_train.diagonal()
        q_diag = sigma_w2 * k_diag + sigma_b2
        q_train = sigma_w2 * state.k_train + sigma_b2
        g_train = _apply_kernel_rowchunked(
            kernel, q_diag.unsqueeze(1), q_diag.unsqueeze(0),
            q_train, row_chunk=chunk,
        )
        # Defensive symmetrisation (composed kernels can drift slightly).
        g_train = 0.5 * (g_train + g_train.T)
        del q_train

        g_cross: Tensor | None = None
        if state.k_cross is not None and state.k_test_diag is not None:
            q_test_diag = sigma_w2 * state.k_test_diag + sigma_b2
            q_cross = sigma_w2 * state.k_cross + sigma_b2
            g_cross = _apply_kernel_rowchunked(
                kernel, q_diag.unsqueeze(1), q_test_diag.unsqueeze(0),
                q_cross, row_chunk=chunk,
            )

        return g_train, g_cross

    @staticmethod
    def _test_diag_from_raw(
        x_test: Tensor, kernel: KernelBase
    ) -> Tensor:
        """Compute the test self-kernel diagonal from raw inputs."""
        d = x_test.shape[1]
        scale = kernel.sigma_w**2 / d
        bias = kernel.sigma_b**2
        q = scale * (x_test * x_test).sum(dim=1) + bias
        return kernel.kernel_from_qs(q, q, q)

    @staticmethod
    def _test_diag_from_state(
        state: LayerState, kernel: KernelBase
    ) -> Tensor | None:
        """Compute the test self-kernel diagonal for a new no-eigen layer.

        After features: ``K_diag = f(σ_w²/m ||f_test||² + σ_b²)``.
        After kernel: ``K_diag = f(σ_w² K_test_diag + σ_b²)``.
        """
        if state.is_features:
            assert state.f_test is not None
            m = state.f_test.shape[1]
            scale = kernel.sigma_w**2 / m
            bias = kernel.sigma_b**2
            # f_test is (n_test, m); row norms.
            q = scale * (state.f_test * state.f_test).sum(dim=1) + bias
            return kernel.kernel_from_qs(q, q, q)
        if state.k_test_diag is not None:
            sigma_w2 = kernel.sigma_w**2
            sigma_b2 = kernel.sigma_b**2
            q = sigma_w2 * state.k_test_diag + sigma_b2
            return kernel.kernel_from_qs(q, q, q)
        return None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @property
    def _model(self) -> KernelSpectralModel:
        return self.model  # type: ignore[return-value]

    @property
    def _config(self) -> KernelSpectralConfig:
        return self.config  # type: ignore[return-value]

    @torch.no_grad()
    def _collect_dataset(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for x, y in loader:
            x, y = self._to_device(x, y)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)


def _eigen_layer_entry(
    layer_idx: int, spec: KernelLayerSpec, eigvals_np: np.ndarray
) -> dict[str, Any]:
    return {
        "layer_idx": layer_idx,
        "kind": "kernel_eigen",
        "kernel": spec.kernel,
        "m": int(spec.m),
        "do_eigenreduction": True,
        "layer_name": f"layer_{layer_idx} ({spec.kernel} kernel, m={spec.m})",
        "eigenvalues": eigvals_np.astype(float).tolist(),
    }


def _deep_nngp_layer_entry(
    layer_idx: int, spec: KernelLayerSpec
) -> dict[str, Any]:
    return {
        "layer_idx": layer_idx,
        "kind": "deep_nngp",
        "kernel": spec.kernel,
        "do_eigenreduction": False,
        "layer_name": f"layer_{layer_idx} ({spec.kernel} kernel, deep NNGP)",
    }
