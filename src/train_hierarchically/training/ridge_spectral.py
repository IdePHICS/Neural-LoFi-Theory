"""Ridge spectral trainer: random features + eigenvector projection + RidgeCV.

Layer-wise procedure:
    1. Estimate input normalization: c_l = sqrt(mean(‖h‖²)) over training set
       Store c_l in ``model.input_norm_coefficients[l]`` for inference.
    2. Expand via random projection: Z_l = σ_l(h @ W_l / c_l) / √P_l
       (``/ c_l`` keeps pre-activation magnitudes O(1); ``/ √P_l`` keeps
       ``‖z_μ‖²`` finite as p → ∞.)
    3. Compute signed covariance: C = (y * Z_l).T @ Z_l / N
    4. Eigendecompose C, keep top-K_l eigenvectors as V_l
    5. Project: h = Z_l @ V_l
    6. Repeat for each layer

Final readout is sklearn RidgeCV (1-D output, ±1 targets).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import eigsh as sp_eigsh
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.ridge_spectral import RidgeSpectralModel
from ..utils.helpers import ACTIVATIONS
from ._eigen_utils import signed_covariance_eigen
from .base import BaseTrainer, BaseTrainerConfig, fit_ridge_readout

log = logging.getLogger(__name__)


@dataclass
class RidgeSpectralConfig(BaseTrainerConfig):
    """Configuration for ridge spectral training.

    Parameters
    ----------
    ridge_alpha_min : float
        Log10 of the smallest regularization candidate. Default ``-6``.
    ridge_alpha_max : float
        Log10 of the largest regularization candidate. Default ``6``.
    ridge_alpha_num : int
        Number of regularization candidates. Default ``500``.
    """

    ridge_alpha_min: float = -6.0
    ridge_alpha_max: float = 6.0
    ridge_alpha_num: int = 500
    store_spectra: bool = False
    n_top_eigenvectors: int = 4
    store_full_eigenvalues: bool = False


class RidgeSpectralTrainer(BaseTrainer):
    """Trainer: random features + eigenvector projection + ridge readout.

    For each layer *l* (l = 1 … L):

    1. Compute random features: ``H = σ_l(h @ W_l)``
    2. Signed covariance: ``C = (1/N) H^T diag(y) H``
    3. Eigendecompose *C*, keep top-K_l eigenvectors → ``V_l``
    4. Project: ``h = H @ V_l``

    After all layers, fit ``RidgeCV(h, y)`` for the final 1-D readout.

    Parameters
    ----------
    model : RidgeSpectralModel
        Model to train.
    config : RidgeSpectralConfig
        Training configuration.
    """

    def __init__(
        self,
        model: RidgeSpectralModel,
        config: RidgeSpectralConfig,
    ) -> None:
        super().__init__(model, config)
        if not isinstance(model, RidgeSpectralModel):
            raise TypeError(
                "RidgeSpectralTrainer requires a RidgeSpectralModel, "
                f"got {type(model).__name__}"
            )

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Fit all layers and the ridge readout.

        Parameters
        ----------
        loader : DataLoader
            Training data yielding ``(X, y)`` batches.
        test_loader : DataLoader | None
            Optional test loader for intermediate evaluation.

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model and a results dict with keys ``"layers"``
            and ``"final"``.
        """
        test_loader: DataLoader | None = kwargs.get("test_loader")  # type: ignore[assignment]
        model: RidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]

        x_all, y_all = self._collect_dataset(loader)

        # Pre-load test data
        x_test: Tensor | None = None
        y_test_np: np.ndarray | None = None
        if test_loader is not None:
            x_test, y_test_t = self._collect_dataset(test_loader)
            y_test_np = y_test_t.cpu().numpy()

        if config.verbose:
            if model.n_layers == 0:
                log.info("Ridge spectral fit: 0 layers (pure ridge on raw inputs)")
            else:
                log.info(
                    "Ridge spectral fit: %d layers, specs=%s",
                    model.n_layers,
                    [(s.p, s.k, s.activation) for s in model.layer_specs],
                )

        # ----------------------------------------------------------
        # Layer-wise: random features → eigendecompose → project
        # ----------------------------------------------------------
        h_train = x_all  # (N, D)
        h_test = x_test
        layer_history: list[dict[str, Any]] = []

        # Batch size for large-P layers to avoid GPU OOM.
        # When N * P * 4 bytes exceeds this limit, use batched processing.
        # Conservative: 4 GB leaves room for W, intermediate tensors, and
        # PyTorch allocator overhead.
        _BATCH_MEM_LIMIT = 4 * 1024**3  # 4 GB

        for layer_idx in range(model.n_layers):
            spec = model.layer_specs[layer_idx]
            N = h_train.shape[0]
            device = h_train.device

            # --- Scattering layers: apply transform + flatten ----------
            if spec.kind == "scattering":
                z_train = model._apply_scattering(h_train, layer_idx)
                z_test = (
                    model._apply_scattering(h_test, layer_idx)
                    if h_test is not None
                    else None
                )

                if spec.do_eigenreduction:
                    eigvals, eigvecs = self._signed_covariance_eigen(
                        z_train, y_all, k=spec.k
                    )
                    model.eigenvector_projections[layer_idx] = nn.Parameter(
                        eigvecs, requires_grad=False
                    )

                h_train = z_train @ model.eigenvector_projections[layer_idx]
                h_test = (
                    z_test @ model.eigenvector_projections[layer_idx]
                    if z_test is not None
                    else None
                )

                layer_name = (
                    f"layer_{layer_idx} (scattering "
                    f"J={spec.scattering_j}, L={spec.scattering_l}, "
                    f"ord={spec.scattering_max_order}, K={spec.k})"
                )
                layer_entry: dict[str, Any] = {
                    "layer_idx": layer_idx,
                    "kind": spec.kind,
                    "layer_name": layer_name,
                }
                if spec.do_eigenreduction:
                    layer_entry["eigenvalues"] = eigvals.cpu().numpy().tolist()
                if config.store_spectra:
                    n_top = config.n_top_eigenvectors
                    layer_entry["spectra"] = self._full_spectra(
                        z_train, y_all, n_top
                    )

            # --- Random-features layers: batched or unbatched ----------
            else:
                w = model.random_projections[layer_idx]
                sigma = ACTIVATIONS[spec.activation]

                # c_l = sqrt(E[‖h‖²]): estimated from training data, stored
                # on the model so the same normalization is used at inference.
                norm_coeff = h_train.pow(2).sum(dim=1).mean().sqrt()
                model.input_norm_coefficients[layer_idx].data.fill_(
                    norm_coeff.item()
                )

                need_batch = (N * spec.p * 4) > _BATCH_MEM_LIMIT
                batch_size = (
                    max(1, _BATCH_MEM_LIMIT // (spec.p * 4))
                    if need_batch
                    else N
                )

                if not need_batch:
                    inv_sqrt_p = 1.0 / math.sqrt(spec.p)
                    z_train = sigma(h_train @ w / norm_coeff) * inv_sqrt_p
                    z_test = (
                        sigma(h_test @ w / norm_coeff) * inv_sqrt_p
                        if h_test is not None
                        else None
                    )

                    if spec.do_eigenreduction:
                        eigvals, eigvecs = self._signed_covariance_eigen(
                            z_train, y_all, k=spec.k
                        )
                        model.eigenvector_projections[layer_idx] = nn.Parameter(
                            eigvecs, requires_grad=False
                        )

                    h_train = z_train @ model.eigenvector_projections[layer_idx]
                    h_test = (
                        z_test @ model.eigenvector_projections[layer_idx]
                        if z_test is not None
                        else None
                    )

                    layer_entry = {
                        "layer_idx": layer_idx,
                        "kind": spec.kind,
                        "layer_name": (
                            f"layer_{layer_idx} (P={spec.p}, K={spec.k})"
                        ),
                    }
                    if spec.do_eigenreduction:
                        layer_entry["eigenvalues"] = (
                            eigvals.cpu().numpy().tolist()
                        )
                    if config.store_spectra:
                        n_top = config.n_top_eigenvectors
                        layer_entry["spectra"] = self._full_spectra(
                            z_train, y_all, n_top
                        )
                else:
                    # Batched path: Z = σ(h @ W) doesn't fit in GPU.
                    if config.verbose:
                        log.info(
                            "  Layer %d: using batched processing "
                            "(batch_size=%d for N=%d, P=%d)",
                            layer_idx, batch_size, N, spec.p,
                        )

                    # Memory guard for CPU Z accumulation (32 GB)
                    _CPU_Z_MEM_LIMIT = 32 * 1024**3
                    cpu_z_fits = (N * spec.p * 4) <= _CPU_Z_MEM_LIMIT
                    use_linop = cpu_z_fits and not config.store_spectra

                    inv_sqrt_p = 1.0 / math.sqrt(spec.p)

                    if use_linop:
                        # Accumulate full Z on CPU, then use
                        # LinearOperator (no P×P matrix).
                        z_cpu_all = torch.empty(
                            N, spec.p, dtype=torch.float32, device="cpu",
                        )
                        for i in range(0, N, batch_size):
                            h_batch = h_train[i: i + batch_size]
                            z_batch = sigma(h_batch @ w / norm_coeff) * inv_sqrt_p
                            z_cpu_all[i: i + batch_size] = z_batch.cpu().float()

                        if spec.do_eigenreduction:
                            y_cpu = y_all.cpu().float()
                            eigvals, eigvecs = signed_covariance_eigen(
                                z_cpu_all, y_cpu, spec.k,
                            )
                            eigvals = eigvals.to(
                                dtype=h_train.dtype, device=device,
                            )
                            eigvecs = eigvecs.to(
                                dtype=h_train.dtype, device=device,
                            )
                            model.eigenvector_projections[layer_idx] = (
                                nn.Parameter(eigvecs, requires_grad=False)
                            )

                        del z_cpu_all
                    else:
                        # Fall back to P×P covariance accumulation
                        # (needed for store_spectra or huge Z).
                        cov_acc = torch.zeros(
                            spec.p, spec.p,
                            dtype=torch.float32, device="cpu",
                        )
                        signed_cov_acc = torch.zeros(
                            spec.p, spec.p,
                            dtype=torch.float32, device="cpu",
                        )
                        for i in range(0, N, batch_size):
                            h_batch = h_train[i: i + batch_size]
                            y_batch = y_all[i: i + batch_size]
                            z_batch = sigma(h_batch @ w / norm_coeff) * inv_sqrt_p
                            z_cpu = z_batch.cpu().float()
                            y_cpu = y_batch.cpu().float().view(-1, 1)
                            cov_acc += z_cpu.T @ z_cpu
                            signed_cov_acc += (y_cpu * z_cpu).T @ z_cpu
                        cov_acc /= N
                        signed_cov_acc /= N

                        if spec.do_eigenreduction:
                            c_np = signed_cov_acc.numpy()
                            p_ = c_np.shape[0]
                            v0 = np.ones(p_, dtype=c_np.dtype) / np.sqrt(
                                p_
                            )
                            eigvals_np, eigvecs_np = sp_eigsh(
                                c_np, k=spec.k, which="LM", v0=v0,
                            )
                            order = np.argsort(np.abs(eigvals_np))[::-1]
                            eigvals = torch.from_numpy(
                                eigvals_np[order],
                            ).to(dtype=h_train.dtype, device=device)
                            eigvecs = torch.from_numpy(
                                eigvecs_np[:, order],
                            ).to(dtype=h_train.dtype, device=device)
                            model.eigenvector_projections[layer_idx] = (
                                nn.Parameter(eigvecs, requires_grad=False)
                            )

                    # Pass 2: project h_train = z @ V batch-by-batch
                    V = model.eigenvector_projections[layer_idx]
                    h_new = torch.empty(
                        N, V.shape[1], dtype=h_train.dtype, device=device,
                    )
                    for i in range(0, N, batch_size):
                        h_batch = h_train[i: i + batch_size]
                        z_batch = sigma(h_batch @ w / norm_coeff) * inv_sqrt_p
                        h_new[i: i + batch_size] = z_batch @ V
                    h_train = h_new

                    # Test set projection
                    if h_test is not None:
                        h_test = (sigma(h_test @ w / norm_coeff) * inv_sqrt_p) @ V

                    layer_entry = {
                        "layer_idx": layer_idx,
                        "kind": spec.kind,
                        "layer_name": (
                            f"layer_{layer_idx} (P={spec.p}, K={spec.k})"
                        ),
                    }
                    if spec.do_eigenreduction:
                        layer_entry["eigenvalues"] = (
                            eigvals.cpu().numpy().tolist()
                        )
                    if config.store_spectra and not use_linop:
                        n_top = config.n_top_eigenvectors
                        layer_entry["spectra"] = (
                            self._full_spectra_from_covariances(
                                cov_acc, signed_cov_acc, n_top,
                            )
                        )

            layer_history.append(layer_entry)

            if config.verbose and spec.do_eigenreduction:
                top_eig = eigvals[: min(5, len(eigvals))].cpu().numpy()
                log.info(
                    "  Layer %d (%s): K=%d, top eigenvalues=%s",
                    layer_idx, spec.kind, spec.k, top_eig,
                )

        # ----------------------------------------------------------
        # Ridge readout
        # ----------------------------------------------------------
        final = fit_ridge_readout(
            model,
            h_train,
            y_all,
            h_test=h_test,
            y_test_np=y_test_np,
            alpha_min=config.ridge_alpha_min,
            alpha_max=config.ridge_alpha_max,
            alpha_num=config.ridge_alpha_num,
            verbose=config.verbose,
        )

        results: dict[str, Any] = {
            "layers": layer_history,
            "final": final,
        }

        return model, results

    # ------------------------------------------------------------------
    # Signed covariance eigendecomposition
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _signed_covariance_eigen(
        z: Tensor, y: Tensor, k: int
    ) -> tuple[Tensor, Tensor]:
        """Eigendecompose the signed covariance matrix.

        Computes ``C = (1/N) Z^T diag(y) Z`` and returns the top-*k*
        eigenvectors sorted by descending absolute eigenvalue.

        Delegates to :func:`._eigen_utils.signed_covariance_eigen` which
        uses a ``LinearOperator`` path (implicit matvec, no P*P matrix)
        when P is large and k is small, or falls back to the dense path.

        Parameters
        ----------
        z : Tensor
            Feature matrix ``(N, P)``.
        y : Tensor
            Targets ``(N,)``.
        k : int
            Number of top eigenvectors to keep.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(eigenvalues, eigenvectors)`` where eigenvalues has shape
            ``(k,)`` and eigenvectors has shape ``(P, k)``.
        """
        return signed_covariance_eigen(z, y, k)

    # ------------------------------------------------------------------
    # Full spectral analysis
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _full_spectra_from_covariances(
        cov: Tensor,
        signed_cov: Tensor,
        n_top: int,
    ) -> dict[str, Any]:
        """Compute top eigenvectors from pre-accumulated covariance matrices.

        Used by the batched training path where covariances are accumulated
        batch-by-batch and already reside on CPU.
        """
        result: dict[str, Any] = {}
        P = cov.shape[0]
        for prefix, mat in [("cov", cov), ("signed_cov", signed_cov)]:
            mat_np = mat.cpu().float().numpy()
            k_use = min(n_top, P - 1)
            eigvals_np, eigvecs_np = sp_eigsh(
                mat_np, k=k_use, which="LM",
            )
            order = np.argsort(np.abs(eigvals_np))[::-1]
            result[f"{prefix}_spectrum"] = []
            result[f"{prefix}_top_eigenvalues"] = (
                eigvals_np[order].tolist()
            )
            result[f"{prefix}_top_eigenvectors"] = (
                eigvecs_np[:, order].tolist()
            )
        return result

    @staticmethod
    @torch.no_grad()
    def _full_spectra(
        z: Tensor, y: Tensor, n_top: int
    ) -> dict[str, Any]:
        """Compute full eigenvalue spectra and top eigenvectors.

        Returns a dict with keys for both the unsigned covariance
        ``Z^T Z / N`` and the signed covariance ``(y * Z)^T Z / N``.

        Parameters
        ----------
        z : Tensor
            Feature matrix ``(N, P)``.
        y : Tensor
            Targets ``(N,)``.
        n_top : int
            Number of top eigenvectors to keep (by absolute eigenvalue).

        Returns
        -------
        dict[str, Any]
            ``cov_spectrum``, ``signed_cov_spectrum``: full eigenvalue arrays.
            ``cov_top_eigenvalues``, ``cov_top_eigenvectors``: top-n_top for
            the unsigned covariance.
            ``signed_cov_top_eigenvalues``, ``signed_cov_top_eigenvectors``:
            top-n_top for the signed covariance.
        """
        P = z.shape[1]
        use_sparse = P > 20000  # Use scipy sparse eigsh for large P

        if use_sparse:
            # Move to CPU for large P to avoid GPU OOM
            z_cpu = z.cpu().float()
            y_cpu = y.view(z.shape[0]).cpu().float()
            n = z_cpu.shape[0]
            cov = (z_cpu.T @ z_cpu) / n
            signed_cov = (y_cpu.unsqueeze(1) * z_cpu).T @ z_cpu / n
        else:
            n = z.shape[0]
            y_flat = y.view(n).to(z.dtype)
            cov = (z.T @ z) / n
            signed_cov = (y_flat.unsqueeze(1) * z).T @ z / n

        result: dict[str, Any] = {}
        for prefix, mat in [("cov", cov), ("signed_cov", signed_cov)]:
            if use_sparse:
                # For large P, use scipy sparse eigsh (top-K only)
                mat_np = mat.cpu().float().numpy()
                k_use = min(n_top, P - 1)
                eigvals_np, eigvecs_np = sp_eigsh(
                    mat_np, k=k_use, which="LM",
                )
                # Sort by descending absolute eigenvalue
                order = np.argsort(np.abs(eigvals_np))[::-1]
                result[f"{prefix}_spectrum"] = []  # skip full spectrum
                result[f"{prefix}_top_eigenvalues"] = (
                    eigvals_np[order].tolist()
                )
                result[f"{prefix}_top_eigenvectors"] = (
                    eigvecs_np[:, order].tolist()
                )
            else:
                try:
                    eigvals, eigvecs = torch.linalg.eigh(mat)
                except torch.cuda.OutOfMemoryError:
                    eigvals, eigvecs = torch.linalg.eigh(mat.cpu())
                    eigvals = eigvals.to(mat.device)
                    eigvecs = eigvecs.to(mat.device)
                result[f"{prefix}_spectrum"] = (
                    eigvals.cpu().numpy().tolist()
                )
                order = torch.argsort(
                    eigvals.abs(), descending=True,
                )[:n_top]
                result[f"{prefix}_top_eigenvalues"] = (
                    eigvals[order].cpu().numpy().tolist()
                )
                result[f"{prefix}_top_eigenvectors"] = (
                    eigvecs[:, order].cpu().numpy().tolist()
                )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_dataset(
        self, loader: DataLoader
    ) -> tuple[Tensor, Tensor]:
        """Load the entire dataset from *loader* into two tensors."""
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for x, y in loader:
            x, y = self._to_device(x, y)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)
