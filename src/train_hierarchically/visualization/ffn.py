"""Visualization tools for feedforward ridge spectral models.

Provides :class:`FFNVisualizer` — a wrapper around a trained
:class:`~train_hierarchically.models.ridge_spectral.RidgeSpectralModel`
that implements Fourier-parameterized gradient-ascent maximizers and
Jacobian-based pixel importance.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..models.ridge_spectral import RidgeSpectralModel
from ..utils.helpers import ACTIVATIONS
from .fourier import (
    build_decay_mask,
    infer_spatial_shape,
    smooth_spatial,
    spatial_hw,
)


class FFNVisualizer:
    """Visualization helper for a trained :class:`RidgeSpectralModel`.

    Parameters
    ----------
    model : RidgeSpectralModel
        Trained model (weights frozen).
    device : torch.device
        Device for computation.
    """

    def __init__(
        self, model: RidgeSpectralModel, device: torch.device,
    ) -> None:
        self.model = model
        self.device = device

    # ------------------------------------------------------------------
    # Forward with per-layer normalization
    # ------------------------------------------------------------------

    def forward_normalized(
        self, x: Tensor, up_to_layer: int,
    ) -> Tensor:
        """Forward through the model, normalizing hidden reps at each layer.

        After each layer's activation, divides by ``||h||`` so that all
        representations live on the unit sphere.  This stabilises the
        gradient signal for deeper layers.
        """
        h = x
        for i in range(up_to_layer + 1):
            w = self.model.random_projections[i]
            sigma = ACTIVATIONS[self.model.activation_names[i]]
            z = sigma(h @ w)
            z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z = z / z_norm
            if i < up_to_layer:
                h = z @ self.model.eigenvector_projections[i]
            else:
                h = z
        return h

    # ------------------------------------------------------------------
    # Fourier-parameterized gradient-ascent maximizer
    # ------------------------------------------------------------------

    def fourier_maximizer(
        self,
        v: Tensor,
        up_to_layer: int,
        input_dim: int,
        *,
        n_steps: int = 1000,
        lr: float = 0.05,
        fourier_decay: float = 1.0,
        fourier_alpha: float = 1.0,
        n_restarts: int = 5,
        normalize_hidden: bool = False,
    ) -> np.ndarray:
        """Find an input image that maximally activates direction *v*.

        Optimizes in Fourier space: ``x = IRFFT(z · mask)`` where *mask*
        is a frequency-dependent decay that suppresses high frequencies
        by construction.

        All restarts are **batched** into a single forward pass for
        efficiency on GPU.

        Uses **cosine similarity** ``v^T h / ||h||`` as the objective
        (scale-invariant, safe with or without ``normalize_hidden``) and
        **gradient normalization** for uniform step sizes across
        frequencies.

        Parameters
        ----------
        v : Tensor
            Target direction in hidden space (P-dimensional, unit norm).
        up_to_layer : int
            Layer index (0-based).
        input_dim : int
            Flattened input dimensionality.
        n_steps : int
            Optimization steps per restart.
        lr : float
            Adam learning rate.
        fourier_decay : float
            ``f0`` — frequency scale for the decay mask.
        fourier_alpha : float
            Decay exponent (higher = smoother).
        n_restarts : int
            Number of random restarts (batched, best is kept).
        normalize_hidden : bool
            If True, normalize hidden representations at each layer.

        Returns
        -------
        ndarray
            Flat array ``(input_dim,)``.  Normalized to ``[0, 1]`` for
            colour images, raw values for grayscale.
        """
        shape = infer_spatial_shape(input_dim)
        C, H, W = spatial_hw(shape)
        is_color = C == 3
        Wh = W // 2 + 1
        R = n_restarts

        decay_mask = build_decay_mask(H, W, fourier_decay, fourier_alpha, self.device)

        # Batched restarts: z has shape (R, C, H, Wh)
        torch.manual_seed(0)
        z_real = torch.randn(R, C, H, Wh, device=self.device, requires_grad=True)
        z_imag = torch.randn(R, C, H, Wh, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([z_real, z_imag], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.05,
        )

        for _step in range(n_steps):
            optimizer.zero_grad()

            # (R, C, H, W) → (R, D)
            z = torch.complex(z_real, z_imag) * decay_mask  # broadcast mask
            x = torch.fft.irfft2(z, s=(H, W)).reshape(R, -1)

            if normalize_hidden:
                h = self.forward_normalized(x, up_to_layer)  # (R, P)
            else:
                h = self.model.forward_to_layer(
                    x, up_to_layer, project=False,
                )  # (R, P)

            # Per-restart cosine similarity: (R,)
            h_norms = h.norm(dim=-1).clamp(min=1e-8)
            objectives = (h @ v) / h_norms  # (R,)

            # Maximize the sum (each restart contributes independently)
            loss = -objectives.sum()
            loss.backward()

            # Per-restart gradient normalization
            with torch.no_grad():
                gr, gi = z_real.grad, z_imag.grad
                if gr is not None and gi is not None:
                    # Norm per restart: (R,)
                    g_norms = torch.sqrt(
                        gr.reshape(R, -1).pow(2).sum(dim=-1)
                        + gi.reshape(R, -1).pow(2).sum(dim=-1),
                    ).clamp(min=1e-8)
                    # Reshape for broadcasting: (R, 1, 1, 1)
                    g_norms = g_norms.view(R, 1, 1, 1)
                    z_real.grad = gr / g_norms
                    z_imag.grad = gi / g_norms

            optimizer.step()
            scheduler.step()

        # Pick best restart
        with torch.no_grad():
            z = torch.complex(z_real, z_imag) * decay_mask
            x_all = torch.fft.irfft2(z, s=(H, W)).reshape(R, -1)  # (R, D)
            h_all = self.model.forward_to_layer(
                x_all, up_to_layer, project=False,
            )  # (R, P)
            h_norms = h_all.norm(dim=-1).clamp(min=1e-8)
            final_objs = (h_all @ v) / h_norms  # (R,)
            best_idx = int(final_objs.argmax())

            x_best = x_all[best_idx].cpu().numpy()
            if is_color:
                lo, hi = float(x_best.min()), float(x_best.max())
                if hi > lo:
                    x_best = (x_best - lo) / (hi - lo)
                else:
                    x_best = np.zeros_like(x_best)

        return x_best

    # ------------------------------------------------------------------
    # Jacobian-based pixel importance (core + public methods)
    # ------------------------------------------------------------------

    def _jacobian_importance_raw(
        self,
        X: Tensor,
        up_to_layer: int,
        eigenvectors: np.ndarray,
        n_use: int,
        *,
        batch_size: int = 256,
    ) -> list[Tensor]:
        """Core Jacobian loop shared by importance methods.

        For each of the first *n_use* eigenvectors, computes
        ``Σ_i |J(x_i)^T v_k|`` (un-normalised) over all samples.

        Parameters
        ----------
        X : Tensor
            Input data ``(N, D)`` — may be a single sample or a batch.
        up_to_layer : int
            Layer index (0-based).
        eigenvectors : ndarray
            ``(P, n_top)`` eigenvectors in hidden space.
        n_use : int
            Number of eigenvectors to process.
        batch_size : int
            Batch size for forward/backward passes.

        Returns
        -------
        list[Tensor]
            One ``(D,)`` tensor per eigenvector, containing the
            accumulated ``|grad|`` across all samples in *X*.
        """
        D = X.shape[1]
        N = X.shape[0]
        accumulators: list[Tensor] = [
            torch.zeros(D, device=self.device) for _ in range(n_use)
        ]

        for k_idx in range(n_use):
            v = torch.tensor(
                eigenvectors[:, k_idx], dtype=torch.float32,
                device=self.device,
            )
            for i in range(0, N, batch_size):
                batch = X[i: i + batch_size].detach().requires_grad_(True)
                h = self.model.forward_to_layer(
                    batch, up_to_layer, project=False,
                )
                scalar = (h @ v).sum()
                scalar.backward()
                accumulators[k_idx] += batch.grad.abs().sum(dim=0)
                batch.grad = None

        return accumulators

    def jacobian_importance(
        self,
        X: Tensor,
        up_to_layer: int,
        eigenvectors: np.ndarray,
        n_top: int,
        *,
        batch_size: int = 256,
        fourier_decay: float = 1.0,
        fourier_alpha: float = 1.0,
    ) -> dict[str, list[float]]:
        """Per-eigenvector pixel importance via Jacobian back-propagation.

        Computes ``(1/N) Σ_i |J(x_i)^T v_k|`` for each eigenvector,
        then applies a Fourier low-pass filter for spatial smoothing.

        Parameters
        ----------
        X : Tensor
            Input data ``(N, D)``.
        up_to_layer : int
            Layer index (0-based).
        eigenvectors : ndarray
            ``(P, n_top)`` eigenvectors in hidden space.
        n_top : int
            Number of eigenvectors to use.
        batch_size : int
            Batch size for forward/backward passes.
        fourier_decay : float
            ``f0`` for post-hoc Fourier smoothing.
        fourier_alpha : float
            Decay exponent for smoothing.

        Returns
        -------
        dict[str, list[float]]
            Per-eigenvector importance maps (``"k0"``, ``"k1"``, …)
            and their mean (``"mean"``).
        """
        D = X.shape[1]
        N = X.shape[0]
        n_use = min(n_top, eigenvectors.shape[1])
        shape = infer_spatial_shape(D)

        accumulators = self._jacobian_importance_raw(
            X, up_to_layer, eigenvectors, n_use, batch_size=batch_size,
        )

        result: dict[str, list[float]] = {}
        mean_importance = torch.zeros(D, device=self.device)

        for k_idx in range(n_use):
            imp_k = accumulators[k_idx] / N
            imp_np = smooth_spatial(
                imp_k.detach().cpu().numpy(), shape,
                fourier_decay, fourier_alpha,
            )
            imp_np = np.abs(imp_np)
            result[f"k{k_idx}"] = imp_np.tolist()
            mean_importance += torch.tensor(
                imp_np, device=self.device, dtype=torch.float32,
            )

        mean_importance /= n_use
        result["mean"] = mean_importance.detach().cpu().numpy().tolist()
        return result

    # ------------------------------------------------------------------
    # Data-driven feature visualization
    # ------------------------------------------------------------------

    def projection_weighted_mean(
        self,
        X: Tensor,
        up_to_layer: int,
        eigenvectors: np.ndarray,
        n_top: int,
        *,
        batch_size: int = 512,
    ) -> dict[str, list[float]]:
        """Projection-weighted mean images for each eigenvector.

        For eigenvector *v_k*, computes
        ``x_feat_k = (p_k^T @ X) / sum(|p_k|)`` where
        ``p_k[i] = v_k^T Z_l(x_i)`` is the projection of the hidden
        representation onto *v_k*.

        This signed image shows which input pixels correlate with
        positive vs negative projection onto the eigenvector.

        Parameters
        ----------
        X : Tensor
            Input data ``(N, D)``.
        up_to_layer : int
            Layer index (0-based).
        eigenvectors : ndarray
            ``(P, n_top)`` eigenvectors in hidden space.
        n_top : int
            Number of eigenvectors to compute.
        batch_size : int
            Batch size for forward passes.

        Returns
        -------
        dict[str, list[float]]
            Keys ``"k0"``, ``"k1"``, …, ``"mean"`` — each a flat
            ``(D,)`` array of pixel values.
        """
        D = X.shape[1]
        N = X.shape[0]
        n_use = min(n_top, eigenvectors.shape[1])

        V = torch.tensor(
            eigenvectors[:, :n_use], dtype=torch.float32,
            device=self.device,
        )  # (P, n_use)

        # Accumulators: numerator[k, d] = Σ_i p_ki * x_id
        #               denom[k] = Σ_i |p_ki|
        numerator = torch.zeros(n_use, D, device=self.device)
        denominator = torch.zeros(n_use, device=self.device)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                x_batch = X[i: i + batch_size]  # (B, D)
                z_batch = self.model.forward_to_layer(
                    x_batch, up_to_layer, project=False,
                )  # (B, P)
                proj = z_batch @ V  # (B, n_use)
                # numerator += proj^T @ x_batch → (n_use, D)
                numerator += proj.T @ x_batch
                denominator += proj.abs().sum(dim=0)

        result: dict[str, list[float]] = {}
        mean_img = torch.zeros(D, device=self.device)
        for k in range(n_use):
            denom_k = denominator[k].clamp(min=1e-8)
            feat_k = numerator[k] / denom_k
            result[f"k{k}"] = feat_k.cpu().numpy().tolist()
            mean_img += feat_k
        mean_img /= n_use
        result["mean"] = mean_img.cpu().numpy().tolist()
        return result

    def top_bottom_activating(
        self,
        X: Tensor,
        up_to_layer: int,
        eigenvectors: np.ndarray,
        n_top: int,
        *,
        M: int = 50,
        batch_size: int = 512,
    ) -> dict[str, dict[str, list[float]]]:
        """Top/bottom activating image averages for each eigenvector.

        For each eigenvector *v_k*, finds the *M* training images with
        the highest and lowest projections, averages them, and computes
        the difference.

        Parameters
        ----------
        X : Tensor
            Input data ``(N, D)``.
        up_to_layer : int
            Layer index (0-based).
        eigenvectors : ndarray
            ``(P, n_top)`` eigenvectors in hidden space.
        n_top : int
            Number of eigenvectors to compute.
        M : int
            Number of top/bottom images to average.
        batch_size : int
            Batch size for forward passes.

        Returns
        -------
        dict[str, dict[str, list[float]]]
            Keys ``"k0"``, ``"k1"``, … — each a dict with
            ``"top"``, ``"bottom"``, ``"diff"`` flat arrays.
        """
        N = X.shape[0]
        n_use = min(n_top, eigenvectors.shape[1])

        V = torch.tensor(
            eigenvectors[:, :n_use], dtype=torch.float32,
            device=self.device,
        )

        # First pass: compute all projections (small memory: N × n_use)
        all_proj = torch.zeros(N, n_use, device=self.device)
        with torch.no_grad():
            for i in range(0, N, batch_size):
                x_batch = X[i: i + batch_size]
                z_batch = self.model.forward_to_layer(
                    x_batch, up_to_layer, project=False,
                )
                all_proj[i: i + batch_size] = z_batch @ V

        X_cpu = X.cpu()
        all_proj_cpu = all_proj.cpu()
        M_use = min(M, N // 2)

        result: dict[str, dict[str, list[float]]] = {}
        for k in range(n_use):
            proj_k = all_proj_cpu[:, k]
            top_idx = proj_k.argsort(descending=True)[:M_use]
            bot_idx = proj_k.argsort()[:M_use]

            top_mean = X_cpu[top_idx].mean(dim=0).numpy()
            bot_mean = X_cpu[bot_idx].mean(dim=0).numpy()
            diff = (top_mean - bot_mean)

            result[f"k{k}"] = {
                "top": top_mean.tolist(),
                "bottom": bot_mean.tolist(),
                "diff": diff.tolist(),
            }
        return result

    # ------------------------------------------------------------------
    # Per-sample importance
    # ------------------------------------------------------------------

    def per_sample_importance(
        self,
        samples: Tensor,
        up_to_layer: int,
        eigenvectors: np.ndarray,
        n_top: int,
        *,
        fourier_decay: float = 0.15,
        fourier_alpha: float = 3.0,
    ) -> list[np.ndarray]:
        """Mean Jacobian importance for individual samples.

        For each sample, computes the mean (across eigenvectors) of
        ``|J(x)^T v_k|`` — the pixel-level sensitivity to the
        projection onto each eigenvector.  Applies Fourier low-pass
        smoothing to suppress high-frequency noise.

        Parameters
        ----------
        samples : Tensor
            Input samples ``(S, D)`` — a small number of images.
        up_to_layer : int
            Layer index (0-based).
        eigenvectors : ndarray
            ``(P, n_top)`` eigenvectors.
        n_top : int
            Number of eigenvectors to average over.
        fourier_decay : float
            ``f0`` for Fourier low-pass smoothing.
        fourier_alpha : float
            Decay exponent for smoothing.

        Returns
        -------
        list[ndarray]
            One ``(D,)`` importance array per sample.
        """
        D = samples.shape[1]
        n_use = min(n_top, eigenvectors.shape[1])
        shape = infer_spatial_shape(D)
        results: list[np.ndarray] = []

        for si in range(samples.shape[0]):
            x_single = samples[si: si + 1]  # (1, D)
            # Reuse core loop: one sample, batch_size=1
            accumulators = self._jacobian_importance_raw(
                x_single, up_to_layer, eigenvectors, n_use,
                batch_size=1,
            )
            # Mean across eigenvectors (each accumulator has N=1 sample)
            imp_sum = sum(accumulators, torch.zeros(D, device=self.device))
            imp_sum /= n_use
            imp_np = np.abs(smooth_spatial(
                imp_sum.detach().cpu().numpy(), shape,
                fourier_decay, fourier_alpha,
            ))
            results.append(imp_np)

        return results
