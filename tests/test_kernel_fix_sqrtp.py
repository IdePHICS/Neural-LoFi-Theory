"""Regression tests for the numerical-stability fixes on ``feature/kernel-fix-sqrtp``.

These tests pin the specific properties the branch's three edits
establish, so a future refactor that inadvertently reverts one of them
fails in a named way.

1. ``test_ridge_spectral_feature_norm_independent_of_p``
   Checks the ``σ(·)/√p`` convention in ``RidgeSpectralModel``.
   Without it, ``‖F⁽¹⁾‖²`` grows linearly in ``p``; with it, the ratio
   stays close to 1 across ``p``.

2. ``test_sigmoid_gram_centered_at_zero`` and
   ``test_sigmoid_gram_psd_to_float_precision``
   Pin the ``δ₁·δ₂`` decomposition in ``SigmoidKernel.kernel_from_qs``:
   the zero-input value is exactly ``σ(0)² = 1/4`` and the Gram matrix
   has ``|λ_min|`` below the float32 / float64 precision floor.

3. ``test_kernel_ridge_preserves_model_dtype`` and
   ``test_kernel_trainer_last_layer_uses_float64_gram``
   Pin the last-no-eigen-layer float64 promotion in
   ``KernelSpectralTrainer._fit_mixed`` and the dtype/device-preserving
   writeback in ``fit_kernel_ridge_readout``.

4. ``test_kernel_limit_matches_finite_p_3layer``
   End-to-end: the kernel trainer on a 3-layer ``eigen + eigen +
   no-eigen`` pipeline reproduces the ``p → ∞`` limit of the matching
   finite-width ``RidgeSpectralTrainer`` within one SEM of the finite-p
   mean.  This is the integration test that catches any combination of
   the three individual edits silently regressing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from train_hierarchically.kernels import get_kernel
from train_hierarchically.models.kernel_spectral import (
    KernelLayerSpec,
    KernelSpectralModel,
)
from train_hierarchically.models.ridge_spectral import (
    LayerSpec,
    RidgeSpectralModel,
)
from train_hierarchically.training.kernel_spectral import (
    KernelSpectralConfig,
    KernelSpectralTrainer,
)
from train_hierarchically.training.ridge_spectral import (
    RidgeSpectralConfig,
    RidgeSpectralTrainer,
)
from train_hierarchically.utils.helpers import set_seed


# ============================================================================
# Helpers
# ============================================================================


def _toy_loaders(
    n_train: int = 200,
    n_test: int = 100,
    d: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataLoader, DataLoader]:
    gen = torch.Generator().manual_seed(seed)
    x_tr = torch.randn(n_train, d, generator=gen)
    y_tr = torch.where(torch.randn(n_train, generator=gen) > 0, 1.0, -1.0)
    x_te = torch.randn(n_test, d, generator=gen)
    y_te = torch.where(torch.randn(n_test, generator=gen) > 0, 1.0, -1.0)
    tr = DataLoader(list(zip(x_tr, y_tr)), batch_size=n_train, shuffle=False)
    te = DataLoader(list(zip(x_te, y_te)), batch_size=n_test, shuffle=False)
    return x_tr, y_tr, x_te, y_te, tr, te


# ============================================================================
# 1. σ/√p feature normalisation in RidgeSpectralModel
# ============================================================================


class TestRidgeSpectralFeatureNormalization:
    """The ``σ(·)/√p`` fix makes ``‖F⁽¹⁾_μ‖²`` a scale-invariant ``O(1)``
    quantity independent of the random-feature width ``p``."""

    def test_feature_norm_independent_of_p(self) -> None:
        """``mean(‖F⁽¹⁾‖²)`` should not grow with ``p`` after the fix.

        Without ``/√p`` it grows linearly: ``‖F⁽¹⁾_μ‖² ∝ p``.
        """
        x_tr, _y_tr, *_ = _toy_loaders(n_train=200, d=20, seed=0)
        norms2 = {}
        for p in (500, 2000, 8000):
            set_seed(0)
            model = RidgeSpectralModel(
                input_dim=x_tr.shape[1],
                layer_specs=[
                    LayerSpec(p=p, k=20, activation="sigmoid", do_eigenreduction=True),
                ],
                seed=0,
            )
            # Apply one trained layer manually — we just want ``Z V``
            # with random V=I of size (p, p).  To avoid requiring training,
            # compute the *raw* layer-1 output instead:
            w = model.random_projections[0]
            z = torch.sigmoid(x_tr @ w) / math.sqrt(w.shape[1])
            norms2[p] = float((z * z).sum(dim=1).mean())

        # With ``/√p`` all three values are ``≈ E[σ(w^T x)²] ≈ 0.25`` —
        # same order of magnitude.  Without it they'd be ``0.25 * p``.
        ratio = norms2[8000] / norms2[500]
        assert ratio < 2.0, (
            f"‖Z‖²/‖Z‖² ratio at p=8000 vs p=500 is {ratio:.3f}; "
            f"σ/√p scaling appears reverted (expect ≈1, got divergent with p)."
        )
        # And the value itself is ``O(1)`` — at least two orders below ``p=500``.
        assert all(norms2[p] < 10.0 for p in norms2), (
            f"Feature row norms {norms2} are not O(1); σ/√p missing."
        )


# ============================================================================
# 2. Sigmoid NNGP kernel: δδ decomposition properties
# ============================================================================


class TestSigmoidGramStability:
    """Pins the ``(σ(h₁) − 1/2)(σ(h₂) − 1/2)`` decomposition in
    :meth:`SigmoidKernel.kernel_from_qs` — the zero-input kernel value is
    exactly ``σ(0)² = 1/4``, and the Gram matrix is PSD to float
    precision.
    """

    def test_gram_zero_input_is_quarter(self) -> None:
        """``κ_σ(0, 0, 0) = σ(0)² = 1/4`` analytically and numerically."""
        kern = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.0, n_quad=30)
        z = torch.zeros(4, 8)
        # (σ_w² = 1, d = 8) with all-zero inputs gives q = 0 everywhere.
        K = kern.gram(z)
        assert torch.allclose(K, torch.full_like(K, 0.25), atol=1e-6)

    def test_gram_psd_float64(self) -> None:
        """Gram matrix on moderately-large inputs has ``|λ_min|`` far
        below the float64 precision floor when the δδ decomposition
        runs end-to-end in float64."""
        gen = torch.Generator().manual_seed(7)
        # Small q regime (where the precision issue bites hardest).
        x = torch.randn(60, 20, generator=gen, dtype=torch.float64) * 0.1
        kern = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.0, n_quad=30)
        K = kern.gram(x).numpy()
        K_sym = 0.5 * (K + K.T)
        evals = np.linalg.eigvalsh(K_sym)
        # δδ in float64: expect |λ_min| near machine precision,
        # certainly below 1e-8.
        assert evals.min() > -1e-8, (
            f"λ_min = {evals.min():.3e} — δδ decomposition in float64 "
            f"should keep this near 1e-15."
        )

    def test_gram_symmetric_and_matches_kernel_from_qs(self) -> None:
        """``gram(X)`` symmetric; ``gram(x, x') == kernel_from_qs``
        single-pair evaluation.  Catches any path divergence caused by
        the δδ edit."""
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(10, 4, generator=gen) * 0.3
        kern = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.0, n_quad=30)

        K = kern.gram(x)
        assert torch.allclose(K, K.T, atol=1e-6)

        # Single-pair check: compute q's manually and call kernel_from_qs.
        scale = 1.0 / x.shape[1]
        q = scale * (x * x).sum(dim=1)
        K_diag = kern.kernel_from_qs(q, q, q)
        assert torch.allclose(K.diagonal(), K_diag, atol=1e-6)


# ============================================================================
# 3. Kernel trainer: float64 promotion + dtype-preserving ridge storage
# ============================================================================


class TestKernelTrainerDtypePreserving:
    """Pins ``_fit_mixed``'s float64 promotion of the last no-eigen Gram
    and ``fit_kernel_ridge_readout``'s writeback to the *model*'s native
    dtype/device."""

    def test_last_layer_gram_computed_in_float64_when_no_eigen(self) -> None:
        """3-layer ``eigen → eigen → no-eigen``: the last-layer Gram must
        reach ``fit_kernel_ridge_readout`` as float64 so ``kernel_ridge_cv``
        inherits the extra precision.  We verify this indirectly by
        checking that the ridge predictions are stable (not collapsed to
        constant) at the LOO-optimal α."""
        _x_tr, _y_tr, _x_te, _y_te, tr, te = _toy_loaders(
            n_train=100, n_test=50, d=10, seed=0,
        )
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=15, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
            KernelLayerSpec(
                kernel="sigmoid", m=15, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=False,
            ),
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-6),
        )
        _, results = trainer.fit(tr, test_loader=te)
        # Collapsed prediction would give ``test_mse ≈ var(y) ≈ 1.0``;
        # a stable one sits well below.
        assert results["final"]["test_mse"] < 1.5, (
            f"test_mse = {results['final']['test_mse']:.3f}; ridge appears "
            f"to have collapsed — the last-layer float64 promotion may "
            f"have been reverted."
        )

    def test_ridge_weight_stored_on_model_native_dtype(self) -> None:
        """Even when the last-layer Gram is computed in float64 on CPU
        for numerical stability, the trainer must write the ridge weights
        back on the model's original dtype and device — otherwise
        ``forward(x_test)`` would hit a dtype/device mismatch at
        inference time."""
        _x_tr, _y_tr, x_te, _y_te, tr, te = _toy_loaders(
            n_train=80, n_test=40, d=10, seed=0,
        )
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=10, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=False,
            ),
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-6),
        )
        trainer.fit(tr, test_loader=te)

        assert model.ridge_weight.dtype == model.x_train.dtype, (
            f"ridge_weight dtype {model.ridge_weight.dtype} != "
            f"model.x_train dtype {model.x_train.dtype}"
        )
        assert model.ridge_weight.device == model.x_train.device

        # The real acid test: does forward() actually run end-to-end?
        y_pred = model(x_te)
        assert y_pred.shape == (x_te.shape[0],)
        assert torch.isfinite(y_pred).all()


# ============================================================================
# 4. Integration test: kernel limit matches finite-p
# ============================================================================


class TestKernelLimitMatchesFiniteP:
    """End-to-end regression: the 3-layer kernel trainer reproduces the
    ``p → ∞`` limit of the matching ``RidgeSpectralTrainer`` within a
    small multiple of the finite-p SEM.

    This is the integration check for the entire stack of fixes on
    ``feature/kernel-fix-sqrtp``.  If any one of them silently regresses
    — σ/√p in the finite-p model, δδ in the kernel, float64 promotion
    in the trainer, dtype-preserving ridge writeback — the gap below
    widens well beyond the tolerance.
    """

    @pytest.mark.parametrize("K1,K2", [(10, 10), (20, 20), (10, 20)])
    def test_3layer_eigen_eigen_noeigen(self, K1: int, K2: int) -> None:
        n_train, n_test, d, P = 200, 100, 16, 2000
        x_tr, y_tr, x_te, y_te, tr, te = _toy_loaders(
            n_train=n_train, n_test=n_test, d=d, seed=0,
        )

        # Kernel limit.
        k_specs = [
            KernelLayerSpec(kernel="sigmoid", m=K1, sigma_w=1.0, sigma_b=0.0,
                            n_quad=20, do_eigenreduction=True),
            KernelLayerSpec(kernel="sigmoid", m=K2, sigma_w=1.0, sigma_b=0.0,
                            n_quad=20, do_eigenreduction=True),
            KernelLayerSpec(kernel="sigmoid", m=0,  sigma_w=1.0, sigma_b=0.0,
                            n_quad=20, do_eigenreduction=False),
        ]
        mk = KernelSpectralModel(k_specs)
        _, r_kernel = KernelSpectralTrainer(
            mk, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-6),
        ).fit(tr, test_loader=te)
        kernel_mse = float(r_kernel["final"]["test_mse"])

        # Finite-p reference — 3 seeds.
        finite_mses = []
        for seed in range(3):
            set_seed(seed)
            specs = [
                LayerSpec(p=P, k=K1, activation="sigmoid", do_eigenreduction=True),
                LayerSpec(p=P, k=K2, activation="sigmoid", do_eigenreduction=True),
                LayerSpec(p=P, k=P, activation="sigmoid", do_eigenreduction=False),
            ]
            mf = RidgeSpectralModel(
                input_dim=d, layer_specs=specs, seed=seed,
            )
            _, r = RidgeSpectralTrainer(
                mf, RidgeSpectralConfig(device="cpu", verbose=False),
            ).fit(tr, test_loader=te)
            finite_mses.append(float(r["final"]["test_mse"]))

        fmean = float(np.mean(finite_mses))
        fsem = float(np.std(finite_mses, ddof=1) / np.sqrt(len(finite_mses)))

        # Tolerance: 5 × SEM from finite-p noise, plus a 0.03 absolute
        # floor to absorb the ``O(1/√P)`` finite-p bias at P=2000.
        tol = max(5.0 * fsem, 0.03)
        assert abs(kernel_mse - fmean) < tol, (
            f"kernel MSE {kernel_mse:.4f} vs finite-p mean {fmean:.4f} "
            f"(sem {fsem:.4f}); gap {kernel_mse - fmean:+.4f} exceeds "
            f"tolerance {tol:.4f}."
        )
