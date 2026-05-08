"""Tests for the unified :class:`KernelSpectralTrainer`.

Covers the ``do_eigenreduction`` dispatch and the n×n whitened
signed-covariance eigenreduction implemented by
:func:`spectral_signed_eig`:

- All layers with ``do_eigenreduction=False`` run the standard deep-NNGP
  path and produce the same ``test_mse`` (to float-round-off) as a
  direct call to :func:`deep_nngp_kernel` + kernel ridge.
- Eigenreduction layers honour the spec's invariants
  (``A^T G A ≈ I_m``, ``F = G A``) and collapse to deep NNGP in the
  sanity limit ``m = n - 1``.
- The fitted model's ``forward(x_test)`` recomputes the kernel chain
  and matches the trainer's test predictions in every dispatch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_hierarchically.models.kernel_spectral import (
    KernelLayerSpec,
    KernelSpectralModel,
)
from train_hierarchically.training.deep_nngp import deep_nngp_kernel
from train_hierarchically.training.kernel_ridge import (
    kernel_ridge_cv,
    kernel_ridge_predict,
)
from train_hierarchically.training.kernel_spectral import (
    KernelSpectralConfig,
    KernelSpectralTrainer,
    spectral_signed_eig,
)


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _toy_loaders(
    n_train: int = 60, n_test: int = 40, d: int = 16, seed: int = 0
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    gen = _rng(seed)
    x_tr = torch.randn(n_train, d, generator=gen)
    y_tr = torch.where(torch.randn(n_train, generator=gen) > 0, 1.0, -1.0)
    x_te = torch.randn(n_test, d, generator=gen)
    y_te = torch.where(torch.randn(n_test, generator=gen) > 0, 1.0, -1.0)
    tr = DataLoader(list(zip(x_tr, y_tr)), batch_size=32, shuffle=False)
    te = DataLoader(list(zip(x_te, y_te)), batch_size=32, shuffle=False)
    return tr, te, x_te, y_te


class TestAllNoEigenreduction:
    """All-False dispatch: trainer matches standalone deep_nngp_kernel."""

    def test_1layer(self) -> None:
        tr, te, x_te, y_te = _toy_loaders(n_train=80, n_test=40, d=10)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.2,
                n_quad=15, do_eigenreduction=False,
            ),
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)

        # Reference: direct deep-NNGP kernel + standard kernel-ridge with
        # LOO-CV alpha selection — must match the trainer to float rounding.
        x_tr = torch.cat([b[0] for b in tr])
        y_tr = torch.cat([b[1] for b in tr])
        k_train, k_cross = deep_nngp_kernel(
            model.kernels, x_tr, x_te, n_layers=1,
        )
        fit = kernel_ridge_cv(
            k_train.numpy().astype(np.float32),
            y_tr.numpy(),
            alphas=np.logspace(-6, 6, 500),
        )
        y_pred_ref = kernel_ridge_predict(fit, k_cross.numpy())
        test_mse_ref = float(np.mean((y_te.numpy() - y_pred_ref) ** 2))
        assert abs(results["final"]["test_mse"] - test_mse_ref) < 1e-5

    def test_3layers(self) -> None:
        tr, te, _x_te, _y_te = _toy_loaders(n_train=80, n_test=40, d=10)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=False,
            )
            for _ in range(3)
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        # Sanity: test_mse finite and non-negative.
        assert 0 <= results["final"]["test_mse"] < 10.0
        assert results["layers"][0]["kind"] == "deep_nngp"
        assert len(results["layers"]) == 3

    def test_forward_matches_trainer_predictions(self) -> None:
        tr, te, x_te, y_te = _toy_loaders(n_train=60, n_test=30, d=8)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.1,
                n_quad=15, do_eigenreduction=False,
            )
            for _ in range(2)
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)

        # forward() must recompute the same test predictions.
        y_pred = model(x_te)
        mse = float(((y_pred.numpy() - y_te.numpy()) ** 2).mean())
        assert abs(mse - results["final"]["test_mse"]) < 1e-5


class TestEigenreductionWorks:
    """Eigenreduction layers now work; verify basic correctness."""

    def test_all_eigen_produces_results(self) -> None:
        tr, te, *_ = _toy_loaders(n_train=60, n_test=30, d=10)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=15, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            )
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        assert 0 <= results["final"]["test_mse"] < 10.0
        assert results["layers"][0]["kind"] == "kernel_eigen"

    def test_mixed_eigen_noeigen(self) -> None:
        """2-layer: eigen + no-eigen (the 'Optimal NLoFi' ending)."""
        tr, te, x_te, y_te = _toy_loaders(n_train=60, n_test=30, d=10)
        specs = [
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
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        assert results["layers"][0]["kind"] == "kernel_eigen"
        assert results["layers"][1]["kind"] == "deep_nngp"
        assert 0 <= results["final"]["test_mse"] < 10.0

    def test_mixed_noeigen_eigen(self) -> None:
        """2-layer: no-eigen + eigen (kernel→eigen transition)."""
        tr, te, x_te, y_te = _toy_loaders(n_train=60, n_test=30, d=10)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=False,
            ),
            KernelLayerSpec(
                kernel="sigmoid", m=15, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        assert results["layers"][0]["kind"] == "deep_nngp"
        assert results["layers"][1]["kind"] == "kernel_eigen"
        assert 0 <= results["final"]["test_mse"] < 10.0

    def test_forward_matches_1layer_eigen(self) -> None:
        tr, te, x_te, y_te = _toy_loaders(n_train=80, n_test=40, d=12)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=20, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            )
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        y_pred = model(x_te)
        mse = float(((y_pred.numpy() - y_te.numpy()) ** 2).mean())
        assert abs(mse - results["final"]["test_mse"]) < 1e-4

    def test_forward_matches_2layer_eigen_eigen(self) -> None:
        """2-layer eigen+eigen: the spec's multilayer projection pipeline."""
        tr, te, x_te, y_te = _toy_loaders(n_train=80, n_test=40, d=12)
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=20, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
            KernelLayerSpec(
                kernel="sigmoid", m=15, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=True,
            ),
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-4)
        )
        _, results = trainer.fit(tr, test_loader=te)
        y_pred = model(x_te)
        mse = float(((y_pred.numpy() - y_te.numpy()) ** 2).mean())
        assert abs(mse - results["final"]["test_mse"]) < 1e-4


class TestSpecInvariants:
    """Directly check the spec's Section 3.2 invariants for a single layer.

    These are pure ``spectral_signed_eig`` properties — exercised on a
    synthetic Gram matrix without running the full trainer.
    """

    @staticmethod
    def _random_gram(n: int = 50, d: int = 30, seed: int = 0) -> np.ndarray:
        """Build a PSD Gram from i.i.d. Gaussian data (rank min(n, d))."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d)).astype(np.float32)
        g = (x @ x.T) / d
        return 0.5 * (g + g.T)

    def test_a_is_kernel_orthonormal(self) -> None:
        """``A^T G A ≈ I_m`` (spec Section 3.2 Step 4)."""
        g = self._random_gram(n=50, d=40, seed=1)
        y = np.where(np.random.default_rng(2).standard_normal(50) > 0, 1.0, -1.0)
        lam, a, f = spectral_signed_eig(g, y.astype(np.float32), k=10, eps=1e-6)
        gram = a.T @ g @ a
        assert np.allclose(gram, np.eye(10), atol=1e-3), f"|A^T G A - I|={np.abs(gram - np.eye(10)).max()}"
        assert f.shape == (50, 10)
        # Sorted by |lambda|.
        assert np.all(np.diff(np.abs(lam)) <= 1e-6)

    def test_f_equals_g_a(self) -> None:
        """``F = G A`` (spec Section 3.2 Step 5)."""
        g = self._random_gram(n=40, d=30, seed=3)
        y = np.where(np.random.default_rng(4).standard_normal(40) > 0, 1.0, -1.0)
        _, a, f = spectral_signed_eig(g, y.astype(np.float32), k=8, eps=1e-6)
        assert np.allclose(f, g @ a, atol=1e-3)

    def test_sanity_k_equals_n(self) -> None:
        """``F F^T ≈ G`` when ``k = n`` (spec §4.3 sanity limit)."""
        # Use a well-conditioned Gram (d >> n) so all eigenvalues are > eps.
        g = self._random_gram(n=30, d=200, seed=5)
        y = np.where(np.random.default_rng(6).standard_normal(30) > 0, 1.0, -1.0)
        # k = n: B B^T = I, so F F^T = G^{1/2} G^{1/2} = G.
        _, _, f = spectral_signed_eig(g, y.astype(np.float32), k=30, eps=1e-8)
        err = np.abs(f @ f.T - g).max()
        assert err < 1e-3, f"max|F F^T - G| = {err}"
