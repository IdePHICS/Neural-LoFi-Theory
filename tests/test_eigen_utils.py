"""Tests for training._eigen_utils — signed covariance eigendecomposition."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from train_hierarchically.training._eigen_utils import (
    _signed_covariance_eigen_dense,
    _signed_covariance_eigen_linop,
    signed_covariance_eigen,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _random_z_y(
    n: int, p: int, *, seed: int = 42, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, p, generator=gen, dtype=dtype)
    y = torch.where(
        torch.randn(n, generator=gen) > 0,
        torch.ones(n, dtype=dtype),
        -torch.ones(n, dtype=dtype),
    )
    return z, y


def _reference_eigen(
    z: torch.Tensor, y: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth via full torch.linalg.eigh."""
    n = z.shape[0]
    y_flat = y.view(n).to(z.dtype)
    c = ((y_flat.unsqueeze(1) * z).T @ z) / n
    eigvals, eigvecs = torch.linalg.eigh(c)
    order = torch.argsort(eigvals.abs(), descending=True)[:k]
    return eigvals[order], eigvecs[:, order]


def _make_train_test_data(
    n: int, d: int, *, seed: int = 0, n_test: int = 200
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    y = torch.where(torch.randn(n, generator=gen) > 0, 1.0, -1.0)
    gen_t = torch.Generator().manual_seed(seed + 99)
    X_t = torch.randn(n_test, d, generator=gen_t)
    y_t = torch.where(torch.randn(n_test, generator=gen_t) > 0, 1.0, -1.0)
    return X, y, X_t, y_t


# ------------------------------------------------------------------ #
# Dense vs LinOp equivalence
# ------------------------------------------------------------------ #


class TestDenseLinopEquivalence:
    @pytest.mark.parametrize("P", [50, 200, 1000])
    def test_varying_p(self, P: int) -> None:
        z, y = _random_z_y(1000, P)
        k = min(5, P - 1)
        vals_d, vecs_d = _signed_covariance_eigen_dense(z, y, k)
        vals_l, vecs_l = _signed_covariance_eigen_linop(z, y, k)

        torch.testing.assert_close(vals_d, vals_l, rtol=1e-4, atol=1e-6)
        cos = torch.abs(torch.sum(vecs_d * vecs_l, dim=0))
        assert (cos > 0.999).all(), f"Cosine similarities: {cos}"

    @pytest.mark.parametrize("k", [1, 5, 20])
    def test_varying_k(self, k: int) -> None:
        z, y = _random_z_y(1000, 200)
        vals_d, vecs_d = _signed_covariance_eigen_dense(z, y, k)
        vals_l, vecs_l = _signed_covariance_eigen_linop(z, y, k)

        torch.testing.assert_close(vals_d, vals_l, rtol=1e-4, atol=1e-6)
        cos = torch.abs(torch.sum(vecs_d * vecs_l, dim=0))
        assert (cos > 0.999).all()

    def test_mixed_sign_eigenvalues(self) -> None:
        """Signed covariance can have both positive and negative eigenvalues.

        Construct Z where col 0 is correlated with y (positive eigenvalue)
        and col 1 is anti-correlated with y (negative eigenvalue).
        Both paths must recover the same mixed-sign spectrum.
        """
        gen = torch.Generator().manual_seed(333)
        n, P = 800, 30
        y = torch.sign(torch.randn(n, generator=gen)).float()

        z1 = y.unsqueeze(1) + 0.1 * torch.randn(n, 1, generator=gen)
        z2 = -y.unsqueeze(1) + 0.1 * torch.randn(n, 1, generator=gen)
        z_rest = 0.1 * torch.randn(n, P - 2, generator=gen)
        z = torch.cat([z1, z2, z_rest], dim=1).float()

        vals_d, vecs_d = _signed_covariance_eigen_dense(z, y, k=5)
        vals_l, vecs_l = _signed_covariance_eigen_linop(z, y, k=5)

        # Top two eigenvalues should have opposite signs
        assert vals_d[0] * vals_d[1] < 0, "Expected mixed-sign top eigenvalues"

        # Both paths agree
        torch.testing.assert_close(vals_d, vals_l, rtol=1e-4, atol=1e-6)
        cos = torch.abs(torch.sum(vecs_d * vecs_l, dim=0))
        assert (cos > 0.999).all()

        # Verify against full eigh reference
        ref_vals, _ = _reference_eigen(z, y, k=5)
        torch.testing.assert_close(vals_d, ref_vals, rtol=1e-4, atol=1e-6)

    def test_eigh_branch_keeps_largest_magnitude(self) -> None:
        """Eigh branch (k >= P//4) must rank by |λ|, not algebraically.

        Construct a class-conditional-variance feature so the signed
        covariance has eigenvalues of both signs comparable in magnitude.
        Then dropping the most-negative one (algebraic top-k) versus the
        smallest-|λ| one (correct top-k) gives observably different results.
        """
        gen = torch.Generator().manual_seed(2024)
        n, P = 600, 40
        y = torch.sign(torch.randn(n, generator=gen)).float()
        pos = (y == 1).float().unsqueeze(1)
        neg = (y == -1).float().unsqueeze(1)

        # Direction 0: large variance only on y=-1 → strong negative λ
        z0 = 0.05 * pos * torch.randn(n, 1, generator=gen) + 5.0 * neg * torch.randn(
            n, 1, generator=gen
        )
        # Directions 1..P-1: variance excess on y=+1 → positive λs
        z_rest = 2.0 * pos * torch.randn(
            n, P - 1, generator=gen
        ) + 1.0 * neg * torch.randn(n, P - 1, generator=gen)
        z = torch.cat([z0, z_rest], dim=1).float()

        # k = P-1 forces the eigh branch (k >> P//4).
        k = P - 1
        vals, _ = _signed_covariance_eigen_dense(z, y, k=k)

        # Must match abs-sorted reference (top-k by |λ|).
        ref_vals, _ = _reference_eigen(z, y, k=k)
        torch.testing.assert_close(vals, ref_vals, rtol=1e-4, atol=1e-6)

        # Spectrum must contain both signs (otherwise the bug is silent).
        assert (vals > 0).any() and (vals < 0).any(), (
            f"Test setup didn't produce mixed-sign top-k: {vals[:5].tolist()}"
        )

        # Demonstrate the buggy `[-k:]` would have given a different result:
        # compute the full spectrum and compare what the two strategies keep.
        y_flat = y.view(n).float()
        c = ((y_flat.unsqueeze(1) * z).T @ z) / n
        all_vals_asc = np.sort(np.linalg.eigh(c.numpy())[0])
        buggy_kept = np.sort(all_vals_asc[-k:])
        correct_kept = np.sort(np.abs(all_vals_asc).argsort()[::-1][:k])
        # If buggy and correct top-k are the same set, the test is vacuous.
        buggy_vals = all_vals_asc[-k:]
        correct_vals = all_vals_asc[np.argsort(np.abs(all_vals_asc))[::-1][:k]]
        assert not np.allclose(np.sort(buggy_vals), np.sort(correct_vals)), (
            "Buggy and correct top-k coincide on this seed — test is vacuous"
        )

        assert torch.all(torch.isfinite(vals))


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_k_equals_1(self) -> None:
        z, y = _random_z_y(200, 50)
        vals, vecs = signed_covariance_eigen(z, y, k=1)
        assert vals.shape == (1,)
        assert vecs.shape == (50, 1)

    def test_k_close_to_p(self) -> None:
        """k = P-2 should force the dense path (k > P//2)."""
        P = 30
        z, y = _random_z_y(200, P)
        k = P - 2
        vals, vecs = signed_covariance_eigen(z, y, k=k)
        assert vals.shape == (k,)
        assert vecs.shape == (P, k)

    def test_n_less_than_p(self) -> None:
        z, y = _random_z_y(50, 200)
        vals, vecs = signed_covariance_eigen(z, y, k=5)
        assert vals.shape == (5,)
        assert vecs.shape == (200, 5)

    def test_n_greater_than_p(self) -> None:
        z, y = _random_z_y(500, 30)
        vals, vecs = signed_covariance_eigen(z, y, k=5)
        assert vals.shape == (5,)
        assert vecs.shape == (30, 5)

    def test_all_same_y(self) -> None:
        z, _ = _random_z_y(300, 50)
        y = torch.ones(300)
        vals, vecs = signed_covariance_eigen(z, y, k=3)
        assert vals.shape == (3,)
        assert vecs.shape == (50, 3)

        ref_vals, _ = _reference_eigen(z, y, k=3)
        torch.testing.assert_close(vals, ref_vals, rtol=1e-4, atol=1e-6)

    def test_deterministic_v0(self) -> None:
        z, y = _random_z_y(200, 80)
        v1, e1 = signed_covariance_eigen(z, y, k=5)
        v2, e2 = signed_covariance_eigen(z, y, k=5)
        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(e1, e2)


# ------------------------------------------------------------------ #
# Dispatcher logic (routing + boundary conditions)
# ------------------------------------------------------------------ #


class TestDispatcher:
    def test_small_p_uses_dense(self) -> None:
        z, y = _random_z_y(200, 100)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_dense",
            wraps=_signed_covariance_eigen_dense,
        ) as mock_dense:
            signed_covariance_eigen(z, y, k=5)
            mock_dense.assert_called_once()

    def test_large_k_uses_dense(self) -> None:
        z, y = _random_z_y(200, 100)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_dense",
            wraps=_signed_covariance_eigen_dense,
        ) as mock_dense:
            signed_covariance_eigen(z, y, k=60)
            mock_dense.assert_called_once()

    def test_large_p_small_k_uses_linop(self) -> None:
        z, y = _random_z_y(200, 10000)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_linop",
            wraps=_signed_covariance_eigen_linop,
        ) as mock_linop:
            signed_covariance_eigen(z, y, k=10)
            mock_linop.assert_called_once()

    def test_custom_threshold(self) -> None:
        z, y = _random_z_y(200, 100)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_linop",
            wraps=_signed_covariance_eigen_linop,
        ) as mock_linop:
            signed_covariance_eigen(z, y, k=5, linop_threshold=50)
            mock_linop.assert_called_once()

    def test_p_exactly_at_threshold_uses_dense(self) -> None:
        """P == linop_threshold (5000) → dense (<=, not <)."""
        z, y = _random_z_y(200, 5000)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_dense",
            wraps=_signed_covariance_eigen_dense,
        ) as mock_dense:
            signed_covariance_eigen(z, y, k=10)
            mock_dense.assert_called_once()

    def test_p_just_above_threshold_uses_linop(self) -> None:
        """P == linop_threshold + 1 (5001) → linop (when k small)."""
        z, y = _random_z_y(200, 5001)
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_linop",
            wraps=_signed_covariance_eigen_linop,
        ) as mock_linop:
            signed_covariance_eigen(z, y, k=10)
            mock_linop.assert_called_once()

    def test_k_equals_p_half_uses_linop(self) -> None:
        """k == P//2 stays in linop (condition is k > P//2, strict)."""
        P = 100
        z, y = _random_z_y(200, P)
        k = P // 2  # 50 — not > 50, so linop
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_linop",
            wraps=_signed_covariance_eigen_linop,
        ) as mock_linop:
            signed_covariance_eigen(z, y, k=k, linop_threshold=0)
            mock_linop.assert_called_once()

    def test_k_above_p_half_forces_dense(self) -> None:
        """k == P//2 + 1 → dense even when P > threshold."""
        P = 100
        z, y = _random_z_y(200, P)
        k = P // 2 + 1  # 51 > 50, forces dense
        with patch(
            "train_hierarchically.training._eigen_utils._signed_covariance_eigen_dense",
            wraps=_signed_covariance_eigen_dense,
        ) as mock_dense:
            signed_covariance_eigen(z, y, k=k, linop_threshold=0)
            mock_dense.assert_called_once()


# ------------------------------------------------------------------ #
# Output properties
# ------------------------------------------------------------------ #


class TestOutputProperties:
    def test_shapes(self) -> None:
        P, k = 80, 10
        z, y = _random_z_y(300, P)
        vals, vecs = signed_covariance_eigen(z, y, k=k)
        assert vals.shape == (k,)
        assert vecs.shape == (P, k)

    def test_sorted_by_descending_abs(self) -> None:
        z, y = _random_z_y(300, 80)
        vals, _ = signed_covariance_eigen(z, y, k=10)
        abs_vals = vals.abs()
        assert (abs_vals[:-1] >= abs_vals[1:] - 1e-8).all()

    def test_orthonormality(self) -> None:
        z, y = _random_z_y(300, 80)
        _, vecs = signed_covariance_eigen(z, y, k=10)
        vtv = vecs.T @ vecs
        torch.testing.assert_close(
            vtv, torch.eye(10), rtol=1e-4, atol=1e-4
        )

    def test_dtype_preservation_float32(self) -> None:
        z, y = _random_z_y(200, 80, dtype=torch.float32)
        vals, vecs = signed_covariance_eigen(z, y, k=5)
        assert vals.dtype == torch.float32
        assert vecs.dtype == torch.float32

    def test_dtype_preservation_float64(self) -> None:
        z, y = _random_z_y(200, 80, dtype=torch.float64)
        vals, vecs = signed_covariance_eigen(z, y, k=5)
        assert vals.dtype == torch.float64
        assert vecs.dtype == torch.float64


# ------------------------------------------------------------------ #
# Numerical stability
# ------------------------------------------------------------------ #


class TestNumericalStability:
    def test_float32_vs_float64_consistency(self) -> None:
        """Both dtypes should produce the same top-k eigenstructure."""
        gen = torch.Generator().manual_seed(999)
        z32 = torch.randn(500, 100, generator=gen, dtype=torch.float32)
        y32 = torch.sign(torch.randn(500, generator=gen)).float()

        vals32, vecs32 = signed_covariance_eigen(z32, y32, k=10)
        vals64, vecs64 = signed_covariance_eigen(
            z32.double(), y32.double(), k=10
        )

        torch.testing.assert_close(
            vals32, vals64.float(), rtol=1e-3, atol=1e-6
        )
        cos = torch.abs(torch.sum(vecs32 * vecs64.float(), dim=0))
        assert (cos > 0.998).all(), f"Min cosine: {cos.min():.6f}"

        # No NaN/Inf in either
        assert not torch.isnan(vals32).any()
        assert not torch.isnan(vals64).any()
        assert not torch.isinf(vals32).any()

    def test_scaling_robustness(self) -> None:
        """Eigenvalues should scale quadratically with Z."""
        z, y = _random_z_y(300, 80, seed=777)
        vals_base, _ = signed_covariance_eigen(z, y, k=5)

        # Z * alpha → eigenvalues * alpha^2
        for alpha in [1e3, 1e-3]:
            z_scaled = z * alpha
            vals_scaled, _ = signed_covariance_eigen(z_scaled, y, k=5)
            expected = vals_base * alpha**2
            torch.testing.assert_close(
                vals_scaled, expected, rtol=0.01, atol=1e-10
            )

    def test_rank_deficient_z(self) -> None:
        """Near-zero eigenvalues from rank-deficient Z don't produce NaN."""
        gen = torch.Generator().manual_seed(555)
        n = 500

        # 10 independent columns + 5 near-duplicates + 1 rank-1 block + noise
        rank_part = torch.randn(n, 10, generator=gen)
        redundant = rank_part[:, :5] + 1e-3 * torch.randn(
            n, 5, generator=gen
        )
        lowrank = torch.randn(n, 1, generator=gen) @ torch.randn(
            1, 40, generator=gen
        )
        noise = 1e-4 * torch.randn(n, 45, generator=gen)

        z = torch.cat([rank_part, redundant, lowrank, noise], dim=1).float()
        y = torch.sign(torch.randn(n, generator=gen)).float()

        vals, vecs = signed_covariance_eigen(z, y, k=20)

        assert not torch.isnan(vals).any()
        assert not torch.isnan(vecs).any()
        # Top eigenvalue should dominate the tail
        assert vals[0].abs() > 10 * vals[-1].abs()
        # Still sorted
        assert (vals[:-1].abs() >= vals[1:].abs() - 1e-8).all()

    def test_duplicate_rows_stability(self) -> None:
        """Near-identical rows shouldn't break eigendecomposition."""
        gen = torch.Generator().manual_seed(111)
        P = 60

        pattern_a = torch.randn(P, generator=gen)
        pattern_b = torch.randn(P, generator=gen)
        z_a = pattern_a.unsqueeze(0).repeat(250, 1) + 1e-3 * torch.randn(
            250, P, generator=gen
        )
        z_b = pattern_b.unsqueeze(0).repeat(250, 1) + 1e-3 * torch.randn(
            250, P, generator=gen
        )
        z = torch.cat([z_a, z_b], dim=0).float()
        y = torch.cat([torch.ones(250), -torch.ones(250)]).float()

        vals, vecs = signed_covariance_eigen(z, y, k=5)

        # With near-duplicate rows, the covariance is near-rank-2.
        # Top 2 eigenvalues should be much larger than the rest.
        assert vals[0].abs() > 100 * vals[2].abs()
        assert not torch.isnan(vals).any()

        # Deterministic on re-run
        vals2, vecs2 = signed_covariance_eigen(z, y, k=5)
        torch.testing.assert_close(vals, vals2)
        torch.testing.assert_close(vecs, vecs2)

    def test_orthonormality_under_different_scales(self) -> None:
        """Eigenvector orthonormality should hold at extreme data scales."""
        z_base, y = _random_z_y(400, 100, seed=888)

        for scale in [1e-4, 1.0, 1e4]:
            z = z_base * scale
            _, vecs = signed_covariance_eigen(z, y, k=20)
            gram = vecs.T @ vecs
            identity = torch.eye(20, dtype=vecs.dtype)

            frob_err = (gram - identity).norm()
            assert frob_err < 1e-3, (
                f"Frobenius error {frob_err:.2e} at scale={scale}"
            )

    def test_no_nan_inf_on_extreme_y(self) -> None:
        """y with all +1 or all -1 shouldn't produce numerical issues."""
        z, _ = _random_z_y(300, 80, seed=12)
        for y_val in [1.0, -1.0]:
            y = torch.full((300,), y_val)
            vals, vecs = signed_covariance_eigen(z, y, k=5)
            assert not torch.isnan(vals).any()
            assert not torch.isinf(vals).any()
            assert not torch.isnan(vecs).any()


# ------------------------------------------------------------------ #
# Reproducibility & cross-path consistency
# ------------------------------------------------------------------ #


class TestReproducibility:
    def test_eigen_triple_run_bitwise_identical(self) -> None:
        """Same input → bitwise identical output on every call."""
        z, y = _random_z_y(500, 200, seed=123)
        results = [signed_covariance_eigen(z, y, k=10) for _ in range(3)]
        for i in range(1, 3):
            torch.testing.assert_close(results[0][0], results[i][0])
            torch.testing.assert_close(results[0][1], results[i][1])

    def test_ffn_multi_layer_determinism(self) -> None:
        """Same model seed → identical per-layer eigenvalues and final MSE."""
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.ridge_spectral import (
            LayerSpec,
            RidgeSpectralModel,
        )
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralConfig,
            RidgeSpectralTrainer,
        )

        D, N = 100, 500
        specs = [
            LayerSpec(p=200, k=10, activation="relu"),
            LayerSpec(p=150, k=8, activation="relu"),
        ]
        X, y, _, _ = _make_train_test_data(N, D)
        ds = TensorDataset(X, y)

        results_list = []
        for _ in range(2):
            model = RidgeSpectralModel(
                input_dim=D, layer_specs=specs, seed=0
            )
            config = RidgeSpectralConfig(device="cpu", verbose=False)
            trainer = RidgeSpectralTrainer(model, config)
            _, res = trainer.fit(DataLoader(ds, batch_size=N))
            results_list.append(res)

        for layer_idx in range(len(specs)):
            assert (
                results_list[0]["layers"][layer_idx]["eigenvalues"]
                == results_list[1]["layers"][layer_idx]["eigenvalues"]
            )
        assert (
            results_list[0]["final"]["train_mse"]
            == results_list[1]["final"]["train_mse"]
        )
        assert (
            results_list[0]["final"]["ridge_alpha"]
            == results_list[1]["final"]["ridge_alpha"]
        )

    def test_different_seeds_produce_different_results(self) -> None:
        """Sanity: different model seeds → different eigenvalues."""
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.ridge_spectral import (
            LayerSpec,
            RidgeSpectralModel,
        )
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralConfig,
            RidgeSpectralTrainer,
        )

        D, P, K, N = 100, 300, 10, 500
        spec = LayerSpec(p=P, k=K, activation="relu")
        X, y, _, _ = _make_train_test_data(N, D)
        ds = TensorDataset(X, y)

        eigs_by_seed = []
        for seed in [0, 1]:
            model = RidgeSpectralModel(
                input_dim=D, layer_specs=[spec], seed=seed
            )
            config = RidgeSpectralConfig(device="cpu", verbose=False)
            trainer = RidgeSpectralTrainer(model, config)
            _, res = trainer.fit(DataLoader(ds, batch_size=N))
            eigs_by_seed.append(res["layers"][0]["eigenvalues"])

        assert eigs_by_seed[0] != eigs_by_seed[1]

    def test_dense_vs_linop_extended_equivalence(self) -> None:
        """Dense and linop paths produce matching eigenvalues, eigenvectors,
        ridge alpha, and test metrics."""
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.ridge_spectral import (
            LayerSpec,
            RidgeSpectralModel,
        )
        from train_hierarchically.training import _eigen_utils
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralConfig,
            RidgeSpectralTrainer,
        )

        D, P, K, N = 100, 300, 10, 500
        spec = LayerSpec(p=P, k=K, activation="relu")
        X, y, X_t, y_t = _make_train_test_data(N, D)
        ds = TensorDataset(X, y)
        ds_t = TensorDataset(X_t, y_t)

        def _run(threshold: int) -> dict:
            model = RidgeSpectralModel(
                input_dim=D, layer_specs=[spec], seed=0
            )
            config = RidgeSpectralConfig(device="cpu", verbose=False)
            trainer = RidgeSpectralTrainer(model, config)
            original = _eigen_utils._LINOP_THRESHOLD
            _eigen_utils._LINOP_THRESHOLD = threshold
            try:
                _, res = trainer.fit(
                    DataLoader(ds, batch_size=N),
                    test_loader=DataLoader(ds_t, batch_size=200),
                )
            finally:
                _eigen_utils._LINOP_THRESHOLD = original
            return res

        res_dense = _run(100000)  # force dense
        res_linop = _run(0)  # force linop

        # Eigenvalues match
        eigs_d = torch.tensor(res_dense["layers"][0]["eigenvalues"])
        eigs_l = torch.tensor(res_linop["layers"][0]["eigenvalues"])
        torch.testing.assert_close(eigs_d, eigs_l, rtol=1e-3, atol=1e-6)

        # Ridge alpha matches
        assert res_dense["final"]["ridge_alpha"] == pytest.approx(
            res_linop["final"]["ridge_alpha"], rel=1e-3
        )
        # Test MSE matches
        assert res_dense["final"]["test_mse"] == pytest.approx(
            res_linop["final"]["test_mse"], rel=1e-3
        )
        # Accuracy matches
        assert res_dense["final"]["test_accuracy"] == pytest.approx(
            res_linop["final"]["test_accuracy"], abs=0.02
        )

    def test_cnn_chunked_vs_non_chunked(self) -> None:
        """CNN chunked and non-chunked paths produce consistent results."""
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.cnn_ridge_spectral import (
            CNNLayerSpec,
            CNNRidgeSpectralModel,
        )
        from train_hierarchically.training.cnn_ridge_spectral import (
            CNNRidgeSpectralConfig,
            CNNRidgeSpectralTrainer,
        )

        N, C, H, W = 100, 1, 16, 16
        P, K = 16, 4
        specs = [
            CNNLayerSpec(kind="conv", p=P, k=K, kernel_size=3, padding=1),
            CNNLayerSpec(kind="flatten"),
        ]

        gen = torch.Generator().manual_seed(0)
        X = torch.randn(N, C, H, W, generator=gen)
        y = torch.where(torch.randn(N, generator=gen) > 0, 1.0, -1.0)
        ds = TensorDataset(X, y)

        def _run_cnn(chunk_size: int) -> dict:
            model = CNNRidgeSpectralModel(
                in_channels=C,
                spatial_size=(H, W),
                layer_specs=specs,
                seed=0,
            )
            config = CNNRidgeSpectralConfig(
                device="cpu",
                verbose=False,
                covariance_chunk_size=chunk_size,
            )
            trainer = CNNRidgeSpectralTrainer(model, config)
            _, res = trainer.fit(DataLoader(ds, batch_size=N))
            return res

        res_full = _run_cnn(0)
        res_chunked = _run_cnn(25)

        eigs_full = res_full["layers"][0]["eigenvalues"]
        eigs_chunked = res_chunked["layers"][0]["eigenvalues"]

        for ev_f, ev_c in zip(eigs_full, eigs_chunked):
            assert ev_f == pytest.approx(ev_c, rel=1e-4, abs=1e-6)

        assert res_full["final"]["train_mse"] == pytest.approx(
            res_chunked["final"]["train_mse"], rel=1e-3
        )


# ------------------------------------------------------------------ #
# CNN-specific spatial consistency
# ------------------------------------------------------------------ #


class TestCNNSpatialConsistency:
    def test_spatial_reshaping_matches_manual_covariance(self) -> None:
        """CNN spatial path (N,P,H,W) → (N*H*W,P) produces correct eigen."""
        N, P, H, W = 50, 32, 8, 8
        gen = torch.Generator().manual_seed(77)
        z_spatial = torch.randn(N, P, H, W, generator=gen)
        y = torch.sign(torch.randn(N, generator=gen)).float()

        # Same reshape as CNN trainer
        n_patches = H * W
        z_flat = z_spatial.permute(0, 2, 3, 1).reshape(N * n_patches, P)
        y_flat = y.unsqueeze(1).expand(N, n_patches).reshape(N * n_patches)

        vals, vecs = signed_covariance_eigen(z_flat, y_flat, k=8)

        # Manual reference via full eigh
        ref_vals, ref_vecs = _reference_eigen(z_flat, y_flat, k=8)
        torch.testing.assert_close(vals, ref_vals, rtol=1e-4, atol=1e-6)
        cos = torch.abs(torch.sum(vecs * ref_vecs, dim=0))
        assert (cos > 0.999).all()


# ------------------------------------------------------------------ #
# Backward compatibility
# ------------------------------------------------------------------ #


class TestBackwardCompat:
    def test_static_method_exists_and_works(self) -> None:
        """RidgeSpectralTrainer._signed_covariance_eigen is still callable
        (used by scripts/track_eigvec_alignment.py)."""
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralTrainer,
        )

        assert hasattr(RidgeSpectralTrainer, "_signed_covariance_eigen")

        z, y = _random_z_y(500, 200)
        eigvals, eigvecs = RidgeSpectralTrainer._signed_covariance_eigen(
            z, y, k=15
        )

        assert eigvals.shape == (15,)
        assert eigvecs.shape == (200, 15)
        assert torch.all(torch.isfinite(eigvals))

        ref_vals, _ = _reference_eigen(z, y, k=15)
        torch.testing.assert_close(eigvals, ref_vals, rtol=1e-4, atol=1e-6)

    def test_cnn_eigen_from_covariance_unchanged(self) -> None:
        """CNN chunked path's _eigen_from_covariance still works on
        pre-formed P x P matrices."""
        from train_hierarchically.training.cnn_ridge_spectral import (
            CNNRidgeSpectralTrainer,
        )

        P, k = 50, 10
        gen = torch.Generator().manual_seed(123)
        A = torch.randn(P, P, generator=gen)
        cov = (A @ A.T) / P

        eigvals, eigvecs = CNNRidgeSpectralTrainer._eigen_from_covariance(
            cov, k=k
        )

        assert eigvals.shape == (k,)
        assert eigvecs.shape == (P, k)
        assert (eigvals[:-1].abs() >= eigvals[1:].abs() - 1e-8).all()

        vtv = eigvecs.T @ eigvecs
        torch.testing.assert_close(vtv, torch.eye(k), rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------ #
# Regression (fixed expected values)
# ------------------------------------------------------------------ #


class TestRegression:
    def test_eigenvalues_fixed_seed(self) -> None:
        """Guard against silent algorithm changes.

        Expected values computed once with the current implementation
        (seed=9999, n=1000, P=200, k=5).
        """
        expected_eigenvalues = np.array(
            [0.98016465, -0.9395784, 0.91144675, -0.90372217, 0.8710196],
            dtype=np.float32,
        )
        expected_eigvec_col0_head = np.array(
            [0.06881607, -0.13961461, 0.13304839, 0.17351948, -0.14386578],
            dtype=np.float32,
        )

        z, y = _random_z_y(n=1000, p=200, seed=9999)
        vals, vecs = signed_covariance_eigen(z, y, k=5)

        np.testing.assert_allclose(
            vals.cpu().numpy(),
            expected_eigenvalues,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Eigenvalue regression — algorithm may have changed.",
        )
        np.testing.assert_allclose(
            vecs[:5, 0].cpu().numpy(),
            expected_eigvec_col0_head,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Eigenvector regression — algorithm may have changed.",
        )


# ------------------------------------------------------------------ #
# store_spectra integration
# ------------------------------------------------------------------ #


class TestStoreSpectra:
    def test_store_spectra_produces_spectra_keys(self) -> None:
        """store_spectra=True → layer result contains spectra dict."""
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.ridge_spectral import (
            LayerSpec,
            RidgeSpectralModel,
        )
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralConfig,
            RidgeSpectralTrainer,
        )

        D, P, K, N = 50, 200, 10, 300
        spec = LayerSpec(p=P, k=K, activation="relu")
        model = RidgeSpectralModel(input_dim=D, layer_specs=[spec], seed=0)
        config = RidgeSpectralConfig(
            device="cpu",
            verbose=False,
            store_spectra=True,
            n_top_eigenvectors=4,
        )
        trainer = RidgeSpectralTrainer(model, config)

        X, y, _, _ = _make_train_test_data(N, D)
        ds = TensorDataset(X, y)
        _, results = trainer.fit(DataLoader(ds, batch_size=N))

        layer = results["layers"][0]
        assert "spectra" in layer
        spectra = layer["spectra"]
        assert "signed_cov_top_eigenvalues" in spectra
        assert "cov_top_eigenvalues" in spectra
        assert len(spectra["signed_cov_top_eigenvalues"]) <= 4


# ------------------------------------------------------------------ #
# FFN trainer integration
# ------------------------------------------------------------------ #


class TestFFNIntegration:
    def test_fit_runs(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.ridge_spectral import (
            LayerSpec,
            RidgeSpectralModel,
        )
        from train_hierarchically.training.ridge_spectral import (
            RidgeSpectralConfig,
            RidgeSpectralTrainer,
        )

        D, P, K, N = 100, 300, 10, 500
        spec = LayerSpec(p=P, k=K, activation="relu")
        model = RidgeSpectralModel(input_dim=D, layer_specs=[spec], seed=0)
        config = RidgeSpectralConfig(device="cpu", verbose=False)
        trainer = RidgeSpectralTrainer(model, config)

        X, y, _, _ = _make_train_test_data(N, D)
        ds = TensorDataset(X, y)
        model, results = trainer.fit(DataLoader(ds, batch_size=N))

        assert "final" in results
        assert "train_mse" in results["final"]
        assert results["final"]["train_mse"] >= 0


# ------------------------------------------------------------------ #
# CNN trainer integration
# ------------------------------------------------------------------ #


class TestCNNIntegration:
    def test_fit_runs(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from train_hierarchically.models.cnn_ridge_spectral import (
            CNNLayerSpec,
            CNNRidgeSpectralModel,
        )
        from train_hierarchically.training.cnn_ridge_spectral import (
            CNNRidgeSpectralConfig,
            CNNRidgeSpectralTrainer,
        )

        N, C, H, W = 200, 1, 16, 16
        P, K = 32, 8
        specs = [
            CNNLayerSpec(kind="conv", p=P, k=K, kernel_size=3, padding=1),
            CNNLayerSpec(kind="flatten"),
        ]
        model = CNNRidgeSpectralModel(
            in_channels=C,
            spatial_size=(H, W),
            layer_specs=specs,
            seed=0,
        )
        config = CNNRidgeSpectralConfig(device="cpu", verbose=False)
        trainer = CNNRidgeSpectralTrainer(model, config)

        gen = torch.Generator().manual_seed(0)
        X = torch.randn(N, C, H, W, generator=gen)
        y = torch.where(torch.randn(N, generator=gen) > 0, 1.0, -1.0)
        ds = TensorDataset(X, y)
        model, results = trainer.fit(DataLoader(ds, batch_size=N))

        assert "final" in results
        assert results["final"]["train_mse"] >= 0
