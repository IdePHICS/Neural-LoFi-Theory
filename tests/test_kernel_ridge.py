"""Tests for :mod:`train_hierarchically.training.kernel_ridge`.

Pins the math down:

1. Dual coefficients satisfy ``(K + alpha I) c = y - bias`` exactly
   (after centering).
2. LOO MSE computed via the eigendecomposition shortcut matches a
   direct leave-one-out refit.
3. Predicting on new points matches the explicit
   ``K_cross.T @ c + bias`` formula.
4. Converges to the finite-p ridge in the random-feature limit: a large
   random-feature Gram and its kernel-ridge fit on ``K = h h^T`` produce
   nearly identical test predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from train_hierarchically.training.kernel_ridge import (
    kernel_ridge_cv,
    kernel_ridge_predict,
)


def _psd_kernel(n: int, d: int, seed: int = 0) -> np.ndarray:
    """A nontrivial PSD Gram matrix: outer product of random features."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, d)).astype(np.float64)
    return h @ h.T


class TestDualCoefficientsAreExact:
    def test_normal_equations(self) -> None:
        n, d = 20, 40
        K = _psd_kernel(n, d)
        y = np.random.default_rng(1).standard_normal(n)
        alphas = np.logspace(-2, 2, 30)
        fit = kernel_ridge_cv(K, y, alphas=alphas)

        residual = (K + fit.alpha * np.eye(n)) @ fit.dual_coef - (y - fit.bias)
        assert np.max(np.abs(residual)) < 1e-8

    def test_center_controls_bias(self) -> None:
        n, d = 12, 30
        K = _psd_kernel(n, d)
        y = np.random.default_rng(2).standard_normal(n) + 5.0  # offset
        alphas = np.array([1.0])

        fit_c = kernel_ridge_cv(K, y, alphas=alphas, center=True)
        fit_u = kernel_ridge_cv(K, y, alphas=alphas, center=False)
        assert abs(fit_c.bias - float(y.mean())) < 1e-12
        assert fit_u.bias == 0.0


class TestLooMatchesRefit:
    """LOO MSE from the eigendecomp shortcut must equal brute-force LOO."""

    def test_small(self) -> None:
        n, d = 15, 25
        K = _psd_kernel(n, d)
        rng = np.random.default_rng(3)
        y = rng.standard_normal(n)
        alphas = np.logspace(-1, 1, 6)

        fit = kernel_ridge_cv(K, y, alphas=alphas, center=False)

        # Brute-force LOO: refit on n-1 samples for each hold-out index.
        def loo_mse_at(alpha: float) -> float:
            residuals = np.empty(n)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                K_sub = K[np.ix_(mask, mask)]
                y_sub = y[mask]
                c = np.linalg.solve(
                    K_sub + alpha * np.eye(n - 1), y_sub
                )
                k_i = K[i, mask]
                y_pred_i = k_i @ c
                residuals[i] = y[i] - y_pred_i
            return float(np.mean(residuals * residuals))

        for idx, alpha in enumerate(alphas):
            bf = loo_mse_at(alpha)
            assert abs(fit.cv_curve[idx] - bf) < 1e-8, (
                f"alpha={alpha:.3g}: shortcut={fit.cv_curve[idx]:.6f}, "
                f"refit={bf:.6f}"
            )


class TestPredictFormula:
    def test_cross_prediction(self) -> None:
        n, n_test, d = 15, 7, 25
        K = _psd_kernel(n, d)
        rng = np.random.default_rng(4)
        y = rng.standard_normal(n)
        K_cross = rng.standard_normal((n, n_test))
        K_cross = np.abs(K_cross)  # any finite values work; positivity not required
        alphas = np.logspace(0, 2, 20)

        fit = kernel_ridge_cv(K, y, alphas=alphas)
        y_pred = kernel_ridge_predict(fit, K_cross)
        expected = K_cross.T @ fit.dual_coef + fit.bias
        assert np.allclose(y_pred, expected, atol=1e-10)


class TestRandomFeatureLimit:
    """Kernel ridge on K = h h^T matches finite RF ridge on h (dual form).

    This is the mathematical identity ``(h^T h + a I)^{-1} h^T y =
    h^T (h h^T + a I)^{-1} y`` expressed as predictors, cross-checking
    that our kernel ridge reproduces the large-p limit.
    """

    def test_matches_ridge_no_intercept(self) -> None:
        """Our kernel ridge centers *y* (not *K*); equivalent sklearn Ridge is
        ``fit_intercept=False`` applied to the y-centered targets."""
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(5)
        n, p = 30, 200
        h_train = rng.standard_normal((n, p)) / np.sqrt(p)
        h_test = rng.standard_normal((10, p)) / np.sqrt(p)
        y_train = rng.standard_normal(n)
        alpha = 0.5

        y_bar = float(y_train.mean())
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
            h_train, y_train - y_bar,
        )
        y_pred_ridge = ridge.predict(h_test) + y_bar

        K_train = h_train @ h_train.T
        K_cross = h_train @ h_test.T
        fit = kernel_ridge_cv(K_train, y_train, alphas=np.array([alpha]))
        y_pred_kr = kernel_ridge_predict(fit, K_cross)

        assert np.max(np.abs(y_pred_ridge - y_pred_kr)) < 1e-6


class TestInputValidation:
    def test_non_square_raises(self) -> None:
        K = np.random.default_rng(0).standard_normal((5, 3))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="square"):
            kernel_ridge_cv(K, y, alphas=np.array([1.0]))

    def test_length_mismatch_raises(self) -> None:
        K = np.eye(4)
        y = np.zeros(5)
        with pytest.raises(ValueError, match="does not match"):
            kernel_ridge_cv(K, y, alphas=np.array([1.0]))
