"""Tests for training.base.fit_ridge_readout — shared ridge readout."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from train_hierarchically.training.base import fit_ridge_readout

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _linear_data(
    n: int = 100,
    d: int = 10,
    *,
    seed: int = 42,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate y = X @ w + noise.  Returns (X, y_tensor, w)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d).astype(np.float32)
    X_np = rng.standard_normal((n, d)).astype(np.float32)
    y_np = X_np @ w + noise_std * rng.standard_normal(n).astype(np.float32)
    return torch.from_numpy(X_np), torch.from_numpy(y_np), w


def _binary_data(
    n: int = 200,
    d: int = 10,
    *,
    seed: int = 42,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate well-separated binary ±1 labels."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d).astype(np.float32)
    w /= np.linalg.norm(w)
    X_np = rng.standard_normal((n, d)).astype(np.float32)
    proj = X_np @ w
    y_np = np.where(proj >= 0, 1.0, -1.0).astype(np.float32)
    # push samples away from decision boundary
    X_np += margin * y_np[:, None] * w[None, :]
    return torch.from_numpy(X_np), torch.from_numpy(y_np), w


# ------------------------------------------------------------------ #
# Tests — return value structure
# ------------------------------------------------------------------ #


class TestReturnStructure:
    def test_train_only_keys(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert set(result.keys()) == {"train_mse", "ridge_alpha", "accuracy"}

    def test_train_and_test_keys(self):
        X, y, _ = _linear_data(n=80)
        Xt, yt, _ = _linear_data(n=20, seed=99)
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=Xt, y_test_np=yt.numpy(),
            verbose=False,
        )
        expected = {
            "train_mse", "ridge_alpha", "accuracy",
            "test_mse", "test_sem", "test_accuracy", "test_accuracy_sem",
        }
        assert set(result.keys()) == expected


# ------------------------------------------------------------------ #
# Tests — model parameters
# ------------------------------------------------------------------ #


class TestModelParameters:
    def test_ridge_weight_is_frozen_parameter(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        fit_ridge_readout(model, X, y, verbose=False)
        assert isinstance(model.ridge_weight, nn.Parameter)
        assert model.ridge_weight.requires_grad is False
        assert model.ridge_weight.shape == (X.shape[1],)

    def test_ridge_bias_is_frozen_parameter(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        fit_ridge_readout(model, X, y, verbose=False)
        assert isinstance(model.ridge_bias, nn.Parameter)
        assert model.ridge_bias.requires_grad is False
        assert model.ridge_bias.shape == (1,)

    def test_dtype_matches_input(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        fit_ridge_readout(model, X, y, verbose=False)
        assert model.ridge_weight.dtype == X.dtype
        assert model.ridge_bias.dtype == X.dtype

    def test_device_matches_input(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        fit_ridge_readout(model, X, y, verbose=False)
        assert model.ridge_weight.device == X.device
        assert model.ridge_bias.device == X.device

    def test_no_ridge_attrs_before_call(self):
        model = nn.Module()
        assert not hasattr(model, "ridge_weight")
        assert not hasattr(model, "ridge_bias")

    def test_overwrites_existing_attrs(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        model.ridge_weight = nn.Parameter(torch.zeros(1))
        model.ridge_bias = nn.Parameter(torch.zeros(1))
        fit_ridge_readout(model, X, y, verbose=False)
        assert model.ridge_weight.shape == (X.shape[1],)
        assert model.ridge_bias.shape == (1,)


# ------------------------------------------------------------------ #
# Tests — numerical correctness
# ------------------------------------------------------------------ #


class TestNumericalCorrectness:
    def test_train_mse_near_zero_noiseless(self):
        X, y, _ = _linear_data(noise_std=0.0)
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert result["train_mse"] < 1e-4

    def test_test_mse_near_zero_noiseless(self):
        X, y, w = _linear_data(n=80, noise_std=0.0)
        rng = np.random.default_rng(99)
        Xt_np = rng.standard_normal((20, 10)).astype(np.float32)
        yt_np = Xt_np @ w
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=torch.from_numpy(Xt_np),
            y_test_np=yt_np,
            verbose=False,
        )
        assert result["test_mse"] < 1e-3

    def test_ridge_alpha_is_positive(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert isinstance(result["ridge_alpha"], float)
        assert result["ridge_alpha"] > 0

    def test_binary_accuracy_perfect_separable(self):
        X, y, _ = _binary_data(margin=3.0)
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert result["accuracy"] >= 0.99

    def test_test_binary_accuracy_high_separable(self):
        # Use the same w for train and test so the decision boundary
        # generalises perfectly.
        X, y, w = _binary_data(n=200, margin=5.0, seed=42)
        rng = np.random.default_rng(99)
        d = X.shape[1]
        w_norm = w / np.linalg.norm(w)
        Xt_np = rng.standard_normal((50, d)).astype(np.float32)
        proj = Xt_np @ w_norm
        yt_np = np.where(proj >= 0, 1.0, -1.0).astype(np.float32)
        Xt_np += 5.0 * yt_np[:, None] * w_norm[None, :]
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=torch.from_numpy(Xt_np),
            y_test_np=yt_np,
            verbose=False,
        )
        assert result["test_accuracy"] >= 0.9

    def test_stored_weights_reproduce_train_mse(self):
        X, y, _ = _linear_data(noise_std=0.1)
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert isinstance(model.ridge_weight, nn.Parameter)
        assert isinstance(model.ridge_bias, nn.Parameter)
        y_pred = (X @ model.ridge_weight + model.ridge_bias).detach().numpy()
        mse_manual = float(np.mean((y.numpy() - y_pred) ** 2))
        np.testing.assert_allclose(mse_manual, result["train_mse"], rtol=1e-5)


# ------------------------------------------------------------------ #
# Tests — SEM formulas
# ------------------------------------------------------------------ #


class TestSEMFormulas:
    def test_test_sem_nonnegative(self):
        X, y, w = _linear_data(n=80, noise_std=0.1)
        Xt, yt, _ = _linear_data(n=20, seed=99, noise_std=0.1)
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=Xt, y_test_np=yt.numpy(),
            verbose=False,
        )
        assert result["test_sem"] >= 0.0

    def test_test_accuracy_sem_nonnegative(self):
        X, y, _ = _binary_data(n=200)
        Xt, yt, _ = _binary_data(n=50, seed=99)
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=Xt, y_test_np=yt.numpy(),
            verbose=False,
        )
        assert result["test_accuracy_sem"] >= 0.0

    def test_test_accuracy_sem_formula(self):
        X, y, _ = _binary_data(n=200)
        Xt, yt, _ = _binary_data(n=50, seed=99)
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            h_test=Xt, y_test_np=yt.numpy(),
            verbose=False,
        )
        p = result["test_accuracy"]
        n_test = 50
        expected = float(np.sqrt(p * (1 - p) / n_test))
        np.testing.assert_allclose(
            result["test_accuracy_sem"], expected, atol=1e-10,
        )


# ------------------------------------------------------------------ #
# Tests — configuration
# ------------------------------------------------------------------ #


class TestConfiguration:
    def test_custom_alpha_range(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        result = fit_ridge_readout(
            model, X, y,
            alpha_min=0.0, alpha_max=2.0, alpha_num=10,
            verbose=False,
        )
        assert 1.0 <= result["ridge_alpha"] <= 100.0

    def test_verbose_false_no_error(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=False)
        assert "train_mse" in result

    def test_verbose_true_no_error(self):
        X, y, _ = _linear_data()
        model = nn.Module()
        result = fit_ridge_readout(model, X, y, verbose=True)
        assert "train_mse" in result
