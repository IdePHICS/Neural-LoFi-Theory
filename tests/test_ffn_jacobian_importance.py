"""Tests for visualization.ffn — Jacobian importance refactoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from train_hierarchically.models.ridge_spectral import (
    LayerSpec,
    RidgeSpectralModel,
)
from train_hierarchically.visualization.ffn import FFNVisualizer
from train_hierarchically.visualization.fourier import (
    infer_spatial_shape,
    smooth_spatial,
)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

INPUT_DIM = 49  # 7x7 grayscale
P = 20
K = 5
N = 16
N_TOP = 3
SEED = 42
DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def model_and_data():
    """Create a small model with non-trivial eigenvector projections."""
    spec = LayerSpec(p=P, k=K, activation="sigmoid")
    model = RidgeSpectralModel(input_dim=INPUT_DIM, layer_specs=[spec], seed=SEED)

    # Replace zero eigenvector projections with random orthonormal cols
    gen = torch.Generator().manual_seed(SEED + 10)
    rand_mat = torch.randn(P, K, generator=gen)
    q, _ = torch.linalg.qr(rand_mat)
    model.eigenvector_projections[0] = torch.nn.Parameter(
        q, requires_grad=False,
    )

    gen2 = torch.Generator().manual_seed(SEED + 1)
    X = torch.randn(N, INPUT_DIM, generator=gen2)

    rng = np.random.default_rng(SEED + 2)
    eigvecs = rng.standard_normal((P, N_TOP)).astype(np.float32)
    # Normalise columns
    eigvecs /= np.linalg.norm(eigvecs, axis=0, keepdims=True)

    return model, X, eigvecs


@pytest.fixture()
def viz(model_and_data):
    model, _, _ = model_and_data
    return FFNVisualizer(model, DEVICE)


# ------------------------------------------------------------------ #
# Tests — _jacobian_importance_raw: shape and properties
# ------------------------------------------------------------------ #


class TestRawShape:
    def test_returns_correct_number_of_tensors(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        assert len(result) == N_TOP

    def test_each_tensor_has_correct_shape(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        for t in result:
            assert t.shape == (INPUT_DIM,)

    def test_values_are_nonnegative(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        for t in result:
            assert (t >= 0).all()

    def test_values_are_nonzero(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        total = sum(t.sum().item() for t in result)
        assert total > 0

    def test_n_use_truncates(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X, 0, eigvecs, n_use=1)
        assert len(result) == 1


class TestRawDeterminism:
    def test_is_deterministic(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        r1 = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        r2 = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        for a, b in zip(r1, r2):
            assert torch.equal(a, b)


class TestRawBatching:
    def test_batch_size_invariance(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        r1 = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP, batch_size=1)
        r2 = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP, batch_size=N)
        r3 = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP, batch_size=4)
        for a, b, c in zip(r1, r2, r3):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)

    def test_additivity_over_splits(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        half = N // 2
        r_full = viz._jacobian_importance_raw(X, 0, eigvecs, N_TOP)
        r_a = viz._jacobian_importance_raw(X[:half], 0, eigvecs, N_TOP)
        r_b = viz._jacobian_importance_raw(X[half:], 0, eigvecs, N_TOP)
        for full, a, b in zip(r_full, r_a, r_b):
            torch.testing.assert_close(full, a + b, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ #
# Tests — jacobian_importance: output structure
# ------------------------------------------------------------------ #


class TestJacobianImportance:
    def test_returns_correct_keys(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.jacobian_importance(X, 0, eigvecs, N_TOP)
        expected_keys = {f"k{i}" for i in range(N_TOP)} | {"mean"}
        assert set(result.keys()) == expected_keys

    def test_values_are_lists_of_correct_length(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.jacobian_importance(X, 0, eigvecs, N_TOP)
        for key in result:
            assert isinstance(result[key], list)
            assert len(result[key]) == INPUT_DIM

    def test_mean_is_average_of_components(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.jacobian_importance(X, 0, eigvecs, N_TOP)
        components = np.array([result[f"k{i}"] for i in range(N_TOP)])
        expected_mean = components.mean(axis=0)
        np.testing.assert_allclose(
            result["mean"], expected_mean, atol=1e-5,
        )

    def test_n_top_exceeds_available(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.jacobian_importance(X, 0, eigvecs, n_top=10)
        # Should use min(10, 3) = 3
        assert len(result) == N_TOP + 1  # k0, k1, k2, mean


# ------------------------------------------------------------------ #
# Tests — per_sample_importance
# ------------------------------------------------------------------ #


class TestPerSampleImportance:
    def test_returns_correct_number(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.per_sample_importance(X[:3], 0, eigvecs, N_TOP)
        assert len(result) == 3

    def test_each_array_has_correct_shape(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.per_sample_importance(X[:2], 0, eigvecs, N_TOP)
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (INPUT_DIM,)

    def test_values_nonnegative(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz.per_sample_importance(X[:2], 0, eigvecs, N_TOP)
        for arr in result:
            assert (arr >= 0).all()

    def test_matches_manual_raw_computation(self, viz, model_and_data):
        """Verify per_sample matches calling _jacobian_importance_raw
        per sample and averaging across eigenvectors."""
        _, X, eigvecs = model_and_data
        shape = infer_spatial_shape(INPUT_DIM)
        f_decay, f_alpha = 0.15, 3.0

        result = viz.per_sample_importance(
            X[:3], 0, eigvecs, N_TOP,
            fourier_decay=f_decay, fourier_alpha=f_alpha,
        )

        for si in range(3):
            accums = viz._jacobian_importance_raw(
                X[si: si + 1], 0, eigvecs, N_TOP, batch_size=1,
            )
            imp_sum = sum(accums, torch.zeros(INPUT_DIM))
            imp_sum /= N_TOP
            imp_np = np.abs(smooth_spatial(
                imp_sum.detach().cpu().numpy(), shape,
                f_decay, f_alpha,
            ))
            np.testing.assert_allclose(result[si], imp_np, atol=1e-5)


# ------------------------------------------------------------------ #
# Tests — single-sample edge case
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_raw_single_sample(self, viz, model_and_data):
        _, X, eigvecs = model_and_data
        result = viz._jacobian_importance_raw(X[:1], 0, eigvecs, 1)
        assert len(result) == 1
        assert result[0].shape == (INPUT_DIM,)
        assert (result[0] >= 0).all()
