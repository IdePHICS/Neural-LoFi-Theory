"""Tests for :func:`train_hierarchically.training.deep_nngp.deep_nngp_kernel`.

Covers:

- ``n_layers == 1`` equals a direct ``kernel.gram`` call.
- Iteration step matches a reference loop using only ``kernel.kernel_from_qs``.
- Test self-kernel diagonal is updated consistently with the train kernel.
- Row-chunk invariance: different ``row_chunk`` values give identical output.
- ``n_layers < 1`` raises.
"""

from __future__ import annotations

import pytest
import torch

from train_hierarchically.kernels import get_kernel
from train_hierarchically.kernels.sigmoid import SigmoidKernel
from train_hierarchically.training.deep_nngp import deep_nngp_kernel


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _reference_iterate(
    kernel: SigmoidKernel, K_train: torch.Tensor, K_cross: torch.Tensor,
    K_test_diag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-step NNGP iteration written as a small, explicit loop."""
    sw2 = kernel.sigma_w**2
    sb2 = kernel.sigma_b**2

    K_train_diag = K_train.diagonal()
    q_tr_diag = sw2 * K_train_diag + sb2
    q_te_diag = sw2 * K_test_diag + sb2
    q_tr = sw2 * K_train + sb2
    q_cross = sw2 * K_cross + sb2

    K_train_new = kernel.kernel_from_qs(
        q_tr_diag.unsqueeze(1), q_tr_diag.unsqueeze(0), q_tr
    )
    K_cross_new = kernel.kernel_from_qs(
        q_tr_diag.unsqueeze(1), q_te_diag.unsqueeze(0), q_cross
    )
    K_test_diag_new = kernel.kernel_from_qs(
        q_te_diag, q_te_diag, q_te_diag
    )
    return K_train_new, K_cross_new, K_test_diag_new


class TestLayer1EqualsGram:
    def test_train_only(self) -> None:
        k = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.2, n_quad=20)
        x = torch.randn(12, 8, generator=_rng(0))
        K_ref = k.gram(x)
        K_test, K_cross = deep_nngp_kernel(k, x, None, n_layers=1)
        torch.testing.assert_close(K_test, K_ref, atol=1e-6, rtol=1e-6)
        assert K_cross is None

    def test_train_test(self) -> None:
        k = get_kernel("sigmoid", sigma_w=0.8, sigma_b=0.5, n_quad=20)
        x = torch.randn(10, 6, generator=_rng(0))
        y = torch.randn(5, 6, generator=_rng(1))
        K_train_ref = k.gram(x)
        K_cross_ref = k.gram(x, y)
        K_train, K_cross = deep_nngp_kernel(k, x, y, n_layers=1)
        torch.testing.assert_close(K_train, K_train_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(K_cross, K_cross_ref, atol=1e-6, rtol=1e-6)


class TestIterationMatchesReference:
    """Each recursion step must match an explicit single-step reference."""

    def test_one_step(self) -> None:
        k = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.3, n_quad=15)
        x = torch.randn(8, 6, generator=_rng(0))
        y = torch.randn(6, 6, generator=_rng(1))

        K0_train = k.gram(x)
        K0_cross = k.gram(x, y)
        d = y.shape[1]
        scale = k.sigma_w**2 / d
        q_y = scale * (y * y).sum(dim=1) + k.sigma_b**2
        K0_test_diag = k.kernel_from_qs(q_y, q_y, q_y)

        K1_ref_train, K1_ref_cross, _ = _reference_iterate(
            k, K0_train, K0_cross, K0_test_diag
        )
        K1_train, K1_cross = deep_nngp_kernel(k, x, y, n_layers=2)
        torch.testing.assert_close(K1_train, K1_ref_train, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(K1_cross, K1_ref_cross, atol=1e-6, rtol=1e-6)

    def test_three_steps(self) -> None:
        k = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.0, n_quad=20)
        x = torch.randn(7, 5, generator=_rng(0))
        y = torch.randn(4, 5, generator=_rng(1))

        # Reference: iterate by hand twice to reach L=3.
        K_train = k.gram(x)
        K_cross = k.gram(x, y)
        d = y.shape[1]
        q_y = k.sigma_w**2 / d * (y * y).sum(dim=1) + k.sigma_b**2
        K_test_diag = k.kernel_from_qs(q_y, q_y, q_y)

        for _ in range(2):  # L=1 -> L=2 -> L=3
            K_train, K_cross, K_test_diag = _reference_iterate(
                k, K_train, K_cross, K_test_diag
            )

        K3_train, K3_cross = deep_nngp_kernel(k, x, y, n_layers=3)
        torch.testing.assert_close(K3_train, K_train, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(K3_cross, K_cross, atol=1e-6, rtol=1e-6)


class TestRowChunkInvariance:
    """``row_chunk`` is a pure tiling — identical output for any value."""

    def test_train_only(self) -> None:
        k = get_kernel("sigmoid", sigma_w=1.0, sigma_b=0.0, n_quad=15)
        x = torch.randn(23, 7, generator=_rng(0))
        K_ref, _ = deep_nngp_kernel(k, x, None, n_layers=3, row_chunk=1000)
        for rc in (1, 3, 7, 23):
            K, _ = deep_nngp_kernel(k, x, None, n_layers=3, row_chunk=rc)
            torch.testing.assert_close(K, K_ref, atol=1e-6, rtol=1e-6)

    def test_cross(self) -> None:
        k = get_kernel("sigmoid", sigma_w=0.9, sigma_b=0.2, n_quad=15)
        x = torch.randn(17, 5, generator=_rng(0))
        y = torch.randn(9, 5, generator=_rng(1))
        K_ref, KC_ref = deep_nngp_kernel(k, x, y, n_layers=2, row_chunk=1000)
        for rc in (1, 2, 5, 17):
            K, KC = deep_nngp_kernel(k, x, y, n_layers=2, row_chunk=rc)
            torch.testing.assert_close(K, K_ref, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(KC, KC_ref, atol=1e-6, rtol=1e-6)


class TestInputValidation:
    def test_zero_layers_raises(self) -> None:
        k = get_kernel("sigmoid", n_quad=10)
        x = torch.randn(3, 4)
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            deep_nngp_kernel(k, x, None, n_layers=0)
