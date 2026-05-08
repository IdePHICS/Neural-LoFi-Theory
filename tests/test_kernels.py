"""Tests for :mod:`train_hierarchically.kernels`.

Covers three layers:

1. **Registry** (``get_kernel`` / ``register_kernel`` / ``available_kernels``).
2. **Spectral-sqrt 2x2 helper** — internal building block used by the
   sigmoid kernel.
3. **Sigmoid kernel** — Monte-Carlo agreement, analytic special cases,
   broadcasting, quadrature convergence.
4. **Shared Gram routine** (``KernelBase.gram``) — symmetry, PSD, chunking
   invariance, fast vs. streamed path equivalence, upper-triangular
   mirror correctness, input validation.

Every test either pins a *mathematical* property (identity, inequality,
or closed form) or cross-checks two implementation paths against each
other.  No assertions on implementation-specific values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from train_hierarchically.kernels import (
    KernelBase,
    available_kernels,
    get_kernel,
    register_kernel,
)
from train_hierarchically.kernels.sigmoid import (
    SigmoidKernel,
    _spectral_sqrt_2x2,
)

# ============================================================================
# Helpers
# ============================================================================


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _mc_sigmoid_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma_w: float,
    sigma_b: float,
    n_mc: int,
    seed: int = 0,
) -> torch.Tensor:
    """Monte-Carlo estimate of ``K(x, y) = E[sigma(h) sigma(h')]``.

    ``h = W x + b`` with ``W ~ N(0, sigma_w^2 / d)``, ``b ~ N(0, sigma_b^2)``.
    Uses paired samples (same ``W``, ``b`` for *x* and *y*) so the
    estimator of *K(x, y)* is low-variance.
    """
    assert x.ndim == 2 and y.ndim == 2
    d = x.shape[1]
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n_mc, d, generator=gen) * (sigma_w / math.sqrt(d))
    b = torch.randn(n_mc, generator=gen) * sigma_b
    # h_x: (nx, n_mc)  h_y: (ny, n_mc); broadcasting b over n_mc.
    h_x = torch.sigmoid(x.float() @ w.T + b)
    h_y = torch.sigmoid(y.float() @ w.T + b)
    return (h_x @ h_y.T) / n_mc


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_sigmoid_is_registered(self) -> None:
        assert "sigmoid" in available_kernels()

    def test_get_kernel_builds_sigmoid(self) -> None:
        k = get_kernel("sigmoid")
        assert isinstance(k, SigmoidKernel)
        assert k.sigma_w == 1.0
        assert k.sigma_b == 0.0
        assert k.n_quad == 30

    def test_get_kernel_forwards_kwargs(self) -> None:
        k = get_kernel("sigmoid", sigma_w=2.5, sigma_b=0.7, n_quad=12)
        assert k.sigma_w == 2.5
        assert k.sigma_b == 0.7
        assert k.n_quad == 12

    def test_unknown_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown kernel"):
            get_kernel("nonexistent")

    def test_name_is_case_insensitive(self) -> None:
        assert isinstance(get_kernel("SIGMOID"), SigmoidKernel)
        assert isinstance(get_kernel(" sigmoid "), SigmoidKernel)

    def test_double_registration_raises(self) -> None:
        with pytest.raises(KeyError, match="already registered"):

            @register_kernel("sigmoid")
            class _Dup(KernelBase):
                pass


class TestKernelBaseAbstract:
    def test_kernel_from_qs_not_implemented(self) -> None:
        k = KernelBase(sigma_w=1.0, sigma_b=0.0)
        with pytest.raises(NotImplementedError):
            k.kernel_from_qs(
                torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.5)
            )


# ============================================================================
# Spectral square-root helper (2x2)
# ============================================================================


class TestSpectralSqrt2x2:
    """``Sigma^{1/2}`` must square back to ``Sigma`` for every PSD case."""

    def _square_roundtrip(
        self, q1: torch.Tensor, q2: torch.Tensor, q12: torch.Tensor
    ) -> None:
        a, b, c = _spectral_sqrt_2x2(q1, q2, q12)
        # (Sigma^{1/2})^2 = [[a² + b², a b + b c], [... , b² + c²]]
        recon_11 = a * a + b * b
        recon_12 = a * b + b * c
        recon_22 = b * b + c * c
        torch.testing.assert_close(recon_11, q1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recon_22, q2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recon_12, q12, atol=1e-5, rtol=1e-5)

    def test_random_psd_matrices(self) -> None:
        gen = _rng(0)
        # Sample q1, q2 > 0 and |q12| <= sqrt(q1 q2) (PSD constraint).
        q1 = 0.1 + 2.0 * torch.rand(128, generator=gen, dtype=torch.float64)
        q2 = 0.1 + 2.0 * torch.rand(128, generator=gen, dtype=torch.float64)
        # Correlation rho ∈ [-1, 1] → q12 = rho · sqrt(q1 q2).
        rho = -1.0 + 2.0 * torch.rand(128, generator=gen, dtype=torch.float64)
        q12 = rho * (q1 * q2).sqrt()
        self._square_roundtrip(q1, q2, q12)

    def test_diagonal_case(self) -> None:
        """q12 = 0 ⇒ Σ^{1/2} = diag(sqrt(q1), sqrt(q2))."""
        q1 = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        q2 = torch.tensor([4.0, 1.0, 16.0], dtype=torch.float64)
        q12 = torch.zeros(3, dtype=torch.float64)
        a, b, c = _spectral_sqrt_2x2(q1, q2, q12)
        torch.testing.assert_close(a, q1.sqrt(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(c, q2.sqrt(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(b, torch.zeros_like(q1), atol=1e-12, rtol=1e-12)

    def test_identity_case(self) -> None:
        q1 = torch.tensor([1.0], dtype=torch.float64)
        q2 = torch.tensor([1.0], dtype=torch.float64)
        q12 = torch.tensor([0.0], dtype=torch.float64)
        a, b, c = _spectral_sqrt_2x2(q1, q2, q12)
        torch.testing.assert_close(a, torch.tensor([1.0], dtype=torch.float64))
        torch.testing.assert_close(b, torch.tensor([0.0], dtype=torch.float64))
        torch.testing.assert_close(c, torch.tensor([1.0], dtype=torch.float64))

    def test_perfectly_correlated_rank1_case(self) -> None:
        """Σ = v v^T with v = (s, s) still round-trips (singular matrix)."""
        s = torch.tensor([1.5, 2.0], dtype=torch.float64)
        q1 = s * s  # [2.25, 4.0]
        q2 = s * s
        q12 = s * s  # perfect correlation → rank-1
        self._square_roundtrip(q1, q2, q12)

    def test_broadcast_shape(self) -> None:
        q1 = torch.ones(4, 1, dtype=torch.float64)
        q2 = torch.ones(1, 5, dtype=torch.float64)
        q12 = torch.zeros(4, 5, dtype=torch.float64) + 0.3
        a, b, c = _spectral_sqrt_2x2(q1, q2, q12)
        assert a.shape == (4, 5)
        assert b.shape == (4, 5)
        assert c.shape == (4, 5)


# ============================================================================
# Sigmoid kernel — mathematical correctness
# ============================================================================


class TestSigmoidAnalytic:
    """Closed-form identities that the kernel must satisfy exactly."""

    def test_zero_weights_zero_bias_gives_quarter(self) -> None:
        """σ_w = σ_b = 0 ⇒ h = h' = 0 ⇒ K = σ(0)² = 0.25."""
        k = SigmoidKernel(sigma_w=0.0, sigma_b=0.0, n_quad=30)
        x = torch.randn(3, 8, generator=_rng(0))
        y = torch.randn(2, 8, generator=_rng(1))
        g = k.gram(x, y)
        torch.testing.assert_close(
            g, torch.full((3, 2), 0.25), atol=1e-6, rtol=1e-6
        )

    def test_zero_weights_bias_gives_constant(self) -> None:
        """σ_w = 0 ⇒ h = h' = b ~ N(0, σ_b²); K is input-independent."""
        k = SigmoidKernel(sigma_w=0.0, sigma_b=1.3, n_quad=40)
        x = torch.randn(4, 6, generator=_rng(0))
        g = k.gram(x)
        # All entries are identical.
        assert torch.allclose(g, g.flatten()[0] * torch.ones_like(g), atol=1e-6)
        # And the constant equals E[σ(b)²] via 1-D quadrature.
        sigma_b = 1.3
        from numpy.polynomial.hermite import hermgauss

        nodes_np, weights_np = hermgauss(80)
        z = nodes_np * math.sqrt(2.0)  # standard-normal substitution
        w = weights_np / math.sqrt(math.pi)
        integrand = (1.0 / (1.0 + np.exp(-sigma_b * z))) ** 2
        k_analytic = float((integrand * w).sum())
        assert abs(g[0, 0].item() - k_analytic) < 1e-5

    def test_diagonal_equals_expected_sigma_squared(self) -> None:
        """K(x, x) = E[σ(h)²] with h ~ N(0, q1). Use 1-D G-H for ground truth."""
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.4, n_quad=30)
        d = 12
        x = torch.randn(5, d, generator=_rng(0))
        q = (1.0 / d) * (x * x).sum(dim=1) + 0.4**2  # q1 per row

        from numpy.polynomial.hermite import hermgauss

        nodes_np, weights_np = hermgauss(60)
        z = nodes_np * math.sqrt(2.0)
        wts = weights_np / math.sqrt(math.pi)

        g_diag = k.gram(x).diagonal()
        for i in range(5):
            sigma_h = math.sqrt(q[i].item())
            integrand = (1.0 / (1.0 + np.exp(-sigma_h * z))) ** 2
            expected = float((integrand * wts).sum())
            assert abs(g_diag[i].item() - expected) < 1e-5, (
                f"diag[{i}]={g_diag[i].item():.6f} vs analytic {expected:.6f}"
            )


class TestSigmoidMonteCarlo:
    """Kernel matches a large-sample Monte Carlo estimate."""

    @pytest.mark.parametrize(
        "sigma_w,sigma_b", [(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (0.5, 1.0)]
    )
    def test_cross_gram_matches_mc(
        self, sigma_w: float, sigma_b: float
    ) -> None:
        k = SigmoidKernel(sigma_w=sigma_w, sigma_b=sigma_b, n_quad=30)
        d = 30
        x = torch.randn(4, d, generator=_rng(0))
        y = torch.randn(5, d, generator=_rng(1))
        g = k.gram(x, y).double()
        # 400k MC samples: per-entry SE of σ²σ² products ≈ 0.25/√N ≈ 4e-4.
        # Tolerance of 3e-3 is ~7 SE → essentially never spurious.
        g_mc = _mc_sigmoid_kernel(
            x, y, sigma_w=sigma_w, sigma_b=sigma_b, n_mc=400_000, seed=42,
        ).double()
        assert (g - g_mc).abs().max().item() < 3e-3, (
            f"max diff {(g - g_mc).abs().max().item():.4e}"
        )


class TestSigmoidGramProperties:
    def test_gram_is_symmetric(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.3, n_quad=20)
        x = torch.randn(20, 10, generator=_rng(0))
        g = k.gram(x)
        # Symmetrised exactly by the base class.
        assert (g - g.T).abs().max().item() == 0.0

    def test_gram_is_psd(self) -> None:
        k = SigmoidKernel(sigma_w=1.2, sigma_b=0.0, n_quad=25)
        x = torch.randn(40, 12, generator=_rng(0))
        g = k.gram(x).double()
        # All eigenvalues ≥ 0 (up to numerical slack).
        eigvals = torch.linalg.eigvalsh(g)
        assert eigvals.min().item() > -1e-6, (
            f"min eigenvalue {eigvals.min().item():.4e}"
        )

    def test_cauchy_schwarz(self) -> None:
        """|K(x, y)|² ≤ K(x, x) K(y, y) — mandatory for any PSD kernel."""
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.5, n_quad=20)
        x = torch.randn(30, 15, generator=_rng(0))
        g = k.gram(x).double()
        diag = g.diagonal()
        rhs = torch.outer(diag, diag)
        # Allow 1e-5 slack for float32 round-off.
        assert (g * g - rhs).max().item() < 1e-5

    def test_symmetric_and_cross_agree(self) -> None:
        """gram(x) == gram(x, x) (symmetric shortcut vs explicit)."""
        k = SigmoidKernel(sigma_w=0.8, sigma_b=0.2, n_quad=20)
        x = torch.randn(12, 8, generator=_rng(0))
        g_sym = k.gram(x)
        g_cross = k.gram(x, x)
        # Both paths symmetrise the sym case; cross path does not, but the
        # values should still agree to within float32 reduction order.
        torch.testing.assert_close(g_sym, g_cross, atol=1e-5, rtol=1e-5)


class TestSigmoidQuadratureConvergence:
    """Higher ``n_quad`` should monotonically reduce error vs a reference."""

    def test_quad_convergence(self) -> None:
        # Reference: extremely high n_quad (effectively exact).
        ref = SigmoidKernel(sigma_w=1.0, sigma_b=0.4, n_quad=80)
        x = torch.randn(6, 20, generator=_rng(0))
        g_ref = ref.gram(x).double()

        errs = []
        for n_quad in (6, 10, 15, 25):
            k = SigmoidKernel(sigma_w=1.0, sigma_b=0.4, n_quad=n_quad)
            g = k.gram(x).double()
            errs.append((g - g_ref).abs().max().item())

        # Strictly decreasing (spectral-sqrt rule is super-algebraic on
        # this symmetric integrand).
        for i in range(len(errs) - 1):
            assert errs[i + 1] < errs[i] + 1e-10, (
                f"n_quad series not monotone: {errs}"
            )
        # And the smallest one is near machine precision for float32.
        assert errs[-1] < 1e-5

    def test_nquad_low_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_quad must be >= 2"):
            SigmoidKernel(n_quad=1)


class TestSigmoidBroadcasting:
    def test_kernel_from_qs_broadcasts(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.1, n_quad=20)
        q1 = torch.tensor([[1.0], [2.0], [3.0]])
        q2 = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        q12 = torch.full((3, 4), 0.3)
        out = k.kernel_from_qs(q1, q2, q12)
        assert out.shape == (3, 4)

        # Scalar call.
        s = k.kernel_from_qs(
            torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.2)
        )
        assert s.shape == ()
        assert 0.0 < s.item() < 1.0


# ============================================================================
# Shared Gram routine — code paths and guarantees
# ============================================================================


class TestGramChunking:
    """Row-chunking must be a pure reordering — identical results."""

    def test_chunk_invariance_symmetric(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=20)
        x = torch.randn(23, 11, generator=_rng(0))
        g_ref = k.gram(x, row_chunk=1000).double()
        for rc in (1, 3, 7, 23):
            g = k.gram(x, row_chunk=rc).double()
            torch.testing.assert_close(g, g_ref, atol=1e-6, rtol=1e-6)

    def test_chunk_invariance_cross(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.2, n_quad=15)
        x = torch.randn(19, 9, generator=_rng(0))
        y = torch.randn(25, 9, generator=_rng(1))
        g_ref = k.gram(x, y, row_chunk=1000).double()
        for rc in (1, 2, 5, 19):
            g = k.gram(x, y, row_chunk=rc).double()
            torch.testing.assert_close(g, g_ref, atol=1e-6, rtol=1e-6)

    def test_auto_chunk_is_consistent(self) -> None:
        """``row_chunk=None`` picks from the memory budget; values identical."""
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=20)
        x = torch.randn(17, 9, generator=_rng(0))
        g_explicit = k.gram(x, row_chunk=17)
        g_auto = k.gram(x, row_chunk=None)
        torch.testing.assert_close(g_auto, g_explicit, atol=1e-6, rtol=1e-6)


class TestGramStreamedPath:
    """The sigmoid kernel has two internal j-axis paths (fast 4-D and streamed).

    Force the streamed path by shrinking ``memory_budget_bytes`` and verify
    it produces values equivalent to the fast path.
    """

    def test_streamed_matches_fast(self) -> None:
        x = torch.randn(8, 6, generator=_rng(0))
        y = torch.randn(7, 6, generator=_rng(1))

        k_fast = SigmoidKernel(sigma_w=1.0, sigma_b=0.3, n_quad=30)
        g_fast = k_fast.gram(x, y).double()

        k_stream = SigmoidKernel(sigma_w=1.0, sigma_b=0.3, n_quad=30)
        # 1 KiB forces j-block = 1 → streamed path exercised.
        k_stream.memory_budget_bytes = 1024
        g_stream = k_stream.gram(x, y).double()
        torch.testing.assert_close(g_stream, g_fast, atol=1e-6, rtol=1e-6)

    def test_streamed_matches_fast_symmetric(self) -> None:
        x = torch.randn(12, 7, generator=_rng(0))

        k_fast = SigmoidKernel(sigma_w=0.9, sigma_b=0.0, n_quad=25)
        g_fast = k_fast.gram(x).double()

        k_stream = SigmoidKernel(sigma_w=0.9, sigma_b=0.0, n_quad=25)
        k_stream.memory_budget_bytes = 1024
        g_stream = k_stream.gram(x).double()
        torch.testing.assert_close(g_stream, g_fast, atol=1e-6, rtol=1e-6)


class TestGramUpperTriangularMirror:
    """The symmetric shortcut computes only upper-tri blocks and mirrors.

    Must match a direct naive full computation via ``gram(x, x)``.
    """

    @pytest.mark.parametrize("n,row_chunk", [(10, 3), (15, 5), (20, 1)])
    def test_full_vs_mirrored(self, n: int, row_chunk: int) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.2, n_quad=20)
        x = torch.randn(n, 8, generator=_rng(0))
        g_sym = k.gram(x, row_chunk=row_chunk).double()

        # Independent path: explicit (x, y=x) — computes every pair.
        g_full = k.gram(x, x, row_chunk=row_chunk).double()
        torch.testing.assert_close(g_sym, g_full, atol=1e-5, rtol=1e-5)


class TestGramInputValidation:
    def test_dim_mismatch_raises(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=10)
        x = torch.randn(3, 5)
        y = torch.randn(4, 7)
        with pytest.raises(ValueError, match="must match"):
            k.gram(x, y)


class TestGramDtypeDevice:
    def test_preserves_dtype_cpu(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=10)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(6, 4, dtype=dtype, generator=_rng(0))
            g = k.gram(x)
            assert g.dtype == dtype
            assert g.device == x.device

    def test_output_shape(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=10)
        x = torch.randn(11, 3, generator=_rng(0))
        y = torch.randn(7, 3, generator=_rng(1))
        assert k.gram(x).shape == (11, 11)
        assert k.gram(x, y).shape == (11, 7)


class TestPickRowChunk:
    """``_pick_row_chunk`` must respect the memory budget."""

    def test_budget_limits_chunk(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=30)
        # Very small budget → chunk=1.
        k.memory_budget_bytes = 1
        assert k._pick_row_chunk(1000, 1000, torch.float32) == 1

    def test_chunk_never_exceeds_n(self) -> None:
        k = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=10)
        # Huge budget → clamped to n.
        k.memory_budget_bytes = 1 << 50
        assert k._pick_row_chunk(17, 10, torch.float32) == 17

    def test_chunk_uses_per_pair_estimate(self) -> None:
        """A kernel with a larger per-pair footprint gets a smaller chunk."""
        k_small = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=10)
        k_big = SigmoidKernel(sigma_w=1.0, sigma_b=0.0, n_quad=30)
        k_small.memory_budget_bytes = 1 << 20  # 1 MiB
        k_big.memory_budget_bytes = 1 << 20
        # m = 100 puts the small kernel inside the budget (chunk > 1) while
        # the big kernel's per-row footprint saturates it (chunk == 1).
        c_small = k_small._pick_row_chunk(10_000, 100, torch.float32)
        c_big = k_big._pick_row_chunk(10_000, 100, torch.float32)
        assert c_small > c_big  # 3·Q² → 9× larger footprint at Q=30 vs Q=10
