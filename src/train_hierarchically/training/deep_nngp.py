"""Standard deep NNGP kernel recursion.

For an L-layer fully-connected network with per-layer weight variance
``sigma_w^2 / d_in`` and bias variance ``sigma_b^2``, the infinite-width
(NNGP) kernel obeys

    q_l(x, x') = sigma_w^2 * K_l(x, x') + sigma_b^2
    K_{l+1}(x, x') = E[sigma(h) sigma(h')]
                     with (h, h') ~ N(0, [[q_l(x,x), q_l(x,x')],
                                           [q_l(x,x'), q_l(x',x')]])

with ``K_0`` given by the layer-0 NNGP kernel of the inputs.  This is
the ``p -> infinity`` limit of the "L-layer random-feature network, no
eigenreduction, ridge on the full ``p``-dim features" architecture.

The iterated kernel depends only on scalar summaries of the previous
layer — norms and inner products — so the recursion is cheap: each
step is an element-wise transformation of an ``n x n`` kernel matrix
plus a diagonal update for test points.

When ``auto_rescale=True`` the recursion uses data-dependent
``(sigma_w_l, sigma_b_l)`` chosen so the training q-diagonal is one at
every layer: ``sigma_w_0^2 = d / mean(||x_train||^2)`` at layer 0 and
``sigma_w_l^2 = 1 / mean(K_{l-1}_diag(train))`` at layers ``l >= 1``,
both with ``sigma_b_l = 0``.  This matches the convention used by the
NLoFi-kernel scripts (``generate_nlofi_kernel_vs_deep_nngp.py`` and
peers): rescaling each composed Gram by ``1 / mean(K_diag)`` before
applying the kernel keeps off-diagonal ``cos theta`` values in a
numerically reasonable regime, regardless of depth.  Equivalent to
fixed-sigma_w only for positively homogeneous kernels (``relu``,
``relu_normalized``); for non-homogeneous kernels (``sigmoid``) it
defines a *different* network architecture, hence the explicit flag.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
from torch import Tensor

from ..kernels.base import KernelBase

log = logging.getLogger(__name__)


def _clone_kernel_with(
    kernel: KernelBase, sigma_w: float, sigma_b: float = 0.0
) -> KernelBase:
    """Return a shallow copy of *kernel* with overridden ``sigma_w`` / ``sigma_b``.

    Bypasses ``__init__`` so subclasses with extra constructor knobs
    (``n_quad`` etc.) do not need to be re-validated.  Used by the
    ``auto_rescale`` paths to set per-layer scales just-in-time without
    mutating the caller's kernel objects.
    """
    cloned = type(kernel).__new__(type(kernel))
    cloned.__dict__.update(kernel.__dict__)
    cloned.sigma_w = float(sigma_w)
    cloned.sigma_b = float(sigma_b)
    return cloned


def _auto_rescale_layer0_sigma_w(x_train: Tensor) -> float:
    """``sigma_w_0`` such that ``sigma_w_0^2 / d * ||x_train||^2`` averages 1."""
    d = x_train.shape[1]
    c2 = float((x_train * x_train).sum(dim=1).mean().item())
    return math.sqrt(d / max(c2, 1e-30))


def _auto_rescale_deeper_sigma_w(k_train_diag: Tensor) -> float:
    """``sigma_w_l`` such that ``sigma_w_l^2 * K_{l-1}_diag`` averages 1."""
    c2 = float(k_train_diag.mean().item())
    return 1.0 / math.sqrt(max(c2, 1e-30))


def _default_row_chunk(kernel: KernelBase, n: int, m: int, dtype: torch.dtype) -> int:
    """Pick a row-chunk size for composition steps inside the kernel's budget.

    Each ``kernel_from_qs`` call allocates ``O(chunk * m * Q^2)`` working
    memory; we size the chunk so that this stays under
    ``kernel.memory_budget_bytes``.
    """
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    q = int(getattr(kernel, "n_quad", 1))
    per_row = max(1, 3 * m * q * q * bytes_per)
    return max(1, min(n, kernel.memory_budget_bytes // per_row))


def _apply_kernel_rowchunked(
    kernel: KernelBase,
    q1: Tensor,
    q2: Tensor,
    q12: Tensor,
    *,
    row_chunk: int,
) -> Tensor:
    """Evaluate ``kernel.kernel_from_qs`` in row-chunks over ``q12.shape[0]``.

    The kernel's ``kernel_from_qs`` allocates large intermediate tensors
    (e.g. the ``(S, Q, Q)`` quadrature grid for sigmoid); chunking rows
    keeps peak memory bounded for large ``n``.
    """
    n = q12.shape[0]
    out = torch.empty_like(q12)
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        out[start:stop] = kernel.kernel_from_qs(
            q1[start:stop], q2, q12[start:stop]
        )
    return out


@torch.no_grad()
def deep_nngp_kernel(
    kernel: KernelBase | list[KernelBase],
    x_train: Tensor,
    x_test: Tensor | None = None,
    *,
    n_layers: int,
    row_chunk: int | None = None,
    progress: Callable[[str], None] | None = None,
    auto_rescale: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compose the deep NNGP recursion for *n_layers* hidden layers.

    Parameters
    ----------
    kernel : KernelBase | list[KernelBase]
        Either a single kernel used at every layer, or a list of length
        ``n_layers`` with one kernel per layer (allowing per-layer
        ``sigma_w`` / ``sigma_b`` / ``n_quad`` variation).
    x_train : Tensor
        Training inputs of shape ``(n, d)``.
    x_test : Tensor | None
        Optional test inputs of shape ``(n_test, d)``.  When provided, the
        test-train cross kernel is returned alongside.
    n_layers : int
        Total number of hidden layers.  ``n_layers == 1`` returns just the
        layer-0 kernel (no composition).  Must be ``>= 1``.
    row_chunk : int | None
        Row-chunk size for the composition steps.  When ``None``, an
        automatic value is picked from each layer kernel's
        ``memory_budget_bytes``.
    progress : callable | None
        Optional callback invoked once per completed layer with a short
        human-readable status string.
    auto_rescale : bool
        When ``True``, override each layer's ``(sigma_w, sigma_b)`` with
        data-dependent values that pin the training q-diagonal to 1
        throughout the depth: layer 0 uses
        ``sigma_w^2 = d / mean(||x_train||^2)``, layers ``>= 1`` use
        ``sigma_w^2 = 1 / mean(K_{l-1}_diag(train))``, both with
        ``sigma_b = 0``.  Whatever the user put in the kernel's
        ``sigma_w`` / ``sigma_b`` attributes is silently overridden.
        Default ``False`` preserves the standard fixed-sigma_w convention.

    Returns
    -------
    tuple[Tensor, Tensor | None]
        ``(K_train, K_cross)`` where ``K_train`` has shape ``(n, n)`` and
        ``K_cross`` has shape ``(n, n_test)``, or ``None`` when
        ``x_test is None``.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    # Normalise to a per-layer list.
    if isinstance(kernel, KernelBase):
        kernels = [kernel] * n_layers
    else:
        kernels = list(kernel)
        if len(kernels) != n_layers:
            raise ValueError(
                f"Expected {n_layers} kernels, got {len(kernels)}"
            )

    progress = progress or (lambda _msg: None)

    # Layer-0 kernel on raw inputs.  When auto_rescale is on, replace the
    # layer-0 kernel with a clone whose sigma_w pins the training
    # q-diagonal at 1 (sigma_b = 0).  Equivalent to the script's
    # ``q = (X X^T) / mean(||x_train||^2)`` convention.
    k0 = kernels[0]
    if auto_rescale:
        k0 = _clone_kernel_with(k0, _auto_rescale_layer0_sigma_w(x_train), 0.0)

    K_train = k0.gram(x_train)
    K_cross = k0.gram(x_train, x_test) if x_test is not None else None
    progress(
        f"layer 0 kernel: n={x_train.shape[0]} "
        f"sigma_w={k0.sigma_w:.4g} sigma_b={k0.sigma_b:.4g}"
    )

    # Layer-0 test self-kernel K_0(x_t, x_t) is cheap: evaluate pointwise.
    K_test_diag: Tensor | None = None
    if x_test is not None:
        d = x_test.shape[1]
        scale = k0.sigma_w**2 / d
        bias = k0.sigma_b**2
        q0_test = scale * (x_test * x_test).sum(dim=1) + bias  # (n_test,)
        K_test_diag = k0.kernel_from_qs(q0_test, q0_test, q0_test)

    for layer in range(1, n_layers):
        k_l = kernels[layer]
        if auto_rescale:
            k_l = _clone_kernel_with(
                k_l, _auto_rescale_deeper_sigma_w(K_train.diagonal()), 0.0
            )
        sigma_w2 = k_l.sigma_w**2
        sigma_b2 = k_l.sigma_b**2

        chunk = row_chunk
        if chunk is None:
            chunk = _default_row_chunk(
                k_l, K_train.shape[0], K_train.shape[1], K_train.dtype
            )

        # q-values derived from the previous layer's kernel.
        K_train_diag = K_train.diagonal()
        q_train_diag = sigma_w2 * K_train_diag + sigma_b2  # (n,)
        q_train = sigma_w2 * K_train + sigma_b2  # (n, n)

        K_train = _apply_kernel_rowchunked(
            k_l,
            q_train_diag.unsqueeze(1),
            q_train_diag.unsqueeze(0),
            q_train,
            row_chunk=chunk,
        )
        del q_train

        if K_cross is not None:
            assert K_test_diag is not None
            q_test_diag = sigma_w2 * K_test_diag + sigma_b2  # (n_test,)
            q_cross = sigma_w2 * K_cross + sigma_b2  # (n, n_test)
            K_cross = _apply_kernel_rowchunked(
                k_l,
                q_train_diag.unsqueeze(1),
                q_test_diag.unsqueeze(0),
                q_cross,
                row_chunk=chunk,
            )
            del q_cross
            K_test_diag = k_l.kernel_from_qs(
                q_test_diag, q_test_diag, q_test_diag
            )

        progress(
            f"layer {layer} composition "
            f"sigma_w={k_l.sigma_w:.4g} sigma_b={k_l.sigma_b:.4g}"
        )

    return K_train, K_cross


__all__ = ["deep_nngp_kernel"]
