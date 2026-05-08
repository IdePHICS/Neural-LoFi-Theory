"""Shared base class for NNGP kernels.

Concrete kernels subclass :class:`KernelBase` and implement
:meth:`KernelBase.kernel_from_qs` — a pointwise function of the three
NNGP scalars ``(q1, q2, q12)`` for one pair of inputs.  The shared
:meth:`KernelBase.gram` routine precomputes those scalars across all
pairs (chunked over rows to bound memory) and dispatches to
``kernel_from_qs``.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KernelBase:
    """Infinite-width (NNGP) kernel with a vectorised Gram routine.

    Attributes
    ----------
    name : str
        Registered kernel name; set on subclasses.
    memory_budget_bytes : int
        Soft upper bound on peak working-tensor memory per chunk.  Used
        to derive an automatic ``row_chunk`` when the caller does not
        specify one.  Defaults to 1 GiB.
    """

    name: str = "base"

    memory_budget_bytes: int = 1 << 30  # 1 GiB

    def __init__(self, *, sigma_w: float = 1.0, sigma_b: float = 0.0) -> None:
        self.sigma_w = float(sigma_w)
        self.sigma_b = float(sigma_b)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def kernel_from_qs(self, q1: Tensor, q2: Tensor, q12: Tensor) -> Tensor:
        """Pointwise NNGP kernel.

        Parameters
        ----------
        q1, q2 : Tensor
            Diagonal NNGP scalars ``(sigma_w^2 / d) ||x||^2 + sigma_b^2``.
        q12 : Tensor
            Off-diagonal scalar ``(sigma_w^2 / d) x . x' + sigma_b^2``.

        All inputs broadcast to a common shape ``S``; the return value has
        shape ``S``.
        """
        raise NotImplementedError

    def _per_pair_working_bytes(self, dtype: torch.dtype) -> int:
        """Peak working-memory (bytes) per pair of inputs inside ``gram``.

        Subclasses with extra inner axes (e.g. quadrature) override this
        so the auto-picked ``row_chunk`` stays inside
        ``memory_budget_bytes``.
        """
        return torch.tensor([], dtype=dtype).element_size()

    def _pick_row_chunk(self, n: int, m: int, dtype: torch.dtype) -> int:
        """Row-chunk size fitting inside ``memory_budget_bytes``."""
        per_row = max(1, m * self._per_pair_working_bytes(dtype))
        rows = max(1, self.memory_budget_bytes // per_row)
        return int(min(n, rows))

    # ------------------------------------------------------------------
    # Shared gram routine
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gram(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        row_chunk: int | None = None,
    ) -> Tensor:
        """Assemble the Gram matrix via row-chunked calls to ``kernel_from_qs``.

        Parameters
        ----------
        x : Tensor
            Inputs of shape ``(n, d)``.
        y : Tensor | None
            Inputs of shape ``(m, d)``.  ``None`` triggers symmetric
            evaluation against *x*.
        row_chunk : int | None
            Number of rows to evaluate per chunk (memory bound).  When
            ``None``, an auto chunk is picked from
            :attr:`memory_budget_bytes`.

        Returns
        -------
        Tensor
            Gram matrix of shape ``(n, m)`` (or ``(n, n)`` when
            ``y is None``), on the same device as *x*.
        """
        symmetric = y is None
        if symmetric:
            y = x

        n, d = x.shape
        m = y.shape[0]
        if y.shape[1] != d:
            raise ValueError(f"x has dim {d} but y has dim {y.shape[1]}; must match.")

        scale = self.sigma_w**2 / d
        bias = self.sigma_b**2

        # Diagonal NNGP scalars q_i = scale * ||x_i||^2 + bias.
        q_diag_x = scale * (x * x).sum(dim=1) + bias
        q_diag_y = q_diag_x if symmetric else scale * (y * y).sum(dim=1) + bias

        if row_chunk is None:
            row_chunk = self._pick_row_chunk(n, m, x.dtype)
        row_chunk = max(1, int(row_chunk))

        out = torch.empty(n, m, dtype=x.dtype, device=x.device)
        y_t = y.T if symmetric else y.T.contiguous()

        for start in range(0, n, row_chunk):
            stop = min(start + row_chunk, n)
            x_chunk = x[start:stop]

            if symmetric:
                # Compute only the upper-triangular block of rows
                # [start:stop] × [start:m] and mirror the already-computed
                # left columns from previous iterations.
                y_slice = y[start:]
                dots = x_chunk @ y_slice.T
                q12 = scale * dots + bias
                q1 = q_diag_x[start:stop].unsqueeze(1)
                q2 = q_diag_y[start:].unsqueeze(0)
                out[start:stop, start:] = self.kernel_from_qs(q1, q2, q12)
                if start > 0:
                    out[start:stop, :start] = out[:start, start:stop].T
            else:
                dots = x_chunk @ y_t
                q12 = scale * dots + bias
                q1 = q_diag_x[start:stop].unsqueeze(1)
                q2 = q_diag_y.unsqueeze(0)
                out[start:stop] = self.kernel_from_qs(q1, q2, q12)

        # Defensive symmetrisation: float32 kernels can still emit tiny
        # asymmetries that trip downstream Cholesky / eigsh.
        if symmetric:
            out = 0.5 * (out + out.T)

        return out
