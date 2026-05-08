"""Kernel ridge regression with efficient leave-one-out CV over alpha.

This is the strict ``p -> infinity`` limit of random-feature ridge:

    finite p:  w = h_train^T (h_train h_train^T + a I)^{-1} y
               y_pred = h_test w
    p -> inf:  y_pred = K_cross^T (K_train + a_kr I)^{-1} y_train,  a_kr = a/p

(Here ``h_train h_train^T`` has entries of order ``p`` while the NNGP
Gram ``K`` has entries of order 1, so the ridge coefficient rescales.)

The CV scan is O(n^2 * n_alpha) after one O(n^3) eigendecomposition via
the identity (for leave-one-out residuals of a linear smoother):

    r_i(a) = (y_i - [S(a) y]_i) / (1 - S_ii(a))
    S(a) = K (K + a I)^{-1} = Q diag(l_i / (l_i + a)) Q^T

Both diag(S) and ``S y`` are trivially computable once ``K`` is
diagonalised.

The returned object mirrors the minimal surface used by
:func:`train_hierarchically.training.base.fit_ridge_readout` so it can
drop into kernel-based training flows without changing downstream
schemas.

For the multi-lambda regime where the readout regularisation is *given*
(not selected), :func:`kernel_ridge_test_metrics_multi_lam` provides an
``O(n^3 + n^2 |Lambda|)`` test-only path: one eigendecomposition of
``K_train`` followed by a vectorised ``alpha_lambda = Q diag(1 / (mu +
lambda)) Q^T y`` per lambda.  This is the package equivalent of the
script ``ridge_solve_multi_lam`` used by ``generate_nlofi_appendix_data``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from torch import Tensor


@dataclass(slots=True)
class KernelRidgeFit:
    """Result of a kernel-ridge fit with CV over alpha.

    Attributes
    ----------
    alpha : float
        Selected regularisation strength.
    dual_coef : np.ndarray
        Solution to ``(K_train + alpha I) c = y_train - bias``; shape
        ``(n,)``.
    bias : float
        Mean-centered intercept (``mean(y_train)``).
    loo_mse : float
        Leave-one-out MSE at the selected alpha (diagnostic).
    train_mse : float
        In-sample MSE (computed from the fitted dual coefficients).
    cv_curve : np.ndarray
        Per-alpha LOO MSE values, shape ``(len(alphas),)``.
    """

    alpha: float
    dual_coef: np.ndarray
    bias: float
    loo_mse: float
    train_mse: float
    cv_curve: np.ndarray


def kernel_ridge_cv(
    K_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alphas: np.ndarray,
    center: bool = True,
) -> KernelRidgeFit:
    """Kernel ridge regression with leave-one-out CV over *alphas*.

    Parameters
    ----------
    K_train : ndarray, shape (n, n)
        Symmetric PSD kernel Gram matrix.
    y_train : ndarray, shape (n,)
        Training targets.
    alphas : ndarray
        Positive candidate alpha values.
    center : bool
        Subtract ``y_train.mean()`` before solving; report it as
        ``bias``.  This is what ``sklearn.RidgeCV`` does by default.

    Returns
    -------
    KernelRidgeFit
        Selected alpha and the solution dual coefficients.
    """
    n = K_train.shape[0]
    if K_train.shape != (n, n):
        raise ValueError(f"K_train must be square; got {K_train.shape}")
    if y_train.shape != (n,):
        raise ValueError(
            f"y_train shape {y_train.shape} does not match K_train ({n},)"
        )

    y = y_train.astype(np.float64, copy=True)
    if center:
        bias = float(y.mean())
        y = y - bias
    else:
        bias = 0.0

    # Symmetrise defensively (float32 ops can leave tiny asymmetries) and
    # eigendecompose once.
    k_sym = 0.5 * (K_train + K_train.T)
    eigvals, q = np.linalg.eigh(k_sym.astype(np.float64))
    # eigvals are ascending; clamp tiny negatives from floating-point to 0.
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    qty = q.T @ y  # (n,)  coefficients of y in the eigenbasis

    # Per-alpha LOO scan.  For each alpha:
    #   s_coef[i] = lambda_i / (lambda_i + alpha)
    #   diag_S[j] = sum_i q[j,i]^2 * s_coef[i]
    #   [S y][j]  = sum_i q[j,i] * s_coef[i] * qty[i]
    # Construct Q_sq = q ** 2 (n, n) once; reuse per alpha.
    q_sq = q * q

    best: tuple[float, float] | None = None  # (alpha, loo_mse)
    cv_curve = np.empty(len(alphas), dtype=np.float64)

    for idx, alpha in enumerate(alphas):
        a = float(alpha)
        s_coef = eigvals / (eigvals + a)  # (n,)
        diag_s = q_sq @ s_coef  # (n,)
        s_y = q @ (s_coef * qty)  # (n,)
        denom = 1.0 - diag_s
        # Guard against division by zero when a kernel eigenmode is
        # negligible AND the corresponding leverage is 1 (happens only
        # pathologically; treat as infinite LOO residual).
        denom = np.where(denom > 1e-12, denom, 1e-12)
        residuals = (y - s_y) / denom
        loo_mse = float(np.mean(residuals * residuals))
        cv_curve[idx] = loo_mse
        if best is None or loo_mse < best[1]:
            best = (a, loo_mse)

    assert best is not None
    alpha_star, loo_mse_star = best

    # Compute dual coefficients at alpha_star.
    c = q @ (qty / (eigvals + alpha_star))  # (K + a I)^-1 y
    # In-sample predictions.
    y_pred_train = k_sym @ c + bias
    train_mse = float(np.mean((y_train - y_pred_train) ** 2))

    return KernelRidgeFit(
        alpha=alpha_star,
        dual_coef=c,
        bias=bias,
        loo_mse=loo_mse_star,
        train_mse=train_mse,
        cv_curve=cv_curve,
    )


def kernel_ridge_predict(
    fit: KernelRidgeFit, K_cross: np.ndarray
) -> np.ndarray:
    """Predict on test points.

    Parameters
    ----------
    fit : KernelRidgeFit
        Result of :func:`kernel_ridge_cv`.
    K_cross : ndarray, shape (n_train, n_test)
        Cross Gram matrix ``K(X_train, X_test)``.
    """
    # y_pred[j] = sum_i K_cross[i, j] * c[i] + bias
    return K_cross.T @ fit.dual_coef + fit.bias


def fit_kernel_ridge_readout(
    model,  # noqa: ANN001 — duck-typed nn.Module with ridge_* parameters
    K_train: Tensor,
    y_train: Tensor,
    *,
    K_cross: Tensor | None = None,
    y_test_np: np.ndarray | None = None,
    alpha_min: float = -6.0,
    alpha_max: float = 6.0,
    alpha_num: int = 500,
    verbose: bool = True,
) -> dict[str, object]:
    """Kernel ridge readout with the same return schema as ``fit_ridge_readout``.

    Installs ``dual_coef`` on *model* under the existing
    ``ridge_weight`` / ``ridge_bias`` names so
    :meth:`KernelSpectralModel.forward` needs no change.

    Under the storage convention used by ``forward``:
      y_pred(x_test) = K_cross.T @ ridge_weight + ridge_bias
    so we set ``ridge_weight = dual_coef`` (shape ``(n_train,)``) and
    ``ridge_bias = fit.bias``.
    """
    import logging

    import torch
    from torch import nn

    log = logging.getLogger(__name__)

    alphas = np.logspace(alpha_min, alpha_max, alpha_num)
    # ``.double()`` rather than ``.float()``: if the trainer promoted the
    # final Gram to float64 (to avoid ~1e-6 quadrature roundoff on tiny
    # eigenvalues of the sigmoid NNGP), we must not demote it here.  For
    # a float32 input this is a lossless upcast.
    k_train_np = K_train.cpu().double().numpy()
    y_train_np = y_train.cpu().numpy()

    fit = kernel_ridge_cv(k_train_np, y_train_np, alphas=alphas)

    # Store on the *model's* dtype/device — the kernel matrices we just
    # ingested may have been promoted to CPU/float64 for numerical
    # stability, but the model itself runs in its native dtype (usually
    # float32 on MPS/CUDA) and the stored readout must match.
    if model.x_train.numel() > 0:
        device = model.x_train.device
        dtype = model.x_train.dtype
    else:
        device, dtype = K_train.device, K_train.dtype
    model.ridge_weight = nn.Parameter(
        torch.from_numpy(fit.dual_coef).to(device=device, dtype=dtype),
        requires_grad=False,
    )
    model.ridge_bias = nn.Parameter(
        torch.tensor([fit.bias], device=device, dtype=dtype),
        requires_grad=False,
    )

    # Binary classification: threshold at 0 (targets in {-1, +1}).
    y_pred_train = k_train_np @ fit.dual_coef + fit.bias
    train_acc = float(
        np.mean(y_train_np == np.where(y_pred_train >= 0, 1.0, -1.0))
    )

    if verbose:
        log.info(
            "Kernel ridge: alpha=%.4e, loo_mse=%.6f, train_mse=%.6f",
            fit.alpha, fit.loo_mse, fit.train_mse,
        )

    final: dict[str, object] = {
        "train_mse": fit.train_mse,
        "ridge_alpha": fit.alpha,
        "accuracy": train_acc,
        "loo_mse": fit.loo_mse,
    }

    if K_cross is not None and y_test_np is not None:
        k_cross_np = K_cross.cpu().double().numpy()
        y_pred_test = kernel_ridge_predict(fit, k_cross_np)
        test_residuals = (y_test_np - y_pred_test) ** 2
        test_mse = float(np.mean(test_residuals))
        test_sem = float(np.std(test_residuals) / np.sqrt(len(y_test_np)))

        y_pred_labels = np.where(y_pred_test >= 0, 1.0, -1.0)
        test_acc = float(np.mean(y_test_np == y_pred_labels))
        test_acc_sem = float(
            np.sqrt(max(test_acc * (1.0 - test_acc), 0.0) / len(y_test_np))
        )

        final["test_mse"] = test_mse
        final["test_sem"] = test_sem
        final["test_accuracy"] = test_acc
        final["test_accuracy_sem"] = test_acc_sem
        # Explicit ``test_clf_err`` field aligns with the NLoFi script
        # JSON schema; redundant with ``1 - test_accuracy`` but lets
        # downstream consumers (and JSON regen tests) read either name.
        final["test_clf_err"] = 1.0 - test_acc

        if verbose:
            log.info(
                "  Test: mse=%.6f, sem=%.6f, accuracy=%.4f±%.4f",
                test_mse, test_sem, test_acc, test_acc_sem,
            )

    return final


def _to_numpy(a: ArrayLike) -> np.ndarray:
    """Materialise a torch tensor or array-like as a contiguous numpy array."""
    if isinstance(a, Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def kernel_ridge_test_metrics_multi_lam(
    K_train: ArrayLike,
    K_cross: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
    *,
    lambdas: Sequence[float],
    center_kernel: bool = True,
    use_torch_gpu_eigh: bool = False,
) -> dict[float, dict[str, float]]:
    """Test-only kernel-ridge metrics across many lambdas via one eigendecomp.

    Solves ``(K_train + lambda I) alpha = y_train`` for every ``lambda``
    in ``lambdas`` using the eigen-form ``alpha = Q diag(1 / (mu +
    lambda)) Q^T y_train`` so the dominant cost is one ``O(n^3)``
    ``eigh`` of ``K_train`` plus ``O(n^2)`` per lambda.

    When ``center_kernel=True`` (default) the readout is the strict
    ``P -> infinity`` limit of ``sklearn.linear_model.RidgeCV(
    fit_intercept=True)`` on random features: train ``K`` is double-
    centered, cross ``K`` is centered with the same training-side row
    means and total mean, ``y`` is mean-centered, and the predictions add
    ``mean(y_train)`` back as an intercept.  This is what one usually
    means by "ridge with intercept" in the kernel limit.

    When ``center_kernel=False`` no centering is applied: the readout
    matches the NLoFi-appendix script's ``ridge_solve_multi_lam``
    convention (no intercept, ``y_pred = K_cross^T alpha``).  Use this
    when reproducing the script's bit-exact JSON caches.

    Classification error follows ``sign(y_pred) != y_test``: at
    ``y_pred == 0`` the prediction sign is 0, which is unequal to
    either ``+1`` or ``-1`` and therefore counts as an error.

    Parameters
    ----------
    K_train : array-like, shape (n, n)
        Symmetric PSD kernel Gram on training points.  Defensively
        symmetrised internally; the caller need not pre-symmetrise.
    K_cross : array-like, shape (n, n_test)
        Cross Gram ``K(X_train, X_test)``.
    y_train : array-like, shape (n,)
        Training targets.
    y_test : array-like, shape (n_test,)
        Test targets.  Used for ``test_mse`` and ``test_clf_err``.
    lambdas : sequence of float
        Ridge regularisation values; each produces its own alpha and its
        own metrics dict.
    center_kernel : bool, default ``True``
        Whether to do the doubly-centered (intercept) readout.  See
        above for the two conventions.

    Returns
    -------
    dict[float, dict[str, float]]
        ``{lam: {"test_mse", "test_sem", "test_accuracy",
        "test_accuracy_sem", "test_clf_err"}}``.  ``test_clf_err = 1 -
        test_accuracy`` is included for direct alignment with the
        script's JSON cache schema.
    """
    k_train = _to_numpy(K_train).astype(np.float64, copy=False)
    k_cross = _to_numpy(K_cross).astype(np.float64, copy=False)
    y_tr = _to_numpy(y_train).astype(np.float64, copy=False).reshape(-1)
    y_te = _to_numpy(y_test).astype(np.float64, copy=False).reshape(-1)

    n = k_train.shape[0]
    if k_train.shape != (n, n):
        raise ValueError(f"K_train must be square; got {k_train.shape}")
    if k_cross.shape[0] != n:
        raise ValueError(
            f"K_cross has {k_cross.shape[0]} train rows; expected {n}"
        )
    if y_tr.shape != (n,):
        raise ValueError(
            f"y_train shape {y_tr.shape} does not match K_train ({n},)"
        )
    n_test = k_cross.shape[1]
    if y_te.shape != (n_test,):
        raise ValueError(
            f"y_test shape {y_te.shape} does not match K_cross "
            f"({n_test},)"
        )

    # Optional double-centering — strict P->inf limit of RidgeCV(
    # fit_intercept=True).  See the docstring; centering is applied
    # entirely in float64 with copies so the caller's arrays are not
    # mutated.
    if center_kernel:
        row_means_train = k_train.mean(axis=1, keepdims=True)         # (n, 1)
        col_means_train = k_train.mean(axis=0, keepdims=True)         # (1, n)
        total_mean_train = float(k_train.mean())
        k_train = (
            k_train - row_means_train - col_means_train + total_mean_train
        )
        # K_cross uses TRAIN-side row means (these cancel the train
        # feature mean, mu = mean(H_train)) and its own column means
        # (which equal mean(H_train) . H_test_j, also expressible as
        # mean over train rows of K_cross).  Total mean is the train one.
        col_means_cross = k_cross.mean(axis=0, keepdims=True)         # (1, n_test)
        k_cross = (
            k_cross - row_means_train - col_means_cross + total_mean_train
        )
        bias = float(y_tr.mean())
        y_tr = y_tr - bias
    else:
        bias = 0.0

    # One eigendecomposition of K_train, then per-lambda solve in the
    # eigenbasis.  Defensive symmetrisation matches the script.  When
    # ``use_torch_gpu_eigh`` is on we route the eigh through
    # ``torch.linalg.eigh`` on the active CUDA device — at n>=5000 this
    # is roughly 25x faster than ``np.linalg.eigh`` on CPU, and the
    # downstream per-lambda solve is cheap enough that staying on the
    # device-host boundary doesn't matter.
    k_sym = 0.5 * (k_train + k_train.T)
    if use_torch_gpu_eigh:
        import torch as _torch
        if _torch.cuda.is_available():
            device = _torch.device(
                "cuda", _torch.cuda.current_device(),
            )
        else:
            device = _torch.device("cpu")
        k_sym_t = _torch.from_numpy(k_sym).to(
            device=device, dtype=_torch.float64,
        )
        mu_t, q_t = _torch.linalg.eigh(k_sym_t)
        mu = mu_t.detach().cpu().numpy()
        q = q_t.detach().cpu().numpy()
    else:
        mu, q = np.linalg.eigh(k_sym)
    qty = q.T @ y_tr  # (n,)
    k_proj = k_cross.T @ q  # (n_test, n)

    out: dict[float, dict[str, float]] = {}
    for lam in lambdas:
        lam_f = float(lam)
        alpha_eig = qty / (mu + lam_f)  # (n,) dual coefficients in eigenbasis
        y_pred = k_proj @ alpha_eig + bias  # (n_test,)
        sq_residuals = (y_pred - y_te) ** 2
        test_mse = float(sq_residuals.mean())
        test_sem = float(sq_residuals.std() / np.sqrt(n_test))

        # Classification error: sign(y_pred) ≠ y_test.  Matches the
        # script's torch.sign convention; y_pred == 0 maps to sign 0,
        # which is unequal to ±1 and therefore counts as an error.
        y_pred_signs = np.sign(y_pred)
        test_clf_err = float((y_pred_signs != y_te).mean())
        test_acc = 1.0 - test_clf_err
        test_acc_sem = float(
            np.sqrt(max(test_acc * (1.0 - test_acc), 0.0) / n_test)
        )

        out[lam_f] = {
            "test_mse": test_mse,
            "test_sem": test_sem,
            "test_accuracy": test_acc,
            "test_accuracy_sem": test_acc_sem,
            "test_clf_err": test_clf_err,
        }
    return out


__all__ = [
    "KernelRidgeFit",
    "fit_kernel_ridge_readout",
    "kernel_ridge_cv",
    "kernel_ridge_predict",
    "kernel_ridge_test_metrics_multi_lam",
]
