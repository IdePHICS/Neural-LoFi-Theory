# ruff: noqa: N803, N806
"""Heatmap generator: NLoFi-kernel test metrics over (K0, K1) × lambda.

Fixed ``n_train = 1000``.  For each ``(seed, K0, K1)`` we run the
3-layer NLoFi-kernel chain (eigen at layers 0, 1; no-eigen at layer 2)
once — building the layer-2 Gram via the package primitives — and then
call :func:`kernel_ridge_test_metrics_multi_lam` *once* with the
configured ``lambdas`` to obtain test MSE and classification error at
every lambda from a single ``eigh`` of the final Gram.  Per-cell wall
time at ``n_train = 1000`` is ~0.1–0.3 s on a single GPU, dominated by
two ``eigh`` calls (signed-cov on layer-0 Gram, signed-cov on layer-1
Gram) plus the final readout's eigh.

Package primitives used (this script is intentionally a thin orchestrator
around them; same calls as :class:`KernelSpectralTrainer._fit_mixed`):

* :func:`train_hierarchically.kernels.get_kernel("relu")`
* :func:`train_hierarchically.training.deep_nngp._clone_kernel_with`
* :func:`train_hierarchically.training.deep_nngp._auto_rescale_layer0_sigma_w`
* :func:`train_hierarchically.training.kernel_spectral._auto_rescale_kernel_for_state`
* :func:`train_hierarchically.training.kernel_spectral.spectral_signed_eig`
* :func:`train_hierarchically.training.kernel_ridge.kernel_ridge_test_metrics_multi_lam`

Output: ``results/heatmap_K0_K1_lambda/`` with one JSON cache per
``(seed, K0, K1)``:

    nlofik_heatmap__n{n}__K{k0}_{k1}__seed{s}.json

Each JSON contains ``test_mse``, ``test_clf_err``, ``test_sem``,
``test_accuracy_sem`` keyed by ``f"{lam:g}"`` so the plotter can read
all lambdas in one pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from train_hierarchically.datasets import load_tensors
from train_hierarchically.kernels import get_kernel
from train_hierarchically.training.deep_nngp import (
    _auto_rescale_layer0_sigma_w,
    _clone_kernel_with,
    deep_nngp_kernel,
)
from train_hierarchically.training.kernel_ridge import (
    kernel_ridge_test_metrics_multi_lam,
)
from train_hierarchically.training.kernel_spectral import (
    LayerState,
    _auto_rescale_kernel_for_state,
    spectral_signed_eig,
)
from train_hierarchically.utils.helpers import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATASET = "cifar10"
PRESET = "animal_vehicle"
N_TRAIN = 1000
N_TEST = 2000
# 16 log-spaced K values capped at n_train - 1 = 999 (kernel-trick rank
# limit).  K = 940 is the largest K below 999 in this geometric grid.
K_GRID: tuple[int, ...] = (
    10, 14, 19, 26, 34, 46, 62, 84, 113, 153, 207, 280, 380, 514, 696, 940,
)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
LAMBDA_LO_LOG10: float = -3.0
LAMBDA_HI_LOG10: float = 0.0
N_LAMBDAS: int = 10
DEVICE = "cuda:0"
OUT_DIR = Path("results/heatmap_K0_K1_lambda")
EIGEN_EPS = 1e-6


def _lambda_grid(
    n: int = N_LAMBDAS,
    lo: float = LAMBDA_LO_LOG10,
    hi: float = LAMBDA_HI_LOG10,
) -> tuple[float, ...]:
    """Log-spaced lambda grid; default 10 values in ``[1e-3, 1]``."""
    return tuple(float(x) for x in np.logspace(lo, hi, n))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    out_dir: Path, n: int, k0: int, k1: int, seed: int,
) -> Path:
    return out_dir / f"nlofik_heatmap__n{n}__K{k0}_{k1}__seed{seed}.json"


def _baseline_path(
    out_dir: Path, n: int, seed: int, n_layers: int,
) -> Path:
    """Cache file for the strict no-eigenreduce kernel baseline (one per
    ``(n, data_seed, n_layers)``); used as the deep-NNGP reference in
    the plotter.
    """
    return out_dir / (
        f"baseline_no_eigen__n{n}__L{n_layers}__dseed{seed}.json"
    )


def _baseline_complete(
    path: Path, lambdas: tuple[float, ...], center_kernel: bool,
) -> bool:
    if not path.exists():
        return False
    try:
        with open(path) as f:
            row = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    mse_dict = row.get("test_mse", {})
    err_dict = row.get("test_clf_err", {})
    if not isinstance(mse_dict, dict) or not isinstance(err_dict, dict):
        return False
    if bool(row.get("center_kernel", False)) != bool(center_kernel):
        return False
    expected = [f"{lam:g}" for lam in lambdas]
    return all(e in mse_dict and e in err_dict for e in expected)


def _cache_complete(
    path: Path, lambdas: tuple[float, ...], center_kernel: bool,
    per_direction_whiten: bool = False,
) -> bool:
    """Cache is *complete* iff JSON parses, has both metric dicts, all
    expected lambda keys are present, AND its ``center_kernel`` /
    ``per_direction_whiten`` fields match the requested convention.
    Files written before either field existed default to False."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            row = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    mse_dict = row.get("test_mse", {})
    err_dict = row.get("test_clf_err", {})
    if not isinstance(mse_dict, dict) or not isinstance(err_dict, dict):
        return False
    saved_center = bool(row.get("center_kernel", False))
    if saved_center != bool(center_kernel):
        return False
    saved_whiten = bool(row.get("per_direction_whiten", False))
    if saved_whiten != bool(per_direction_whiten):
        return False
    expected = [f"{lam:g}" for lam in lambdas]
    return all(e in mse_dict and e in err_dict for e in expected)


def _metric_dict_to_json(d: dict[float, float]) -> dict[str, float]:
    return {f"{lam:g}": float(v) for lam, v in d.items()}


def _whiten_per_direction(
    f_train: torch.Tensor, f_test: torch.Tensor, eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide each kept direction by its train-set std (no centering).

    Flattens the per-direction variance profile of the eigenvector-projected
    features so all K columns enter the next kernel evaluation with unit
    variance.  Test features are scaled by the same train-derived sigma to
    keep train/test on the same axis.
    """
    sigma = f_train.std(dim=0, keepdim=True).clamp_min(eps)
    return f_train / sigma, f_test / sigma


# ---------------------------------------------------------------------------
# Core: NLoFi-kernel chain via package primitives
# ---------------------------------------------------------------------------


def _layer0_features(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
    k0: int, eps: float = EIGEN_EPS, *,
    use_torch_gpu_eigh: bool = False,
    per_direction_whiten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer 0: auto-rescaled relu Gram + signed-cov eigen-reduce to k0.

    Returns ``(F1_train, F1_test)`` of shapes ``(n, k0)``, ``(n_test, k0)``.
    The kernel chain after this point is data-independent of the layer-0
    rescale (it absorbed the input-norm scaling into ``F1_train``).

    If ``per_direction_whiten`` is True, each of the ``k0`` projected
    columns is divided by its train-set std before being returned, so the
    layer-1 kernel sees unit-variance directions.
    """
    k0_kernel = _clone_kernel_with(
        get_kernel("relu"), _auto_rescale_layer0_sigma_w(x_train), 0.0,
    )
    g0_train = k0_kernel.gram(x_train)
    g0_cross = k0_kernel.gram(x_train, x_test)

    _, alpha0_np, f1_train_np = spectral_signed_eig(
        g0_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        k0,
        eps,
        use_torch_gpu_eigh=use_torch_gpu_eigh,
    )
    f1_train = torch.from_numpy(f1_train_np).to(
        device=x_train.device, dtype=x_train.dtype,
    )
    alpha0 = torch.from_numpy(alpha0_np).to(
        device=x_train.device, dtype=x_train.dtype,
    )
    f1_test = g0_cross.T @ alpha0  # (n_test, k0)
    if per_direction_whiten:
        f1_train, f1_test = _whiten_per_direction(f1_train, f1_test)
    return f1_train, f1_test


def _layer1_features(
    f1_train: torch.Tensor, y_train: torch.Tensor, f1_test: torch.Tensor,
    k1: int, eps: float = EIGEN_EPS, *,
    use_torch_gpu_eigh: bool = False,
    per_direction_whiten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer 1: auto-rescaled relu Gram on F1 + eigen-reduce to k1.

    If ``per_direction_whiten`` is True, the ``k1`` projected columns are
    divided by their train-set std before being returned, so the layer-2
    kernel sees unit-variance directions.
    """
    state1 = LayerState(f_train=f1_train, f_test=f1_test)
    k1_kernel = _auto_rescale_kernel_for_state(get_kernel("relu"), state1)
    g1_train = k1_kernel.gram(f1_train)
    g1_cross = k1_kernel.gram(f1_train, f1_test)

    _, alpha1_np, f2_train_np = spectral_signed_eig(
        g1_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        k1,
        eps,
        use_torch_gpu_eigh=use_torch_gpu_eigh,
    )
    f2_train = torch.from_numpy(f2_train_np).to(
        device=f1_train.device, dtype=f1_train.dtype,
    )
    alpha1 = torch.from_numpy(alpha1_np).to(
        device=f1_train.device, dtype=f1_train.dtype,
    )
    f2_test = g1_cross.T @ alpha1  # (n_test, k1)
    if per_direction_whiten:
        f2_train, f2_test = _whiten_per_direction(f2_train, f2_test)
    return f2_train, f2_test


def _layer2_gram(
    f2_train: torch.Tensor, f2_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer 2: auto-rescaled relu Gram on F2 (no eigen-reduce)."""
    state2 = LayerState(f_train=f2_train, f_test=f2_test)
    k2_kernel = _auto_rescale_kernel_for_state(get_kernel("relu"), state2)
    g2_train = k2_kernel.gram(f2_train)
    g2_cross = k2_kernel.gram(f2_train, f2_test)
    return g2_train, g2_cross


# ---------------------------------------------------------------------------
# No-eigenreduce baseline (deep NNGP via the package's deep_nngp_kernel)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_baseline_no_eigen(
    *,
    seed: int, n_train: int,
    x_test: torch.Tensor, y_test: torch.Tensor,
    dataset: str, preset: str,
    lambdas: tuple[float, ...],
    n_layers: int,
    dtype: torch.dtype, device: str, out_dir: Path,
    center_kernel: bool = True,
    use_torch_gpu_eigh: bool = False,
) -> None:
    """Compute the strict no-eigenreduce kernel baseline for one data seed.

    Calls :func:`deep_nngp_kernel` with ``auto_rescale=True`` for an
    ``n_layers``-deep ReLU NNGP composition (no signed eigen-reduction
    anywhere) and runs the same multi-lambda readout as the rest of the
    heatmap.  Cached as
    ``baseline_no_eigen__n{n}__L{n_layers}__dseed{ds}.json`` so the
    plotter can pick it up without reloading any data.
    """
    cp = _baseline_path(out_dir, n_train, seed, n_layers)
    if _baseline_complete(cp, lambdas, center_kernel):
        log.info("seed=%d  baseline cached (no-eigen, L=%d)", seed, n_layers)
        return

    t_load = time.time()
    x_tr, y_tr = load_tensors(
        dataset_type=dataset, split="train", n_samples=n_train,
        device=device, seed=seed, class_preset=preset,
    )
    x_tr = x_tr.to(dtype)
    y_tr = y_tr.to(dtype).view(-1)
    elapsed_load = time.time() - t_load

    t_kernel = time.time()
    K_train, K_cross = deep_nngp_kernel(
        get_kernel("relu"), x_tr, x_test, n_layers=n_layers,
        auto_rescale=True,
    )
    elapsed_kernel = time.time() - t_kernel

    t_readout = time.time()
    metrics = kernel_ridge_test_metrics_multi_lam(
        K_train.detach().cpu().numpy(),
        K_cross.detach().cpu().numpy(),
        y_tr.detach().cpu().numpy(),
        y_test.detach().cpu().numpy(),
        lambdas=lambdas,
        center_kernel=center_kernel,
        use_torch_gpu_eigh=use_torch_gpu_eigh,
    )
    elapsed_readout = time.time() - t_readout

    row = {
        "kind": "baseline_no_eigen",
        "dataset": dataset, "preset": preset,
        "n_train": int(n_train), "n_layers": int(n_layers),
        "seed": int(seed),
        "auto_rescale": True,
        "center_kernel": bool(center_kernel),
        "lambdas": [float(lam) for lam in lambdas],
        "test_mse": _metric_dict_to_json(
            {lam: m["test_mse"] for lam, m in metrics.items()},
        ),
        "test_clf_err": _metric_dict_to_json(
            {lam: m["test_clf_err"] for lam, m in metrics.items()},
        ),
        "test_sem": _metric_dict_to_json(
            {lam: m["test_sem"] for lam, m in metrics.items()},
        ),
        "test_accuracy_sem": _metric_dict_to_json(
            {lam: m["test_accuracy_sem"] for lam, m in metrics.items()},
        ),
        "elapsed_load_s": elapsed_load,
        "elapsed_kernel_s": elapsed_kernel,
        "elapsed_readout_s": elapsed_readout,
    }
    with open(cp, "w") as f:
        json.dump(row, f, indent=2)
    log.info(
        "seed=%d  baseline (no-eigen, L=%d)  kernel=%.1fs readout=%.1fs",
        seed, n_layers, elapsed_kernel, elapsed_readout,
    )


# ---------------------------------------------------------------------------
# Main driver: one (n_train, seed) cell across the (K0, K1) grid
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_one_seed(
    *,
    seed: int, n_train: int,
    x_test: torch.Tensor, y_test: torch.Tensor,
    dataset: str, preset: str,
    k_grid: tuple[int, ...], lambdas: tuple[float, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
    center_kernel: bool = True,
    use_torch_gpu_eigh: bool = False,
    per_direction_whiten: bool = False,
) -> None:
    """Process one (n_train, seed) cell across all (K0, K1) in ``k_grid``.

    Layer-0 and layer-1 intermediate features are shared across
    ``K1``-values for fixed ``K0`` (since layer-1's signed eigh depends
    only on ``K0`` through the F1 features).  Skip-if-cached at the
    per-cell granularity so partial reruns work.
    """
    valid_ks = tuple(k for k in k_grid if k <= n_train - 1)
    pairs_to_run: list[tuple[int, int]] = []
    for k0 in valid_ks:
        for k1 in valid_ks:
            cp = _cache_path(out_dir, n_train, k0, k1, seed)
            if not _cache_complete(
                cp, lambdas, center_kernel, per_direction_whiten,
            ):
                pairs_to_run.append((k0, k1))
    if not pairs_to_run:
        log.info("seed=%d  all cached", seed)
        return

    # Group missing pairs by k0 so layer-1 eigen can be recomputed only
    # when k0 changes (layer-1 Gram depends on the F1 features which
    # depend on k0).
    k0_groups: dict[int, list[int]] = {}
    for k0, k1 in pairs_to_run:
        k0_groups.setdefault(k0, []).append(k1)

    t_load = time.time()
    x_tr, y_tr = load_tensors(
        dataset_type=dataset, split="train", n_samples=n_train,
        device=device, seed=seed, class_preset=preset,
    )
    x_tr = x_tr.to(dtype)
    y_tr = y_tr.to(dtype).view(-1)
    log.info(
        "seed=%d  loaded n_train=%d in %.1fs",
        seed, n_train, time.time() - t_load,
    )

    for k0 in sorted(k0_groups):
        t_k0 = time.time()
        f1_tr, f1_te = _layer0_features(
            x_tr, y_tr, x_test, k0,
            use_torch_gpu_eigh=use_torch_gpu_eigh,
            per_direction_whiten=per_direction_whiten,
        )
        for k1 in k0_groups[k0]:
            t_cell = time.time()
            f2_tr, f2_te = _layer1_features(
                f1_tr, y_tr, f1_te, k1,
                use_torch_gpu_eigh=use_torch_gpu_eigh,
                per_direction_whiten=per_direction_whiten,
            )
            g2_tr, g2_cr = _layer2_gram(f2_tr, f2_te)
            metrics = kernel_ridge_test_metrics_multi_lam(
                g2_tr.detach().cpu().numpy(),
                g2_cr.detach().cpu().numpy(),
                y_tr.detach().cpu().numpy(),
                y_test.detach().cpu().numpy(),
                lambdas=lambdas,
                center_kernel=center_kernel,
                use_torch_gpu_eigh=use_torch_gpu_eigh,
            )
            elapsed = time.time() - t_cell
            row = {
                "kind": "nlofik_heatmap",
                "dataset": dataset, "preset": preset,
                "n_train": int(n_train),
                "k0": int(k0), "k1": int(k1), "seed": int(seed),
                "auto_rescale": True,
                "center_kernel": bool(center_kernel),
                "per_direction_whiten": bool(per_direction_whiten),
                "lambdas": [float(lam) for lam in lambdas],
                "test_mse": _metric_dict_to_json(
                    {lam: m["test_mse"] for lam, m in metrics.items()},
                ),
                "test_clf_err": _metric_dict_to_json(
                    {lam: m["test_clf_err"] for lam, m in metrics.items()},
                ),
                "test_sem": _metric_dict_to_json(
                    {lam: m["test_sem"] for lam, m in metrics.items()},
                ),
                "test_accuracy_sem": _metric_dict_to_json(
                    {lam: m["test_accuracy_sem"] for lam, m in metrics.items()},
                ),
                "elapsed_s": elapsed,
            }
            cp = _cache_path(out_dir, n_train, k0, k1, seed)
            with open(cp, "w") as f:
                json.dump(row, f, indent=2)
            log.info(
                "  seed=%d K=(%d,%d)  mse[lam=min,max]=%.4f,%.4f  "
                "clf[min,max]=%.3f,%.3f  (%.2fs)",
                seed, k0, k1,
                row["test_mse"][f"{lambdas[0]:g}"],
                row["test_mse"][f"{lambdas[-1]:g}"],
                row["test_clf_err"][f"{lambdas[0]:g}"],
                row["test_clf_err"][f"{lambdas[-1]:g}"],
                elapsed,
            )
        log.info(
            "seed=%d K0=%d done (%d K1 vals, %.1fs)",
            seed, k0, len(k0_groups[k0]), time.time() - t_k0,
        )

    del x_tr, y_tr


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--preset", type=str, default=PRESET)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float64"],
    )
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument(
        "--k-grid", type=int, nargs="+", default=list(K_GRID),
        help="Log-spaced K values (each capped at n_train - 1).",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help="Each seed varies BOTH the data subsample and (in the "
             "finite-P case) the RF weights.  In the kernel limit only "
             "data subsample varies.",
    )
    p.add_argument(
        "--n-lambdas", type=int, default=N_LAMBDAS,
        help="Number of log-spaced lambda values.",
    )
    p.add_argument(
        "--lambda-range", type=float, nargs=2,
        default=(LAMBDA_LO_LOG10, LAMBDA_HI_LOG10),
        metavar=("LO_LOG10", "HI_LOG10"),
        help="Log10-bounds for the lambda grid; default -3 0 (i.e. [1e-3, 1]).",
    )
    p.add_argument(
        "--no-center-kernel", action="store_true",
        help="Use the no-centering kernel-ridge readout (matches the "
             "older scripts' ``ridge_solve_multi_lam``).  Default uses "
             "double-centering, the strict P->inf limit of "
             "``RidgeCV(fit_intercept=True)``.",
    )
    p.add_argument(
        "--torch-gpu-eigh", action="store_true",
        help="Route every eigh inside ``spectral_signed_eig`` and "
             "``kernel_ridge_test_metrics_multi_lam`` through "
             "``torch.linalg.eigh`` on CUDA.  Roughly 25x faster at "
             "n=10000 than the default scipy CPU dispatch — needed for "
             "large-n heatmaps.",
    )
    p.add_argument(
        "--fix-data-seed", type=int, default=None, metavar="N",
        help="Force every per-seed run to load the same training subsample "
             "(``load_tensors(seed=N)``).  Overrides ``--seeds`` to a single "
             "value ``[N]`` and switches off data-subsample averaging — "
             "useful for studying the heatmap structure on one fixed dataset.",
    )
    p.add_argument(
        "--baseline-n-layers", type=int, default=3,
        help="Depth of the strict no-eigenreduce baseline (deep ReLU NNGP "
             "via ``deep_nngp_kernel``).  Default 3 to match the heatmap's "
             "3-layer NLoFi-kernel architecture.",
    )
    p.add_argument(
        "--skip-baseline", action="store_true",
        help="Do not compute the no-eigenreduce baseline (cells only).",
    )
    p.add_argument(
        "--per-direction-whiten", action="store_true",
        help="After each layer's signed-eig projection, divide each kept "
             "direction by its train-set std before feeding it to the next "
             "kernel.  Flattens the per-direction variance profile, so all "
             "K columns enter the next layer with unit variance.  Different "
             "model than the default — kernel and finite-P pipelines do "
             "NOT correspond to the same NNGP limit when this is on.",
    )
    args = p.parse_args()
    center_kernel = not args.no_center_kernel
    if args.fix_data_seed is not None:
        if list(args.seeds) != list(SEEDS):
            log.warning(
                "--fix-data-seed=%d overrides --seeds; using a single seed.",
                args.fix_data_seed,
            )
        args.seeds = [int(args.fix_data_seed)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    # Pin the *default* CUDA device to whichever GPU the user picked, so
    # the package helpers (``spectral_signed_eig`` and
    # ``kernel_ridge_test_metrics_multi_lam``) launching their internal
    # eigh through ``torch.cuda.current_device()`` actually run on the
    # requested GPU when multiple sweeps share the box.
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_device(device)
    elif isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    lambdas = _lambda_grid(
        n=args.n_lambdas, lo=args.lambda_range[0], hi=args.lambda_range[1],
    )
    log.info(
        "n_train=%d K-grid (%d values)=%s seeds=%s lambdas=%s "
        "center_kernel=%s torch_gpu_eigh=%s per_direction_whiten=%s",
        args.n_train, len(args.k_grid), args.k_grid,
        args.seeds, [f"{lam:g}" for lam in lambdas],
        center_kernel, args.torch_gpu_eigh, args.per_direction_whiten,
    )

    x_test, y_test = load_tensors(
        dataset_type=args.dataset, split="test", n_samples=args.n_test,
        device=device, seed=0, class_preset=args.preset,
    )
    x_test = x_test.to(dtype)
    y_test = y_test.to(dtype).view(-1)

    t_start = time.time()
    for seed in args.seeds:
        run_one_seed(
            seed=int(seed), n_train=int(args.n_train),
            x_test=x_test, y_test=y_test,
            dataset=args.dataset, preset=args.preset,
            k_grid=tuple(int(k) for k in args.k_grid),
            lambdas=lambdas,
            dtype=dtype, device=device, out_dir=out_dir,
            center_kernel=center_kernel,
            use_torch_gpu_eigh=args.torch_gpu_eigh,
            per_direction_whiten=args.per_direction_whiten,
        )
        if not args.skip_baseline:
            run_baseline_no_eigen(
                seed=int(seed), n_train=int(args.n_train),
                x_test=x_test, y_test=y_test,
                dataset=args.dataset, preset=args.preset,
                lambdas=lambdas, n_layers=int(args.baseline_n_layers),
                dtype=dtype, device=device, out_dir=out_dir,
                center_kernel=center_kernel,
                use_torch_gpu_eigh=args.torch_gpu_eigh,
            )
        log.info(
            "elapsed so far: %.1f min", (time.time() - t_start) / 60.0,
        )
    log.info("Total elapsed: %.1f min", (time.time() - t_start) / 60.0)


if __name__ == "__main__":
    main()
