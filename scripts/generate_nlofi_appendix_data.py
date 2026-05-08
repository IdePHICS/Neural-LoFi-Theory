# ruff: noqa: N803, N806
"""Data generator for the NLoFi-NNGP convergence appendix figure.

Architecture (same as ``check_nlofi_kernel_convergence.py``): three layers
with signed-cov eigen-reduction at the first two and a kernel-ridge
readout at the third.  Sweeps over

* ``(K0, K1)`` pairs (rows of the figure),
* ``λ`` values (columns of the figure),
* ``n_train`` (separate lines within each subplot),
* ``P`` (x-axis), and
* ``seed`` (averaged over).

Optimisations:

* **Multi-λ batching.** The final Gram is independent of λ, so we
  diagonalise once and solve ``(G + λ I) α = y`` for all λ simultaneously
  via the eigen-form ``α = Q diag(1/(μ + λ)) Q^T y``.  One JSON cache
  per ``(K0, K1, n, P, seed)`` (or ``(K0, K1, n, ds)`` for the kernel
  block) holding all λ.

* **Layer-0 Gram sharing.** Layer-0's Gram depends only on
  ``(n_train, P, seed)``; for each fixed triple we build it once and run
  the remaining two layers separately for each ``(K0, K1)`` pair.  We
  use ``Generator.get_state``/``set_state`` so each ``(K0, K1)`` pipeline
  consumes the same layer-1/layer-2 randomness it would have had in
  isolation.

Output: ``results/nlofi_appendix/`` with one JSON per evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from train_hierarchically.datasets import load_tensors
from train_hierarchically.utils.helpers import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (CLI overridable)
# ---------------------------------------------------------------------------

DATASET = "cifar10"
PRESET = "animal_vehicle"
K_PAIRS: tuple[tuple[int, int], ...] = ((50, 50), (100, 100), (200, 200))
LAMBDAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0)
N_TRAIN_VALUES: tuple[int, ...] = (500, 1000, 2000, 5000)
P_VALUES: tuple[int, ...] = (
    100, 300, 1000, 3000, 10000, 30000, 100000, 200000,
)
N_TEST = 2000
SEEDS: tuple[int, ...] = tuple(range(10))
DATA_SEEDS: tuple[int, ...] = tuple(range(10))
DEVICE_PREF = "cuda:1"
OUT_DIR = Path("results/nlofi_appendix")
EIGEN_EPS = 1e-6
CHUNK_P = 20000


# ---------------------------------------------------------------------------
# relu NNGP kernel (Cho-Saul J_1, closed form)
# ---------------------------------------------------------------------------


def relu_arc_cosine(q: torch.Tensor) -> torch.Tensor:
    q_diag = q.diagonal()
    outer = q_diag.unsqueeze(0) * q_diag.unsqueeze(1)
    radial = torch.sqrt(torch.clamp_min(outer, 1e-30))
    cos_t = torch.clamp(q / radial, -1.0, 1.0)
    theta = torch.arccos(cos_t)
    return radial / (2.0 * math.pi) * (
        torch.sin(theta) + (math.pi - theta) * cos_t
    )


# ---------------------------------------------------------------------------
# Signed-cov eigen-reduction (kernel-trick / whitened n×n form)
# ---------------------------------------------------------------------------


@torch.no_grad()
def spectral_signed_eig(
    G: torch.Tensor, y: torch.Tensor, k: int, eps: float = EIGEN_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype_in = G.dtype
    G64 = (0.5 * (G + G.T)).to(torch.float64)
    y64 = y.to(torch.float64)
    mu, Q = torch.linalg.eigh(G64)
    mu_clip = torch.clamp(mu, min=eps)
    sqrt_mu = torch.sqrt(mu_clip)
    inv_sqrt = 1.0 / sqrt_mu
    G_half = (Q * sqrt_mu) @ Q.T
    G_half = 0.5 * (G_half + G_half.T)
    M = (G_half * y64.view(1, -1)) @ G_half
    M = 0.5 * (M + M.T)
    eigvals, B = torch.linalg.eigh(M)
    idx = torch.argsort(eigvals.abs(), descending=True)[:k]
    eigvals = eigvals[idx]
    B = B[:, idx]
    A = (Q * inv_sqrt) @ (Q.T @ B)
    F = G_half @ B
    return eigvals.to(dtype_in), A.to(dtype_in), F.to(dtype_in)


# ---------------------------------------------------------------------------
# Gram builders
# ---------------------------------------------------------------------------


@torch.no_grad()
def gram_kernel(
    F_train: torch.Tensor, F_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    F_full = torch.cat([F_train, F_test], dim=0)
    n = F_train.shape[0]
    c = torch.sqrt(F_train.pow(2).sum(dim=1).mean())
    q = (F_full @ F_full.T) / (c ** 2)
    K = relu_arc_cosine(q)
    return K[:n, :n], K[:n, n:]


@torch.no_grad()
def gram_finite(
    F_train: torch.Tensor, F_test: torch.Tensor,
    P: int, gen: torch.Generator, dtype: torch.dtype, device: str,
    chunk_p: int = CHUNK_P,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = F_train.shape[1]
    n, n_t = F_train.shape[0], F_test.shape[0]
    c = torch.sqrt(F_train.pow(2).sum(dim=1).mean())
    inv_sqrt_p = 1.0 / math.sqrt(P)
    G_train = torch.zeros(n, n, device=device, dtype=dtype)
    G_cross = torch.zeros(n, n_t, device=device, dtype=dtype)
    for start in range(0, P, chunk_p):
        stop = min(start + chunk_p, P)
        cp = stop - start
        W = torch.randn(
            d, cp, generator=gen, device=device, dtype=dtype,
        )
        z_tr = torch.relu(F_train @ W / c) * inv_sqrt_p
        z_te = torch.relu(F_test @ W / c) * inv_sqrt_p
        G_train += z_tr @ z_tr.T
        G_cross += z_tr @ z_te.T
        del W, z_tr, z_te
    return G_train, G_cross


# ---------------------------------------------------------------------------
# Multi-λ ridge solve via eigendecomposition of the final Gram
# ---------------------------------------------------------------------------


@torch.no_grad()
def ridge_solve_multi_lam(
    G_train: torch.Tensor, G_cross: torch.Tensor,
    y_train: torch.Tensor, y_test: torch.Tensor,
    lambdas: tuple[float, ...],
) -> tuple[dict[float, float], dict[float, float]]:
    """Solve ``(G + λ I) α = y`` for many λ via one eigendecomposition.

    Returns ``(mses, clf_errs)``: per-λ test MSE and per-λ test
    classification error (0-1 loss assuming ``y_test ∈ {-1, +1}``).
    """
    G_sym = 0.5 * (G_train + G_train.T)
    mu, Q = torch.linalg.eigh(G_sym.to(torch.float64))
    qy = Q.T @ y_train.to(torch.float64)
    G_proj = G_cross.T.to(torch.float64) @ Q  # (n_test, n)
    y_test_64 = y_test.to(torch.float64)
    mse_out: dict[float, float] = {}
    err_out: dict[float, float] = {}
    for lam in lambdas:
        alpha = qy / (mu + lam)
        y_pred = G_proj @ alpha
        mse_out[float(lam)] = float(
            (y_pred - y_test_64).pow(2).mean().item()
        )
        err_out[float(lam)] = float(
            (torch.sign(y_pred) != y_test_64).to(torch.float64).mean().item()
        )
    return mse_out, err_out


# ---------------------------------------------------------------------------
# Pipelines (multi-λ)
# ---------------------------------------------------------------------------


@torch.no_grad()
def nlofi_kernel_multi_lam(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    k0: int, k1: int, lambdas: tuple[float, ...],
) -> tuple[dict[float, float], dict[float, float]]:
    F_train, F_test = X_train, X_test
    for k in (k0, k1):
        G_train, G_cross = gram_kernel(F_train, F_test)
        _, A, F_train = spectral_signed_eig(G_train, y_train, k)
        F_test = G_cross.T @ A
    G_train, G_cross = gram_kernel(F_train, F_test)
    return ridge_solve_multi_lam(G_train, G_cross, y_train, y_test, lambdas)


@torch.no_grad()
def nlofi_finite_continue(
    G0_train: torch.Tensor, G0_cross: torch.Tensor,
    y_train: torch.Tensor, y_test: torch.Tensor,
    k0: int, k1: int, P: int, lambdas: tuple[float, ...],
    gen: torch.Generator, dtype: torch.dtype, device: str,
) -> tuple[dict[float, float], dict[float, float]]:
    """Layers 1, 2 + multi-λ readout, given a precomputed layer-0 Gram."""
    _, A0, F1_train = spectral_signed_eig(G0_train, y_train, k0)
    F1_test = G0_cross.T @ A0
    G1_train, G1_cross = gram_finite(F1_train, F1_test, P, gen, dtype, device)
    _, A1, F2_train = spectral_signed_eig(G1_train, y_train, k1)
    F2_test = G1_cross.T @ A1
    G2_train, G2_cross = gram_finite(F2_train, F2_test, P, gen, dtype, device)
    return ridge_solve_multi_lam(G2_train, G2_cross, y_train, y_test, lambdas)


# ---------------------------------------------------------------------------
# Cache filenames
# ---------------------------------------------------------------------------


def _kernel_cache(
    out_dir: Path, n_train: int, k0: int, k1: int, ds: int,
) -> Path:
    return out_dir / f"kernel__K{k0}_{k1}__n{n_train}__ds{ds}.json"


def _rf_cache(
    out_dir: Path, n_train: int, k0: int, k1: int, P: int, seed: int,
) -> Path:
    return out_dir / f"rf__K{k0}_{k1}__n{n_train}__P{P}__seed{seed}.json"


def _cache_complete(path: Path) -> bool:
    """A cache file is *complete* iff it parses and has both ``mse`` and
    ``clf_err`` top-level dicts.  Triggers re-runs for legacy entries
    that predate the dual-metric upgrade."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            row = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "mse" in row and "clf_err" in row


def _mse_to_json(mses: dict[float, float]) -> dict[str, float]:
    return {f"{lam:g}": float(m) for lam, m in mses.items()}


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------


def run_kernel_one_seed(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int, ds: int,
    k_pairs: tuple[tuple[int, int], ...],
    lambdas: tuple[float, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
    log_prefix: str = "",
) -> None:
    """Kernel pipeline for one ``(n_train, ds)`` cell across all K-pairs."""
    needed = [
        (k0, k1) for k0, k1 in k_pairs
        if not _cache_complete(_kernel_cache(out_dir, n_train, k0, k1, ds))
    ]
    if not needed:
        return
    X_tr, y_tr = load_tensors(
        dataset_type=dataset, split="train", n_samples=n_train,
        device=device, seed=ds, class_preset=preset,
    )
    X_tr = X_tr.to(dtype)
    y_tr = y_tr.to(dtype).view(-1)
    for k0, k1 in needed:
        t0 = time.time()
        mses, clf_errs = nlofi_kernel_multi_lam(
            X_tr, y_tr, X_test, y_test, k0, k1, lambdas,
        )
        elapsed = time.time() - t0
        row = {
            "kind": "kernel",
            "dataset": dataset, "preset": preset,
            "n_train": int(n_train), "k0": int(k0), "k1": int(k1),
            "data_seed": int(ds),
            "mse": _mse_to_json(mses),
            "clf_err": _mse_to_json(clf_errs),
            "elapsed_s": elapsed,
        }
        with open(_kernel_cache(out_dir, n_train, k0, k1, ds), "w") as f:
            json.dump(row, f, indent=2)
        log.info(
            "%skernel n=%d K=(%d,%d) ds=%d  mse@λ=%s  (%.1fs)",
            log_prefix, n_train, k0, k1, ds,
            ", ".join(f"{lam:g}:{m:.4f}" for lam, m in mses.items()),
            elapsed,
        )
    del X_tr, y_tr


def run_kernel_block(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int,
    k_pairs: tuple[tuple[int, int], ...],
    lambdas: tuple[float, ...], data_seeds: tuple[int, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
) -> None:
    """Single-GPU sequential dispatch — back-compat wrapper."""
    for ds in data_seeds:
        run_kernel_one_seed(
            X_test, y_test,
            dataset=dataset, preset=preset, n_train=n_train, ds=ds,
            k_pairs=k_pairs, lambdas=lambdas,
            dtype=dtype, device=device, out_dir=out_dir,
        )


def run_finite_one_seed(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int, seed: int,
    k_pairs: tuple[tuple[int, int], ...],
    lambdas: tuple[float, ...], p_values: tuple[int, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
    log_prefix: str = "",
) -> None:
    """One ``(n_train, seed)`` cell: load data with ``seed=seed`` (matching
    the kernel block) and loop ``P × K-pair`` sharing layer-0 Gram.

    Loading the data with ``seed=seed`` ties the finite-P seed to the
    data subsample, so averaging over seeds yields the same expectation
    as the kernel block's average over data seeds — finite-P → kernel
    convergence holds in expectation.
    """
    # Skip if all (P, K-pair) caches for this seed already exist.
    needed_pk: list[tuple[int, list[tuple[int, int, Path]]]] = []
    for P in p_values:
        missing = []
        for k0, k1 in k_pairs:
            cp = _rf_cache(out_dir, n_train, k0, k1, P, seed)
            if not _cache_complete(cp):
                missing.append((k0, k1, cp))
        if missing:
            needed_pk.append((P, missing))
    if not needed_pk:
        return

    X_tr, y_tr = load_tensors(
        dataset_type=dataset, split="train", n_samples=n_train,
        device=device, seed=seed, class_preset=preset,
    )
    X_tr = X_tr.to(dtype)
    y_tr = y_tr.to(dtype).view(-1)

    for P, missing in needed_pk:
        # Layer-0 Gram shared across K-pairs at fixed (n, P, seed).
        t_layer0 = time.time()
        gen = torch.Generator(device=device).manual_seed(seed)
        G0_train, G0_cross = gram_finite(
            X_tr, X_test, P, gen, dtype, device,
        )
        state_after_layer0 = gen.get_state()
        elapsed_layer0 = time.time() - t_layer0
        log.info(
            "%s  layer0 n=%d P=%d seed=%d  (%.1fs, missing %d K-pairs)",
            log_prefix, n_train, P, seed, elapsed_layer0, len(missing),
        )

        for idx, (k0, k1, cp) in enumerate(missing):
            gen.set_state(state_after_layer0)
            t0 = time.time()
            mses, clf_errs = nlofi_finite_continue(
                G0_train, G0_cross, y_tr, y_test,
                k0, k1, P, lambdas, gen, dtype, device,
            )
            elapsed = time.time() - t0
            row = {
                "kind": "finite",
                "dataset": dataset, "preset": preset,
                "n_train": int(n_train),
                "k0": int(k0), "k1": int(k1),
                "P": int(P), "seed": int(seed),
                "mse": _mse_to_json(mses),
                "clf_err": _mse_to_json(clf_errs),
                "elapsed_s": elapsed,
                "elapsed_layer0_s": elapsed_layer0 if idx == 0 else None,
            }
            with open(cp, "w") as f:
                json.dump(row, f, indent=2)
            log.info(
                "%s    K=(%d,%d) n=%d P=%d seed=%d  mse@λ=%s  (%.1fs)",
                log_prefix, k0, k1, n_train, P, seed,
                ", ".join(
                    f"{lam:g}:{m:.4f}" for lam, m in mses.items()
                ),
                elapsed,
            )
        del G0_train, G0_cross
    del X_tr, y_tr


def run_finite_block(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int,
    k_pairs: tuple[tuple[int, int], ...],
    lambdas: tuple[float, ...], p_values: tuple[int, ...],
    seeds: tuple[int, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
) -> None:
    """Single-GPU sequential dispatch over seeds — back-compat wrapper."""
    for seed in seeds:
        run_finite_one_seed(
            X_test, y_test,
            dataset=dataset, preset=preset, n_train=n_train, seed=seed,
            k_pairs=k_pairs, lambdas=lambdas, p_values=p_values,
            dtype=dtype, device=device, out_dir=out_dir,
        )


def _worker(
    rank: int, world_size: int, gpu_ids: list[int], args_dict: dict,
    tasks: list[tuple[int, int]],
) -> None:
    """Per-GPU worker: process its slice of (n_train, seed) tasks."""
    gpu_id = gpu_ids[rank]
    device = f"cuda:{gpu_id}"
    log_prefix = f"[gpu{gpu_id}] "
    dtype = (
        torch.float64 if args_dict["dtype"] == "float64" else torch.float32
    )

    X_test, y_test = load_tensors(
        dataset_type=args_dict["dataset"], split="test",
        n_samples=args_dict["n_test"], device=device, seed=0,
        class_preset=args_dict["preset"],
    )
    X_test = X_test.to(dtype)
    y_test = y_test.to(dtype).view(-1)

    out_dir = Path(args_dict["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    k_pairs = tuple(tuple(p) for p in args_dict["k_pairs"])  # type: ignore[misc]
    lambdas = tuple(args_dict["lambdas"])
    p_values = tuple(args_dict["p_values"])

    my_tasks = [t for i, t in enumerate(tasks) if i % world_size == rank]
    log.info(
        "%sworker rank=%d/%d  %d tasks",
        log_prefix, rank, world_size, len(my_tasks),
    )

    for n_train, seed in my_tasks:
        if not args_dict["skip_kernel"]:
            run_kernel_one_seed(
                X_test, y_test,
                dataset=args_dict["dataset"], preset=args_dict["preset"],
                n_train=n_train, ds=seed,
                k_pairs=k_pairs, lambdas=lambdas,
                dtype=dtype, device=device, out_dir=out_dir,
                log_prefix=log_prefix,
            )
        if not args_dict["skip_finite"]:
            run_finite_one_seed(
                X_test, y_test,
                dataset=args_dict["dataset"], preset=args_dict["preset"],
                n_train=n_train, seed=seed,
                k_pairs=k_pairs, lambdas=lambdas, p_values=p_values,
                dtype=dtype, device=device, out_dir=out_dir,
                log_prefix=log_prefix,
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--preset", type=str, default=PRESET)
    p.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU ids (e.g. '0,1') for multi-GPU dispatch "
             "across (n_train, seed) tasks.  If unset, falls back to "
             "--device for single-GPU.",
    )
    p.add_argument("--device", type=str, default=DEVICE_PREF)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument(
        "--k-pairs", type=str, default=None,
        help="JSON list of [K0, K1] pairs; default uses K_PAIRS.",
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=list(LAMBDAS),
    )
    p.add_argument(
        "--p-values", type=int, nargs="+", default=list(P_VALUES),
    )
    p.add_argument(
        "--n-train-values", type=int, nargs="+",
        default=list(N_TRAIN_VALUES),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument(
        "--data-seeds", type=int, nargs="+", default=list(DATA_SEEDS),
        help="Dataset sub-sample seeds for kernel error bars.",
    )
    p.add_argument(
        "--skip-kernel", action="store_true",
        help="Skip kernel evaluations (only finite RF).",
    )
    p.add_argument(
        "--skip-finite", action="store_true",
        help="Skip finite RF evaluations (only kernel).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.k_pairs is None:
        k_pairs: tuple[tuple[int, int], ...] = K_PAIRS
    else:
        k_pairs = tuple(tuple(p) for p in json.loads(args.k_pairs))  # type: ignore[misc]

    # The fix ties data subsample to seed in the finite block, so the two
    # sets must agree.  Warn if the user passed mismatched values.
    if tuple(args.seeds) != tuple(args.data_seeds):
        log.warning(
            "seeds=%s and data_seeds=%s differ; the finite block will use "
            "`seeds`, the kernel block will use `data_seeds`.  For finite-P → "
            "kernel convergence in expectation these must be the same set.",
            args.seeds, args.data_seeds,
        )
    seeds_finite = tuple(args.seeds)
    seeds_kernel = tuple(args.data_seeds)

    log.info(
        "K_pairs=%s λ=%s n_train=%s P=%s seeds(finite)=%s seeds(kernel)=%s",
        k_pairs, args.lambdas, args.n_train_values,
        args.p_values, seeds_finite, seeds_kernel,
    )

    t_start = time.time()
    if args.gpus is not None and "," in args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
        world_size = len(gpu_ids)

        # The two seed sets are unioned: each (n, seed) task runs whatever
        # is missing among kernel/finite for that seed.
        all_seeds = sorted(set(seeds_finite) | set(seeds_kernel))
        tasks: list[tuple[int, int]] = [
            (int(n), int(s))
            for n in args.n_train_values
            for s in all_seeds
        ]
        log.info(
            "Multi-GPU spawn: %s (%d workers, %d tasks)",
            gpu_ids, world_size, len(tasks),
        )
        args_dict = {
            "dataset": args.dataset, "preset": args.preset,
            "dtype": args.dtype, "n_test": args.n_test,
            "out_dir": str(out_dir),
            "k_pairs": [list(p) for p in k_pairs],
            "lambdas": list(args.lambdas),
            "p_values": list(args.p_values),
            "skip_kernel": args.skip_kernel,
            "skip_finite": args.skip_finite,
        }
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            _worker,
            args=(world_size, gpu_ids, args_dict, tasks),
            nprocs=world_size, join=True,
        )
    else:
        device = resolve_device(args.device)
        log.info("Single-GPU: %s dtype=%s out=%s", device, dtype, out_dir)
        X_test, y_test = load_tensors(
            dataset_type=args.dataset, split="test", n_samples=args.n_test,
            device=device, seed=0, class_preset=args.preset,
        )
        X_test = X_test.to(dtype)
        y_test = y_test.to(dtype).view(-1)
        lambdas = tuple(args.lambdas)
        p_values = tuple(args.p_values)
        for n_train in args.n_train_values:
            log.info("======================================")
            log.info("n_train = %d", n_train)
            log.info("======================================")
            if not args.skip_kernel:
                run_kernel_block(
                    X_test, y_test,
                    dataset=args.dataset, preset=args.preset, n_train=n_train,
                    k_pairs=k_pairs, lambdas=lambdas,
                    data_seeds=seeds_kernel,
                    dtype=dtype, device=device, out_dir=out_dir,
                )
            if not args.skip_finite:
                run_finite_block(
                    X_test, y_test,
                    dataset=args.dataset, preset=args.preset, n_train=n_train,
                    k_pairs=k_pairs, lambdas=lambdas, p_values=p_values,
                    seeds=seeds_finite, dtype=dtype, device=device,
                    out_dir=out_dir,
                )
            log.info("elapsed so far: %.1f min", (time.time() - t_start) / 60.0)

    log.info("Total elapsed: %.1f min", (time.time() - t_start) / 60.0)


if __name__ == "__main__":
    main()
