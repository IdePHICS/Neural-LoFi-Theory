# ruff: noqa: N803, N806
"""Data generator: NLoFi-finite-RF (with K-grid optimization) vs deep NNGP.

Two blocks of evaluations, both at fixed ``λ`` (default ``10^-3``):

1. **Deep NNGP** (no eigen-reduce; 3-layer relu kernel composition with
   final ridge readout).  One value per ``(n_train, data_seed)``.

2. **NLoFi finite RF** (3 layers, eigen-reduce at layers 0 and 1, ridge
   at layer 2).  Sweeps over a *diagonal* K-grid ``K_0 = K_1 ∈
   {50, 100, 200, 500}``, the P grid, and RF seeds.  Layer-0 Gram is
   shared across the K-grid for each ``(n_train, P, seed)`` triple
   (verified bit-exact in the appendix-script smoke test).

The plot only uses ``min_K test_mse`` per ``(n_train, P)`` — the K-grid
is the inner optimization — but every individual evaluation is cached,
so we never recompute on re-runs.

Output layout: ``results/nlofi_vs_deep_nngp/`` containing

    deep_nngp__n{n}__ds{ds}__lam{lam}.json       (kernel reference)
    nlofi__n{n}__K{k}__P{P}__seed{s}__lam{lam}.json (per-K finite RF)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch

from train_hierarchically.datasets import load_tensors
from train_hierarchically.utils.helpers import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (CLI overridable)
# ---------------------------------------------------------------------------

DATASET = "cifar10"
PRESET = "animal_vehicle"
N_LAYERS = 3
LAMBDA = 1e-3
K_GRID: tuple[int, ...] = (50, 100, 200, 500)
P_VALUES: tuple[int, ...] = (300, 1000, 3000, 10000, 30000, 100000)
N_TRAIN_VALUES: tuple[int, ...] = (500, 1000, 2000, 5000)
N_TEST = 2000
SEEDS: tuple[int, ...] = tuple(range(10))
DATA_SEEDS: tuple[int, ...] = tuple(range(10))
DEVICE_PREF = "cuda:1"
OUT_DIR = Path("results/nlofi_vs_deep_nngp")
EIGEN_EPS = 1e-6
CHUNK_P = 20000


# ---------------------------------------------------------------------------
# relu NNGP kernel (Cho-Saul J_1)
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
# Signed-cov eigen-reduction (kernel-trick)
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
        W = torch.randn(d, cp, generator=gen, device=device, dtype=dtype)
        z_tr = torch.relu(F_train @ W / c) * inv_sqrt_p
        z_te = torch.relu(F_test @ W / c) * inv_sqrt_p
        G_train += z_tr @ z_tr.T
        G_cross += z_tr @ z_te.T
        del W, z_tr, z_te
    return G_train, G_cross


# ---------------------------------------------------------------------------
# Ridge readout
# ---------------------------------------------------------------------------


@torch.no_grad()
def ridge_test_mse(
    G_train: torch.Tensor, G_cross: torch.Tensor,
    y_train: torch.Tensor, y_test: torch.Tensor, lam: float,
) -> float:
    G_sym = 0.5 * (G_train + G_train.T)
    mu, Q = torch.linalg.eigh(G_sym.to(torch.float64))
    qy = Q.T @ y_train.to(torch.float64)
    G_proj = G_cross.T.to(torch.float64) @ Q
    alpha = qy / (mu + lam)
    y_pred = G_proj @ alpha
    return float((y_pred - y_test.to(torch.float64)).pow(2).mean().item())


# ---------------------------------------------------------------------------
# Deep NNGP (no eigen-reduce)
# ---------------------------------------------------------------------------


@torch.no_grad()
def deep_nngp_mse(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    n_layers: int, lam: float,
) -> float:
    """Closed-form ``n_layers``-deep relu NNGP with ridge readout.

    Repeated kernel composition: build the joint ``(n + n_test) × (n +
    n_test)`` covariance, divide by per-layer scale ``c`` (so the diagonal
    has unit average on the training block), apply ``relu_arc_cosine``,
    and feed the result back as the next layer's covariance.
    """
    n = X_train.shape[0]
    F_full = torch.cat([X_train, X_test], dim=0)
    c2 = X_train.pow(2).sum(dim=1).mean()
    K_full = relu_arc_cosine((F_full @ F_full.T) / c2)
    for _ in range(n_layers - 1):
        c2 = K_full[:n, :n].diagonal().mean()
        K_full = relu_arc_cosine(K_full / c2)
    return ridge_test_mse(
        K_full[:n, :n], K_full[:n, n:], y_train, y_test, lam,
    )


# ---------------------------------------------------------------------------
# NLoFi continue (layer-1 + layer-2 + ridge given a layer-0 Gram)
# ---------------------------------------------------------------------------


@torch.no_grad()
def nlofi_continue_mse(
    G0_train: torch.Tensor, G0_cross: torch.Tensor,
    y_train: torch.Tensor, y_test: torch.Tensor,
    k0: int, k1: int, P: int, lam: float,
    gen: torch.Generator, dtype: torch.dtype, device: str,
) -> float:
    _, A0, F1_train = spectral_signed_eig(G0_train, y_train, k0)
    F1_test = G0_cross.T @ A0
    G1_train, G1_cross = gram_finite(F1_train, F1_test, P, gen, dtype, device)
    _, A1, F2_train = spectral_signed_eig(G1_train, y_train, k1)
    F2_test = G1_cross.T @ A1
    G2_train, G2_cross = gram_finite(F2_train, F2_test, P, gen, dtype, device)
    return ridge_test_mse(G2_train, G2_cross, y_train, y_test, lam)


# ---------------------------------------------------------------------------
# Cache filenames
# ---------------------------------------------------------------------------


def _deep_cache(out_dir: Path, n: int, ds: int, lam: float) -> Path:
    return out_dir / f"deep_nngp__n{n}__ds{ds}__lam{lam:g}.json"


def _nlofi_cache(
    out_dir: Path, n: int, k: int, P: int, seed: int, lam: float,
) -> Path:
    return out_dir / f"nlofi__n{n}__K{k}__P{P}__seed{seed}__lam{lam:g}.json"


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------


def run_deep_nngp_block(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int, n_layers: int,
    lam: float, data_seeds: tuple[int, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
) -> None:
    for ds in data_seeds:
        cp = _deep_cache(out_dir, n_train, ds, lam)
        if cp.exists():
            continue
        X_tr, y_tr = load_tensors(
            dataset_type=dataset, split="train", n_samples=n_train,
            device=device, seed=ds, class_preset=preset,
        )
        X_tr = X_tr.to(dtype)
        y_tr = y_tr.to(dtype).view(-1)
        t0 = time.time()
        mse = deep_nngp_mse(X_tr, y_tr, X_test, y_test, n_layers, lam)
        elapsed = time.time() - t0
        row = {
            "kind": "deep_nngp",
            "dataset": dataset, "preset": preset,
            "n_train": int(n_train), "n_layers": int(n_layers),
            "data_seed": int(ds), "lambda": float(lam),
            "test_mse": float(mse),
            "elapsed_s": elapsed,
        }
        with open(cp, "w") as f:
            json.dump(row, f, indent=2)
        log.info(
            "deep-NNGP n=%d L=%d λ=%g ds=%d  mse=%.4f  (%.1fs)",
            n_train, n_layers, lam, ds, mse, elapsed,
        )
        del X_tr, y_tr


def run_nlofi_block(
    X_test: torch.Tensor, y_test: torch.Tensor,
    *, dataset: str, preset: str, n_train: int,
    k_grid: tuple[int, ...], lam: float,
    p_values: tuple[int, ...], seeds: tuple[int, ...],
    dtype: torch.dtype, device: str, out_dir: Path,
) -> None:
    """Per (n, P, seed): build layer-0 Gram once, run the K-grid sharing it."""
    X_tr, y_tr = load_tensors(
        dataset_type=dataset, split="train", n_samples=n_train,
        device=device, seed=0, class_preset=preset,
    )
    X_tr = X_tr.to(dtype)
    y_tr = y_tr.to(dtype).view(-1)

    for P in p_values:
        for seed in seeds:
            needed = [
                (k, _nlofi_cache(out_dir, n_train, k, P, seed, lam))
                for k in k_grid
                if not _nlofi_cache(out_dir, n_train, k, P, seed, lam).exists()
            ]
            if not needed:
                continue

            t_layer0 = time.time()
            gen = torch.Generator(device=device).manual_seed(seed)
            G0_train, G0_cross = gram_finite(
                X_tr, X_test, P, gen, dtype, device,
            )
            state_after_layer0 = gen.get_state()
            elapsed_layer0 = time.time() - t_layer0
            log.info(
                "  layer0 n=%d P=%d seed=%d  (%.1fs, %d K missing)",
                n_train, P, seed, elapsed_layer0, len(needed),
            )

            for idx, (k, cp) in enumerate(needed):
                gen.set_state(state_after_layer0)
                t0 = time.time()
                mse = nlofi_continue_mse(
                    G0_train, G0_cross, y_tr, y_test,
                    k, k, P, lam, gen, dtype, device,
                )
                elapsed = time.time() - t0
                row = {
                    "kind": "nlofi",
                    "dataset": dataset, "preset": preset,
                    "n_train": int(n_train),
                    "k0": int(k), "k1": int(k),
                    "P": int(P), "seed": int(seed),
                    "lambda": float(lam),
                    "test_mse": float(mse),
                    "elapsed_s": elapsed,
                    "elapsed_layer0_s": elapsed_layer0 if idx == 0 else None,
                }
                with open(cp, "w") as f:
                    json.dump(row, f, indent=2)
                log.info(
                    "    K=%d n=%d P=%d seed=%d  mse=%.4f  (%.1fs)",
                    k, n_train, P, seed, mse, elapsed,
                )
            del G0_train, G0_cross
    del X_tr, y_tr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--preset", type=str, default=PRESET)
    p.add_argument("--device", type=str, default=DEVICE_PREF)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--k-grid", type=int, nargs="+", default=list(K_GRID))
    p.add_argument("--p-values", type=int, nargs="+", default=list(P_VALUES))
    p.add_argument("--n-train-values", type=int, nargs="+",
                   default=list(N_TRAIN_VALUES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--data-seeds", type=int, nargs="+",
                   default=list(DATA_SEEDS))
    p.add_argument("--skip-deep-nngp", action="store_true")
    p.add_argument("--skip-nlofi", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    log.info("Device=%s dtype=%s out=%s", device, dtype, out_dir)
    log.info(
        "L=%d λ=%g K-grid=%s n_train=%s P=%s seeds=%s data_seeds=%s",
        args.n_layers, args.lam, args.k_grid, args.n_train_values,
        args.p_values, args.seeds, args.data_seeds,
    )

    X_test, y_test = load_tensors(
        dataset_type=args.dataset, split="test", n_samples=args.n_test,
        device=device, seed=0, class_preset=args.preset,
    )
    X_test = X_test.to(dtype)
    y_test = y_test.to(dtype).view(-1)

    k_grid = tuple(args.k_grid)
    p_values = tuple(args.p_values)
    seeds = tuple(args.seeds)
    data_seeds = tuple(args.data_seeds)

    t_start = time.time()
    for n_train in args.n_train_values:
        log.info("======================================")
        log.info("n_train = %d", n_train)
        log.info("======================================")
        if not args.skip_deep_nngp:
            run_deep_nngp_block(
                X_test, y_test,
                dataset=args.dataset, preset=args.preset, n_train=n_train,
                n_layers=args.n_layers, lam=args.lam, data_seeds=data_seeds,
                dtype=dtype, device=device, out_dir=out_dir,
            )
        if not args.skip_nlofi:
            run_nlofi_block(
                X_test, y_test,
                dataset=args.dataset, preset=args.preset, n_train=n_train,
                k_grid=k_grid, lam=args.lam, p_values=p_values, seeds=seeds,
                dtype=dtype, device=device, out_dir=out_dir,
            )
        log.info("elapsed so far: %.1f min", (time.time() - t_start) / 60.0)
    log.info("Total elapsed: %.1f min", (time.time() - t_start) / 60.0)


if __name__ == "__main__":
    main()
