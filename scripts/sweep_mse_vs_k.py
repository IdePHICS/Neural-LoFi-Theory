# ruff: noqa: N803, N806
"""Sweep test MSE/error over K values for a fixed P.

For a given (P, n_train) pair (set via config or --override), generates
log-spaced K values from 2 to P and runs ridge spectral training for each
(K, seed) combination.  Supports both 1-layer and 2-layer architectures
(determined by the config file's ``layers`` list).

Data is loaded once per seed and reused across K values for efficiency.
Results are saved as individual JSON files with skip-if-exists logic.

Intended to be called via ``parallel_run.py`` with a sweep over ``all_p``
and ``dataset.n_train``.

Usage
-----
# Direct (single P, single n_train):
python scripts/sweep_mse_vs_k.py --conf conf/mse_vs_k_1layer.yaml \
    --override all_p=1000 dataset.n_train=10000

# Via parallel_run.py (distributes across P × n_train):
python scripts/parallel_run.py \
    --script scripts/sweep_mse_vs_k.py \
    --conf conf/mse_vs_k_1layer.yaml \
    --sweep conf/sweeps/p_ntrain.yaml \
    --n_workers 8
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from _ridge_spectral_core import build_results_filename, run_ridge_spectral
from config_utils import parse_classes, with_config
from omegaconf import OmegaConf

from train_hierarchically.datasets import load_tensors
from train_hierarchically.utils.helpers import resolve_device

log = logging.getLogger(__name__)


def _generate_k_values(p: int, n_points: int) -> list[int]:
    """Generate *n_points* log-spaced K values from 2 to *p* - 1 (inclusive)."""
    upper = max(2, p - 1)
    raw = np.geomspace(2, upper, n_points)
    return sorted(set(int(k) for k in np.unique(np.round(raw).astype(int))))


def _build_layer_dicts(
    base_layers: list[dict[str, Any]], k: int,
) -> list[dict[str, Any]]:
    """Build layer dicts with the first layer's k set to *k*."""
    layers = []
    for i, ld in enumerate(base_layers):
        ld_copy = dict(ld)
        if i == 0:
            ld_copy["k"] = k
        layers.append(ld_copy)
    return layers


@with_config("conf/mse_vs_k_1layer.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    P = cfg.all_p
    n_k = cfg.n_k_points
    seeds = list(range(cfg.seed_start, cfg.seed_end + 1))
    base_layers: list[dict[str, Any]] = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]

    k_values = _generate_k_values(P, n_k)
    n_layers = len(base_layers)

    log.info(
        "Sweep: P=%d, n_train=%d, n_test=%d, %d layers, %d K values, %d seeds",
        P, cfg.dataset.n_train, cfg.dataset.n_test, n_layers, len(k_values), len(seeds),
    )
    log.info("K values: %s", k_values)

    device = resolve_device(cfg.device)
    classes = parse_classes(cfg)

    total = len(seeds) * len(k_values)
    done = 0
    skipped = 0

    for seed in seeds:
        # Load data once per seed (both load_tensors and the model use
        # dedicated generators seeded with `seed`, so data is reproducible
        # regardless of global RNG state).
        X_train, y_train = load_tensors(
            dataset_type=cfg.dataset.name,
            split="train",
            n_samples=cfg.dataset.n_train,
            device=device,
            seed=seed,
            root=cfg.root,
            class_preset=cfg.dataset.get("preset"),
            classes=classes,
            remap_labels=cfg.remap_labels,
        )
        X_test, y_test = load_tensors(
            dataset_type=cfg.dataset.name,
            split="test",
            n_samples=cfg.dataset.n_test,
            device=device,
            seed=seed,
            root=cfg.root,
            class_preset=cfg.dataset.get("preset"),
            classes=classes,
            remap_labels=cfg.remap_labels,
        )
        data = (X_train, y_train, X_test, y_test)

        for k in k_values:
            layer_dicts = _build_layer_dicts(base_layers, k)

            # Build filename using cfg with this seed
            cfg_copy = cfg.copy()
            cfg_copy.seed = seed
            filename = build_results_filename(cfg_copy, layer_dicts)
            filepath = out_dir / filename

            if filepath.exists() and not force_run:
                skipped += 1
                continue

            log.info(
                "Running: seed=%d, K=%d (%d/%d)",
                seed, k, done + skipped + 1, total,
            )
            run_ridge_spectral(
                cfg, seed=seed, layer_dicts=layer_dicts,
                out_path=filepath, data=data,
            )
            done += 1

    log.info("Sweep complete: %d runs, %d skipped (of %d total)", done, skipped, total)


if __name__ == "__main__":
    main()
