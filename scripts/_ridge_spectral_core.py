# ruff: noqa: N803, N806
"""Shared core for ridge spectral training.

Provides ``run_ridge_spectral`` — a single function that loads data, builds
a model, trains it, and returns the results dict.  Used by both
``train_ridge_spectral_dnn.py`` (single run) and ``optimize_k1k2.py``
(Nelder-Mead over k values).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config_utils import parse_classes
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from train_hierarchically.datasets import load_tensors
from train_hierarchically.models.ridge_spectral import LayerSpec, RidgeSpectralModel
from train_hierarchically.training.ridge_spectral import (
    RidgeSpectralConfig,
    RidgeSpectralTrainer,
)
from train_hierarchically.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


def build_results_filename(cfg: DictConfig, layer_dicts: list[dict[str, Any]]) -> str:
    """Build a descriptive filename encoding all run parameters."""
    preset = cfg.dataset.get("preset")
    raw_classes = cfg.dataset.get("classes")
    if preset:
        task = preset
    elif raw_classes:
        task = "classes_" + "_".join(str(c) for c in raw_classes)
    else:
        task = "all"

    if layer_dicts:
        layer_parts = []
        for ld in layer_dicts:
            p_str = "pD" if ld["p"] is None else f"p{ld['p']}"
            do_eig = ld.get("do_eigenreduction", True)
            if do_eig:
                layer_parts.append(f"{p_str}k{ld['k']}{ld['activation']}")
            else:
                layer_parts.append(f"{p_str}{ld['activation']}_noeig")
        layers_str = "-".join(layer_parts)
    else:
        layers_str = "nolayers"

    ridge_str = (
        f"ridge{cfg.ridge_alpha_min}to{cfg.ridge_alpha_max}x{cfg.ridge_alpha_num}"
    )

    return (
        f"{task}_seed{cfg.seed}"
        f"_ntrain{cfg.dataset.n_train}_ntest{cfg.dataset.n_test}"
        f"_{layers_str}_{ridge_str}.json"
    )


def run_ridge_spectral(
    cfg: DictConfig,
    *,
    seed: int | None = None,
    layer_dicts: list[dict[str, Any]] | None = None,
    out_path: Path | None = None,
    data: tuple | None = None,
) -> dict[str, Any]:
    """Run one ridge spectral training and return the results dict.

    Parameters
    ----------
    cfg
        Full experiment configuration (OmegaConf DictConfig).
    seed
        Override for ``cfg.seed``.  When ``None``, uses ``cfg.seed``.
    layer_dicts
        Override for ``cfg.layers`` (already converted via
        ``OmegaConf.to_object``).  When ``None``, reads from ``cfg``.
    out_path
        If given and the file exists, loads and returns the cached result
        (file-level cache).  If given and does not exist, saves the result
        after training.
    data
        Pre-loaded ``(X_train, y_train, X_test, y_test)`` tensors.
        When provided, skips ``load_tensors`` calls.

    Returns
    -------
    dict[str, Any]
        The results dict produced by ``RidgeSpectralTrainer.fit``.
    """
    # --- File-level cache: load if exists ---
    if out_path is not None and out_path.exists():
        log.info("Cache hit: %s", out_path)
        try:
            with open(out_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            log.warning("Corrupt cache file, re-running: %s", out_path)
            out_path.unlink()

    # --- Resolve overrides ---
    if seed is not None:
        cfg = cfg.copy()
        cfg.seed = seed

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    if layer_dicts is None:
        layer_dicts = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]

    # --- Load datasets as tensors ---
    if data is not None:
        X_train, y_train, X_test, y_test = data
    else:
        classes = parse_classes(cfg)
        X_train, y_train = load_tensors(
            dataset_type=cfg.dataset.name,
            split="train",
            n_samples=cfg.dataset.n_train,
            device=device,
            seed=cfg.seed,
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
            seed=cfg.seed,
            root=cfg.root,
            class_preset=cfg.dataset.get("preset"),
            classes=classes,
            remap_labels=cfg.remap_labels,
        )
    input_dim = X_train.shape[1]

    # --- Build layer specs ---
    layer_specs: list[LayerSpec] = []
    prev_dim = input_dim
    for ld in layer_dicts:
        p = ld["p"] if ld["p"] is not None else prev_dim
        layer_specs.append(
            LayerSpec(
                p=p,
                k=ld["k"],
                activation=ld["activation"],
                do_eigenreduction=ld.get("do_eigenreduction", True),
            )
        )
        prev_dim = ld["k"]

    log.info(
        "Model: input_dim=%d, layers=%s",
        input_dim,
        [(s.p, s.k, s.activation) for s in layer_specs],
    )

    # --- Build model, trainer ---
    model = RidgeSpectralModel(
        input_dim=input_dim,
        layer_specs=layer_specs,
        seed=cfg.seed,
    )
    config = RidgeSpectralConfig(
        device=device,
        verbose=True,
        ridge_alpha_min=cfg.ridge_alpha_min,
        ridge_alpha_max=cfg.ridge_alpha_max,
        ridge_alpha_num=cfg.ridge_alpha_num,
    )
    trainer = RidgeSpectralTrainer(model=model, config=config)

    # --- DataLoaders ---
    train_loader = DataLoader(
        list(zip(X_train, y_train)), batch_size=512, shuffle=False
    )
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=512, shuffle=False)

    # --- Train ---
    _, results = trainer.fit(train_loader, test_loader=test_loader)

    # --- Save if path given ---
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", out_path)

    return results
