# ruff: noqa: N803, N806, N812
"""Covariance spectrum eigenvectors for the ridge spectral model.

Uses RidgeSpectralTrainer to fit the model, then saves per-layer spectral
data (full eigenvalue spectra, top eigenvectors, gradient-ascent maximizers).

Naming convention:
    - "input"   — raw input data X
    - "layer_1" — first random-feature expansion σ(X @ W_1)
    - "layer_2" — second expansion σ(h_1 @ W_2)
    - …
"""
from __future__ import annotations

import json
import logging
import warnings
from typing import cast

import numpy as np
import torch
from config_utils import parse_classes, with_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_hierarchically.datasets import load_tensors
from train_hierarchically.models.ridge_spectral import LayerSpec, RidgeSpectralModel
from train_hierarchically.training.ridge_spectral import (
    RidgeSpectralConfig,
    RidgeSpectralTrainer,
)
from train_hierarchically.utils.helpers import set_seed
from train_hierarchically.visualization import FFNVisualizer

warnings.filterwarnings("ignore", message="dtype.*align", category=DeprecationWarning)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(data: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_run_dirname(cfg, layer_specs: list[LayerSpec]) -> str:
    """Build a deterministic directory name encoding experiment parameters."""
    preset = cfg.dataset.get("preset", "all")
    n_layers = len(layer_specs)
    P = layer_specs[0].p
    activation = layer_specs[0].activation

    # Encode K schedule: single value if uniform, dash-separated if heterogeneous
    k_values = [spec.k for spec in layer_specs]
    if len(set(k_values)) == 1:
        keep_str = str(k_values[0])
    else:
        keep_str = "-".join(str(k) for k in k_values)

    parts = [
        f"{preset}",
        f"ntrain{cfg.dataset.n_train}",
        f"P{P}",
        f"layers{n_layers}",
        f"keep{keep_str}",
        f"{activation}",
        f"seed{cfg.seed}",
    ]
    return "_".join(parts)


@with_config("conf/covariance_spectrum_eigenvectors.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    n_samples: int = cfg.dataset.n_train
    n_top: int = cfg.n_top
    ga_n_top: int = cfg.get("ga_n_top", n_top)
    ga_steps: int = cfg.ga_steps
    ga_lr: float = cfg.ga_lr
    ga_fourier_decay: float = cfg.get("ga_fourier_decay", 0.3)
    ga_fourier_alpha: float = cfg.get("ga_fourier_alpha", 2.0)
    ga_n_restarts: int = cfg.get("ga_n_restarts", 5)
    ga_normalize_hidden: bool = cfg.get("ga_normalize_hidden", False)

    # Load data
    log.info("Loading %s (split=%s, N=%d)", cfg.dataset.name, cfg.split, n_samples)
    X, y = load_tensors(
        dataset_type=cfg.dataset.name,
        split=cfg.split,
        n_samples=n_samples,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=cfg.dataset.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
    )
    D = X.shape[1]

    # Build layer specs from per-layer config list
    layer_dicts = cast(list[dict], OmegaConf.to_object(cfg.layers))
    layer_specs: list[LayerSpec] = []
    prev_dim = D
    for ld in layer_dicts:
        p = ld["p"] if ld["p"] is not None else prev_dim
        k = ld["k"]
        layer_specs.append(LayerSpec(
            p=p,
            k=k,
            activation=ld["activation"],
            do_eigenreduction=ld.get("do_eigenreduction", True),
        ))
        prev_dim = k
    n_layers = len(layer_specs)

    # Build deterministic run directory
    run_name = _build_run_dirname(cfg, layer_specs)
    out_dir = out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / f"eigenvalues_layer_{n_layers}.json"
    if marker.exists() and not force_run:
        log.info("Results already exist at %s — skipping.", out_dir)
        return

    # ------------------------------------------------------------------
    # Build model and train via RidgeSpectralTrainer
    # ------------------------------------------------------------------
    model = RidgeSpectralModel(
        input_dim=D, layer_specs=layer_specs, seed=cfg.seed,
    ).to(device)

    trainer_config = RidgeSpectralConfig(
        device=cfg.device,
        verbose=True,
        store_spectra=True,
        n_top_eigenvectors=n_top,
    )
    trainer = RidgeSpectralTrainer(model, trainer_config)

    train_loader = DataLoader(
        list(zip(X, y)), batch_size=512, shuffle=False,
    )
    model, results = trainer.fit(train_loader)

    # Save model state dict for later analysis (class projections, etc.)
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")

    # Create visualizer
    viz = FFNVisualizer(model, device)

    # ------------------------------------------------------------------
    # Save input spectra (computed directly, not part of training)
    # ------------------------------------------------------------------
    input_spectra = RidgeSpectralTrainer._full_spectra(X, y, n_top)
    _save_json(
        {
            "cov": input_spectra["cov_spectrum"],
            "signed_cov": input_spectra["signed_cov_spectrum"],
        },
        out_dir / "eigenvalues_input.json",
    )
    for name in ["cov", "signed_cov"]:
        _save_json(
            {
                "eigenvalues": input_spectra[f"{name}_top_eigenvalues"],
                "eigenvectors": input_spectra[f"{name}_top_eigenvectors"],
            },
            out_dir / f"top_eigenvectors_input_{name}.json",
        )

    # ------------------------------------------------------------------
    # Save per-layer spectra, maximizers, and Jacobian importance
    # ------------------------------------------------------------------
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx + 1}"
        layer_data = results["layers"][layer_idx]
        spectra = layer_data["spectra"]

        spec = layer_specs[layer_idx]
        log.info(
            "Layer %d (saved as %s): P=%d, K=%d",
            layer_idx, layer_name, spec.p, spec.k,
        )

        # Full eigenvalue spectra
        _save_json(
            {
                "cov": spectra["cov_spectrum"],
                "signed_cov": spectra["signed_cov_spectrum"],
            },
            out_dir / f"eigenvalues_{layer_name}.json",
        )

        # Top eigenvectors
        for name in ["cov", "signed_cov"]:
            _save_json(
                {
                    "eigenvalues": spectra[f"{name}_top_eigenvalues"],
                    "eigenvectors": spectra[f"{name}_top_eigenvectors"],
                },
                out_dir / f"top_eigenvectors_{layer_name}_{name}.json",
            )

        # Fourier gradient-ascent maximizers
        log.info(
            "  Gradient ascent (Fourier): steps=%d, lr=%g, f0=%g, alpha=%g",
            ga_steps, ga_lr, ga_fourier_decay, ga_fourier_alpha,
        )
        for name in ["cov", "signed_cov"]:
            top_vecs = np.array(spectra[f"{name}_top_eigenvectors"])
            maxs: dict = {}
            for k_idx in range(ga_n_top):
                v = torch.tensor(
                    top_vecs[:, k_idx], dtype=torch.float32, device=device
                )
                maxs[f"k{k_idx}"] = viz.fourier_maximizer(
                    v, layer_idx, D,
                    n_steps=ga_steps,
                    lr=ga_lr,
                    fourier_decay=ga_fourier_decay,
                    fourier_alpha=ga_fourier_alpha,
                    n_restarts=ga_n_restarts,
                    normalize_hidden=ga_normalize_hidden,
                ).tolist()
            _save_json(maxs, out_dir / f"maximizers_{layer_name}_{name}.json")

        # Jacobian-based pixel importance (per-eigenvector + mean)
        log.info("  Jacobian importance: n_top=%d", n_top)
        for name in ["cov", "signed_cov"]:
            top_vecs = np.array(spectra[f"{name}_top_eigenvectors"])
            imp = viz.jacobian_importance(
                X, layer_idx, top_vecs, n_top,
                fourier_decay=ga_fourier_decay,
                fourier_alpha=ga_fourier_alpha,
            )
            _save_json(
                imp, out_dir / f"jacobian_importance_{layer_name}_{name}.json",
            )

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
