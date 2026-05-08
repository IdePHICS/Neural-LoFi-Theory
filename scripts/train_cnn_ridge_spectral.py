from __future__ import annotations

import json
import logging
from typing import Any, cast

from config_utils import with_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_hierarchically.datasets import DatasetConfig, build_dataset
from train_hierarchically.models.cnn_ridge_spectral import (
    CNNLayerSpec,
    CNNRidgeSpectralModel,
)
from train_hierarchically.training.cnn_ridge_spectral import (
    CNNRidgeSpectralConfig,
    CNNRidgeSpectralTrainer,
)
from train_hierarchically.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)

# Spatial sizes for supported datasets (after default transforms)
_SPATIAL_SIZES: dict[str, tuple[int, int]] = {
    "mnist": (28, 28),
    "fashion-mnist": (28, 28),
    "cifar10": (32, 32),
    "pcam": (96, 96),
    "celeba": (64, 64),
}

_IN_CHANNELS: dict[str, int] = {
    "mnist": 1,
    "fashion-mnist": 1,
    "cifar10": 3,
    "pcam": 3,
    "celeba": 3,
}


def _build_results_filename(cfg, layer_dicts: list[dict[str, Any]]) -> str:
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
            kind = ld.get("kind", "conv")
            if kind == "conv":
                do_eig = ld.get("do_eigenreduction", True)
                p = ld.get("p", "na")
                k = ld.get("k", "na")
                k_str = f"k{k}" if do_eig else "noeig"
                layer_parts.append(
                    "_".join(
                        [
                            "conv",
                            f"p{p}{k_str}",
                            f"ker{ld.get('kernel_size', 5)}",
                            f"act{ld.get('activation', cfg.get('activation', 'sigmoid'))}",
                        ]
                    )
                )
            elif kind == "fully_connected":
                do_eig = ld.get("do_eigenreduction", True)
                p = ld.get("p", "na")
                k = ld.get("k", "na")
                k_str = f"k{k}" if do_eig else "noeig"
                layer_parts.append(
                    "_".join(
                        [
                            "fc",
                            f"p{p}{k_str}",
                            f"act{ld.get('activation', cfg.get('activation', 'sigmoid'))}",
                        ]
                    )
                )
            elif kind == "scattering":
                do_eig = ld.get("do_eigenreduction", True)
                k = ld.get("k", "na")
                k_str = f"k{k}" if do_eig else "noeig"
                layer_parts.append(
                    "_".join(
                        [
                            "scat",
                            f"J{ld.get('scattering_j', 2)}",
                            f"L{ld.get('scattering_l', 8)}",
                            f"ord{ld.get('scattering_max_order', 2)}",
                            k_str,
                        ]
                    )
                )
            elif kind == "pooling":
                pool_kernel = ld.get("pool_kernel_size", ld.get("kernel_size", 2))
                pool_stride = ld.get("pool_stride", pool_kernel)
                pool_mode = ld.get("pool_mode", "max")
                layer_parts.append(
                    f"pool_{pool_mode}_k{pool_kernel}_s{pool_stride}"
                )
            else:
                layer_parts.append("flatten")
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


@with_config("conf/train_cnn_ridge_spectral_deep.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    layer_dicts: list[dict[str, Any]] = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]
    results_filename = _build_results_filename(cfg, layer_dicts)
    results_path = out_dir / results_filename

    if results_path.exists() and not force_run:
        log.info("Results already exist at %s — skipping.", results_path)
        return

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    log.info("Using device: %s", device)

    raw = cfg.dataset.get("classes")
    classes: list[int] | None = (
        cast(list[int], OmegaConf.to_object(raw)) if raw else None
    )

    ds_name: str = cfg.dataset.name

    # --- Load datasets (NOT flattened — need (C, H, W)) ---
    train_config = DatasetConfig(
        dataset_type=ds_name,
        split="train",
        root=cfg.root,
        n_samples=cfg.dataset.n_train,
        seed=cfg.seed,
        flatten=False,
        class_preset=cfg.dataset.get("preset"),
        classes=classes,
        remap_labels=cfg.remap_labels,
    )
    train_dataset = build_dataset(train_config)

    test_config = DatasetConfig(
        dataset_type=ds_name,
        split="test",
        root=cfg.root,
        n_samples=cfg.dataset.n_test,
        seed=cfg.seed,
        flatten=False,
        class_preset=cfg.dataset.get("preset"),
        classes=classes,
        remap_labels=cfg.remap_labels,
    )
    test_dataset = build_dataset(test_config)

    # --- Infer spatial / channel info ---
    in_channels = _IN_CHANNELS[ds_name]
    spatial_size = _SPATIAL_SIZES[ds_name]

    # --- Build layer specs ---
    layer_specs: list[CNNLayerSpec] = []
    for ld in layer_dicts:
        kind = ld.get("kind", "conv")
        layer_specs.append(
            CNNLayerSpec(
                kind=kind,
                p=ld.get("p", 0),
                k=ld.get("k", 0),
                kernel_size=ld.get("kernel_size", cfg.get("kernel_size", 5)),
                padding=ld.get("padding", cfg.get("padding", 2)),
                stride=ld.get("stride", cfg.get("stride", 1)),
                activation=ld.get("activation", cfg.get("activation", "sigmoid")),
                do_eigenreduction=ld.get("do_eigenreduction", True),
                pool_mode=ld.get("pool_mode", "max"),
                pool_kernel_size=ld.get(
                    "pool_kernel_size", ld.get("kernel_size", 2)
                ),
                pool_stride=ld.get("pool_stride"),
                pool_padding=ld.get("pool_padding", 0),
                scattering_j=ld.get("scattering_j", 2),
                scattering_l=ld.get("scattering_l", 8),
                scattering_max_order=ld.get("scattering_max_order", 2),
            )
        )

    specs_summary = [
        (
            s.kind,
            s.p,
            s.k,
            s.kernel_size,
            s.activation,
            s.do_eigenreduction,
        )
        for s in layer_specs
    ]
    log.info(
        "Model: in_channels=%d, spatial=%s, layers=%s",
        in_channels,
        spatial_size,
        specs_summary,
    )

    # --- Build model & trainer ---
    model = CNNRidgeSpectralModel(
        in_channels=in_channels,
        spatial_size=spatial_size,
        layer_specs=layer_specs,
        seed=cfg.seed,
    )

    log.info("covariance_chunk_size=%d", cfg.get("covariance_chunk_size", 0))
    config = CNNRidgeSpectralConfig(
        device=device,
        verbose=True,
        ridge_alpha_min=cfg.ridge_alpha_min,
        ridge_alpha_max=cfg.ridge_alpha_max,
        ridge_alpha_num=cfg.ridge_alpha_num,
        covariance_chunk_size=cfg.get("covariance_chunk_size", 0),
        store_full_eigenvalues=cfg.get("store_full_eigenvalues", False),
    )
    trainer = CNNRidgeSpectralTrainer(model=model, config=config)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # --- Train ---
    _, results = trainer.fit(train_loader, test_loader=test_loader)

    # --- Save results ---
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)

    log.info("--- Layer-wise Diagnostics ---")
    for entry in results["layers"]:
        idx = entry["layer_idx"]
        name = entry["layer_name"]
        eigvals = entry.get("eigenvalues", [])
        n_eig = len(eigvals) if hasattr(eigvals, "__len__") else 0
        log.info("  layer %2d (%s): %d eigenvalues", idx, name, n_eig)

    log.info("--- Final Metrics ---\n%s", results["final"])


if __name__ == "__main__":
    main()
