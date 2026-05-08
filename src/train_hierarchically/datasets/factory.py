from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig
from .wrappers import filter_classes, limit_dataset

Builder = Callable[[DatasetConfig], Dataset]
_REGISTRY: dict[str, Builder] = {}


def register(dataset_type: str):
    """
    Decorator to register dataset builders.
    """

    def _wrap(fn: Builder) -> Builder:
        key = dataset_type.lower().strip()
        if key in _REGISTRY:
            raise KeyError(f"Dataset builder already registered for {key!r}")
        _REGISTRY[key] = fn
        return fn

    return _wrap


def build_dataset(config: DatasetConfig) -> Dataset:
    """
    Main entry point: returns a ``torch.utils.data.Dataset``.

    Processing order:
      1. Build the raw dataset via its registered builder.
      2. Filter to ``config.classes`` (if set).
      3. Sub-sample to ``config.n_samples``.

    Each sample is a ``(input, target)`` tuple following standard PyTorch
    conventions.
    """
    key = config.dataset_type
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset_type {key!r}. "
            f"Registered types: {sorted(_REGISTRY.keys())}"
        )

    ds = _REGISTRY[key](config)

    # --- class filter (before n_samples limit) ---
    if config.classes is not None:
        ds = filter_classes(
            ds, config.classes, remap=config.remap_labels, label_map=config.label_map
        )

    # --- sub-sample ---
    ds = limit_dataset(ds, config.n_samples, seed=config.seed, shuffle=True)

    return ds


def load_tensors(
    *,
    dataset_type: str,
    split: str,
    n_samples: int,
    device: torch.device | str,
    seed: int = 0,
    root: str | Path = "./data",
    flatten: bool = True,
    class_preset: str | None = None,
    classes: list[int] | None = None,
    remap_labels: bool = True,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a dataset as ``(X, y)`` float tensors on *device*.

    Convenience wrapper around :func:`build_dataset` that materialises
    the full dataset into two tensors.

    Parameters
    ----------
    dataset_type : str
        Registered dataset name (e.g. ``"mnist"``, ``"cifar10"``).
    split : str
        ``"train"`` or ``"test"``.
    n_samples : int
        Number of samples to load.
    device : torch.device | str
        Target device for the returned tensors.
    seed : int
        Random seed for sub-sampling.
    root : str | Path
        Root data directory.
    flatten : bool
        Whether to flatten spatial dims (default ``True``).
    class_preset : str | None
        Named class-grouping preset.
    classes : list[int] | None
        Explicit class filter.
    remap_labels : bool
        Whether to remap labels (default ``True``).
    batch_size : int
        Batch size for the internal DataLoader (default 512).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(X, y)`` where ``X`` is float32 and ``y`` is float32.
    """
    ds_cfg = DatasetConfig(
        dataset_type=dataset_type,
        split=split,
        root=root,
        n_samples=n_samples,
        seed=seed,
        flatten=flatten,
        class_preset=class_preset,
        classes=classes,
        remap_labels=remap_labels,
    )
    ds = build_dataset(ds_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    xs, ys = zip(*list(loader))
    return (
        torch.cat(xs, dim=0).float().to(device),
        torch.cat(ys, dim=0).float().to(device),
    )


def available_datasets() -> list[str]:
    """Return sorted list of registered dataset type names."""
    return sorted(_REGISTRY.keys())
