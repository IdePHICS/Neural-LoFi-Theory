# ruff: noqa: N803, N806
"""Shared core for kernel spectral training (NNGP variant).

Mirrors :mod:`_ridge_spectral_core` but for the kernel-trick trainer.
Provides the filename helpers, content-addressed cache key/path
utilities, the disk-cache loader, and :func:`run_kernel_spectral` — the
single entry point used by both the one-shot training script and
(eventually) the Bayesian optimiser.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from config_utils import parse_classes
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from train_hierarchically.datasets import load_tensors
from train_hierarchically.models.kernel_spectral import (
    KernelLayerSpec,
    KernelSpectralModel,
)
from train_hierarchically.training.kernel_spectral import (
    KernelSpectralConfig,
    KernelSpectralTrainer,
    Layer1Bundle,
)
from train_hierarchically.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result file naming
# ---------------------------------------------------------------------------


def _layer_token(ld: dict[str, Any]) -> str:
    """Per-layer token used in the results filename.

    The ``do_eigenreduction`` flag is mirrored from
    :class:`LayerSpec.do_eigenreduction`: eigenreducing layers encode
    ``m`` in the name, deep-NNGP layers encode ``noeig``.
    """
    base = (
        f"{ld['kernel']}sw{ld.get('sigma_w', 1.0)}"
        f"sb{ld.get('sigma_b', 0.0)}q{ld.get('n_quad', 30)}"
    )
    if ld.get("do_eigenreduction", True):
        return f"{base}m{ld['m']}"
    return f"{base}noeig"


def build_results_filename(cfg: DictConfig, layer_dicts: list[dict[str, Any]]) -> str:
    """Deterministic JSON filename encoding all reproducibility-relevant config."""
    preset = cfg.dataset.get("preset")
    raw_classes = cfg.dataset.get("classes")
    if preset:
        task = preset
    elif raw_classes:
        task = "classes_" + "_".join(str(c) for c in raw_classes)
    else:
        task = "all"

    layers_str = (
        "-".join(_layer_token(ld) for ld in layer_dicts) if layer_dicts else "nolayers"
    )
    ridge_str = (
        f"ridge{cfg.ridge_alpha_min}to{cfg.ridge_alpha_max}x{cfg.ridge_alpha_num}"
    )
    eps_tok = f"_eps{cfg.eps}" if "eps" in cfg else ""
    return (
        f"{task}_seed{cfg.seed}"
        f"_ntrain{cfg.dataset.n_train}_ntest{cfg.dataset.n_test}"
        f"{eps_tok}_{layers_str}_{ridge_str}.json"
    )


# ---------------------------------------------------------------------------
# Layer-1 disk cache
# ---------------------------------------------------------------------------


def kernel_cache_key(
    *,
    dataset_name: str,
    preset: str | None,
    classes: list[int] | None,
    seed: int,
    n_max: int,
    n_test: int,
    kernel_name: str,
    sigma_w: float,
    sigma_b: float,
    n_quad: int,
    dtype: str,
) -> str:
    """SHA-1 of all layer-1-affecting parameters (used as cache directory)."""
    payload = json.dumps(
        {
            "dataset": dataset_name,
            "preset": preset,
            "classes": list(classes) if classes else None,
            "seed": int(seed),
            "n_max": int(n_max),
            "n_test": int(n_test),
            "kernel": kernel_name,
            "sigma_w": float(sigma_w),
            "sigma_b": float(sigma_b),
            "n_quad": int(n_quad),
            "dtype": dtype,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def kernel_cache_dir(base_dir: Path, key: str) -> Path:
    """Directory holding the cache for *key* under *base_dir*."""
    return Path(base_dir) / key


def load_layer1_cache(cache_dir: Path) -> Layer1Bundle | None:
    """Load a layer-1 cache produced by ``precompute_kernel.py``.

    Returns ``None`` when the cache directory or its metadata file is
    missing.  Partial caches (e.g. no eigen bundle) are returned with the
    available pieces; the trainer falls back to running its own
    eigendecomposition.
    """
    cache_dir = Path(cache_dir)
    if not (cache_dir / "metadata.json").exists():
        return None

    g_path = cache_dir / "G_full.npy"
    if not g_path.exists():
        return None

    g_full = np.load(g_path, mmap_mode="r")
    g_cross: np.ndarray | None = None
    g_cross_path = cache_dir / "G_cross_full.npy"
    if g_cross_path.exists():
        g_cross = np.load(g_cross_path, mmap_mode="r")

    alpha: np.ndarray | None = None
    lam: np.ndarray | None = None
    eig_path = cache_dir / "layer1_eigen.npz"
    if eig_path.exists():
        eig = np.load(eig_path)
        alpha = eig["alpha"]
        lam = eig["eigenvalues"]

    return Layer1Bundle(
        g_train=g_full, g_cross=g_cross, alpha_full=alpha, lambda_full=lam
    )


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------


def _resolve_layer1_cache(
    cfg: DictConfig, layer_dicts: list[dict[str, Any]]
) -> Layer1Bundle | None:
    """Look up the layer-1 cache pointed to by *cfg* (or return None)."""
    base = cfg.get("kernel_cache_dir")
    if base is None or not layer_dicts:
        return None

    base_path = Path(base).expanduser()
    if not base_path.exists():
        return None

    layer0 = layer_dicts[0]
    classes = parse_classes(cfg)
    key = kernel_cache_key(
        dataset_name=cfg.dataset.name,
        preset=cfg.dataset.get("preset"),
        classes=classes,
        seed=cfg.seed,
        n_max=int(cfg.dataset.get("n_max", cfg.dataset.n_train)),
        n_test=int(cfg.dataset.n_test),
        kernel_name=layer0["kernel"],
        sigma_w=layer0.get("sigma_w", 1.0),
        sigma_b=layer0.get("sigma_b", 0.0),
        n_quad=layer0.get("n_quad", 30),
        dtype=cfg.get("dtype", "float32"),
    )
    cache_dir = kernel_cache_dir(base_path, key)
    bundle = load_layer1_cache(cache_dir)
    if bundle is not None:
        log.info("Layer-1 cache hit: %s", cache_dir)
    else:
        log.info("Layer-1 cache miss at %s", cache_dir)
    return bundle


def _load_split_tensors(cfg: DictConfig, device: Any, split: str, n: int) -> Any:
    """Load one dataset split as ``(X, y)`` tensors."""
    return load_tensors(
        dataset_type=cfg.dataset.name,
        split=split,
        n_samples=n,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=cfg.dataset.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
    )


def run_kernel_spectral(
    cfg: DictConfig,
    *,
    seed: int | None = None,
    layer_dicts: list[dict[str, Any]] | None = None,
    out_path: Path | None = None,
    data: tuple | None = None,
) -> dict[str, Any]:
    """Run one kernel spectral training and return the results dict.

    Parameters
    ----------
    cfg
        Full experiment configuration (OmegaConf DictConfig).
    seed
        Override for ``cfg.seed``.
    layer_dicts
        Override for ``cfg.layers`` (already converted via
        ``OmegaConf.to_object``).
    out_path
        File-level cache.  When the file exists, the cached JSON is
        returned directly; otherwise the result is written there at the
        end of training.
    data
        Optional pre-loaded ``(X_train, y_train, X_test, y_test)`` tuple.
    """
    if out_path is not None and out_path.exists():
        log.info("Cache hit: %s", out_path)
        try:
            with open(out_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            log.warning("Corrupt cache file, re-running: %s", out_path)
            out_path.unlink()

    if seed is not None:
        cfg = cfg.copy()
        cfg.seed = seed

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    if layer_dicts is None:
        layer_dicts = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]

    if data is not None:
        X_train, y_train, X_test, y_test = data
    else:
        X_train, y_train = _load_split_tensors(
            cfg, device, "train", cfg.dataset.n_train
        )
        X_test, y_test = _load_split_tensors(cfg, device, "test", cfg.dataset.n_test)

    layer_specs = [
        KernelLayerSpec(
            kernel=ld["kernel"],
            m=ld.get("m", 0),
            sigma_w=ld.get("sigma_w", 1.0),
            sigma_b=ld.get("sigma_b", 0.0),
            n_quad=ld.get("n_quad", 30),
            do_eigenreduction=ld.get("do_eigenreduction", True),
        )
        for ld in layer_dicts
    ]

    log.info(
        "Model: input_dim=%d, layers=%s",
        X_train.shape[1],
        [
            (s.kernel, s.m, s.sigma_w, s.sigma_b, s.do_eigenreduction)
            for s in layer_specs
        ],
    )

    model = KernelSpectralModel(layer_specs=layer_specs)
    trainer = KernelSpectralTrainer(
        model=model,
        config=KernelSpectralConfig(
            device=device,
            verbose=True,
            ridge_alpha_min=cfg.ridge_alpha_min,
            ridge_alpha_max=cfg.ridge_alpha_max,
            ridge_alpha_num=cfg.ridge_alpha_num,
            eps=cfg.get("eps", 1e-6),
            m_max=cfg.get("m_max"),
            row_chunk=int(cfg.get("row_chunk") or 256),
            layer1_cache=_resolve_layer1_cache(cfg, layer_dicts),
            memory_budget_bytes=cfg.get("memory_budget_bytes"),
            auto_rescale=bool(cfg.get("auto_rescale", False)),
        ),
    )

    train_loader = DataLoader(
        list(zip(X_train, y_train)), batch_size=512, shuffle=False
    )
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=512, shuffle=False)

    _, results = trainer.fit(train_loader, test_loader=test_loader)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", out_path)

    return results
