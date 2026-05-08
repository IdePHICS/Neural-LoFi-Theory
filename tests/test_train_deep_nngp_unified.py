"""Tests for the unified ``train_deep_nngp`` entry point.

Covers the migration from the old flat ``n_layers + kernel`` schema to
the unified ``layers: [...]`` list schema shared with
``train_kernel_spectral``.

* ``TestValidation`` — the all-no-eigen guard in
  :func:`train_deep_nngp.validate_all_no_eigen` accepts valid deep-NNGP
  configs and rejects anything with an eigen layer.
* ``TestShippedConfig`` — the bundled ``conf/train_deep_nngp.yaml``
  parses, has a non-empty ``layers`` list, and every layer has
  ``do_eigenreduction: false``.
* ``TestUnifiedPipeline`` — an all-no-eigen ``layers`` list through the
  kernel-spectral trainer produces the same predictions as a direct
  ``deep_nngp_kernel + kernel_ridge_cv`` reference.  This pins the
  claim that removing the flat schema did not change the numerics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_hierarchically.models.kernel_spectral import (
    KernelLayerSpec,
    KernelSpectralModel,
)
from train_hierarchically.training.deep_nngp import deep_nngp_kernel
from train_hierarchically.training.kernel_ridge import (
    kernel_ridge_cv,
    kernel_ridge_predict,
)
from train_hierarchically.training.kernel_spectral import (
    KernelSpectralConfig,
    KernelSpectralTrainer,
)

# The script lives under ``scripts/`` which isn't on the default import
# path — add it before importing the entry-point module.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import train_deep_nngp  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================


def _toy_data(seed: int = 0, n_train: int = 60, n_test: int = 30, d: int = 8):
    gen = torch.Generator().manual_seed(seed)
    x_tr = torch.randn(n_train, d, generator=gen)
    y_tr = torch.where(torch.randn(n_train, generator=gen) > 0, 1.0, -1.0)
    x_te = torch.randn(n_test, d, generator=gen)
    y_te = torch.where(torch.randn(n_test, generator=gen) > 0, 1.0, -1.0)
    return x_tr, y_tr, x_te, y_te


# ============================================================================
# 1. validate_all_no_eigen
# ============================================================================


class TestValidation:
    """``validate_all_no_eigen`` allows deep-NNGP configs, rejects others."""

    def test_accepts_all_no_eigen(self) -> None:
        layers = [
            {"kernel": "sigmoid", "m": 0, "sigma_w": 1.0, "sigma_b": 0.0,
             "n_quad": 20, "do_eigenreduction": False}
            for _ in range(3)
        ]
        # No exception.
        train_deep_nngp.validate_all_no_eigen(layers)

    def test_rejects_single_eigen_layer(self) -> None:
        layers = [
            {"kernel": "sigmoid", "m": 0, "sigma_w": 1.0, "sigma_b": 0.0,
             "n_quad": 20, "do_eigenreduction": False},
            {"kernel": "sigmoid", "m": 50, "sigma_w": 1.0, "sigma_b": 0.0,
             "n_quad": 20, "do_eigenreduction": True},
            {"kernel": "sigmoid", "m": 0, "sigma_w": 1.0, "sigma_b": 0.0,
             "n_quad": 20, "do_eigenreduction": False},
        ]
        with pytest.raises(ValueError, match=r"do_eigenreduction=False"):
            train_deep_nngp.validate_all_no_eigen(layers)

    def test_rejects_default_missing_flag(self) -> None:
        """A layer without an explicit ``do_eigenreduction`` flag is
        treated as eigen (spec default), so the validator must reject it."""
        layers = [{"kernel": "sigmoid", "m": 0, "sigma_w": 1.0}]
        with pytest.raises(ValueError):
            train_deep_nngp.validate_all_no_eigen(layers)

    def test_error_message_lists_offending_indices(self) -> None:
        layers = [
            {"kernel": "sigmoid", "m": 0, "do_eigenreduction": False},
            {"kernel": "sigmoid", "m": 10, "do_eigenreduction": True},
            {"kernel": "sigmoid", "m": 10, "do_eigenreduction": True},
        ]
        with pytest.raises(ValueError, match=r"\[1, 2\]"):
            train_deep_nngp.validate_all_no_eigen(layers)


# ============================================================================
# 2. Shipped conf/train_deep_nngp.yaml
# ============================================================================


class TestShippedConfig:
    """The bundled deep-NNGP config is a valid layers-list schema."""

    @staticmethod
    def _load_shipped() -> OmegaConf:
        path = Path(__file__).resolve().parent.parent / "conf" / "train_deep_nngp.yaml"
        return OmegaConf.load(str(path))

    def test_uses_layers_list_schema(self) -> None:
        cfg = self._load_shipped()
        assert "layers" in cfg, "new schema must expose a top-level ``layers`` list"
        layer_dicts = OmegaConf.to_object(cfg.layers)
        assert isinstance(layer_dicts, list) and len(layer_dicts) > 0

    def test_legacy_fields_removed(self) -> None:
        """``n_layers`` was the flat-schema depth field and should be gone
        (depth now comes from the list length)."""
        cfg = self._load_shipped()
        assert "n_layers" not in cfg, (
            "legacy ``n_layers`` field is still present — it should be "
            "removed; depth is implied by len(layers)"
        )

    def test_all_layers_no_eigen(self) -> None:
        cfg = self._load_shipped()
        layer_dicts = OmegaConf.to_object(cfg.layers)
        # Validator is our single source of truth.
        train_deep_nngp.validate_all_no_eigen(layer_dicts)


# ============================================================================
# 3. Pipeline equivalence: layers-list ↔ direct deep_nngp_kernel
# ============================================================================


class TestUnifiedPipeline:
    """Running an all-no-eigen ``layers`` list through the kernel-spectral
    trainer produces the same predictions as a direct
    ``deep_nngp_kernel + kernel_ridge_cv`` reference — the numerics did
    not change when the flat adapter was removed.
    """

    def test_matches_direct_reference(self) -> None:
        x_tr, y_tr, x_te, y_te = _toy_data(seed=0, n_train=60, n_test=30, d=8)
        n_layers = 3

        tr = DataLoader(list(zip(x_tr, y_tr)), batch_size=32, shuffle=False)
        te = DataLoader(list(zip(x_te, y_te)), batch_size=32, shuffle=False)

        # --- Path A: unified kernel-spectral trainer with all-no-eigen ---
        specs = [
            KernelLayerSpec(
                kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                n_quad=15, do_eigenreduction=False,
            )
            for _ in range(n_layers)
        ]
        model = KernelSpectralModel(specs)
        trainer = KernelSpectralTrainer(
            model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-6),
        )
        _, r_trainer = trainer.fit(tr, test_loader=te)
        mse_trainer = float(r_trainer["final"]["test_mse"])

        # --- Path B: direct deep_nngp_kernel + kernel_ridge_cv ---
        kernels = [s.build() for s in specs]
        k_train, k_cross = deep_nngp_kernel(
            kernels, x_tr, x_te, n_layers=n_layers,
        )
        fit = kernel_ridge_cv(
            k_train.numpy().astype(np.float32),
            y_tr.numpy(),
            alphas=np.logspace(-6, 6, 500),
        )
        y_pred_ref = kernel_ridge_predict(fit, k_cross.numpy())
        mse_ref = float(np.mean((y_te.numpy() - y_pred_ref) ** 2))

        # Unified path must agree with the direct computation to float
        # round-off (both call the same underlying kernel routines).
        assert abs(mse_trainer - mse_ref) < 1e-5, (
            f"unified trainer MSE {mse_trainer:.6f} disagrees with "
            f"direct deep_nngp_kernel+kernel_ridge_cv reference "
            f"{mse_ref:.6f} — the layers-list schema changed the "
            f"numerics relative to the legacy path."
        )

    def test_depth_is_controlled_by_list_length(self) -> None:
        """Appending a layer should deepen the pipeline — no separate
        ``n_layers`` knob needed."""
        x_tr, y_tr, x_te, y_te = _toy_data(seed=1, n_train=40, n_test=20, d=6)
        tr = DataLoader(list(zip(x_tr, y_tr)), batch_size=32, shuffle=False)
        te = DataLoader(list(zip(x_te, y_te)), batch_size=32, shuffle=False)

        def mk(n_layers: int) -> float:
            specs = [
                KernelLayerSpec(
                    kernel="sigmoid", m=0, sigma_w=1.0, sigma_b=0.0,
                    n_quad=15, do_eigenreduction=False,
                )
                for _ in range(n_layers)
            ]
            model = KernelSpectralModel(specs)
            trainer = KernelSpectralTrainer(
                model, KernelSpectralConfig(device="cpu", verbose=False, eps=1e-6),
            )
            _, r = trainer.fit(tr, test_loader=te)
            return float(r["final"]["test_mse"])

        mse_1 = mk(1)
        mse_3 = mk(3)
        # Both run to completion with finite MSE; model depths are
        # controlled by the layers list length without a separate
        # n_layers knob.
        assert 0.0 <= mse_1 < 5.0
        assert 0.0 <= mse_3 < 5.0
        # Different depths should give different predictions (sanity).
        assert mse_1 != mse_3


# ============================================================================
# 4. Entry-point script: legacy adapter module was removed
# ============================================================================


class TestLegacyAdapterRemoved:
    """``scripts/_deep_nngp_core.py`` was the adapter from the flat
    schema to the unified one; it should no longer exist."""

    def test_adapter_module_gone(self) -> None:
        path = Path(__file__).resolve().parent.parent / "scripts" / "_deep_nngp_core.py"
        assert not path.exists(), (
            f"Legacy adapter {path} still present — it should have been "
            f"removed when the flat schema was retired."
        )

    def test_train_deep_nngp_imports_from_kernel_spectral_core(self) -> None:
        """The entry point should source its runner from the unified
        core, not a deep-NNGP-specific adapter."""
        import inspect

        src = inspect.getsource(train_deep_nngp)
        assert "_kernel_spectral_core" in src, (
            "train_deep_nngp should now import its runner from "
            "_kernel_spectral_core; it still references a separate "
            "deep-NNGP adapter."
        )
        assert "_deep_nngp_core" not in src
