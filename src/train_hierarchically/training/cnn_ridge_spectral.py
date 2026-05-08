"""CNN ridge spectral trainer: random convolutions + eigenreduction + RidgeCV.

Layer-wise procedure (CNN analog of :class:`RidgeSpectralTrainer`):
    1. Random conv + activation: Z_l = σ_l(conv2d(H, W_l))
    2. Reshape Z to (N * spatial, P) and compute signed covariance
    3. Keep top-K_l eigenvectors → V_l stored as (K_l, P_l, 1, 1)
    4. 1×1 projection: H = conv2d(Z, V_l)
    5. Repeat for each layer

Final readout is sklearn RidgeCV on the flattened feature map.

Memory-efficient mode (``covariance_chunk_size > 0``): for each layer,
streams chunks of the *raw* input ``x_all`` through all previously fitted
layers on-the-fly, so neither the full ``(N, P, H', W')`` intermediate
nor the ``(N, K, H', W')`` projected tensor is ever stored — only the
small ``(P, P)`` covariance accumulator lives in memory.  The trade-off
is O(L²) convolutions instead of O(L), but L is typically 2–3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.cnn_ridge_spectral import (
    CNNRidgeSpectralModel,
    _n_scattering_coefficients,
)
from ..utils.helpers import ACTIVATIONS
from ._eigen_utils import (
    signed_covariance_eigen,
    signed_covariance_full_eigenvalues,
    signed_covariance_full_eigenvalues_from_covariance,
)
from .base import BaseTrainer, fit_ridge_readout
from .ridge_spectral import RidgeSpectralConfig

log = logging.getLogger(__name__)

_conv2d = torch.nn.functional.conv2d
_max_pool2d = torch.nn.functional.max_pool2d
_avg_pool2d = torch.nn.functional.avg_pool2d


@dataclass
class CNNRidgeSpectralConfig(RidgeSpectralConfig):
    """Config for CNN ridge spectral training.

    Parameters
    ----------
    covariance_chunk_size : int
        Number of images per chunk when accumulating the signed covariance
        matrix.  ``0`` (default) disables chunking and processes the full
        dataset at once (original behaviour).
    """

    covariance_chunk_size: int = 0


class CNNRidgeSpectralTrainer(BaseTrainer):
    """Trainer for :class:`CNNRidgeSpectralModel`.

    Parameters
    ----------
    model : CNNRidgeSpectralModel
        Model to train.
    config : RidgeSpectralConfig | CNNRidgeSpectralConfig
        Training configuration.
    """

    def __init__(
        self,
        model: CNNRidgeSpectralModel,
        config: RidgeSpectralConfig,
    ) -> None:
        super().__init__(model, config)
        if not isinstance(model, CNNRidgeSpectralModel):
            raise TypeError(
                "CNNRidgeSpectralTrainer requires a "
                "CNNRidgeSpectralModel, "
                f"got {type(model).__name__}"
            )

    @torch.no_grad()
    def _conv_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        y_all_device: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        model: CNNRidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]
        chunk_size = (
            config.covariance_chunk_size
            if isinstance(config, CNNRidgeSpectralConfig)
            else 0
        )
        run_device = config.device

        spec = model.layer_specs[layer_idx]
        w_conv = model.random_conv_weights[layer_idx]
        sigma = ACTIVATIONS[spec.activation]
        eigvals: Tensor | None = None
        input_shape = tuple(h_train.shape[1:])
        assert (
            h_train.ndim == 4
            and h_train.shape[-1] == h_train.shape[-2]
            and (
                h_test is None
                or (h_test.ndim == 4 and h_test.shape[-1] == h_test.shape[-2])
            )
        ), (
            f"Layer {layer_idx} (conv) expects square image tensors "
            "(N, C, H, W) for train/test."
        )

        if chunk_size > 0:
            n_train = h_train.shape[0]

            store_full = bool(getattr(config, "store_full_eigenvalues", False))
            full_eigvals: Tensor | None = None

            if spec.do_eigenreduction:
                c_accum: Tensor | None = None
                total_patches = 0

                for start in range(0, n_train, chunk_size):
                    end = min(start + chunk_size, n_train)
                    h_chunk = h_train[start:end].to(run_device)
                    z_chunk = sigma(
                        _conv2d(
                            h_chunk,
                            w_conv,
                            padding=spec.padding,
                            stride=spec.stride,
                        )
                    )
                    c_part, n_patches = self._covariance_chunk(
                        z_chunk, y_all_device[start:end]
                    )
                    c_accum = c_part if c_accum is None else c_accum + c_part
                    total_patches += n_patches

                if c_accum is None or total_patches == 0:
                    raise RuntimeError(
                        "Could not accumulate covariance in chunked mode."
                    )

                c_accum = c_accum.to("cpu") / total_patches
                if store_full:
                    full_eigvals = signed_covariance_full_eigenvalues_from_covariance(
                        c_accum
                    )
                eigvals, eigvecs = self._eigen_from_covariance(c_accum, k=spec.k)

                v = eigvecs.T.unsqueeze(-1).unsqueeze(-1).to(run_device)
                model.eigenvector_projections[layer_idx] = nn.Parameter(
                    v, requires_grad=False
                )

            v_param = model.eigenvector_projections[layer_idx]

            new_train_chunks: list[Tensor] = []
            for start in range(0, n_train, chunk_size):
                end = min(start + chunk_size, n_train)
                h_chunk = h_train[start:end].to(run_device)
                z_chunk = sigma(
                    _conv2d(
                        h_chunk,
                        w_conv,
                        padding=spec.padding,
                        stride=spec.stride,
                    )
                )
                new_train_chunks.append(_conv2d(z_chunk, v_param).cpu())
            h_train = torch.cat(new_train_chunks)

            if h_test is not None:
                n_test = h_test.shape[0]
                new_test_chunks: list[Tensor] = []
                for start in range(0, n_test, chunk_size):
                    end = min(start + chunk_size, n_test)
                    h_chunk = h_test[start:end].to(run_device)
                    z_chunk = sigma(
                        _conv2d(
                            h_chunk,
                            w_conv,
                            padding=spec.padding,
                            stride=spec.stride,
                        )
                    )
                    new_test_chunks.append(_conv2d(z_chunk, v_param).cpu())
                h_test = torch.cat(new_test_chunks)

        else:
            z_train = sigma(
                _conv2d(
                    h_train,
                    w_conv,
                    padding=spec.padding,
                    stride=spec.stride,
                )
            )
            z_test = (
                sigma(
                    _conv2d(
                        h_test,
                        w_conv,
                        padding=spec.padding,
                        stride=spec.stride,
                    )
                )
                if h_test is not None
                else None
            )

            store_full = bool(getattr(config, "store_full_eigenvalues", False))
            full_eigvals: Tensor | None = None
            if spec.do_eigenreduction:
                eigvals, eigvecs = self._signed_covariance_eigen_spatial(
                    z_train, y_all_device, k=spec.k
                )
                if store_full:
                    full_eigvals = self._signed_covariance_full_eigen_spatial(
                        z_train, y_all_device
                    )
                v = eigvecs.T.unsqueeze(-1).unsqueeze(-1)
                model.eigenvector_projections[layer_idx] = nn.Parameter(
                    v, requires_grad=False
                )

            v_param = model.eigenvector_projections[layer_idx]
            h_train = _conv2d(z_train, v_param)
            h_test = _conv2d(z_test, v_param) if z_test is not None else None

        output_shape = tuple(h_train.shape[1:])

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "conv",
            "layer_name": (
                f"layer_{layer_idx} (P={spec.p}, K={spec.k}, kernel={spec.kernel_size})"
            ),
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }
        if spec.do_eigenreduction and eigvals is not None:
            layer_entry["eigenvalues"] = eigvals.cpu().numpy().tolist()
        if spec.do_eigenreduction and full_eigvals is not None:
            layer_entry["eigenvalues_full"] = (
                full_eigvals.cpu().numpy().tolist()
            )

        if config.verbose:
            message = f"  Layer {layer_idx}: conv train {input_shape} -> {output_shape}"
            if spec.do_eigenreduction and eigvals is not None:
                top = eigvals[: min(5, len(eigvals))].cpu().numpy()
                message += f", top eigenvalues={top}"
            log.info(message)

        return h_train, h_test, layer_entry

    @torch.no_grad()
    def _flatten_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]
        input_shape = tuple(h_train.shape[1:])
        assert (
            h_train.ndim == 4
            and h_train.shape[-1] == h_train.shape[-2]
            and (
                h_test is None
                or (h_test.ndim == 4 and h_test.shape[-1] == h_test.shape[-2])
            )
        ), (
            f"Layer {layer_idx} (flatten) expects square image tensors "
            "(N, C, H, W) for train/test."
        )

        h_train = h_train.flatten(1)
        h_test = h_test.flatten(1) if h_test is not None else None
        output_shape = tuple(h_train.shape[1:])

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "flatten",
            "layer_name": f"layer_{layer_idx} (flatten)",
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }

        if config.verbose:
            message = (
                f"  Layer {layer_idx}: flatten train {input_shape} -> {output_shape}"
            )
            log.info(message)

        return h_train, h_test, layer_entry

    @torch.no_grad()
    def _l2norm_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]
        input_shape = tuple(h_train.shape[1:])
        assert (
            h_train.ndim == 4
            and h_train.shape[-1] == h_train.shape[-2]
            and (
                h_test is None
                or (h_test.ndim == 4 and h_test.shape[-1] == h_test.shape[-2])
            )
        ), (
            f"Layer {layer_idx} (l2norm) expects square image tensors "
            "(N, C, H, W) for train/test."
        )

        epsilon = 1e-8

        h_train_norm = torch.sqrt(torch.sum(h_train**2, dim=1, keepdim=True) + epsilon)
        h_train = h_train / h_train_norm

        if h_test is not None:
            h_test_norm = torch.sqrt(
                torch.sum(h_test**2, dim=1, keepdim=True) + epsilon
            )
            h_test = h_test / h_test_norm

        output_shape = tuple(h_train.shape[1:])

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "l2norm",
            "layer_name": f"layer_{layer_idx} (l2norm)",
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }

        if config.verbose:
            message = (
                f"  Layer {layer_idx}: l2norm train {input_shape} -> {output_shape}"
            )
            print(message)

        return h_train, h_test, layer_entry

    @torch.no_grad()
    def _pooling_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        model: CNNRidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]
        spec = model.layer_specs[layer_idx]
        input_shape = tuple(h_train.shape[1:])
        assert (
            h_train.ndim == 4
            and h_train.shape[-1] == h_train.shape[-2]
            and (
                h_test is None
                or (h_test.ndim == 4 and h_test.shape[-1] == h_test.shape[-2])
            )
        ), (
            f"Layer {layer_idx} (pooling) expects square image tensors "
            "(N, C, H, W) for train/test."
        )

        pool_stride = (
            spec.pool_stride if spec.pool_stride is not None else spec.pool_kernel_size
        )
        pool_mode = spec.pool_mode.strip().lower()
        pool_fn = _avg_pool2d if pool_mode == "avg" else _max_pool2d

        h_train = pool_fn(
            h_train,
            kernel_size=spec.pool_kernel_size,
            stride=pool_stride,
            padding=spec.pool_padding,
        )
        h_test = (
            pool_fn(
                h_test,
                kernel_size=spec.pool_kernel_size,
                stride=pool_stride,
                padding=spec.pool_padding,
            )
            if h_test is not None
            else None
        )
        output_shape = tuple(h_train.shape[1:])

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "pooling",
            "layer_name": (
                f"layer_{layer_idx} (pooling mode={pool_mode}, "
                f"kernel={spec.pool_kernel_size}, stride={pool_stride})"
            ),
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }

        if config.verbose:
            message = (
                f"  Layer {layer_idx}: pooling train {input_shape} -> {output_shape}"
            )
            message += (
                f", mode={pool_mode}, kernel={spec.pool_kernel_size}, "
                f"stride={pool_stride}"
            )
            log.info(message)

        return h_train, h_test, layer_entry

    @torch.no_grad()
    def _scattering_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        y_all_device: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        """Apply wavelet scattering, eigenreduce, and project."""
        model: CNNRidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]

        spec = model.layer_specs[layer_idx]
        scat_idx = model._scattering_layer_map[layer_idx]
        scat = model.scattering_modules[scat_idx]
        input_shape = tuple(h_train.shape[1:])

        # Apply scattering transform
        z_train = scat(h_train)
        z_test = scat(h_test) if h_test is not None else None

        # Reshape 5D -> 4D if needed: (N, C, paths, H', W') -> (N, C*paths, H', W')
        if z_train.ndim == 5:
            n, c, paths, h_out, w_out = z_train.shape
            z_train = z_train.reshape(n, c * paths, h_out, w_out)
            if z_test is not None:
                nt = z_test.shape[0]
                z_test = z_test.reshape(nt, c * paths, h_out, w_out)

        eigvals: Tensor | None = None
        store_full = bool(getattr(config, "store_full_eigenvalues", False))
        full_eigvals: Tensor | None = None
        if spec.do_eigenreduction:
            eigvals, eigvecs = self._signed_covariance_eigen_spatial(
                z_train, y_all_device, k=spec.k
            )
            if store_full:
                full_eigvals = self._signed_covariance_full_eigen_spatial(
                    z_train, y_all_device
                )
            v = eigvecs.T.unsqueeze(-1).unsqueeze(-1)
            model.eigenvector_projections[layer_idx] = nn.Parameter(
                v, requires_grad=False
            )

        v_param = model.eigenvector_projections[layer_idx]
        h_train = _conv2d(z_train, v_param)
        h_test = _conv2d(z_test, v_param) if z_test is not None else None

        output_shape = tuple(h_train.shape[1:])
        n_paths = _n_scattering_coefficients(
            spec.scattering_j, spec.scattering_l, spec.scattering_max_order
        )

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "scattering",
            "layer_name": (
                f"layer_{layer_idx} "
                f"(scattering J={spec.scattering_j}, "
                f"L={spec.scattering_l}, "
                f"order={spec.scattering_max_order}, "
                f"paths={n_paths}, K={spec.k})"
            ),
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }
        if spec.do_eigenreduction and eigvals is not None:
            layer_entry["eigenvalues"] = eigvals.cpu().numpy().tolist()
        if spec.do_eigenreduction and full_eigvals is not None:
            layer_entry["eigenvalues_full"] = (
                full_eigvals.cpu().numpy().tolist()
            )

        if config.verbose:
            message = (
                f"  Layer {layer_idx}: scattering train {input_shape} -> {output_shape}"
            )
            if spec.do_eigenreduction and eigvals is not None:
                top = eigvals[: min(5, len(eigvals))].cpu().numpy()
                message += f", top eigenvalues={top}"
            log.info(message)

        return h_train, h_test, layer_entry

    @torch.no_grad()
    def _fully_connected_layer(
        self,
        *,
        layer_idx: int,
        h_train: Tensor,
        y_all_device: Tensor,
        h_test: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        model: CNNRidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]
        run_device = config.device

        spec = model.layer_specs[layer_idx]
        w_fc = model.random_fc_weights[layer_idx]
        sigma = ACTIVATIONS[spec.activation]
        input_shape = tuple(h_train.shape[1:])
        assert h_train.ndim == 2 and (h_test is None or h_test.ndim == 2), (
            f"Layer {layer_idx} (fully_connected) expects flattened "
            "vectors (N, D) for train/test."
        )

        h_train_2d = h_train
        h_test_2d = h_test
        h_train_2d = h_train_2d.to(device=run_device, dtype=torch.float32)
        h_test_2d = (
            h_test_2d.to(device=run_device, dtype=torch.float32)
            if h_test_2d is not None
            else None
        )

        z_train = sigma(h_train_2d @ w_fc)
        z_test = sigma(h_test_2d @ w_fc) if h_test_2d is not None else None

        store_full = bool(getattr(config, "store_full_eigenvalues", False))
        eigvals: Tensor | None = None
        full_eigvals: Tensor | None = None
        if spec.do_eigenreduction:
            eigvals, eigvecs = self._signed_covariance_eigen_dense(
                z_train,
                y_all_device.to(device=z_train.device, dtype=z_train.dtype),
                k=spec.k,
            )
            if store_full:
                full_eigvals = self._signed_covariance_full_eigen_dense(
                    z_train,
                    y_all_device.to(device=z_train.device, dtype=z_train.dtype),
                )
            model.fc_eigenvector_projections[layer_idx] = nn.Parameter(
                eigvecs, requires_grad=False
            )

        v_param = model.fc_eigenvector_projections[layer_idx]
        h_train = z_train @ v_param
        h_test = z_test @ v_param if z_test is not None else None
        output_shape = tuple(h_train.shape[1:])

        layer_entry: dict[str, Any] = {
            "layer_idx": layer_idx,
            "kind": "fully_connected",
            "layer_name": (
                f"layer_{layer_idx} (fully_connected P={spec.p}, K={spec.k})"
            ),
            "train_input_shape": input_shape,
            "train_output_shape": output_shape,
        }
        if spec.do_eigenreduction and eigvals is not None:
            layer_entry["eigenvalues"] = eigvals.cpu().numpy().tolist()
        if spec.do_eigenreduction and full_eigvals is not None:
            layer_entry["eigenvalues_full"] = (
                full_eigvals.cpu().numpy().tolist()
            )

        if config.verbose:
            message = (
                f"  Layer {layer_idx}: fully_connected train "
                f"{input_shape} -> {output_shape}"
            )
            if spec.do_eigenreduction and eigvals is not None:
                top = eigvals[: min(5, len(eigvals))].cpu().numpy()
                message += f", top eigenvalues={top}"
            log.info(message)

        return h_train, h_test, layer_entry

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Fit all layers and the ridge readout.

        Parameters
        ----------
        loader : DataLoader
            Training data yielding ``(X, y)`` batches.  ``X`` must be
            ``(batch, C, H, W)`` (not flattened).
        test_loader : DataLoader | None
            Optional test loader for evaluation.

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model and a results dict.
        """
        test_loader: DataLoader | None = kwargs.get("test_loader")  # type: ignore[assignment]
        model: CNNRidgeSpectralModel = self.model  # type: ignore[assignment]
        config: RidgeSpectralConfig = self.config  # type: ignore[assignment]

        chunk_size = (
            config.covariance_chunk_size
            if isinstance(config, CNNRidgeSpectralConfig)
            else 0
        )

        run_device = config.device
        x_all, y_all = self._collect_dataset(loader)
        h_train = x_all.to(device=run_device, dtype=torch.float32)
        y_all_device = y_all.to(device=run_device, dtype=torch.float32)

        x_test: Tensor | None = None
        y_test_np: np.ndarray | None = None
        if test_loader is not None:
            x_test, y_test_t = self._collect_dataset(test_loader)
            y_test_np = y_test_t.cpu().numpy()
        h_test: Tensor | None = (
            x_test.to(device=run_device, dtype=torch.float32)
            if x_test is not None
            else None
        )

        if config.verbose:
            specs_summary = [
                (
                    s.kind,
                    s.p,
                    s.k,
                    s.activation,
                    s.do_eigenreduction,
                )
                for s in model.layer_specs
            ]
            log.info(
                "CNN ridge spectral fit: %d layers, specs=%s",
                model.n_layers,
                specs_summary,
            )
            if chunk_size > 0:
                log.info("  Chunked mode: chunk_size=%d", chunk_size)

        layer_history: list[dict[str, Any]] = []

        for layer_idx in range(model.n_layers):
            spec = model.layer_specs[layer_idx]
            kind = spec.kind

            if kind == "scattering":
                h_train, h_test, layer_entry = self._scattering_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    y_all_device=y_all_device,
                    h_test=h_test,
                )
            elif kind == "conv":
                h_train, h_test, layer_entry = self._conv_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    y_all_device=y_all_device,
                    h_test=h_test,
                )
            elif kind == "flatten":
                h_train, h_test, layer_entry = self._flatten_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    h_test=h_test,
                )
            elif kind == "pooling":
                h_train, h_test, layer_entry = self._pooling_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    h_test=h_test,
                )
            elif kind == "l2norm":
                h_train, h_test, layer_entry = self._l2norm_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    h_test=h_test,
                )
            elif kind == "fully_connected":
                h_train, h_test, layer_entry = self._fully_connected_layer(
                    layer_idx=layer_idx,
                    h_train=h_train,
                    y_all_device=y_all_device,
                    h_test=h_test,
                )
            else:
                raise ValueError(f"Unsupported layer kind: {kind!r}")

            layer_history.append(layer_entry)

        final = fit_ridge_readout(
            model,
            h_train,
            y_all,
            h_test=h_test,
            y_test_np=y_test_np,
            alpha_min=config.ridge_alpha_min,
            alpha_max=config.ridge_alpha_max,
            alpha_num=config.ridge_alpha_num,
            verbose=config.verbose,
        )

        results: dict[str, Any] = {
            "layers": layer_history,
            "final": final,
        }

        return model, results

    # ------------------------------------------------------------------
    # Chunked covariance helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _covariance_chunk(z: Tensor, y: Tensor) -> tuple[Tensor, int]:
        """Unnormalised signed covariance from one chunk.

        Returns ``(C_part, B*S)`` where ``C_part = Σ y_i Z_i^T Z_i``
        (unnormalised) and ``B*S`` is the number of spatial patches.
        """
        b, p, h_out, w_out = z.shape
        n_patches = h_out * w_out
        y_f = y.to(z.dtype)
        z_flat = z.permute(0, 2, 3, 1).reshape(b * n_patches, p)
        y_flat = y_f.unsqueeze(1).expand(b, n_patches).reshape(b * n_patches)
        weighted = y_flat.unsqueeze(1) * z_flat  # (B*S, P)
        c_part = weighted.T @ z_flat  # (P, P)
        return c_part, b * n_patches

    @staticmethod
    @torch.no_grad()
    def _eigen_from_covariance(c: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Eigendecompose a (P, P) covariance matrix, keep top-k.

        Uses ``scipy.sparse.linalg.eigsh`` with ``which='LM'`` (largest
        magnitude) via the implicitly restarted Lanczos method, computing
        only *k* eigenpairs instead of the full O(P³) decomposition.
        """
        c_np = c.cpu().numpy()
        p = c_np.shape[0]
        v0 = np.ones(p, dtype=c_np.dtype) / np.sqrt(p)
        eigvals_np, eigvecs_np = eigsh(c_np, k=k, which="LM", v0=v0)

        eigvals = torch.from_numpy(eigvals_np).to(dtype=c.dtype)
        eigvecs = torch.from_numpy(eigvecs_np).to(dtype=c.dtype)

        order = torch.argsort(eigvals.abs(), descending=True)
        return eigvals[order], eigvecs[:, order]

    # ------------------------------------------------------------------
    # Non-chunked signed covariance (original path)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _signed_covariance_eigen_dense(
        z: Tensor, y: Tensor, k: int
    ) -> tuple[Tensor, Tensor]:
        """Eigendecompose signed covariance from dense features ``(N, P)``."""
        return signed_covariance_eigen(z, y, k)

    @staticmethod
    @torch.no_grad()
    def _signed_covariance_eigen_spatial(
        z: Tensor, y: Tensor, k: int
    ) -> tuple[Tensor, Tensor]:
        """Eigendecompose signed covariance from (N, P, H, W)."""
        n, p, h_out, w_out = z.shape
        n_patches = h_out * w_out
        y_f = y.to(z.dtype)
        z_flat = z.permute(0, 2, 3, 1).reshape(n * n_patches, p)
        y_flat = y_f.unsqueeze(1).expand(n, n_patches).reshape(n * n_patches)
        return signed_covariance_eigen(z_flat, y_flat, k)

    @staticmethod
    @torch.no_grad()
    def _signed_covariance_full_eigen_dense(z: Tensor, y: Tensor) -> Tensor:
        """All eigenvalues from dense features ``(N, P)``."""
        return signed_covariance_full_eigenvalues(z, y)

    @staticmethod
    @torch.no_grad()
    def _signed_covariance_full_eigen_spatial(z: Tensor, y: Tensor) -> Tensor:
        """All eigenvalues from (N, P, H, W) features."""
        n, p, h_out, w_out = z.shape
        n_patches = h_out * w_out
        y_f = y.to(z.dtype)
        z_flat = z.permute(0, 2, 3, 1).reshape(n * n_patches, p)
        y_flat = y_f.unsqueeze(1).expand(n, n_patches).reshape(n * n_patches)
        return signed_covariance_full_eigenvalues(z_flat, y_flat)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_dataset(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        """Load the entire dataset from *loader* into CPU tensors.

        Overrides the base implementation to keep data on CPU, avoiding
        GPU memory pressure.  Chunks are moved to the accelerator
        on-the-fly during training.
        """
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)
