"""CNN ridge spectral model: random convolutions + eigenvector projection.

Ridge readout on top of random convolutional features.

CNN analog of :class:`RidgeSpectralModel`.  Each layer performs:

1. Random conv (frozen): ``Z_l = σ_l(conv2d(H_{l-1}, W_l))``
2. Eigenvector channel reduction (1×1 conv): ``H_l = conv2d(Z_l, V_l)``

Final output::

    output = flatten(H_L) @ ridge_weight + ridge_bias

The random conv projects each spatial patch (C_in × k × k) onto *P*
random directions and applies an activation — this is the convolutional
analog of the dense random expansion ``σ(h @ W)`` in the feedforward
:class:`RidgeSpectralModel`.

The eigenvector projection reduces channels from *P* to *K* using the
top eigenvectors of the signed (label-weighted) covariance of the
*P*-dimensional feature vectors across all spatial locations.  This is
stored as a (K, P, 1, 1) parameter and applied as a pointwise (1×1)
convolution — the convolutional analog of ``z @ V`` in the feedforward
model.

Wavelet scattering layers (``kind="scattering"``) use Kymatio's
``Scattering2D`` as a deterministic, multiscale feature extractor.
The scattering transform is inherently nonlinear (wavelet modulus)
and produces structured features at multiple scales and orientations.
Eigenreduction then selects the task-relevant linear combinations
of scattering channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
try:
    from kymatio.scattering2d.frontend.torch_frontend import (
        ScatteringTorch2D as Scattering2D,
    )
except ImportError as e:
    Scattering2D = None
    print("Warning: Kymatio is not installed. Scattering layers will not work.")
from torch import Tensor, nn

from ..utils.helpers import ACTIVATIONS
from .scattering_utils import n_scattering_coefficients, scattering_output_spatial

# Keep old names as aliases for backward compatibility with external imports
_n_scattering_coefficients = n_scattering_coefficients
_scattering_output_spatial = scattering_output_spatial


@dataclass(slots=True)
class CNNLayerSpec:
    """Specification for one CNN ridge-spectral layer.

    Parameters
    ----------
    kind : str
        Layer kind. Supported values: ``"conv"``, ``"flatten"``,
        ``"fully_connected"``, ``"pooling"``, ``"l2_norm"``. Defaults to ``"conv"``.
    p : int
        Expansion width. Used by ``conv`` (output channels) and
        ``fully_connected`` (hidden width).
    k : int
        Number of top eigenvectors to keep when
        ``do_eigenreduction=True`` (for ``conv`` and ``fully_connected``).
    backprop_channels : int
        For backprop-trained models: overrides the output dimensions for
        this layer. Priority: ``backprop_channels`` > ``k`` (if
        ``do_eigenreduction=True``) > ``p``. If 0, falls back to the
        default logic.
    kernel_size : int
        Spatial extent of the convolutional kernel for ``conv``.
    padding : int
        Zero-padding applied before convolution for ``conv``.
    stride : int
        Stride for ``conv``.
    activation : str
        Activation function name (key into ``ACTIVATIONS``), used by
        ``conv`` and ``fully_connected``.
    do_eigenreduction : bool
        If ``False``, skip eigenvector projection (identity projection).
    pool_mode : str
        Pooling mode for ``pooling``: ``"max"`` or ``"avg"``.
    pool_kernel_size : int
        Pooling kernel size for ``pooling``.
    pool_stride : int | None
        Pooling stride for ``pooling``. If ``None``, defaults to
        ``pool_kernel_size``.
    pool_padding : int
        Pooling padding for ``pooling``.
    """

    kind: str = "conv"
    p: int = 0
    k: int = 0
    backprop_channels: int = 0
    kernel_size: int = 5
    padding: int = 2
    stride: int = 1
    activation: str = "sigmoid"
    do_eigenreduction: bool = True
    pool_mode: str = "max"
    pool_kernel_size: int = 2
    pool_stride: int | None = None
    pool_padding: int = 0
    scattering_j: int = 2
    scattering_l: int = 8
    scattering_max_order: int = 2


class CNNRidgeSpectralModel(nn.Module):
    """CNN ridge spectral model with mixed layer kinds.

    Supported layer kinds are:

    - ``conv``: random conv + activation + 1x1 eigenvector projection
    - ``pooling``: max/avg pooling
    - ``flatten``: flatten to ``(N, D)``
    - ``fully_connected``: random dense layer + activation + projection
    - ``l2norm``: L2 normalization across channels at each spatial location

    Architecture per ``conv`` layer *l*::

        Z_l = σ_l(conv2d(H_{l-1}, W_l, padding_l, stride_l))
        H_l = conv2d(Z_l, V_l)          # 1×1 eigenvector projection

    Final output::

        output = flatten(H_L) @ ridge_weight + ridge_bias

    ``W_l`` is a frozen random convolutional kernel of shape
    ``(P_l, C_in, k_l, k_l)``.  ``V_l`` is a ``(K_l, P_l, 1, 1)``
    weight set during training by the signed-covariance
    eigendecomposition.  ``ridge_weight`` / ``ridge_bias`` come from
    ``sklearn.linear_model.RidgeCV``.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    spatial_size : tuple[int, int]
        ``(H, W)`` of the input images — needed to pre-compute the
        flattened output dimension for the ridge readout.
    layer_specs : list[CNNLayerSpec]
        Per-layer configuration.
    seed : int
        Random seed for generating the frozen convolution kernels.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        layer_specs: list[CNNLayerSpec],
        seed: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.layer_specs = layer_specs
        self.n_layers = len(layer_specs)

        # Per-layer params for conv and fully-connected kinds.
        self.random_conv_weights = nn.ParameterList()
        self.eigenvector_projections = nn.ParameterList()
        self.random_fc_weights = nn.ParameterList()
        self.fc_eigenvector_projections = nn.ParameterList()
        self.scattering_modules = nn.ModuleList()
        self._scattering_layer_map: dict[int, int] = {}
        self.activation_names: list[str] = []
        self.layer_kinds: list[str] = []

        gen = torch.Generator()
        gen.manual_seed(seed)

        c_in = in_channels
        h, w = spatial_size
        feature_dim = c_in * h * w
        is_flat = False

        for layer_idx, spec in enumerate(layer_specs):
            kind = spec.kind
            self.layer_kinds.append(kind)
            self.activation_names.append(spec.activation)

            empty = nn.Parameter(torch.empty(0), requires_grad=False)
            conv_weight = empty
            conv_proj = empty
            fc_weight = empty
            fc_proj = empty

            if kind == "scattering":
                if is_flat:
                    raise ValueError(
                        "scattering layer cannot come after flatten/fully_connected"
                    )
                if spec.do_eigenreduction and spec.k <= 0:
                    raise ValueError(
                        "scattering layer with eigenreduction requires k > 0"
                    )

                scat = Scattering2D(
                    J=spec.scattering_j,
                    shape=(h, w),
                    L=spec.scattering_l,
                    max_order=spec.scattering_max_order,
                )
                scat_idx = len(self.scattering_modules)
                self.scattering_modules.append(scat)
                self._scattering_layer_map[layer_idx] = scat_idx

                n_paths = _n_scattering_coefficients(
                    spec.scattering_j,
                    spec.scattering_l,
                    spec.scattering_max_order,
                )
                out_channels_scat = c_in * n_paths
                h, w = _scattering_output_spatial(h, w, spec.scattering_j)

                if spec.do_eigenreduction:
                    v = torch.zeros(spec.k, out_channels_scat, 1, 1)
                    conv_proj = nn.Parameter(v, requires_grad=False)
                    c_in = spec.k
                else:
                    v = torch.eye(out_channels_scat).unsqueeze(-1).unsqueeze(-1)
                    conv_proj = nn.Parameter(v, requires_grad=False)
                    c_in = out_channels_scat

                feature_dim = c_in * h * w

            elif kind == "conv":
                if is_flat:
                    raise ValueError(
                        "conv layer cannot come after flatten/fully_connected"
                    )
                if spec.p <= 0:
                    raise ValueError("conv layer requires p > 0")
                if spec.do_eigenreduction and spec.k <= 0:
                    raise ValueError("conv layer with eigenreduction requires k > 0")

                wt = (
                    torch.randn(
                        spec.p,
                        c_in,
                        spec.kernel_size,
                        spec.kernel_size,
                        generator=gen,
                    )
                    / (spec.p) ** 0.5
                )
                conv_weight = nn.Parameter(wt, requires_grad=False)

                if spec.do_eigenreduction:
                    v = torch.zeros(spec.k, spec.p, 1, 1)
                    conv_proj = nn.Parameter(v, requires_grad=False)
                    out_channels = spec.k
                else:
                    v = torch.eye(spec.p).unsqueeze(-1).unsqueeze(-1)
                    conv_proj = nn.Parameter(v, requires_grad=False)
                    out_channels = spec.p

                h = (h + 2 * spec.padding - spec.kernel_size) // spec.stride + 1
                w = (w + 2 * spec.padding - spec.kernel_size) // spec.stride + 1
                c_in = out_channels
                feature_dim = c_in * h * w

            elif kind == "pooling":
                if is_flat:
                    raise ValueError(
                        "pooling layer cannot come after flatten/fully_connected"
                    )
                pool_stride = (
                    spec.pool_stride
                    if spec.pool_stride is not None
                    else spec.pool_kernel_size
                )
                h = (
                    h + 2 * spec.pool_padding - spec.pool_kernel_size
                ) // pool_stride + 1
                w = (
                    w + 2 * spec.pool_padding - spec.pool_kernel_size
                ) // pool_stride + 1
                if h <= 0 or w <= 0:
                    raise ValueError(
                        "pooling configuration collapses spatial dimensions"
                    )
                feature_dim = c_in * h * w

            elif kind == "l2norm":
                if is_flat:
                    raise ValueError(
                        "l2norm layer cannot come after flatten/fully_connected"
                    )
                # Spatial dimensions and channels are unchanged

            elif kind == "flatten":
                if is_flat:
                    raise ValueError(
                        "flatten layer expects image features and cannot come "
                        "after flatten/fully_connected"
                    )
                is_flat = True
                feature_dim = c_in * h * w

            elif kind == "fully_connected":
                if not is_flat:
                    raise ValueError(
                        "fully_connected layer expects flattened features; "
                        "insert a flatten layer before it"
                    )
                if spec.p <= 0:
                    raise ValueError("fully_connected layer requires p > 0")
                if spec.do_eigenreduction and spec.k <= 0:
                    raise ValueError(
                        "fully_connected layer with eigenreduction requires k > 0"
                    )

                wt = torch.randn(feature_dim, spec.p, generator=gen) / (spec.p) ** 0.5
                fc_weight = nn.Parameter(wt, requires_grad=False)

                if spec.do_eigenreduction:
                    v = torch.zeros(spec.p, spec.k)
                    fc_proj = nn.Parameter(v, requires_grad=False)
                    feature_dim = spec.k
                else:
                    v = torch.eye(spec.p)
                    fc_proj = nn.Parameter(v, requires_grad=False)
                    feature_dim = spec.p

            self.random_conv_weights.append(conv_weight)
            self.eigenvector_projections.append(conv_proj)
            self.random_fc_weights.append(fc_weight)
            self.fc_eigenvector_projections.append(fc_proj)

        self._flat_dim = feature_dim if is_flat else c_in * h * w

        # Ridge readout: weight and bias
        self.ridge_weight = nn.Parameter(
            torch.zeros(self._flat_dim), requires_grad=False
        )
        self.ridge_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def flat_dim(self) -> int:
        """Dimensionality of the flattened feature map before ridge readout."""
        return self._flat_dim

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass through all configured layers + ridge readout.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, C, H, W)``.

        Returns
        -------
        Tensor
            Predictions of shape ``(batch,)``.
        """
        h = x
        for i in range(self.n_layers):
            spec = self.layer_specs[i]
            kind = self.layer_kinds[i]

            if kind == "scattering":
                if h.ndim != 4:
                    raise ValueError(
                        f"Layer {i} (scattering) expects 4D image "
                        f"features (N, C, H, W), "
                        f"got shape={tuple(h.shape)}"
                    )
                scat_idx = self._scattering_layer_map[i]
                scat = self.scattering_modules[scat_idx]
                z = scat(h)
                # Kymatio output may be 5D: (N, C, paths, H', W')
                if z.ndim == 5:
                    n, c, paths, h_out, w_out = z.shape
                    z = z.reshape(n, c * paths, h_out, w_out)
                v = self.eigenvector_projections[i]
                h = torch.nn.functional.conv2d(z, v)

            elif kind == "conv":
                if h.ndim != 4:
                    raise ValueError(
                        f"Layer {i} (conv) expects 4D image features (N, C, H, W), "
                        f"got shape={tuple(h.shape)}"
                    )
                w_conv = self.random_conv_weights[i]
                if h.shape[1] != w_conv.shape[1]:
                    raise ValueError(
                        f"Layer {i} (conv) expected {w_conv.shape[1]} input channels, "
                        f"got {h.shape[1]}"
                    )
                v = self.eigenvector_projections[i]
                sigma = ACTIVATIONS[self.activation_names[i]]

                z = sigma(
                    torch.nn.functional.conv2d(
                        h, w_conv, padding=spec.padding, stride=spec.stride
                    )
                )
                h = torch.nn.functional.conv2d(z, v)

            elif kind == "pooling":
                if h.ndim != 4:
                    raise ValueError(
                        f"Layer {i} (pooling) expects 4D image features "
                        f"(N, C, H, W), got shape={tuple(h.shape)}"
                    )
                pool_stride = (
                    spec.pool_stride
                    if spec.pool_stride is not None
                    else spec.pool_kernel_size
                )
                pool_mode = spec.pool_mode.strip().lower()
                if pool_mode == "avg":
                    h = torch.nn.functional.avg_pool2d(
                        h,
                        kernel_size=spec.pool_kernel_size,
                        stride=pool_stride,
                        padding=spec.pool_padding,
                    )
                else:
                    h = torch.nn.functional.max_pool2d(
                        h,
                        kernel_size=spec.pool_kernel_size,
                        stride=pool_stride,
                        padding=spec.pool_padding,
                    )

            elif kind == "l2norm":
                if h.ndim != 4:
                    raise ValueError(
                        f"Layer {i} (l2norm) expects 4D image features "
                        f"(N, C, H, W), got shape={tuple(h.shape)}"
                    )
                epsilon = 1e-8
                norm = torch.sqrt(torch.sum(h**2, dim=1, keepdim=True) + epsilon)
                h = h / norm

            elif kind == "flatten":
                if h.ndim != 4:
                    raise ValueError(
                        f"Layer {i} (flatten) expects 4D image features (N, C, H, W), "
                        f"got shape={tuple(h.shape)}"
                    )
                h = h.flatten(1)

            elif kind == "fully_connected":
                if h.ndim != 2:
                    raise ValueError(
                        f"Layer {i} (fully_connected) expects 2D vector features "
                        f"(N, D), got shape={tuple(h.shape)}"
                    )
                w_fc = self.random_fc_weights[i]
                if h.shape[1] != w_fc.shape[0]:
                    raise ValueError(
                        f"Layer {i} (fully_connected) expected {w_fc.shape[0]} "
                        f"input features, got {h.shape[1]}"
                    )
                v = self.fc_eigenvector_projections[i]
                sigma = ACTIVATIONS[self.activation_names[i]]
                z = sigma(h @ w_fc)
                h = z @ v

            else:
                raise ValueError(f"Unsupported layer kind: {kind!r}")

        h = h.flatten(1) if h.ndim > 2 else h
        return h @ self.ridge_weight + self.ridge_bias
