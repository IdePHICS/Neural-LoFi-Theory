"""Backprop-trained CNN built from :class:`CNNLayerSpec`.

This model mirrors the mixed-layer architecture used by
``CNNRidgeSpectralModel`` (conv/pooling/l2norm/flatten/fully_connected),
but uses standard trainable layers end-to-end.
"""

from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn

from ..utils.helpers import ACTIVATION_MODULES
from .cnn_ridge_spectral import CNNLayerSpec


class ChannelL2Norm(nn.Module):
    """Per-location L2 normalization across channels for 4D feature maps."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ChannelL2Norm expects input shape (N, C, H, W), "
                f"got {tuple(x.shape)}"
            )
        denom = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        return x / denom.clamp_min(self.eps)


class CNNModel(nn.Module):
    """Trainable CNN with mixed layer kinds from ``CNNLayerSpec``.

    Channel dimensions for ``conv`` and ``fully_connected`` follow:
    ``backprop_channels`` > (``k`` when ``do_eigenreduction=True``) > ``p``.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        layer_specs: list[CNNLayerSpec],
        output_dim: int = 1,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layer_specs = layer_specs
        self.layers = nn.ModuleList()

        c_in = in_channels
        h, w = spatial_size
        feature_dim = c_in * h * w
        is_flat = False

        for i, spec in enumerate(layer_specs):
            kind = spec.kind

            if kind == "conv":
                if is_flat:
                    raise ValueError(
                        "conv layer cannot come after flatten/fully_connected"
                    )
                c_out = _compute_backprop_channels(spec)
                if c_out <= 0:
                    raise ValueError(
                        f"Layer {i} (conv) requires output channels > 0; "
                        "set backprop_channels, k (with do_eigenreduction=True), or p"
                    )

                self.layers.append(
                    nn.ModuleDict(
                        {
                            "conv": nn.Conv2d(
                                c_in,
                                c_out,
                                kernel_size=spec.kernel_size,
                                stride=spec.stride,
                                padding=spec.padding,
                                bias=False,
                            ),
                            "activation": _make_activation(spec.activation),
                        }
                    )
                )

                h = (h + 2 * spec.padding - spec.kernel_size) // spec.stride + 1
                w = (w + 2 * spec.padding - spec.kernel_size) // spec.stride + 1
                if h <= 0 or w <= 0:
                    raise ValueError(
                        f"Layer {i} (conv) collapses spatial size to {(h, w)}"
                    )

                c_in = c_out
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
                pool_mode = spec.pool_mode.strip().lower()
                if pool_mode == "avg":
                    self.layers.append(
                        nn.AvgPool2d(
                            kernel_size=spec.pool_kernel_size,
                            stride=pool_stride,
                            padding=spec.pool_padding,
                        )
                    )
                else:
                    self.layers.append(
                        nn.MaxPool2d(
                            kernel_size=spec.pool_kernel_size,
                            stride=pool_stride,
                            padding=spec.pool_padding,
                        )
                    )

                h = (
                    h + 2 * spec.pool_padding - spec.pool_kernel_size
                ) // pool_stride + 1
                w = (
                    w + 2 * spec.pool_padding - spec.pool_kernel_size
                ) // pool_stride + 1
                if h <= 0 or w <= 0:
                    raise ValueError(
                        f"Layer {i} (pooling) collapses spatial size to {(h, w)}"
                    )
                feature_dim = c_in * h * w

            elif kind == "l2norm":
                if is_flat:
                    raise ValueError(
                        "l2norm layer cannot come after flatten/fully_connected"
                    )
                self.layers.append(ChannelL2Norm())

            elif kind == "flatten":
                if is_flat:
                    raise ValueError("Multiple flatten layers are not supported")
                self.layers.append(nn.Flatten())
                is_flat = True
                feature_dim = c_in * h * w

            elif kind == "fully_connected":
                if not is_flat:
                    raise ValueError(
                        "fully_connected layer expects flattened features; "
                        "insert a flatten layer before it"
                    )
                out_dim = _compute_backprop_channels(spec)
                if out_dim <= 0:
                    raise ValueError(
                        f"Layer {i} (fully_connected) requires output dim > 0; "
                        "set backprop_channels, k (with do_eigenreduction=True), or p"
                    )
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "linear": nn.Linear(feature_dim, out_dim, bias=False),
                            "activation": _make_activation(spec.activation),
                        }
                    )
                )
                feature_dim = out_dim

            else:
                raise ValueError(f"Unsupported layer kind: {kind!r}")

        if not is_flat:
            raise ValueError(
                "Layer specs must include a flatten layer before "
                "fully_connected/readout."
            )

        self.classifier = nn.Linear(feature_dim, output_dim, bias=classifier_bias)

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Return feature tensors after each configured layer."""
        h = x
        outputs: list[Tensor] = []

        for spec, layer in zip(self.layer_specs, self.layers, strict=True):
            kind = spec.kind

            if kind == "conv":
                block = cast(nn.ModuleDict, layer)
                conv_layer = cast(nn.Conv2d, block["conv"])
                activation = block["activation"]
                h = conv_layer(h)
                h = activation(h)

            elif kind in {"pooling", "l2norm", "flatten"}:
                h = layer(h)

            elif kind == "fully_connected":
                block = cast(nn.ModuleDict, layer)
                linear = cast(nn.Linear, block["linear"])
                activation = block["activation"]
                h = linear(h)
                h = activation(h)

            else:
                raise ValueError(f"Unsupported layer kind: {kind!r}")

            outputs.append(h)

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.forward_features(x)
        h = outputs[-1]
        if h.ndim > 2:
            h = h.flatten(1)
        return self.classifier(h)


def _make_activation(name: str) -> nn.Module:
    base = ACTIVATION_MODULES.get(name)
    if base is None:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Available: {sorted(ACTIVATION_MODULES.keys())}"
        )
    return copy.deepcopy(base)


def _compute_backprop_channels(spec: CNNLayerSpec) -> int:
    """Output-size priority for backprop model layers."""
    if spec.backprop_channels > 0:
        return spec.backprop_channels
    if spec.do_eigenreduction and spec.k > 0:
        return spec.k
    return spec.p
