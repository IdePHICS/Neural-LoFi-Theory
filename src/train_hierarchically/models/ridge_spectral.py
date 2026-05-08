"""Ridge spectral model: random features + eigenvector projection + ridge readout.

Random-feature layers apply::

    z = σ(h @ W / c) / √p

where *W* is i.i.d. N(0, 1), *c = sqrt(E[‖h‖²])* is the RMS row-norm of
the layer input estimated from training data and stored as
``input_norm_coefficients[l]``, and the ``1/√p`` factor keeps
``‖z_μ‖² → κ_σ`` (O(1)) in the ``p → ∞`` limit.  The ``/ c`` step
centres the pre-activation magnitudes around O(1) so that sigmoid /
erf / tanh operate in their informative regime regardless of depth.
"""

from __future__ import annotations

import math
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


@dataclass(slots=True)
class LayerSpec:
    """Specification for one layer in the ridge spectral model.

    For ``kind="random_features"`` (default), the layer computes
    ``h = σ(h_prev @ W) @ V`` where *W* is a random projection and *V*
    contains the top-K eigenvectors of the signed covariance.

    For ``kind="scattering"``, the layer applies a fixed wavelet scattering
    transform (via kymatio) to the input images and flattens the result.
    Eigenreduction then selects the top-K directions in the flattened
    scattering feature space.  Must be layer index 0.

    Parameters
    ----------
    p : int
        Random expansion width (number of random features).
        Ignored for scattering layers.
    k : int
        Number of top eigenvectors to keep after projection.
    activation : str
        Activation function name. Options include ``"sigmoid"``, ``"tanh"``,
        ``"relu"``, ``"relu6"``, ``"erf"``, ``"gelu"``, ``"step"``,
        ``"capped_relu"``, ``"relu_normalized"``, ``"poly2"`` (x²),
        ``"poly3"`` (x³), ``"cos"``, ``"sin"``, ``"rbf"``, ``"identity"``.  Ignored for
        scattering layers.  ``"relu_normalized"`` applies ReLU then
        per-sample L2 normalization (spherical arc-cosine kernel feature
        map).  ``"rbf"`` is the Han et al. single-activation RFF
        ``√2 sin(√2 t + π/4)`` whose NNGP kernel is the RBF kernel
        ``exp(-‖x-y‖²)``.
    do_eigenreduction : bool
        Whether to apply eigenvector projection.
    kind : str
        ``"random_features"`` (default) or ``"scattering"``.
    scattering_j : int
        Number of wavelet scales (J parameter).  Only used when
        ``kind="scattering"``.
    scattering_l : int
        Number of wavelet orientations (L parameter).  Only used when
        ``kind="scattering"``.
    scattering_max_order : int
        Maximum scattering path order (1 or 2).  Only used when
        ``kind="scattering"``.
    """

    p: int
    k: int
    activation: str = "sigmoid"
    do_eigenreduction: bool = True
    kind: str = "random_features"
    scattering_j: int = 2
    scattering_l: int = 8
    scattering_max_order: int = 2



class RidgeSpectralModel(nn.Module):
    """Depth-L random features + eigenvector projection + ridge readout.

    Architecture per layer *l* (l = 1 … L)::

        h_l = σ_l(h_{l-1} @ W_l) @ V_l

    When the first layer is a scattering layer, *W_l* and *σ_l* are
    replaced by a fixed wavelet scattering transform followed by
    flattening.

    Final output::

        output = h_L @ ridge_weight + ridge_bias

    where ``W_l`` is a frozen random projection, ``V_l`` contains the top-K_l
    eigenvectors of the signed covariance matrix (set during training), and
    ``ridge_weight`` / ``ridge_bias`` come from sklearn ``RidgeCV``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of flattened input.
    layer_specs : list[LayerSpec]
        Per-layer configuration (p, k, activation).
    seed : int
        Random seed for generating projection matrices.
    input_shape : tuple[int, int, int] | None
        Spatial input shape ``(C, H, W)``.  Required when the first layer
        is ``kind="scattering"``.
    """

    def __init__(
        self,
        input_dim: int,
        layer_specs: list[LayerSpec],
        seed: int = 0,
        *args: Any,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.layer_specs = layer_specs
        self.n_layers = len(layer_specs)
        self.input_shape = input_shape

        # Random projections W_l (frozen), eigenvector projections V_l,
        # and per-layer input normalization coefficients c_l = sqrt(E[‖h‖²])
        self.random_projections = nn.ParameterList()
        self.eigenvector_projections = nn.ParameterList()
        self.input_norm_coefficients = nn.ParameterList()
        self.activation_names: list[str] = []

        # Scattering modules (lazy-loaded, only when needed)
        self.scattering_modules = nn.ModuleList()
        self._scattering_layer_map: dict[int, int] = {}

        gen = torch.Generator()
        gen.manual_seed(seed)

        prev_dim = input_dim
        for i, spec in enumerate(layer_specs):
            if spec.kind == "scattering":
                prev_dim = self._init_scattering_layer(i, spec)
            elif spec.kind == "random_features":
                prev_dim = self._init_random_features_layer(
                    spec, prev_dim, gen
                )
            else:
                raise ValueError(f"Unsupported layer kind: {spec.kind!r}")

        # Ridge readout: weight and bias
        # For 0-layer models, readout operates directly on input_dim
        readout_dim = layer_specs[-1].k if layer_specs else input_dim
        self.ridge_weight = nn.Parameter(
            torch.zeros(readout_dim), requires_grad=False
        )
        self.ridge_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _init_scattering_layer(
        self, layer_idx: int, spec: LayerSpec
    ) -> int:
        """Initialize a scattering layer.  Returns the output dimension."""
        if layer_idx != 0:
            raise ValueError(
                "Scattering layer must be the first layer (index 0), "
                f"got index {layer_idx}"
            )
        if self.input_shape is None:
            raise ValueError(
                "input_shape=(C, H, W) is required when using a "
                "scattering layer"
            )
        if spec.do_eigenreduction and spec.k <= 0:
            raise ValueError(
                "Scattering layer with eigenreduction requires k > 0"
            )

        c_in, h, w = self.input_shape
        scat = Scattering2D(
            J=spec.scattering_j,
            shape=(h, w),
            L=spec.scattering_l,
            max_order=spec.scattering_max_order,
        )
        scat_idx = len(self.scattering_modules)
        self.scattering_modules.append(scat)
        self._scattering_layer_map[layer_idx] = scat_idx

        n_paths = n_scattering_coefficients(
            spec.scattering_j, spec.scattering_l, spec.scattering_max_order
        )
        h_out, w_out = scattering_output_spatial(h, w, spec.scattering_j)
        out_dim = c_in * n_paths * h_out * w_out

        # Empty placeholders for index alignment
        self.random_projections.append(
            nn.Parameter(torch.empty(0), requires_grad=False)
        )
        self.input_norm_coefficients.append(
            nn.Parameter(torch.ones(1), requires_grad=False)
        )

        # Eigenvector projection
        if spec.do_eigenreduction:
            v = torch.zeros(out_dim, spec.k)
            self.eigenvector_projections.append(
                nn.Parameter(v, requires_grad=False)
            )
            prev_dim = spec.k
        else:
            v = torch.eye(out_dim)
            self.eigenvector_projections.append(
                nn.Parameter(v, requires_grad=False)
            )
            prev_dim = out_dim

        self.activation_names.append("")
        return prev_dim

    def _init_random_features_layer(
        self, spec: LayerSpec, prev_dim: int, gen: torch.Generator
    ) -> int:
        """Initialize a random-features layer.  Returns the output dimension."""
        # W_l: (prev_dim, P_l) — i.i.d. N(0, 1), frozen
        w = torch.randn(prev_dim, spec.p, generator=gen)
        self.random_projections.append(nn.Parameter(w, requires_grad=False))

        # V_l: (P_l, K_l) — placeholder, set during fitting
        v = torch.zeros(spec.p, spec.k)
        self.eigenvector_projections.append(
            nn.Parameter(v, requires_grad=False)
            if spec.do_eigenreduction
            else nn.Parameter(torch.eye(spec.p, spec.p), requires_grad=False)
        )

        # c_l: input normalization coefficient, set during training
        self.input_norm_coefficients.append(
            nn.Parameter(torch.ones(1), requires_grad=False)
        )

        self.activation_names.append(spec.activation)
        return spec.k

    def _apply_scattering(self, x: Tensor, layer_idx: int) -> Tensor:
        """Apply scattering transform and flatten to (N, D').

        Parameters
        ----------
        x : Tensor
            Input of shape ``(N, C, H, W)`` or ``(N, D)`` (flat).
        layer_idx : int
            Layer index in the model.

        Returns
        -------
        Tensor
            Flattened scattering features of shape ``(N, D')``.
        """
        if x.ndim == 2 and self.input_shape is not None:
            x = x.reshape(x.shape[0], *self.input_shape)

        scat_idx = self._scattering_layer_map[layer_idx]
        scat = self.scattering_modules[scat_idx]
        z = scat(x)

        # Kymatio output may be 5D: (N, C, paths, H', W')
        if z.ndim == 5:
            n, c, paths, h_out, w_out = z.shape
            z = z.reshape(n, c * paths, h_out, w_out)

        # Flatten spatial: (N, C*paths, H', W') -> (N, D')
        return z.reshape(z.shape[0], -1)

    def forward_to_layer(
        self, x: Tensor, layer: int, *, project: bool = True
    ) -> Tensor:
        """Forward through layers 0..layer, return the hidden representation.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, input_dim)`` or ``(batch, C, H, W)``
            when the first layer is scattering.
        layer : int
            Target layer index (0-based).
        project : bool
            If ``True`` (default), return post-projection ``h_l = Z_l @ V_l``
            (K-dimensional). If ``False``, return pre-projection features.

        Returns
        -------
        Tensor
            Hidden representation at the target layer.
        """
        h = x
        for i in range(layer + 1):
            spec = self.layer_specs[i]
            if spec.kind == "scattering":
                z = self._apply_scattering(h, i)
            else:
                w = self.random_projections[i]
                sigma = ACTIVATIONS[self.activation_names[i]]
                c = self.input_norm_coefficients[i]
                z = sigma(h @ w / c) / math.sqrt(w.shape[1])

            if i < layer:
                h = z @ self.eigenvector_projections[i]
            else:
                h = z @ self.eigenvector_projections[i] if project else z
        return h

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass through all layers + ridge readout.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, input_dim)`` or ``(batch, C, H, W)``
            when the first layer is scattering.

        Returns
        -------
        Tensor
            Predictions of shape ``(batch,)``.
        """
        h = x
        for i in range(self.n_layers):
            spec = self.layer_specs[i]
            if spec.kind == "scattering":
                z = self._apply_scattering(h, i)
                h = z @ self.eigenvector_projections[i]
            else:
                w = self.random_projections[i]
                v = self.eigenvector_projections[i]
                sigma = ACTIVATIONS[self.activation_names[i]]
                c = self.input_norm_coefficients[i]
                h = (sigma(h @ w / c) / math.sqrt(w.shape[1])) @ v
        return h @ self.ridge_weight + self.ridge_bias
