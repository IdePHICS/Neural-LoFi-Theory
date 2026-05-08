"""Kernel-trick variant of :class:`RidgeSpectralModel`.

In the infinite-width (NNGP) limit the random-feature matrix
``Phi in R^{n x p}`` becomes implicit; only the Gram matrix
``G = K(X, X) in R^{n x n}`` is observed.  Eigenvectors of the signed
covariance that lie in the row span of ``Phi`` can be written as
``H = Phi^T A``; the corresponding n×n whitened eigenproblem is
``G^{1/2} Y G^{1/2} B = B Lambda``, with dual coefficients
``A = G^{-1/2} B`` satisfying ``A^T G A = I_m`` and projected features
``F_l = G_l A_l`` (equivalently ``G^{1/2} B``).

Per-layer features are therefore ``F_l`` of shape ``(n, m_l)`` — rows
index samples, columns index kept eigendirections.  This module stores
the per-layer ``A`` matrices, the cached training features needed at
test time for layers ``l >= 2``, and the ridge readout so that
:meth:`KernelSpectralModel.forward` is a self-contained predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..kernels import KernelBase, get_kernel


@dataclass(slots=True)
class KernelLayerSpec:
    """Per-layer kernel configuration.

    Parameters
    ----------
    kernel : str
        Registered kernel name (e.g. ``"sigmoid"``).
    m : int
        Number of top eigenvectors to keep when
        ``do_eigenreduction=True``; ignored otherwise.
    sigma_w : float
        Weight scale (``W ~ N(0, sigma_w^2 / d)``).
    sigma_b : float
        Bias standard deviation.
    n_quad : int
        Number of 1-D Gauss-Hermite nodes for kernels that need
        quadrature.  Ignored by kernels with closed-form expressions.
    do_eigenreduction : bool
        When ``True`` the layer performs the signed-covariance
        eigenreduction on the Gram matrix (NLoFi-NNGP path, analogous to
        :class:`LayerSpec.do_eigenreduction`).  When ``False`` the layer
        propagates the previous layer's kernel forward via the standard
        deep-NNGP recursion ``K_{l+1} = f(sigma_w^2 K_l + sigma_b^2, …)``
        without any ``1/m`` scaling.
    """

    kernel: str
    m: int
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    n_quad: int = 30
    do_eigenreduction: bool = True

    def build(self) -> KernelBase:
        """Instantiate the configured kernel."""
        return get_kernel(
            self.kernel,
            sigma_w=self.sigma_w,
            sigma_b=self.sigma_b,
            n_quad=self.n_quad,
        )


class KernelSpectralModel(nn.Module):
    """Deep NNGP feature extractor + ridge readout.

    State stored after fitting:

    - ``x_train`` : ``(n, d)`` raw training inputs; the layer-1 kernel
      evaluates against them at test time.
    - ``alphas`` : list of ``(n, m_l)`` kernel-trick dual coefficients
      ``A_l = G_l^{-1/2} B_l``.
    - ``train_features`` : list of ``(n, m_l)`` per-layer train features
      ``F_l = G_l A_l`` used as the "kernel side" for layers ``l >= 2``
      at test time.
    - ``ridge_weight`` / ``ridge_bias`` : final readout (filled by
      :func:`fit_ridge_readout`).

    Parameters
    ----------
    layer_specs : list[KernelLayerSpec]
        Per-layer kernel configurations.
    """

    def __init__(self, layer_specs: list[KernelLayerSpec]) -> None:
        super().__init__()
        self.layer_specs = layer_specs
        self.n_layers = len(layer_specs)
        self.kernels: list[KernelBase] = [s.build() for s in layer_specs]
        self.all_no_eigenreduction = (
            len(layer_specs) > 0
            and not any(s.do_eigenreduction for s in layer_specs)
        )

        # State buffers — populated by the trainer via ``set_state``.
        self.register_buffer("x_train", torch.empty(0), persistent=False)
        self.alphas = nn.ParameterList()
        self.train_features = nn.ParameterList()

        # Ridge-readout dimension depends on the last layer:
        #   - do_eigenreduction=True  -> m_L (feature-space ridge),
        #   - do_eigenreduction=False -> n_train (kernel-space ridge).
        # The trainer overwrites ``ridge_weight`` / ``ridge_bias`` at the
        # end of :func:`fit_ridge_readout`; 0 is a safe placeholder.
        self.ridge_weight = nn.Parameter(torch.zeros(0), requires_grad=False)
        self.ridge_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------
    # Trainer-side population
    # ------------------------------------------------------------------

    def set_state(
        self,
        x_train: Tensor,
        alphas: list[Tensor] | None = None,
        train_features: list[Tensor] | None = None,
    ) -> None:
        """Install the fitted state in-place.

        Parameters
        ----------
        x_train : Tensor
            Raw training inputs ``(n, d)``; required in all modes because
            the layer-1 kernel evaluates against them at test time.
        alphas : list[Tensor] | None
            Per-layer eigenvectors of shape ``(n, m_l)``.  Only used for
            layers with ``do_eigenreduction=True``; may be omitted when
            no layer performs eigenreduction.
        train_features : list[Tensor] | None
            Per-layer features ``F_l_train`` of shape ``(n, m_l)``, used
            as the "kernel side" for layers ``l >= 2`` at test time.
            Same optional semantics as ``alphas``.
        """
        self.x_train = x_train.detach()

        if alphas is not None:
            if len(alphas) != self.n_layers:
                raise ValueError(
                    f"Expected {self.n_layers} alphas, got {len(alphas)}"
                )
            self.alphas = nn.ParameterList(
                [nn.Parameter(a.detach(), requires_grad=False) for a in alphas]
            )
        if train_features is not None:
            if len(train_features) != self.n_layers:
                raise ValueError(
                    f"Expected {self.n_layers} train_features, "
                    f"got {len(train_features)}"
                )
            self.train_features = nn.ParameterList(
                [
                    nn.Parameter(f.detach(), requires_grad=False)
                    for f in train_features
                ]
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def features_to_layer(self, x_test: Tensor, layer: int) -> Tensor:
        """Return ``F_layer(x_test)`` of shape ``(n_test, m_layer)``.

        ``layer`` is 0-based; only the Gram matrices needed for the
        requested layer are computed.  Requires eigenreduction to have
        been performed at every layer ``0..layer`` (otherwise there are
        no ``alphas`` to project with).
        """
        if layer >= self.n_layers or layer < 0:
            raise IndexError(f"layer must be in [0, {self.n_layers - 1}], got {layer}")
        if self.x_train.numel() == 0:
            raise RuntimeError("Model has no fitted state; call set_state first")
        if not self.alphas or len(self.alphas) <= layer:
            raise RuntimeError(
                "features_to_layer requires eigenreduction state; "
                "this model was fit with do_eigenreduction=False"
            )

        # Layer 0: kernel against raw inputs.  k(x_test, X_train) A = F.
        g_cross = self.kernels[0].gram(self.x_train, x_test)  # (n, n_test)
        features = g_cross.T @ self.alphas[0]  # (n_test, m_0)

        for layer_idx in range(1, layer + 1):
            train_in = self.train_features[layer_idx - 1]  # (n, m_{l-1})
            test_in = features  # (n_test, m_{l-1})
            g_cross = self.kernels[layer_idx].gram(train_in, test_in)  # (n, n_test)
            features = g_cross.T @ self.alphas[layer_idx]  # (n_test, m_l)

        return features

    @torch.no_grad()
    def forward(self, x_test: Tensor, **_: Any) -> Tensor:
        """Predict ``y_pred(x_test)`` of shape ``(n_test,)``."""
        if self.n_layers == 0:
            raise RuntimeError("KernelSpectralModel has no layers")
        if self.x_train.numel() == 0:
            raise RuntimeError("Model has no fitted state; call set_state first")

        if self.all_no_eigenreduction:
            from ..training.deep_nngp import deep_nngp_kernel

            _, k_cross = deep_nngp_kernel(
                self.kernels, self.x_train, x_test, n_layers=self.n_layers
            )
            assert k_cross is not None
            return k_cross.T @ self.ridge_weight + self.ridge_bias

        return self._forward_mixed(x_test)

    @torch.no_grad()
    def _forward_mixed(self, x_test: Tensor) -> Tensor:
        """Replay the layer-by-layer computation for mixed eigen/no-eigen."""
        from ..training.deep_nngp import _apply_kernel_rowchunked, _default_row_chunk

        # Track state: either features (f_train, f_test) or kernel
        # (k_train, k_cross, k_test_diag).
        # Start with raw inputs.
        f_train: Tensor | None = None
        f_test: Tensor | None = None
        k_train: Tensor | None = None
        k_cross: Tensor | None = None
        k_test_diag: Tensor | None = None
        is_features = False

        for layer_idx in range(self.n_layers):
            spec = self.layer_specs[layer_idx]
            kern = self.kernels[layer_idx]
            has_alpha = (
                layer_idx < len(self.alphas)
                and self.alphas[layer_idx].numel() > 0
            )

            # Step 1: compute this layer's cross Gram (and train Gram
            # if a no-eigen layer needs the kernel chain).
            if layer_idx == 0 and not is_features and k_train is None:
                # Raw-input case.
                g_cross = kern.gram(self.x_train, x_test)
                if not spec.do_eigenreduction:
                    g_train = kern.gram(self.x_train)
                    d = x_test.shape[1]
                    sc = kern.sigma_w**2 / d
                    bi = kern.sigma_b**2
                    q_te = sc * (x_test * x_test).sum(dim=1) + bi
                    k_test_diag = kern.kernel_from_qs(q_te, q_te, q_te)
            elif is_features:
                assert f_train is not None and f_test is not None
                # f_train: (n, m_prev); f_test: (n_test, m_prev).
                g_cross = kern.gram(f_train, f_test)
                if not spec.do_eigenreduction:
                    g_train = kern.gram(f_train)
                    m = f_test.shape[1]
                    sc = kern.sigma_w**2 / m
                    bi = kern.sigma_b**2
                    q_te = sc * (f_test * f_test).sum(dim=1) + bi
                    k_test_diag = kern.kernel_from_qs(q_te, q_te, q_te)
            else:
                # Kernel-state case: NNGP composition.
                assert k_train is not None and k_cross is not None
                sw2 = kern.sigma_w**2
                sb2 = kern.sigma_b**2
                chunk = _default_row_chunk(
                    kern, k_train.shape[0], k_train.shape[1], k_train.dtype,
                )
                kd = k_train.diagonal()
                qd = sw2 * kd + sb2
                qt = sw2 * k_train + sb2
                g_train = _apply_kernel_rowchunked(
                    kern, qd.unsqueeze(1), qd.unsqueeze(0), qt,
                    row_chunk=chunk,
                )
                g_train = 0.5 * (g_train + g_train.T)
                assert k_test_diag is not None
                qtd = sw2 * k_test_diag + sb2
                qc = sw2 * k_cross + sb2
                g_cross = _apply_kernel_rowchunked(
                    kern, qd.unsqueeze(1), qtd.unsqueeze(0), qc,
                    row_chunk=chunk,
                )
                k_test_diag = kern.kernel_from_qs(qtd, qtd, qtd)

            # Step 2: apply the layer.
            if spec.do_eigenreduction:
                assert has_alpha
                alpha = self.alphas[layer_idx]
                # g_cross is (n_train, n_test); f_test_row = k(x, X_train) A.
                f_test = g_cross.T @ alpha  # (n_test, m)
                f_train = self.train_features[layer_idx]  # (n, m)
                is_features = True
                k_train = k_cross = k_test_diag = None
            else:
                k_train = g_train  # type: ignore[possibly-undefined]
                k_cross = g_cross
                is_features = False
                f_train = f_test = None

        # Final readout.
        if self.layer_specs[-1].do_eigenreduction:
            assert f_test is not None
            return f_test @ self.ridge_weight + self.ridge_bias
        assert k_cross is not None
        return k_cross.T @ self.ridge_weight + self.ridge_bias
