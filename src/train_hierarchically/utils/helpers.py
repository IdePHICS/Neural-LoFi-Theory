"""General-purpose helper functions."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

_SQRT2 = math.sqrt(2.0)
_PI_OVER_4 = math.pi / 4.0

# ---------------------------------------------------------------------------
# Activation registries
# ---------------------------------------------------------------------------


class _Erf(nn.Module):
    """Element-wise error function (no built-in ``nn.Erf`` in PyTorch)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.erf(x)


def _step(x: Tensor) -> Tensor:
    return (x > 0).to(x.dtype)


class _Step(nn.Module):
    """Heaviside step: 1 if x>0 else 0.  Gradient-free; ridge-spectral only."""

    def forward(self, x: Tensor) -> Tensor:
        return (x > 0).to(x.dtype)


def _capped_relu(x: Tensor) -> Tensor:
    return x.clamp(0.0, 1.0)


class _CappedReLU(nn.Module):
    """ReLU saturated at 1: max(0, min(x, 1))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(0.0, 1.0)


def _relu_normalized(x: Tensor) -> Tensor:
    # Spherical arc-cosine kernel feature map: ReLU followed by per-sample
    # L2 normalization on the last dim. Assumes 2D input (N, P); behavior on
    # higher-rank tensors (e.g. conv outputs) is not what the kernel intends.
    z = F.relu(x)
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class _ReLUNormalized(nn.Module):
    """ReLU then per-sample L2 normalize on the last dim (FFN-only)."""

    def forward(self, x: Tensor) -> Tensor:
        return _relu_normalized(x)


def _poly2(x: Tensor) -> Tensor:
    return x * x


class _Poly2(nn.Module):
    """Polynomial activation x²."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x


def _poly3(x: Tensor) -> Tensor:
    return x * x * x


class _Poly3(nn.Module):
    """Polynomial activation x³."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x * x


class _Cos(nn.Module):
    """Cosine activation; with sin/cos features, recovers the RBF kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x)


class _Sin(nn.Module):
    """Sine activation; with sin/cos features, recovers the RBF kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def _rbf(x: Tensor) -> Tensor:
    # Han et al. single-activation RBF: √2 sin(√2 t + π/4). Uses the identity
    # √2 sin(θ + π/4) = sin θ + cos θ so one activation reproduces the
    # (sin, cos) RFF construction with a deterministic phase shift.
    return _SQRT2 * torch.sin(_SQRT2 * x + _PI_OVER_4)


class _RBF(nn.Module):
    """Han et al. single-activation RBF: √2 sin(√2 t + π/4)."""

    def forward(self, x: Tensor) -> Tensor:
        return _SQRT2 * torch.sin(_SQRT2 * x + _PI_OVER_4)


# NOTE: ``torch.erf`` is used because ``F.erf`` does not exist in PyTorch.
# ``F.gelu`` defaults to the exact (erf-based) GELU, which is the form with
# a closed-form NNGP kernel.
ACTIVATIONS: dict[str, Callable[..., Tensor]] = {
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "relu": F.relu,
    "relu6": F.relu6,
    "erf": torch.erf,
    "gelu": F.gelu,
    "step": _step,
    "capped_relu": _capped_relu,
    "relu_normalized": _relu_normalized,
    "poly2": _poly2,
    "poly3": _poly3,
    "cos": torch.cos,
    "sin": torch.sin,
    "rbf": _rbf,
    "identity": lambda x: x,
}

ACTIVATION_MODULES: dict[str, nn.Module] = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "erf": _Erf(),
    "gelu": nn.GELU(),
    "step": _Step(),
    "capped_relu": _CappedReLU(),
    "relu_normalized": _ReLUNormalized(),
    "poly2": _Poly2(),
    "poly3": _Poly3(),
    "cos": _Cos(),
    "sin": _Sin(),
    "rbf": _RBF(),
    "identity": nn.Identity(),
}


def resolve_device(device: str) -> str:
    """Resolve a device string, expanding ``"auto"`` to the best available."""
    logger = logging.getLogger(__name__)
    if device == "auto":
        if torch.backends.mps.is_available():
            logger.info("Device resolved to 'mps'")
            return "mps"
        if torch.cuda.is_available():
            logger.info("Device resolved to 'cuda'")
            return "cuda"
        logger.info("Device resolved to 'cpu'")
        return "cpu"
    logger.info("Using device '%s'", device)
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and stdlib."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
