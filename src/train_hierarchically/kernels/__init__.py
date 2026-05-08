"""NNGP kernel registry.

Kernels implement :class:`KernelBase` (in :mod:`.base`) and are discovered
through the ``@register_kernel("name")`` decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .base import KernelBase

_REGISTRY: dict[str, type[KernelBase]] = {}

_K = TypeVar("_K", bound=KernelBase)


def register_kernel(name: str) -> Callable[[type[_K]], type[_K]]:
    """Decorator registering a :class:`KernelBase` subclass under *name*."""

    def _wrap(cls: type[_K]) -> type[_K]:
        key = name.lower().strip()
        if key in _REGISTRY:
            raise KeyError(f"Kernel already registered for {key!r}")
        _REGISTRY[key] = cls
        return cls

    return _wrap


def get_kernel(name: str, **kwargs: Any) -> KernelBase:
    """Instantiate a registered kernel by name.

    Parameters
    ----------
    name : str
        Registered kernel name (e.g. ``"sigmoid"``).
    **kwargs
        Keyword arguments forwarded to the kernel constructor (e.g.
        ``sigma_w``, ``sigma_b``, ``n_quad``).
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown kernel {name!r}.  Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def available_kernels() -> list[str]:
    """Sorted list of registered kernel names."""
    return sorted(_REGISTRY.keys())


# Import bundled kernels so their @register_kernel decorators fire.
from . import relu  # noqa: E402, F401
from . import sigmoid  # noqa: E402, F401

__all__ = ["KernelBase", "available_kernels", "get_kernel", "register_kernel"]
