from __future__ import annotations

import torch
from torchvision import transforms

# Per-dataset empirical channel statistics (computed on training sets).
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

_FASHION_MNIST_MEAN = (0.2860,)
_FASHION_MNIST_STD = (0.3530,)

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_PCAM_MEAN = (0.7008, 0.5384, 0.6916)
_PCAM_STD = (0.2350, 0.2774, 0.2129)

_CELEBA_MEAN = (0.5063, 0.4258, 0.3832)
_CELEBA_STD = (0.3107, 0.2904, 0.2897)


def _maybe_flatten(steps: list, *, flatten: bool) -> transforms.Compose:
    """Append a Flatten step when *flatten* is True, then compose."""
    if flatten:
        steps.append(torch.nn.Flatten(start_dim=0))
    return transforms.Compose(steps)


def default_mnist_transform(*, flatten: bool = False) -> transforms.Compose:
    """MNIST: 28×28 grayscale, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ],
        flatten=flatten,
    )


def default_fashion_mnist_transform(
    *, flatten: bool = False
) -> transforms.Compose:
    """Fashion-MNIST: 28×28 grayscale, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STD),
        ],
        flatten=flatten,
    )


def default_cifar10_transform(*, flatten: bool = False) -> transforms.Compose:
    """CIFAR-10: 32×32 RGB, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ],
        flatten=flatten,
    )


def default_pcam_transform(*, flatten: bool = False) -> transforms.Compose:
    """PCAM: 96×96 RGB, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_PCAM_MEAN, _PCAM_STD),
        ],
        flatten=flatten,
    )


def default_celeba_transform(*, flatten: bool = False) -> transforms.Compose:
    """CelebA: center-cropped to 178×178, resized to 64×64 RGB, normalised."""
    return _maybe_flatten(
        [
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(_CELEBA_MEAN, _CELEBA_STD),
        ],
        flatten=flatten,
    )
