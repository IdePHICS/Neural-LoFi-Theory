from __future__ import annotations

from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_cifar10_transform


@register("cifar10")
def build_cifar10(config: DatasetConfig) -> Dataset:
    tfm = config.transform or default_cifar10_transform(flatten=config.flatten)

    if config.split == "train":
        train = True
    elif config.split in {"val", "test"}:
        train = False
    else:
        raise ValueError(f"Unsupported split for CIFAR10: {config.split!r}")

    ds = datasets.CIFAR10(
        root=str(config.root_path),
        train=train,
        download=True,
        transform=tfm,
    )

    return ds
