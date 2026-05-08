from __future__ import annotations

from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_fashion_mnist_transform


@register("fashion-mnist")
def build_fashion_mnist(config: DatasetConfig) -> Dataset:
    tfm = config.transform or default_fashion_mnist_transform(flatten=config.flatten)

    if config.split == "train":
        train = True
    elif config.split in {"val", "test"}:
        train = False
    else:
        raise ValueError(f"Unsupported split for FashionMNIST: {config.split!r}")

    ds = datasets.FashionMNIST(
        root=str(config.root_path),
        train=train,
        download=True,
        transform=tfm,
    )

    return ds
