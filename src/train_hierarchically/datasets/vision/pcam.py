from __future__ import annotations

from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_pcam_transform


@register("pcam")
def build_pcam(config: DatasetConfig) -> Dataset:
    tfm = config.transform or default_pcam_transform(flatten=config.flatten)

    if config.split == "train":
        split = "train"
    elif config.split == "val":
        split = "val"
    elif config.split == "test":
        split = "test"
    else:
        raise ValueError(f"Unsupported split for PCAM: {config.split!r}")

    ds = datasets.PCAM(
        root=str(config.root_path),
        split=split,
        download=True,
        transform=tfm,
    )

    return ds
