from __future__ import annotations

from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import Dataset, Subset

# ------------------------------------------------------------------
# Size Liminting
# ------------------------------------------------------------------


def limit_dataset(
    ds: Dataset, n: int, *, seed: int = 0, shuffle: bool = True
) -> Dataset:
    """Sub-sample *ds* to at most *n* elements."""
    if n <= 0:
        return ds

    total = len(cast(Sized, ds))
    if n >= total:
        return ds

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(total, generator=g)[:n].tolist()
    else:
        idx = list(range(n))

    return Subset(ds, idx)


# ------------------------------------------------------------------
# Class filtering
# ------------------------------------------------------------------


class _MappedSubset(Dataset):
    """Subset that optionally remaps targets via ``label_map``."""

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        label_map: dict[int, int] | dict[int, int | float] | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[self.indices[idx]]
        y_int = int(y)

        if self.label_map is None:
            return x, y_int

        return x, self.label_map[y_int]

    def __len__(self) -> int:
        return len(self.indices)


def filter_classes(
    ds: Dataset,
    classes: list[int],
    *,
    remap: bool = True,
    label_map: dict[int, int] | dict[int, int | float] | None = None,
) -> Dataset:
    """Keep only samples whose target is in *classes* and optionally relabel them.

    Parameters
    ----------
    ds : Dataset
        A PyTorch dataset returning ``(input, target)`` tuples.
    classes : list[int]
        Class indices to retain.
    remap : bool
        If ``True`` (default) and ``label_map`` is not provided, relabel kept
        classes to consecutive integers ``0 … len(classes) - 1``
        (sorted by original label).
    label_map : dict[int, int | float] | None
        Explicit mapping for target relabelling
        (e.g. ``{3: -1.0, 7: 1.0}``).  If provided, it takes
        precedence over ``remap`` and is applied to the kept classes.

    Returns
    -------
    Dataset
        Filtered dataset, optionally relabelled.

    Notes
    -----
    - Returned targets are scalar labels (not one-hot vectors).
    - If both ``remap`` and ``label_map`` are set, ``label_map`` is used.
    """
    keep = set(classes)
    indices: list[int] = []

    for i in range(len(cast(Sized, ds))):
        _, y = ds[i]
        if int(y) in keep:
            indices.append(i)

    if not indices:
        raise ValueError(
            f"No samples found for classes {sorted(keep)}. "
            "Check that the class indices match the dataset."
        )

    if label_map is None and not remap:
        return Subset(ds, indices)

    if label_map is not None:
        missing = keep.difference(label_map.keys())
        if missing:
            raise ValueError(
                f"label_map is missing mappings for classes {sorted(missing)}. "
                f"Provided keys: {sorted(label_map.keys())}"
            )
        return _MappedSubset(ds, indices, label_map=label_map)

    remap_map = {c: new for new, c in enumerate(sorted(keep))}
    return _MappedSubset(ds, indices, label_map=remap_map)
