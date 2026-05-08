from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torchvision import transforms

_VALID_TYPES = {
    "mnist",
    "fashion-mnist",
    "cifar10",
    "pcam",
    "celeba",
}

_VALID_SPLITS = {"train", "val", "test"}


@dataclass(slots=True)
class DatasetConfig:
    """
    Unified configuration for dataset creation.

    dataset_type:
      One of the registered vision dataset names (e.g. "mnist", "cifar10", "celeba", …).

    split: "train" | "val" | "test"
      Used to select the appropriate torchvision split.

    class_preset : str | None
      Named grouping shortcut (e.g. ``"even_odd"``, ``"upper_lower"``).
      When set, populates ``classes`` and ``label_map`` automatically.
      Cannot be combined with explicit ``classes`` or ``label_map``.
      See :mod:`train_hierarchically.datasets.presets` for available names.

    classes : list[int] | None
      If given, only samples whose target is in this list are kept.
      Applied **before** the ``n_samples`` limit.

    remap_labels : bool
      When ``classes`` is set and ``remap_labels`` is True (default),
      the original class indices are remapped to consecutive integers
      ``0 … len(classes) - 1`` (sorted order).

    Each returned sample is an ``(input, target)`` tuple following standard
    PyTorch Dataset conventions.
    """

    dataset_type: str
    split: str = "train"
    root: str | Path = "./data"

    n_samples: int = 20_000
    seed: int = 0
    flatten: bool = False

    class_preset: str | None = None
    classes: list[int] | None = None
    remap_labels: bool = True
    label_map: dict[int, int | float] | None = None

    transform: transforms.Compose | None = None

    root_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_type = self.dataset_type.lower().strip()
        if self.dataset_type not in _VALID_TYPES:
            raise ValueError(
                f"dataset_type must be one of {sorted(_VALID_TYPES)}, "
                f"got {self.dataset_type!r}"
            )

        self.split = self.split.lower().strip()
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {self.split!r}"
            )

        self.root_path = Path(self.root).expanduser()

        self._resolve_class_preset()
        self._maybe_autoset_binary_label_map()

        self.validate()

    def _resolve_class_preset(self) -> None:
        """Populate ``classes`` and ``label_map`` from a named preset.

        Raises ``ValueError`` if ``class_preset`` is set alongside explicit
        ``classes`` or ``label_map``, as those combinations are ambiguous.
        """
        if self.class_preset is None:
            return
        if self.classes is not None:
            raise ValueError(
                "Cannot set both 'class_preset' and 'classes'. "
                "Use 'class_preset' alone or specify 'classes' directly."
            )
        if self.label_map is not None:
            raise ValueError(
                "Cannot set both 'class_preset' and 'label_map'. "
                "Use 'class_preset' alone or specify 'label_map' directly."
            )
        from .presets import resolve_preset  # local import to avoid circular deps

        preset = resolve_preset(self.dataset_type, self.class_preset)
        self.classes = list(preset.classes)
        self.label_map = preset.label_map

    def _maybe_autoset_binary_label_map(self) -> None:
        """If binary class filtering is requested, auto-map to {-1, +1}.

        Mapping rule (deterministic): sorted(classes)[0] -> -1, sorted(classes)[1] -> +1
        """
        if not self.remap_labels:
            return
        if self.label_map is not None:
            return
        if self.classes is None:
            return

        unique_classes = sorted(set(self.classes))
        if len(unique_classes) == 2:
            self.label_map = {
                unique_classes[0]: -1.0,
                unique_classes[1]: +1.0,
            }

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.classes is not None and len(self.classes) == 0:
            raise ValueError("classes must be None or a non-empty list")
        if self.label_map is not None and len(self.label_map) == 0:
            raise ValueError("label_map must be None or a non-empty dict")
        if self.label_map is not None and self.classes is not None:
            class_set = set(self.classes)
            map_keys = set(self.label_map.keys())
            if map_keys != class_set:
                raise ValueError(
                    "label_map keys must match exactly the selected classes. "
                    f"classes={sorted(class_set)}, label_map keys={sorted(map_keys)}"
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(type={self.dataset_type!r}, split={self.split!r}, "
            f"n_samples={self.n_samples}, root={str(self.root_path)!r})"
        )
