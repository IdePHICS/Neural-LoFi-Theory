from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassPreset:
    """A named grouping of dataset classes with an optional label mapping.

    Parameters
    ----------
    classes:
        Ordered class indices to retain from the dataset.
    label_map:
        Mapping from original class index to output label (int or float).
        ``None`` means consecutive-integer remapping (0 … K-1) is applied
        by the standard ``remap_labels`` logic.
    description:
        Human-readable explanation of the grouping.
    """

    classes: tuple[int, ...]
    label_map: dict[int, int | float] | None
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in presets  –  PRESETS[dataset_type][preset_name]
# ---------------------------------------------------------------------------

#: Fashion-MNIST class index → name reference
#: 0: T-shirt/top  1: Trouser    2: Pullover  3: Dress  4: Coat
#: 5: Sandal       6: Shirt      7: Sneaker   8: Bag    9: Ankle boot

#: CIFAR-10 class index → name reference
#: 0: airplane  1: automobile  2: bird  3: cat   4: deer
#: 5: dog       6: frog        7: horse 8: ship  9: truck

PRESETS: dict[str, dict[str, ClassPreset]] = {
    "mnist": {
        "even_odd": ClassPreset(
            classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            label_map={
                0: -1.0, 2: -1.0, 4: -1.0, 6: -1.0, 8: -1.0,  # even → -1
                1: 1.0,  3: 1.0,  5: 1.0,  7: 1.0,  9: 1.0,   # odd  → +1
            },
            description="Even digits → -1, odd digits → +1",
        ),
        "less_than_five": ClassPreset(
            classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            label_map={0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0,  # <5 → -1
                       5: 1.0,  6: 1.0,  7: 1.0,  8: 1.0,  9: 1.0}, # ≥5 → +1
            description="Digits < 5 → -1, digits ≥ 5 → +1",
        ),
    },
    "fashion-mnist": {
        # Upper-body clothing vs. lower-body / footwear (bag excluded)
        "upper_lower": ClassPreset(
            classes=(0, 1, 2, 3, 4, 5, 6, 7, 9),
            label_map={
                0: -1.0, 2: -1.0, 3: -1.0, 4: -1.0, 6: -1.0,  # upper body → -1
                1: 1.0,  5: 1.0,  7: 1.0,  9: 1.0,             # lower/footwear → +1
            },
            description=(
                "Upper-body clothing (T-shirt, Pullover, Dress, Coat, Shirt) → -1; "
                "lower / footwear (Trouser, Sandal, Sneaker, Ankle boot) → +1. "
                "Bag (class 8) is excluded."
            ),
        ),
        # Clothing items vs. footwear-only (bag excluded)
        "clothing_footwear": ClassPreset(
            classes=(0, 1, 2, 3, 4, 5, 6, 7, 9),
            label_map={
                0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0,  # clothing → -1
                4: -1.0, 6: -1.0,                      # clothing → -1
                5: 1.0, 7: 1.0, 9: 1.0,                # footwear → +1
            },
            description=(
                "Clothing items (T-shirt, Trouser, Pullover, Dress, Coat, Shirt) → -1; "
                "footwear (Sandal, Sneaker, Ankle boot) → +1. "
                "Bag (class 8) is excluded."
            ),
        ),
    },
    "cifar10": {
        "animal_vehicle": ClassPreset(
            classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            label_map={
                2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0,  # animals → -1
                0: 1.0,  1: 1.0,  8: 1.0,  9: 1.0,                       # vehicles → +1
            },
            description=(
                "Animals (bird, cat, deer, dog, frog, horse) → -1; "
                "vehicles (airplane, automobile, ship, truck) → +1."
            ),
        ),
    },
    #: CelebA — binary face attributes
    #: Target is a single attribute extracted from the 40-attribute vector.
    #: The builder selects the attribute based on the preset name.
    "celeba": {
        "male_female": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Female → -1; Male → +1 (attribute 20: 'Male').",
        ),
        "smiling": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Not smiling → -1; Smiling → +1 (attribute 31: 'Smiling').",
        ),
        "eyeglasses": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="No eyeglasses → -1; Eyeglasses → +1 (attribute 15).",
        ),
        "mustache": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="No mustache → -1; Mustache → +1 (attribute 22).",
        ),
        "no_beard": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Has beard → -1; No beard → +1 (attribute 24: 'No_Beard').",
        ),
        "brown_hair": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Not brown hair → -1; Brown hair → +1 (attribute 11).",
        ),
        "high_cheekbones": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="No high cheekbones → -1; High cheekbones → +1 (attribute 19).",
        ),
        "mouth_open": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description=(
                "Mouth closed → -1; Mouth slightly open → +1 "
                "(attribute 21: 'Mouth_Slightly_Open')."
            ),
        ),
        "young": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Not young → -1; Young → +1 (attribute 39: 'Young').",
        ),
        "attractive": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Not attractive → -1; Attractive → +1 (attribute 2).",
        ),
    },
    #: PCAM class index → name reference
    #: 0: normal (no tumor)  1: tumor
    "pcam": {
        "tumor_normal": ClassPreset(
            classes=(0, 1),
            label_map={0: -1.0, 1: 1.0},
            description="Normal tissue → -1; tumor tissue → +1.",
        ),
    },
}


def available_presets(dataset_type: str) -> list[str]:
    """Return sorted preset names for *dataset_type*, or ``[]`` if none exist."""
    return sorted(PRESETS.get(dataset_type, {}).keys())


def resolve_preset(dataset_type: str, preset_name: str) -> ClassPreset:
    """Look up *preset_name* for *dataset_type*.

    Raises
    ------
    ValueError
        If the dataset or the preset name is not found.
    """
    dataset_presets = PRESETS.get(dataset_type)
    if dataset_presets is None:
        raise ValueError(
            f"No presets defined for dataset_type {dataset_type!r}. "
            f"Datasets with presets: {sorted(PRESETS.keys())}"
        )
    preset = dataset_presets.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown preset {preset_name!r} for {dataset_type!r}. "
            f"Available presets: {sorted(dataset_presets.keys())}"
        )
    return preset
