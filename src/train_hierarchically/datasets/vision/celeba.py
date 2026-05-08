from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_celeba_transform

log = logging.getLogger(__name__)

# CelebA attribute indices for preset-based binary classification.
# Each preset selects one of the 40 face attributes as the binary target.
_PRESET_ATTR_INDEX: dict[str, int] = {
    "male_female": 20,  # "Male"
    "smiling": 31,  # "Smiling"
    "eyeglasses": 15,  # "Eyeglasses"
    "mustache": 22,  # "Mustache"
    "no_beard": 24,  # "No_Beard"
    "brown_hair": 11,  # "Brown_Hair"
    "high_cheekbones": 19,  # "High_Cheekbones"
    "mouth_open": 21,  # "Mouth_Slightly_Open"
    "young": 39,  # "Young"
    "attractive": 2,  # "Attractive"
}

_DEFAULT_ATTR_INDEX = 20  # "Male" — used when no preset is specified

# Internet Archive mirror — reliable HTTPS, no auth, no rate limits.
_IA_BASE = "https://archive.org/download/celeba"
_CELEBA_FILES: list[tuple[str, str]] = [
    (f"{_IA_BASE}/Img/img_align_celeba.zip", "img_align_celeba.zip"),
    (f"{_IA_BASE}/Anno/list_attr_celeba.txt", "list_attr_celeba.txt"),
    (f"{_IA_BASE}/Anno/identity_CelebA.txt", "identity_CelebA.txt"),
    (f"{_IA_BASE}/Anno/list_bbox_celeba.txt", "list_bbox_celeba.txt"),
    (
        f"{_IA_BASE}/Anno/list_landmarks_align_celeba.txt",
        "list_landmarks_align_celeba.txt",
    ),
    (
        f"{_IA_BASE}/Eval/list_eval_partition.txt",
        "list_eval_partition.txt",
    ),
]


def _download_with_retries(
    url: str,
    dest: Path,
    *,
    max_attempts: int = 5,
    timeout_s: float = 120.0,
) -> None:
    """Download a file with retries for transient HTTP/network failures."""
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; train-hierarchically/1.0)"}

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                with tmp_dest.open("wb") as f:
                    shutil.copyfileobj(response, f)
            tmp_dest.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            if tmp_dest.exists():
                tmp_dest.unlink()

            should_retry = attempt < max_attempts
            if isinstance(err, urllib.error.HTTPError) and err.code < 500:
                should_retry = False

            if not should_retry:
                raise RuntimeError(
                    f"Failed to download {url} after {attempt} attempt(s)."
                ) from err

            backoff_s = min(30.0, 2.0 ** (attempt - 1))
            log.warning(
                "Download failed (%s). Retrying in %.1fs [%d/%d]: %s",
                type(err).__name__,
                backoff_s,
                attempt,
                max_attempts,
                url,
            )
            time.sleep(backoff_s)


def _download_celeba(root: str | Path) -> None:
    """Download CelebA from Internet Archive into torchvision layout."""
    celeba_dir = Path(root) / "celeba"
    celeba_dir.mkdir(parents=True, exist_ok=True)

    for url, filename in _CELEBA_FILES:
        dest = celeba_dir / filename
        if dest.exists():
            continue
        log.info("Downloading %s → %s", filename, dest)
        _download_with_retries(url, dest)

    # Extract images if not already done.
    img_dir = celeba_dir / "img_align_celeba"
    zip_path = celeba_dir / "img_align_celeba.zip"
    if not img_dir.exists() and zip_path.exists():
        log.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(celeba_dir))


class _CelebASingleAttr(Dataset):
    """Wraps torchvision CelebA to return a single attribute as target."""

    def __init__(
        self, celeba_dataset: datasets.CelebA, attr_index: int
    ) -> None:
        self.dataset = celeba_dataset
        self.attr_index = attr_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        img, attrs = self.dataset[index]
        return img, int(attrs[self.attr_index])


@register("celeba")
def build_celeba(config: DatasetConfig) -> Dataset:
    tfm = config.transform or default_celeba_transform(
        flatten=config.flatten
    )

    if config.split == "train":
        split = "train"
    elif config.split == "val":
        split = "valid"
    elif config.split == "test":
        split = "test"
    else:
        raise ValueError(
            f"Unsupported split for CelebA: {config.split!r}"
        )

    attr_index = _PRESET_ATTR_INDEX.get(
        config.class_preset or "", _DEFAULT_ATTR_INDEX
    )

    # Download from Internet Archive (skips files already present).
    _download_celeba(config.root_path)

    ds = datasets.CelebA(
        root=str(config.root_path),
        split=split,
        target_type="attr",
        download=False,
        transform=tfm,
    )

    return _CelebASingleAttr(ds, attr_index)
