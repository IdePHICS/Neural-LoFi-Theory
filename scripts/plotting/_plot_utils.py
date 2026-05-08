"""Shared utilities for plotting scripts.

Provides helpers for loading JSON data, inferring image shapes, reshaping
eigenvectors/maximizers for display, and managing figure output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------

_DIRNAME_RE = re.compile(
    r"^(?P<preset>\w+)"
    r"_ntrain(?P<ntrain>\d+)"
    r"_P(?P<P>\d+)"
    r"_layers(?P<layers>\d+)"
    r"(?:_keep(?P<keep>\d+(?:-\d+)*))?"
    r"_(?P<activation>\w+)"
    r"_seed(?P<seed>\d+)$"
)


def parse_run_dirname(name: str) -> dict[str, int | str] | None:
    """Parse a run directory name into a parameter dict."""
    m = _DIRNAME_RE.match(name)
    if m is None:
        return None
    keep_raw = m.group("keep")
    if keep_raw is None:
        keep: int | str | None = None
    elif "-" in keep_raw:
        keep = keep_raw  # e.g. "50-10-5" for heterogeneous K
    else:
        keep = int(keep_raw)
    return {
        "preset": m.group("preset"),
        "ntrain": int(m.group("ntrain")),
        "P": int(m.group("P")),
        "layers": int(m.group("layers")),
        "keep": keep,
        "activation": m.group("activation"),
        "seed": int(m.group("seed")),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Image shape helpers
# ---------------------------------------------------------------------------


def infer_image_shape(dim: int) -> tuple[int, int] | tuple[int, int, int]:
    """Infer image shape from flattened dimension.

    Returns (H, W) for grayscale or (C, H, W) for colour images.
    """
    side = int(np.sqrt(dim))
    if side * side == dim:
        return (side, side)
    for c in (3, 1):
        spatial = dim // c
        s = int(np.sqrt(spatial))
        if s * s * c == dim:
            return (c, s, s)
    return (side, side)


def reshape_to_image(
    v: np.ndarray,
    shape: tuple,
    *,
    grayscale: bool = False,
) -> np.ndarray:
    """Reshape a flat vector to a displayable image array.

    For (H, W) shapes returns as-is. For (C, H, W) transposes to (H, W, C).

    Parameters
    ----------
    grayscale : bool
        If True and the image is multi-channel, reduce to (H, W) by taking
        the signed channel norm: ``sign(mean) * ||pixel||_2``.  This allows
        diverging colormaps (e.g. RdBu_r) to work correctly with RGB data.
    """
    if len(shape) == 2:
        return v[: shape[0] * shape[1]].reshape(shape)
    img = v[: int(np.prod(shape))].reshape(shape)  # (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    if grayscale and img.shape[-1] > 1:
        channel_mean = img.mean(axis=-1)
        channel_norm = np.linalg.norm(img, axis=-1)
        return np.sign(channel_mean) * channel_norm
    return img


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_symmetric(img: np.ndarray) -> tuple[float, float]:
    """Compute symmetric vmin/vmax for diverging colormaps.

    Returns ``(-absmax, +absmax)`` so that zero maps to the colourmap centre.
    """
    absmax = float(np.abs(img).max()) or 1.0
    return -absmax, absmax


def normalize_rgb(img: np.ndarray) -> np.ndarray:
    """Rescale an image to [0, 1] for RGB display.

    Uses joint min-max normalization across all channels so that
    colour relationships are preserved.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def is_rgb(img: np.ndarray) -> bool:
    """Check if an image array is RGB (H, W, 3)."""
    return img.ndim == 3 and img.shape[-1] == 3


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------


def discover_layers(run_dir: Path) -> list[str]:
    """Find all layer names present in a run directory.

    Returns a sorted list like ``["input", "layer_1", "layer_2", ...]``.
    """
    layers: list[str] = []
    if (run_dir / "eigenvalues_input.json").exists():
        layers.append("input")
    idx = 1
    while (run_dir / f"eigenvalues_layer_{idx}.json").exists():
        layers.append(f"layer_{idx}")
        idx += 1
    return layers


# ---------------------------------------------------------------------------
# Title / label helpers
# ---------------------------------------------------------------------------


def make_title(params: dict) -> str:
    """Build a compact title string from parsed run-directory parameters."""
    parts = [
        f"{params['preset']}",
        f"N={params['ntrain']}",
        f"P={params['P']}",
        f"act={params['activation']}",
        f"seed={params['seed']}",
    ]
    if params.get("keep") is not None:
        keep = params["keep"]
        if isinstance(keep, str) and "-" in keep:
            parts.append(f"K=[{keep.replace('-', ', ')}]")
        else:
            parts.append(f"K={keep}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_STYLE_PATH = Path(__file__).resolve().parents[2] / "neurips.mplstyle"


def setup_style(draft: bool = False) -> None:
    """Load the NeurIPS matplotlib style.

    Parameters
    ----------
    draft : bool
        If True, load the style but disable LaTeX rendering (fast iteration
        with correct visual style — colours, sizes, grid — but no TeX fonts).
    """
    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    if draft:
        plt.rcParams["text.usetex"] = False


class NeurIPSFigure:
    """Context manager that creates a NeurIPS-sized figure and saves/shows on exit.

    Parameters
    ----------
    width : float or str
        Figure width as a fraction of *TEXTWIDTH* (e.g. ``1.0`` = full text,
        ``0.5`` = half) or a preset string: ``"full"`` / ``"text"`` (5.5 in),
        ``"column"`` (3.25 in), ``"half"`` (2.75 in), ``"third"`` (1.83 in).
    aspect : float
        Height / width ratio per panel.  Default ``0.675`` matches the mplstyle.
    ncols, nrows : int
        Subplot grid; forwarded to :func:`plt.subplots`.
    draft : bool
        Skip LaTeX rendering (faster for iteration).
    save : bool
        Save on exit; requires *out_path*.
    out_path : Path or None
        Destination PDF path.  A PNG sidecar is also written when *also_png*.
    also_png : bool
        Write a companion PNG alongside the PDF (default ``True``).
    **subplot_kwargs
        Forwarded verbatim to :func:`plt.subplots`.

    Examples
    --------
    ::

        out = Path("imgs/fig.pdf")
        with NeurIPSFigure(width=1.0, save=True, out_path=out) as (fig, ax):
            ax.plot(x, y)
            ax.set_xlabel("x")

        # two-panel layout at full text width
        with NeurIPSFigure(width=1.0, ncols=2, save=True, out_path=out) as (fig, (ax1, ax2)):
            ax1.plot(...)
            ax2.plot(...)
    """

    TEXTWIDTH: float = 5.5     # NeurIPS full text width (inches)
    COLUMNWIDTH: float = 3.25  # NeurIPS single column width (inches)

    _PRESETS: dict[str, float] = {
        "full": 5.5,
        "text": 5.5,
        "column": 3.25,
        "half": 2.75,
        "third": 5.5 / 3.0,
    }

    def __init__(
        self,
        width: float | str = 1.0,
        aspect: float = 0.675,
        ncols: int = 1,
        nrows: int = 1,
        draft: bool = False,
        save: bool = False,
        out_path: Path | None = None,
        also_png: bool = True,
        **subplot_kwargs,
    ) -> None:
        if isinstance(width, str):
            w_in = self._PRESETS.get(width)
            if w_in is None:
                raise ValueError(
                    f"Unknown width preset {width!r}; choose from {list(self._PRESETS)}"
                )
        else:
            w_in = self.TEXTWIDTH * float(width)

        self._figsize = (w_in, w_in * aspect * nrows / ncols)
        self._nrows = nrows
        self._ncols = ncols
        self._draft = draft
        self._save = save
        self._out_path = out_path
        self._also_png = also_png
        self._subplot_kwargs = subplot_kwargs
        self._fig: plt.Figure | None = None

    def __enter__(self) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        setup_style(draft=self._draft)
        self._fig, axes = plt.subplots(
            self._nrows,
            self._ncols,
            figsize=self._figsize,
            **self._subplot_kwargs,
        )
        return self._fig, axes

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None or self._fig is None:
            return
        if self._save and self._out_path is not None:
            save_or_show(self._fig, True, self._out_path, also_png=self._also_png)
        else:
            plt.show()
            plt.close(self._fig)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_or_show(
    fig: plt.Figure,
    save: bool,
    out_path: Path,
    *,
    also_png: bool = True,
) -> None:
    """Save figure to *out_path* (PDF + optional PNG) or show interactively."""
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
        if also_png and out_path.suffix == ".pdf":
            png_path = out_path.with_suffix(".png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"Saved {png_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def add_common_args(parser) -> None:
    """Add standard CLI arguments shared across plotting scripts."""
    parser.add_argument(
        "--n_train", type=int, required=True,
        help="Filter runs by n_train value",
    )
    parser.add_argument(
        "--p", type=int, required=True,
        help="Filter runs by P (feature dimension) value",
    )
    parser.add_argument(
        "--dataset", type=str, default="fashion-mnist",
        help="Dataset name (default: fashion-mnist)",
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="Save plots instead of showing interactively",
    )
    parser.add_argument(
        "--draft", action="store_true", default=False,
        help="Skip neurips.mplstyle (faster, no LaTeX)",
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Override base results directory",
    )


def find_matching_runs(
    base_dir: Path,
    n_train: int,
    p: int,
    keep: int | None = None,
) -> list[tuple[Path, dict]]:
    """Find run directories matching the given filters."""
    matched = []
    if not base_dir.exists():
        return matched
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        params = parse_run_dirname(d.name)
        if params is None:
            continue
        if params["ntrain"] != n_train or params["P"] != p:
            continue
        if keep is not None and params.get("keep") != keep:
            continue
        matched.append((d, params))
    return matched
