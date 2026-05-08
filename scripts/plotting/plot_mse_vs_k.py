"""Plot test MSE and classification error vs K for ridge spectral models.

Reads JSON results from ``results/mse_vs_k/<dataset>/layers_<L>/`` and
produces two figures per (n_layers, n_train) combination:

1. Test MSE (± SEM) vs K, one curve per P value.
2. Test error % (± SEM) vs K, one curve per P value.

Expected filename patterns::

    # 1-layer (with eigenreduction):
    animal_vehicle_seed0_ntrain10000_ntest5000_p100k2sigmoid_ridge-6to6x500.json

    # 2-layer (first layer eigenreduced, second full-rank):
    animal_vehicle_seed0_ntrain10000_ntest5000_p100k2sigmoid-p100sigmoid_noeig_ridge-6to6x500.json

Usage
-----
python scripts/plotting/plot_mse_vs_k.py --dataset cifar10 --save
python scripts/plotting/plot_mse_vs_k.py --dataset cifar10 --layers 1 --save
python scripts/plotting/plot_mse_vs_k.py \
    --dataset cifar10 --layers 2 --n_train 10000 --save
python scripts/plotting/plot_mse_vs_k.py --dataset cifar10 --layers 1 2 --save --draft
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
from _plot_utils import NeurIPSFigure

# ---------------------------------------------------------------------------
# Filename regexes
# ---------------------------------------------------------------------------

# 1-layer with eigenreduction
_RE_1LAYER = re.compile(
    r"^(?P<preset>\w+)"
    r"_seed(?P<seed>\d+)"
    r"_ntrain(?P<ntrain>\d+)"
    r"_ntest(?P<ntest>\d+)"
    r"_p(?P<p>\d+)k(?P<k>\d+)(?P<act>[a-zA-Z]+)"
    r"_ridge(?P<ridge>.+)\.json$"
)

# 2-layer: first layer eigenreduced, second noeig
_RE_2LAYER = re.compile(
    r"^(?P<preset>\w+)"
    r"_seed(?P<seed>\d+)"
    r"_ntrain(?P<ntrain>\d+)"
    r"_ntest(?P<ntest>\d+)"
    r"_p(?P<p1>\d+)k(?P<k>\d+)(?P<act1>[a-zA-Z]+)"
    r"-p(?P<p2>\d+)(?P<act2>[a-zA-Z]+)_noeig"
    r"_ridge(?P<ridge>.+)\.json$"
)


def _parse_1layer(name: str) -> dict[str, int | str] | None:
    m = _RE_1LAYER.match(name)
    if m is None:
        return None
    return {
        "seed": int(m.group("seed")),
        "n_train": int(m.group("ntrain")),
        "p": int(m.group("p")),
        "k": int(m.group("k")),
        "activation": m.group("act"),
    }


def _parse_2layer(name: str) -> dict[str, int | str] | None:
    m = _RE_2LAYER.match(name)
    if m is None:
        return None
    return {
        "seed": int(m.group("seed")),
        "n_train": int(m.group("ntrain")),
        "p": int(m.group("p1")),
        "k": int(m.group("k")),
        "activation": m.group("act1"),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(
    data_dir: Path,
    *,
    n_layers: int,
    activation: str = "",
) -> tuple[
    dict[tuple[int, int, int], list[float]],
    dict[tuple[int, int, int], list[float]],
]:
    """Load test MSE and test accuracy from result files.

    Parameters
    ----------
    data_dir : Path
        Directory containing JSON result files.
    n_layers : int
        1 or 2, selects the filename regex.
    activation : str
        If non-empty, only load files whose parsed activation matches.

    Returns
    -------
    mse_data : dict[(p, k, n_train), list[float]]
    acc_data : dict[(p, k, n_train), list[float]]
    """
    parse_fn = _parse_1layer if n_layers == 1 else _parse_2layer

    mse_data: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    acc_data: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    n_loaded = 0
    n_skipped = 0

    for entry in os.scandir(data_dir):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        params = parse_fn(entry.name)
        if params is None:
            n_skipped += 1
            continue
        if activation and params["activation"] != activation:
            continue
        try:
            with open(entry.path) as f:
                data = json.load(f)
            mse = float(data["final"]["test_mse"])
            acc = float(data["final"]["test_accuracy"])
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
            print(f"Warning: skipping {entry.name} ({exc})", file=sys.stderr)
            continue

        key = (params["p"], params["k"], params["n_train"])
        mse_data[key].append(mse)
        acc_data[key].append(acc)
        n_loaded += 1

    if n_skipped:
        print(
            f"Note: {n_skipped} file(s) did not match the {n_layers}-layer pattern.",
            file=sys.stderr,
        )
    print(f"Loaded {n_loaded} result files from {data_dir}")

    return dict(mse_data), dict(acc_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))



def _fmt_p(p: int) -> str:
    if p >= 1000 and p % 1000 == 0:
        return rf"{p // 1000}\mathrm{{k}}"
    return str(p)


def _choose_colors(n: int) -> list:
    prop_cycle = mpl.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in prop_cycle]
    return [colors[i % len(colors)] for i in range(n)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panels(
    data_getter,
    p_values: list[int],
    n_train_values: list[int],
    *,
    ylabel: str,
    setup_ax,
    out: Path,
    draft: bool,
    save: bool,
) -> None:
    ncols = len(n_train_values)
    colors = _choose_colors(len(p_values))
    width = "full" if ncols > 1 else "column"

    with NeurIPSFigure(
        width=width, ncols=ncols, draft=draft, save=save, out_path=out,
        sharey=True,
    ) as (_, raw_axes):
        ax_list: list = [raw_axes] if ncols == 1 else list(raw_axes)

        for col, n_train in enumerate(n_train_values):
            ax = ax_list[col]
            for i, p in enumerate(p_values):
                k_vals, means, sems = data_getter(p, n_train)
                if not k_vals:
                    continue
                x = np.array(k_vals, dtype=float)
                y = np.array(means)
                e = np.array(sems)
                (line,) = ax.plot(
                    x, y,
                    marker=".", markersize=2,
                    linewidth=1.5, color=colors[i],
                    label=f"${_fmt_p(p)}$",
                )
                ax.fill_between(x, y - e, y + e, alpha=0.2, color=line.get_color())

            ax.set_xlabel("$k$")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            setup_ax(ax)
            ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
            ax.set_title(f"$n = {n_train}$")

        ax_list[-1].legend(title="$p$")


def plot_mse_vs_k(
    mse_data: dict[tuple[int, int, int], list[float]],
    p_values: list[int],
    n_train_values: list[int],
    *,
    n_layers: int,
    activation: str = "",
    dataset: str = "",
    save_path: Path | None = None,
    save: bool = False,
    draft: bool = False,
) -> None:
    """Plot test MSE ± SEM vs K, one curve per P, panels per n_train."""
    out = save_path or _default_save_path(
        "mse_vs_k", dataset, n_layers, n_train_values, activation,
    )

    def get_data(p: int, n_train: int):
        all_k = sorted(k for (pp, k, nt) in mse_data if pp == p and nt == n_train)
        k_vals, means, sems = [], [], []
        for k in all_k:
            values = mse_data.get((p, k, n_train), [])
            if not values:
                continue
            k_vals.append(k)
            means.append(float(np.mean(values)))
            sems.append(_sem(values))
        return k_vals, means, sems

    _plot_panels(
        get_data, p_values, n_train_values,
        ylabel="Test MSE",
        setup_ax=lambda _: None,
        out=out, draft=draft, save=save,
    )


def plot_error_vs_k(
    acc_data: dict[tuple[int, int, int], list[float]],
    p_values: list[int],
    n_train_values: list[int],
    *,
    n_layers: int,
    activation: str = "",
    dataset: str = "",
    save_path: Path | None = None,
    save: bool = False,
    draft: bool = False,
) -> None:
    """Plot test error (%) ± SEM vs K, one curve per P, panels per n_train."""
    out = save_path or _default_save_path(
        "error_vs_k", dataset, n_layers, n_train_values, activation,
    )

    def get_data(p: int, n_train: int):
        all_k = sorted(k for (pp, k, nt) in acc_data if pp == p and nt == n_train)
        k_vals, means, sems = [], [], []
        for k in all_k:
            values = acc_data.get((p, k, n_train), [])
            if not values:
                continue
            errors = [100.0 * (1.0 - a) for a in values]
            k_vals.append(k)
            means.append(float(np.mean(errors)))
            sems.append(
                float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
                if len(errors) > 1 else 0.0
            )
        return k_vals, means, sems

    def setup_error_ax(ax) -> None:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))

    _plot_panels(
        get_data, p_values, n_train_values,
        ylabel=rf"Test error (\%)",
        setup_ax=setup_error_ax,
        out=out, draft=draft, save=save,
    )


def _default_save_path(
    prefix: str,
    dataset: str,
    n_layers: int,
    n_train: int | list[int],
    activation: str = "",
) -> Path:
    img_dir = Path("imgs") / "mse_vs_k" / dataset
    nt_str = (
        "-".join(str(n) for n in n_train)
        if isinstance(n_train, list)
        else str(n_train)
    )
    parts = [prefix, f"layers_{n_layers}", f"ntrain_{nt_str}"]
    if activation:
        parts.append(activation)
    return img_dir / f"{'_'.join(parts)}.pdf"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_RESULTS_ROOT = Path("results")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test MSE and error vs K from mse_vs_k sweep results.",
    )
    parser.add_argument(
        "--dataset", default="cifar10",
        help="Dataset name (default: cifar10).",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[1, 2],
        help="Number of layers to plot (default: 1 2).",
    )
    parser.add_argument(
        "--n_train", type=int, nargs="+", default=None,
        help="n_train values to plot (default: auto-discover from data).",
    )
    parser.add_argument(
        "--p_values", type=int, nargs="+", default=None,
        help="P values to plot (default: auto-discover from data).",
    )
    parser.add_argument(
        "--activation", default="",
        help="Activation name for title annotation.",
    )
    parser.add_argument(
        "--base_dir", type=Path, default=None,
        help="Override results base directory.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save figures instead of displaying.",
    )
    parser.add_argument(
        "--draft", action="store_true",
        help="Skip neurips.mplstyle (faster, no LaTeX).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = args.base_dir or (_RESULTS_ROOT / "mse_vs_k" / args.dataset)

    for n_layers in args.layers:
        data_dir = base / f"layers_{n_layers}"
        if not data_dir.is_dir():
            print(
                f"Warning: directory not found: {data_dir}, skipping.",
                file=sys.stderr,
            )
            continue

        print(f"\n--- Loading {n_layers}-layer results from {data_dir} ---")
        mse_data, acc_data = load_results(
            data_dir, n_layers=n_layers, activation=args.activation,
        )

        if not mse_data:
            print(f"No data found for {n_layers}-layer.", file=sys.stderr)
            continue

        # Discover available P and n_train values
        all_p = sorted({p for p, _, _ in mse_data})
        all_ntrain = sorted({nt for _, _, nt in mse_data})
        p_values = args.p_values or all_p
        n_train_values = args.n_train or all_ntrain

        print(f"P values: {p_values}")
        print(f"n_train values: {n_train_values}")

        print(f"\nPlotting {n_layers}-layer ...")
        common = dict(
            n_layers=n_layers,
            activation=args.activation,
            dataset=args.dataset,
            save=args.save,
            draft=args.draft,
        )
        plot_mse_vs_k(mse_data, p_values, n_train_values, **common)
        plot_error_vs_k(acc_data, p_values, n_train_values, **common)


if __name__ == "__main__":
    main()
