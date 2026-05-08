"""Companion to ``plot_nlofi_appendix``: log-log convergence-rate figure.

Plots ``|mean_seeds (MSE_finite,s − MSE_kernel,s)|`` vs ``P`` on log-log
axes for each ``(K, λ)`` cell.  Pairs seeds across the finite-P and
kernel blocks (which after the data-tie fix use the same data subsample
per seed), so the difference cancels data-subsample variability and
isolates the finite-P → kernel bias.

The y-error reflects ``SEM(Δ_s)`` over the paired seeds.  When ``|mean| −
SEM`` drops below the configured floor the lower bar is clipped to the
floor and a small down-triangle marker is drawn there to indicate the
uncertainty crosses zero.

Mirrors the layout of ``plot_nlofi_appendix``: 3 K-pairs × 4 λ panels,
one line per ``n_train``.  Selectable via ``--metric mse`` (default) or
``--metric clf_err``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_utils import NeurIPSFigure  # noqa: E402


_DEFAULT_DATA_DIR = Path("results/nlofi_appendix")
_DEFAULT_OUT = Path("imgs/nlofi_appendix/nlofi_convergence_rate.pdf")
_DEFAULT_OUT_CLFERR = Path(
    "imgs/nlofi_appendix/nlofi_convergence_rate_test_error.pdf",
)

_METRIC_YLABEL: dict[str, str] = {
    "mse": r"$|\mathrm{Test\ MSE}_{\mathrm{finite}-P} - \mathrm{Test\ MSE}_{\mathrm{kernel}}|$",
    "clf_err": r"$|\mathrm{Test\ Err}_{\mathrm{finite}-P} - \mathrm{Test\ Err}_{\mathrm{kernel}}|$",
}

_RF_RE = re.compile(
    r"^rf__K(?P<k0>\d+)_(?P<k1>\d+)__n(?P<n>\d+)__P(?P<P>\d+)__seed"
    r"(?P<seed>\d+)\.json$"
)
_KERNEL_RE = re.compile(
    r"^kernel__K(?P<k0>\d+)_(?P<k1>\d+)__n(?P<n>\d+)__ds(?P<ds>\d+)\.json$"
)


# ---------------------------------------------------------------------------
# Loading (per-seed)
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_per_seed(
    data_dir: Path, metric: str = "mse",
) -> tuple[
    dict[tuple[int, int, str, int, int], dict[int, float]],
    # finite[K0, K1, λ, n, P][seed] = value
    dict[tuple[int, int, str, int], dict[int, float]],
    # kernel[K0, K1, λ, n][ds] = value
]:
    finite: dict[
        tuple[int, int, str, int, int], dict[int, float]
    ] = defaultdict(dict)
    kernel: dict[
        tuple[int, int, str, int], dict[int, float]
    ] = defaultdict(dict)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")
    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith("rf__"):
            m = _RF_RE.match(name)
            if m is None:
                continue
            row = _read(Path(entry.path))
            d = row.get(metric, {})
            for lam_str, val in d.items():
                finite[(
                    int(m["k0"]), int(m["k1"]),
                    lam_str, int(m["n"]), int(m["P"]),
                )][int(m["seed"])] = float(val)
        elif name.startswith("kernel__"):
            m = _KERNEL_RE.match(name)
            if m is None:
                continue
            row = _read(Path(entry.path))
            d = row.get(metric, {})
            for lam_str, val in d.items():
                kernel[(
                    int(m["k0"]), int(m["k1"]),
                    lam_str, int(m["n"]),
                )][int(m["ds"])] = float(val)
    return finite, kernel


# ---------------------------------------------------------------------------
# Paired-seed aggregation
# ---------------------------------------------------------------------------


def paired_diff_at_p(
    finite: dict[tuple[int, int, str, int, int], dict[int, float]],
    kernel: dict[tuple[int, int, str, int], dict[int, float]],
    *, k0: int, k1: int, lam_str: str, n_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For one (K, λ, n) cell, collect Δ_s = finite_s − kernel_s across all P.

    Returns four equal-length arrays:
        P_arr           – P values (sorted ascending)
        mean_signed     – seed-mean of Δ_s (can be negative)
        sem             – SEM of Δ_s over paired seeds
        n_pairs         – number of paired seeds at each P
    """
    kdict = kernel.get((k0, k1, lam_str, n_train), {})
    if not kdict:
        return (
            np.array([]), np.array([]), np.array([]), np.array([], dtype=int),
        )
    rows = []
    for (kk0, kk1, ll, nn, P), seed_dict in finite.items():
        if kk0 != k0 or kk1 != k1 or ll != lam_str or nn != n_train:
            continue
        common = set(seed_dict.keys()) & set(kdict.keys())
        if not common:
            continue
        deltas = np.array(
            [seed_dict[s] - kdict[s] for s in sorted(common)],
            dtype=float,
        )
        if deltas.size == 0:
            continue
        mean = float(deltas.mean())
        sem = (
            float(deltas.std(ddof=1) / np.sqrt(deltas.size))
            if deltas.size > 1 else 0.0
        )
        rows.append((P, mean, sem, deltas.size))
    rows.sort()
    if not rows:
        return (
            np.array([]), np.array([]), np.array([]), np.array([], dtype=int),
        )
    return (
        np.array([r[0] for r in rows], dtype=int),
        np.array([r[1] for r in rows], dtype=float),
        np.array([r[2] for r in rows], dtype=float),
        np.array([r[3] for r in rows], dtype=int),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _lam_label(lam_str: str) -> str:
    val = float(lam_str)
    if val >= 1.0 and val.is_integer():
        return rf"$\lambda = {int(val)}$"
    if val < 1.0:
        log10 = np.log10(val)
        if abs(log10 - round(log10)) < 1e-9:
            return rf"$\lambda = 10^{{{int(round(log10))}}}$"
    return rf"$\lambda = {val:g}$"


def _kpair_label(k0: int, k1: int) -> str:
    if k0 == k1:
        return rf"$k_1{{=}}k_2{{=}}{k0}$"
    return rf"$k_1{{=}}{k0},k_2{{=}}{k1}$"


def _plot_one_cell(
    ax: plt.Axes,
    *,
    P_arr: np.ndarray, mean_signed: np.ndarray, sem: np.ndarray,
    color, floor: float, marker_size: float = 2.0, line_width: float = 0.9,
) -> None:
    """Plot |mean(Δ)| with asymmetric, floor-clipped error bars on log-y."""
    if P_arr.size == 0:
        return
    y = np.abs(mean_signed)
    # Don't allow y itself to drop below floor (rare; only if mean is exactly 0)
    y_plot = np.maximum(y, floor)

    upper_err = sem
    raw_lower = y - sem
    clipped = raw_lower < floor
    lower_err = np.where(clipped, y_plot - floor, sem)

    ax.errorbar(
        P_arr, y_plot, yerr=[lower_err, upper_err],
        marker="o", markersize=marker_size, linewidth=line_width,
        color=color, capsize=1.0,
    )

    # Down-triangles where the lower bar was clipped (i.e. error crosses 0)
    if clipped.any():
        ax.scatter(
            P_arr[clipped], np.full(clipped.sum(), floor),
            marker="v", s=8, color=color, alpha=0.85, zorder=3,
        )


# ---------------------------------------------------------------------------
# Top-level figure builder
# ---------------------------------------------------------------------------


def make_figure(
    *,
    finite: dict[tuple[int, int, str, int, int], dict[int, float]],
    kernel: dict[tuple[int, int, str, int], dict[int, float]],
    k_pairs: list[tuple[int, int]],
    lambdas: list[str],
    n_trains: list[int],
    out_path: Path,
    save: bool,
    draft: bool,
    width_frac: float,
    aspect: float,
    metric: str,
    floor: float,
) -> None:
    nrows = len(k_pairs)
    ncols = len(lambdas)

    with NeurIPSFigure(
        width=width_frac, aspect=aspect,
        ncols=ncols, nrows=nrows,
        sharex=True, sharey="row",
        draft=draft, save=save, out_path=out_path,
        constrained_layout=False,
    ) as (fig, axes):
        axes = np.atleast_2d(axes)

        cmap = plt.get_cmap("viridis")
        n_colors = max(len(n_trains), 2)
        n_train_color = {
            n: cmap(0.15 + 0.7 * (i / max(n_colors - 1, 1)))
            for i, n in enumerate(n_trains)
        }

        legend_handles_n: list = []
        legend_labels_n: list[str] = []

        for r, (k0, k1) in enumerate(k_pairs):
            for c, lam_str in enumerate(lambdas):
                ax = axes[r, c]
                for n_train in n_trains:
                    color = n_train_color[n_train]
                    P_arr, mean_signed, sem, _ = paired_diff_at_p(
                        finite, kernel,
                        k0=k0, k1=k1, lam_str=lam_str, n_train=n_train,
                    )
                    _plot_one_cell(
                        ax,
                        P_arr=P_arr, mean_signed=mean_signed, sem=sem,
                        color=color, floor=floor,
                    )
                    if r == 0 and c == 0 and P_arr.size:
                        legend_handles_n.append(
                            plt.Line2D(
                                [0], [0], color=color, marker="o",
                                markersize=2.0, linewidth=0.9,
                            ),
                        )
                        legend_labels_n.append(
                            rf"$n{{=}}{n_train}$"
                            if n_train % 1000
                            else rf"$n{{=}}{n_train//1000}\mathrm{{k}}$"
                        )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--",
                        linewidth=0.4, alpha=0.5)

                if r == 0:
                    ax.set_title(_lam_label(lam_str), fontsize=8, pad=2.0)
                if c == ncols - 1:
                    ax.text(
                        1.02, 0.5, _kpair_label(k0, k1),
                        transform=ax.transAxes,
                        rotation=0, ha="left", va="center",
                        fontsize=plt.rcParams["axes.labelsize"],
                    )
                if r == nrows - 1:
                    ax.set_xlabel(r"$p$")
                if c == 0 and r == nrows // 2:
                    ax.set_ylabel(_METRIC_YLABEL[metric])

        # Floor-marker legend entry
        from matplotlib.lines import Line2D
        legend_handles_kind = [
            Line2D([0], [0], color="0.3", linestyle="-", marker="o",
                   markersize=2.0, linewidth=0.9,
                   label=r"$|\mathrm{mean}(\Delta_s)|$"),
            Line2D([0], [0], color="0.3", linestyle="None", marker="v",
                   markersize=4.0, label=r"SEM crosses 0"),
        ]
        legend_labels_kind = [h.get_label() for h in legend_handles_kind]

        fig.legend(
            legend_handles_n + legend_handles_kind,
            legend_labels_n + legend_labels_kind,
            loc="lower center",
            ncol=len(legend_handles_n) + len(legend_handles_kind),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
            fontsize=7,
        )

        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), pad=0.2)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    p.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output PDF (PNG sidecar also written).",
    )
    p.add_argument(
        "--k-pairs", type=str, default=None,
        help="JSON list of [K0, K1] rows; default: all distinct pairs.",
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=None,
        help="λ columns; default: all distinct λ in cache.",
    )
    p.add_argument(
        "--n-trains", type=int, nargs="+", default=None,
        help="n_train values to plot as separate lines.",
    )
    p.add_argument("--width", type=float, default=0.75)
    p.add_argument("--aspect", type=float, default=0.85)
    p.add_argument(
        "--metric", choices=["mse", "clf_err"], default="mse",
    )
    p.add_argument(
        "--floor", type=float, default=1e-4,
        help="Log-y floor; SEM-clipped points get a down-triangle marker "
             "at this y-value.  Defaults to 1e-4 (suits MSE; reduce for "
             "clf-err if needed).",
    )
    p.add_argument("--save", action="store_true", default=False)
    p.add_argument("--draft", action="store_true", default=False)
    return p.parse_args()


def _all_keys(
    finite: dict[tuple[int, int, str, int, int], dict[int, float]],
    kernel: dict[tuple[int, int, str, int], dict[int, float]],
) -> tuple[list[tuple[int, int]], list[str], list[int]]:
    k_pairs = sorted(
        {(k[0], k[1]) for k in finite}
        | {(k[0], k[1]) for k in kernel}
    )
    lambdas = sorted(
        {k[2] for k in finite} | {k[2] for k in kernel},
        key=lambda s: float(s),
    )
    n_trains = sorted(
        {k[3] for k in finite} | {k[3] for k in kernel}
    )
    return list(k_pairs), list(lambdas), list(n_trains)


def main() -> None:
    args = _parse_args()
    finite, kernel = load_per_seed(args.data_dir, metric=args.metric)
    auto_kp, auto_lam, auto_n = _all_keys(finite, kernel)

    if args.k_pairs is None:
        k_pairs = auto_kp
    else:
        k_pairs = [tuple(p) for p in json.loads(args.k_pairs)]
    if args.lambdas is None:
        lambdas = auto_lam
    else:
        lambdas = [f"{float(x):g}" for x in args.lambdas]
    if args.n_trains is None:
        n_trains = auto_n
    else:
        n_trains = list(args.n_trains)

    if args.out == _DEFAULT_OUT and args.metric == "clf_err":
        out_path = _DEFAULT_OUT_CLFERR
    else:
        out_path = args.out

    print(
        f"data_dir={args.data_dir}\n"
        f"  metric:  {args.metric}\n"
        f"  K-pairs: {k_pairs}\n"
        f"  λ:       {lambdas}\n"
        f"  n_train: {n_trains}\n"
        f"  finite cells:  {len(finite)} (sum seeds: {sum(len(d) for d in finite.values())})\n"
        f"  kernel cells:  {len(kernel)} (sum seeds: {sum(len(d) for d in kernel.values())})\n"
        f"  floor:   {args.floor:g}\n"
        f"  out:     {out_path}"
    )

    if not k_pairs or not lambdas or not n_trains:
        raise SystemExit(
            f"No data for metric={args.metric!r} in {args.data_dir}.",
        )

    make_figure(
        finite=finite, kernel=kernel,
        k_pairs=k_pairs, lambdas=lambdas, n_trains=n_trains,
        out_path=out_path, save=args.save, draft=args.draft,
        width_frac=args.width, aspect=args.aspect, metric=args.metric,
        floor=args.floor,
    )


if __name__ == "__main__":
    main()
