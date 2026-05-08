"""Appendix figure: NLoFi-finite-RF → NLoFi-NNGP convergence.

Reads the per-evaluation JSON caches written by
``scripts/generate_nlofi_appendix_data.py`` under ``results/nlofi_appendix/``
and produces a 3 (K-pair) × 4 (λ) grid of subplots.  Each subplot shows
the chosen metric versus P (log-x) for four ``n_train`` values, with a
dashed horizontal reference per ``n_train`` for the NLoFi-NNGP limit.

The metric is selected via ``--metric``: ``mse`` (test MSE, default) or
``clf_err`` (test classification error / 0-1 loss).  When ``clf_err`` is
selected and ``--out`` is left at the default, the output filename is
adjusted automatically.

LaTeX rendering is on by default (via ``neurips.mplstyle``); use
``--draft`` for a fast, no-TeX preview.
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
_DEFAULT_OUT = Path("imgs/nlofi_appendix/nlofi_convergence.pdf")
_DEFAULT_OUT_CLFERR = Path("imgs/nlofi_appendix/nlofi_convergence_test_error.pdf")

_METRIC_YLABEL: dict[str, str] = {
    "mse": "Test MSE",
    "clf_err": "Test Error",
}

_RF_RE = re.compile(
    r"^rf__K(?P<k0>\d+)_(?P<k1>\d+)__n(?P<n>\d+)__P(?P<P>\d+)__seed"
    r"(?P<seed>\d+)\.json$"
)
_KERNEL_RE = re.compile(
    r"^kernel__K(?P<k0>\d+)_(?P<k1>\d+)__n(?P<n>\d+)__ds(?P<ds>\d+)\.json$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_caches(
    data_dir: Path,
    metric: str = "mse",
) -> tuple[
    dict[tuple[int, int, str, int, int], list[float]],  # finite[K0,K1,lam,n,P]
    dict[tuple[int, int, str, int], list[float]],       # kernel[K0,K1,lam,n]
]:
    """Walk *data_dir* once, parsing every rf__/kernel__ JSON.

    Returns nested per-key lists (one float per seed for finite, per
    data-seed for kernel).  The JSON field used is selected by
    ``metric`` (``mse`` or ``clf_err``).  Files without the requested
    field are silently skipped.
    """
    finite: dict[
        tuple[int, int, str, int, int], list[float]
    ] = defaultdict(list)
    kernel: dict[
        tuple[int, int, str, int], list[float]
    ] = defaultdict(list)

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
            metric_dict = row.get(metric, {})
            k = (
                int(m["k0"]), int(m["k1"]),
                int(m["n"]), int(m["P"]),
            )
            for lam_str, val in metric_dict.items():
                finite[(k[0], k[1], lam_str, k[2], k[3])].append(float(val))
        elif name.startswith("kernel__"):
            m = _KERNEL_RE.match(name)
            if m is None:
                continue
            row = _read(Path(entry.path))
            metric_dict = row.get(metric, {})
            k = (
                int(m["k0"]), int(m["k1"]),
                int(m["n"]),
            )
            for lam_str, val in metric_dict.items():
                kernel[(k[0], k[1], lam_str, k[2])].append(float(val))

    return finite, kernel


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_sem(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return (
        float(arr.mean()),
        float(arr.std(ddof=1) / np.sqrt(arr.size)),
    )


def aggregate_finite(
    finite: dict[tuple[int, int, str, int, int], list[float]],
    *, k0: int, k1: int, lam_str: str, n_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P, mean, sem, count) for one (K0, K1, λ, n) cell."""
    rows = []
    for (kk0, kk1, lam, nn, P), xs in finite.items():
        if kk0 == k0 and kk1 == k1 and lam == lam_str and nn == n_train:
            mean, sem = _mean_sem(xs)
            rows.append((P, mean, sem, len(xs)))
    rows.sort()
    if not rows:
        return (
            np.array([]), np.array([]), np.array([]), np.array([], dtype=int),
        )
    P_arr = np.array([r[0] for r in rows])
    mean = np.array([r[1] for r in rows])
    sem = np.array([r[2] for r in rows])
    count = np.array([r[3] for r in rows], dtype=int)
    return P_arr, mean, sem, count


def aggregate_kernel(
    kernel: dict[tuple[int, int, str, int], list[float]],
    *, k0: int, k1: int, lam_str: str, n_train: int,
) -> tuple[float, float, int]:
    xs = kernel.get((k0, k1, lam_str, n_train), [])
    mean, sem = _mean_sem(xs)
    return mean, sem, len(xs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _lam_label(lam_str: str) -> str:
    """Convert the JSON key (``%g`` formatted) to a LaTeX math string."""
    val = float(lam_str)
    if val >= 1.0 and val.is_integer():
        return rf"$\lambda = {int(val)}$"
    if val < 1.0:
        # Render small λ as 10^{...} when it's a clean power of 10
        log10 = np.log10(val)
        if abs(log10 - round(log10)) < 1e-9:
            return rf"$\lambda = 10^{{{int(round(log10))}}}$"
    return rf"$\lambda = {val:g}$"


def _kpair_label(k0: int, k1: int) -> str:
    if k0 == k1:
        return rf"$k_1{{=}}k_2{{=}}{k0}$"
    return rf"$k_1{{=}}{k0},k_2{{=}}{k1}$"


def make_figure(
    *,
    finite: dict[tuple[int, int, str, int, int], list[float]],
    kernel: dict[tuple[int, int, str, int], list[float]],
    k_pairs: list[tuple[int, int]],
    lambdas: list[str],
    n_trains: list[int],
    out_path: Path,
    save: bool,
    draft: bool,
    width_frac: float,
    aspect: float,
    metric: str,
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
                    P_arr, mean, sem, _ = aggregate_finite(
                        finite, k0=k0, k1=k1, lam_str=lam_str, n_train=n_train,
                    )
                    if P_arr.size:
                        line, = ax.plot(
                            P_arr, mean, marker="o", markersize=2.0,
                            linewidth=0.9, color=color,
                        )
                        ax.fill_between(
                            P_arr, mean - sem, mean + sem,
                            color=color, alpha=0.2, linewidth=0,
                        )
                        if r == 0 and c == 0:
                            legend_handles_n.append(line)
                            legend_labels_n.append(
                                rf"$n{{=}}{n_train}$" if n_train % 1000 else rf"$n{{=}}{n_train//1000}\mathrm{{k}}$"
                            )

                    kmean, ksem, _ = aggregate_kernel(
                        kernel, k0=k0, k1=k1, lam_str=lam_str, n_train=n_train,
                    )
                    if not np.isnan(kmean):
                        ax.axhline(
                            kmean, color=color, linestyle="--",
                            linewidth=0.8, alpha=0.85,
                        )
                        if ksem > 0:
                            ax.axhspan(
                                kmean - ksem, kmean + ksem,
                                color=color, alpha=0.10, linewidth=0,
                            )

                ax.set_xscale("log")
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
                # Anchor the y-axis label to the middle row's leftmost
                # panel so it stays aligned with the tick labels regardless
                # of figure width / aspect.
                if c == 0 and r == nrows // 2:
                    ax.set_ylabel(_METRIC_YLABEL[metric])

        from matplotlib.lines import Line2D
        legend_handles_kind = [
            Line2D([0], [0], color="0.3", linestyle="-", marker="o",
                   markersize=2.0, linewidth=0.9,
                   label=r"finite RF"),
            Line2D([0], [0], color="0.3", linestyle="--", linewidth=0.8,
                   label=r"NLoFi-NNGP"),
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
        # Make sharex/sharey panels touch — zero spacing between subplots.
        fig.subplots_adjust(wspace=0.0, hspace=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
        help="Directory with rf__*.json and kernel__*.json caches.",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output PDF path (PNG sidecar also written).",
    )
    parser.add_argument(
        "--k-pairs", type=str, default=None,
        help=(
            "JSON list of [K0, K1] rows; default: all distinct pairs found "
            "in the cache directory, sorted by (K0, K1)."
        ),
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=None,
        help=(
            "λ columns; default: all distinct λ found in the cache "
            "directory, sorted ascending."
        ),
    )
    parser.add_argument(
        "--n-trains", type=int, nargs="+", default=None,
        help=(
            "n_train values to plot as separate lines; default: all "
            "distinct n_train found in the cache, sorted ascending."
        ),
    )
    parser.add_argument(
        "--width", type=float, default=0.75,
        help=(
            "Figure width as a fraction of NeurIPS textwidth (5.5 in); "
            "default 0.75 (= 4.125 in).  Use a smaller value for a more "
            "compressed figure."
        ),
    )
    parser.add_argument(
        "--aspect", type=float, default=0.85,
        help="Per-panel aspect (height / width); default 0.85.",
    )
    parser.add_argument(
        "--metric", choices=["mse", "clf_err"], default="mse",
        help=(
            "Which metric to plot: 'mse' for test MSE (default) or "
            "'clf_err' for test classification error (0-1 loss)."
        ),
    )
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--draft", action="store_true", default=False,
        help="Disable LaTeX rendering for fast preview.",
    )
    return parser.parse_args()


def _all_keys(
    finite: dict[tuple[int, int, str, int, int], list[float]],
    kernel: dict[tuple[int, int, str, int], list[float]],
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
    finite, kernel = load_caches(args.data_dir, metric=args.metric)

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
        f"  finite cache rows: {sum(len(v) for v in finite.values())}\n"
        f"  kernel cache rows: {sum(len(v) for v in kernel.values())}\n"
        f"  out:     {out_path}"
    )

    if not k_pairs or not lambdas or not n_trains:
        raise SystemExit(
            f"No data found for metric={args.metric!r} in {args.data_dir}. "
            f"Existing JSON caches may not contain this field; "
            f"re-run scripts/generate_nlofi_appendix_data.py to refresh them."
        )

    make_figure(
        finite=finite, kernel=kernel,
        k_pairs=k_pairs, lambdas=lambdas, n_trains=n_trains,
        out_path=out_path, save=args.save, draft=args.draft,
        width_frac=args.width, aspect=args.aspect, metric=args.metric,
    )


if __name__ == "__main__":
    main()
