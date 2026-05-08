"""NLoFi (min-over-K K-grid) vs deep-NNGP horizontal references.

Single panel.  For each ``n_train``:
* a finite-RF curve where each point is the *minimum* test MSE over the
  diagonal K-grid for that ``P`` (mean ± SEM over RF seeds), and
* a horizontal dashed line for the deep-NNGP test MSE at the same
  ``n_train`` (mean ± SEM over data seeds, shaded band).

Reads the JSON caches written by
``scripts/generate_nlofi_vs_deep_nngp.py``.
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
from _plot_utils import save_or_show, setup_style  # noqa: E402


_DEFAULT_DATA_DIR = Path("results/nlofi_vs_deep_nngp")
_DEFAULT_OUT = Path("imgs/nlofi_vs_deep_nngp/nlofi_vs_deep_nngp.pdf")

_NLOFI_DIAG_RE = re.compile(
    r"^nlofi__n(?P<n>\d+)__K(?P<k>\d+)__P(?P<P>\d+)__seed"
    r"(?P<seed>\d+)__lam(?P<lam>[^.]+(?:\.[^.]+)*)\.json$"
)
_NLOFI_2D_RE = re.compile(
    r"^nlofi2d__n(?P<n>\d+)__K(?P<k0>\d+)_(?P<k1>\d+)__P(?P<P>\d+)__seed"
    r"(?P<seed>\d+)__lam(?P<lam>[^.]+(?:\.[^.]+)*)\.json$"
)
_DEEP_RE = re.compile(
    r"^deep_nngp__n(?P<n>\d+)__ds(?P<ds>\d+)__lam(?P<lam>[^.]+(?:\.[^.]+)*)"
    r"\.json$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_caches(
    data_dir: Path, lam: float,
) -> tuple[
    dict[tuple[int, int, int, int], list[tuple[int, float]]],
    # nlofi[n, K0, K1, P] = [(seed, mse)]
    dict[int, list[float]],  # deep[n] = [mse, ...]
]:
    nlofi: dict[
        tuple[int, int, int, int], list[tuple[int, float]]
    ] = defaultdict(list)
    deep: dict[int, list[float]] = defaultdict(list)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")
    target_lam = float(lam)
    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith("nlofi2d__"):
            m = _NLOFI_2D_RE.match(name)
            if m is None:
                continue
            try:
                if abs(float(m["lam"]) - target_lam) > 1e-12:
                    continue
            except ValueError:
                continue
            row = _read(Path(entry.path))
            nlofi[(
                int(m["n"]), int(m["k0"]), int(m["k1"]), int(m["P"]),
            )].append((int(m["seed"]), float(row["test_mse"])))
        elif name.startswith("nlofi__"):
            m = _NLOFI_DIAG_RE.match(name)
            if m is None:
                continue
            try:
                if abs(float(m["lam"]) - target_lam) > 1e-12:
                    continue
            except ValueError:
                continue
            row = _read(Path(entry.path))
            k = int(m["k"])
            nlofi[(int(m["n"]), k, k, int(m["P"]))].append(
                (int(m["seed"]), float(row["test_mse"])),
            )
        elif name.startswith("deep_nngp__"):
            m = _DEEP_RE.match(name)
            if m is None:
                continue
            try:
                if abs(float(m["lam"]) - target_lam) > 1e-12:
                    continue
            except ValueError:
                continue
            row = _read(Path(entry.path))
            deep[int(m["n"])].append(float(row["test_mse"]))
    return nlofi, deep


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


def aggregate_min_over_kpair(
    nlofi: dict[tuple[int, int, int, int], list[tuple[int, float]]],
    *, n_train: int, p_values: list[int],
    k_pairs: list[tuple[int, int]] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]],
    list[list[tuple[int, int, float, float]]],
]:
    """Per-P min over (K0, K1) of the seed-mean MSE.

    Returns:
      ``P_arr``       – array of P values that have data,
      ``best_mean``   – seed-mean MSE at the (K0, K1) that argmins per-P,
      ``best_sem``    – SEM at that (K0, K1),
      ``best_kpair``  – the chosen (K0, K1) per P,
      ``per_p_table`` – list (per P) of [(K0, K1, mean, sem), …] over every
                        (K0, K1) seen in the cache for that (n, P).
    """
    P_arr = np.array(p_values, dtype=int)
    best_mean = np.full(len(p_values), np.nan)
    best_sem = np.full(len(p_values), np.nan)
    best_kpair: list[tuple[int, int]] = [(-1, -1)] * len(p_values)
    per_p_table: list[list[tuple[int, int, float, float]]] = [
        [] for _ in p_values
    ]

    if k_pairs is None:
        candidate_pairs_per_p: list[list[tuple[int, int]]] = []
        for P in p_values:
            kps = sorted({
                (k0, k1) for (n, k0, k1, p) in nlofi
                if n == n_train and p == P
            })
            candidate_pairs_per_p.append(kps)
    else:
        candidate_pairs_per_p = [list(k_pairs)] * len(p_values)

    for ip, P in enumerate(p_values):
        for k0, k1 in candidate_pairs_per_p[ip]:
            xs = [v for _, v in nlofi.get((n_train, k0, k1, P), [])]
            if not xs:
                continue
            mean, sem = _mean_sem(xs)
            per_p_table[ip].append((k0, k1, mean, sem))
        if not per_p_table[ip]:
            continue
        idx = int(np.argmin([row[2] for row in per_p_table[ip]]))
        k0, k1, mean, sem = per_p_table[ip][idx]
        best_kpair[ip] = (k0, k1)
        best_mean[ip] = mean
        best_sem[ip] = sem
    return P_arr, best_mean, best_sem, best_kpair, per_p_table


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def make_figure(
    *,
    nlofi: dict[tuple[int, int, int, int], list[tuple[int, float]]],
    deep: dict[int, list[float]],
    n_trains: list[int],
    p_values: list[int],
    lam: float,
    out_path: Path,
    save: bool,
    draft: bool,
    fig_width_in: float,
    aspect: float,
    show_all_k: bool,
    annotate_best: bool,
) -> None:
    setup_style(draft=draft)

    fig_h = fig_width_in * aspect
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_h))

    cmap = plt.get_cmap("viridis")
    n_colors = max(len(n_trains), 2)
    n_train_color = {
        n: cmap(0.15 + 0.7 * (i / max(n_colors - 1, 1)))
        for i, n in enumerate(n_trains)
    }

    legend_handles_n: list = []
    legend_labels_n: list[str] = []

    for n_train in n_trains:
        color = n_train_color[n_train]
        P_arr, mean, sem, best_kpair, per_p_table = aggregate_min_over_kpair(
            nlofi, n_train=n_train, p_values=p_values,
        )
        valid = ~np.isnan(mean)
        if valid.any():
            line, = ax.plot(
                P_arr[valid], mean[valid], marker="o", markersize=4,
                linewidth=1.6, color=color,
            )
            ax.fill_between(
                P_arr[valid],
                mean[valid] - sem[valid], mean[valid] + sem[valid],
                color=color, alpha=0.22, linewidth=0,
            )
            legend_handles_n.append(line)
            legend_labels_n.append(rf"$n_{{\mathrm{{train}}}} = {n_train}$")

            if annotate_best:
                for ip in np.where(valid)[0]:
                    k0, k1 = best_kpair[ip]
                    ax.annotate(
                        rf"$({k0},{k1})$",
                        xy=(P_arr[ip], mean[ip]),
                        xytext=(2, 4), textcoords="offset points",
                        fontsize=6, color=color, alpha=0.9,
                    )

            if show_all_k:
                # Faint individual (K0, K1) curves for this n_train.
                kp_curves: dict[tuple[int, int], list[tuple[int, float]]] = {}
                for ip, P in enumerate(p_values):
                    for k0, k1, m, _ in per_p_table[ip]:
                        kp_curves.setdefault((k0, k1), []).append((P, m))
                for kp, pts in kp_curves.items():
                    pts.sort()
                    Ps = [pp for pp, _ in pts]
                    Ms = [mm for _, mm in pts]
                    ax.plot(
                        Ps, Ms, color=color, linewidth=0.6, alpha=0.25,
                        linestyle=":", zorder=1,
                    )

        d_xs = deep.get(n_train, [])
        if d_xs:
            d_mean, d_sem = _mean_sem(d_xs)
            ax.axhline(
                d_mean, color=color, linestyle="--",
                linewidth=1.0, alpha=0.85,
            )
            if d_sem > 0:
                ax.axhspan(
                    d_mean - d_sem, d_mean + d_sem,
                    color=color, alpha=0.10, linewidth=0,
                )

    ax.set_xscale("log")
    ax.set_xlabel(r"$P$")
    ax.set_ylabel(r"test MSE")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    if lam < 1.0:
        log10 = np.log10(lam)
        if abs(log10 - round(log10)) < 1e-9:
            lam_lbl = rf"\lambda = 10^{{{int(round(log10))}}}"
        else:
            lam_lbl = rf"\lambda = {lam:g}"
    else:
        lam_lbl = rf"\lambda = {lam:g}"
    ax.set_title(
        rf"NLoFi $\min_{{K_0, K_1}}$ vs deep NNGP, ${lam_lbl}$",
    )

    from matplotlib.lines import Line2D
    legend_handles_kind = [
        Line2D([0], [0], color="0.3", linestyle="-", marker="o",
               markersize=4, linewidth=1.6,
               label=r"NLoFi (best $K_0, K_1$)"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.0,
               label=r"deep NNGP"),
    ]
    legend_labels_kind = [h.get_label() for h in legend_handles_kind]

    ax.legend(
        legend_handles_n + legend_handles_kind,
        legend_labels_n + legend_labels_kind,
        loc="best", ncol=2, frameon=False, fontsize=8,
    )

    fig.tight_layout()
    save_or_show(fig, save, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
    )
    p.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
    )
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument(
        "--n-trains", type=int, nargs="+", default=None,
        help="Default: all distinct n_train in the cache.",
    )
    p.add_argument(
        "--p-values", type=int, nargs="+", default=None,
        help="Default: all distinct P in the cache, sorted ascending.",
    )
    p.add_argument("--width", type=float, default=5.5)
    p.add_argument("--aspect", type=float, default=0.7)
    p.add_argument(
        "--show-all-k", action="store_true",
        help="Overlay faint dotted lines for every (K0, K1) in the cache.",
    )
    p.add_argument(
        "--annotate-best", action="store_true",
        help="Print the best (K0, K1) at each P next to the marker.",
    )
    p.add_argument("--save", action="store_true", default=False)
    p.add_argument("--draft", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    nlofi, deep = load_caches(args.data_dir, args.lam)

    auto_n = sorted({k[0] for k in nlofi} | set(deep.keys()))
    auto_p = sorted({k[3] for k in nlofi})

    n_trains = list(args.n_trains) if args.n_trains else auto_n
    p_values = list(args.p_values) if args.p_values else auto_p

    distinct_kpairs = sorted({(k[1], k[2]) for k in nlofi})

    print(
        f"data_dir={args.data_dir}\n"
        f"  λ:       {args.lam}\n"
        f"  n_train: {n_trains}\n"
        f"  K-pairs: {distinct_kpairs}\n"
        f"  P:       {p_values}\n"
        f"  nlofi rows: {sum(len(v) for v in nlofi.values())}\n"
        f"  deep rows:  {sum(len(v) for v in deep.values())}"
    )

    make_figure(
        nlofi=nlofi, deep=deep,
        n_trains=n_trains, p_values=p_values,
        lam=args.lam, out_path=args.out, save=args.save, draft=args.draft,
        fig_width_in=args.width, aspect=args.aspect,
        annotate_best=args.annotate_best,
        show_all_k=args.show_all_k,
    )


if __name__ == "__main__":
    main()
