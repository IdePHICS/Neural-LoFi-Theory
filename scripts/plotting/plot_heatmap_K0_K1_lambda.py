"""Heatmap plotter for ``scripts/heatmap_K0_K1_lambda.py`` outputs.

Two figures per metric are emitted:

1. **Summary** (1 row × 5 columns) — first panel is the "best per cell"
   heatmap (per (K0, K1), the test metric at the lambda that minimises
   it; with a small numeric annotation showing argmin lambda).  The
   remaining four panels show heatmaps at four representative lambda
   values pulled from the sweep cache.

2. **Individual** — one heatmap per lambda, plus the best-per-cell
   heatmap (the one used as panel 1 of the summary), each saved as a
   separate ``.{pdf,png}`` pair.

Everything runs from the JSON cache written by the generator
(``results/heatmap_K0_K1_lambda/nlofik_heatmap__n{n}__K{k0}_{k1}__seed{s}.json``).
Aggregation is plain seed-mean — paired-seed semantics already give a
clean signal.

Output layout:

    imgs/heatmap_K0_K1_lambda/
        summary_mse.{pdf,png}
        summary_clferr.{pdf,png}
        individual_mse/
            best_per_cell.{pdf,png}
            lambda_1e-3.{pdf,png}
            lambda_2.15e-3.{pdf,png}
            ...
        individual_clferr/
            <same pattern>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_utils import save_or_show, setup_style  # noqa: E402


_DEFAULT_DATA_DIR = Path("results/heatmap_K0_K1_lambda")
_DEFAULT_OUT_DIR = Path("imgs/heatmap_K0_K1_lambda")

_FILE_RE = re.compile(
    r"^nlofik_heatmap__n(?P<n>\d+)__K(?P<k0>\d+)_(?P<k1>\d+)__seed"
    r"(?P<seed>\d+)\.json$"
)

_BASELINE_RE = re.compile(
    r"^baseline_no_eigen__n(?P<n>\d+)__L(?P<L>\d+)__dseed(?P<seed>\d+)\.json$"
)

# Four lambda values to surface in the summary figure (alongside the
# best-per-cell panel).  Aligned with the default 10-point log grid
# `np.logspace(-3, 0, 10)` whose elements include {1e-3, 1e-2, 1e-1, 1.0}
# at indices {0, 3, 6, 9}.  The plotter snaps to the nearest cached
# lambda when the user runs a different grid.
_SUMMARY_LAMBDAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_caches(
    data_dir: Path, n_train: int,
) -> dict[tuple[int, int, str], list[tuple[int, float, float]]]:
    """Walk *data_dir*, parse every ``nlofik_heatmap__`` cache for
    ``n_train``.  Returns a flat dict
    ``{(k0, k1, lam_str): [(seed, mse, clf_err), …]}``.
    """
    out: dict[tuple[int, int, str], list[tuple[int, float, float]]] = defaultdict(list)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        m = _FILE_RE.match(entry.name)
        if m is None:
            continue
        if int(m["n"]) != n_train:
            continue
        row = _read(Path(entry.path))
        mse_dict = row.get("test_mse", {})
        err_dict = row.get("test_clf_err", {})
        if not mse_dict:
            continue
        k0 = int(m["k0"])
        k1 = int(m["k1"])
        seed = int(m["seed"])
        for lam_str, mse_val in mse_dict.items():
            err_val = err_dict.get(lam_str, float("nan"))
            out[(k0, k1, lam_str)].append((seed, float(mse_val), float(err_val)))
    return out


def _all_keys(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
) -> tuple[list[int], list[int], list[str]]:
    k0_set = sorted({k[0] for k in cache})
    k1_set = sorted({k[1] for k in cache})
    lam_set = sorted({k[2] for k in cache}, key=float)
    return k0_set, k1_set, lam_set


def load_baselines(
    data_dir: Path, n_train: int,
) -> list[dict[str, Any]]:
    """Read every ``baseline_no_eigen__*`` cache for ``n_train``.

    Returns a list of raw JSON dicts (one per data seed); the plotter
    aggregates them in :func:`aggregate_baseline`.
    """
    out: list[dict[str, Any]] = []
    if not data_dir.is_dir():
        return out
    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        m = _BASELINE_RE.match(entry.name)
        if m is None or int(m["n"]) != n_train:
            continue
        out.append(_read(Path(entry.path)))
    return out


def aggregate_baseline(
    baselines: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Per-metric aggregation across baseline seeds.

    Returns a dict ``{metric: {best_lambda, best_mean, best_sem,
    n_seeds, per_lambda: {lam: (mean, sem)}}}`` for ``metric in
    {"mse", "clf_err"}``.  Returns ``None`` when there are no baseline
    files cached.
    """
    if not baselines:
        return None
    out: dict[str, dict[str, Any]] = {}
    for metric, key in (("mse", "test_mse"), ("clf_err", "test_clf_err")):
        # Collect per-lambda lists across seeds.
        per_lam: dict[str, list[float]] = defaultdict(list)
        for b in baselines:
            d = b.get(key, {})
            for lam_s, val in d.items():
                per_lam[lam_s].append(float(val))
        if not per_lam:
            continue
        means: dict[str, float] = {}
        sems: dict[str, float] = {}
        for lam_s, vs in per_lam.items():
            arr = np.asarray(vs, dtype=float)
            means[lam_s] = float(arr.mean())
            sems[lam_s] = (
                float(arr.std(ddof=1) / np.sqrt(arr.size))
                if arr.size > 1 else 0.0
            )
        # Argmin lambda over the seed-mean.
        best_lam = min(means.keys(), key=lambda s: means[s])
        out[metric] = {
            "best_lambda": float(best_lam),
            "best_mean": means[best_lam],
            "best_sem": sems[best_lam],
            "per_lambda": {
                lam_s: (means[lam_s], sems[lam_s]) for lam_s in per_lam
            },
            "n_seeds": len(per_lam[next(iter(per_lam))]),
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _build_grid(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *, k0_axis: list[int], k1_axis: list[int], lam_str: str, metric_idx: int,
) -> np.ndarray:
    """Seed-mean of the requested metric on the (K0, K1) grid for one lambda.

    ``metric_idx``: 1 for ``test_mse``, 2 for ``test_clf_err``.
    """
    grid = np.full((len(k0_axis), len(k1_axis)), np.nan)
    for i, k0 in enumerate(k0_axis):
        for j, k1 in enumerate(k1_axis):
            rows = cache.get((k0, k1, lam_str), [])
            if not rows:
                continue
            vals = [r[metric_idx] for r in rows]
            grid[i, j] = float(np.mean(vals))
    return grid


def _build_best_per_cell(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *, k0_axis: list[int], k1_axis: list[int], lambdas: list[str],
    metric_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(K0, K1) min over lambda of seed-mean metric, plus argmin lambda.

    Returns ``(min_grid, argmin_lambda_grid)`` of shape ``(|K0|, |K1|)``.
    ``argmin_lambda_grid`` stores the actual lambda value (float) at
    each cell.
    """
    n0, n1 = len(k0_axis), len(k1_axis)
    min_grid = np.full((n0, n1), np.nan)
    arg_grid = np.full((n0, n1), np.nan)
    lam_floats = np.array([float(s) for s in lambdas])

    per_lam_grids: list[np.ndarray] = [
        _build_grid(
            cache, k0_axis=k0_axis, k1_axis=k1_axis,
            lam_str=lam_s, metric_idx=metric_idx,
        )
        for lam_s in lambdas
    ]
    stack = np.stack(per_lam_grids, axis=-1)  # (n0, n1, n_lambda)

    for i in range(n0):
        for j in range(n1):
            col = stack[i, j]
            if np.all(np.isnan(col)):
                continue
            idx = int(np.nanargmin(col))
            min_grid[i, j] = col[idx]
            arg_grid[i, j] = lam_floats[idx]
    return min_grid, arg_grid


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _format_log_lambda(lam: float) -> str:
    """Pretty TeX-y label for a lambda value, e.g. ``\\lambda = 10^{-3}``."""
    if lam <= 0:
        return rf"$\lambda = {lam:g}$"
    log10 = np.log10(lam)
    if abs(log10 - round(log10)) < 1e-6:
        return rf"$\lambda = 10^{{{int(round(log10))}}}$"
    return rf"$\lambda = {lam:.2g}$"


def _short_lambda_tag(lam: float) -> str:
    """Short, filename-friendly tag for a lambda value (e.g. ``1e-3``)."""
    if lam <= 0:
        return f"{lam:g}"
    log10 = np.log10(lam)
    if abs(log10 - round(log10)) < 1e-6:
        return f"1e{int(round(log10)):+d}"
    return f"{lam:.3g}"


def _set_log_axes(
    ax: plt.Axes,
    k0_axis: list[int], k1_axis: list[int],
    *, show_x_label: bool = True, show_y_label: bool = True,
) -> None:
    """Use log K-axes with K0 on the horizontal and K1 on the vertical axis."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [10, 30, 100, 300, 940]
    ax.set_xticks([t for t in ticks if t >= min(k0_axis) and t <= max(k0_axis)])
    ax.set_yticks([t for t in ticks if t >= min(k1_axis) and t <= max(k1_axis)])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    ax.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    if show_x_label:
        ax.set_xlabel(r"$K_0$")
    if show_y_label:
        ax.set_ylabel(r"$K_1$")


def _draw_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    k0_axis: list[int], k1_axis: list[int],
    *,
    vmin: float | None = None, vmax: float | None = None,
    cmap: str = "viridis",
    log_color: bool = False,
) -> "plt.cm.ScalarMappable":
    """``pcolormesh`` heatmap with K0 on horizontal and K1 on vertical axes.

    Input ``grid`` is indexed as ``grid[i_K0, j_K1]`` (matching the
    convention used by the loaders).  pcolormesh expects ``z[row=y,
    col=x]``, so the array is transposed at the call site to put K0 on
    the horizontal axis.
    """
    def _log_edges(values: list[int]) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        log_v = np.log10(v)
        edges = np.empty(len(v) + 1)
        edges[1:-1] = 0.5 * (log_v[:-1] + log_v[1:])
        edges[0] = log_v[0] - (log_v[1] - log_v[0]) / 2
        edges[-1] = log_v[-1] + (log_v[-1] - log_v[-2]) / 2
        return 10 ** edges

    x_edges = _log_edges(k0_axis)
    y_edges = _log_edges(k1_axis)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_color else None

    return ax.pcolormesh(
        x_edges, y_edges, grid.T,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
        edgecolors="face", linewidth=0, rasterized=True,
    )


# ---------------------------------------------------------------------------
# Top-level plot routines
# ---------------------------------------------------------------------------


def _pick_summary_lambdas(
    available: list[str],
    targets: tuple[float, ...] = _SUMMARY_LAMBDAS,
    n_panels: int = 4,
) -> list[str]:
    """Pick ``n_panels`` distinct cached lambdas for the summary figure.

    First tries to snap each hard-coded target to its nearest cached
    value.  If that yields ``n_panels`` distinct entries, returns them.
    Otherwise falls back to ``n_panels`` log-evenly-spaced lambdas from
    the cached set so the panels span the actual sweep range without
    duplicates — handles cases like ``cache=[1e-5..1e-1]`` where the
    default target ``1.0`` would collide with ``1e-1``.
    """
    if not available:
        return []
    avail_floats = np.array([float(s) for s in available])
    log_avail = np.log10(avail_floats)

    # First pass: snap each target to the nearest cached value.
    snapped: list[str] = []
    for t in targets:
        if t <= 0:
            continue
        idx = int(np.argmin(np.abs(log_avail - np.log10(t))))
        snapped.append(available[idx])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    distinct = [x for x in snapped if not (x in seen or seen.add(x))]
    if len(distinct) >= n_panels:
        return distinct[:n_panels]

    # Fallback: ``n_panels`` log-evenly-spaced values from what's cached.
    log_grid = np.linspace(log_avail.min(), log_avail.max(), n_panels)
    fallback: list[str] = []
    fb_seen: set[str] = set()
    for log_t in log_grid:
        idx = int(np.argmin(np.abs(log_avail - log_t)))
        s = available[idx]
        if s not in fb_seen:
            fb_seen.add(s)
            fallback.append(s)
    return fallback


def _global_color_range(
    grids: list[np.ndarray],
) -> tuple[float, float]:
    """Min/max across a list of grids ignoring NaNs."""
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def _format_lambda_for_title(lam: float) -> str:
    """Compact LaTeX label for a lambda value (decade or general)."""
    if lam <= 0:
        return rf"{lam:g}"
    log10 = np.log10(lam)
    if abs(log10 - round(log10)) < 1e-6:
        return rf"10^{{{int(round(log10))}}}"
    return rf"{lam:.2g}"


def _baseline_subtitle(
    baseline_metric: dict[str, Any] | None, metric_label: str,
    *, lam_str: str | None = None,
) -> str:
    """Compact subtitle showing the no-eigenreduce baseline.

    If ``lam_str`` is given (and present in the per-lambda map), the
    subtitle shows the baseline value at that *specific* lambda — used
    for the per-lambda heatmaps so the visible baseline matches the
    panel.  Without it, the best-over-lambda baseline is shown — used
    for the best-per-cell panel.
    """
    if baseline_metric is None:
        return r"no-eigenreduce baseline: <not cached>"
    per_lam = baseline_metric.get("per_lambda", {})
    if lam_str is not None and lam_str in per_lam:
        mean, sem = per_lam[lam_str]
        return (
            rf"no-eigen baseline at $\lambda\!=\!"
            rf"{_format_lambda_for_title(float(lam_str))}$: "
            rf"{metric_label} $=\!{mean:.4f}\pm{sem:.4f}$"
            rf" ($n_{{\mathrm{{seeds}}}}={baseline_metric['n_seeds']}$)"
        )
    return (
        rf"no-eigen baseline: $\lambda^*\!=\!"
        rf"{_format_lambda_for_title(baseline_metric['best_lambda'])}$, "
        rf"{metric_label} $=\!{baseline_metric['best_mean']:.4f}\pm"
        rf"{baseline_metric['best_sem']:.4f}$"
        rf" ($n_{{\mathrm{{seeds}}}}={baseline_metric['n_seeds']}$)"
    )


def make_summary_figure(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *,
    k0_axis: list[int], k1_axis: list[int],
    lambdas_all: list[str],
    metric: str,                     # "mse" or "clf_err"
    out_path: Path,
    save: bool, draft: bool,
    baseline: dict[str, Any] | None = None,
) -> None:
    setup_style(draft=draft)
    metric_idx = 1 if metric == "mse" else 2
    metric_label = (
        r"test MSE" if metric == "mse" else r"test classification error"
    )

    summary_lambdas = _pick_summary_lambdas(lambdas_all)
    if len(summary_lambdas) < 4:
        # Pad with whatever exists at the upper end.
        summary_lambdas = summary_lambdas + lambdas_all[-(4 - len(summary_lambdas)):]
    summary_lambdas = summary_lambdas[:4]

    # Build grids.
    best_grid, arg_grid = _build_best_per_cell(
        cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas_all,
        metric_idx=metric_idx,
    )
    lam_grids = [
        _build_grid(
            cache, k0_axis=k0_axis, k1_axis=k1_axis,
            lam_str=lam_s, metric_idx=metric_idx,
        )
        for lam_s in summary_lambdas
    ]

    vmin, vmax = _global_color_range([best_grid] + lam_grids)

    fig, axes = plt.subplots(
        1, 5, figsize=(13.5, 3.2), constrained_layout=True,
    )

    # Panel 0: best per cell — second title line shows the
    # no-eigenreduce baseline at *its* optimal lambda (argmin), so the
    # comparison "is the broad valley really beating no-eigen?" is
    # immediate.
    mesh0 = _draw_heatmap(
        axes[0], best_grid, k0_axis, k1_axis,
        vmin=vmin, vmax=vmax, cmap="viridis_r",
    )
    _set_log_axes(axes[0], k0_axis, k1_axis)
    title0 = rf"min over $\lambda$ ({metric_label})"
    if baseline is not None:
        title0 += "\n" + (
            rf"baseline at $\lambda^*\!=\!"
            rf"{_format_lambda_for_title(baseline['best_lambda'])}$: "
            rf"${baseline['best_mean']:.4f}\pm"
            rf"{baseline['best_sem']:.4f}$"
        )
    axes[0].set_title(title0, fontsize=9)

    # Panels 1–4: heatmaps at chosen lambdas.  Each title carries the
    # corresponding no-eigenreduce baseline value at *that* lambda so
    # the comparison "where is the baseline on this colour scale?" is
    # immediate.
    per_lam_bl = baseline.get("per_lambda", {}) if baseline is not None else {}
    for col, (lam_s, grid) in enumerate(zip(summary_lambdas, lam_grids), 1):
        _draw_heatmap(
            axes[col], grid, k0_axis, k1_axis,
            vmin=vmin, vmax=vmax, cmap="viridis_r",
        )
        _set_log_axes(axes[col], k0_axis, k1_axis, show_y_label=(col == 0))
        title = f"{_format_log_lambda(float(lam_s))} ({metric_label})"
        if lam_s in per_lam_bl:
            mean, sem = per_lam_bl[lam_s]
            title += "\n" + (
                rf"baseline $={mean:.4f}\pm{sem:.4f}$"
            )
        axes[col].set_title(title, fontsize=9)
        axes[col].set_ylabel("")

    fig.colorbar(
        mesh0, ax=axes.tolist(), shrink=0.85, pad=0.02,
        label=metric_label,
    )

    # No suptitle: each per-lambda panel already shows its own baseline
    # value in its title (set above), and the best-per-cell panel has
    # its own clear "min over lambda" label.

    save_or_show(fig, save, out_path)


def make_individual_heatmap(
    grid: np.ndarray,
    k0_axis: list[int], k1_axis: list[int],
    *,
    title: str, metric_label: str,
    out_path: Path,
    save: bool, draft: bool,
    vmin: float | None = None, vmax: float | None = None,
    cmap: str = "viridis_r",
    baseline: dict[str, Any] | None = None,
    baseline_lam_str: str | None = None,
) -> None:
    """Single-panel heatmap.

    When ``baseline_lam_str`` is given, the suptitle shows the
    no-eigenreduce baseline value *at that specific lambda* (used for
    the per-lambda heatmaps).  Otherwise the best-over-lambda baseline
    is shown (used for the best-per-cell panel).
    """
    setup_style(draft=draft)
    fig, ax = plt.subplots(figsize=(4.0, 3.6), constrained_layout=True)
    mesh = _draw_heatmap(
        ax, grid, k0_axis, k1_axis,
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    _set_log_axes(ax, k0_axis, k1_axis)
    ax.set_title(title, fontsize=9)
    fig.colorbar(mesh, ax=ax, shrink=0.85, label=metric_label)
    if baseline is not None:
        fig.suptitle(
            _baseline_subtitle(
                baseline, metric_label, lam_str=baseline_lam_str,
            ),
            fontsize=8, y=1.02,
        )
    save_or_show(fig, save, out_path)


def make_best_lambda_heatmap(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *,
    k0_axis: list[int], k1_axis: list[int], lambdas: list[str],
    metric_idx: int, metric_name: str, n_train: int,
    out_path: Path, save: bool, draft: bool,
    cmap: str = "viridis_r",
) -> None:
    """Single-panel heatmap of the per-cell minimum over lambda.

    For each ``(k_1, k_2)`` cell, picks the lambda that minimises the
    seed-mean metric at that cell (so different cells may use different
    lambdas) and plots the resulting value.  A red star marks the cell
    with the global minimum and is annotated with the metric value
    (matches the ``best`` row in ``best_and_corner.txt``).  Title shows
    only ``N``, axes are log-scaled with minor tick markers, and
    ``clf_err`` values are displayed as percentages.
    """
    best_grid, _arg_grid = _build_best_per_cell(
        cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas,
        metric_idx=metric_idx,
    )

    is_pct = metric_name == "clf_err"
    pct_sym = r"\%" if plt.rcParams.get("text.usetex", False) else "%"

    setup_style(draft=draft)
    # NeurIPS textwidth = 5.5"; 0.32 * textwidth ≈ 1.76" wide.
    fig_w = 0.32 * 5.5
    fig_h = fig_w * (3.6 / 4.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_layout_engine(
        "constrained",
        rect=(0.0, 0.0, 1.0, 1.0),
        w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0,
    )
    mesh = _draw_heatmap(
        ax, best_grid, k0_axis, k1_axis, cmap=cmap, log_color=True,
    )

    if not np.all(np.isnan(best_grid)):
        flat_idx = int(np.nanargmin(best_grid))
        i_min, j_min = np.unravel_index(flat_idx, best_grid.shape)
        x_star = k0_axis[i_min]
        y_star = k1_axis[j_min]
        v_star = float(best_grid[i_min, j_min])
        ax.plot(
            x_star, y_star,
            marker="*", color="red", markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            linestyle="none", zorder=5,
        )
        annot = (
            f"{v_star * 100:.1f}{pct_sym}" if is_pct
            else f"{v_star:.3f}"
        )
        # Flip annotation below the star when the star is near the top
        # of the plot (avoids colliding with the title).
        log_pos = (
            (np.log10(y_star) - np.log10(min(k1_axis)))
            / (np.log10(max(k1_axis)) - np.log10(min(k1_axis)))
        )
        if log_pos > 0.7:
            xytext = (4, -3)
            va = "top"
        else:
            xytext = (4, 3)
            va = "baseline"
        ax.annotate(
            annot, xy=(x_star, y_star), xytext=xytext,
            textcoords="offset points",
            color="red", fontsize=6, va=va,
            path_effects=[
                path_effects.withStroke(
                    linewidth=1.8, foreground="white",
                ),
            ],
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(
        plt.matplotlib.ticker.LogFormatterMathtext(),
    )
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.LogFormatterMathtext(),
    )
    ax.tick_params(axis="both", which="both", direction="in", pad=2)
    ax.set_xlabel(r"$k_1$", labelpad=1)
    ax.set_ylabel(r"$k_2$", labelpad=1)
    ax.set_title(rf"$n = {n_train}$", fontsize=9, pad=2)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.yaxis.set_major_locator(
        plt.matplotlib.ticker.MaxNLocator(nbins=5),
    )
    cbar.ax.yaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    if is_pct:
        cbar.ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.PercentFormatter(
                xmax=1.0, decimals=1, symbol="",
            ),
        )
    else:
        cbar.ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.FormatStrFormatter("%.3f"),
        )
    save_or_show(fig, save, out_path)


def _summarise_best_and_corner(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *,
    k0_axis: list[int], k1_axis: list[int], lambdas: list[str],
    metric_idx: int, metric_name: str,
) -> dict[str, dict[str, object] | None]:
    """For one metric, return ``(best, no_eigenreduce_corner)`` summary.

    *best*: cell ``(K0*, K1*, lambda*)`` with the lowest seed-mean of
    ``metric``.  *no_eigenreduce_corner*: the same lookup restricted to
    the corner ``(K0_max, K1_max)`` — the closest grid point to the
    "no eigenreduction" kernel limit (eigen-reduce keeping all
    eigenvectors is mathematically the deep-NNGP no-eigen baseline).
    """
    if not k0_axis or not k1_axis:
        return {"best": None, "no_eigenreduce_corner": None}

    k0_max = max(k0_axis)
    k1_max = max(k1_axis)

    def _record(rows: list[tuple[int, float, float]]) -> tuple[float, float, int]:
        vals = [r[metric_idx] for r in rows]
        m = float(np.mean(vals))
        sem = (
            float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            if len(vals) > 1 else 0.0
        )
        return m, sem, len(vals)

    best: dict[str, object] | None = None
    corner: dict[str, object] | None = None
    best_m = float("inf")
    corner_m = float("inf")

    for (k0, k1, lam_s), rows in cache.items():
        if not rows:
            continue
        m, sem, n = _record(rows)
        if m < best_m:
            best_m = m
            best = {
                "k0": int(k0), "k1": int(k1),
                "lambda": float(lam_s),
                f"{metric_name}_mean": m,
                f"{metric_name}_sem": sem,
                "n_seeds": n,
            }
        if k0 == k0_max and k1 == k1_max and m < corner_m:
            corner_m = m
            corner = {
                "k0": int(k0), "k1": int(k1),
                "lambda": float(lam_s),
                f"{metric_name}_mean": m,
                f"{metric_name}_sem": sem,
                "n_seeds": n,
            }
    return {"best": best, "no_eigenreduce_corner": corner}


def write_best_and_corner_summary(
    cache: dict[tuple[int, int, str], list[tuple[int, float, float]]],
    *,
    k0_axis: list[int], k1_axis: list[int], lambdas: list[str],
    n_train: int, data_dir: Path,
    out_dir: Path,
    baseline: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write a human-readable text summary plus a machine-readable JSON
    of the best ``(K0, K1, lambda)`` and the no-eigenreduce-corner
    ``(K0_max, K1_max, lambda*)`` for both ``test_mse`` and
    ``test_clf_err``.

    Returns ``(txt_path, json_path)``.
    """
    summary: dict[str, object] = {
        "n_train": int(n_train),
        "data_dir": str(data_dir),
        "k0_axis": list(k0_axis),
        "k1_axis": list(k1_axis),
        "n_lambdas": len(lambdas),
        "k0_max": max(k0_axis) if k0_axis else None,
        "k1_max": max(k1_axis) if k1_axis else None,
    }
    for metric, midx in (("mse", 1), ("clf_err", 2)):
        block = _summarise_best_and_corner(
            cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas,
            metric_idx=midx, metric_name=("test_" + metric),
        )
        # Strict no-eigenreduce baseline (deep_nngp_kernel) per metric.
        if baseline is not None and metric in baseline:
            b = baseline[metric]
            metric_name = "test_" + metric
            block["no_eigenreduce_baseline"] = {  # type: ignore[index]
                "lambda": float(b["best_lambda"]),
                f"{metric_name}_mean": float(b["best_mean"]),
                f"{metric_name}_sem": float(b["best_sem"]),
                "n_seeds": int(b["n_seeds"]),
                "source": "deep_nngp_kernel(auto_rescale=True), "
                          "no signed eigen-reduce anywhere",
            }
        else:
            block["no_eigenreduce_baseline"] = None  # type: ignore[index]
        summary[metric] = block

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "best_and_corner.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plain-text human-readable version.
    lines: list[str] = []
    lines.append(
        f"Best (K0, K1, lambda) and no-eigenreduce corner — n_train={n_train}",
    )
    lines.append(f"data_dir = {data_dir}")
    lines.append(
        f"K0_max = {summary['k0_max']}, K1_max = {summary['k1_max']}, "
        f"|lambdas| = {len(lambdas)}",
    )
    lines.append("")

    for metric_label, mkey, metric_name in (
        ("test MSE", "mse", "test_mse"),
        ("test classification error", "clf_err", "test_clf_err"),
    ):
        block = summary[mkey]  # type: ignore[index]
        lines.append(f"=== {metric_label} ===")
        for kind, label in (
            ("best", "best (K0, K1, lambda)        "),
            ("no_eigenreduce_corner", "K=K=K_max corner             "),
        ):
            entry = block[kind]  # type: ignore[index]
            if entry is None:
                lines.append(f"  {label}: <not available>")
                continue
            lines.append(
                f"  {label}: K0={entry['k0']:>4d}  K1={entry['k1']:>4d}  "
                f"lambda={entry['lambda']:.3e}  "
                f"{metric_name}={entry[f'{metric_name}_mean']:.6f} "
                f"+/- {entry[f'{metric_name}_sem']:.6f}  "
                f"(n_seeds={entry['n_seeds']})",
            )
        # Strict no-eigenreduce baseline (independent deep NNGP).
        bl_entry = block.get("no_eigenreduce_baseline")  # type: ignore[index]
        if bl_entry is None:
            lines.append(
                "  no-eigenreduce baseline      : <not cached "
                "(re-run heatmap script to generate)>",
            )
        else:
            lines.append(
                f"  no-eigenreduce baseline      : "
                f"K0=  N/A  K1=  N/A  "
                f"lambda={bl_entry['lambda']:.3e}  "
                f"{metric_name}={bl_entry[f'{metric_name}_mean']:.6f} "
                f"+/- {bl_entry[f'{metric_name}_sem']:.6f}  "
                f"(n_seeds={bl_entry['n_seeds']})",
            )
            lines.append(
                f"      [source: deep_nngp_kernel(auto_rescale=True), "
                f"no eigen-reduce anywhere]",
            )
        # Improvements.
        b = block["best"]  # type: ignore[index]
        c = block["no_eigenreduce_corner"]  # type: ignore[index]
        if b is not None and c is not None:
            delta = (
                b[f"{metric_name}_mean"]  # type: ignore[index]
                - c[f"{metric_name}_mean"]
            )
            lines.append(
                f"  delta(best − K=K=K_max corner)             "
                f"= {delta:+.6f}",
            )
        if b is not None and bl_entry is not None:
            delta_bl = (
                b[f"{metric_name}_mean"]
                - bl_entry[f"{metric_name}_mean"]
            )
            lines.append(
                f"  delta(best − no-eigenreduce baseline)      "
                f"= {delta_bl:+.6f}",
            )

        # Per-lambda baseline table — one row per cached lambda, sorted
        # ascending.  Useful for cross-checking each panel of the
        # summary figure against the literal value plotted above it.
        if baseline is not None and mkey in baseline:
            per_lam = baseline[mkey].get("per_lambda", {})  # type: ignore[index]
            if per_lam:
                lines.append("  no-eigenreduce baseline at each lambda:")
                for lam_s in sorted(per_lam.keys(), key=float):
                    mean, sem = per_lam[lam_s]
                    lines.append(
                        f"      lambda={float(lam_s):.3e}  "
                        f"{metric_name}={mean:.6f} +/- {sem:.6f}",
                    )
        lines.append("")

    txt_path = out_dir / "best_and_corner.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path, json_path


def make_argmin_lambda_heatmap(
    arg_grid: np.ndarray,
    k0_axis: list[int], k1_axis: list[int],
    *,
    title: str,
    out_path: Path,
    save: bool, draft: bool,
    log_lo: float, log_hi: float,
) -> None:
    """Render the per-cell ``log10(lambda*)`` heatmap.

    Colour scale spans the ``log10`` range of the configured lambda grid
    (``log_lo`` to ``log_hi``) so the same colour means the same lambda
    across runs, regardless of whether the local optimum sits at a grid
    edge.
    """
    setup_style(draft=draft)
    log_grid = np.log10(np.where(np.isnan(arg_grid), np.nan, arg_grid))
    fig, ax = plt.subplots(figsize=(4.0, 3.6), constrained_layout=True)
    mesh = _draw_heatmap(
        ax, log_grid, k0_axis, k1_axis,
        vmin=log_lo, vmax=log_hi, cmap="plasma",
    )
    _set_log_axes(ax, k0_axis, k1_axis)
    ax.set_title(title, fontsize=9)
    fig.colorbar(
        mesh, ax=ax, shrink=0.85,
        label=r"$\log_{10} \lambda^{*}$",
    )
    save_or_show(fig, save, out_path)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--save", action="store_true", default=False)
    p.add_argument("--draft", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cache = load_caches(args.data_dir, args.n_train)
    if not cache:
        raise SystemExit(
            f"No data for n_train={args.n_train} in {args.data_dir}",
        )
    k0_axis, k1_axis, lambdas = _all_keys(cache)

    n_seeds = max(len(rows) for rows in cache.values())
    print(
        f"data_dir={args.data_dir}\n"
        f"  n_train: {args.n_train}\n"
        f"  K0 axis ({len(k0_axis)}): {k0_axis}\n"
        f"  K1 axis ({len(k1_axis)}): {k1_axis}\n"
        f"  lambdas ({len(lambdas)}): {lambdas}\n"
        f"  cells: {len(cache)} (max seeds/cell: {n_seeds})"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strict no-eigenreduce baseline (deep NNGP via deep_nngp_kernel),
    # cached one-per-seed by the generator.  None if no baseline files
    # are present in the cache directory.
    baseline_raw = load_baselines(args.data_dir, args.n_train)
    baseline = aggregate_baseline(baseline_raw)
    if baseline is None:
        print(
            "  baseline:    none cached (re-run heatmap script to add)",
        )
    else:
        for m in ("mse", "clf_err"):
            if m in baseline:
                b = baseline[m]
                print(
                    f"  baseline {m}:  λ*={b['best_lambda']:.3e}  "
                    f"value={b['best_mean']:.6f}±{b['best_sem']:.6f} "
                    f"({b['n_seeds']} seeds, no-eigenreduce)",
                )

    for metric in ("mse", "clf_err"):
        metric_idx = 1 if metric == "mse" else 2
        metric_label = (
            r"test MSE" if metric == "mse"
            else r"test classification error"
        )

        baseline_metric = baseline.get(metric) if baseline is not None else None

        # Summary figure.
        summary_path = out_dir / f"summary_{metric}.pdf"
        make_summary_figure(
            cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas_all=lambdas,
            metric=metric, out_path=summary_path,
            save=args.save, draft=args.draft,
            baseline=baseline_metric,
        )

        # Individual heatmaps.
        ind_dir = out_dir / f"individual_{metric}"
        ind_dir.mkdir(parents=True, exist_ok=True)

        all_lam_grids = [
            _build_grid(
                cache, k0_axis=k0_axis, k1_axis=k1_axis,
                lam_str=lam_s, metric_idx=metric_idx,
            )
            for lam_s in lambdas
        ]
        best_grid, arg_grid = _build_best_per_cell(
            cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas,
            metric_idx=metric_idx,
        )
        # Use shared vmin/vmax across the lambda heatmaps so they're
        # directly comparable; the best-per-cell uses its own range
        # (it's a min, so values are systematically lower).
        v_lo, v_hi = _global_color_range(all_lam_grids)
        v_lo_b, v_hi_b = _global_color_range([best_grid])

        for lam_s, grid in zip(lambdas, all_lam_grids):
            tag = _short_lambda_tag(float(lam_s))
            ind_path = ind_dir / f"lambda_{tag}.pdf"
            make_individual_heatmap(
                grid, k0_axis, k1_axis,
                title=(
                    f"{metric_label}  ·  {_format_log_lambda(float(lam_s))}"
                ),
                metric_label=metric_label,
                out_path=ind_path, save=args.save, draft=args.draft,
                vmin=v_lo, vmax=v_hi,
                baseline=baseline_metric,
                baseline_lam_str=lam_s,
            )
        make_individual_heatmap(
            best_grid, k0_axis, k1_axis,
            title=rf"min over $\lambda$ of {metric_label}",
            metric_label=metric_label,
            out_path=ind_dir / "best_per_cell.pdf",
            save=args.save, draft=args.draft,
            vmin=v_lo_b, vmax=v_hi_b,
            baseline=baseline_metric,
        )
        # Argmin lambda heatmap: which lambda achieves the per-cell min.
        lam_floats = np.array([float(s) for s in lambdas])
        make_argmin_lambda_heatmap(
            arg_grid, k0_axis, k1_axis,
            title=(
                rf"argmin $\lambda$ for {metric_label}"
            ),
            out_path=ind_dir / "argmin_lambda.pdf",
            save=args.save, draft=args.draft,
            log_lo=float(np.log10(lam_floats.min())),
            log_hi=float(np.log10(lam_floats.max())),
        )

        # Min-over-lambda single-panel heatmap with lowercase k_1, k_2
        # axes, a single colorbar, log markers, and an annotated red
        # star at the global min.
        make_best_lambda_heatmap(
            cache, k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas,
            metric_idx=metric_idx, metric_name=metric, n_train=args.n_train,
            out_path=out_dir / f"best_lambda_{metric}.pdf",
            save=args.save, draft=args.draft,
        )

    # Best-and-corner summary across both metrics — one .txt + one .json
    # in the imgs out_dir so a quick glance gives the headline numbers.
    txt_path, json_path = write_best_and_corner_summary(
        cache,
        k0_axis=k0_axis, k1_axis=k1_axis, lambdas=lambdas,
        n_train=args.n_train, data_dir=args.data_dir, out_dir=out_dir,
        baseline=baseline,
    )
    print(f"Wrote summary: {txt_path}")
    print(f"Wrote summary: {json_path}")


if __name__ == "__main__":
    main()
