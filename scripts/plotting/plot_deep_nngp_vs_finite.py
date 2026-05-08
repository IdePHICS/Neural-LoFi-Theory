"""Overlay the standard deep-NNGP curves on the finite-p=10k RF reference.

Loads per-seed JSONs produced by ``scripts/train_deep_nngp.py`` under
``results/train_deep_nngp/<dataset>/layers_<L>/`` and the corresponding
finite-p no-eigenreduction baselines from
``results/train_ridge_spectral/<dataset>/layers_<L>/``.

The goal of this plot is a validation check: at every ``n_train``, the
deep NNGP curve at depth ``L`` should agree with the finite ``p``
no-eigen baseline at the same depth, to within SEM, as ``p`` grows.

Usage
-----
python scripts/plotting/plot_deep_nngp_vs_finite.py --save
python scripts/plotting/plot_deep_nngp_vs_finite.py --layers 1 3 --p 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_utils import save_or_show, setup_style  # noqa: E402

_RESULTS_ROOT = Path("results")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


# Deep-NNGP filename schema (unified with kernel-spectral; see
# ``_kernel_spectral_core._layer_token``).  We only parse down to the
# run-identifying fields; per-layer tokens are parsed separately because
# their count varies.
_DEEP_NNGP_RE = re.compile(
    r"^(?P<preset>\w+)"
    r"_seed(?P<seed>\d+)"
    r"_ntrain(?P<ntrain>\d+)"
    r"_ntest(?P<ntest>\d+)"
    r"_(?P<layers_str>[^_].*?)"
    r"_ridge(?P<ridge>[-\d.a-z]+)\.json$"
)

_LAYER_TOKEN_RE = re.compile(
    r"^(?P<kernel>[a-z]+)sw(?P<sw>[\d.]+)sb(?P<sb>[\d.]+)q(?P<nq>\d+)"
    r"(?:m(?P<m>\d+)|noeig)$"
)


# Finite-p no-eigen schema with repeated layer-tokens (one per layer).
def _make_noeig_regex(n_layers: int) -> re.Pattern:
    layer_pat = r"p(?P<p{i}>\d+)(?P<act{i}>[a-z]+)_noeig"
    parts = "-".join(layer_pat.format(i=i) for i in range(n_layers))
    return re.compile(
        r"^(?P<preset>\w+)"
        r"_seed(?P<seed>\d+)"
        r"_ntrain(?P<ntrain>\d+)"
        r"_ntest(?P<ntest>\d+)"
        rf"_{parts}"
        r"_ridge(?P<ridge>.+)\.json$"
    )


def _read_metric(path: Path, key: str) -> float | None:
    try:
        with open(path) as f:
            return float(json.load(f)["final"][key])
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _filter_finite_p(
    match: re.Match, *, p: int, activation: str, n_layers: int
) -> bool:
    for i in range(n_layers):
        if int(match.group(f"p{i}")) != p:
            return False
        if match.group(f"act{i}") != activation:
            return False
    return True


def _parse_layer_tokens(
    layers_str: str,
) -> list[dict[str, str]] | None:
    """Split and parse the hyphen-separated layer tokens; return None on failure."""
    parsed: list[dict[str, str]] = []
    for tok in layers_str.split("-"):
        m = _LAYER_TOKEN_RE.match(tok)
        if m is None:
            return None
        parsed.append(m.groupdict())
    return parsed


def load_deep_nngp(
    result_dir: Path,
    *,
    n_trains: list[int],
    metric_key: str,
    sigma_w: float,
    sigma_b: float,
    n_quad: int,
    n_layers: int,
) -> list[dict]:
    """Per-n_train values for deep-NNGP JSONs matching the kernel config.

    Only files whose every layer-token parses as ``noeig`` with matching
    ``(sigma_w, sigma_b, n_quad)`` and length ``n_layers`` are kept.
    """
    by_n: dict[int, list[float]] = {n: [] for n in n_trains}
    if not result_dir.is_dir():
        return []
    sw_s = f"{float(sigma_w)}"
    sb_s = f"{float(sigma_b)}"
    for entry in result_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        m = _DEEP_NNGP_RE.match(entry.name)
        if m is None:
            continue
        toks = _parse_layer_tokens(m["layers_str"])
        if toks is None or len(toks) != n_layers:
            continue
        if not all(
            t["m"] is None
            and t["sw"] == sw_s
            and t["sb"] == sb_s
            and int(t["nq"]) == int(n_quad)
            for t in toks
        ):
            continue
        n = int(m["ntrain"])
        if n not in by_n:
            continue
        v = _read_metric(entry, metric_key)
        if v is not None:
            by_n[n].append(v)
    return [
        {"n_train": n, "values": np.array(vs)}
        for n, vs in sorted(by_n.items())
        if len(vs) > 0
    ]


def load_finite_p_noeig(
    result_dir: Path,
    *,
    n_trains: list[int],
    p_filter: int,
    activation: str,
    n_layers: int,
    metric_key: str,
) -> list[dict]:
    regex = _make_noeig_regex(n_layers)
    by_n: dict[int, list[float]] = {n: [] for n in n_trains}
    if not result_dir.is_dir():
        return []
    for entry in result_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        m = regex.match(entry.name)
        if m is None:
            continue
        if not _filter_finite_p(
            m, p=p_filter, activation=activation, n_layers=n_layers,
        ):
            continue
        n = int(m["ntrain"])
        if n not in by_n:
            continue
        v = _read_metric(entry, metric_key)
        if v is not None:
            by_n[n].append(v)
    return [
        {"n_train": n, "values": np.array(vs)}
        for n, vs in sorted(by_n.items())
        if len(vs) > 0
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    if len(values) <= 1:
        return float(np.mean(values)), 0.0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1) / np.sqrt(len(values))),
    )


def _plot_curve(
    ax: plt.Axes,
    series: list[dict],
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    transform=lambda v: v,  # noqa: E731
) -> None:
    if not series:
        return
    ns = np.array([r["n_train"] for r in series], dtype=float)
    transformed = [transform(r["values"]) for r in series]
    means = np.array([v.mean() for v in transformed])
    sems = np.array([
        v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        for v in transformed
    ])
    line, = ax.plot(
        ns, means, marker=marker, markersize=5, linewidth=1.6,
        linestyle=linestyle, color=color, label=label,
    )
    ax.fill_between(ns, means - sems, means + sems, alpha=0.2, color=line.get_color())


def plot_overlay(
    *,
    deep_nngp_mse: dict[int, list[dict]],
    deep_nngp_acc: dict[int, list[dict]],
    finite_mse: dict[tuple[int, int], list[dict]],
    finite_acc: dict[tuple[int, int], list[dict]],
    p_refs: list[int],
    activation: str,
    dataset: str,
    save_path: Path | None,
    save: bool,
    draft: bool,
) -> None:
    setup_style(draft=draft)

    fig, (ax_mse, ax_err) = plt.subplots(
        1, 2, figsize=(12, 4.2), constrained_layout=True,
    )

    # Distinct colours per (depth, source).  NNGP gets a solid colour,
    # each finite-p reference its own colour + linestyle so curves are
    # individually readable when they overlap.
    nngp_colors = {1: "C3", 2: "C1", 3: "C0", 4: "C2"}
    p_colors = ["C0", "C2", "C4", "C5"]  # cycled across p values
    p_styles = ["--", ":", "-.", (0, (3, 1, 1, 1))]

    for L in sorted(deep_nngp_mse):
        nngp_color = nngp_colors.get(L, f"C{L}")
        label = f"Deep NNGP ({L}L)"
        _plot_curve(
            ax_mse, deep_nngp_mse[L], label=label,
            color=nngp_color, marker="o",
        )
        _plot_curve(
            ax_err, deep_nngp_acc[L], label=label,
            color=nngp_color, marker="o",
            transform=lambda v: 100.0 * (1.0 - v),
        )
        for i, p in enumerate(p_refs):
            key = (L, p)
            color = p_colors[i % len(p_colors)]
            linestyle = p_styles[i % len(p_styles)]
            lbl = f"Finite $p={p}$ ({L}L RF)"
            if key in finite_mse:
                _plot_curve(
                    ax_mse, finite_mse[key], label=lbl,
                    color=color, marker="s", linestyle=linestyle,
                )
            if key in finite_acc:
                _plot_curve(
                    ax_err, finite_acc[key], label=lbl,
                    color=color, marker="s", linestyle=linestyle,
                    transform=lambda v: 100.0 * (1.0 - v),
                )

    for ax, ylabel in [(ax_mse, "Test MSE"), (ax_err, "Test error (\\%)")]:
        ax.set_xlabel("$n_{\\mathrm{train}}$")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
        ax.legend(fontsize=8, ncol=2)
    ax_err.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_err.yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))

    ps_str = "_".join(f"p{p}" for p in p_refs)
    fig.suptitle(
        f"Deep NNGP ({activation}) vs finite {', '.join(f'p={p}' for p in p_refs)}"
        f" — {dataset}"
    )

    default_out = (
        Path("imgs") / "deep_nngp_vs_finite"
        / f"deep_nngp_vs_{ps_str}_{dataset}_{activation}.pdf"
    )
    save_or_show(fig, save, save_path or default_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the standard deep-NNGP kernel against the finite-p "
            "no-eigenreduction reference."
        )
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[1, 3],
        help="Depths to overlay (default: 1 3).",
    )
    parser.add_argument(
        "--p", type=int, nargs="+", default=[5000, 10_000],
        help="Finite reference p values (default: 5000 10000).",
    )
    parser.add_argument(
        "--activation", default="sigmoid",
        help="Activation name (must match both NNGP kernel and finite refs).",
    )
    parser.add_argument("--sigma_w", type=float, default=1.0)
    parser.add_argument("--sigma_b", type=float, default=0.0)
    parser.add_argument("--n_quad", type=int, default=20)
    parser.add_argument(
        "--n_trains", type=int, nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000],
    )
    parser.add_argument("--deep_nngp_dir", type=Path, default=None)
    parser.add_argument("--baseline_dir", type=Path, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--draft", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    deep_nngp_base = args.deep_nngp_dir or (
        _RESULTS_ROOT / "train_deep_nngp" / args.dataset
    )
    finite_base = args.baseline_dir or (
        _RESULTS_ROOT / "train_ridge_spectral" / args.dataset
    )

    deep_nngp_mse: dict[int, list[dict]] = {}
    deep_nngp_acc: dict[int, list[dict]] = {}
    finite_mse: dict[tuple[int, int], list[dict]] = {}
    finite_acc: dict[tuple[int, int], list[dict]] = {}

    p_label = ", ".join(f"p={p}" for p in args.p)
    print(
        f"\nDeep NNGP ({args.activation}, sigma_w={args.sigma_w}, "
        f"sigma_b={args.sigma_b}, n_quad={args.n_quad}) vs {p_label}\n"
    )

    for L in args.layers:
        d_mse = load_deep_nngp(
            deep_nngp_base / f"layers_{L}",
            n_trains=args.n_trains, metric_key="test_mse",
            sigma_w=args.sigma_w, sigma_b=args.sigma_b, n_quad=args.n_quad,
            n_layers=L,
        )
        d_acc = load_deep_nngp(
            deep_nngp_base / f"layers_{L}",
            n_trains=args.n_trains, metric_key="test_accuracy",
            sigma_w=args.sigma_w, sigma_b=args.sigma_b, n_quad=args.n_quad,
            n_layers=L,
        )
        deep_nngp_mse[L] = d_mse
        deep_nngp_acc[L] = d_acc

        for p in args.p:
            f_mse = load_finite_p_noeig(
                finite_base / f"layers_{L}",
                n_trains=args.n_trains, p_filter=p,
                activation=args.activation, n_layers=L, metric_key="test_mse",
            )
            f_acc = load_finite_p_noeig(
                finite_base / f"layers_{L}",
                n_trains=args.n_trains, p_filter=p,
                activation=args.activation, n_layers=L,
                metric_key="test_accuracy",
            )
            finite_mse[(L, p)] = f_mse
            finite_acc[(L, p)] = f_acc

        print(f"L={L}:")
        for r_n in d_mse:
            n = r_n["n_train"]
            n_mean, n_sem = _mean_sem(r_n["values"])
            parts = [
                f"n={n:>6}  nngp={n_mean:.4f}±{n_sem:.4f} (seeds={len(r_n['values'])})"
            ]
            for p in args.p:
                row = next(
                    (x for x in finite_mse[(L, p)] if x["n_train"] == n), None,
                )
                if row is not None:
                    f_mean, f_sem = _mean_sem(row["values"])
                    parts.append(
                        f"p={p}={f_mean:.4f}±{f_sem:.4f} Δ={n_mean - f_mean:+.4f}"
                    )
            print("  " + "  |  ".join(parts))

    if not any(deep_nngp_mse.values()):
        print("\nNo deep-NNGP results found; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plot_overlay(
        deep_nngp_mse=deep_nngp_mse, deep_nngp_acc=deep_nngp_acc,
        finite_mse=finite_mse, finite_acc=finite_acc,
        p_refs=args.p, activation=args.activation, dataset=args.dataset,
        save_path=args.output, save=args.save, draft=args.draft,
    )


if __name__ == "__main__":
    main()
