"""Plot eigenvalue-weighted mean Jacobian importance across layers.

Reads precomputed JSON outputs from ``compute_data_driven_features`` /
``covariance_spectrum_eigenvectors`` and renders a single-row figure where
each panel shows the per-pixel importance map for one layer (Input, Layer 1,
Layer 2, ...). Per-layer min-max normalization makes the spatial structure
comparable across panels.

Usage
-----
PYTHONPATH=scripts/plotting python scripts/plotting/plot_weighted_mean_jacobian.py \\
    --run_dir results/covariance_spectrum_eigenvectors/celeba/high_cheekbones_ntrain200000_P50000_layers3_keep100-50-20_relu_seed0 \\
    --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _plot_utils import (
    NeurIPSFigure,
    discover_layers,
    infer_image_shape,
    load_json,
    parse_run_dirname,
    reshape_to_image,
)


def _compute_weighted_importance(
    run_dir: Path,
    layer: str,
    matrix_type: str,
    n_k: int | None,
) -> np.ndarray | None:
    """Return the eigenvalue-weighted importance map for *layer* as a 2-D array."""
    eigvec_path = run_dir / f"top_eigenvectors_{layer}_{matrix_type}.json"
    if not eigvec_path.exists():
        return None
    edata = load_json(eigvec_path)
    evals = np.abs(np.array(edata["eigenvalues"]))

    if layer == "input":
        vecs = np.array(edata["eigenvectors"])
        k_avail = min(vecs.shape[1], len(evals))
        k_use = k_avail if n_k is None else min(k_avail, n_k)
        weights = evals[:k_use]
        # sum_k |λ_k| * v_k(d)^2  (input importance has no precomputed table)
        imp = (vecs[:, :k_use] ** 2) @ weights
        total = float(weights.sum())
    else:
        jac_path = run_dir / f"jacobian_importance_{layer}_{matrix_type}.json"
        if not jac_path.exists():
            return None
        jdata = load_json(jac_path)
        k_keys = sorted(
            (k for k in jdata if k.startswith("k")),
            key=lambda s: int(s[1:]),
        )
        if not k_keys:
            return None
        k_avail = min(len(k_keys), len(evals))
        k_use = k_avail if n_k is None else min(k_avail, n_k)
        weights = evals[:k_use]
        stacked = np.stack(
            [np.array(jdata[f"k{ki}"]) for ki in range(k_use)], axis=1,
        )  # (D, k_use)
        imp = stacked @ weights
        total = float(weights.sum())

    if total > 0:
        imp = imp / total

    shape = infer_image_shape(len(imp))
    img = reshape_to_image(imp, shape)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img


def _layer_label(layer: str) -> str:
    if layer == "input":
        return "Input"
    return layer.replace("_", " ").title()


def plot(
    run_dir: Path,
    matrix_type: str,
    n_k: int | None,
    out_path: Path,
    save: bool,
    draft: bool,
    width: float,
    task_label: str | None,
) -> None:
    layers = discover_layers(run_dir)
    if not layers:
        print(f"No layers discovered in {run_dir}", file=sys.stderr)
        sys.exit(1)

    panels: list[tuple[str, np.ndarray]] = []
    for layer in layers:
        img = _compute_weighted_importance(run_dir, layer, matrix_type, n_k)
        if img is not None:
            panels.append((_layer_label(layer), img))

    if not panels:
        print(
            f"No weighted-importance data found in {run_dir} "
            f"for matrix_type={matrix_type}",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(panels)
    # NeurIPSFigure: height = width * aspect * nrows / ncols.
    # aspect=1.0 → each panel is square regardless of n.
    with NeurIPSFigure(
        width=width,
        aspect=1.0,
        ncols=n,
        nrows=1,
        draft=draft,
        save=save,
        out_path=out_path,
        squeeze=False,
        gridspec_kw={"wspace": 0.06},
    ) as (fig, axes):
        axes_row = axes[0]
        im = None
        for col_idx, (ax, (label, img)) in enumerate(zip(axes_row, panels)):
            lo, hi = float(img.min()), float(img.max())
            img_n = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
            im = ax.imshow(
                img_n, cmap="inferno", vmin=0, vmax=1, interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)
            ax.set_title(label)
            if col_idx == 0 and task_label:
                ax.set_ylabel(task_label, rotation=90, labelpad=6)

        cbar = fig.colorbar(
            im, ax=axes_row.tolist(),
            location="right", fraction=0.02, pad=0.015,
        )
        cbar.set_label("Importance")
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot eigenvalue-weighted mean Jacobian importance "
            "across layers (NeurIPS figure)."
        ),
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True,
        help="Path to a covariance_spectrum_eigenvectors run directory.",
    )
    parser.add_argument(
        "--matrix_type", type=str, default="signed_cov",
        choices=["signed_cov", "cov"],
        help="Covariance matrix variant to read (default: signed_cov).",
    )
    parser.add_argument(
        "--n_k", type=int, default=None,
        help="Cap the number of eigenvectors used (default: use all available).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Output PDF path. Default: "
            "imgs/weighted_mean_jacobian/<run_name>_<matrix_type>.pdf"
        ),
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="Save figure (PDF + PNG sidecar) instead of showing.",
    )
    parser.add_argument(
        "--draft", action="store_true", default=False,
        help="Skip neurips.mplstyle (faster; no LaTeX).",
    )
    parser.add_argument(
        "--width", type=float, default=1.0,
        help="Figure width as a fraction of NeurIPS text width (default: 1.0).",
    )
    parser.add_argument(
        "--task_label", type=str, default=None,
        help=(
            "Vertical label on the leftmost panel. Default: prettified preset "
            "name parsed from the run-dir (e.g. 'High Cheekbones'). "
            "Pass an empty string to disable."
        ),
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    parsed = parse_run_dirname(run_dir.name)
    if parsed is None:
        print(
            f"Warning: {run_dir.name!r} does not match the standard run-dir "
            "naming pattern; continuing anyway.",
            file=sys.stderr,
        )

    if args.task_label is None:
        task_label = (
            str(parsed["preset"]).replace("_", " ").title() if parsed else None
        )
    else:
        task_label = args.task_label or None

    out_path = args.out
    if out_path is None:
        out_path = (
            Path("imgs") / "weighted_mean_jacobian"
            / f"{run_dir.name}_{args.matrix_type}.pdf"
        )

    plot(
        run_dir=run_dir,
        matrix_type=args.matrix_type,
        n_k=args.n_k,
        out_path=out_path,
        save=args.save,
        draft=args.draft,
        width=args.width,
        task_label=task_label,
    )


if __name__ == "__main__":
    main()
