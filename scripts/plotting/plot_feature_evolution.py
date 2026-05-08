"""Plot feature evolution across layers.

Produces a grid where rows are eigenvector indices (k=0..n_top-1) and
columns are layers (input, layer_1, ..., layer_L).  For the input column
the eigenvectors are reshaped directly to pixel space; for deeper layers
the gradient-ascent maximizers (already in pixel space) are shown.

Usage
-----
python scripts/plotting/plot_feature_evolution.py \
    --n_train 50000 --p 2000 --dataset fashion-mnist --save

python scripts/plotting/plot_feature_evolution.py \
    --n_train 50000 --p 2000 --dataset cifar10 --save --matrix_type both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from _plot_utils import (
    add_common_args,
    discover_layers,
    find_matching_runs,
    infer_image_shape,
    is_rgb,
    load_json,
    make_title,
    normalize_rgb,
    normalize_symmetric,
    reshape_to_image,
    save_or_show,
    setup_style,
)

_RESULTS_BASE = Path("results/covariance_spectrum_eigenvectors")


def _load_column_images(
    run_dir: Path,
    layer: str,
    matrix_type: str,
    n_top: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Load images and eigenvalues for one column of the grid.

    For ``"input"`` layers, loads eigenvectors and reshapes directly.
    For deeper layers, loads gradient-ascent maximizers.

    Returns (images, eigenvalues) where images are display-ready arrays.
    """
    images: list[np.ndarray] = []
    eigenvalues: list[float] = []

    if layer == "input":
        path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
        if not path.exists():
            return images, eigenvalues
        data = load_json(path)
        vecs = np.array(data["eigenvectors"])  # (D, n_top)
        evals = data["eigenvalues"]
        shape = infer_image_shape(vecs.shape[0])
        for ki in range(min(n_top, vecs.shape[1])):
            images.append(reshape_to_image(vecs[:, ki], shape, grayscale=True))
            eigenvalues.append(evals[ki] if ki < len(evals) else 0.0)
    else:
        # Deeper layers: use maximizers (already in input pixel space)
        max_path = run_dir / f"maximizers_{layer}_{matrix_type}.json"
        if not max_path.exists():
            return images, eigenvalues
        maxdata = load_json(max_path)
        # Try to get eigenvalues from eigenvector file
        eigvec_path = run_dir / f"top_eigenvectors_{layer}_{matrix_type}.json"
        evals = []
        if eigvec_path.exists():
            evals = load_json(eigvec_path).get("eigenvalues", [])
        first_key = next(iter(maxdata), None)
        if first_key is None:
            return images, eigenvalues
        shape = infer_image_shape(len(maxdata[first_key]))
        for ki in range(n_top):
            key = f"k{ki}"
            if key not in maxdata:
                break
            images.append(reshape_to_image(np.array(maxdata[key]), shape, grayscale=False))
            eigenvalues.append(evals[ki] if ki < len(evals) else 0.0)

    return images, eigenvalues


def plot_evolution(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Generate the feature evolution grid for one run and matrix type."""
    layers = discover_layers(run_dir)
    if not layers:
        print(f"  No layers found in {run_dir.name}", file=sys.stderr)
        return

    # Probe all layers to find the minimum available n_top
    # (input may have many eigenvectors but deeper layers only ga_n_top maximizers)
    probe_limit = 50
    layer_counts: list[int] = []
    for layer in layers:
        imgs, _ = _load_column_images(run_dir, layer, matrix_type, probe_limit)
        if imgs:
            layer_counts.append(len(imgs))
    n_top = min(layer_counts) if layer_counts else 4

    # Load all columns
    columns: list[tuple[str, list[np.ndarray], list[float]]] = []
    for layer in layers:
        imgs, evals = _load_column_images(run_dir, layer, matrix_type, n_top)
        if imgs:
            columns.append((layer, imgs, evals))

    if not columns:
        print(
            f"  No data for matrix_type={matrix_type} in {run_dir.name}",
            file=sys.stderr,
        )
        return

    n_rows = n_top
    n_cols = len(columns)
    cell_size = 1.6
    fig = plt.figure(figsize=(cell_size * n_cols + 0.8, cell_size * n_rows + 0.8))
    gs = GridSpec(
        n_rows, n_cols, figure=fig, wspace=0.08, hspace=0.25,
    )

    for col_idx, (layer_name, imgs, evals) in enumerate(columns):
        for row_idx in range(min(n_rows, len(imgs))):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = imgs[row_idx]
            if is_rgb(img):
                ax.imshow(np.clip(normalize_rgb(img), 0, 1))
            else:
                vmin, vmax = normalize_symmetric(img)
                ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            # Eigenvalue subtitle
            if row_idx < len(evals) and evals[row_idx] != 0.0:
                ax.set_xlabel(
                    f"$\\lambda$={evals[row_idx]:.2g}",
                    fontsize=7, labelpad=1,
                )

            # Column header
            if row_idx == 0:
                label = "Input" if layer_name == "input" else layer_name.replace("_", " ").title()
                ax.set_title(label, fontsize=9)

            # Row label
            if col_idx == 0:
                ax.set_ylabel(f"$k={row_idx}$", fontsize=9, rotation=0, labelpad=20)

    label = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Feature Evolution — {label}\n{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "feature_evolution" / dataset_name / run_dir.name
    out_path = img_dir / f"evolution_{matrix_type}.pdf"
    save_or_show(fig, save, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature evolution across layers.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--matrix_type",
        type=str,
        default="signed_cov",
        choices=["signed_cov", "cov", "both"],
        help="Which matrix type to plot (default: signed_cov)",
    )
    args = parser.parse_args()

    setup_style(draft=args.draft)

    base_dir = (
        Path(args.base_dir) if args.base_dir
        else _RESULTS_BASE / args.dataset
    )
    matched = find_matching_runs(base_dir, args.n_train, args.p)
    if not matched:
        print(f"No runs found for n_train={args.n_train}, P={args.p} under {base_dir}", file=sys.stderr)
        sys.exit(1)

    types = ["signed_cov", "cov"] if args.matrix_type == "both" else [args.matrix_type]

    print(f"Found {len(matched)} matching run(s)")
    for run_dir, params in matched:
        for mt in types:
            print(f"  Plotting: {run_dir.name} ({mt})")
            plot_evolution(run_dir, params, args.dataset, mt, save=args.save)


if __name__ == "__main__":
    main()
