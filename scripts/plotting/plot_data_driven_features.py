"""Plot data-driven feature visualizations across layers.

Produces grids of projection-weighted mean images and top/bottom
activating image averages computed by compute_data_driven_features.py.

Usage
-----
PYTHONPATH=scripts/plotting python scripts/plotting/plot_data_driven_features.py \
    --n_train 50000 --p 10000 --dataset celeba --save --draft --matrix_type signed_cov
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


def _load_projection_weighted(
    run_dir: Path,
    layer: str,
    matrix_type: str,
    n_top: int,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Load projection-weighted mean images for a layer.

    Returns (per_k_images, mean_image).
    """
    if layer == "input":
        # For input layer, use eigenvectors directly (same as feature_evolution)
        path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
        if not path.exists():
            return [], None
        data = load_json(path)
        vecs = np.array(data["eigenvectors"])
        shape = infer_image_shape(vecs.shape[0])
        images = []
        for ki in range(min(n_top, vecs.shape[1])):
            images.append(reshape_to_image(vecs[:, ki], shape, grayscale=True))
        return images, None

    path = run_dir / f"data_driven_features_{layer}_{matrix_type}.json"
    if not path.exists():
        return [], None
    data = load_json(path)
    first_key = next(iter(data), None)
    if first_key is None:
        return [], None
    dim = len(data[first_key])
    shape = infer_image_shape(dim)
    images = []
    for ki in range(n_top):
        key = f"k{ki}"
        if key not in data:
            break
        images.append(reshape_to_image(np.array(data[key]), shape, grayscale=True))
    mean_img = None
    if "mean" in data:
        mean_img = reshape_to_image(np.array(data["mean"]), shape, grayscale=True)
    return images, mean_img


def _load_top_bottom(
    run_dir: Path,
    layer: str,
    matrix_type: str,
    n_top: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load top/bottom/diff images for a layer.

    Returns list of (top, bottom, diff) tuples per eigenvector.
    """
    if layer == "input":
        return []
    path = run_dir / f"top_bottom_features_{layer}_{matrix_type}.json"
    if not path.exists():
        return []
    data = load_json(path)
    results = []
    for ki in range(n_top):
        key = f"k{ki}"
        if key not in data:
            break
        entry = data[key]
        shape = infer_image_shape(len(entry["top"]))
        top = reshape_to_image(np.array(entry["top"]), shape)
        bot = reshape_to_image(np.array(entry["bottom"]), shape)
        diff = reshape_to_image(np.array(entry["diff"]), shape, grayscale=True)
        results.append((top, bot, diff))
    return results


def plot_projection_weighted_evolution(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot projection-weighted mean images across layers."""
    layers = discover_layers(run_dir)
    if not layers:
        return

    # Probe to find available n_top
    probe_limit = 20
    layer_counts = []
    for layer in layers:
        imgs, _ = _load_projection_weighted(run_dir, layer, matrix_type, probe_limit)
        if imgs:
            layer_counts.append(len(imgs))
    n_top = min(layer_counts) if layer_counts else 4

    columns: list[tuple[str, list[np.ndarray], np.ndarray | None]] = []
    for layer in layers:
        imgs, mean_img = _load_projection_weighted(
            run_dir, layer, matrix_type, n_top,
        )
        if imgs:
            columns.append((layer, imgs, mean_img))

    if not columns:
        return

    # Include mean row for layers that have it
    has_mean = any(m is not None for _, _, m in columns)
    n_rows = n_top + (1 if has_mean else 0)
    n_cols = len(columns)
    cell_size = 1.6
    fig = plt.figure(figsize=(cell_size * n_cols + 0.8, cell_size * n_rows + 0.8))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.08, hspace=0.25)

    for col_idx, (layer_name, imgs, mean_img) in enumerate(columns):
        for row_idx in range(min(n_top, len(imgs))):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = imgs[row_idx]
            vmin, vmax = normalize_symmetric(img)
            ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                label = (
                    "Input" if layer_name == "input"
                    else layer_name.replace("_", " ").title()
                )
                ax.set_title(label, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"$k={row_idx}$", fontsize=9, rotation=0, labelpad=20)

        if has_mean:
            ax = fig.add_subplot(gs[n_top, col_idx])
            if mean_img is not None:
                vmin, vmax = normalize_symmetric(mean_img)
                ax.imshow(mean_img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel("mean", fontsize=9, rotation=0, labelpad=20)

    label = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Projection-Weighted Features — {label}\n{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"projection_weighted_{matrix_type}.pdf")


def plot_top_bottom_grid(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot top/bottom activating averages and differences across layers."""
    layers = discover_layers(run_dir)
    hidden_layers = [l for l in layers if l != "input"]
    if not hidden_layers:
        return

    # Collect data
    all_data: list[tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    n_top = 4
    for layer in hidden_layers:
        tb = _load_top_bottom(run_dir, layer, matrix_type, n_top)
        if tb:
            label = layer.replace("_", " ").title()
            all_data.append((label, tb))

    if not all_data:
        return

    n_k = min(len(tb) for _, tb in all_data)
    n_show = min(n_k, 4)
    n_layers = len(all_data)

    # Grid: rows = k, columns = layers × 3 (top, bottom, diff)
    n_cols = n_layers * 3
    cell_size = 1.3
    fig = plt.figure(
        figsize=(cell_size * n_cols + 1.5, cell_size * n_show + 1.2),
    )
    gs = GridSpec(n_show, n_cols, figure=fig, wspace=0.05, hspace=0.3)

    for li, (layer_label, tb_data) in enumerate(all_data):
        for ki in range(n_show):
            top_img, bot_img, diff_img = tb_data[ki]
            col_base = li * 3

            # Top
            ax = fig.add_subplot(gs[ki, col_base])
            if is_rgb(top_img):
                ax.imshow(np.clip(normalize_rgb(top_img), 0, 1))
            else:
                ax.imshow(top_img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if ki == 0:
                ax.set_title(f"{layer_label}\ntop", fontsize=7)
            if col_base == 0:
                ax.set_ylabel(f"$k={ki}$", fontsize=9, rotation=0, labelpad=18)

            # Bottom
            ax = fig.add_subplot(gs[ki, col_base + 1])
            if is_rgb(bot_img):
                ax.imshow(np.clip(normalize_rgb(bot_img), 0, 1))
            else:
                ax.imshow(bot_img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if ki == 0:
                ax.set_title("bottom", fontsize=7)

            # Diff
            ax = fig.add_subplot(gs[ki, col_base + 2])
            vmin, vmax = normalize_symmetric(diff_img)
            ax.imshow(diff_img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ki == 0:
                ax.set_title("diff", fontsize=7)

    label = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Top/Bottom Activating Faces — {label}\n{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"top_bottom_{matrix_type}.pdf")


def plot_mean_only_projection_weighted(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot only the mean projection-weighted image per layer (no input)."""
    layers = discover_layers(run_dir)
    hidden_layers = [l for l in layers if l != "input"]
    if not hidden_layers:
        return

    means: list[tuple[str, np.ndarray]] = []
    for layer in hidden_layers:
        path = run_dir / f"data_driven_features_{layer}_{matrix_type}.json"
        if not path.exists():
            continue
        data = load_json(path)
        if "mean" not in data:
            continue
        shape = infer_image_shape(len(data["mean"]))
        img = reshape_to_image(np.array(data["mean"]), shape, grayscale=True)
        label = layer.replace("_", " ").title()
        means.append((label, img))

    if not means:
        return

    n = len(means)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), squeeze=False)

    for idx, (label, img) in enumerate(means):
        ax = axes[0, idx]
        vmin, vmax = normalize_symmetric(img)
        ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=9)

    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Mean Projection-Weighted Feature — {label_text}\n{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"mean_projection_{matrix_type}.pdf")


def plot_mean_only_jacobian(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot only the mean Jacobian importance per layer (input + hidden)."""
    layers = discover_layers(run_dir)
    if not layers:
        return

    means: list[tuple[str, np.ndarray]] = []
    for layer in layers:
        if layer == "input":
            path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
            if not path.exists():
                continue
            data = load_json(path)
            vecs = np.array(data["eigenvectors"])
            evals = np.array(data["eigenvalues"])
            n_k = min(vecs.shape[1], len(evals))
            imp = np.zeros(vecs.shape[0])
            for ki in range(n_k):
                imp += np.abs(evals[ki]) * vecs[:, ki] ** 2
            imp /= n_k
            shape = infer_image_shape(len(imp))
            img = reshape_to_image(imp, shape)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            means.append(("Input", img))
        else:
            path = run_dir / f"jacobian_importance_{layer}_{matrix_type}.json"
            if not path.exists():
                continue
            data = load_json(path)
            if "mean" not in data:
                continue
            raw = np.array(data["mean"])
            shape = infer_image_shape(len(raw))
            img = reshape_to_image(raw, shape)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            label = layer.replace("_", " ").title()
            means.append((label, img))

    if not means:
        return

    n = len(means)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), squeeze=False)

    for idx, (label, img) in enumerate(means):
        ax = axes[0, idx]
        # Per-layer normalization for spatial comparison
        img_min, img_max = float(img.min()), float(img.max())
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)
        im = ax.imshow(img_norm, cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=9)

    fig.colorbar(
        im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.03,
        label="normalised (per-layer)",
    )

    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Mean Jacobian Importance — {label_text}\n{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"mean_jacobian_{matrix_type}.pdf")


def plot_weighted_mean_jacobian(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot eigenvalue-weighted mean Jacobian importance per layer.

    Computes weighted_mean_d = sum_k |lambda_k| * imp_k_d / sum_k |lambda_k|
    instead of the uniform (1/K) average.
    """
    layers = discover_layers(run_dir)
    if not layers:
        return

    means: list[tuple[str, np.ndarray]] = []
    for layer in layers:
        eigvec_path = run_dir / f"top_eigenvectors_{layer}_{matrix_type}.json"
        if not eigvec_path.exists():
            continue
        edata = load_json(eigvec_path)
        evals = np.abs(np.array(edata["eigenvalues"]))

        if layer == "input":
            vecs = np.array(edata["eigenvectors"])
            n_k = min(vecs.shape[1], len(evals))
            imp = np.zeros(vecs.shape[0])
            total_weight = 0.0
            for ki in range(n_k):
                w = evals[ki]
                imp += w * w * vecs[:, ki] ** 2
                total_weight += w
            if total_weight > 0:
                imp /= total_weight
            shape = infer_image_shape(len(imp))
            img = reshape_to_image(imp, shape)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            means.append(("Input", img))
        else:
            jac_path = run_dir / f"jacobian_importance_{layer}_{matrix_type}.json"
            if not jac_path.exists():
                continue
            jdata = load_json(jac_path)
            # Find available per-k keys
            k_keys = sorted(
                [k for k in jdata if k.startswith("k")],
                key=lambda s: int(s[1:]),
            )
            if not k_keys:
                continue
            n_k = min(len(k_keys), len(evals))
            dim = len(jdata[k_keys[0]])
            imp = np.zeros(dim)
            total_weight = 0.0
            for ki in range(n_k):
                w = evals[ki]
                imp += w * np.array(jdata[f"k{ki}"])
                total_weight += w
            if total_weight > 0:
                imp /= total_weight
            shape = infer_image_shape(dim)
            img = reshape_to_image(imp, shape)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            label = layer.replace("_", " ").title()
            means.append((label, img))

    if not means:
        return

    n = len(means)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), squeeze=False)

    for idx, (label, img) in enumerate(means):
        ax = axes[0, idx]
        img_min, img_max = float(img.min()), float(img.max())
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)
        im = ax.imshow(img_norm, cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=9)

    fig.colorbar(
        im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.03,
        label="normalised (per-layer)",
    )

    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Eigenvalue-Weighted Jacobian Importance — {label_text}\n"
        f"{make_title(params)}",
        fontsize=10,
    )

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(
        fig, save, img_dir / f"weighted_mean_jacobian_{matrix_type}.pdf",
    )


def plot_sample_importance_overlay(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
    *,
    overlay_alpha: float = 0.4,
) -> None:
    """Plot per-sample Jacobian importance overlaid on original images.

    Loads precomputed per-sample importance maps and overlays them
    with alpha blending on the original images.
    """
    path = run_dir / f"sample_importance_{matrix_type}.json"
    if not path.exists():
        return

    data = load_json(path)
    sample_indices = data.get("sample_indices", [])
    labels = data.get("labels", [])
    layers = data.get("layers", [])
    images = data.get("images", [])  # Original images (flattened)
    importances = data.get("importances", {})  # {layer: [per_sample_importance]}

    if not sample_indices or not layers:
        return

    n_samples = len(sample_indices)
    n_layers = len(layers)
    dim = len(images[0])
    shape = infer_image_shape(dim)

    fig, axes = plt.subplots(
        n_samples, n_layers + 1,
        figsize=(2.2 * (n_layers + 1), 2.2 * n_samples + 0.8),
        squeeze=False,
    )

    for si in range(n_samples):
        # Original image
        orig = reshape_to_image(np.array(images[si]), shape)
        ax = axes[si, 0]
        if is_rgb(orig):
            ax.imshow(np.clip(normalize_rgb(orig), 0, 1))
        else:
            ax.imshow(orig, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if si == 0:
            ax.set_title("Original", fontsize=8)
        # Show label if available, otherwise just index
        if labels:
            lbl = int(labels[si])
            sign = "+" if lbl > 0 else "\u2212"
            color = "green" if lbl > 0 else "red"
            ax.set_ylabel(
                f"$y={sign}1$", fontsize=8, rotation=0, labelpad=22, color=color,
            )
        else:
            ax.set_ylabel(
                f"#{sample_indices[si]}", fontsize=7, rotation=0, labelpad=20,
            )

        for li, layer in enumerate(layers):
            ax = axes[si, li + 1]
            imp_raw = np.array(importances[layer][si])
            imp_img = reshape_to_image(imp_raw, shape)
            if imp_img.ndim == 3:
                imp_img = imp_img.mean(axis=-1)

            # Normalize importance to [0, 1]
            imp_min, imp_max = float(imp_img.min()), float(imp_img.max())
            if imp_max > imp_min:
                imp_norm = (imp_img - imp_min) / (imp_max - imp_min)
            else:
                imp_norm = np.zeros_like(imp_img)

            # Show original image underneath
            if is_rgb(orig):
                ax.imshow(np.clip(normalize_rgb(orig), 0, 1))
            else:
                ax.imshow(orig, cmap="gray")
            # Overlay importance with alpha
            ax.imshow(imp_norm, cmap="hot", alpha=overlay_alpha, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if si == 0:
                label = (
                    "Input" if layer == "input"
                    else layer.replace("_", " ").title()
                )
                ax.set_title(label, fontsize=8)

    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Per-Sample Importance Overlay — {label_text}\n{make_title(params)}",
        fontsize=10,
    )
    fig.tight_layout()

    img_dir = Path("imgs") / "data_driven_features" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"sample_overlay_{matrix_type}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot data-driven feature visualizations.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--matrix_type",
        type=str,
        default="signed_cov",
        choices=["signed_cov", "cov", "both"],
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.4,
        help="Alpha for importance overlay on sample images (default: 0.4)",
    )
    args = parser.parse_args()

    setup_style(draft=args.draft)

    base_dir = (
        Path(args.base_dir) if args.base_dir
        else _RESULTS_BASE / args.dataset
    )
    matched = find_matching_runs(base_dir, args.n_train, args.p)
    if not matched:
        print(
            f"No runs found for n_train={args.n_train}, P={args.p}",
            file=sys.stderr,
        )
        sys.exit(1)

    types = (
        ["signed_cov", "cov"] if args.matrix_type == "both"
        else [args.matrix_type]
    )

    print(f"Found {len(matched)} matching run(s)")
    for run_dir, params in matched:
        for mt in types:
            print(f"  Plotting: {run_dir.name} ({mt})")
            plot_projection_weighted_evolution(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            plot_mean_only_projection_weighted(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            plot_mean_only_jacobian(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            plot_weighted_mean_jacobian(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            plot_top_bottom_grid(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            plot_sample_importance_overlay(
                run_dir, params, args.dataset, mt, save=args.save,
                overlay_alpha=args.overlay_alpha,
            )


if __name__ == "__main__":
    main()
