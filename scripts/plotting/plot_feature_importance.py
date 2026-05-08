"""Plot pixel-level feature importance maps (AGOP analogue).

Computes importance as ``importance[d] = sum_k |lambda_k| * v_k[d]^2``,
the diagonal of the weighted eigenvector outer product — analogous to
the AGOP diagonal in Radhakrishnan et al. (Science, 2024).

Produces:
(a) Weighted importance heatmap (input layer).
(b) Per-eigenvector importance grid.
(c) Importance evolution across layers (using maximizer squared values
    for deeper layers).

Usage
-----
python scripts/plotting/plot_feature_importance.py \
    --n_train 50000 --p 2000 --dataset fashion-mnist --save
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
    load_json,
    make_title,
    reshape_to_image,
    save_or_show,
    setup_style,
)

_RESULTS_BASE = Path("results/covariance_spectrum_eigenvectors")


def _compute_importance_input(
    run_dir: Path,
    matrix_type: str,
) -> np.ndarray | None:
    """Compute pixel importance from input-layer eigenvectors.

    Returns a flat array of importance values (one per pixel), or None.
    """
    path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
    if not path.exists():
        return None
    data = load_json(path)
    vecs = np.array(data["eigenvectors"])  # (D, n_top)
    evals = np.array(data["eigenvalues"])  # (n_top,)
    n_top = min(vecs.shape[1], len(evals))
    importance = np.zeros(vecs.shape[0])
    for ki in range(n_top):
        importance += np.abs(evals[ki]) * vecs[:, ki] ** 2
    return importance


def _load_jacobian_importance(
    run_dir: Path,
    layer: str,
    matrix_type: str,
) -> np.ndarray | None:
    """Load mean Jacobian importance for a layer (backward compat)."""
    path = run_dir / f"jacobian_importance_{layer}_{matrix_type}.json"
    if not path.exists():
        return None
    data = load_json(path)
    # New format: dict with "k0", "k1", ..., "mean"
    if "mean" in data:
        return np.array(data["mean"])
    # Old format: dict with "importance" key (flat array)
    if "importance" in data:
        return np.array(data["importance"])
    return None


def _load_jacobian_per_eigenvector(
    run_dir: Path,
    layer: str,
    matrix_type: str,
) -> list[np.ndarray] | None:
    """Load per-eigenvector Jacobian importance maps for a layer.

    Returns a list of arrays [k0, k1, ...], or None if not available.
    """
    path = run_dir / f"jacobian_importance_{layer}_{matrix_type}.json"
    if not path.exists():
        return None
    data = load_json(path)
    if "mean" not in data:
        return None  # old format, no per-eigenvector data
    maps = []
    k = 0
    while f"k{k}" in data:
        maps.append(np.array(data[f"k{k}"]))
        k += 1
    return maps if maps else None


def _compute_importance_maximizer(
    run_dir: Path,
    layer: str,
    matrix_type: str,
) -> np.ndarray | None:
    """Compute eigenvalue-weighted importance from maximizers.

    Loads eigenvalues from the corresponding eigenvector file and computes
    ``importance[d] = sum_k |lambda_k| * m_k[d]^2``, analogous to the
    input-layer formula.
    """
    max_path = run_dir / f"maximizers_{layer}_{matrix_type}.json"
    if not max_path.exists():
        return None
    maxdata = load_json(max_path)
    if not maxdata:
        return None

    # Load eigenvalues for weighting
    eig_path = run_dir / f"top_eigenvectors_{layer}_{matrix_type}.json"
    evals: np.ndarray | None = None
    if eig_path.exists():
        evals = np.array(load_json(eig_path)["eigenvalues"])

    first_key = next(iter(maxdata))
    dim = len(maxdata[first_key])
    importance = np.zeros(dim)
    for ki, key in enumerate(maxdata):
        v = np.array(maxdata[key])
        weight = float(np.abs(evals[ki])) if evals is not None and ki < len(evals) else 1.0
        importance += weight * v ** 2
    return importance


def plot_importance_heatmap(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    save: bool,
) -> None:
    """Plot weighted pixel importance heatmaps for the input layer."""
    img_dir = Path("imgs") / "feature_importance" / dataset_name / run_dir.name

    for matrix_type in ["cov", "signed_cov"]:
        importance = _compute_importance_input(run_dir, matrix_type)
        if importance is None:
            continue

        shape = infer_image_shape(len(importance))
        img = reshape_to_image(importance, shape)
        # For multi-channel, take mean across channels for a single heatmap
        if img.ndim == 3:
            img = img.mean(axis=-1)

        fig, ax = plt.subplots(figsize=(3.5, 3))
        im = ax.imshow(img, cmap="inferno")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        label = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
        ax.set_title(
            f"Pixel Importance — {label}\n{make_title(params)}",
            fontsize=9,
        )

        save_or_show(fig, save, img_dir / f"importance_{matrix_type}.pdf")


def plot_per_eigenvector_importance(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
) -> None:
    """Plot per-eigenvector importance maps for the input layer."""
    path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
    if not path.exists():
        return

    data = load_json(path)
    vecs = np.array(data["eigenvectors"])  # (D, n_top)
    evals = data["eigenvalues"]
    n_top = vecs.shape[1]
    shape = infer_image_shape(vecs.shape[0])

    fig, axes = plt.subplots(1, n_top, figsize=(2.2 * n_top, 2.5), squeeze=False)

    for ki in range(n_top):
        ax = axes[0, ki]
        imp = vecs[:, ki] ** 2
        img = reshape_to_image(imp, shape)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        ax.imshow(img, cmap="inferno")
        ax.set_xticks([])
        ax.set_yticks([])
        lam = evals[ki] if ki < len(evals) else 0
        ax.set_title(f"$k={ki}$\n$\\lambda$={lam:.2g}", fontsize=8)

    label = "Signed Cov" if matrix_type == "signed_cov" else "Cov"
    fig.suptitle(
        f"Per-Eigenvector Importance — {label}\n{make_title(params)}",
        fontsize=9,
    )

    img_dir = Path("imgs") / "feature_importance" / dataset_name / run_dir.name
    save_or_show(fig, save, img_dir / f"per_eigvec_{matrix_type}.pdf")


def _normalize_img(
    img: np.ndarray,
    mode: str,
    *,
    global_min: float = 0.0,
    global_max: float = 1.0,
) -> np.ndarray:
    """Normalize an importance image according to the given mode."""
    if mode == "global":
        if global_max > global_min:
            return (img - global_min) / (global_max - global_min)
        return np.zeros_like(img)
    if mode == "log_global":
        img_log = np.log1p(img)
        log_min, log_max = np.log1p(global_min), np.log1p(global_max)
        if log_max > log_min:
            return (img_log - log_min) / (log_max - log_min)
        return np.zeros_like(img)
    # per_layer (default)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return np.zeros_like(img)


def _collect_importance_heatmaps(
    run_dir: Path,
    matrix_type: str,
    *,
    color: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Collect importance heatmaps for all layers.

    When *color* is False, RGB channels are averaged to grayscale.
    When *color* is True, the (H, W, 3) array is kept as-is.
    """
    layers = discover_layers(run_dir)
    heatmaps: list[tuple[str, np.ndarray]] = []
    for layer in layers:
        if layer == "input":
            importance = _compute_importance_input(run_dir, matrix_type)
        else:
            importance = _load_jacobian_importance(run_dir, layer, matrix_type)
            if importance is None:
                importance = _compute_importance_maximizer(
                    run_dir, layer, matrix_type,
                )
        if importance is None:
            continue
        shape = infer_image_shape(len(importance))
        img = reshape_to_image(importance, shape)
        if not color and img.ndim == 3:
            img = img.mean(axis=-1)
        label = "Input" if layer == "input" else layer.replace("_", " ").title()
        heatmaps.append((label, img))
    return heatmaps


def plot_importance_evolution(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
    *,
    normalization: str = "per_layer",
) -> None:
    """Plot importance heatmap evolution across layers.

    Generates both grayscale (inferno) and RGB color variants.
    """
    img_dir = Path("imgs") / "feature_importance" / dataset_name / run_dir.name
    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"

    for color in (False, True):
        heatmaps = _collect_importance_heatmaps(
            run_dir, matrix_type, color=color,
        )
        if not heatmaps:
            return

        # Compute global stats for non-per-layer modes (grayscale only)
        if not color:
            all_vals = np.concatenate(
                [img.ravel() for _, img in heatmaps],
            )
            g_min, g_max = float(all_vals.min()), float(all_vals.max())

        n = len(heatmaps)
        fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), squeeze=False)

        for idx, (label, img) in enumerate(heatmaps):
            ax = axes[0, idx]
            raw_max = float(np.max(img))

            if color:
                # RGB importance — normalize jointly to [0, 1]
                hi = float(img.max())
                img_disp = img / hi if hi > 0 else np.zeros_like(img)
                ax.imshow(np.clip(img_disp, 0, 1))
            else:
                img_norm = _normalize_img(
                    img, normalization, global_min=g_min, global_max=g_max,
                )
                im = ax.imshow(img_norm, cmap="inferno", vmin=0, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=8)
            ax.set_xlabel(f"max = {raw_max:.2g}", fontsize=7, labelpad=2)

        if not color:
            cb_label = {
                "per_layer": "normalised (per-layer)",
                "global": "normalised (global)",
                "log_global": "log importance",
            }.get(normalization, "normalised")
            fig.colorbar(
                im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.03,
                label=cb_label,
            )

        norm_tag = f" [{normalization}]" if not color else " [color]"
        fig.suptitle(
            f"Importance Evolution — {label_text}{norm_tag}\n{make_title(params)}",
            fontsize=9,
        )

        suffix = "_color" if color else f"_{normalization}"
        save_or_show(
            fig, save,
            img_dir / f"evolution_{matrix_type}{suffix}.pdf",
        )


def _collect_jacobian_layer_data(
    run_dir: Path,
    matrix_type: str,
    *,
    color: bool = False,
) -> list[tuple[str, list[np.ndarray] | None, np.ndarray | None]]:
    """Collect per-eigenvector and mean importance maps for all layers."""
    layers = discover_layers(run_dir)
    layer_data: list[tuple[str, list[np.ndarray] | None, np.ndarray | None]] = []
    for layer in layers:
        label = "Input" if layer == "input" else layer.replace("_", " ").title()
        if layer == "input":
            path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
            if not path.exists():
                continue
            data = load_json(path)
            vecs = np.array(data["eigenvectors"])
            evals = np.array(data["eigenvalues"])
            n_k = min(vecs.shape[1], len(evals))
            maps = []
            for ki in range(n_k):
                maps.append(np.abs(evals[ki]) * vecs[:, ki] ** 2)
            mean = sum(maps) / len(maps) if maps else None
            layer_data.append((label, maps, mean))
        else:
            per_k = _load_jacobian_per_eigenvector(run_dir, layer, matrix_type)
            mean_arr = _load_jacobian_importance(run_dir, layer, matrix_type)
            if per_k is None and mean_arr is None:
                continue
            layer_data.append((label, per_k, mean_arr))
    return layer_data


def _importance_to_img(
    raw: np.ndarray, *, color: bool = False,
) -> np.ndarray:
    """Reshape flat importance array to displayable image."""
    shape = infer_image_shape(len(raw))
    img = reshape_to_image(raw, shape)
    if not color and img.ndim == 3:
        img = img.mean(axis=-1)
    return img


def plot_jacobian_grid(
    run_dir: Path,
    params: dict,
    dataset_name: str,
    matrix_type: str,
    save: bool,
    *,
    normalization: str = "per_layer",
) -> None:
    """Plot per-eigenvector Jacobian importance grid.

    Generates both grayscale (inferno) and RGB color variants.
    """
    img_dir = Path("imgs") / "feature_importance" / dataset_name / run_dir.name
    label_text = "Signed Cov" if matrix_type == "signed_cov" else "Cov"

    for color in (False, True):
        layer_data = _collect_jacobian_layer_data(
            run_dir, matrix_type, color=color,
        )
        if not layer_data:
            return

        k_counts = [len(maps) for _, maps, _ in layer_data if maps is not None]
        if not k_counts:
            return
        n_top = min(k_counts)
        n_show = min(n_top, 4)

        # Collect all cell images for global normalization
        all_cell_imgs: list[np.ndarray] = []
        if not color:
            for _, per_k_maps, mean_map in layer_data:
                if per_k_maps is not None:
                    for ki in range(n_show):
                        if ki < len(per_k_maps):
                            all_cell_imgs.append(
                                _importance_to_img(per_k_maps[ki], color=False),
                            )
                if mean_map is not None:
                    all_cell_imgs.append(
                        _importance_to_img(mean_map, color=False),
                    )
            if all_cell_imgs:
                all_vals = np.concatenate([c.ravel() for c in all_cell_imgs])
                g_min, g_max = float(all_vals.min()), float(all_vals.max())
            else:
                g_min, g_max = 0.0, 1.0

        n_rows = n_show + 1
        n_cols = len(layer_data)
        cell_size = 1.5
        fig = plt.figure(
            figsize=(cell_size * n_cols + 1.5, cell_size * n_rows + 1.0),
        )
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.08, hspace=0.3)

        for col_idx, (layer_label, per_k_maps, mean_map) in enumerate(layer_data):
            for row_idx in range(n_show):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                if per_k_maps is not None and row_idx < len(per_k_maps):
                    img = _importance_to_img(per_k_maps[row_idx], color=color)
                    if color and img.ndim == 3:
                        hi = float(img.max())
                        ax.imshow(np.clip(img / hi if hi > 0 else img, 0, 1))
                    else:
                        img_norm = _normalize_img(
                            img, normalization,
                            global_min=g_min, global_max=g_max,
                        )
                        ax.imshow(img_norm, cmap="inferno", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(
                        f"$k={row_idx}$", fontsize=9, rotation=0, labelpad=20,
                    )
                if row_idx == 0:
                    ax.set_title(layer_label, fontsize=8)

            ax = fig.add_subplot(gs[n_show, col_idx])
            if mean_map is not None:
                img = _importance_to_img(mean_map, color=color)
                if color and img.ndim == 3:
                    hi = float(img.max())
                    ax.imshow(np.clip(img / hi if hi > 0 else img, 0, 1))
                else:
                    img_norm = _normalize_img(
                        img, normalization,
                        global_min=g_min, global_max=g_max,
                    )
                    ax.imshow(img_norm, cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel("mean", fontsize=9, rotation=0, labelpad=20)

        norm_tag = " [color]" if color else f" [{normalization}]"
        fig.suptitle(
            f"Jacobian Importance — {label_text}{norm_tag}\n{make_title(params)}",
            fontsize=10,
        )

        suffix = "_color" if color else f"_{normalization}"
        save_or_show(
            fig, save,
            img_dir / f"jacobian_grid_{matrix_type}{suffix}.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pixel-level feature importance maps.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--matrix_type",
        type=str,
        default="signed_cov",
        choices=["signed_cov", "cov", "both"],
        help="Which matrix type to plot (default: signed_cov)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="all",
        choices=["per_layer", "global", "log_global", "all"],
        help="Normalization mode for importance plots (default: all)",
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
    norms = (
        ["per_layer", "global", "log_global"]
        if args.normalization == "all"
        else [args.normalization]
    )

    print(f"Found {len(matched)} matching run(s)")
    for run_dir, params in matched:
        print(f"  Plotting: {run_dir.name}")
        plot_importance_heatmap(run_dir, params, args.dataset, save=args.save)
        for mt in types:
            plot_per_eigenvector_importance(
                run_dir, params, args.dataset, mt, save=args.save,
            )
            for norm in norms:
                plot_importance_evolution(
                    run_dir, params, args.dataset, mt, save=args.save,
                    normalization=norm,
                )
                plot_jacobian_grid(
                    run_dir, params, args.dataset, mt, save=args.save,
                    normalization=norm,
                )


if __name__ == "__main__":
    main()
