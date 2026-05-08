# ruff: noqa: N803, N806, N812
"""Compute data-driven feature visualizations for existing runs.

Loads saved model_state_dict.pt + eigenvectors + dataset, computes
projection-weighted mean images and top/bottom activating image averages,
and saves them alongside existing data.

Usage
-----
# Compute for all runs under a dataset:
python scripts/compute_data_driven_features.py \
    --base-dir results/covariance_spectrum_eigenvectors/celeba

# Compute for a single run:
python scripts/compute_data_driven_features.py \
    --run-dir results/covariance_spectrum_eigenvectors/celeba/high_cheekbones_ntrain50000_P10000_layers3_keep100-50-20_relu_seed0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_hierarchically.datasets import DatasetConfig, build_dataset
from train_hierarchically.models.ridge_spectral import (
    LayerSpec,
    RidgeSpectralModel,
)
from train_hierarchically.visualization import FFNVisualizer

# Add plotting utils to path for run-dir parsing
sys.path.insert(0, str(Path(__file__).resolve().parent / "plotting"))
from _plot_utils import discover_layers, load_json, parse_run_dirname  # noqa: E402

log = logging.getLogger(__name__)

_DATASET_NAMES = {"celeba", "cifar10", "fashion-mnist", "mnist", "pcam"}


def _load_dataset(
    dataset_name: str, preset: str, n_samples: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load dataset and return (X, y) as tensors."""
    cfg = DatasetConfig(
        dataset_type=dataset_name,
        split="train",
        root="./data",
        n_samples=n_samples,
        seed=seed,
        flatten=True,
        class_preset=preset,
    )
    ds = build_dataset(cfg)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    xs, ys = zip(*list(loader))
    return torch.cat(xs, 0).float(), torch.cat(ys, 0).float()


def _rebuild_model(
    params: dict, input_dim: int, run_dir: Path,
) -> RidgeSpectralModel | None:
    """Reconstruct model from saved state dict, handling heterogeneous K."""
    state_path = run_dir / "model_state_dict.pt"
    if not state_path.exists():
        return None
    n_layers = params["layers"]
    P = params["P"]
    keep = params["keep"]
    activation = params["activation"]
    seed = params["seed"]

    # Handle heterogeneous K (e.g., "100-50-20")
    if isinstance(keep, str) and "-" in keep:
        k_values = [int(x) for x in keep.split("-")]
    elif keep is not None:
        k_values = [int(keep)] * n_layers
    else:
        k_values = [P] * n_layers

    layer_specs = [
        LayerSpec(p=P, k=k_values[i], activation=activation)
        for i in range(n_layers)
    ]
    model = RidgeSpectralModel(
        input_dim=input_dim, layer_specs=layer_specs, seed=seed,
    )
    model.load_state_dict(
        torch.load(state_path, map_location="cpu", weights_only=True),
    )
    model.eval()
    return model


def _infer_dataset_name(run_dir: Path) -> str | None:
    parent = run_dir.parent.name
    return parent if parent in _DATASET_NAMES else None


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def compute_run(
    run_dir: Path,
    dataset_name: str,
    *,
    device: str = "cuda",
    n_top: int = 4,
    M: int = 50,
    force: bool = False,
) -> None:
    """Compute data-driven features for a single run."""
    params = parse_run_dirname(run_dir.name)
    if params is None:
        log.warning("Cannot parse dirname: %s", run_dir.name)
        return

    layers = discover_layers(run_dir)
    hidden_layers = [l for l in layers if l != "input"]
    if not hidden_layers:
        log.warning("No hidden layers in %s", run_dir.name)
        return

    # Check marker
    last_layer = hidden_layers[-1]
    marker = run_dir / f"data_driven_features_{last_layer}_signed_cov.json"
    if marker.exists() and not force:
        log.info("Already computed for %s — skipping", run_dir.name)
        return

    log.info("Processing: %s", run_dir.name)

    # Load data
    preset = params["preset"]
    n_samples = params["ntrain"]
    seed = params["seed"]
    X, y = _load_dataset(dataset_name, preset, n_samples, seed)
    D = X.shape[1]

    # Rebuild model
    model = _rebuild_model(params, D, run_dir)
    if model is None:
        log.warning("No model_state_dict.pt in %s", run_dir.name)
        return

    dev = torch.device(device)
    model = model.to(dev)
    X = X.to(dev)

    viz = FFNVisualizer(model, dev)

    for layer_idx, layer_name in enumerate(hidden_layers):
        for matrix_type in ["signed_cov", "cov"]:
            eigvec_path = (
                run_dir / f"top_eigenvectors_{layer_name}_{matrix_type}.json"
            )
            if not eigvec_path.exists():
                continue

            data = load_json(eigvec_path)
            vecs = np.array(data["eigenvectors"])  # (P, n_top_saved)

            log.info(
                "  %s (%s): projection-weighted means + top/bottom",
                layer_name, matrix_type,
            )

            # Projection-weighted means
            pwm = viz.projection_weighted_mean(
                X, layer_idx, vecs, n_top,
            )
            _save_json(
                pwm,
                run_dir / f"data_driven_features_{layer_name}_{matrix_type}.json",
            )

            # Top/bottom activating averages
            tb = viz.top_bottom_activating(
                X, layer_idx, vecs, n_top, M=M,
            )
            _save_json(
                tb,
                run_dir / f"top_bottom_features_{layer_name}_{matrix_type}.json",
            )

    # Per-sample importance overlay (pick a few representative samples)
    # Select samples from each class near the median projection of k=0
    log.info("  Per-sample importance overlay")
    n_sample_imgs = 6
    rng = np.random.RandomState(seed)
    pos_mask = (y.cpu().numpy() > 0)
    neg_mask = ~pos_mask
    pos_idx = rng.choice(np.where(pos_mask)[0], size=n_sample_imgs // 2, replace=False)
    neg_idx = rng.choice(np.where(neg_mask)[0], size=n_sample_imgs // 2, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_idx])

    samples = X[sample_idx]  # (n_sample_imgs, D)
    sample_images = samples.cpu().numpy().tolist()

    sample_labels = y[sample_idx].cpu().numpy().tolist()

    for matrix_type in ["signed_cov"]:
        per_sample_data: dict = {
            "sample_indices": sample_idx.tolist(),
            "labels": sample_labels,
            "images": sample_images,
            "layers": [],
            "importances": {},
        }
        all_layers = ["input"] + hidden_layers
        per_sample_data["layers"] = all_layers

        # Input layer: use eigenvector-based importance per sample
        eigvec_path = run_dir / f"top_eigenvectors_input_{matrix_type}.json"
        if eigvec_path.exists():
            edata = load_json(eigvec_path)
            vecs_input = np.array(edata["eigenvectors"])
            evals_input = np.array(edata["eigenvalues"])
            n_k = min(vecs_input.shape[1], len(evals_input))
            input_imps = []
            for si in range(len(sample_idx)):
                x_np = samples[si].cpu().numpy()
                imp = np.zeros_like(x_np)
                for ki in range(n_k):
                    imp += np.abs(evals_input[ki]) * (vecs_input[:, ki] * x_np) ** 2
                imp /= n_k
                input_imps.append(np.sqrt(imp).tolist())
            per_sample_data["importances"]["input"] = input_imps

        # Hidden layers: per-sample Jacobian importance
        for layer_idx, layer_name in enumerate(hidden_layers):
            eigvec_path = (
                run_dir / f"top_eigenvectors_{layer_name}_{matrix_type}.json"
            )
            if not eigvec_path.exists():
                continue
            edata = load_json(eigvec_path)
            vecs = np.array(edata["eigenvectors"])

            log.info("  Per-sample importance: %s (%s)", layer_name, matrix_type)
            imp_list = viz.per_sample_importance(
                samples, layer_idx, vecs, n_top,
            )
            per_sample_data["importances"][layer_name] = [
                imp.tolist() for imp in imp_list
            ]

        _save_json(
            per_sample_data,
            run_dir / f"sample_importance_{matrix_type}.json",
        )

    log.info("Done: %s", run_dir.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Compute data-driven feature visualizations.",
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-top", type=int, default=4)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        dataset_name = _infer_dataset_name(run_dir)
        if dataset_name is None:
            print(f"Cannot infer dataset from {run_dir}", file=sys.stderr)
            sys.exit(1)
        compute_run(
            run_dir, dataset_name,
            device=args.device, n_top=args.n_top, M=args.M,
            force=args.force,
        )
    elif args.base_dir:
        base = Path(args.base_dir)
        dataset_name = base.name
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if parse_run_dirname(d.name) is None:
                continue
            compute_run(
                d, dataset_name,
                device=args.device, n_top=args.n_top, M=args.M,
                force=args.force,
            )
    else:
        print("Provide --run-dir or --base-dir", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
