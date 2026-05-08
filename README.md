# Predicting real data with NLoFi

Code accompanying the arXiv paper. Reproduces the figures comparing
**NLoFi** (kernel-spectral random features with signed-covariance
eigenreduction) against the **Deep-NNGP** baseline on real image
datasets, plus the CelebA feature visualisations.

The codebase is a small, self-contained Python package
(`train_hierarchically`) plus a set of generator and plotting scripts.
Datasets are pulled via `torchvision`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package targets Python ≥ 3.9. CUDA is optional; everything runs on
CPU but the kernel and finite-P sweeps benefit from a GPU.

## Layout

```
src/train_hierarchically/   # Python package
  ├── datasets/              # Torchvision wrappers + class presets
  ├── kernels/               # NNGP kernels: relu, relu_normalized, sigmoid
  ├── models/                # Ridge-spectral and kernel-spectral models
  ├── training/              # Trainers + kernel-ridge readout
  ├── utils/                 # Metrics, helpers
  └── visualization/         # FFN feature visualizer (Fourier maximizer, Jacobian)
scripts/                    # Generators + shared helpers
  └── plotting/              # Figure scripts (read JSON caches)
conf/                       # OmegaConf YAML configs
tests/                      # Unit tests
```

## Reproducing the figures

Each figure has a *generator* (writes JSON results under `results/<name>/`)
and a *plotter* (reads those JSON files, writes PDFs under `imgs/<name>/`).

### NLoFi (finite-P) vs Deep-NNGP

```bash
python scripts/generate_nlofi_vs_deep_nngp.py \
    --n-train-values 500 1000 2000 5000 10000 \
    --seeds 0 1 2 --lam 1e-3 --gpus 0
python scripts/plotting/plot_nlofi_vs_deep_nngp.py --save
python scripts/plotting/plot_deep_nngp_vs_finite.py --save
```

### NLoFi appendix (multi-λ + convergence rate)

```bash
python scripts/generate_nlofi_appendix_data.py \
    --n-train-values 500 1000 2000 5000 \
    --seeds 0 1 2 3 4
python scripts/plotting/plot_nlofi_appendix.py --save
python scripts/plotting/plot_nlofi_convergence_rate.py --save
```

### K0 / K1 / λ heatmap

The heatmap script is fully CLI-driven (no YAML config). Canonical run:

```bash
python scripts/heatmap_K0_K1_lambda.py \
    --n-train 1000 --seeds 0 1 2 3 \
    --n-lambdas 9 --lambda-range -3 1 \
    --out-dir results/heatmap_K0_K1_lambda
python scripts/plotting/plot_heatmap_K0_K1_lambda.py
```

Common variations: `--per-direction-whiten`, `--no-center-kernel`,
`--fix-data-seed N`, `--torch-gpu-eigh`.

### MSE vs k (2-layer)

```bash
python scripts/sweep_mse_vs_k.py \
    --conf conf/mse_vs_k_2layer_relu.yaml \
    --override all_p=1000 dataset.n_train=10000

# Or sweep across (P, n_train) cells in parallel:
python scripts/parallel_run.py \
    --script scripts/sweep_mse_vs_k.py \
    --conf conf/mse_vs_k_2layer_relu.yaml \
    --sweep conf/sweeps/p_ntrain.yaml \
    --n_workers 8

python scripts/plotting/plot_mse_vs_k.py --save
```

### CelebA feature visualisation

Step 1 — train a 3-layer ridge-spectral model and dump signed-covariance
eigenvectors:

```bash
python scripts/covariance_spectrum_eigenvectors.py \
    --conf conf/celeba_3layer_eigenvectors.yaml
```

Step 2 — augment with data-driven features (projection-weighted means,
top/bottom-activating images):

```bash
python scripts/compute_data_driven_features.py \
    --base-dir results/covariance_spectrum_eigenvectors/celeba
```

Step 3 — render the figures:

```bash
python scripts/plotting/plot_weighted_mean_jacobian.py \
    --run_dir results/covariance_spectrum_eigenvectors/celeba/<run_name> --save
python scripts/plotting/plot_feature_evolution.py --save
python scripts/plotting/plot_feature_importance.py --save
python scripts/plotting/plot_data_driven_features.py --save
```

### CNN ridge-spectral (signed patch eigenfilters)

```bash
python scripts/train_cnn_ridge_spectral.py --conf conf/train_cnn_ridge_spectral_eigen.yaml
python scripts/plotting/plot_cnn_signed_patch_eigenfilters.py \
    --conf conf/plot_cnn_signed_patch_eigenfilters_celeba.yaml
```

`scripts/train_cnn_ridge_spectral.py`'s default config name does not
exist in this bundle — pass an explicit `--conf` as shown above.

## Tests

```bash
pip install -e .[dev]
pytest
```

Tests verify:
- the kernel formulas (`test_kernels.py`),
- the kernel/finite-P convergence (`test_kernel_fix_sqrtp.py`,
  `test_kernel_nlofi_integration.py`),
- the kernel-ridge readout and centred-kernel variant
  (`test_kernel_ridge.py`, `test_kernel_spectral_unified.py`),
- the deep-NNGP composition (`test_deep_nngp.py`,
  `test_train_deep_nngp_unified.py`),
- shared eigendecomposition utilities (`test_eigen_utils.py`),
- and the FFN feature visualizer (`test_ffn_jacobian_importance.py`).

## License

MIT — see `LICENSE`.
