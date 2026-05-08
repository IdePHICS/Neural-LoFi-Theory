# Neural Low-Degree Filtering (Neural LoFi)

<p align="center">
  <img src="LoFiGirl.jpg" width="900" alt="Neural LoFi meme: from noisy high-dimensional learning to structured feature discovery">
</p>

Code for **Deep Learning as Neural Low-Degree Filtering: A Spectral Theory of
Hierarchical Feature Learning**.

Neural LoFi is a tractable surrogate for deep feature learning. Instead of
following the full dynamics of backpropagation, each layer becomes an explicit
spectral step: form a label-weighted moment operator in the current
representation, keep the leading task-correlated directions, and lift them
through a nonlinear random feature map.

## What is in this repo?

- **NLoFi**: kernel-spectral random features with signed-covariance
  eigenreduction.
- **Deep-NNGP baseline**: fixed-kernel deep random-feature comparison.
- **Synthetic hierarchy experiments**: sample-complexity transitions,
  feature recovery, and spectral outliers.
- **Real-data experiments**: CIFAR-10 animal-vs-vehicle classification,
  feature emergence, and convolutional filter visualizations.
- **Plotting scripts**: reproduce the figures and JSON caches used in the
  paper.

  
## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python ≥ 3.9. CUDA optional.

## Layout

- `src/train_hierarchically/` — the package (kernels, models, trainers,
  datasets, visualisation).
- `scripts/` — generators that produce the JSON caches consumed by the
  plotters under `scripts/plotting/`.
- `conf/` — OmegaConf YAML configs.
- `tests/` — `pytest` suite covering the kernels, kernel-ridge readout,
  deep-NNGP composition, and feature visualiser.

Each generator script has a self-contained docstring with its CLI;
`python scripts/<name>.py --help` lists the flags.

## License

MIT — see `LICENSE`.
