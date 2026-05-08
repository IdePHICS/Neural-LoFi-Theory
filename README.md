# Predicting real data with NLoFi

Code accompanying the arXiv paper. Implements **NLoFi** (kernel-spectral
random features with signed-covariance eigenreduction) and the
**Deep-NNGP** baseline.

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
