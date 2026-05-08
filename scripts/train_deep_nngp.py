# ruff: noqa: N803, N806
"""Train a deep-NNGP (all layers no-eigen) model from a config file.

Deep NNGP is just the kernel-spectral pipeline with
``do_eigenreduction: false`` on every layer.  This script is a thin
convenience wrapper over :func:`run_kernel_spectral` that keeps the
historical ``scripts/train_deep_nngp.py`` entry point working and
validates at load time that the config actually requests an all-no-
eigen pipeline (so a misconfigured config surfaces a clear error here
rather than producing a mixed run under the wrong script name).
"""

from __future__ import annotations

import logging
from typing import Any

from _kernel_spectral_core import build_results_filename, run_kernel_spectral
from config_utils import with_config
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def validate_all_no_eigen(layer_dicts: list[dict[str, Any]]) -> None:
    """Raise if any layer in *layer_dicts* has ``do_eigenreduction=True``.

    Deep NNGP is the kernel-spectral pipeline with every layer
    no-eigen; this guard makes a misconfigured run (eigen layer under
    the deep-NNGP script name) fail loudly rather than silently
    producing a mixed pipeline.
    """
    eigen_layers = [
        i for i, ld in enumerate(layer_dicts)
        if ld.get("do_eigenreduction", True)
    ]
    if eigen_layers:
        raise ValueError(
            f"train_deep_nngp requires every layer to have "
            f"do_eigenreduction=False; got eigen layers at indices "
            f"{eigen_layers}.  Use train_kernel_spectral.py for mixed "
            f"pipelines."
        )


@with_config("conf/train_deep_nngp.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    layer_dicts: list[dict[str, Any]] = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]
    validate_all_no_eigen(layer_dicts)

    results_path = out_dir / build_results_filename(cfg, layer_dicts)
    if results_path.exists() and not force_run:
        log.info("Results already exist at %s — skipping.", results_path)
        return

    results = run_kernel_spectral(cfg, layer_dicts=layer_dicts, out_path=results_path)
    log.info("\n--- Final Metrics ---\n%s", results["final"])


if __name__ == "__main__":
    main()
