# ruff: noqa: N803, N806
from __future__ import annotations

import logging
from typing import Any

from _ridge_spectral_core import build_results_filename, run_ridge_spectral
from config_utils import with_config
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@with_config("conf/train_ridge_spectral.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    # --- Build results path and check skip ---
    layer_dicts: list[dict[str, Any]] = OmegaConf.to_object(cfg.layers)  # type: ignore[assignment]
    results_filename = build_results_filename(cfg, layer_dicts)
    results_path = out_dir / results_filename

    if results_path.exists() and not force_run:
        log.info("Results already exist at %s — skipping.", results_path)
        return

    # --- Run training via shared core ---
    results = run_ridge_spectral(cfg, layer_dicts=layer_dicts, out_path=results_path)

    # --- Log results ---
    log.info("\n--- Layer-wise Diagnostics ---")
    for entry in results["layers"]:
        idx = entry["layer_idx"]
        name = entry["layer_name"]
        eigvals = entry.get("eigenvalues", [])
        n_eig = len(eigvals) if hasattr(eigvals, "__len__") else 0
        log.info("  layer %2d (%s): %d eigenvalues", idx, name, n_eig)

    log.info("\n--- Final Metrics ---\n%s", results["final"])


if __name__ == "__main__":
    main()
