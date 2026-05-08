"""Shared config-loading utility for experiment scripts.

Each script decorates its ``main`` with ``@with_config("config_name")``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def parse_classes(cfg: DictConfig) -> list[int] | None:
    """Extract the class list from an experiment config.

    Handles the OmegaConf ``ListConfig`` → plain ``list[int]`` conversion
    required by ``DatasetConfig`` (which uses ``slots=True``).
    """
    raw = cfg.dataset.get("classes")
    return cast(list[int], OmegaConf.to_object(raw)) if raw else None


# --- Custom OmegaConf resolvers ---
OmegaConf.register_new_resolver(
    "len",
    lambda key, _root_: len(OmegaConf.select(_root_, key)),
    replace=True,
)
OmegaConf.register_new_resolver(
    "sub",
    lambda a, b: int(a) - int(b),
    replace=True,
)
OmegaConf.register_new_resolver(
    "div",
    lambda a, b: float(a) / float(b),
    replace=True,
)


def resolve_results_path(raw_path: str) -> Path:
    """Resolve a ``results/…`` path, respecting ``RESULTS_BASE_PATH``.

    When ``RESULTS_BASE_PATH`` is set (e.g. on a SLURM cluster), the
    leading ``results/`` component is stripped and the env-var is used as
    root — the same convention that ``load_config`` uses for
    ``output_dir``.  Without the env-var the path is returned as-is.
    """
    results_base = os.environ.get("RESULTS_BASE_PATH")
    if results_base is not None:
        p = Path(raw_path)
        if p.parts and p.parts[0] == "results":
            p = Path(*p.parts[1:])
        return Path(results_base) / p
    return Path(raw_path)


def load_config(
    config_name: str,
    use_timestamp: bool = True,
) -> tuple[DictConfig, Path, bool]:
    """Load a YAML config and apply CLI overrides.

    Parameters
    ----------
    config_name
        Name of the YAML file in *conf_dir* (without the ``.yaml`` suffix).
    conf_dir
        Directory containing the YAML configs.

    Returns
    -------
    tuple[DictConfig, Path, bool]
        The resolved configuration, the output directory, and whether
        ``--force-run`` was passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        default=None,
        help="Config file path or name (without .yaml) to override the default",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. seed=42 dataset.n_train=5000",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        default=False,
        help="Force run even if results already exist",
    )
    args = parser.parse_args()

    # --- Resolve which config file to load ---
    base_path = f"{args.conf or config_name}"
    if isinstance(base_path, str):
        base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    cfg = OmegaConf.load(base_path)
    assert isinstance(cfg, DictConfig)

    # --- Apply CLI overrides ---
    # Use OmegaConf.update() instead of from_dotlist + merge so that
    # dotted keys with numeric segments (e.g. ``layers.0.p=1000``) are
    # resolved as list-index accesses rather than dict-key lookups.
    for override in args.override:
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        OmegaConf.update(cfg, key, value)

    # --- Resolve interpolations ---
    OmegaConf.resolve(cfg)

    # --- Build output directory ---
    output_base: str = str(cfg.get("output_dir", f"results/{config_name}"))

    # If RESULTS_BASE_PATH is set (e.g. by a SLURM script), use it as the
    # root for output.  Config ``output_dir`` values conventionally start
    # with ``results/…``; strip that prefix so the env-var replaces it.
    results_base = os.environ.get("RESULTS_BASE_PATH")

    if results_base is not None:
        output_path = Path(output_base)
        # Strip leading "results" component when present
        if output_path.parts and output_path.parts[0] == "results":
            output_path = Path(*output_path.parts[1:])
        output_base = str(Path(results_base) / output_path)

    if use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(output_base) / timestamp
    else:
        out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove output_dir from config (it's metadata, not a parameter)
    if "output_dir" in cfg:
        del cfg["output_dir"]

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Output directory: %s", out_dir)

    return cfg, out_dir, args.force_run


def with_config(
    config_name: str,
    use_timestamp: bool = True,
) -> Callable[[Callable[..., Any]], Callable[[], None]]:
    """Decorator that loads config and passes ``(cfg, out_dir)`` to the function.

    Usage::

        @with_config("train_backprop")
        def main(cfg: DictConfig, out_dir: Path) -> None:
            ...

        if __name__ == "__main__":
            main()
    """

    def decorator(
        fn: Callable[..., Any],
    ) -> Callable[[], None]:
        @functools.wraps(fn)
        def wrapper() -> None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                datefmt="%H:%M:%S",
            )
            cfg, out_dir, force_run = load_config(config_name, use_timestamp)
            sig = inspect.signature(fn)
            if "force_run" in sig.parameters:
                fn(cfg, out_dir, force_run=force_run)
            else:
                fn(cfg, out_dir)

        return wrapper

    return decorator
