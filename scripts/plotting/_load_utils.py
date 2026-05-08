# ruff: noqa: E402, F401
"""Re-export from the canonical location at ``scripts/_load_utils.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load the canonical module by file path to avoid circular resolution
# (this file's own name shadows the target when using normal imports).
_canonical = Path(__file__).resolve().parent.parent / "_load_utils.py"
_spec = importlib.util.spec_from_file_location("_load_utils_canonical", _canonical)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

ResultIndex = _mod.ResultIndex
discover_matching = _mod.discover_matching
infer_template = _mod.infer_template
load_and_group = _mod.load_and_group
load_direct = _mod.load_direct
scan_and_parse = _mod.scan_and_parse

__all__ = [
    "ResultIndex",
    "discover_matching",
    "infer_template",
    "load_and_group",
    "load_direct",
    "scan_and_parse",
]
