"""Shared utilities for loading and grouping JSON result files.

Provides two main entry points:

- ``load_and_group`` — single-pass scan + filter + load + group.
- ``ResultIndex``    — scan once, query many times with different filters.

Each calling script provides its own parse function (regex-based).  This
module handles directory scanning (via ``os.scandir`` for speed), JSON
metric extraction, and grouping.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ParseFn = Callable[[str], dict[str, Any] | None]
FilterFn = Callable[[dict[str, Any]], bool] | None
KeyFn = Callable[[dict[str, Any]], Hashable]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _extract_metric(data: dict, metric_path: tuple[str, ...]) -> Any:
    """Traverse nested dict by *metric_path*.  Raises ``KeyError`` on miss."""
    val: Any = data
    for key in metric_path:
        val = val[key]
    return val


def scan_and_parse(
    data_dir: Path,
    parse_fn: ParseFn,
    *,
    suffix: str = ".json",
) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    """Scan *data_dir* once and return parsed entries.

    Parameters
    ----------
    data_dir : Path
        Directory to scan.
    parse_fn : callable
        ``filename -> params_dict | None``.
    suffix : str
        Only consider entries ending with this suffix.

    Returns
    -------
    entries : list[tuple[Path, dict]]
        Sorted by filename.  Only files where *parse_fn* returned non-None.
    n_skipped : int
        Number of files that matched *suffix* but failed parsing.
    """
    entries: list[tuple[str, Path, dict[str, Any]]] = []
    n_skipped = 0

    for entry in os.scandir(data_dir):
        name = entry.name
        if not name.endswith(suffix) or not entry.is_file():
            continue
        params = parse_fn(name)
        if params is None:
            n_skipped += 1
            continue
        entries.append((name, Path(entry.path), params))

    entries.sort(key=lambda t: t[0])
    return [(path, params) for _, path, params in entries], n_skipped


# ---------------------------------------------------------------------------
# Single-pass loader
# ---------------------------------------------------------------------------


def load_and_group(
    data_dir: Path,
    parse_fn: ParseFn,
    key_fn: KeyFn,
    *,
    filter_fn: FilterFn = None,
    metric_path: tuple[str, ...] = ("final", "test_mse"),
) -> dict[Hashable, list[float]]:
    """Scan a directory, parse filenames, filter, load JSON, and group.

    Parameters
    ----------
    data_dir : Path
        Directory containing result JSON files.
    parse_fn : callable
        ``filename -> params_dict | None``.
    key_fn : callable
        ``params_dict -> hashable_key`` for grouping.
    filter_fn : callable or None
        ``params_dict -> bool``.  Applied before opening JSON.
    metric_path : tuple[str, ...]
        Nested keys into the JSON to extract the metric value.

    Returns
    -------
    dict[hashable_key, list[float]]
    """
    parsed, n_skipped = scan_and_parse(data_dir, parse_fn)

    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for path, params in parsed:
        if filter_fn is not None and not filter_fn(params):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            value = float(_extract_metric(data, metric_path))
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
            print(f"Warning: skipping {path.name} ({exc})", file=sys.stderr)
            continue
        grouped[key_fn(params)].append(value)

    if n_skipped:
        print(
            f"Note: {n_skipped} file(s) did not match the filename pattern.",
            file=sys.stderr,
        )
    return dict(grouped)


# ---------------------------------------------------------------------------
# Multi-query index
# ---------------------------------------------------------------------------


class ResultIndex:
    """Pre-scanned directory index for repeated queries.

    The directory is scanned and all JSON files are loaded once in the
    constructor.  Subsequent :meth:`query` calls filter and group from
    the in-memory cache with zero disk I/O.

    Parameters
    ----------
    data_dir : Path
        Directory containing result JSON files.
    parse_fn : callable
        ``filename -> params_dict | None``.
    metric_path : tuple[str, ...]
        Nested keys into the JSON to extract the metric value.
    """

    def __init__(
        self,
        data_dir: Path,
        parse_fn: ParseFn,
        *,
        metric_path: tuple[str, ...] = ("final", "test_mse"),
    ) -> None:
        parsed, n_skipped = scan_and_parse(data_dir, parse_fn)

        self._entries: list[tuple[dict[str, Any], float]] = []
        for path, params in parsed:
            try:
                with open(path) as f:
                    data = json.load(f)
                value = float(_extract_metric(data, metric_path))
            except (
                json.JSONDecodeError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                print(
                    f"Warning: skipping {path.name} ({exc})",
                    file=sys.stderr,
                )
                continue
            self._entries.append((params, value))

        if n_skipped:
            print(
                f"Note: {n_skipped} file(s) did not match the filename pattern.",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self._entries)

    def query(
        self,
        key_fn: KeyFn,
        *,
        filter_fn: FilterFn = None,
    ) -> dict[Hashable, list[float]]:
        """Filter and group from the pre-loaded cache.

        Parameters
        ----------
        key_fn : callable
            ``params -> hashable_key``.
        filter_fn : callable or None
            ``params -> bool``.

        Returns
        -------
        dict[hashable_key, list[float]]
        """
        grouped: dict[Hashable, list[float]] = defaultdict(list)
        for params, value in self._entries:
            if filter_fn is not None and not filter_fn(params):
                continue
            grouped[key_fn(params)].append(value)
        return dict(grouped)


# ---------------------------------------------------------------------------
# Fast direct-construction helpers
# ---------------------------------------------------------------------------


def infer_template(
    data_dir: Path,
    regex: re.Pattern[str],
    *,
    markers: Sequence[str] = (),
    validate: Callable[[re.Match[str]], bool] | None = None,
) -> dict[str, str] | None:
    """Return the ``groupdict`` of the first filename matching *markers* + *regex*.

    Uses ``os.scandir`` with substring pre-filtering so that the expensive
    regex is only applied to the small fraction of entries that pass the
    cheap marker checks.  Returns as soon as the first match is found.

    Parameters
    ----------
    data_dir : Path
        Directory to scan.
    regex : re.Pattern
        Compiled regex applied after the marker filter.
    markers : sequence of str
        Every marker must appear as a substring of the filename.
    validate : callable or None
        Extra predicate on the ``re.Match`` object.
    """
    for entry in os.scandir(data_dir):
        name = entry.name
        if not entry.is_file():
            continue
        if not all(m in name for m in markers):
            continue
        match = regex.match(name)
        if match is None:
            continue
        if validate is not None and not validate(match):
            continue
        return match.groupdict()
    return None


def discover_matching(
    data_dir: Path,
    regex: re.Pattern[str],
    *,
    markers: Sequence[str] = (),
    validate: Callable[[re.Match[str]], bool] | None = None,
) -> list[dict[str, str]]:
    """Like :func:`infer_template` but collect *all* matching ``groupdict``s.

    No JSON files are opened — only filenames are inspected.
    """
    results: list[dict[str, str]] = []
    for entry in os.scandir(data_dir):
        name = entry.name
        if not entry.is_file():
            continue
        if not all(m in name for m in markers):
            continue
        match = regex.match(name)
        if match is None:
            continue
        if validate is not None and not validate(match):
            continue
        results.append(match.groupdict())
    return results


def load_direct(
    data_dir: Path,
    filenames: Iterable[tuple[Hashable, str]],
    *,
    metric_path: tuple[str, ...] = ("final", "test_mse"),
    existing: frozenset[str] | None = None,
) -> dict[Hashable, list[float]]:
    """Load metrics from directly-constructed filenames, grouped by key.

    Each ``(key, filename)`` pair is resolved against *data_dir*.  Only
    existing files are opened — non-existent paths are silently skipped.

    Parameters
    ----------
    data_dir : Path
        Base directory containing the files.
    filenames : iterable of (key, filename)
        Grouping key paired with the filename to try.
    metric_path : tuple of str
        Nested keys into the loaded JSON.
    existing : frozenset of str or None
        Pre-collected set of filenames known to exist in *data_dir*.
        When provided, membership is checked in O(1) instead of issuing
        a ``stat`` syscall per file — much faster on network filesystems.
    """
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for key, fname in filenames:
        if existing is not None:
            if fname not in existing:
                continue
        elif not (data_dir / fname).is_file():
            continue
        try:
            with open(data_dir / fname) as f:
                data = json.load(f)
            value = float(_extract_metric(data, metric_path))
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
            print(f"Warning: skipping {fname} ({exc})", file=sys.stderr)
            continue
        grouped[key].append(value)
    return dict(grouped)
