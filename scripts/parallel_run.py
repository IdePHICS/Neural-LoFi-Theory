"""Parallel sweep runner with optional MPI multi-node support.

Computes the Cartesian product of all parameter values in a sweep YAML file
and dispatches each combination to a pool of worker processes. Each worker
invokes the target script as a subprocess:

    python <script> --conf=<conf> --override param1=value1 param2=value2 ...

The total number of runs equals the product of sweep parameter list lengths.

MPI support
-----------
When launched via ``mpirun`` / ``mpiexec`` (requires ``mpi4py``), the full
list of configurations is deterministically split across MPI ranks using
round-robin assignment (rank gets indices ``rank, rank+size, rank+2*size,
...``).  Each rank then processes its subset with local multiprocessing
workers, exactly as in the single-node case.

**No inter-rank communication** occurs after startup — every rank reads the
same YAML, computes the same Cartesian product, and independently selects its
slice.  There is no barrier, broadcast, or reduce, so there are no MPI
timeout concerns regardless of how long runs take.

Without ``mpi4py`` (or when run with a single rank) the script behaves
identically to the non-MPI version.

Arguments:
    --script       Target experiment script to run.
    --conf         Base configuration file passed via --conf= to the script.
    --sweep        YAML file with a top-level ``sweep`` key mapping parameter
                   names to lists of values (or a single value).
    --n_workers    Number of parallel worker processes **per rank** (default:
                   number of configs assigned to that rank).
    --timeout      Per-run timeout in seconds (default: 3600).

Sweep YAML example::

    sweep:
      seed: [1, 2, 3]
      layer1.p: [500, 1000]

    This produces 6 runs (3 seeds x 2 layer widths).

Usage (single node)::

    python parallel_run.py \\
        --script train_ridge_spectral_dnn.py \\
        --conf configs/ridge_spectral.yaml \\
        --sweep configs/sweep.yaml \\
        --n_workers 4 \\
        --timeout 7200

Usage (multi-node via MPI)::

    mpirun -n 4 python parallel_run.py \\
        --script train_ridge_spectral_dnn.py \\
        --conf configs/ridge_spectral.yaml \\
        --sweep configs/sweep.yaml \\
        --n_workers 2 \\
        --timeout 7200
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from multiprocessing import Manager, Process, Queue
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Optional MPI — graceful fallback to environment variables, then single-process
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    _comm = None
    # Detect rank/size from environment when mpi4py is unavailable.
    # srun sets SLURM_PROCID / SLURM_NTASKS; OpenMPI sets OMPI_COMM_WORLD_*.
    _rank = int(
        os.environ.get(
            "SLURM_PROCID",
            os.environ.get("OMPI_COMM_WORLD_RANK", "0"),
        )
    )
    _size = int(
        os.environ.get(
            "SLURM_NTASKS",
            os.environ.get("OMPI_COMM_WORLD_SIZE", "1"),
        )
    )

_USE_MPI = _size > 1


def _log(msg: str) -> None:
    """Print with an MPI rank prefix when running multi-rank."""
    prefix = f"[Rank {_rank}] " if _USE_MPI else ""
    print(f"{prefix}{msg}", flush=True)


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def worker(
    worker_id: int,
    queue: Queue,
    script_path: str,
    base_conf: str,
    timeout: int,
    rank_prefix: str,
    extra_overrides: list[str] | None = None,
    force_run: bool = False,
) -> None:
    """Worker process that runs configurations from the queue, logging to stdout."""
    tag = f"{rank_prefix}Worker {worker_id}"
    while True:
        item = queue.get()
        if item is None:
            print(f"[{tag}] Shutting down.", flush=True)
            break
        config_num, param_dict = item
        print(f"[{tag}] Processing config {config_num}: {param_dict}", flush=True)
        cmd = [
            sys.executable,
            script_path,
            f"--conf={base_conf}",
        ]
        if force_run:
            cmd.append("--force-run")
        cmd.append("--override")
        if extra_overrides:
            cmd.extend(extra_overrides)
        for param, value in param_dict.items():
            cmd.append(f"{param}={value}")
        print(f"[{tag}] Running: {' '.join(cmd)}", flush=True)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            print(f"[{tag}] STDOUT:\n{result.stdout}", flush=True)
            if result.stderr:
                print(f"[{tag}] STDERR:\n{result.stderr}", flush=True)
            if result.returncode != 0:
                print(f"[{tag}] FAILED (return code {result.returncode})", flush=True)
            else:
                print(f"[{tag}] SUCCESS", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{tag}] TIMED OUT", flush=True)
        except Exception as e:
            print(f"[{tag}] ERROR: {e}", flush=True)


def main() -> None:
    """Main entry point for the parallel run script."""
    parser = argparse.ArgumentParser(
        description="Run multiple experiment configurations in parallel"
    )
    parser.add_argument("--script", type=str, required=True, help="Core script to run")
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Path to sweep configuration file",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of worker processes per rank (defaults to number of "
        "configs assigned to this rank)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per run in seconds (default 7200)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Extra key=value overrides forwarded to every run",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        default=False,
        help="Force re-run even if results already exist",
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.script).exists():
        _log(f"Script not found: {args.script}")
        sys.exit(1)
    if not Path(args.conf).exists():
        _log(f"Config not found: {args.conf}")
        sys.exit(1)
    if not Path(args.sweep).exists():
        _log(f"Sweep config not found: {args.sweep}")
        sys.exit(1)

    sweep_config = load_yaml(args.sweep)
    sweep = sweep_config.get("sweep", {})

    # ------------------------------------------------------------------
    # Compute full Cartesian product of sweep parameters.
    # Every rank computes the same deterministic list — no broadcast needed.
    # ------------------------------------------------------------------
    keys = list(sweep.keys())
    values_lists = [v if isinstance(v, list) else [v] for v in sweep.values()]
    all_configs = [dict(zip(keys, vals)) for vals in itertools.product(*values_lists)]
    n_total = len(all_configs)

    # ------------------------------------------------------------------
    # Split across MPI ranks (round-robin).  Without MPI, rank=0 gets all.
    # ------------------------------------------------------------------
    my_configs = [
        (global_idx, cfg)
        for global_idx, cfg in enumerate(all_configs)
        if global_idx % _size == _rank
    ]
    n_local = len(my_configs)
    n_workers = args.n_workers or n_local

    if _USE_MPI:
        _log(
            f"MPI enabled — {_size} rank(s). "
            f"Total configs: {n_total}. This rank handles {n_local} configs "
            f"with {n_workers} local worker(s)."
        )
    else:
        _log(f"Dispatching {n_total} runs over {n_workers} workers.")

    # ------------------------------------------------------------------
    # Dispatch local subset via multiprocessing queue
    # ------------------------------------------------------------------
    rank_prefix = f"Rank {_rank} / " if _USE_MPI else ""
    manager = Manager()
    queue = manager.Queue()

    for global_idx, param_dict in my_configs:
        queue.put((global_idx, param_dict))
    for _ in range(n_workers):
        queue.put(None)

    workers = []
    for worker_id in range(n_workers):
        p = Process(
            target=worker,
            args=(
                worker_id,
                queue,
                args.script,
                args.conf,
                args.timeout,
                rank_prefix,
                args.override,
                args.force_run,
            ),
        )
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    _log(f"All local workers completed ({n_local}/{n_total} configs).")

    # ------------------------------------------------------------------
    # No MPI barrier / reduce / gather.  Each rank exits independently.
    # This avoids any MPI timeout issues — ranks never wait on each other.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
