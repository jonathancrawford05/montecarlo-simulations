"""
Monte Carlo mortality simulation with hybrid parallel processing.

This module implements a stochastic mortality simulation that uses multiprocessing
with vectorized batch operations for efficient large-scale simulations.
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np


def _init_worker():
    """
    Initialize worker process with single-threaded NumPy.

    Prevents thread oversubscription when using multiprocessing with
    multi-threaded BLAS/MKL libraries. Each worker process will use
    single-threaded NumPy, while parallelism comes from multiprocessing.
    """
    # Set thread limits for common BLAS implementations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def check_threading_config() -> dict:
    """
    Check current NumPy/BLAS threading configuration.

    Returns information about thread pools used by NumPy's underlying
    BLAS/LAPACK libraries. Useful for diagnosing performance issues.

    Returns
    -------
    dict
        Threading configuration with keys:
        - 'threadpool_info': List of thread pool configs (if threadpoolctl installed)
        - 'env_vars': Current thread-related environment variables
        - 'warning': Any warnings about potential issues
    """
    result = {
        "env_vars": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not set"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not set"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "not set"),
        },
        "warning": None,
    }

    try:
        import threadpoolctl

        info = threadpoolctl.threadpool_info()
        result["threadpool_info"] = info

        # Check for potential oversubscription
        total_threads = sum(lib.get("num_threads", 1) for lib in info)
        if total_threads > cpu_count():
            result["warning"] = (
                f"Potential thread oversubscription: {total_threads} threads "
                f"across BLAS libraries, but only {cpu_count()} CPU cores. "
                "Consider setting OMP_NUM_THREADS=1 before importing numpy."
            )
    except ImportError:
        result["threadpool_info"] = "threadpoolctl not installed (pip install threadpoolctl)"

    return result


def get_optimal_params(n_rows: int) -> tuple[int, int]:
    """
    Get optimal (n_processes, batch_size) based on benchmark calibration.

    Calibrated on a 10-core MacBook with the following results:
    - 10K rows: 1 process, batch_size=50 (overhead dominated)
    - 100K rows: 1 process, batch_size=50 (overhead dominated)
    - 1M rows: 2 processes, batch_size=25 (parallelism helps)
    - 20M rows: 2 processes, batch_size=10 (memory constrained)

    Parameters
    ----------
    n_rows : int
        Number of rows in the dataset.

    Returns
    -------
    tuple[int, int]
        Optimal (n_processes, batch_size) for the given data size.
    """
    if n_rows < 500_000:
        # Small data: multiprocessing overhead exceeds benefit
        return (1, 50)
    elif n_rows < 5_000_000:
        # Medium data: parallelism helps, memory is manageable
        return (2, 25)
    else:
        # Large data: memory constrained, use small batches
        return (2, 10)


def _worker_vectorized_batch(args):
    """Worker function - each process runs vectorized batches."""
    volumes, baseline_qx, shocked_qx, large_mask, n_trials, batch_size = args
    n_rows = len(volumes)

    # Pre-allocate for this worker's trials
    results = {
        "claim_volume_baseline": np.zeros(n_trials),
        "claim_volume_shocked": np.zeros(n_trials),
        "claim_count_baseline": np.zeros(n_trials, dtype=int),
        "claim_count_shocked": np.zeros(n_trials, dtype=int),
        "claim_count_baseline_10PLUS": np.zeros(n_trials, dtype=int),
        "claim_count_shocked_10PLUS": np.zeros(n_trials, dtype=int),
        "volume_baseline_10PLUS": np.zeros(n_trials),
        "volume_shocked_10PLUS": np.zeros(n_trials),
    }

    # Process in batches
    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)
        batch_n = batch_end - batch_start

        rand_numbers = np.random.rand(n_rows, batch_n)

        dead_baseline = rand_numbers < baseline_qx[:, np.newaxis]
        dead_shocked = rand_numbers < shocked_qx[:, np.newaxis]

        results["claim_count_baseline"][batch_start:batch_end] = dead_baseline.sum(
            axis=0
        )
        results["claim_count_shocked"][batch_start:batch_end] = dead_shocked.sum(axis=0)
        results["claim_volume_baseline"][batch_start:batch_end] = (
            dead_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["claim_volume_shocked"][batch_start:batch_end] = (
            dead_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)

        large_baseline = dead_baseline & large_mask[:, np.newaxis]
        large_shocked = dead_shocked & large_mask[:, np.newaxis]

        results["claim_count_baseline_10PLUS"][batch_start:batch_end] = (
            large_baseline.sum(axis=0)
        )
        results["claim_count_shocked_10PLUS"][batch_start:batch_end] = (
            large_shocked.sum(axis=0)
        )
        results["volume_baseline_10PLUS"][batch_start:batch_end] = (
            large_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["volume_shocked_10PLUS"][batch_start:batch_end] = (
            large_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)

    return results


def stochastic_runs_hybrid(
    data,
    n_trials,
    volume_col,
    baseline_qx_col,
    shocked_qx_col,
    n_processes="auto",
    batch_size="auto",
):
    """
    Hybrid parallel Monte Carlo simulation with vectorized batches.

    Uses multiprocessing with each worker running vectorized batch operations
    for optimal performance on large datasets.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing policy information.
    n_trials : int
        Number of Monte Carlo trials to run.
    volume_col : str
        Column name for claim volumes.
    baseline_qx_col : str
        Column name for baseline mortality rates (qx).
    shocked_qx_col : str
        Column name for shocked mortality rates (qx).
    n_processes : int or "auto", default="auto"
        Number of parallel processes. "auto" selects optimal based on data size.
    batch_size : int or "auto", default="auto"
        Number of trials per batch within each worker. "auto" selects optimal
        based on data size to balance speed and memory usage.

    Returns
    -------
    dict
        Dictionary containing simulation results:
        - claim_volume_baseline: Total claim volumes under baseline scenario
        - claim_volume_shocked: Total claim volumes under shocked scenario
        - claim_count_baseline: Claim counts under baseline scenario
        - claim_count_shocked: Claim counts under shocked scenario
        - claim_count_baseline_10PLUS: Large claim counts (>=10M) baseline
        - claim_count_shocked_10PLUS: Large claim counts (>=10M) shocked
        - volume_baseline_10PLUS: Large claim volumes baseline
        - volume_shocked_10PLUS: Large claim volumes shocked
    """
    n_rows = len(data)

    # Apply calibrated defaults if "auto"
    if n_processes == "auto" or batch_size == "auto":
        optimal_processes, optimal_batch = get_optimal_params(n_rows)
        if n_processes == "auto":
            n_processes = optimal_processes
        if batch_size == "auto":
            batch_size = optimal_batch

    # Extract data once
    volumes = data[volume_col].values
    baseline_qx = data[baseline_qx_col].values
    shocked_qx = data[shocked_qx_col].values
    large_mask = volumes >= 10_000_000

    # Split trials across processes
    trials_per_process = n_trials // n_processes
    remainder = n_trials % n_processes

    # Create worker arguments
    worker_args = []
    for i in range(n_processes):
        # Give remainder trials to first few workers
        worker_trials = trials_per_process + (1 if i < remainder else 0)
        worker_args.append(
            (volumes, baseline_qx, shocked_qx, large_mask, worker_trials, batch_size)
        )

    # Run parallel with thread-safe workers
    # initializer prevents BLAS/MKL thread oversubscription
    with Pool(n_processes, initializer=_init_worker) as pool:
        worker_results = pool.map(_worker_vectorized_batch, worker_args)

    # Concatenate results from all workers
    final_results = {}
    for key in worker_results[0].keys():
        final_results[key] = np.concatenate([w[key] for w in worker_results])

    return {k: v.tolist() for k, v in final_results.items()}
