"""
Monte Carlo mortality simulation with hybrid parallel processing.

This module implements a stochastic mortality simulation that uses multiprocessing
with vectorized batch operations for efficient large-scale simulations.
"""

from multiprocessing import Pool, cpu_count

import numpy as np


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
    n_processes=None,
    batch_size=50,
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
    n_processes : int, optional
        Number of parallel processes. Defaults to cpu_count().
    batch_size : int, default=50
        Number of trials per batch within each worker.
        Smaller batch size reduces memory usage (50 for 20M rows ≈ 8GB per batch).

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
    if n_processes is None:
        n_processes = cpu_count()

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

    # Run parallel
    with Pool(n_processes) as pool:
        worker_results = pool.map(_worker_vectorized_batch, worker_args)

    # Concatenate results from all workers
    final_results = {}
    for key in worker_results[0].keys():
        final_results[key] = np.concatenate([w[key] for w in worker_results])

    return {k: v.tolist() for k, v in final_results.items()}
