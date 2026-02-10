"""
Core simulation logic shared between local and distributed implementations.

This module contains the vectorized Bernoulli mortality simulation that can be
called from either the multiprocessing backend (simulation.py) or the Spark
backend (spark_simulation.py).
"""

import numpy as np

# Result keys returned by simulation functions
RESULT_KEYS = [
    "claim_volume_baseline",
    "claim_volume_shocked",
    "claim_count_baseline",
    "claim_count_shocked",
    "claim_count_baseline_10PLUS",
    "claim_count_shocked_10PLUS",
    "volume_baseline_10PLUS",
    "volume_shocked_10PLUS",
]

# Threshold for "large" claims (10M+)
LARGE_CLAIM_THRESHOLD = 10_000_000


def simulate_trials(
    volumes: np.ndarray,
    baseline_qx: np.ndarray,
    shocked_qx: np.ndarray,
    n_trials: int,
    batch_size: int = 50,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Run vectorized Bernoulli mortality simulation.

    Each life dies with probability qx (independently). The simulation computes
    aggregate claim volumes and counts across all trials.

    Parameters
    ----------
    volumes : np.ndarray
        Claim volume for each life, shape (n_lives,).
    baseline_qx : np.ndarray
        Baseline mortality rate for each life, shape (n_lives,).
    shocked_qx : np.ndarray
        Shocked mortality rate for each life, shape (n_lives,).
    n_trials : int
        Number of Monte Carlo trials to run.
    batch_size : int, default=50
        Number of trials to process in each vectorized batch.
        Larger batches are faster but use more memory.
    random_state : int, Generator, or None, default=None
        Random state for reproducibility. Can be:
        - int: seed for numpy.random.default_rng()
        - Generator: existing numpy random generator
        - None: use numpy's global random state

    Returns
    -------
    dict[str, np.ndarray]
        Simulation results with keys:
        - claim_volume_baseline: Total claim volume per trial (baseline)
        - claim_volume_shocked: Total claim volume per trial (shocked)
        - claim_count_baseline: Number of deaths per trial (baseline)
        - claim_count_shocked: Number of deaths per trial (shocked)
        - claim_count_baseline_10PLUS: Large claim count per trial (baseline)
        - claim_count_shocked_10PLUS: Large claim count per trial (shocked)
        - volume_baseline_10PLUS: Large claim volume per trial (baseline)
        - volume_shocked_10PLUS: Large claim volume per trial (shocked)
    """
    n_rows = len(volumes)
    large_mask = volumes >= LARGE_CLAIM_THRESHOLD

    # Set up random number generation
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Pre-allocate result arrays
    results = {
        "claim_volume_baseline": np.zeros(n_trials),
        "claim_volume_shocked": np.zeros(n_trials),
        "claim_count_baseline": np.zeros(n_trials, dtype=np.int64),
        "claim_count_shocked": np.zeros(n_trials, dtype=np.int64),
        "claim_count_baseline_10PLUS": np.zeros(n_trials, dtype=np.int64),
        "claim_count_shocked_10PLUS": np.zeros(n_trials, dtype=np.int64),
        "volume_baseline_10PLUS": np.zeros(n_trials),
        "volume_shocked_10PLUS": np.zeros(n_trials),
    }

    # Process in batches to manage memory
    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)
        batch_n = batch_end - batch_start

        # Generate random numbers: shape (n_lives, batch_n)
        rand_numbers = rng.random((n_rows, batch_n))

        # Determine deaths: each life dies if rand < qx
        dead_baseline = rand_numbers < baseline_qx[:, np.newaxis]
        dead_shocked = rand_numbers < shocked_qx[:, np.newaxis]

        # Aggregate counts and volumes
        results["claim_count_baseline"][batch_start:batch_end] = dead_baseline.sum(axis=0)
        results["claim_count_shocked"][batch_start:batch_end] = dead_shocked.sum(axis=0)
        results["claim_volume_baseline"][batch_start:batch_end] = (
            dead_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["claim_volume_shocked"][batch_start:batch_end] = (
            dead_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)

        # Large claims (10M+)
        large_baseline = dead_baseline & large_mask[:, np.newaxis]
        large_shocked = dead_shocked & large_mask[:, np.newaxis]

        results["claim_count_baseline_10PLUS"][batch_start:batch_end] = large_baseline.sum(axis=0)
        results["claim_count_shocked_10PLUS"][batch_start:batch_end] = large_shocked.sum(axis=0)
        results["volume_baseline_10PLUS"][batch_start:batch_end] = (
            large_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["volume_shocked_10PLUS"][batch_start:batch_end] = (
            large_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)

    return results


def concatenate_results(result_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Concatenate results from multiple simulation runs.

    Parameters
    ----------
    result_list : list of dict
        List of result dictionaries from simulate_trials().

    Returns
    -------
    dict[str, np.ndarray]
        Combined results with all trials concatenated.
    """
    if not result_list:
        raise ValueError("result_list cannot be empty")

    combined = {}
    for key in RESULT_KEYS:
        combined[key] = np.concatenate([r[key] for r in result_list])

    return combined


def results_to_lists(results: dict[str, np.ndarray]) -> dict[str, list]:
    """
    Convert numpy arrays to Python lists for JSON serialization.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Simulation results with numpy arrays.

    Returns
    -------
    dict[str, list]
        Same results with Python lists instead of numpy arrays.
    """
    return {k: v.tolist() for k, v in results.items()}
