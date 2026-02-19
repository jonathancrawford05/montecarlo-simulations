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

# Supported random modes
RANDOM_MODES = ("standard", "deterministic")


def deterministic_random(
    life_ids: np.ndarray,
    trial_ids: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate reproducible random numbers from (life_id, trial_id) pairs.

    Uses a splitmix64-style integer hash so that each (life_id, trial_id)
    pair always maps to the same float in [0, 1), regardless of portfolio
    composition, batch size, partition count, or run order.

    Memory: O(n_lives × len(trial_ids)) — call once per batch, not per
    full simulation, to avoid materialising an enormous array.

    Parameters
    ----------
    life_ids : np.ndarray, shape (n_lives,)
        Unique, stable integer identifier for each life.
    trial_ids : np.ndarray, shape (batch_size,)
        Global trial indices for this batch.
    seed : int, default=0
        Hash salt. Use the same value across all runs for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_lives, batch_size)
        Deterministic pseudo-random floats in [0, 1).
    """
    life_ids = np.asarray(life_ids, dtype=np.uint64)
    trial_ids = np.asarray(trial_ids, dtype=np.uint64)
    seed = np.uint64(seed)

    lives = life_ids[:, None]   # (n_lives, 1)
    trials = trial_ids[None, :] # (1, batch_size)

    # splitmix64-style hash combining life_id and trial_id
    x = lives * np.uint64(0x9E3779B97F4A7C15) + trials + seed
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))

    # Map to [0, 1) using 53-bit mantissa precision
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def simulate_trials(
    volumes: np.ndarray,
    baseline_qx: np.ndarray,
    shocked_qx: np.ndarray,
    n_trials: int,
    batch_size: int = 50,
    random_state: int | np.random.Generator | None = None,
    random_mode: str = "standard",
    life_ids: np.ndarray | None = None,
    global_trial_start: int = 0,
    seed: int = 0,
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
        Standard mode only. Seed or Generator for numpy's RNG.
        Ignored when random_mode="deterministic".
    random_mode : {"standard", "deterministic"}, default="standard"
        - "standard": sequential numpy RNG (fast, not subset-reproducible).
        - "deterministic": hash-based RNG keyed on (life_id, trial_id).
          Guarantees identical results for any life regardless of which
          other lives are present or how trials are partitioned.
    life_ids : np.ndarray or None, default=None
        Required when random_mode="deterministic". Unique integer identifier
        for each life, shape (n_lives,).
    global_trial_start : int, default=0
        Index of the first trial in this call within the overall simulation.
        Used in deterministic mode to form globally unique trial IDs.
        For local (non-Spark) runs this is always 0; for Spark workers each
        partition receives its correct offset.
    seed : int, default=0
        Hash salt for deterministic mode. Must be consistent across all
        calls in the same simulation run.

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
    if random_mode not in RANDOM_MODES:
        raise ValueError(
            f"random_mode must be one of {RANDOM_MODES}, got {random_mode!r}"
        )
    if random_mode == "deterministic" and life_ids is None:
        raise ValueError("life_ids is required when random_mode='deterministic'")

    n_rows = len(volumes)
    large_mask = volumes >= LARGE_CLAIM_THRESHOLD

    # Standard mode: set up numpy RNG once
    rng = None
    if random_mode == "standard":
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
        if random_mode == "deterministic":
            batch_trial_ids = np.arange(
                global_trial_start + batch_start,
                global_trial_start + batch_end,
                dtype=np.uint64,
            )
            rand_numbers = deterministic_random(life_ids, batch_trial_ids, seed)
        else:
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
