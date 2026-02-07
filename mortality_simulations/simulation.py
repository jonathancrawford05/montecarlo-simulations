"""
Monte Carlo mortality simulation with hybrid parallel processing.

This module implements a stochastic mortality simulation that uses multiprocessing
with vectorized batch operations for efficient large-scale simulations.

Includes confidence analysis tools for quantifying uncertainty in simulation
results, particularly for tail quantiles used in risk management.
"""

import os
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy import stats


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


def _worker_vectorized_batch_multi_year(args):
    """Worker function for multi-year simulation with vectorized batches."""
    (
        volumes_by_year,
        baseline_qx_by_year,
        shocked_qx_by_year,
        large_mask_by_year,
        n_trials,
        batch_size,
        return_yearly,
    ) = args
    n_rows, n_years = volumes_by_year.shape

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

    if return_yearly:
        results["claim_volume_baseline_by_year"] = np.zeros((n_years, n_trials))
        results["claim_volume_shocked_by_year"] = np.zeros((n_years, n_trials))
        results["claim_count_baseline_by_year"] = np.zeros((n_years, n_trials), dtype=int)
        results["claim_count_shocked_by_year"] = np.zeros((n_years, n_trials), dtype=int)
        results["claim_count_baseline_10PLUS_by_year"] = np.zeros(
            (n_years, n_trials), dtype=int
        )
        results["claim_count_shocked_10PLUS_by_year"] = np.zeros(
            (n_years, n_trials), dtype=int
        )
        results["volume_baseline_10PLUS_by_year"] = np.zeros((n_years, n_trials))
        results["volume_shocked_10PLUS_by_year"] = np.zeros((n_years, n_trials))

    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)
        batch_n = batch_end - batch_start

        alive_baseline = np.ones((n_rows, batch_n), dtype=bool)
        alive_shocked = np.ones((n_rows, batch_n), dtype=bool)

        for year_idx in range(n_years):
            rand_numbers = np.random.rand(n_rows, batch_n)

            baseline_qx = baseline_qx_by_year[:, year_idx]
            shocked_qx = shocked_qx_by_year[:, year_idx]
            volumes = volumes_by_year[:, year_idx]
            large_mask = large_mask_by_year[:, year_idx]

            dead_baseline = (rand_numbers < baseline_qx[:, np.newaxis]) & alive_baseline
            dead_shocked = (rand_numbers < shocked_qx[:, np.newaxis]) & alive_shocked

            baseline_count = dead_baseline.sum(axis=0)
            shocked_count = dead_shocked.sum(axis=0)
            baseline_volume = (dead_baseline * volumes[:, np.newaxis]).sum(axis=0)
            shocked_volume = (dead_shocked * volumes[:, np.newaxis]).sum(axis=0)

            large_baseline = dead_baseline & large_mask[:, np.newaxis]
            large_shocked = dead_shocked & large_mask[:, np.newaxis]

            baseline_count_large = large_baseline.sum(axis=0)
            shocked_count_large = large_shocked.sum(axis=0)
            baseline_volume_large = (large_baseline * volumes[:, np.newaxis]).sum(axis=0)
            shocked_volume_large = (large_shocked * volumes[:, np.newaxis]).sum(axis=0)

            results["claim_count_baseline"][batch_start:batch_end] += baseline_count
            results["claim_count_shocked"][batch_start:batch_end] += shocked_count
            results["claim_volume_baseline"][batch_start:batch_end] += baseline_volume
            results["claim_volume_shocked"][batch_start:batch_end] += shocked_volume
            results["claim_count_baseline_10PLUS"][
                batch_start:batch_end
            ] += baseline_count_large
            results["claim_count_shocked_10PLUS"][
                batch_start:batch_end
            ] += shocked_count_large
            results["volume_baseline_10PLUS"][
                batch_start:batch_end
            ] += baseline_volume_large
            results["volume_shocked_10PLUS"][
                batch_start:batch_end
            ] += shocked_volume_large

            if return_yearly:
                results["claim_count_baseline_by_year"][
                    year_idx, batch_start:batch_end
                ] = baseline_count
                results["claim_count_shocked_by_year"][
                    year_idx, batch_start:batch_end
                ] = shocked_count
                results["claim_volume_baseline_by_year"][
                    year_idx, batch_start:batch_end
                ] = baseline_volume
                results["claim_volume_shocked_by_year"][
                    year_idx, batch_start:batch_end
                ] = shocked_volume
                results["claim_count_baseline_10PLUS_by_year"][
                    year_idx, batch_start:batch_end
                ] = baseline_count_large
                results["claim_count_shocked_10PLUS_by_year"][
                    year_idx, batch_start:batch_end
                ] = shocked_count_large
                results["volume_baseline_10PLUS_by_year"][
                    year_idx, batch_start:batch_end
                ] = baseline_volume_large
                results["volume_shocked_10PLUS_by_year"][
                    year_idx, batch_start:batch_end
                ] = shocked_volume_large

            alive_baseline &= ~dead_baseline
            alive_shocked &= ~dead_shocked

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

    # Execute simulation
    if n_processes == 1:
        # Single process: bypass Pool entirely to avoid overhead
        # (process spawning, pickling, IPC)
        worker_results = [_worker_vectorized_batch(worker_args[0])]
    else:
        # Multi-process: use Pool with thread-safe workers
        # initializer prevents BLAS/MKL thread oversubscription
        with Pool(n_processes, initializer=_init_worker) as pool:
            worker_results = pool.map(_worker_vectorized_batch, worker_args)

    # Concatenate results from all workers
    final_results = {}
    for key in worker_results[0].keys():
        final_results[key] = np.concatenate([w[key] for w in worker_results])

    return {k: v.tolist() for k, v in final_results.items()}


def stochastic_runs_multi_year(
    data,
    n_trials,
    volume_col,
    baseline_qx_cols,
    shocked_qx_cols,
    volume_cols=None,
    n_processes="auto",
    batch_size="auto",
    return_yearly=True,
):
    """
    Multi-year hybrid Monte Carlo simulation with vectorized batches.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing policy information.
    n_trials : int
        Number of Monte Carlo trials to run.
    volume_col : str
        Column name for level claim volumes (used if volume_cols is None).
    baseline_qx_cols : list[str]
        Column names for baseline mortality rates per year.
    shocked_qx_cols : list[str]
        Column names for shocked mortality rates per year.
    volume_cols : list[str] or None, default=None
        Column names for per-year claim volumes. If None, volume_col is used
        for all years.
    n_processes : int or "auto", default="auto"
        Number of parallel processes. "auto" selects optimal based on data size.
    batch_size : int or "auto", default="auto"
        Number of trials per batch within each worker. "auto" selects optimal
        based on data size to balance speed and memory usage.
    return_yearly : bool, default=True
        Whether to return per-year results in addition to totals.

    Returns
    -------
    dict
        Dictionary containing total simulation results and optional per-year
        breakdowns with *_by_year keys.
    """
    if not baseline_qx_cols or not shocked_qx_cols:
        raise ValueError("baseline_qx_cols and shocked_qx_cols must be provided.")
    if len(baseline_qx_cols) != len(shocked_qx_cols):
        raise ValueError("baseline_qx_cols and shocked_qx_cols must be the same length.")
    if volume_cols is not None and len(volume_cols) != len(baseline_qx_cols):
        raise ValueError("volume_cols must match the length of baseline_qx_cols.")

    n_rows = len(data)
    n_years = len(baseline_qx_cols)

    if n_processes == "auto" or batch_size == "auto":
        optimal_processes, optimal_batch = get_optimal_params(n_rows)
        if n_processes == "auto":
            n_processes = optimal_processes
        if batch_size == "auto":
            batch_size = optimal_batch

    baseline_qx_by_year = data[baseline_qx_cols].values
    shocked_qx_by_year = data[shocked_qx_cols].values

    if volume_cols is None:
        volumes = data[volume_col].values
        volumes_by_year = np.repeat(volumes[:, np.newaxis], n_years, axis=1)
    else:
        volumes_by_year = data[volume_cols].values

    large_mask_by_year = volumes_by_year >= 10_000_000

    trials_per_process = n_trials // n_processes
    remainder = n_trials % n_processes

    worker_args = []
    for i in range(n_processes):
        worker_trials = trials_per_process + (1 if i < remainder else 0)
        worker_args.append(
            (
                volumes_by_year,
                baseline_qx_by_year,
                shocked_qx_by_year,
                large_mask_by_year,
                worker_trials,
                batch_size,
                return_yearly,
            )
        )

    if n_processes == 1:
        worker_results = [_worker_vectorized_batch_multi_year(worker_args[0])]
    else:
        with Pool(n_processes, initializer=_init_worker) as pool:
            worker_results = pool.map(_worker_vectorized_batch_multi_year, worker_args)

    final_results = {}
    for key in worker_results[0].keys():
        if key.endswith("_by_year"):
            final_results[key] = np.concatenate([w[key] for w in worker_results], axis=1)
        else:
            final_results[key] = np.concatenate([w[key] for w in worker_results])

    return {k: v.tolist() for k, v in final_results.items()}


def analyze_simulation_confidence(
    results: dict,
    metric: str,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
) -> dict:
    """
    Compute confidence interval for a quantile from simulation results.

    Uses the exact binomial method based on order statistics to determine
    which simulation outcomes bound the true quantile with the specified
    confidence level.

    Parameters
    ----------
    results : dict
        Output from stochastic_runs_hybrid containing simulation results.
    metric : str
        Key in results dict to analyze (e.g., "claim_volume_shocked").
    quantile : float, default=0.95
        Target quantile to estimate (0 to 1).
    confidence_level : float, default=0.95
        Confidence level for the interval (0 to 1).

    Returns
    -------
    dict
        Analysis results containing:
        - 'n_simulations': Number of simulations used
        - 'quantile': Target quantile
        - 'point_estimate': Sample quantile value
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'ci_width': Width of confidence interval
        - 'ci_width_relative': Width as percentage of point estimate
        - 'confidence_level': Confidence level used
        - 'order_stat_lower': Lower order statistic index used
        - 'order_stat_upper': Upper order statistic index used

    Examples
    --------
    >>> results = stochastic_runs_hybrid(data, n_trials=10000, ...)
    >>> ci = analyze_simulation_confidence(results, "claim_volume_shocked")
    >>> print(f"95th percentile: {ci['point_estimate']:,.0f}")
    >>> print(f"95% CI: [{ci['ci_lower']:,.0f}, {ci['ci_upper']:,.0f}]")
    """
    if metric not in results:
        available = list(results.keys())
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    data = np.array(results[metric])
    n = len(data)

    if n < 10:
        raise ValueError(f"Need at least 10 simulations, got {n}")

    # Sort data for order statistics
    sorted_data = np.sort(data)

    # Point estimate of quantile
    point_estimate = np.quantile(data, quantile)

    # Find order statistics for CI using binomial method
    # For quantile p, the j-th order statistic X_(j) satisfies:
    # P(X_(j) <= xi_p) = P(Binomial(n, p) >= j)
    alpha = 1 - confidence_level

    # Lower bound: find largest j such that P(Bin(n,p) < j) <= alpha/2
    # Upper bound: find smallest k such that P(Bin(n,p) >= k) <= alpha/2
    j_lower = stats.binom.ppf(alpha / 2, n, quantile)
    k_upper = stats.binom.ppf(1 - alpha / 2, n, quantile)

    # Convert to 0-based indices and bound to valid range
    idx_lower = max(0, int(j_lower) - 1)
    idx_upper = min(n - 1, int(k_upper))

    ci_lower = sorted_data[idx_lower]
    ci_upper = sorted_data[idx_upper]
    ci_width = ci_upper - ci_lower

    # Relative width (as percentage)
    if point_estimate != 0:
        ci_width_relative = (ci_width / point_estimate) * 100
    else:
        ci_width_relative = float("inf") if ci_width > 0 else 0.0

    return {
        "n_simulations": n,
        "quantile": quantile,
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_width),
        "ci_width_relative": float(ci_width_relative),
        "confidence_level": confidence_level,
        "order_stat_lower": idx_lower + 1,  # 1-based for reporting
        "order_stat_upper": idx_upper + 1,
    }


def compute_portfolio_moments(
    volumes: np.ndarray,
    qx: np.ndarray,
) -> dict:
    """
    Compute exact analytical moments of the aggregate claim distribution.

    For a portfolio of independent lives where life i has death probability
    qx_i and claim volume v_i, the total claim volume S = sum(v_i * D_i)
    where D_i ~ Bernoulli(qx_i). This function computes the exact moments
    of S using the individual life parameters.

    Parameters
    ----------
    volumes : np.ndarray
        Claim volume for each life (1-D array of length n_lives).
    qx : np.ndarray
        Mortality rate for each life (1-D array of length n_lives).

    Returns
    -------
    dict
        Exact distribution moments:
        - 'mean': E[S] = sum(v_i * qx_i)
        - 'variance': Var(S) = sum(v_i^2 * qx_i * (1 - qx_i))
        - 'std': Standard deviation sqrt(Var(S))
        - 'skewness': Standardised third central moment
        - 'third_central_moment': sum(v_i^3 * qx_i * (1-qx_i) * (1-2*qx_i))
        - 'n_lives': Number of lives in the portfolio
        - 'cv': Coefficient of variation (std / mean)

    Notes
    -----
    These moments are exact for a sum of independent heterogeneous
    Bernoulli random variables (Poisson binomial), weighted by volumes.
    No simulation is required.
    """
    volumes = np.asarray(volumes, dtype=float)
    qx = np.asarray(qx, dtype=float)

    if len(volumes) != len(qx):
        raise ValueError(
            f"volumes and qx must have same length, got {len(volumes)} and {len(qx)}"
        )

    pq = qx * (1 - qx)  # Bernoulli variance factor per life

    mean = np.sum(volumes * qx)
    variance = np.sum(volumes**2 * pq)
    std = np.sqrt(variance)
    third_central_moment = np.sum(volumes**3 * pq * (1 - 2 * qx))
    skewness = third_central_moment / std**3 if std > 0 else 0.0

    return {
        "mean": float(mean),
        "variance": float(variance),
        "std": float(std),
        "skewness": float(skewness),
        "third_central_moment": float(third_central_moment),
        "n_lives": len(volumes),
        "cv": float(std / mean) if mean > 0 else float("inf"),
    }


def estimate_quantile_ci_width(
    moments: dict,
    n_simulations: int,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
) -> dict:
    """
    Estimate confidence interval width for a quantile analytically.

    Uses the exact portfolio moments and the asymptotic distribution of
    sample quantiles to compute the expected CI width for a given number
    of simulations. Applies the Cornish-Fisher expansion to account for
    skewness in the aggregate claim distribution.

    Parameters
    ----------
    moments : dict
        Output from compute_portfolio_moments().
    n_simulations : int
        Number of Monte Carlo simulations.
    quantile : float, default=0.95
        Target quantile (0 to 1).
    confidence_level : float, default=0.95
        Confidence level for the interval (0 to 1).

    Returns
    -------
    dict
        Analytical CI estimates:
        - 'quantile_estimate_normal': Quantile under normal approximation
        - 'quantile_estimate_cf': Quantile under Cornish-Fisher expansion
        - 'ci_width_absolute': Expected CI width (absolute)
        - 'ci_width_relative': Expected CI width as % of quantile estimate
        - 'se_quantile': Standard error of the sample quantile
        - 'n_simulations': Simulations used
        - 'quantile': Target quantile
        - 'confidence_level': Confidence level
        - 'method': 'cornish_fisher' or 'normal'

    Notes
    -----
    The standard error of the p-th sample quantile from n iid draws is:

        SE(ξ̂_p) = sqrt(p(1-p) / n) / f(ξ_p)

    Under a normal approximation, f(ξ_p) = phi(z_p) / sigma, giving:

        SE = sqrt(p(1-p) / n) * sigma / phi(z_p)

    The Cornish-Fisher expansion corrects for skewness by adjusting both
    the quantile estimate and the local density. The adjusted quantile is:

        ξ_p ≈ mu + sigma * [z_p + (z_p^2 - 1) * gamma / 6]

    and the density correction factor is (1 + 2*z_p * gamma/6), which
    accounts for the stretching/compression of the distribution at the
    quantile point due to skewness.
    """
    mu = moments["mean"]
    sigma = moments["std"]
    gamma = moments["skewness"]

    if sigma <= 0:
        return {
            "quantile_estimate_normal": mu,
            "quantile_estimate_cf": mu,
            "ci_width_absolute": 0.0,
            "ci_width_relative": 0.0,
            "se_quantile": 0.0,
            "n_simulations": n_simulations,
            "quantile": quantile,
            "confidence_level": confidence_level,
            "method": "degenerate",
        }

    z_p = stats.norm.ppf(quantile)
    z_ci = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    phi_zp = stats.norm.pdf(z_p)

    # Normal approximation quantile
    quantile_normal = mu + sigma * z_p

    # Cornish-Fisher corrected quantile (accounts for skewness)
    cf_adjustment = (z_p**2 - 1) * gamma / 6
    w_p = z_p + cf_adjustment
    quantile_cf = mu + sigma * w_p

    # Density correction factor for Cornish-Fisher
    # dw/dz = 1 + 2*z_p*gamma/6, so the local density at the CF quantile
    # is phi(z_p) / (sigma * |dw/dz|)
    density_correction = 1 + 2 * z_p * gamma / 6

    # Guard against non-positive density correction (extreme skewness)
    if density_correction <= 0:
        # Fall back to normal approximation
        se = np.sqrt(quantile * (1 - quantile) / n_simulations) * sigma / phi_zp
        quantile_est = quantile_normal
        method = "normal"
    else:
        se = (
            np.sqrt(quantile * (1 - quantile) / n_simulations)
            * sigma
            * density_correction
            / phi_zp
        )
        quantile_est = quantile_cf
        method = "cornish_fisher"

    ci_width = 2 * z_ci * se

    if quantile_est != 0:
        ci_width_relative = (ci_width / quantile_est) * 100
    else:
        ci_width_relative = float("inf") if ci_width > 0 else 0.0

    return {
        "quantile_estimate_normal": float(quantile_normal),
        "quantile_estimate_cf": float(quantile_cf),
        "ci_width_absolute": float(ci_width),
        "ci_width_relative": float(ci_width_relative),
        "se_quantile": float(se),
        "n_simulations": n_simulations,
        "quantile": quantile,
        "confidence_level": confidence_level,
        "method": method,
    }


def plan_simulation_count(
    volumes: np.ndarray,
    qx: np.ndarray,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
    target_ci_width_relative: float = 1.0,
    simulation_scenarios: list[int] | None = None,
) -> dict:
    """
    Plan the required number of simulations from portfolio data alone.

    This is the recommended entry point for answering "how many simulations
    do I need?" It uses the exact analytical moments of the aggregate claim
    distribution (no simulation required) to project CI widths and recommend
    a simulation count for the target precision.

    Unlike :func:`estimate_required_simulations`, which extrapolates from a
    pilot run and is subject to sampling variability, this function gives
    deterministic results derived directly from the portfolio structure.

    Parameters
    ----------
    volumes : array-like
        Claim volume for each life.
    qx : array-like
        Mortality rate for each life.
    quantile : float, default=0.95
        Target quantile (0 to 1).
    confidence_level : float, default=0.95
        Confidence level for the interval (0 to 1).
    target_ci_width_relative : float, default=1.0
        Target CI width as percentage of quantile estimate.
    simulation_scenarios : list[int], optional
        Simulation counts to evaluate. Default: [1_000, 10_000, 100_000, 1_000_000].

    Returns
    -------
    dict
        Planning results containing:
        - 'portfolio_moments': Exact moments from compute_portfolio_moments()
        - 'target_ci_width_relative': Target CI width (%)
        - 'recommended_n': Smallest scenario achieving target, or extrapolated n
        - 'scenarios': List of dicts with CI width at each scenario count
        - 'quantile': Target quantile
        - 'confidence_level': Confidence level

    Examples
    --------
    >>> plan = plan_simulation_count(
    ...     data["volume"].values,
    ...     data["shocked_qx"].values,
    ...     target_ci_width_relative=1.0,
    ... )
    >>> print(f"Recommended: {plan['recommended_n']:,} simulations")
    """
    if simulation_scenarios is None:
        simulation_scenarios = [1_000, 10_000, 100_000, 1_000_000]

    volumes = np.asarray(volumes, dtype=float)
    qx = np.asarray(qx, dtype=float)

    moments = compute_portfolio_moments(volumes, qx)

    scenarios = []
    for n_sim in simulation_scenarios:
        ci_est = estimate_quantile_ci_width(
            moments, n_sim, quantile, confidence_level
        )
        scenarios.append({
            "n_simulations": n_sim,
            "ci_width_relative": ci_est["ci_width_relative"],
            "ci_width_absolute": ci_est["ci_width_absolute"],
            "se_quantile": ci_est["se_quantile"],
            "quantile_estimate": ci_est["quantile_estimate_cf"],
        })

    # Find recommended n: smallest scenario that achieves the target
    suitable = [
        s for s in scenarios
        if s["ci_width_relative"] <= target_ci_width_relative
    ]
    if suitable:
        recommended_n = min(suitable, key=lambda x: x["n_simulations"])["n_simulations"]
    else:
        # Extrapolate from the largest scenario using 1/sqrt(n) scaling
        best = min(scenarios, key=lambda x: x["ci_width_relative"])
        ratio = best["ci_width_relative"] / target_ci_width_relative
        recommended_n = int(np.ceil(best["n_simulations"] * ratio ** 2))

    return {
        "portfolio_moments": moments,
        "target_ci_width_relative": target_ci_width_relative,
        "recommended_n": recommended_n,
        "scenarios": scenarios,
        "quantile": quantile,
        "confidence_level": confidence_level,
    }


# Minimum pilot sizes for estimate_required_simulations stability
_PILOT_MIN = 1_000
_PILOT_RECOMMENDED = 5_000


def estimate_required_simulations(
    pilot_results: dict,
    metric: str,
    quantile: float = 0.95,
    target_ci_width_relative: float = 1.0,
    confidence_level: float = 0.95,
) -> dict:
    """
    Estimate required simulations by extrapolating from a pilot run.

    Measures the empirical CI width from the pilot and extrapolates via
    1/sqrt(n) scaling. Best used to **validate** that a completed run
    achieved the target precision.

    For **planning** (before running any simulations), prefer
    :func:`plan_simulation_count`, which uses exact analytical moments
    and produces deterministic results with no sampling variability.

    Parameters
    ----------
    pilot_results : dict
        Output from stochastic_runs_hybrid (pilot run with smaller n_trials).
    metric : str
        Key in results dict to analyze.
    quantile : float, default=0.95
        Target quantile (0 to 1).
    target_ci_width_relative : float, default=1.0
        Target CI width as percentage of point estimate (e.g., 1.0 = ±0.5%).
    confidence_level : float, default=0.95
        Confidence level for the interval.

    Returns
    -------
    dict
        Estimation results containing:
        - 'pilot_n': Number of simulations in pilot
        - 'pilot_ci_width_relative': Current CI width (%)
        - 'target_ci_width_relative': Target CI width (%)
        - 'estimated_n_required': Estimated simulations needed
        - 'scaling_factor': Ratio of required to pilot simulations
        - 'quantile': Target quantile
        - 'confidence_level': Confidence level
        - 'pilot_stability': 'low' (<1,000), 'moderate' (1,000-4,999),
          or 'high' (>=5,000)

    Notes
    -----
    The asymptotic standard error of a sample quantile is:
        SE(ξ̂_p) ≈ sqrt(p(1-p)/n) / f(ξ_p)

    Since CI width ∝ 1/sqrt(n), to reduce width by factor k requires
    k² times more simulations.

    Pilot stability:
        The empirical CI width is itself a random variable. Small pilots
        (<1,000 sims) can produce estimates that vary by 60%+ across
        replicates. At 5,000+ the CV drops below 25%.

    See Also
    --------
    plan_simulation_count : Deterministic planning using analytical moments.

    Examples
    --------
    >>> pilot = stochastic_runs_hybrid(data, n_trials=5000, ...)
    >>> est = estimate_required_simulations(
    ...     pilot, "claim_volume_shocked", target_ci_width_relative=1.0
    ... )
    >>> print(f"Need {est['estimated_n_required']:,} simulations for 1% CI width")
    """
    # Get current CI from pilot
    current_ci = analyze_simulation_confidence(
        pilot_results, metric, quantile, confidence_level
    )

    pilot_n = current_ci["n_simulations"]
    current_width_rel = current_ci["ci_width_relative"]

    # Assess pilot stability
    if pilot_n < _PILOT_MIN:
        pilot_stability = "low"
        warnings.warn(
            f"Pilot size ({pilot_n:,}) is below the minimum recommended "
            f"({_PILOT_MIN:,}). Estimates may be unreliable (CV > 30%). "
            f"Consider using plan_simulation_count() for deterministic "
            f"planning, or increase pilot to {_PILOT_RECOMMENDED:,}+ "
            f"simulations.",
            UserWarning,
            stacklevel=2,
        )
    elif pilot_n < _PILOT_RECOMMENDED:
        pilot_stability = "moderate"
    else:
        pilot_stability = "high"

    if current_width_rel == 0:
        return {
            "pilot_n": pilot_n,
            "pilot_ci_width_relative": current_width_rel,
            "target_ci_width_relative": target_ci_width_relative,
            "estimated_n_required": pilot_n,
            "scaling_factor": 1.0,
            "quantile": quantile,
            "confidence_level": confidence_level,
            "pilot_stability": pilot_stability,
            "note": "CI width is zero; pilot may be sufficient or data has no variability",
        }

    # CI width scales as 1/sqrt(n), so n scales as (width_ratio)^2
    width_ratio = current_width_rel / target_ci_width_relative
    scaling_factor = width_ratio ** 2
    estimated_n = int(np.ceil(pilot_n * scaling_factor))

    return {
        "pilot_n": pilot_n,
        "pilot_ci_width_relative": float(current_width_rel),
        "target_ci_width_relative": target_ci_width_relative,
        "estimated_n_required": estimated_n,
        "scaling_factor": float(scaling_factor),
        "quantile": quantile,
        "confidence_level": confidence_level,
        "pilot_stability": pilot_stability,
    }


def generate_confidence_summary(
    results: dict,
    metric: str,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
    simulation_scenarios: list[int] | None = None,
    portfolio_data: dict | None = None,
) -> dict:
    """
    Generate a comprehensive confidence analysis summary.

    Analyzes the current simulation results and projects confidence intervals
    for different simulation counts (10K, 100K, 1M by default). When portfolio
    data is provided, includes analytical projections derived from the exact
    moments of the heterogeneous Bernoulli portfolio, accounting for skewness
    via the Cornish-Fisher expansion.

    Parameters
    ----------
    results : dict
        Output from stochastic_runs_hybrid containing simulation results.
    metric : str
        Key in results dict to analyze (e.g., "claim_volume_shocked").
    quantile : float, default=0.95
        Target quantile to analyze.
    confidence_level : float, default=0.95
        Confidence level for intervals.
    simulation_scenarios : list[int], optional
        Simulation counts to project. Default: [10_000, 100_000, 1_000_000].
    portfolio_data : dict, optional
        Portfolio structure for analytical projections. Must contain:
        - 'volumes': array-like of claim volumes per life
        - 'qx': array-like of mortality rates per life
        When provided, the summary includes analytical CI projections that
        account for individual-life heterogeneity and distribution skewness.

    Returns
    -------
    dict
        Summary containing:
        - 'current_analysis': Full analysis of current results
        - 'scenarios': List of projected CI widths for each scenario
        - 'recommendation': Suggested simulation count based on use case
        - 'metric': Metric analyzed
        - 'quantile': Quantile analyzed
        - 'portfolio_moments': (if portfolio_data provided) Exact moments
        - 'analytical_scenarios': (if portfolio_data provided) Analytical projections

    Examples
    --------
    >>> results = stochastic_runs_hybrid(data, n_trials=10000, ...)
    >>> summary = generate_confidence_summary(
    ...     results, "claim_volume_shocked",
    ...     portfolio_data={
    ...         "volumes": data["volume"].values,
    ...         "qx": data["shocked_qx"].values,
    ...     },
    ... )
    >>> print_confidence_summary(summary)
    """
    if simulation_scenarios is None:
        simulation_scenarios = [10_000, 100_000, 1_000_000]

    # Current analysis (empirical, from simulation results)
    current = analyze_simulation_confidence(
        results, metric, quantile, confidence_level
    )

    current_n = current["n_simulations"]
    current_width_rel = current["ci_width_relative"]

    # Empirical projections (1/sqrt(n) scaling from observed CI)
    scenarios = []
    for n_sim in simulation_scenarios:
        scaling = np.sqrt(current_n / n_sim)
        projected_width = current_width_rel * scaling
        projected_abs_width = current["ci_width"] * scaling

        scenarios.append({
            "n_simulations": n_sim,
            "projected_ci_width_relative": float(projected_width),
            "projected_ci_width_absolute": float(projected_abs_width),
            "is_current": n_sim == current_n,
        })

    # Generate recommendation based on common use cases
    recommendation = _generate_recommendation(scenarios, quantile)

    summary = {
        "metric": metric,
        "quantile": quantile,
        "confidence_level": confidence_level,
        "current_analysis": current,
        "scenarios": scenarios,
        "recommendation": recommendation,
    }

    # Analytical projections (if portfolio data provided)
    if portfolio_data is not None:
        volumes = np.asarray(portfolio_data["volumes"])
        qx = np.asarray(portfolio_data["qx"])

        moments = compute_portfolio_moments(volumes, qx)
        summary["portfolio_moments"] = moments

        analytical_scenarios = []
        for n_sim in simulation_scenarios:
            ci_est = estimate_quantile_ci_width(
                moments, n_sim, quantile, confidence_level
            )
            analytical_scenarios.append(ci_est)

        summary["analytical_scenarios"] = analytical_scenarios

        # Analytical recommendation uses analytical CI widths
        analytical_scenario_dicts = [
            {
                "n_simulations": s["n_simulations"],
                "projected_ci_width_relative": s["ci_width_relative"],
            }
            for s in analytical_scenarios
        ]
        summary["analytical_recommendation"] = _generate_recommendation(
            analytical_scenario_dicts, quantile
        )

    return summary


def _generate_recommendation(scenarios: list[dict], quantile: float) -> dict:
    """Generate recommendation based on projected CI widths."""
    recommendations = {
        "exploratory": {
            "threshold": 5.0,
            "description": "Exploratory analysis, rough estimates",
        },
        "reporting": {
            "threshold": 2.0,
            "description": "Management reporting, internal risk metrics",
        },
        "regulatory": {
            "threshold": 1.0,
            "description": "Regulatory capital, external reporting",
        },
        "precision": {
            "threshold": 0.5,
            "description": "High-precision requirements, model validation",
        },
    }

    result = {}
    for use_case, config in recommendations.items():
        threshold = config["threshold"]
        # Find smallest n that achieves threshold
        suitable = [s for s in scenarios if s["projected_ci_width_relative"] <= threshold]
        if suitable:
            best = min(suitable, key=lambda x: x["n_simulations"])
            result[use_case] = {
                "recommended_n": best["n_simulations"],
                "achieves_ci_width": best["projected_ci_width_relative"],
                "threshold": threshold,
                "description": config["description"],
            }
        else:
            # Extrapolate required n
            # Find the scenario with smallest width and extrapolate
            best_available = min(scenarios, key=lambda x: x["projected_ci_width_relative"])
            ratio = best_available["projected_ci_width_relative"] / threshold
            extrapolated_n = int(best_available["n_simulations"] * (ratio ** 2))
            result[use_case] = {
                "recommended_n": extrapolated_n,
                "achieves_ci_width": threshold,
                "threshold": threshold,
                "description": config["description"],
                "extrapolated": True,
            }

    return result


def print_confidence_summary(summary: dict) -> None:
    """
    Print a formatted confidence analysis summary.

    When the summary includes analytical projections (from portfolio_data),
    displays both empirical and analytical CI widths side-by-side for
    comparison.

    Parameters
    ----------
    summary : dict
        Output from generate_confidence_summary().
    """
    current = summary["current_analysis"]
    has_analytical = "analytical_scenarios" in summary
    print("=" * 70)
    print("SIMULATION CONFIDENCE ANALYSIS")
    print(f"Metric: {summary['metric']}")
    print(f"Quantile: {summary['quantile'] * 100:.0f}th percentile")
    print(f"Confidence Level: {summary['confidence_level'] * 100:.0f}%")
    print("=" * 70)

    # Portfolio moments section (if available)
    if has_analytical:
        moments = summary["portfolio_moments"]
        print(f"\nPORTFOLIO STRUCTURE ({moments['n_lives']:,} lives)")
        print("-" * 50)
        print(f"  Analytical mean:    {moments['mean']:>20,.2f}")
        print(f"  Analytical std dev: {moments['std']:>20,.2f}")
        print(f"  Coeff. of variation:{moments['cv']:>19.4f}")
        print(f"  Skewness:           {moments['skewness']:>20.4f}")

        # Show analytical vs simulation quantile estimates
        a_scen = summary["analytical_scenarios"][0]
        print(f"\n  Quantile estimates:")
        print(f"    Simulation (empirical):  {current['point_estimate']:>16,.2f}")
        print(f"    Normal approximation:    {a_scen['quantile_estimate_normal']:>16,.2f}")
        print(f"    Cornish-Fisher adjusted: {a_scen['quantile_estimate_cf']:>16,.2f}")

    print(f"\nCURRENT RESULTS ({current['n_simulations']:,} simulations)")
    print("-" * 50)
    print(f"  Point estimate:     {current['point_estimate']:>20,.2f}")
    print(f"  CI lower bound:     {current['ci_lower']:>20,.2f}")
    print(f"  CI upper bound:     {current['ci_upper']:>20,.2f}")
    print(f"  CI width:           {current['ci_width']:>20,.2f}")
    print(f"  CI width (relative):{current['ci_width_relative']:>19.2f}%")

    # Side-by-side projection table
    print(f"\nPROJECTED CI WIDTH BY SIMULATION COUNT")
    print("-" * 70)
    if has_analytical:
        print(
            f"  {'Simulations':>15}  "
            f"{'Empirical (%)':>15}  "
            f"{'Analytical (%)':>15}  "
            f"{'Status':<10}"
        )
        for emp, ana in zip(summary["scenarios"], summary["analytical_scenarios"]):
            status = "current" if emp["is_current"] else ""
            print(
                f"  {emp['n_simulations']:>15,}  "
                f"{emp['projected_ci_width_relative']:>14.2f}%  "
                f"{ana['ci_width_relative']:>14.2f}%  "
                f"{status:<10}"
            )
    else:
        print(f"  {'Simulations':>15}  {'CI Width (%)':>15}  {'Status':<15}")
        for scenario in summary["scenarios"]:
            status = "current" if scenario["is_current"] else ""
            print(
                f"  {scenario['n_simulations']:>15,}  "
                f"{scenario['projected_ci_width_relative']:>14.2f}%  "
                f"{status:<15}"
            )

    # Recommendations
    rec_source = "analytical_recommendation" if has_analytical else "recommendation"
    rec_label = "ANALYTICAL" if has_analytical else "EMPIRICAL"
    recs = summary.get(rec_source, summary["recommendation"])

    print(f"\nRECOMMENDATIONS BY USE CASE ({rec_label})")
    print("-" * 50)
    for use_case, rec in recs.items():
        extra = " (extrapolated)" if rec.get("extrapolated") else ""
        print(f"  {rec['description']}:")
        print(
            f"    -> {rec['recommended_n']:,} simulations "
            f"(CI <= {rec['threshold']}%){extra}"
        )
    print("=" * 70)
