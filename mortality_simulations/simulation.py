"""
Monte Carlo mortality simulation with hybrid parallel processing.

This module implements a stochastic mortality simulation that uses multiprocessing
with vectorized batch operations for efficient large-scale simulations.

Includes confidence analysis tools for quantifying uncertainty in simulation
results, particularly for tail quantiles used in risk management.
"""

import os
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


def estimate_required_simulations(
    pilot_results: dict,
    metric: str,
    quantile: float = 0.95,
    target_ci_width_relative: float = 1.0,
    confidence_level: float = 0.95,
) -> dict:
    """
    Estimate number of simulations required for a target confidence interval width.

    Uses pilot run results to estimate the density at the quantile, then
    applies the asymptotic formula for quantile standard error to determine
    the required sample size.

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

    Notes
    -----
    The asymptotic standard error of a sample quantile is:
        SE(ξ̂_p) ≈ sqrt(p(1-p)/n) / f(ξ_p)

    Since CI width ∝ 1/sqrt(n), to reduce width by factor k requires
    k² times more simulations.

    Examples
    --------
    >>> pilot = stochastic_runs_hybrid(data, n_trials=1000, ...)
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

    if current_width_rel == 0:
        return {
            "pilot_n": pilot_n,
            "pilot_ci_width_relative": current_width_rel,
            "target_ci_width_relative": target_ci_width_relative,
            "estimated_n_required": pilot_n,
            "scaling_factor": 1.0,
            "quantile": quantile,
            "confidence_level": confidence_level,
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
    }


def generate_confidence_summary(
    results: dict,
    metric: str,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
    simulation_scenarios: list[int] | None = None,
) -> dict:
    """
    Generate a comprehensive confidence analysis summary.

    Analyzes the current simulation results and projects confidence intervals
    for different simulation counts (10K, 100K, 1M by default).

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

    Returns
    -------
    dict
        Summary containing:
        - 'current_analysis': Full analysis of current results
        - 'scenarios': List of projected CI widths for each scenario
        - 'recommendation': Suggested simulation count based on use case
        - 'metric': Metric analyzed
        - 'quantile': Quantile analyzed

    Examples
    --------
    >>> results = stochastic_runs_hybrid(data, n_trials=10000, ...)
    >>> summary = generate_confidence_summary(results, "claim_volume_shocked")
    >>> for scenario in summary['scenarios']:
    ...     print(f"{scenario['n_simulations']:>10,}: CI width ≈ {scenario['projected_ci_width_relative']:.2f}%")
    """
    if simulation_scenarios is None:
        simulation_scenarios = [10_000, 100_000, 1_000_000]

    # Current analysis
    current = analyze_simulation_confidence(
        results, metric, quantile, confidence_level
    )

    current_n = current["n_simulations"]
    current_width_rel = current["ci_width_relative"]

    # Project CI widths for each scenario
    # CI width scales as 1/sqrt(n)
    scenarios = []
    for n_sim in simulation_scenarios:
        scaling = np.sqrt(current_n / n_sim)
        projected_width = current_width_rel * scaling

        # Also project absolute CI width
        projected_abs_width = current["ci_width"] * scaling

        scenarios.append({
            "n_simulations": n_sim,
            "projected_ci_width_relative": float(projected_width),
            "projected_ci_width_absolute": float(projected_abs_width),
            "is_current": n_sim == current_n,
        })

    # Generate recommendation based on common use cases
    recommendation = _generate_recommendation(scenarios, quantile)

    return {
        "metric": metric,
        "quantile": quantile,
        "confidence_level": confidence_level,
        "current_analysis": current,
        "scenarios": scenarios,
        "recommendation": recommendation,
    }


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

    Parameters
    ----------
    summary : dict
        Output from generate_confidence_summary().
    """
    current = summary["current_analysis"]
    print("=" * 70)
    print(f"SIMULATION CONFIDENCE ANALYSIS")
    print(f"Metric: {summary['metric']}")
    print(f"Quantile: {summary['quantile'] * 100:.0f}th percentile")
    print(f"Confidence Level: {summary['confidence_level'] * 100:.0f}%")
    print("=" * 70)

    print(f"\nCURRENT RESULTS ({current['n_simulations']:,} simulations)")
    print("-" * 50)
    print(f"  Point estimate:     {current['point_estimate']:>20,.2f}")
    print(f"  CI lower bound:     {current['ci_lower']:>20,.2f}")
    print(f"  CI upper bound:     {current['ci_upper']:>20,.2f}")
    print(f"  CI width:           {current['ci_width']:>20,.2f}")
    print(f"  CI width (relative):{current['ci_width_relative']:>19.2f}%")

    print(f"\nPROJECTED CI WIDTH BY SIMULATION COUNT")
    print("-" * 50)
    print(f"  {'Simulations':>15}  {'CI Width (%)':>15}  {'Status':<15}")
    for scenario in summary["scenarios"]:
        status = "← current" if scenario["is_current"] else ""
        print(
            f"  {scenario['n_simulations']:>15,}  "
            f"{scenario['projected_ci_width_relative']:>14.2f}%  "
            f"{status:<15}"
        )

    print(f"\nRECOMMENDATIONS BY USE CASE")
    print("-" * 50)
    for use_case, rec in summary["recommendation"].items():
        extra = " (extrapolated)" if rec.get("extrapolated") else ""
        print(f"  {rec['description']}:")
        print(f"    → {rec['recommended_n']:,} simulations (CI ≤ {rec['threshold']}%){extra}")
    print("=" * 70)
