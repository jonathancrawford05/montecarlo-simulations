"""
Joint life mortality simulation with contagion modeling.

This module implements Monte Carlo simulation for joint life (last-to-die) policies
using Gaussian copulas to model mortality correlation between paired lives.

Key concepts:
- Last-to-die: Claim triggers only when both lives have died
- Contagion: When one life dies, the surviving life's mortality rate increases
- Gaussian copula: Models dependency between two lives via correlated uniforms
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import norm

from mortality_simulations.simulation import _init_worker, get_optimal_params


# Pre-defined parameter configurations
PARAM_CONFIGS = {
    "correlation_only": {
        "rho": 0.25,
        "contagion_multiplier": 1.0,
        "description": (
            "Uses correlation (rho) only to model joint death dependency. "
            "Simpler single-parameter approach. Recommended for single-period models."
        ),
    },
    "contagion_only": {
        "rho": 0.0,
        "contagion_multiplier": 1.5,
        "description": (
            "Uses contagion multiplier only (independent base mortality). "
            "Models pure widow/widower effect without shared environment correlation."
        ),
    },
    "low_correlation_with_contagion": {
        "rho": 0.05,
        "contagion_multiplier": 1.35,
        "description": (
            "Low correlation for shared environment, plus time-weighted contagion. "
            "Separates the two effects while avoiding double-counting. "
            "Contagion of 1.35 reflects annual average of decaying widow effect."
        ),
    },
    "moderate_combined": {
        "rho": 0.10,
        "contagion_multiplier": 1.25,
        "description": (
            "Moderate values of both parameters. "
            "Use when you have evidence for both shared environment and causal contagion."
        ),
    },
    "multi_period_literature": {
        "rho": 0.05,
        "contagion_multiplier": 1.5,
        "description": (
            "Based on actuarial literature for multi-period projections. "
            "Low rho for shared environment, standard contagion for widow effect. "
            "In multi-period models, contagion applies to future periods after first death."
        ),
    },
}


def get_joint_life_params(config: str = "correlation_only") -> dict:
    """
    Get pre-defined parameter configuration for joint life simulation.

    Provides recommended (rho, contagion_multiplier) combinations based on
    different modeling approaches. Helps avoid double-counting effects.

    Parameters
    ----------
    config : str, default="correlation_only"
        Configuration name. Options:
        - "correlation_only": Single parameter (rho=0.25, contagion=1.0)
        - "contagion_only": Pure widow effect (rho=0.0, contagion=1.5)
        - "low_correlation_with_contagion": Separated effects (rho=0.05, contagion=1.35)
        - "moderate_combined": Both effects moderate (rho=0.10, contagion=1.25)
        - "multi_period_literature": For multi-year projections (rho=0.05, contagion=1.5)

    Returns
    -------
    dict
        Configuration with keys: 'rho', 'contagion_multiplier', 'description'

    Examples
    --------
    >>> params = get_joint_life_params("correlation_only")
    >>> results = stochastic_runs_joint_life(
    ...     data=data,
    ...     rho=params["rho"],
    ...     contagion_multiplier=params["contagion_multiplier"],
    ...     ...
    ... )
    """
    if config not in PARAM_CONFIGS:
        available = ", ".join(PARAM_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config}'. Available: {available}")

    return PARAM_CONFIGS[config].copy()


def list_param_configs() -> dict:
    """
    List all available parameter configurations with descriptions.

    Returns
    -------
    dict
        Dictionary of all configurations and their descriptions.
    """
    return {name: cfg["description"] for name, cfg in PARAM_CONFIGS.items()}


def generate_correlated_uniforms(n_pairs: int, n_trials: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated uniform random variables using Gaussian copula.

    Uses the Cholesky decomposition approach:
    Z1 ~ N(0,1)
    Z2 = rho * Z1 + sqrt(1 - rho^2) * Z2_indep
    U1 = Phi(Z1), U2 = Phi(Z2)

    Parameters
    ----------
    n_pairs : int
        Number of joint life pairs (policies).
    n_trials : int
        Number of simulation trials.
    rho : float
        Correlation parameter for Gaussian copula (-1 to 1).
        Positive rho = contagion (deaths are positively correlated).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        U_A, U_B: Correlated uniform arrays of shape (n_pairs, n_trials).
    """
    if not -1 <= rho <= 1:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    # Generate independent standard normals
    Z1 = np.random.randn(n_pairs, n_trials)
    Z2_indep = np.random.randn(n_pairs, n_trials)

    # Induce correlation via Cholesky factor
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2_indep

    # Transform to uniforms via normal CDF
    U_A = norm.cdf(Z1)
    U_B = norm.cdf(Z2)

    return U_A, U_B


def prepare_joint_life_data(
    data: pd.DataFrame,
    policy_col: str,
    life_id_col: str,
    volume_col: str,
    baseline_qx_col: str,
    shocked_qx_col: str,
) -> dict:
    """
    Prepare joint life data by pairing lives within each policy.

    Transforms row-per-life data into paired arrays suitable for vectorized
    joint life simulation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with one row per life. Must contain joint life policies
        with exactly 2 lives per policy.
    policy_col : str
        Column identifying the policy (links lives together).
    life_id_col : str
        Column identifying the life within a policy ('A' or 'B').
    volume_col : str
        Column with claim volumes.
    baseline_qx_col : str
        Column with baseline mortality rates.
    shocked_qx_col : str
        Column with shocked mortality rates.

    Returns
    -------
    dict
        Paired data arrays:
        - 'policy_numbers': Array of policy identifiers
        - 'volume': Claim volume per policy (from life A, assumed same for both)
        - 'qx_A_baseline': Baseline qx for life A
        - 'qx_B_baseline': Baseline qx for life B
        - 'qx_A_shocked': Shocked qx for life A
        - 'qx_B_shocked': Shocked qx for life B
        - 'n_policies': Number of joint life policies
    """
    # Validate life IDs
    valid_life_ids = {"A", "B"}
    actual_life_ids = set(data[life_id_col].unique())
    if not actual_life_ids.issubset(valid_life_ids):
        raise ValueError(
            f"life_id_col must contain only 'A' and 'B', got {actual_life_ids}"
        )

    # Split by life ID
    life_a = data[data[life_id_col] == "A"].set_index(policy_col)
    life_b = data[data[life_id_col] == "B"].set_index(policy_col)

    # Find policies with both lives
    common_policies = life_a.index.intersection(life_b.index)

    if len(common_policies) == 0:
        raise ValueError("No policies found with both life A and life B")

    # Check for policies with missing lives
    a_only = set(life_a.index) - set(common_policies)
    b_only = set(life_b.index) - set(common_policies)
    if a_only or b_only:
        import warnings
        warnings.warn(
            f"Found {len(a_only)} policies with only life A, "
            f"{len(b_only)} with only life B. These will be excluded."
        )

    # Align data
    life_a = life_a.loc[common_policies]
    life_b = life_b.loc[common_policies]

    return {
        "policy_numbers": common_policies.values,
        "volume": life_a[volume_col].values,  # Assume same volume for both lives
        "qx_A_baseline": life_a[baseline_qx_col].values,
        "qx_B_baseline": life_b[baseline_qx_col].values,
        "qx_A_shocked": life_a[shocked_qx_col].values,
        "qx_B_shocked": life_b[shocked_qx_col].values,
        "n_policies": len(common_policies),
    }


def apply_contagion(
    qx_survivor: np.ndarray,
    partner_died: np.ndarray,
    contagion_multiplier: float = 1.5,
) -> np.ndarray:
    """
    Apply contagion effect to surviving life's mortality rate.

    When one life dies, the surviving life's qx is multiplied by the
    contagion factor (widow/widower effect).

    Parameters
    ----------
    qx_survivor : np.ndarray
        Original mortality rates for the surviving life.
    partner_died : np.ndarray
        Boolean mask indicating where the partner has died.
    contagion_multiplier : float, default=1.5
        Factor by which to multiply qx when partner dies.
        Literature suggests 1.5-2.0 for immediate effect.

    Returns
    -------
    np.ndarray
        Adjusted mortality rates with contagion applied.
    """
    adjusted_qx = qx_survivor.copy()
    adjusted_qx[partner_died] = np.minimum(
        adjusted_qx[partner_died] * contagion_multiplier,
        1.0  # Cap at 100% mortality
    )
    return adjusted_qx


def _worker_joint_life_batch(args):
    """
    Worker function for joint life simulation with vectorized batches.

    Handles last-to-die logic: claim triggers only when both lives have died.
    """
    (
        qx_A_baseline, qx_B_baseline,
        qx_A_shocked, qx_B_shocked,
        volumes, large_mask,
        n_trials, batch_size, rho, contagion_multiplier
    ) = args

    n_policies = len(volumes)

    # Pre-allocate results
    results = {
        # Last-to-die claims (both must die)
        "claim_volume_baseline": np.zeros(n_trials),
        "claim_volume_shocked": np.zeros(n_trials),
        "claim_count_baseline": np.zeros(n_trials, dtype=int),
        "claim_count_shocked": np.zeros(n_trials, dtype=int),
        # Large claims (volume >= 10M)
        "claim_count_baseline_10PLUS": np.zeros(n_trials, dtype=int),
        "claim_count_shocked_10PLUS": np.zeros(n_trials, dtype=int),
        "volume_baseline_10PLUS": np.zeros(n_trials),
        "volume_shocked_10PLUS": np.zeros(n_trials),
        # Intermediate states for analysis
        "both_alive_baseline": np.zeros(n_trials, dtype=int),
        "both_alive_shocked": np.zeros(n_trials, dtype=int),
        "one_dead_baseline": np.zeros(n_trials, dtype=int),
        "one_dead_shocked": np.zeros(n_trials, dtype=int),
    }

    # Process in batches
    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)
        batch_n = batch_end - batch_start

        # Generate correlated uniforms for this batch
        U_A, U_B = generate_correlated_uniforms(n_policies, batch_n, rho)

        # --- Baseline scenario ---
        dead_A_baseline = U_A < qx_A_baseline[:, np.newaxis]
        dead_B_baseline = U_B < qx_B_baseline[:, np.newaxis]

        # Apply contagion: if A dies, B's mortality increases (and vice versa)
        if contagion_multiplier > 1.0:
            # Adjusted qx for B given A died
            qx_B_adj = apply_contagion(
                qx_B_baseline[:, np.newaxis] * np.ones((1, batch_n)),
                dead_A_baseline,
                contagion_multiplier
            )
            # Re-evaluate B's death with adjusted qx
            dead_B_baseline = U_B < qx_B_adj

            # Similarly for A given B died (symmetric contagion)
            qx_A_adj = apply_contagion(
                qx_A_baseline[:, np.newaxis] * np.ones((1, batch_n)),
                dead_B_baseline,
                contagion_multiplier
            )
            dead_A_baseline = U_A < qx_A_adj

        # Last-to-die: claim only when BOTH are dead
        both_dead_baseline = dead_A_baseline & dead_B_baseline
        one_dead_baseline = dead_A_baseline ^ dead_B_baseline  # XOR
        both_alive_baseline = ~dead_A_baseline & ~dead_B_baseline

        results["claim_count_baseline"][batch_start:batch_end] = both_dead_baseline.sum(axis=0)
        results["claim_volume_baseline"][batch_start:batch_end] = (
            both_dead_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["both_alive_baseline"][batch_start:batch_end] = both_alive_baseline.sum(axis=0)
        results["one_dead_baseline"][batch_start:batch_end] = one_dead_baseline.sum(axis=0)

        # Large claims baseline
        large_baseline = both_dead_baseline & large_mask[:, np.newaxis]
        results["claim_count_baseline_10PLUS"][batch_start:batch_end] = large_baseline.sum(axis=0)
        results["volume_baseline_10PLUS"][batch_start:batch_end] = (
            large_baseline * volumes[:, np.newaxis]
        ).sum(axis=0)

        # --- Shocked scenario ---
        dead_A_shocked = U_A < qx_A_shocked[:, np.newaxis]
        dead_B_shocked = U_B < qx_B_shocked[:, np.newaxis]

        if contagion_multiplier > 1.0:
            qx_B_adj = apply_contagion(
                qx_B_shocked[:, np.newaxis] * np.ones((1, batch_n)),
                dead_A_shocked,
                contagion_multiplier
            )
            dead_B_shocked = U_B < qx_B_adj

            qx_A_adj = apply_contagion(
                qx_A_shocked[:, np.newaxis] * np.ones((1, batch_n)),
                dead_B_shocked,
                contagion_multiplier
            )
            dead_A_shocked = U_A < qx_A_adj

        both_dead_shocked = dead_A_shocked & dead_B_shocked
        one_dead_shocked = dead_A_shocked ^ dead_B_shocked
        both_alive_shocked = ~dead_A_shocked & ~dead_B_shocked

        results["claim_count_shocked"][batch_start:batch_end] = both_dead_shocked.sum(axis=0)
        results["claim_volume_shocked"][batch_start:batch_end] = (
            both_dead_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)
        results["both_alive_shocked"][batch_start:batch_end] = both_alive_shocked.sum(axis=0)
        results["one_dead_shocked"][batch_start:batch_end] = one_dead_shocked.sum(axis=0)

        # Large claims shocked
        large_shocked = both_dead_shocked & large_mask[:, np.newaxis]
        results["claim_count_shocked_10PLUS"][batch_start:batch_end] = large_shocked.sum(axis=0)
        results["volume_shocked_10PLUS"][batch_start:batch_end] = (
            large_shocked * volumes[:, np.newaxis]
        ).sum(axis=0)

    return results


def stochastic_runs_joint_life(
    data: pd.DataFrame,
    n_trials: int,
    policy_col: str,
    life_id_col: str,
    volume_col: str,
    baseline_qx_col: str,
    shocked_qx_col: str,
    rho: float = 0.15,
    contagion_multiplier: float = 1.5,
    n_processes: str | int = "auto",
    batch_size: str | int = "auto",
) -> dict:
    """
    Run Monte Carlo simulation for joint life (last-to-die) policies.

    Uses Gaussian copula to model mortality correlation between paired lives,
    with optional contagion effect (widow/widower mortality increase).

    Parameters
    ----------
    data : pd.DataFrame
        Input data with one row per life. Joint life policies must have
        exactly 2 rows (life A and life B) sharing the same policy number.
    n_trials : int
        Number of Monte Carlo trials to run.
    policy_col : str
        Column identifying the policy (links lives together).
    life_id_col : str
        Column identifying the life within a policy ('A' or 'B').
    volume_col : str
        Column with claim volumes.
    baseline_qx_col : str
        Column with baseline mortality rates.
    shocked_qx_col : str
        Column with shocked mortality rates.
    rho : float, default=0.15
        Correlation parameter for Gaussian copula (-1 to 1).
        Typical values from literature:
        - 0.05: Long-term shared environment
        - 0.15: Moderate contagion (recommended default)
        - 0.25-0.30: Strong immediate widow effect
    contagion_multiplier : float, default=1.5
        Factor by which surviving life's qx increases when partner dies.
        Literature suggests 1.5-2.0 for immediate effect.
        Set to 1.0 to disable contagion adjustment.
    n_processes : int or "auto", default="auto"
        Number of parallel processes.
    batch_size : int or "auto", default="auto"
        Number of trials per batch.

    Returns
    -------
    dict
        Simulation results:
        - claim_volume_baseline/shocked: Total claim volumes (last-to-die)
        - claim_count_baseline/shocked: Number of claims (both lives dead)
        - claim_count_baseline/shocked_10PLUS: Large claim counts
        - volume_baseline/shocked_10PLUS: Large claim volumes
        - both_alive_baseline/shocked: Policies where both survive
        - one_dead_baseline/shocked: Policies where exactly one died
    """
    # Prepare paired data
    paired_data = prepare_joint_life_data(
        data, policy_col, life_id_col, volume_col, baseline_qx_col, shocked_qx_col
    )

    n_policies = paired_data["n_policies"]

    # Apply calibrated defaults if "auto"
    if n_processes == "auto" or batch_size == "auto":
        optimal_processes, optimal_batch = get_optimal_params(n_policies)
        if n_processes == "auto":
            n_processes = optimal_processes
        if batch_size == "auto":
            batch_size = optimal_batch

    # Extract arrays
    qx_A_baseline = paired_data["qx_A_baseline"]
    qx_B_baseline = paired_data["qx_B_baseline"]
    qx_A_shocked = paired_data["qx_A_shocked"]
    qx_B_shocked = paired_data["qx_B_shocked"]
    volumes = paired_data["volume"]
    large_mask = volumes >= 10_000_000

    # Split trials across processes
    trials_per_process = n_trials // n_processes
    remainder = n_trials % n_processes

    worker_args = []
    for i in range(n_processes):
        worker_trials = trials_per_process + (1 if i < remainder else 0)
        worker_args.append((
            qx_A_baseline, qx_B_baseline,
            qx_A_shocked, qx_B_shocked,
            volumes, large_mask,
            worker_trials, batch_size, rho, contagion_multiplier
        ))

    # Execute simulation
    if n_processes == 1:
        worker_results = [_worker_joint_life_batch(worker_args[0])]
    else:
        with Pool(n_processes, initializer=_init_worker) as pool:
            worker_results = pool.map(_worker_joint_life_batch, worker_args)

    # Concatenate results
    final_results = {}
    for key in worker_results[0].keys():
        final_results[key] = np.concatenate([w[key] for w in worker_results])

    return {k: v.tolist() for k, v in final_results.items()}


def stochastic_runs_portfolio(
    data: pd.DataFrame,
    n_trials: int,
    volume_col: str,
    baseline_qx_col: str,
    shocked_qx_col: str,
    life_status_col: str = None,
    policy_col: str = None,
    life_id_col: str = None,
    rho: float = 0.15,
    contagion_multiplier: float = 1.5,
    n_processes: str | int = "auto",
    batch_size: str | int = "auto",
) -> dict:
    """
    Run Monte Carlo simulation for a mixed portfolio of single and joint lives.

    Automatically splits the portfolio by life status, runs appropriate
    simulation for each segment, and combines results.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with one row per life.
    n_trials : int
        Number of Monte Carlo trials to run.
    volume_col : str
        Column with claim volumes.
    baseline_qx_col : str
        Column with baseline mortality rates.
    shocked_qx_col : str
        Column with shocked mortality rates.
    life_status_col : str, optional
        Column indicating 'single' or 'joint' life status.
        If None, all lives are treated as single.
    policy_col : str, optional
        Column identifying the policy (required for joint lives).
    life_id_col : str, optional
        Column identifying the life within a policy (required for joint lives).
    rho : float, default=0.15
        Correlation parameter for joint life copula.
    contagion_multiplier : float, default=1.5
        Mortality multiplier for surviving joint life.
    n_processes : int or "auto", default="auto"
        Number of parallel processes.
    batch_size : int or "auto", default="auto"
        Number of trials per batch.

    Returns
    -------
    dict
        Combined simulation results with keys:
        - claim_volume_baseline/shocked: Total claim volumes
        - claim_count_baseline/shocked: Total claim counts
        - claim_count_baseline/shocked_10PLUS: Large claim counts
        - volume_baseline/shocked_10PLUS: Large claim volumes
        - single_life_*: Results from single life segment only
        - joint_life_*: Results from joint life segment only
    """
    from mortality_simulations.simulation import stochastic_runs_hybrid

    # If no life_status_col, treat everything as single life
    if life_status_col is None:
        results = stochastic_runs_hybrid(
            data=data,
            n_trials=n_trials,
            volume_col=volume_col,
            baseline_qx_col=baseline_qx_col,
            shocked_qx_col=shocked_qx_col,
            n_processes=n_processes,
            batch_size=batch_size,
        )
        # Add prefixed copies for consistency
        for key, value in list(results.items()):
            results[f"single_life_{key}"] = value
        return results

    # Split by life status
    single_mask = data[life_status_col] == "single"
    joint_mask = data[life_status_col] == "joint"

    single_data = data[single_mask]
    joint_data = data[joint_mask]

    results = {}

    # Run single life simulation
    if len(single_data) > 0:
        single_results = stochastic_runs_hybrid(
            data=single_data,
            n_trials=n_trials,
            volume_col=volume_col,
            baseline_qx_col=baseline_qx_col,
            shocked_qx_col=shocked_qx_col,
            n_processes=n_processes,
            batch_size=batch_size,
        )
        for key, value in single_results.items():
            results[f"single_life_{key}"] = value
    else:
        # No single lives - create zero arrays
        single_results = {
            "claim_volume_baseline": [0.0] * n_trials,
            "claim_volume_shocked": [0.0] * n_trials,
            "claim_count_baseline": [0] * n_trials,
            "claim_count_shocked": [0] * n_trials,
            "claim_count_baseline_10PLUS": [0] * n_trials,
            "claim_count_shocked_10PLUS": [0] * n_trials,
            "volume_baseline_10PLUS": [0.0] * n_trials,
            "volume_shocked_10PLUS": [0.0] * n_trials,
        }
        for key, value in single_results.items():
            results[f"single_life_{key}"] = value

    # Run joint life simulation
    if len(joint_data) > 0:
        if policy_col is None or life_id_col is None:
            raise ValueError(
                "policy_col and life_id_col are required when joint lives are present"
            )

        joint_results = stochastic_runs_joint_life(
            data=joint_data,
            n_trials=n_trials,
            policy_col=policy_col,
            life_id_col=life_id_col,
            volume_col=volume_col,
            baseline_qx_col=baseline_qx_col,
            shocked_qx_col=shocked_qx_col,
            rho=rho,
            contagion_multiplier=contagion_multiplier,
            n_processes=n_processes,
            batch_size=batch_size,
        )
        for key, value in joint_results.items():
            results[f"joint_life_{key}"] = value
    else:
        # No joint lives - create zero arrays
        joint_results = {
            "claim_volume_baseline": [0.0] * n_trials,
            "claim_volume_shocked": [0.0] * n_trials,
            "claim_count_baseline": [0] * n_trials,
            "claim_count_shocked": [0] * n_trials,
            "claim_count_baseline_10PLUS": [0] * n_trials,
            "claim_count_shocked_10PLUS": [0] * n_trials,
            "volume_baseline_10PLUS": [0.0] * n_trials,
            "volume_shocked_10PLUS": [0.0] * n_trials,
        }
        for key, value in joint_results.items():
            results[f"joint_life_{key}"] = value

    # Combine results (element-wise sum across trials)
    combined_keys = [
        "claim_volume_baseline", "claim_volume_shocked",
        "claim_count_baseline", "claim_count_shocked",
        "claim_count_baseline_10PLUS", "claim_count_shocked_10PLUS",
        "volume_baseline_10PLUS", "volume_shocked_10PLUS",
    ]

    for key in combined_keys:
        single_vals = results.get(f"single_life_{key}", [0.0] * n_trials)
        joint_vals = results.get(f"joint_life_{key}", [0.0] * n_trials)
        results[key] = [s + j for s, j in zip(single_vals, joint_vals)]

    return results
