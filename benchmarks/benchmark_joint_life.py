"""
Benchmarks for joint life simulation performance.

Run with: poetry run python benchmarks/benchmark_joint_life.py
"""

import time
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from mortality_simulations import stochastic_runs_joint_life, stochastic_runs_portfolio


def generate_joint_life_data(n_policies: int) -> pd.DataFrame:
    """Generate synthetic joint life test data."""
    np.random.seed(42)
    data = []

    for i in range(n_policies):
        policy_num = f"POL_{i:08d}"
        volume = np.random.uniform(100_000, 50_000_000)

        # Life A
        data.append({
            "policy_number": policy_num,
            "life_id": "A",
            "life_status": "joint",
            "volume": volume,
            "baseline_qx": np.random.uniform(0.001, 0.05),
            "shocked_qx": np.random.uniform(0.002, 0.08),
        })

        # Life B
        data.append({
            "policy_number": policy_num,
            "life_id": "B",
            "life_status": "joint",
            "volume": volume,
            "baseline_qx": np.random.uniform(0.001, 0.05),
            "shocked_qx": np.random.uniform(0.002, 0.08),
        })

    return pd.DataFrame(data)


def generate_mixed_portfolio(n_single: int, n_joint_policies: int) -> pd.DataFrame:
    """Generate mixed portfolio with single and joint lives."""
    np.random.seed(42)

    # Single lives
    single_data = pd.DataFrame({
        "policy_number": [f"SINGLE_{i:08d}" for i in range(n_single)],
        "life_id": ["A"] * n_single,
        "life_status": ["single"] * n_single,
        "volume": np.random.uniform(100_000, 50_000_000, n_single),
        "baseline_qx": np.random.uniform(0.001, 0.05, n_single),
        "shocked_qx": np.random.uniform(0.002, 0.08, n_single),
    })

    # Joint lives
    joint_data = generate_joint_life_data(n_joint_policies)

    return pd.concat([single_data, joint_data], ignore_index=True)


def benchmark_joint_life(
    n_policies: int,
    n_trials: int,
    n_processes: int,
    batch_size: int,
    rho: float,
    contagion_multiplier: float,
    n_runs: int = 3,
) -> dict:
    """Benchmark joint life simulation configuration."""
    data = generate_joint_life_data(n_policies)
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = stochastic_runs_joint_life(
            data=data,
            n_trials=n_trials,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=rho,
            contagion_multiplier=contagion_multiplier,
            n_processes=n_processes,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "trials_per_second": n_trials / np.mean(times),
    }


def run_joint_life_benchmark_suite():
    """Run benchmark suite for joint life simulation."""
    print("=" * 60)
    print("Joint Life Simulation Benchmark Suite")
    print(f"CPU Cores Available: {cpu_count()}")
    print("=" * 60)

    # Test configurations
    policy_counts = [1_000, 10_000, 100_000]
    n_trials = 100
    process_counts = [1, 2]
    batch_sizes = [25, 50]

    results = []

    for n_policies in policy_counts:
        print(f"\n--- {n_policies:,} Joint Life Policies ({n_policies*2:,} rows) ---")

        for n_processes in process_counts:
            for batch_size in batch_sizes:
                print(
                    f"  Testing: {n_processes} processes, "
                    f"batch_size={batch_size}...",
                    end=" ",
                    flush=True,
                )

                try:
                    result = benchmark_joint_life(
                        n_policies=n_policies,
                        n_trials=n_trials,
                        n_processes=n_processes,
                        batch_size=batch_size,
                        rho=0.15,
                        contagion_multiplier=1.5,
                        n_runs=3,
                    )
                    print(f"{result['mean_time']:.2f}s ({result['trials_per_second']:.1f} trials/s)")

                    results.append({
                        "n_policies": n_policies,
                        "n_processes": n_processes,
                        "batch_size": batch_size,
                        **result,
                    })
                except Exception as e:
                    print(f"ERROR: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Best configurations by policy count:")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    for n_policies in policy_counts:
        subset = results_df[results_df["n_policies"] == n_policies]
        if not subset.empty:
            best = subset.loc[subset["mean_time"].idxmin()]
            print(
                f"\n{n_policies:,} policies: "
                f"{int(best['n_processes'])} processes, "
                f"batch_size={int(best['batch_size'])}, "
                f"{best['mean_time']:.2f}s"
            )

    return results_df


def run_correlation_impact_analysis():
    """Analyze impact of correlation parameter on results."""
    print("\n" + "=" * 60)
    print("Correlation Impact Analysis")
    print("=" * 60)

    n_policies = 10_000
    n_trials = 1000
    data = generate_joint_life_data(n_policies)

    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{n_policies:,} joint life policies, {n_trials:,} trials")
    print("-" * 50)
    print(f"{'rho':>6} | {'Avg Claims':>12} | {'Avg Volume':>15}")
    print("-" * 50)

    for rho in rho_values:
        np.random.seed(42)  # Reset seed for fair comparison
        results = stochastic_runs_joint_life(
            data=data,
            n_trials=n_trials,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=rho,
            contagion_multiplier=1.0,  # Disable contagion to isolate correlation effect
            n_processes=1,
        )

        avg_claims = np.mean(results["claim_count_baseline"])
        avg_volume = np.mean(results["claim_volume_baseline"])

        print(f"{rho:>6.2f} | {avg_claims:>12.1f} | {avg_volume:>15,.0f}")


def run_contagion_impact_analysis():
    """Analyze impact of contagion multiplier on results."""
    print("\n" + "=" * 60)
    print("Contagion Impact Analysis")
    print("=" * 60)

    n_policies = 10_000
    n_trials = 1000
    data = generate_joint_life_data(n_policies)

    multipliers = [1.0, 1.25, 1.5, 1.75, 2.0]

    print(f"\n{n_policies:,} joint life policies, {n_trials:,} trials, rho=0.15")
    print("-" * 50)
    print(f"{'Multiplier':>10} | {'Avg Claims':>12} | {'Avg Volume':>15}")
    print("-" * 50)

    for mult in multipliers:
        np.random.seed(42)
        results = stochastic_runs_joint_life(
            data=data,
            n_trials=n_trials,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=0.15,
            contagion_multiplier=mult,
            n_processes=1,
        )

        avg_claims = np.mean(results["claim_count_baseline"])
        avg_volume = np.mean(results["claim_volume_baseline"])

        print(f"{mult:>10.2f} | {avg_claims:>12.1f} | {avg_volume:>15,.0f}")


if __name__ == "__main__":
    run_joint_life_benchmark_suite()
    run_correlation_impact_analysis()
    run_contagion_impact_analysis()
