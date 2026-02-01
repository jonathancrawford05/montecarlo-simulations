"""
Benchmarks for calibrating simulation performance across different data sizes.

Run with: poetry run python benchmarks/benchmark_simulation.py
"""

import time
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from mortality_simulations import stochastic_runs_hybrid


def generate_test_data(n_rows: int) -> pd.DataFrame:
    """Generate synthetic test data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "volume": np.random.uniform(100_000, 50_000_000, n_rows),
            "baseline_qx": np.random.uniform(0.001, 0.05, n_rows),
            "shocked_qx": np.random.uniform(0.002, 0.08, n_rows),
        }
    )


def estimate_memory_usage(n_rows: int, batch_size: int) -> float:
    """Estimate memory usage in GB for a single batch."""
    # Each batch creates: rand_numbers (n_rows x batch_size) float64
    # Plus boolean arrays for dead_baseline, dead_shocked, large_baseline, large_shocked
    float_bytes = n_rows * batch_size * 8  # float64
    bool_bytes = n_rows * batch_size * 4  # 4 boolean arrays
    return (float_bytes + bool_bytes) / (1024**3)


def benchmark_configuration(
    data: pd.DataFrame,
    n_trials: int,
    n_processes: int,
    batch_size: int,
    n_runs: int = 3,
) -> dict:
    """Benchmark a specific configuration."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = stochastic_runs_hybrid(
            data=data,
            n_trials=n_trials,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
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


def run_calibration_suite():
    """Run a suite of benchmarks to calibrate for your MacBook."""
    print("=" * 60)
    print("Mortality Simulation Benchmark Suite")
    print(f"CPU Cores Available: {cpu_count()}")
    print("=" * 60)

    # Test configurations
    row_sizes = [10_000, 100_000, 1_000_000]
    n_trials = 100
    process_counts = [1, 2, 4, cpu_count()]
    batch_sizes = [10, 25, 50, 100]

    results = []

    for n_rows in row_sizes:
        print(f"\n--- Data Size: {n_rows:,} rows ---")
        data = generate_test_data(n_rows)

        for n_processes in process_counts:
            if n_processes > cpu_count():
                continue

            for batch_size in batch_sizes:
                mem_estimate = estimate_memory_usage(n_rows, batch_size)

                print(
                    f"  Testing: {n_processes} processes, "
                    f"batch_size={batch_size} (~{mem_estimate:.2f} GB/batch)...",
                    end=" ",
                    flush=True,
                )

                try:
                    result = benchmark_configuration(
                        data, n_trials, n_processes, batch_size, n_runs=3
                    )
                    print(f"{result['mean_time']:.2f}s ({result['trials_per_second']:.1f} trials/s)")

                    results.append(
                        {
                            "n_rows": n_rows,
                            "n_processes": n_processes,
                            "batch_size": batch_size,
                            "mem_estimate_gb": mem_estimate,
                            **result,
                        }
                    )
                except MemoryError:
                    print("MEMORY ERROR")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Best configurations by data size:")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    for n_rows in row_sizes:
        subset = results_df[results_df["n_rows"] == n_rows]
        if not subset.empty:
            best = subset.loc[subset["mean_time"].idxmin()]
            print(
                f"\n{n_rows:,} rows: "
                f"{best['n_processes']} processes, "
                f"batch_size={best['batch_size']}, "
                f"{best['mean_time']:.2f}s"
            )

    return results_df


if __name__ == "__main__":
    run_calibration_suite()
