"""
Spark-distributed Monte Carlo mortality simulation.

This module provides a Spark-native implementation of the mortality simulation
using pandas_udf for distributed execution across a Spark cluster. Use this
when the driver-only multiprocessing approach is insufficient.

Requires PySpark to be installed and a SparkSession to be available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd

from mortality_simulations._core import (
    RESULT_KEYS,
    concatenate_results,
    results_to_lists,
    simulate_trials,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def stochastic_runs_spark(
    spark: SparkSession,
    data: pd.DataFrame,
    n_trials: int,
    volume_col: str,
    baseline_qx_col: str,
    shocked_qx_col: str,
    n_partitions: int = 100,
    batch_size: int = 50,
) -> dict[str, list]:
    """
    Distributed Monte Carlo simulation using Spark pandas_udf.

    Distributes simulation trials across Spark workers, with each worker
    running vectorized Bernoulli simulations on the full portfolio. This
    leverages the full cluster rather than just the driver node.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    data : pandas.DataFrame
        Portfolio data containing volumes and mortality rates.
    n_trials : int
        Total number of Monte Carlo trials to run.
    volume_col : str
        Column name for claim volumes.
    baseline_qx_col : str
        Column name for baseline mortality rates (qx).
    shocked_qx_col : str
        Column name for shocked mortality rates (qx).
    n_partitions : int, default=100
        Number of Spark partitions. Controls parallelism across workers.
        More partitions = more parallel tasks, but also more overhead.
        Recommended: 2-4x the number of executor cores.
    batch_size : int, default=50
        Number of trials per vectorized batch within each partition.
        Larger batches are faster but use more memory per executor.

    Returns
    -------
    dict[str, list]
        Simulation results with same structure as stochastic_runs_hybrid:
        - claim_volume_baseline: Total claim volumes under baseline scenario
        - claim_volume_shocked: Total claim volumes under shocked scenario
        - claim_count_baseline: Claim counts under baseline scenario
        - claim_count_shocked: Claim counts under shocked scenario
        - claim_count_baseline_10PLUS: Large claim counts (>=10M) baseline
        - claim_count_shocked_10PLUS: Large claim counts (>=10M) shocked
        - volume_baseline_10PLUS: Large claim volumes baseline
        - volume_shocked_10PLUS: Large claim volumes shocked

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> from mortality_simulations.spark_simulation import stochastic_runs_spark
    >>>
    >>> spark = SparkSession.builder.getOrCreate()
    >>> results = stochastic_runs_spark(
    ...     spark,
    ...     data,
    ...     n_trials=1_000_000,
    ...     volume_col="volume",
    ...     baseline_qx_col="baseline_qx",
    ...     shocked_qx_col="shocked_qx",
    ...     n_partitions=200,  # Adjust based on cluster size
    ... )

    Notes
    -----
    Memory usage per executor: approximately n_rows * batch_size * 20 bytes.
    For a 10M row portfolio with batch_size=50, this is ~10 GB per executor.

    The portfolio data is broadcast to all executors, so ensure the driver
    has enough memory to serialize it (typically not an issue for <100M rows).
    """
    from pyspark.sql import functions as F
    from pyspark.sql.pandas.functions import pandas_udf
    from pyspark.sql.types import (
        ArrayType,
        DoubleType,
        LongType,
        StructField,
        StructType,
    )

    # Extract portfolio data as numpy arrays
    volumes = data[volume_col].values.astype(np.float64)
    baseline_qx = data[baseline_qx_col].values.astype(np.float64)
    shocked_qx = data[shocked_qx_col].values.astype(np.float64)

    # Broadcast portfolio data to all workers
    volumes_bc = spark.sparkContext.broadcast(volumes)
    baseline_qx_bc = spark.sparkContext.broadcast(baseline_qx)
    shocked_qx_bc = spark.sparkContext.broadcast(shocked_qx)

    # Distribute trials across partitions
    trials_per_partition = n_trials // n_partitions
    remainder = n_trials % n_partitions

    # Create DataFrame with partition assignments
    partition_data = []
    for i in range(n_partitions):
        partition_trials = trials_per_partition + (1 if i < remainder else 0)
        if partition_trials > 0:
            partition_data.append((i, partition_trials))

    partitions_df = spark.createDataFrame(
        partition_data, ["partition_id", "n_trials"]
    ).repartition(n_partitions, "partition_id")

    # Define output schema for the pandas_udf
    result_schema = StructType([
        StructField("claim_volume_baseline", ArrayType(DoubleType()), False),
        StructField("claim_volume_shocked", ArrayType(DoubleType()), False),
        StructField("claim_count_baseline", ArrayType(LongType()), False),
        StructField("claim_count_shocked", ArrayType(LongType()), False),
        StructField("claim_count_baseline_10PLUS", ArrayType(LongType()), False),
        StructField("claim_count_shocked_10PLUS", ArrayType(LongType()), False),
        StructField("volume_baseline_10PLUS", ArrayType(DoubleType()), False),
        StructField("volume_shocked_10PLUS", ArrayType(DoubleType()), False),
    ])

    # Capture batch_size in closure
    _batch_size = batch_size

    @pandas_udf(result_schema)
    def simulate_partition(
        iterator: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        """Run simulation for each partition's trials."""
        # Get broadcast data (once per executor)
        vols = volumes_bc.value
        base_qx = baseline_qx_bc.value
        shock_qx = shocked_qx_bc.value

        for batch_df in iterator:
            results_list = []

            for _, row in batch_df.iterrows():
                partition_id = int(row["partition_id"])
                partition_trials = int(row["n_trials"])

                # Use partition_id as seed for reproducibility
                # (different partitions get different random streams)
                rng = np.random.default_rng(seed=partition_id)

                result = simulate_trials(
                    volumes=vols,
                    baseline_qx=base_qx,
                    shocked_qx=shock_qx,
                    n_trials=partition_trials,
                    batch_size=_batch_size,
                    random_state=rng,
                )

                results_list.append({
                    "claim_volume_baseline": result["claim_volume_baseline"].tolist(),
                    "claim_volume_shocked": result["claim_volume_shocked"].tolist(),
                    "claim_count_baseline": result["claim_count_baseline"].tolist(),
                    "claim_count_shocked": result["claim_count_shocked"].tolist(),
                    "claim_count_baseline_10PLUS": result["claim_count_baseline_10PLUS"].tolist(),
                    "claim_count_shocked_10PLUS": result["claim_count_shocked_10PLUS"].tolist(),
                    "volume_baseline_10PLUS": result["volume_baseline_10PLUS"].tolist(),
                    "volume_shocked_10PLUS": result["volume_shocked_10PLUS"].tolist(),
                })

            yield pd.DataFrame(results_list)

    # Execute distributed simulation
    results_df = partitions_df.select(
        simulate_partition(F.struct("partition_id", "n_trials")).alias("results")
    ).select("results.*")

    # Collect and concatenate results
    collected = results_df.collect()

    # Flatten results from all partitions
    final_results: dict[str, list] = {key: [] for key in RESULT_KEYS}
    for row in collected:
        for key in RESULT_KEYS:
            final_results[key].extend(row[key])

    # Clean up broadcast variables
    volumes_bc.unpersist()
    baseline_qx_bc.unpersist()
    shocked_qx_bc.unpersist()

    return final_results


def estimate_spark_memory(
    n_rows: int,
    batch_size: int,
    n_partitions: int,
    executor_memory_gb: float,
) -> dict:
    """
    Estimate memory requirements for Spark simulation.

    Helps calibrate batch_size and n_partitions based on cluster resources.

    Parameters
    ----------
    n_rows : int
        Number of rows in the portfolio.
    batch_size : int
        Trials per vectorized batch.
    n_partitions : int
        Number of Spark partitions.
    executor_memory_gb : float
        Memory per Spark executor in GB.

    Returns
    -------
    dict
        Memory estimates with keys:
        - 'memory_per_batch_gb': Peak memory per batch
        - 'broadcast_size_gb': Size of broadcast portfolio data
        - 'fits_in_executor': Whether batch fits in executor memory
        - 'recommended_batch_size': Suggested batch_size if current doesn't fit
        - 'headroom_factor': Fraction of executor memory used by batch
    """
    # Memory per batch: rand_numbers (8) + dead masks (2) + volume products (8) + large masks (2)
    bytes_per_element = 20
    memory_per_batch_bytes = n_rows * batch_size * bytes_per_element
    memory_per_batch_gb = memory_per_batch_bytes / (1024 ** 3)

    # Broadcast data: 3 arrays of float64
    broadcast_bytes = n_rows * 8 * 3
    broadcast_gb = broadcast_bytes / (1024 ** 3)

    # Leave 50% headroom for Python overhead, GC, etc.
    usable_memory_gb = executor_memory_gb * 0.5
    fits = memory_per_batch_gb < usable_memory_gb

    # Calculate recommended batch_size if current doesn't fit
    if fits:
        recommended_batch_size = batch_size
    else:
        max_batch_bytes = usable_memory_gb * (1024 ** 3)
        recommended_batch_size = max(1, int(max_batch_bytes / (n_rows * bytes_per_element)))

    return {
        "memory_per_batch_gb": round(memory_per_batch_gb, 2),
        "broadcast_size_gb": round(broadcast_gb, 3),
        "fits_in_executor": fits,
        "recommended_batch_size": recommended_batch_size,
        "headroom_factor": round(memory_per_batch_gb / executor_memory_gb, 3),
    }


def recommend_spark_config(
    n_rows: int,
    n_trials: int,
    executor_cores: int,
    executor_memory_gb: float,
    num_executors: int,
) -> dict:
    """
    Recommend Spark configuration for simulation.

    Parameters
    ----------
    n_rows : int
        Number of rows in the portfolio.
    n_trials : int
        Total number of Monte Carlo trials.
    executor_cores : int
        Cores per Spark executor.
    executor_memory_gb : float
        Memory per Spark executor in GB.
    num_executors : int
        Number of Spark executors available.

    Returns
    -------
    dict
        Recommended configuration with keys:
        - 'n_partitions': Recommended number of partitions
        - 'batch_size': Recommended batch size
        - 'trials_per_partition': Trials each partition will run
        - 'estimated_parallelism': Effective parallel tasks
        - 'memory_estimate': Output from estimate_spark_memory
    """
    # Target: 2-4 partitions per core for good load balancing
    total_cores = executor_cores * num_executors
    n_partitions = min(n_trials, total_cores * 3)

    # Ensure we don't have more partitions than trials
    n_partitions = max(1, min(n_partitions, n_trials))

    # Find batch_size that fits in executor memory
    # Start with 100 and decrease until it fits
    for batch_size in [100, 50, 25, 10, 5, 1]:
        mem_est = estimate_spark_memory(
            n_rows, batch_size, n_partitions, executor_memory_gb
        )
        if mem_est["fits_in_executor"]:
            break

    trials_per_partition = n_trials // n_partitions

    return {
        "n_partitions": n_partitions,
        "batch_size": batch_size,
        "trials_per_partition": trials_per_partition,
        "estimated_parallelism": min(n_partitions, total_cores),
        "memory_estimate": mem_est,
    }
