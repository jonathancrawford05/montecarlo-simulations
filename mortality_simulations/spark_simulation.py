"""
Spark-distributed Monte Carlo mortality simulation.

This module provides a Spark-native implementation of the mortality simulation
using mapInPandas for distributed execution across a Spark cluster. Use this
when the driver-only multiprocessing approach is insufficient.

Requires PySpark 3.0+ and a SparkSession to be available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mortality_simulations._core import (
    RESULT_KEYS,
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
    random_mode: str = "standard",
    life_id_col: str | None = None,
    seed: int = 0,
) -> dict[str, list]:
    """
    Distributed Monte Carlo simulation using Spark mapInPandas.

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
    random_mode : {"standard", "deterministic"}, default="standard"
        - "standard": sequential numpy RNG seeded per partition.
        - "deterministic": hash-based RNG keyed on (life_id, trial_id).
          Guarantees identical results for any life regardless of portfolio
          subset, batch size, or partition count. Requires ``life_id_col``.
    life_id_col : str or None, default=None
        Column containing a unique integer identifier for each life.
        Required when random_mode="deterministic".
    seed : int, default=0
        Hash salt for deterministic mode. Use the same value across all
        runs you want to compare. Ignored in standard mode.

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
    Standard mode (existing behaviour):

    >>> results = stochastic_runs_spark(
    ...     spark, data, n_trials=1_000_000,
    ...     volume_col="volume",
    ...     baseline_qx_col="baseline_qx",
    ...     shocked_qx_col="shocked_qx",
    ...     n_partitions=120, batch_size=25,
    ... )

    Deterministic mode (subset-reproducible):

    >>> results = stochastic_runs_spark(
    ...     spark, data, n_trials=1_000_000,
    ...     volume_col="volume",
    ...     baseline_qx_col="baseline_qx",
    ...     shocked_qx_col="shocked_qx",
    ...     n_partitions=120, batch_size=25,
    ...     life_id_col="policy_id",
    ...     random_mode="deterministic",
    ...     seed=42,
    ... )

    Notes
    -----
    Memory per executor batch: n_rows * batch_size * 20 bytes.
    For a 10M row portfolio with batch_size=25, this is ~5 GB per executor.

    The portfolio data is broadcast to all executors
    (3 arrays × n_rows × 8 bytes ≈ 240 MB for 10M rows).
    """
    if random_mode == "deterministic" and life_id_col is None:
        raise ValueError("life_id_col is required when random_mode='deterministic'")

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
    life_ids = (
        data[life_id_col].values.astype(np.uint64)
        if life_id_col is not None
        else None
    )

    # Broadcast portfolio data to all workers
    volumes_bc = spark.sparkContext.broadcast(volumes)
    baseline_qx_bc = spark.sparkContext.broadcast(baseline_qx)
    shocked_qx_bc = spark.sparkContext.broadcast(shocked_qx)
    life_ids_bc = spark.sparkContext.broadcast(life_ids) if life_ids is not None else None

    # Distribute trials across partitions, tracking global trial start offset
    # so deterministic mode can form globally unique trial IDs per partition.
    trials_per_partition = n_trials // n_partitions
    remainder = n_trials % n_partitions

    partition_data = []
    global_trial_start = 0
    for i in range(n_partitions):
        partition_trials = trials_per_partition + (1 if i < remainder else 0)
        if partition_trials > 0:
            # (partition_id, n_trials_this_partition, global_trial_start_offset)
            partition_data.append((i, partition_trials, global_trial_start))
            global_trial_start += partition_trials

    partitions_df = spark.createDataFrame(
        partition_data, ["partition_id", "n_trials", "trial_start"]
    ).repartition(n_partitions, "partition_id")

    # Define output schema for mapInPandas
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

    # Capture closure variables
    _batch_size = batch_size
    _random_mode = random_mode
    _seed = seed
    _volumes_bc = volumes_bc
    _baseline_qx_bc = baseline_qx_bc
    _shocked_qx_bc = shocked_qx_bc
    _life_ids_bc = life_ids_bc

    def process_partition(iterator):
        """Process each partition's trials using mapInPandas."""
        vols = _volumes_bc.value
        base_qx = _baseline_qx_bc.value
        shock_qx = _shocked_qx_bc.value
        lids = _life_ids_bc.value if _life_ids_bc is not None else None

        for batch_df in iterator:
            results_list = []

            for _, row in batch_df.iterrows():
                partition_id = int(row["partition_id"])
                partition_trials = int(row["n_trials"])
                trial_start = int(row["trial_start"])

                result = simulate_trials(
                    volumes=vols,
                    baseline_qx=base_qx,
                    shocked_qx=shock_qx,
                    n_trials=partition_trials,
                    batch_size=_batch_size,
                    random_state=np.random.default_rng(seed=partition_id),
                    random_mode=_random_mode,
                    life_ids=lids,
                    global_trial_start=trial_start,
                    seed=_seed,
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

    # Execute distributed simulation using mapInPandas (Spark 3.0+)
    results_df = partitions_df.mapInPandas(process_partition, schema=result_schema)

    # Collect and flatten results from all partitions
    final_results: dict[str, list] = {key: [] for key in RESULT_KEYS}
    for row in results_df.collect():
        for key in RESULT_KEYS:
            final_results[key].extend(row[key])

    # Clean up broadcast variables
    volumes_bc.unpersist()
    baseline_qx_bc.unpersist()
    shocked_qx_bc.unpersist()
    if life_ids_bc is not None:
        life_ids_bc.unpersist()

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
