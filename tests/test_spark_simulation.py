"""Tests for the spark_simulation module.

Tests for utility functions run without Spark. Tests requiring a SparkSession
are marked with @pytest.mark.spark and skipped if PySpark is not available.
"""

import numpy as np
import pandas as pd
import pytest

# Check if PySpark is available
try:
    from pyspark.sql import SparkSession

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

from mortality_simulations.spark_simulation import (
    estimate_spark_memory,
    recommend_spark_config,
)


class TestEstimateSparkMemory:
    """Tests for estimate_spark_memory utility function."""

    def test_returns_expected_keys(self):
        """Result contains all expected keys."""
        result = estimate_spark_memory(
            n_rows=1_000_000,
            batch_size=50,
            n_partitions=100,
            executor_memory_gb=16.0,
        )
        expected_keys = {
            "memory_per_batch_gb",
            "broadcast_size_gb",
            "fits_in_executor",
            "recommended_batch_size",
            "headroom_factor",
        }
        assert set(result.keys()) == expected_keys

    def test_memory_scales_with_rows(self):
        """Memory increases proportionally with row count."""
        result_small = estimate_spark_memory(
            n_rows=1_000_000, batch_size=50, n_partitions=100, executor_memory_gb=16.0
        )
        result_large = estimate_spark_memory(
            n_rows=10_000_000, batch_size=50, n_partitions=100, executor_memory_gb=16.0
        )
        # 10x rows should mean ~10x memory
        assert result_large["memory_per_batch_gb"] > result_small["memory_per_batch_gb"] * 9

    def test_memory_scales_with_batch_size(self):
        """Memory increases proportionally with batch size."""
        result_small = estimate_spark_memory(
            n_rows=1_000_000, batch_size=10, n_partitions=100, executor_memory_gb=16.0
        )
        result_large = estimate_spark_memory(
            n_rows=1_000_000, batch_size=100, n_partitions=100, executor_memory_gb=16.0
        )
        # 10x batch_size should mean ~10x memory
        assert result_large["memory_per_batch_gb"] > result_small["memory_per_batch_gb"] * 9

    def test_fits_in_executor_true_when_small(self):
        """Small batch fits in large executor."""
        result = estimate_spark_memory(
            n_rows=100_000, batch_size=10, n_partitions=100, executor_memory_gb=64.0
        )
        assert result["fits_in_executor"] is True

    def test_fits_in_executor_false_when_large(self):
        """Large batch doesn't fit in small executor."""
        result = estimate_spark_memory(
            n_rows=10_000_000, batch_size=500, n_partitions=100, executor_memory_gb=4.0
        )
        assert result["fits_in_executor"] is False

    def test_recommends_smaller_batch_when_needed(self):
        """Recommends smaller batch_size when current doesn't fit."""
        result = estimate_spark_memory(
            n_rows=10_000_000, batch_size=500, n_partitions=100, executor_memory_gb=4.0
        )
        assert result["recommended_batch_size"] < 500

    def test_recommends_same_batch_when_fits(self):
        """Keeps same batch_size when it fits."""
        result = estimate_spark_memory(
            n_rows=100_000, batch_size=50, n_partitions=100, executor_memory_gb=64.0
        )
        assert result["recommended_batch_size"] == 50

    def test_broadcast_size_calculation(self):
        """Broadcast size calculated correctly."""
        n_rows = 1_000_000
        result = estimate_spark_memory(
            n_rows=n_rows, batch_size=50, n_partitions=100, executor_memory_gb=16.0
        )
        # 3 arrays × n_rows × 8 bytes
        expected_gb = (3 * n_rows * 8) / (1024**3)
        assert abs(result["broadcast_size_gb"] - expected_gb) < 0.001

    def test_headroom_factor_calculation(self):
        """Headroom factor is memory_per_batch / executor_memory."""
        result = estimate_spark_memory(
            n_rows=1_000_000, batch_size=50, n_partitions=100, executor_memory_gb=16.0
        )
        expected = result["memory_per_batch_gb"] / 16.0
        assert abs(result["headroom_factor"] - expected) < 0.001


class TestRecommendSparkConfig:
    """Tests for recommend_spark_config utility function."""

    def test_returns_expected_keys(self):
        """Result contains all expected keys."""
        result = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=100_000,
            executor_cores=4,
            executor_memory_gb=16.0,
            num_executors=10,
        )
        expected_keys = {
            "n_partitions",
            "batch_size",
            "trials_per_partition",
            "estimated_parallelism",
            "memory_estimate",
        }
        assert set(result.keys()) == expected_keys

    def test_partitions_scale_with_cores(self):
        """More cores results in more partitions."""
        result_small = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=100_000,
            executor_cores=2,
            executor_memory_gb=16.0,
            num_executors=5,
        )
        result_large = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=100_000,
            executor_cores=8,
            executor_memory_gb=16.0,
            num_executors=10,
        )
        assert result_large["n_partitions"] >= result_small["n_partitions"]

    def test_partitions_capped_by_trials(self):
        """Can't have more partitions than trials."""
        result = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=50,  # Very few trials
            executor_cores=8,
            executor_memory_gb=16.0,
            num_executors=100,
        )
        assert result["n_partitions"] <= 50

    def test_batch_size_fits_in_memory(self):
        """Recommended batch_size fits in executor memory."""
        result = recommend_spark_config(
            n_rows=10_000_000,
            n_trials=1_000_000,
            executor_cores=4,
            executor_memory_gb=8.0,  # Limited memory
            num_executors=10,
        )
        assert result["memory_estimate"]["fits_in_executor"] is True

    def test_trials_per_partition_calculation(self):
        """Trials per partition is n_trials / n_partitions."""
        result = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=100_000,
            executor_cores=4,
            executor_memory_gb=16.0,
            num_executors=10,
        )
        expected = 100_000 // result["n_partitions"]
        assert result["trials_per_partition"] == expected

    def test_parallelism_capped_by_partitions(self):
        """Estimated parallelism doesn't exceed partitions."""
        result = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=10,  # Very few trials
            executor_cores=8,
            executor_memory_gb=16.0,
            num_executors=100,
        )
        assert result["estimated_parallelism"] <= result["n_partitions"]

    def test_parallelism_capped_by_cores(self):
        """Estimated parallelism doesn't exceed total cores."""
        result = recommend_spark_config(
            n_rows=1_000_000,
            n_trials=1_000_000,
            executor_cores=4,
            executor_memory_gb=16.0,
            num_executors=10,
        )
        total_cores = 4 * 10
        assert result["estimated_parallelism"] <= total_cores


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestStochasticRunsSpark:
    """Integration tests for stochastic_runs_spark (require PySpark)."""

    @pytest.fixture(scope="class")
    def spark(self):
        """Create a local SparkSession for testing."""
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("mortality_sim_test")
            .getOrCreate()
        )
        yield spark
        spark.stop()

    @pytest.fixture
    def sample_data(self):
        """Create sample portfolio data."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame(
            {
                "volume": np.random.uniform(100_000, 50_000_000, n),
                "baseline_qx": np.random.uniform(0.001, 0.05, n),
                "shocked_qx": np.random.uniform(0.002, 0.08, n),
            }
        )

    def test_returns_expected_keys(self, spark, sample_data):
        """Result contains all expected keys."""
        from mortality_simulations.spark_simulation import stochastic_runs_spark

        result = stochastic_runs_spark(
            spark,
            sample_data,
            n_trials=100,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_partitions=4,
        )
        from mortality_simulations._core import RESULT_KEYS

        assert set(result.keys()) == set(RESULT_KEYS)

    def test_returns_correct_trial_count(self, spark, sample_data):
        """Each result list has correct length."""
        from mortality_simulations.spark_simulation import stochastic_runs_spark

        n_trials = 500
        result = stochastic_runs_spark(
            spark,
            sample_data,
            n_trials=n_trials,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_partitions=10,
        )
        for key in result:
            assert len(result[key]) == n_trials

    def test_results_are_lists(self, spark, sample_data):
        """Results are Python lists (for compatibility with hybrid version)."""
        from mortality_simulations.spark_simulation import stochastic_runs_spark

        result = stochastic_runs_spark(
            spark,
            sample_data,
            n_trials=100,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_partitions=4,
        )
        for key in result:
            assert isinstance(result[key], list)

    def test_shocked_generally_higher(self, spark, sample_data):
        """Shocked claims generally exceed baseline claims."""
        from mortality_simulations.spark_simulation import stochastic_runs_spark

        result = stochastic_runs_spark(
            spark,
            sample_data,
            n_trials=1000,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_partitions=10,
        )
        assert np.mean(result["claim_volume_shocked"]) > np.mean(
            result["claim_volume_baseline"]
        )

    def test_compatible_with_confidence_analysis(self, spark, sample_data):
        """Results work with confidence analysis functions."""
        from mortality_simulations import analyze_simulation_confidence
        from mortality_simulations.spark_simulation import stochastic_runs_spark

        result = stochastic_runs_spark(
            spark,
            sample_data,
            n_trials=1000,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_partitions=10,
        )

        ci = analyze_simulation_confidence(result, "claim_volume_shocked")
        assert "point_estimate" in ci
        assert "ci_lower" in ci
        assert "ci_upper" in ci
