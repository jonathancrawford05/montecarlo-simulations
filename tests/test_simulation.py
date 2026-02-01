"""Tests for the mortality simulation module."""

import numpy as np
import pandas as pd
import pytest

from mortality_simulations import stochastic_runs_hybrid


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_rows = 1000
    return pd.DataFrame(
        {
            "volume": np.random.uniform(100_000, 50_000_000, n_rows),
            "baseline_qx": np.random.uniform(0.001, 0.05, n_rows),
            "shocked_qx": np.random.uniform(0.002, 0.08, n_rows),
        }
    )


class TestStochasticRunsHybrid:
    """Tests for the stochastic_runs_hybrid function."""

    def test_returns_expected_keys(self, sample_data):
        """Test that results contain all expected keys."""
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=2,
            batch_size=5,
        )

        expected_keys = {
            "claim_volume_baseline",
            "claim_volume_shocked",
            "claim_count_baseline",
            "claim_count_shocked",
            "claim_count_baseline_10PLUS",
            "claim_count_shocked_10PLUS",
            "volume_baseline_10PLUS",
            "volume_shocked_10PLUS",
        }
        assert set(results.keys()) == expected_keys

    def test_returns_correct_number_of_trials(self, sample_data):
        """Test that results have correct length for each trial."""
        n_trials = 25
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=n_trials,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=2,
        )

        for key, values in results.items():
            assert len(values) == n_trials, f"{key} has wrong length"

    def test_shocked_counts_generally_higher(self, sample_data):
        """Test that shocked mortality produces higher average claims."""
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=100,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=2,
        )

        avg_baseline = np.mean(results["claim_count_baseline"])
        avg_shocked = np.mean(results["claim_count_shocked"])

        # Shocked qx is higher, so shocked counts should generally be higher
        assert avg_shocked > avg_baseline

    def test_single_process_execution(self, sample_data):
        """Test that simulation works with single process."""
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
        )

        assert len(results["claim_count_baseline"]) == 10

    def test_results_are_lists(self, sample_data):
        """Test that results are returned as Python lists."""
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
        )

        for key, values in results.items():
            assert isinstance(values, list), f"{key} should be a list"
