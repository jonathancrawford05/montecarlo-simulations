"""Tests for the mortality simulation module."""

import numpy as np
import pandas as pd
import pytest

from mortality_simulations import (
    analyze_simulation_confidence,
    check_threading_config,
    estimate_required_simulations,
    generate_confidence_summary,
    get_optimal_params,
    print_confidence_summary,
    stochastic_runs_hybrid,
)


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

    def test_single_process_bypasses_pool(self, sample_data):
        """Test that single process mode is faster (bypasses Pool overhead)."""
        import time

        # Warm up
        _ = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=5,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
        )

        # Time single process (should bypass Pool)
        start = time.perf_counter()
        for _ in range(3):
            _ = stochastic_runs_hybrid(
                data=sample_data,
                n_trials=10,
                volume_col="volume",
                baseline_qx_col="baseline_qx",
                shocked_qx_col="shocked_qx",
                n_processes=1,
            )
        single_time = time.perf_counter() - start

        # Single process should complete quickly (no Pool spawn overhead)
        # With 1000 rows and 10 trials, this should be < 0.5s total for 3 runs
        assert single_time < 1.0, f"Single process took {single_time:.2f}s, expected < 1s"

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

    def test_auto_params_small_data(self, sample_data):
        """Test that auto params work correctly for small datasets."""
        # sample_data is 1000 rows, should use 1 process, batch_size=50
        results = stochastic_runs_hybrid(
            data=sample_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes="auto",
            batch_size="auto",
        )

        assert len(results["claim_count_baseline"]) == 10


class TestGetOptimalParams:
    """Tests for the get_optimal_params function."""

    def test_small_data_single_process(self):
        """Small datasets should use single process."""
        n_processes, batch_size = get_optimal_params(10_000)
        assert n_processes == 1
        assert batch_size == 50

    def test_medium_data_two_processes(self):
        """Medium datasets should use 2 processes."""
        n_processes, batch_size = get_optimal_params(1_000_000)
        assert n_processes == 2
        assert batch_size == 25

    def test_large_data_memory_constrained(self):
        """Large datasets should use small batches for memory."""
        n_processes, batch_size = get_optimal_params(20_000_000)
        assert n_processes == 2
        assert batch_size == 10

    def test_boundary_500k(self):
        """Test boundary at 500K rows."""
        # Just below boundary
        n_processes, batch_size = get_optimal_params(499_999)
        assert n_processes == 1

        # At boundary
        n_processes, batch_size = get_optimal_params(500_000)
        assert n_processes == 2

    def test_boundary_5m(self):
        """Test boundary at 5M rows."""
        # Just below boundary
        n_processes, batch_size = get_optimal_params(4_999_999)
        assert batch_size == 25

        # At boundary
        n_processes, batch_size = get_optimal_params(5_000_000)
        assert batch_size == 10


class TestCheckThreadingConfig:
    """Tests for the check_threading_config function."""

    def test_returns_expected_keys(self):
        """Test that config contains expected keys."""
        config = check_threading_config()

        assert "env_vars" in config
        assert "warning" in config
        assert "threadpool_info" in config

    def test_env_vars_structure(self):
        """Test that env_vars has expected structure."""
        config = check_threading_config()

        assert "OMP_NUM_THREADS" in config["env_vars"]
        assert "MKL_NUM_THREADS" in config["env_vars"]
        assert "OPENBLAS_NUM_THREADS" in config["env_vars"]


@pytest.fixture
def simulation_results(sample_data):
    """Run simulation to get results for confidence analysis tests."""
    np.random.seed(42)
    return stochastic_runs_hybrid(
        data=sample_data,
        n_trials=1000,
        volume_col="volume",
        baseline_qx_col="baseline_qx",
        shocked_qx_col="shocked_qx",
        n_processes=1,
    )


class TestAnalyzeSimulationConfidence:
    """Tests for the analyze_simulation_confidence function."""

    def test_returns_expected_keys(self, simulation_results):
        """Test that results contain all expected keys."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )

        expected_keys = {
            "n_simulations",
            "quantile",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "ci_width",
            "ci_width_relative",
            "confidence_level",
            "order_stat_lower",
            "order_stat_upper",
        }
        assert set(ci.keys()) == expected_keys

    def test_default_quantile_is_95th(self, simulation_results):
        """Test default quantile is 0.95."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        assert ci["quantile"] == 0.95

    def test_default_confidence_is_95(self, simulation_results):
        """Test default confidence level is 0.95."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        assert ci["confidence_level"] == 0.95

    def test_ci_bounds_order(self, simulation_results):
        """Test that CI lower <= point estimate <= CI upper."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

    def test_ci_width_positive(self, simulation_results):
        """Test that CI width is non-negative."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        assert ci["ci_width"] >= 0

    def test_ci_width_relative_calculation(self, simulation_results):
        """Test relative CI width is calculated correctly."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        expected_rel = (ci["ci_width"] / ci["point_estimate"]) * 100
        assert abs(ci["ci_width_relative"] - expected_rel) < 0.001

    def test_different_quantiles(self, simulation_results):
        """Test analysis works for different quantiles."""
        ci_50 = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked", quantile=0.50
        )
        ci_99 = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked", quantile=0.99
        )

        # 99th percentile should be higher than 50th
        assert ci_99["point_estimate"] > ci_50["point_estimate"]

    def test_different_confidence_levels(self, simulation_results):
        """Test that higher confidence produces wider CI."""
        ci_90 = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked", confidence_level=0.90
        )
        ci_99 = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked", confidence_level=0.99
        )

        # 99% CI should be wider than 90% CI
        assert ci_99["ci_width"] > ci_90["ci_width"]

    def test_invalid_metric_raises(self, simulation_results):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            analyze_simulation_confidence(
                simulation_results, "nonexistent_metric"
            )

    def test_insufficient_simulations_raises(self):
        """Test that too few simulations raises ValueError."""
        small_results = {"claim_volume_shocked": [100, 200, 300]}
        with pytest.raises(ValueError, match="at least 10"):
            analyze_simulation_confidence(small_results, "claim_volume_shocked")

    def test_n_simulations_correct(self, simulation_results):
        """Test that n_simulations matches input length."""
        ci = analyze_simulation_confidence(
            simulation_results, "claim_volume_shocked"
        )
        assert ci["n_simulations"] == len(simulation_results["claim_volume_shocked"])


class TestEstimateRequiredSimulations:
    """Tests for the estimate_required_simulations function."""

    def test_returns_expected_keys(self, simulation_results):
        """Test that results contain all expected keys."""
        est = estimate_required_simulations(
            simulation_results, "claim_volume_shocked"
        )

        expected_keys = {
            "pilot_n",
            "pilot_ci_width_relative",
            "target_ci_width_relative",
            "estimated_n_required",
            "scaling_factor",
            "quantile",
            "confidence_level",
        }
        assert expected_keys.issubset(set(est.keys()))

    def test_tighter_target_requires_more_simulations(self, simulation_results):
        """Test that tighter CI requires more simulations."""
        est_5pct = estimate_required_simulations(
            simulation_results, "claim_volume_shocked",
            target_ci_width_relative=5.0
        )
        est_1pct = estimate_required_simulations(
            simulation_results, "claim_volume_shocked",
            target_ci_width_relative=1.0
        )

        # 1% target should require more simulations than 5%
        assert est_1pct["estimated_n_required"] > est_5pct["estimated_n_required"]

    def test_scaling_factor_quadratic(self, simulation_results):
        """Test that scaling follows quadratic relationship."""
        est = estimate_required_simulations(
            simulation_results, "claim_volume_shocked",
            target_ci_width_relative=1.0
        )

        # n scales as (current_width / target_width)^2
        expected_scaling = (est["pilot_ci_width_relative"] / 1.0) ** 2
        assert abs(est["scaling_factor"] - expected_scaling) < 0.01

    def test_pilot_n_correct(self, simulation_results):
        """Test that pilot_n matches input length."""
        est = estimate_required_simulations(
            simulation_results, "claim_volume_shocked"
        )
        assert est["pilot_n"] == len(simulation_results["claim_volume_shocked"])


class TestGenerateConfidenceSummary:
    """Tests for the generate_confidence_summary function."""

    def test_returns_expected_keys(self, simulation_results):
        """Test that summary contains all expected keys."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )

        expected_keys = {
            "metric",
            "quantile",
            "confidence_level",
            "current_analysis",
            "scenarios",
            "recommendation",
        }
        assert set(summary.keys()) == expected_keys

    def test_default_scenarios(self, simulation_results):
        """Test default simulation scenarios are 10K, 100K, 1M."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )

        scenario_ns = [s["n_simulations"] for s in summary["scenarios"]]
        assert scenario_ns == [10_000, 100_000, 1_000_000]

    def test_custom_scenarios(self, simulation_results):
        """Test custom simulation scenarios."""
        custom = [5_000, 50_000, 500_000]
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked",
            simulation_scenarios=custom
        )

        scenario_ns = [s["n_simulations"] for s in summary["scenarios"]]
        assert scenario_ns == custom

    def test_ci_width_decreases_with_more_simulations(self, simulation_results):
        """Test that projected CI width decreases with more simulations."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )

        widths = [s["projected_ci_width_relative"] for s in summary["scenarios"]]
        # Each subsequent scenario should have smaller width
        assert widths[0] > widths[1] > widths[2]

    def test_recommendation_keys(self, simulation_results):
        """Test that recommendations contain expected use cases."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )

        expected_use_cases = {"exploratory", "reporting", "regulatory", "precision"}
        assert set(summary["recommendation"].keys()) == expected_use_cases

    def test_recommendation_thresholds(self, simulation_results):
        """Test that recommendation thresholds are correct."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )

        assert summary["recommendation"]["exploratory"]["threshold"] == 5.0
        assert summary["recommendation"]["reporting"]["threshold"] == 2.0
        assert summary["recommendation"]["regulatory"]["threshold"] == 1.0
        assert summary["recommendation"]["precision"]["threshold"] == 0.5

    def test_is_current_flag(self, simulation_results):
        """Test that is_current flag is set correctly."""
        n_sims = len(simulation_results["claim_volume_shocked"])
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked",
            simulation_scenarios=[n_sims, 100_000, 1_000_000]
        )

        current_flags = [s["is_current"] for s in summary["scenarios"]]
        assert current_flags == [True, False, False]


class TestPrintConfidenceSummary:
    """Tests for the print_confidence_summary function."""

    def test_prints_without_error(self, simulation_results, capsys):
        """Test that print function executes without error."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )
        print_confidence_summary(summary)

        captured = capsys.readouterr()
        assert "SIMULATION CONFIDENCE ANALYSIS" in captured.out
        assert "claim_volume_shocked" in captured.out

    def test_prints_all_sections(self, simulation_results, capsys):
        """Test that all sections are printed."""
        summary = generate_confidence_summary(
            simulation_results, "claim_volume_shocked"
        )
        print_confidence_summary(summary)

        captured = capsys.readouterr()
        assert "CURRENT RESULTS" in captured.out
        assert "PROJECTED CI WIDTH" in captured.out
        assert "RECOMMENDATIONS BY USE CASE" in captured.out
