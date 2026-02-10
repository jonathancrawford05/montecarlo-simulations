"""Tests for the _core simulation module."""

import numpy as np
import pytest

from mortality_simulations._core import (
    LARGE_CLAIM_THRESHOLD,
    RESULT_KEYS,
    concatenate_results,
    results_to_lists,
    simulate_trials,
)


class TestSimulateTrials:
    """Tests for the core simulate_trials function."""

    @pytest.fixture
    def portfolio(self):
        """Create a small test portfolio."""
        np.random.seed(42)
        n = 100
        return {
            "volumes": np.random.uniform(100_000, 50_000_000, n),
            "baseline_qx": np.random.uniform(0.001, 0.05, n),
            "shocked_qx": np.random.uniform(0.002, 0.08, n),
        }

    def test_returns_expected_keys(self, portfolio):
        """Result contains all expected keys."""
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
        )
        assert set(result.keys()) == set(RESULT_KEYS)

    def test_returns_numpy_arrays(self, portfolio):
        """Result values are numpy arrays."""
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
        )
        for key in RESULT_KEYS:
            assert isinstance(result[key], np.ndarray)

    def test_correct_trial_count(self, portfolio):
        """Each result array has correct length."""
        n_trials = 500
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=n_trials,
        )
        for key in RESULT_KEYS:
            assert len(result[key]) == n_trials

    def test_reproducible_with_seed(self, portfolio):
        """Same seed produces same results."""
        result1 = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
            random_state=42,
        )
        result2 = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
            random_state=42,
        )
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_different_seeds_produce_different_results(self, portfolio):
        """Different seeds produce different results."""
        result1 = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
            random_state=42,
        )
        result2 = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
            random_state=123,
        )
        # At least one key should differ
        any_different = any(
            not np.array_equal(result1[key], result2[key]) for key in RESULT_KEYS
        )
        assert any_different

    def test_accepts_generator(self, portfolio):
        """Can pass a numpy Generator as random_state."""
        rng = np.random.default_rng(42)
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
            random_state=rng,
        )
        assert len(result["claim_volume_baseline"]) == 100

    def test_shocked_generally_higher(self, portfolio):
        """Shocked claims generally exceed baseline claims."""
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=1000,
            random_state=42,
        )
        # Mean shocked should exceed mean baseline
        assert np.mean(result["claim_volume_shocked"]) > np.mean(
            result["claim_volume_baseline"]
        )

    def test_large_claims_threshold(self, portfolio):
        """Large claims are filtered by threshold."""
        # Create portfolio with known large/small volumes
        volumes = np.array([1_000_000, 5_000_000, 15_000_000, 20_000_000])
        qx = np.array([1.0, 1.0, 1.0, 1.0])  # 100% mortality for determinism

        result = simulate_trials(
            volumes, qx, qx, n_trials=1, batch_size=1, random_state=42
        )

        # Only 2 policies are >= 10M
        assert result["claim_count_baseline_10PLUS"][0] == 2
        assert result["volume_baseline_10PLUS"][0] == 15_000_000 + 20_000_000

    def test_batch_size_does_not_affect_statistics(self, portfolio):
        """Different batch sizes produce statistically similar distributions."""
        result_small = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=10000,
            batch_size=10,
            random_state=42,
        )
        result_large = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=10000,
            batch_size=500,
            random_state=123,  # Different seed is fine
        )
        # Means should be within 5% of each other
        for key in ["claim_volume_baseline", "claim_volume_shocked"]:
            mean_small = np.mean(result_small[key])
            mean_large = np.mean(result_large[key])
            relative_diff = abs(mean_small - mean_large) / mean_small
            assert relative_diff < 0.05, f"{key}: means differ by {relative_diff:.1%}"

    def test_count_dtypes_are_integer(self, portfolio):
        """Count results have integer dtype."""
        result = simulate_trials(
            portfolio["volumes"],
            portfolio["baseline_qx"],
            portfolio["shocked_qx"],
            n_trials=100,
        )
        assert np.issubdtype(result["claim_count_baseline"].dtype, np.integer)
        assert np.issubdtype(result["claim_count_shocked"].dtype, np.integer)
        assert np.issubdtype(result["claim_count_baseline_10PLUS"].dtype, np.integer)
        assert np.issubdtype(result["claim_count_shocked_10PLUS"].dtype, np.integer)


class TestConcatenateResults:
    """Tests for concatenate_results function."""

    def test_concatenates_arrays(self):
        """Concatenates multiple result dicts."""
        r1 = {key: np.array([1, 2, 3]) for key in RESULT_KEYS}
        r2 = {key: np.array([4, 5]) for key in RESULT_KEYS}

        combined = concatenate_results([r1, r2])

        for key in RESULT_KEYS:
            assert len(combined[key]) == 5
            np.testing.assert_array_equal(combined[key], [1, 2, 3, 4, 5])

    def test_empty_list_raises(self):
        """Raises ValueError for empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            concatenate_results([])

    def test_single_result(self):
        """Single result returns same arrays."""
        r1 = {key: np.array([1, 2, 3]) for key in RESULT_KEYS}
        combined = concatenate_results([r1])

        for key in RESULT_KEYS:
            np.testing.assert_array_equal(combined[key], r1[key])


class TestResultsToLists:
    """Tests for results_to_lists function."""

    def test_converts_to_lists(self):
        """Converts numpy arrays to Python lists."""
        results = {key: np.array([1.0, 2.0, 3.0]) for key in RESULT_KEYS}
        converted = results_to_lists(results)

        for key in RESULT_KEYS:
            assert isinstance(converted[key], list)
            assert converted[key] == [1.0, 2.0, 3.0]


class TestConstants:
    """Tests for module constants."""

    def test_large_claim_threshold(self):
        """Large claim threshold is 10M."""
        assert LARGE_CLAIM_THRESHOLD == 10_000_000

    def test_result_keys_complete(self):
        """All expected result keys are defined."""
        expected = {
            "claim_volume_baseline",
            "claim_volume_shocked",
            "claim_count_baseline",
            "claim_count_shocked",
            "claim_count_baseline_10PLUS",
            "claim_count_shocked_10PLUS",
            "volume_baseline_10PLUS",
            "volume_shocked_10PLUS",
        }
        assert set(RESULT_KEYS) == expected
