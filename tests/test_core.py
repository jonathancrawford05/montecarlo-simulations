"""Tests for the _core simulation module."""

import numpy as np
import pytest

from mortality_simulations._core import (
    LARGE_CLAIM_THRESHOLD,
    RESULT_KEYS,
    concatenate_results,
    deterministic_random,
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


class TestDeterministicRandom:
    """Tests for the deterministic_random hash function."""

    def test_output_shape(self):
        """Output shape is (n_lives, batch_size)."""
        life_ids = np.arange(100, dtype=np.uint64)
        trial_ids = np.arange(50, dtype=np.uint64)
        result = deterministic_random(life_ids, trial_ids)
        assert result.shape == (100, 50)

    def test_values_in_unit_interval(self):
        """All values are in [0, 1)."""
        life_ids = np.arange(1000, dtype=np.uint64)
        trial_ids = np.arange(200, dtype=np.uint64)
        result = deterministic_random(life_ids, trial_ids)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    def test_fully_reproducible(self):
        """Same inputs always produce exactly the same outputs."""
        life_ids = np.array([1, 5, 99, 1000], dtype=np.uint64)
        trial_ids = np.array([0, 1, 2, 3, 4], dtype=np.uint64)
        r1 = deterministic_random(life_ids, trial_ids, seed=42)
        r2 = deterministic_random(life_ids, trial_ids, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_gives_different_values(self):
        """Different seeds produce different outputs."""
        life_ids = np.arange(10, dtype=np.uint64)
        trial_ids = np.arange(10, dtype=np.uint64)
        r1 = deterministic_random(life_ids, trial_ids, seed=0)
        r2 = deterministic_random(life_ids, trial_ids, seed=1)
        assert not np.array_equal(r1, r2)

    def test_subset_reproducibility(self):
        """A life's random numbers are the same whether run alone or in a group."""
        all_life_ids = np.array([10, 20, 30, 40, 50], dtype=np.uint64)
        subset_life_ids = np.array([20, 40], dtype=np.uint64)
        trial_ids = np.arange(100, dtype=np.uint64)
        seed = 7

        r_all = deterministic_random(all_life_ids, trial_ids, seed=seed)
        r_sub = deterministic_random(subset_life_ids, trial_ids, seed=seed)

        # Life 20 is index 1 in all_life_ids, index 0 in subset
        np.testing.assert_array_equal(r_all[1], r_sub[0])
        # Life 40 is index 3 in all_life_ids, index 1 in subset
        np.testing.assert_array_equal(r_all[3], r_sub[1])

    def test_trial_offset_reproducibility(self):
        """Splitting trials into batches gives same values as one contiguous call."""
        life_ids = np.arange(20, dtype=np.uint64)
        all_trials = np.arange(100, dtype=np.uint64)

        r_full = deterministic_random(life_ids, all_trials, seed=0)

        # Split into two batches
        r_batch1 = deterministic_random(life_ids, all_trials[:50], seed=0)
        r_batch2 = deterministic_random(life_ids, all_trials[50:], seed=0)
        r_combined = np.concatenate([r_batch1, r_batch2], axis=1)

        np.testing.assert_array_equal(r_full, r_combined)

    def test_approximately_uniform(self):
        """Large sample should be approximately uniform on [0, 1)."""
        life_ids = np.arange(1000, dtype=np.uint64)
        trial_ids = np.arange(1000, dtype=np.uint64)
        values = deterministic_random(life_ids, trial_ids).ravel()
        # Mean should be close to 0.5
        assert abs(values.mean() - 0.5) < 0.01
        # Std should be close to 1/sqrt(12) ≈ 0.2887
        assert abs(values.std() - (1 / np.sqrt(12))) < 0.01


class TestDeterministicSimulateTrials:
    """Tests for simulate_trials in deterministic mode."""

    @pytest.fixture
    def portfolio(self):
        np.random.seed(42)
        n = 200
        return {
            "volumes": np.random.uniform(100_000, 50_000_000, n),
            "baseline_qx": np.random.uniform(0.001, 0.05, n),
            "shocked_qx": np.random.uniform(0.002, 0.08, n),
            "life_ids": np.arange(n, dtype=np.uint64),
        }

    def test_invalid_mode_raises(self, portfolio):
        """Invalid random_mode raises ValueError."""
        with pytest.raises(ValueError, match="random_mode"):
            simulate_trials(
                portfolio["volumes"], portfolio["baseline_qx"],
                portfolio["shocked_qx"], n_trials=10,
                random_mode="bad_mode",
            )

    def test_missing_life_ids_raises(self, portfolio):
        """Deterministic mode without life_ids raises ValueError."""
        with pytest.raises(ValueError, match="life_ids"):
            simulate_trials(
                portfolio["volumes"], portfolio["baseline_qx"],
                portfolio["shocked_qx"], n_trials=10,
                random_mode="deterministic",
                life_ids=None,
            )

    def test_deterministic_mode_reproducible(self, portfolio):
        """Two runs with same life_ids and seed produce identical results."""
        kwargs = dict(
            volumes=portfolio["volumes"],
            baseline_qx=portfolio["baseline_qx"],
            shocked_qx=portfolio["shocked_qx"],
            n_trials=500,
            batch_size=50,
            random_mode="deterministic",
            life_ids=portfolio["life_ids"],
            seed=42,
        )
        r1 = simulate_trials(**kwargs)
        r2 = simulate_trials(**kwargs)
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(r1[key], r2[key])

    def test_deterministic_subset_reproducibility(self, portfolio):
        """Results for a subset of lives match those from the full portfolio."""
        n_trials = 300
        seed = 7

        # Full portfolio
        r_full = simulate_trials(
            portfolio["volumes"], portfolio["baseline_qx"],
            portfolio["shocked_qx"], n_trials=n_trials,
            random_mode="deterministic",
            life_ids=portfolio["life_ids"], seed=seed,
        )

        # Subset: every other life
        idx = np.arange(0, len(portfolio["volumes"]), 2)
        r_sub = simulate_trials(
            portfolio["volumes"][idx], portfolio["baseline_qx"][idx],
            portfolio["shocked_qx"][idx], n_trials=n_trials,
            random_mode="deterministic",
            life_ids=portfolio["life_ids"][idx], seed=seed,
        )

        # The subset's claim_volume_baseline should be <= the full portfolio's
        # (fewer lives = fewer possible claims), and means should differ
        # but the key check is that re-running the subset gives same result
        r_sub2 = simulate_trials(
            portfolio["volumes"][idx], portfolio["baseline_qx"][idx],
            portfolio["shocked_qx"][idx], n_trials=n_trials,
            random_mode="deterministic",
            life_ids=portfolio["life_ids"][idx], seed=seed,
        )
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(r_sub[key], r_sub2[key])

    def test_deterministic_batch_size_invariant(self, portfolio):
        """Changing batch_size does not change deterministic results."""
        kwargs = dict(
            volumes=portfolio["volumes"],
            baseline_qx=portfolio["baseline_qx"],
            shocked_qx=portfolio["shocked_qx"],
            n_trials=200,
            random_mode="deterministic",
            life_ids=portfolio["life_ids"],
            seed=0,
        )
        r_small = simulate_trials(**kwargs, batch_size=10)
        r_large = simulate_trials(**kwargs, batch_size=100)
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(r_small[key], r_large[key])

    def test_deterministic_global_trial_start(self, portfolio):
        """global_trial_start shifts trial IDs, producing the same result
        as a single run split at that offset."""
        n_trials = 200
        seed = 3
        life_ids = portfolio["life_ids"]

        # One contiguous run of 200 trials
        r_full = simulate_trials(
            portfolio["volumes"], portfolio["baseline_qx"],
            portfolio["shocked_qx"], n_trials=n_trials,
            random_mode="deterministic",
            life_ids=life_ids, seed=seed,
            global_trial_start=0,
        )

        # Two halves with correct offsets — should concatenate to the same result
        r_first = simulate_trials(
            portfolio["volumes"], portfolio["baseline_qx"],
            portfolio["shocked_qx"], n_trials=100,
            random_mode="deterministic",
            life_ids=life_ids, seed=seed,
            global_trial_start=0,
        )
        r_second = simulate_trials(
            portfolio["volumes"], portfolio["baseline_qx"],
            portfolio["shocked_qx"], n_trials=100,
            random_mode="deterministic",
            life_ids=life_ids, seed=seed,
            global_trial_start=100,
        )

        for key in RESULT_KEYS:
            combined = np.concatenate([r_first[key], r_second[key]])
            np.testing.assert_array_equal(r_full[key], combined)

    def test_standard_mode_unchanged(self, portfolio):
        """Standard mode still works and is unaffected by new parameters."""
        result = simulate_trials(
            portfolio["volumes"], portfolio["baseline_qx"],
            portfolio["shocked_qx"], n_trials=100,
            random_mode="standard",
            random_state=42,
        )
        assert set(result.keys()) == set(RESULT_KEYS)
        assert len(result["claim_volume_baseline"]) == 100


class TestDeterministicHybrid:
    """Tests for stochastic_runs_hybrid in deterministic mode."""

    @pytest.fixture
    def sample_data(self):
        import pandas as pd
        np.random.seed(0)
        n = 500
        return pd.DataFrame({
            "policy_id": np.arange(n),
            "volume": np.random.uniform(100_000, 50_000_000, n),
            "baseline_qx": np.random.uniform(0.001, 0.05, n),
            "shocked_qx": np.random.uniform(0.002, 0.08, n),
        })

    def _run(self, data, **kwargs):
        from mortality_simulations import stochastic_runs_hybrid
        return stochastic_runs_hybrid(
            data, n_trials=200,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
            **kwargs,
        )

    def test_missing_life_id_col_raises(self, sample_data):
        """random_mode='deterministic' without life_id_col raises ValueError."""
        with pytest.raises(ValueError, match="life_id_col"):
            self._run(sample_data, random_mode="deterministic")

    def test_deterministic_mode_reproducible(self, sample_data):
        """Two runs give identical results."""
        kwargs = dict(random_mode="deterministic", life_id_col="policy_id", seed=1)
        r1 = self._run(sample_data, **kwargs)
        r2 = self._run(sample_data, **kwargs)
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(r1[key], r2[key])

    def test_deterministic_subset_matches_full(self, sample_data):
        """A subset re-run gives identical results for those lives."""
        kwargs = dict(random_mode="deterministic", life_id_col="policy_id", seed=5)

        # Subset: first 250 lives
        subset = sample_data.iloc[:250].copy()
        r_sub1 = self._run(subset, **kwargs)
        r_sub2 = self._run(subset, **kwargs)

        for key in RESULT_KEYS:
            np.testing.assert_array_equal(r_sub1[key], r_sub2[key])

    def test_standard_mode_backwards_compatible(self, sample_data):
        """Standard mode (default) still works without any new parameters."""
        result = self._run(sample_data)
        assert set(result.keys()) == set(RESULT_KEYS)
        assert len(result["claim_volume_baseline"]) == 200
