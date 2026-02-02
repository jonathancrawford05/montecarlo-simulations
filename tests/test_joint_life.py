"""Tests for the joint life mortality simulation module."""

import numpy as np
import pandas as pd
import pytest

from mortality_simulations import (
    apply_contagion,
    generate_correlated_uniforms,
    get_joint_life_params,
    list_param_configs,
    PARAM_CONFIGS,
    prepare_joint_life_data,
    stochastic_runs_joint_life,
    stochastic_runs_portfolio,
)


@pytest.fixture
def joint_life_data():
    """Create sample joint life data for testing."""
    np.random.seed(42)
    n_policies = 500

    # Create paired data (2 rows per policy)
    data = []
    for i in range(n_policies):
        policy_num = f"POL_{i:04d}"
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

        # Life B (different qx values)
        data.append({
            "policy_number": policy_num,
            "life_id": "B",
            "life_status": "joint",
            "volume": volume,
            "baseline_qx": np.random.uniform(0.001, 0.05),
            "shocked_qx": np.random.uniform(0.002, 0.08),
        })

    return pd.DataFrame(data)


@pytest.fixture
def mixed_portfolio_data(joint_life_data):
    """Create mixed portfolio with single and joint lives."""
    np.random.seed(123)
    n_single = 300

    # Single life data
    single_data = pd.DataFrame({
        "policy_number": [f"SINGLE_{i:04d}" for i in range(n_single)],
        "life_id": ["A"] * n_single,  # Not used for single
        "life_status": ["single"] * n_single,
        "volume": np.random.uniform(100_000, 50_000_000, n_single),
        "baseline_qx": np.random.uniform(0.001, 0.05, n_single),
        "shocked_qx": np.random.uniform(0.002, 0.08, n_single),
    })

    return pd.concat([joint_life_data, single_data], ignore_index=True)


class TestGenerateCorrelatedUniforms:
    """Tests for the Gaussian copula uniform generation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n_pairs, n_trials = 100, 50
        U_A, U_B = generate_correlated_uniforms(n_pairs, n_trials, rho=0.5)

        assert U_A.shape == (n_pairs, n_trials)
        assert U_B.shape == (n_pairs, n_trials)

    def test_uniform_range(self):
        """Test that outputs are in [0, 1]."""
        U_A, U_B = generate_correlated_uniforms(1000, 100, rho=0.5)

        assert np.all(U_A >= 0) and np.all(U_A <= 1)
        assert np.all(U_B >= 0) and np.all(U_B <= 1)

    def test_correlation_positive(self):
        """Test that positive rho produces positive correlation."""
        np.random.seed(42)
        U_A, U_B = generate_correlated_uniforms(10000, 1, rho=0.8)

        correlation = np.corrcoef(U_A.flatten(), U_B.flatten())[0, 1]
        # Gaussian copula correlation is approximately rho for high rho
        assert correlation > 0.5

    def test_correlation_zero(self):
        """Test that rho=0 produces near-zero correlation."""
        np.random.seed(42)
        U_A, U_B = generate_correlated_uniforms(10000, 1, rho=0.0)

        correlation = np.corrcoef(U_A.flatten(), U_B.flatten())[0, 1]
        assert abs(correlation) < 0.05

    def test_correlation_negative(self):
        """Test that negative rho produces negative correlation."""
        np.random.seed(42)
        U_A, U_B = generate_correlated_uniforms(10000, 1, rho=-0.8)

        correlation = np.corrcoef(U_A.flatten(), U_B.flatten())[0, 1]
        assert correlation < -0.5

    def test_invalid_rho_raises(self):
        """Test that invalid rho values raise ValueError."""
        with pytest.raises(ValueError):
            generate_correlated_uniforms(100, 10, rho=1.5)

        with pytest.raises(ValueError):
            generate_correlated_uniforms(100, 10, rho=-1.5)


class TestPrepareJointLifeData:
    """Tests for joint life data preprocessing."""

    def test_basic_pairing(self, joint_life_data):
        """Test that data is correctly paired."""
        result = prepare_joint_life_data(
            joint_life_data,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
        )

        assert result["n_policies"] == 500
        assert len(result["qx_A_baseline"]) == 500
        assert len(result["qx_B_baseline"]) == 500

    def test_missing_life_b_warning(self):
        """Test warning when some policies lack life B."""
        data = pd.DataFrame({
            "policy_number": ["POL1", "POL1", "POL2"],
            "life_id": ["A", "B", "A"],
            "volume": [1000, 1000, 2000],
            "baseline_qx": [0.01, 0.02, 0.03],
            "shocked_qx": [0.02, 0.03, 0.04],
        })

        with pytest.warns(UserWarning, match="only life A"):
            result = prepare_joint_life_data(
                data,
                policy_col="policy_number",
                life_id_col="life_id",
                volume_col="volume",
                baseline_qx_col="baseline_qx",
                shocked_qx_col="shocked_qx",
            )

        # Only POL1 should be included
        assert result["n_policies"] == 1

    def test_invalid_life_id_raises(self):
        """Test that invalid life IDs raise ValueError."""
        data = pd.DataFrame({
            "policy_number": ["POL1", "POL1"],
            "life_id": ["X", "Y"],  # Invalid
            "volume": [1000, 1000],
            "baseline_qx": [0.01, 0.02],
            "shocked_qx": [0.02, 0.03],
        })

        with pytest.raises(ValueError, match="only 'A' and 'B'"):
            prepare_joint_life_data(
                data,
                policy_col="policy_number",
                life_id_col="life_id",
                volume_col="volume",
                baseline_qx_col="baseline_qx",
                shocked_qx_col="shocked_qx",
            )


class TestApplyContagion:
    """Tests for contagion mortality adjustment."""

    def test_contagion_increases_mortality(self):
        """Test that contagion increases qx where partner died."""
        qx = np.array([0.01, 0.02, 0.03])
        partner_died = np.array([True, False, True])

        adjusted = apply_contagion(qx, partner_died, contagion_multiplier=2.0)

        assert adjusted[0] == 0.02  # 0.01 * 2.0
        assert adjusted[1] == 0.02  # Unchanged
        assert adjusted[2] == 0.06  # 0.03 * 2.0

    def test_contagion_capped_at_one(self):
        """Test that adjusted qx is capped at 1.0."""
        qx = np.array([0.8])
        partner_died = np.array([True])

        adjusted = apply_contagion(qx, partner_died, contagion_multiplier=2.0)

        assert adjusted[0] == 1.0  # Capped, not 1.6

    def test_no_contagion_when_multiplier_one(self):
        """Test that multiplier=1.0 leaves qx unchanged."""
        qx = np.array([0.01, 0.02])
        partner_died = np.array([True, True])

        adjusted = apply_contagion(qx, partner_died, contagion_multiplier=1.0)

        np.testing.assert_array_equal(adjusted, qx)


class TestStochasticRunsJointLife:
    """Tests for the joint life simulation function."""

    def test_returns_expected_keys(self, joint_life_data):
        """Test that results contain all expected keys."""
        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=10,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
        )

        expected_keys = {
            "claim_volume_baseline", "claim_volume_shocked",
            "claim_count_baseline", "claim_count_shocked",
            "claim_count_baseline_10PLUS", "claim_count_shocked_10PLUS",
            "volume_baseline_10PLUS", "volume_shocked_10PLUS",
            "both_alive_baseline", "both_alive_shocked",
            "one_dead_baseline", "one_dead_shocked",
        }

        assert set(results.keys()) == expected_keys

    def test_returns_correct_trial_count(self, joint_life_data):
        """Test that results have correct length."""
        n_trials = 25
        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=n_trials,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
        )

        for key, values in results.items():
            assert len(values) == n_trials, f"{key} has wrong length"

    def test_last_to_die_fewer_claims(self, joint_life_data):
        """Test that last-to-die produces fewer claims than single life."""
        # For last-to-die, BOTH must die, so claims should be much lower
        # than if we just summed individual deaths
        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=100,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=0.0,
            contagion_multiplier=1.0,
            n_processes=1,
        )

        # With independent deaths and qx~0.025, P(both die) ~ 0.025^2 ~ 0.0006
        # So claim count should be much smaller than n_policies (500)
        avg_claims = np.mean(results["claim_count_baseline"])
        assert avg_claims < 50  # Should be much less than 500

    def test_correlation_increases_joint_deaths(self, joint_life_data):
        """Test that higher correlation increases joint death probability."""
        np.random.seed(42)

        # Low correlation
        results_low = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=100,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=0.0,
            contagion_multiplier=1.0,
            n_processes=1,
        )

        np.random.seed(42)

        # High correlation
        results_high = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=100,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=0.5,
            contagion_multiplier=1.0,
            n_processes=1,
        )

        # Higher correlation should lead to more "both dead" outcomes
        avg_low = np.mean(results_low["claim_count_baseline"])
        avg_high = np.mean(results_high["claim_count_baseline"])

        assert avg_high > avg_low

    def test_shocked_produces_more_claims(self, joint_life_data):
        """Test that shocked scenario produces more claims."""
        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=100,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            n_processes=1,
        )

        avg_baseline = np.mean(results["claim_count_baseline"])
        avg_shocked = np.mean(results["claim_count_shocked"])

        assert avg_shocked > avg_baseline

    def test_results_are_lists(self, joint_life_data):
        """Test that results are returned as Python lists."""
        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=10,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
        )

        for key, values in results.items():
            assert isinstance(values, list), f"{key} should be a list"


class TestStochasticRunsPortfolio:
    """Tests for the mixed portfolio simulation function."""

    def test_single_life_only(self):
        """Test portfolio with only single lives."""
        np.random.seed(42)
        data = pd.DataFrame({
            "policy_number": [f"POL_{i}" for i in range(100)],
            "life_status": ["single"] * 100,
            "volume": np.random.uniform(100_000, 1_000_000, 100),
            "baseline_qx": np.random.uniform(0.01, 0.05, 100),
            "shocked_qx": np.random.uniform(0.02, 0.08, 100),
        })

        results = stochastic_runs_portfolio(
            data=data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            life_status_col="life_status",
            n_processes=1,
        )

        # Should have combined and single_life keys
        assert "claim_count_baseline" in results
        assert "single_life_claim_count_baseline" in results

    def test_joint_life_only(self, joint_life_data):
        """Test portfolio with only joint lives."""
        results = stochastic_runs_portfolio(
            data=joint_life_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            life_status_col="life_status",
            policy_col="policy_number",
            life_id_col="life_id",
            n_processes=1,
        )

        # Should have combined and joint_life keys
        assert "claim_count_baseline" in results
        assert "joint_life_claim_count_baseline" in results

    def test_mixed_portfolio(self, mixed_portfolio_data):
        """Test portfolio with both single and joint lives."""
        results = stochastic_runs_portfolio(
            data=mixed_portfolio_data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            life_status_col="life_status",
            policy_col="policy_number",
            life_id_col="life_id",
            n_processes=1,
        )

        # Should have all key types
        assert "claim_count_baseline" in results
        assert "single_life_claim_count_baseline" in results
        assert "joint_life_claim_count_baseline" in results

        # Combined should equal sum of parts
        for i in range(10):
            single = results["single_life_claim_count_baseline"][i]
            joint = results["joint_life_claim_count_baseline"][i]
            combined = results["claim_count_baseline"][i]
            assert combined == single + joint

    def test_no_life_status_col_treats_as_single(self):
        """Test that missing life_status_col treats all as single life."""
        np.random.seed(42)
        data = pd.DataFrame({
            "volume": np.random.uniform(100_000, 1_000_000, 100),
            "baseline_qx": np.random.uniform(0.01, 0.05, 100),
            "shocked_qx": np.random.uniform(0.02, 0.08, 100),
        })

        results = stochastic_runs_portfolio(
            data=data,
            n_trials=10,
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            life_status_col=None,  # No status column
            n_processes=1,
        )

        assert "claim_count_baseline" in results
        assert "single_life_claim_count_baseline" in results

    def test_joint_without_policy_col_raises(self, joint_life_data):
        """Test that joint lives without policy_col raises error."""
        with pytest.raises(ValueError, match="policy_col.*required"):
            stochastic_runs_portfolio(
                data=joint_life_data,
                n_trials=10,
                volume_col="volume",
                baseline_qx_col="baseline_qx",
                shocked_qx_col="shocked_qx",
                life_status_col="life_status",
                policy_col=None,  # Missing
                life_id_col="life_id",
            )


class TestParamConfigs:
    """Tests for parameter configuration helper functions."""

    def test_get_joint_life_params_returns_expected_keys(self):
        """Test that get_joint_life_params returns expected keys."""
        params = get_joint_life_params("correlation_only")

        assert "rho" in params
        assert "contagion_multiplier" in params
        assert "description" in params

    def test_get_joint_life_params_all_configs(self):
        """Test that all configurations are valid."""
        for config_name in PARAM_CONFIGS.keys():
            params = get_joint_life_params(config_name)

            assert 0 <= params["rho"] <= 1
            assert params["contagion_multiplier"] >= 1.0
            assert len(params["description"]) > 0

    def test_get_joint_life_params_invalid_raises(self):
        """Test that invalid config name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config"):
            get_joint_life_params("invalid_config")

    def test_list_param_configs_returns_all(self):
        """Test that list_param_configs returns all configurations."""
        configs = list_param_configs()

        assert len(configs) == len(PARAM_CONFIGS)
        for name in PARAM_CONFIGS.keys():
            assert name in configs

    def test_correlation_only_has_no_contagion(self):
        """Test that correlation_only config disables contagion."""
        params = get_joint_life_params("correlation_only")

        assert params["contagion_multiplier"] == 1.0
        assert params["rho"] > 0

    def test_contagion_only_has_no_correlation(self):
        """Test that contagion_only config disables correlation."""
        params = get_joint_life_params("contagion_only")

        assert params["rho"] == 0.0
        assert params["contagion_multiplier"] > 1.0

    def test_params_can_be_used_in_simulation(self, joint_life_data):
        """Test that params work correctly with simulation."""
        params = get_joint_life_params("correlation_only")

        results = stochastic_runs_joint_life(
            data=joint_life_data,
            n_trials=10,
            policy_col="policy_number",
            life_id_col="life_id",
            volume_col="volume",
            baseline_qx_col="baseline_qx",
            shocked_qx_col="shocked_qx",
            rho=params["rho"],
            contagion_multiplier=params["contagion_multiplier"],
            n_processes=1,
        )

        assert len(results["claim_count_baseline"]) == 10

