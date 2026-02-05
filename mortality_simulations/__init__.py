"""Mortality Monte Carlo Simulations package."""

from mortality_simulations.simulation import (
    analyze_simulation_confidence,
    check_threading_config,
    compute_portfolio_moments,
    estimate_quantile_ci_width,
    estimate_required_simulations,
    generate_confidence_summary,
    get_optimal_params,
    print_confidence_summary,
    stochastic_runs_hybrid,
)
from mortality_simulations.joint_life import (
    stochastic_runs_joint_life,
    stochastic_runs_portfolio,
    generate_correlated_uniforms,
    prepare_joint_life_data,
    apply_contagion,
    get_joint_life_params,
    list_param_configs,
    PARAM_CONFIGS,
)

__version__ = "0.1.0"
__all__ = [
    # Single life simulation
    "stochastic_runs_hybrid",
    "get_optimal_params",
    "check_threading_config",
    # Joint life simulation
    "stochastic_runs_joint_life",
    "stochastic_runs_portfolio",
    "generate_correlated_uniforms",
    "prepare_joint_life_data",
    "apply_contagion",
    # Parameter configurations
    "get_joint_life_params",
    "list_param_configs",
    "PARAM_CONFIGS",
    # confidence analysis
    "analyze_simulation_confidence",
    "estimate_required_simulations",
    "generate_confidence_summary",
    "print_confidence_summary",
    "compute_portfolio_moments",
    "estimate_quantile_ci_width",
    # Parameter configurations
    "get_joint_life_params",
    "list_param_configs",
    "PARAM_CONFIGS",
]
