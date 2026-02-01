"""Mortality Monte Carlo Simulations package."""

from mortality_simulations.simulation import (
    check_threading_config,
    get_optimal_params,
    stochastic_runs_hybrid,
)
from mortality_simulations.joint_life import (
    stochastic_runs_joint_life,
    stochastic_runs_portfolio,
    generate_correlated_uniforms,
    prepare_joint_life_data,
    apply_contagion,
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
]
