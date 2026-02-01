"""Mortality Monte Carlo Simulations package."""

from mortality_simulations.simulation import (
    check_threading_config,
    get_optimal_params,
    stochastic_runs_hybrid,
)

__version__ = "0.1.0"
__all__ = ["stochastic_runs_hybrid", "get_optimal_params", "check_threading_config"]
